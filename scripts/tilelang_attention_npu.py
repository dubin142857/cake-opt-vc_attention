"""Ordinary dense, noncausal BNSD attention for Ascend A2/A3 (DAV2201).

This uses *tile-ai/tilelang-ascend*, not the upstream CUDA TileLang wheel or
the upstream Ascend 950 dialect. QK and PV use native FP16/BF16 Cube GEMM;
the online softmax, normalizer and running output use FP32 Vector arithmetic.
As in ordinary FlashAttention, the softmax tile is rounded to the input dtype
before PV. There is no integer/FP8 quantizer, ExpCast, VSmooth, or VC attention.

API: build_attention(...).prepare(q, k, v), then run_prepared(prepared).
Preparation pads and snapshots tail inputs, and allocates reusable output and
per-active-Cube-core exchange buffers. A prepared call belongs to its creation
NPU stream; do not use it concurrently or modify its buffers while in flight.
The returned output aliases reusable storage and is overwritten on the next run.
Compilation, padding and allocation are outside run_prepared's timed region.

Reference implementation and API source (MIT licensed):
https://github.com/tile-ai/tilelang-ascend/blob/1d7525399141b242435e6e1c60cc006df0a3d7a6/examples/flash_attention/flash_attn_bhsd.py
https://github.com/tile-ai/tilelang-ascend/blob/1d7525399141b242435e6e1c60cc006df0a3d7a6/tilelang/language/ascend.py
T.gemm_v0 is this fork's AscendC L1 -> Cube -> L0C primitive. Its T.gemm
retains the generic upstream tl.gemm interface; it is not used here.

Upstream portions: Copyright (c) Tile-AI. MIT License:
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Only static/CPU validation is possible on the development Windows host.
An NPU machine with CANN must compile and validate this implementation.
"""

from dataclasses import asdict, dataclass, field
import ctypes
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any


SOURCE_COMMIT = "1d7525399141b242435e6e1c60cc006df0a3d7a6"
SOURCE_REPOSITORY = "https://github.com/tile-ai/tilelang-ascend"
SOURCE_URL = f"{SOURCE_REPOSITORY}/tree/{SOURCE_COMMIT}"
DEFAULT_SHAPE = (1, 28, 37741, 128)
IMPLEMENTATION_REVISION = "broadcast_softmax_tail_mask_v2"


@dataclass(frozen=True)
class AttentionSpec:
    batch: int = 1
    heads: int = 28
    queries: int = 37741
    keys: int = 37741
    dim: int = 128
    dtype: str = "bfloat16"
    block_m: int = 64
    block_n: int = 64

    def __post_init__(self):
        for name in ("batch", "heads", "queries", "keys", "dim", "block_m", "block_n"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.dtype not in ("float16", "bfloat16"):
            raise ValueError("dtype must be float16 or bfloat16; no implicit conversion")
        if self.dim not in (64, 128):
            raise ValueError("Only head dimensions 64 and 128 are supported")
        # Fixed conservative tiles keep L1/L0C/UB usage within A2/A3 limits.
        if (self.block_m, self.block_n) != (64, 64):
            raise ValueError("This baseline supports block_m=64, block_n=64 only")

    @property
    def padded_queries(self):
        return ((self.queries + self.block_m - 1) // self.block_m) * self.block_m

    @property
    def padded_keys(self):
        return ((self.keys + self.block_n - 1) // self.block_n) * self.block_n

    @property
    def query_tiles(self):
        return self.batch * self.heads * (self.padded_queries // self.block_m)

    def memory_plan(self, cube_cores):
        if isinstance(cube_cores, bool) or not isinstance(cube_cores, int) or cube_cores < 1:
            raise ValueError("cube_cores must be a positive integer from the runtime")
        active = min(cube_cores, self.query_tiles)
        b, h, qp, kp, d = self.batch, self.heads, self.padded_queries, self.padded_keys, self.dim
        # Inputs with already aligned sequence axes are referenced, not copied.
        q_copy = 2 * b * h * qp * d if qp != self.queries else 0
        kv_copy = 4 * b * h * kp * d if kp != self.keys else 0
        output = 2 * b * h * qp * d
        scores = active * self.block_m * self.block_n * 4
        probabilities = active * self.block_m * self.block_n * 2
        partial_output = active * self.block_m * d * 4
        return {
            "active_cube_cores": active,
            "input_padding_allocated_bytes": q_copy + kv_copy,
            "padded_output_bytes": output,
            "score_workspace_bytes": scores,
            "probability_workspace_bytes": probabilities,
            "pv_workspace_bytes": partial_output,
            "workspace_bytes": scores + probabilities + partial_output,
            "prepared_allocated_bytes": q_copy + kv_copy + output + scores + probabilities + partial_output,
            "original_input_bytes": 2 * b * h * (self.queries + 2 * self.keys) * d,
        }


def _backend_provenance(tilelang):
    """Require pin evidence from the installed module, never the caller's cwd."""
    module_dir = Path(tilelang.__file__).resolve().parent
    candidates = []
    checkout = module_dir.parent
    if (checkout / ".git").exists():
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            candidates.append(("module_checkout", commit))
        except (OSError, subprocess.SubprocessError):
            pass
    # Non-PyPI source-built wheels preserve the commit in tilelang/VERSION.
    for version_file in (module_dir / "VERSION", checkout / "VERSION"):
        if version_file.is_file():
            for commit in re.findall(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", version_file.read_text()):
                candidates.append((str(version_file), commit))
    if not candidates or any(commit != SOURCE_COMMIT for _, commit in candidates):
        raise RuntimeError(
            f"Install tile-ai/tilelang-ascend at {SOURCE_COMMIT}; could not verify "
            f"the installed package pin from {module_dir}. Found {candidates!r}. "
            "Use its source checkout or source-built wheel, not upstream pip tilelang."
        )
    from tilelang import language as T
    required = ("gemm_v0", "alloc_L1", "alloc_L0C", "alloc_ub", "Scope", "set_cross_flag", "wait_cross_flag")
    if any(not hasattr(T, name) for name in required):
        raise RuntimeError("Installed TileLang lacks the pinned AscendC A2/A3 language API")
    return {
        "repository": SOURCE_REPOSITORY,
        "required_commit": SOURCE_COMMIT,
        "verified_commit": SOURCE_COMMIT,
        "pin_evidence": candidates,
        "module_path": str(module_dir),
        "version": str(tilelang.__version__),
        "target": "ascendc",
        "compiler_arch": "dav-2201",
        "execution_backend": "cython",
    }


def _runtime_target(torch, device):
    import torch_npu  # noqa: F401; registers the npu device and torch.npu
    if os.environ.get("TL_RUN_MODE", "npu") != "npu":
        raise RuntimeError("This entry point requires TL_RUN_MODE=npu, not simulator execution")
    if not torch.npu.is_available():
        raise RuntimeError("No Ascend NPU is available")
    if device is None:
        index = torch.npu.current_device()
    elif isinstance(device, int) and not isinstance(device, bool):
        index = device
    else:
        resolved = torch.device(device)
        if resolved.type != "npu":
            raise ValueError("device must be an NPU device or integer NPU index")
        index = resolved.index if resolved.index is not None else torch.npu.current_device()
    torch.npu.set_device(index)
    try:
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtGetSocName.argtypes = []
        acl.aclrtGetSocName.restype = ctypes.c_char_p
        raw = acl.aclrtGetSocName()
    except (OSError, AttributeError) as exc:
        raise RuntimeError("Cannot query actual runtime SoC through aclrtGetSocName") from exc
    if not raw:
        raise RuntimeError("aclrtGetSocName returned null")
    soc = raw.decode("utf-8")
    upper = soc.upper()
    if upper.startswith("ASCEND910B"):
        platform = "A2"
    elif upper.startswith(("ASCEND910_93", "ASCEND910C")):
        platform = "A3"
    else:
        raise RuntimeError(f"Unsupported runtime SoC {soc!r}; this kernel requires A2/A3 DAV2201")
    props = torch.npu.get_device_properties(index)
    cube_cores = int(getattr(props, "cube_core_num", 0))
    vector_cores = getattr(props, "vector_core_num", None)
    if cube_cores < 1:
        raise RuntimeError("Runtime did not report cube_core_num; refusing to guess from 910B_23")
    return {
        "device_index": index,
        "runtime_soc": soc,
        "runtime_device_name": torch.npu.get_device_name(index),
        "platform": platform,
        "cube_core_num": cube_cores,
        "vector_core_num": int(vector_cores) if vector_cores is not None else None,
    }


def make_primfunc(spec, active_cores):
    """Construct the pinned fork's genuine AscendC Cube/Vector TileLang IR."""
    from tilelang import language as T
    if not isinstance(spec, AttentionSpec):
        raise TypeError("spec must be AttentionSpec")
    plan = spec.memory_plan(active_cores)
    if plan["active_cube_cores"] != active_cores:
        raise ValueError("active_cores must not exceed the number of query tiles")
    batch, heads, queries, keys, dim = spec.batch, spec.heads, spec.queries, spec.keys, spec.dim
    dtype = spec.dtype
    bm, bn, half = spec.block_m, spec.block_n, spec.block_m // 2
    qp, kp = spec.padded_queries, spec.padded_keys
    qblocks, kblocks = qp // bm, kp // bn
    tasks = spec.query_tiles
    scale = 1.0 / math.sqrt(dim)

    @T.prim_func
    def attention(
        Q: T.Tensor((batch, heads, qp, dim), dtype),
        K: T.Tensor((batch, heads, kp, dim), dtype),
        V: T.Tensor((batch, heads, kp, dim), dtype),
        O: T.Tensor((batch, heads, qp, dim), dtype),
        Scores: T.Tensor((active_cores, bm, bn), "float"),
        Prob: T.Tensor((active_cores, bm, bn), dtype),
        PV: T.Tensor((active_cores, bm, dim), "float"),
    ):
        T.func_attr({"global_symbol": "tilelang_plain_attention_dav2201", "tir.noalias": True})
        with T.Kernel(active_cores, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1((bm, dim), dtype)
            k_l1 = T.alloc_L1((bn, dim), dtype)
            v_l1 = T.alloc_L1((bn, dim), dtype)
            p_l1 = T.alloc_L1((bm, bn), dtype)
            scores_l0c = T.alloc_L0C((bm, bn), "float")
            pv_l0c = T.alloc_L0C((bm, dim), "float")
            scores_ub = T.alloc_ub((half, bn), "float")
            prob_ub = T.alloc_ub((half, bn), dtype)
            out_ub = T.alloc_ub((half, dim), "float")
            pv_ub = T.alloc_ub((half, dim), "float")
            output_ub = T.alloc_ub((half, dim), dtype)
            # Match the official AscendC broadcast examples: one contiguous
            # FP32 scalar per row, represented as [rows, 1]. Reading an
            # individual row_max[i] as a tile-op scalar generates GetValue()
            # and an implicit PIPE_ALL barrier for every row on this backend.
            row_max = T.alloc_ub((half, 1), "float")
            prev_max_alpha = T.alloc_ub((half, 1), "float")
            row_sum = T.alloc_ub((half, 1), "float")
            tile_sum = T.alloc_ub((half, 1), "float")
            score_broadcast = T.alloc_ub((half, bn), "float")
            output_broadcast = T.alloc_ub((half, dim), "float")
            tail_mask = T.alloc_ub((half, bn), "float")

            # Each Cube and its two Vector subblocks process the same sequence
            # of tasks. All cross-core signals are consumed before reuse.
            with T.Scope("C"):
                for task_iter in T.serial(T.ceildiv(tasks - cid, active_cores)):
                    task = task_iter * active_cores + cid
                    qb = task % qblocks
                    h = (task // qblocks) % heads
                    b = task // (qblocks * heads)
                    T.copy(Q[b, h, qb * bm : (qb + 1) * bm, :], q_l1)
                    T.barrier_all()
                    for kb in T.serial(kblocks):
                        T.copy(K[b, h, kb * bn : (kb + 1) * bn, :], k_l1)
                        T.barrier_all()
                        T.gemm_v0(q_l1, k_l1, scores_l0c, transpose_B=True, init=True)
                        T.barrier_all()
                        T.copy(scores_l0c, Scores[cid, :, :])
                        T.barrier_all()
                        T.set_cross_flag("FIX", 0)
                        T.wait_cross_flag(1)
                        T.barrier_all()
                        T.copy(Prob[cid, :, :], p_l1)
                        T.copy(V[b, h, kb * bn : (kb + 1) * bn, :], v_l1)
                        T.barrier_all()
                        T.gemm_v0(p_l1, v_l1, pv_l0c, init=True)
                        T.barrier_all()
                        T.copy(pv_l0c, PV[cid, :, :])
                        T.barrier_all()
                        T.set_cross_flag("FIX", 2)
                        T.wait_cross_flag(3)
                        T.barrier_all()

            with T.Scope("V"):
                # Construct the constant additive mask once per physical
                # Vector subblock, outside *both* attention loops. At this pin
                # T.Parallel with if_then_else can lower to scalar Get/SetValue;
                # it must never scan every score tile in the main KV loop.
                # The hot tail path below is one native vector tile.add.
                if keys != kp:
                    for mask_i, mask_j in T.Parallel(half, bn):
                        tail_mask[mask_i, mask_j] = T.if_then_else(
                            mask_j < keys % bn, 0.0, -T.infinity("float")
                        )
                    T.barrier_all()
                for task_iter in T.serial(T.ceildiv(tasks - cid, active_cores)):
                    task = task_iter * active_cores + cid
                    qb = task % qblocks
                    h = (task // qblocks) % heads
                    b = task // (qblocks * heads)
                    T.tile.fill(out_ub, 0.0)
                    T.tile.fill(row_sum, 0.0)
                    T.tile.fill(row_max, -T.infinity("float"))
                    T.barrier_all()
                    for kb in T.serial(kblocks):
                        T.wait_cross_flag(0)
                        T.copy(Scores[cid, vid * half : (vid + 1) * half, :], scores_ub)
                        T.barrier_all()
                        T.tile.mul(scores_ub, scores_ub, scale)
                        T.barrier_all()
                        # Only the last KV block contains padding. Native
                        # vector add uses the prebuilt [0, ..., -inf] mask;
                        # valid logits and all earlier blocks are unchanged.
                        if keys != kp:
                            if kb == kblocks - 1:
                                T.tile.add(scores_ub, scores_ub, tail_mask)
                                T.barrier_all()
                        T.copy(row_max, prev_max_alpha)
                        T.barrier_all()
                        T.reduce_max(scores_ub, row_max, dim=-1)
                        T.barrier_all()
                        T.tile.max(row_max, row_max, prev_max_alpha)
                        T.barrier_all()
                        T.tile.sub(prev_max_alpha, prev_max_alpha, row_max)
                        T.barrier_all()
                        T.tile.exp(prev_max_alpha, prev_max_alpha)
                        T.barrier_all()
                        T.tile.broadcast(score_broadcast, row_max, axis=1)
                        T.barrier_all()
                        T.tile.sub(scores_ub, scores_ub, score_broadcast)
                        T.barrier_all()
                        T.tile.broadcast(output_broadcast, prev_max_alpha, axis=1)
                        T.barrier_all()
                        T.tile.mul(out_ub, out_ub, output_broadcast)
                        T.barrier_all()
                        T.tile.exp(scores_ub, scores_ub)
                        T.barrier_all()
                        T.reduce_sum(scores_ub, tile_sum, dim=-1)
                        T.barrier_all()
                        T.tile.mul(row_sum, row_sum, prev_max_alpha)
                        T.barrier_all()
                        T.tile.add(row_sum, row_sum, tile_sum)
                        T.barrier_all()
                        T.copy(scores_ub, prob_ub)
                        T.barrier_all()
                        T.copy(prob_ub, Prob[cid, vid * half : (vid + 1) * half, :])
                        T.barrier_all()
                        T.set_cross_flag("MTE3", 1)
                        T.wait_cross_flag(2)
                        T.copy(PV[cid, vid * half : (vid + 1) * half, :], pv_ub)
                        T.barrier_all()
                        T.tile.add(out_ub, out_ub, pv_ub)
                        T.barrier_all()
                        T.set_cross_flag("V", 3)
                        T.barrier_all()
                    T.tile.broadcast(output_broadcast, row_sum, axis=1)
                    T.barrier_all()
                    T.tile.div(out_ub, out_ub, output_broadcast)
                    T.barrier_all()
                    T.copy(out_ub, output_ub)
                    T.barrier_all()
                    T.copy(output_ub, O[b, h, qb * bm + vid * half : qb * bm + (vid + 1) * half, :])
                    T.barrier_all()
    return attention


def _stream_id(torch, index):
    return int(torch.npu.current_stream(index).npu_stream)


@dataclass
class PreparedAttention:
    """Reusable single-stream storage. Tail pads snapshot inputs at prepare()."""
    owner: Any = field(repr=False)
    tensors: tuple = field(repr=False)
    output: Any = field(repr=False)
    stream_id: int

    def metadata(self):
        return {
            **self.owner.spec.memory_plan(self.owner.runtime["cube_core_num"]),
            "stream_id": self.stream_id,
            "concurrent_use_supported": False,
            "output_is_reused_view": True,
            "tail_input_padding_is_snapshot": True,
        }


class AttentionKernel:
    def __init__(self, spec, torch, tilelang, runtime, backend):
        self.spec, self.torch = spec, torch
        self.runtime, self.backend = runtime, backend
        self.active_cores = spec.memory_plan(runtime["cube_core_num"])["active_cube_cores"]
        self.prim_func = make_primfunc(spec, self.active_cores)
        # Manual cross-core synchronization follows the official Expert example.
        # AUTO_SYNC concerns intra-core hazards, not C/V exchange ownership.
        self.pass_configs = {
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        }
        self.kernel = tilelang.compile(
            self.prim_func, out_idx=[], workspace_idx=[], target="ascendc",
            platform=runtime["platform"], execution_backend="cython",
            pass_configs=self.pass_configs,
        )

    def metadata(self):
        return {
            "implementation": "tilelang_native_float_online_attention",
            "implementation_revision": IMPLEMENTATION_REVISION,
            "attention": "dense_noncausal_softmax(QK^T/sqrt(D))V",
            "layout": "BNSD",
            "spec": asdict(self.spec),
            "padded_queries": self.spec.padded_queries,
            "padded_keys": self.spec.padded_keys,
            "scale": 1.0 / math.sqrt(self.spec.dim),
            "qk_pv_dtype": self.spec.dtype,
            "softmax_accumulator_dtype": "float32",
            "probability_rounding": self.spec.dtype,
            "implicit_input_conversion": False,
            "qk_pv_primitive": "T.gemm_v0",
            "row_ops": "whole_tile_fp32_broadcast_sub_mul_div",
            "tail_mask_strategy": "constant_ub_additive_mask_last_kv_block_only",
            "runtime": dict(self.runtime),
            "backend": dict(self.backend),
            "memory": self.spec.memory_plan(self.runtime["cube_core_num"]),
            "pass_configs": {str(k.value): v for k, v in self.pass_configs.items()},
        }

    def _check_inputs(self, q, k, v):
        torch, spec = self.torch, self.spec
        shapes = ((spec.batch, spec.heads, spec.queries, spec.dim),
                  (spec.batch, spec.heads, spec.keys, spec.dim))
        dtype = getattr(torch, spec.dtype)
        for name, tensor, expected in (("Q", q, shapes[0]), ("K", k, shapes[1]), ("V", v, shapes[1])):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have BNSD shape {expected}")
            if tensor.dtype != dtype:
                raise ValueError(f"{name} must have dtype {dtype}; no implicit conversion")
            if tensor.device.type != "npu" or tensor.device.index != self.runtime["device_index"]:
                raise ValueError(f"{name} must be on npu:{self.runtime['device_index']}")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous BNSD")

    def prepare(self, q, k, v):
        self._check_inputs(q, k, v)
        torch, s = self.torch, self.spec
        index = self.runtime["device_index"]
        with torch.npu.device(index):
            def pad(tensor, length):
                if tensor.shape[2] == length:
                    return tensor
                padded = torch.zeros((s.batch, s.heads, length, s.dim), dtype=tensor.dtype, device=tensor.device)
                padded[:, :, :tensor.shape[2], :].copy_(tensor)
                return padded
            qp, kp, vp = pad(q, s.padded_queries), pad(k, s.padded_keys), pad(v, s.padded_keys)
            output = torch.empty_like(qp)
            scores = torch.empty((self.active_cores, s.block_m, s.block_n), dtype=torch.float32, device=q.device)
            probs = torch.empty_like(scores, dtype=q.dtype)
            pv = torch.empty((self.active_cores, s.block_m, s.dim), dtype=torch.float32, device=q.device)
            stream = _stream_id(torch, index)
        return PreparedAttention(self, (qp, kp, vp, output, scores, probs, pv),
                                 output[:, :, :s.queries, :], stream)

    def run_prepared(self, prepared):
        if not isinstance(prepared, PreparedAttention) or prepared.owner is not self:
            raise ValueError("prepared must come from this AttentionKernel.prepare")
        with self.torch.npu.device(self.runtime["device_index"]):
            if _stream_id(self.torch, self.runtime["device_index"]) != prepared.stream_id:
                raise RuntimeError("PreparedAttention must run on its creation stream; prepare separate storage on this stream")
            self.kernel(*prepared.tensors)
        return prepared.output

    def __call__(self, q, k, v):
        return self.run_prepared(self.prepare(q, k, v))

    @property
    def source(self):
        return self.kernel.get_kernel_source()

    def export_source(self, output_dir):
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        paths = {"ascendc": destination / "attention_ascendc.cpp",
                 "tir": destination / "attention.tir",
                 "metadata": destination / "attention_metadata.json"}
        paths["ascendc"].write_text(self.source, encoding="utf-8")
        paths["tir"].write_text(self.prim_func.script(), encoding="utf-8")
        paths["metadata"].write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")
        return {name: str(path) for name, path in paths.items()}


def build_attention(batch=1, heads=28, queries=37741, keys=37741, dim=128,
                    dtype="bfloat16", device=None, block_m=64, block_n=64):
    """Compile native FP16/BF16 attention for the selected actual A2/A3 SoC."""
    spec = AttentionSpec(batch, heads, queries, keys, dim, dtype, block_m, block_n)
    import torch
    runtime = _runtime_target(torch, device)
    import tilelang
    backend = _backend_provenance(tilelang)
    with torch.npu.device(runtime["device_index"]):
        return AttentionKernel(spec, torch, tilelang, runtime, backend)


def cpu_math_self_test():
    """Dependency-free recurrence/tail test; does not run or compile NPU code.

    Float64 Python arithmetic checks the unrounded online-softmax algorithm.
    Device BF16/FP16 probability rounding and compiler synchronization require
    independent NPU checks; this intentionally makes no claim about them.
    """
    import random
    rng = random.Random(1729)
    errors = []
    for dim, queries, keys in ((64, 3, 1), (64, 5, 65), (128, 3, 73)):
        q = [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(queries)]
        k = [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(keys)]
        v = [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(keys)]
        for row in q:
            logits = [sum(a * b for a, b in zip(row, key)) / math.sqrt(dim) for key in k]
            maximum = max(logits)
            weights = [math.exp(x - maximum) for x in logits]
            mass = sum(weights)
            reference = [sum(weights[j] * v[j][d] for j in range(keys)) / mass for d in range(dim)]
            m, l, out = -math.inf, 0.0, [0.0] * dim
            for start in range(0, math.ceil(keys / 64) * 64, 64):
                scores = [logits[j] if j < keys else -math.inf for j in range(start, start + 64)]
                new_max = max(m, max(scores))
                alpha = math.exp(m - new_max)
                p = [math.exp(x - new_max) for x in scores]
                l = alpha * l + sum(p)
                out = [alpha * out[d] + sum(p[j] * v[start + j][d] for j in range(64) if start + j < keys)
                       for d in range(dim)]
                m = new_max
            errors.append(max(abs(out[d] / l - reference[d]) for d in range(dim)))
    if max(errors) > 2e-14:
        raise AssertionError(f"Online softmax/tail recurrence failed: {max(errors)}")
    spec = AttentionSpec()
    return {"kind": "cpu_math_only", "max_abs_error": max(errors), "rows_checked": len(errors),
            "tail_key_lengths": [1, 65, 73], "head_dimensions": [64, 128],
            "default_shape": list(DEFAULT_SHAPE), "memory_plan_for_20_runtime_cores": spec.memory_plan(20),
            "npu_compile_or_run_performed": False}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-check", action="store_true", help="Check recurrence/tail math without torch/CANN/TileLang")
    args = parser.parse_args()
    if not args.cpu_check:
        parser.error("Use --cpu-check locally; use the independent benchmark launcher for NPU execution")
    print(json.dumps(cpu_math_self_test(), indent=2))
