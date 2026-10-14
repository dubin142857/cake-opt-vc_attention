"""Explicit MindIE-SD Laser baseline. No automatic attention fallback.

API inspected at Ascend/MindIE-SD 71098c20ff90838d1fd1ca0a40515c751a3869b0.
Uses the installed upstream preprocessing/postprocessing, not a copied kernel.
"""
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import math
from pathlib import Path
import torch


def distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_npu(q):
    if q.device.type != "npu":
        raise ValueError("MindIE Laser benchmark requires NPU inputs")


@dataclass
class LaserPayload:
    tensors: tuple
    dtype: torch.dtype
    shape: tuple
    scale: float
    pre_tokens: int


class MindIELaser:
    def __init__(self, preprocess, postprocess, native, metadata):
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.native = native
        self.metadata = metadata

    def prepare(self, q, k, v):
        _require_npu(q)
        if any(x.ndim != 4 for x in (q, k, v)):
            raise ValueError("Laser adapter expects HND/BNSD rank-four tensors")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("Laser adapter supports FP16/BF16 inputs")
        if any(x.dtype != q.dtype or x.device != q.device for x in (k, v)):
            raise ValueError("Q/K/V dtype and device must match")
        if k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
            raise ValueError("Dense MHA requires matching B/H/D and K/V shapes")
        if min(*q.shape, *k.shape) <= 0 or q.shape[-1] not in (64, 128):
            raise ValueError("Nonempty D64/128 inputs required")
        tensors = tuple(self.preprocess(q, k, v))
        b, h, nq, d = q.shape
        nk = k.shape[2]
        expected = [(b, h, ((n + 255) // 256) * 256, 128) for n in (nq, nk, nk)]
        if len(tensors) != 3 or any(tuple(x.shape) != shape or x.dtype != torch.float16
                                    or x.device != q.device or not x.is_contiguous()
                                    for x, shape in zip(tensors, expected)):
            raise RuntimeError("Installed MindIE Laser preprocessing changed; expected contiguous FP16 S256/D128 padding")
        # Laser's pre_tokens is tail-padding length here, NOT CANN's causal window.
        padding = tensors[1].shape[2] - nk
        return LaserPayload(tensors, q.dtype, tuple(q.shape), 1 / math.sqrt(d),
                            padding if padding else 2147483647)

    def run_prepared(self, payload):
        q, k, v = payload.tensors
        result = self.native(q, k, v, None, None, None, payload.scale,
                             payload.shape[1], "BNSD", 1.0, payload.pre_tokens, 1, True)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError("mindiesd::la must return (softmax statistics, attention output)")
        # Output is the SECOND tensor. Upstream handles BF16 round-trip and trim.
        out = self.postprocess(result[1], payload.dtype, payload.shape[2], payload.shape[3])
        if tuple(out.shape) != payload.shape or out.dtype != payload.dtype or out.device != q.device:
            raise RuntimeError("MindIE Laser output shape/dtype/device differs from the requested contract")
        return out.contiguous()  # same contiguous HND output contract as VC

    def __call__(self, q, k, v):
        return self.run_prepared(self.prepare(q, k, v))

    def describe_payload(self, payload):
        return {"input_shape_hnd": list(payload.shape), "input_dtype": str(payload.dtype),
                "native_shapes": [list(x.shape) for x in payload.tensors],
                "native_dtype": str(payload.tensors[0].dtype), "scale": payload.scale,
                "pre_tokens": payload.pre_tokens, "next_tokens": 1,
                "is_high_precision": True, "causal": False, "quantized": False,
                "fixed_core_scope": "preconverted/padded QKV; includes allocation, native dispatch, output cast/trim/contiguous"}


def load_mindie_laser():
    """Import before first NPU workload so MindIE can set its custom OPP paths."""
    try:
        package = importlib.import_module("mindiesd")
        module = importlib.import_module("mindiesd.layers.flash_attn.ascend_laser_attention")
        cls = module.AscendLaserAttention
        native = torch.ops.mindiesd.la.default
        schema = native._schema
        expected = ["query", "key", "value", "atten_mask", "alibi_mask", "drop_mask",
                    "scale_value", "head_num", "input_layout", "keep_prob", "pre_tokens", "next_tokens"]
        names = [a.name for a in schema.arguments]
        if len(names) != 13 or names[:12] != expected or names[-1] not in ("is_highPrecision", "is_high_precision") or len(schema.returns) != 2:
            raise RuntimeError(f"Unrecognized mindiesd::la schema: {schema}")
        source = Path(module.__file__).resolve()
        metadata = {"package": "mindiesd", "version": distribution_version("mindiesd") or getattr(package, "__version__", "unknown"),
                    "package_path": str(Path(package.__file__).resolve()), "wrapper_source": str(source),
                    "wrapper_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "native_schema": str(schema), "dispatch": "torch.ops.mindiesd.la.default",
                    "auto_dispatch": False, "public_attention_forward": False,
                    "upstream_repository": "https://github.com/Ascend/MindIE-SD",
                    "inspected_commit": "71098c20ff90838d1fd1ca0a40515c751a3869b0",
                    "loaded_libraries": sorted(str(x) for x in torch.ops.loaded_libraries if "mindie" in str(x).lower() or "PTAExtension" in str(x)),
                    "full_scope": "installed Laser preprocess + explicit la + installed postprocess + contiguous output"}
        return MindIELaser(cls.la_preprocess_input, cls.la_postprocess_output, native, metadata)
    except Exception as exc:
        raise RuntimeError("MindIE Laser unavailable. Install MindIE-SD (Python package mindiesd) with its LaserAttention native ops; "
                           "MindIE-LLM alone is insufficient. Match the target CANN/torch_npu versions. "
                           f"Original error: {type(exc).__name__}: {exc}") from exc


@torch.no_grad()
def validate_laser(adapter, device, dense_reference, error):
    """Real-NPU tail/scale/dtype checks; all keys, sampled queries, outside timing."""
    results = []
    generator = torch.Generator().manual_seed(91024)
    for dtype in (torch.float16, torch.bfloat16):
        for nq, nk, d in ((256, 256, 128), (257, 385, 64)):
            original = [torch.randn((1, 2, n, d), generator=generator).to(dtype) for n in (nq, nk, nk)]
            original[2][..., 0] = 0
            original[2][..., 1] = 2
            q, k, v = [x.to(device) for x in original]
            print(f"Validate MindIE Laser: {dtype}, Q={nq}, K={nk}, D={d}", flush=True)
            prepared = adapter.prepare(q, k, v)
            out = adapter.run_prepared(prepared).cpu()
            full = adapter(q, k, v).cpu()
            consistency = error(out, full)
            if consistency["relative_rmse"] > .001:
                raise AssertionError(f"Laser full/prepared inconsistency: {consistency}")
            ids = torch.tensor([0, nq // 2, nq - 1])
            dense = dense_reference(*original, query_indices=ids)
            metric = error(out.index_select(2, ids), dense)
            if metric["relative_rmse"] > .01:
                raise AssertionError(f"Laser dense-reference implementation check failed: {metric}")
            torch.testing.assert_close(out[..., 0], torch.zeros_like(out[..., 0]), rtol=0, atol=0)
            torch.testing.assert_close(out[..., 1], torch.full_like(out[..., 1], 2), rtol=.004, atol=.003)
            results.append({**adapter.describe_payload(prepared), "dense_fp32": metric, "full_vs_prepared": consistency})
    return results
