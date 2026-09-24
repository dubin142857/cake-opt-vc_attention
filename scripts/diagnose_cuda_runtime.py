"""Locate a slow VC call: one full-length forward, synchronized phase timings.

This reproduces the Python CUDA wrapper's stages explicitly for diagnosis.
The first call includes setup; it is not a steady-state performance comparison.
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from benchmark_cuda import input_shapes, require_batched_runtime


def operation_counts(args):
    # Current default CUDA geometry: Q/KV=128, Q chunk <=1024, all heads batched.
    qtiles = math.ceil(args.queries / 128)
    ktiles = math.ceil(args.tokens / 128)
    chunks = math.ceil(qtiles / 8)
    return {"q_chunks": chunks, "kv_tiles": ktiles,
            "batched_gemm_calls": 2 * chunks * ktiles,
            "batched_update_kernel_launches": 2 * chunks * ktiles,
            "legacy_serial_gemm_calls": 2 * args.batch * args.heads * qtiles * ktiles}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--queries", type=int, default=37741)
    parser.add_argument("--tokens", type=int, default=37741)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-v-smooth", action="store_true", help="explicitly skip clustering and V centering for a separate diagnostic")
    parser.add_argument("--output", default="results/h200_vc_runtime.json")
    args = parser.parse_args(argv)
    if min(args.batch, args.queries, args.tokens, args.heads) < 1:
        parser.error("shapes must be positive")
    return args


@torch.no_grad()
def run(args):
    runtime = require_batched_runtime()
    from vc_attention import Config, LayoutCache
    from vc_attention.core import _validate
    from vc_attention.cuda_backend import _extension, preflight
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = Config(backend="cuda_fp8", expcast=True, v_smooth=not args.no_v_smooth, cuda_schedule="batched")
    report = {"runtime": runtime, "arguments": vars(args), "input_shapes": input_shapes(args),
              "call_counts_from_code": operation_counts(args), "phases": {},
              "notes": ["Single staged forward, first-use costs included. This is not a benchmark speedup.",
                        "Every phase synchronizes for diagnosis; totals need not equal the uninstrumented API.",
                        "Native forward includes Q/K transforms, quantization, allocation, QK/PV GEMMs and online updates.",
                        "Call counts are derived from current default geometry, not profiler measurements."]}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def phase(name, fn):
        print(f"START {name}", flush=True)
        report["active_phase"] = name
        save()  # Partial report remains available even if a later phase fails.
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        value = fn()
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - begin
        report["phases"][name] = {"wall_seconds": seconds,
                                  "allocated_mib": torch.cuda.memory_allocated(device) / (1024**2)}
        report["active_phase"] = None
        save()
        print(f"DONE  {name}: {seconds:.3f} s", flush=True)
        return value

    print(json.dumps({"input_shapes": report["input_shapes"], "call_counts": report["call_counts_from_code"]}, indent=2), flush=True)
    report["preflight"] = phase("small_native_fp8_preflight", lambda: preflight(device))
    ext = _extension()
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    q, k, v = phase("allocate_QKV_on_GPU", lambda: tuple(
        torch.randn(args.batch, args.heads, n, args.dim, device=device, dtype=dtype)
        for n in (args.queries, args.tokens, args.tokens)))
    print(f"Q/K/V devices: {q.device}, {k.device}, {v.device}", flush=True)
    phase("finite_input_validation", lambda: _validate(q, k, v, False, None, native_finite=True))
    vf = phase("V_to_FP32", lambda: v.float())
    cache = LayoutCache(cfg)
    pi, active, refreshed = phase("V_clustering", lambda: cache.layout(
        vf, step=0, total_steps=1, request_id="runtime-diagnostic", layer_id=0))
    del vf
    report["layout"] = {"smoothing_active": active, "refreshed": refreshed}
    if pi is not None:
        k, v = phase("KV_permutation", lambda: (
            k.gather(-2, pi.unsqueeze(-1).expand_as(k)),
            v.gather(-2, pi.unsqueeze(-1).expand_as(v))))
    output = phase("native_batched_CUDA_forward", lambda: ext.forward(
        q, k, v, cfg.q_block, cfg.k_quant_block, cfg.kv_block, active, cfg.expcast,
        {"float32": 0, "float16": 1, "bfloat16": 2}[cfg.mean_dtype], args.dim**-0.5,
        cfg.k_smooth, cfg.qk_hadamard, True))
    finite = phase("output_finite_check", lambda: bool(torch.isfinite(output).all()))
    report["output"] = {"shape": list(output.shape), "device": str(output.device), "finite": finite}
    save()
    if not finite:
        raise RuntimeError("Native output contains NaN/Inf")
    print(f"Finished one full-size native call. Report: {path}", flush=True)
    return report


if __name__ == "__main__":
    run(parse_args())
