"""One warmed full-size call inside a CUDA profiler range for Nsight Compute.

Use ncu --profile-from-start off and a fused_attention kernel filter. Preflight,
allocation and warmup happen before cudaProfilerStart, avoiding small-test captures.
Hardware counters/replay timing do not replace benchmark_cuda.py API latency.
"""
import argparse
import json

import torch

from benchmark_cuda import require_batched_runtime
from vc_attention import Config, attention
from vc_attention.cuda_backend import preflight


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variant", choices=("interleaved", "d128"), default="interleaved")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--queries", type=int, default=37741)
    parser.add_argument("--tokens", type=int, default=37741)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    if args.queries < 64 or min(args.batch, args.heads, args.tokens, args.warmup) < 1:
        parser.error("queries must be >=64; batch/heads/tokens/warmup must be positive")
    return args


@torch.no_grad()
def run(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    require_batched_runtime("fused")
    preflight(device, schedule="fused")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    qkv = tuple(torch.randn(args.batch, args.heads, n, 128, device=device, dtype=getattr(torch, args.dtype))
                for n in (args.queries, args.tokens, args.tokens))
    cfg = Config(backend="cuda_fp8", cuda_schedule="fused", expcast=True, v_smooth=True,
                 cuda_fused_interleave=args.variant == "interleaved")
    for _ in range(args.warmup):
        attention(*qkv, cfg)
    torch.cuda.synchronize(device)
    print(f"Capturing one {args.variant} call; BHND Q={list(qkv[0].shape)}, K/V={list(qkv[1].shape)}", flush=True)
    torch.cuda.profiler.start()
    try:
        output, stats = attention(*qkv, cfg)
        torch.cuda.synchronize(device)
    finally:
        torch.cuda.profiler.stop()
    assert stats["cuda_fused_variant"] == args.variant
    if not torch.isfinite(output).all().item():
        raise RuntimeError("Profiled output contains NaN/Inf")
    print(json.dumps({"arguments": vars(args), "stats": stats, "finite": True}, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
