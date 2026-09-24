"""Record CPU/CUDA timelines for the selected VC core and optional comparisons.

Profiler overhead is intentional; use benchmark_cuda.py for latency numbers.
"""
import argparse
import json
from pathlib import Path

import torch

from benchmark_cuda import make_backends, output_hnd, require_batched_runtime
from vc_attention.cuda_backend import preflight


def summarize_trace(path):
    trace = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {}
    for event in trace.get("traceEvents", []):
        category = event.get("cat")
        if category not in ("kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime") or event.get("ph") != "X":
            continue
        name = event.get("name", "unknown")
        row = rows.setdefault((category, name), {"category": category, "name": name, "calls": 0, "total_us": 0.})
        row["calls"] += 1
        row["total_us"] += event.get("dur", 0.)
    return sorted(rows.values(), key=lambda row: row["total_us"], reverse=True)


def summarize_gpu(rows, iterations):
    kernels = [row for row in rows if row["category"] == "kernel"]
    core_us = sum(row["total_us"] for row in kernels if "fused_attention" in row["name"])
    all_us = sum(row["total_us"] for row in kernels)
    return {"core_ms_per_call": core_us / (1000 * iterations),
            "other_kernels_ms_per_call": (all_us - core_us) / (1000 * iterations),
            "kernel_sum_ms_per_call": all_us / (1000 * iterations),
            "core_percent_of_kernel_sum": 100 * core_us / all_us if all_us else 0,
            "kernel_launches_per_call": sum(row["calls"] for row in kernels) / iterations}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--vc-schedule", choices=("fused", "batched"), default="fused")
    parser.add_argument("--vc-schedule-compare", action="store_true", help="also profile the expensive legacy serial path")
    parser.add_argument("--vc-fused-compare", action="store_true", help="also profile the retained 0.4.0 fused core")
    parser.add_argument("--vc-specialize-compare", action="store_true", help="also profile the retained 0.5.0 generic pipeline")
    parser.add_argument("--vc-interleave-compare", action="store_true", help="also profile the retained 0.6.0 D128 schedule")
    parser.add_argument("--vc-reuse-layout", action="store_true")
    parser.add_argument("--output-dir", default="results/h200_profile")
    args = parser.parse_args(argv)
    if min(args.batch, args.heads, args.queries, args.tokens, args.warmup, args.iterations) < 1:
        parser.error("shapes, warmup and iterations must be positive")
    if args.vc_fused_compare and args.vc_schedule != "fused":
        parser.error("--vc-fused-compare requires --vc-schedule fused")
    if args.vc_specialize_compare and (args.vc_schedule != "fused" or args.dim != 128):
        parser.error("--vc-specialize-compare requires --vc-schedule fused and --dim 128")
    if args.vc_interleave_compare and (args.vc_schedule != "fused" or args.dim != 128):
        parser.error("--vc-interleave-compare requires --vc-schedule fused and --dim 128")
    return args


@torch.no_grad()
def run(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runtime = require_batched_runtime(args.vc_schedule)
    info = preflight(device, schedule=args.vc_schedule)
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError("CUDA profiling/CUPTI is unavailable in this PyTorch environment")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    qkv = tuple(torch.randn(args.batch, args.heads, n, args.dim, device=device, dtype=dtype)
                for n in (args.queries, args.tokens, args.tokens))
    backends = make_backends(["vc"], qkv, False, vc_schedule_compare=args.vc_schedule_compare,
                             vc_reuse_layout=args.vc_reuse_layout, vc_schedule=args.vc_schedule,
                             vc_fused_compare=args.vc_fused_compare,
                             vc_specialize_compare=args.vc_specialize_compare,
                             vc_interleave_compare=args.vc_interleave_compare)
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = {"environment": {"torch": torch.__version__, "torch_cuda": torch.version.cuda, **info, "runtime": runtime},
              "arguments": vars(args), "results": {},
              "notes": ["Profiling adds overhead. Use benchmark_cuda.py for latency/speedup.",
                        "Kernel and CUDA runtime durations are separate; never add CPU and GPU times.",
                        "Reuse rows alone exclude initial clustering setup; all rows recompute current tensor values."]}
    for backend in backends:
        print(f"Profiling {backend.name}", flush=True)
        for _ in range(args.warmup):
            backend.call()
        torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as profile:
            for iteration in range(args.iterations):
                with torch.profiler.record_function(f"{backend.name}/call_{iteration}"):
                    result = backend.call()
            torch.cuda.synchronize(device)
        # Shape/finite checks occur after the profiled region.
        output_hnd(result, backend.layout, qkv[0].shape, dtype)
        trace_path = directory / f"{backend.name}.trace.json"
        profile.export_chrome_trace(str(trace_path))
        rows = summarize_trace(trace_path)
        if not any(row["category"] == "kernel" for row in rows):
            raise RuntimeError(f"{trace_path}: no GPU kernel events captured; check CUPTI availability")
        table = profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
        (directory / f"{backend.name}.operators.txt").write_text(table, encoding="utf-8")
        summary = summarize_gpu(rows, args.iterations)
        report["results"][backend.name] = {**backend.info, "trace": str(trace_path), "events": rows, "gpu_summary": summary}
        print(f"{backend.name} GPU kernel summary: " + json.dumps(summary), flush=True)
        print("CUDA kernels: calls/call, summed GPU ms/call (profiled)")
        for row in [x for x in rows if x["category"] == "kernel"][:12]:
            print(f"{row['calls'] / args.iterations:8.1f} {row['total_us'] / (1000 * args.iterations):10.4f} {row['name']}")
        print(f"Trace: {trace_path}", flush=True)
    path = directory / "summary.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Profile report: {path}; use benchmark_cuda.py for speed comparisons.")
    return report


if __name__ == "__main__":
    run(parse_args())
