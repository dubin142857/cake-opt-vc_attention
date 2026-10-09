"""CUPTI launch durations, isolated from Python/API timing and preprocessing.

Each profiled call is synchronized inside a named CPU range so all GPU work,
including auxiliary FA4 kernels, can be attributed without guessing launch counts.
No CUDA event or API time is substituted when CUPTI is unavailable.
"""
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch


def summarize_trace(events, expected_calls, markers):
    if expected_calls < 1:
        raise RuntimeError("At least one profiled call is required")
    ranges = sorted((e for e in events if e.get("ph") == "X" and
                     e.get("cat") == "user_annotation" and
                     e.get("name", "").startswith("vc31_call_")), key=lambda e: e["ts"])
    if len(ranges) != expected_calls:
        raise RuntimeError(f"Expected {expected_calls} call ranges, captured {len(ranges)}")
    gpu = [e for e in events if e.get("ph") == "X" and
           e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    calls, used = [], set()
    for span in ranges:
        begin, end = span["ts"], span["ts"] + span["dur"]
        work = [(i, e) for i, e in enumerate(gpu)
                if e["ts"] >= begin and e["ts"] + e["dur"] <= end]
        if any(i in used for i, _ in work):
            raise RuntimeError("Overlapping profiler call ranges")
        used.update(i for i, _ in work)
        kernels = [e for _, e in work if e["cat"] == "kernel"]
        attention = [e for e in kernels if any(m.lower() in e.get("name", "").lower() for m in markers)]
        if not attention:
            names = [e.get("name") for e in kernels]
            raise RuntimeError(f"Missing attention GPU activity in {span['name']}: {names}; check CUPTI or kernel names")
        if not all(math.isfinite(e["dur"]) and e["dur"] > 0 for _, e in work):
            raise RuntimeError("Nonpositive/nonfinite CUPTI duration")
        calls.append({
            "attention_ms": sum(e["dur"] for e in attention) / 1000,
            "all_kernel_ms": sum(e["dur"] for e in kernels) / 1000,
            "all_gpu_work_ms": sum(e["dur"] for _, e in work) / 1000,
            "attention_launches": len(attention), "kernel_launches": len(kernels),
            "kernels": [{"name": e["name"], "ms": e["dur"] / 1000, "resources": e.get("args", {})}
                        for e in kernels],
        })
    if len(used) != len(gpu):
        raise RuntimeError(f"{len(gpu)-len(used)} GPU activities fell outside synchronized call ranges; refusing partial timing")
    med = lambda key: statistics.median(c[key] for c in calls)
    return {"attention_median_ms": med("attention_ms"),
            "all_kernel_median_ms": med("all_kernel_ms"),
            "all_gpu_work_median_ms": med("all_gpu_work_ms"),
            "single_attention_launch": all(c["attention_launches"] == 1 for c in calls),
            "calls": calls}


def run_window(call, seconds, device, *, annotate=False):
    torch.cuda.synchronize(device)
    start, count, output = time.perf_counter(), 0, None
    while count == 0 or time.perf_counter() - start < seconds:
        if annotate:
            with torch.profiler.record_function(f"vc31_call_{count}"):
                output = call()
                torch.cuda.synchronize(device)
        else:
            output = call()
            torch.cuda.synchronize(device)
        count += 1
    # Keep the last result alive until all submitted GPU work has finished.
    return count, time.perf_counter() - start


@torch.no_grad()
def profile_kernels(calls, args, device, on_result=None, kernel_markers=None):
    if not calls:
        raise ValueError("No kernel backends selected")
    directory = Path(args.trace_dir or args.output.with_suffix("")) / "kernel_windows"
    directory.mkdir(parents=True, exist_ok=True)
    rng, order_log = random.Random(args.seed + 3100), []
    report = {"variants": {name: [] for name in calls}, "order": order_log,
              "warmup_seconds": args.kernel_warmup_seconds, "capture_seconds": args.kernel_seconds,
              "scope": "CUPTI activity with synchronized isolated calls; preprocessing/JIT outside. "
                       "Per-call attention, all-kernel and all-GPU-work durations kept separately. "
                       "API allocation/dispatch and stream gaps are not counted. No CUDA graph.",
              "status": "running"}
    for trial in range(args.kernel_trials):
        order = list(calls); rng.shuffle(order); order_log.append(order)
        for name in order:
            print(f"Kernel {trial+1}/{args.kernel_trials}: {name}; warmup {args.kernel_warmup_seconds:g}s, "
                  f"CUPTI window {args.kernel_seconds:g}s", flush=True)
            call = calls[name]
            run_window(call, args.kernel_warmup_seconds, device)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                count, elapsed = run_window(call, args.kernel_seconds, device, annotate=True)
            path = directory / f"{name}.{trial}.json"
            prof.export_chrome_trace(str(path))
            markers = (("qk_int8_sv_f8_attn_kernel",) if name == "sage_cpp" else
                       ("flash", "fmha", "attention") if name == "flash4" else
                       ("sage_vc_attention_kernel",))
            if kernel_markers and name in kernel_markers:
                markers = kernel_markers[name]
            row = summarize_trace(json.loads(path.read_text(encoding="utf-8"))["traceEvents"], count, markers)
            if name != "flash4" and any(c["kernel_launches"] != 1 for c in row["calls"]):
                raise RuntimeError(f"{name} prepared API launched unexpected auxiliary kernels")
            row.update(trace=str(path), capture_wall_seconds=elapsed, launches=count)
            report["variants"][name].append(row)
            print(f"  {name}: attention {row['attention_median_ms']:.4f} ms; "
                  f"all kernels {row['all_kernel_median_ms']:.4f} ms; {count} calls", flush=True)
            if on_result:
                on_result(report)
    report["medians"] = {
        name: {key: statistics.median(t[key] for t in trials)
               for key in ("attention_median_ms", "all_kernel_median_ms", "all_gpu_work_median_ms")}
        for name, trials in report["variants"].items()}
    report["aggregation"] = "Median of window medians; raw per-call activities retained"
    report["status"] = "passed"
    if on_result:
        on_result(report)
    print("CUPTI GPU durations only (ms); separate from the API table:")
    print(f"{'variant':<26} {'attention':>12} {'all kernels':>12} {'all GPU work':>13} {'Sage/kernel':>13} {'FA4/kernel':>12}")
    for name, row in report["medians"].items():
        ms = row["all_kernel_median_ms"]
        ratios = [f"{report['medians'][base]['all_kernel_median_ms']/ms:.3f}x" if base in report['medians'] else "--"
                  for base in ("sage_cpp", "flash4")]
        print(f"{name:<26} {row['attention_median_ms']:>12.4f} {ms:>12.4f} {row['all_gpu_work_median_ms']:>13.4f} {ratios[0]:>13} {ratios[1]:>12}")
    print("Ratios use ALL kernel durations per call, including auxiliary kernels; no API-wall/ GPU-time mixing.")
    return report
