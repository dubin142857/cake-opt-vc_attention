"""Focused H200 diagnosis: INT8 QK + FP8 PV; V smoothing/grouping always OFF.

Compare original Sage, matched vc28 control/ExpCast, and optional FA4. Keep
uninstrumented API timings, CUPTI stage/kernel durations and sampled clock64
intervals separate. No ncu, nvdisasm, download, or package rebuild is required.
--kernel-phases builds a small diagnostic extension with the installed nvcc.
"""
import argparse
import json
import math
from pathlib import Path
import random
import statistics
from types import SimpleNamespace

import torch

from benchmark_cuda import measure_trial, query_indices, fp32_reference, output_hnd
from profile_int8_h200 import load_inputs, require_int8_state, metrics, make_flash4
from profile_sage_vc import (load_ops, load_sage_core_ops, sage_core_calls, staged_prepare,
                            assert_payload_equal, DirectStages, EventStages, stats)
from kernel_timing import profile_kernels, run_window
from attention_phase_probe import load_probe, summarize_clocks, PHASES

MODE = "vc28_stream_i2f"
STAGES = ("k_mean", "qk_int8", "v_pad", "v_quantize", "attention")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in (("batch", 1), ("heads", 28), ("queries", 37741), ("tokens", 37741),
                          ("trials", 5), ("repeats", 10), ("check-queries", 128), ("seed", 0),
                          ("kernel-trials", 3), ("phase-trials", 3),
                          ("phase-cta-stride", 64), ("phase-tile-stride", 8)):
        p.add_argument("--"+name, type=int, default=default)
    p.add_argument("--dim", type=int, choices=(64, 128), default=128)
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--qkv-file", type=Path)
    p.add_argument("--qkv-layout", choices=("NHD", "HND"), default="NHD")
    p.add_argument("--warmup-seconds", type=float, default=10.)
    p.add_argument("--kernel-warmup-seconds", type=float, default=2.)
    p.add_argument("--kernel-seconds", type=float, default=1.)
    p.add_argument("--skip-flash4", action="store_true")
    p.add_argument("--skip-cupti", action="store_true", help="Explicitly omit CUPTI; never substitute event timing")
    p.add_argument("--kernel-phases", action="store_true", help="Compile/validate clock64 probe of the actual vc28 loop")
    p.add_argument("--trace-dir", type=Path, default=Path("results/h200_expcast_focus_traces"))
    p.add_argument("--output", type=Path, default=Path("results/h200_expcast_focus.json"))
    a = p.parse_args(argv)
    for name in ("batch", "heads", "queries", "tokens", "trials", "repeats", "kernel_trials",
                 "phase_trials", "phase_cta_stride", "phase_tile_stride"):
        if getattr(a, name) < 1:
            p.error(name + " must be positive")
    if a.check_queries < 0:
        p.error("check-queries must be nonnegative; 0 checks all rows")
    for name in ("warmup_seconds", "kernel_warmup_seconds", "kernel_seconds"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            p.error(name + " must be positive and finite")
    return a


def make_calls(vc, sage_api, native, qkv, base):
    require_int8_state(base)
    if base["block_values"] or base["means"].numel() or base["permutation"] is not None:
        raise ValueError("Focused diagnosis forbids V smoothing/grouping")
    states = {name: {**base, "expcast": cast} for name, cast in (("control", False), ("expcast", True))}
    core = {name: (lambda s=s: vc.forward_prepared(s, kernel_mode=MODE)) for name, s in states.items()}
    full = {name: (lambda cast=s["expcast"]: vc.sageattn_vc(
        *qkv, tensor_layout="HND", expcast=cast, v_smooth=False, kernel_mode=MODE))
        for name, s in states.items()}
    core["sage_cpp"] = sage_core_calls(SimpleNamespace(vc=vc), base, native, native)["sage_cpp"]
    full["sage_cpp"] = lambda: sage_api(*qkv, tensor_layout="HND", is_causal=False)
    return states, core, full


def staged_call(qkv, ops, recorder, cast):
    state = staged_prepare(*qkv, ops, recorder, expcast=cast, smooth=False)
    return recorder.call("attention", lambda: ops.vc.forward_prepared(state, kernel_mode=MODE))


class CuptiStages:
    """Synchronize ONLY diagnostic capture so stage ownership is unambiguous."""
    def __init__(self, device):
        self.device = device

    def call(self, name, fn):
        torch.cuda.synchronize(self.device)
        with torch.profiler.record_function("expcast_stage/"+name):
            value = fn()
            torch.cuda.synchronize(self.device)
        return value


def summarize_stage_trace(events, expected):
    ranges = [e for e in events if e.get("ph")=="X" and e.get("cat")=="user_annotation"
              and e.get("name", "").startswith("expcast_stage/")]
    if sorted(e["name"].split("/")[-1] for e in ranges) != sorted(expected):
        raise RuntimeError("Missing/duplicated CUPTI stage ranges")
    work = [e for e in events if e.get("ph")=="X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    used, result = set(), {}
    for span in ranges:
        indices = {i for i, e in enumerate(work) if e["ts"] >= span["ts"] and
                   e["ts"]+e["dur"] <= span["ts"]+span["dur"]}
        if used & indices:
            raise RuntimeError("Overlapping stage ranges")
        used |= indices
        kernels = [work[i] for i in sorted(indices) if work[i]["cat"]=="kernel"]
        if not kernels or any(not math.isfinite(work[i]["dur"]) or work[i]["dur"]<=0 for i in indices):
            raise RuntimeError("Missing/nonpositive GPU activity for " + span["name"])
        result[span["name"].split("/")[-1]] = {
            "kernel_ms": sum(e["dur"] for e in kernels)/1000,
            "gpu_work_ms": sum(work[i]["dur"] for i in indices)/1000,
            "launches": len(kernels), "kernels": [
                {"name": e["name"], "ms": e["dur"]/1000, "resources": e.get("args", {})} for e in kernels]}
    if len(used) != len(work):
        raise RuntimeError("GPU work outside stage ranges; refusing partial attribution")
    return result


def expcast_savings(samples):
    """Paired trial deltas; positive means ExpCast saved time. No phase attribution."""
    control, cast = samples["control"], samples["expcast"]
    if len(control) != len(cast) or not control:
        raise ValueError("Expected paired nonempty control/ExpCast trials")
    deltas = [a-b for a,b in zip(control,cast)]
    return {"paired_trial_saved_ms": deltas, "median_saved_ms": statistics.median(deltas),
            "min_saved_ms": min(deltas), "max_saved_ms": max(deltas),
            "ratio_of_medians": statistics.median(control)/statistics.median(cast)}


def capture_stages(qkv, ops, cast, args, device, expected, trial):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        out = staged_call(qkv, ops, CuptiStages(device), cast)
    path = args.trace_dir / f"stages_{'expcast' if cast else 'control'}.{trial}.json"
    prof.export_chrome_trace(str(path))
    summary = summarize_stage_trace(json.loads(path.read_text(encoding="utf-8"))["traceEvents"], expected)
    return {"stages": summary, "trace": str(path)}


def probe_call(ext, state, trace):
    return lambda: ext.forward(*(state[k] for k in ("q", "k", "v", "qs", "ks", "vs",
                                                   "output_type", "scale", "expcast")), trace)


@torch.no_grad()
def validate_probe(ext, vc, device):
    """Full-output parity at both dtypes/dimensions and masked/aligned lengths."""
    rows = []
    generator = torch.Generator(device=device).manual_seed(3300)
    for dtype in (torch.float16, torch.bfloat16):
        for dim in (64, 128):
            for m, n in ((1, 1), (65, 128), (65, 129), (193, 513)):
                q = torch.randn(1, 2, m, dim, dtype=dtype, device=device, generator=generator)
                k = torch.randn(1, 2, n, dim, dtype=dtype, device=device, generator=generator)
                v = torch.randn(k.shape, dtype=dtype, device=device, generator=generator)
                base = vc.prepare(q, k, v, v_smooth=False)
                for cast in (False, True):
                    state = {**base, "expcast": cast}
                    expected = vc.forward_prepared(state, kernel_mode=MODE)
                    row = {"dtype": str(dtype), "dim": dim, "queries": m, "keys": n, "expcast": cast}
                    for trace in (False, True):
                        out, clocks = probe_call(ext, state, trace)()
                        row["probe" if trace else "plain"] = metrics(out, expected, limit=.001)
                        if trace:
                            # Validate every timestamp, including final masked tiles.
                            summarize_clocks(clocks.cpu(), n, ext.phase_tile_stride)
                    rows.append(row)
    return rows


@torch.no_grad()
def phase_diagnosis(states, core, vc, args, device, save):
    ext, build = load_probe(args.phase_cta_stride, args.phase_tile_stride)
    ext.phase_tile_stride = args.phase_tile_stride
    result = {"build": build, "status": "validating"}
    save(result)
    result["validation"] = validate_probe(ext, vc, device)
    result["resources"] = ext.resources(states["control"]["q"].shape[-1], states["control"]["output_type"])
    calls = {name+suffix: probe_call(ext, state, trace) for name, state in states.items()
             for suffix, trace in (("_plain", False), ("_probe", True))}
    # Whole-workload parity and current-stream behavior before using any clocks.
    result["accuracy"] = {}
    for name, state in states.items():
        expected = core[name]()
        for suffix in ("_plain", "_probe"):
            out, clocks = calls[name+suffix]()
            result["accuracy"][name+suffix] = {**metrics(out, expected, limit=.001),
                                               "exact": torch.equal(out, expected)}
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            side_out, side_clocks = calls[name+"_probe"]()
        stream.synchronize()
        metrics(side_out, expected, limit=.001)
        summarize_clocks(side_clocks.cpu(), state["k"].shape[2], args.phase_tile_stride)
    result.update(status="measuring", samples={name: [] for name in calls}, clocks={name: [] for name in states})
    save(result)
    for call in calls.values():
        run_window(call, args.kernel_warmup_seconds, device)
    rng = random.Random(args.seed+3300)
    for trial in range(args.phase_trials):
        order = list(calls); rng.shuffle(order)
        for name in order:
            (out, clocks), elapsed = measure_trial(calls[name], args.repeats, device)
            result["samples"][name].append(elapsed)
            if name.endswith("_probe"):
                base = name.removesuffix("_probe")
                summary = summarize_clocks(clocks.cpu(), states[base]["k"].shape[2], args.phase_tile_stride)
                raw_path = args.trace_dir / f"{base}.clocks.{trial}.pt"
                torch.save(clocks.cpu(), raw_path)
                summary["raw_timestamps"] = str(raw_path)
                result["clocks"][base].append(summary)
        save(result)
        print(f"Kernel phase trial {trial+1}/{args.phase_trials} complete", flush=True)
    result["timings"] = {name: stats(values) for name, values in result["samples"].items()}
    result["perturbation"] = {}
    for name in states:
        plain = result["timings"][name+"_plain"]["median_ms"]
        traced = result["timings"][name+"_probe"]["median_ms"]
        result["perturbation"][name] = {"probe_over_plain": traced/plain,
                                      "warning": abs(traced/plain-1) > .10}
        print(f"Probe perturbation {name}: plain {plain:.4f}, probe {traced:.4f} ms; "
              f"ratio {traced/plain:.3f}x (diagnostic native API, not a speedup)")
        if result["perturbation"][name]["warning"]:
            print("  WARNING: >10% probe timing perturbation; phase values are exploratory, "
                  "repeat with sparser CTA/tile sampling before choosing an optimization.")
    result["perturbation_scope"] = ("Diagnostic native API includes output/trace allocation; probe also "
                                    "initializes timestamps. JIT plain may differ from installed binary. "
                                    "The resource table and whole-output parity are independent checks.")
    print("Sampled instrumented MAIN-tile thread elapsed cycles; NOT production kernel percentages:")
    print(f"{'phase':<26} {'control cycles':>16} {'ExpCast cycles':>16}")
    for phase in PHASES:
        values = []
        for name in states:
            samples = [r["main_tiles"]["phases"][phase]["median_cycles"]
                       for r in result["clocks"][name] if r["main_tiles"] is not None]
            values.append(f"{statistics.median(samples):.1f}" if samples else "--")
        print(f"{phase:<26} {values[0]:>16} {values[1]:>16}")
    print("PV issue -> history during PV -> remaining wait overlap hardware execution. "
          "Do not equate wait with total PV cost or convert these cycles into attention ms.")
    result["status"] = "passed"
    save(result)
    return result


@torch.no_grad()
def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "starting", "route": "INT8 QK + FP8 PV, V smoothing/grouping OFF",
              "kernel_mode": MODE, "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              "measurement_scopes": {
                  "api": "Uninstrumented synchronized wall time; full API includes fresh preprocessing",
                  "stage_events": "CUDA event intervals in a continuous pipeline, including stream gaps",
                  "stage_cupti": "GPU kernel durations with diagnostic synchronization at stage boundaries",
                  "kernel_windows": "CUPTI prepared-core durations without phase probes",
                  "kernel_phases": "Instrumented sampled thread cycles, not global ms or hardware utilization"}}
    def save():
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    try:
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise RuntimeError("Hopper required; no CPU fallback")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(args.seed)
        from sageattention_local import vc, vc_reference, sageattn
        report["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                                 "gpu": torch.cuda.get_device_name(device),
                                 "native_version": vc._extension().sage_vc_kernel_version}
        qkv, report["inputs"] = load_inputs(args, device)
        report["args"] = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
        q = qkv[0]
        print(f"ExpCast focus: NHD {(args.batch,args.queries,args.heads,args.dim)}, keys={args.tokens}, "
              f"{args.dtype}; V smoothing/grouping OFF; K centering retained", flush=True)
        base = vc.prepare(*qkv, v_smooth=False)
        native, _ = load_sage_core_ops()
        states, core, full = make_calls(vc, sageattn, native, qkv, base)
        flash4 = None if args.skip_flash4 else make_flash4(qkv)
        if flash4:
            full["flash4"] = flash4.call
            report["flash4"] = flash4.info
        ops = load_ops()
        staged = staged_prepare(*qkv, ops, DirectStages(), expcast=True, smooth=False)
        assert_payload_equal(staged, base)
        ids = query_indices(args.queries, args.check_queries)
        index = torch.tensor(ids, device=device)
        golden = fp32_reference(*qkv, ids, 32)
        report["accuracy_scope"] = {"dense_queries": len(ids), "all_keys": args.tokens,
                                     "implementation_gates": "1% decoded reference; 0.1% same algorithm, not model quality"}
        report["accuracy"] = {}
        control_out = None
        for name, fn in full.items():
            print("Checking", name, flush=True)
            out = output_hnd(fn(), "BSHD" if name=="flash4" else "BHND", q.shape, q.dtype)
            error = {"dense_fp32": metrics(out.index_select(2,index).float().cpu(), golden)}
            if name in core:
                torch.testing.assert_close(out, core[name](), rtol=0, atol=0)
                error["full_core_exact"] = True
            if name in states:
                oracle = vc_reference.reference(states[name], query_indices=ids, chunk=32).to(q.dtype)
                error["decoded_reference"] = metrics(out.index_select(2,index), oracle.to(device), limit=.01)
                torch.testing.assert_close(out, staged_call(qkv,ops,DirectStages(),states[name]["expcast"]),rtol=0,atol=0)
            if name=="control":
                control_out = out
            if name=="sage_cpp":
                error["control_vs_sage_all_rows"] = metrics(control_out, out, limit=.001)
                control_out = None
            report["accuracy"][name] = error
        save()
        print("Correctness passed; warming device before uninstrumented API timing", flush=True)
        run_window(full["expcast"], args.warmup_seconds, device)
        for fn in full.values():
            fn()
        samples = {"full": {n: [] for n in full}, "core": {n: [] for n in core}}
        report["api_order"] = []
        rng = random.Random(args.seed+3301)
        for trial in range(args.trials):
            order = [(scope,n) for scope,group in samples.items() for n in group]; rng.shuffle(order)
            report["api_order"].append(order)
            for scope,n in order:
                _, ms = measure_trial((full if scope=="full" else core)[n], args.repeats, device)
                samples[scope][n].append(ms)
            print(f"API trial {trial+1}/{args.trials} complete", flush=True)
        report["api"] = {scope: {n: stats(values) for n,values in group.items()} for scope,group in samples.items()}
        report["expcast_savings"] = {scope: expcast_savings(group) for scope,group in samples.items()}
        print("Uninstrumented API wall time; identical payloads for control/ExpCast core:")
        print(f"{'variant':<14} {'full API ms':>14} {'fixed core ms':>15} {'FP32 RMSE %':>14}")
        for n in full:
            core_text = f"{report['api']['core'][n]['median_ms']:.4f}" if n in core else "--"
            print(f"{n:<14} {report['api']['full'][n]['median_ms']:>14.4f} {core_text:>15} "
                  f"{100*report['accuracy'][n]['dense_fp32']['relative_rmse']:>14.6f}")
        for scope, row in report["expcast_savings"].items():
            print(f"ExpCast {scope}: paired median saving {row['median_saved_ms']:+.4f} ms; "
                  f"trial range [{row['min_saved_ms']:+.4f}, {row['max_saved_ms']:+.4f}] ms; "
                  f"control/ExpCast {row['ratio_of_medians']:.4f}x")
        report["stage_events"] = {n: [] for n in states}
        for trial in range(args.trials):
            order = list(states); rng.shuffle(order)
            for n in order:
                recorder = EventStages(device)
                out = staged_call(qkv, ops, recorder, states[n]["expcast"])
                report["stage_events"][n].append(recorder.finish())
        print("Stage event intervals ms (include stream gaps; not pure kernel work):")
        for n, trials in report["stage_events"].items():
            print(n, " ".join(f"{s}={statistics.median(t[s]['cuda_event_ms'] for t in trials):.4f}"
                               for s in STAGES if s in trials[0]))
        save()
        if not args.skip_cupti:
            report["stage_cupti"] = {n: [] for n in states}
            for trial in range(args.kernel_trials):
                order = list(states); rng.shuffle(order)
                for n in order:
                    row = capture_stages(qkv, ops, states[n]["expcast"], args, device,
                                         report["stage_events"][n][0], trial)
                    report["stage_cupti"][n].append(row)
                    save()
            print("CUPTI stage kernel sums ms (diagnostic stage synchronization; NOT API wall time):")
            for n, trials in report["stage_cupti"].items():
                print(n, " ".join(f"{s}={statistics.median(t['stages'][s]['kernel_ms'] for t in trials):.4f}"
                                   for s in STAGES if s in trials[0]["stages"]))
                # Print the real Q/K quantization kernels separately from subtraction.
                by_name = {}
                for t in trials:
                    for kernel in t["stages"]["qk_int8"]["kernels"]:
                        by_name.setdefault(kernel["name"], []).append(kernel["ms"])
                for kernel, durations in by_name.items():
                    print(f"  qk_int8 / {kernel}: median launch {statistics.median(durations):.4f} ms")
            def save_kernels(value):
                report["kernel_windows"] = value; save()
            profile_kernels({**core, **({"flash4": flash4.call} if flash4 else {})}, args, device, save_kernels)
        if args.kernel_phases:
            def save_phases(value):
                report["kernel_phases"] = value; save()
            phase_diagnosis(states, core, vc, args, device, save_phases)
        report["status"] = "passed"
        report["scope"] = ("ExpCast only; K mean retained, V grouping/smoothing absent. "
            "qk_int8 includes input-dtype K subtraction plus separate Q/K quantizers. "
            "API/event/CUPTI/clock64 clocks are separate; do not add them. "
            "Full minus core medians is NOT a measured preprocessing stage. "
            "ExpCast/control accuracy compares different algorithms; no cross-algorithm regression gate.")
        save()
    except Exception as exc:
        report.update(status="failed", error=str(exc)); save()
        raise
    print("Saved:", args.output, flush=True)


if __name__ == "__main__":
    main()
