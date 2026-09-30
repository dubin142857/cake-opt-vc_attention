"""Time public APIs, individual preprocessing stages, and fixed-payload Sage+VC cores.

Works with installed 2.2.0+vc3 or newer; the profiler itself needs no rebuild.
CUDA event intervals include stream idle/host launch gaps, not just kernel work.
Optional CUPTI traces report kernel durations separately. CPU/GPU times overlap.
"""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time
from types import SimpleNamespace

import torch

from benchmark_cuda import (SAGE_VC_VARIANTS, external_backend, input_shapes, measure_trial,
                            output_hnd, query_indices, fp32_reference, error_metrics)


STAGES = ("cluster_sort", "reorder_kv", "k_mean", "qk_int8", "v_pad", "v_quantize", "attention")


class DirectStages:
    def __init__(self, trace=False):
        self.trace = trace

    def call(self, name, function):
        with torch.profiler.record_function("sage_vc/" + name) if self.trace else nullcontext():
            return function()


class EventStages:
    """Events are primed before timing; never synchronize between stages."""
    def __init__(self, device):
        self.device = device
        self.events = {name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                       for name in STAGES}
        for pair in self.events.values():
            for event in pair:
                event.record()
        torch.cuda.synchronize(device)
        self.reset()

    def reset(self):
        self.cpu_ms = {}

    def call(self, name, function):
        if name in self.cpu_ms:
            raise RuntimeError(f"Stage recorded twice: {name}")
        begin, end = self.events[name]
        begin.record()
        start = time.perf_counter()
        result = function()
        self.cpu_ms[name] = (time.perf_counter() - start) * 1000
        end.record()
        return result

    def finish(self):
        torch.cuda.synchronize(self.device)
        return {name: {"cuda_event_ms": self.events[name][0].elapsed_time(self.events[name][1]),
                       "cpu_enqueue_ms": elapsed} for name, elapsed in self.cpu_ms.items()}


def load_ops():
    from sageattention_local import vc
    from sageattention_local.triton.quant_per_thread import per_thread_int8
    from sageattention_local.quant import per_channel_fp8
    return SimpleNamespace(vc=vc, ext=vc._extension(), qk=per_thread_int8, v_global=per_channel_fp8)


@torch.no_grad()
def staged_prepare(q, k, v, ops, recorder, *, expcast, smooth, fixed_permutation=None):
    """Mirror vc3 prepare, with parity checked against the installed implementation.

fixed_permutation is used ONLY for the untimed parity check. Timed stage calls
always recompute clustering; the production package is never monkey-patched.
"""
    call = recorder.call
    permutation = None
    if smooth:
        permutation = fixed_permutation
        if permutation is None:
            permutation = call("cluster_sort", lambda: ops.ext.group_values(v, 8, 2))
        def reorder():
            index = permutation.unsqueeze(-1)
            return k.gather(2, index.expand_as(k)), v.gather(2, index.expand_as(v))
        k, v = call("reorder_kv", reorder)
    km = call("k_mean", lambda: k.mean(dim=2, keepdim=True))
    q8, qs, k8, ks = call("qk_int8", lambda: ops.qk(q, k, km, BLKQ=64, WARPQ=16,
                                                 BLKK=128, WARPK=128, tensor_layout="HND"))
    if smooth:
        v8, vs, means = call("v_quantize", lambda: ops.ext.prepare_values(v.contiguous()))
    else:
        pad = (-v.shape[2]) % 128
        if pad:
            v = call("v_pad", lambda: torch.cat((v, torch.zeros(v.shape[0], v.shape[1], pad, v.shape[3],
                                                              device=v.device, dtype=v.dtype)), dim=2))
        v8, vs, _ = call("v_quantize", lambda: ops.v_global(v, tensor_layout="HND", smooth_v=False))
        means = torch.empty(0, device=v.device, dtype=torch.float16)
    return {"q": q8, "k": k8, "v": v8, "qs": qs, "ks": ks, "vs": vs, "means": means,
            "expcast": expcast, "block_values": smooth, "output_type": 0 if q.dtype == torch.float16 else 1,
            "scale": q.shape[-1] ** -0.5, "permutation": permutation}


def staged_forward(qkv, ops, recorder, *, expcast, smooth):
    state = staged_prepare(*qkv, ops, recorder, expcast=expcast, smooth=smooth)
    return recorder.call("attention", lambda: ops.vc.forward_prepared(state))


def assert_payload_equal(actual, expected):
    """Compare all consumed bytes/scales; ignore unused padded Q/K scale slots."""
    for key in ("q", "k", "v", "qs", "ks", "vs", "means", "permutation"):
        a, e = actual[key], expected[key]
        if a is None or e is None:
            if a is not e:
                raise AssertionError(f"Staged/installed payload differs: {key}")
            continue
        if a.shape != e.shape or a.dtype != e.dtype or a.device != e.device:
            raise AssertionError(f"Staged/installed metadata differs: {key}")
        if key in ("qs", "ks"):
            rows = torch.arange(expected["q" if key == "qs" else "k"].shape[2], device=e.device)
            ids = (rows // 16) * 8 + rows % 8 if key == "qs" else (rows // 128) * 4 + (rows % 8) // 2
            a, e = (x.index_select(2, ids.unique()) for x in (a, e))
        if not torch.equal(a.contiguous().view(torch.uint8), e.contiguous().view(torch.uint8)):
            raise AssertionError(f"Staged/installed payload bytes differ: {key}; refusing misleading stage timings")
    for key in ("expcast", "block_values", "output_type", "scale"):
        if actual[key] != expected[key]:
            raise AssertionError(f"Staged/installed configuration differs: {key}")


def fixed_variants(bases):
    # All tensor objects (including QK scales and V metadata) are shared by the
    # two ExpCast settings within a V mode. Only the scalar switch changes.
    return {name: {**bases[smooth], "expcast": cast}
            for name, (cast, smooth) in SAGE_VC_VARIANTS.items()}


def stats(samples):
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples),
            "trial_mean_ms": samples}


def summarize_stages(trials):
    names = set(trials[0][0]["stages"])
    rows = {}
    for name in STAGES:
        if name not in names:
            continue
        rows[name] = {}
        for metric in ("cuda_event_ms", "cpu_enqueue_ms"):
            values = [statistics.mean(sample["stages"][name][metric] for sample in trial) for trial in trials]
            rows[name][metric] = stats(values)
    return {"stages": rows,
            "instrumented_wall": stats([statistics.mean(x["wall_ms"] for x in trial) for trial in trials]),
            "event_sum": stats([statistics.mean(sum(s["cuda_event_ms"] for s in x["stages"].values())
                                                for x in trial) for trial in trials]),
            "raw_trials": trials}


def trace_summary(events, core_names=("sage_vc_attention_kernel",)):
    """Keep device and host-runtime events separate; exclude nested CPU operators."""
    groups = {}
    resources = []
    for event in events:
        category = event.get("cat")
        if event.get("ph") != "X" or category not in ("kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime"):
            continue
        key = (category, event.get("name", "unknown"))
        row = groups.setdefault(key, {"category": category, "name": key[1], "calls": 0, "total_us": 0.})
        row["calls"] += 1
        row["total_us"] += event.get("dur", 0.)
        if category == "kernel" and any(marker in key[1] for marker in core_names):
            metadata = event.get("args", {})
            resources.append({"name": key[1], "duration_ms": event.get("dur", 0.) / 1000,
                              **{key: metadata.get(key) for key in (
                                  "registers per thread", "shared memory", "grid", "block",
                                  "est. achieved occupancy %")}})
    rows = sorted(groups.values(), key=lambda x: x["total_us"], reverse=True)
    kernels = [x for x in rows if x["category"] == "kernel"]
    if not kernels:
        raise RuntimeError("No GPU kernel events captured; check CUPTI/PyTorch profiler availability")
    core = sum(x["total_us"] for x in kernels if any(marker in x["name"] for marker in core_names))
    if not core:
        raise RuntimeError("Trace does not contain the requested Sage VC native core; refusing a zero-time result")
    total = sum(x["total_us"] for x in kernels)
    return {"core_kernel_ms": core / 1000, "other_kernel_sum_ms": (total-core) / 1000,
            "kernel_sum_ms": total / 1000, "kernel_launches": sum(x["calls"] for x in kernels),
            "events": rows, "core_resources": resources}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--queries", type=int, default=37741)
    parser.add_argument("--tokens", type=int, default=37741)
    parser.add_argument("--dim", type=int, choices=(64,128), default=128)
    parser.add_argument("--dtype", choices=("float16","bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="results/h200_sage_vc_profile.json")
    parser.add_argument("--trace-dir", help="Optional CUPTI CPU/GPU timelines, one staged call per variant")
    parser.add_argument("--compare-kernels", action="store_true",
                        help="Compare previous/optimized cores on identical payloads; validate ALL output rows")
    parser.add_argument("--compare-sage-core", action="store_true",
                        help="vc15+: compare original Sage C++, registered Sage, VC control and vc14/vc15 ExpCast on the same payload")
    parser.add_argument("--compare-sage-layout", action="store_true",
                        help="vc16+: compare original-loop candidates with vc15 and Sage, including fresh full APIs")
    parser.add_argument("--compare-probability-candidates", action="store_true",
                        help="vc17+: compare four probability candidates on the vc16 body, plus vc16 and Sage")
    parser.add_argument("--compare-register-candidates", action="store_true",
                        help="vc18+: compare D128 register caps and N64 PV fragments, control and ExpCast, with resources and full APIs")
    parser.add_argument("--compare-pipeline-candidates", action="store_true",
                        help="vc19+: vc16 versus K/V double buffering and late-copy control, with fresh full APIs and compiled resources")
    parser.add_argument("--kernel-samples", type=int, default=20,
                        help="GPU launches per variant/trial for --compare-sage-layout with --trace-dir")
    parser.add_argument("--kernel-trials", type=int, default=3,
                        help="Randomized CUPTI trials for --compare-sage-layout with --trace-dir")
    parser.add_argument("--expcast-only",action="store_true",
                        help="Measure Sage/control/ExpCast only; skip V grouping/smoothing entirely")
    parser.add_argument("--kernel-baseline",choices=("vc4","vc6","vc7","vc9","vc10","vc11","vc12","vc13_serial","vc13_overlap","vc14"),
                        help="Same-payload baseline; defaults to vc14 on vc15, vc12 on vc14, vc13_overlap after the vc13.1 rollback, vc12 on vc13, vc11 on vc12, vc10 on vc11, vc9 on vc10, vc7 on vc8/9, vc6 on vc7, otherwise vc4")
    parser.add_argument("--compare-probability-schedules", action="store_true",
                        help="vc13+: compare vc12, serial scratch and overlap on identical payloads; validate ALL output rows")
    parser.add_argument("--compare-v-scales", action="store_true",
                        help="vc6+: compare old block and new channel V quantization, including API time and dense FP32 errors")
    parser.add_argument("--compare-grouping", action="store_true",
                        help="vc9+: compare legacy/fused clustering and full APIs, label agreement and dense FP32 errors")
    parser.add_argument("--check-queries", type=int, default=128, help="Dense FP32 check rows for optional comparisons; 0 means all")
    parser.add_argument("--reference-chunk", type=int, default=32)
    parser.add_argument("--ncu-variant", choices=tuple(SAGE_VC_VARIANTS),
                        help="Only capture one warmed fixed-payload native call inside cudaProfilerStart/Stop")
    args = parser.parse_args(argv)
    if sum((args.compare_register_candidates,args.compare_probability_candidates,args.compare_pipeline_candidates)) > 1:
        parser.error("Choose one of register, probability or pipeline candidates in one comparison")
    if (args.compare_register_candidates or args.compare_pipeline_candidates) and args.dim != 128:
        parser.error("Register/pipeline comparisons currently report D128 resources; use --dim 128")
    if args.compare_register_candidates or args.compare_pipeline_candidates:
        args.expcast_only = True
    if args.compare_probability_candidates or args.compare_register_candidates or args.compare_pipeline_candidates:
        args.compare_sage_layout = True
    if min(args.batch,args.heads,args.queries,args.tokens,args.warmup,args.repeats,args.trials,args.kernel_samples,args.kernel_trials) < 1:
        parser.error("shapes, warmup, repeats and trials must be positive")
    if args.check_queries < 0 or args.reference_chunk < 1:
        parser.error("check-queries must be nonnegative and reference-chunk positive")
    if args.kernel_baseline and not args.compare_kernels:
        parser.error("--kernel-baseline requires --compare-kernels")
    if args.expcast_only and (args.compare_v_scales or args.compare_grouping or args.ncu_variant):
        parser.error("--expcast-only cannot be combined with V/grouping comparisons or --ncu-variant")
    if args.ncu_variant and (args.compare_sage_layout or args.compare_sage_core or args.compare_probability_schedules or args.trace_dir or args.compare_kernels or args.compare_v_scales or args.compare_grouping):
        parser.error("--ncu-variant uses a separate profiler run")
    return args


def validate_device(args):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or torch.cuda.get_device_capability(device) != (9,0):
        raise RuntimeError("Sage VC profiling requires Hopper, with sageattention-local 2.2.0+vc3 or newer installed")
    torch.cuda.set_device(device)
    if torch.cuda.current_stream(device) != torch.cuda.default_stream(device):
        raise RuntimeError("Use the default CUDA stream, as required by the public Sage VC API")
    if args.trace_dir and torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError("CUDA profiling/CUPTI unavailable; omit --trace-dir for event/wall timing")
    return device


def output_difference(actual, reference, chunk=512):
    """All query rows, bounded scratch memory; no timing or dense oracle."""
    if actual.shape != reference.shape or actual.ndim != 4 or chunk < 1:
        raise ValueError("Expected matching BHND outputs and positive chunk")
    sums = torch.zeros(2,device=actual.device,dtype=torch.float64)
    peak = torch.zeros((),device=actual.device,dtype=torch.float64)
    for begin in range(0,actual.shape[2],chunk):
        a = actual[:,:,begin:begin+chunk].double()
        b = reference[:,:,begin:begin+chunk].double()
        diff = a-b
        sums += torch.stack((diff.square().sum(),b.square().sum()))
        peak = torch.maximum(peak,diff.abs().amax())
    error, norm = sums.sqrt().tolist()
    maximum = peak.item()
    if not all(math.isfinite(x) for x in (error,norm,maximum)):
        raise RuntimeError("Nonfinite output in same-payload kernel comparison")
    return {"rel_rmse": error/max(norm,1e-12), "max_abs": maximum,
            "checked_queries": actual.shape[2], "exact_match": maximum == 0}


def kernel_calls(ops, fixed, baseline="vc4"):
    # Each pair closes over the very same state object, including every scale.
    return {(name,mode): lambda state=state,mode=mode: ops.vc.forward_prepared(state,kernel_mode=mode)
            for name,state in fixed.items() for mode in (baseline,"optimized")}


def load_sage_core_ops():
    from sageattention_local import _qattn_sm90, sm90_compile
    name = "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf"
    return getattr(_qattn_sm90,name), getattr(sm90_compile,name)


def sage_core_calls(ops, state, native, registered):
    """Every call allocates its output; all calls share the very same input tensors."""
    if state["block_values"]:
        raise ValueError("Original Sage core comparison requires unsmoothed, channel-quantized V")
    def sage_call(function):
        query = state["q"]
        with torch.cuda.device(query.device) if query.is_cuda else nullcontext():
            output = torch.empty(query.shape,device=query.device,
                                 dtype=torch.float16 if state["output_type"] == 0 else torch.bfloat16)
            function(query,state["k"],state["v"],output,state["qs"],state["ks"],state["vs"],
                     1,0,3,state["scale"],0)
        return output
    control, cast = {**state,"expcast":False}, {**state,"expcast":True}
    return {"sage_cpp":lambda:sage_call(native),
            "sage_registered":lambda:sage_call(registered),
            "vc_control":lambda:ops.vc.forward_prepared(control),
            "vc14_expcast":lambda:ops.vc.forward_prepared(cast,kernel_mode="vc14"),
            "vc_optimized_expcast":lambda:ops.vc.forward_prepared(cast)}


@torch.no_grad()
def compare_sage_core(ops, state, args, device):
    if getattr(ops.ext,"sage_vc_kernel_version",0) < 12:
        raise RuntimeError("--compare-sage-core needs vc15; rebuild with bash scripts/build_sage_local.sh")
    calls = sage_core_calls(ops,state,*load_sage_core_ops())
    for call in calls.values():
        for _ in range(args.warmup): call()
    torch.cuda.synchronize(device)
    errors, regression = {}, {}
    original = calls["sage_cpp"]()
    for name,call in calls.items():
        if name == "sage_cpp": continue
        actual = call()
        errors[name] = output_difference(actual,original)
        if name in ("sage_registered","vc_control") and errors[name]["rel_rmse"] > .001:
            raise AssertionError(f"{name}/original Sage relative RMSE exceeds 0.1%: {errors[name]}")
        del actual
    del original
    previous, actual = calls["vc14_expcast"](), calls["vc_optimized_expcast"]()
    regression = output_difference(actual,previous)
    if regression["rel_rmse"] > .001:
        raise AssertionError(f"Optimized/vc14 ExpCast relative RMSE exceeds 0.1%: {regression}")
    del previous,actual
    times = {name:[] for name in calls}
    rng, orders = random.Random(args.seed), []
    for trial in range(args.trials):
        order = list(calls)
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            _, elapsed = measure_trial(calls[name],args.repeats,device)
            times[name].append(elapsed)
        print(f"Original Sage core comparison trial {trial+1}/{args.trials} complete",flush=True)
    rows = {name:{**stats(samples),"vs_original_sage":errors.get(name)} for name,samples in times.items()}
    print("\nIdentical quantized inputs, fresh output allocation, synchronized API wall time:")
    print(f"{'core API':<25} {'median ms':>12} {'Sage/core':>12} {'vs Sage RMSE %':>16}")
    for name,row in rows.items():
        row["sage_cpp_over_variant"] = rows["sage_cpp"]["median_ms"]/row["median_ms"]
        error = f"{errors[name]['rel_rmse']*100:.6f}" if name in errors else "--"
        print(f"{name:<25} {row['median_ms']:>12.4f} {row['sage_cpp_over_variant']:>11.3f}x {error:>16}")
    print("ExpCast/Sage errors compare different algorithms; only control/Sage and optimized/vc14 use the 0.1% regression gate.")
    return {"rows":rows,"trial_order":orders,"optimized_vs_vc14":regression,
            "optimized_kernel":getattr(ops.vc,"DEFAULT_EXPCAST_KERNEL",None),
            "scope":"Same Q/K/V bytes and scales; no preprocessing; every call includes fresh output allocation and native dispatch. Synchronized wall time, not pure GPU duration."}


def sage_layout_calls(ops, state, qkv, native, sage_api, probability_candidates=False, register_candidates=False, pipeline_candidates=False):
    """All native variants share operands. Full APIs recompute preprocessing."""
    modes = {False:("vc15","vc16","vc16_early_v","vc16_pack_first","vc16_pack_early"),
             True:("vc15","vc16","vc16_early_v","vc16_serial")}
    if probability_candidates:
        modes = {False:("vc16",),True:("vc16","vc17_g1","vc17_g4","vc17_c4","vc17_direct")}
    if register_candidates:
        modes = {cast:("vc16",*ops.vc.REGISTER_CANDIDATES) for cast in (False,True)}
    if pipeline_candidates:
        modes = {cast:("vc16",*ops.vc.PIPELINE_CANDIDATES) for cast in (False,True)}
    calls, full, metadata = {}, {}, {}
    for cast, variants in modes.items():
        payload = {**state,"expcast":cast}
        for mode in variants:
            name = ("expcast_" if cast else "control_") + mode
            calls[name] = lambda payload=payload,mode=mode:ops.vc.forward_prepared(payload,kernel_mode=mode)
            full[name] = lambda cast=cast,mode=mode:ops.vc.sageattn_vc(*qkv,tensor_layout="HND",expcast=cast,v_smooth=False,kernel_mode=mode)
            metadata[name] = {"expcast":cast,"kernel_mode":mode,"reference":("expcast_" if cast else "control_")+modes[cast][0]}
            if register_candidates or pipeline_candidates:
                metadata[name].update(register_limit=ops.vc.REGISTER_CANDIDATES.get(mode,168) if register_candidates else 168,
                    resource_key=("expcast_" if cast else "control_")+f"sage_layout_{ops.vc.SAGE_LAYOUT_MODES[mode]}")
            if pipeline_candidates:
                config = ops.vc.PIPELINE_CANDIDATES.get(mode,dict(k_buffers=1,v_buffers=1,early_tma_prefetch=0))
                metadata[name]["expected_pipeline_resources"] = {
                    **config,"dynamic_shared_bytes":(64+128*(config["k_buffers"]+config["v_buffers"]))*128}
    calls["sage_cpp"] = sage_core_calls(ops,state,native,native)["sage_cpp"]
    full["sage_cpp"] = lambda:sage_api(*qkv,tensor_layout="HND",is_causal=False)
    metadata["sage_cpp"] = {"expcast":False,"reference":"control_"+modes[False][0],"full_api":"sage_local public sageattn"}
    return calls,full,metadata


def kernel_sample_summary(events, expected, markers):
    summary = trace_summary(events,markers)
    durations = [row["duration_ms"] for row in summary["core_resources"]]
    if len(durations) != expected or summary["kernel_launches"] != expected:
        raise RuntimeError(f"Expected {expected} single-kernel calls, captured {len(durations)} core / {summary['kernel_launches']} total launches")
    if not all(math.isfinite(x) and x > 0 for x in durations):
        raise RuntimeError("Nonpositive/nonfinite CUPTI kernel durations")
    return {"median_ms":statistics.median(durations),"min_ms":min(durations),"max_ms":max(durations),
            "launch_ms":durations,"resources":summary["core_resources"][0]}


@torch.no_grad()
def trace_layout_samples(calls, args, device):
    directory = Path(args.trace_dir)/"sage_layout_samples"
    directory.mkdir(parents=True,exist_ok=True)
    trials = {name:[] for name in calls}
    rng,orders = random.Random(args.seed+1600), []
    for trial in range(args.kernel_trials):
        order=list(calls);rng.shuffle(order);orders.append(order)
        for name in order:
            call=calls[name]
            for _ in range(args.warmup): call()
            torch.cuda.synchronize(device)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(args.kernel_samples): output=call()
                torch.cuda.synchronize(device)
            trace=directory/f"{name}.{trial}.trace.json"
            prof.export_chrome_trace(str(trace))
            markers=("qk_int8_sv_f8_attn_kernel",) if name=="sage_cpp" else ("sage_vc_attention_kernel",)
            row=kernel_sample_summary(json.loads(trace.read_text(encoding="utf-8"))["traceEvents"],args.kernel_samples,markers)
            trials[name].append({"path":str(trace),**row})
            print(f"CUPTI trial {trial+1}/{args.kernel_trials}: {name} median {row['median_ms']:.4f} ms ({args.kernel_samples} launches)",flush=True)
    summaries = {}
    for name,rows in trials.items():
        summary = stats([x["median_ms"] for x in rows])
        summary["trial_median_ms"] = summary.pop("trial_mean_ms")
        summaries[name] = {**summary,"trials":rows}
    return {"variants":summaries,
            "trial_order":orders,"scope":"CUPTI GPU kernel duration only; median of per-trial launch medians. Profiler-active measurements, distinct from uninstrumented API wall times."}


@torch.no_grad()
def compare_sage_layout(ops,state,qkv,args,device,sage_api):
    pipeline = args.compare_pipeline_candidates
    if pipeline and getattr(ops.ext,"sage_vc_kernel_version",0) < 16:
        raise RuntimeError("--compare-pipeline-candidates needs vc19; rebuild with bash scripts/build_sage_local.sh")
    if args.compare_register_candidates and getattr(ops.ext,"sage_vc_kernel_version",0) < 15:
        raise RuntimeError("--compare-register-candidates needs vc18; rebuild with bash scripts/build_sage_local.sh")
    if args.compare_probability_candidates and getattr(ops.ext,"sage_vc_kernel_version",0) < 14:
        raise RuntimeError("--compare-probability-candidates needs vc17; rebuild with bash scripts/build_sage_local.sh")
    if getattr(ops.ext,"sage_vc_kernel_version",0) < 13:
        raise RuntimeError("--compare-sage-layout needs vc16; rebuild with bash scripts/build_sage_local.sh")
    native,_ = load_sage_core_ops()
    calls,full,metadata = sage_layout_calls(ops,state,qkv,native,sage_api,args.compare_probability_candidates,args.compare_register_candidates,pipeline)
    if args.compare_register_candidates or pipeline:
        resources = ops.vc.kernel_resources(0 if args.dtype=="float16" else 1)
        for meta in metadata.values():
            if "resource_key" in meta:
                meta["compiled_resources"] = resources[meta["resource_key"]]
                if meta["compiled_resources"]["requested_register_limit"] != meta["register_limit"]:
                    raise RuntimeError(f"Register cap metadata does not match the loaded binary: {meta}")
                for key,value in meta.get("expected_pipeline_resources",{}).items():
                    if meta["compiled_resources"].get(key) != value:
                        raise RuntimeError(f"Pipeline metadata does not match the loaded binary: {key}, {meta}")
    for call in (*calls.values(),*full.values()):
        for _ in range(args.warmup): call()
    torch.cuda.synchronize(device)
    errors={}
    for cast in (False,True):
        reference_name=next(meta["reference"] for meta in metadata.values() if meta["expcast"]==cast)
        reference=calls[reference_name]()
        for name,meta in metadata.items():
            if meta["expcast"]!=cast: continue
            output=calls[name]()
            error=output_difference(output,reference)
            if error["rel_rmse"]>.001:
                raise AssertionError(f"{name}/{reference_name} exceeds 0.1%: {error}")
            api_output=full[name]()
            api_error=output_difference(api_output,output)
            if api_error["rel_rmse"]>.001:
                raise AssertionError(f"{name} fresh API/fixed-payload output differs: {api_error}")
            errors[name]={"vs_same_algorithm_baseline":error,"full_vs_fixed":api_error}
            del output,api_output
        del reference
    jobs={(kind,name):call for kind,group in (("core",calls),("api",full)) for name,call in group.items()}
    times={key:[] for key in jobs};orders=[];rng=random.Random(args.seed+16)
    for trial in range(args.trials):
        order=list(jobs);rng.shuffle(order);orders.append(order)
        for key in order:
            _,elapsed=measure_trial(jobs[key],args.repeats,device)
            times[key].append(elapsed)
        print(f"Sage layout trial {trial+1}/{args.trials} complete",flush=True)
    rows={name:{**meta,"core":stats(times["core",name]),"full_api":stats(times["api",name]),**errors[name]}
          for name,meta in metadata.items()}
    print("\nOriginal-loop candidates, identical native payloads; fresh preprocessing for full API (ms):")
    baseline_label="vc16" if args.compare_probability_candidates or args.compare_register_candidates or pipeline else "vc15"
    print(f"{'variant':<28} {'core API':>11} {'full API':>11} {baseline_label+'/core':>11} {'vs base RMSE %':>17}")
    for name,row in rows.items():
        ref=row["reference"]
        row["baseline_core_over_variant"]=rows[ref]["core"]["median_ms"]/row["core"]["median_ms"]
        row["sage_core_over_variant"]=rows["sage_cpp"]["core"]["median_ms"]/row["core"]["median_ms"]
        row["baseline_full_api_over_variant"]=rows[ref]["full_api"]["median_ms"]/row["full_api"]["median_ms"]
        row["sage_full_api_over_variant"]=rows["sage_cpp"]["full_api"]["median_ms"]/row["full_api"]["median_ms"]
        print(f"{name:<28} {row['core']['median_ms']:>11.4f} {row['full_api']['median_ms']:>11.4f} {row['baseline_core_over_variant']:>10.3f}x {100*row['vs_same_algorithm_baseline']['rel_rmse']:>17.6f}")
    print("vc15 remains the default. Rows compare schedules within the same algorithm; 0.1% is an implementation gate, not model quality.")
    if args.compare_register_candidates:
        print("\nCompiled D128 resources for the actual candidates (occupancy prediction, not utilization):")
        print(f"{'variant':<28} {'PV N':>5} {'cap':>5} {'regs':>5} {'local B':>9} {'shared B':>10} {'blocks/SM':>10} {'Sage/full':>11}")
        for name,row in rows.items():
            if "compiled_resources" not in row: continue
            r=row["compiled_resources"]
            print(f"{name:<28} {r['pv_fragment_channels']:>5} {row['register_limit']:>5} {r['registers_per_thread']:>5} {r['local_bytes_per_thread']:>9} "
                  f"{r['static_shared_bytes']+r['dynamic_shared_bytes']:>10} {r['predicted_active_blocks_per_sm']:>10} {row['sage_full_api_over_variant']:>10.3f}x")
        print("Register-only rows change the compiler cap; pv64 rows also split D128 PV into two N64 fragments. Two-level FP32 accumulation is retained. local B is stack/spill storage, not measured traffic; occupancy is not a speedup.")
    if pipeline:
        print("\nTMA pipeline resources: cap 168, unchanged vc16 QK/P/PV arithmetic; D128 N128 PV")
        print(f"{'variant':<28} {'K/V':>5} {'early':>6} {'regs':>5} {'local B':>9} {'shared B':>10} {'blocks/SM':>10} {'Sage/full':>11}")
        for name,row in rows.items():
            if "compiled_resources" not in row: continue
            r=row["compiled_resources"]
            buffers=f"{r['k_buffers']}/{r['v_buffers']}"
            print(f"{name:<28} {buffers:>5} {r['early_tma_prefetch']:>6} {r['registers_per_thread']:>5} {r['local_bytes_per_thread']:>9} "
                  f"{r['static_shared_bytes']+r['dynamic_shared_bytes']:>10} {r['predicted_active_blocks_per_sm']:>10} {row['sage_full_api_over_variant']:>10.3f}x")
        for cast in ("control","expcast"):
            early,late=rows[f"{cast}_vc19_kv"],rows[f"{cast}_vc19_kv_late"]
            early["late_copy_core_over_early"]=late["core"]["median_ms"]/early["core"]["median_ms"]
            print(f"{cast}: same KV buffers/synchronization, late/early core = {early['late_copy_core_over_early']:.4f}x")
        print("blocks/SM is an occupancy prediction, not measured utilization. Speedup requires lower full API time; no H200 speed claim from compilation alone.")
    return {"variants":rows,"trial_order":orders,"scope":"Core APIs include allocation/dispatch; full APIs redo all preprocessing. All output rows validated outside timing; default is unchanged."},calls


def kernel_baseline(ops,args):
    version = getattr(ops.ext,"sage_vc_kernel_version",0)
    restored = getattr(getattr(ops,"vc",None),"DEFAULT_EXPCAST_KERNEL",None) == "vc12"
    baseline = getattr(args,"kernel_baseline",None) or ("vc14" if version >= 12 else "vc12" if version >= 11 else "vc13_overlap" if version >= 10 and restored else "vc12" if version >= 10 else "vc11" if version >= 9 else "vc10" if version >= 8 else "vc9" if version >= 7 else "vc7" if version >= 6 else "vc6" if version >= 5 else "vc4")
    if baseline == "vc14" and version < 12:
        raise RuntimeError("--kernel-baseline vc14 needs vc15; rebuild with bash scripts/build_sage_local.sh")
    if baseline == "vc13_overlap" and (version < 10 or not (restored or version >= 11)):
        raise RuntimeError("--kernel-baseline vc13_overlap needs vc13.1; rebuild with bash scripts/build_sage_local.sh")
    if baseline in ("vc12", "vc13_serial") and version < 10:
        raise RuntimeError("--kernel-baseline vc12/vc13_serial needs vc13; rebuild with bash scripts/build_sage_local.sh")
    if baseline == "vc11" and version < 9:
        raise RuntimeError("--kernel-baseline vc11 needs vc12; rebuild with bash scripts/build_sage_local.sh")
    if baseline == "vc10" and version < 8:
        raise RuntimeError("--kernel-baseline vc10 needs vc11; rebuild with bash scripts/build_sage_local.sh")
    if baseline == "vc9" and version < 7:
        raise RuntimeError("--kernel-baseline vc9 needs vc10; rebuild with bash scripts/build_sage_local.sh")
    if baseline == "vc7" and version < 6:
        raise RuntimeError("--kernel-baseline vc7 needs vc8; rebuild with bash scripts/build_sage_local.sh")
    if baseline == "vc6" and getattr(ops.ext,"sage_vc_kernel_version",0) < 5:
        raise RuntimeError("--kernel-baseline vc6 needs vc7; rebuild with bash scripts/build_sage_local.sh")
    return baseline


def value_scale_states(ops, qkv, base):
    """Reuse the exact grouping AND QK bytes/scales; only V quantization changes."""
    v = qkv[2].gather(2, base["permutation"].unsqueeze(-1).expand_as(qkv[2])).contiguous()
    payload, scales, means = ops.ext.prepare_values(v, channel_scale=False)
    return {"channel": base, "block": {**base, "v": payload, "vs": scales, "means": means}}


def require_fused_grouping(ops):
    if getattr(ops.ext,"sage_vc_preprocess_version",0) < 1:
        raise RuntimeError("--compare-grouping needs vc9; rebuild with bash scripts/build_sage_local.sh")


def grouping_states(ops, qkv):
    """Labels and accuracy payloads must describe the exact same grouping run."""
    require_fused_grouping(ops)
    states, labels, permutations = {}, {}, {}
    for mode,group in (("legacy",ops.ext.group_values_legacy_debug),("fused",ops.ext.group_values_debug)):
        permutation, label, _ = group(qkv[2],8,2)
        states[mode] = staged_prepare(*qkv,ops,DirectStages(),expcast=False,smooth=True,
                                     fixed_permutation=permutation)
        labels[mode], permutations[mode] = label, permutation
    agreement = {"label_mismatch_fraction": (labels["fused"] != labels["legacy"]).double().mean().item(),
                 "permutation_exact_match": torch.equal(permutations["fused"],permutations["legacy"]),
                 "checked_tokens": labels["fused"].numel()}
    return states, agreement


@torch.no_grad()
def compare_grouping(ops, qkv, args, device):
    states, agreement = grouping_states(ops,qkv)
    indices = query_indices(qkv[0].shape[2],args.check_queries)
    ids = torch.tensor(indices,device=device)
    print(f"Grouping comparison: dense FP32 for {ids.numel()} query rows, ALL keys; outside timing",flush=True)
    golden = fp32_reference(*qkv,indices,args.reference_chunk)
    calls, accuracy, differences = {}, {}, {}
    for mode,group in (("legacy",ops.ext.group_values_legacy),("fused",ops.ext.group_values)):
        calls["cluster_sort",mode] = lambda group=group: group(qkv[2],8,2)
    for cast in (False,True):
        name = "sage_vc" if cast else "sage_vc_smooth"
        outputs = {}
        for mode,base in states.items():
            # Accuracy uses the recorded permutation. Timed API calls redo
            # clustering from original QKV, with no frozen layout or cache.
            outputs[mode] = ops.vc.forward_prepared({**base,"expcast":cast})
            accuracy[name,mode] = error_metrics(outputs[mode].index_select(2,ids).float().cpu(),golden)
            calls[name,mode] = lambda cast=cast,mode=mode: ops.vc.sageattn_vc(
                *qkv,expcast=cast,v_smooth=True,grouping_mode=mode)
        differences[name] = output_difference(outputs["fused"],outputs["legacy"])
        del outputs
    for call in calls.values():
        for _ in range(args.warmup): call()
    torch.cuda.synchronize(device)
    times, orders, rng = {key:[] for key in calls}, [], random.Random(args.seed)
    for trial in range(args.trials):
        order = list(calls)
        rng.shuffle(order)
        orders.append(order)
        for key in order:
            _, elapsed = measure_trial(calls[key],args.repeats,device)
            times[key].append(elapsed)
        print(f"Grouping comparison trial {trial+1}/{args.trials} complete",flush=True)
    rows = {}
    print("\nLegacy vs fused grouping: same original QKV, unchanged attention kernel")
    print(f"Label mismatch {100*agreement['label_mismatch_fraction']:.6f}%; exact permutation={agreement['permutation_exact_match']}")
    print(f"{'scope':<20} {'legacy ms':>12} {'fused ms':>12} {'old/new':>10} {'legacy FP32 %':>15} {'fused FP32 %':>15}")
    for name in ("cluster_sort","sage_vc_smooth","sage_vc"):
        old,new = (stats(times[name,mode]) for mode in ("legacy","fused"))
        speedup = old["median_ms"]/new["median_ms"]
        row = {"legacy":old,"fused":new,"legacy_over_fused":speedup}
        errors = ""
        if name != "cluster_sort":
            row["vs_dense_fp32"] = {mode:accuracy[name,mode] for mode in ("legacy","fused")}
            row["fused_vs_legacy_all_outputs"] = differences[name]
            errors = "".join(f"{100*accuracy[name,mode]['relative_rmse_vs_fp32']:>15.6f}" for mode in ("legacy","fused"))
        rows[name] = row
        print(f"{name:<20} {old['median_ms']:>12.4f} {new['median_ms']:>12.4f} {speedup:>9.3f}x {errors}")
    print("API includes fresh grouping; cluster_sort includes stable sorting/allocation. Grouping can change quantized payloads; no same-payload regression gate or model-quality claim.")
    return {"rows":rows,"grouping_agreement":agreement,"trial_order":orders,"query_indices":indices,
            "scope":"Same original QKV; final accuracy uses exact debug permutations. API reclusters every call. FP32 summation order can change grouping and quantization."}


@torch.no_grad()
def compare_v_scales(ops, qkv, base, args, device):
    if getattr(ops.ext,"sage_vc_kernel_version",0) < 4:
        raise RuntimeError("--compare-v-scales needs vc6; rebuild with bash scripts/build_sage_local.sh")
    states = value_scale_states(ops,qkv,base)
    indices = query_indices(qkv[0].shape[2],args.check_queries)
    ids = torch.tensor(indices,device=device)
    print(f"V scale comparison: FP32 reference for {ids.numel()} query rows, ALL keys; outside timing",flush=True)
    golden = fp32_reference(*qkv,indices,args.reference_chunk)
    calls, accuracy, differences = {}, {}, {}
    for cast in (False,True):
        name = "sage_vc" if cast else "sage_vc_smooth"
        outputs = {}
        for mode,base_state in states.items():
            state = {**base_state,"expcast":cast}
            calls[name,mode,"core"] = lambda state=state: ops.vc.forward_prepared(state)
            calls[name,mode,"api"] = lambda cast=cast,mode=mode: ops.vc.sageattn_vc(
                *qkv, expcast=cast, v_smooth=True, v_scale_mode=mode)
            outputs[mode] = calls[name,mode,"core"]()
            accuracy[name,mode] = error_metrics(outputs[mode].index_select(2,ids).cpu(),golden)
        differences[name] = output_difference(outputs["channel"],outputs["block"])
        del outputs
    for call in calls.values():
        for _ in range(args.warmup): call()
    torch.cuda.synchronize(device)
    times, orders, rng = {key:[] for key in calls}, [], random.Random(args.seed)
    for trial in range(args.trials):
        order = list(calls)
        rng.shuffle(order)
        orders.append(order)
        for key in order:
            _, elapsed = measure_trial(calls[key],args.repeats,device)
            times[key].append(elapsed)
        print(f"V scale comparison trial {trial+1}/{args.trials} complete",flush=True)
    rows = {}
    print("\nV scale quantization comparison (different V payloads; same original inputs):")
    print(f"{'variant':<20} {'V scale':<9} {'full API ms':>13} {'core API ms':>13} {'FP32 rel RMSE %':>18}")
    for name in ("sage_vc_smooth","sage_vc"):
        rows[name] = {"channel_vs_block_all_outputs":differences[name]}
        for mode in ("block","channel"):
            api, core = stats(times[name,mode,"api"]), stats(times[name,mode,"core"])
            error = accuracy[name,mode]
            rows[name][mode] = {"full_api":api,"fixed_core_api":core,"vs_dense_fp32":error}
            print(f"{name:<20} {mode:<9} {api['median_ms']:>13.4f} {core['median_ms']:>13.4f} "
                  f"{100*error['relative_rmse_vs_fp32']:>18.6f}")
    print("V quantization changed: channel/block output differences are NOT an implementation regression gate.")
    return {"variants":rows,"trial_order":orders,"query_indices":ids.tolist(),
            "scope":"Same original QKV; fixed cores share grouping and QK payloads; API recomputes grouping; quantization changes; no model quality gate."}


@torch.no_grad()
def compare_kernels(ops, fixed, args, device):
    if getattr(ops.ext,"sage_vc_kernel_version",0) < 3:
        raise RuntimeError("--compare-kernels needs vc5; rebuild with bash scripts/build_sage_local.sh")
    baseline = kernel_baseline(ops,args)
    calls = kernel_calls(ops,fixed,baseline)
    for key,call in calls.items():
        for _ in range(args.warmup):
            call()
    torch.cuda.synchronize(device)
    errors = {}
    for name in fixed:
        old, new = calls[name,baseline](), calls[name,"optimized"]()
        errors[name] = output_difference(new,old)
        if errors[name]["rel_rmse"] > .001:
            raise AssertionError(f"Optimized/{baseline} relative RMSE exceeds 0.1%: {name}, {errors[name]}")
        del old,new
    times = {key: [] for key in calls}
    rng, orders = random.Random(args.seed), []
    for trial in range(args.trials):
        order = list(calls)
        rng.shuffle(order)
        orders.append(order)
        for key in order:
            _, elapsed = measure_trial(calls[key],args.repeats,device)
            times[key].append(elapsed)
        print(f"Kernel comparison trial {trial+1}/{args.trials} complete",flush=True)
    rows = {}
    print(f"\nSame-payload native APIs: {baseline} vs optimized (ms; excludes all preprocessing)")
    if baseline == "vc14":
        print("vc14/vc15 differ only in four-byte assembly: the second paired conversion appends the first pair. P bytes, decoded-P FP32 sums and PV accumulation are unchanged; no shared copy.")
    elif baseline == "vc12" and getattr(ops.ext,"sage_vc_kernel_version",0) >= 11:
        resolved = getattr(ops.vc,"DEFAULT_EXPCAST_KERNEL","vc14")
        print(f"vc12/{resolved} differ only in FP32-to-byte conversion/packing. P rounding, decoded-P FP32 denominator, Q/K/V payloads and PV accumulation are unchanged; no shared probability copy.")
    elif baseline in ("vc12", "vc13_serial", "vc13_overlap"):
        resolved = getattr(ops.vc,"DEFAULT_EXPCAST_KERNEL","vc13_overlap")
        print(f"Loaded optimized ExpCast path: {resolved}. vc12 has no shared copy; vc13_serial sums the copy before PV; vc13_overlap sums during PV. Quantized payloads and P encoding are identical.")
    elif baseline == "vc11":
        print("vc11/optimized fold the ExpCast-only affine transform in FP32. Q/K/V payloads are identical; FP8 P rounding may differ near boundaries. The denominator still sums decoded P; extreme ranges use vc11.")
    elif baseline == "vc10":
        print("vc10/optimized compare ExpCast-only scheduling; vc12+ also folds the affine transform and may change FP8 P rounding. Q/K/V payloads and two-level PV accumulation are unchanged.")
    elif baseline == "vc9":
        print("vc9/optimized change only ExpCast without V smoothing: independent tile sums and history scaling overlapped with PV. denominator FP32 addition order differs; vc12+ can also change FP8 P rounding.")
    elif baseline == "vc7":
        print("vc7/optimized compare D128 channel-V PV scheduling; vc10+ additionally changes ExpCast-only probability scheduling. Quantized Q/K/V payloads are identical.")
    elif baseline == "vc6":
        print("vc6/optimized compare ExpCast scheduling and compact D128 channel-V accumulation; vc10+ changes ExpCast-only history addition order. Quantized payloads are identical.")
    elif getattr(ops.ext,"sage_vc_kernel_version",0) >= 4:
        print("Channel V: vc4 applies the shared scale per tile; optimized applies it once in the epilogue. Both retain two-level FP32 accumulation.")
    print(f"{'variant':<23} {baseline+' ms':>11} {'optimized ms':>14} {'old/new':>10} {'output rel RMSE %':>19} {'max abs':>12}")
    for name in fixed:
        old, new = stats(times[name,baseline]), stats(times[name,"optimized"])
        ratio = old["median_ms"]/new["median_ms"]
        rows[name] = {baseline:old,"optimized":new,"speedup":ratio,"output_difference":errors[name]}
        print(f"{name:<23} {old['median_ms']:>11.4f} {new['median_ms']:>14.4f} {ratio:>9.3f}x "
              f"{errors[name]['rel_rmse']*100:>19.6f} {errors[name]['max_abs']:>12.8f}")
    print("Accuracy compares ALL output rows on identical quantized tensors; 0.1% is an implementation regression gate, not model quality.")
    return {"variants":rows,"baseline_mode":baseline,"trial_order":orders,"implementation_rel_rmse_limit":.001,
            "scope":"Same process/payload; synchronized native API wall time; includes dispatch/allocation; no preprocessing or dense FP32 oracle."}


@torch.no_grad()
def compare_probability_schedules(ops, fixed, args, device):
    """Correct-output full kernels: separate copy cost from overlap benefit."""
    if getattr(ops.ext, "sage_vc_kernel_version", 0) < 10:
        raise RuntimeError("Probability schedule comparison needs vc13")
    # Older installed vc13 exposes this experiment through optimized.
    overlap_mode = "vc13_overlap" if getattr(ops.vc,"DEFAULT_EXPCAST_KERNEL",None) in ("vc12", "vc14", "vc15") else "optimized"
    modes = ("vc12", "vc13_serial", overlap_mode)
    fixed = {name: state for name, state in fixed.items() if not state["block_values"]}
    calls = {(name, mode): lambda state=state, mode=mode:
             ops.vc.forward_prepared(state, kernel_mode=mode)
             for name, state in fixed.items() for mode in modes}
    for call in calls.values():
        for _ in range(args.warmup):
            call()
    torch.cuda.synchronize(device)
    errors = {}
    for name in fixed:
        reference = calls[name, "vc12"]()
        for mode in modes[1:]:
            errors[name, mode] = output_difference(calls[name, mode](), reference)
            if errors[name, mode]["rel_rmse"] > .001:
                raise AssertionError(f"{mode}/vc12 relative RMSE exceeds 0.1%: {name}, {errors[name, mode]}")
        del reference
    times, orders = {key: [] for key in calls}, []
    rng = random.Random(args.seed + 13)
    for trial in range(args.trials):
        # Keep each input's three schedules adjacent; randomize both inputs
        # and schedule order to reduce drift bias. Save every raw observation.
        names = list(fixed)
        rng.shuffle(names)
        order = []
        for name in names:
            shuffled = list(modes)
            rng.shuffle(shuffled)
            for mode in shuffled:
                key = (name, mode)
                order.append(key)
                _, elapsed = measure_trial(calls[key], args.repeats, device)
                times[key].append(elapsed)
        orders.append(order)
        print(f"Probability schedule trial {trial+1}/{args.trials} complete", flush=True)
    rows = {}
    print("\nFull native API, identical payloads: vc12 / serial shared-copy sum / sum during PV (ms)")
    print(f"{'variant':<23} {'vc12':>10} {'serial':>10} {'overlap':>10} {'vc12/new':>10} {'serial/new':>12} {'new RMSE %':>12}")
    for name in fixed:
        measurements = {mode: stats(times[name, mode]) for mode in modes}
        baseline, serial, overlap = (measurements[mode]["median_ms"] for mode in modes)
        rows[name] = {"timings": measurements,
                      "output_vs_vc12": {mode: errors[name, mode] for mode in modes[1:]},
                      "vc12_over_overlap": baseline/overlap,
                      "serial_over_overlap": serial/overlap,
                      "paired_vc12_over_overlap": [a/b for a, b in zip(times[name, "vc12"], times[name, overlap_mode])],
                      "paired_serial_over_overlap": [a/b for a, b in zip(times[name, "vc13_serial"], times[name, overlap_mode])]}
        print(f"{name:<23} {baseline:>10.4f} {serial:>10.4f} {overlap:>10.4f} {baseline/overlap:>9.3f}x "
              f"{serial/overlap:>11.3f}x {errors[name, overlap_mode]['rel_rmse']*100:>12.6f}")
    print("serial/new tests overlap with the SAME shared-copy path; vc12/new includes the copy cost.")
    print("These are full correct-output native APIs, not isolated sum durations. All output rows checked; preprocessing excluded.")
    return {"overlap_mode": overlap_mode, "variants": rows, "trial_order": orders, "implementation_rel_rmse_limit": .001,
            "scope": "Same original and quantized inputs; full native API wall time, no preprocessing; three compiled schedules; timings are not additive component costs."}


def print_report(report):
    print("\nUninstrumented API wall time and fixed-payload native API wall time (ms):")
    print(f"{'variant':<23} {'full API':>12} {'fixed core API':>16}")
    for name, row in report["end_to_end"].items():
        fixed = report["fixed_payload_core"].get(name)
        value = f"{fixed['median_ms']:.4f}" if fixed else "--"
        print(f"{name:<23} {row['median_ms']:>12.4f} {value:>16}")
    print("\nStaged pipeline: CUDA-event intervals in ms (includes stream gaps; not pure kernel durations):")
    print(f"{'variant':<23}" + "".join(f"{name:>13}" for name in STAGES))
    for name, row in report["pipeline"].items():
        values = [f"{row['stages'][s]['cuda_event_ms']['median_ms']:.4f}" if s in row["stages"] else "--" for s in STAGES]
        print(f"{name:<23}" + "".join(f"{value:>13}" for value in values))
    print("\nFixed-payload ExpCast comparison; identical Q/K/V payloads within each pair:")
    for plain, cast in (("sage_vc_control","sage_vc_expcast"),("sage_vc_smooth","sage_vc")):
        if plain not in report["fixed_payload_core"]:
            continue
        a, b = (report["fixed_payload_core"][key]["median_ms"] for key in (plain,cast))
        print(f"  {plain} -> {cast}: {a:.4f} -> {b:.4f} ms, delta {b-a:+.4f} ms, plain/cast={a/b:.4f}x")
    print("CPU enqueue times and raw trials are in JSON. Do not add CPU and GPU times.")
    print("Fixed core excludes all preprocessing and still includes native dispatch/allocation. It is NOT end-to-end speedup.")


@torch.no_grad()
def run(args):
    device = validate_device(args)
    torch.backends.cuda.matmul.allow_tf32 = False
    from sageattention_local import sageattn, sage_vc_build_info
    info, ops = sage_vc_build_info(), load_ops()
    if args.compare_probability_schedules and getattr(ops.ext,"sage_vc_kernel_version",0) < 10:
        raise RuntimeError("--compare-probability-schedules needs vc13; rebuild with bash scripts/build_sage_local.sh")
    if args.compare_kernels and getattr(ops.ext,"sage_vc_kernel_version",0) < 3:
        raise RuntimeError("--compare-kernels needs vc5; rebuild with bash scripts/build_sage_local.sh")
    if args.compare_kernels:
        kernel_baseline(ops,args)
    if args.compare_sage_core and getattr(ops.ext,"sage_vc_kernel_version",0) < 12:
        raise RuntimeError("--compare-sage-core needs vc15; rebuild with bash scripts/build_sage_local.sh")
    if args.compare_sage_layout and getattr(ops.ext,"sage_vc_kernel_version",0) < 13:
        raise RuntimeError("--compare-sage-layout needs vc16; rebuild with bash scripts/build_sage_local.sh")
    if args.compare_probability_candidates and getattr(ops.ext,"sage_vc_kernel_version",0) < 14:
        raise RuntimeError("--compare-probability-candidates needs vc17; rebuild with bash scripts/build_sage_local.sh")
    if args.compare_v_scales and getattr(ops.ext,"sage_vc_kernel_version",0) < 4:
        raise RuntimeError("--compare-v-scales needs vc6; rebuild with bash scripts/build_sage_local.sh")
    if args.compare_grouping:
        require_fused_grouping(ops)
    torch.manual_seed(args.seed)
    dtype = getattr(torch,args.dtype)
    qkv = tuple(torch.randn(args.batch,args.heads,n,args.dim,device=device,dtype=dtype)
                for n in (args.queries,args.tokens,args.tokens))
    if args.ncu_variant:
        cast, smooth = SAGE_VC_VARIANTS[args.ncu_variant]
        state = ops.vc.prepare(*qkv,expcast=cast,v_smooth=smooth)
        for _ in range(args.warmup):
            ops.vc.forward_prepared(state)
        torch.cuda.synchronize(device)
        print(f"NCU: one warmed {args.ncu_variant} core, BHND={list(qkv[0].shape)}; preprocessing excluded",flush=True)
        torch.cuda.profiler.start()
        try:
            output = ops.vc.forward_prepared(state)
            torch.cuda.synchronize(device)
        finally:
            torch.cuda.profiler.stop()
        output_hnd(output,"BHND",qkv[0].shape,dtype)
        return {"ncu_variant": args.ncu_variant, "build": info, "finite": True}
    environment = {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(device), "device": str(device), "tf32_enabled": False,
                   "input_shapes": input_shapes(args), "build": info}
    if args.dim == 128 and getattr(ops.ext,"sage_vc_kernel_version",0) >= 6:
        environment["compiled_d128_channel_resources"] = getattr(ops.vc,"kernel_resources",ops.ext.kernel_resources)(0 if dtype == torch.float16 else 1)
    print(json.dumps(environment,indent=2),flush=True)
    if args.compare_register_candidates or args.compare_pipeline_candidates:
        print("Initial stage tables use the installed default kernel; the requested vc16/candidate table follows separately.",flush=True)
    bases = {}
    variants = {name: flags for name,flags in SAGE_VC_VARIANTS.items() if not args.expcast_only or not flags[1]}
    for smooth in ((False,) if args.expcast_only else (False,True)):
        print(f"Checking stage/public payload parity, v_smooth={smooth} (outside timing)",flush=True)
        base = ops.vc.prepare(*qkv,expcast=False,v_smooth=smooth)
        staged = staged_prepare(*qkv,ops,DirectStages(),expcast=False,smooth=smooth,fixed_permutation=base["permutation"])
        assert_payload_equal(staged,base)
        del staged
        bases[smooth] = base
    fixed = {name:{**bases[smooth],"expcast":cast} for name,(cast,smooth) in variants.items()}
    backends = {name: external_backend(name,qkv,None) for name in ("sage_local",*variants)}
    for name, backend in backends.items():
        print(f"Warmup {name}",flush=True)
        for _ in range(args.warmup):
            output = backend.call()
            if name in fixed:
                ops.vc.forward_prepared(fixed[name])
        output_hnd(output,"BHND",qkv[0].shape,dtype)
    recorder = EventStages(device)
    times = {name: [] for name in backends}
    core_times = {name: [] for name in fixed}
    pipelines = {name: [] for name in fixed}
    rng, orders = random.Random(args.seed), []
    for trial in range(args.trials):
        # Shuffle all API/core jobs together to reduce warmup/clock/order bias.
        jobs = [("api",name) for name in backends] + [("core",name) for name in fixed]
        rng.shuffle(jobs)
        stage_order = list(fixed)
        rng.shuffle(stage_order)
        orders.append({"api_and_core": jobs, "pipeline": stage_order})
        for kind, name in jobs:
            call = backends[name].call if kind == "api" else lambda name=name: ops.vc.forward_prepared(fixed[name])
            output, elapsed = measure_trial(call,args.repeats,device)
            output_hnd(output,"BHND",qkv[0].shape,dtype)
            (times if kind == "api" else core_times)[name].append(elapsed)
            print(f"Trial {trial+1}/{args.trials}: {kind} {name} {elapsed:.4f} ms",flush=True)
        for name in stage_order:
            cast, smooth = SAGE_VC_VARIANTS[name]
            samples = []
            for _ in range(args.repeats):
                recorder.reset()
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                output = staged_forward(qkv,ops,recorder,expcast=cast,smooth=smooth)
                stages = recorder.finish()
                samples.append({"stages": stages,"wall_ms": (time.perf_counter()-start)*1000})
            output_hnd(output,"BHND",qkv[0].shape,dtype)
            pipelines[name].append(samples)
            print(f"Trial {trial+1}/{args.trials}: stages {name} complete",flush=True)
    report = {"environment": environment,"arguments": vars(args),"payload_parity": "passed; fixed permutation; all consumed bytes/scales",
              "end_to_end": {name: {**stats(values),"backend": backends[name].info} for name,values in times.items()},
              "fixed_payload_core": {name: stats(values) for name,values in core_times.items()},
              "pipeline": {name: summarize_stages(values) for name,values in pipelines.items()},"trial_order": orders,
              "notes": ["Public API times are synchronized wall time, with fresh clustering on every smoothed call.",
                        "Staged mirror is verified against installed prepare, using its exact permutation only for the untimed parity check.",
                        "No synchronization between stages. CUDA-event intervals include device idle/host launch gaps.",
                        "CPU enqueue times overlap GPU time; never sum them. Instrumentation changes latency.",
                        "Fixed payloads exclude ALL preprocessing, reuse the same quantized tensors within each ExpCast pair, and still include native host dispatch/output allocation.",
                        "No layout-cache speedup or model-quality claim is made. Optional grouping/V-scale comparisons compute a sampled dense FP32 oracle."]}
    # Write useful timing results before optional CUPTI tracing, which can be unavailable.
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print_report(report)
    print(f"Timing report: {path}",flush=True)
    if args.compare_kernels:
        report["kernel_comparison"] = compare_kernels(ops,fixed,args,device)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if args.compare_sage_core:
        report["sage_core_comparison"] = compare_sage_core(ops,bases[False],args,device)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    layout_calls = None
    if args.compare_sage_layout:
        report["sage_layout_comparison"],layout_calls = compare_sage_layout(ops,bases[False],qkv,args,device,sageattn)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if args.compare_probability_schedules:
        report["probability_schedules"] = compare_probability_schedules(ops,fixed,args,device)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if args.compare_v_scales:
        report["value_scale_comparison"] = compare_v_scales(ops,qkv,bases[True],args,device)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if args.compare_grouping:
        report["grouping_comparison"] = compare_grouping(ops,qkv,args,device)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if args.trace_dir:
        directory = Path(args.trace_dir)
        directory.mkdir(parents=True,exist_ok=True)
        report["traces"] = {}
        for name, (cast,smooth) in variants.items():
            torch.cuda.synchronize(device)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function(name):
                    output = staged_forward(qkv,ops,DirectStages(trace=True),expcast=cast,smooth=smooth)
                torch.cuda.synchronize(device)
            output_hnd(output,"BHND",qkv[0].shape,dtype)
            trace = directory / (name + ".trace.json")
            prof.export_chrome_trace(str(trace))
            summary = trace_summary(json.loads(trace.read_text(encoding="utf-8"))["traceEvents"])
            report["traces"][name] = {"path": str(trace), **summary}
            print(f"Trace {name}: core GPU {summary['core_kernel_ms']:.4f} ms; other kernels {summary['other_kernel_sum_ms']:.4f} ms; {summary['kernel_launches']} launches",flush=True)
            for resource in summary["core_resources"]:
                print(f"  Registers/thread: {resource['registers per thread']}; "
                      f"shared bytes: {resource['shared memory']}; block: {resource['block']}; "
                      f"estimated occupancy: {resource['est. achieved occupancy %']}% "
                      "(estimate, not a hardware utilization counter)", flush=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        if args.compare_sage_core:
            report["sage_core_traces"] = {}
            calls = sage_core_calls(ops,bases[False],*load_sage_core_ops())
            for name,call in calls.items():
                call()
                torch.cuda.synchronize(device)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    output = call()
                    torch.cuda.synchronize(device)
                output_hnd(output,"BHND",qkv[0].shape,dtype)
                trace = directory / (name + ".core.trace.json")
                prof.export_chrome_trace(str(trace))
                markers = ("qk_int8_sv_f8_attn_kernel",) if name.startswith("sage_") else ("sage_vc_attention_kernel",)
                summary = trace_summary(json.loads(trace.read_text(encoding="utf-8"))["traceEvents"],markers)
                report["sage_core_traces"][name] = {"path":str(trace),**summary}
                print(f"Same-payload trace {name}: core GPU {summary['core_kernel_ms']:.4f} ms; {summary['kernel_launches']} launches",flush=True)
                path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        if layout_calls is not None:
            report["sage_layout_kernel_samples"] = trace_layout_samples(layout_calls,args,device)
            path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
            print("\nRepeated CUPTI kernel durations (profiler active; separate from API wall time):")
            for name,row in report["sage_layout_kernel_samples"]["variants"].items():
                print(f"{name:<28} {row['median_ms']:>10.4f} ms",flush=True)
    return report


if __name__ == "__main__":
    run(parse_args())
