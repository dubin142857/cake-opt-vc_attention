"""Isolated TileLang optimization experiment: INT8 QK, direct ExpCast, FP8 PV.

Uses the installed Sage quantizer only; never builds/loads the paper C++ JIT or
VC attention extension. Original and optimized kernels share natural-order V.
"""
import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace

import torch

from benchmark_cuda import fp32_reference, measure_trial, output_hnd, query_indices
from kernel_timing import profile_kernels, run_window
from profile_int8_h200 import load_inputs, make_flash4, metrics
from profile_sage_vc import load_sage_core_ops, output_difference, sage_core_calls, stats
import tilelang_expcast as baseline_backend
import tilelang_expcast_opt as optimized_backend


CANDIDATES = {
    "cached_s1": (1, "cached"), "cached_s2": (2, "cached"),
    "mma_qk_s1": (1, "mma_qk"), "mma_qk_s2": (2, "mma_qk"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("batch", 1), ("heads", 28), ("queries", 37741),
                          ("tokens", 37741), ("dim", 128), ("trials", 5),
                          ("repeats", 10), ("seed", 0), ("check-queries", 128),
                          ("kernel-trials", 3)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variants", nargs="+", choices=tuple(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--include-overlap", action="store_true", help="Also run original async PV schedules")
    parser.add_argument("--qkv-file", type=Path)
    parser.add_argument("--qkv-layout", choices=("NHD", "HND"), default="NHD")
    parser.add_argument("--skip-sage", action="store_true")
    parser.add_argument("--skip-flash4", action="store_true")
    parser.add_argument("--skip-cupti", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--kernel-warmup-seconds", type=float, default=2.)
    parser.add_argument("--kernel-seconds", type=float, default=1.)
    parser.add_argument("--trace-dir", type=Path, default=Path("results/tilelang_opt_traces"))
    parser.add_argument("--output", type=Path, default=Path("results/h200_tilelang_opt.json"))
    args = parser.parse_args(argv)
    if any(getattr(args, name) < 1 for name in ("batch", "heads", "queries", "tokens", "trials", "repeats", "kernel_trials")):
        parser.error("Dimensions and repeat/trial counts must be positive")
    if args.dim not in (64, 128) or args.check_queries < 0:
        parser.error("D64/128 and nonnegative reference query count required")
    if any(not 0 < value < float("inf") for value in (args.kernel_seconds, args.kernel_warmup_seconds)):
        parser.error("Timing windows must be positive finite seconds")
    args.variants = list(dict.fromkeys(args.variants))
    return args


def configurations(args):
    configs = {"baseline_s1": (baseline_backend, 1, "baseline"),
               "baseline_s2": (baseline_backend, 2, "baseline")}
    configs.update({name: (optimized_backend, *CANDIDATES[name]) for name in args.variants})
    if args.include_overlap:
        configs.update({f"overlap_s{stage}": (baseline_backend, stage, "overlap") for stage in (1, 2)})
    return configs


def implementations(configs):
    return {name: lambda state, backend=backend, stage=stage, variant=variant:
            backend.forward_prepared(state, stage, variant)
            for name, (backend, stage, variant) in configs.items()}


def oracle_state(natural, reference):
    """Untimed storage adaptation only; decoded oracle expects Sage's V order."""
    permutation = reference.value_permutation(natural["v"].shape[-1], natural["v"].device)
    inverse = torch.argsort(permutation)
    v = natural["v"].view(torch.uint8).index_select(-1, inverse).view(torch.float8_e4m3fn)
    return {**natural, "v": v, "v_layout": "sage_permuted"}


def check_value_payload(v, natural, reference):
    """Independently run original V preprocessing; no VC native extension."""
    from sageattention_local.quant import per_channel_fp8
    padding = (-v.shape[2]) % 128
    if padding:
        v = torch.cat((v, torch.zeros(*v.shape[:2], padding, v.shape[-1], device=v.device, dtype=v.dtype)), dim=2)
    sage_v, sage_scale, _ = per_channel_fp8(v, tensor_layout="HND", smooth_v=False)
    decoded_bytes = sage_v.view(torch.uint8).index_select(-1, reference.value_permutation(sage_v.shape[-1], v.device))
    torch.testing.assert_close(decoded_bytes, natural["v"].view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(sage_scale, natural["vs"], rtol=0, atol=0)
    return {"logical_v_bytes_exact": True, "v_scales_exact": True, "outside_timing": True}


def save_source(state, name, configuration, directory):
    backend, stage, variant = configuration
    batch, heads, queries, dim = state["q"].shape
    dtype = "float16" if state["output_type"] == 0 else "bfloat16"
    keys = state["k"].shape[2]
    kernel = backend.compile_kernel(batch, heads, queries, keys, dim, dtype, stage, variant)
    path = directory / f"{name}_{dtype}_b{batch}_h{heads}_m{queries}_n{keys}_d{dim}.cu"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(kernel.get_kernel_source(), encoding="utf-8")
    return str(path)


def uniform_inputs(dtype, dim, keys, device):
    q = torch.zeros(1, 2, 65, dim, device=device, dtype=dtype)
    k = torch.zeros(1, 2, keys, dim, device=device, dtype=dtype)
    token = torch.arange(keys, device=device).view(1, 1, keys, 1)
    channel = torch.arange(dim, device=device).view(1, 1, 1, dim)
    head = torch.arange(2, device=device).view(1, 2, 1, 1)
    v = (1 + 2 * (token // 128) + (token % 16) / 16 + (channel % 8) / 8 + head / 4).to(dtype)
    return q, k, v


def gate(row, key, limit, label):
    if row[key]["relative_rmse"] > limit:
        row["status"] = "failed"
        raise AssertionError(f"{label}: {key} relative RMSE exceeds {100 * limit:g}%: {row[key]}")


def all_row_metrics(actual, expected):
    result = output_difference(actual, expected)
    return {"relative_rmse": result["rel_rmse"], "max_abs": result["max_abs"],
            "checked_queries": result["checked_queries"], "exact_match": result["exact_match"]}


@torch.no_grad()
def validate(configs, reference, device, report, save, source_dir):
    calls = implementations(configs)
    validation = report["validation"] = {"status": "running", "cases": [],
        "gates": "Baseline/decoded oracle <=1%; candidate/stage-matched baseline <=0.1%; non-default stream exact"}
    generator = torch.Generator(device=device).manual_seed(3333)
    for dtype in (torch.float16, torch.bfloat16):
        for dim in (64, 128):
            cases = [("random", m, n) for m, n in ((1, 1), (65, 129), (193, 256))]
            cases += [("uniform_tiles", 65, n) for n in (128, 129, 257)]
            for pattern, queries, keys in cases:
                row = {"pattern": pattern, "dtype": str(dtype), "dim": dim,
                       "queries": queries, "keys": keys, "status": "running", "variants": {}}
                validation["cases"].append(row)
                save()
                if pattern == "uniform_tiles":
                    qkv = uniform_inputs(dtype, dim, keys, device)
                else:
                    q = torch.randn(1, 2, queries, dim, device=device, dtype=dtype, generator=generator)
                    k = torch.randn(1, 2, keys, dim, device=device, dtype=dtype, generator=generator)
                    v = torch.randn(k.shape, device=device, dtype=dtype, generator=generator)
                    if keys == 1:
                        q.zero_(); k.zero_()
                    qkv = q, k, v
                natural = baseline_backend.prepare(*qkv)
                row["payload_check"] = check_value_payload(qkv[2], natural, reference)
                oracle = reference.reference(oracle_state(natural, reference), chunk=32).to(dtype)
                if pattern == "uniform_tiles":
                    mean = (natural["v"].float()[..., :keys].mean(-1) * natural["vs"]).unsqueeze(2).expand_as(oracle).to(dtype)
                    torch.testing.assert_close(oracle, mean, rtol=.001, atol=.001)
                baselines = {}
                for name, call in calls.items():
                    print("Validate", name, pattern, str(dtype), dim, queries, keys, flush=True)
                    error = row["variants"][name] = {"status": "running"}
                    validation["active_variant"] = name
                    save()
                    error["generated_source"] = save_source(natural, name, configs[name], source_dir)
                    out = call(natural)
                    error["decoded_reference"] = metrics(out, oracle)
                    save()
                    gate(error, "decoded_reference", .01, name)
                    stage = configs[name][1]
                    if name.startswith("baseline"):
                        baselines[stage] = out
                    else:
                        error["baseline_all_rows"] = all_row_metrics(out, baselines[stage])
                        save()
                        gate(error, "baseline_all_rows", .001, name)
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        side = call(natural)
                    stream.synchronize()
                    error["non_default_stream"] = all_row_metrics(side, out)
                    save()
                    torch.testing.assert_close(side, out, rtol=0, atol=0)
                    error["status"] = "passed"
                    save()
                row["status"] = "passed"
                save()
    validation.pop("active_variant", None)
    validation["status"] = "passed"
    save()


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    configs = configurations(args)
    report = {"status": "starting", "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "contract": "INT8 QK; direct paper ExpCast FP32 math/RNE; FP8 PV; decoded-P denominator; V smoothing OFF",
        "tilelang_v_layout": baseline_backend.V_LAYOUT,
        "isolation": "No paper_expcast_native or VC C++ extension; uses existing installed quantizers and original Sage baseline",
        "configurations": {name: {"backend": backend.__name__, "stages": stage, "variant": variant}
                           for name, (backend, stage, variant) in configs.items()}}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def update(key, value):
        report[key] = value
        save()

    save()
    try:
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise RuntimeError("Hopper required")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(args.seed)
        import tilelang
        from sageattention_local import vc_reference
        if tilelang.__version__.split("+")[0] != "0.1.15":
            raise RuntimeError("Use the existing TileLang 0.1.15 environment")
        report["environment"] = {"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
                                 "cuda": torch.version.cuda, "tilelang": tilelang.__version__}
        save()
        validate(configs, vc_reference, device, report, save, args.trace_dir / "validation_sources")
        if args.validate_only:
            report["status"] = "validation_passed"
            save()
            print("Validation passed:", args.output)
            return
        qkv, report["inputs"] = load_inputs(args, device)
        report["args"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        natural = baseline_backend.prepare(*qkv)
        sage_state = oracle_state(natural, vc_reference)
        report["payload_check"] = check_value_payload(qkv[2], natural, vc_reference)
        calls = implementations(configs)
        core = {name: lambda call=call: call(natural) for name, call in calls.items()}
        full = {name: lambda call=call: call(baseline_backend.prepare(*qkv)) for name, call in calls.items()}
        if not args.skip_sage:
            from sageattention_local import sageattn
            sage_native, _ = load_sage_core_ops()
            core["sage_cpp"] = sage_core_calls(SimpleNamespace(), sage_state, sage_native, sage_native)["sage_cpp"]
            full["sage_cpp"] = lambda: sageattn(*qkv, tensor_layout="HND", is_causal=False)
        if not args.skip_flash4:
            fa4 = make_flash4(qkv)
            report["flash4"] = fa4.info
            full["flash4"] = fa4.call
        ids = query_indices(args.queries, args.check_queries)
        index = torch.tensor(ids, device=device)
        golden = fp32_reference(*qkv, ids, 32)
        oracle = vc_reference.reference(sage_state, query_indices=ids, chunk=32).to(qkv[0].dtype)
        report["accuracy_scope"] = {"dense_reference_queries": len(ids), "all_keys": args.tokens,
            "same_algorithm_comparison": "All main-shape query rows, matched pipeline stage, <=0.1% relative RMSE",
            "decoded_reference": "Sampled main-shape rows, <=1%; all rows in all 24 validation cases",
            "dense_reference": "Diagnostic only; implementation checks are not model-quality gates"}
        report["accuracy"], report["tilelang_sources"] = {}, {}
        baselines = {}
        for name, call in full.items():
            print("Check main shape", name, flush=True)
            row = report["accuracy"][name] = {"status": "running"}
            save()
            if name in configs:
                report["tilelang_sources"][name] = save_source(natural, name, configs[name], args.trace_dir / "main_sources")
                save()
            out = output_hnd(call(), "BSHD" if name == "flash4" else "BHND", qkv[0].shape, qkv[0].dtype)
            row["dense_fp32"] = metrics(out.index_select(2, index).float().cpu(), golden)
            if name in core:
                prepared = output_hnd(core[name](), "BHND", qkv[0].shape, qkv[0].dtype)
                row["full_vs_prepared_all_rows"] = all_row_metrics(out, prepared)
                save()
                torch.testing.assert_close(out, prepared, rtol=0, atol=0)
                del prepared
            if name in configs:
                row["decoded_reference"] = metrics(out.index_select(2, index), oracle)
                save()
                gate(row, "decoded_reference", .01, name)
                stage = configs[name][1]
                if name.startswith("baseline"):
                    baselines[stage] = out
                else:
                    row["baseline_all_rows"] = all_row_metrics(out, baselines[stage])
                    save()
                    gate(row, "baseline_all_rows", .001, name)
            row["status"] = "passed"
            save()
        del baselines, out
        report["status"] = "accuracy_passed"
        save()
        # Compile, check, and synchronize all implementations before timing.
        for call in full.values():
            run_window(call, args.kernel_warmup_seconds, device)
        samples = {"full": {name: [] for name in full}, "core": {name: [] for name in core}}
        report["trial_order"] = []
        rng = random.Random(args.seed + 3340)
        for trial in range(args.trials):
            order = [(scope, name) for scope, group in samples.items() for name in group]
            rng.shuffle(order)
            report["trial_order"].append(order)
            for scope, name in order:
                _, elapsed = measure_trial((full if scope == "full" else core)[name], args.repeats, device)
                samples[scope][name].append(elapsed)
                report["api_samples"] = samples
                update("api", {s: {n: stats(v) for n, v in group.items() if v}
                               for s, group in samples.items()})
            print(f"Trial {trial + 1}/{args.trials} complete", flush=True)
        report["schedule_comparison"] = {}
        for name in configs:
            if name.startswith("baseline"):
                continue
            base = f"baseline_s{configs[name][1]}"
            report["schedule_comparison"][name] = {
                scope + "_baseline_over_candidate": report["api"][scope][base]["median_ms"] / report["api"][scope][name]["median_ms"]
                for scope in ("full", "core")}
        print("Synchronized API time, ms; full includes fresh preprocessing, core reuses prepared payloads:")
        print(f"{'variant':<20} {'full API':>12} {'core API':>12} {'Sage/API':>12} {'FA4/API':>12}")
        for name in full:
            elapsed = report["api"]["full"][name]["median_ms"]
            fixed = f"{report['api']['core'][name]['median_ms']:.4f}" if name in core else "--"
            ratios = [f"{report['api']['full'][base]['median_ms'] / elapsed:.3f}x" if base in full else "--"
                      for base in ("sage_cpp", "flash4")]
            print(f"{name:<20} {elapsed:>12.4f} {fixed:>12} {ratios[0]:>12} {ratios[1]:>12}")
        print("Speed ratios >1 mean faster; full API and CUPTI GPU durations are reported separately.")
        save()
        if not args.skip_cupti:
            capture = {**core, **({"flash4": full["flash4"]} if "flash4" in full else {})}
            profile_kernels(capture, args, device, on_result=lambda value: update("kernel_windows", value),
                kernel_markers={name: ("vc_tilelang",) for name in configs})
        report["status"] = "passed"
        save()
        print("Saved:", args.output)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
