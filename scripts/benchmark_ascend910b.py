"""Synchronized API wall time on ONE NPU; explicit MindIE-SD Laser dispatch."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import torch
from validate_ascend910b import error, to_cpu
from mindie_laser_backend import load_mindie_laser, validate_laser, distribution_version

BACKENDS = ("vc_expcast", "vc_control", "cann_fusion_attention", "mindie_laser")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--backends", choices=BACKENDS, nargs="+", default=list(BACKENDS))
    parser.add_argument("--require-laser", action="store_true", help="Fail if mindiesd/its native Laser op is unavailable")
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--keys", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--check-queries", type=int, default=32)
    parser.add_argument("--output", default="results/ascend910b_benchmark.json")
    args = parser.parse_args()
    if min(args.queries, args.keys, args.heads, args.repeats, args.trials, args.check_queries) <= 0 or args.warmup < 0:
        parser.error("Lengths, repeat/trial counts and checked rows must be positive")
    if args.require_laser and "mindie_laser" not in args.backends:
        parser.error("--require-laser requires mindie_laser in --backends")
    # Baseline-only runs may use the source reference without installing our .so.
    sys.path.append(str(Path(__file__).resolve().parents[1] / "ascend"))
    from vc_attention_ascend.reference import payload_reference, dense_reference
    report = {"status": "running", "config": vars(args), "torch": torch.__version__,
              "shape_nhd": [1, args.queries, args.heads, args.dim],
              "scope": "synchronized API wall ms, not kernel durations; full includes fresh preprocessing; fixed core excludes input preprocessing",
              "contract": "dense noncausal MHA, original HND inputs and contiguous HND outputs; VC INT8 Q32/K128, E4M3 V decoded to FP16, no V smoothing",
              "finite_scan_in_timing": False, "baseline_error": None, "skipped": {},
              "mindie_laser": None, "accuracy": {}, "timings": {}}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    try:
        import torch_npu
        report["torch_npu"] = torch_npu.__version__
        laser = None
        if "mindie_laser" in args.backends:
            # MindIE sets ASCEND_CUSTOM_OPP_PATH; import before any NPU workloads.
            try:
                laser = load_mindie_laser()
                report["mindie_laser"] = laser.metadata
                print("MindIE Laser:", json.dumps(laser.metadata), flush=True)
            except RuntimeError as exc:
                report["skipped"]["mindie_laser"] = str(exc)
                save()
                if args.require_laser:
                    raise
                print("SKIP mindie_laser:", exc, flush=True)
        torch.npu.set_device(args.device)
        report["device"] = torch.npu.get_device_name(args.device)
        report["cann_root"] = os.environ.get("ASCEND_HOME_PATH")
        if hasattr(torch.npu, "matmul"):
            torch.npu.matmul.allow_hf32 = False
        if laser is not None:
            # Once loaded, execution/correctness failures are fatal even in auto mode.
            report["laser_validation"] = validate_laser(laser, f"npu:{args.device}", dense_reference, error)
            save()
        dtype = getattr(torch, args.dtype)
        generator = torch.Generator().manual_seed(0)
        original = [torch.randn((1, args.heads, n, args.dim), generator=generator).to(dtype)
                    for n in (args.queries, args.keys, args.keys)]
        q, k, v = [x.to(f"npu:{args.device}") for x in original]
        full, core, refs = {}, {}, {}
        ids = torch.linspace(0, args.queries - 1, min(args.queries, args.check_queries)).long().unique()
        if any(name.startswith("vc_") for name in args.backends):
            import vc_attention_ascend as vc
            report["vc_version"] = vc.__version__
            p = vc.prepare(q, k, v, layout="HND")
            vc.load_native()
            for name, cast in (("vc_expcast", True), ("vc_control", False)):
                if name in args.backends:
                    full[name] = lambda cast=cast: vc.attention(q, k, v, layout="HND", expcast=cast, check_finite=False)
                    core[name] = lambda cast=cast: vc.run_prepared(p, expcast=cast)
                    print(f"CPU decoded reference: {name}; outside NPU timing", flush=True)
                    refs[name] = payload_reference(to_cpu(p), expcast=cast, query_indices=ids).to(dtype)
        if "cann_fusion_attention" in args.backends:
            if hasattr(torch_npu, "npu_fusion_attention"):
                full["cann_fusion_attention"] = lambda: torch_npu.npu_fusion_attention(
                    q, k, v, args.heads, "BNSD", scale=1 / math.sqrt(args.dim),
                    keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647, sparse_mode=0)[0].contiguous()
            else:
                report["skipped"]["cann_fusion_attention"] = "torch_npu.npu_fusion_attention is unavailable"
        if laser is not None:
            lp = laser.prepare(q, k, v)
            report["mindie_laser_payload"] = laser.describe_payload(lp)
            full["mindie_laser"] = lambda: laser(q, k, v)
            core["mindie_laser"] = lambda: laser.run_prepared(lp)
        if not full:
            raise RuntimeError("No requested backend is available; see skipped in the JSON report")
        print(f"CPU dense FP32 reference: {len(ids)} query rows, all {args.keys} keys; outside NPU timing", flush=True)
        dense = dense_reference(*original, query_indices=ids)
        for name, fn in list(full.items()):
            print("Warmup", name, flush=True)
            try:
                out = fn()
                for _ in range(args.warmup):
                    out = fn()
                torch.npu.synchronize()
            except Exception as exc:
                if name != "cann_fusion_attention":
                    raise
                report["baseline_error"] = repr(exc)
                report["skipped"][name] = repr(exc)
                del full[name]
                print(f"SKIP {name}: {exc}", flush=True)
                save()
                continue
            if out.shape != q.shape or out.dtype != dtype or out.device != q.device:
                raise AssertionError(f"{name} returned wrong shape/dtype/device")
            cpu_out = out.cpu()
            metrics = {"dense_fp32": error(cpu_out.index_select(2, ids), dense)}
            if name in refs:
                decoded = error(cpu_out.index_select(2, ids), refs[name])
                if decoded["relative_rmse"] > .01:
                    raise AssertionError(f"{name} failed decoded-reference gate: {decoded}")
                metrics["decoded_reference"] = decoded
            if name == "mindie_laser" and metrics["dense_fp32"]["relative_rmse"] > .01:
                raise AssertionError(f"Laser dense-reference implementation gate failed: {metrics}")
            if name in core:
                for _ in range(max(1, args.warmup)):
                    core_out = core[name]()
                torch.npu.synchronize()
                consistency = error(core_out.cpu(), cpu_out)
                if consistency["relative_rmse"] > .001:
                    raise AssertionError(f"{name} full/prepared mismatch: {consistency}")
                metrics["full_vs_prepared_all_rows"] = consistency
            report["accuracy"][name] = metrics
            save()
        if not full:
            raise RuntimeError("No requested backend is available; see skipped in the JSON report")
        calls = [("full", n, f) for n, f in full.items()] + [("core", n, f) for n, f in core.items()]
        rng = random.Random(91023)
        for trial in range(args.trials):
            rng.shuffle(calls)
            for scope, name, fn in calls:
                torch.npu.synchronize()
                start = time.perf_counter()
                for _ in range(args.repeats):
                    fn()
                torch.npu.synchronize()
                ms = (time.perf_counter() - start) * 1000 / args.repeats
                report["timings"].setdefault(scope, {}).setdefault(name, []).append(ms)
                print(f"Trial {trial + 1}/{args.trials}: {scope} {name}: {ms:.4f} ms", flush=True)
                save()
        medians = {scope: {name: statistics.median(values) for name, values in rows.items()}
                   for scope, rows in report["timings"].items()}
        speedups = {base: {name: medians["full"][base] / ms for name, ms in medians["full"].items()}
                    for base in ("mindie_laser", "cann_fusion_attention") if base in medians["full"]}
        report.update(status="passed", medians_ms=medians, full_api_speedups=speedups,
                      mindiesd_version=distribution_version("mindiesd"))
        print("variant                    full API ms   fixed core ms   Laser/API    CANN/API   dense RMSE %")
        for name in full:
            fixed = medians.get("core", {}).get(name)
            fixed_text = "--" if fixed is None else f"{fixed:.4f}"
            ratios = [f"{speedups[b][name]:.3f}x" if b in speedups else "--" for b in ("mindie_laser", "cann_fusion_attention")]
            print(f"{name:27s} {medians['full'][name]:11.4f} {fixed_text:>15s} {ratios[0]:>11s} {ratios[1]:>11s} {100 * report['accuracy'][name]['dense_fp32']['relative_rmse']:14.6f}")
        print("Laser/API and CANN/API = baseline FULL API / row FULL API; >1 means row is faster. No kernel/API-time mixing.")
        print("Laser uses explicit mindiesd::la, dense high-precision mode; BF16->FP16 and padding included in full API.")
        print("Fixed core reuses each backend's own payload, not identical quantized operands across algorithms. No model-quality claim.")
        save()
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        save()
        raise
    print("Saved:", path)


if __name__ == "__main__":
    main()
