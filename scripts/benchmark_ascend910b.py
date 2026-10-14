"""Synchronized API wall time on ONE NPU; no GPU/NPU cross-device speed claims."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics
import time
import torch
from validate_ascend910b import error, to_cpu


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
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
    import torch_npu
    import vc_attention_ascend as vc
    from vc_attention_ascend.reference import payload_reference, dense_reference
    if min(args.queries, args.keys, args.heads, args.repeats, args.trials, args.check_queries) <= 0 or args.warmup < 0:
        parser.error("Lengths, repeat/trial counts and checked rows must be positive")
    torch.npu.set_device(args.device)
    if hasattr(torch.npu, "matmul"):
        torch.npu.matmul.allow_hf32 = False
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator().manual_seed(0)
    original = [torch.randn((1, args.heads, n, args.dim), generator=generator).to(dtype)
                for n in (args.queries, args.keys, args.keys)]
    q, k, v = [x.to(f"npu:{args.device}") for x in original]
    p = vc.prepare(q, k, v, layout="HND")
    ids = torch.linspace(0, args.queries - 1, min(args.queries, args.check_queries)).long().unique()
    print(f"CPU FP32 reference: {len(ids)} query rows, all {args.keys} keys; outside NPU timing", flush=True)
    dense = dense_reference(*original, query_indices=ids)
    refs = {}
    for cast in (False, True):
        print(f"CPU decoded reference: expcast={cast}; outside NPU timing", flush=True)
        refs[cast] = payload_reference(to_cpu(p), expcast=cast, query_indices=ids).to(dtype)
    full = {"vc_expcast": lambda: vc.attention(q, k, v, layout="HND", check_finite=False),
            "vc_control": lambda: vc.attention(q, k, v, layout="HND", expcast=False, check_finite=False)}
    core = {"vc_expcast": lambda: vc.run_prepared(p),
            "vc_control": lambda: vc.run_prepared(p, expcast=False)}
    baseline_error = None
    if hasattr(torch_npu, "npu_fusion_attention"):
        full["cann_fusion_attention"] = lambda: torch_npu.npu_fusion_attention(
            q, k, v, args.heads, "BNSD", scale=1 / math.sqrt(args.dim),
            keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647, sparse_mode=0)[0]
    else:
        baseline_error = "torch_npu.npu_fusion_attention is unavailable"
    report = {"status": "running", "config": vars(args), "torch": torch.__version__,
              "torch_npu": torch_npu.__version__, "vc_version": vc.__version__,
              "device": torch.npu.get_device_name(args.device),
              "shape_nhd": [1, args.queries, args.heads, args.dim],
              "scope": "synchronized API wall ms; full includes fresh preprocessing; fixed core includes tiling/allocation/dispatch/output cast",
              "contract": "INT8 Q32/K128 blocks, FP32 K centering, E4M3 V decoded to FP16, FP16 Cube PV, V smoothing off",
              "finite_scan_in_timing": False, "baseline_error": baseline_error, "accuracy": {}, "timings": {}}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
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
                del full[name]
                save()
                continue
            sample = out.cpu().index_select(2, ids)
            metrics = {"dense_fp32": error(sample, dense)}
            if name.startswith("vc_"):
                decoded = error(sample, refs[name == "vc_expcast"])
                if decoded["relative_rmse"] > .01:
                    raise AssertionError(f"{name} failed decoded-reference gate: {decoded}")
                metrics["decoded_reference"] = decoded
            report["accuracy"][name] = metrics
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
        report.update(status="passed", medians_ms=medians)
        print("variant                    full API ms   fixed core ms   dense RMSE %")
        for name in full:
            fixed = medians["core"].get(name)
            fixed_text = "--" if fixed is None else f"{fixed:.4f}"
            print(f"{name:27s} {medians['full'][name]:11.4f} {fixed_text:>15s} {100 * report['accuracy'][name]['dense_fp32']['relative_rmse']:14.6f}")
        print("Same NPU only. Control/ExpCast share Q/K/V payloads; CANN uses original inputs. No model quality gate.")
        save()
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        save()
        raise
    print("Saved:", path)


if __name__ == "__main__":
    main()
