"""M32/M64 accuracy, synchronized API timing and sampled Ascend system ticks.

Probe phases drain asynchronous work and perturb the kernel. They are neither
production kernel percentages nor a conversion from ticks to milliseconds.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import torch
from validate_ascend910b import error, reuse_inputs, to_cpu
from vc_attention_ascend.reference import K_TILE
from mindie_laser_backend import load_mindie_laser, validate_laser


ROLE_PHASES = {
    "aic": ("qk_api_complete", "wait_p", "pv_api_complete", "wait_o_used"),
    "aiv0": ("wait_s", "load_scores", "probability", "p_publish", "history",
             "wait_o", "load_pv", "accumulate"),
    "aiv1": ("wait_s", "load_scores", "probability", "p_publish", "history",
             "wait_o", "load_pv", "accumulate"),
}
#  ★tile_m 现在只支持 64 (手写 cube 的几何 M=64 x N=128; ROWS=M/2=32 恰好是一个
#    Q_SCALE_BLOCK, 向量侧的 scale 逻辑依赖这一条)。原来这里是 (32, 64) 的笛卡尔积 ——
#    删 32 支路时漏了这一处, 表现为探针脚本一跑就被 api.py 的校验拒掉。
VARIANTS = {f"{mode}_m{tile}": (cast, tile)
            for mode, cast in (("control", False), ("expcast", True))
            for tile in (64,)}


def summarize_records(records):
    """Aggregate valid sampled records only; keep AIC and both AIVs separate."""
    records = torch.as_tensor(records).detach().cpu()
    if records.ndim != 4 or tuple(records.shape[1:]) != (3, 4, 16):
        raise ValueError("Expected probe records [cores,3,4,16]")
    if records.dtype != torch.int64:
        raise TypeError("Probe records must use int64 system ticks")
    result = {"unit": "GetSystemCycle system ticks", "valid_samples": 0,
              "invalid_slots": 0, "roles": {},
              "scope": "Sampled instrumented task/key tiles; drained phases perturb execution. No ms conversion or production percentages."}
    for role_id, (role, phases) in enumerate(ROLE_PHASES.items()):
        buckets = {kind: {name: [] for name in phases} for kind in ("main", "tail")}
        for row in records[:, role_id].reshape(-1, 16).tolist():
            if row[0] == 0:
                result["invalid_slots"] += 1
                continue
            #  tile 起始 key 必须是 tile 宽度的倍数, 有效 key 数落在 1..宽度。
            #  ★宽度取自 reference.K_TILE, 不写死 —— 它是内核 vc_tiling.h 的 N。
            if (row[0] != 1 or row[1] < 0 or row[2] < 0 or row[2] % K_TILE
                    or not 1 <= row[3] <= K_TILE):
                raise ValueError(f"Malformed valid probe record: {row[:4]}")
            if any(value < 0 for value in row[4:4 + len(phases)]):
                raise ValueError("Negative elapsed system tick count")
            result["valid_samples"] += 1
            kind = "main" if row[3] == K_TILE else "tail"
            for phase, value in zip(phases, row[4:]):
                buckets[kind][phase].append(value)
        result["roles"][role] = {
            kind: {"samples": len(next(iter(values.values()))), "phases": {
                phase: {"median_ticks": statistics.median(ticks),
                        "min_ticks": min(ticks), "max_ticks": max(ticks)}
                for phase, ticks in values.items() if ticks}}
            for kind, values in buckets.items()}
    return result


def gate(actual, expected, *, limit, label, row, save):
    """Persist the measured error before raising; never silently relax a gate."""
    if actual.shape != expected.shape:
        row[label] = {"error": "shape mismatch", "actual_shape": list(actual.shape),
                      "expected_shape": list(expected.shape)}
        save()
        raise AssertionError(f"{label}: output/reference shape mismatch: {row[label]}")
    metric = error(actual, expected)
    row[label] = metric
    save()
    if metric["relative_rmse"] > limit:
        raise AssertionError(f"{label}: RMSE exceeds {100 * limit:g}%: {metric}")
    return metric


@torch.no_grad()
def validate_tiles(vc, reference, device, report, save):
    generator = torch.Generator().manual_seed(91064)
    report["small_validation"] = []
    # The H65 case forces task reuse across heads with sharply differing scales.
    shapes = ((2, 1, 1), (2, 33, 129), (2, 65, 129), (2, 1025, 129), (65, 33, 129))
    for dtype in (torch.float16, torch.bfloat16):
        for dim in (64, 128):
            for heads, queries, keys in shapes:
                if heads == 65:
                    inputs = reuse_inputs(heads, queries, keys, dim, dtype, generator)
                else:
                    inputs = [torch.randn((1, heads, n, dim), generator=generator).to(dtype)
                              for n in (queries, keys, keys)]
                p = vc.prepare(*(x.to(device) for x in inputs), layout="HND")
                cpu_p = to_cpu(p)
                for cast in (False, True):
                    row = {"dtype": str(dtype), "dim": dim, "heads": heads,
                           "queries": queries, "keys": keys, "expcast": cast,
                           "status": "running", "repeat_scope": "cast API output, not native raw FP32 bits"}
                    report["small_validation"].append(row)
                    save()
                    print("Validate M32/M64:", json.dumps(row), flush=True)
                    oracle = reference(cpu_p, expcast=cast).to(dtype)
                    baseline = None
                    for tile in (64,):   # tile_m 现在只支持 64
                        out = vc.run_prepared(p, expcast=cast, tile_m=tile).cpu()
                        gate(out, oracle, limit=.01, label=f"m{tile}_decoded_reference",
                             row=row, save=save)
                        if baseline is None:
                            baseline = out
                        else:
                            gate(out, baseline, limit=.001, label="m64_vs_m32_all_rows", row=row, save=save)
                        again = vc.run_prepared(p, expcast=cast, tile_m=tile).cpu()
                        row[f"m{tile}_repeat_exact"] = torch.equal(out, again)
                        save()
                        if not row[f"m{tile}_repeat_exact"]:
                            raise AssertionError(f"M{tile} fixed-payload cast output changed on repeat")
                        if heads == 2 and queries in (1, 65):
                            # Single-K-tile and main+tail sampling, both dimensions
                            # and both output dtypes, before the long-shape probe.
                            probe_out, records = vc.profile_prepared(p, expcast=cast, tile_m=tile)
                            gate(probe_out.cpu(), out, limit=.001,
                                 label=f"m{tile}_probe_vs_plain_all_rows", row=row, save=save)
                            row[f"m{tile}_probe_records"] = summarize_records(records)
                            save()
                            if row[f"m{tile}_probe_records"]["valid_samples"] == 0:
                                raise AssertionError(f"M{tile} small-shape probe produced no valid samples")
                    row["status"] = "passed"
                    save()


def time_call(fn, repeats):
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--queries", type=int, default=37741)
    parser.add_argument("--keys", type=int, default=37741)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--probe-repeats", type=int, default=2)
    parser.add_argument("--check-queries", type=int, default=32)
    parser.add_argument("--require-laser", action="store_true")
    parser.add_argument("--output", default="results/ascend910b_profile.json")
    args = parser.parse_args()
    if min(args.queries, args.keys, args.heads, args.repeats, args.trials,
           args.probe_repeats, args.check_queries) <= 0 or args.warmup < 0:
        parser.error("Sizes and measurement counts must be positive; warmup must be nonnegative")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "config": vars(args), "shape_nhd": [1, args.queries, args.heads, args.dim],
              "contract": "INT8 Q32/K128 scales unchanged; decoded-E4M3/FP16 V; no V smoothing; M64 is an execution tile candidate",
              "timing_scope": "Uninstrumented synchronized API wall milliseconds; full includes fresh preprocessing; fixed core includes native padding/allocation/dispatch and final cast",
              "probe_scope": "Separate instrumented drained kernel, sampled GetSystemCycle system ticks. No tick-to-ms or production percentage conversion.",
              "gates": {"decoded_reference": .01, "same_algorithm": .001},
              "model_quality_claim": False, "accuracy": {}, "probes": {},
              "timings": {"full": {}, "core": {}}, "skipped": {}}

    def save():
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    try:
        import torch_npu
        import vc_attention_ascend as vc
        from vc_attention_ascend.reference import dense_reference, payload_reference
        # Register MindIE custom operators before creating any NPU tensors.
        laser = None
        try:
            laser = load_mindie_laser()
            report["mindie_laser"] = laser.metadata
        except RuntimeError as exc:
            report["skipped"]["mindie_laser"] = str(exc)
            save()
            if args.require_laser:
                raise
            print("SKIP mindie_laser:", exc, flush=True)
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        torch.npu.set_device(args.device)
        device = f"npu:{args.device}"
        report["environment"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                                 "vc_version": vc.__version__, "device": torch.npu.get_device_name(args.device),
                                 "cann_root": os.environ.get("ASCEND_HOME_PATH")}
        if hasattr(torch.npu, "matmul"):
            torch.npu.matmul.allow_hf32 = False
        vc.load_native()
        validate_tiles(vc, payload_reference, device, report, save)
        if laser is not None:
            report["laser_validation"] = validate_laser(laser, device, dense_reference, error)
            save()
        dtype = getattr(torch, args.dtype)
        generator = torch.Generator().manual_seed(args.seed)
        original = [torch.randn((1, args.heads, n, args.dim), generator=generator).to(dtype)
                    for n in (args.queries, args.keys, args.keys)]
        q, k, v = [x.to(device) for x in original]
        p = vc.prepare(q, k, v, layout="HND")
        ids = torch.linspace(0, args.queries - 1, min(args.queries, args.check_queries)).long().unique()
        print(f"CPU reference: {len(ids)} query rows, ALL {args.keys} keys; outside timing", flush=True)
        dense = dense_reference(*original, query_indices=ids)
        cpu_p = to_cpu(p)
        refs = {cast: payload_reference(cpu_p, expcast=cast, query_indices=ids).to(dtype)
                for cast in (False, True)}
        del cpu_p
        full = {name: (lambda cast=cast, tile=tile: vc.attention(q, k, v, layout="HND",
                expcast=cast, tile_m=tile, check_finite=False)) for name, (cast, tile) in VARIANTS.items()}
        core = {name: (lambda cast=cast, tile=tile: vc.run_prepared(p, expcast=cast, tile_m=tile))
                for name, (cast, tile) in VARIANTS.items()}
        if hasattr(torch_npu, "npu_fusion_attention"):
            full["cann_fusion_attention"] = lambda: torch_npu.npu_fusion_attention(
                q, k, v, args.heads, "BNSD", scale=1 / math.sqrt(args.dim), keep_prob=1.0,
                pre_tockens=2147483647, next_tockens=2147483647, sparse_mode=0)[0].contiguous()
        else:
            report["skipped"]["cann_fusion_attention"] = "torch_npu.npu_fusion_attention is unavailable"
        if laser is not None:
            lp = laser.prepare(q, k, v)
            report["mindie_laser_payload"] = laser.describe_payload(lp)
            full["mindie_laser"] = lambda: laser(q, k, v)
            core["mindie_laser"] = lambda: laser.run_prepared(lp)
        core_outputs = {}
        for name, fn in full.items():
            report["current"] = {"phase": "long_shape_accuracy", "variant": name}
            save()
            print("Validate full/prepared:", name, flush=True)
            native_out = fn()
            if native_out.shape != q.shape or native_out.dtype != dtype or native_out.device != q.device:
                raise AssertionError(f"{name} returned wrong shape/dtype/device")
            output = native_out.cpu()
            del native_out
            row = report["accuracy"][name] = {"dense_fp32": error(output.index_select(2, ids), dense),
                                           "checked_dense_rows": len(ids), "all_keys": args.keys}
            save()
            if name in VARIANTS:
                cast, tile = VARIANTS[name]
                gate(output.index_select(2, ids), refs[cast], limit=.01,
                     label="decoded_reference", row=row, save=save)
            elif row["dense_fp32"]["relative_rmse"] > .01:
                raise AssertionError(f"{name} failed dense FP32 implementation gate: {row}")
            if name in core:
                prepared_out = core[name]().cpu()
                gate(prepared_out, output, limit=.001, label="full_vs_prepared_all_rows", row=row, save=save)
                if name in VARIANTS:
                    core_outputs[name] = prepared_out
                    repeated = core[name]().cpu()
                    row["prepared_repeat_exact"] = torch.equal(prepared_out, repeated)
                    save()
                    if not row["prepared_repeat_exact"]:
                        raise AssertionError(f"{name} fixed-payload cast output changed on repeat")
                    del repeated
            del output
        #  ★原来这里是 m64 对 m32 的交叉校验 (两种 tile 几何必须一致到 0.1%)。
        #    tile_m 现在只剩 64, 那道闸跑不了 —— 但**直接删掉等于削弱检查**,
        #    故换一道等价强度的不变式, 而不是留个空位。
        #  新判据: **control 必须对参考的在线更新窗口不变**。
        #    依据是 reference.py 的文档串原话 "普通 exp 路径对宽度不变, 故只有 ExpCast 受影响" ——
        #    这对 control 是数学恒等 (softmax 对分块归一化不变), 对 expcast 不是
        #    (E4M3 编码相对窗口最大值定义, 见 F190)。
        #  ★所以这道闸只对 control 设限; expcast 只记录差异不设限, 它的差异是**预期的**。
        from vc_attention_ascend.reference import payload_reference, K_TILE, ONLINE_TILE
        report["window_invariance"] = {}
        cpu_p = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in p.items()}
        for mode, cast in (("control", False), ("expcast", True)):
            row = report["window_invariance"][mode] = {}
            a = payload_reference(cpu_p, expcast=cast, tile_width=K_TILE)
            b = payload_reference(cpu_p, expcast=cast, tile_width=ONLINE_TILE)
            diff = (a.double() - b.double())
            rel = float((diff.norm() / a.double().norm().clamp_min(1e-30)).item())
            row.update(tile_width_a=K_TILE, tile_width_b=ONLINE_TILE, relative_diff=rel)
            if mode == "control":
                #  ★闸取 1e-3, 不是 1e-6 —— 自检当场否掉了后者。
                #    control 只在**精确算术**下对窗口不变; 实际是 fp32 重结合,
                #    实测残差 1.37e-4 (账本早已量化过这一项为"约 2e-4")。
                #    闸必须落在重结合残差 (~1.4e-4) 与 expcast 的差 (~2.3e-2) **之间**:
                #    低于前者会恒红, 高于后者就分不开两条路径。
                row["gate"] = 1e-3
                assert rel < 1e-3, (
                    f"control 对参考窗口的相对差 {rel:.3e} 超过 1e-3 —— "
                    f"fp32 重结合只应给出约 1.4e-4; 更大说明归一化不再是数学恒等, "
                    f"或参考的窗口语义变了")
            else:
                row["gate"] = None
                row["note"] = ("expcast 对窗口**不**不变 (实测约 2.3e-2), 这是 E4M3 编码"
                               "相对窗口最大值定义使然 (F190), 故只记录不设限 —— "
                               "它同时是上面那条闸的**负例对照**: 两者差两个量级, 判据分得开")
        # Probe correctness and record schema are also checked before timing.
        for name, (cast, tile) in VARIANTS.items():
            detail = report["probes"][name] = {"trials": []}
            collected_records = []
            for repeat in range(args.probe_repeats):
                report["current"] = {"phase": "probe_accuracy", "variant": name, "repeat": repeat + 1}
                save()
                print(f"Validate probe {repeat + 1}/{args.probe_repeats}: {name}", flush=True)
                probe_out, records = vc.profile_prepared(p, expcast=cast, tile_m=tile)
                row = {"repeat": repeat + 1}
                detail["trials"].append(row)
                gate(probe_out.cpu(), core_outputs[name], limit=.001,
                     label="probe_vs_plain_all_rows", row=row, save=save)
                row["records"] = records.cpu().tolist()
                row["summary"] = summarize_records(records)
                collected_records.append(records.cpu())
                if row["summary"]["valid_samples"] == 0:
                    raise AssertionError(f"{name} produced no valid probe samples")
                save()
            detail["summary"] = summarize_records(torch.cat(collected_records, dim=0))
            detail["summary"]["aggregation"] = "All valid sampled records across probe repetitions; not weighted by whole-kernel duration"
            print("  valid samples:", detail["summary"]["valid_samples"], flush=True)
            save()
        del core_outputs
        report["accuracy_status"] = "passed"
        report["current"] = {"phase": "warmup"}
        save()
        calls = [(scope, name, fn) for scope, functions in (("full", full), ("core", core))
                 for name, fn in functions.items()]
        for scope, name, fn in calls:
            print("Warmup:", scope, name, flush=True)
            for _ in range(args.warmup):
                fn()
            torch.npu.synchronize()
        rng = random.Random(91064)
        for trial in range(args.trials):
            rng.shuffle(calls)
            for scope, name, fn in calls:
                ms = time_call(fn, args.repeats)
                report["timings"][scope].setdefault(name, []).append(ms)
                print(f"Trial {trial + 1}/{args.trials}: {scope} {name}: {ms:.4f} ms", flush=True)
                save()
        medians = {scope: {name: statistics.median(values) for name, values in rows.items()}
                   for scope, rows in report["timings"].items()}
        report["medians_ms"] = medians
        for mode, row in report["tile_comparison"].items():
            row["m32_over_m64"] = {scope: medians[scope][f"{mode}_m32"] / medians[scope][f"{mode}_m64"]
                                     for scope in ("full", "core")}
        print("Uninstrumented synchronized API milliseconds:")
        print("variant                     full API    fixed core   Laser/API    CANN/API   dense RMSE %")
        for name in full:
            ratios = [f"{medians['full'][base] / medians['full'][name]:.3f}x" if base in medians["full"] else "--"
                      for base in ("mindie_laser", "cann_fusion_attention")]
            fixed = f"{medians['core'][name]:.4f}" if name in medians["core"] else "--"
            print(f"{name:26s} {medians['full'][name]:10.4f} {fixed:>13s} {ratios[0]:>11s} {ratios[1]:>11s} "
                  f"{100 * report['accuracy'][name]['dense_fp32']['relative_rmse']:14.6f}")
        print("Ratios use FULL API only. Fixed core includes M64 native padding/allocation; excludes preprocessing.")
        # Calibrate probe perturbation separately using randomized paired API calls.
        for name, (cast, tile) in VARIANTS.items():
            detail = report["probes"][name]
            values = {"plain_ms": [], "probe_ms": []}
            for repeat in range(args.probe_repeats):
                pair = [("plain_ms", core[name]),
                        ("probe_ms", lambda cast=cast, tile=tile: vc.profile_prepared(p, expcast=cast, tile_m=tile))]
                rng.shuffle(pair)
                for label, fn in pair:
                    values[label].append(time_call(fn, 1))
            plain, probe = (statistics.median(values[key]) for key in ("plain_ms", "probe_ms"))
            detail["calibration"] = {**values, "plain_median_ms": plain, "probe_median_ms": probe,
                                     "probe_over_plain": probe / plain,
                                     "scope": "Synchronized API wall time, including probe record allocation; separate from system ticks"}
            print(f"Probe perturbation {name}: plain {plain:.4f}, probe {probe:.4f} ms, ratio {probe / plain:.3f}x")
            for role, buckets in detail["summary"]["roles"].items():
                for kind, data in buckets.items():
                    if data["samples"]:
                        med = {phase: metric["median_ticks"] for phase, metric in data["phases"].items()}
                        print(f"  {role} {kind}, {data['samples']} sampled records, median system ticks: {json.dumps(med)}")
            save()
        report.update(status="passed", current=None)
        save()
        print("Probe ticks are sampled instrumented elapsed time, not core clock cycles or production percentages.")
        print("All accuracy gates passed; these implementation checks do not establish model quality.")
        print("Saved:", path, flush=True)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save()
        raise


if __name__ == "__main__":
    main()
