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
VARIANTS = {f"{mode}_m{tile}": (cast, tile)
            for mode, cast in (("control", False), ("expcast", True))
            for tile in (32, 64)}

#  ====================================================================================
#  decoded_reference 闸门的修订 (2026-10-27, 经用户确认)
#  ------------------------------------------------------------------------------------
#  原判据: 内核输出与同契约 oracle (payload_reference) 的相对 RMSE < 1%, control 与
#  expcast 一视同仁。该判据在 S 为 int32 时工作良好 (实测 control 6.8e-5 / expcast 4.3e-5)。
#
#  **为什么必须改**: 当 QK 改为经 fixpipe 反量化直出 fp16 (S 由 int32 变 fp16) 后,
#  expcast 路径的该指标升到 1.1089%, 越过 1% 闸门。逐项排查后确认这**不是质量退化**:
#
#    1. S 本身是吻合的。用 V=单位阵把内核实际算出的 S 提取出来逐元素比对, 单 tile 下
#       与 oracle 的差分 RMS 仅 3.6e-4 (相对 S 的跨度 5.5 是 6.6e-5), 与 fp16 舍入噪声一致。
#    2. 失配不随 tile 数累积。keys 由 256 到 32768 (1 到 128 个 tile) 全程平坦。
#    3. 机理是 **ExpCast 的 E4M3 硬量化放大 S 的末位扰动**: E4M3 只有 121 个离散码,
#       相邻码相对步长 6.2%。把 S 扰动一个 fp16 ulp, 1.662% 的码发生跳变, 输出相对 RMSE
#       被放大 11.3 倍 (半个 ulp 时 16.8 倍); 普通 exp 是光滑函数, 几乎不受影响
#       (同一扰动下 exp 路径只有 5.7e-4)。
#    4. 对**真实 dense attention** 的质量未退: 9 个点 (keys 1024/2048/4096 x 种子 1/2/3,
#       每点 256 个 query 行) 实测 expcast 由 0.043676 变为 0.043786, 即误差预算的 +0.25%;
#       control 为 +0.01%。而该误差预算由 E4M3 的 ExpCast 量化主导 (4.37%)。
#
#  已逐一排除的其它解释: 舍入次数 (fp32 两次 / fp64 一次 / 完全不舍入三者几乎相同)、
#  按通道反量化向量的编码 (SetQuantScalar 与 SetQuantVector 给出完全相同的数字)、
#  反量化系数的尾数截断 (scale 取 2 的幂后失配只降到 0.001498 而非消失)、
#  尾数受限的中间表示 (中间尾数降到 11/10/9/8/7 位时失配单调变差, fp32 中间为最优拟合)。
#
#  结论: 该闸门在 fp16-S 契约下**实质要求逐位复现 S**, 而硬件 fixpipe 的舍入与
#  torch 的 .half() 在最后一位偶有分歧, ExpCast 恰好把这些分歧放大 —— 结构上不可达。
#
#  **为什么不能直接删掉**: 它同时负责抓真实缺陷 (错的 scale、错位的块索引、算错的 tile)。
#  故改为三条合取, 真实缺陷仍会被其中至少一条抓住:
#
#    (a) control 路径仍守 1% —— 它是光滑路径, 不受 ExpCast 放大影响, 判据原样不动;
#    (b) expcast 路径改守"相对 control 的放大倍数 <= 15" —— 机理给出 11.3 至 16.8 倍,
#        留约一倍余量。真实缺陷会同时破坏 control 与该比值;
#    (c) 新增 dense_fp32 对既有基线的回归检查 —— 这是唯一直接对应模型质量的量。
#
#  ★留痕要求来自用户 (2026-10-27): "放宽可以, 但是务必留痕, 说明愿意与必要性"。
#  ★本修订**只针对 expcast 路径的 decoded_reference**; control 的 1% 与其余所有闸门不动。
#  ====================================================================================
DECODED_LIMIT = .01          # control 路径: 与同契约 oracle 的相对 RMSE 上限 (未改动)
EXPCAST_AMPLIFY_LIMIT = 15.  # expcast 路径: 相对 control 的放大倍数上限 (机理 11.3-16.8 倍)
#  dense_fp32 的回归基线: int32-S 实现 (h10/h11) 实测值。超过 1.05 倍即判回归。
#  实测离散度 (9 个点): control 0.029105-0.030419, expcast 0.043084-0.044468。
DENSE_BASELINE = {"control": .029799, "expcast": .043645}
DENSE_REGRESSION_FACTOR = 1.05


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
                    for tile in (32, 64):
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
        #  control 的 decoded_reference 要先测出来, 才能判 expcast 的放大倍数;
        #  full 的迭代顺序由 VARIANTS 决定, control 在前。此处显式断言该前提。
        decoded_control = {}
        assert list(VARIANTS).index("control_m32") < list(VARIANTS).index("expcast_m32"), \
            "VARIANTS 的顺序必须让 control 先于 expcast"
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
                mode = "expcast" if cast else "control"
                #  (a) control 仍守 1%; (b) expcast 改守相对 control 的放大倍数。
                #  判据的来由与必要性见文件头 VARIANTS 之后的注释块。
                if not cast:
                    metric = gate(output.index_select(2, ids), refs[cast], limit=DECODED_LIMIT,
                                  label="decoded_reference", row=row, save=save)
                    decoded_control[f"m{tile}"] = metric["relative_rmse"]
                else:
                    metric = error(output.index_select(2, ids), refs[cast])
                    row["decoded_reference"] = metric
                    base = decoded_control.get(f"m{tile}")
                    if base is None:
                        raise AssertionError(
                            f"{name}: control 的 decoded_reference 尚未测得, 无法判放大倍数 —— "
                            "control 必须先于 expcast 跑")
                    ratio = metric["relative_rmse"] / max(base, 1e-12)
                    row["decoded_amplification"] = {"ratio": ratio, "control": base,
                                                    "limit": EXPCAST_AMPLIFY_LIMIT}
                    save()
                    if ratio > EXPCAST_AMPLIFY_LIMIT:
                        raise AssertionError(
                            f"{name}: ExpCast 相对 control 的放大倍数 {ratio:.1f} 超过 "
                            f"{EXPCAST_AMPLIFY_LIMIT:g} (control {base:.6f}, expcast "
                            f"{metric['relative_rmse']:.6f}): {metric}")
                #  (c) dense_fp32 对既有基线的回归检查 —— 唯一直接对应模型质量的量。
                dense_limit = DENSE_BASELINE[mode] * DENSE_REGRESSION_FACTOR
                row["dense_regression"] = {"baseline": DENSE_BASELINE[mode], "limit": dense_limit}
                save()
                if row["dense_fp32"]["relative_rmse"] > dense_limit:
                    raise AssertionError(
                        f"{name}: dense_fp32 {row['dense_fp32']['relative_rmse']:.6f} 超过基线 "
                        f"{DENSE_BASELINE[mode]:.6f} 的 {DENSE_REGRESSION_FACTOR} 倍 "
                        f"({dense_limit:.6f}) —— 这是对真实 dense attention 的质量回归")
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
        report["tile_comparison"] = {}
        for mode in ("control", "expcast"):
            row = report["tile_comparison"][mode] = {}
            gate(core_outputs[f"{mode}_m64"], core_outputs[f"{mode}_m32"], limit=.001,
                 label="m64_vs_m32_all_rows", row=row, save=save)
            row["exact_match"] = torch.equal(core_outputs[f"{mode}_m64"], core_outputs[f"{mode}_m32"])
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
