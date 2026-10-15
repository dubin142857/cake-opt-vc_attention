#!/usr/bin/env python3
"""Isolated installed / unchanged-source / ExpCast Laser experiment on A2/A3."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from types import SimpleNamespace

import benchmark_mindie_laser_local as baseline
from prepare_mindie_laser_source import validate_prepared_source

common = baseline.common


def build_arguments(args, name):
    """Identical native names are safe only in independent worker processes."""
    key = "baseline" if name == "installed" else name
    build_path = Path(getattr(args, key + "_build")).resolve()
    build = json.loads(build_path.read_text(encoding="utf-8"))
    bridge_path = Path(args.bridge_manifest).resolve()
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    return SimpleNamespace(
        vendor_root=build["vendor_root"], build_manifest=str(build_path),
        source_manifest=build["source_manifest"]["path"],
        bridge_path=bridge["bridge_path"], bridge_manifest=str(bridge_path),
        device=args.device, warmup=args.warmup, repeats=args.repeats,
        trials=args.trials, cpu_threads=args.cpu_threads,
        _worker="installed" if name == "installed" else "local",
        _run_dir=str(Path(args._run_dir) / name),
    )


def worker(args):
    name = args._worker
    local_args = build_arguments(args, name)
    validator = None
    if name.startswith("expcast"):
        from mindie_laser_expcast_validation import validate_small
        validator = validate_small
    after_long = None
    if name.endswith("_probe"):
        def after_long(adapter, payload, report, save, torch):
            from mindie_laser_probe_patch import decode_records, TRACE_BYTES
            q, k, v = payload.tensors
            statistics_tensor, output = adapter.native(
                q, k, v, None, None, None, payload.scale, payload.shape[1],
                "BNSD", 1.0, payload.pre_tokens, 1, True)
            torch.npu.synchronize()
            # The original unused statistics output is a DIAGNOSTIC-only sink.
            if statistics_tensor.numel() * statistics_tensor.element_size() < TRACE_BYTES:
                raise AssertionError("Laser statistics buffer is too small for diagnostic records")
            raw = statistics_tensor.flatten()[:TRACE_BYTES // 4].cpu().contiguous().view(torch.uint8).numpy().tobytes()
            report["probability_probe"] = decode_records(raw)
            save()
            del output
    return baseline.run_worker(local_args, small_validator=validator,
                               dense_limit=None if name.startswith("expcast") else .01,
                               after_long=after_long)


def compare_algorithm_samples(first_path, second_path):
    """Reuse input-identity checks; do not apply same-algorithm error thresholds."""
    result = common.compare_samples(first_path, second_path)
    result.pop("passed", None)
    result.pop("limit", None)
    result["scope"] = "Different probability algorithms; sampled output diagnostic, not a regression or model-quality gate"
    return result


def table(report):
    lines = ["", "Laser ExpCast experiment; synchronized wall ms/API (production variants only)",
             "variant          full API    prepared core    baseline/API    dense RMSE %"]
    reference = report["workers"]["baseline"]["timing_ms"]["full"]["median"]
    for name in ("installed", "baseline", "expcast"):
        row = report["workers"][name]
        full = row["timing_ms"]["full"]["median"]
        core = row["timing_ms"]["prepared_core"]["median"]
        error = 100 * row["long_dense_fp32"]["relative_rmse"]
        lines.append(f"{name:<15} {full:>10.4f} {core:>16.4f} {reference/full:>15.4f}x {error:>15.6f}")
    lines += ["ExpCast replaces bulk exp only: original FP16 QK/PV, no V reordering or smoothing.",
              "prepared core includes output conversion/trim/contiguous; excludes fresh input conversion/padding.",
              "Dense error: 32 query rows, all 37741 keys. It is an algorithm diagnostic, not model quality.",
              "Workers run in separate process groups; ratios are not paired/interleaved trials or kernel-only timings."]
    for name in ("baseline_probe", "expcast_probe"):
        if name not in report["workers"]:
            continue
        row = report["workers"][name]
        production = report["workers"][name.removesuffix("_probe")]
        ratio = row["timing_ms"]["prepared_core"]["median"] / production["timing_ms"]["prepared_core"]["median"]
        lines.append(f"Diagnostic {name}: probe/production prepared API={ratio:.4f}x (perturbation, not speedup)")
        lines.append(json.dumps(row["probability_probe"], ensure_ascii=False))
    return "\n".join(lines)


def parent(args):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args._run_dir = str(Path(tempfile.mkdtemp(prefix=output.stem + "_", dir=output.parent)).resolve())
    order = ["installed", "baseline", "expcast"]
    if args.baseline_probe_build:
        order += ["baseline_probe", "expcast_probe"]
    report = {"status": "running", "shape_bnsd": list(baseline.SHAPE), "dtype": "bfloat16",
              "seed": 0, "process_order": order, "artifact_dir": args._run_dir,
              "workers": {}, "provenance": {}, "proof_scope": baseline.PROOF_SCOPE,
              "implementation_contract": "Bulk probability FP32 Muls+Adds, RNE code in [0,120], E4M3/256 decoded to FP16; unchanged FP16 QK/PV and online history",
              "quality_scope": "Synthetic tests; no model-quality claim; ExpCast is not the complete quantized VC paper pipeline"}
    save = lambda: common.write_json(output, report)
    save()
    try:
        for name in order:
            if name == "installed":
                continue
            resolved = build_arguments(args, name)
            manifest = json.loads(Path(resolved.source_manifest).read_text(encoding="utf-8"))
            validate_prepared_source(Path(manifest["source_dir"]), Path(resolved.source_manifest))
            report["provenance"][name] = baseline.provenance(resolved, expected_variant=name)
        save()
        for name in order:
            resolved = build_arguments(args, name)
            directory = Path(resolved._run_dir) / resolved._worker
            directory.mkdir(parents=True, exist_ok=True)
            env = baseline.worker_environment(os.environ, resolved._worker, resolved.vendor_root, resolved._run_dir)
            for key in ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME"):
                Path(env[key]).mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--_worker", name,
                       "--_run-dir", args._run_dir]
            for key in ("baseline_build", "expcast_build", "bridge_manifest", "device", "warmup", "repeats", "trials", "cpu_threads"):
                command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
            for key in ("baseline_probe_build", "expcast_probe_build"):
                if getattr(args, key):
                    command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
            print(f"Starting isolated {name}: {directory}", flush=True)
            with (directory / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace")
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            path = directory / "report.json"
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            report["workers"][name] = {**data, "command": command, "returncode": code}
            save()
            if code or data.get("status") != "passed":
                raise RuntimeError(f"{name} failed; see {directory / 'worker.log'}")
        if len({row["pid"] for row in report["workers"].values()}) != len(order):
            raise AssertionError("Provider workers must be distinct processes")
        report["baseline_vs_installed"] = {}
        report["expcast_vs_baseline"] = {}
        for scope in ("full", "prepared_core"):
            sample = lambda name: report["workers"][name][scope + "_sample"]
            unchanged = common.compare_samples(sample("installed"), sample("baseline"))
            report["baseline_vs_installed"][scope] = unchanged
            save()
            if not unchanged["passed"]:
                raise AssertionError("Unchanged source baseline differs from installed Laser by more than 0.1%")
            report["expcast_vs_baseline"][scope] = compare_algorithm_samples(sample("baseline"), sample("expcast"))
            for name in order:
                if name.endswith("_probe"):
                    metric = common.compare_samples(sample(name.removesuffix("_probe")), sample(name))
                    report.setdefault("probe_vs_production", {}).setdefault(name, {})[scope] = metric
                    save()
                    if not metric["passed"]:
                        raise AssertionError(f"{name} instrumentation changed sampled output")
        report["status"] = "passed"
        save()
        print(table(report), flush=True)
        print(f"Report: {output}", flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save()
        traceback.print_exc()
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-build", default="results/mindie_laser_build.json")
    parser.add_argument("--expcast-build", default="results/mindie_laser_expcast_build.json")
    parser.add_argument("--bridge-manifest", default="results/mindie_laser_bridge.json")
    parser.add_argument("--baseline-probe-build")
    parser.add_argument("--expcast-probe-build")
    parser.add_argument("--output", default="results/mindie_laser_expcast_compare.json")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--_worker", choices=("installed", "baseline", "expcast", "baseline_probe", "expcast_probe"), help=argparse.SUPPRESS)
    parser.add_argument("--_run-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.device < 0 or args.warmup < 0 or args.trials < 3 or min(args.repeats, args.cpu_threads) < 1:
        parser.error("Nonnegative device/warmup, trials >=3 and positive repeats/threads required")
    if bool(args.baseline_probe_build) != bool(args.expcast_probe_build):
        parser.error("Supply both probe builds or neither")
    if args._worker and not args._run_dir:
        parser.error("Internal worker requires --_run-dir")
    return worker(args) if args._worker else parent(args)


if __name__ == "__main__":
    sys.exit(main())
