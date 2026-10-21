#!/usr/bin/env python3
"""Calibrated sparse PV diagnostics for the validated V-prefetch Laser kernel."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import traceback
from types import SimpleNamespace

import benchmark_mindie_laser_quant as quant
from mindie_laser_quant_pairing import run_paired, paired_comparison
from profile_mindie_laser_vector_scale_bulk import inspect_providers, require_exact

common = quant.common
ORIGINAL, PRODUCTION, PROBE = "original_laser", "vector_pv_prefetch", "pv_probe"
MODES = {"plain": 0, "issue": 1, "detail": 2}
PAIRS = ((PRODUCTION, "plain"), ("plain", "issue"), ("plain", "detail"),
         ("issue", "detail"), (PRODUCTION, "detail"))
MANIFESTS = {
    "production_build": "results/mindie_laser_quant_vscale_pv_prefetch_build.json",
    "probe_build": "results/mindie_laser_cube_pv_probe_build.json",
    "preprocess_build": "results/mindie_laser_quant_preprocess_build.json",
    "baseline_build": "results/mindie_laser_build.json",
    "bridge_manifest": "results/mindie_laser_bridge.json",
}
SCOPE = (
    "Original FP16 Laser is the primary production baseline. Its FULL/prepared/native API times "
    "are paired with current V-only PV-prefetch INT8, with no profiler. Diagnostic plain/issue/detail "
    "are separate calibration rows on identical V-only prepared operands, never optimization variants. "
    "All three diagnostic modes include fresh trace allocation/zeroing. Sparse system ticks are "
    "not core cycles, production percentages, additive phase costs or kernel milliseconds."
)


def resolve_providers(args):
    import prepare_mindie_laser_quant_vscale_pv_prefetch_source as source
    import prepare_mindie_laser_cube_probe_source as probe_source
    original = quant.build_arguments(args, "baseline")
    original.marker_env, original.marker_token = quant.baseline.MARKER_ENV, quant.baseline.MARKER_TOKEN
    options = SimpleNamespace(**vars(args), quant_build=args.production_build)
    production = quant.build_arguments(options, "qk_int8_fused")
    production.marker_env, production.marker_token = source.MARKER_ENV, source.MARKER_TOKEN
    options.quant_build = args.probe_build
    probe = quant.build_arguments(options, "qk_int8_fused")
    probe.marker_env, probe.marker_token = probe_source.marker_environment("pv"), probe_source.marker_token("pv")
    providers = {ORIGINAL: original, PRODUCTION: production, PROBE: probe}
    if len({str(Path(p.vendor_root).resolve()) for p in providers.values()}) != 3:
        raise ValueError("Original, V-prefetch production and diagnostic require three distinct vendors")
    return providers


def verify_builds(args, providers):
    from mindie_laser_quant_fused import validate_preprocess_manifest
    import prepare_mindie_laser_quant_vscale_pv_prefetch_source as source
    import prepare_mindie_laser_cube_probe_source as probe_source
    return {
        ORIGINAL: quant.baseline.provenance(providers[ORIGINAL]),
        PRODUCTION: quant.quant_provenance(providers[PRODUCTION], identity=source.IDENTITY,
                                           source_validator=source.validate_prepared_source),
        PROBE: quant.quant_provenance(providers[PROBE], identity=probe_source.identity("pv"),
            source_validator=lambda path, manifest: probe_source.validate_prepared_source(path, manifest, kind="pv")),
        "preprocessor": validate_preprocess_manifest(args.preprocess_build),
    }


def worker_environment(base, providers, directory):
    if set(providers) != {ORIGINAL, PRODUCTION, PROBE}:
        raise ValueError("PV diagnostic worker requires original, production and diagnostic providers")
    roots = [str(Path(providers[name].vendor_root).resolve()) for name in (ORIGINAL, PRODUCTION, PROBE)]
    if len(set(roots)) != 3:
        raise ValueError("Three distinct provider roots required")
    env = quant.baseline.worker_environment(base, "local", roots[0], directory)
    paths = [p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
             if p and not any(common.within(p, root) for root in roots)]
    env["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(roots)
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(Path(root)/"op_api/lib") for root in roots]+paths)
    for key in list(env):
        if key.startswith("VC_") and key.endswith("MARKER_FILE"):
            env.pop(key)
    for name, provider in providers.items():
        env[provider.marker_env] = str(Path(directory)/"local"/(name+"_markers.jsonl"))
    return env


def probe_call(native, payload, mode):
    return native(payload.q, payload.k, payload.v, payload.q_scale, payload.k_scale,
                  payload.scale, payload.keys, False, mode)


def checked_trace(result, payload, mode, *, require=False):
    import torch
    from mindie_laser_pv_records import TRACE_SHAPE, decode_trace, require_long_coverage
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError("PV probe must return (padded attention, trace)")
    out, trace = result
    if (not isinstance(out, torch.Tensor) or out.shape != payload.q.shape
            or out.device != payload.q.device or out.dtype != torch.float32):
        raise RuntimeError("PV probe must return padded FP32 attention")
    if (not isinstance(trace, torch.Tensor) or tuple(trace.shape) != TRACE_SHAPE
            or trace.device != payload.q.device or trace.dtype != torch.int64):
        raise RuntimeError("PV probe returned invalid trace shape/dtype/device")
    records = decode_trace(trace.cpu(), mode, require_records=require and mode > 0)
    if require and mode > 0 and tuple(payload.shape) == (1, 28, 37741, 128) and payload.keys == 37741:
        require_long_coverage(records)
    return records


class DeferredProbeAdapter:
    """All modes launch before comparisons; preserve queued scale update tests."""
    def __init__(self, production, native, rows):
        self.production, self.native, self.rows = production, native, rows
        self.pending = []

    def __getattr__(self, name):
        return getattr(self.production, name)

    def run_native(self, payload):
        expected = self.production.run_native(payload)
        results = {label: probe_call(self.native, payload, mode) for label, mode in MODES.items()}
        self.pending.append((payload, expected, results))
        return results["detail"][0]

    def __call__(self, *inputs):
        payload = self.prepare(*inputs)
        return self.postprocess(self.run_native(payload), payload)

    def flush(self):
        from profile_mindie_laser_quant import compare_all_rows
        for payload, expected, results in self.pending:
            for label, result in results.items():
                records = checked_trace(result, payload, MODES[label])
                error = compare_all_rows(result[0], expected)
                require_exact(error)
                self.rows.append(dict(mode=label, records=len(records), **error))
        self.pending.clear()


def validate_probe(production, native, device, report, save):
    from mindie_laser_vscale_reference import validate_vector_scale
    from mindie_laser_pv_prefetch_checks import validate_pv_prefetch
    checked = DeferredProbeAdapter(production, native, report.setdefault("probe_exact_validation", []))
    def checkpoint():
        checked.flush()
        save()
    validate_vector_scale({"baseline": production, "reuse": checked}, device, report, checkpoint)
    validate_pv_prefetch(production, checked, device, report, checkpoint,
        scope_name="pv_probe",
        description="Current V-only production versus plain/issue/detail diagnostic; unchanged algorithm")
    checked.flush()
    report["validation_scope"] = (
        "Current Vector-scale decoded oracle, same-address queued Q/K scale updates, 38 PV boundary cases; "
        "ALL padded outputs exact in plain/issue/detail. No model-quality claim.")
    save()


def make_production_calls(original, production, payloads, inputs):
    from mindie_laser_fp16_trace_worker import native_call
    return {
        "full": {ORIGINAL: lambda: original(*inputs), PRODUCTION: lambda: production(*inputs)},
        "prepared": {ORIGINAL: lambda: original.run_prepared(payloads[ORIGINAL]),
                     PRODUCTION: lambda: production.run_prepared(payloads[PRODUCTION])},
        "native": {ORIGINAL: lambda: native_call(original, payloads[ORIGINAL]),
                   PRODUCTION: lambda: production.run_native(payloads[PRODUCTION])},
    }


def production_table(report):
    lines = ["Production only: same-process randomized paired synchronized API milliseconds; NO profiler:",
             "Primary baseline: pinned original FP16 MindIE Laser; diagnostic modes are excluded.",
             f"{'scope':<10} {'variant':<22} {'API ms':>10} {'Laser/row':>11} {'saved ms':>11} {'faster':>8} {'dense RMSE %':>14}"]
    for scope, timing in report["production_timings"].items():
        baseline = timing["rows"][ORIGINAL]["median_ms"]
        pair = report["production_comparisons"][scope]
        for name in (ORIGINAL, PRODUCTION):
            value = timing["rows"][name]["median_ms"]
            saved = f"{pair['median_saved_ms']:+.4f}" if name == PRODUCTION else "--"
            faster = f"{pair['faster_trials']}/{pair['trials']}" if name == PRODUCTION else "--"
            dense = 100*report["dense_fp32_samples"][name]["relative_rmse"]
            lines.append(f"{scope:<10} {name:<22} {value:10.4f} {baseline/value:10.4f}x {saved:>11} {faster:>8} {dense:14.6f}")
    lines += ["FULL includes fresh per-algorithm preprocessing; prepared/native use each algorithm's own payload.",
              "Native includes allocation/dispatch. Scopes are independent, not additive or kernel-only.",
              "Dense RMSE samples queries/all keys; it is not a model-quality verdict."]
    return "\n".join(lines)


def calibration_table(report):
    lines = ["V-only PV-prefetch diagnostic calibration: identical payload, paired native API wall ms:",
             f"{'comparison':<34} {'production/plain':>17} {'measured':>11} {'paired added':>14}"]
    for name, pair in report["calibration"].items():
        lines.append(f"{name:<34} {pair['baseline_median_ms']:17.4f} {pair['candidate_median_ms']:11.4f} "
                     f"{-pair['median_saved_ms']:+14.4f}")
    return "\n".join(lines+["Probe overhead, not optimization speedup; all diagnostic modes allocate/zero trace storage."])


def worker(args):
    directory = Path(args._run_dir)/"paired"/"local"
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", stage="provenance", pid=os.getpid(), scope=SCOPE,
        primary_baseline=ORIGINAL, production=PRODUCTION, shape_bnsd=args.shape, dtype="bfloat16", seed=0,
        production_timings={}, production_comparisons={}, traces=[])
    save = lambda: common.write_json(directory/"report.json", report)
    save()
    try:
        providers = resolve_providers(args)
        evidence = report["provenance"] = verify_builds(args, providers)
        import torch
        import torch_npu
        from mindie_laser_quant_backend import load_quant_laser
        from mindie_laser_quant_vscale_patch import VSCALE_CONTRACT
        from mindie_laser_quant_vscale_pv_prefetch_patch import CONTRACT as PV_CONTRACT
        from profile_mindie_laser_quant import compare_all_rows, verify_runtime_target
        from mindie_laser_fp16_trace_worker import verify_runtime_target as verify_fp16_target
        from mindie_laser_pv_records import summarize_records, sample_table
        torch.set_num_threads(min(args.cpu_threads, torch.get_num_threads()))
        torch.npu.set_device(args.device)
        torch.npu.set_option({"ACL_OP_COMPILER_CACHE_MODE": "disable",
                              "ACL_OP_COMPILER_CACHE_DIR": os.environ["ACL_OP_COMPILER_CACHE_DIR"]})
        original, transforms = quant.baseline.load_adapter(providers[ORIGINAL], evidence[ORIGINAL]["source_manifest"])
        report["transform_validation"] = transforms
        production = load_quant_laser(providers[PRODUCTION].bridge_path, preprocessing="fused",
            preprocess_build=args.preprocess_build, torch_namespace="vc_laser_quant_vscale_pv_prefetch",
            v_quantize=False, expcast=False, check_finite=False)
        production.metadata.update(qk_result_contract=VSCALE_CONTRACT, pv_schedule=PV_CONTRACT)
        torch.ops.load_library(providers[PROBE].bridge_path)
        native = torch.ops.vc_laser_cube_pv_probe.la.default
        if ([str(x.type) for x in native._schema.arguments] != ["Tensor"]*5+["float", "int", "bool", "int"]
                or [str(x.type) for x in native._schema.returns] != ["Tensor", "Tensor"]):
            raise RuntimeError(f"Unexpected PV diagnostic ABI: {native._schema}")
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtGetSocName.restype = ctypes.c_char_p
        soc = acl.aclrtGetSocName()
        if not soc:
            raise RuntimeError("aclrtGetSocName returned null")
        soc, cann = soc.decode(), os.environ["ASCEND_HOME_PATH"]
        prep = json.loads(Path(args.preprocess_build).read_text(encoding="utf-8"))
        report["runtime_targets"] = {ORIGINAL: verify_fp16_target(evidence[ORIGINAL]["build_manifest"], soc, cann),
            **{name: verify_runtime_target(evidence[name]["build_manifest"], prep, soc, cann)
               for name in (PRODUCTION, PROBE)}}
        report["environment"] = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
            runtime_soc=soc, device_index=args.device, native_schema=str(native._schema))
        report["adapters"] = {ORIGINAL: original.metadata, PRODUCTION: production.metadata}
        context = dict(cann_root=cann, soc=evidence[ORIGINAL]["build_manifest"]["soc"])
        device = f"npu:{args.device}"
        with torch.no_grad():
            report["stage"], report["original_validation"] = "structured_validation", {}
            quant.baseline.validate_small(original, device, report["original_validation"], save, torch)
            validate_probe(production, native, device, report, save)
            report["stage"] = "long_shape_validation"
            generator = torch.Generator().manual_seed(0)
            host = [torch.randn(args.shape, generator=generator).to(torch.bfloat16) for _ in range(3)]
            report["input_sha256"] = [common.tensor_sha256(x) for x in host]
            inputs = [x.to(device) for x in host]
            payloads = {ORIGINAL: original.prepare(*inputs), PRODUCTION: production.prepare(*inputs)}
            fixed = payloads[PRODUCTION]
            calls = make_production_calls(original, production, payloads, inputs)
            probes = {label: lambda mode=mode: probe_call(native, fixed, mode) for label, mode in MODES.items()}
            calibration_calls = {PRODUCTION: calls["native"][PRODUCTION], **probes}
            raw, markers = {}, {}
            report["host_markers"], report["initial_records"], report["same_payload_all_output"] = {}, {}, {}
            for label, call in {**calls["native"], **probes}.items():
                provider = providers[label if label in (ORIGINAL, PRODUCTION) else PROBE]
                marker = Path(os.environ[provider.marker_env])
                offset = marker.stat().st_size if marker.exists() else 0
                result = call()
                torch.npu.synchronize()
                with marker.open("rb") as stream:
                    stream.seek(offset)
                    fresh = stream.read().decode("utf-8")
                report["host_markers"][label] = (quant.baseline.parse_markers(fresh, os.getpid()) if label == ORIGINAL
                    else quant.parse_quant_markers(fresh, os.getpid(), provider.marker_token))
                markers[marker] = marker.stat().st_size
                if label in (ORIGINAL, PRODUCTION):
                    raw[label] = result
                else:
                    report["initial_records"][label] = checked_trace(result, fixed, MODES[label], require=True)
                    error = compare_all_rows(result[0], raw[PRODUCTION])
                    require_exact(error)
                    report["same_payload_all_output"][label] = error
                save()
            ids = torch.linspace(0, args.shape[2]-1, min(32, args.shape[2])).long().unique()
            dense = common.dense_reference(*host, ids)
            report["dense_fp32_samples"], report["full_prepared_samples"] = {}, {}
            for name, adapter in ((ORIGINAL, original), (PRODUCTION, production)):
                payload = payloads[name]
                out = (original.postprocess(raw[name][1], payload.dtype, payload.shape[2], payload.shape[3])
                       if name == ORIGINAL else production.postprocess(raw[name], payload))
                sampled = out.index_select(2, ids.to(device)).cpu()
                report["dense_fp32_samples"][name] = common.error_metrics(sampled, dense)
                if name == ORIGINAL:
                    quant.baseline.record_gate(report["original_validation"], "dense_long", sampled, dense, .01, save)
                full = calls["full"][name]().index_select(2, ids.to(device)).cpu()
                prepared = calls["prepared"][name]().index_select(2, ids.to(device)).cpu()
                quant.baseline.record_gate(report["full_prepared_samples"], name, full, prepared, .001, save)
                quant.baseline.record_gate(report["full_prepared_samples"], name+"_native", sampled, prepared, .001, save)
                del out, sampled, full, prepared
            report["accuracy_scope"] = dict(query_ids=ids.tolist(), all_keys=args.shape[2],
                diagnostic_equivalence="ALL padded native rows/features", model_quality_claim=False)
            del raw, result, dense, host
            inspect_providers(providers, evidence, report, directory, **context)
            quant.inspect_fused_preprocessor(providers[PRODUCTION], report)
            for provider in providers.values():
                # Include all untimed checks above in the final marker extent.
                marker = Path(os.environ[provider.marker_env])
                markers[marker] = marker.stat().st_size
                os.environ.pop(provider.marker_env, None)
            for scope, group in calls.items():
                report["stage"] = "production_timing_"+scope
                def progress(snapshot, scope=scope):
                    report["production_timings"][scope] = snapshot
                    save()
                    print(f"Production {scope}: paired trial {snapshot['completed_trials']}/{args.trials}", flush=True)
                timing = run_paired(group, synchronize=torch.npu.synchronize, warmup=args.warmup,
                    repeats=args.repeats, trials=args.trials, seed=args.seed, on_trial=progress)
                report["production_timings"][scope] = timing
                report["production_comparisons"][scope] = paired_comparison(timing, ORIGINAL, PRODUCTION)
                save()
            print(production_table(report), flush=True)
            report["stage"] = "probe_calibration"
            def calibration_progress(snapshot):
                report["probe_timings"] = snapshot
                save()
                print(f"PV probe calibration {snapshot['completed_trials']}/{args.trials}", flush=True)
            timing = run_paired(calibration_calls, synchronize=torch.npu.synchronize, warmup=args.warmup,
                repeats=args.repeats, trials=args.trials, seed=args.seed, on_trial=calibration_progress)
            report["probe_timings"] = timing
            report["calibration"] = {f"{a} -> {b}": paired_comparison(timing, a, b) for a, b in PAIRS}
            print(calibration_table(report), flush=True)
            report["stage"] = "sparse_samples"
            rng = random.Random(args.seed+1)
            for trial in range(args.samples):
                labels = ["issue", "detail"]
                rng.shuffle(labels)
                for label in labels:
                    result = probes[label]()
                    torch.npu.synchronize()
                    records = checked_trace(result, fixed, MODES[label], require=True)
                    report["traces"].append(dict(trial=trial, variant=label, records=records))
                    del result
                save()
                print(f"Sparse PV sample {trial+1}/{args.samples}", flush=True)
            report["phase_summary"] = {label: summarize_records([item["records"] for item in report["traces"]
                if item["variant"] == label]) for label in ("issue", "detail")}
            print(sample_table(report["phase_summary"]), flush=True)
            if any(path.stat().st_size != size for path, size in markers.items()):
                raise RuntimeError("Source marker writes continued during measurement")
            inspect_providers(providers, evidence, report, directory, **context)
            quant.inspect_fused_preprocessor(providers[PRODUCTION], report)
        report.update(status="passed", stage="complete")
        save()
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save()
        traceback.print_exc()
        return 1


def launch(args, env, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for key in ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker", "--_run-dir", args._run_dir]
    for key in (*MANIFESTS, "device", "warmup", "repeats", "trials", "samples", "cpu_threads", "seed"):
        command += ["--"+key.replace("_", "-"), str(getattr(args, key))]
    command += ["--shape", *map(str, args.shape)]
    with (directory/"worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    path = directory/"report.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status": "missing_report"}
    return dict(result, returncode=code, command=command, worker_log=str(directory/"worker.log"))


def parent(args):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args._run_dir = tempfile.mkdtemp(prefix="mindie_laser_pv_phases_", dir=output.parent)
    report = dict(status="running", scope=SCOPE, primary_baseline=ORIGINAL, production=PRODUCTION,
                  run_dir=args._run_dir, workers={})
    save = lambda: common.write_json(output, report)
    save()
    try:
        providers = resolve_providers(args)
        report["provenance"] = verify_builds(args, providers)
        save()
        directory = Path(args._run_dir)/"paired"
        env = worker_environment(os.environ, providers, directory)
        result = report["workers"]["paired"] = launch(args, env, directory/"local")
        save()
        if result.get("returncode") != 0 or result.get("status") != "passed":
            raise RuntimeError("PV diagnostics failed; see worker_log")
        report["status"] = "passed"
        save()
        print(f"Report: {output}", flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        traceback.print_exc()
        return 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for key, default in MANIFESTS.items():
        result.add_argument("--"+key.replace("_", "-"), default=default)
    result.add_argument("--output", default="results/mindie_laser_pv_phases.json")
    result.add_argument("--shape", type=int, nargs=4, default=[1, 28, 37741, 128])
    for name, default in (("device", 0), ("warmup", 5), ("repeats", 10), ("trials", 9),
                          ("samples", 5), ("cpu-threads", 4), ("seed", 2026)):
        result.add_argument("--"+name, type=int, default=default)
    result.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--_run-dir", help=argparse.SUPPRESS)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if (min(args.shape) <= 0 or args.shape[-1] not in (64, 128) or args.device < 0 or args.warmup < 0
            or min(args.repeats, args.trials, args.samples, args.cpu_threads) <= 0):
        cli.error("Need nonempty D64/128, positive sample/trial counts, nonnegative device/warmup")
    return worker(args) if args._worker else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
