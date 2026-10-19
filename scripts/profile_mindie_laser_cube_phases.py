#!/usr/bin/env python3
"""Calibrated Cube QK/output and PV/history diagnostics for INT8 and FP16 Laser."""
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
from mindie_laser_scale_reuse_checks import paired_environment, verify_provider_paths

common = quant.common
MODES = {"plain": 0, "issue": 1, "completed": 2}
PAIRS = (("baseline", "plain"), ("plain", "issue"), ("plain", "completed"),
         ("issue", "completed"), ("baseline", "completed"))
SCOPE = (
    "Within each family: identical prepared payload, randomized paired synchronized native API wall time, "
    "without profiler/preprocessing/output conversion. All diagnostic modes allocate and zero a trace. "
    "Mode 0 retains the compiled diagnostic footprint. Mode 1 measures scalar issue windows; mode 2 "
    "adds sparse completion barriers around whole QK/output or PV/history calls. Neither is a production "
    "phase percentage. FP16 and INT8 workers use different payloads in separate processes, not paired speedups."
)


def resolve_providers(args, family):
    import prepare_mindie_laser_cube_probe_source as source
    import prepare_mindie_laser_quant_reuse_source as reuse
    options = SimpleNamespace(**vars(args))
    options.quant_build = args.reuse_build
    old = quant.build_arguments(options, "qk_int8_fused" if family == "int8" else "baseline")
    old.marker_env = reuse.MARKER_ENV if family == "int8" else quant.baseline.MARKER_ENV
    old.marker_token = reuse.MARKER_TOKEN if family == "int8" else quant.baseline.MARKER_TOKEN
    options.quant_build = getattr(args, family + "_probe_build")
    probe = quant.build_arguments(options, "qk_int8_fused")
    probe.marker_env, probe.marker_token = source.marker_environment(family), source.marker_token(family)
    if Path(old.vendor_root).resolve() == Path(probe.vendor_root).resolve():
        raise ValueError("Original and diagnostic operators must use distinct vendors")
    return {"baseline": old, "probe": probe}


def verify_family(args, family):
    import prepare_mindie_laser_cube_probe_source as source
    import prepare_mindie_laser_quant_reuse_source as reuse
    providers = resolve_providers(args, family)
    evidence = {"baseline": quant.quant_provenance(providers["baseline"], identity=reuse.IDENTITY,
        source_validator=reuse.validate_prepared_source) if family == "int8"
        else quant.baseline.provenance(providers["baseline"])}
    evidence["probe"] = quant.quant_provenance(providers["probe"], identity=source.identity(family),
        source_validator=lambda path, manifest: source.validate_prepared_source(path, manifest, kind=family))
    if family == "int8":
        from mindie_laser_quant_fused import validate_preprocess_manifest
        evidence["preprocessor"] = validate_preprocess_manifest(args.preprocess_build)
    return evidence


def inspect_pair(providers, evidence, report, directory, *, cann_root, soc):
    """Audit both mapped vendors; original FP16 has a separate bridge manifest."""
    if any(name == "mindiesd" or name.startswith("mindiesd.") for name in sys.modules):
        raise RuntimeError("Local paired worker must not import mindiesd")
    text = Path("/proc/self/maps").read_text(encoding="utf-8")
    (directory / "proc_self_maps.txt").write_text(text, encoding="utf-8")
    paths = common.mapped_libraries(text)
    background = quant.baseline.inspect_batch_invariant(paths, cann_root, soc)
    relevant = verify_provider_paths(paths, providers, background=background["libraries"])
    for name, provider in providers.items():
        build = evidence[name]["build_manifest"]
        expected = {str(Path(item["path"]).resolve()): item["sha256"] for item in build["host_libraries"]}
        expected[str(Path(provider.bridge_path).resolve())] = evidence[name]["bridge_sha256"]
        for path in relevant:
            if common.within(path, provider.vendor_root) or path == str(Path(provider.bridge_path).resolve()):
                if path not in expected or common.sha256_file(path) != expected[path]:
                    raise RuntimeError("Mapped library differs from audited build: " + path)
    report["mapped_library_paths"] = paths
    report["ambient_registered_provider"] = background["metadata"]
    report["loaded_library_sha256"] = {path: common.sha256_file(path) for path in relevant}


def baseline_call(adapter, payload, family):
    if family == "int8":
        return adapter.native(payload.q, payload.k, payload.v, payload.q_scale, payload.k_scale,
                              payload.scale, payload.keys, False)
    q, k, v = payload.tensors
    return adapter.native(q, k, v, None, None, None, payload.scale, payload.shape[1], "BNSD",
                          1.0, payload.pre_tokens, 1, True)


def probe_call(native, payload, family, keys, mode):
    if family == "int8":
        return native(payload.q, payload.k, payload.v, payload.q_scale, payload.k_scale,
                      payload.scale, payload.keys, False, mode)
    return native(*payload.tensors, payload.scale, keys, mode)


def baseline_output(result, family):
    return result if family == "int8" else result[1]


def checked_trace(result, payload, family, mode, *, require=False):
    import torch
    from mindie_laser_cube_records import decode_trace
    q = payload.q if family == "int8" else payload.tensors[0]
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError("Cube probe must return (attention, trace)")
    out, trace = result
    if (not isinstance(out, torch.Tensor) or out.shape != q.shape or out.device != q.device
            or out.dtype != torch.float32):
        raise RuntimeError("Cube probe must return padded FP32 attention")
    if (not isinstance(trace, torch.Tensor) or tuple(trace.shape) != (4, 2, 3, 24)
            or trace.device != q.device or trace.dtype != torch.int64):
        raise RuntimeError("Cube probe returned an invalid trace shape/dtype/device")
    records = decode_trace(trace.cpu(), mode, require_records=require and mode > 0)
    if require and mode > 0 and {row["stage"] for row in records} != {0, 1}:
        raise ValueError("Long-shape Cube trace requires both QK and PV records")
    return records


def validate_modes(adapter, native, family, device, report, save):
    import torch
    from profile_mindie_laser_quant import compare_all_rows
    report["structured_validation"] = []
    cases = ((1, 1, 1, 1, 64), (1, 2, 17, 129, 64), (1, 2, 256, 256, 128),
             (1, 2, 257, 385, 128), (2, 2, 65, 4097, 64), (1, 2, 129, 8193, 128))
    for dtype in (torch.float16, torch.bfloat16):
        for batch, heads, nq, nk, dim in cases:
            generator = torch.Generator().manual_seed(873)
            inputs = [torch.randn(shape, generator=generator).to(dtype).to(device) for shape in
                      ((batch, heads, nq, dim), (batch, heads, nk, dim), (batch, heads, nk, dim))]
            inputs[2][..., 0], inputs[2][..., 1] = 0, 2
            payload = adapter.prepare(*inputs)
            raw = baseline_call(adapter, payload, family)
            expected = baseline_output(raw, family)
            for label, mode in MODES.items():
                print(f"{family} Cube probe validation {dtype} Q{nq} K{nk} D{dim}: {label}", flush=True)
                result = probe_call(native, payload, family, nk, mode)
                records = checked_trace(result, payload, family, mode)
                error = compare_all_rows(result[0], expected)
                report["structured_validation"].append(dict(dtype=str(dtype), shape_q=list(inputs[0].shape),
                    keys=nk, mode=mode, records=len(records), **error))
                save()
                if not error["exact_match"]:
                    raise AssertionError("Cube diagnostics changed full padded native output")
                del result
            del expected, raw, payload, inputs


def timing_table(report):
    lines = [f"{report['family']} Cube probe calibration: paired synchronized native API wall ms:",
             f"{'comparison':<24} {'baseline':>10} {'measured':>10} {'paired added':>13}"]
    for name, pair in report.get("calibration", {}).items():
        lines.append(f"{name:<24} {pair['baseline_median_ms']:10.4f} {pair['candidate_median_ms']:10.4f} "
                     f"{-pair['median_saved_ms']:+13.4f}")
    return "\n".join(lines + ["Probe perturbation, not optimization speedup; trace allocation/zeroing is included."])


def sample_table(summaries):
    lines = ["Sparse Cube system-counter samples; raw ticks, NOT production phase percentages:",
             f"{'mode/stage/position':<31} {'desc':>5} {'Kblks':>6} {'tiles':>6} {'hist':>5} {'norm':>5} {'count':>6} "
             f"{'issue':>10} {'tail':>10} {'elapsed':>10} {'predrain':>10} {'empty':>10}"]
    for mode, summary in summaries.items():
        for group in summary["groups"]:
            label = f"{mode}/{group['stage_name']}/{group['position_name']}"
            ticks = group["ticks"]
            values = [f"{ticks[field]['median']:.1f}" if field in ticks else "--" for field in
                      ("issue_ticks", "completion_ticks", "elapsed_ticks", "pre_drain_ticks", "empty_ticks")]
            lines.append(f"{label:<31} {group['descriptors']:5} {group['k_blocks']:6} {group['logical_tiles']:6} "
                         f"{group['history_calls']:5} {group['normalize_calls']:5} {group['samples']:6} "
                         + " ".join(f"{value:>10}" for value in values))
    return "\n".join(lines + ["QK includes address mapping, loads, matmul and score output/dequantization.",
        "PV includes address mapping, PV matmul, history scaling and tail normalization where present.",
        "completion tail is residual queued work, NOT pure FIX time. Drains change overlap; intervals are not additive."])


def worker(args):
    from profile_mindie_laser_quant import compare_all_rows, verify_runtime_target
    from mindie_laser_fp16_trace_worker import verify_runtime_target as verify_fp16_target
    from mindie_laser_cube_records import summarize_records
    family = args._worker
    providers = resolve_providers(args, family)
    directory = Path(args._run_dir) / family / "local"
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", stage="provenance", family=family, scope=SCOPE,
                  shape_bnsd=args.shape, dtype="bfloat16", seed=0, pid=os.getpid(), traces=[])
    save = lambda: common.write_json(directory / "report.json", report)
    save()
    try:
        evidence = verify_family(args, family)
        report["provenance"] = evidence
        import torch
        import torch_npu
        torch.set_num_threads(min(args.cpu_threads, torch.get_num_threads()))
        torch.npu.set_device(args.device)
        torch.npu.set_option({"ACL_OP_COMPILER_CACHE_MODE": "disable",
                              "ACL_OP_COMPILER_CACHE_DIR": os.environ["ACL_OP_COMPILER_CACHE_DIR"]})
        if family == "int8":
            from mindie_laser_quant_backend import load_quant_laser
            adapter = load_quant_laser(providers["baseline"].bridge_path, preprocessing="fused",
                preprocess_build=args.preprocess_build, torch_namespace="vc_laser_quant_reuse",
                v_quantize=False, expcast=False, check_finite=False)
        else:
            adapter, transform_check = quant.baseline.load_adapter(providers["baseline"], evidence["baseline"]["source_manifest"])
            report["transform_validation"] = transform_check
        namespace = "vc_laser_cube_" + family + "_probe"
        torch.ops.load_library(providers["probe"].bridge_path)
        native = getattr(torch.ops, namespace).la.default
        types = ["Tensor"]*5+["float", "int", "bool", "int"] if family == "int8" else ["Tensor"]*3+["float", "int", "int"]
        if ([str(x.type) for x in native._schema.arguments] != types
                or [str(x.type) for x in native._schema.returns] != ["Tensor", "Tensor"]):
            raise RuntimeError(f"Unexpected {family} diagnostic ABI: {native._schema}")
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtGetSocName.restype = ctypes.c_char_p
        soc = acl.aclrtGetSocName()
        if not soc:
            raise RuntimeError("aclrtGetSocName returned null")
        soc, cann = soc.decode(), os.environ["ASCEND_HOME_PATH"]
        if family == "int8":
            preprocess = json.loads(Path(args.preprocess_build).read_text(encoding="utf-8"))
            report["runtime_targets"] = {name: verify_runtime_target(item["build_manifest"], preprocess, soc, cann)
                                         for name, item in evidence.items() if name != "preprocessor"}
        else:
            report["runtime_targets"] = {name: verify_fp16_target(item["build_manifest"], soc, cann)
                                         for name, item in evidence.items()}
        report["environment"] = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
            runtime_soc=soc, device_index=args.device, native_schema=str(native._schema))
        context = dict(cann_root=cann, soc=evidence["baseline"]["build_manifest"]["soc"])
        device = f"npu:{args.device}"
        with torch.no_grad():
            report["stage"] = "structured_validation"
            validate_modes(adapter, native, family, device, report, save)
            report["stage"] = "long_shape_validation"
            save()
            generator = torch.Generator().manual_seed(0)
            originals = [torch.randn(args.shape, generator=generator).to(torch.bfloat16) for _ in range(3)]
            report["input_sha256"] = [common.tensor_sha256(x) for x in originals]
            payload = adapter.prepare(*(x.to(device) for x in originals))
            report["payload"] = adapter.describe_payload(payload)
            calls = {"baseline": lambda: baseline_call(adapter, payload, family),
                     **{label: lambda mode=mode: probe_call(native, payload, family, args.shape[2], mode)
                        for label, mode in MODES.items()}}
            report["same_payload_all_output"], report["host_markers"] = {}, {}
            markers, expected = {}, None
            for label, call in calls.items():
                provider = providers["baseline" if label == "baseline" else "probe"]
                marker = Path(os.environ[provider.marker_env])
                offset = marker.stat().st_size if marker.exists() else 0
                result = call()
                torch.npu.synchronize()
                with marker.open("rb") as stream:
                    stream.seek(offset)
                    fresh = stream.read().decode("utf-8")
                report["host_markers"][label] = quant.parse_quant_markers(fresh, os.getpid(), provider.marker_token)
                markers[marker] = marker.stat().st_size
                if label == "baseline":
                    expected = baseline_output(result, family)
                else:
                    records = checked_trace(result, payload, family, MODES[label], require=True)
                    error = compare_all_rows(result[0], expected)
                    report["same_payload_all_output"][label] = error
                    report.setdefault("initial_records", {})[label] = records
                    if not error["exact_match"]:
                        raise AssertionError(f"{family}/{label} changed ALL-row native output")
                del result
                save()
            # Dense error describes the respective algorithm, not probe equivalence.
            ids = torch.linspace(0, args.shape[2]-1, min(32, args.shape[2])).long().unique()
            out = (adapter.postprocess(expected, payload) if family == "int8" else
                   adapter.postprocess(expected, payload.dtype, payload.shape[2], payload.shape[3]))
            sample = out.index_select(2, ids.to(device)).cpu()
            dense = common.dense_reference(*originals, ids)
            report["dense_fp32_sample"] = common.error_metrics(sample, dense)
            report["dense_scope"] = dict(query_ids=ids.tolist(), all_keys=args.shape[2], model_quality_claim=False)
            del out, sample, dense, expected, originals
            inspect_pair(providers, evidence, report, directory, **context)
            if family == "int8":
                quant.inspect_fused_preprocessor(providers["baseline"], report)
            for provider in providers.values():
                os.environ.pop(provider.marker_env, None)
            report["stage"] = "calibration"
            def progress(snapshot):
                report["timings"] = snapshot
                save()
                print(f"{family} Cube calibration {snapshot['completed_trials']}/{args.trials}", flush=True)
            timing = run_paired(calls, synchronize=torch.npu.synchronize, warmup=args.warmup,
                repeats=args.repeats, trials=args.trials, seed=args.seed, on_trial=progress)
            report["timings"] = timing
            report["calibration"] = {f"{a} -> {b}": paired_comparison(timing, a, b) for a, b in PAIRS}
            if any(path.stat().st_size != size for path, size in markers.items()):
                raise RuntimeError("Source marker writes continued during calibration")
            print(timing_table(report), flush=True)
            report["stage"] = "sample_records"
            save()
            rng = random.Random(args.seed + 1)
            for trial in range(args.samples):
                labels = ["issue", "completed"]
                rng.shuffle(labels)
                for label in labels:
                    result = calls[label]()
                    torch.npu.synchronize()
                    records = checked_trace(result, payload, family, MODES[label], require=True)
                    report["traces"].append(dict(trial=trial, variant=label, records=records))
                    del result
                save()
                print(f"{family} Cube sample {trial+1}/{args.samples}", flush=True)
            report["phase_summary"] = {label: summarize_records([item["records"] for item in report["traces"]
                if item["variant"] == label]) for label in ("issue", "completed")}
            print(sample_table(report["phase_summary"]), flush=True)
            inspect_pair(providers, evidence, report, directory, **context)
            if family == "int8":
                quant.inspect_fused_preprocessor(providers["baseline"], report)
        report.update(status="passed", stage="complete")
        save()
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save()
        traceback.print_exc()
        return 1


def launch(args, family, providers):
    directory = Path(args._run_dir) / family / "local"
    directory.mkdir(parents=True, exist_ok=True)
    env = paired_environment(os.environ, providers, directory.parent)
    for key in ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker", family, "--_run-dir", args._run_dir]
    for key in ("reuse_build", "int8_probe_build", "fp16_probe_build", "preprocess_build", "baseline_build",
                "bridge_manifest", "device", "warmup", "repeats", "trials", "samples", "cpu_threads", "seed"):
        command += ["--"+key.replace("_", "-"), str(getattr(args, key))]
    command += ["--shape", *map(str, args.shape)]
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    path = directory / "report.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status": "missing_report"}
    return dict(result, returncode=code, command=command, worker_log=str(directory / "worker.log"))


def parent(args):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args._run_dir = tempfile.mkdtemp(prefix="mindie_laser_cube_phases_", dir=output.parent)
    report = dict(status="running", scope=SCOPE, run_dir=args._run_dir, workers={}, provenance={})
    save = lambda: common.write_json(output, report)
    save()
    try:
        for family in ("int8", "fp16"):
            report["provenance"][family] = verify_family(args, family)
        save()
        for family in ("int8", "fp16"):
            print(f"Starting isolated {family} Cube diagnostics", flush=True)
            result = launch(args, family, resolve_providers(args, family))
            report["workers"][family] = result
            save()
            if result["returncode"] or result["status"] != "passed":
                raise RuntimeError(f"{family} diagnostic failed; see {result['worker_log']}")
        first, second = (report["workers"][name] for name in ("int8", "fp16"))
        if first["input_sha256"] != second["input_sha256"] or first["environment"]["runtime_soc"] != second["environment"]["runtime_soc"]:
            raise RuntimeError("FP16/INT8 diagnostics did not use identical original inputs/runtime SoC")
        report["same_original_inputs"] = True
        report["status"] = "passed"
        save()
        print("FP16/INT8 use different prepared operands and separate processes; do not report paired speedup.", flush=True)
        print(f"Report: {output}", flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        traceback.print_exc()
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("reuse-build", "results/mindie_laser_quant_reuse_build.json"),
        ("int8-probe-build", "results/mindie_laser_cube_int8_probe_build.json"),
        ("fp16-probe-build", "results/mindie_laser_cube_fp16_probe_build.json"),
        ("preprocess-build", "results/mindie_laser_quant_preprocess_build.json"),
        ("baseline-build", "results/mindie_laser_build.json"),
        ("bridge-manifest", "results/mindie_laser_bridge.json"),
        ("output", "results/mindie_laser_cube_phases.json")):
        parser.add_argument("--"+name, default=default)
    parser.add_argument("--shape", type=int, nargs=4, default=[1, 28, 37741, 128])
    for name, default in (("device", 0), ("warmup", 5), ("repeats", 10), ("trials", 9),
                          ("samples", 5), ("cpu-threads", 4), ("seed", 2026)):
        parser.add_argument("--"+name, type=int, default=default)
    parser.add_argument("--_worker", choices=("int8", "fp16"), help=argparse.SUPPRESS)
    parser.add_argument("--_run-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (min(args.shape) <= 0 or args.shape[-1] not in (64, 128) or args.device < 0 or args.warmup < 0
            or min(args.repeats, args.trials, args.samples, args.cpu_threads) <= 0):
        parser.error("Need nonempty D64/128, positive sample/trial counts, nonnegative device/warmup")
    return worker(args) if args._worker else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
