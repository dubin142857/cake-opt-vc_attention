#!/usr/bin/env python3
"""Calibrate sparse Cube/Vector wait and completed Vector phase diagnostics."""
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
from mindie_laser_scale_reuse_checks import paired_environment, inspect_providers

common = quant.common
MODES = {"plain": 0, "waits": 1, "phases": 2}
PAIRS = (("reuse", "plain"), ("plain", "waits"), ("plain", "phases"),
         ("waits", "phases"), ("reuse", "phases"))
SCOPE = (
    "Identical fixed INT8 Q/K, FP16 V and FP32 scales; synchronized native API wall time. "
    "The diagnostic ABI additionally allocates/zeros an INT64 trace and returns it. "
    "Mode 0 still contains compiled diagnostic branches/resources. Mode 1 timestamps original "
    "cross-core waits; mode 2 additionally inserts sparse Vector completion fences. "
    "No preprocessing, output conversion, profiler or trace decoding inside timing."
)


def resolve_providers(args):
    from importlib import import_module
    providers, sources = {}, {}
    for name, module, path in (("baseline", "quant_reuse", args.reuse_build),
                               ("probe", "quant_probe", args.probe_build)):
        source = import_module("prepare_mindie_laser_" + module + "_source")
        options = SimpleNamespace(**vars(args), quant_build=path)
        provider = quant.build_arguments(options, "qk_int8_fused")
        provider.marker_env, provider.marker_token = source.MARKER_ENV, source.MARKER_TOKEN
        providers[name], sources[name] = provider, source
    if Path(providers["baseline"].vendor_root).resolve() == Path(providers["probe"].vendor_root).resolve():
        raise ValueError("Diagnostic and scale-reuse builds must use independent vendors")
    return providers, sources


def verify_builds(args):
    from mindie_laser_quant_fused import validate_preprocess_manifest
    providers, sources = resolve_providers(args)
    return {**{name: quant.quant_provenance(provider, identity=sources[name].IDENTITY,
                source_validator=sources[name].validate_prepared_source)
               for name, provider in providers.items()},
            "preprocessor": validate_preprocess_manifest(args.preprocess_build)}


def probe_call(native, payload, mode):
    return native(payload.q, payload.k, payload.v, payload.q_scale, payload.k_scale,
                  payload.scale, payload.keys, False, mode)


def reuse_call(native, payload):
    # Match the diagnostic's direct-dispatch level; validation is outside timing.
    return native(payload.q, payload.k, payload.v, payload.q_scale, payload.k_scale,
                  payload.scale, payload.keys, False)


def checked_trace(result, payload, mode, *, require=False):
    import torch
    from mindie_laser_phase_records import decode_trace
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError("Diagnostic native API must return (attention, trace)")
    out, trace = result
    if (not isinstance(out, torch.Tensor) or out.shape != payload.q.shape
            or out.dtype != torch.float32 or out.device != payload.q.device):
        raise RuntimeError("Unexpected diagnostic attention output")
    if (not isinstance(trace, torch.Tensor) or tuple(trace.shape) != (4, 3, 8, 24)
            or trace.dtype != torch.int64 or trace.device != payload.q.device):
        raise RuntimeError("Unexpected diagnostic trace shape/dtype/device")
    return decode_trace(trace.cpu(), mode, require_waits=require and mode > 0,
                        require_phases=require and mode == 2)


def validate_modes(old, native, device, report, save):
    """Check exact output, untouched padding and trace decoding in each mode."""
    import torch
    from profile_mindie_laser_quant import compare_all_rows
    report["structured_validation"] = []
    # Include both section classes and an >8192-key multi-section case; use two
    # batches/heads so cross-core wait sampling is not only a single tiny task.
    cases = ((1, 1, 1, 1, 64), (1, 2, 17, 129, 64), (1, 2, 257, 385, 128),
             (2, 2, 65, 4097, 64), (1, 2, 129, 8193, 128))
    for dtype in (torch.float16, torch.bfloat16):
        for batch, heads, nq, nk, dim in cases:
            generator = torch.Generator().manual_seed(872)
            inputs = [torch.randn(shape, generator=generator).to(dtype).to(device) for shape in
                      ((batch, heads, nq, dim), (batch, heads, nk, dim), (batch, heads, nk, dim))]
            payload = old.prepare(*inputs)
            expected = old.run_native(payload)
            for label, mode in MODES.items():
                print(f"Diagnostic validation {dtype} Q{nq} K{nk} D{dim}: {label}", flush=True)
                result = probe_call(native, payload, mode)
                records = checked_trace(result, payload, mode)
                error = compare_all_rows(result[0], expected)
                row = dict(dtype=str(dtype), shape_q=list(inputs[0].shape), keys=nk,
                           mode=mode, records=len(records), **error)
                report["structured_validation"].append(row)
                save()
                if not error["exact_match"]:
                    raise AssertionError("Diagnostic mode changed native output (all padded rows checked)")
                del result
            del expected, payload, inputs


def calibration_table(report):
    lines = ["Probe calibration: same-payload paired synchronized native API ms (no profiler):",
             f"{'comparison':<23} {'baseline':>10} {'measured':>10} {'paired added':>13} {'faster':>9}"]
    for name, pair in report.get("calibration", {}).items():
        lines.append(f"{name:<23} {pair['baseline_median_ms']:10.4f} {pair['candidate_median_ms']:10.4f} "
                     f"{-pair['median_saved_ms']:+13.4f} {pair['faster_trials']:4}/{pair['trials']}")
    lines += ["Added time is a paired median; it is not the difference of table medians.",
              "reuse/plain includes trace allocation/zeroing and compiled instrumentation footprint.",
              "plain/waits and plain/phases measure diagnostic perturbation, not optimization speedup."]
    return "\n".join(lines)


def phase_table(summaries):
    lines = ["Sparse instrumented system-counter intervals (raw ticks; NOT production percentages):",
             "Wait position: first/second/middle/last logical round. Inactive samples remain separate.",
             f"{'mode/role/position':<33} {'active':>6} {'samples':>8} {'wait median':>13} {'min':>11} {'max':>11}"]
    for mode, summary in summaries.items():
        for group in summary["groups"]:
            if group["kind"] != 1:
                continue
            tick = group["ticks"]["wait_ticks"]
            name = f"{mode}/{group['role_name']}/{group['sample_position']}"
            lines.append(f"{name:<33} {str(group['active']):>6} {group['samples']:8} "
                         f"{tick['median']:13.1f} {tick['min']:11} {tick['max']:11}")
    lines += ["Vector completed four-row sample medians (separate groups by K blocks):",
              f"{'role/class':<22} {'Kblks':>6} {'count':>6} {'empty':>9} {'load':>9} {'max/history':>12} {'subtract':>10} {'exp':>9} {'sum':>9}"]
    for group in summaries.get("phases", {}).get("groups", []):
        if group["kind"] != 2:
            continue
        fields = ("empty_ticks", "score_load_ticks", "max_history_ticks", "subtract_ticks", "exp_ticks", "sum_ticks")
        values = [group["ticks"][field]["median"] for field in fields]
        label = f"{group['role_name']}/{group['section_class']}"
        lines.append(f"{label:<22} {group['key_blocks']:6} {group['samples']:6} "
                     + " ".join(f"{value:10.1f}" for value in values))
    lines += ["load includes address/slot dependency/DMA/completion; max/history includes history exponentials.",
              "Vector completion fences change overlap. Do not sum samples into kernel milliseconds.",
              "Cross-core wait intervals do not measure the other core's entire computation."]
    return "\n".join(lines)


def worker(args):
    from profile_mindie_laser_quant import compare_all_rows, payload_evidence, verify_runtime_target
    from mindie_laser_phase_records import summarize_records
    providers, _ = resolve_providers(args)
    directory = Path(args._run_dir) / "paired" / "local"
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", stage="load", pid=os.getpid(), shape_bnsd=args.shape,
                  dtype="bfloat16", scope=SCOPE, traces=[], calibration={},
                  contract="Scale-reuse INT8 QK, original exp, FP16 PV; no V quantization/reorder/smoothing")
    save = lambda: common.write_json(directory / "report.json", report)
    save()
    try:
        import torch
        import torch_npu
        from mindie_laser_quant_backend import load_quant_laser
        torch.set_num_threads(min(args.cpu_threads, torch.get_num_threads()))
        torch.npu.set_device(args.device)
        torch.npu.set_option({"ACL_OP_COMPILER_CACHE_MODE": "disable",
                              "ACL_OP_COMPILER_CACHE_DIR": os.environ["ACL_OP_COMPILER_CACHE_DIR"]})
        old = load_quant_laser(providers["baseline"].bridge_path, preprocessing="fused",
            preprocess_build=args.preprocess_build, torch_namespace="vc_laser_quant_reuse",
            v_quantize=False, expcast=False, check_finite=False)
        torch.ops.load_library(providers["probe"].bridge_path)
        native = torch.ops.vc_laser_quant_probe.la.default
        if ([str(x.type) for x in native._schema.arguments] != ["Tensor"]*5+["float", "int", "bool", "int"]
                or [str(x.type) for x in native._schema.returns] != ["Tensor", "Tensor"]):
            raise RuntimeError(f"Unexpected diagnostic ABI: {native._schema}")
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtGetSocName.restype = ctypes.c_char_p
        runtime_soc = acl.aclrtGetSocName()
        if not runtime_soc:
            raise RuntimeError("aclrtGetSocName returned null")
        runtime_soc = runtime_soc.decode()
        preprocess = json.loads(Path(args.preprocess_build).read_text(encoding="utf-8"))
        builds = {name: json.loads(Path(p.build_manifest).read_text(encoding="utf-8"))
                  for name, p in providers.items()}
        report["runtime_targets"] = {name: verify_runtime_target(build, preprocess, runtime_soc,
            os.environ["ASCEND_HOME_PATH"]) for name, build in builds.items()}
        report["environment"] = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
            runtime_soc=runtime_soc, device_index=args.device, native_schema=str(native._schema))
        context = dict(cann_root=os.environ["ASCEND_HOME_PATH"], soc=builds["baseline"]["soc"])
        device = f"npu:{args.device}"
        with torch.no_grad():
            validate_modes(old, native, device, report, save)
            report["stage"] = "long_shape_all_output_validation"
            save()
            generator = torch.Generator().manual_seed(0)
            originals = [torch.randn(args.shape, generator=generator).to(torch.bfloat16) for _ in range(3)]
            report["input_sha256"] = [common.tensor_sha256(x) for x in originals]
            payload = old.prepare(*(x.to(device) for x in originals))
            del originals
            report["shared_payload"] = payload_evidence(payload)
            calls = {"reuse": lambda: reuse_call(old.native, payload),
                     **{label: lambda mode=mode: probe_call(native, payload, mode)
                        for label, mode in MODES.items()}}
            report["host_markers"], report["same_payload_all_output"] = {}, {}
            marker_sizes = {}
            expected = None
            for label, call in calls.items():
                provider = providers["baseline" if label == "reuse" else "probe"]
                marker = Path(os.environ[provider.marker_env])
                offset = marker.stat().st_size if marker.exists() else 0
                result = call()
                torch.npu.synchronize()
                with marker.open("rb") as stream:
                    stream.seek(offset)
                    fresh = stream.read().decode("utf-8")
                report["host_markers"][label] = quant.parse_quant_markers(fresh, os.getpid(), provider.marker_token)
                marker_sizes[marker] = marker.stat().st_size
                if label == "reuse":
                    expected = result
                else:
                    records = checked_trace(result, payload, MODES[label], require=True)
                    error = compare_all_rows(result[0], expected)
                    report["same_payload_all_output"][label] = error
                    if not error["exact_match"]:
                        raise AssertionError(f"{label} differs from scale reuse on full native output")
                    report.setdefault("initial_records", {})[label] = records
                del result
                save()
            del expected
            inspect_providers(providers, report, directory, **context)
            quant.inspect_fused_preprocessor(providers["baseline"], report)
            for provider in providers.values():
                os.environ.pop(provider.marker_env, None)
            report["stage"] = "paired_calibration"
            def progress(snapshot):
                report["timings"] = snapshot
                save()
                print(f"Probe calibration trial {snapshot['completed_trials']}/{args.trials}", flush=True)
            timing = run_paired(calls, synchronize=torch.npu.synchronize, warmup=args.warmup,
                repeats=args.repeats, trials=args.trials, seed=args.seed, on_trial=progress)
            report["timings"] = timing
            report["calibration"] = {f"{a} -> {b}": paired_comparison(timing, a, b) for a, b in PAIRS}
            if any(path.stat().st_size != size for path, size in marker_sizes.items()):
                raise RuntimeError("Source marker writes continued during calibration")
            print(calibration_table(report), flush=True)
            report["stage"] = "sample_records"
            save()
            rng = random.Random(args.seed + 1)
            for trial in range(args.samples):
                labels = ["waits", "phases"]
                rng.shuffle(labels)
                for label in labels:
                    result = calls[label]()
                    torch.npu.synchronize()
                    records = checked_trace(result, payload, MODES[label], require=True)
                    report["traces"].append(dict(trial=trial, variant=label, records=records))
                    del result
                save()
                print(f"Sparse phase sample {trial+1}/{args.samples}", flush=True)
            report["phase_summary"] = {label: summarize_records(
                [item["records"] for item in report["traces"] if item["variant"] == label])
                for label in ("waits", "phases")}
            print(phase_table(report["phase_summary"]), flush=True)
            inspect_providers(providers, report, directory, **context)
            quant.inspect_fused_preprocessor(providers["baseline"], report)
        report.update(status="passed", stage="complete")
        save()
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save()
        traceback.print_exc()
        return 1


def parent(args):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args._run_dir = tempfile.mkdtemp(prefix="mindie_laser_phases_", dir=output.parent)
    report = dict(status="running", run_dir=args._run_dir, scope=SCOPE)
    save = lambda: common.write_json(output, report)
    save()
    try:
        report["provenance"] = verify_builds(args)
        save()
        providers, _ = resolve_providers(args)
        directory = Path(args._run_dir) / "paired" / "local"
        directory.mkdir(parents=True)
        env = paired_environment(os.environ, providers, directory.parent)
        for key in ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker", "--_run-dir", args._run_dir]
        for key in ("reuse_build", "probe_build", "preprocess_build", "device", "warmup", "repeats",
                    "trials", "samples", "cpu_threads", "seed"):
            command += ["--"+key.replace("_", "-"), str(getattr(args, key))]
        command += ["--shape", *map(str, args.shape)]
        report["command"], report["worker_log"] = command, str(directory / "worker.log")
        save()
        with (directory / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        report["returncode"] = code
        worker_report = directory / "report.json"
        report["worker"] = json.loads(worker_report.read_text(encoding="utf-8")) if worker_report.is_file() else {}
        save()
        if code or report["worker"].get("status") != "passed":
            raise RuntimeError(f"Diagnostic worker failed; see {directory / 'worker.log'}")
        report["status"] = "passed"
        save()
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
                          ("probe-build", "results/mindie_laser_quant_probe_build.json"),
                          ("preprocess-build", "results/mindie_laser_quant_preprocess_build.json"),
                          ("output", "results/mindie_laser_phases.json")):
        parser.add_argument("--"+name, default=default)
    parser.add_argument("--shape", type=int, nargs=4, default=[1, 28, 37741, 128])
    for name, default in (("device", 0), ("warmup", 5), ("repeats", 10), ("trials", 9),
                          ("samples", 5), ("cpu-threads", 4), ("seed", 2026)):
        parser.add_argument("--"+name, type=int, default=default)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_run-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (min(args.shape) <= 0 or args.shape[-1] not in (64, 128) or args.device < 0 or args.warmup < 0
            or min(args.repeats, args.trials, args.samples, args.cpu_threads) <= 0):
        parser.error("Need nonempty D64/128, positive sample/trial counts, nonnegative device/warmup")
    return worker(args) if args._worker else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
