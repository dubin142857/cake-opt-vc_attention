#!/usr/bin/env python3
"""Paired INT8 scale scheduling experiments, plus isolated original FP16 counters."""
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

import benchmark_mindie_laser_quant as quant
from mindie_laser_quant_pairing import run_paired, paired_comparison
from mindie_laser_scale_reuse_checks import (
    paired_environment, inspect_providers, make_calls, validate_scale_reuse,
)

common = quant.common
SCOPES = {
    "native": "Identical INT8 Q/K, FP16 V and FP32 scales; native allocation/dispatch only",
    "prepared": "Identical fixed payload + native allocation/dispatch + output postprocess",
    "full": "Fresh fused Q/K preprocessing, conversion/padding, attention and postprocess per call",
}


def experiment(args):
    """Keep each candidate's identity and source validator bound together."""
    from importlib import import_module
    kind = getattr(args, "experiment", "scale-reuse")
    if kind == "scale-reuse":
        names = ("quant", "quant_reuse")
        paths = (args.quant_build, args.reuse_build)
        labels = ("baseline", "scale reuse")
    elif kind == "gain-precompute":
        names = ("quant_reuse", "quant_gain")
        paths = (args.reuse_build, args.gain_build)
        labels = ("scale reuse", "gain ready")
    else:
        raise ValueError(f"Unknown paired experiment: {kind}")
    return {key: dict(source=import_module("prepare_mindie_laser_"+name+"_source"),
                     path=path, namespace="vc_laser_"+name, label=label)
            for key, name, path, label in zip(("baseline", "reuse"), names, paths, labels)}


def resolve_providers(args):
    providers = {}
    for name, spec in experiment(args).items():
        options = SimpleNamespace(**vars(args))
        options.quant_build = spec["path"]
        provider = quant.build_arguments(options, "qk_int8_fused")
        provider.marker_env, provider.marker_token = spec["source"].MARKER_ENV, spec["source"].MARKER_TOKEN
        providers[name] = provider
    if Path(providers["baseline"].vendor_root).resolve() == Path(providers["reuse"].vendor_root).resolve():
        raise ValueError("Baseline and candidate must have independent vendor roots")
    return providers


def verify_builds(args):
    from mindie_laser_quant_fused import validate_preprocess_manifest
    providers = resolve_providers(args)
    specs = experiment(args)
    return {
        **{name: quant.quant_provenance(provider, identity=specs[name]["source"].IDENTITY,
            source_validator=specs[name]["source"].validate_prepared_source)
           for name, provider in providers.items()},
        "preprocessor": validate_preprocess_manifest(args.preprocess_build),
        "fp16": quant.baseline.provenance(quant.build_arguments(args, "baseline")),
    }


def table(report):
    labels = report.get("variant_labels", {"baseline": "baseline", "reuse": "scale reuse"})
    lines = ["Same-process paired wall ms/API; no profiler active:",
             f"{'scope':<15} {labels['baseline']:>12} {labels['reuse']:>12} {'old/new':>10} {'saved ms':>12} {'faster':>8}"]
    for scope, timing in report.get("timings", {}).items():
        old, new = (timing["rows"][name]["median_ms"] for name in ("baseline", "reuse"))
        pair = report["comparisons"][scope]
        lines.append(f"{scope:<15} {old:12.4f} {new:12.4f} {old/new:9.4f}x "
                     f"{pair['median_saved_ms']:+12.4f} {pair['faster_trials']:3}/{pair['trials']}")
    lines += ["native/prepared reuse identical payload objects; full freshly preprocesses every call.",
              "Positive paired saved means candidate is faster. Scopes are not additive or kernel-only times."]
    return "\n".join(lines)


def worker(args):
    from profile_mindie_laser_quant import compare_all_rows, payload_evidence, verify_runtime_target
    providers = resolve_providers(args)
    specs = experiment(args)
    directory = Path(args._run_dir) / "paired" / "local"
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", stage="load", pid=os.getpid(), shape_bnsd=args.shape,
                  dtype="bfloat16", seed=0, scopes=SCOPES, timings={}, comparisons={},
                  experiment=args.experiment,
                  variant_labels={name: spec["label"] for name, spec in specs.items()},
                  contract="INT8 QK, original exp, FP16 PV; no V quantization/reorder/smoothing")
    save = lambda: common.write_json(directory / "report.json", report)
    save()
    try:
        import torch
        import torch_npu
        from mindie_laser_quant_backend import load_quant_laser, MindIEQuantLaser
        from mindie_laser_quant_reference import validate_small
        torch.set_num_threads(min(args.cpu_threads, torch.get_num_threads()))
        old = load_quant_laser(providers["baseline"].bridge_path, preprocessing="fused",
            preprocess_build=args.preprocess_build, torch_namespace=specs["baseline"]["namespace"],
            v_quantize=False, expcast=False, check_finite=False)
        torch.ops.load_library(providers["reuse"].bridge_path)
        namespace = specs["reuse"]["namespace"]
        native = getattr(torch.ops, namespace).la.default
        if ([str(x.type) for x in native._schema.arguments] != ["Tensor"]*5+["float", "int", "bool"]
                or [str(x.type) for x in native._schema.returns] != ["Tensor"]):
            raise RuntimeError(f"Unexpected candidate schema: {native._schema}")
        adapters = {"baseline": old, "reuse": MindIEQuantLaser(native,
            quantizer=old.quantizer, v_quantize=False, expcast=False, check_finite=False,
            metadata=dict(old.metadata, dispatch=f"torch.ops.{namespace}.la.default",
                          bridge_path=providers["reuse"].bridge_path, native_schema=str(native._schema)))}
        torch.npu.set_option({"ACL_OP_COMPILER_CACHE_MODE": "disable",
                              "ACL_OP_COMPILER_CACHE_DIR": os.environ["ACL_OP_COMPILER_CACHE_DIR"]})
        torch.npu.set_device(args.device)
        device = f"npu:{args.device}"
        import ctypes
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
            runtime_soc=runtime_soc, device_index=args.device, custom_opp_path=os.environ["ASCEND_CUSTOM_OPP_PATH"])
        report["adapters"] = {name: a.metadata for name, a in adapters.items()}
        provider_context = dict(cann_root=os.environ["ASCEND_HOME_PATH"], soc=builds["baseline"]["soc"])
        with torch.no_grad():
            report["stage"], report["validation"] = "structured_validation", {}
            for name, adapter in adapters.items():
                report["validation"][name] = {}
                validate_small(adapter, device, report["validation"][name], save, torch)
            validate_scale_reuse(adapters, device, report, save)
            if args.experiment == "gain-precompute":
                from mindie_laser_gain_checks import validate_block_gains
                validate_block_gains(adapters, device, report, save)
            report["stage"] = "long_shape_validation"
            save()
            generator = torch.Generator().manual_seed(0)
            original = [torch.randn(args.shape, generator=generator).to(torch.bfloat16) for _ in range(3)]
            report["input_sha256"] = [common.tensor_sha256(x) for x in original]
            inputs = [x.to(device) for x in original]
            payload = old.prepare(*inputs)
            report["shared_payload"] = payload_evidence(payload)
            calls = make_calls(adapters, payload, inputs)
            raw, markers = {}, {}
            report["host_markers"] = {}
            for name in adapters:
                marker = Path(os.environ[providers[name].marker_env])
                offset = marker.stat().st_size if marker.exists() else 0
                raw[name] = calls["native"][name]()
                torch.npu.synchronize()
                with marker.open("rb") as stream:
                    stream.seek(offset)
                    fresh = stream.read().decode("utf-8")
                report["host_markers"][name] = quant.parse_quant_markers(
                    fresh, os.getpid(), providers[name].marker_token)
                (directory / (name + "_long_markers.jsonl")).write_text(fresh, encoding="utf-8")
                markers[marker] = marker.stat().st_size
            inspect_providers(providers, report, directory, **provider_context)
            quant.inspect_fused_preprocessor(providers["baseline"], report)
            for p in providers.values():
                os.environ.pop(p.marker_env, None)
            report["same_payload_all_output"] = compare_all_rows(raw["reuse"], raw["baseline"])
            save()
            if not report["same_payload_all_output"]["exact_match"]:
                raise AssertionError("Candidate must preserve ALL native output elements exactly")
            ids = torch.linspace(0, args.shape[2]-1, min(32, args.shape[2])).long().unique()
            dense = common.dense_reference(*original, ids)
            report["dense_fp32_samples"] = {name: common.error_metrics(
                adapters[name].postprocess(out, payload).index_select(2, ids.to(device)).cpu(), dense)
                for name, out in raw.items()}
            report["accuracy_scope"] = dict(query_ids=ids.tolist(), all_keys=args.shape[2],
                note="Dense RMSE is an algorithm diagnostic, not a model-quality gate")
            del raw, original, dense
            save()
            for scope, group in calls.items():
                report["stage"] = "timing_"+scope
                def progress(snapshot, scope=scope):
                    report["timings"][scope] = snapshot
                    save()
                    print(f"{scope}: paired trial {snapshot['completed_trials']}/{args.trials}", flush=True)
                timing = run_paired(group, synchronize=torch.npu.synchronize, warmup=args.warmup,
                    repeats=args.repeats, trials=args.trials, seed=args.seed, on_trial=progress)
                report["timings"][scope] = timing
                report["comparisons"][scope] = paired_comparison(timing, "baseline", "reuse")
                save()
            if any(path.stat().st_size != size for path, size in markers.items()):
                raise RuntimeError("Source marker writes continued during timing")
            report["timing_status"] = "passed"
            print(table(report), flush=True)
            save()
            if args.trace:
                from mindie_laser_quant_trace import capture_native_traces
                from summarize_mindie_laser_quant_profile import render_report
                report["stage"] = "optional_profiler"
                report["trace"] = capture_native_traces(calls["native"], torch=torch, torch_npu=torch_npu,
                    directory=directory / "profiler", repeats=3)
                save()
                print(render_report(report["trace"]), flush=True)
            inspect_providers(providers, report, directory, **provider_context)
            quant.inspect_fused_preprocessor(providers["baseline"], report)
        report.update(status="passed", stage="complete")
        save()
        return 0
    except Exception as exc:
        if getattr(exc, "partial_report", None) is not None:
            report["trace"] = exc.partial_report
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save()
        traceback.print_exc()
        return 1


def launch(args, mode, env, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for key in ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker", mode,
               "--_run-dir", args._run_dir]
    for key in ("quant_build", "reuse_build", "gain_build", "experiment", "preprocess_build", "baseline_build", "bridge_manifest",
                "device", "warmup", "repeats", "trials", "cpu_threads", "seed"):
        command += ["--"+key.replace("_", "-"), str(getattr(args, key))]
    command += ["--shape", *map(str, args.shape)]
    if args.trace:
        command += ["--trace"]
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
    result.update(returncode=code, command=command, worker_log=str(directory / "worker.log"))
    return result


def parent(args):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    kind = getattr(args, "experiment", "scale-reuse")
    args._run_dir = tempfile.mkdtemp(prefix="mindie_laser_"+kind.replace("-", "_")+"_", dir=output.parent)
    report = dict(status="running", run_dir=args._run_dir, workers={},
        experiment=kind,
        scope="Paired identical-payload INT8 kernel A/B; original FP16 counters in a separate process, not paired speedup")
    save = lambda: common.write_json(output, report)
    save()
    try:
        report["provenance"] = verify_builds(args)
        save()
        providers = resolve_providers(args)
        paired_dir = Path(args._run_dir) / "paired"
        env = paired_environment(os.environ, providers, paired_dir)
        report["workers"]["paired"] = launch(args, "paired", env, paired_dir / "local")
        save()
        if (report["workers"]["paired"].get("returncode")
                or report["workers"]["paired"].get("status") != "passed"):
            raise RuntimeError("Paired candidate worker failed; see its worker_log in report")
        fp16 = quant.build_arguments(args, "baseline")
        env = quant.baseline.worker_environment(os.environ, "local", fp16.vendor_root, fp16._run_dir)
        for key in list(env):
            if key.startswith("VC_") and key.endswith("MARKER_FILE") and key != quant.baseline.MARKER_ENV:
                env.pop(key)
        report["workers"]["fp16"] = launch(args, "fp16", env, Path(fp16._run_dir) / "local")
        save()
        if (report["workers"]["fp16"].get("returncode")
                or report["workers"]["fp16"].get("status") != "passed"):
            raise RuntimeError("Original FP16 worker failed; paired timings remain saved")
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
    for name, default in (
        ("quant-build", "results/mindie_laser_quant_build.json"),
        ("reuse-build", "results/mindie_laser_quant_reuse_build.json"),
        ("gain-build", "results/mindie_laser_quant_gain_build.json"),
        ("preprocess-build", "results/mindie_laser_quant_preprocess_build.json"),
        ("baseline-build", "results/mindie_laser_build.json"),
        ("bridge-manifest", "results/mindie_laser_bridge.json"),
        ("output", "results/mindie_laser_scale_reuse.json"),
    ):
        parser.add_argument("--"+name, default=default)
    parser.add_argument("--experiment", choices=("scale-reuse", "gain-precompute"), default="scale-reuse")
    parser.add_argument("--shape", type=int, nargs=4, default=list(quant.baseline.SHAPE))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--_worker", choices=("paired", "fp16"), help=argparse.SUPPRESS)
    parser.add_argument("--_run-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (min(args.shape) <= 0 or args.shape[-1] not in (64, 128) or args.device < 0
            or args.warmup < 0 or min(args.repeats, args.trials, args.cpu_threads) <= 0):
        parser.error("Need nonempty BNSD D64/128, nonnegative device/warmup and positive repeats/trials/threads")
    if args._worker == "fp16":
        from mindie_laser_fp16_trace_worker import run
        return run(args)
    return worker(args) if args._worker == "paired" else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
