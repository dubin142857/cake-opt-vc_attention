#!/usr/bin/env python3
"""Original Laser / stable INT8 / independent binary16 ExpCast paired APIs."""
from __future__ import annotations

import argparse
import importlib
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
from profile_mindie_laser_vector_scale_bulk import inspect_providers, require_exact, SCOPES, BASELINE_DESCRIPTION

common = quant.common
ORIGINAL, STABLE, CONTROL, EXPCAST = "original_laser", "vector_pv_prefetch", "fp16_expcast_control", "fp16_expcast"
NAMES = (ORIGINAL, STABLE, CONTROL, EXPCAST)
PROVIDERS = (ORIGINAL, STABLE, EXPCAST)
MANIFESTS = dict(baseline_build="results/mindie_laser_build.json",
    bridge_manifest="results/mindie_laser_bridge.json",
    production_build="results/mindie_laser_quant_vscale_pv_prefetch_build.json",
    expcast_build="results/mindie_laser_quant_vscale_pv_expcast_build.json",
    preprocess_build="results/mindie_laser_quant_preprocess_build.json")


def specifications(args):
    return {name: dict(path=path, source=importlib.import_module("prepare_mindie_laser_quant_"+suffix+"_source"),
                      namespace="vc_laser_quant_"+suffix)
            for name, path, suffix in ((STABLE, args.production_build, "vscale_pv_prefetch"),
                                      (EXPCAST, args.expcast_build, "vscale_pv_expcast"))}


def resolve_providers(args):
    original = quant.build_arguments(args, "baseline")
    original.marker_env, original.marker_token = quant.baseline.MARKER_ENV, quant.baseline.MARKER_TOKEN
    providers = {ORIGINAL: original}
    for name, spec in specifications(args).items():
        options = SimpleNamespace(**vars(args), quant_build=spec["path"])
        provider = quant.build_arguments(options, "qk_int8_fused")
        provider.marker_env, provider.marker_token = spec["source"].MARKER_ENV, spec["source"].MARKER_TOKEN
        providers[name] = provider
    if len({str(Path(p.vendor_root).resolve()) for p in providers.values()}) != 3:
        raise ValueError("Original, stable and FP16 ExpCast require three distinct vendor roots")
    return providers


def verify_builds(args, providers):
    from mindie_laser_quant_fused import validate_preprocess_manifest
    evidence = {ORIGINAL: quant.baseline.provenance(providers[ORIGINAL]),
                "preprocessor": validate_preprocess_manifest(args.preprocess_build)}
    for name, spec in specifications(args).items():
        evidence[name] = quant.quant_provenance(providers[name], identity=spec["source"].IDENTITY,
                                               source_validator=spec["source"].validate_prepared_source)
    return evidence


def worker_environment(base, providers, directory):
    roots = [str(Path(providers[name].vendor_root).resolve()) for name in PROVIDERS]
    if len(set(roots)) != 3:
        raise ValueError("Three distinct provider roots required")
    env = quant.baseline.worker_environment(base, "local", roots[0], directory)
    rest = [p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if p and not any(common.within(p, root) for root in roots)]
    env["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(roots)
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(Path(root)/"op_api/lib") for root in roots]+rest)
    for key in list(env):
        if key.startswith("VC_") and key.endswith("MARKER_FILE"):
            env.pop(key)
    for name, provider in providers.items():
        env[provider.marker_env] = str(Path(directory)/"local"/(name+"_markers.jsonl"))
    return env


class DeviceGuard:
    """Save first failure using host data only; never retry a failed device."""
    def __init__(self, report, save, synchronize):
        self.report, self.save, self.synchronize = report, save, synchronize
        self.failed = False

    def __call__(self, label, function):
        if self.failed:
            raise RuntimeError("Further device access aborted after first failure")
        self.report["active_device_operation"] = dict(label=label, phase="about_to_call")
        self.save()
        try:
            result = function()
            self.report["active_device_operation"]["phase"] = "awaiting_sync"
            self.save()
            self.synchronize()
            self.report["active_device_operation"]["phase"] = "completed"
            return result
        except Exception as error:
            self.failed = True
            self.report.setdefault("first_device_failure", dict(operation=dict(self.report["active_device_operation"]),
                validation=self.report.get("active_validation"), error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(), further_device_access_aborted=True))
            self.save()
            raise


def check_schema(native):
    schema = native._schema
    if ([str(x.type) for x in schema.arguments] != ["Tensor"]*5+["float", "int", "bool"] or
            [str(x.type) for x in schema.returns] != ["Tensor"]):
        raise RuntimeError(f"Unexpected INT8 Laser schema: {schema}")


def make_calls(adapters, payloads, inputs):
    from mindie_laser_fp16_trace_worker import native_call
    if tuple(adapters) != NAMES or set(payloads) != set(NAMES):
        raise ValueError("Require all four benchmark rows")
    if not (payloads[STABLE] is payloads[CONTROL] is payloads[EXPCAST]):
        raise ValueError("All INT8 fixed scopes must share identical tensor objects")
    if payloads[ORIGINAL] is payloads[STABLE]:
        raise ValueError("Original FP16 uses its own payload")
    return {
        "full": {name: lambda a=a: a(*inputs) for name, a in adapters.items()},
        "prepared": {name: lambda a=a, p=payloads[name]: a.run_prepared(p) for name, a in adapters.items()},
        "native": {name: (lambda a=a, p=payloads[name]: native_call(a, p)) if name == ORIGINAL
                   else (lambda a=a, p=payloads[name]: a.run_native(p)) for name, a in adapters.items()},
    }


def postprocess_output(name, adapter, raw, payload):
    if name == ORIGINAL:
        if not isinstance(raw, (tuple,list)) or len(raw) != 2:
            raise RuntimeError("Original native Laser must return statistics/output")
        return adapter.postprocess(raw[1], payload.dtype, payload.shape[2], payload.shape[3])
    return adapter.postprocess(raw, payload)


def comparisons(timing):
    return {"vs_original": {name: paired_comparison(timing, ORIGINAL, name) for name in NAMES[1:]},
            "vs_stable": {name: paired_comparison(timing, STABLE, name) for name in (CONTROL, EXPCAST)},
            "feature_toggle": paired_comparison(timing, CONTROL, EXPCAST)}


def table(report):
    lines = ["Same-process randomized synchronized API ms; NO profiler:",
             "FP16 ExpCast is an experimental binary16 adaptation; not native FP8 or the paper's E4M3 path."]
    for scope in SCOPES:
        timing = report["timings"][scope]
        base = timing["rows"][ORIGINAL]["median_ms"]
        lines += [f"{scope}: original FP16 Laser is the primary baseline",
                  f"{'variant':<25} {'API ms':>10} {'Laser/row':>11} {'saved ms':>11} {'faster':>8} {'dense RMSE %':>14}"]
        for name in NAMES:
            value = timing["rows"][name]["median_ms"]
            pair = report["comparisons"][scope]["vs_original"].get(name)
            saved = f"{pair['median_saved_ms']:+.4f}" if pair else "--"
            faster = f"{pair['faster_trials']}/{pair['trials']}" if pair else "--"
            error = 100*report["dense_fp32_samples"][name]["relative_rmse"]
            lines.append(f"{name:<25} {value:10.4f} {base/value:10.4f}x {saved:>11} {faster:>8} {error:14.6f}")
        pair = report["comparisons"][scope]["feature_toggle"]
        lines.append(f"same-provider control/ExpCast: paired saved {pair['median_saved_ms']:+.4f} ms; "
                     f"faster {pair['faster_trials']}/{pair['trials']}")
    lines += ["Fixed INT8 scopes share quantized tensors; full freshly preprocesses every call.",
              "Control/stable checks ALL padded outputs exactly. Different algorithms have no equality gate.",
              "Scopes are independent, not additive or kernel-only timings. Dense RMSE is not model quality."]
    return "\n".join(lines)


def worker(args):
    directory = Path(args._run_dir)/"paired/local"
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", stage="provenance", pid=os.getpid(), primary_baseline=ORIGINAL,
        original_baseline=BASELINE_DESCRIPTION, shape_bnsd=args.shape, dtype="bfloat16", scopes=SCOPES,
        timings={}, comparisons={}, no_v_quantization=True, no_v_reordering=True, no_v_smoothing=True,
        instrumentation=False, timing_scope="Synchronized API wall times; no profiler")
    save = lambda: common.write_json(directory/"report.json", report)
    save()
    try:
        providers = resolve_providers(args)
        evidence = report["provenance"] = verify_builds(args, providers)
        import ctypes
        import torch
        import torch_npu
        from mindie_laser_quant_backend import load_quant_laser, MindIEQuantLaser
        from profile_mindie_laser_quant import compare_all_rows, payload_evidence, verify_runtime_target
        from mindie_laser_fp16_trace_worker import verify_runtime_target as verify_fp16_target
        from mindie_laser_fp16_expcast_reference import validate, checked
        from mindie_laser_fp16_expcast_math import CONTRACT
        torch.set_num_threads(min(args.cpu_threads, torch.get_num_threads()))
        original, transforms = quant.baseline.load_adapter(providers[ORIGINAL], evidence[ORIGINAL]["source_manifest"])
        stable = load_quant_laser(providers[STABLE].bridge_path, preprocessing="fused",
            preprocess_build=args.preprocess_build, torch_namespace="vc_laser_quant_vscale_pv_prefetch",
            v_quantize=False, expcast=False, check_finite=False)
        from mindie_laser_quant_vscale_patch import VSCALE_CONTRACT
        stable.metadata["qk_result_contract"] = VSCALE_CONTRACT
        torch.ops.load_library(providers[EXPCAST].bridge_path)
        native = torch.ops.vc_laser_quant_vscale_pv_expcast.la.default
        check_schema(native)
        new = {name: MindIEQuantLaser(native, quantizer=stable.quantizer, v_quantize=False,
            expcast=name == EXPCAST, check_finite=False, metadata=dict(stable.metadata,
            dispatch="torch.ops.vc_laser_quant_vscale_pv_expcast.la.default", fp16_expcast_contract=CONTRACT,
            bridge_path=providers[EXPCAST].bridge_path, native_schema=str(native._schema)))
            for name in (CONTROL, EXPCAST)}
        adapters = {ORIGINAL: original, STABLE: stable, **new}
        report["source_transform_validation"], report["adapters"] = transforms, {n:a.metadata for n,a in adapters.items()}
        torch.npu.set_option({"ACL_OP_COMPILER_CACHE_MODE": "disable",
                              "ACL_OP_COMPILER_CACHE_DIR": os.environ["ACL_OP_COMPILER_CACHE_DIR"]})
        torch.npu.set_device(args.device)
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtGetSocName.restype = ctypes.c_char_p
        raw_soc = acl.aclrtGetSocName()
        if not raw_soc:
            raise RuntimeError("aclrtGetSocName returned null")
        soc, cann = raw_soc.decode(), os.environ["ASCEND_HOME_PATH"]
        prep = json.loads(Path(args.preprocess_build).read_text(encoding="utf-8"))
        builds = {name:evidence[name]["build_manifest"] for name in PROVIDERS}
        report["runtime_targets"] = {ORIGINAL:verify_fp16_target(builds[ORIGINAL], soc, cann),
            **{name:verify_runtime_target(builds[name], prep, soc, cann) for name in (STABLE, EXPCAST)}}
        report["environment"] = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
            runtime_soc=soc, device_index=args.device, custom_opp_path=os.environ["ASCEND_CUSTOM_OPP_PATH"])
        context = dict(cann_root=cann, soc=builds[ORIGINAL]["soc"])
        device, call = f"npu:{args.device}", DeviceGuard(report, save, torch.npu.synchronize)
        with torch.no_grad():
            report["stage"] = "structured_validation"
            validate(adapters, device, report, save, call)
            report["original_validation"] = {}
            call("original_small_validation", lambda: quant.baseline.validate_small(
                original, device, report["original_validation"], save, torch))
            report["stage"] = "long_shape_validation"
            report.pop("active_validation", None)
            generator = torch.Generator().manual_seed(0)
            host = [torch.randn(args.shape, generator=generator).to(torch.bfloat16) for _ in range(3)]
            if not all(bool(torch.isfinite(x).all()) for x in host):
                raise ValueError("Finite benchmark inputs required")
            report["input_sha256"] = [common.tensor_sha256(x) for x in host]
            inputs = call("long_inputs", lambda: tuple(x.to(device) for x in host))
            fixed = call("long_int8_prepare", lambda: stable.prepare(*inputs))
            payloads = {ORIGINAL:call("long_original_prepare", lambda: original.prepare(*inputs)),
                        STABLE:fixed, CONTROL:fixed, EXPCAST:fixed}
            report["payloads"] = {ORIGINAL:original.describe_payload(payloads[ORIGINAL]), "shared_int8":payload_evidence(fixed)}
            calls = make_calls(adapters, payloads, inputs)
            raw, marker_sizes = {}, {}
            report["host_markers"] = {}
            for name in NAMES:
                provider = providers[EXPCAST if name == CONTROL else name]
                marker = Path(os.environ[provider.marker_env])
                offset = marker.stat().st_size if marker.exists() else 0
                raw[name] = call("long_native_"+name, calls["native"][name])
                with marker.open("rb") as stream:
                    stream.seek(offset)
                    fresh = stream.read().decode("utf-8")
                report["host_markers"][name] = (quant.baseline.parse_markers(fresh, os.getpid()) if name == ORIGINAL
                    else quant.parse_quant_markers(fresh, os.getpid(), provider.marker_token))
                marker_sizes[marker] = marker.stat().st_size
            inspect_providers(providers, evidence, report, directory, **context)
            quant.inspect_fused_preprocessor(providers[STABLE], report)
            for provider in providers.values():
                os.environ.pop(provider.marker_env, None)
            report["control_vs_stable_all_padded"] = call("long_control_exact", lambda: compare_all_rows(raw[CONTROL], raw[STABLE]))
            require_exact(report["control_vs_stable_all_padded"])
            ids = torch.linspace(0, args.shape[2]-1, min(32, args.shape[2])).long().unique()
            dense = common.dense_reference(*host, ids)
            report["dense_fp32_samples"], report["full_prepared_samples"] = {}, {}
            for name, adapter in adapters.items():
                def postprocess():
                    p = payloads[name]
                    out = postprocess_output(name, adapter, raw[name], p)
                    return out.index_select(2, ids.to(device)).cpu()
                sampled = call("long_sample_"+name, postprocess)
                error = common.error_metrics(sampled, dense)
                report["dense_fp32_samples"][name] = checked(error, name+" dense finite", .01 if name == ORIGINAL else float("inf"))
                full = call("long_full_"+name, lambda n=name: calls["full"][n]().index_select(2, ids.to(device)).cpu())
                prepared = call("long_prepared_"+name, lambda n=name: calls["prepared"][n]().index_select(2, ids.to(device)).cpu())
                report["full_prepared_samples"][name] = checked(common.error_metrics(full, prepared), name+" full/prepared", .001)
                checked(common.error_metrics(sampled, prepared), name+" native/prepared", .001)
            report["accuracy_scope"] = dict(query_indices=ids.tolist(), all_keys=args.shape[2],
                control_vs_stable="ALL padded native rows", dense_quality_gate=False)
            del raw, host, dense
            for scope, group in calls.items():
                report["stage"] = "timing_"+scope
                def progress(snapshot, scope=scope):
                    report["timings"][scope] = snapshot
                    save()
                    print(f"{scope}: paired trial {snapshot['completed_trials']}/{args.trials}", flush=True)
                report["timings"][scope] = call("paired_"+scope, lambda: run_paired(group,
                    synchronize=torch.npu.synchronize, warmup=args.warmup, repeats=args.repeats,
                    trials=args.trials, seed=args.seed, on_trial=progress))
                report["comparisons"][scope] = comparisons(report["timings"][scope])
                save()
            if any(path.stat().st_size != size for path,size in marker_sizes.items()):
                raise RuntimeError("Source markers continued writing during timing")
            inspect_providers(providers, evidence, report, directory, **context)
            quant.inspect_fused_preprocessor(providers[STABLE], report)
        report.update(status="passed", stage="complete")
        save()
        print(table(report), flush=True)
        return 0
    except Exception as error:
        # No synchronize, copies, profiler, cleanup launches or output checks here.
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc(), further_device_access_aborted=True)
        save()
        traceback.print_exc()
        return 1


def launch(args, env, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for key in ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker", "--_run-dir", args._run_dir]
    for key in (*MANIFESTS, "device", "warmup", "repeats", "trials", "seed", "cpu_threads"):
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
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(status="missing_report")
    result.update(returncode=code, command=command, worker_log=str(directory/"worker.log"))
    return result


def parent(args):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args._run_dir = tempfile.mkdtemp(prefix="mindie_laser_fp16_expcast_", dir=output.parent)
    report = dict(status="running", primary_baseline=ORIGINAL, run_dir=args._run_dir, workers={})
    save = lambda: common.write_json(output, report)
    save()
    try:
        providers = resolve_providers(args)
        report["provenance"] = verify_builds(args, providers)
        save()
        directory = Path(args._run_dir)/"paired"
        result = report["workers"]["paired"] = launch(args, worker_environment(os.environ, providers, directory), directory/"local")
        save()
        if result.get("status") != "passed" or result.get("returncode") != 0:
            raise RuntimeError("FP16 ExpCast validation/timing failed; see worker_log. No retry launched.")
        report["status"] = "passed"
        save()
        print(f"Report: {output}", flush=True)
        return 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        traceback.print_exc()
        return 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for key, default in MANIFESTS.items():
        result.add_argument("--"+key.replace("_", "-"), default=default)
    result.add_argument("--shape", nargs=4, type=int, default=[1,28,37741,128])
    result.add_argument("--output", default="results/mindie_laser_fp16_expcast.json")
    for key, default in (("device",0), ("warmup",5), ("repeats",10), ("trials",9), ("seed",2026), ("cpu-threads",4)):
        result.add_argument("--"+key, type=int, default=default)
    result.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--_run-dir", help=argparse.SUPPRESS)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if (min(args.shape) < 1 or args.shape[-1] not in (64,128) or args.device < 0 or
            args.warmup < 0 or min(args.repeats,args.trials,args.cpu_threads) < 1):
        cli.error("Require positive BNSD D64/128, nonnegative device/warmup and positive repeats/trials/threads")
    return worker(args) if args._worker else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
