#!/usr/bin/env python3
"""Same-process paired diagnostics for the existing INT8 Laser build; no rebuild."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

import benchmark_mindie_laser_quant as quant
from mindie_laser_quant_pairing import run_paired, paired_comparison

common = quant.common
FIELDS = ("q", "k", "v", "q_scale", "k_scale")
SCOPES = {
    "shared_native": "Both adapters call the SAME native API on the SAME fused payload object/storage; fresh output allocation; no preprocessing/postprocess",
    "own_native": "Same native API, independently prepared fixed torch/fused payloads; fresh output allocation; no preprocessing/postprocess",
    "prepared": "Each adapter's fixed payload + native allocation/dispatch + output postprocess",
    "full": "Each call freshly prepares original QKV, calls native attention and postprocesses output",
    "preprocessing": "Each call freshly converts/pads/quantizes original QKV; no attention",
}


def make_calls(adapters, payloads, inputs):
    """Keep payload identity explicit; never re-prepare in fixed-payload timers."""
    return {
        "shared_native": {name: lambda a=adapter: a.run_native(payloads["fused"])
                          for name, adapter in adapters.items()},
        "own_native": {name: lambda a=adapter, p=payloads[name]: a.run_native(p)
                       for name, adapter in adapters.items()},
        "prepared": {name: lambda a=adapter, p=payloads[name]: a.run_prepared(p)
                     for name, adapter in adapters.items()},
        "full": {name: lambda a=adapter: a(*inputs) for name, adapter in adapters.items()},
        "preprocessing": {name: lambda a=adapter: a.prepare(*inputs) for name, adapter in adapters.items()},
    }


def compare_all_rows(actual, reference, *, chunk=256):
    """Bound CPU temporary memory and check *all* padded output elements."""
    import torch
    if actual.shape != reference.shape or actual.dtype != reference.dtype or actual.ndim != 4:
        raise ValueError("Native output shape/dtype mismatch")
    squared_error = squared_reference = maximum = 0.0
    exact = True
    for begin in range(0, actual.shape[2], chunk):
        a = actual[:, :, begin:begin + chunk].detach().cpu().double()
        b = reference[:, :, begin:begin + chunk].detach().cpu().double()
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            raise ValueError("Non-finite native output")
        difference = a - b
        squared_error += difference.square().sum().item()
        squared_reference += b.square().sum().item()
        maximum = max(maximum, difference.abs().max().item())
        exact = exact and torch.equal(a, b)
    return {"relative_rmse": math.sqrt(squared_error / max(squared_reference, 1e-24)),
            "max_abs": maximum, "exact_match": exact, "elements": actual.numel(),
            "scope": "All padded output rows/heads/features, outside timing"}


def payload_evidence(payload):
    return {name: {"data_ptr": getattr(payload, name).data_ptr(),
                   "shape": list(getattr(payload, name).shape),
                   "dtype": str(getattr(payload, name).dtype),
                   "sha256": common.tensor_sha256(getattr(payload, name).detach().cpu())}
            for name in FIELDS}


def verify_runtime_target(build, preprocess, runtime_soc, cann_root):
    """Do not reuse a direct-launch binary built for another exact device SKU."""
    from check_ascend910b import resolve_soc
    selection = resolve_soc(runtime_soc, None, Path(cann_root))
    if selection["errors"]:
        raise ValueError("; ".join(selection["errors"]))
    if preprocess.get("soc", "").casefold() != selection["compiler_soc"].casefold():
        raise ValueError("Fused quantizer build targets another exact runtime SoC; rebuild for " + runtime_soc)
    family = "ascend910_93" if runtime_soc.lower().startswith("ascend910_93") else "ascend910b"
    if build.get("soc", "").lower() != family:
        raise ValueError("Laser attention build targets another runtime SoC family")
    for manifest in (build, preprocess):
        root = manifest.get("cann_root") or manifest.get("definitions", {}).get("CUSTOM_ASCEND_CANN_PACKAGE_PATH")
        if not root or Path(root).resolve() != Path(cann_root).resolve():
            raise ValueError("Build manifest targets another CANN Toolkit installation")
    return selection


def table(report):
    lines = ["Same-process randomized paired API wall milliseconds (no profiler active):",
             f"{'scope':<19} {'torch prep':>12} {'fused prep':>12} {'old/fused':>11} {'paired saved':>14} {'faster':>9}"]
    for scope, timing in report.get("timings", {}).items():
        first, second = (timing["rows"][name]["median_ms"] for name in ("torch", "fused"))
        pair = report["comparisons"][scope]
        lines.append(f"{scope:<19} {first:12.4f} {second:12.4f} {first / second:10.4f}x "
                     f"{pair['median_saved_ms']:+14.4f} {pair['faster_trials']:3}/{pair['trials']:<3}")
    lines += ["shared_native uses identical tensor objects and dispatch: adapter labels are NOT different kernels.",
              "own_native differs in prepared buffers/codes; it does not isolate quantization code from pointer/cache effects.",
              "Positive paired saved ms means fused is faster. Independent scopes are NOT additive or kernel phase percentages."]
    return "\n".join(lines)


def worker(args):
    resolved = quant.build_arguments(args, "qk_int8_fused")
    directory = Path(resolved._run_dir) / "local"
    directory.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "stage": "load_runtime", "pid": os.getpid(),
              "shape_bnsd": args.shape, "dtype": "bfloat16", "seed": 0,
              "scopes": SCOPES, "timings": {}, "comparisons": {},
              "contract": "Native INT8 QK, original exp, FP16 PV; no V quantization/smoothing/reordering"}
    save = lambda: common.write_json(directory / "report.json", report)
    save()
    try:
        import torch
        import torch_npu
        from mindie_laser_quant_backend import load_quant_laser
        from mindie_laser_quant_reference import validate_small
        torch.set_num_threads(min(args.cpu_threads, torch.get_num_threads()))
        adapters = {name: load_quant_laser(resolved.bridge_path, preprocessing=name,
                        preprocess_build=resolved.preprocess_build if name == "fused" else None,
                        v_quantize=False, expcast=False, check_finite=False)
                    for name in ("torch", "fused")}
        if adapters["torch"].native is not adapters["fused"].native:
            raise AssertionError("Paired adapters must share the identical native operator object")
        torch.npu.set_option({"ACL_OP_COMPILER_CACHE_MODE": "disable",
                              "ACL_OP_COMPILER_CACHE_DIR": os.environ["ACL_OP_COMPILER_CACHE_DIR"]})
        torch.npu.set_device(args.device)
        device = f"npu:{args.device}"
        import ctypes
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtGetSocName.restype = ctypes.c_char_p
        soc_name = acl.aclrtGetSocName()
        if not soc_name:
            raise RuntimeError("aclrtGetSocName returned null")
        build = json.loads(Path(resolved.build_manifest).read_text(encoding="utf-8"))
        preprocess = json.loads(Path(resolved.preprocess_build).read_text(encoding="utf-8"))
        report["runtime_target"] = verify_runtime_target(build, preprocess, soc_name.decode(), os.environ["ASCEND_HOME_PATH"])
        report["environment"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                                 "device_index": args.device, "device": torch.npu.get_device_name(args.device),
                                 "runtime_soc": soc_name.decode(),
                                 "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
                                 "compiler_cache": "disable"}
        report["adapters"] = {name: adapter.metadata for name, adapter in adapters.items()}
        with torch.no_grad():
            report["stage"] = "structured_validation"
            report["validation"] = {}
            for name, adapter in adapters.items():
                report["validation"][name] = {}
                validate_small(adapter, device, report["validation"][name], save, torch)
            report["stage"] = "long_shape_provider_and_output_checks"
            save()
            generator = torch.Generator().manual_seed(0)
            original = [torch.randn(args.shape, generator=generator).to(torch.bfloat16) for _ in range(3)]
            if not all(bool(torch.isfinite(value).all()) for value in original):
                raise ValueError("Finite original inputs required")
            report["input_sha256"] = [common.tensor_sha256(value) for value in original]
            inputs = [value.to(device) for value in original]
            payloads = {name: adapter.prepare(*inputs) for name, adapter in adapters.items()}
            report["payloads"] = {name: payload_evidence(payload) for name, payload in payloads.items()}
            report["independent_payload_hash_equal"] = {
                field: report["payloads"]["torch"][field]["sha256"] == report["payloads"]["fused"][field]["sha256"]
                for field in FIELDS}
            report["shared_payload"] = {"source": "fused", "same_object": True,
                                        "same_native_operator_object": True, "fields": list(FIELDS)}
            marker = directory / "markers.jsonl"
            offset = marker.stat().st_size if marker.exists() else 0
            calls = make_calls(adapters, payloads, inputs)
            raw = calls["shared_native"]["torch"]()
            torch.npu.synchronize()
            source = json.loads(Path(resolved.source_manifest).read_text(encoding="utf-8"))
            with marker.open("rb") as stream:
                stream.seek(offset)
                fresh = stream.read().decode("utf-8")
            report["host_markers"] = quant.parse_quant_markers(fresh, os.getpid(), source["marker_token"])
            (directory / "long_shape_markers.jsonl").write_text(fresh, encoding="utf-8")
            build = json.loads(Path(resolved.build_manifest).read_text(encoding="utf-8"))
            provider = {"cann_root": build.get("cann_root") or build.get("definitions", {}).get("CUSTOM_ASCEND_CANN_PACKAGE_PATH"),
                        "soc": build.get("soc")}
            quant.baseline.inspect_provider("local", resolved.vendor_root, resolved.bridge_path, report, directory, **provider)
            quant.inspect_fused_preprocessor(resolved, report)
            os.environ.pop(quant.QUANT_MARKER_ENV, None)
            marker_size = marker.stat().st_size
            candidate = calls["shared_native"]["fused"]()
            report["shared_all_output_check"] = compare_all_rows(candidate, raw)
            if not report["shared_all_output_check"]["exact_match"]:
                raise AssertionError("Identical native operation/payload produced different output; stop before timing")
            del candidate
            ids = torch.linspace(0, args.shape[2] - 1, min(32, args.shape[2])).long().unique()
            index = ids.to(device)
            samples = {"fused": adapters["fused"].postprocess(raw, payloads["fused"]).index_select(2, index).cpu()}
            del raw
            samples["torch"] = adapters["torch"].run_prepared(payloads["torch"]).index_select(2, index).cpu()
            dense = common.dense_reference(*original, ids)
            report["dense_fp32_samples"] = {name: common.error_metrics(sample, dense) for name, sample in samples.items()}
            report["accuracy_scope"] = {"queries": ids.tolist(), "all_keys": args.shape[2],
                "note": "Independent payload/dense comparisons are algorithm diagnostics, not model quality gates"}
            report["independent_payload_output_difference"] = common.error_metrics(samples["fused"], samples["torch"])
            del original, samples, dense
            save()
            for scope, group in calls.items():
                report["stage"] = "timing_" + scope
                def progress(snapshot, label=scope):
                    report["timings"][label] = snapshot
                    save()
                    print(f"  {label}: paired trial {snapshot['completed_trials']}/{args.trials} complete", flush=True)
                print(f"Paired timing {scope}: {SCOPES[scope]}", flush=True)
                timing = run_paired(group, synchronize=torch.npu.synchronize,
                    warmup=args.warmup, repeats=args.repeats, trials=args.trials,
                    seed=args.seed, on_trial=progress)
                report["timings"][scope] = timing
                report["comparisons"][scope] = paired_comparison(timing, "torch", "fused")
                save()
            if marker.stat().st_size != marker_size:
                raise AssertionError("Tiling marker writes continued during timing")
            report["timing_status"] = "passed"
            print(table(report), flush=True)
            save()  # Persist uninstrumented timings before the optional profiler.
            if args.trace:
                from mindie_laser_quant_trace import capture_native_traces
                report["stage"] = "optional_npu_profiler"
                save()
                report["trace"] = capture_native_traces(
                    {"native_torch_payload": calls["own_native"]["torch"],
                     "native_fused_payload": calls["own_native"]["fused"],
                     "fused_preprocessing": calls["preprocessing"]["fused"]},
                    torch=torch, torch_npu=torch_npu, directory=directory / "profiler", repeats=3)
                save()
                from summarize_mindie_laser_quant_profile import render_report
                print(render_report(report["trace"]), flush=True)
            else:
                report["trace"] = {"status": "disabled"}
            quant.baseline.inspect_provider("local", resolved.vendor_root, resolved.bridge_path, report, directory, **provider)
            quant.inspect_fused_preprocessor(resolved, report)
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


def parent(args):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args._run_dir = tempfile.mkdtemp(prefix="mindie_laser_quant_profile_", dir=output.parent)
    report = {"status": "running", "run_dir": args._run_dir, "mode": "paired_same_process"}
    common.write_json(output, report)
    try:
        resolved = quant.build_arguments(args, "qk_int8_fused")
        from mindie_laser_quant_fused import validate_preprocess_manifest
        report["provenance"] = quant.quant_provenance(resolved)
        report["preprocessing_provenance"] = validate_preprocess_manifest(resolved.preprocess_build)
        common.write_json(output, report)
        directory = Path(resolved._run_dir) / "local"
        directory.mkdir(parents=True, exist_ok=True)
        env = quant.worker_environment(args, "qk_int8_fused", args._run_dir)
        for key in ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker", "--_run-dir", args._run_dir]
        for key in ("quant_build", "preprocess_build", "device", "warmup", "repeats", "trials", "cpu_threads", "seed"):
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        command += ["--shape", *map(str, args.shape)]
        if args.trace:
            command.append("--trace")
        print(f"Reusing built libraries; one isolated paired worker: {directory}", flush=True)
        with (directory / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        path = directory / "report.json"
        if path.is_file():
            report.update(json.loads(path.read_text(encoding="utf-8")))
        report.update(command=command, returncode=code)
        if code or report.get("status") != "passed":
            raise RuntimeError(f"Paired worker failed; see {directory / 'worker.log'}")
        common.write_json(output, report)
        print(f"Report: {output}", flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", parent_error=f"{type(exc).__name__}: {exc}")
        common.write_json(output, report)
        traceback.print_exc()
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-build", default="results/mindie_laser_quant_build.json")
    parser.add_argument("--preprocess-build", default="results/mindie_laser_quant_preprocess_build.json")
    parser.add_argument("--output", default="results/mindie_laser_quant_profile.json")
    parser.add_argument("--shape", type=int, nargs=4, default=list(quant.baseline.SHAPE))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--trace", action="store_true", help="Capture NPU traces separately after wall timing")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_run-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.device < 0 or args.warmup < 0 or args.trials < 3 or min(args.repeats, args.cpu_threads, *args.shape) < 1
            or args.shape[-1] not in (64, 128) or args.shape[2] >= 2**31):
        parser.error("Need nonempty BNSD with D64/128, trials>=3, repeats/threads>=1, device/warmup>=0")
    if args._worker and not args._run_dir:
        parser.error("Internal worker requires --_run-dir")
    return worker(args) if args._worker else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
