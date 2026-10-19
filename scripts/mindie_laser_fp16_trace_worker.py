"""Independent original-FP16 Laser worker for scale-reuse diagnostics.

The parent supplies an isolated original-Laser vendor environment and verified
build manifests. This module never builds or installs operators. Importing it
does not import torch or initialize an NPU runtime.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import traceback

import benchmark_mindie_laser_quant as quant
from mindie_laser_quant_pairing import run_paired


baseline = quant.baseline
common = quant.common
ROW = "original_fp16"
SCOPES = {
    "full": "Fresh FP16 input conversion/padding, native allocation/dispatch, output cast/trim/contiguous",
    "prepared": "Fixed original-FP16 payload, native allocation/dispatch, output cast/trim/contiguous",
    "native": "Fixed original-FP16 payload; raw native statistics/output tuple, allocation/dispatch; no output postprocess",
}


def native_call(adapter, payload):
    """Exactly the native invocation inside MindIELaser.run_prepared.

    Keep both returned tensors alive only until the caller discards this tuple.
    No cast, slicing or contiguous conversion belongs in the native timer/trace.
    """
    q, k, v = payload.tensors
    return adapter.native(q, k, v, None, None, None, payload.scale,
                          payload.shape[1], "BNSD", 1.0,
                          payload.pre_tokens, 1, True)


def make_calls(adapter, payload, inputs):
    return {
        "full": {ROW: lambda: adapter(*inputs)},
        "prepared": {ROW: lambda: adapter.run_prepared(payload)},
        "native": {ROW: lambda: native_call(adapter, payload)},
    }


def verify_runtime_target(build, runtime_soc, cann_root):
    """Original Laser targets a family; the exact device must belong to it."""
    from check_ascend910b import resolve_soc
    selection = resolve_soc(runtime_soc, None, Path(cann_root))
    if selection["errors"]:
        raise ValueError("; ".join(selection["errors"]))
    family = "ascend910_93" if runtime_soc.lower().startswith("ascend910_93") else "ascend910b"
    if build.get("soc", "").lower() != family:
        raise ValueError("Original FP16 Laser build targets another runtime SoC family")
    root = build.get("cann_root") or build.get("definitions", {}).get("CUSTOM_ASCEND_CANN_PACKAGE_PATH")
    if not root or Path(root).resolve() != Path(cann_root).resolve():
        raise ValueError("Original FP16 Laser build targets another CANN Toolkit installation")
    return selection


def run(args):
    """Write <run_dir>/baseline/local/report.json; return a worker exit code.

    Required parent attributes match quant.build_arguments(args, 'baseline'),
    plus shape. The marker/cache environment must be configured before process
    launch using baseline.worker_environment(..., 'local', vendor, resolved._run_dir).
    """
    resolved = quant.build_arguments(args, "baseline")
    directory = Path(resolved._run_dir) / "local"
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "stage": "verify_provenance", "pid": os.getpid(),
        "mode": "original_fp16_separate_process", "shape_bnsd": list(args.shape),
        "dtype": "bfloat16", "seed": 0, "scopes": SCOPES, "timings": {},
        "comparison_scope": (
            "Original FP16 Laser uses the same original seeded inputs, but its own FP16 payload. "
            "Its timings run in a separate process from INT8 candidates and are not paired speedups. "
            "Native device counters include the whole kernel and are not additive phase percentages."),
        "contract": "Pinned original FP16 QK/PV, original exp, dense high-precision mode; no quantization/smoothing/reordering",
    }
    save = lambda: common.write_json(directory / "report.json", report)
    save()
    try:
        # Recheck existing artifacts in the worker; registration precedes first
        # NPU allocation so provider resolution uses the isolated environment.
        report["provenance"] = baseline.provenance(resolved)
        source = json.loads(Path(resolved.source_manifest).read_text(encoding="utf-8"))
        build = json.loads(Path(resolved.build_manifest).read_text(encoding="utf-8"))
        report["stage"] = "load_runtime"
        save()
        import torch
        import torch_npu
        torch.set_num_threads(min(args.cpu_threads, torch.get_num_threads()))
        adapter, transforms = baseline.load_adapter(resolved, source)
        report["adapter"], report["source_transform_validation"] = adapter.metadata, transforms
        torch.npu.set_option({"ACL_OP_COMPILER_CACHE_MODE": "disable",
                              "ACL_OP_COMPILER_CACHE_DIR": os.environ["ACL_OP_COMPILER_CACHE_DIR"]})
        torch.npu.set_device(args.device)
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtGetSocName.restype = ctypes.c_char_p
        soc_name = acl.aclrtGetSocName()
        if not soc_name:
            raise RuntimeError("aclrtGetSocName returned null")
        runtime_soc = soc_name.decode()
        cann_root = os.environ["ASCEND_HOME_PATH"]
        report["runtime_target"] = verify_runtime_target(build, runtime_soc, cann_root)
        provider = {"cann_root": cann_root, "soc": build["soc"]}
        report["environment"] = {
            "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "device_index": args.device, "device": torch.npu.get_device_name(args.device),
            "runtime_soc": runtime_soc, "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
            "compiler_cache": "disable",
        }
        device = f"npu:{args.device}"
        with torch.no_grad():
            report["stage"] = "structured_validation"
            save()
            baseline.validate_small(adapter, device, report, save, torch)
            report["stage"] = "long_shape_provider_and_output_checks"
            save()
            generator = torch.Generator().manual_seed(0)
            original = [torch.randn(args.shape, generator=generator).to(torch.bfloat16) for _ in range(3)]
            report["input_sha256"] = [common.tensor_sha256(value) for value in original]
            inputs = [value.to(device) for value in original]
            payload = adapter.prepare(*inputs)
            report["prepared_payload"] = adapter.describe_payload(payload)
            calls = make_calls(adapter, payload, inputs)
            marker = directory / "markers.jsonl"
            if os.environ.get(baseline.MARKER_ENV) != str(marker):
                raise ValueError("FP16 worker requires its isolated original-Laser marker path")
            torch.npu.synchronize()
            offset = marker.stat().st_size if marker.exists() else 0
            raw = calls["native"][ROW]()
            torch.npu.synchronize()
            if not isinstance(raw, (tuple, list)) or len(raw) != 2:
                raise RuntimeError("Original FP16 native API must return (statistics, output)")
            with marker.open("rb") as stream:
                stream.seek(offset)
                fresh = stream.read().decode("utf-8")
            report["host_markers"] = baseline.parse_markers(fresh, os.getpid())
            (directory / "long_shape_markers.jsonl").write_text(fresh, encoding="utf-8")
            baseline.inspect_provider("local", resolved.vendor_root, resolved.bridge_path,
                                      report, directory, **provider)
            report["provider_evidence_passed"] = True
            os.environ.pop(baseline.MARKER_ENV, None)
            marker_size = marker.stat().st_size
            report["marker_disabled_before_timing"] = True

            ids = torch.linspace(0, args.shape[2] - 1, min(32, args.shape[2])).long().unique()
            index = ids.to(device)
            output = adapter.postprocess(raw[1], payload.dtype, payload.shape[2], payload.shape[3])
            sample = output.index_select(2, index).cpu()
            del output, raw
            prepared_sample = adapter.run_prepared(payload).index_select(2, index).cpu()
            full_sample = adapter(*inputs).index_select(2, index).cpu()
            baseline.record_gate(report, "raw_postprocessed_vs_prepared_sample", sample,
                                 prepared_sample, .001, save)
            baseline.record_gate(report, "full_vs_prepared_sample", full_sample,
                                 prepared_sample, .001, save)
            dense = common.dense_reference(*original, ids)
            baseline.record_gate(report, "dense_fp32_sample", sample, dense, .01, save)
            report["accuracy_scope"] = {"queries": ids.tolist(), "all_keys": args.shape[2],
                "all_heads_and_features": True, "note": "Sampled implementation check, not all-output equivalence or model quality"}
            del original, sample, prepared_sample, full_sample, dense
            save()
            for scope, group in calls.items():
                report["stage"] = "timing_" + scope
                def progress(snapshot, label=scope):
                    report["timings"][label] = snapshot
                    save()
                    print(f"  original FP16 {label}: trial {snapshot['completed_trials']}/{args.trials} complete", flush=True)
                report["timings"][scope] = run_paired(
                    group, synchronize=torch.npu.synchronize, warmup=args.warmup,
                    repeats=args.repeats, trials=args.trials,
                    seed=getattr(args, "seed", 2026), on_trial=progress)
                save()
            if marker.stat().st_size != marker_size:
                raise AssertionError("Original FP16 tiling marker writes continued during timing")
            report["timing_status"] = "passed"
            print("Original FP16 Laser; separate-process synchronized API wall ms:", flush=True)
            for scope, timing in report["timings"].items():
                print(f"  {scope:<12} {timing['rows'][ROW]['median_ms']:.4f} ms", flush=True)
            print("FP16/INT8 runs use different payloads and processes; these are not paired kernel speedups.", flush=True)
            save()  # All uninstrumented timings survive a profiler failure.
            if getattr(args, "trace", True):
                report["stage"] = "optional_npu_profiler"
                save()
                from mindie_laser_quant_trace import capture_native_traces
                report["trace"] = capture_native_traces(
                    {"original_fp16_native": calls["native"][ROW]}, torch=torch, torch_npu=torch_npu,
                    directory=directory / "profiler", repeats=3)
                save()
                from summarize_mindie_laser_quant_profile import render_report
                print(render_report(report["trace"]), flush=True)
            else:
                report["trace"] = {"status": "disabled"}
            baseline.inspect_provider("local", resolved.vendor_root, resolved.bridge_path,
                                      report, directory, **provider)
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
