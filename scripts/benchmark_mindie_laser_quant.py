#!/usr/bin/env python3
"""Original Laser vs INT8 QK, with optional decoded-E4M3 V and ExpCast experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

import benchmark_mindie_laser_local as baseline

common = baseline.common
VARIANTS = ("installed", "baseline", "qk_int8", "qk_int8_fused",
            "qkv_int8_e4m3", "qkv_int8_e4m3_expcast")
DEFAULT_VARIANTS = VARIANTS[:4]
QUANT_MARKER_ENV = "VC_MINDIE_LASER_QUANT_MARKER_FILE"
QUANT_OP = "VcLaserQuantAttention"
QUANT_API = "aclnnVcLaserQuantAttention"
QUANT_SCOPE = (
    "Pinned MindIE Laser with native INT8 QK/INT32 accumulation and FP16 PV. "
    "V E4M3 is a numerical quantization experiment decoded to FP16 before PV; "
    "there is no native FP8 matmul or FP8-storage performance claim. No V reordering or K smoothing."
)


def build_arguments(args, name):
    quantized = name not in ("installed", "baseline")
    path = Path(args.quant_build if quantized else args.baseline_build).resolve()
    build = json.loads(path.read_text(encoding="utf-8"))
    if quantized:
        bridge, bridge_manifest = build["bridge_path"], None
    else:
        bridge_manifest = str(Path(args.bridge_manifest).resolve())
        bridge = json.loads(Path(bridge_manifest).read_text(encoding="utf-8"))["bridge_path"]
    source = build["source_manifest"]
    return SimpleNamespace(
        vendor_root=build["vendor_root"], build_manifest=str(path),
        source_manifest=source["path"] if isinstance(source, dict) else source,
        bridge_path=bridge, bridge_manifest=bridge_manifest,
        preprocess_build=str(Path(args.preprocess_build).resolve()) if name == "qk_int8_fused" else None,
        device=args.device, warmup=args.warmup, repeats=args.repeats,
        trials=args.trials, cpu_threads=args.cpu_threads,
        _worker="installed" if name == "installed" else "local",
        _run_dir=str(Path(args._run_dir) / name),
    )


def quant_provenance(args, *, identity=None, source_validator=None):
    from prepare_mindie_laser_quant_source import validate_prepared_source
    identity = identity or {"native_api": QUANT_API, "op_type": QUANT_OP}
    source_validator = source_validator or validate_prepared_source
    build_path, source_path = Path(args.build_manifest), Path(args.source_manifest)
    build = json.loads(build_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if build.get("status") != "passed":
        raise ValueError("Quantized Laser build manifest must report passed")
    for item in (source, build):
        if any(item.get(key) != identity[key] for key in ("native_api", "op_type")):
            raise ValueError("Quantized source/build must use the requested unique operator symbols")
    if (source.get("upstream", {}).get("commit") != baseline.UPSTREAM_COMMIT
            or source.get("upstream", {}).get("url") != baseline.SOURCE_LOCK.url):
        raise ValueError("Quantized Laser must derive from the pinned MindIE-SD source")
    if not source.get("marker_token"):
        raise ValueError("Quantized source manifest requires its marker token")
    if Path(build["vendor_root"]).resolve() != Path(args.vendor_root).resolve():
        raise ValueError("Quantized build vendor_root differs from requested provider")
    source_validator(Path(source["source_dir"]), source_path)
    digest = common.sha256_file(source_path)
    if build.get("source_manifest", {}).get("sha256") != digest:
        raise ValueError("Quantized source manifest SHA256 mismatch")
    bridge_digest = common.sha256_file(args.bridge_path)
    if build.get("bridge_sha256") != bridge_digest:
        raise ValueError("Quantized bridge SHA256 mismatch")
    return {"source_manifest": source, "build_manifest": build,
            "source_manifest_sha256": digest, "build_manifest_sha256": common.sha256_file(build_path),
            "bridge_sha256": bridge_digest,
            "device_binary_sha256": baseline.checked_files(build.get("device_binaries"), args.vendor_root, "device_binaries"),
            "host_library_sha256": baseline.checked_files(build.get("host_libraries"), args.vendor_root, "host_libraries"),
            "proof_scope": baseline.PROOF_SCOPE}


def inspect_fused_preprocessor(args, report):
    """Require the independently built quantizer to be mapped in this worker."""
    from mindie_laser_quant_fused import validate_preprocess_manifest
    evidence = validate_preprocess_manifest(args.preprocess_build)
    build = json.loads(Path(args.preprocess_build).read_text(encoding="utf-8"))
    library = str(Path(build["library_path"]).resolve())
    if library not in report.get("mapped_library_paths", []):
        raise RuntimeError("Fused quantizer library is absent from process library mappings")
    report["preprocessor_provider_evidence"] = {"manifest": evidence, "mapped_library": library}
    report["loaded_library_sha256"][library] = common.sha256_file(library)


def worker_environment(args, name, run_dir):
    resolved = build_arguments(args, name)
    env = baseline.worker_environment(os.environ, resolved._worker, resolved.vendor_root, resolved._run_dir)
    env.pop(QUANT_MARKER_ENV, None)
    if name not in ("installed", "baseline"):
        env.pop(baseline.MARKER_ENV, None)
        env[QUANT_MARKER_ENV] = str(Path(resolved._run_dir) / "local" / "markers.jsonl")
    return env


def parse_quant_markers(text, pid, token):
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not records:
        raise RuntimeError("Quantized first long-shape call produced no fresh tiling marker")
    for record in records:
        if (record.get("token") != token or record.get("phase") != "tiling"
                or type(record.get("pid")) is not int or record["pid"] != pid
                or type(record.get("tiling_key")) is not int or record["tiling_key"] < 0):
            raise ValueError("Malformed or foreign-process quantized Laser marker")
    return records


def benchmark_calls(calls, args, torch, report, save, *, timing_key="timing_ms"):
    timings = report.setdefault(timing_key, {})
    timings.update({name: {"trials": []} for name in calls})
    for call in calls.values():
        for _ in range(args.warmup):
            call()
    torch.npu.synchronize()
    for trial in range(args.trials):
        for name, call in calls.items():
            torch.npu.synchronize()
            started = time.perf_counter()
            for _ in range(args.repeats):
                call()
            torch.npu.synchronize()
            elapsed = 1000 * (time.perf_counter() - started) / args.repeats
            row = timings[name]
            row["trials"].append(elapsed)
            row.update(median=statistics.median(row["trials"]), min=min(row["trials"]), max=max(row["trials"]))
            save()
            print(f"{report['mode']} {name} trial {trial + 1}: {elapsed:.4f} ms", flush=True)


def benchmark_quant_components(adapter, inputs, payload, args, torch, report, save):
    """Time independent calls after primary timings, without retaining another large output."""
    report["component_timing_scope"] = {
        "qk_quantize": ("fresh Q/K quantization, padding and scale/output allocation; fused input ND "
                        "conversion included; final payload format normalization excluded"),
        "native_api": "fixed quantized payload, native dispatch/allocation, padded FP32 output; no output postprocess",
        "output_postprocess": "one retained padded FP32 output, V scale/cast/trim/contiguous only",
        "limits": ("Separate synchronized wall-time calls after primary timings; not kernel durations, "
                   "not additive. The retained raw output exists only during output_postprocess timing.")}
    save()
    benchmark_calls({"qk_quantize": lambda: adapter.quantize_qk(inputs[0], inputs[1]),
                     "native_api": lambda: adapter.run_native(payload)},
                    args, torch, report, save, timing_key="component_timing_ms")
    raw = adapter.run_native(payload)
    torch.npu.synchronize()
    try:
        benchmark_calls({"output_postprocess": lambda: adapter.postprocess(raw, payload)},
                        args, torch, report, save, timing_key="component_timing_ms")
    finally:
        del raw


def worker(args):
    name = args._worker
    resolved = build_arguments(args, name)
    directory = Path(resolved._run_dir) / resolved._worker
    directory.mkdir(parents=True, exist_ok=True)
    quantized = name not in ("installed", "baseline")
    report = {"status": "running", "mode": name, "pid": os.getpid(),
              "implementation_contract": QUANT_SCOPE if quantized else "Original FP16 Laser QK/PV",
              "shape_bnsd": list(args.shape), "dtype": "bfloat16", "seed": 0}
    save = lambda: common.write_json(directory / "report.json", report)
    save()
    try:
        import torch
        import torch_npu
        torch.set_num_threads(min(args.cpu_threads, torch.get_num_threads()))
        source = json.loads(Path(resolved.source_manifest).read_text(encoding="utf-8"))
        build = json.loads(Path(resolved.build_manifest).read_text(encoding="utf-8"))
        if quantized:
            from mindie_laser_quant_backend import load_quant_laser
            from mindie_laser_quant_reference import cpu_payload, decoded_reference, validate_small
            adapter = load_quant_laser(
                resolved.bridge_path, v_quantize=name not in ("qk_int8", "qk_int8_fused"),
                expcast=name.endswith("_expcast"), check_finite=False,
                preprocessing="fused" if name == "qk_int8_fused" else "torch",
                preprocess_build=resolved.preprocess_build)
            report["source_transform_validation"] = (
                "CPU structured checks: scales within 1 FP32 ULP; INT8 RNE boundary-aware "
                "division check using actual scales; exact zero/padding and decoded V; "
                "exact-scale RNE probes. Native oracle uses actual device payloads.")
        else:
            adapter, transforms = baseline.load_adapter(resolved, source)
            validate_small = baseline.validate_small
            report["source_transform_validation"] = transforms
        report["adapter"] = adapter.metadata
        torch.npu.set_option({"ACL_OP_COMPILER_CACHE_MODE": "disable",
                              "ACL_OP_COMPILER_CACHE_DIR": os.environ["ACL_OP_COMPILER_CACHE_DIR"]})
        torch.npu.set_device(args.device)
        device = f"npu:{args.device}"
        report["environment"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                                 "device": torch.npu.get_device_name(args.device),
                                 "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
                                 "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
                                 "compiler_cache": "disable"}
        definitions = build.get("definitions", {})
        provider = {"cann_root": build.get("cann_root") or definitions.get("CUSTOM_ASCEND_CANN_PACKAGE_PATH"),
                    "soc": build.get("soc")}
        save()
        with torch.no_grad():
            validate_small(adapter, device, report, save, torch)
            generator = torch.Generator().manual_seed(0)
            original = [torch.randn(args.shape, generator=generator).to(torch.bfloat16) for _ in range(3)]
            if not all(bool(torch.isfinite(x).all()) for x in original):
                raise ValueError("Benchmark source tensors must be finite")
            report["finite_validation"] = "Original CPU tensors checked once, outside timing; repeated scans excluded"
            fingerprints = [common.tensor_sha256(x) for x in original]
            inputs = [x.to(device) for x in original]
            ids = torch.linspace(0, args.shape[2] - 1, min(32, args.shape[2])).long().unique()
            marker = directory / "markers.jsonl"
            torch.npu.synchronize()
            offset = marker.stat().st_size if marker.exists() else 0
            print(f"{name}: first requested-shape full call {args.shape}", flush=True)
            output = adapter(*inputs)
            torch.npu.synchronize()
            if name != "installed":
                with marker.open("rb") as stream:
                    stream.seek(offset)
                    fresh = stream.read().decode("utf-8")
                (directory / "long_shape_markers.jsonl").write_text(fresh, encoding="utf-8")
                report["host_markers"] = (parse_quant_markers(fresh, os.getpid(), source["marker_token"])
                                          if quantized else baseline.parse_markers(fresh, os.getpid()))
                report["tiling_keys"] = sorted({item["tiling_key"] for item in report["host_markers"]})
            baseline.inspect_provider(resolved._worker, resolved.vendor_root, resolved.bridge_path,
                                      report, directory, **provider)
            if name == "qk_int8_fused":
                inspect_fused_preprocessor(resolved, report)
            report["provider_evidence_passed"] = True
            os.environ.pop(baseline.MARKER_ENV, None)
            os.environ.pop(QUANT_MARKER_ENV, None)
            marker_size = marker.stat().st_size if marker.exists() else 0
            report["marker_disabled_before_timing"] = True
            sample = output.index_select(2, ids.to(device)).cpu()
            del output
            payload = adapter.prepare(*inputs)
            core_sample = adapter.run_prepared(payload).index_select(2, ids.to(device)).cpu()
            report["prepared_payload"] = adapter.describe_payload(payload)
            baseline.record_gate(report, "long_full_vs_prepared_sample", sample, core_sample, .001, save)
            for scope, values in (("full", sample), ("prepared_core", core_sample)):
                path = directory / (scope + "_sample.pt")
                torch.save({"query_ids": ids, "output": values, "shape_bnsd": list(args.shape),
                            "seed": 0, "input_sha256": fingerprints}, path)
                report[scope + "_sample"] = str(path)
            print(f"{name}: FP32 CPU references, {len(ids)} query rows and all keys", flush=True)
            dense = common.dense_reference(*original, ids)
            report["long_dense_fp32"] = common.error_metrics(sample, dense)
            report["long_dense_fp32"]["scope"] = "Original input FP32 softmax; synthetic algorithm diagnostic"
            if not quantized and report["long_dense_fp32"]["relative_rmse"] > .01:
                raise AssertionError("Original Laser dense error exceeds 1%")
            if quantized:
                host_payload = cpu_payload(payload)
                report["prepared_input_sha256"] = {
                    field: common.tensor_sha256(getattr(host_payload, field))
                    for field in ("q", "k", "v", "q_scale", "k_scale", "v_scale")}
                report["prepared_input_hash_scope"] = "Contiguous CPU bytes of actual native payload, outside timing"
                decoded = decoded_reference(host_payload, ids, expcast=adapter.expcast)
                report["long_decoded_reference"] = common.error_metrics(sample, decoded)
                report["long_decoded_reference"]["scope"] = (
                    "Decoded INT8/E4M3 operands, truncated FIX gain, FP16 score/scale, global maximum, "
                    "FP32 sum/PV; diagnostic, not a model of native online/reduction rounding")
                del decoded, host_payload
            report["query_ids"], report["all_keys"] = ids.tolist(), args.shape[2]
            save()
            del original, dense, sample, core_sample
            benchmark_calls({"preprocessing": lambda: adapter.prepare(*inputs),
                             "full": lambda: adapter(*inputs),
                             "prepared_core": lambda: adapter.run_prepared(payload)}, args, torch, report, save)
            if quantized:
                benchmark_quant_components(adapter, inputs, payload, args, torch, report, save)
            if (marker.stat().st_size if marker.exists() else 0) != marker_size:
                raise AssertionError("Marker writes continued during timing")
            baseline.inspect_provider(resolved._worker, resolved.vendor_root, resolved.bridge_path,
                                      report, directory, **provider)
            if name == "qk_int8_fused":
                inspect_fused_preprocessor(resolved, report)
        report["status"] = "passed"
        save()
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save()
        traceback.print_exc()
        return 1


def compare_algorithm_samples(first, second):
    result = common.compare_samples(first, second)
    result.pop("passed", None)
    result.pop("limit", None)
    result["scope"] = "Different quantization/probability algorithms; diagnostic, no same-algorithm threshold"
    return result


def verify_quant_payload_identity(workers):
    """Require identical decoded inputs for each intended quantization ablation."""
    pairs = (("qk_int8", "qkv_int8_e4m3", ("q", "k", "q_scale", "k_scale")),
             ("qkv_int8_e4m3", "qkv_int8_e4m3_expcast", ("q", "k", "v", "q_scale", "k_scale", "v_scale")))
    checks = []
    for first, second, fields in pairs:
        if first not in workers or second not in workers:
            continue
        for field in fields:
            left = workers[first].get("prepared_input_sha256", {}).get(field)
            right = workers[second].get("prepared_input_sha256", {}).get(field)
            if (not isinstance(left, str) or len(left) != 64
                    or any(char not in "0123456789abcdef" for char in left) or left != right):
                raise AssertionError(f"Quantized ablation payload differs or lacks a valid SHA256: {first}/{second} {field}")
        checks.append({"first": first, "second": second, "fields": list(fields), "exact_sha256_match": True})
    return checks


def table(report):
    reference_name = "baseline" if "baseline" in report["workers"] else report["process_order"][0]
    lines = ["", "Quantized MindIE Laser; synchronized wall ms/API",
             f"variant                      preprocessing      full API    prepared     {reference_name}/full    dense RMSE %"]
    reference = report["workers"][reference_name]["timing_ms"]["full"]["median"]
    for name in report["process_order"]:
        row = report["workers"][name]
        timing = row["timing_ms"]
        full = timing["full"]["median"]
        lines.append(f"{name:<28} {timing['preprocessing']['median']:>13.4f} {full:>13.4f}"
                     f" {timing['prepared_core']['median']:>11.4f} {reference / full:>16.4f}x"
                     f" {100 * row['long_dense_fp32']['relative_rmse']:>15.6f}")
    component_rows = [name for name in report["process_order"]
                      if report["workers"][name].get("component_timing_ms")]
    if component_rows:
        lines += ["", "Independent component API wall times (ms; measured after primary timings):",
                  "variant                      QK quantize     native API    output postprocess"]
        for name in component_rows:
            timing = report["workers"][name]["component_timing_ms"]
            lines.append(f"{name:<28} {timing['qk_quantize']['median']:>11.4f}"
                         f" {timing['native_api']['median']:>14.4f}"
                         f" {timing['output_postprocess']['median']:>21.4f}")
        lines.append("Component calls are not additive or kernel-only times; native API includes allocation/dispatch.")
    lines += [QUANT_SCOPE,
              "Prepared includes output V scale where needed and conversion/trim; full freshly quantizes every call.",
              "Preprocessing measured separately; full minus prepared is not used as its estimate.",
              "Process-grouped synchronized wall timings, not paired/interleaved or kernel-only measurements.",
              "Accuracy samples cover at most 32 query rows, all keys, heads and features; no model-quality claim."]
    return "\n".join(lines)


def parent(args):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args._run_dir = str(Path(tempfile.mkdtemp(prefix=output.stem + "_", dir=output.parent)).resolve())
    report = {"status": "running", "shape_bnsd": list(args.shape), "dtype": "bfloat16", "seed": 0,
              "process_order": args.variants, "artifact_dir": args._run_dir,
              "workers": {}, "provenance": {}, "proof_scope": baseline.PROOF_SCOPE,
              "implementation_contract": QUANT_SCOPE,
              "timing_scope": {"preprocessing": "fresh conversion, padding, quantization and decoded V (where applicable)",
                               "full": "fresh preprocessing + native allocation/dispatch + V output scale/cast/trim/contiguous",
                               "prepared_core": "fixed native input payload + native allocation/dispatch + V output scale/cast/trim/contiguous"}}
    save = lambda: common.write_json(output, report)
    save()
    try:
        for name in args.variants:
            resolved = build_arguments(args, name)
            report["provenance"][name] = (baseline.provenance(resolved) if name in ("installed", "baseline")
                                          else quant_provenance(resolved))
            if name == "qk_int8_fused":
                from mindie_laser_quant_fused import validate_preprocess_manifest
                report["provenance"][name]["preprocessing"] = validate_preprocess_manifest(resolved.preprocess_build)
        save()
        for name in args.variants:
            resolved = build_arguments(args, name)
            directory = Path(resolved._run_dir) / resolved._worker
            directory.mkdir(parents=True, exist_ok=True)
            env = worker_environment(args, name, args._run_dir)
            for key in ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME"):
                Path(env[key]).mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--_worker", name,
                       "--_run-dir", args._run_dir]
            for key in ("baseline_build", "quant_build", "preprocess_build", "bridge_manifest", "device", "warmup", "repeats", "trials", "cpu_threads"):
                command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
            command += ["--shape", *map(str, args.shape)]
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
            report["workers"][name] = dict(data, command=command, returncode=code)
            save()
            if code or data.get("status") != "passed":
                raise RuntimeError(f"{name} failed; see {directory / 'worker.log'}")
        if len({row["pid"] for row in report["workers"].values()}) != len(args.variants):
            raise AssertionError("Provider workers must have distinct process identities")
        report["quant_payload_identity"] = verify_quant_payload_identity(report["workers"])
        save()
        for scope in ("full", "prepared_core"):
            sample = lambda name: report["workers"][name][scope + "_sample"]
            if "baseline" in args.variants and "installed" in args.variants:
                metric = common.compare_samples(sample("installed"), sample("baseline"))
                report.setdefault("baseline_vs_installed", {})[scope] = metric
                save()
                if not metric["passed"]:
                    raise AssertionError("Original source baseline differs from installed Laser by more than 0.1%")
            reference = "baseline" if "baseline" in args.variants else args.variants[0]
            for name in args.variants:
                if name in ("installed", "baseline", reference):
                    continue
                report.setdefault("algorithm_vs_" + reference, {}).setdefault(name, {})[scope] = compare_algorithm_samples(sample(reference), sample(name))
        if "qk_int8" in args.variants and "qk_int8_fused" in args.variants:
            comparison = {"scope": ("Same original input, independently prepared quantized operands; "
                                    "FP32 division may change INT8 codes near RNE boundaries. "
                                    "Output comparisons use sampled query rows and all keys; "
                                    "not an all-output identity or model-quality gate.")}
            for scope in ("full", "prepared_core"):
                first = report["workers"]["qk_int8"]
                second = report["workers"]["qk_int8_fused"]
                error = compare_algorithm_samples(first[scope + "_sample"], second[scope + "_sample"])
                error["scope"] = comparison["scope"]
                comparison[scope] = error
                comparison[scope + "_old_over_fused"] = (first["timing_ms"][scope]["median"] /
                                                          second["timing_ms"][scope]["median"])
            report["fused_preprocessing_comparison"] = comparison
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
    parser.add_argument("--quant-build", default="results/mindie_laser_quant_build.json")
    parser.add_argument("--preprocess-build", default="results/mindie_laser_quant_preprocess_build.json")
    parser.add_argument("--bridge-manifest", default="results/mindie_laser_bridge.json")
    parser.add_argument("--output", default="results/mindie_laser_quant_compare.json")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(DEFAULT_VARIANTS),
                        help="Default: installed baseline qk_int8 qk_int8_fused; select E4M3 V/ExpCast experiments explicitly")
    parser.add_argument("--shape", type=int, nargs=4, default=baseline.SHAPE, metavar=("B", "H", "S", "D"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--_worker", choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--_run-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.device < 0 or args.warmup < 0 or args.trials < 3 or min(args.repeats, args.cpu_threads) < 1:
        parser.error("Nonnegative device/warmup, trials >=3 and positive repeats/threads required")
    if min(args.shape) < 1 or args.shape[-1] not in (64, 128):
        parser.error("--shape requires positive B/H/S and D64/128")
    if len(set(args.variants)) != len(args.variants):
        parser.error("--variants must not repeat a provider")
    if args._worker and not args._run_dir:
        parser.error("Internal worker requires --_run-dir")
    return worker(args) if args._worker else parent(args)


if __name__ == "__main__":
    sys.exit(main())
