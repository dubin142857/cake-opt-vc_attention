#!/usr/bin/env python3
"""Original Laser vs INT8 QK, with optional decoded-E4M3 V and ExpCast experiments."""
from __future__ import annotations

import argparse
import json
import math
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


def experiment_contract(variants):
    if all(name in ("installed", "baseline") for name in variants):
        return "Original FP16 Laser QK/PV; no quantization or ExpCast"
    return QUANT_SCOPE


def physical_device(logical_device):
    """ASCEND_RT_VISIBLE_DEVICES remaps logical ordinals, including logical 0."""
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible is None:
        return logical_device
    tokens = visible.split(",")
    if not all(token.strip().isdigit() for token in tokens):
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES must list nonnegative physical device IDs")
    devices = [int(token) for token in tokens]
    if logical_device < 0 or logical_device >= len(devices):
        raise ValueError("--device is outside ASCEND_RT_VISIBLE_DEVICES logical ordinals")
    return devices[logical_device]


def synthetic_inputs(torch, shape, std=1.0, seed=0):
    """Same seed/FP32 distribution for every provider; round only after scaling."""
    if not math.isfinite(std) or std < 0:
        raise ValueError("std must be finite and nonnegative")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0,2**63)")
    generator = torch.Generator().manual_seed(seed)
    return [(torch.randn(shape, generator=generator) * std).to(torch.bfloat16)
            for _ in range(3)]


def worker_command(args, name):
    command = [sys.executable, str(Path(__file__).resolve()), "--_worker", name,
               "--_run-dir", args._run_dir]
    for key in ("baseline_build", "quant_build", "preprocess_build", "bridge_manifest", "device", "warmup", "repeats", "trials", "cpu_threads"):
        command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    if getattr(args, "merge_build", None):
        command += ["--merge-build", str(Path(args.merge_build).resolve())]
    if getattr(args,"pack_build",None):
        command += ["--pack-build",str(Path(args.pack_build).resolve())]
    if getattr(args, "trace_prepared", False):
        command += ["--trace-prepared"]
    if getattr(args,"replay_inputs",()):
        command += ["--replay-inputs",*(str(Path(p).resolve()) for p in args.replay_inputs)]
    return command + ["--shape", *map(str, args.shape), "--std", str(args.std),
                      "--seed", str(getattr(args,"seed",0)),
                      "--baseline-source-variant", getattr(args, "baseline_source_variant", "baseline"),
                      "--split-kv", str(getattr(args, "split_kv", 1))]


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


def inspect_fused_preprocessor(args, report, *, operation="quantize"):
    """Require each audited direct-launch library in actual worker mappings."""
    from mindie_laser_quant_fused import validate_preprocess_manifest
    manifest = args.preprocess_build if operation == "quantize" else getattr(args,operation+'_build')
    evidence = (validate_preprocess_manifest(manifest) if operation == "quantize" else
                validate_preprocess_manifest(manifest,operation=operation))
    build = json.loads(Path(manifest).read_text(encoding="utf-8"))
    library = str(Path(build["library_path"]).resolve())
    if library not in report.get("mapped_library_paths", []):
        raise RuntimeError(f"Fused {operation} library is absent from process library mappings")
    field = 'preprocessor_provider_evidence' if operation == 'quantize' else operation+'_provider_evidence'
    report[field] = {"manifest": evidence, "mapped_library": library}
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




def validate_fused_merge_direct(merger, device, torch, *, raw_stats=False):
    """Adversarial row/half-row/tail indexing checks, independent FP64 CPU merge."""
    checks = []
    generator = torch.Generator().manual_seed(65001)
    for queries, dimension in ((17,64), (65,64), (129,128), (256,128)):
        for dtype in (torch.float16, torch.bfloat16):
            for extra in (0,1):
                lse = torch.randn((4,3,256), generator=generator) * 10
                if extra:
                    lse = lse.abs() + 2
                else:
                    lse[0::2, :, 0::2] = 10000
                    lse[1::2, :, 1::2] = -10000
                raw = torch.randn((4,3,256,128), generator=generator)
                p = SimpleNamespace(shape=(2,3,queries,dimension), dtype=dtype)
                payload = SimpleNamespace(original=p,splits=2,extra_zero_keys=extra,raw_stats=raw_stats)
                sums = torch.rand((4,3,256), generator=generator) * 15 + 1 if raw_stats else None
                statistics = torch.cat((lse, sums), dim=-1) if raw_stats else lse
                actual = merger((statistics.to(device), raw.to(device)), payload).cpu()
                logs = lse.double().reshape(2,2,3,256)
                values = raw.double().reshape(2,2,3,256,128)
                shift = logs.max(0).values
                weights = (logs-shift).exp()
                denominator = ((weights * sums.double().reshape(2,2,3,256)).sum(0)
                               if raw_stats else weights.sum(0))
                if extra:
                    denominator = denominator - extra * (-shift).exp()
                expected = ((values*weights.unsqueeze(-1)).sum(0)/denominator.unsqueeze(-1))
                expected = expected[:,:,:queries,:dimension].half().to(dtype)
                metric = common.error_metrics(actual, expected)
                if metric["relative_rmse"] > .001 or not bool(torch.isfinite(actual).all()):
                    raise AssertionError(f"Fused merge direct probe failed: {queries}/{dimension}/{dtype}/{extra}: {metric}")
                checks.append(dict(queries=queries, dimension=dimension, dtype=str(dtype), extra=extra, raw_stats=raw_stats, **metric))
    return checks


def check_partition_lse(adapter, payload, query_ids, torch):
    """Independent CPU reference from actual FP16 operands, outside all timings."""
    if hasattr(payload, "original"):
        lse, _ = adapter.run_native(payload)
        scale, padding = payload.original.scale, payload.pre_tokens
    else:
        lse, _ = adapter.native(*payload.tensors, None, None, None, payload.scale,
                               payload.shape[1], "BNSD", 1.0, payload.pre_tokens, 1, True)
        scale, padding = payload.scale, payload.pre_tokens
    q, k, _ = payload.tensors
    valid = k.shape[2] - (0 if padding == 2147483647 else padding)
    host_q = q.index_select(2, query_ids.to(q.device)).float().cpu()
    host_k = k[:, :, :valid].float().cpu()
    # Native QK accumulation is FP32 -> FP16, then separate FP16 scale.
    scores = ((host_q @ host_k.transpose(-1, -2)).half() * torch.tensor(scale).half()).float()
    if getattr(payload,"masked_tail_keys",0):
        scores[scores.shape[0]//2:,...,-payload.masked_tail_keys:] = -float("inf")
    expected = scores.logsumexp(-1)
    if getattr(payload, "raw_stats", False):
        host_stats = lse.cpu()
        maxima, sums = host_stats[..., :q.shape[2]], host_stats[..., q.shape[2]:]
        if not bool(torch.isfinite(host_stats).all()) or not bool((sums > 0).all()):
            raise AssertionError("Raw max/rowsum is nonfinite, nonpositive, or unwritten")
        actual = (maxima + sums.log()).index_select(2, query_ids)
    else:
        actual = lse.index_select(2, query_ids.to(q.device)).cpu()
    if not bool(torch.isfinite(actual).all()):
        raise AssertionError("Partition LSE contains nonfinite or unwritten data")
    error = (actual - expected).abs()
    metrics = dict(max_absolute=float(error.max()), mean_absolute=float(error.mean()),
                   sampled_query_rows=query_ids.numel(), partitions_batch=q.shape[0],
                   reference="FP16 QK output then FP16 scale; FP32 logsumexp, all valid partition keys")
    if metrics["max_absolute"] > .2:
        raise AssertionError(f"Partition LSE differs from CPU reference: {metrics}")
    return metrics


def validate_partition_short_keys(adapter, device, report, save, torch, replay_paths=()):
    """Regression for dummy-key underflow and full final padding tiles (E71/72)."""
    report["partition_short_key_checks"] = []
    cases = [(nk, False) for nk in (1, 2, 3, 77, 511, 513, 1025, 8193)] + [(3, True)]
    for nk, negative in cases:
        nq = 129 if nk <= 3 else 17
        generator = torch.Generator().manual_seed(1337)
        original = [(torch.randn((1,1,n,64), generator=generator)*5).bfloat16()
                    for n in (nq,nk,nk)]
        if negative:
            original[0].fill_(2)
            original[1].fill_(-2)
            original[2] = torch.arange(nk).view(1,1,nk,1).expand(1,1,nk,64).bfloat16().clone()
        original[2][...,0] = 2
        payload = adapter.prepare(*(x.to(device) for x in original))
        actual = adapter.run_prepared(payload).cpu()
        row = dict(keys=nk, queries=nq, seed=1337, negative_scores=negative,
                   nonfinite_elements=int((~torch.isfinite(actual)).sum()),
                   payload=adapter.describe_payload(payload))
        report["partition_short_key_checks"].append(row)
        save()  # Preserve finite evidence even when metric calculation cannot proceed.
        if row["nonfinite_elements"]:
            raise AssertionError(f"Short-key regression: K={nk}, negative={negative}: nonfinite output")
        expected = common.dense_reference(*original, torch.arange(nq))
        baseline.record_gate(row,"dense_fp32_all_rows_all_keys",actual,expected,.01,save)
        row["lse"] = check_partition_lse(adapter,payload,torch.arange(nq),torch)
        torch.testing.assert_close(actual[...,0],torch.full_like(actual[...,0],2),rtol=0,atol=0)
        if negative or nk == 1:
            torch.testing.assert_close(actual,expected.to(actual.dtype),rtol=0,atol=0)
        row["constant_and_single_key_checks"] = "passed"
        save()
    report["replayed_input_checks"] = []
    for path in replay_paths:
        path = Path(path).resolve()
        original = torch.load(path,map_location="cpu",weights_only=True)["inputs"]
        row = dict(path=str(path),artifact_sha256=common.sha256_file(path),
                   input_sha256=[common.tensor_sha256(x) for x in original])
        report["replayed_input_checks"].append(row)
        actual = adapter(*(x.to(device) for x in original)).cpu()
        row["nonfinite_elements"] = int((~torch.isfinite(actual)).sum())
        save()
        if row["nonfinite_elements"]:
            raise AssertionError(f"Replayed counterexample still nonfinite: {path}")
        expected = common.dense_reference(*original,torch.arange(original[0].shape[2]))
        baseline.record_gate(row,"dense_fp32_all_rows_all_keys",actual,expected,.01,save)


def validate_fused_packing(adapter,device,report,save,torch):
    """Compare actual consumer operands bitwise, including zeros and subnormals."""
    from mindie_laser_backend import SplitKVLaser
    control = SplitKVLaser(adapter.adapter,2,raw_stats=True,masked_padding=True)
    report['fused_pack_checks'] = []
    cases = [(17,nk,64,dtype,'contiguous') for nk in (1,2,3,385,511,513,8193)
             for dtype in (torch.float16,torch.bfloat16)]
    cases += [(65,1025,128,dtype,layout) for dtype in (torch.float16,torch.bfloat16)
              for layout in ('contiguous','strided','offset','cast_edges')]
    for nq,nk,d,dtype,layout in cases:
        generator=torch.Generator().manual_seed(7301)
        original=[(torch.randn((2,3,n,d),generator=generator)*5).to(dtype) for n in (nq,nk,nk)]
        if layout == 'cast_edges':
            values=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype)
            values=values[torch.isfinite(values) & (values.float().abs()<=65504)]
            original=[values.repeat((x.numel()+values.numel()-1)//values.numel())[:x.numel()].reshape(x.shape) for x in original]
        inputs=[]
        for x in original:
            if layout=='strided':
                x=x.transpose(-1,-2).contiguous().to(device).transpose(-1,-2)
            elif layout=='offset':
                storage=torch.empty(x.numel()+1,dtype=dtype,device=device)
                storage[1:].copy_(x.flatten().to(device))
                x=storage[1:].view(x.shape)
            else:x=x.to(device)
            inputs.append(x)
        actual,expected=adapter.prepare(*inputs),control.prepare(*inputs)
        row=dict(queries=nq,keys=nk,dimension=d,dtype=str(dtype),layout=layout,
                 input_strides=[list(x.stride()) for x in inputs],input_alignment32=[x.data_ptr()%32 for x in inputs],
                 mismatched_elements=[])
        report['fused_pack_checks'].append(row)
        for got,want in zip(actual.tensors,expected.tensors):
            if got.shape!=want.shape or got.dtype!=want.dtype or got.device!=want.device or not got.is_contiguous():
                raise AssertionError('Fused packing shape/dtype/device/layout differs')
            row['mismatched_elements'].append(int((got.cpu().view(torch.int16)!=want.cpu().view(torch.int16)).sum()))
            save()
        if any(row['mismatched_elements']):raise AssertionError(f'Fused payload differs bitwise: {row}')
        assert (actual.pre_tokens,actual.extra_zero_keys,actual.masked_tail_keys,actual.duplicated_single_key)==(expected.pre_tokens,expected.extra_zero_keys,expected.masked_tail_keys,expected.duplicated_single_key)
    return control


def worker(args):
    name = args._worker
    resolved = build_arguments(args, name)
    directory = Path(resolved._run_dir) / resolved._worker
    directory.mkdir(parents=True, exist_ok=True)
    quantized = name not in ("installed", "baseline")
    report = {"status": "running", "mode": name, "pid": os.getpid(),
              "scheduling": {"splits": getattr(args, "split_kv", 1), "source_variant": getattr(args, "baseline_source_variant", "baseline")},
              "implementation_contract": QUANT_SCOPE if quantized else "Original FP16 Laser QK/PV",
              "shape_bnsd": list(args.shape), "dtype": "bfloat16", "seed": getattr(args,"seed",0), "std": args.std}
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
            if getattr(args,"baseline_source_variant","baseline") in ("raw_stats","raw_masked") and getattr(args,"merge_build",None):
                from mindie_laser_backend import load_local_raw_stats
                adapter = load_local_raw_stats(dict(provider=vars(resolved),
                    merge_manifest=str(Path(args.merge_build).resolve()),device=args.device,
                    pack_manifest=str(Path(args.pack_build).resolve()) if getattr(args,'pack_build',None) else None,
                    allow_legacy_unsafe=args.baseline_source_variant == "raw_stats"))
                transforms = adapter.metadata["source_transform_validation"]
                report["merge_build"] = adapter.metadata["merge_build"]
                report["public_loader"] = True
            else:
                adapter, transforms = baseline.load_adapter(resolved, source)
                if getattr(args, "split_kv", 1) > 1:
                    from mindie_laser_backend import SplitKVLaser
                    merger, merge_evidence = None, None
                    if getattr(args, "merge_build", None):
                        from mindie_laser_quant_fused import load_fused_merger
                        merger, merge_evidence = load_fused_merger(args.merge_build)
                        report["merge_build"] = merge_evidence
                    adapter = SplitKVLaser(adapter, args.split_kv, fused_merge=merger,
                                           raw_stats=getattr(args,"baseline_source_variant","baseline") in ("raw_stats","raw_masked"),
                                           masked_padding=getattr(args,"baseline_source_variant","baseline") == "raw_masked")
            report["baseline_source_variant"] = getattr(args, "baseline_source_variant", "baseline")
            report["split_kv"] = getattr(args, "split_kv", 1)
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
            if getattr(args, "merge_build", None):
                report["fused_merge_direct_probes"] = validate_fused_merge_direct(adapter.fused_merge, device, torch,
                    raw_stats=getattr(adapter,"raw_stats",False))
                save()
            if getattr(args,'pack_build',None):
                validate_fused_packing(adapter,device,report,save,torch)
            validate_small(adapter, device, report, save, torch)
            if not quantized and getattr(args, "baseline_source_variant", "baseline") in ("lse", "raw_stats", "raw_masked"):
                report["partition_small_checks"] = []
                for probe_std in (0., 5.):
                    host_q, host_k, host_v = [x[:, :, :n].contiguous() for x, n in zip(
                        synthetic_inputs(torch, (2, 1, 385, 64), probe_std), (17, 385, 385))]
                    host_v[..., 0] = 2
                    probe_payload = adapter.prepare(*(x.to(device) for x in (host_q, host_k, host_v)))
                    actual = adapter.run_prepared(probe_payload).cpu()
                    ref = common.dense_reference(host_q, host_k, host_v, torch.arange(17))
                    row = {"std": probe_std, "lse": check_partition_lse(adapter, probe_payload, torch.arange(17), torch)}
                    report["partition_small_checks"].append(row)
                    baseline.record_gate(row, "dense_fp32", actual, ref, .01, save)
                    torch.testing.assert_close(actual[..., 0], torch.full_like(actual[..., 0], 2), rtol=0, atol=0)
            if not quantized and getattr(args,"baseline_source_variant","baseline") == "raw_masked":
                validate_partition_short_keys(adapter,device,report,save,torch,getattr(args,"replay_inputs",()))
            original = synthetic_inputs(torch, args.shape, args.std, seed=getattr(args,"seed",0))
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
            for operation in ('pack','merge'):
                if getattr(args,operation+'_build',None):
                    inspect_fused_preprocessor(args,report,operation=operation)
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
            if getattr(args,'pack_build',None):
                from mindie_laser_backend import SplitKVLaser
                control=SplitKVLaser(adapter.adapter,2,raw_stats=True,masked_padding=True,fused_merge=adapter.fused_merge)
                control_payload=control.prepare(*inputs)
                report['long_pack_mismatched_elements']=[int((a.cpu().view(torch.int16)!=b.cpu().view(torch.int16)).sum()) for a,b in zip(payload.tensors,control_payload.tensors)]
                actual_all=adapter.run_prepared(payload).cpu()
                control_all=control.run_prepared(control_payload).cpu()
                report['long_output_mismatched_elements']=int((actual_all.view(torch.int16)!=control_all.view(torch.int16)).sum())
                save()
                if any(report['long_pack_mismatched_elements']) or report['long_output_mismatched_elements']:
                    raise AssertionError('Fused packing changed long payload or full output bits')
                del control,control_payload,actual_all,control_all
            if not quantized and getattr(args, "merge_build", None):
                from mindie_laser_backend import SplitKVLaser
                raw_partition = adapter.run_native(payload)
                actual_merge = adapter.merge_output(raw_partition, payload).cpu()
                torch_merge = SplitKVLaser(adapter.adapter, payload.splits, raw_stats=payload.raw_stats).merge_output(raw_partition, payload).cpu()
                baseline.record_gate(report, "fused_vs_torch_merge_all_output", actual_merge, torch_merge, .001, save)
                del raw_partition, actual_merge, torch_merge
            baseline.record_gate(report, "long_full_vs_prepared_sample", sample, core_sample, .001, save)
            if not quantized and getattr(args, "baseline_source_variant", "baseline") in ("lse", "raw_stats", "raw_masked"):
                report["partition_lse_check"] = check_partition_lse(adapter, payload, ids, torch)
                save()
            for scope, values in (("full", sample), ("prepared_core", core_sample)):
                path = directory / (scope + "_sample.pt")
                torch.save({"query_ids": ids, "output": values, "shape_bnsd": list(args.shape),
                            "seed": getattr(args,"seed",0), "std": args.std, "input_sha256": fingerprints}, path)
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
            elif getattr(args, "split_kv", 1) > 1:
                raw_partition = adapter.run_native(payload)
                torch.npu.synchronize()
                benchmark_calls({"partition_native_api": lambda: adapter.run_native(payload),
                                 "merge_postprocess": lambda: adapter.merge_output(raw_partition, payload)},
                                args, torch, report, save, timing_key="component_timing_ms")
                report["component_timing_scope"] = "Separate synchronized API calls; not kernel-only, not additive"
            if getattr(args, "trace_prepared", False):
                # Trace after all primary API/component timings; reuse the fixed
                # payload. No new preprocessing, reference or D2H in this scope.
                from mindie_laser_quant_trace import capture_native_traces
                report["prepared_trace"] = capture_native_traces(
                    {"prepared_pipeline": lambda: adapter.run_prepared(payload)},
                    torch=torch, torch_npu=torch_npu, directory=directory / "prepared_trace", repeats=3)
                if getattr(args,'pack_build',None):
                    report['packing_trace']=capture_native_traces(
                        {'preprocessing':lambda:adapter.prepare(*inputs),'full_pipeline':lambda:adapter(*inputs)},
                        torch=torch,torch_npu=torch_npu,directory=directory/'packing_trace',repeats=3)
                save()
            if (marker.stat().st_size if marker.exists() else 0) != marker_size:
                raise AssertionError("Marker writes continued during timing")
            baseline.inspect_provider(resolved._worker, resolved.vendor_root, resolved.bridge_path,
                                      report, directory, **provider)
            if name == "qk_int8_fused":
                inspect_fused_preprocessor(resolved, report)
            if getattr(args, "merge_build", None):
                evidence = report["merge_build"]
                library = evidence["library_path"]
                if library not in report["mapped_library_paths"] or common.sha256_file(library) != evidence["library_sha256"]:
                    raise RuntimeError("Fused merge provider missing or changed in actual worker mappings")
                report["loaded_library_sha256"][library] = evidence["library_sha256"]
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
    title = ("Original FP16 MindIE Laser" if all(name in ("installed", "baseline") for name in report["process_order"])
             else "Quantized MindIE Laser")
    lines = ["", title + "; synchronized wall ms/API",
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
            if "qk_quantize" in timing:
                lines.append(f"{name:<28} {timing['qk_quantize']['median']:>11.4f}"
                             f" {timing['native_api']['median']:>14.4f}"
                             f" {timing['output_postprocess']['median']:>21.4f}")
            else:
                lines.append(name + ": " + "; ".join(
                    f"{component}={value['median']:.4f}" for component, value in timing.items()))
        lines.append("Component calls are not additive or kernel-only times; native API includes allocation/dispatch.")
    lines += [experiment_contract(report["process_order"]),
              "Prepared includes output V scale where needed and conversion/trim; full freshly preprocesses every call.",
              "Preprocessing measured separately; full minus prepared is not used as its estimate.",
              "Process-grouped synchronized wall timings, not paired/interleaved or kernel-only measurements.",
              "Accuracy samples cover at most 32 query rows, all keys, heads and features; no model-quality claim."]
    for name in report["process_order"]:
        row = report["workers"][name]
        if row.get("baseline_source_variant", "baseline") != "baseline" or row.get("split_kv", 1) > 1:
            lines.append(f"{name}: source={row.get('baseline_source_variant')}; KV splits={row.get('split_kv', 1)}; changed reduction order")
    return "\n".join(lines)


def parent(args):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args._run_dir = str(Path(tempfile.mkdtemp(prefix=output.stem + "_", dir=output.parent)).resolve())
    report = {"status": "running", "shape_bnsd": list(args.shape), "dtype": "bfloat16", "seed": getattr(args,"seed",0),
              "std": args.std, "process_order": args.variants, "artifact_dir": args._run_dir,
              "workers": {}, "provenance": {}, "proof_scope": baseline.PROOF_SCOPE,
              "implementation_contract": experiment_contract(args.variants),
              "timing_scope": {"preprocessing": "fresh conversion, padding, quantization and decoded V (where applicable)",
                               "full": "fresh preprocessing + native allocation/dispatch + V output scale/cast/trim/contiguous",
                               "prepared_core": "fixed native input payload + native allocation/dispatch + V output scale/cast/trim/contiguous"}}
    save = lambda: common.write_json(output, report)
    save()
    try:
        if getattr(args, "merge_build", None):
            from mindie_laser_quant_fused import validate_preprocess_manifest
            report["merge_build"] = validate_preprocess_manifest(args.merge_build, operation="merge")
        if getattr(args,'pack_build',None):
            from mindie_laser_quant_fused import validate_preprocess_manifest
            report['pack_build']=validate_preprocess_manifest(args.pack_build,operation='pack')
        for name in args.variants:
            resolved = build_arguments(args, name)
            report["provenance"][name] = (baseline.provenance(resolved, expected_variant=getattr(args, "baseline_source_variant", "baseline")) if name in ("installed", "baseline")
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
            command = worker_command(args, name)
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
        print(f"Source variant={args.baseline_source_variant}; KV splits={args.split_kv}", flush=True)
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
    parser.add_argument("--baseline-source-variant", choices=("baseline", "lse", "raw_stats", "raw_masked"), default="baseline")
    parser.add_argument("--split-kv", type=int, default=1)
    parser.add_argument("--merge-build", help="Audited direct-launch fused split2 merge build manifest")
    parser.add_argument('--pack-build',help='Audited split2 operand-packing build; requires raw_masked and fused merge')
    parser.add_argument("--trace-prepared", action="store_true", help="Capture all prepared-pipeline device tasks after API timing")
    parser.add_argument("--quant-build", default="results/mindie_laser_quant_build.json")
    parser.add_argument("--preprocess-build", default="results/mindie_laser_quant_preprocess_build.json")
    parser.add_argument("--bridge-manifest", default="results/mindie_laser_bridge.json")
    parser.add_argument("--output", default="results/mindie_laser_quant_compare.json")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(DEFAULT_VARIANTS),
                        help="Default: installed baseline qk_int8 qk_int8_fused; select E4M3 V/ExpCast experiments explicitly")
    parser.add_argument("--shape", type=int, nargs=4, default=baseline.SHAPE, metavar=("B", "H", "S", "D"))
    parser.add_argument("--std", type=float, default=1.0,
                        help="FP32 Gaussian standard deviation before BF16 rounding; default preserves old inputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-inputs",nargs="*",default=[],help="Saved CPU inputs counterexamples; checked before timing for raw_masked")
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--_worker", choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--_run-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.split_kv < 1 or (args.split_kv > 1 and args.baseline_source_variant not in ("lse", "raw_stats", "raw_masked")):
        parser.error("split-KV requires a positive split count and audited lse source")
    if args.baseline_source_variant in ("raw_stats","raw_masked") and args.split_kv != 2:
        parser.error("Raw stats source requires --split-kv 2")
    if args.merge_build and args.split_kv != 2:
        parser.error("Fused merge requires --split-kv 2")
    selected_variants = [args._worker] if args._worker else args.variants
    if args.pack_build and (args.baseline_source_variant!='raw_masked' or not args.merge_build or selected_variants!=['baseline']):
        parser.error('--pack-build requires raw_masked baseline with --merge-build and --split-kv 2')
    if args.replay_inputs and (args.baseline_source_variant != "raw_masked" or selected_variants != ["baseline"]):
        parser.error("--replay-inputs requires --variants baseline --baseline-source-variant raw_masked")
    if (args.split_kv > 1 or args.baseline_source_variant != "baseline") and selected_variants != ["baseline"]:
        parser.error("LSE/split-KV experiments require --variants baseline only")
    try:
        physical = physical_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    if physical == 0:
        parser.error("Physical NPU 0 is reserved for the administrator CI runner")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be in [0,2**63)")
    if not math.isfinite(args.std) or args.std < 0:
        parser.error("--std must be finite and nonnegative")
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
