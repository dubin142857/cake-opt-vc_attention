#!/usr/bin/env python3
"""One explicit native Laser call for an externally launched msOpProf process.

Host descriptions never import torch. This worker does not fork, compile,
download, warm up attention, or start any profiler/instrumented kernel.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import benchmark_mindie_laser_quant as quant

common = quant.common
ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("original_laser", "vector_pv_prefetch", "fp16_expcast_control", "fp16_expcast")
MANIFESTS = dict(baseline_build="results/mindie_laser_build.json",
    bridge_manifest="results/mindie_laser_bridge.json",
    stable_build="results/mindie_laser_quant_vscale_pv_prefetch_build.json",
    expcast_build="results/mindie_laser_quant_vscale_pv_expcast_build.json",
    preprocess_build="results/mindie_laser_quant_preprocess_build.json")


def provider_info(variant, *, device=0, shape=(1, 28, 37741, 128), output=None):
    """Return JSON-only host evidence after checking the selected built artifacts."""
    if variant not in VARIANTS:
        raise ValueError("Unsupported msOpProf variant: " + str(variant))
    if type(device) is not int or device < 0:
        raise ValueError("Device must be a nonnegative physical index")
    if len(shape) != 4 or any(type(x) is not int or x <= 0 for x in shape) or shape[-1] not in (64, 128):
        raise ValueError("Expected a positive BNSD shape with D64/128")
    output = Path(output or ROOT/"results"/("msopprof_"+variant+"_worker.json")).resolve()
    directory = output.parent/(output.stem+"_artifacts")
    paths = {key:str((ROOT/value).resolve()) for key,value in MANIFESTS.items()}
    options = SimpleNamespace(**paths, device=device, warmup=0, repeats=1, trials=1,
                              cpu_threads=4, _run_dir=str(directory))
    original = variant == "original_laser"
    source = None
    if original:
        provider = quant.build_arguments(options, "baseline")
        evidence = quant.baseline.provenance(provider)
        marker_env, token = quant.baseline.MARKER_ENV, quant.baseline.MARKER_TOKEN
        op_type, namespace = quant.baseline.OP_TYPE, "vc_laser"
    else:
        suffix = "vscale_pv_prefetch" if variant == "vector_pv_prefetch" else "vscale_pv_expcast"
        source = importlib.import_module("prepare_mindie_laser_quant_"+suffix+"_source")
        options.quant_build = paths["stable_build" if variant == "vector_pv_prefetch" else "expcast_build"]
        provider = quant.build_arguments(options, "qk_int8_fused")
        evidence = quant.quant_provenance(provider, identity=source.IDENTITY,
                                         source_validator=source.validate_prepared_source)
        from mindie_laser_quant_fused import validate_preprocess_manifest
        evidence["preprocessor"] = validate_preprocess_manifest(paths["preprocess_build"])
        marker_env, token = source.MARKER_ENV, source.MARKER_TOKEN
        op_type, namespace = source.OP_TYPE, "vc_laser_quant_"+suffix
    build = evidence["build_manifest"]
    cann = build.get("cann_root") or build.get("definitions", {}).get("CUSTOM_ASCEND_CANN_PACKAGE_PATH")
    if not cann:
        raise ValueError("Selected build does not identify its CANN Toolkit")
    return dict(variant=variant, device=device, shape_bnsd=list(shape), dtype="bfloat16", seed=0,
        output=str(output), artifact_dir=str(directory), provider=vars(provider),
        namespace=namespace, kernel_prefix=op_type+"_", op_type=op_type,
        cann_root=str(Path(cann).resolve()), provenance=evidence,
        marker_env=marker_env, marker_token=token, marker_path=str(directory/"markers.jsonl"),
        expcast=variant == "fp16_expcast", native_attention_calls=1,
        scope="One prepared native API call; preprocessing and CPU reference are outside the kernel-name filter")


def worker_environment(base, info):
    """Construct the one-vendor environment BEFORE the external worker starts."""
    if base.get("ASCEND_RT_VISIBLE_DEVICES", "").strip():
        raise ValueError("Unset ASCEND_RT_VISIBLE_DEVICES: msOpProf uses a physical device index")
    env = quant.baseline.worker_environment(base, "local", info["provider"]["vendor_root"], info["artifact_dir"])
    for key in list(env):
        if key.startswith("VC_") and key.endswith("MARKER_FILE"):
            env.pop(key)
    env[info["marker_env"]] = info["marker_path"]
    env["ASCEND_HOME_PATH"] = info["cann_root"]
    return env


def check_environment(info, env):
    """Fail before loading torch rather than mutating an already-loaded runtime."""
    if env.get("ASCEND_RT_VISIBLE_DEVICES", "").strip():
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES is incompatible with the physical-device collection contract")
    expected = Path(info["provider"]["vendor_root"]).resolve()
    paths = [p for p in env.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if p]
    if len(paths) != 1 or Path(paths[0]).resolve() != expected:
        raise ValueError("Worker requires exactly its selected custom OPP provider")
    if env.get(info["marker_env"]) != info["marker_path"]:
        raise ValueError("Worker requires its unique tiling marker path")
    if Path(env.get("ASCEND_HOME_PATH", ".")).resolve() != Path(info["cann_root"]).resolve():
        raise ValueError("Worker CANN path differs from the audited build")
    if not env.get("ACL_OP_COMPILER_CACHE_DIR"):
        raise ValueError("Worker requires a private disabled-compiler-cache directory")
    if any(name == "mindiesd" or name.startswith("mindiesd.") for name in sys.modules):
        raise ValueError("Worker must not import installed mindiesd")


class DeviceStages:
    """Record the first failure; never attempt another operation or cleanup sync."""
    def __init__(self, report, save, synchronize):
        self.report, self.save, self.synchronize = report, save, synchronize
        self.failed = False

    def call(self, label, operation):
        if self.failed:
            raise RuntimeError("Device work stopped after first failure")
        self.report["active_device_operation"] = dict(label=label, phase="about_to_call")
        self.save()
        try:
            result = operation()
            self.report["active_device_operation"]["phase"] = "awaiting_sync"
            self.save()
            self.synchronize()
            self.report["active_device_operation"]["phase"] = "completed"
            self.save()
            return result
        except Exception as exc:
            self.failed = True
            self.report["first_device_failure"] = dict(operation=dict(self.report["active_device_operation"]),
                error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                further_device_access_aborted=True)
            self.save()
            raise


def native_once(adapter, payload, variant, stages):
    if variant == "original_laser":
        from mindie_laser_fp16_trace_worker import native_call
        return stages.call("native_attention", lambda:native_call(adapter, payload))
    return stages.call("native_attention", lambda:adapter.run_native(payload))


def output_tensor(raw, payload, variant, torch):
    if variant == "original_laser":
        if not isinstance(raw, (tuple,list)) or len(raw) != 2:
            raise ValueError("Original Laser must return (statistics, output)")
        raw = raw[1]
        # Upstream la.cpp allocates FP32 statistics and FP32 attention_out,
        # although its QK/PV operands are FP16.
        expected_dtype = torch.float32
        expected_shape = payload.tensors[0].shape
    else:
        expected_dtype = torch.float32
        expected_shape = payload.q.shape
    if not isinstance(raw, torch.Tensor) or raw.dtype != expected_dtype or raw.shape != expected_shape:
        raise ValueError("Unexpected native Laser output shape/dtype")
    return raw


def cpu_postprocess(raw_cpu, adapter, payload, variant):
    if raw_cpu.device.type != "cpu":
        raise ValueError("Output postprocessing must run on CPU outside the collected kernel")
    if variant == "original_laser":
        return adapter.postprocess(raw_cpu, payload.dtype, payload.shape[2], payload.shape[3])
    if payload.v_quantize:
        raise ValueError("msOpProf route must preserve unquantized FP16 V")
    return adapter.postprocess(raw_cpu, payload)


def collect(args):
    output = Path(args.output).resolve()
    report = dict(status="running", stage="host_provenance", variant=args.variant,
        native_attention_calls=0, pid=os.getpid(), instrumentation="external msOpProf only; no source probes")
    save = lambda:common.write_json(output, report)
    save()
    try:
        info = provider_info(args.variant, device=args.device, shape=tuple(args.shape), output=output)
        report["host"] = info
        check_environment(info, os.environ)
        directory = Path(info["artifact_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        report["stage"] = "load_runtime"
        save()
        import torch
        import torch_npu
        from mindie_laser_fp16_trace_worker import verify_runtime_target as original_target
        from profile_mindie_laser_quant import verify_runtime_target as quant_target
        from profile_mindie_laser_vector_scale_bulk import inspect_providers
        provider = SimpleNamespace(**info["provider"])
        provider.marker_env, provider.marker_token = info["marker_env"], info["marker_token"]
        evidence = info["provenance"]
        torch.set_num_threads(min(4, torch.get_num_threads()))
        if args.variant == "original_laser":
            adapter, transforms = quant.baseline.load_adapter(provider, evidence["source_manifest"])
            report["source_transform_validation"] = transforms
        else:
            from mindie_laser_quant_backend import load_quant_laser
            adapter = load_quant_laser(provider.bridge_path, preprocessing="fused",
                preprocess_build=provider.preprocess_build, torch_namespace=info["namespace"],
                v_quantize=False, expcast=info["expcast"], check_finite=False)
            from mindie_laser_quant_vscale_patch import VSCALE_CONTRACT
            adapter.metadata["qk_result_contract"] = VSCALE_CONTRACT
        report["adapter"] = adapter.metadata
        stages = DeviceStages(report, save, torch.npu.synchronize)
        torch.npu.set_option({"ACL_OP_COMPILER_CACHE_MODE":"disable",
            "ACL_OP_COMPILER_CACHE_DIR":os.environ["ACL_OP_COMPILER_CACHE_DIR"]})
        torch.npu.set_device(args.device)
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtGetSocName.restype = ctypes.c_char_p
        soc_bytes = acl.aclrtGetSocName()
        if not soc_bytes:
            raise RuntimeError("aclrtGetSocName returned null")
        soc = soc_bytes.decode()
        if args.variant == "original_laser":
            target = original_target(evidence["build_manifest"], soc, info["cann_root"])
        else:
            prep = json.loads(Path(provider.preprocess_build).read_text(encoding="utf-8"))
            target = quant_target(evidence["build_manifest"], prep, soc, info["cann_root"])
        report["runtime_target"] = target
        report["environment"] = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
            runtime_soc=soc, device_index=args.device, shape_bnsd=list(args.shape), dtype="bfloat16")
        with torch.no_grad():
            generator = torch.Generator().manual_seed(0)
            host = [torch.randn(args.shape, generator=generator).to(torch.bfloat16) for _ in range(3)]
            report["input_sha256"] = [common.tensor_sha256(x) for x in host]
            report["stage"] = "input_transfer_and_preprocessing"
            inputs = stages.call("input_transfer", lambda:tuple(x.to(f"npu:{args.device}") for x in host))
            payload = stages.call("prepare_once", lambda:adapter.prepare(*inputs))
            report["payload"] = adapter.describe_payload(payload)
            marker = Path(info["marker_path"])
            offset = marker.stat().st_size if marker.exists() else 0
            report["stage"] = "single_native_call"
            report["native_attention_calls"] = 1
            save()
            raw = native_once(adapter, payload, args.variant, stages)
            with marker.open("rb") as stream:
                stream.seek(offset)
                fresh = stream.read().decode("utf-8")
            report["host_markers"] = (quant.baseline.parse_markers(fresh, os.getpid()) if args.variant == "original_laser"
                else quant.parse_quant_markers(fresh, os.getpid(), info["marker_token"]))
            report["stage"] = "provider_inspection"
            inspect_providers({args.variant:provider}, {args.variant:evidence}, report, directory,
                cann_root=info["cann_root"], soc=evidence["build_manifest"]["soc"])
            if args.variant != "original_laser":
                quant.inspect_fused_preprocessor(provider, report)
            report["provider_evidence_passed"] = True
            # Transfer the entire raw output once. All following slicing,
            # dtype conversion, finiteness checks and dense math are CPU-only.
            result = output_tensor(raw, payload, args.variant, torch)
            host_raw = stages.call("copy_output_to_cpu", lambda:result.detach().cpu())
            report["stage"] = "cpu_output_validation"
            actual = cpu_postprocess(host_raw, adapter, payload, args.variant)
            if tuple(actual.shape) != tuple(args.shape) or actual.dtype != torch.bfloat16:
                raise ValueError("CPU postprocessed output violates requested BNSD/BF16 contract")
            if not bool(torch.isfinite(actual).all()):
                raise ValueError("Non-finite output from profiled native attention")
            ids = torch.linspace(0, args.shape[2]-1, min(32,args.shape[2])).long().unique()
            sample = actual.index_select(2, ids).contiguous()
            # Same seeded inputs and valid rows let the collector compare its
            # unprofiled smoke result with the external profiler replay result.
            # Do not hash padding or the native auxiliary statistics tensor.
            report["output_sample_sha256"] = common.tensor_sha256(sample)
            dense = common.dense_reference(*host, ids)
            report["dense_fp32_sample"] = common.error_metrics(sample, dense)
            report["accuracy_scope"] = dict(query_ids=ids.tolist(), all_keys=args.shape[2],
                all_heads_and_features=True, full_output_finite=True,
                note="Dense error is diagnostic, not an implementation or model-quality gate; no extra attention launches")
        report.update(status="passed", stage="complete")
        save()
        print(f"msOpProf worker {args.variant}: one native call passed; {output}", flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save()  # Host I/O only: do not sync, retry, or inspect NPU state here.
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shape", type=int, nargs=4, default=(1,28,37741,128))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--describe-host", action="store_true")
    args = parser.parse_args()
    if args.describe_host:
        print(json.dumps(provider_info(args.variant, device=args.device, shape=tuple(args.shape), output=args.output),
                         indent=2, ensure_ascii=False))
        return 0
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
