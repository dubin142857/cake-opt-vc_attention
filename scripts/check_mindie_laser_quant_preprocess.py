#!/usr/bin/env python3
"""Fail early on fused-quantizer registration, launch, or exact tiny-probe errors.

Torch is imported only after argument parsing. This does not run Laser attention
or establish accuracy/performance for the full benchmark workload.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_dtype_probe(quantize, dtype, device):
    """Check an injected quantizer; CPU injection is for regression tests only."""
    import torch

    # Both formats represent these values exactly. The maximum gives scale=1,
    # making RNE ties unambiguous; the entirely padded second block also uses 1.
    values = [127, -127, 1, -2, .5, 1.5, 2.5, -.5, -1.5, -2.5]
    expected_codes = [127, -127, 1, -2, 0, 2, 2, 0, -2, -2]
    source = torch.zeros((1, 1, 1, 64), dtype=dtype)
    source[0, 0, 0, :len(values)] = torch.tensor(values, dtype=dtype)
    source = source.to(device)
    quant, scales = quantize(source)
    if (not isinstance(quant, torch.Tensor) or not isinstance(scales, torch.Tensor)
            or quant.shape != (1, 1, 256, 128) or quant.dtype != torch.int8
            or scales.shape != (1, 1, 2) or scales.dtype != torch.float32
            or quant.device != source.device or scales.device != source.device
            or not quant.is_contiguous() or not scales.is_contiguous()):
        raise RuntimeError("Tiny quantizer probe returned invalid shape, dtype, device, or layout")
    if source.device.type == "npu":
        torch.npu.synchronize(source.device)
    quant, scales = quant.cpu(), scales.cpu()
    expected_quant = torch.zeros((1, 1, 256, 128), dtype=torch.int8)
    expected_quant[0, 0, 0, :len(expected_codes)] = torch.tensor(expected_codes, dtype=torch.int8)
    if not torch.equal(scales, torch.ones((1, 1, 2), dtype=torch.float32)):
        raise RuntimeError(f"Tiny quantizer probe requires exact unit scales, got {scales.tolist()}")
    if not torch.equal(quant, expected_quant):
        mismatches = (quant != expected_quant).nonzero()
        index = tuple(mismatches[0].tolist())
        raise RuntimeError(
            f"Tiny quantizer probe failed exact RNE/zero-padding check: "
            f"{len(mismatches)} mismatches; first at {index}: "
            f"got {quant[index].item()}, expected {expected_quant[index].item()}")
    return {"status": "passed", "dtype": str(dtype), "input_shape": list(source.shape),
            "quant_shape": list(quant.shape), "scale_shape": list(scales.shape),
            "scale_values": scales.tolist(), "exact_rne_and_padding": True}


def check(build_manifest, device, output):
    report = {"status": "running", "stage": "import_runtime", "device_index": device,
              "build_manifest": str(Path(build_manifest).resolve()), "probes": [],
              "scope": "Native fused Q/K quantizer registration and tiny FP16/BF16 execution; no attention"}
    write_report(output, report)

    def stage(name):
        report["stage"] = name
        write_report(output, report)
        print(f"Fused preprocessing smoke: {name}", flush=True)

    try:
        stage("import_runtime")
        import torch
        import torch_npu
        from mindie_laser_quant_fused import load_fused_quantizer

        report.update(torch=torch.__version__, torch_npu=torch_npu.__version__)
        # Match the benchmark worker: constructor registration precedes device
        # initialization. Do not hide a bad generated registration stub by
        # changing initialization order in this check alone.
        stage("load_library")
        quantize, evidence = load_fused_quantizer(build_manifest)
        report["preprocessing"] = evidence
        stage("initialize_device")
        torch.npu.set_device(device)
        for name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            stage(name)
            report["probes"].append(run_dtype_probe(quantize, dtype, f"npu:{device}"))
        report.update(status="passed", stage="complete")
        write_report(output, report)
        print(f"Fused preprocessing smoke PASSED (FP16 + BF16): {output}", flush=True)
    except Exception as error:
        report.update(status="failed", first_failed_stage=report["stage"],
                      error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        write_report(output, report)
        print(f"Fused preprocessing smoke FAILED at {report['first_failed_stage']}: "
              f"{report['error']}", file=sys.stderr, flush=True)
        print(report["traceback"], file=sys.stderr, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-manifest", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="results/mindie_laser_quant_preprocess_smoke.json")
    args = parser.parse_args()
    report = check(args.build_manifest, args.device, args.output)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
