"""CPU regressions for the early native-registration/execution smoke gate."""
import builtins
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_mindie_laser_quant_preprocess as smoke
import mindie_laser_quant_fused as fused


def cpu_quantizer(source):
    padded = F.pad(source.float(), (0, 64, 0, 255))
    return padded.round().to(torch.int8), torch.ones((1, 1, 2), dtype=torch.float32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tiny_probe_checks_rne_and_both_padding_dimensions_on_cpu(dtype):
    report = smoke.run_dtype_probe(cpu_quantizer, dtype, "cpu")
    assert report["status"] == "passed"
    assert report["input_shape"] == [1, 1, 1, 64]
    assert report["quant_shape"] == [1, 1, 256, 128]
    assert report["scale_values"] == [[[1.0, 1.0]]]
    assert report["exact_rne_and_padding"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("corruption", ["rounding", "channel_padding", "token_padding", "second_block",
                                        "scale", "padded_scale", "shape", "dtype", "layout"])
def test_tiny_probe_rejects_bad_payloads(corruption):
    def quantize(source):
        quant, scales = cpu_quantizer(source)
        if corruption == "rounding":
            quant[0, 0, 0, 4] = 1  # +0.5 must round to even zero.
        elif corruption == "channel_padding":
            quant[0, 0, 0, 64] = 1
        elif corruption == "token_padding":
            quant[0, 0, 1, 0] = 1
        elif corruption == "second_block":
            quant[0, 0, 128, 0] = 1
        elif corruption == "scale":
            scales[0, 0, 0] = 2
        elif corruption == "padded_scale":
            scales[0, 0, 1] = 0
        elif corruption == "shape":
            quant = quant[:, :, :128]
        elif corruption == "dtype":
            quant = quant.float()
        else:
            quant = quant.transpose(-1, -2).contiguous().transpose(-1, -2)
        return quant, scales

    with pytest.raises(RuntimeError, match="Tiny quantizer probe"):
        smoke.run_dtype_probe(quantize, torch.bfloat16, "cpu")


@pytest.fixture
def native_standins(tmp_path, monkeypatch):
    events = []
    output = tmp_path / "smoke.json"
    evidence = {"preprocessing_build": {"soc": "Ascend910_9382", "library_sha256": "fixture-sha"},
                "preprocessing_schema": "fixture-schema"}
    quantizer = object()
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__version__="fixture"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=lambda index: events.append(("device", index))),
                        raising=False)

    def load(path):
        assert path == "fixture-build.json"
        events.append(("load", path))
        return quantizer, evidence

    def probe(candidate, dtype, device):
        assert candidate is quantizer
        persisted = json.loads(output.read_text(encoding="utf-8"))
        assert persisted["status"] == "running"
        assert persisted["stage"] == ("fp16" if dtype == torch.float16 else "bf16")
        events.append(("probe", str(dtype), device))
        return {"status": "passed", "dtype": str(dtype)}

    monkeypatch.setattr(fused, "load_fused_quantizer", load)
    monkeypatch.setattr(smoke, "run_dtype_probe", probe)
    return output, events, evidence


def test_smoke_preserves_provenance_and_matches_worker_registration_order(native_standins):
    output, events, evidence = native_standins
    report = smoke.check("fixture-build.json", 3, output)
    assert report["status"] == "passed"
    assert report["stage"] == "complete"
    assert report["preprocessing"] == evidence
    assert events == [("load", "fixture-build.json"), ("device", 3),
                      ("probe", "torch.float16", "npu:3"), ("probe", "torch.bfloat16", "npu:3")]
    assert json.loads(output.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize("failed_stage", ["load_library", "initialize_device", "fp16", "bf16"])
def test_smoke_persists_first_failure_and_traceback(native_standins, monkeypatch, capsys, failed_stage):
    output, events, _ = native_standins

    def fail(*args, **kwargs):
        persisted = json.loads(output.read_text(encoding="utf-8"))
        assert persisted["status"] == "running" and persisted["stage"] == failed_stage
        raise RuntimeError("RegisterAscendBinary 107000 fixture")

    if failed_stage == "load_library":
        monkeypatch.setattr(fused, "load_fused_quantizer", fail)
    elif failed_stage == "initialize_device":
        monkeypatch.setattr(torch.npu, "set_device", fail)
    else:
        original_probe = smoke.run_dtype_probe

        def probe(candidate, dtype, device):
            if failed_stage == "fp16" or dtype == torch.bfloat16:
                fail()
            return original_probe(candidate, dtype, device)

        monkeypatch.setattr(smoke, "run_dtype_probe", probe)
    report = smoke.check("fixture-build.json", 0, output)
    assert report["status"] == "failed"
    assert report["first_failed_stage"] == failed_stage
    assert "RegisterAscendBinary 107000 fixture" in report["error"]
    assert "Traceback (most recent call last)" in report["traceback"]
    assert len(report["probes"]) == (1 if failed_stage == "bf16" else 0)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert f"FAILED at {failed_stage}" in capsys.readouterr().err
    if failed_stage == "load_library":
        assert events == []


def test_missing_npu_runtime_leaves_failed_report(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def missing_npu(name, *args, **kwargs):
        if name == "torch_npu":
            raise ModuleNotFoundError("No module named 'torch_npu'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_npu)
    output = tmp_path / "smoke.json"
    report = smoke.check("fixture-build.json", 0, output)
    assert report["status"] == "failed"
    assert report["first_failed_stage"] == "import_runtime"
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_help_does_not_import_torch_or_npu():
    # -S excludes installed packages, including torch, from the new interpreter.
    result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/check_mindie_laser_quant_preprocess.py"),
                             "--help"], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--build-manifest" in result.stdout and "--device" in result.stdout


def test_cli_failed_check_returns_nonzero(tmp_path):
    output = tmp_path / "failed.json"
    result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/check_mindie_laser_quant_preprocess.py"),
                             "--build-manifest", "missing.json", "--output", str(output)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    assert json.loads(output.read_text(encoding="utf-8"))["first_failed_stage"] == "import_runtime"
    assert "FAILED at import_runtime" in result.stderr
