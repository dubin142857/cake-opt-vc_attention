"""Offline presentation of existing profiles cannot imply successful capture."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import summarize_mindie_laser_quant_profile as summary


def capture():
    return {
        "status": "captured", "variants": {"native_fused_payload": {
            "status": "captured", "submitted_calls": 3,
            "device_summary": {
                "submitted_calls": 3, "device_tasks": 3,
                "all_device_tasks_sum_ms_per_call": 86.5031,
                "kernel_groups_total": 1, "kernel_groups": [{
                    "name": "VcLaserQuantAttention", "op_type": "VcLaserQuantAttention",
                    "task_type": "MIX_AIC", "tasks": 3,
                    "sum_duration_ms": 259.5093, "sum_duration_ms_per_call": 86.5031,
                    "median_task_duration_ms": 86.5,
                    "counters_as_exported": {"aic_mac_ratio": {
                        "median": .4, "min": .3, "max": .5, "samples": 3}}}]}}}}


def report():
    return {"status": "passed", "timing_status": "passed", "trace": capture(),
            "timings": {"full": {"rows": {
                "torch": {"median_ms": 106.0615}, "fused": {"median_ms": 90.948}}}},
            "comparisons": {"full": {"median_saved_ms": 14.9952, "faster_trials": 9, "trials": 9}}}


def test_existing_saved_report_offline(tmp_path, capsys):
    path = tmp_path / "old_report.json"
    path.write_text(json.dumps(report()), encoding="utf-8")
    before = path.read_bytes()
    assert summary.main(["--input", str(path)]) == 0
    text = capsys.readouterr().out
    assert "106.0615" in text and "+14.9952" in text and "9/9" in text
    assert "legacy schedule unverified" in text
    assert "capture_finalized: unverified (legacy report)" in text
    assert 'task_type="MIX_AIC"; tasks=3' in text
    assert "aic_mac_ratio: median=0.4; min=0.3; max=0.5; samples=3" in text
    assert "40%" not in text
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["old_report.json"]


def test_capture_root_and_finalized_schedule():
    data = capture()
    data["schedule"] = {"wait": 0, "warmup": 0, "active": 1, "repeat": 1,
                        "unit": "whole callable batch"}
    data["variants"]["native_fused_payload"]["capture_finalized"] = True
    text = summary.render_report(data)
    assert "NPU trace status: captured" in text
    assert "whole callable batch" in text
    assert "capture_finalized: true" in text
    assert "legacy" not in text


def test_missing_counters_explicit():
    data = capture()
    del data["variants"]["native_fused_payload"]["device_summary"]["kernel_groups"][0]["counters_as_exported"]
    text = summary.render_report(data)
    assert "counters_as_exported: missing (no numeric counter samples)" in text
    assert "aic_mac_ratio" not in text


def test_failed_trace_preserves_valid_wall_times(tmp_path, capsys):
    data = report()
    data["trace"] = {"status": "failed", "error": "export failed", "variants": {
        "native": {"status": "failed", "capture_finalized": False,
                   "failed_stage": "profiler_finalize", "error": {"message": "stop failed"}}}}
    path = tmp_path / "failed.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert summary.main(["--input", str(path)]) == 1
    text = capsys.readouterr().out
    assert "Uninstrumented timing_status: passed" in text
    assert "NPU trace status: failed" in text
    assert "NPU trace status: passed" not in text
    assert "device_summary: unavailable" in text


@pytest.mark.parametrize("payload", ["[]", "{}", "{bad json}", '{"status":"passed","trace":NaN}',
                                    '{"status":"passed","timings":{"x":{"rows":{"torch":{"median_ms":Infinity}}}}}'])
def test_malformed_json_nonzero(payload, tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    assert summary.main(["--input", str(path)]) == 2
    assert "Cannot summarize report" in capsys.readouterr().err


@pytest.mark.parametrize("key,value", [("median", float("nan")), ("min", float("inf")),
                                     ("samples", -1), ("samples", True)])
def test_invalid_counter_rejected(key, value):
    data = capture()
    data["variants"]["native_fused_payload"]["device_summary"]["kernel_groups"][0]["counters_as_exported"]["aic_mac_ratio"][key] = value
    with pytest.raises(ValueError):
        summary.render_report(data)


def test_no_runtime_imports(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report()), encoding="utf-8")
    code = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'torch_npu'):
        raise AssertionError('runtime import attempted: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.path.insert(0, sys.argv[1])
import summarize_mindie_laser_quant_profile as module
raise SystemExit(module.main(['--input', sys.argv[2]]))
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(SCRIPTS), str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "aic_mac_ratio" in result.stdout


def test_raw_only_explicit_and_no_counters_invented():
    text = summary.render_report({"status": "raw_only", "variants": {
        "native": {"status": "raw_only", "reason": "No kernel_details.csv"}}})
    assert "NPU trace status: raw_only" in text
    assert "device_summary: unavailable" in text
