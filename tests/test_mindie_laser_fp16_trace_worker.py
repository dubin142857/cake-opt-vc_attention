"""Keep original-FP16 diagnostics separate from quantized native scope."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mindie_laser_fp16_trace_worker as worker


def test_raw_native_scope_retains_original_tuple_without_postprocessing():
    calls = []
    tensors = tuple(object() for _ in range(3))
    returned = (object(), object())
    payload = SimpleNamespace(tensors=tensors, scale=.125, shape=(1, 28, 37741, 128), pre_tokens=147)
    def native(*args):
        calls.append(args)
        return returned
    adapter = SimpleNamespace(native=native)
    assert worker.native_call(adapter, payload) is returned
    assert calls == [(*tensors, None, None, None, .125, 28, "BNSD", 1.0, 147, 1, True)]


def test_fixed_fp16_scopes_do_not_prepare_or_call_full_wrapper():
    observed = []
    payload = SimpleNamespace(tensors=tuple(object() for _ in range(3)), scale=1.,
                              shape=(1, 2, 17, 64), pre_tokens=127)
    inputs = tuple(object() for _ in range(3))
    class Adapter:
        def native(self, *args):
            observed.append(("native", args))
            return "raw"
        def run_prepared(self, value):
            observed.append(("prepared", value))
            return "prepared"
        def prepare(self, *args):
            raise AssertionError("A fixed scope must not prepare inputs")
        def __call__(self, *args):
            observed.append(("full", args))
            return "full"
    calls = worker.make_calls(Adapter(), payload, inputs)
    assert set(calls) == {"native", "prepared", "full"}
    assert calls["native"][worker.ROW]() == "raw"
    assert calls["prepared"][worker.ROW]() == "prepared"
    assert observed[1] == ("prepared", payload)
    assert not any(event[0] == "full" for event in observed)
    assert calls["full"][worker.ROW]() == "full"
    assert observed[-1] == ("full", inputs)


def test_runtime_check_accepts_family_but_not_toolkit_mismatch(tmp_path):
    build = {"soc": "ascend910_93", "definitions": {"CUSTOM_ASCEND_CANN_PACKAGE_PATH": str(tmp_path)}}
    selection = worker.verify_runtime_target(build, "Ascend910_9382", tmp_path)
    assert selection["compiler_soc"] == "Ascend910_9382"
    with pytest.raises(ValueError, match="SoC family"):
        worker.verify_runtime_target(dict(build, soc="ascend910b"), "Ascend910_9382", tmp_path)
    with pytest.raises(ValueError, match="CANN Toolkit"):
        worker.verify_runtime_target(build, "Ascend910_9382", tmp_path / "other")


def test_worker_persists_provenance_failure_before_npu_loading(tmp_path, monkeypatch):
    resolved = SimpleNamespace(_run_dir=str(tmp_path / "baseline"))
    monkeypatch.setattr(worker.quant, "build_arguments", lambda args, mode: resolved)
    def fail(*args):
        raise ValueError("stale original bridge")
    monkeypatch.setattr(worker.baseline, "provenance", fail)
    args = SimpleNamespace(shape=(1, 28, 37741, 128))
    assert worker.run(args) == 1
    report = json.loads((tmp_path / "baseline/local/report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["stage"] == "verify_provenance"
    assert "stale original bridge" in report["error"]
    assert report["timings"] == {}
