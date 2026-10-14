"""CPU fake-native tests of the reuse acceptance gates; not NPU validation."""
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ascend"))
import validate_ascend910b as validation
from vc_attention_ascend.api import prepare
from vc_attention_ascend.reference import payload_reference


@pytest.fixture
def prepared():
    q, k, v = validation.reuse_inputs(3, 33, 129, 64, torch.bfloat16,
                                     torch.Generator().manual_seed(49))
    p = prepare(q, k, v, layout="HND", _reference_only=True)
    return p, v[:, :, :1], payload_reference(p)


def recorder():
    snapshots = []

    def save(results, current):
        snapshots.append(copy.deepcopy({"results": results, "current": current}))

    return snapshots, save


def test_every_repeat_checks_all_rows_and_excludes_padding(prepared):
    payload, constant, expected = prepared
    snapshots, save = recorder()
    results, calls = [], []

    def forward(p, cast):
        calls.append((p, cast))
        # Native padding is deliberately unspecified; only valid queries count.
        padding = torch.full((1, 3, 31, 64), float("nan"))
        return torch.cat((expected, padding), dim=2)

    validation.check_reuse_payload(forward, payload, True, {"suite": "fake"}, results, save,
                                   repeats=5, constant_v=constant)
    assert len(calls) == len(results) == 5
    assert all(p is payload and cast is True for p, cast in calls)
    assert all(row["status"] == "passed" and row["determinism"]["bitwise_equal"] for row in results)
    assert [row["repeat"] for row in results] == [1, 2, 3, 4, 5]
    assert snapshots[-1]["current"] is None


@pytest.mark.parametrize("failing_repeat", [1, 2, 5])
def test_sparse_corruption_cannot_hide_in_global_or_head_rmse(prepared, failing_repeat):
    payload, constant, expected = prepared
    broken = expected.clone()
    broken[0, 0, 32] *= 1.04
    metric = validation.reuse_errors(broken.bfloat16(), expected.bfloat16())
    assert metric["relative_rmse"] < .01
    assert all(head["relative_rmse"] < .01 for head in metric["per_head"])
    assert metric["worst_row"]["relative_rmse"] > .01
    snapshots, save = recorder()
    results, calls = [], 0

    def forward(p, cast):
        nonlocal calls
        calls += 1
        return broken if calls == failing_repeat else expected

    with pytest.raises(AssertionError, match="Decoded-reference implementation RMSE exceeds 1%"):
        validation.check_reuse_payload(forward, payload, True, {"suite": "fake"}, results, save,
                                       constant_v=constant)
    assert calls == failing_repeat
    assert results[-1]["status"] == "failed"
    assert results[-1]["repeat"] == failing_repeat
    assert results[-1]["decoded_reference"]["worst_row"]["query"] == 32
    assert snapshots[-2]["current"]["status"] == "running"
    assert snapshots[-1]["current"]["status"] == "failed"


def test_one_raw_fp32_ulp_on_second_repeat_fails_despite_identical_bf16(prepared):
    payload, _, expected = prepared
    changed = expected.clone()
    changed[0, 0, 0, 0] = torch.nextafter(changed[0, 0, 0, 0], torch.tensor(float("inf")))
    assert torch.equal(changed.bfloat16(), expected.bfloat16())
    outputs = iter((expected, changed))
    snapshots, save = recorder()
    results = []
    with pytest.raises(AssertionError, match="not bitwise deterministic"):
        validation.check_reuse_payload(lambda p, cast: next(outputs), payload, True, {}, results, save)
    assert results[-1]["repeat"] == 2
    assert results[-1]["determinism"]["different_elements"] == 1
    assert results[-1]["determinism"]["first_changed_indices"] == [[0, 0, 0, 0]]
    assert snapshots[-1]["current"]["status"] == "failed"


def test_constant_v_invariant_is_an_independent_gate(prepared):
    payload, constant, expected = prepared
    results = []
    with pytest.raises(AssertionError, match="Constant-V invariant RMSE exceeds 1%"):
        validation.check_reuse_payload(lambda p, cast: expected, payload, True, {}, results,
                                       lambda *_: None, constant_v=constant * 1.05)
    assert results[-1]["decoded_reference"]["relative_rmse"] < .01
    assert results[-1]["constant_v_invariant"]["relative_rmse"] > .01


def test_native_exception_is_saved_and_reraised_unchanged(prepared):
    payload, _, _ = prepared
    failure = RuntimeError("simulated native launch failure")
    snapshots, save = recorder()
    results = []

    def forward(p, cast):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        validation.check_reuse_payload(forward, payload, True, {}, results, save)
    assert caught.value is failure
    assert snapshots[-1]["current"]["error_type"] == "RuntimeError"
    assert snapshots[-1]["results"][-1]["status"] == "failed"


def test_suite_covers_dtypes_dimensions_tails_and_cross_head_scale_reuse(monkeypatch):
    cases, launches = [], []
    marker = object()

    class FakeNative:
        def forward(self, *args):
            launches.append(args)
            return marker

    def fake_prepare(q, k, v, *, layout):
        assert layout == "HND"
        assert torch.equal(v, v[:, :, :1].expand_as(v))
        assert float(v[0, -1, 0, 0] / v[0, 0, 0, 0]) == 1024
        return {"q": q, "k": k, "v": v, "q_scale": None, "k_scale": None,
                "v_scale": None, "lut": None, "scale": .125, "keys": k.shape[2],
                "dtype": q.dtype, "queries": q.shape[2], "heads": q.shape[1], "dim": q.shape[3]}

    def fake_check(forward, payload, cast, case, results, save, *, constant_v):
        assert case["dtype"] == str(payload["dtype"])
        assert case["heads"] == payload["heads"] == constant_v.shape[1]
        assert case["dim"] == payload["dim"] == constant_v.shape[3]
        assert case["queries"] == payload["queries"]
        assert case["expcast"] is cast
        assert forward(payload, cast) is marker
        assert launches[-1][0] is payload["q"]
        assert launches[-1][7:] == (payload["queries"], 129, .125, cast)
        cases.append(case)

    monkeypatch.setattr(validation, "check_reuse_payload", fake_check)
    validation.validate_core_reuse("cpu", lambda *_: None, [], FakeNative(), fake_prepare)
    assert len(cases) == len(launches) == 16
    assert {(case["heads"], case["queries"], case["tasks"]) for case in cases} == {(2, 1025, 66), (65, 33, 130)}
    assert {case["dtype"] for case in cases} == {"torch.float16", "torch.bfloat16"}
    assert {case["dim"] for case in cases} == {64, 128}
    assert {case["expcast"] for case in cases} == {False, True}
