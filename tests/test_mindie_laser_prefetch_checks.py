"""Regression harness coverage and its behavior on buffer-lifetime failures."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mindie_laser_prefetch_checks as checks


class Adapter:
    def __init__(self, *, stale=False, wrong_tail=False, nonfinite=False):
        self.stale = stale
        self.wrong_tail = wrong_tail
        self.nonfinite = nonfinite
        self.cached = None
        self.calls = []

    def prepare(self, q, k, v):
        return SimpleNamespace(q=q.float().clone(), k=k.float().clone(), v=v.float().clone(),
                               q_scale=q.float().abs().amax().reshape(1),
                               k_scale=k.float().abs().amax().reshape(1))

    def run_native(self, payload):
        self.calls.append((id(payload), payload.q.data_ptr(), payload.q.clone()))
        output = payload.q + payload.k.mean() + payload.v.mean() + payload.q_scale * payload.k_scale
        if self.stale:
            if self.cached is None:
                self.cached = output.clone()
            output = self.cached.clone()
        if self.wrong_tail:
            output = output.clone()
            output[..., -1, -1] += 1
        if self.nonfinite:
            output = output.clone()
            output[..., -1, -1] = float("nan")
        return output


def tiny_inputs(case, dtype, device, revision):
    del case
    base = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)
    return tuple((base * (revision + 1) + index + revision).to(dtype).to(device)
                 for index in range(3))


def test_case_coverage_crosses_vector_batches_and_section_boundaries():
    cases = checks.PREFETCH_CASES
    assert {128, 129, 4096, 4097, 8193} <= {case["keys"] for case in cases}
    assert {64, 128} == {case["dim"] for case in cases}
    assert any(case["batch"] > 1 for case in cases)
    assert any(case["heads"] > 1 for case in cases)
    assert any(case["queries"] > 512 for case in cases)
    assert any(case["queries"] % 4 for case in cases)
    assert {"random", "zero_qk", "constant", "outliers"} == {case["pattern"] for case in cases}


def test_identical_payload_all_rows_and_in_place_refresh(monkeypatch):
    monkeypatch.setattr(checks, "_make_inputs", tiny_inputs)
    adapters = {name: Adapter() for name in ("baseline", "reuse")}
    report, snapshots = {}, []
    checks.validate_score_prefetch(adapters, "cpu", report, lambda: snapshots.append(copy.deepcopy(report)))
    rows = report["score_prefetch_validation"]
    assert len(rows) == 2 * len(checks.PREFETCH_CASES) * 4
    assert {row["dtype"] for row in rows} == {"torch.float16", "torch.bfloat16"}
    assert all(row["status"] == "passed" and row["exact_match"] and row["pointers_unchanged"] for row in rows)
    assert all(row["elements"] == 8 for row in rows)
    assert rows[0]["order"] == ["baseline", "reuse"]
    assert rows[1]["order"] == ["reuse", "baseline"]
    for offset in range(0, len(rows), 4):
        before = adapters["baseline"].calls[offset:offset + 4]
        other = adapters["reuse"].calls[offset:offset + 4]
        assert len({call[0] for call in before + other}) == 1
        assert len({call[1] for call in before + other}) == 1
        assert torch.equal(before[0][2], before[1][2])
        assert not torch.equal(before[1][2], before[2][2])
        assert torch.equal(before[2][2], before[3][2])
    assert snapshots[0]["score_prefetch_validation_scope"]["same_payload_object"]
    assert snapshots[-1]["score_prefetch_validation"][-1]["status"] == "passed"


@pytest.mark.parametrize("failure, expected_iteration, match", [
    ("stale", 2, "changed padded native outputs"),
    ("wrong_tail", 0, "changed padded native outputs"),
    ("nonfinite", 0, "Non-finite"),
])
def test_failure_preserves_case_and_partial_results(monkeypatch, failure, expected_iteration, match):
    monkeypatch.setattr(checks, "_make_inputs", tiny_inputs)
    report, snapshots = {}, []
    adapters = {"baseline": Adapter(), "reuse": Adapter(**{failure: True})}
    with pytest.raises((AssertionError, ValueError), match=match):
        checks.validate_score_prefetch(adapters, "cpu", report, lambda: snapshots.append(copy.deepcopy(report)))
    rows = snapshots[-1]["score_prefetch_validation"]
    assert len(rows) == expected_iteration + 1
    assert rows[-1]["iteration"] == expected_iteration and rows[-1]["status"] == "failed"
    assert rows[-1]["name"] == checks.PREFETCH_CASES[0]["name"]
    assert match in rows[-1]["error"]
    assert all(row["status"] == "passed" for row in rows[:-1])


def test_generated_patterns_are_finite_and_change_on_refresh():
    for pattern in ("random", "zero_qk", "constant", "outliers"):
        case = dict(batch=1, heads=2, queries=17, keys=129, dim=64, pattern=pattern)
        for dtype in (torch.float16, torch.bfloat16):
            old = checks._make_inputs(case, dtype, "cpu", 0)
            new = checks._make_inputs(case, dtype, "cpu", 1)
            assert all(torch.isfinite(tensor).all() and tensor.dtype == dtype for tensor in (*old, *new))
            assert not torch.equal(old[2], new[2])
            if pattern == "zero_qk":
                assert all(torch.count_nonzero(tensor) == 0 for tensor in (old[0], old[1], new[0], new[1]))
            elif pattern == "outliers":
                assert old[1][0, 0, 128, 0] == 8
                assert new[1][0, 0, 128, 1] == -6


def test_missing_adapter_fails_before_attempting_calls():
    with pytest.raises(ValueError, match="baseline and reuse"):
        checks.validate_score_prefetch({"baseline": Adapter()}, "cpu", {}, lambda: None)
