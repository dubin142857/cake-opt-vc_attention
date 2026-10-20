"""CPU fault injection for the remote Q-L0A residency acceptance gate."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mindie_laser_qresident_checks as checks
import profile_mindie_laser_quant as profile


class Adapter:
    """Two rows model one Q block; groups of four model a Q0/Q1 pair."""
    def __init__(self, *, bug=None, events=None):
        self.bug = bug
        self.events = [] if events is None else events
        self.originals, self.payloads, self.cache = [], [], {}

    def prepare(self, q, k, v):
        payload = SimpleNamespace(q=F.pad(q.float(), (0, 0, 0, 4)),
            k=F.pad(k.float(), (0, 0, 0, 4)), v=F.pad(v.float(), (0, 0, 0, 4)),
            q_scale=q.float().abs().amax().reshape(1),
            k_scale=k.float().abs().amax().reshape(1))
        self.originals.append((payload, {name: getattr(payload, name).clone() for name in checks.FIELDS}))
        self.events.append(("prepare",))
        return payload

    def run_native(self, payload):
        self.payloads.append(payload)
        self.events.append(("native", self, payload,
                            {name: getattr(payload, name).clone() for name in checks.FIELDS}))
        q = payload.q
        if self.bug == "swapped_pair":
            q = q.reshape(*q.shape[:2], -1, 2, 2, q.shape[-1]).flip(-3).reshape_as(q)
        elif self.bug == "stale_descriptor":
            q = q[:, :, :4].repeat(1, 1, q.shape[2] // 4, 1)
        elif self.bug == "stale_after_pv":
            q = q.clone()
            # Poison the second pair, modeling PV clobbering L0A before Q is
            # incorrectly reused without loading its new descriptor.
            q[:, :, 4:8] = -17
        elif self.bug == "stale_invocation":
            if id(payload) not in self.cache:
                self.cache[id(payload)] = q.clone()
            q = self.cache[id(payload)]
        output = q + payload.k.mean() + payload.v.mean() + payload.q_scale * payload.k_scale
        if self.bug == "padded_tail":
            output[..., -1, -1] += 1
        if self.bug == "nonfinite":
            output[..., -1, -1] = float("nan")
        if self.bug == "wrong_dtype":
            output = output.half()
        return output


def tiny_inputs(case, dtype, device, revision):
    del case
    base = torch.arange(16).view(1, 1, 8, 2).float()
    return tuple((base * (revision + 1) + index + revision).to(dtype).to(device)
                 for index in range(3))


def test_shapes_cover_q_pairs_sections_pv_and_descriptor_switches():
    cases = checks.Q_RESIDENT_CASES
    assert len(cases) == 6
    assert {257, 4095, 4096, 4097, 8193} <= {case["keys"] for case in cases}
    assert {case["dim"] for case in cases} == {64, 128}
    assert all(case["queries"] >= 257 and case["queries"] % 256 for case in cases)
    assert any(case["queries"] > 512 and case["keys"] > 8192 for case in cases)
    switch = next(case for case in cases if case["name"] == "descriptor_head_batch_switch")
    assert switch["batch"] * switch["heads"] * ((switch["queries"] + 255) // 256) > 64


def test_generated_q_pairs_are_distinct_and_refresh_changes_both_slots():
    case = dict(batch=2, heads=3, queries=513, keys=4097, dim=64)
    for dtype in (torch.float16, torch.bfloat16):
        old = checks._make_inputs(case, dtype, "cpu", 0)
        new = checks._make_inputs(case, dtype, "cpu", 1)
        assert all(t.dtype == dtype and torch.isfinite(t).all() for t in (*old, *new))
        assert all(not torch.equal(a, b) for a, b in zip(old, new))
        q = old[0]
        assert torch.all(q[:, :, :128, 0] == .75)
        assert torch.all(q[:, :, 128:256, 0] == -1.25)
        assert torch.equal(new[0][..., 0], -q[..., 0])
        assert not torch.equal(q[:, :, :128], q[:, :, 128:256])
        assert not torch.equal(q[:, :, :256], q[:, :, 256:512])
        assert not torch.equal(q[0, 0, :, 1], q[1, 0, :, 1])
        assert not torch.equal(q[0, 0, :, 1], q[0, 1, :, 1])
        assert q[:, :, :128].abs().max() < q[:, :, 128:256].abs().max()


def test_same_payload_queued_calls_all_rows_refresh_and_originals_preserved(monkeypatch):
    monkeypatch.setattr(checks, "_make_inputs", tiny_inputs)
    events = []
    adapters = {name: Adapter(events=events) for name in ("baseline", "reuse")}
    original_compare = profile.compare_all_rows
    def compare(a, b, **kwargs):
        events.append(("compare",))
        return original_compare(a, b, **kwargs)
    monkeypatch.setattr(profile, "compare_all_rows", compare)
    report, snapshots = {}, []
    checks.validate_q_resident(adapters, "cpu", report, lambda: snapshots.append(copy.deepcopy(report)))
    rows = report["q_resident_validation"]
    assert len(rows) == 12
    assert {row["dtype"] for row in rows} == {"torch.float16", "torch.bfloat16"}
    assert "ky!=2" in report["q_resident_validation_scope"]["public_mapping"]
    for case in rows:
        assert case["status"] == "passed" and len(case["rounds"]) == 4
        for iteration, row in enumerate(case["rounds"]):
            assert row["exact_match"] and row["pointers_unchanged"] and row["elements"] == 24
            assert row["order"] == (["baseline", "reuse"] if iteration % 2 == 0 else ["reuse", "baseline"])
            if iteration:
                assert row["baseline_change"]["exact_match"] == (iteration != 2)
    for start in range(0, len(events), 17):
        block = events[start:start + 17]
        assert [event[0] for event in block] == ["prepare"] * 2 + ["native"] * 8 + ["compare"] * 7
        calls = block[2:10]
        assert len({id(call[2]) for call in calls}) == 1
        for a, b in zip(calls[::2], calls[1::2]):
            assert all(torch.equal(a[3][field], b[3][field]) for field in checks.FIELDS)
        for field in checks.FIELDS:
            assert torch.equal(calls[0][3][field], calls[2][3][field])
            assert not torch.equal(calls[0][3][field], calls[4][3][field])
            assert torch.equal(calls[4][3][field], calls[6][3][field])
    for original, saved in adapters["baseline"].originals:
        assert all(torch.equal(getattr(original, field), saved[field]) for field in checks.FIELDS)
    for index, payload in enumerate(adapters["reuse"].payloads):
        saved = adapters["baseline"].originals[(index // 4) * 2][1]
        assert all(torch.equal(getattr(payload, field), saved[field]) for field in checks.FIELDS)
    assert snapshots[-1]["q_resident_validation"][-1]["status"] == "passed"


@pytest.mark.parametrize("bug, iteration, message", [
    ("swapped_pair", 0, "changed padded native outputs"),
    ("stale_descriptor", 0, "changed padded native outputs"),
    ("stale_after_pv", 0, "changed padded native outputs"),
    ("stale_invocation", 2, "changed padded native outputs"),
    ("padded_tail", 0, "changed padded native outputs"),
    ("nonfinite", None, "Non-finite"),
    ("wrong_dtype", None, "entire padded native FP32"),
])
def test_gate_detects_residency_and_output_failures_with_partial_evidence(monkeypatch, bug, iteration, message):
    monkeypatch.setattr(checks, "_make_inputs", tiny_inputs)
    baseline, candidate = Adapter(), Adapter(bug=bug)
    report, snapshots = {}, []
    with pytest.raises((AssertionError, ValueError), match=message):
        checks.validate_q_resident({"baseline": baseline, "reuse": candidate}, "cpu",
                                  report, lambda: snapshots.append(copy.deepcopy(report)))
    failed = snapshots[-1]["q_resident_validation"][-1]
    assert failed["status"] == "failed" and message in failed["error"]
    if iteration is not None:
        assert len(failed["rounds"]) == iteration + 1 and failed["rounds"][-1]["exact_match"] is False
    saved = baseline.originals[0][1]
    assert all(torch.equal(getattr(candidate.payloads[-1], field), saved[field]) for field in checks.FIELDS)


def test_invalid_adapter_set_fails_before_execution():
    with pytest.raises(ValueError, match="baseline and reuse"):
        checks.validate_q_resident({"baseline": Adapter()}, "cpu", {}, lambda: None)
