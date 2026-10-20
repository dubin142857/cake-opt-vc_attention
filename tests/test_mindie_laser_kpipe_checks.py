"""Exercise the remote K-lookahead gates without an NPU."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mindie_laser_kpipe_checks as checks
import profile_mindie_laser_quant as profile


class Adapter:
    def __init__(self, *, bug=None, events=None):
        self.bug = bug
        self.events = [] if events is None else events
        self.originals, self.payloads, self.cache = [], [], {}

    def prepare(self, q, k, v):
        payload = SimpleNamespace(q=F.pad(q.float(), (0, 0, 0, 3)),
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
        key = payload.k.roll(1, dims=2) if self.bug == "wrong_tile" else payload.k
        weights = torch.arange(1, key.shape[2] + 1).view(1, 1, -1, 1)
        output = payload.q + (key * weights).mean() + payload.v.mean() + payload.q_scale * payload.k_scale
        if self.bug == "stale":
            if id(payload) not in self.cache:
                self.cache[id(payload)] = output.clone()
            output = self.cache[id(payload)].clone()
        if self.bug == "padded_tail":
            output[..., -1, -1] += 1
        if self.bug == "nonfinite":
            output[..., -1, -1] = float("nan")
        if self.bug == "wrong_dtype":
            output = output.half()
        return output


def tiny_inputs(case, dtype, device, revision):
    del case
    data = torch.arange(8).view(1, 1, 4, 2).float()
    return tuple((data * (revision + 1) + index + revision).to(dtype).to(device)
                 for index in range(3))


def test_public_shape_coverage_has_boundaries_batch_and_descriptor_switch():
    cases = checks.K_PREFETCH_CASES
    assert len(cases) == 8
    assert {127, 129, 257, 4095, 4096, 4097, 8193} <= {case["keys"] for case in cases}
    assert {case["dim"] for case in cases} == {64, 128}
    assert any(case["batch"] > 1 and case["heads"] > 1 for case in cases)
    assert all(case["queries"] % 256 for case in cases)
    switch = next(case for case in cases if case["name"] == "descriptor_head_batch_switch")
    q_groups = switch["batch"] * switch["heads"] * ((switch["queries"] + 255) // 256)
    assert q_groups > 64
    assert ((switch["keys"] + 255) // 256 * 2) == 8


def test_marker_inputs_distinguish_adjacent_k_tiles_and_scale_ranges():
    case = dict(batch=2, heads=3, queries=257, keys=4097, dim=64)
    for dtype in (torch.float16, torch.bfloat16):
        old = checks._make_inputs(case, dtype, "cpu", 0)
        new = checks._make_inputs(case, dtype, "cpu", 1)
        assert all(t.dtype == dtype and torch.isfinite(t).all() for t in (*old, *new))
        assert all(not torch.equal(before, after) for before, after in zip(old, new))
        k = old[1]
        assert k[0, 0, 0, 0] == -1
        assert k[0, 0, 128, 0] == -.375
        assert k[0, 0, 256, 0] == .25
        assert not torch.equal(k[:, :, :128], k[:, :, 128:256])
        # First four full blocks have increasing programmed ranges, reflected
        # in these deterministic inputs even after FP16/BF16 rounding.
        maximum = [k[0, 0, block * 128:(block + 1) * 128].abs().max().item()
                   for block in range(4)]
        assert maximum == sorted(maximum) and len(set(maximum)) == 4
        assert not torch.equal(old[2][:, :, :128], old[2][:, :, 128:256])


def test_queue_then_compare_same_payload_both_orders_and_preserved_originals(monkeypatch):
    monkeypatch.setattr(checks, "_make_inputs", tiny_inputs)
    events = []
    adapters = {name: Adapter(events=events) for name in ("baseline", "reuse")}
    original_compare = profile.compare_all_rows
    def compare(a, b, **kwargs):
        events.append(("compare",))
        return original_compare(a, b, **kwargs)
    monkeypatch.setattr(profile, "compare_all_rows", compare)
    report, saves = {}, []
    checks.validate_k_prefetch(adapters, "cpu", report, lambda: saves.append(copy.deepcopy(report)))
    cases = report["k_prefetch_validation"]
    assert len(cases) == len(checks.K_PREFETCH_CASES) * 2
    assert {case["dtype"] for case in cases} == {"torch.float16", "torch.bfloat16"}
    assert "odd slot parity" in report["k_prefetch_validation_scope"]["parity"]
    assert report["k_prefetch_validation_scope"]["queued_native_calls_before_cpu_comparison"] == 8
    for case in cases:
        assert case["status"] == "passed" and len(case["rounds"]) == 4
        for iteration, row in enumerate(case["rounds"]):
            assert row["exact_match"] and row["pointers_unchanged"] and row["elements"] == 14
            assert row["order"] == (["baseline", "reuse"] if iteration % 2 == 0 else ["reuse", "baseline"])
            if iteration:
                assert row["baseline_change"]["exact_match"] == (iteration != 2)

    # Each case has two preparations, then eight launches, then seven CPU
    # comparisons. All output checks happen after the same-pointer refresh.
    for start in range(0, len(events), 17):
        block = events[start:start + 17]
        assert [e[0] for e in block] == ["prepare"] * 2 + ["native"] * 8 + ["compare"] * 7
        native = block[2:10]
        assert len({id(e[2]) for e in native}) == 1
        for first, second in zip(native[::2], native[1::2]):
            assert first[2] is second[2]
            assert all(torch.equal(first[3][name], second[3][name]) for name in checks.FIELDS)
        assert not torch.equal(native[0][3]["k"], native[4][3]["k"])
        assert torch.equal(native[0][3]["k"], native[2][3]["k"])
        assert torch.equal(native[4][3]["k"], native[6][3]["k"])
    for original, snapshot in adapters["baseline"].originals:
        assert all(torch.equal(getattr(original, name), snapshot[name]) for name in checks.FIELDS)
    for index, payload in enumerate(adapters["reuse"].payloads):
        snapshot = adapters["baseline"].originals[(index // 4) * 2][1]
        assert all(torch.equal(getattr(payload, name), snapshot[name]) for name in checks.FIELDS)
    assert saves[-1]["k_prefetch_validation"][-1]["status"] == "passed"


@pytest.mark.parametrize("bug, iteration, error", [
    ("wrong_tile", 0, "changed padded native outputs"),
    ("padded_tail", 0, "changed padded native outputs"),
    ("stale", 2, "changed padded native outputs"),
    ("nonfinite", None, "Non-finite"),
    ("wrong_dtype", None, "entire padded native FP32"),
])
def test_wrong_tile_padding_stale_or_invalid_output_fails_with_saved_evidence(monkeypatch, bug, iteration, error):
    monkeypatch.setattr(checks, "_make_inputs", tiny_inputs)
    baseline, candidate = Adapter(), Adapter(bug=bug)
    report, snapshots = {}, []
    with pytest.raises((AssertionError, ValueError), match=error):
        checks.validate_k_prefetch({"baseline": baseline, "reuse": candidate}, "cpu",
                                  report, lambda: snapshots.append(copy.deepcopy(report)))
    failed = snapshots[-1]["k_prefetch_validation"][-1]
    assert failed["status"] == "failed" and error in failed["error"]
    if iteration is not None:
        assert len(failed["rounds"]) == iteration + 1
        assert failed["rounds"][-1]["exact_match"] is False
    initial = baseline.originals[0][1]
    assert all(torch.equal(getattr(candidate.payloads[-1], name), initial[name]) for name in checks.FIELDS)


def test_baselines_must_respond_to_refresh_and_adapters_are_explicit(monkeypatch):
    monkeypatch.setattr(checks, "_make_inputs", tiny_inputs)
    with pytest.raises(AssertionError, match="ineffective"):
        checks.validate_k_prefetch({name: Adapter(bug="stale") for name in ("baseline", "reuse")},
                                  "cpu", {}, lambda: None)
    with pytest.raises(ValueError, match="baseline and reuse"):
        checks.validate_k_prefetch({"baseline": Adapter()}, "cpu", {}, lambda: None)
