"""CPU tests of the remote exact-output/mutable-storage cache candidate gates."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mindie_laser_cache_checks as checks
import profile_mindie_laser_quant as profile


class Adapter:
    """Tiny deterministic native substitute; every Q/K scale affects output."""
    def __init__(self, *, bug=None, events=None):
        self.bug = bug
        self.events = events if events is not None else []
        self.cached = {}
        self.original = []
        self.payloads = []

    def prepare(self, q, k, v):
        batch, heads, nq, _ = q.shape
        nk = k.shape[2]
        pq, pk = (nq + 255) // 256 * 256, (nk + 255) // 256 * 256
        payload = SimpleNamespace(q=torch.zeros(batch, heads, pq, 3, dtype=torch.int8),
            k=torch.zeros(batch, heads, pk, 3, dtype=torch.int8),
            v=torch.zeros(batch, heads, pk, 3, dtype=torch.float16),
            q_scale=torch.full((batch, heads, pq // 128), .02),
            k_scale=torch.full((batch, heads, pk // 128), .03))
        self.original.append(payload)
        self.events.append(("prepare", payload))
        return payload

    def run_native(self, payload):
        self.payloads.append(payload)
        self.events.append(("native", self, payload, payload.q_scale.clone(), payload.k_scale.clone()))
        q, k = payload.q_scale, payload.k_scale
        if self.bug in ("stale_q", "stale_k", "stale_both"):
            key = (q.data_ptr(), k.data_ptr())
            if key not in self.cached:
                self.cached[key] = (q.clone(), k.clone())
            cq, ck = self.cached[key]
            if self.bug in ("stale_q", "stale_both"): q = cq
            if self.bug in ("stale_k", "stale_both"): k = ck
        if self.bug == "wrong_q": q = q.roll(1, dims=-1)
        if self.bug == "wrong_k": k = k.roll(1, dims=-1)
        weights = torch.arange(1, k.shape[-1] + 1, dtype=torch.float32)
        value = (q * (k * weights).sum(-1, keepdim=True)).repeat_interleave(128, -1)
        output = value.unsqueeze(-1).expand_as(payload.q).clone()
        if self.bug == "padded_tail": output[..., -1, -1] += 1
        if self.bug == "wrong_dtype": output = output.half()
        return output


@pytest.fixture(autouse=True)
def lightweight_original_inputs(monkeypatch):
    original = torch.randn
    monkeypatch.setattr(torch, "randn", lambda shape, **kw: original((*shape[:3], 1), **kw))


def test_offset_views_keep_owner_contiguity_values_and_real_addresses():
    source = torch.arange(102, dtype=torch.float32).view(1, 3, 34)
    for offset in (1, 15, 16):
        view = checks.offset_scale(source, offset)
        assert view.is_contiguous() and view.storage_offset() == offset
        assert torch.equal(view, source)
        assert view.data_ptr() == view.untyped_storage().data_ptr() + 4 * offset
        location = checks.scale_location(view)
        assert location["address_mod_64"] == view.data_ptr() % 64
        assert location["blocks_per_head"] == 34
        assert location["head_start_address_mod_64"] == [
            (view.data_ptr() + h * 34 * 4) % 64 for h in range(3)]
        assert len(set(location["head_start_address_mod_64"])) == 3
        view.fill_(.25)
        assert torch.equal(source, torch.arange(102).view(1, 3, 34).float())
    with pytest.raises(ValueError, match="FP32"):
        checks.offset_scale(source.half(), 1)
    with pytest.raises(ValueError, match="offsets"):
        checks.offset_scale(source, 0)


def test_gate_queues_both_orders_before_cpu_comparisons_and_preserves_originals(monkeypatch):
    events = []
    baseline, candidate = Adapter(events=events), Adapter(events=events)
    compare = profile.compare_all_rows
    def recorded_compare(actual, reference, **kwargs):
        events.append(("compare",))
        return compare(actual, reference, **kwargs)
    monkeypatch.setattr(profile, "compare_all_rows", recorded_compare)
    report = {}
    saves = []
    checks.validate_scale_cache({"baseline": baseline, "reuse": candidate}, "cpu",
                                report, lambda: saves.append(len(events)))
    cases = report["scale_cache_storage"]
    assert len(cases) == 6
    assert {row["dtype"] for row in cases} == {"torch.float16", "torch.bfloat16"}
    assert {row["keys"] for row in cases} == {129, 4097}
    for field in checks.SCALE_FIELDS:
        assert {row["scales"][field]["storage_offset_elements"] for row in cases} == {1, 15, 16}
    assert any(row["shape_bhqd"][:2] == [2, 3] for row in cases)
    assert any(row["scales"]["k_scale"]["blocks_per_head"] == 34 for row in cases)
    for case in cases:
        assert case["status"] == "passed" and len(case["rounds"]) == 4
        for index, row in enumerate(case["rounds"]):
            assert row["exact_match"] and row["pointers_unchanged"]
            assert row["order"] == (["reuse", "baseline"] if index % 2 == 0 else ["baseline", "reuse"])
            if index:
                assert row["baseline_change_after_rewrite"]["exact_match"] is False

    boundaries = [i for i, event in enumerate(events) if event[0] == "prepare"] + [len(events)]
    for start, end in zip(boundaries, boundaries[1:]):
        batch = events[start + 1:end]
        # ALL 4 x 2 launches happen before the first CPU comparison.
        assert [event[0] for event in batch[:8]] == ["native"] * 8
        assert all(event[0] == "compare" for event in batch[8:])
        for first, second in zip(batch[:8:2], batch[1:8:2]):
            assert first[2] is second[2]
            assert torch.equal(first[3], second[3]) and torch.equal(first[4], second[4])
        assert len({id(event[2]) for event in batch[:8]}) == 1
    for original in baseline.original:
        assert torch.all(original.q_scale == .02) and torch.all(original.k_scale == .03)
    for restored in candidate.payloads:
        assert torch.all(restored.q_scale == .02) and torch.all(restored.k_scale == .03)
    assert saves


@pytest.mark.parametrize("bug", ["stale_q", "stale_k", "wrong_q", "wrong_k", "padded_tail"])
def test_gate_rejects_stale_cache_wrong_scale_selection_and_padded_row_corruption(bug):
    baseline, candidate = Adapter(), Adapter(bug=bug)
    report = {}
    with pytest.raises(AssertionError, match="changed padded native outputs"):
        checks.validate_scale_cache({"baseline": baseline, "reuse": candidate}, "cpu", report, lambda: None)
    assert report["scale_cache_storage"][0]["rounds"][-1]["exact_match"] is False
    # Failure cleanup must also preserve original scales and restore isolated views.
    for restored in candidate.payloads:
        assert torch.all(restored.q_scale == .02) and torch.all(restored.k_scale == .03)


def test_gate_rejects_vacuous_both_stale_and_wrong_output_contract():
    with pytest.raises(AssertionError, match="ineffective"):
        checks.validate_scale_cache({name: Adapter(bug="stale_both") for name in ("baseline", "reuse")},
                                    "cpu", {}, lambda: None)
    with pytest.raises(AssertionError, match="whole padded native FP32"):
        checks.validate_scale_cache({"baseline": Adapter(), "reuse": Adapter(bug="wrong_dtype")},
                                    "cpu", {}, lambda: None)
    with pytest.raises(ValueError, match="baseline and reuse"):
        checks.validate_scale_cache({"baseline": Adapter()}, "cpu", {}, lambda: None)
