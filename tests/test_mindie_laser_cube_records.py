"""Cube diagnostics must preserve work boundaries and reject corrupt records."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mindie_laser_cube_records as probe


def empty_trace():
    return [[[[0] * 24 for _ in range(3)] for _ in range(2)] for _ in range(4)]


def record(*, core=0, stage=0, position=0, mode=2, total=10, **updates):
    row = dict.fromkeys(probe.FIELD_NAMES, 0)
    row.update(magic=probe.MAGIC, version=1, mode=mode, core=core, stage=stage,
               position=position, total_rounds=total,
               round=(0, total // 2, total - 1)[position],
               descriptors=2, k_blocks=32, logical_tiles=64,
               record_start=100, issue_end=150, record_end=180,
               issue_ticks=50, completion_ticks=30, elapsed_ticks=80,
               empty_ticks=3, pre_drain_ticks=25, valid_mask=3)
    if stage == 1:
        row.update(history_calls=1, normalize_calls=1)
    if mode == 1:
        row.update(record_end=150, completion_ticks=0, elapsed_ticks=50,
                   empty_ticks=0, pre_drain_ticks=0, valid_mask=1)
    row.update(updates)
    return row


def put(trace, row, *, location=None):
    core, stage, position = location or (row["core"], row["stage"], row["position"])
    trace[core][stage][position] = [row[field] for field in probe.FIELD_NAMES]
    return trace


def decode_one(**kwargs):
    row = record(**kwargs)
    return probe.decode_trace(put(empty_trace(), row), kwargs.get("mode", 2))


def test_zero_trace_is_allowed_and_presence_gate_does_not_require_every_core_or_stage():
    for mode in range(3):
        assert probe.decode_trace(empty_trace(), mode) == []
    for mode in (1, 2):
        with pytest.raises(ValueError, match="required Cube group"):
            probe.decode_trace(empty_trace(), mode, require_records=True)
    with pytest.raises(ValueError, match="incompatible"):
        probe.decode_trace(empty_trace(), 0, require_records=True)
    # A short shape may only exercise one physical core and one sample slot.
    trace = put(empty_trace(), record(total=1, stage=1))
    rows = probe.decode_trace(trace, 2, require_records=True)
    assert len(rows) == 1
    assert rows[0]["stage_name"] == "PV"
    assert rows[0]["sample_position"] == "first"
    assert rows[0]["mode_name"] == "completed"


def test_tensor_like_is_copied_to_cpu_without_importing_torch():
    calls = []

    class TensorLike:
        def detach(self):
            calls.append("detach")
            return self

        def cpu(self):
            calls.append("cpu")
            return self

        def tolist(self):
            calls.append("tolist")
            return put(empty_trace(), record(mode=1))

    assert len(probe.decode_trace(TensorLike(), 1)) == 1
    assert calls == ["detach", "cpu", "tolist"]


def test_duplicate_round_targets_use_lowest_position():
    for total, allowed in ((1, (0,)), (2, (0, 1)), (3, (0, 1, 2)), (10, (0, 1, 2))):
        for position in allowed:
            assert len(decode_one(total=total, position=position)) == 1
        for position in set(range(3)) - set(allowed):
            with pytest.raises(ValueError, match="first-match sampling"):
                decode_one(total=total, position=position)


@pytest.mark.parametrize("change,match", [
    ({"magic": 1}, "magic/version"),
    ({"version": 2}, "magic/version"),
    ({"mode": 1}, "storage/mode"),
    ({"reserved0": 1}, "reserved"),
    ({"reserved1": 1}, "reserved"),
    ({"total_rounds": 0}, "outside total_rounds"),
    ({"round": 10}, "outside total_rounds"),
    ({"round": 1}, "first-match sampling"),
    ({"descriptors": 0}, "positive descriptors"),
    ({"k_blocks": 0}, "positive descriptors"),
    ({"logical_tiles": 0}, "positive descriptors"),
    ({"history_calls": 1}, "QK group"),
    ({"normalize_calls": 1}, "QK group"),
    ({"record_start": 151}, "timestamp order"),
    ({"record_end": 149}, "timestamp order"),
    ({"issue_ticks": 49}, "issue ticks"),
    ({"completion_ticks": 31}, "completion ticks"),
    ({"elapsed_ticks": 81}, "elapsed ticks"),
    ({"valid_mask": 1}, "valid mask"),
    ({"valid_mask": 7}, "valid mask"),
])
def test_malformed_completed_records_rejected(change, match):
    row = record()
    row.update(change)
    with pytest.raises(ValueError, match=match):
        probe.decode_trace(put(empty_trace(), row), 2)


@pytest.mark.parametrize("change", [
    {"completion_ticks": 1, "record_end": 151, "elapsed_ticks": 51},
    {"empty_ticks": 1}, {"pre_drain_ticks": 1},
])
def test_issue_mode_rejects_completion_or_drain_values(change):
    with pytest.raises(ValueError, match="issue-only"):
        decode_one(mode=1, **change)


def test_issue_mode_rejects_completed_mask_and_plain_rejects_any_nonzero_record():
    with pytest.raises(ValueError, match="valid mask"):
        decode_one(mode=1, valid_mask=3)
    with pytest.raises(ValueError, match="all zero"):
        probe.decode_trace(put(empty_trace(), record(mode=0)), 0)


@pytest.mark.parametrize("field", ["history_calls", "normalize_calls"])
def test_pv_helper_calls_cannot_exceed_logical_descriptor_count(field):
    with pytest.raises(ValueError, match="exceed descriptor count"):
        decode_one(stage=1, **{field: 3})


def test_zero_tick_intervals_are_legal_and_predrain_is_outside_record_span():
    rows = decode_one(record_start=100, issue_end=100, record_end=100,
                      issue_ticks=0, completion_ticks=0, elapsed_ticks=0,
                      empty_ticks=999, pre_drain_ticks=1000)
    assert rows[0]["elapsed_ticks"] == 0
    assert rows[0]["pre_drain_ticks"] > rows[0]["elapsed_ticks"]


def test_wrong_storage_location_or_partial_write_rejected():
    for location in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        with pytest.raises(ValueError, match="storage/mode"):
            probe.decode_trace(put(empty_trace(), record(), location=location), 2)
    trace = empty_trace()
    trace[0][0][0][-1] = 1
    with pytest.raises(ValueError, match="partially written"):
        probe.decode_trace(trace, 2)


@pytest.mark.parametrize("value", [True, 1.0, float("nan"), float("inf"), -1, 1 << 63, "1", None])
def test_invalid_scalar_rejected_even_in_unused_slot(value):
    trace = empty_trace()
    trace[3][1][2][-1] = value
    with pytest.raises(ValueError, match="integer|INT64"):
        probe.decode_trace(trace, 2)


@pytest.mark.parametrize("trace", [[], [0] * (4 * 2 * 3 * 24), [[[[]]]], "not a tensor"])
def test_bad_shape_rejected(trace):
    with pytest.raises(ValueError, match="Trace shape"):
        probe.decode_trace(trace, 2)


@pytest.mark.parametrize("mode", [-1, 3, True, 1.0])
def test_bad_mode_rejected(mode):
    with pytest.raises(ValueError):
        probe.decode_trace(empty_trace(), mode)


def test_summary_pools_samples_without_mutation_and_separates_mode_and_stage():
    first = empty_trace()
    put(first, record(core=0, issue_end=140, issue_ticks=40, completion_ticks=40))
    put(first, record(core=1, issue_end=160, issue_ticks=60, completion_ticks=20))
    put(first, record(stage=1))
    second = empty_trace()
    put(second, record(core=2))
    decoded = [probe.decode_trace(first, 2), probe.decode_trace(second, 2), decode_one(mode=1)]
    before = deepcopy(decoded)
    summary = probe.summarize_records(decoded)
    assert decoded == before
    assert summary["trial_count"] == 3 and summary["record_count"] == 5
    assert summary["unit"] == "system_ticks"
    assert "not core clock cycles" in summary["scope"]
    issue, qk, pv = summary["groups"]
    assert issue["mode_name"] == "issue"
    assert set(issue["ticks"]) == {"issue_ticks", "elapsed_ticks"}
    assert qk["cores"] == [0, 1, 2]
    assert qk["samples"] == 3 and qk["trials"] == 2
    assert qk["ticks"]["issue_ticks"] == {"samples": 3, "median": 50, "min": 40, "max": 60}
    assert pv["stage_name"] == "PV" and pv["history_calls"] == 1
    assert pv["trials"] == 1
    assert "total_ticks" not in qk and "percentage" not in qk


@pytest.mark.parametrize("field", probe.WORK_FIELDS)
def test_summary_never_merges_different_logical_work(field):
    first = record(stage=1)
    second = record(core=1, stage=1)
    second[field] += 1
    trace = put(put(empty_trace(), first), second)
    groups = probe.summarize_records([probe.decode_trace(trace, 2)])["groups"]
    assert len(groups) == 2
    assert groups[0][field] != groups[1][field]
    assert all(row["samples"] == 1 for row in groups)


def test_summary_separates_position_and_reports_sparse_or_empty_captures():
    trace = empty_trace()
    for position in range(3):
        put(trace, record(position=position))
    summary = probe.summarize_records([probe.decode_trace(trace, 2), []])
    assert [row["sample_position"] for row in summary["groups"]] == ["first", "middle", "last"]
    assert summary["trial_count"] == 2
    assert all(row["trials"] == 1 for row in summary["groups"])
    assert all(row["total_rounds"] == [10] for row in summary["groups"])
    empty = probe.summarize_records([[], []])
    assert empty["groups"] == [] and empty["record_count"] == 0
    assert empty["trial_count"] == 2
