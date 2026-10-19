"""The diagnostic decoder must reject corrupt traces, not manufacture phases."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mindie_laser_phase_records as probe


def empty_trace():
    return [[[[0] * 24 for _ in range(8)] for _ in range(3)] for _ in range(4)]


def record(*, core=0, role=0, slot=0, mode=2, total=10, **updates):
    row = dict.fromkeys(probe.FIELD_NAMES, 0)
    row.update(magic=probe.MAGIC, version=1, mode=mode, core=core, role=role,
               slot=slot, total_rounds=total, flags=1, record_start=100, record_end=200)
    if slot < 4:
        row.update(kind=1, valid_mask=1, wait_ticks=100,
                   round=(0, 1, total // 2, total - 1)[slot])
    else:
        row.update(kind=2, valid_mask=126, round=0, flags=1 | ((slot - 4) << 1),
                   key_blocks=32, rows=4, empty_ticks=5, score_load_ticks=10,
                   max_history_ticks=20, subtract_ticks=10, exp_ticks=30, sum_ticks=25)
    row.update(updates)
    return row


def put(trace, row, *, location=None):
    core, role, slot = location or (row["core"], row["role"], row["slot"])
    trace[core][role][slot] = [row[field] for field in probe.FIELD_NAMES]
    return trace


def decode_one(**kwargs):
    row = record(**kwargs)
    return probe.decode_trace(put(empty_trace(), row), mode=kwargs.get("mode", 2))


def test_zero_trace_for_all_modes_and_optional_presence_gates():
    for mode in range(3):
        assert probe.decode_trace(empty_trace(), mode) == []
    with pytest.raises(ValueError, match="required cross-core wait"):
        probe.decode_trace(empty_trace(), 1, require_waits=True)
    with pytest.raises(ValueError, match="required Vector phase"):
        probe.decode_trace(empty_trace(), 2, require_phases=True)
    with pytest.raises(ValueError, match="incompatible"):
        probe.decode_trace(empty_trace(), 0, require_waits=True)
    with pytest.raises(ValueError, match="incompatible"):
        probe.decode_trace(empty_trace(), 1, require_phases=True)


def test_short_capture_need_not_contain_all_cores_or_roles():
    rows = decode_one(total=1, role=2)
    assert len(rows) == 1
    assert rows[0]["role_name"] == "AIV1"
    assert rows[0]["kind_name"] == "wait"
    assert rows[0]["section_class"] is None


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

    assert len(probe.decode_trace(TensorLike(), 1, require_waits=True)) == 1
    assert calls == ["detach", "cpu", "tolist"]


def test_wait_timestamp_equality_and_first_match_round_sampling():
    for total, allowed in ((1, (0,)), (2, (0, 1)), (3, (0, 1, 3)), (10, (0, 1, 2, 3))):
        for slot in allowed:
            assert len(decode_one(total=total, slot=slot)) == 1
        for slot in set(range(4)) - set(allowed):
            with pytest.raises(ValueError, match="first-match sampling|outside total_rounds"):
                decode_one(total=total, slot=slot)
    with pytest.raises(ValueError, match="wait ticks do not match"):
        decode_one(wait_ticks=99)


@pytest.mark.parametrize("slot,section_class", enumerate(probe.SECTION_CLASSES, start=4))
def test_phase_section_flags_slots_and_nonzero_line_metadata(slot, section_class):
    rows = decode_one(role=1, slot=slot, lines=4, section=123, key_blocks=33)
    assert rows[0]["section_class"] == section_class
    assert rows[0]["lines"] == 4


def test_phase_components_are_not_required_to_sum_to_span():
    # The enclosing interval includes unmeasured instructions and DMA issue.
    row = decode_one(role=2, slot=4, record_end=220)[0]
    assert sum(row[field] for field in probe.TICK_FIELDS[1:]) != row["record_end"] - row["record_start"]
    # Neither should equality accidentally be imposed by the summary.
    result = probe.summarize_records([[row]])
    assert "total_ticks" not in result["groups"][0]
    assert "wait_ticks" not in result["groups"][0]["ticks"]


@pytest.mark.parametrize("change,match", [
    ({"magic": 1}, "magic/version"),
    ({"version": 2}, "magic/version"),
    ({"mode": 1}, "storage/mode"),
    ({"flags": 8}, "flag"),
    ({"round": 10}, "outside total_rounds"),
    ({"total_rounds": 0}, "outside total_rounds"),
    ({"section": 1 << 32}, "UINT32"),
    ({"record_start": 201}, "timestamp order"),
    ({"valid_mask": 3}, "wait-only mask"),
    ({"kind": 2}, "wait-only mask"),
    ({"exp_ticks": 1}, "unmarked phase"),
])
def test_invalid_wait_records_rejected(change, match):
    row = record()
    row.update(change)
    with pytest.raises(ValueError, match=match):
        probe.decode_trace(put(empty_trace(), row), 2)


@pytest.mark.parametrize("change,match", [
    ({"role": 0}, "AIV role"),
    ({"valid_mask": 62}, "all six phase"),
    ({"kind": 1}, "all six phase"),
    ({"flags": 0}, "must be active"),
    ({"flags": 3}, "section-class slot"),
    ({"key_blocks": 0}, "four rows"),
    ({"rows": 3}, "four rows"),
    ({"wait_ticks": 1}, "unmarked wait"),
    ({"sum_ticks": 101}, "exceeds record span"),
])
def test_invalid_phase_records_rejected(change, match):
    row = record(role=1, slot=4)
    row.update(change)
    with pytest.raises(ValueError, match=match):
        probe.decode_trace(put(empty_trace(), row), 2)


def test_wait_only_mode_rejects_a_phase_record_and_plain_rejects_any_write():
    with pytest.raises(ValueError, match="mode 2"):
        decode_one(role=1, slot=4, mode=1)
    with pytest.raises(ValueError, match="all zero"):
        probe.decode_trace(put(empty_trace(), record(mode=0)), 0)


def test_wrong_storage_location_and_partial_writes_rejected():
    with pytest.raises(ValueError, match="storage/mode"):
        probe.decode_trace(put(empty_trace(), record(), location=(1, 0, 0)), 2)
    trace = empty_trace()
    trace[0][0][0][-1] = 1
    with pytest.raises(ValueError, match="partially written"):
        probe.decode_trace(trace, 2)


@pytest.mark.parametrize("value", [True, 1.0, float("nan"), float("inf"), -1, 1 << 63, "1", None])
def test_invalid_scalar_type_or_range_rejected_even_in_otherwise_empty_slot(value):
    trace = empty_trace()
    trace[3][2][7][0] = value
    with pytest.raises(ValueError, match="integer|INT64"):
        probe.decode_trace(trace, 2)


@pytest.mark.parametrize("trace", [[], [0] * (4 * 3 * 8 * 24), [[[[]]]], "not a tensor"])
def test_shape_rejected(trace):
    with pytest.raises(ValueError, match="Trace shape"):
        probe.decode_trace(trace, 2)


@pytest.mark.parametrize("mode", [-1, 3, True, 1.0])
def test_bad_mode_rejected(mode):
    with pytest.raises(ValueError):
        probe.decode_trace(empty_trace(), mode)


def test_summary_pools_records_and_preserves_role_class_and_sparse_trial_counts():
    first = empty_trace()
    put(first, record(core=0, wait_ticks=10, record_end=110))
    put(first, record(core=1, wait_ticks=30, record_end=130))
    put(first, record(role=1, slot=4, exp_ticks=5))
    second = empty_trace()
    put(second, record(core=2, wait_ticks=20, record_end=120))
    put(second, record(role=2, slot=7, exp_ticks=7))
    decoded = [probe.decode_trace(first, 2), probe.decode_trace(second, 2)]
    before = deepcopy(decoded)
    summary = probe.summarize_records(decoded)
    assert decoded == before
    assert summary["trial_count"] == 2 and summary["record_count"] == 5
    assert summary["unit"] == "system_ticks"
    assert "not core clock cycles" in summary["scope"]
    waits, steady, both = summary["groups"]
    assert waits["cores"] == [0, 1, 2]
    assert waits["samples"] == 3 and waits["trials"] == 2
    assert waits["ticks"] == {"wait_ticks": {"samples": 3, "median": 20, "min": 10, "max": 30}}
    assert steady["section_class"] == "steady" and steady["role_name"] == "AIV0"
    assert both["section_class"] == "head_and_tail" and both["role_name"] == "AIV1"
    assert steady["trials"] == both["trials"] == 1
    assert "wait_ticks" not in steady["ticks"]


def test_empty_summary_is_explicitly_empty():
    summary = probe.summarize_records([[], []])
    assert summary["groups"] == []
    assert summary["record_count"] == 0
    assert summary["trial_count"] == 2


def test_summary_separates_wait_position_and_inactive_work():
    trace = empty_trace()
    put(trace, record(core=0, slot=0, flags=1, wait_ticks=10, record_end=110))
    put(trace, record(core=0, slot=2, flags=1, wait_ticks=80, record_end=180))
    put(trace, record(core=1, slot=2, flags=0, wait_ticks=1, record_end=101))
    groups = probe.summarize_records([probe.decode_trace(trace, 2)])["groups"]
    assert len(groups) == 3
    by_position = {(row["sample_position"], row["active"]): row for row in groups}
    assert set(by_position) == {("first", True), ("middle", True), ("middle", False)}
    assert by_position["first", True]["ticks"]["wait_ticks"]["median"] == 10
    assert by_position["middle", True]["ticks"]["wait_ticks"]["median"] == 80
    assert by_position["middle", False]["ticks"]["wait_ticks"]["median"] == 1
    assert all(row["rows"] is None and row["key_blocks"] is None for row in groups)


def test_summary_separates_phase_work_sizes_with_same_section_class():
    trace = empty_trace()
    put(trace, record(core=0, role=1, slot=4, key_blocks=16, exp_ticks=10))
    put(trace, record(core=1, role=1, slot=4, key_blocks=32, exp_ticks=50))
    put(trace, record(core=2, role=1, slot=4, key_blocks=32, exp_ticks=30))
    groups = probe.summarize_records([probe.decode_trace(trace, 2)])["groups"]
    assert len(groups) == 2
    small, large = groups
    assert small["key_blocks"] == 16 and large["key_blocks"] == 32
    assert small["ticks"]["exp_ticks"]["median"] == 10
    assert large["ticks"]["exp_ticks"]["median"] == 40
    assert small["section_class"] == large["section_class"] == "steady"
    assert small["rows"] == large["rows"] == 4
    assert small["active"] is large["active"] is True
    assert small["cores"] == [0] and large["cores"] == [1, 2]
    assert small["samples"] == 1 and large["samples"] == 2
