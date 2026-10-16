"""Timing protocol and paired summaries without importing Torch or requiring NPU."""
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mindie_laser_quant_pairing as pairing


def test_module_does_not_import_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    spec = importlib.util.spec_from_file_location("isolated_pairing", pairing.__file__)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_timing_excludes_warmups_callbacks_and_pre_sync_but_includes_post_sync():
    elapsed, events, snapshots = [0.0], [], []

    def call(name, seconds):
        events.append(name)
        elapsed[0] += seconds

    def synchronize():
        events.append("sync")
        elapsed[0] += 0.002

    def clock():
        events.append("clock")
        return elapsed[0]

    def on_trial(report):
        events.append("callback")
        snapshots.append(copy.deepcopy(report))
        elapsed[0] += 10.0

    report = pairing.run_paired(
        {"a": lambda: call("a", 0.01), "b": lambda: call("b", 0.02)},
        synchronize=synchronize, warmup=2, repeats=2, trials=3,
        clock=clock, on_trial=on_trial,
    )
    assert events[:5] == ["a", "a", "b", "b", "sync"]
    expected = events[:5]
    for order in report["orders"]:
        for name in order:
            expected += ["sync", "clock", name, name, "sync", "clock"]
        expected.append("callback")
    assert events == expected
    assert report["rows"]["a"]["trials_ms"] == pytest.approx([11.0] * 3)
    assert report["rows"]["b"]["trials_ms"] == pytest.approx([21.0] * 3)
    assert report["rows"]["a"]["median_ms"] == pytest.approx(11.0)
    assert report["rows"]["b"]["min_ms"] == pytest.approx(21.0)
    assert report["rows"]["b"]["max_ms"] == pytest.approx(21.0)
    assert [row["completed_trials"] for row in snapshots] == [1, 2, 3]
    for index, snapshot in enumerate(snapshots, start=1):
        assert len(snapshot["orders"]) == index
        assert all(len(row["trials_ms"]) == index for row in snapshot["rows"].values())
    json.dumps(report, allow_nan=False)


def test_outputs_released_before_next_call_and_before_post_sync():
    live, counts = [], Counter()

    class Output:
        def __init__(self):
            assert not live
            live.append(True)
            counts["created"] += 1

        def __del__(self):
            live.pop()
            counts["deleted"] += 1

    def synchronize():
        assert not live

    clock = iter(range(100))
    pairing.run_paired(
        {"first": Output, "second": Output}, synchronize=synchronize,
        warmup=2, repeats=3, trials=4, clock=lambda: next(clock),
    )
    assert not live
    assert counts == {"created": 28, "deleted": 28}


def _run_orders(names, trials, seed):
    clock = iter(range(1000))
    return pairing.run_paired(
        {name: lambda: None for name in names}, synchronize=lambda: None,
        warmup=0, repeats=1, trials=trials, seed=seed, clock=lambda: next(clock),
    )["orders"]


@pytest.mark.parametrize("count,trials", [(2, 9), (4, 9), (4, 12), (4, 3)])
def test_seeded_orders_balance_each_position_and_do_not_touch_global_rng(count, trials):
    names = [f"call_{index}" for index in range(count)]
    state = random.getstate()
    orders = _run_orders(names, trials, 2026)
    assert random.getstate() == state
    assert orders == _run_orders(names, trials, 2026)
    assert orders != _run_orders(names, trials, 2027)
    assert all(sorted(order) == names for order in orders)
    for position in range(count):
        counts = Counter(order[position] for order in orders)
        values = [counts[name] for name in names]
        assert max(values) - min(values) <= 1
    for start in range(0, trials - count + 1, count):
        for position in range(count):
            assert {order[position] for order in orders[start:start + count]} == set(names)


@pytest.mark.parametrize("keyword,value", [
    ("warmup", -1), ("warmup", True), ("warmup", 1.5),
    ("repeats", 0), ("repeats", -1), ("repeats", True),
    ("trials", 0), ("trials", -1), ("trials", 1.5),
    ("seed", True), ("seed", 1.5), ("synchronize", None),
    ("clock", None), ("on_trial", False),
])
def test_invalid_arguments_fail_before_any_call(keyword, value):
    events = []
    options = {"synchronize": lambda: events.append("sync"), keyword: value}
    with pytest.raises(ValueError):
        pairing.run_paired({"a": lambda: events.append("call")}, **options)
    assert events == []


@pytest.mark.parametrize("calls", [{}, [], {"": lambda: None}, {1: lambda: None}, {"a": None}])
def test_invalid_call_mapping_fails_before_synchronization(calls):
    events = []
    with pytest.raises(ValueError):
        pairing.run_paired(calls, synchronize=lambda: events.append("sync"))
    assert events == []


@pytest.mark.parametrize("timestamps", [
    [0.0, float("nan")], [float("nan"), 1.0], [0.0, float("inf")],
    [0.0, -1.0], [0.0, 0.0], [float("-inf"), 1.0],
])
def test_bad_clock_measurements_are_rejected(timestamps):
    clock = iter(timestamps)
    with pytest.raises(ValueError, match="finite and positive"):
        pairing.run_paired(
            {"a": lambda: None}, synchronize=lambda: None,
            warmup=0, repeats=1, trials=1, clock=lambda: next(clock),
        )


def _report(old, new):
    return {
        "orders": [["old", "new"] for _ in old],
        "rows": {"old": {"trials_ms": old}, "new": {"trials_ms": new}},
    }


def test_comparison_uses_trial_pairs_not_difference_or_ratio_of_medians():
    report = _report([1.0, 20.0, 10.0], [2.0, 10.0, 9.0])
    result = pairing.paired_comparison(report, "old", "new")
    assert result["paired_saved_ms"] == [-1.0, 10.0, 1.0]
    assert result["median_saved_ms"] == 1.0
    assert result["min_saved_ms"] == -1.0
    assert result["max_saved_ms"] == 10.0
    assert result["paired_speedups"] == pytest.approx([0.5, 2.0, 10 / 9])
    assert result["median_speedup"] == pytest.approx(10 / 9)
    assert (result["faster_trials"], result["slower_trials"], result["tied_trials"]) == (2, 1, 0)
    assert "not confidence intervals" in result["caveat"]
    json.dumps(result, allow_nan=False)

    # Here neither the paired median delta nor the paired median ratio equals
    # the corresponding calculation from the two independent medians.
    result = pairing.paired_comparison(_report([1.0, 10.0, 20.0], [2.0, 100.0, 19.0]), "old", "new")
    assert result["median_saved_ms"] == -1.0
    assert result["median_saved_ms"] != result["baseline_median_ms"] - result["candidate_median_ms"]
    assert result["median_speedup"] == 0.5
    assert result["median_speedup"] != result["baseline_median_ms"] / result["candidate_median_ms"]


def test_comparison_counts_exact_ties():
    result = pairing.paired_comparison(_report([2.0], [2.0]), "old", "new")
    assert (result["faster_trials"], result["slower_trials"], result["tied_trials"]) == (0, 0, 1)
    assert result["median_speedup"] == 1.0


@pytest.mark.parametrize("old,new", [
    ([], []), ([1.0], []), ([float("nan")], [1.0]), ([1.0], [float("inf")]),
    ([0.0], [1.0]), ([1.0], [-1.0]), ([1e308], [1e-308]),
])
def test_comparison_rejects_invalid_or_unpaired_trials(old, new):
    with pytest.raises(ValueError):
        pairing.paired_comparison(_report(old, new), "old", "new")


@pytest.mark.parametrize("orders", [[], [["old"]], [["old", "new", "new"]]])
def test_comparison_requires_one_observation_per_call_in_each_trial(orders):
    report = _report([2.0], [1.0])
    report["orders"] = orders
    with pytest.raises(ValueError, match="record both calls exactly once"):
        pairing.paired_comparison(report, "old", "new")


def test_comparison_requires_distinct_names():
    with pytest.raises(ValueError, match="different calls"):
        pairing.paired_comparison(_report([1.0], [1.0]), "old", "old")
