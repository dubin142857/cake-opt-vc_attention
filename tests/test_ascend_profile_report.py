"""CPU-only report/gate tests; do not exercise or validate Ascend kernels."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from profile_ascend910b import gate, summarize_records, validate_tiles


def records():
    return torch.zeros((2, 3, 4, 16), dtype=torch.int64)


def put(data, core, role, slot, key, count, phases):
    data[core, role, slot, :4] = torch.tensor([1, core * 3, key, count])
    data[core, role, slot, 4:4 + len(phases)] = torch.tensor(phases)


def test_role_tail_and_invalid_separation():
    data = records()
    put(data, 0, 0, 0, 0, 128, [10, 20, 30, 40])
    put(data, 1, 0, 0, 128, 128, [30, 40, 50, 60])
    put(data, 0, 0, 1, 256, 7, [1000, 2000, 3000, 4000])
    put(data, 0, 1, 0, 0, 128, [101, 102, 103, 104, 105, 106, 107, 108])
    put(data, 0, 2, 0, 0, 128, [201, 202, 203, 204, 205, 206, 207, 208])
    # Unwritten slots can have irrelevant contents; validity is authoritative.
    data[1, 2, 3, 1:] = -999
    result = summarize_records(data)
    assert result["valid_samples"] == 5
    assert result["invalid_slots"] == 19
    assert result["roles"]["aic"]["main"]["samples"] == 2
    assert result["roles"]["aic"]["main"]["phases"]["qk_api_complete"]["median_ticks"] == 20
    assert result["roles"]["aic"]["tail"]["phases"]["qk_api_complete"]["median_ticks"] == 1000
    assert result["roles"]["aiv0"]["main"]["phases"]["probability"]["median_ticks"] == 103
    assert result["roles"]["aiv1"]["main"]["phases"]["probability"]["median_ticks"] == 203
    assert result["roles"]["aiv1"]["tail"] == {"samples": 0, "phases": {}}
    assert result["unit"] == "GetSystemCycle system ticks"


def test_zero_elapsed_is_valid_and_empty_slots_produce_no_fake_phases():
    data = records()
    empty = summarize_records(data)
    assert empty["valid_samples"] == 0
    assert all(not group["phases"] for role in empty["roles"].values() for group in role.values())
    put(data, 0, 1, 0, 0, 1, [0] * 8)
    assert summarize_records(data)["roles"]["aiv0"]["tail"]["phases"]["probability"]["median_ticks"] == 0


@pytest.mark.parametrize("column,value", [(0, 2), (1, -1), (2, 1), (3, 0), (3, 129), (4, -1)])
def test_corrupt_valid_records_are_rejected(column, value):
    data = records()
    put(data, 0, 0, 0, 0, 128, [1] * 4)
    data[0, 0, 0, column] = value
    with pytest.raises(ValueError):
        summarize_records(data)


def test_schema_and_integer_ticks_are_required():
    with pytest.raises(ValueError, match="cores,3,4,16"):
        summarize_records(torch.zeros((2, 3, 16), dtype=torch.int64))
    with pytest.raises(TypeError, match="int64"):
        summarize_records(records().float())


def test_gate_preserves_evidence_before_failure():
    row, snapshots = {}, []
    with pytest.raises(AssertionError, match="exceeds 0.1%"):
        gate(torch.full((2, 3), 1.01), torch.ones(2, 3), limit=.001,
             label="same_payload", row=row, save=lambda: snapshots.append(dict(row)))
    assert snapshots[-1]["same_payload"]["relative_rmse"] > .001
    metric = gate(torch.ones(2, 3), torch.ones(2, 3), limit=.001,
                  label="exact", row=row, save=lambda: None)
    assert metric == {"relative_rmse": 0.0, "max_abs": 0.0}


def test_nonfinite_output_is_not_accepted():
    with pytest.raises(AssertionError, match="Nonfinite"):
        gate(torch.tensor([float("nan")]), torch.ones(1), limit=.001,
             label="invalid", row={}, save=lambda: None)


def test_shape_mismatch_is_not_silently_broadcast():
    row = {}
    with pytest.raises(AssertionError, match="shape mismatch"):
        gate(torch.ones(1, 1), torch.ones(2, 3), limit=.001,
             label="invalid", row=row, save=lambda: None)
    assert row["invalid"]["actual_shape"] == [1, 1]


def test_cpu_fake_small_suite_checks_probe_shapes_and_repeats():
    calls, probes = [], []

    class FakeVC:
        @staticmethod
        def prepare(q, k, v, *, layout):
            assert layout == "HND"
            return {"q": q, "dtype": q.dtype}

        @staticmethod
        def run_prepared(p, *, expcast, tile_m):
            calls.append((p["q"].shape, p["q"].dtype, expcast, tile_m))
            return p["q"].clone()

        @staticmethod
        def profile_prepared(p, *, expcast, tile_m):
            probes.append((p["q"].shape, p["q"].dtype, expcast, tile_m))
            data = records()
            put(data, 0, 0, 0, 0, 1, [1] * 4)
            return p["q"].clone(), data

    report = {}
    validate_tiles(FakeVC(), lambda p, **kwargs: p["q"], "cpu", report, lambda: None)
    assert len(report["small_validation"]) == 40
    assert all(row["status"] == "passed" for row in report["small_validation"])
    assert len(calls) == 160  # Each case checks both tile sizes twice.
    assert len(probes) == 32
    assert {shape[2] for shape, *_ in probes} == {1, 65}
    assert {shape[-1] for shape, *_ in probes} == {64, 128}
    assert {dtype for _, dtype, *_ in probes} == {torch.float16, torch.bfloat16}
    assert {tile for *_, tile in probes} == {32, 64}
