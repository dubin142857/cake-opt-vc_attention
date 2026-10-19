"""Decode sparse Cube diagnostics without importing torch or an NPU runtime.

The probe observes logical QK/PV groups, not individual matrix instructions.
Mode 2 adds completion fences and changes overlap; its counter intervals are
not production stage costs, percentages, or a decomposition of API latency.
"""
from __future__ import annotations

from collections import defaultdict
from numbers import Integral
from statistics import median


MAGIC = 0x56434C4350524F42
VERSION = 1
TRACE_SHAPE = (4, 2, 3, 24)
TRACE_BYTES = 4 * 2 * 3 * 24 * 8
CONTRACT = "laser_cube_groups_system_ticks_v1"
FIELD_NAMES = (
    "magic", "version", "mode", "core", "stage", "position", "round",
    "total_rounds", "descriptors", "k_blocks", "logical_tiles",
    "history_calls", "normalize_calls", "record_start", "issue_end",
    "record_end", "issue_ticks", "completion_ticks", "elapsed_ticks",
    "empty_ticks", "pre_drain_ticks", "valid_mask", "reserved0", "reserved1",
)
TICK_FIELDS = (
    "issue_ticks", "completion_ticks", "elapsed_ticks", "empty_ticks",
    "pre_drain_ticks",
)
WORK_FIELDS = (
    "descriptors", "k_blocks", "logical_tiles", "history_calls", "normalize_calls",
)
STAGE_NAMES = ("QK", "PV")
POSITION_NAMES = ("first", "middle", "last")
MODE_NAMES = ("plain", "issue", "completed")
SCOPE = (
    "Sparse logical QK/PV groups on Cube cores 0..3. Issue intervals include "
    "scalar submission and any existing dependencies in the group; they are "
    "not pure matrix instruction issue. Completed mode adds PIPE_ALL drains "
    "and changes overlap. Pre-drain and empty-bracket intervals are separate "
    "diagnostics, not corrections to subtract. System ticks are not core "
    "clock cycles. Do not add groups, extrapolate to kernel milliseconds, or "
    "interpret the samples as production stage percentages. Work metadata "
    "counts logical source operations, not hardware instruction counts."
)


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if not 0 <= value <= (1 << 63) - 1:
        raise ValueError(f"{name} is outside the nonnegative INT64 range")
    return value


def _nested(trace):
    # Copy accelerator tensors before inspecting them. Array-like objects
    # are supported as well, without a dependency on torch or numpy.
    if hasattr(trace, "detach"):
        trace = trace.detach()
    if hasattr(trace, "cpu"):
        trace = trace.cpu()
    if hasattr(trace, "tolist"):
        trace = trace.tolist()

    def visit(value, shape, path):
        if not isinstance(value, (list, tuple)) or len(value) != shape[0]:
            raise ValueError(f"Trace shape must be {TRACE_SHAPE}; invalid dimension at {path}")
        if len(shape) == 1:
            return [_integer(item, f"{path}[{i}]") for i, item in enumerate(value)]
        return [visit(item, shape[1:], f"{path}[{i}]") for i, item in enumerate(value)]

    return visit(trace, TRACE_SHAPE, "trace")


def decode_trace(trace, mode, require_records=False):
    """Validate trace storage and return all complete nonzero records.

    Inactive cores, ineligible logical groups, and duplicate sample positions
    remain zero. ``require_records`` only requires at least one record; it
    does not silently assume that every core or stage must have been active.
    """
    mode = _integer(mode, "mode")
    if mode not in (0, 1, 2):
        raise ValueError("mode must be 0 (plain), 1 (issue), or 2 (completed)")
    if require_records and mode == 0:
        raise ValueError("Required records are incompatible with plain mode")
    data = _nested(trace)
    records = []
    for core, stages in enumerate(data):
        for stage, positions in enumerate(stages):
            for position, values in enumerate(positions):
                if not any(values):
                    continue
                location = f"core {core}, stage {stage}, position {position}"
                if mode == 0:
                    raise ValueError(f"Plain mode trace must be all zero ({location})")
                record = dict(zip(FIELD_NAMES, values))

                def check(condition, reason):
                    if not condition:
                        raise ValueError(f"Invalid Laser Cube record at {location}: {reason}")

                check(record["magic"] == MAGIC and record["version"] == VERSION,
                      "magic/version mismatch or partially written record")
                check((record["core"], record["stage"], record["position"], record["mode"])
                      == (core, stage, position, mode), "metadata does not match storage/mode")
                check(record["reserved0"] == record["reserved1"] == 0,
                      "reserved fields must be zero")
                check(record["total_rounds"] > 0 and record["round"] < record["total_rounds"],
                      "round is outside total_rounds")
                targets = (0, record["total_rounds"] // 2, record["total_rounds"] - 1)
                check(record["round"] == targets[position]
                      and targets.index(record["round"]) == position,
                      "round/position violates first-match sampling")
                check(all(record[field] > 0 for field in WORK_FIELDS[:3]),
                      "eligible groups require positive descriptors, k_blocks and logical_tiles")
                check(stage != 0 or record["history_calls"] == record["normalize_calls"] == 0,
                      "QK group cannot contain PV history/normalization calls")
                check(record["history_calls"] <= record["descriptors"]
                      and record["normalize_calls"] <= record["descriptors"],
                      "PV history/normalization calls exceed descriptor count")
                check(record["record_start"] <= record["issue_end"] <= record["record_end"],
                      "timestamp order")
                check(record["issue_ticks"] == record["issue_end"] - record["record_start"],
                      "issue ticks do not match timestamps")
                check(record["completion_ticks"] == record["record_end"] - record["issue_end"],
                      "completion ticks do not match timestamps")
                check(record["elapsed_ticks"] == record["record_end"] - record["record_start"],
                      "elapsed ticks do not match timestamps")
                check(record["valid_mask"] == (1 if mode == 1 else 3),
                      "valid mask does not match probe mode")
                if mode == 1:
                    check(record["completion_ticks"] == record["empty_ticks"]
                          == record["pre_drain_ticks"] == 0,
                          "issue-only record contains completion/drain values")
                record["stage_name"] = STAGE_NAMES[stage]
                record["sample_position"] = POSITION_NAMES[position]
                record["position_name"] = POSITION_NAMES[position]
                record["mode_name"] = MODE_NAMES[mode]
                records.append(record)
    if require_records and not records:
        raise ValueError("Trace contains no required Cube group records")
    return records


def summarize_records(trials):
    """Pool captures by mode, stage, sample position, and exact logical work.

    A pooled median describes individual sparse samples, not a median kernel
    or a sum of stage costs. Modes and unequal work are never combined.
    """
    trials = list(trials)
    groups = defaultdict(list)
    for trial_index, records in enumerate(trials):
        for record in records:
            key = (record["mode"], record["stage"], record["position"],
                   *(record[field] for field in WORK_FIELDS))
            groups[key].append((trial_index, record))
    summaries = []
    for key, samples in sorted(groups.items()):
        mode, stage, position, *work = key
        fields = TICK_FIELDS if mode == 2 else ("issue_ticks", "elapsed_ticks")
        ticks = {}
        for field in fields:
            values = [record[field] for _, record in samples]
            ticks[field] = {"samples": len(values), "median": median(values),
                            "min": min(values), "max": max(values)}
        summaries.append({
            "mode": mode, "mode_name": MODE_NAMES[mode],
            "stage": stage, "stage_name": STAGE_NAMES[stage],
            "position": position, "sample_position": POSITION_NAMES[position],
            "position_name": POSITION_NAMES[position],
            **dict(zip(WORK_FIELDS, work)), "samples": len(samples),
            "trials": len({trial for trial, _ in samples}),
            "cores": sorted({record["core"] for _, record in samples}),
            "total_rounds": sorted({record["total_rounds"] for _, record in samples}),
            "ticks": ticks,
        })
    return {"contract": CONTRACT, "scope": SCOPE, "unit": "system_ticks",
            "aggregation": "pooled sparse records; no stage shares or additive production total",
            "trial_count": len(trials), "record_count": sum(len(rows) for rows in trials),
            "groups": summaries}
