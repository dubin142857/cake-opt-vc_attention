"""Strict decoding of the independent Laser phase probe's INT64 output.

This module deliberately has no torch dependency.  Sampled system-counter
ticks are not core clock cycles, production phase percentages or API latency.
"""
from __future__ import annotations

from collections import defaultdict
from numbers import Integral
from statistics import median


MAGIC = 0x56434C50524F4245
VERSION = 1
TRACE_SHAPE = (4, 3, 8, 24)
TRACE_BYTES = 4 * 3 * 8 * 24 * 8
CONTRACT = "laser_cube_vector_phase_system_ticks_v1"
FIELD_NAMES = (
    "magic", "version", "mode", "core", "role", "kind", "slot", "round",
    "section", "flags", "total_rounds", "key_blocks", "rows", "lines",
    "wait_ticks", "empty_ticks", "score_load_ticks", "max_history_ticks",
    "subtract_ticks", "exp_ticks", "sum_ticks", "record_start", "record_end",
    "valid_mask",
)
TICK_FIELDS = FIELD_NAMES[14:21]
ROLE_NAMES = ("Cube", "AIV0", "AIV1")
SECTION_CLASSES = ("steady", "head", "tail", "head_and_tail")
SCOPE = (
    "Sampled cores 0..3: original cross-core waits and instrumented Vector "
    "completion intervals. Extra synchronization and recording perturb the "
    "pipeline. System ticks are not core clock cycles; groups, stages and "
    "counter values must not be added or converted into production phase "
    "percentages or attention milliseconds."
)


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if not 0 <= value <= (1 << 63) - 1:
        raise ValueError(f"{name} is outside the nonnegative INT64 range")
    return value


def _nested(trace):
    # A CPU/NPU torch Tensor is copied before reading.  Array-like objects
    # with tolist() work as well, without importing either dependency here.
    if hasattr(trace, "detach"):
        trace = trace.detach()
    if hasattr(trace, "cpu"):
        trace = trace.cpu()
    if hasattr(trace, "tolist"):
        trace = trace.tolist()

    def shape_check(value, shape, path):
        if not isinstance(value, (list, tuple)) or len(value) != shape[0]:
            raise ValueError(f"Trace shape must be {TRACE_SHAPE}; invalid dimension at {path}")
        if len(shape) == 1:
            return [_integer(item, f"{path}[{i}]") for i, item in enumerate(value)]
        return [shape_check(item, shape[1:], f"{path}[{i}]") for i, item in enumerate(value)]

    return shape_check(trace, TRACE_SHAPE, "trace")


def decode_trace(trace, mode, require_waits=False, require_phases=False):
    """Return complete nonzero records after validating storage and metadata.

    Zero slots are expected: short cases need not exercise every core, AIV
    role, sampled round or section class. Optional requirements demand at
    least one record of the specified kind, not coverage of every role.
    """
    mode = _integer(mode, "mode")
    if mode not in (0, 1, 2):
        raise ValueError("mode must be 0 (plain), 1 (waits), or 2 (waits and Vector phases)")
    if (require_waits and mode == 0) or (require_phases and mode != 2):
        raise ValueError("Requested records are incompatible with probe mode")
    data = _nested(trace)
    records = []
    for core, roles in enumerate(data):
        for role, slots in enumerate(roles):
            for slot, values in enumerate(slots):
                if not any(values):
                    continue
                location = f"core {core}, role {role}, slot {slot}"
                if mode == 0:
                    raise ValueError(f"Plain mode trace must be all zero ({location})")
                record = dict(zip(FIELD_NAMES, values))

                def check(condition, reason):
                    if not condition:
                        raise ValueError(f"Invalid Laser phase record at {location}: {reason}")

                check(record["magic"] == MAGIC and record["version"] == VERSION,
                      "magic/version mismatch or partially written record")
                check((record["core"], record["role"], record["slot"], record["mode"])
                      == (core, role, slot, mode), "metadata does not match storage/mode")
                check(record["flags"] <= 7, "unknown flag bits")
                check(record["total_rounds"] > 0 and record["round"] < record["total_rounds"],
                      "round is outside total_rounds")
                check(all(record[field] <= (1 << 32) - 1 for field in
                          ("section", "key_blocks", "rows", "lines")),
                      "section/shape metadata exceeds UINT32")
                check(record["record_end"] >= record["record_start"], "timestamp order")
                if slot < 4:
                    check(record["kind"] == 1 and record["valid_mask"] == 1,
                          "wait slot must have kind 1 and wait-only mask")
                    targets = (0, 1, record["total_rounds"] // 2, record["total_rounds"] - 1)
                    check(record["round"] == targets[slot]
                          and targets.index(record["round"]) == slot,
                          "wait sample round/slot violates first-match sampling")
                    check(record["wait_ticks"] == record["record_end"] - record["record_start"],
                          "wait ticks do not match timestamps")
                    check(all(record[field] == 0 for field in TICK_FIELDS[1:]),
                          "wait slot contains unmarked phase values")
                    section_class = None
                else:
                    check(mode == 2 and role in (1, 2), "Vector phases require mode 2 and an AIV role")
                    check(record["kind"] == 2 and record["valid_mask"] == 126,
                          "phase slot must have kind 2 and all six phase fields")
                    check(record["flags"] & 1, "phase sample must be active")
                    check(slot == 4 + ((record["flags"] >> 1) & 3),
                          "head/tail flags do not match section-class slot")
                    check(record["key_blocks"] > 0 and record["rows"] == 4,
                          "phase sample must contain four rows and positive key_blocks")
                    check(record["wait_ticks"] == 0, "phase slot contains an unmarked wait value")
                    # Each interval lies within the enclosing record, but a
                    # sum of intervals is intentionally not an ABI invariant.
                    check(all(record[field] <= record["record_end"] - record["record_start"]
                              for field in TICK_FIELDS[1:]), "phase interval exceeds record span")
                    section_class = SECTION_CLASSES[slot - 4]
                record["role_name"] = ROLE_NAMES[role]
                record["kind_name"] = "wait" if slot < 4 else "phase"
                record["section_class"] = section_class
                records.append(record)
    if require_waits and not any(record["kind"] == 1 for record in records):
        raise ValueError("Trace contains no required cross-core wait samples")
    if require_phases and not any(record["kind"] == 2 for record in records):
        raise ValueError("Trace contains no required Vector phase samples")
    return records


def summarize_records(trials):
    """Pool captures without mixing wait positions or different Vector work.

    Medians summarize sampled records (not medians of per-trial medians).
    Samples/trials are reported so sparse captures remain visible.
    """
    trials = list(trials)
    groups = defaultdict(list)
    for trial_index, records in enumerate(trials):
        for record in records:
            active = bool(record["flags"] & 1)
            # Wait position and activity distinguish startup/drain/inactive
            # handoffs. Vector section class alone does not fix the amount of
            # work, so keep key-block count separate as well.
            key = (record["role"], record["kind"], record["slot"], active,
                   record["key_blocks"] if record["kind"] == 2 else 0)
            groups[key].append((trial_index, record))
    summaries = []
    for (role, kind, slot, active, key_blocks), samples in sorted(groups.items()):
        section_class = SECTION_CLASSES[slot - 4] if kind == 2 else None
        ticks = {}
        for bit, field in enumerate(TICK_FIELDS):
            values = [record[field] for _, record in samples if record["valid_mask"] & (1 << bit)]
            if values:
                ticks[field] = {"samples": len(values), "median": median(values),
                                "min": min(values), "max": max(values)}
        summaries.append({"role": role, "role_name": ROLE_NAMES[role],
                          "kind": kind, "kind_name": "wait" if kind == 1 else "phase",
                          "slot": slot, "active": active,
                          "sample_position": ("first", "second", "middle", "last")[slot]
                          if kind == 1 else None,
                          "key_blocks": key_blocks if kind == 2 else None,
                          "rows": 4 if kind == 2 else None,
                          "section_class": section_class, "samples": len(samples),
                          "trials": len({trial for trial, _ in samples}),
                          "cores": sorted({record["core"] for _, record in samples}),
                          "ticks": ticks})
    return {"contract": CONTRACT, "scope": SCOPE, "unit": "system_ticks",
            "aggregation": "pooled sampled records; no phase shares or additive total",
            "trial_count": len(trials), "record_count": sum(len(rows) for rows in trials),
            "groups": summaries}
