"""Strict sparse PV trace decoding; raw system ticks never represent phase shares."""
from __future__ import annotations

from collections import defaultdict
from numbers import Integral
from statistics import median

MAGIC = 0x56434C5056505242
VERSION = 2
TRACE_SHAPE = (4, 3, 3, 64)
FIELD_NAMES = (
    "magic", "version", "mode", "core", "round_position", "k_position", "round", "rounds",
    "descriptor", "descriptors", "kx", "ky", "k", "on_start", "on_end", "v_prefetch", "group_valid",
    "call_begin", "call_issue_end", "call_completed_end", "call_issue_ticks", "call_completion_ticks",
    "call_total_ticks", "history_issue_ticks", "output_issue_ticks", "tile_issue_ticks",
    "p_free_dependency_ticks", "p_copy_issue_ticks", "v_ready_dependency_ticks",
    "p_ready_dependency_ticks", "l0b_free_dependency_ticks", "v_l1_to_l0b_issue_ticks",
    "v_l0b_ready_dependency_ticks", "next_v_free_dependency_ticks", "next_v_copy_issue_ticks",
    "l0a_free_dependency_ticks", "p_l1_to_l0a_issue_ticks", "p_l0a_ready_dependency_ticks",
    "mad_fix_issue_dependency_ticks", "release_issue_ticks", "prime_v_free_dependency_ticks",
    "prime_v_copy_issue_ticks", "phase_mask", "p_copy_calls", "v_l1_l0_load_calls",
    "p_l1_l0_load_calls", "mad_calls", "fix_calls", "next_v_copy_calls", "history_calls",
    "output_calls", "valid_flags", *tuple(f"reserved{i}" for i in range(12)),
)
PHASE_FIELDS = FIELD_NAMES[26:40]
MODE_NAMES = ("plain", "issue", "detail")
POSITION_NAMES = ("first", "middle", "last")
CALL_FIELDS = FIELD_NAMES[20:23]
SCOPE = (
    "Sparse Cube-only raw system ticks. Internal brackets observe original scalar issue/dependency windows, "
    "not pure DMA/MAD/FIX completion. Neither mode adds a completion drain; completed-call fields are "
    "unavailable. Whole calls cover all descriptors, while tiles/history/output/prime represent descriptor 0. "
    "Do not add windows, subtract probe overhead from samples, or convert them into kernel percentages/ms."
)


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 2**63-1:
        raise ValueError(f"{name} requires nonnegative INT64")
    return int(value)


def _nested(trace):
    if hasattr(trace, "detach"): trace = trace.detach()
    if hasattr(trace, "cpu"): trace = trace.cpu()
    if hasattr(trace, "tolist"): trace = trace.tolist()
    def walk(value, shape):
        if not isinstance(value, (tuple, list)) or len(value) != shape[0]:
            raise ValueError(f"PV trace shape must be {TRACE_SHAPE}")
        return ([_integer(x, "trace field") for x in value] if len(shape) == 1
                else [walk(x, shape[1:]) for x in value])
    return walk(trace, TRACE_SHAPE)


def _position(index, size):
    targets = (0, size//2, size-1)
    return targets.index(index) if index in targets else None


def decode_trace(trace, mode, require_records=False):
    mode = _integer(mode, "mode")
    if mode not in (0, 1, 2):
        raise ValueError("PV mode must be plain0/issue1/detail2")
    if require_records and mode == 0:
        raise ValueError("Plain mode cannot require records")
    records = []
    for core, rounds in enumerate(_nested(trace)):
        for rp, keys in enumerate(rounds):
            for kp, values in enumerate(keys):
                if not any(values): continue
                r = dict(zip(FIELD_NAMES, values))
                def check(condition, reason):
                    if not condition:
                        raise ValueError(f"Invalid PV record core{core}/round{rp}/k{kp}: {reason}")
                check(mode > 0, "plain trace must be zero")
                check(r["magic"] == MAGIC and r["version"] == VERSION, "magic/version")
                check((r["mode"], r["core"], r["round_position"], r["k_position"]) == (mode, core, rp, kp),
                      "storage/mode metadata")
                check(not any(values[52:]), "reserved fields")
                check(r["rounds"] > 0 and r["round"] < r["rounds"] and
                      _position(r["round"], r["rounds"]) == rp, "round/first-match position")
                check(r["kx"] > 0 and r["k"] < r["kx"] and _position(r["k"], r["kx"]) == kp,
                      "K/first-match position")
                check(r["descriptor"] == 0 and r["descriptors"] > 0 and r["ky"] > 0,
                      "descriptor/Ky metadata")
                check(all(r[f] in (0, 1) for f in ("on_start", "on_end", "v_prefetch", "group_valid")),
                      "boolean metadata")
                group = kp == 0
                check(r["group_valid"] == int(group), "whole-call validity only first K slot")
                detailed = mode == 2
                history, output = detailed and group and not r["on_start"], detailed and group and bool(r["on_end"])
                next_v = detailed and bool(r["v_prefetch"]) and r["k"]+1 < r["kx"]
                flags = int(group) | (int(detailed)<<2) | (int(history)<<3) | (int(output)<<4) | (int(next_v)<<5)
                check(r["valid_flags"] == flags, "valid flags")
                check(r["call_completed_end"] == r["call_completion_ticks"] == r["call_total_ticks"] == 0,
                      "passive probe cannot contain completed call fields")
                if group:
                    check(r["call_begin"] <= r["call_issue_end"] and
                          r["call_issue_ticks"] == r["call_issue_end"]-r["call_begin"], "call issue timestamp/delta")
                else:
                    check(not any(values[17:25]) and not any(values[40:42]) and not any(values[49:51]),
                          "non-first K record duplicates whole-call/descriptor-only windows")
                check(r["phase_mask"] < 1<<len(PHASE_FIELDS), "unknown phase bits")
                for i, field in enumerate(PHASE_FIELDS):
                    check(bool(r["phase_mask"] & (1<<i)) or r[field] == 0, "invalid phase window is nonzero")
                if not detailed:
                    check(not any(values[23:51]), "issue mode contains internal windows/counts")
                else:
                    check(r["p_copy_calls"] == 2 and r["v_l1_l0_load_calls"] == 8 and
                          r["p_l1_l0_load_calls"] == 8*r["ky"] and r["mad_calls"] == r["ky"], "logical tile call counts")
                    check(r["fix_calls"] == (r["ky"] if r["k"]+1 == r["kx"] else 0) and
                          r["next_v_copy_calls"] == int(next_v), "FIX/next V counts")
                    check(r["history_calls"] == int(history) and r["output_calls"] == int(output), "descriptor history/output counts")
                    check(history or r["history_issue_ticks"] == 0, "inactive history window")
                    check(output or r["output_issue_ticks"] == 0, "inactive output window")
                    next_mask = (1<<(33-26)) | (1<<(34-26))
                    base_mask = ((1<<len(PHASE_FIELDS))-1) ^ next_mask
                    check(r["phase_mask"] == base_mask | (next_mask if next_v else 0), "missing/inactive phase bits")
                    check(group and r["v_prefetch"] or not any(values[40:42]), "inactive V prime windows")
                r.update(role="Cube", mode_name=MODE_NAMES[mode], round_position_name=POSITION_NAMES[rp],
                         k_position_name=POSITION_NAMES[kp])
                records.append(r)
    if require_records and not records:
        raise ValueError("No required PV records")
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["core"], record["round_position"])].append(record)
    consistency = ("round", "rounds", "descriptors", "kx", "ky", "on_start", "on_end", "v_prefetch")
    for key, rows in grouped.items():
        if not any(r["group_valid"] for r in rows) or len({tuple(r[f] for f in consistency) for r in rows}) != 1:
            raise ValueError(f"PV K records disagree or omit whole-call first slot at {key}")
    return records


def require_long_coverage(records):
    """The default long workload must sample four cores and all nine slots each."""
    expected = {(c, r, k) for c in range(4) for r in range(3) for k in range(3)}
    observed = {(r["core"], r["round_position"], r["k_position"]) for r in records}
    if observed != expected or any(not r["v_prefetch"] for r in records):
        raise ValueError("Long PV trace requires four cores, first/middle/last rounds/K and V-prefetch route")


def _stats(values):
    return dict(samples=len(values), median=median(values), min=min(values), max=max(values))


def summarize_records(trials):
    trials = list(trials)
    calls, tiles = defaultdict(list), defaultdict(list)
    for trial, records in enumerate(trials):
        for r in records:
            common = (r["mode"], r["round_position"], r["descriptors"], r["kx"], r["ky"],
                      r["on_start"], r["on_end"], r["v_prefetch"])
            if r["group_valid"]: calls[common].append((trial, r))
            if r["mode"] == 2: tiles[(*common, r["k_position"])].append((trial, r))
    fields = ("mode", "round_position", "descriptors", "kx", "ky", "on_start", "on_end", "v_prefetch")
    groups = {"calls": [], "tiles": []}
    for kind, mapping in (("calls", calls), ("tiles", tiles)):
        for key, samples in sorted(mapping.items()):
            row = dict(zip(fields+( ("k_position",) if kind == "tiles" else ()), key))
            row.update(samples=len(samples), trials=len({trial for trial, _ in samples}),
                       cores=sorted({r["core"] for _, r in samples}), ticks={}, inactive={})
            window_fields = CALL_FIELDS+("history_issue_ticks", "output_issue_ticks", "prime_v_free_dependency_ticks", "prime_v_copy_issue_ticks") if kind == "calls" else ("tile_issue_ticks",)+PHASE_FIELDS
            for field in window_fields:
                values = []
                for _, r in samples:
                    valid = True
                    if field in PHASE_FIELDS: valid = bool(r["phase_mask"] & (1<<PHASE_FIELDS.index(field)))
                    elif field == "history_issue_ticks": valid = bool(r["valid_flags"] & 8)
                    elif field == "output_issue_ticks": valid = bool(r["valid_flags"] & 16)
                    elif field.startswith("prime_v_"): valid = r["mode"] == 2 and bool(r["v_prefetch"])
                    elif field in ("call_completion_ticks", "call_total_ticks"): valid = False
                    if valid: values.append(r[field])
                row["inactive"][field] = len(samples)-len(values)
                if values: row["ticks"][field] = _stats(values)
            groups[kind].append(row)
    return dict(unit="system_ticks", scope=SCOPE, trial_count=len(trials),
                record_count=sum(map(len, trials)), **groups)


def sample_table(summaries):
    lines = ["Sparse PV raw system ticks; NOT production percentages or pure hardware phase costs:"]
    for label, summary in summaries.items():
        for kind in ("calls", "tiles"):
            for row in summary[kind]:
                position = POSITION_NAMES[row["round_position"]]
                if kind == "tiles": position += "/K"+POSITION_NAMES[row["k_position"]]
                lines.append(f"{label}/{kind}/{position}: descriptor0 Kx={row['kx']} Ky={row['ky']} "
                             f"group_descriptors={row['descriptors']} start={row['on_start']} end={row['on_end']} samples={row['samples']}")
                for field, value in row["ticks"].items():
                    lines.append(f"  {field}: median={value['median']:.1f}; min={value['min']}; max={value['max']}; n={value['samples']}")
                inactive = {field: count for field, count in row["inactive"].items() if count}
                if inactive: lines.append("  inactive/excluded: "+str(inactive))
    return "\n".join(lines+[SCOPE])
