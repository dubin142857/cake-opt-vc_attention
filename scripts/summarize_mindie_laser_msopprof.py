#!/usr/bin/env python3
"""Summarize existing msOpProf CSVs without importing Torch or launching a device.

CSV layouts and metric names follow Ascend's msopprof_user_guide.md and
msopprof_performance_data.md.  In particular, per-core CSVs normally have no
kernel name: their identity must come from an unambiguous sibling OpBasicInfo.
No values are added across cores, replay passes, tasks, or metric groups.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import statistics


GROUPS = ("OpBasicInfo", "PipeUtilization", "Memory", "MemoryL0", "MemoryUB",
          "L2Cache", "ArithmeticUtilization", "ResourceConflictRatio")
SOURCES = [
    "https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md",
    "https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_performance_data.md",
]
NAME_KEYS = {"opname", "kernelname"}
ID_KEYS = NAME_KEYS | {"optype", "taskid", "streamid", "deviceid", "pid", "blockid",
                       "subblockid", "coreid", "coretype", "tasktype", "launchid", "passid"}
MISSING = {"", "na", "n/a", "nan", "inf", "+inf", "-inf", "none", "null", "--", "-"}
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _key(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _group(path):
    # Official single-task files are Group.csv; multi-task files can carry a
    # timestamp suffix. Do not match MemoryL0 as Memory.
    for group in GROUPS:
        if path.stem.lower() == group.lower() or path.stem.lower().startswith(group.lower() + "_"):
            return group
    return "unrecognized"


def _number(raw):
    text = raw.strip()
    if text.lower() in MISSING:
        return None
    suffix = "%" if text.endswith("%") else ""
    body = text[:-1].strip() if suffix else text
    if not NUMBER.fullmatch(body):
        return None
    value = int(body) if re.fullmatch(r"[+-]?\d+", body) else float(body)
    return (value, suffix) if math.isfinite(value) else None


def _units(header):
    # Keep all parenthetical annotations; e.g. (KB)(estimate). Ratio fields
    # without a unit remain unspecified; they are never multiplied by 100.
    return re.findall(r"\(([^()]*)\)", header)


def _matches(name, prefixes):
    return any(name.startswith(prefix) for prefix in prefixes)


def _read_csv(path, root):
    item = {"path": str(path), "relative_path": path.relative_to(root).as_posix(),
            "task_directory": path.parent.relative_to(root).as_posix(),
            "metric_group": _group(path), "errors": []}
    try:
        data = path.read_bytes()
        item.update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        parsed = list(csv.reader(io.StringIO(data.decode("utf-8-sig")), strict=True))
        header_index = next((i for i, row in enumerate(parsed) if any(row)), None)
        if header_index is None:
            raise ValueError("Empty CSV")
        headers, raw_rows = parsed[header_index], parsed[header_index + 1:]
        item.update(headers=headers, raw_rows=raw_rows, header_csv_row=header_index + 1)
        normalized = [_key(x) for x in headers]
        if len(set(headers)) != len(headers) or len(set(normalized)) != len(normalized):
            item["errors"].append("Duplicate or ambiguous header names; raw columns retained")
        if not any(headers):
            item["errors"].append("CSV contains no named headers")
        rows = []
        for index, values in enumerate(raw_rows, header_index + 2):
            if not any(values):
                continue
            if len(values) != len(headers):
                item["errors"].append(f"CSV row {index}: {len(values)} cells for {len(headers)} headers")
                continue
            identifiers = {h: v for h, v in zip(headers, values) if _key(h) in ID_KEYS}
            names = [v.strip() for h, v in zip(headers, values) if _key(h) in NAME_KEYS and v.strip()]
            numeric = []
            for column, (header, raw) in enumerate(zip(headers, values)):
                number = _number(raw)
                if number is not None and _key(header) not in ID_KEYS:
                    numeric.append({"column": column, "field": header, "raw": raw,
                                    "value": number[0], "value_suffix": number[1],
                                    "header_units": _units(header),
                                    "domain": "AIC" if header.lower().startswith("aic_") else
                                              "AIV" if header.lower().startswith("aiv_") else "exported"})
            rows.append({"csv_row": index, "identifiers": identifiers, "kernel_names": names,
                         "numeric": numeric})
        item["rows"] = rows
        if not rows:
            item["errors"].append("No data rows")
    except (OSError, ValueError, UnicodeError, csv.Error) as error:
        item["errors"].append(f"{type(error).__name__}: {error}")
        item.setdefault("headers", [])
        item.setdefault("raw_rows", [])
        item.setdefault("rows", [])
    return item


def _attribute(files, prefixes):
    basics = {}
    for item in files:
        if item["metric_group"] == "OpBasicInfo":
            basics.setdefault(item["task_directory"], []).append(item)
    for item in files:
        siblings = basics.get(item["task_directory"], [])
        named = [(basic, row) for basic in siblings for row in basic["rows"]
                 if row["kernel_names"] and not basic["errors"]]
        # A single basic-info row in the same directory is the documented
        # single-task layout. Multiple rows/files are not guessed to be one task.
        sibling = named[0] if len(named) == 1 and len(siblings) == 1 else None
        for row in item["rows"]:
            names = row["kernel_names"]
            evidence = "row kernel name"
            if not names and sibling is not None:
                names = sibling[1]["kernel_names"]
                evidence = "sibling OpBasicInfo: " + sibling[0]["relative_path"]
            if not names:
                state = "unconfirmed"
            elif all(_matches(name, prefixes) for name in names):
                state = "matched"
            else:
                state = "mismatched"
            row["target"] = {"status": state, "kernel_names": names,
                              "evidence": evidence if names else "no unambiguous task identity"}
        item["matched_rows"] = sum(row["target"]["status"] == "matched" for row in item["rows"])
        item["mismatched_rows"] = sum(row["target"]["status"] == "mismatched" for row in item["rows"])
        item["unconfirmed_rows"] = sum(row["target"]["status"] == "unconfirmed" for row in item["rows"])


def _distributions(item):
    """Within one CSV/task only; retain every contributing core and row index."""
    groups = {}
    for row in item["rows"]:
        # Unknown/other kernels are retained raw, but excluded from target stats.
        if row["target"]["status"] != "matched":
            continue
        context = {h: v for h, v in row["identifiers"].items()
                   if _key(h) not in {"blockid", "coreid"}}
        context["attributed_kernel_names"] = row["target"]["kernel_names"]
        if not {"blockid", "subblockid"} <= {_key(h) for h in row["identifiers"]}:
            # Basic-info rows are separate task observations, not a core sample
            # distribution. Do not combine repeated names without task IDs.
            context["csv_row"] = row["csv_row"]
        for metric in row["numeric"]:
            key = (metric["column"], metric["value_suffix"], json.dumps(context, sort_keys=True))
            entry = groups.setdefault(key, {"field": metric["field"], "domain": metric["domain"],
                "header_units": metric["header_units"], "value_suffix": metric["value_suffix"],
                "context": context, "samples": []})
            entry["samples"].append({"csv_row": row["csv_row"], "identifiers": row["identifiers"],
                                     "value": metric["value"], "raw": metric["raw"]})
    result = []
    for entry in groups.values():
        values = [sample["value"] for sample in entry["samples"]]
        entry.update(count=len(values), minimum=min(values), median=statistics.median(values), maximum=max(values))
        entry["minimum_rows"] = [sample["csv_row"] for sample in entry["samples"] if sample["value"] == min(values)]
        entry["maximum_rows"] = [sample["csv_row"] for sample in entry["samples"] if sample["value"] == max(values)]
        result.append(entry)
    return result


def _usable(item):
    if item["errors"] or not item["matched_rows"]:
        return False
    target = [row for row in item["rows"] if row["target"]["status"] == "matched"]
    if item["metric_group"] == "OpBasicInfo":
        return any(all(any(_key(m["field"]) == field and m["value"] > 0 for m in row["numeric"])
                       for field in ("taskdurationus", "blockdim"))
                   for row in target)
    if item["metric_group"] == "PipeUtilization":
        return any({"blockid", "subblockid"} <= {_key(k) for k in row["identifiers"]}
                   and any(m["field"].lower() in {"aic_time(us)", "aiv_time(us)"} and m["value"] > 0
                           for m in row["numeric"])
                   and any(m["domain"] in {"AIC", "AIV"} and "ratio" in m["field"].lower()
                           for m in row["numeric"])
                   for row in target)
    group = item["metric_group"]
    def specific(field):
        field = field.lower()
        if group == "Memory":
            return any(token in field for token in ("_bw(", "_datas(", "_instructions", "_bw_usage_rate"))
        if group == "MemoryL0":
            return "_l0" in field and "_bw" in field
        if group == "MemoryUB":
            return "_ub_" in field and "_bw" in field
        if group == "L2Cache":
            return "cache_" in field or "hit_rate" in field
        if group == "ArithmeticUtilization":
            return ("_cube_" in field or "_vec_" in field) and any(
                token in field for token in ("ratio", "fops", "instr_number"))
        if group == "ResourceConflictRatio":
            return "_wait_ratio" in field or "_cflt_ratio" in field
        return False
    return any(specific(metric["field"]) for row in target for metric in row["numeric"])


def _collection(root, collection, index):
    result = {"collection_index": index, "variant": collection.get("variant", f"collection_{index}"),
              "collection": collection, "errors": [], "files": [], "status": "partial"}
    prefix = collection.get("kernel_prefix")
    prefixes = [prefix] if isinstance(prefix, str) else prefix
    if not isinstance(prefixes, list) or not prefixes or not all(isinstance(p, str) and p for p in prefixes):
        result["errors"].append("Missing valid kernel_prefix; no target identity can be verified")
        prefixes = []
    output = collection.get("output_dir")
    if not isinstance(output, str) or not output:
        result["errors"].append("Missing output_dir")
        directory = root / "__missing_collection_output__"
    else:
        directory = Path(output)
        if not directory.is_absolute():
            directory = root / directory
        directory = directory.resolve()
    result["output_dir"] = str(directory)
    if not directory.is_dir():
        result["errors"].append("Collection output directory is absent")
    else:
        try:
            paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".csv")
            result["files"] = [_read_csv(path, directory) for path in paths]
            result["visualization_artifacts"] = [str(p) for p in sorted(directory.rglob("visualize_data.bin"))]
        except OSError as error:
            result["errors"].append(f"Cannot enumerate artifacts: {error}")
    files = result["files"]
    _attribute(files, prefixes)
    for item in files:
        item["distributions"] = _distributions(item)
        item["usable_target_metrics"] = _usable(item)
        if item["metric_group"] in GROUPS and item["matched_rows"] and not item["usable_target_metrics"]:
            item["missing_evidence"] = (
                "OpBasicInfo requires target name, positive Task Duration(us), and positive Block Dim; "
                "PipeUtilization requires block_id/sub_block_id, positive core time, and a numeric pipeline ratio; "
                "other groups require at least one numeric group-specific counter. Check retained headers/rows/errors."
            )
        if item["metric_group"] in GROUPS and (item["mismatched_rows"] or item["unconfirmed_rows"]):
            result["errors"].append(f"Unmatched or unconfirmed target rows: {item['relative_path']}")
    result["groups"] = {}
    for group in GROUPS:
        members = [item for item in files if item["metric_group"] == group]
        result["groups"][group] = {"files": [m["relative_path"] for m in members],
            "status": "verified" if any(m["usable_target_metrics"] for m in members) else
                      "partial" if members else "missing"}
    result["missing_groups"] = [group for group, info in result["groups"].items() if info["status"] == "missing"]
    result["partial_groups"] = [group for group, info in result["groups"].items() if info["status"] == "partial"]
    # Require the two essentials in the SAME task directory. Two unrelated
    # tasks cannot jointly satisfy collection verification.
    basic_dirs = {f["task_directory"] for f in files if f["metric_group"] == "OpBasicInfo" and f["usable_target_metrics"]}
    pipe_dirs = {f["task_directory"] for f in files if f["metric_group"] == "PipeUtilization" and f["usable_target_metrics"]}
    result["verified_task_directories"] = sorted(basic_dirs & pipe_dirs)
    result["target_verification"] = "verified" if result["verified_task_directories"] else "partial"
    command_ok = collection.get("status") == "passed" and collection.get("returncode") == 0
    if not command_ok:
        result["errors"].append("Collection command did not report passed with returncode 0; artifacts retained")
    if (result["target_verification"] == "verified" and not result["missing_groups"]
            and not result["partial_groups"] and not result["errors"] and not any(f["errors"] for f in files)):
        result["status"] = "verified"
    return result


def summarize(run_root: Path):
    """Return JSON-serializable evidence; missing/failed output stays partial."""
    root = Path(run_root).resolve()
    report = {"schema": "laser-msopprof-summary-v1", "run_root": str(root), "status": "partial",
        "sources": SOURCES, "collections": [], "errors": [],
        "scope": "Profiler-active, replayed device counters; not paired API speedup or QK/softmax/PV phase attribution.",
        "aggregation": "Per CSV file, task context, core role and original field; never sum overlapping cores/pipes. Raw rows and identifiers retained. No unit conversion or ratio-to-percent inference."}
    manifest_path = root / "manifest.json"
    try:
        raw = manifest_path.read_bytes()
        report["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("collections"), list):
            raise ValueError("manifest must contain a collections list")
        report["manifest_schema"] = manifest.get("schema")
        if manifest.get("schema") != "laser-msopprof-v1":
            report["errors"].append("Unknown or absent manifest schema")
        for index, collection in enumerate(manifest["collections"]):
            if not isinstance(collection, dict):
                report["errors"].append(f"collections[{index}] is not an object")
                continue
            report["collections"].append(_collection(root, collection, index))
        if not report["collections"]:
            report["errors"].append("No collections")
        elif not report["errors"] and all(x["status"] == "verified" for x in report["collections"]):
            report["status"] = "verified"
    except (OSError, ValueError, UnicodeError) as error:
        report["errors"].append(f"Cannot read manifest: {type(error).__name__}: {error}")
    return report


def render_detailed_summary(report):
    lines = [f"msOpProf evidence: {report['status']}", report["scope"], report["aggregation"]]
    lines.extend("ERROR: " + error for error in report["errors"])
    for collection in report["collections"]:
        lines += ["", f"{collection['variant']}: {collection['status']}; target identity/basic+pipe: {collection['target_verification']}",
                  "  Missing groups: " + (", ".join(collection["missing_groups"]) or "none"),
                  "  Partial groups: " + (", ".join(collection["partial_groups"]) or "none")]
        lines.extend("  ERROR: " + error for error in collection["errors"])
        for item in collection["files"]:
            lines.append(f"  {item['metric_group']}: {item['relative_path']}; rows matched/mismatched/unconfirmed="
                         f"{item['matched_rows']}/{item['mismatched_rows']}/{item['unconfirmed_rows']}")
            lines.extend("    ERROR: " + error for error in item["errors"])
            if "missing_evidence" in item:
                lines.append("    PARTIAL: " + item["missing_evidence"])
            for entry in item["distributions"]:
                # All originals remain in JSON. A min/max row pointer makes
                # uneven core behavior visible without hiding it in a median.
                context = json.dumps(entry["context"], ensure_ascii=False, sort_keys=True)
                suffix = entry["value_suffix"]
                lines.append(f"    {entry['domain']} {entry['field']} [{context}]: n={entry['count']} "
                    f"min={entry['minimum']}{suffix} median={entry['median']}{suffix} max={entry['maximum']}{suffix}; "
                    f"min CSV rows={entry['minimum_rows']} max CSV rows={entry['maximum_rows']}")
    lines += ["", "CSV values/headers are exported units. AIC and AIV fields stay separate. "
              "CSV rows/core identities and raw values are in summary JSON.",
              "No register-spill diagnosis, critical-path percentage, or automatic bottleneck conclusion is inferred."]
    return "\n".join(lines) + "\n"


def _compact_text(value, limit=220):
    """Bound long exported identities without making truncation invisible."""
    value = str(value).replace("\n", " ").replace("\r", " ")
    if len(value) <= limit:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return value[:limit - 34] + f"... [sha={digest}; full in JSON]"


def _compact_metric(entry, *, span=False):
    if entry is None:
        return "NA"
    def number(value):
        return format(value, ".6g") + entry.get("value_suffix", "")
    values = (entry["minimum"], entry["median"], entry["maximum"]) if span else (entry["median"],)
    return "/".join(number(value) for value in values) + f"[n={entry['count']}]"


def _compact_line(label, tokens, limit=240):
    line, shown = label + ": ", 0
    for token in tokens:
        addition = ("; " if shown else "") + token
        # Reserve space for an explicit omitted-fields notice.
        if len(line) + len(addition) > limit - 33:
            break
        line += addition
        shown += 1
    if shown != len(tokens):
        line += f"; +{len(tokens) - shown} fields omitted (JSON)"
    return _compact_text(line, limit)


def _compact_file_view(item, kernel_names):
    """Choose one exported context per role; never pool tasks or CSV passes."""
    contexts, omitted = {}, 0
    for entry in item.get("distributions", []):
        context = entry["context"]
        if kernel_names and context.get("attributed_kernel_names") != kernel_names:
            omitted += 1
            continue
        role = next((str(v) for k, v in context.items() if _key(k) == "subblockid"), "task")
        identity = json.dumps(context, sort_keys=True, ensure_ascii=False)
        if role not in contexts:
            contexts[role] = (identity, {})
        if contexts[role][0] != identity:
            omitted += 1
            continue
        # A duplicate field with a different suffix/representation is a
        # different exported distribution, not a value to average together.
        field = entry["field"].lower()
        if field in contexts[role][1]:
            omitted += 1
            continue
        contexts[role][1][field] = entry
    return {role: fields for role, (_, fields) in contexts.items()}, omitted


def render_summary(report):
    """Bounded paste-friendly view; complete originals remain in JSON/details.

    A normal four-variant, one-task capture is approximately 80 lines. Only
    one task, one CSV per group and one context per core role are displayed.
    Additional tasks/passes/contexts are explicitly counted as omitted.
    """
    lines = [f"msOpProf compact evidence: {report['status']}",
        "Profiler/replay counters, not paired API timings. Values keep exported units; ratios are NOT multiplied by 100.",
        "time(us)=min/median/max[n]; other metrics=median[n]. f# indexes collection.files; NA is missing, not zero.",
        "Pipe/Arithmetic/Conflict labels omit the aic_/aiv_ field prefix identified by cube/vector role; units are unchanged."]
    if report.get("errors"):
        lines.append(_compact_text("Report errors: " + "; ".join(report["errors"])))
    for collection in report.get("collections", [])[:4]:
        files = collection.get("files", [])
        tasks = sorted({item["task_directory"] for item in files})
        preferred = collection.get("verified_task_directories", [])
        task = next(iter(preferred or tasks), None)
        lines += ["", _compact_text(f"{collection['variant']}: {collection['status']}; target/basic+pipe={collection['target_verification']}"),
                  _compact_text(f"  task={task if task is not None else 'NA'}; other tasks omitted={max(0, len(tasks) - 1)}")]
        selected, extra_files = {}, 0
        for index, item in enumerate(files):
            if item["task_directory"] != task:
                continue
            group = item["metric_group"]
            if group not in GROUPS:
                extra_files += 1
            elif group in selected:
                extra_files += 1
            else:
                selected[group] = (index, item)
        basic = selected.get("OpBasicInfo", (None, {}))[1]
        kernel_names = next((entry["context"]["attributed_kernel_names"] for entry in basic.get("distributions", [])
                             if entry["context"].get("attributed_kernel_names")), [])
        lines.append(_compact_text("  kernel=" + (" | ".join(kernel_names) or "NA (identity unconfirmed)")))
        views, context_omitted = {}, 0
        for group, (index, item) in selected.items():
            views[group], omitted = _compact_file_view(item, kernel_names)
            context_omitted += omitted
        def source(group):
            return "f" + str(selected[group][0]) if group in selected else "missing"
        def metric(group, role, field, span=False):
            return _compact_metric(views.get(group, {}).get(role, {}).get(field.lower()), span=span)
        def row(group, role, fields, label=None):
            def displayed(field):
                return re.sub(r"^ai[cv]_", "", field) if label in {"Pipe", "Arithmetic", "Conflict"} else field
            return _compact_line(f"  {label or group}[{source(group)}] {role}",
                [f"{displayed(field)}={metric(group, role, field, span)}" for field, span in fields])
        lines.append(row("OpBasicInfo", "task", [(field, False) for field in
            ("Task Duration(us)", "Block Dim", "Mix Block Dim", "Current Freq", "Rated Freq")], "Basic"))
        pipe_roles = list(views.get("PipeUtilization", {}))
        aic_roles = [role for role in pipe_roles if any(field.startswith("aic_") for field in views["PipeUtilization"][role])]
        aiv_roles = [role for role in pipe_roles if any(field.startswith("aiv_") for field in views["PipeUtilization"][role])]
        # Preserve the actual sub_block_id text (normally cube0/vector0/vector1).
        # Do not silently replace a missing vector1 with vector0.
        cube = aic_roles[0] if aic_roles else "cube?"
        vectors = aiv_roles[:2] or ["vector?"]
        omitted_roles = max(0, len(aic_roles) - 1) + max(0, len(aiv_roles) - 2)
        lines.append(row("PipeUtilization", cube, [("aic_time(us)", True)] + [(field, False) for field in
            ("aic_cube_ratio", "aic_mte1_ratio", "aic_mte2_ratio", "aic_fixpipe_ratio", "aic_scalar_ratio")], "Pipe"))
        for role in vectors:
            lines.append(row("PipeUtilization", role, [("aiv_time(us)", True)] + [(field, False) for field in
                ("aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio", "aiv_scalar_ratio")], "Pipe"))
        lines.append(_compact_line(f"  Wait/stall[{source('PipeUtilization')}]", [
            f"{cube} aic_scalar_mte1_stall_time(us)={metric('PipeUtilization', cube, 'aic_scalar_mte1_stall_time(us)')}"
        ] + [f"{role} aiv_scalar_wait_time(us)={metric('PipeUtilization', role, 'aiv_scalar_wait_time(us)')}" for role in vectors]))
        lines.append(row("ArithmeticUtilization", cube, [(field, False) for field in
            ("aic_cube_fp16_ratio", "aic_cube_int8_ratio")], "Arithmetic"))
        for role in vectors:
            lines.append(row("ArithmeticUtilization", role, [(field, False) for field in
                ("aiv_vec_fp16_ratio", "aiv_vec_fp32_ratio", "aiv_vec_misc_ratio")], "Arithmetic"))
        lines.append(_compact_line(f"  Memory[{source('Memory')}] {cube}", [
            f"aic_l1_read_bw(GB/s)={metric('Memory', cube, 'aic_l1_read_bw(GB/s)')}",
            f"aic_main_mem_read_bw(GB/s)={metric('Memory', cube, 'aic_main_mem_read_bw(GB/s)')}",
            f"L0[{source('MemoryL0')}] aic_l0a_read_bw(GB/s)={metric('MemoryL0', cube, 'aic_l0a_read_bw(GB/s)')}" ]))
        for role in vectors:
            lines.append(_compact_line(f"  Memory[{source('Memory')}] {role}", [
                f"aiv_gm_to_ub_bw(GB/s)={metric('Memory', role, 'aiv_gm_to_ub_bw(GB/s)')}",
                f"UB[{source('MemoryUB')}] aiv_ub_read_bw_vector(GB/s)={metric('MemoryUB', role, 'aiv_ub_read_bw_vector(GB/s)')}",
                f"aiv_ub_write_bw_vector(GB/s)={metric('MemoryUB', role, 'aiv_ub_write_bw_vector(GB/s)')}" ]))
        lines.append(_compact_line(f"  L2[{source('L2Cache')}]", [
            f"{cube} aic_total_hit_rate(%)={metric('L2Cache', cube, 'aic_total_hit_rate(%)')}"
        ] + [f"{role} aiv_total_hit_rate(%)={metric('L2Cache', role, 'aiv_total_hit_rate(%)')}" for role in vectors]))
        for role in vectors:
            lines.append(row("ResourceConflictRatio", role, [(field, False) for field in
                ("aiv_vec_total_cflt_ratio", "aiv_vec_bankgroup_cflt_ratio", "aiv_vec_bank_cflt_ratio")], "Conflict"))
        lines.append(_compact_text("  coverage: missing=" + (",".join(collection.get("missing_groups", [])) or "none")
            + "; partial=" + (",".join(collection.get("partial_groups", [])) or "none")
            + f"; extra files={extra_files}, contexts/fields={context_omitted}, roles={omitted_roles} omitted; errors={len(collection.get('errors', []))}"))
    if len(report.get("collections", [])) > 4:
        lines.append(f"Collections omitted={len(report['collections']) - 4}; see JSON/details.")
    lines.append("All omitted fields, exact values, per-core IDs, CSV paths and errors remain in JSON/details; no automatic bottleneck verdict.")
    return "\n".join(lines) + "\n"


def write_summary(run_root: Path, output: Path | None = None):
    """Write full JSON, compact text and a separate detailed text artifact."""
    report = summarize(run_root)
    target = Path(output) if output is not None else Path(run_root) / "summary.json"
    if target.suffix.lower() != ".json":
        target = target / "summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    report["summary_json"] = str(target.resolve())
    report["summary_text"] = str(target.with_suffix(".txt").resolve())
    details = target.with_name(target.stem + "_details.txt")
    report["summary_details_text"] = str(details.resolve())
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    target.with_suffix(".txt").write_text(render_summary(report), encoding="utf-8")
    details.write_text(render_detailed_summary(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, nargs="?", help="Existing capture directory; no new device work")
    parser.add_argument("--report", type=Path, help="Existing collector report; default results/mindie_laser_msopprof.json")
    parser.add_argument("--output", type=Path, help="JSON path or output directory; default RUN_ROOT/summary.json")
    parser.add_argument("--detailed", action="store_true", help="Explicitly print every counter; default is compact")
    args = parser.parse_args()
    if args.run_root is not None and args.report is not None:
        parser.error("Choose run_root or --report, not both")
    source_report = None
    if args.run_root is None:
        source_report = (args.report or Path(__file__).resolve().parents[1]/"results/mindie_laser_msopprof.json").resolve()
        try:
            metadata = json.loads(source_report.read_text(encoding="utf-8"))
            run_root = Path(metadata["run_root"])
            if not run_root.is_absolute():
                run_root = source_report.parent/run_root
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"Cannot read existing capture location from {source_report}: {error}. No NPU work performed.")
    else:
        run_root = args.run_root
    report = write_summary(run_root, args.output)
    if source_report is not None:
        alias = source_report.with_name(source_report.stem+"_summary.txt")
        if alias.resolve() != Path(report["summary_text"]).resolve():
            shutil.copyfile(report["summary_text"], alias)
    print(render_detailed_summary(report) if args.detailed else render_summary(report), end="")
    print("Summary:", report["summary_json"])
    print("All counters:", report["summary_details_text"])
    # A partial summary is a successful offline read, NOT a successful capture.
    # Collection drivers must inspect status rather than infer it from this exit.


if __name__ == "__main__":
    main()
