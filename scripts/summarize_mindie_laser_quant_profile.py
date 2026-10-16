#!/usr/bin/env python3
"""Print existing Laser paired timings and NPU counters, entirely offline."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _number(value, label, *, nonnegative=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{label} must be finite" + (" and nonnegative" if nonnegative else ""))
    if integer and type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _finite_tree(value, label="report"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains nonfinite data")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")


def _status(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty status string")
    return value


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _trace_from_report(report):
    if "trace" in report:
        return _mapping(report["trace"], "trace")
    if "variants" in report:
        return report  # Standalone capture_report.json.
    return None


def render_report(report):
    """Return a validated summary without importing runtime libraries or writing files.

    Counters retain their original exported names and numbers. Their units and
    meaning must be interpreted using the matching CANN version, not guessed
    from a field name or rescaled as percentages here.
    """
    _mapping(report, "report")
    _finite_tree(report)
    if not any(key in report for key in ("timings", "trace", "variants", "error", "parent_error")):
        raise ValueError("Not a Laser profile or capture report")
    status = _status(report.get("status"), "report.status")
    lines = [f"Report status: {status}"]
    if "stage" in report:
        lines.append(f"Report stage: {report['stage']}")
    if "timing_status" in report:
        lines.append(f"Uninstrumented timing_status: {_status(report['timing_status'], 'timing_status')}")
    for key in ("error", "parent_error"):
        if key in report:
            lines.append(f"{key}: {_json(report[key])}")
    timings = _mapping(report.get("timings", {}), "timings")
    comparisons = _mapping(report.get("comparisons", {}), "comparisons")
    if timings:
        lines += ["", "Paired API wall milliseconds (profiler inactive):",
                  f"{'scope':<20} {'torch':>12} {'fused':>12} {'torch/fused':>12} {'paired saved':>14} {'faster':>9}"]
        for scope, timing in timings.items():
            timing = _mapping(timing, f"timings.{scope}")
            rows = _mapping(timing.get("rows"), f"timings.{scope}.rows")
            values = []
            for name in ("torch", "fused"):
                row = _mapping(rows.get(name), f"{scope}.rows.{name}")
                values.append(_number(row.get("median_ms"), f"{scope}.{name}.median_ms", nonnegative=True))
            old, new = values
            ratio = f"{old / new:.4f}x" if new else "undefined"
            saved, faster = "unavailable", "--"
            if scope in comparisons:
                pair = _mapping(comparisons[scope], f"comparisons.{scope}")
                saved = f"{_number(pair.get('median_saved_ms'), scope + '.median_saved_ms'):+.4f}"
                count = _number(pair.get("faster_trials"), scope + ".faster_trials", nonnegative=True, integer=True)
                trials = _number(pair.get("trials"), scope + ".trials", nonnegative=True, integer=True)
                if count > trials:
                    raise ValueError(f"{scope}: faster_trials exceeds trials")
                faster = f"{count}/{trials}"
            lines.append(f"{scope:<20} {old:12.4f} {new:12.4f} {ratio:>12} {saved:>14} {faster:>9}")
        lines.append("Positive paired saved ms favors fused; independent scopes are not additive or kernel phase percentages.")

    trace = _trace_from_report(report)
    if trace is None:
        lines += ["", "NPU trace: absent; no device counters available."]
        return "\n".join(lines)
    trace_status = _status(trace.get("status"), "trace.status")
    lines += ["", f"NPU trace status: {trace_status}"]
    if "error" in trace:
        lines.append(f"NPU trace error: {_json(trace['error'])}")
    schedule = trace.get("schedule")
    if schedule is None:
        lines.append("Schedule: legacy schedule unverified; capture completeness is not established.")
    else:
        _mapping(schedule, "trace.schedule")
        lines.append(f"Schedule: {_json(schedule)}")
    if "pipe_metrics" in trace:
        lines.append(f"pipe_metrics: {_json(trace['pipe_metrics'])}")
    variants = _mapping(trace.get("variants", {}), "trace.variants")
    if not variants:
        lines.append("No captured variants; no device counters available.")
    for name, row in variants.items():
        row = _mapping(row, f"trace.variants.{name}")
        lines += ["", f"Variant: {name}; status={_status(row.get('status'), name + '.status')}"]
        finalized = row.get("capture_finalized")
        if finalized is not None and type(finalized) is not bool:
            raise ValueError(f"{name}.capture_finalized must be boolean")
        lines.append("  capture_finalized: " + (str(finalized).lower() if finalized is not None else "unverified (legacy report)"))
        for key in ("failed_stage", "reason", "error", "directory", "kernel_table_files"):
            if key in row:
                lines.append(f"  {key}: {_json(row[key])}")
        summary = row.get("device_summary")
        if summary is None:
            lines.append("  device_summary: unavailable; no device timing or counters inferred.")
            continue
        summary = _mapping(summary, name + ".device_summary")
        for key in ("submitted_calls", "device_tasks", "kernel_groups_total"):
            _number(summary.get(key), f"{name}.{key}", nonnegative=True, integer=True)
        duration = _number(summary.get("all_device_tasks_sum_ms_per_call"), name + ".all_device_tasks_sum_ms_per_call", nonnegative=True)
        lines.append(f"  submitted_calls={summary['submitted_calls']}; device_tasks={summary['device_tasks']}; all_device_tasks_sum_ms_per_call={duration:.6f}")
        if "source" in summary:
            lines.append(f"  source: {summary['source']}")
        groups = summary.get("kernel_groups")
        if not isinstance(groups, list):
            raise ValueError(f"{name}.kernel_groups must be a list")
        if summary["kernel_groups_total"] < len(groups):
            raise ValueError(f"{name}.kernel_groups_total smaller than available groups")
        lines.append(f"  kernel_groups: {len(groups)} shown / {summary['kernel_groups_total']} exported")
        for index, group in enumerate(groups):
            group = _mapping(group, f"{name}.kernel_groups[{index}]")
            for key in ("name", "op_type", "task_type"):
                if not isinstance(group.get(key), str):
                    raise ValueError(f"{name}.kernel_groups[{index}].{key} must be a string")
            tasks = _number(group.get("tasks"), "kernel.tasks", nonnegative=True, integer=True)
            lines.append(f"  Kernel name={_json(group['name'])}; op_type={_json(group['op_type'])}; task_type={_json(group['task_type'])}; tasks={tasks}")
            for key in ("sum_duration_ms", "sum_duration_ms_per_call", "median_task_duration_ms"):
                value = _number(group.get(key), "kernel." + key, nonnegative=True)
                lines.append(f"    {key}: {value:.6f}")
            counters = _mapping(group.get("counters_as_exported", {}), "counters_as_exported")
            if not counters:
                lines.append("    counters_as_exported: missing (no numeric counter samples)")
            for key, values in counters.items():
                values = _mapping(values, f"counters_as_exported.{key}")
                count = _number(values.get("samples"), key + ".samples", nonnegative=True, integer=True)
                if count == 0:
                    raise ValueError(f"{key}.samples must be positive for an exported counter")
                for field in ("median", "min", "max"):
                    _number(values.get(field), key + "." + field)
                if not values["min"] <= values["median"] <= values["max"]:
                    raise ValueError(f"{key}: inconsistent min/median/max")
                lines.append(f"    counters_as_exported.{key}: median={values['median']}; min={values['min']}; max={values['max']}; samples={count}")
    lines += ["", "Device durations were collected with the profiler active; overlapping tasks are summed.",
              "Counters retain exported names and values; no unit conversion or percentage inference.",
              "These are not QK/softmax/PV phase percentages and must not be mixed with API wall times."]
    return "\n".join(lines)


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/mindie_laser_quant_profile.json",
                        help="Existing combined report or standalone capture_report.json")
    args = parser.parse_args(argv)
    try:
        report = json.loads(Path(args.input).read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)
        output = render_report(report)
        print(output)
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        print(f"Cannot summarize report: {exc}", file=sys.stderr)
        return 2
    trace = _trace_from_report(report)
    if report.get("status") in ("failed", "running") or (trace and trace.get("status") in ("failed", "running")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
