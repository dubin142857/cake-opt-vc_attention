"""Optional torch_npu traces, separate from uninstrumented API timings.

Official API examples:
https://gitee.com/ascend/mstt/blob/337553a74001d0a541bdcf6ff5eb72fdf407ff3a/profiler/README.md
Official CSV fields (Duration(us) describes device execution):
https://www.hiascend.com/doc_center/source/zh/canncommercial/70RC1/foundmodeldev/foundmodelinfer/PT_LMIMOG_0044.html

This module deliberately imports neither torch nor torch_npu at import time.
It does not infer device timings from PyTorch host events. Mixed Cube/Vector
tasks can overlap, so summed device task durations are not elapsed API time.
"""
from collections import defaultdict
from decimal import Decimal
import csv
import json
import math
from pathlib import Path
import re
import statistics
import tempfile


SOURCES = [
    "https://gitee.com/ascend/mstt/blob/337553a74001d0a541bdcf6ff5eb72fdf407ff3a/profiler/README.md",
    "https://www.hiascend.com/doc_center/source/zh/canncommercial/70RC1/foundmodeldev/foundmodelinfer/PT_LMIMOG_0044.html",
    "https://gitee.com/ascend/pytorch/blob/47f1f92999deda63f627141111cd6ffb736deb97/torch_npu/profiler/_profiler_action_controller.py",
]


class NativeTraceExecutionError(RuntimeError):
    """The actual submitted call or device synchronization failed during capture."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report
        self.partial_report = report


def _error(exc):
    return {"type": type(exc).__name__, "message": str(exc)[:4000]}


def summarize_kernel_csv(path, calls, *, all_numeric_fields=False):
    """Read ONE exported kernel table; retain task types rather than merging them.

    Counts and summed durations include every device row, including auxiliary
    kernels. Per-call values divide by actual submitted callables; they
    are averages across this capture, not per-call medians or critical paths.
    """
    if calls <= 0:
        raise ValueError("calls must be positive")
    groups = defaultdict(lambda: {"duration_us": [], "counters": defaultdict(list), "extra": defaultdict(list)})
    intervals = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        # Do not guess units from arbitrary duration-like column names.
        if "Name" not in columns or "Duration(us)" not in columns:
            raise ValueError("kernel_details.csv lacks Name or Duration(us); no timing inferred")
        counters = [key for key in columns if key.startswith(("aic_", "aiv_", "aicore_"))]
        has_timeline = "Start Time(us)" in columns and "Device_id" in columns
        for row in reader:
            duration = float(row["Duration(us)"])
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("Invalid device duration in kernel_details.csv")
            if has_timeline:
                # Absolute device times exceed 1e15 us: binary float loses
                # sub-microsecond gaps. Keep exported decimals until subtraction.
                start = Decimal(row["Start Time(us)"].strip())
                length = Decimal(row["Duration(us)"].strip())
                device = row["Device_id"].strip()
                if not device or not start.is_finite() or start < 0:
                    raise ValueError("Invalid device timestamp or device identifier")
                intervals[device].append((start, start + length))
            key = (row["Name"], row.get("Type", row.get("Op Type", "")),
                   row.get("Task Type", row.get("Accelerator Core", "")))
            group = groups[key]
            group["duration_us"].append(duration)
            for column in counters:
                raw = row.get(column, "")
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    group["counters"][column].append(value)
            if all_numeric_fields:
                identity = {"Name", "Type", "Op Type", "Task Type", "Accelerator Core", "Device_id",
                            "Stream Id", "Task Id", "Start Time(us)", "Duration(us)"}
                for column in columns:
                    if column in identity or column in counters:continue
                    try:value = float(row.get(column, ""))
                    except (TypeError, ValueError):continue
                    if math.isfinite(value):group["extra"][column].append(value)
    if not groups:
        raise ValueError("kernel_details.csv contains no device rows")
    rows = []
    for (name, op_type, task_type), group in groups.items():
        durations = group["duration_us"]
        rows.append({"name": name, "op_type": op_type, "task_type": task_type,
                     "tasks": len(durations), "sum_duration_ms": sum(durations) / 1000,
                     "sum_duration_ms_per_call": sum(durations) / 1000 / calls,
                     "median_task_duration_ms": statistics.median(durations) / 1000,
                     "counters_as_exported": {
                         column: {"samples": len(values), "median": statistics.median(values),
                                  "min": min(values), "max": max(values)}
                         for column, values in group["counters"].items()}})
        if all_numeric_fields:
            rows[-1]["extra_numeric_fields_as_exported"] = {
                column:dict(samples=len(values),median=statistics.median(values),min=min(values),max=max(values))
                for column,values in group["extra"].items()}

    rows.sort(key=lambda row: row["sum_duration_ms"], reverse=True)
    timeline = {"status": "unavailable", "reason": "Explicit Start Time(us)/Device_id columns absent"}
    if has_timeline:
        devices = {}
        for device, spans in intervals.items():
            ordered = sorted(spans)
            left, right = ordered[0]
            union = Decimal(0)
            for start, end in ordered[1:]:
                if start > right:
                    union += right - left
                    left, right = start, end
                else:
                    right = max(right, end)
            union += right - left
            span = max(end for _,end in ordered) - ordered[0][0]
            summed = sum((end-start for start,end in ordered), Decimal(0))
            devices[device] = dict(union_ms_per_call=float(union / 1000 / calls),
                capture_span_ms=float(span / 1000), gap_ms=float((span-union) / 1000),
                overlap_ms=float((summed-union) / 1000), tasks=len(spans))
        timeline = dict(status="verified", devices=devices,
            scope="Per-device interval union, normalized by submitted calls; capture span includes inter-call gaps and is not per-call latency")
    summary = {"source": str(Path(path).resolve()), "submitted_calls": calls,
            "device_timeline": timeline,
            "device_tasks": sum(row["tasks"] for row in rows),
            "all_device_tasks_sum_ms_per_call":
                sum(row["sum_duration_ms"] for row in rows) / calls,
            "counter_values_present": any(row["counters_as_exported"] for row in rows),
            "kernel_groups_total": len(rows), "kernel_groups": rows[:32],
            "scope": "Profiler-active device task durations; overlapping tasks are summed. "
                     "Not API wall time, and not QK/softmax/PV phase attribution."}
    if all_numeric_fields:
        summary["exported_columns"] = columns
        summary["extra_numeric_scope"] = "Raw exported fields only; units/denominators not inferred. Metadata is not proof a requested metric was collected."
    return summary


METRIC_NAMES = ("PipeUtilization", "Memory", "MemoryAccess", "L2Cache", "ResourceConflictRatio")


def _profiler_options(profiler, report, metric="PipeUtilization"):
    key = "pipe_metrics" if metric == "PipeUtilization" else "hardware_metrics"
    # Level0 silently resets AI Core metrics. Explicit Level1 is required.
    config_type = getattr(profiler, "_ExperimentalConfig", None)
    metrics = getattr(profiler, "AiCMetrics", None)
    levels = getattr(profiler, "ProfilerLevel", None)
    if (config_type is None or not hasattr(metrics, metric)
            or not hasattr(levels, "Level1")):
        report[key] = {"requested": False, "reason": "API unavailable"}
        return {}
    kwargs = {"profiler_level": levels.Level1, "aic_metrics": getattr(metrics, metric),
              "l2_cache": metric == "L2Cache"}
    # Prefer CSV/Text explicitly on versions supporting selectable export format.
    export_type = getattr(profiler, "ExportType", None)
    if export_type is not None and hasattr(export_type, "Text"):
        kwargs["export_type"] = export_type.Text
    try:
        config = config_type(**kwargs)
    except (TypeError, ValueError) as exc:
        report[key] = {"requested": False, "reason": "Config rejected", **_error(exc)}
        return {}
    report[key] = {"requested": True, "level": "Level1",
                              "metric": metric,
                              "note": "Requested does not prove counters were collected; inspect CSV values."}
    return {"experimental_config": config}


def capture_native_traces(calls, *, torch, torch_npu, directory, repeats=3, metric="PipeUtilization", all_numeric_fields=False):
    """Capture warmed callables without adding per-call synchronization.

    Profiling is optional. Missing profiler/export support is returned as a
    failed capture and never invalidates previous uninstrumented wall timings.
    Each caller-supplied callable defines its own scope, such as native attention
    or preprocessing. The collector does not add numerical validation/transfers.
    Actual callable/synchronization failures raise NativeTraceExecutionError
    with the partial report; they must not be presented as successful execution.
    """
    if metric not in METRIC_NAMES:raise ValueError("Unsupported hardware metric: " + str(metric))
    if type(all_numeric_fields) is not bool:raise ValueError("all_numeric_fields must be boolean")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "repeats": repeats, "directory": str(directory),
              "variants": {}, "sources": SOURCES,
              "schedule": {"wait": 0, "warmup": 0, "active": 1, "repeat": 1,
                           "skip_first": 0, "unit": "whole callable batch"},
              "note": "Separate profiler-active captures after API timing. Each supplied "
                      "callable defines its scope (native attention or preprocessing). "
                      "Collector adds no warmup, numerical validation or output transfer."}

    def save():
        (directory / "capture_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    profiler = getattr(torch_npu, "profiler", None)
    activity = getattr(profiler, "ProfilerActivity", None)
    if (profiler is None or not callable(getattr(profiler, "profile", None))
            or not callable(getattr(profiler, "schedule", None))
            or not callable(getattr(profiler, "tensorboard_trace_handler", None))
            or not hasattr(activity, "CPU") or not hasattr(activity, "NPU")):
        report.update(status="failed", error="torch_npu CPU/NPU profiler API unavailable")
        save()
        print("Optional NPU trace failed: torch_npu profiler API unavailable; API timings preserved", flush=True)
        return report
    options = _profiler_options(profiler, report, metric)
    if metric != "PipeUtilization" and not report.get("hardware_metrics", {}).get("requested"):
        report.update(status="failed", error="Explicit hardware metric unavailable or configuration rejected", metric=metric)
        save();return report
    if metric != "PipeUtilization" or all_numeric_fields:
        report.update(metric=metric, all_numeric_fields=all_numeric_fields)
    save()
    for index, (name, call) in enumerate(calls.items()):
        label = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))[:80]
        target = Path(tempfile.mkdtemp(prefix=f"{index:02d}_{label}_", dir=directory))
        row = {"status": "running", "directory": str(target), "submitted_calls": 0,
               "capture_finalized": False}
        report["variants"][name] = row
        print(f"NPU trace: {name}; {repeats} calls; profiler active", flush=True)
        save()
        stage = "pre_capture_synchronize"
        try:
            torch.npu.synchronize()
            stage = "profiler_setup"
            handler = profiler.tensorboard_trace_handler(str(target))
            callback = {"completed": False}
            def on_trace_ready(*args, **kwargs):
                handler(*args, **kwargs)
                callback["completed"] = True
            # One profiler step contains the complete batch. Explicitly finish
            # RECORD_AND_SAVE -> NONE after its final device synchronization;
            # exiting a default RECORD window causes incomplete-data warnings.
            schedule = profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0)
            context = profiler.profile(
                activities=[activity.CPU, activity.NPU], record_shapes=False,
                profile_memory=False, with_stack=False, schedule=schedule,
                on_trace_ready=on_trace_ready, **options)
            if not callable(getattr(context, "step", None)):
                raise RuntimeError("torch_npu profiler lacks step(); cannot finalize a scheduled capture")
            with context:
                stage = "call_execution"
                for _ in range(repeats):
                    call()
                    row["submitted_calls"] += 1
                stage = "post_capture_synchronize"
                torch.npu.synchronize()
                stage = "profiler_finalize"
                context.step()
                # Some profiler releases suppress exceptions inside step().
                # A returned step alone does not establish a completed export.
                if not callback["completed"]:
                    raise RuntimeError("Scheduled profiler step did not complete on_trace_ready")
                row["capture_finalized"] = True
            stage = "export_summary"
            tables = list(target.rglob("kernel_details.csv"))
            traces = list(target.rglob("trace_view.json"))
            row["trace_files"] = [str(path) for path in traces[:8]]
            row["kernel_table_files"] = [str(path) for path in tables[:8]]
            if len(tables) == 1:
                row["device_summary"] = summarize_kernel_csv(tables[0], repeats, all_numeric_fields=all_numeric_fields)
                row["status"] = "captured"
            elif len(tables) > 1:
                row.update(status="raw_only", reason="Multiple kernel tables; refusing possible duplicate durations")
            else:
                row.update(status="raw_only", reason="No kernel_details.csv; no device timings inferred")
            if not traces:
                # Some older releases support only explicit Chrome export.
                exporter = getattr(context, "export_chrome_trace", None)
                if callable(exporter):
                    chrome = target / "chrome_trace.json"
                    try:
                        exporter(str(chrome))
                        if chrome.is_file():
                            row["trace_files"].append(str(chrome))
                    except Exception as exc:
                        row["chrome_export_error"] = _error(exc)
            if not tables and not row["trace_files"]:
                row.update(status="failed", reason="Profiler produced no readable device table or Chrome trace")
        except Exception as exc:
            row.update(status="failed", failed_stage=stage, error=_error(exc))
            report["status"] = "failed"
            save()
            print(f"NPU trace {name}: failed at {stage}: {exc}; API timings preserved", flush=True)
            if stage in ("call_execution", "pre_capture_synchronize", "post_capture_synchronize"):
                raise NativeTraceExecutionError(f"{name}: {stage} failed: {exc}", report) from exc
            # An optional capture failure is explicit; do not retry device work
            # or continue another profiler after a failed stop/export operation.
            return report
        summary = row.get("device_summary")
        if summary:
            print(f"NPU trace {name}: summed device tasks "
                  f"{summary['all_device_tasks_sum_ms_per_call']:.4f} ms/call; "
                  f"{summary['device_tasks']} device tasks/{repeats} calls; "
                  f"counter values present={summary['counter_values_present']}", flush=True)
        else:
            print(f"NPU trace {name}: {row['status']}: {row.get('reason', '')}; {target}", flush=True)
        save()
    statuses = [row["status"] for row in report["variants"].values()]
    report["status"] = ("failed" if not statuses or "failed" in statuses else
                        "raw_only" if "raw_only" in statuses else "captured")
    save()
    return report
