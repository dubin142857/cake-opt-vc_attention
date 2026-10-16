"""CPU fixtures for optional NPU capture failure and device-only timing semantics."""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mindie_laser_quant_trace as trace


def write_table(path, rows=None, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or ["Name", "Type", "Task Type", "Duration(us)", "aic_mac_ratio"]
    rows = rows or [
        ["VcLaserQuantAttention", "VcLaserQuantAttention", "AI_CORE", "80000", ".4"],
        ["VcLaserQuantAttention", "VcLaserQuantAttention", "AI_VECTOR", "40000", "N/A"],
        ["VcLaserQuantAttention", "VcLaserQuantAttention", "AI_CORE", "100000", ".8"],
        ["VcLaserQuantAttention", "VcLaserQuantAttention", "AI_VECTOR", "60000", "N/A"],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def test_summary_preserves_task_types_and_does_not_call_sum_elapsed_wall(tmp_path):
    source = tmp_path / "kernel_details.csv"
    write_table(source)
    report = trace.summarize_kernel_csv(source, 2)
    assert report["device_tasks"] == 4
    assert report["all_device_tasks_sum_ms_per_call"] == 140
    assert report["counter_values_present"]
    assert len(report["kernel_groups"]) == 2
    cube = report["kernel_groups"][0]
    assert cube["task_type"] == "AI_CORE"
    assert cube["sum_duration_ms_per_call"] == 90
    assert cube["counters_as_exported"]["aic_mac_ratio"]["median"] == pytest.approx(.6)
    assert "overlapping tasks are summed" in report["scope"]


@pytest.mark.parametrize("duration", ["nan", "inf", "-1", "missing"])
def test_summary_rejects_unusable_device_durations(tmp_path, duration):
    source = tmp_path / "kernel_details.csv"
    write_table(source, [["kernel", duration]], ["Name", "Duration(us)"])
    with pytest.raises(ValueError):
        trace.summarize_kernel_csv(source, 1)


def test_summary_never_guesses_units_or_uses_cpu_duration(tmp_path):
    source = tmp_path / "kernel_details.csv"
    write_table(source, [["kernel", "1000"]], ["Name", "Self CPU Time Total"])
    with pytest.raises(ValueError, match="no timing inferred"):
        trace.summarize_kernel_csv(source, 1)


class FakeProfiler:
    ProfilerActivity = SimpleNamespace(CPU="cpu", NPU="npu")
    AiCMetrics = SimpleNamespace(PipeUtilization="pipes")
    ProfilerLevel = SimpleNamespace(Level1="level1")
    ExportType = SimpleNamespace(Text="text")

    def __init__(self, events, *, files="csv", enter_error=False, exit_error=False):
        self.events, self.files = events, files
        self.enter_error, self.exit_error = enter_error, exit_error
        self.options = []
        self.configs = []

    def _ExperimentalConfig(self, **options):
        self.configs.append(options)
        return options

    def tensorboard_trace_handler(self, directory):
        def handler():
            folder = Path(directory) / "run_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
            if self.files == "csv":
                write_table(folder / "kernel_details.csv")
            elif self.files == "duplicate":
                write_table(folder / "kernel_details.csv")
                write_table(folder / "other" / "kernel_details.csv")
            if self.files != "none":
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "trace_view.json").write_text('{"traceEvents": []}')
        return handler

    def profile(self, **options):
        self.options.append(options)
        outer = self

        class Context:
            def __enter__(self):
                outer.events.append("start")
                if outer.enter_error:
                    raise RuntimeError("profiling unavailable")
                return self

            def __exit__(self, typ, exc, tb):
                outer.events.append("stop")
                if outer.exit_error:
                    raise RuntimeError("export failed")
                options["on_trace_ready"]()

        return Context()


def fake_runtime(events, **kwargs):
    profiler = FakeProfiler(events, **kwargs)
    torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: events.append("sync")))
    return torch, SimpleNamespace(profiler=profiler)


def test_capture_has_no_per_call_sync_and_keeps_host_events_out_of_summary(tmp_path):
    events = []
    torch, npu = fake_runtime(events)
    result = trace.capture_native_traces({"one": lambda: events.append("call")},
                                        torch=torch, torch_npu=npu, directory=tmp_path, repeats=2)
    assert events == ["sync", "start", "call", "call", "sync", "stop"]
    assert result["status"] == "captured"
    assert result["variants"]["one"]["device_summary"]["all_device_tasks_sum_ms_per_call"] == 140
    assert npu.profiler.options[0]["activities"] == ["cpu", "npu"]
    assert not npu.profiler.options[0]["record_shapes"]
    assert not npu.profiler.options[0]["profile_memory"]
    assert npu.profiler.configs == [{"profiler_level": "level1", "aic_metrics": "pipes",
                                      "l2_cache": False, "export_type": "text"}]
    assert json.loads((tmp_path / "capture_report.json").read_text())["status"] == "captured"


def test_missing_api_never_invokes_native_and_reports_failed(tmp_path):
    result = trace.capture_native_traces({"bad": lambda: pytest.fail("called native")},
                                        torch=None, torch_npu=SimpleNamespace(), directory=tmp_path)
    assert result["status"] == "failed"


def test_missing_metrics_api_does_not_claim_counters_requested(tmp_path):
    events = []
    torch, npu = fake_runtime(events)
    npu.profiler.AiCMetrics = None
    result = trace.capture_native_traces({"native": lambda: None}, torch=torch,
                                        torch_npu=npu, directory=tmp_path)
    assert not result["pipe_metrics"]["requested"]
    assert "experimental_config" not in npu.profiler.options[0]


@pytest.mark.parametrize("files,expected", [("trace", "raw_only"), ("duplicate", "raw_only"),
                                            ("none", "failed")])
def test_ambiguous_or_missing_device_output_is_not_successful_timing(tmp_path, files, expected):
    events = []
    torch, npu = fake_runtime(events, files=files)
    result = trace.capture_native_traces({"native": lambda: None}, torch=torch,
                                        torch_npu=npu, directory=tmp_path)
    assert result["status"] == expected
    assert "device_summary" not in result["variants"]["native"]


def test_actual_kernel_failure_propagates_with_persisted_partial_report(tmp_path):
    events = []
    torch, npu = fake_runtime(events)

    def fail():
        raise RuntimeError("LaunchAscendKernel ret 507000")

    with pytest.raises(trace.NativeTraceExecutionError, match="507000") as caught:
        trace.capture_native_traces({"native": fail}, torch=torch, torch_npu=npu, directory=tmp_path)
    assert caught.value.report["status"] == "failed"
    assert caught.value.partial_report is caught.value.report
    assert caught.value.report["variants"]["native"]["failed_stage"] == "call_execution"
    assert json.loads((tmp_path / "capture_report.json").read_text())["status"] == "failed"


def test_async_device_error_at_final_sync_propagates(tmp_path):
    events = []
    torch, npu = fake_runtime(events)
    count = [0]

    def sync():
        count[0] += 1
        if count[0] == 2:
            raise RuntimeError("device execution failed")

    torch.npu.synchronize = sync
    with pytest.raises(trace.NativeTraceExecutionError) as caught:
        trace.capture_native_traces({"native": lambda: None}, torch=torch,
                                    torch_npu=npu, directory=tmp_path)
    assert caught.value.report["variants"]["native"]["failed_stage"] == "post_capture_synchronize"


def test_unavailable_capture_does_not_report_success_or_run_next_variant(tmp_path):
    events = []
    torch, npu = fake_runtime(events, enter_error=True)
    result = trace.capture_native_traces({"first": lambda: events.append("call"),
                                         "second": lambda: events.append("call")},
                                        torch=torch, torch_npu=npu, directory=tmp_path)
    assert result["status"] == "failed"
    assert list(result["variants"]) == ["first"]
    assert "call" not in events


def test_new_capture_never_reuses_stale_csv(tmp_path):
    events = []
    torch, npu = fake_runtime(events)
    first = trace.capture_native_traces({"native": lambda: None}, torch=torch,
                                       torch_npu=npu, directory=tmp_path)
    npu.profiler.files = "none"
    second = trace.capture_native_traces({"native": lambda: None}, torch=torch,
                                        torch_npu=npu, directory=tmp_path)
    assert first["variants"]["native"]["directory"] != second["variants"]["native"]["directory"]
    assert second["status"] == "failed"
