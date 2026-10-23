"""ABI-only isolation, exact padded outputs, and host-only failure handling."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_mindie_laser_pv_abi as validation


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_module_and_help_do_not_import_torch_or_touch_device():
    code = (
        "import builtins, runpy, sys\n"
        "original = builtins.__import__\n"
        "def guarded(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in ('torch', 'torch_npu', 'mindiesd'):\n"
        "        raise AssertionError('Device dependency imported: '+name)\n"
        "    return original(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded\n"
        "sys.argv = ['validate_mindie_laser_pv_abi.py', '--help']\n"
        f"runpy.run_path({str(ROOT / 'scripts/validate_mindie_laser_pv_abi.py')!r}, run_name='__main__')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--worker-timeout" in result.stdout


@pytest.mark.parametrize("kind", validation.KINDS)
def test_defaults_select_two_distinct_providers_and_bounded_small_only_run(monkeypatch, tmp_path, kind):
    import benchmark_mindie_laser_quant as quant
    args = validation.parser().parse_args([] if kind == validation.ABI else ["--kind", kind])
    args._run_dir = str(tmp_path)
    assert args.worker_timeout == 180 and args.device == 0
    assert args.output == f"results/mindie_laser_{kind}.json"
    assert args.abi_build == f"results/mindie_laser_cube_{kind}_probe_build.json"
    assert validation.resolve_defaults(args) is args and args.kind == kind
    assert not hasattr(args, "shape") and not hasattr(args, "trials")
    selected = []
    def resolve(options, name):
        selected.append((options.quant_build, name))
        return SimpleNamespace(vendor_root=str(tmp_path / Path(options.quant_build).stem))
    monkeypatch.setattr(quant, "build_arguments", resolve)
    providers = validation.resolve_providers(args)
    assert tuple(providers) == (validation.PRODUCTION, kind)
    assert selected == [(args.production_build, "qk_int8_fused"), (args.abi_build, "qk_int8_fused")]
    assert len({p.marker_env for p in providers.values()}) == 2
    assert len({p.marker_token for p in providers.values()}) == 2
    monkeypatch.setattr(quant, "build_arguments", lambda *a: SimpleNamespace(vendor_root=str(tmp_path)))
    with pytest.raises(ValueError, match="distinct"):
        validation.resolve_providers(args)


@pytest.mark.parametrize("kind", validation.KINDS)
def test_explicit_paths_are_preserved_but_do_not_change_selected_identity(kind):
    args = validation.parser().parse_args(["--kind", kind, "--abi-build", "custom-build.json",
                                          "--output", "custom-result.json"])
    assert validation.resolve_defaults(args) is args
    assert args.kind == kind and args.abi_build == "custom-build.json"
    assert args.output == "custom-result.json"
    minimal = validation.resolve_defaults(SimpleNamespace(kind=kind))
    assert minimal.abi_build == f"results/mindie_laser_cube_{kind}_probe_build.json"
    with pytest.raises(ValueError, match="Unsupported isolation"):
        validation.resolve_defaults(SimpleNamespace(kind="pv"))


def test_old_instrumented_provider_cannot_be_selected_by_cli():
    with pytest.raises(SystemExit) as caught:
        validation.parser().parse_args(["--kind", "pv"])
    assert caught.value.code == 2


@pytest.mark.parametrize("kind,op", [("pv_abi", "VcLaserCubePvAbiProbeAttention"),
                                     ("pv_const", "VcLaserCubePvConstProbeAttention")])
def test_provenance_uses_exact_identities_and_source_validation(monkeypatch, kind, op):
    import benchmark_mindie_laser_quant as quant
    import prepare_mindie_laser_quant_vscale_pv_prefetch_source as production
    import prepare_mindie_laser_cube_probe_source as abi
    import mindie_laser_quant_fused as prep
    calls = []
    def provenance(provider, *, identity, source_validator):
        calls.append(identity)
        assert source_validator(Path("cache"), Path("manifest")) == identity["op_type"]
        return {"op_type": identity["op_type"]}
    monkeypatch.setattr(quant, "quant_provenance", provenance)
    monkeypatch.setattr(production, "validate_prepared_source", lambda *a: production.IDENTITY["op_type"])
    selected_kind = kind
    def check_abi(*args, kind):
        assert kind == selected_kind
        return abi.identity(kind)["op_type"]
    monkeypatch.setattr(abi, "validate_prepared_source", check_abi)
    monkeypatch.setattr(prep, "validate_preprocess_manifest", lambda path: {"path": path})
    args = validation.parser().parse_args(["--kind", kind])
    result = validation.verify_builds(args, {validation.PRODUCTION: object(), kind: object()})
    assert [c["op_type"] for c in calls] == ["VcLaserQuantVscalePvPrefetchAttention", op]
    assert result["preprocessor"]["path"] == args.preprocess_build


@pytest.mark.parametrize("abi", (False, True))
def test_native_schema_is_checked_including_outputs_and_mode(abi):
    suffix = ", int probe_mode" if abi else ""
    result = "(Tensor, Tensor)" if abi else "Tensor"
    schema = f"la(Tensor q, Tensor k, Tensor v, Tensor qs, Tensor ks, float scale, int keys, bool expcast{suffix}) -> {result}"
    native = SimpleNamespace(_schema=torch._C.parse_schema(schema))
    validation.check_schema(native, abi=abi)
    for invalid in (schema.replace("bool expcast", "int expcast"),
                    schema.replace(" -> " + result, " -> (Tensor, Tensor, Tensor)")):
        with pytest.raises(RuntimeError, match="schema"):
            validation.check_schema(SimpleNamespace(_schema=torch._C.parse_schema(invalid)), abi=abi)


class Runtime:
    """CPU tensors with explicit launch/sync/read failures and access tracking."""
    def __init__(self, failure=None, target=validation.ABI, kind=validation.ABI):
        self.kind = kind
        self.failure, self.target, self.mode = failure, target, "prepare"
        self.error = RuntimeError("first aicore timeout 507014")
        self.poisoned, self.after_failure = False, []
        self.events, self.prepared = [], []
        self.payload = None

    def touch(self, action):
        if self.poisoned:
            self.after_failure.append((self.mode, action))
            raise AssertionError("Second device access after failure")
        self.events.append((self.mode, action))
        if self.mode == self.target and action == self.failure:
            self.poisoned = True
            raise self.error

    def prepare(self, q, k, v):
        from mindie_laser_quant_backend import prepare_quant_tensors
        from mindie_laser_vscale_reference import _mapping_inputs
        self.mode = "prepare"
        self.touch("prepare")
        cases = validation.fixture_cases()
        case = cases[len(self.prepared) // 2]
        expected = _mapping_inputs(case, q.dtype)
        assert all(torch.equal(x, y) for x, y in zip((q, k, v), expected))
        self.payload = prepare_quant_tensors(q, k, v, v_quantize=False)
        self.prepared.append(self.payload)
        return self.payload

    def run_native(self, payload):
        assert payload is self.payload
        self.mode = validation.PRODUCTION
        self.touch("native")
        return torch.zeros_like(payload.q, dtype=torch.float32)

    def native(self, *args):
        self.mode = self.kind
        self.touch("native")
        for value, name in zip(args[:5], ("q", "k", "v", "q_scale", "k_scale")):
            assert value is getattr(self.payload, name)
        assert args[5:] == (self.payload.scale, self.payload.keys, False, 0)
        runtime = self
        class Trace(torch.Tensor):
            @staticmethod
            def __new__(cls):
                return torch.Tensor._make_subclass(cls, torch.zeros(validation.TRACE_SHAPE, dtype=torch.int64), False)
            def cpu(self):
                runtime.touch("trace_cpu")
                return self.as_subclass(torch.Tensor)
        return torch.zeros_like(args[0], dtype=torch.float32), Trace()

    def synchronize(self):
        self.touch("sync")


@pytest.mark.parametrize("kind", validation.KINDS)
def test_six_cases_start_with_original_failing_tiny_for_both_dtypes_and_share_payloads(kind):
    runtime, report, snapshots, markers = Runtime(kind=kind), {}, [], []
    validation.validate_cases(runtime, runtime.native, "cpu", report,
        lambda: snapshots.append(json.loads(json.dumps(report))), runtime.synchronize,
        after_call=lambda name, row: markers.append((name, row["name"], row["dtype"])), kind=kind)
    rows = report["cases"]
    assert [(r["queries"], r["keys"], r["dim"]) for r in rows] == [
        (1, 1, 128), (1, 1, 128), (17, 129, 64), (17, 129, 64), (65, 257, 128), (65, 257, 128)]
    assert all(r["batch"] == 1 and r["heads"] == 2 for r in rows)
    assert [r["dtype"] for r in rows] == ["torch.float16", "torch.bfloat16"] * 3
    assert all(r["status"] == "passed" and r["all_padded_output"]["exact_match"] and r["trace_all_zero"] for r in rows)
    assert rows[0]["all_padded_output"]["elements"] == 1 * 2 * 256 * 128
    expected_events = [("prepare", "prepare"), ("prepare", "sync"),
        (validation.PRODUCTION, "native"), (validation.PRODUCTION, "sync"),
        (kind, "native"), (kind, "sync"), (kind, "trace_cpu")]
    assert runtime.events == expected_events * 6
    assert [name for name, _, _ in markers] == [validation.PRODUCTION, kind] * 6
    assert any(s["active_probe_validation"]["stage"] == "awaiting_sync" for s in snapshots)


@pytest.mark.parametrize("target,failure,stage", [
    ("prepare", "prepare", "about_to_prepare"), ("prepare", "sync", "awaiting_sync"),
    (validation.PRODUCTION, "native", "about_to_launch"),
    (validation.PRODUCTION, "sync", "awaiting_sync"),
    (validation.ABI, "native", "about_to_launch"), (validation.ABI, "sync", "awaiting_sync"),
    (validation.ABI, "trace_cpu", "trace_and_all_output_cpu"),
])
@pytest.mark.parametrize("kind", validation.KINDS)
def test_first_error_preserved_and_failure_save_is_host_only(target, failure, stage, kind):
    target = kind if target == validation.ABI else target
    runtime, report, snapshots = Runtime(failure, target, kind), {}, []
    with pytest.raises(RuntimeError) as caught:
        validation.validate_cases(runtime, runtime.native, "cpu", report,
            lambda: snapshots.append(json.loads(json.dumps(report))), runtime.synchronize, kind=kind)
    assert caught.value is runtime.error
    assert runtime.after_failure == []
    error = report["probe_validation_failure"]
    assert error["mode"] == target and error["stage"] == stage and error["isolated"]
    assert error["shape_bnsd"] == [1, 2, 1, 128] and error["keys"] == 1
    assert error["further_device_access_aborted"]
    assert error["origin"] == validation.kind_config(kind)["failure_origin"]
    assert len(report["cases"]) == 1 and report["cases"][0]["status"] == "failed"
    assert snapshots[-1]["probe_validation_failure"]["error"] == "RuntimeError: first aicore timeout 507014"


def test_failure_save_io_error_cannot_replace_device_error():
    runtime, report = Runtime("sync"), {}
    def save():
        if "probe_validation_failure" in report:
            raise OSError("host report unwritable")
    with pytest.raises(RuntimeError) as caught:
        validation.validate_cases(runtime, runtime.native, "cpu", report, save, runtime.synchronize)
    assert caught.value is runtime.error and not runtime.after_failure
    assert report["probe_failure_report_error"] == "OSError: host report unwritable"


def test_zero_trace_and_all_padded_output_require_exact_finite_values():
    payload = SimpleNamespace(q=torch.zeros(1, 2, 256, 128, dtype=torch.int8))
    expected = torch.zeros_like(payload.q, dtype=torch.float32)
    trace = torch.zeros(validation.TRACE_SHAPE, dtype=torch.int64)
    result = validation.check_result((expected.clone(), trace), payload, expected)
    assert result["all_padded_output"]["elements"] == expected.numel()
    out = expected.clone()
    out[..., -1, -1] = 1e-8  # A padded tail element must not be trimmed away.
    with pytest.raises(AssertionError, match="ALL padded"):
        validation.check_result((out, trace), payload, expected)
    out[..., -1, -1] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        validation.check_result((out, trace), payload, expected)
    trace[-1, -1, -1, -1] = 1
    with pytest.raises(AssertionError, match="entirely zero"):
        validation.check_result((expected, trace), payload, expected)
    for invalid in ((expected.half(), trace), (expected[..., :-1], trace),
                    (expected, trace.float()), (expected, trace[..., :-1]), (expected,)):
        with pytest.raises(RuntimeError):
            validation.check_result(invalid, payload, expected)


def providers(tmp_path, kind=validation.ABI):
    return {name: SimpleNamespace(vendor_root=str(tmp_path / name / "vendors" / name),
        bridge_path=str(tmp_path / name / "bridge.so"), marker_env=f"VC_{name.upper()}_MARKER_FILE")
        for name in (validation.PRODUCTION, kind)}


@pytest.mark.parametrize("timed_out", (False, True))
@pytest.mark.parametrize("kind", validation.KINDS)
def test_launch_retains_partial_context_markers_logs_and_enforces_host_deadline(monkeypatch, tmp_path, timed_out, kind):
    args = validation.parser().parse_args(["--device", "2", "--kind", kind])
    args._run_dir = str(tmp_path)
    group = providers(tmp_path, kind)
    process_calls, signals = [], []
    partial = dict(status="running" if timed_out else "passed", active_probe_validation=dict(
        mode=kind, stage="awaiting_sync", shape_bnsd=[1, 2, 1, 128], keys=1))
    class Process:
        pid = 4242
        returncode = None
        waits = []
        def wait(self, timeout):
            self.waits.append(timeout)
            if len(self.waits) == 1 and timed_out:
                raise subprocess.TimeoutExpired("worker", timeout)
            self.returncode = -15 if timed_out else 0
            return self.returncode
        def poll(self): return self.returncode
        def terminate(self): signals.append("terminate")
        def kill(self): signals.append("kill")
    child = Process()
    def popen(command, **kwargs):
        process_calls.append((command, kwargs))
        directory = Path(args._run_dir) / "paired/local"
        kwargs["stdout"].write(f"{kind} awaiting sync\n".encode())
        (directory / "report.json").write_text(json.dumps(partial), encoding="utf-8")
        for provider in group.values():
            Path(kwargs["env"][provider.marker_env]).write_text("host marker retained\n")
        return child
    monkeypatch.setattr(validation.subprocess, "Popen", popen)
    if os.name == "posix":
        monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    result = validation.launch(args, group)
    assert len(process_calls) == 1 and child.waits[0] == 180
    command, kwargs = process_calls[0]
    for name in validation.MANIFESTS:
        assert command[command.index("--" + name.replace("_", "-")) + 1] == getattr(args, name)
    assert command[command.index("--kind") + 1] == kind
    assert command[command.index("--device") + 1] == "2"
    assert kwargs["env"]["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep) == [str(Path(p.vendor_root).resolve()) for p in group.values()]
    assert "mindie_laser_cube_pv_probe" not in " ".join(command)
    assert result["active_probe_validation"] == partial["active_probe_validation"]
    assert Path(result["worker_log"]).read_text() == f"{kind} awaiting sync\n"
    assert all(Path(kwargs["env"][p.marker_env]).read_text() == "host marker retained\n" for p in group.values())
    assert result["timed_out"] is timed_out
    if timed_out:
        assert result["status"] == "timed_out" and result["prior_worker_status"] == "running"
        assert result["termination"]["device_tasks_stopped"] is None
        assert child.waits == [180, 5] and signals
    else:
        assert result["status"] == "passed" and not signals


def test_host_termination_escalates_after_bounded_wait_without_claiming_device_stop(monkeypatch):
    actions = []
    class Process:
        pid = 4242
        def wait(self, timeout):
            actions.append(("wait", timeout))
            raise subprocess.TimeoutExpired("worker", timeout)
        def poll(self): return None
        def terminate(self): actions.append("terminate")
        def kill(self): actions.append("kill")
    if os.name == "posix":
        monkeypatch.setattr(os, "killpg", lambda pid, sig: actions.append(("signal", pid, sig)))
    result = validation.terminate_worker(Process())
    assert [x for x in actions if isinstance(x, tuple) and x[0] == "wait"] == [("wait", 5), ("wait", 5)]
    assert result["actions"] == ["terminate", "terminate_wait_timeout", "kill", "kill_wait_timeout"]
    assert result["host_returncode"] is None and result["device_tasks_stopped"] is None


@pytest.mark.parametrize("child", [dict(status="passed", returncode=2), dict(status="failed", returncode=0),
    dict(status="missing_report", returncode=0), dict(status="timed_out", returncode=-15, timed_out=True)])
@pytest.mark.parametrize("kind", validation.KINDS)
def test_parent_keeps_failure_report_and_never_retries_new_device_work(monkeypatch, tmp_path, child, kind):
    args = validation.parser().parse_args(["--kind", kind, "--output", str(tmp_path / "result.json")])
    launches = []
    monkeypatch.setattr(validation, "resolve_providers", lambda args: providers(tmp_path, kind))
    monkeypatch.setattr(validation, "verify_builds", lambda *args: {"offline": True})
    monkeypatch.setattr(validation, "launch", lambda *args: launches.append(args) or child)
    assert validation.parent(args) == 1
    result = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert len(launches) == 1 and result["no_additional_device_work"]
    assert result["workers"]["paired"] == child and result["status"] == "failed"
    assert result["kind"] == kind and kind in Path(result["run_dir"]).name


@pytest.mark.parametrize("timeout", ("0", "-1", "nan", "inf"))
def test_unbounded_or_invalid_timeouts_are_rejected_before_work(monkeypatch, timeout):
    monkeypatch.setattr(validation, "parent", lambda args: pytest.fail("Invalid limit must not start work"))
    with pytest.raises(SystemExit) as caught:
        validation.main(["--worker-timeout", timeout])
    assert caught.value.code == 2


def test_marker_first_call_is_required_and_later_cache_reuse_is_explicit(tmp_path):
    path = tmp_path / "markers.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no fresh"):
        validation.fresh_markers(path, 0, pid=5, token="abi")
    marker = dict(pid=5, token="abi", phase="tiling", tiling_key=0)
    path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    offset, first = validation.fresh_markers(path, 0, pid=5, token="abi")
    assert first["records"] == [marker] and first["new_records"] == 1
    assert not first["reused_first_call_evidence"]
    end, cached = validation.fresh_markers(path, offset, pid=5, token="abi")
    assert end == offset and cached["new_records"] == 0 and cached["reused_first_call_evidence"]
    second_marker = dict(marker, tiling_key=1)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(second_marker) + "\n")
    _, second = validation.fresh_markers(path, offset, pid=5, token="abi")
    assert second["records"] == [second_marker] and second["new_records"] == 1


@pytest.mark.parametrize("changed", [dict(token="production"), dict(pid=6), dict(phase="wrong"), dict(tiling_key=-1)])
def test_fresh_malformed_or_foreign_marker_is_not_treated_as_cache_reuse(tmp_path, changed):
    path = tmp_path / "markers.jsonl"
    first = dict(pid=5, token="abi", phase="tiling", tiling_key=0)
    path.write_text(json.dumps(first) + "\n", encoding="utf-8")
    offset, _ = validation.fresh_markers(path, 0, pid=5, token="abi")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(first, **changed)) + "\n")
    with pytest.raises(ValueError, match="Malformed or foreign"):
        validation.fresh_markers(path, offset, pid=5, token="abi")


def test_report_summary_real_host_deadline_does_not_invent_device_exception():
    child = dict(status="timed_out", prior_worker_status="running", timed_out=True,
        returncode=-15, worker_timeout_seconds=180, probe_validation_failure=None, traceback=None,
        active_probe_validation=dict(mode="pv_const", stage="awaiting_sync",
            shape_bnsd=[1, 2, 1, 128], keys=1, dtype="torch.float16", native_q_dtype="torch.int8", isolated=True),
        termination=dict(actions=["terminate", "terminate_wait_timeout", "kill"],
            host_returncode=-15, device_tasks_stopped=None),
        worker_log="retained/worker.log", report_path="retained/report.json")
    report = dict(status="failed", workers={"paired": child})
    original = json.dumps(report)
    summary = validation.report_summary(report)
    assert summary["status"] == "failed" and summary["worker_status"] == "timed_out"
    assert summary["timed_out"] is True and summary["returncode"] == -15
    assert summary["active"] == {key: child["active_probe_validation"][key]
        for key in ("mode", "stage", "shape_bnsd", "keys", "dtype")}
    assert summary["first_recorded_validation_error"] is None
    assert summary["host_deadline_seconds"] == 180
    assert summary["host_termination"]["device_tasks_stopped"] is None
    assert "not evidence of a device timeout" in summary["deadline_scope"]
    assert "507014" not in json.dumps(summary)
    assert summary["worker_log"] == "retained/worker.log"
    assert json.dumps(report) == original


def test_report_summary_retains_first_error_and_distinguishes_host_assertion():
    for error in ("RuntimeError: first aicore timeout 507014", "AssertionError: padded output mismatch"):
        report = dict(status="failed", timed_out=False, returncode=1, error=error,
            probe_validation_failure=dict(mode="pv_const", stage="trace_and_all_output_cpu", error=error),
            active_probe_validation=dict(mode="different_mode", stage="later_stage"))
        summary = validation.report_summary(report)
        assert summary["first_recorded_validation_error"] == error
        assert summary["active"]["mode"] == "pv_const"
        assert summary["timed_out"] is False and summary["returncode"] == 1
        assert "deadline_scope" not in summary


def test_launch_host_deadline_preserves_already_recorded_exception(monkeypatch, tmp_path):
    args = validation.parser().parse_args(["--kind", "pv_const"])
    args._run_dir = str(tmp_path)
    first = "RuntimeError: original launch exception"
    class Process:
        pid = 4242
        def wait(self, timeout):
            raise subprocess.TimeoutExpired("worker", timeout)
        def poll(self): return -15
    def popen(command, **kwargs):
        path = tmp_path / "paired/local/report.json"
        path.write_text(json.dumps(dict(status="failed", error=first,
            probe_validation_failure=dict(error=first, mode="pv_const", stage="about_to_launch"))))
        return Process()
    monkeypatch.setattr(validation.subprocess, "Popen", popen)
    monkeypatch.setattr(validation, "terminate_worker", lambda process:
        dict(actions=["terminate"], host_returncode=-15, device_tasks_stopped=None))
    report = validation.launch(args, providers(tmp_path, "pv_const"))
    assert report["error"] == first
    assert report["probe_validation_failure"]["error"] == first
    assert report["timed_out"] and report["prior_worker_status"] == "failed"
    assert "Host worker exceeded" in report["host_timeout_error"]


def test_summary_cli_reads_existing_report_without_device_imports_or_provenance(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(dict(status="failed", workers={"paired": dict(status="timed_out",
        timed_out=True, returncode=-15, worker_log="retained/worker.log")})), encoding="utf-8")
    before = {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in tmp_path.iterdir()}
    code = (
        "import builtins, runpy, sys\n"
        "original = builtins.__import__\n"
        "def guarded(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in ('torch', 'torch_npu', 'mindiesd') or "
        "name.startswith(('benchmark_mindie', 'prepare_mindie', 'profile_mindie', 'mindie_laser')):\n"
        "        raise AssertionError('Device/provenance dependency imported: '+name)\n"
        "    return original(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded\n"
        f"sys.argv = ['validate_mindie_laser_pv_abi.py', '--summarize-report', {str(path)!r}]\n"
        f"runpy.run_path({str(ROOT / 'scripts/validate_mindie_laser_pv_abi.py')!r}, run_name='__main__')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.split("\n", 1)[1])
    assert summary["worker_status"] == "timed_out" and summary["returncode"] == -15
    assert summary["worker_log"] == "retained/worker.log"
    assert {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in tmp_path.iterdir()} == before


@pytest.mark.parametrize("content", ("invalid JSON", "[]"))
def test_invalid_summary_report_fails_read_only_without_falling_back_to_launch(monkeypatch, tmp_path, content):
    path = tmp_path / "result.json"
    path.write_text(content)
    monkeypatch.setattr(validation, "parent", lambda args: pytest.fail("Summary cannot launch"))
    monkeypatch.setattr(validation, "worker", lambda args: pytest.fail("Summary cannot launch"))
    with pytest.raises(SystemExit) as caught:
        validation.main(["--summarize-report", str(path)])
    assert caught.value.code == 2 and path.read_text() == content


def test_parent_failure_prints_recorded_child_context(monkeypatch, tmp_path, capsys):
    args = validation.parser().parse_args(["--kind", "pv_const", "--output", str(tmp_path / "result.json")])
    child = dict(status="timed_out", timed_out=True, returncode=-15, worker_timeout_seconds=180,
        active_probe_validation=dict(mode="pv_const", stage="awaiting_sync", shape_bnsd=[1, 2, 1, 128], keys=1),
        worker_log="retained/worker.log")
    monkeypatch.setattr(validation, "resolve_providers", lambda args: providers(tmp_path, "pv_const"))
    monkeypatch.setattr(validation, "verify_builds", lambda *args: {})
    monkeypatch.setattr(validation, "launch", lambda *args: child)
    assert validation.parent(args) == 1
    output = capsys.readouterr().err
    assert '"timed_out": true' in output and '"returncode": -15' in output
    assert "awaiting_sync" in output and "retained/worker.log" in output
    assert '"first_recorded_validation_error": null' in output
