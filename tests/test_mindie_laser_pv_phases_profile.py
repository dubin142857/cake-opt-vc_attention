"""Production/PV diagnostic separation, exact gates, isolation and one-click flow."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import profile_mindie_laser_pv_phases as profile
import mindie_laser_pv_records as records


def providers(tmp_path):
    return {name: SimpleNamespace(vendor_root=str(tmp_path/name/"vendors"/name),
        bridge_path=str(tmp_path/name/"bridge.so"), marker_env=f"VC_{name.upper()}_MARKER_FILE")
        for name in (profile.ORIGINAL, profile.PRODUCTION, profile.PROBE)}


def test_defaults_select_validated_v_only_and_one_new_diagnostic():
    args = profile.parser().parse_args([])
    assert args.shape == [1, 28, 37741, 128]
    assert (args.warmup, args.repeats, args.trials, args.samples) == (5, 10, 9, 5)
    assert args.production_build == "results/mindie_laser_quant_vscale_pv_prefetch_build.json"
    assert args.probe_build == "results/mindie_laser_cube_pv_probe_build.json"
    assert args.output == "results/mindie_laser_pv_phases.json"


def test_resolve_uses_original_v_only_and_probe_unique_identities(monkeypatch, tmp_path):
    args = profile.parser().parse_args([])
    args._run_dir = str(tmp_path)
    def resolve(options, name):
        selected = "original" if name == "baseline" else Path(options.quant_build).stem
        return SimpleNamespace(vendor_root=str(tmp_path/selected), selected=selected)
    monkeypatch.setattr(profile.quant, "build_arguments", resolve)
    group = profile.resolve_providers(args)
    assert tuple(group) == (profile.ORIGINAL, profile.PRODUCTION, profile.PROBE)
    assert group[profile.PRODUCTION].selected == "mindie_laser_quant_vscale_pv_prefetch_build"
    assert group[profile.PROBE].selected == "mindie_laser_cube_pv_probe_build"
    assert len({r.marker_env for r in group.values()}) == 3
    assert len({r.marker_token for r in group.values()}) == 3


def test_environment_allows_only_three_verified_providers(tmp_path):
    group = providers(tmp_path)
    env = profile.worker_environment(dict(ASCEND_CUSTOM_OPP_PATH="/foreign/vendors/extra",
        LD_PRELOAD="/foreign/hook.so", LD_AUDIT="/foreign/audit.so", VC_OTHER_MARKER_FILE="foreign"), group, tmp_path)
    assert env["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep) == [r.vendor_root for r in group.values()]
    assert "LD_PRELOAD" not in env and "LD_AUDIT" not in env and "VC_OTHER_MARKER_FILE" not in env
    assert env[group[profile.PROBE].marker_env].endswith("pv_probe_markers.jsonl")
    group[profile.PROBE].vendor_root = group[profile.PRODUCTION].vendor_root
    with pytest.raises(ValueError, match="distinct"):
        profile.worker_environment({}, group, tmp_path)


def test_production_calls_exclude_probe_and_preserve_independent_operands(monkeypatch):
    import mindie_laser_fp16_trace_worker
    log = []
    class Adapter:
        def __init__(self, name): self.name = name
        def __call__(self, *xs): log.append((self.name, "full", xs))
        def run_prepared(self, p): log.append((self.name, "prepared", p))
        def run_native(self, p): log.append((self.name, "native", p))
    monkeypatch.setattr(mindie_laser_fp16_trace_worker, "native_call", lambda a, p: a.run_native(p))
    original, current = Adapter(profile.ORIGINAL), Adapter(profile.PRODUCTION)
    payloads = {profile.ORIGINAL: object(), profile.PRODUCTION: object()}
    inputs = (object(), object(), object())
    calls = profile.make_production_calls(original, current, payloads, inputs)
    for scope, group in calls.items():
        assert tuple(group) == (profile.ORIGINAL, profile.PRODUCTION)
        for call in group.values(): call()
    assert (profile.ORIGINAL, "native", payloads[profile.ORIGINAL]) in log
    assert (profile.PRODUCTION, "native", payloads[profile.PRODUCTION]) in log
    assert (profile.PRODUCTION, "full", inputs) in log


def test_deferred_modes_are_queued_before_trace_reads_and_exact_checks(monkeypatch):
    log, rows = [], []
    class Adapter:
        def __init__(self): self.value = torch.zeros(1, 1, 2, 2)
        def run_native(self, p): log.append("production"); return self.value.clone()
    production = Adapter()
    def call(native, payload, mode):
        log.append(mode)
        return production.value.clone(), object()
    monkeypatch.setattr(profile, "probe_call", call)
    monkeypatch.setattr(profile, "checked_trace", lambda *a, **k: log.append("read") or [])
    checked = profile.DeferredProbeAdapter(production, object(), rows)
    checked.run_native(object())
    production.value.fill_(2)
    checked.run_native(object())
    assert log == ["production", 0, 1, 2, "production", 0, 1, 2]
    checked.flush()
    assert len(rows) == 6 and all(r["exact_match"] for r in rows)
    checked.run_native(object())
    checked.pending[0][2]["issue"][0][..., -1, -1] += .001
    with pytest.raises(AssertionError, match="ALL padded"):
        checked.flush()


class FailingProbeRuntime:
    """CPU tensors plus explicit fake asynchronous-runtime failure boundaries."""
    def __init__(self, failure=None, mode=0):
        self.failure, self.target_mode = failure, mode
        self.mode, self.phase = "production", "smoke"
        self.error = RuntimeError("first device timeout 507014")
        self.poisoned, self.after_failure = False, []
        self.events = []
        self.payload = SimpleNamespace(q=torch.zeros(1, 2, 256, 128, dtype=torch.int8),
            k=torch.zeros(1, 2, 256, 128, dtype=torch.int8), v=torch.zeros(1, 2, 256, 128),
            q_scale=torch.ones(1, 2, 2), k_scale=torch.ones(1, 2, 2),
            shape=(1, 2, 1, 128), keys=1, scale=128**-.5, dtype=torch.float16)

    def touch(self, operation):
        event = (self.phase, operation, self.mode)
        if self.poisoned:
            self.after_failure.append(event)
            raise AssertionError("second NPU access after first failure")
        self.events.append(event)
        if operation == self.failure and self.mode == self.target_mode:
            self.poisoned = True
            raise self.error

    def prepare(self, *inputs):
        self.mode = "production"
        self.touch("prepare")
        from mindie_laser_vscale_reference import MAPPING_CASES, _mapping_inputs
        case = next(case for case in MAPPING_CASES if case["name"] == "tiny")
        expected = _mapping_inputs(case, torch.float16)
        assert tuple(inputs[0].shape) == (1, 2, 1, 128)
        assert all(torch.equal(a, b) for a, b in zip(inputs, expected))
        return self.payload

    def run_native(self, payload):
        self.mode = "production"
        self.touch("native")
        return torch.zeros_like(payload.q, dtype=torch.float32)

    def native(self, *args):
        self.mode = args[-1]
        self.touch("native")
        runtime, mode = self, self.mode
        class Trace(torch.Tensor):
            @staticmethod
            def __new__(cls):
                return torch.Tensor._make_subclass(cls, torch.zeros(records.TRACE_SHAPE, dtype=torch.int64), False)
            def cpu(self):
                runtime.mode = mode
                runtime.touch("trace_cpu")
                return self.as_subclass(torch.Tensor)
        return torch.zeros_like(args[0], dtype=torch.float32), Trace()

    def synchronize(self):
        self.touch("sync")


@pytest.mark.parametrize("failure", ("native", "sync", "trace_cpu"))
@pytest.mark.parametrize("mode", (0, 1, 2))
def test_tiny_smoke_localizes_each_failure_and_never_enters_queued_gates(monkeypatch, failure, mode):
    import mindie_laser_vscale_reference
    import mindie_laser_pv_prefetch_checks
    monkeypatch.setattr(mindie_laser_vscale_reference, "validate_vector_scale",
                        lambda *a: pytest.fail("Queued gates must not run after smoke failure"))
    monkeypatch.setattr(mindie_laser_pv_prefetch_checks, "validate_pv_prefetch",
                        lambda *a, **kw: pytest.fail("Boundary gates must not run after smoke failure"))
    runtime = FailingProbeRuntime(failure, mode)
    report, saved = {}, []
    save = lambda: saved.append(json.loads(json.dumps(report)))
    with pytest.raises(RuntimeError) as caught:
        profile.validate_probe(runtime, runtime.native, "cpu", report, save, synchronize=runtime.synchronize)
    assert caught.value is runtime.error
    assert runtime.after_failure == []
    rows = report["probe_tiny_smoke_validation"]
    assert [row["mode"] for row in rows] == list(profile.MODES)[:mode]
    error = report["probe_validation_failure"]
    assert error["mode"] == list(profile.MODES)[mode] and error["isolated"]
    assert error["shape_bnsd"] == [1, 2, 1, 128] and error["keys"] == 1
    assert error["stage"] == {"native": "about_to_launch", "sync": "awaiting_sync", "trace_cpu": "trace_cpu"}[failure]
    assert saved[-1]["probe_validation_failure"]["error"] == "RuntimeError: first device timeout 507014"
    assert error["further_device_access_aborted"]
    assert [event[2] for event in runtime.events if event[1] == "native" and event[2] != "production"] == list(range(mode+1))


@pytest.mark.parametrize("failure", ("native", "trace_cpu", "external_sync"))
def test_finally_checkpoint_preserves_first_failure_without_reaccessing_pending_outputs(monkeypatch, failure):
    import mindie_laser_vscale_reference
    import mindie_laser_pv_prefetch_checks
    runtime = FailingProbeRuntime()
    report, checkpoints = {}, []
    def validate(adapters, device, report, save):
        checkpoints.append(save)
        runtime.phase = "queued"
        if failure != "external_sync": runtime.failure, runtime.target_mode = failure, 0
        try:
            adapters["reuse"].run_native(runtime.payload)
            if failure == "external_sync":
                runtime.failure, runtime.target_mode = "sync", 2
                runtime.synchronize()
            save()  # trace.cpu failure occurs while flushing pending modes
        finally:
            save()  # The original validator's error path must be host-only.
            save()  # Repeated checkpoints remain host-only too.
    monkeypatch.setattr(mindie_laser_vscale_reference, "validate_vector_scale", validate)
    monkeypatch.setattr(mindie_laser_pv_prefetch_checks, "validate_pv_prefetch",
                        lambda *a, **kw: pytest.fail("Do not continue after a failed queue"))
    with pytest.raises(RuntimeError) as caught:
        profile.validate_probe(runtime, runtime.native, "cpu", report,
            lambda: json.dumps(report), synchronize=runtime.synchronize)
    assert caught.value is runtime.error
    assert not runtime.after_failure
    before = list(runtime.events)
    checkpoints[0]()  # also safe outside exception unwinding once aborted
    assert runtime.events == before
    assert report["probe_validation_failure"]["further_device_access_aborted"]
    assert not report["probe_validation_failure"]["isolated"]
    assert "does not identify" in report["probe_validation_failure"]["attribution"]


def test_tiny_smoke_does_not_add_synchronization_to_existing_queued_checks(monkeypatch):
    import mindie_laser_vscale_reference
    import mindie_laser_pv_prefetch_checks
    runtime = FailingProbeRuntime()
    def validate(adapters, device, report, save):
        runtime.phase = "queued"
        adapters["reuse"].run_native(runtime.payload)
        adapters["reuse"].run_native(runtime.payload)
        assert not any(op == "trace_cpu" for phase, op, mode in runtime.events if phase == "queued")
        save()
    monkeypatch.setattr(mindie_laser_vscale_reference, "validate_vector_scale", validate)
    monkeypatch.setattr(mindie_laser_pv_prefetch_checks, "validate_pv_prefetch", lambda *a, **kw: None)
    report = {}
    profile.validate_probe(runtime, runtime.native, "cpu", report, lambda: None, synchronize=runtime.synchronize)
    assert [mode for phase, op, mode in runtime.events if op == "sync"] == ["production", 0, 1, 2]
    assert len(report["probe_exact_validation"]) == 6
    assert len(report["probe_tiny_smoke_validation"]) == 3


def test_failure_report_io_error_cannot_replace_original_device_error():
    runtime = FailingProbeRuntime("sync", 1)
    report = {}
    def save():
        if "probe_validation_failure" in report:
            raise OSError("report write failed")
    guard = profile.ProbeValidationGuard(report, save)
    with pytest.raises(RuntimeError) as caught:
        profile.validate_tiny_probe_modes(runtime, runtime.native, "cpu", guard, runtime.synchronize)
    assert caught.value is runtime.error
    assert not runtime.after_failure
    assert report["probe_failure_report_error"] == "OSError: report write failed"


def test_checked_trace_enforces_output_and_trace_abi():
    payload = SimpleNamespace(q=torch.zeros(1, 1, 256, 128, dtype=torch.int8),
                              shape=(1, 1, 1, 64), keys=1)
    output = torch.zeros_like(payload.q, dtype=torch.float32)
    trace = torch.zeros(records.TRACE_SHAPE, dtype=torch.int64)
    assert profile.checked_trace((output, trace), payload, 0) == []
    for malformed in ((output.half(), trace), (output[..., :-1], trace),
                      (output, trace.float()), (output, trace[..., :-1]), (output,)):
        with pytest.raises(RuntimeError): profile.checked_trace(malformed, payload, 0)
    trace[0, 0, 0, 0] = records.MAGIC
    with pytest.raises(ValueError, match="plain trace"):
        profile.checked_trace((output, trace), payload, 0)


def test_primary_table_reports_only_production_ratios_and_dense_error():
    from mindie_laser_quant_pairing import run_paired, paired_comparison
    elapsed = [0.]
    durations = {profile.ORIGINAL: .083, profile.PRODUCTION: .078}
    def call(name): elapsed[0] += durations[name]
    timing = run_paired({name: lambda n=name: call(n) for name in durations}, synchronize=lambda: None,
        repeats=1, trials=9, warmup=0, clock=lambda: elapsed[0])
    report = dict(production_timings={"full": timing},
        production_comparisons={"full": paired_comparison(timing, profile.ORIGINAL, profile.PRODUCTION)},
        dense_fp32_samples={profile.ORIGINAL: {"relative_rmse": .002}, profile.PRODUCTION: {"relative_rmse": .014}})
    text = profile.production_table(report)
    assert "1.0641x" in text and "1.400000" in text and "9/9" in text
    assert "pv_probe " not in text and "detail " not in text
    assert "diagnostic modes are excluded" in text
    pair = paired_comparison(timing, profile.ORIGINAL, profile.PRODUCTION)
    text = profile.calibration_table({"calibration": {"plain -> detail": pair}})
    assert "Probe overhead, not optimization speedup" in text


@pytest.mark.parametrize("child", [{"status": "passed", "returncode": 2},
                                  {"status": "failed", "returncode": 0},
                                  {"status": "missing_report", "returncode": 0}])
def test_parent_never_accepts_missing_or_failed_diagnostic(monkeypatch, tmp_path, child):
    args = profile.parser().parse_args(["--output", str(tmp_path/"report.json")])
    monkeypatch.setattr(profile, "resolve_providers", lambda a: providers(tmp_path))
    monkeypatch.setattr(profile, "verify_builds", lambda *a: {})
    monkeypatch.setattr(profile, "worker_environment", lambda *a: {})
    monkeypatch.setattr(profile, "launch", lambda *a: child)
    assert profile.parent(args) == 1
    report = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["workers"]["paired"] == child


def test_launch_passes_every_manifest_and_calibration_setting(monkeypatch, tmp_path):
    args = profile.parser().parse_args(["--device", "2"])
    args._run_dir = str(tmp_path)
    class Process:
        stdout = []
        def wait(self): return 7
    monkeypatch.setattr(profile.subprocess, "Popen", lambda *a, **kw: Process())
    env = {k: str(tmp_path/k) for k in ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME")}
    result = profile.launch(args, env, tmp_path/"local")
    command = result["command"]
    for key, value in profile.MANIFESTS.items():
        assert command[command.index("--"+key.replace("_", "-"))+1] == value
    assert command[command.index("--trials")+1] == "9"
    assert command[command.index("--samples")+1] == "5"
    assert command[command.index("--device")+1] == "2"
    assert result["returncode"] == 7 and result["status"] == "missing_report"


@pytest.mark.parametrize("failure", [None, "preflight", "build", "profile"])
def test_one_click_only_builds_pv_probe_and_stops_on_error(tmp_path, failure):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing")
    if not bash.is_file(): pytest.skip("Bash unavailable")
    root = tmp_path/"project with spaces"
    for part in ("scripts", "results", "cann"): (root/part).mkdir(parents=True)
    for name in ("mindie_laser_quant_vscale_pv_prefetch_build", "mindie_laser_quant_preprocess_build",
                 "mindie_laser_build", "mindie_laser_bridge"):
        (root/"results"/(name+".json")).write_text("{}")
    driver = root/"scripts/run_mindie_laser_pv_phases.sh"
    driver.write_bytes((ROOT/"scripts/run_mindie_laser_pv_phases.sh").read_bytes())
    stub = root/"stub.py"
    stub.write_text("import json,sys\nfrom pathlib import Path\na=sys.argv[1:]\n"
        "s='preflight' if a[0]=='-' else 'metadata' if a[0]=='-c' else 'build' if 'build_' in a[1] else 'profile'\n"
        "with Path('calls.jsonl').open('a') as f:f.write(json.dumps({'stage':s,'args':a})+'\\n')\n"
        "if s=='metadata':print('cached V prefetch source' if 'source_dir' in a[-1] else 'ascend910_93')\n"
        f"raise SystemExit(23 if s=={failure!r} else 0)\n", encoding="utf-8")
    wrapper = root/"mock-python"
    wrapper.write_text(f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "{stub.as_posix()}" "$@"\n',
                       encoding="utf-8", newline="\n")
    wrapper.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("BASH_ENV", "ENV")}
    env.update(PYTHON_BIN=wrapper.as_posix(), ASCEND_HOME_PATH=(root/"cann").as_posix(), NPU_DEVICE="2")
    result = subprocess.run([str(bash), "--noprofile", "--norc", str(driver)], cwd=root, env=env,
        capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == (23 if failure else 0), result.stdout+result.stderr
    calls = [json.loads(row) for row in (root/"calls.jsonl").read_text().splitlines()]
    if failure: assert calls[-1]["stage"] == failure
    else:
        assert [c["stage"] for c in calls] == ["preflight", "metadata", "metadata", "build", "profile"]
        build, measure = calls[-2]["args"], calls[-1]["args"]
        assert build[1] == "scripts/build_mindie_laser_cube_probe.py"
        assert build[build.index("--kind")+1] == "pv"
        assert build[build.index("--reference-dir")+1] == "cached V prefetch source"
        assert measure[1] == "scripts/profile_mindie_laser_pv_phases.py"
        assert "--trace" not in measure
