"""Diagnostic scope, failure preservation and one-command offline workflow."""
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
sys.path.insert(0, str(ROOT / "scripts"))
import profile_mindie_laser_phases as profile


def test_probe_call_preserves_objects_and_disables_expcast():
    payload = SimpleNamespace(**{key: object() for key in ("q", "k", "v", "q_scale", "k_scale")},
                              scale=.125, keys=37741)
    seen = []
    result = object()
    def native(*args):
        seen.append(args)
        return result
    assert profile.probe_call(native, payload, 2) is result
    assert seen == [(payload.q, payload.k, payload.v, payload.q_scale, payload.k_scale,
                     .125, 37741, False, 2)]
    assert profile.reuse_call(native, payload) is result
    assert seen[-1] == seen[0][:-1]


def test_phase_table_labels_raw_ticks_and_separates_positions():
    from mindie_laser_phase_records import summarize_records, decode_trace
    from tests.test_mindie_laser_phase_records import record, empty_trace, put
    trace = empty_trace()
    put(trace, record())
    put(trace, record(role=1, slot=5))
    summary = summarize_records([decode_trace(trace, 2)])
    output = profile.phase_table({"phases": summary})
    assert "system-counter" in output and "max/history" in output
    assert "not measure the other core" in output and "production percentages" in output


def test_trace_shape_dtype_and_empty_mode_contract():
    payload = SimpleNamespace(q=torch.zeros(1, 1, 2, 64, dtype=torch.int8))
    out = torch.zeros(payload.q.shape, dtype=torch.float32)
    trace = torch.zeros((4, 3, 8, 24), dtype=torch.int64)
    assert profile.checked_trace((out, trace), payload, 0) == []
    with pytest.raises(ValueError):
        profile.checked_trace((out, trace), payload, 1, require=True)
    for result in (out, (out, trace.float()), (out, trace[:, :, :1]), (out.half(), trace)):
        with pytest.raises(RuntimeError):
            profile.checked_trace(result, payload, 0)


def test_structured_checks_fail_on_changed_output(monkeypatch):
    original_randn = torch.randn
    monkeypatch.setattr(torch, "randn", lambda shape, **kw: original_randn((1, 1, 2, 2), **kw))
    monkeypatch.setattr(profile, "checked_trace", lambda *a, **k: [])
    class Old:
        def prepare(self, q, k, v):
            return SimpleNamespace(q=q, k=k, v=v, q_scale=q, k_scale=k, scale=1., keys=2)
        def run_native(self, payload): return payload.q.float()
    def bad(q, k, v, qs, ks, scale, keys, expcast, mode):
        return q.float() + (1 if mode == 2 else 0), None
    report = {}
    with pytest.raises(AssertionError, match="changed native output"):
        profile.validate_modes(Old(), bad, "cpu", report, lambda: None)
    assert report["structured_validation"][-1]["exact_match"] is False
    assert report["structured_validation"][-1]["mode"] == 2


def test_preflight_failure_is_saved_without_launch(tmp_path, monkeypatch):
    def fail(args): raise ValueError("stale diagnostic build")
    monkeypatch.setattr(profile, "verify_builds", fail)
    monkeypatch.setattr(profile.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    output = tmp_path / "report.json"
    assert profile.parent(SimpleNamespace(output=output)) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed" and "stale diagnostic build" in report["error"]


def test_parent_import_does_not_initialize_runtime():
    result = subprocess.run([sys.executable, "-c",
        "import sys;sys.path.insert(0,'scripts');import profile_mindie_laser_phases;"
        "assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fail_stage", [None, "preflight", "build", "profile"])
def test_shell_builds_only_probe_and_preserves_failure(tmp_path, fail_stage):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing")
    if not bash.is_file(): pytest.skip("Bash required")
    root = tmp_path / "space in project"
    for name in ("scripts", "results", "cann"):
        (root / name).mkdir(parents=True)
    for name in ("mindie_laser_quant_reuse_build", "mindie_laser_quant_preprocess_build"):
        (root / "results" / (name+".json")).write_text("{}")
    script = root / "scripts/run_mindie_laser_phases.sh"
    script.write_bytes((ROOT / "scripts/run_mindie_laser_phases.sh").read_bytes())
    stub = root / "stub.py"
    stub.write_text("import json,sys\nfrom pathlib import Path\n"
        "a=sys.argv[1:]\ns='preflight' if a[0]=='-' else 'cache' if a[0]=='-c' else 'build' if 'build_' in a[1] else 'profile'\n"
        "with Path('calls.jsonl').open('a') as f:f.write(json.dumps({'stage':s,'args':a})+'\\n')\n"
        "if s=='cache':print('local reuse source' if 'source_dir' in a[-1] else 'Ascend910_9382')\n"
        f"raise SystemExit(23 if s=={fail_stage!r} else 0)\n", encoding="utf-8")
    wrapper = root / "mock-python"
    wrapper.write_text(f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "{stub.as_posix()}" "$@"\n', encoding="utf-8", newline="\n")
    wrapper.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("BASH_ENV", "ENV")}
    env.update(PYTHON_BIN=wrapper.as_posix(), ASCEND_HOME_PATH=(root / "cann").as_posix(), NPU_DEVICE="2")
    result = subprocess.run([str(bash), "--noprofile", "--norc", str(script)], cwd=root, env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == (23 if fail_stage else 0), result.stdout+result.stderr
    calls = [json.loads(row) for row in (root / "calls.jsonl").read_text().splitlines()]
    if fail_stage:
        assert calls[-1]["stage"] == fail_stage
    for call in calls:
        a = call["args"]
        if call["stage"] == "build":
            assert a[:2] == ["-u", "scripts/build_mindie_laser_quant_probe.py"]
            assert a[a.index("--reference-dir")+1] == "local reuse source"
        if call["stage"] == "profile":
            assert a[:2] == ["-u", "scripts/profile_mindie_laser_phases.py"]
            assert a[a.index("--shape")+1:a.index("--shape")+5] == ["1", "28", "37741", "128"]
            assert a[a.index("--device")+1] == "2"
