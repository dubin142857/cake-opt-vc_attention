"""Cube diagnostic routing, output gates and offline entrypoint behavior."""
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
import profile_mindie_laser_cube_phases as profile
from mindie_laser_cube_records import decode_trace, summarize_records
from tests.test_mindie_laser_cube_records import empty_trace, record, put


@pytest.mark.parametrize("family", ["int8", "vscale", "fp16"])
def test_native_argument_routing_and_output_order(family):
    payload = SimpleNamespace(**{key: object() for key in ("q", "k", "v", "q_scale", "k_scale")},
                              scale=.125, keys=37741, shape=(1, 28, 37741, 128), pre_tokens=147)
    payload.tensors = (payload.q, payload.k, payload.v)
    seen = []
    result = (object(), object())
    def native(*args):
        seen.append(args)
        return result
    assert profile.baseline_call(SimpleNamespace(native=native), payload, family) is result
    assert profile.probe_call(native, payload, family, 37741, 2) is result
    if family != "fp16":
        assert seen[0] == (payload.q, payload.k, payload.v, payload.q_scale, payload.k_scale, .125, 37741, False)
        assert seen[1] == seen[0] + (2,)
        assert profile.baseline_output(result, family) is result
    else:
        assert seen[0] == (*payload.tensors, None, None, None, .125, 28, "BNSD", 1., 147, 1, True)
        assert seen[1] == (*payload.tensors, .125, 37741, 2)
        assert profile.baseline_output(result, family) is result[1]


@pytest.mark.parametrize("family", ["int8", "vscale", "fp16"])
def test_trace_contract_requires_both_long_shape_stages(family):
    q = torch.zeros(1, 1, 256, 128, dtype=torch.int8 if family != "fp16" else torch.float16)
    payload = SimpleNamespace(q=q, tensors=(q, q, q))
    out = q.float()
    trace = torch.tensor(empty_trace(), dtype=torch.int64)
    assert profile.checked_trace((out, trace), payload, family, 0, require=True) == []
    with pytest.raises(ValueError, match="required Cube group"):
        profile.checked_trace((out, trace), payload, family, 2, require=True)
    data = put(empty_trace(), record(stage=0))
    with pytest.raises(ValueError, match="both QK and PV"):
        profile.checked_trace((out, torch.tensor(data)), payload, family, 2, require=True)
    put(data, record(stage=1))
    assert len(profile.checked_trace((out, torch.tensor(data)), payload, family, 2, require=True)) == 2
    for bad in ((out.half(), trace), (out, trace.float()), (out, trace[:1]), out):
        with pytest.raises(RuntimeError):
            profile.checked_trace(bad, payload, family, 0)


def test_summary_table_distinguishes_work_and_unmeasured_fields():
    summaries = {}
    for name, mode in (("issue", 1), ("completed", 2)):
        trace = put(empty_trace(), record(stage=0, mode=mode))
        put(trace, record(stage=1, mode=mode))
        summaries[name] = summarize_records([decode_trace(trace, mode)])
    table = profile.sample_table(summaries)
    assert "Kblks" in table and "count" in table and "empty" in table
    assert "NOT production phase percentages" in table and "NOT pure FIX" in table
    assert "issue/QK/first" in table and "completed/PV/first" in table
    assert table.split("issue/QK/first")[1].splitlines()[0].count("--") == 3


@pytest.mark.parametrize("family", ["int8", "vscale", "fp16"])
def test_changed_padded_output_stops_before_timing(monkeypatch, family):
    original_randn = torch.randn
    monkeypatch.setattr(torch, "randn", lambda shape, **kw: original_randn((1, 1, 2, 2), **kw))
    monkeypatch.setattr(profile, "checked_trace", lambda *a, **k: [])
    class Adapter:
        def prepare(self, q, k, v):
            return SimpleNamespace(q=q, k=k, v=v, q_scale=q, k_scale=k, tensors=(q, k, v),
                                   scale=1., keys=2, shape=q.shape, pre_tokens=0)
        def native(self, q, *args):
            return q.float() if family != "fp16" else (None, q.float())
    def bad(q, *args):
        return q.float() + (1 if args[-1] == 2 else 0), None
    report = {}
    with pytest.raises(AssertionError, match="changed full padded native output"):
        profile.validate_modes(Adapter(), bad, family, "cpu", report, lambda: None)
    assert report["structured_validation"][-1]["mode"] == 2
    assert report["structured_validation"][-1]["exact_match"] is False


def test_second_worker_failure_preserves_first_result(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "verify_family", lambda *a: {"verified": True})
    monkeypatch.setattr(profile, "resolve_providers", lambda *a: {})
    monkeypatch.setattr(profile, "launch", lambda args, family, providers:
        dict(returncode=0 if family == "int8" else 9, status="passed" if family == "int8" else "failed",
             worker_log=family+".log", measurement=87.2))
    output = tmp_path / "report.json"
    assert profile.parent(SimpleNamespace(output=output)) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["workers"]["int8"]["measurement"] == 87.2
    assert report["workers"]["fp16"]["returncode"] == 9
    assert "fp16.log" in report["error"]


def test_parent_import_does_not_initialize_runtime():
    result = subprocess.run([sys.executable, "-c",
        "import sys;sys.path.insert(0,'scripts');import profile_mindie_laser_cube_phases;"
        "assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fail_stage", [None, "preflight", "int8", "fp16", "profile"])
def test_shell_reuses_baselines_and_stops_on_failure(tmp_path, fail_stage):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing")
    if not bash.is_file():
        pytest.skip("Bash required")
    root = tmp_path / "space in project"
    for name in ("scripts", "results", "cann"):
        (root / name).mkdir(parents=True)
    for name in ("mindie_laser_quant_reuse_build", "mindie_laser_quant_preprocess_build", "mindie_laser_build", "mindie_laser_bridge"):
        (root / "results" / (name+".json")).write_text("{}")
    script = root / "scripts/run_mindie_laser_cube_phases.sh"
    script.write_bytes((ROOT / "scripts/run_mindie_laser_cube_phases.sh").read_bytes())
    stub = root / "stub.py"
    stub.write_text("import json,sys\nfrom pathlib import Path\n"
        "a=sys.argv[1:]\ns='preflight' if a[0]=='-' else 'cache' if a[0]=='-c' else a[a.index('--kind')+1] if '--kind' in a else 'profile'\n"
        "with Path('calls.jsonl').open('a') as f:f.write(json.dumps({'stage':s,'args':a})+'\\n')\n"
        "if s=='cache':print('local reuse source' if 'source_dir' in a[-1] else 'ascend910_93')\n"
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
    else:
        assert [c["stage"] for c in calls if c["stage"] != "cache"] == ["preflight", "int8", "fp16", "profile"]
    for call in calls:
        a = call["args"]
        if call["stage"] in ("int8", "fp16"):
            assert a[:2] == ["-u", "scripts/build_mindie_laser_cube_probe.py"]
            assert a[a.index("--reference-dir")+1] == "local reuse source"
        if call["stage"] == "profile":
            assert a[:2] == ["-u", "scripts/profile_mindie_laser_cube_phases.py"]
            assert a[a.index("--shape")+1:a.index("--shape")+5] == ["1", "28", "37741", "128"]
            assert a[a.index("--device")+1] == "2"
