"""The recovery entry point reuses production builds without launching probes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = (
    "mindie_laser_quant_vscale_build.json",
    "mindie_laser_quant_vscale_pv_prefetch_build.json",
    "mindie_laser_quant_preprocess_build.json",
    "mindie_laser_build.json",
    "mindie_laser_bridge.json",
)


@pytest.fixture
def stable_runner(tmp_path):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing")
    if not bash.is_file():
        pytest.skip("Bash/Git Bash required")
    root = tmp_path / "project with spaces"
    for part in ("scripts", "results", "cached CANN/bin"):
        (root / part).mkdir(parents=True)
    cann = root / "cached CANN"
    (cann / "bin/setenv.bash").write_text("export LASER_TEST_CANN_LOADED=1\n", encoding="utf-8", newline="\n")
    for name in MANIFESTS:
        (root / "results" / name).write_text(json.dumps({"cann_root": cann.as_posix()}), encoding="utf-8")
    driver = root / "scripts/run_mindie_laser_stable.sh"
    driver.write_bytes((ROOT / "scripts/run_mindie_laser_stable.sh").read_bytes())
    stub = root / "mock_python.py"
    stub.write_text('''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
stage = "metadata" if args[0] == "-c" else "profile"
record = dict(stage=stage, args=args, cann=os.environ.get("ASCEND_HOME_PATH"),
              cann_loaded=os.environ.get("LASER_TEST_CANN_LOADED"))
with Path("calls.jsonl").open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record) + "\\n")
if os.environ.get("LASER_TEST_FAIL") == stage:
    raise SystemExit(23)
if stage == "metadata":
    sys.argv = ["-c", *args[2:]]
    exec(args[1])
else:
    assert args[:2] == ["-u", "scripts/profile_mindie_laser_pv_prefetch.py"], args
    print("production comparison completed")
''', encoding="utf-8")
    wrapper = root / "mock python"
    wrapper.write_text(f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "{stub.as_posix()}" "$@"\n',
                       encoding="utf-8", newline="\n")
    wrapper.chmod(0o755)

    def run(extra_env=None, args=(), missing=None):
        if missing:
            (root / "results" / missing).unlink()
        env = {k: v for k, v in os.environ.items() if k not in
               ("BASH_ENV", "ENV", "ASCEND_HOME_PATH", "NPU_DEVICE", "LASER_TEST_FAIL", "LASER_TEST_CANN_LOADED")}
        env.update(PYTHON_BIN=wrapper.as_posix(), PYTHONIOENCODING="utf-8")
        env.update(extra_env or {})
        result = subprocess.run([str(bash), "--noprofile", "--norc", str(driver), *args],
            cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        trace = root / "calls.jsonl"
        calls = [json.loads(row) for row in trace.read_text(encoding="utf-8").splitlines()] if trace.exists() else []
        return result, calls, root

    return run


@pytest.mark.parametrize("device", [None, "2"])
def test_recovery_reuses_cached_cann_and_only_runs_production_comparison(stable_runner, device):
    result, calls, root = stable_runner({"NPU_DEVICE": device} if device is not None else {})
    assert result.returncode == 0, result.stdout + result.stderr
    assert [call["stage"] for call in calls] == ["metadata", "profile"]
    profile = calls[-1]
    assert Path(profile["cann"]) == root / "cached CANN"
    assert profile["cann_loaded"] == "1"
    args = profile["args"]
    for option, value in (("--device", device or "0"), ("--warmup", "5"), ("--repeats", "10"),
                          ("--trials", "9"), ("--output", "results/mindie_laser_stable.json")):
        assert args[args.index(option) + 1] == value
    assert args[args.index("--shape") + 1:args.index("--shape") + 5] == ["1", "28", "37741", "128"]
    assert "--trace" not in args
    assert args[args.index("--candidate") + 1] == "pv_prefetch"
    assert "production comparison completed" in (root / "results/mindie_laser_stable.log").read_text()


@pytest.mark.parametrize("stage", ["metadata", "profile"])
def test_failed_metadata_or_comparison_is_not_hidden_by_tee(stable_runner, stage):
    result, calls, _ = stable_runner({"LASER_TEST_FAIL": stage})
    assert result.returncode == 23, result.stdout + result.stderr
    assert calls[-1]["stage"] == stage
    assert len(calls) == (1 if stage == "metadata" else 2)


@pytest.mark.parametrize("missing", MANIFESTS)
def test_missing_production_build_stops_before_python_or_automatic_rebuild(stable_runner, missing):
    result, calls, _ = stable_runner(missing=missing)
    assert result.returncode != 0 and missing in result.stdout + result.stderr
    assert calls == []


@pytest.mark.parametrize("extra_env,args", [({}, ("--trace",)), ({"NPU_DEVICE": "-1"}, ()),
                                          ({"NPU_DEVICE": "not-a-device"}, ())])
def test_invalid_options_stop_before_loading_runtime(stable_runner, extra_env, args):
    result, calls, _ = stable_runner(extra_env, args=args)
    assert result.returncode == 2
    assert calls == []
