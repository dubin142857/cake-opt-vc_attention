"""Run the public Bash entry point using host-only Python/CANN stubs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run_script(tmp_path):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing")
    if not bash.is_file():
        pytest.skip("Bash required")
    root = tmp_path/"project with spaces"
    for part in ("scripts", "results", "cached CANN/bin"):
        (root/part).mkdir(parents=True)
    cann = root/"cached CANN"
    environment = cann/"bin/setenv.bash"
    environment.write_text('''if [[ "${ASCEND_HOME_PATH:-}" != "$TEST_CANN_ROOT" ]]; then
  echo "CANN root was not exported before setenv" >&2
  exit 29
fi
export TEST_CANN_READY=yes
export ASCEND_HOME_PATH=/simulated-setenv-override
''', encoding="utf-8", newline="\n")
    manifest = root/"results/mindie_laser_quant_vscale_pv_prefetch_build.json"
    manifest.write_text(json.dumps({"cann_root":cann.as_posix()}), encoding="utf-8")
    driver = root/"scripts/run_mindie_laser_msopprof.sh"
    driver.write_bytes((ROOT/"scripts"/driver.name).read_bytes())
    stub = root/"mock.py"
    stub.write_text('''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "-c":
    stage = "metadata"
elif args[:2] == ["-u", "scripts/collect_mindie_laser_msopprof.py"]:
    stage = "collector"
else:
    raise SystemExit("Unexpected command; no build/download/probe allowed: " + repr(args))
with Path("calls.jsonl").open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(dict(stage=stage, args=args, cann=os.environ.get("ASCEND_HOME_PATH"),
                                ready=os.environ.get("TEST_CANN_READY"))) + "\\n")
if os.environ.get("TEST_FAIL_STAGE") == stage:
    raise SystemExit(23)
if stage == "metadata":
    exec(args[1])
else:
    print("collector complete")
''', encoding="utf-8")
    python = root/"mock python"
    python.write_text(f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "{stub.as_posix()}" "$@"\n',
        encoding="utf-8", newline="\n")
    python.chmod(0o755)

    def run(*, settings=None, fail=None, missing_env=False, missing_manifest=False, arguments=()):
        if missing_env:
            environment.unlink()
        if missing_manifest:
            manifest.unlink()
        env = {key:value for key,value in os.environ.items() if key not in
            ("BASH_ENV", "ENV", "TEST_FAIL_STAGE", "TEST_CANN_READY", "NPU_DEVICE", "LASER_MSOPPROF_EXPCAST")}
        env.update(PYTHON_BIN=python.as_posix(), TEST_CANN_ROOT=cann.as_posix(),
                   ASCEND_HOME_PATH="/unrelated/toolkit", PYTHONIOENCODING="utf-8")
        env.update(settings or {})
        if fail:
            env["TEST_FAIL_STAGE"] = fail
        result = subprocess.run([str(bash), "--noprofile", "--norc", str(driver), *arguments],
            cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
        trace = root/"calls.jsonl"
        calls = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()] if trace.exists() else []
        return result, calls, root
    return run


def test_defaults_two_variants_fixed_shape_no_build_or_probe(run_script):
    result, calls, root = run_script()
    assert result.returncode == 0, result.stdout+result.stderr
    assert [c["stage"] for c in calls] == ["metadata", "collector"]
    call = calls[1]
    assert call["args"] == ["-u", "scripts/collect_mindie_laser_msopprof.py", "--device", "0",
        "--shape", "1", "28", "37741", "128", "--variants", "original_laser", "vector_pv_prefetch",
        "--output", "results/mindie_laser_msopprof.json"]
    # The setenv stub also fails unless this root was exported BEFORE sourcing.
    assert Path(call["cann"]) == root/"cached CANN"
    assert call["ready"] == "yes"


def test_optional_expcast_appends_two_modes_and_preserves_device(run_script):
    result, calls, _ = run_script(settings={"LASER_MSOPPROF_EXPCAST":"1", "NPU_DEVICE":"2"})
    assert result.returncode == 0, result.stdout+result.stderr
    args = calls[-1]["args"]
    assert args[args.index("--device")+1] == "2"
    assert args[args.index("--variants")+1:args.index("--output")] == [
        "original_laser", "vector_pv_prefetch", "fp16_expcast_control", "fp16_expcast"]
    assert len(calls) == 2


@pytest.mark.parametrize("stage", ["metadata", "collector"])
def test_failures_propagate_without_retry(run_script, stage):
    result, calls, _ = run_script(fail=stage)
    assert result.returncode == 23, result.stdout+result.stderr
    assert calls[-1]["stage"] == stage
    assert sum(call["stage"] == stage for call in calls) == 1


def test_missing_recorded_environment_stops_before_collector(run_script):
    result, calls, _ = run_script(missing_env=True)
    assert result.returncode == 1
    assert [call["stage"] for call in calls] == ["metadata"]
    assert "Recorded CANN environment is missing" in result.stderr


def test_missing_build_manifest_stops_before_collector(run_script):
    result, calls, _ = run_script(missing_manifest=True)
    assert result.returncode != 0
    assert [call["stage"] for call in calls] == ["metadata"]


@pytest.mark.parametrize("settings", [{"NPU_DEVICE":"-1"}, {"LASER_MSOPPROF_EXPCAST":"yes"}])
def test_bad_settings_rejected_before_python(run_script, settings):
    result, calls, _ = run_script(settings=settings)
    assert result.returncode == 2 and calls == []


def test_arguments_rejected_before_python(run_script):
    result, calls, _ = run_script(arguments=("--probe",))
    assert result.returncode == 2 and calls == []
