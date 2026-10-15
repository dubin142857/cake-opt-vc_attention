"""Run the real one-command Bash flow with CPU-only subprocess stand-ins."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing")
    if not bash.is_file():
        pytest.skip("Bash/Git Bash required")
    root = tmp_path / "project with spaces"
    scripts, tools = root / "scripts", root / "tools"
    scripts.mkdir(parents=True)
    tools.mkdir()
    script = scripts / "run_mindie_laser_expcast.sh"
    script.write_bytes((ROOT / "scripts/run_mindie_laser_expcast.sh").read_bytes())
    stub = root / "mock_python.py"
    stub.write_text('''import json, os, pathlib, sys
args = sys.argv[1:]
if args[0] == '-u': args = args[1:]
if args[0] == '-':
    exec(sys.stdin.read())
    raise SystemExit(0)
record = {'script': args[0], 'args': args[1:], 'pid': os.getpid(),
          'cann_env_loaded': os.environ.get('LASER_TEST_ENV_LOADED')}
with open('calls.jsonl', 'a', encoding='utf-8') as stream:
    stream.write(json.dumps(record) + '\\n')
if args[0].endswith('check_ascend910b.py'):
    target = pathlib.Path(args[args.index('--output') + 1])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'runtime_soc': os.environ.get('LASER_TEST_SOC', 'Ascend910_9382')}))
if args[0].endswith('build_mindie_laser_local.py') and os.environ.get('LASER_TEST_FAIL'):
    source = args[args.index('--source-dir') + 1]
    if source.endswith('mindie_laser_expcast_source'):
        raise SystemExit(23)
''', encoding="utf-8")
    wrapper = tools / "mock-python"
    wrapper.write_text(f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "{stub.as_posix()}" "$@"\n',
                       encoding="utf-8", newline="\n")
    wrapper.chmod(0o755)
    for name in ("git", "cmake", "g++", "make"):
        target = tools / name
        target.write_text("#!/bin/sh\nexit 88\n", encoding="utf-8", newline="\n")
        target.chmod(0o755)
    cann = root / "cann"
    (cann / "bin").mkdir(parents=True)
    (cann / "bin/setenv.bash").write_text("export LASER_TEST_ENV_LOADED=1\n", encoding="utf-8")

    def run(extra_env=None, args=()):
        env = {key: value for key, value in os.environ.items()
               if key not in ("BASH_ENV", "ENV", "LASER_BUILD_JOBS", "LASER_PHASE_PROBE")}
        env.update(ASCEND_HOME_PATH=cann.as_posix(), PYTHON_BIN=wrapper.as_posix())
        env.update(extra_env or {})
        # Convert the fixture directory to Bash's PATH representation first.
        command = [str(bash), "--noprofile", "--norc", "-c",
                   'export PATH="$(cd "$1" && pwd -P):$PATH"; shift; exec bash "$@"',
                   "launcher-test", tools.as_posix(), script.as_posix(), *args]
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=30)
        trace = root / "calls.jsonl"
        calls = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()] if trace.exists() else []
        return result, calls

    return run


def option(call, name):
    return call["args"][call["args"].index(name) + 1]


def test_default_builds_two_isolated_variants_and_one_shared_bridge(launcher):
    result, calls = launcher()
    assert result.returncode == 0, result.stdout + result.stderr
    preps = [call for call in calls if call["script"].endswith("prepare_mindie_laser_source.py")]
    builds = [call for call in calls if call["script"].endswith("build_mindie_laser_local.py")]
    assert [option(call, "--variant") for call in preps] == ["baseline", "expcast"]
    assert len(builds) == 2
    assert all(option(call, "--jobs") == "4" for call in builds)
    assert all(option(call, "--soc") == "ascend910_93" for call in builds)
    assert option(builds[0], "--source-dir") != option(builds[1], "--source-dir")
    assert option(builds[0], "--install-dir") != option(builds[1], "--install-dir")
    assert "--offline" in preps[1]["args"]
    assert option(preps[1], "--reference-dir") == option(preps[0], "--source-dir")
    assert sum(call["script"].endswith("build_mindie_laser_bridge.py") for call in calls) == 1
    compare = calls[-1]
    assert compare["script"].endswith("benchmark_mindie_laser_expcast.py")
    assert option(compare, "--baseline-build") == "results/mindie_laser_build.json"
    assert option(compare, "--expcast-build") == "results/mindie_laser_expcast_build.json"
    assert "--baseline-probe-build" not in compare["args"]
    assert all(call["cann_env_loaded"] == "1" for call in calls)
    assert len({call["pid"] for call in calls}) == len(calls)


def test_optional_probes_use_two_more_isolated_builds_and_manifest_arguments(launcher):
    result, calls = launcher({"LASER_PHASE_PROBE": "1", "LASER_BUILD_JOBS": "2", "LASER_TEST_SOC": "Ascend910B4"})
    assert result.returncode == 0, result.stdout + result.stderr
    preps = [call for call in calls if call["script"].endswith("prepare_mindie_laser_source.py")]
    builds = [call for call in calls if call["script"].endswith("build_mindie_laser_local.py")]
    assert [option(call, "--variant") for call in preps] == ["baseline", "expcast", "baseline_probe", "expcast_probe"]
    assert len(builds) == 4
    assert len({option(call, "--source-dir") for call in builds}) == 4
    assert len({option(call, "--install-dir") for call in builds}) == 4
    assert all(option(call, "--jobs") == "2" for call in builds)
    assert all(option(call, "--soc") == "ascend910b" for call in builds)
    assert all("--offline" in call["args"] for call in preps[1:])
    assert option(calls[-1], "--baseline-probe-build") == "results/mindie_laser_baseline_probe_build.json"
    assert option(calls[-1], "--expcast-probe-build") == "results/mindie_laser_expcast_probe_build.json"


def test_candidate_failure_through_tee_stops_before_bridge_or_comparison(launcher):
    result, calls = launcher({"LASER_TEST_FAIL": "1"})
    assert result.returncode == 23
    assert not any(call["script"].endswith("build_mindie_laser_bridge.py") for call in calls)
    assert not any("benchmark_" in call["script"] for call in calls)


@pytest.mark.parametrize("env", [{"LASER_BUILD_JOBS": "0"}, {"LASER_PHASE_PROBE": "maybe"}])
def test_invalid_resource_or_probe_option_fails_before_preflight(launcher, env):
    result, calls = launcher(env)
    assert result.returncode == 2
    assert calls == []


def test_unsupported_device_does_not_start_preparation(launcher):
    result, calls = launcher({"LASER_TEST_SOC": "Ascend950"})
    assert result.returncode != 0
    assert len(calls) == 1 and calls[0]["script"].endswith("check_ascend910b.py")
