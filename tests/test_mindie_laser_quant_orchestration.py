"""Exercise the single Bash command with build/runtime subprocess stand-ins."""
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
    script = scripts / "run_mindie_laser_quant.sh"
    script.write_bytes((ROOT / "scripts/run_mindie_laser_quant.sh").read_bytes())
    stub = root / "mock_python.py"
    stub.write_text('''import json, os, pathlib, sys
args = sys.argv[1:]
if args[0] == '-u': args = args[1:]
if args[0] == '-':
    text = sys.stdin.read()
    if 'baseline.provenance' in text:
        raise SystemExit(0 if os.environ.get('LASER_TEST_REUSE') else 1)
    exec(text)
    raise SystemExit(0)
with open('calls.jsonl', 'a', encoding='utf-8') as stream:
    stream.write(json.dumps({'script': args[0], 'args': args[1:]}) + '\\n')
if args[0].endswith('check_ascend910b.py'):
    target = pathlib.Path(args[args.index('--output') + 1])
    target.write_text(json.dumps({'runtime_soc': os.environ.get('LASER_TEST_SOC', 'Ascend910_9382')}))
if args[0].endswith('build_mindie_laser_quant.py') and os.environ.get('LASER_TEST_FAIL'):
    raise SystemExit(23)
if args[0].endswith('build_mindie_laser_quant_preprocess.py') and os.environ.get('LASER_TEST_PREPROCESS_FAIL'):
    raise SystemExit(24)
if args[0].endswith('check_mindie_laser_quant_preprocess.py') and os.environ.get('LASER_TEST_SMOKE_FAIL'):
    raise SystemExit(25)
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
    cann.mkdir()

    def run(extra_env=None):
        env = {key: value for key, value in os.environ.items()
               if key not in ("BASH_ENV", "ENV", "LASER_BUILD_JOBS", "LASER_QUANT_EXPERIMENTS",
                              "LASER_TEST_REUSE", "LASER_TEST_FAIL", "LASER_TEST_PREPROCESS_FAIL",
                              "LASER_TEST_SMOKE_FAIL", "LASER_TEST_SOC", "NPU_DEVICE")}
        env.update(ASCEND_HOME_PATH=cann.as_posix(), PYTHON_BIN=wrapper.as_posix())
        env.update(extra_env or {})
        command = [str(bash), "--noprofile", "--norc", "-c",
                   'export PATH="$(cd "$1" && pwd -P):$PATH"; shift; exec bash "$@"',
                   "quant-launcher-test", tools.as_posix(), script.as_posix()]
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=30)
        trace = root / "calls.jsonl"
        calls = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()] if trace.exists() else []
        return result, calls
    return run


def test_one_command_builds_missing_baseline_and_quant_then_benchmarks(launcher):
    result, calls = launcher()
    assert result.returncode == 0, result.stdout + result.stderr
    assert [Path(call["script"]).name for call in calls] == [
        "check_ascend910b.py", "prepare_mindie_laser_source.py", "build_mindie_laser_local.py",
        "build_mindie_laser_bridge.py", "build_mindie_laser_quant.py",
        "build_mindie_laser_quant_preprocess.py", "check_mindie_laser_quant_preprocess.py",
        "benchmark_mindie_laser_quant.py"]
    assert calls[-4]["args"][calls[-4]["args"].index("--soc") + 1] == "ascend910_93"
    assert calls[-3]["args"][calls[-3]["args"].index("--soc") + 1] == "Ascend910_9382"
    assert calls[-2]["args"] == ["--build-manifest", "results/mindie_laser_quant_preprocess_build.json",
                                 "--device", "0", "--output", "results/mindie_laser_quant_preprocess_smoke.json"]
    assert "--quant-build" in calls[-1]["args"]
    args = calls[-1]["args"]
    assert args[args.index("--preprocess-build") + 1] == "results/mindie_laser_quant_preprocess_build.json"
    assert args[args.index("--variants") + 1:] == ["installed", "baseline", "qk_int8", "qk_int8_fused"]
    assert args[args.index("--output") + 1] == "results/mindie_laser_quant_compare.json"


@pytest.mark.parametrize("experiments,variants", [
    ("0", ["installed", "baseline", "qk_int8", "qk_int8_fused"]),
    ("1", ["installed", "baseline", "qk_int8", "qk_int8_fused", "qkv_int8_e4m3", "qkv_int8_e4m3_expcast"]),
])
def test_experiment_switch_controls_benchmark_variants(launcher, experiments, variants):
    result, calls = launcher({"LASER_TEST_REUSE": "1", "LASER_QUANT_EXPERIMENTS": experiments})
    assert result.returncode == 0, result.stdout + result.stderr
    args = calls[-1]["args"]
    assert args[args.index("--variants") + 1:] == variants
    assert args[args.index("--output") + 1] == "results/mindie_laser_quant_compare.json"


@pytest.mark.parametrize("experiments", ["", "2", "-1", "true"])
def test_invalid_experiment_switch_rejected_before_preflight(launcher, experiments):
    result, calls = launcher({"LASER_QUANT_EXPERIMENTS": experiments})
    assert result.returncode == 2 and calls == []
    assert "LASER_QUANT_EXPERIMENTS must be 0 or 1" in result.stderr


def test_verified_baseline_is_reused_without_recompilation_or_extra_benchmark(launcher):
    result, calls = launcher({"LASER_TEST_REUSE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert [Path(call["script"]).name for call in calls] == [
        "check_ascend910b.py", "build_mindie_laser_quant.py", "build_mindie_laser_quant_preprocess.py",
        "check_mindie_laser_quant_preprocess.py", "benchmark_mindie_laser_quant.py"]


def test_quant_build_failure_stops_pipeline_through_tee(launcher):
    result, calls = launcher({"LASER_TEST_REUSE": "1", "LASER_TEST_FAIL": "1"})
    assert result.returncode == 23
    assert all("benchmark_" not in call["script"] for call in calls)
    assert all("preprocess" not in call["script"] for call in calls)


def test_preprocess_build_failure_stops_without_fallback(launcher):
    result, calls = launcher({"LASER_TEST_REUSE": "1", "LASER_TEST_PREPROCESS_FAIL": "1"})
    assert result.returncode == 24
    assert Path(calls[-1]["script"]).name == "build_mindie_laser_quant_preprocess.py"
    assert all("benchmark_" not in call["script"] for call in calls)


def test_preprocess_smoke_failure_stops_before_any_benchmark_through_tee(launcher):
    result, calls = launcher({"LASER_TEST_REUSE": "1", "LASER_TEST_SMOKE_FAIL": "1", "NPU_DEVICE": "3"})
    assert result.returncode == 25
    assert Path(calls[-1]["script"]).name == "check_mindie_laser_quant_preprocess.py"
    assert calls[-1]["args"][calls[-1]["args"].index("--device") + 1] == "3"
    assert all("benchmark_" not in call["script"] for call in calls)


def test_a2_keeps_family_for_attention_and_exact_soc_for_preprocessing(launcher):
    result, calls = launcher({"LASER_TEST_REUSE": "1", "LASER_TEST_SOC": "Ascend910B2"})
    assert result.returncode == 0, result.stdout + result.stderr
    attention, preprocess = calls[1:3]
    assert attention["args"][attention["args"].index("--soc") + 1] == "ascend910b"
    assert preprocess["args"][preprocess["args"].index("--soc") + 1] == "Ascend910B2"


def test_unsupported_soc_rejected_before_build(launcher):
    result, calls = launcher({"LASER_TEST_SOC": "Ascend310P3"})
    assert result.returncode != 0
    assert [Path(call["script"]).name for call in calls] == ["check_ascend910b.py"]
    assert "requires A2/A3" in result.stderr


def test_invalid_jobs_rejected_before_preflight(launcher):
    result, calls = launcher({"LASER_BUILD_JOBS": "0"})
    assert result.returncode == 2 and calls == []
