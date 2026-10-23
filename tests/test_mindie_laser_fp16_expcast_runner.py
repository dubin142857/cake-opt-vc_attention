"""Offline orchestration: only the new provider builds; failures stop dispatch."""
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
    root = tmp_path / "project with spaces"
    for part in ("scripts", "results", "cached CANN/bin", "stable source"):
        (root / part).mkdir(parents=True)
    (root / "cached CANN/bin/setenv.bash").write_text("export TEST_CANN_READY=yes\n", encoding="utf-8", newline="\n")
    manifest = {"cann_root": (root / "cached CANN").as_posix(),
                "source_dir": (root / "stable source").as_posix(), "soc": "ascend910_93"}
    for name in ("mindie_laser_quant_vscale_pv_prefetch_build", "mindie_laser_quant_preprocess_build",
                 "mindie_laser_build", "mindie_laser_bridge"):
        (root / "results" / (name + ".json")).write_text(json.dumps(manifest), encoding="utf-8")
    driver = root / "scripts/run_mindie_laser_fp16_expcast.sh"
    driver.write_bytes((ROOT / "scripts" / driver.name).read_bytes())
    stub = root / "mock.py"
    stub.write_text('''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
stage = ("preflight" if args[0] == "-" else "metadata" if args[0] == "-c" else
         "build" if args[1] == "scripts/build_mindie_laser_quant_vscale_pv_expcast.py" else "benchmark")
with Path("calls.jsonl").open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(dict(stage=stage, args=args, cann=os.environ.get("ASCEND_HOME_PATH"),
                                ready=os.environ.get("TEST_CANN_READY"))) + "\\n")
if os.environ.get("TEST_FAIL_STAGE") == stage:
    raise SystemExit(23)
if stage == "metadata":
    exec(args[1])
else:
    print(stage + " complete")
''', encoding="utf-8")
    python = root / "mock python"
    python.write_text(f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "{stub.as_posix()}" "$@"\n',
                      encoding="utf-8", newline="\n")
    python.chmod(0o755)

    def run(fail=None, missing=False):
        if missing:
            (root / "results/mindie_laser_quant_vscale_pv_prefetch_build.json").unlink()
        env = {key: value for key, value in os.environ.items()
               if key not in ("BASH_ENV", "ENV", "TEST_FAIL_STAGE", "TEST_CANN_READY")}
        env.update(PYTHON_BIN=python.as_posix(), NPU_DEVICE="2", LASER_BUILD_JOBS="3", PYTHONIOENCODING="utf-8")
        if fail:
            env["TEST_FAIL_STAGE"] = fail
        result = subprocess.run([str(bash), "--noprofile", "--norc", str(driver)],
            cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
        trace = root / "calls.jsonl"
        calls = [json.loads(x) for x in trace.read_text(encoding="utf-8").splitlines()] if trace.exists() else []
        return result, calls, root
    return run


def test_only_new_provider_builds_from_recorded_source_then_four_way_benchmark(run_script):
    result, calls, root = run_script()
    assert result.returncode == 0, result.stdout + result.stderr
    assert [c["stage"] for c in calls] == ["preflight", "metadata", "metadata", "metadata", "build", "benchmark"]
    build, benchmark = calls[-2:]
    assert build["args"][:2] == ["-u", "scripts/build_mindie_laser_quant_vscale_pv_expcast.py"]
    args = build["args"]
    assert Path(args[args.index("--reference-dir") + 1]) == root / "stable source"
    assert args[args.index("--jobs") + 1] == "3"
    assert build["ready"] == benchmark["ready"] == "yes"
    args = benchmark["args"]
    assert args[:2] == ["-u", "scripts/profile_mindie_laser_fp16_expcast.py"]
    assert args[args.index("--device") + 1] == "2"
    assert args[args.index("--shape") + 1:args.index("--shape") + 5] == ["1", "28", "37741", "128"]
    assert "--trace" not in args
    assert (root / "results/mindie_laser_fp16_expcast.log").read_text().strip() == "benchmark complete"


@pytest.mark.parametrize("stage", ["preflight", "metadata", "build", "benchmark"])
def test_failure_stops_without_retry_or_other_device_work(run_script, stage):
    result, calls, _ = run_script(fail=stage)
    assert result.returncode == 23, result.stdout + result.stderr
    assert calls[-1]["stage"] == stage
    assert sum(c["stage"] == stage for c in calls) == 1


def test_missing_stable_build_stops_before_python(run_script):
    result, calls, _ = run_script(missing=True)
    assert result.returncode == 1 and calls == []
    assert "Missing results/mindie_laser_quant_vscale_pv_prefetch_build.json" in result.stderr
