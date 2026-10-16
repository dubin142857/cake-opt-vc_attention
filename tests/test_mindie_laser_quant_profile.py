"""Check fixed-payload scope, all-output gates and the no-build entry point."""
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
import profile_mindie_laser_quant as profile


def test_fixed_timers_reuse_exact_objects_and_never_prepare():
    seen = []
    class Adapter:
        def __init__(self, name): self.name = name
        def run_native(self, payload): seen.append((self.name, "native", payload))
        def run_prepared(self, payload): seen.append((self.name, "prepared", payload))
        def prepare(self, *inputs): seen.append((self.name, "prepare", inputs))
        def __call__(self, *inputs): seen.append((self.name, "full", inputs))
    adapters = {name: Adapter(name) for name in ("torch", "fused")}
    payloads = {name: object() for name in adapters}
    inputs = (object(), object(), object())
    calls = profile.make_calls(adapters, payloads, inputs)
    for name in adapters:
        calls["shared_native"][name]()
        assert seen[-1] == (name, "native", payloads["fused"])
        calls["own_native"][name]()
        assert seen[-1] == (name, "native", payloads[name])
        calls["prepared"][name]()
        assert seen[-1] == (name, "prepared", payloads[name])
    assert not any(item[1] == "prepare" for item in seen)
    for name in adapters:
        calls["full"][name]()
        assert seen[-1] == (name, "full", inputs)
        calls["preprocessing"][name]()
        assert seen[-1] == (name, "prepare", inputs)


def test_all_output_check_includes_last_chunk_and_padding():
    reference = torch.ones(2, 3, 513, 8)
    actual = reference.clone()
    metric = profile.compare_all_rows(actual, reference)
    assert metric["exact_match"] and metric["elements"] == reference.numel()
    actual[-1, -1, -1, -1] = 3
    metric = profile.compare_all_rows(actual, reference)
    assert not metric["exact_match"] and metric["max_abs"] == 2
    assert metric["relative_rmse"] == pytest.approx(2 / reference.numel() ** .5)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_all_output_check_rejects_nonfinite_last_row(value):
    actual = torch.zeros(1, 1, 257, 4)
    actual[0, 0, -1, -1] = value
    with pytest.raises(ValueError, match="Non-finite"):
        profile.compare_all_rows(actual, torch.zeros_like(actual))


def test_runtime_build_mismatch_fails_before_any_native_execution(tmp_path):
    build = {"soc": "ascend910_93", "cann_root": str(tmp_path)}
    preprocess = {"soc": "Ascend910_9382", "cann_root": str(tmp_path)}
    assert profile.verify_runtime_target(build, preprocess, "Ascend910_9382", tmp_path)["expected_npu_arch"] == "2201"
    with pytest.raises(ValueError, match="exact runtime SoC"):
        profile.verify_runtime_target(build, preprocess, "Ascend910_9391", tmp_path)
    with pytest.raises(ValueError, match="SoC family"):
        profile.verify_runtime_target(dict(build, soc="ascend910b"), preprocess, "Ascend910_9382", tmp_path)
    with pytest.raises(ValueError, match="CANN Toolkit"):
        profile.verify_runtime_target(build, dict(preprocess, cann_root=str(tmp_path / "other")), "Ascend910_9382", tmp_path)


def test_parent_provenance_failure_never_launches_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(profile.quant, "build_arguments", lambda *args: SimpleNamespace())
    def fail(*args): raise ValueError("stale source build")
    monkeypatch.setattr(profile.quant, "quant_provenance", fail)
    def unexpected(*args, **kwargs): raise AssertionError("worker must not run")
    monkeypatch.setattr(profile.subprocess, "Popen", unexpected)
    output = tmp_path / "report.json"
    assert profile.parent(SimpleNamespace(output=str(output))) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed" and "stale source" in report["parent_error"]


@pytest.mark.parametrize("exit_code", [0, 23])
def test_one_command_profiles_without_compilation_and_preserves_failure(tmp_path, exit_code):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing")
    if not bash.is_file(): pytest.skip("Bash required")
    root = tmp_path / "space in project"
    (root / "scripts").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "cann").mkdir()
    for name in ("mindie_laser_quant_build.json", "mindie_laser_quant_preprocess_build.json"):
        (root / "results" / name).write_text("{}")
    script = root / "scripts/run_mindie_laser_quant_profile.sh"
    script.write_bytes((ROOT / "scripts/run_mindie_laser_quant_profile.sh").read_bytes())
    stub = root / "stub.py"
    stub.write_text("import json,sys\nfrom pathlib import Path\n"
                    "Path('calls.json').write_text(json.dumps(sys.argv[1:]))\n"
                    f"raise SystemExit({exit_code})\n", encoding="utf-8")
    wrapper = root / "mock-python"
    wrapper.write_text(f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "{stub.as_posix()}" "$@"\n',
                       encoding="utf-8", newline="\n")
    wrapper.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("BASH_ENV", "ENV")}
    env.update(PYTHON_BIN=wrapper.as_posix(), ASCEND_HOME_PATH=(root / "cann").as_posix(), NPU_DEVICE="2")
    result = subprocess.run([str(bash), "--noprofile", "--norc", str(script)], cwd=root, env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == exit_code, result.stdout + result.stderr
    command = json.loads((root / "calls.json").read_text())
    assert command[:2] == ["-u", "scripts/profile_mindie_laser_quant.py"]
    assert command[command.index("--shape") + 1:command.index("--shape") + 5] == ["1", "28", "37741", "128"]
    assert command[command.index("--device") + 1] == "2"
    assert command[command.index("--trials") + 1] == "9" and "--trace" in command


def test_cli_parent_import_does_not_initialize_torch():
    code = "import sys; sys.path.insert(0, 'scripts'); import profile_mindie_laser_quant; assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
