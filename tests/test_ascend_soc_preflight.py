"""SoC selection and launcher regression tests; no NPU or CANN required."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_ascend910b import resolve_soc


def ini(root, relative, soc, arch="2201"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"[version]\nSoC_version = {soc}\nNpuArch={arch}\n", encoding="utf-8")
    return str(path.resolve())


def test_a3_runtime_overrides_wrong_inherited_sku(tmp_path):
    # Reproduces the user's CANN 9.0.1 / A3 log. Never choose 9391 for 9382.
    wanted = ini(tmp_path, "arm64-linux/data/platform_config/arbitrary.ini", "Ascend910_9382")
    ini(tmp_path, "arm64-linux/data/platform_config/Ascend910_9382.ini", "Ascend910_9391")
    selected = resolve_soc("Ascend910_9382", "ascend910_9391", tmp_path)
    assert selected["compiler_soc"] == "Ascend910_9382"
    assert selected["inherited_soc_version"] == "ascend910_9391"
    assert selected["target_label"].startswith("Atlas A3")
    assert selected["matching_soc_ini"] == [wanted]
    assert not selected["errors"]
    assert any("Ignoring inherited SOC_VERSION" in w for w in selected["warnings"])


@pytest.mark.parametrize("soc,expected", [
    ("Ascend910B1", "Ascend910B1"), ("ascend910b2c", "Ascend910B2C"),
    ("Ascend910B3", "Ascend910B3"), ("Ascend910B4", "Ascend910B4"),
    ("ascend910_9382", "Ascend910_9382"), ("Ascend910_9391", "Ascend910_9391")])
def test_canonical_target_and_matching_lowercase_environment(tmp_path, soc, expected):
    ini(tmp_path, "compiler/data/platform_config/device.ini", soc.lower())
    result = resolve_soc(soc, soc.lower(), tmp_path)
    assert result["compiler_soc"] == expected
    assert result["expected_npu_arch"] == "2201"
    assert not result["errors"] and not result["warnings"]


@pytest.mark.parametrize("soc", ["Ascend910", "Ascend910A", "Ascend910B", "910B_23",
                                      "Ascend950PR_9579", "Ascend310P3", "Ascend910_93",
                                      "Ascend910_9382oops"])
def test_marketing_names_and_other_architectures_rejected(tmp_path, soc):
    with pytest.raises(ValueError, match="Unsupported runtime SoC"):
        resolve_soc(soc, "Ascend910B3", tmp_path)


def test_available_toolkit_without_matching_sku_is_rejected(tmp_path):
    ini(tmp_path, "aarch64-linux/data/platform_config/Ascend910_9382.ini", "Ascend910_9391")
    result = resolve_soc("Ascend910_9382", None, tmp_path)
    assert not result["matching_soc_ini"]
    assert any("no exact SoC_version=Ascend910_9382" in e for e in result["errors"])


def test_matching_soc_with_conflicting_architecture_is_rejected(tmp_path):
    ini(tmp_path, "data/platform_config/soc.ini", "Ascend910_9382", "3510")
    result = resolve_soc("Ascend910_9382", None, tmp_path)
    assert result["matching_soc_ini"]
    assert any("NpuArch=3510" in e for e in result["errors"])


def test_no_exposed_ini_keeps_exact_runtime_target_with_warning(tmp_path):
    result = resolve_soc("Ascend910_9382", "Ascend910B3", tmp_path)
    assert result["compiler_soc"] == "Ascend910_9382"
    assert not result["errors"]
    assert any("compiler must validate" in w for w in result["warnings"])


@pytest.mark.parametrize("preflight_fails", [False, True])
def test_launcher_exports_resolved_soc_and_stops_on_failed_preflight(tmp_path, preflight_fails):
    bash = ("C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash"))
    if not bash or not Path(bash).is_file():
        pytest.skip("bash is unavailable")
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/run_ascend910b.sh", tmp_path / "scripts/run_ascend910b.sh")
    shim = tmp_path / "python-double.sh"
    shim.write_text('''#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  scripts/check_ascend910b.py)
    printf 'preflight|%s\\n' "$SOC_VERSION" >> calls.txt
    if [[ "${FAIL_PREFLIGHT:-0}" == 1 ]]; then exit 44; fi
    printf '%s\\n' '{"compiler_soc":"Ascend910_9382"}' > results/ascend910b_environment.json
    ;;
  -c) printf '%s\\n' 'Ascend910_9382' ;;
  *) printf '%s|%s\\n' "$*" "$SOC_VERSION" >> calls.txt ;;
esac
''', encoding="utf-8", newline="\n")
    shim.chmod(0o755)
    env = {**os.environ, "SOC_VERSION": "ascend910_9391", "PYTHON_BIN": "./python-double.sh",
           "FAIL_PREFLIGHT": str(int(preflight_fails))}
    run = subprocess.run([bash, "scripts/run_ascend910b.sh", "--with-laser"], cwd=tmp_path,
                         env=env, capture_output=True, text=True, timeout=30)
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert calls[0] == "preflight|ascend910_9391"
    if preflight_fails:
        assert run.returncode == 44
        assert len(calls) == 1  # no build, validation, or benchmark after failure
    else:
        assert run.returncode == 0, run.stdout + run.stderr
        assert len(calls) == 4
        assert calls[1].startswith("-m pip install ./ascend ")
        assert calls[2].startswith("scripts/validate_ascend910b.py ")
        assert calls[3].startswith("scripts/benchmark_ascend910b.py ")
        assert all(line.endswith("|Ascend910_9382") for line in calls[1:])
