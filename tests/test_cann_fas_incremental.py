"""CANN build planning/failure preservation checks; no device compilation."""
import json
from pathlib import Path
import subprocess

import pytest

from scripts import build_cann_fas_local as driver


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    source, cann = tmp_path / "source", tmp_path / "cann"
    source.mkdir()
    (source / "CMakeLists.txt").write_text("# fixture")
    hook = tmp_path / "hook.cmake"
    hook.write_text("# fixture")
    monkeypatch.setattr(driver, "HOOK", hook)
    monkeypatch.setattr(driver, "build_environment", lambda base, source, cann: {})
    return source, cann, tmp_path / "report.json"


def existing_build(workspace, changes=None, hook=True, complete=True):
    source, cann, _ = workspace
    build_dir = source / "build"
    build_dir.mkdir()
    values = driver.definitions(source, cann, "ascend910_93")
    values.update(CMAKE_HOME_DIRECTORY=str(source), CMAKE_GENERATOR="Unix Makefiles")
    values["ASCEND_COMPUTE_UNIT"] += ";"  # Official build.sh emits this spelling.
    if not hook:
        values.pop("CMAKE_PROJECT_INCLUDE")
    values.update(changes or {})
    (build_dir / "CMakeCache.txt").write_text(
        "\n".join(f"{k}:STRING={v}" for k, v in values.items()), encoding="utf-8")
    if complete:
        for name in ("Makefile", "CMakeFiles/Makefile.cmake", "cmake_install.cmake", "CPackConfig.cmake"):
            path = build_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# already generated")
    sentinel = build_dir / "compiled_object.o"
    sentinel.write_bytes(b"preserve existing work")
    return sentinel


def perform(workspace, runner):
    source, cann, output = workspace
    return driver.build(source, cann, "ascend910_93", 8, output, runner=runner)


def test_reuses_ready_build_without_configure_or_clean(workspace):
    sentinel = existing_build(workspace)
    called = []
    report = perform(workspace, lambda command, *args: called.append(command) or 0.1)
    assert report["status"] == "passed" and report["resume_existing_cache"]
    assert len(called) == 1 and called[0][1] == "--build"
    assert called[0][-2:] == ["--parallel", "8"]
    assert sentinel.read_bytes() == b"preserve existing work"


@pytest.mark.parametrize("complete", [True, False])
def test_adopts_old_or_interrupted_configuration_preserving_objects(workspace, complete):
    sentinel = existing_build(workspace, hook=False, complete=complete)
    called = []
    perform(workspace, lambda command, *args: called.append(command) or 0.2)
    assert len(called) == 2 and called[0][1] == "-S"
    assert any(arg.startswith("-DCMAKE_PROJECT_INCLUDE=") for arg in called[0])
    assert not any("clean" in arg or arg == "build.sh" for command in called for arg in command)
    assert sentinel.exists()


@pytest.mark.parametrize("change", [
    {"ASCEND_COMPUTE_UNIT": "ascend910b"}, {"ASCEND_OP_NAME": "ALL"},
    {"VENDOR_NAME": "other"}, {"CMAKE_BUILD_TYPE": "Debug"}, {"ENABLE_OPS_KERNEL": "OFF"},
    {"CMAKE_HOME_DIRECTORY": "/different/source"}, {"CUSTOM_ASCEND_CANN_PACKAGE_PATH": "/different/cann"},
    {"CMAKE_PROJECT_INCLUDE": "/another/hook.cmake"},
])
def test_refuses_incompatible_cache_without_destroying_it(workspace, change):
    sentinel = existing_build(workspace, change)
    def must_not_run(*args):
        raise AssertionError("Must reject incompatible cache before execution")
    with pytest.raises(ValueError):
        perform(workspace, must_not_run)
    assert sentinel.exists()
    report = json.loads(workspace[2].read_text())
    assert report["status"] == "failed" and not report["steps"]


def test_failed_build_saves_diagnostic_and_retains_work(workspace):
    sentinel = existing_build(workspace)
    def fail(command, *args):
        raise subprocess.CalledProcessError(17, command)
    with pytest.raises(subprocess.CalledProcessError):
        perform(workspace, fail)
    report = json.loads(workspace[2].read_text())
    assert report["status"] == "failed" and report["steps"][-1]["phase"] == "build_package"
    assert report["steps"][-1]["status"] == "failed"
    assert sentinel.exists()


def test_fresh_build_configures_and_builds(workspace):
    called = []
    report = perform(workspace, lambda command, *args: called.append(command) or 0.2)
    assert not report["resume_existing_cache"]
    assert [row["phase"] for row in report["steps"]] == ["configure", "build_package"]
    assert "-DENABLE_OPS_KERNEL=ON" in called[0]
