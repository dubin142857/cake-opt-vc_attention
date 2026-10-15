"""Build lifecycle, cache boundaries and artifact evidence with a mocked compiler."""
import json
from pathlib import Path
import subprocess

import pytest

from scripts import build_mindie_laser_local as driver


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    source, cann = root / "build/mindie_laser_source", tmp_path / "cann"
    (source / "csrc/ops").mkdir(parents=True)
    (source / "csrc/ops/CMakeLists.txt").write_text("project(test)")
    cann.mkdir()
    manifest = source.parent / "mindie_laser_source_manifest.json"
    manifest.write_text('{"schema_version": 1}\n')
    monkeypatch.setattr(driver, "ROOT", root)
    monkeypatch.setattr(driver, "validate_prepared_source", lambda *args: {
        "upstream": {"commit": driver.SOURCE_LOCK.commit},
        "native_api": driver.NATIVE_API, "op_type": driver.OP_TYPE,
    })
    monkeypatch.setattr(driver, "build_environment", lambda base, cann: dict(base, PATH="fake-toolkit"))
    return source, cann, root / "build/mindie_laser_install", root / "results/build.json"


def fake_runner(environment, calls, *, missing_binary=False):
    source, _, install, _ = environment
    build = source / "build/build"

    def run(command, cwd, env, phase):
        calls.append((phase, command, dict(env)))
        if phase == "configure":
            build.mkdir(parents=True, exist_ok=True)
            definitions = {arg[2:].split("=", 1)[0]: arg.split("=", 1)[1]
                           for arg in command if arg.startswith("-D")}
            definitions.update(CMAKE_HOME_DIRECTORY=str(source / "csrc/ops"),
                               CMAKE_GENERATOR="Unix Makefiles")
            (build / "CMakeCache.txt").write_text("\n".join(f"{k}:STRING={v}" for k, v in definitions.items()), encoding="utf-8")
            for name in driver.GENERATED_FILES:
                path = build / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("generated")
            gen = build / "binary/ascend910_93/gen"
            gen.mkdir(parents=True, exist_ok=True)
            (gen / "VcLaserAttention-vc_laser_attention-0.sh").write_text("# compile Laser\n")
            (build / "CMakeFiles/TargetDirectories.txt").write_text("\n".join(
                str(build / "CMakeFiles" / (name + ".dir")) for name in (
                    "ops_kernel", "vc_laser_attention", "vc_laser_attention_ascend910_93",
                    "vc_laser_attention_ascend910_93_0")), encoding="utf-8")
        if phase == "build_package":
            (build / "CANN-custom_ops-test-linux.aarch64.run").write_bytes(b"package")
        if phase == "install_workspace":
            vendor = install / "packages/vendors/vc_laser"
            files = {
                "op_api/lib/libcust_opapi.so": b"api",
                "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so": b"tiling",
            }
            if not missing_binary:
                files["op_impl/ai_core/tbe/kernel/ascend910_93/vc_laser_attention/kernel.o"] = b"device"
            for name, data in files.items():
                path = vendor / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        return 0.1
    return run


def call_build(environment, runner, jobs=2):
    source, cann, install, output = environment
    return driver.build(source, cann, "ascend910_93", jobs, install, output, runner=runner)


def test_build_scopes_to_one_operator_and_installs_only_workspace(environment):
    calls = []
    report = call_build(environment, fake_runner(environment, calls))
    assert [item[0] for item in calls] == ["configure", "build_package", "install_workspace"]
    assert report["status"] == "passed"
    assert report["definitions"]["ASCEND_OP_NAME"] == "vc_laser_attention"
    assert report["definitions"]["VENDOR_NAME"] == "vc_laser"
    assert report["definitions"]["CMAKE_BUILD_TYPE"] == "Release"
    assert report["definitions"]["CHECK_COMPATIBLE"] == "TRUE"
    assert report["device_graph_after_configure"]["complete"] is True
    assert all(item[2]["BUILD_JOB_NUM"] == "2" for item in calls)
    assert all(item[2]["CMAKE_BUILD_PARALLEL_LEVEL"] == "2" for item in calls)
    assert len(report["device_binaries"]) == 1
    assert len(report["host_libraries"]) == 2
    assert report["source_manifest"]["sha256"]
    assert report["native_api"] == "aclnnVcLaserAttention"
    assert report["vendor_root"] == str(environment[2] / "packages/vendors/vc_laser")
    assert not any("build_ops.sh" in " ".join(call[1]) for call in calls)


def test_complete_cache_resumes_without_reconfigure_or_cleanup(environment):
    calls = []
    runner = fake_runner(environment, calls)
    call_build(environment, runner)
    sentinel = environment[0] / "build/build/keep.o"
    sentinel.write_bytes(b"partial object")
    calls.clear()
    report = call_build(environment, runner)
    assert [item[0] for item in calls] == ["build_package", "install_workspace"]
    assert report["resume_existing_cache"] is True
    assert sentinel.read_bytes() == b"partial object"


@pytest.mark.parametrize("missing_parameter", [True, False])
def test_host_only_cache_reconfigures_even_if_compile_scripts_appeared_late(environment, missing_parameter):
    calls = []
    runner = fake_runner(environment, calls)
    call_build(environment, runner)
    build = environment[0] / "build/build"
    # Reproduce the failed remote state: scripts exist after make, but the
    # configure-time graph has only a host build and an empty ops_kernel.
    (build / "CMakeFiles/TargetDirectories.txt").write_text(
        str(build / "CMakeFiles/ops_kernel.dir"), encoding="utf-8")
    if missing_parameter:
        cache = build / "CMakeCache.txt"
        cache.write_text("\n".join(line for line in cache.read_text(encoding="utf-8").splitlines()
                                   if not line.startswith("CHECK_COMPATIBLE:")), encoding="utf-8")
    sentinel = build / "existing_host.o"
    sentinel.write_bytes(b"already compiled host")
    calls.clear()
    report = call_build(environment, runner)
    assert [item[0] for item in calls] == ["configure", "build_package", "install_workspace"]
    assert report["device_graph_before"]["compile_scripts"]
    assert not report["device_graph_before"]["complete"]
    assert report["device_graph_after_configure"]["complete"]
    assert sentinel.read_bytes() == b"already compiled host"


@pytest.mark.parametrize("broken", ["no_scripts", "missing_target"])
def test_empty_device_graph_stops_before_packaging(environment, broken):
    calls = []
    runner = fake_runner(environment, calls)
    build = environment[0] / "build/build"

    def bad_configure(command, cwd, env, phase):
        result = runner(command, cwd, env, phase)
        if phase == "configure":
            if broken == "no_scripts":
                (build / "binary/ascend910_93/gen/VcLaserAttention-vc_laser_attention-0.sh").unlink()
            else:
                # Even the parent target is insufficient without actual index targets.
                target_file = build / "CMakeFiles/TargetDirectories.txt"
                text = target_file.read_text(encoding="utf-8")
                target_file.write_text("\n".join(line for line in text.splitlines()
                                                if not line.endswith("ascend910_93_0.dir")), encoding="utf-8")
        return result

    with pytest.raises(ValueError, match="device build graph"):
        call_build(environment, bad_configure)
    assert [item[0] for item in calls] == ["configure"]
    report = json.loads(environment[3].read_text())
    assert report["status"] == "failed"
    assert report["steps"][-1]["status"] == "failed"
    assert report["device_graph_after_configure"]["complete"] is False


@pytest.mark.parametrize("key,value", [
    ("ASCEND_OP_NAME", "ALL"), ("ASCEND_COMPUTE_UNIT", "ascend910b"),
    ("VENDOR_NAME", "customize"), ("ENABLE_OPS_KERNEL", "OFF"),
    ("CMAKE_HOME_DIRECTORY", "/wrong/source"),
])
def test_incompatible_cache_fails_before_any_build(environment, key, value):
    calls = []
    call_build(environment, fake_runner(environment, calls))
    cache = environment[0] / "build/build/CMakeCache.txt"
    lines = [line for line in cache.read_text(encoding="utf-8").splitlines() if not line.startswith(key + ":")]
    cache.write_text("\n".join([*lines, f"{key}:STRING={value}"]), encoding="utf-8")
    calls.clear()
    with pytest.raises(ValueError, match="differs|incompatible"):
        call_build(environment, fake_runner(environment, calls))
    assert calls == []


def test_failed_compiler_records_failure_and_preserves_files(environment):
    calls = []
    runner = fake_runner(environment, calls)

    def fail(command, cwd, env, phase):
        if phase == "build_package":
            raise subprocess.CalledProcessError(17, command)
        return runner(command, cwd, env, phase)

    with pytest.raises(subprocess.CalledProcessError):
        call_build(environment, fail)
    report = json.loads(environment[3].read_text())
    assert report["status"] == "failed"
    assert report["steps"][-1]["status"] == "failed"
    assert (environment[0] / "build/build/CMakeCache.txt").is_file()


def test_successful_command_without_device_binary_is_not_success(environment):
    with pytest.raises(ValueError, match="device"):
        call_build(environment, fake_runner(environment, [], missing_binary=True))
    assert json.loads(environment[3].read_text())["status"] == "failed"


def test_install_outside_workspace_is_rejected(environment, tmp_path):
    source, cann, _, output = environment
    calls = []
    with pytest.raises(ValueError, match="workspace"):
        driver.build(source, cann, "ascend910_93", 2, tmp_path / "system", output,
                     runner=fake_runner(environment, calls))
    assert calls == []
