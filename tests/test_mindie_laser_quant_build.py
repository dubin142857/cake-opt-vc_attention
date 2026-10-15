"""No NPU/CANN required: isolated build composition, provenance and failures."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_mindie_laser_quant as quant


@pytest.fixture
def staged(tmp_path):
    arguments = (tmp_path / "quant_source", tmp_path / "reference",
                 tmp_path / "quant_source_manifest.json", tmp_path / "cann",
                 "ascend910_93", 4, tmp_path / "quant_install", tmp_path / "quant_bridge",
                 tmp_path / "results/quant_build.json")
    calls = []

    def prepare(source, reference, manifest):
        calls.append(("prepare", source, reference))
        source.mkdir(parents=True, exist_ok=True)
        quant.write_json(manifest, {"upstream": {"commit": "fixture"}, "native_api": quant.IDENTITY["native_api"]})

    def validate(source, manifest):
        calls.append(("validate", source, manifest))
        assert source.is_dir()
        return json.loads(manifest.read_text(encoding="utf-8"))

    def native(source, cann, soc, jobs, install_dir, output, **kwargs):
        calls.append(("native", kwargs))
        report = {"status": "passed", "native_api": quant.IDENTITY["native_api"],
                  "op_type": quant.IDENTITY["op_type"], "cann_root": str(cann), "soc": soc,
                  "vendor_root": str(install_dir / "packages/vendors/vc_laser_quant"),
                  "host_libraries": [{"path": "/fixture/host.so", "sha256": "host-hash"}],
                  "device_binaries": [{"path": "/fixture/device.o", "sha256": "device-hash"}]}
        quant.write_json(output, report)
        return report

    def bridge(source, cann, bridge_dir, jobs, output, **kwargs):
        calls.append(("bridge", kwargs))
        bridge_dir.mkdir(parents=True, exist_ok=True)
        binary = bridge_dir / quant.BRIDGE_NAME
        binary.write_bytes(b"quant bridge fixture")
        report = {"status": "passed", "native_api": quant.IDENTITY["native_api"],
                  "op_type": quant.IDENTITY["op_type"], "bridge_path": str(binary),
                  "bridge_sha256": quant.sha256_file(binary)}
        quant.write_json(output, report)
        return report

    dependencies = {"preparer": prepare, "source_validator": validate,
                    "operator_builder": native, "bridge_builder": bridge}
    return arguments, dependencies, calls


def test_combined_manifest_keeps_binary_evidence_and_separate_source_identity(staged):
    arguments, dependencies, calls = staged
    report = quant.build(*arguments, **dependencies)
    assert report["status"] == "passed"
    assert report["native_api"] == "aclnnVcLaserQuantAttention"
    assert report["op_type"] == "VcLaserQuantAttention"
    assert report["soc"] == "ascend910_93"
    assert report["host_libraries"][0]["sha256"] == "host-hash"
    assert report["device_binaries"][0]["sha256"] == "device-hash"
    assert report["source_manifest"]["sha256"] == quant.sha256_file(arguments[2])
    assert report["bridge_sha256"] == quant.sha256_file(report["bridge_path"])
    assert report["bridge_manifest_sha256"] == quant.sha256_file(report["bridge_manifest"])
    assert report["marker_environment"] == "VC_MINDIE_LASER_QUANT_MARKER_FILE"
    native_call = next(call for call in calls if call[0] == "native")
    assert native_call[1]["identity"] == quant.IDENTITY
    assert native_call[1]["source_validator"] is dependencies["source_validator"]
    assert [call[0] for call in calls] == ["prepare", "validate", "native", "bridge"]
    assert json.loads(arguments[-1].read_text(encoding="utf-8"))["status"] == "passed"


def test_failed_bridge_keeps_operator_evidence_and_never_claims_complete(staged):
    arguments, dependencies, _ = staged
    def fail(*args, **kwargs):
        raise RuntimeError("fixture bridge compiler failed")
    dependencies["bridge_builder"] = fail
    with pytest.raises(RuntimeError, match="compiler failed"):
        quant.build(*arguments, **dependencies)
    report = json.loads(arguments[-1].read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert "compiler failed" in report["error"]
    assert report["device_binaries"][0]["sha256"] == "device-hash"
    assert Path(report["operator_manifest"]).is_file()


def test_foreign_operator_manifest_is_rejected_before_bridge(staged):
    arguments, dependencies, calls = staged
    original = dependencies["operator_builder"]
    def foreign(*args, **kwargs):
        report = original(*args, **kwargs)
        report["native_api"] = "aclnnVcLaserAttention"
        return report
    dependencies["operator_builder"] = foreign
    with pytest.raises(ValueError, match="does not identify"):
        quant.build(*arguments, **dependencies)
    assert not any(call[0] == "bridge" for call in calls)
    assert json.loads(arguments[-1].read_text(encoding="utf-8"))["status"] == "failed"


def test_tampered_bridge_binary_is_rejected(staged):
    arguments, dependencies, _ = staged
    original = dependencies["bridge_builder"]
    def tampered(*args, **kwargs):
        report = original(*args, **kwargs)
        Path(report["bridge_path"]).write_bytes(b"changed after manifest")
        return report
    dependencies["bridge_builder"] = tampered
    with pytest.raises(ValueError, match="hash differs"):
        quant.build(*arguments, **dependencies)
    assert json.loads(arguments[-1].read_text(encoding="utf-8"))["status"] == "failed"


def test_adapter_is_copied_from_pinned_git_objects_including_license(tmp_path, monkeypatch):
    commit = "fixture-pinned-commit"
    calls = []
    monkeypatch.setattr(quant, "_module", lambda name: SimpleNamespace(SOURCE_LOCK=SimpleNamespace(commit=commit)))
    def git(source, *args):
        calls.append(args)
        return (commit + "\n").encode() if args == ("rev-parse", "HEAD") else args[1].encode()
    monkeypatch.setattr(quant, "git_bytes", git)
    output = tmp_path / "adapter"
    records = quant.copy_pinned_adapter(tmp_path, output)
    assert set(records["files"]) == set(quant.ADAPTER_FILES)
    assert (output / "LICENSE.md").read_bytes() == f"{commit}:LICENSE.md".encode()
    assert all(args[0] == "show" for args in calls[1:])
    for record in records["files"].values():
        assert record["sha256"] == quant.sha256_file(record["path"])


def test_adapter_refuses_wrong_checkout_before_copying(tmp_path, monkeypatch):
    monkeypatch.setattr(quant, "_module", lambda name: SimpleNamespace(SOURCE_LOCK=SimpleNamespace(commit="wanted")))
    monkeypatch.setattr(quant, "git_bytes", lambda *args: b"unexpected\n")
    with pytest.raises(ValueError, match="requires pinned source"):
        quant.copy_pinned_adapter(tmp_path, tmp_path / "adapter")
    assert not (tmp_path / "adapter").exists()


def test_cmake_cache_rejects_runtime_changes_and_incomplete_foreign_cache(tmp_path):
    cache = tmp_path / "CMakeCache.txt"
    expected = {"TORCH_ROOT": str(tmp_path / "torch"), "TORCH_ABI": "0"}
    quant.check_cmake_cache(cache, expected)
    cache.write_text(f"TORCH_ROOT:PATH={tmp_path / 'torch'}\nTORCH_ABI:STRING=0\n", encoding="utf-8")
    quant.check_cmake_cache(cache, expected)
    with pytest.raises(ValueError, match="mismatch for TORCH_ABI"):
        quant.check_cmake_cache(cache, {**expected, "TORCH_ABI": "1"})
    cache.write_text("CMAKE_BUILD_TYPE:STRING=Release\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks TORCH_ROOT"):
        quant.check_cmake_cache(cache, expected)


def test_bridge_build_records_commands_runtime_and_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(quant, "copy_pinned_adapter", lambda *args: {"upstream_commit": "fixture", "files": {}})
    runtime = {"torch_root": str(tmp_path / "torch"), "torch_npu_root": str(tmp_path / "torch_npu"),
               "torch_abi": 1, "python_executable": str(tmp_path / "python")}
    bridge_dir = tmp_path / "bridge"
    calls = []
    def runner(command, cwd, env, phase):
        calls.append((command, cwd, phase))
        if phase == "quant bridge compile":
            bridge_dir.mkdir(parents=True, exist_ok=True)
            (bridge_dir / quant.BRIDGE_NAME).write_bytes(b"shared library fixture")
        return .25
    report = quant.build_bridge(tmp_path / "source", tmp_path / "cann", bridge_dir,
                                4, tmp_path / "bridge.json", runner=runner, runtime=runtime)
    assert report["status"] == "passed"
    assert report["bridge_sha256"] == quant.sha256_file(bridge_dir / quant.BRIDGE_NAME)
    configure, compile_ = calls
    assert "-DTORCH_ABI=1" in configure[0]
    assert f"-DPython3_EXECUTABLE={runtime['python_executable']}" in configure[0]
    assert compile_[0][-2:] == ["--parallel", "4"]
    assert all(record["elapsed_seconds"] == .25 for record in report["steps"])


def test_bridge_missing_output_saves_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(quant, "copy_pinned_adapter", lambda *args: {"files": {}})
    runtime = {"torch_root": str(tmp_path / "torch"), "torch_npu_root": str(tmp_path / "torch_npu"),
               "torch_abi": 0, "python_executable": str(tmp_path / "python")}
    manifest = tmp_path / "bridge.json"
    with pytest.raises(RuntimeError, match="did not produce"):
        quant.build_bridge(tmp_path, tmp_path / "cann", tmp_path / "bridge", 4,
                           manifest, runner=lambda *args: 0, runtime=runtime)
    report = json.loads(manifest.read_text(encoding="utf-8"))
    assert report["status"] == "failed" and len(report["steps"]) == 2


def test_cli_defaults_are_isolated_from_baseline():
    arguments = quant.parser().parse_args(["--cann-root", "toolkit", "--soc", "ascend910_93"])
    assert arguments.jobs == 4
    assert arguments.source_dir.name == "mindie_laser_quant_source"
    assert arguments.install_dir.name == "mindie_laser_quant_install"
    assert arguments.bridge_dir.name == "mindie_laser_quant_bridge"
    assert arguments.output.name == "mindie_laser_quant_build.json"


@pytest.mark.parametrize("key,value", [("CMAKE_GENERATOR", "Ninja"), ("CMAKE_BUILD_TYPE", "Debug")])
def test_bridge_cache_rejects_generator_or_build_type_change(tmp_path, key, value):
    cache = tmp_path / "CMakeCache.txt"
    expected = {"CMAKE_GENERATOR": quant.CMAKE_GENERATOR, "CMAKE_BUILD_TYPE": "Release"}
    values = {**expected, key: value}
    cache.write_text("".join(f"{name}:STRING={item}\n" for name, item in values.items()), encoding="utf-8")
    with pytest.raises(ValueError, match=f"mismatch for {key}"):
        quant.check_cmake_cache(cache, expected)


@pytest.mark.parametrize("key", [
    "CMAKE_TOOLCHAIN_FILE", "CMAKE_CXX_COMPILER_LAUNCHER",
    "CMAKE_PROJECT_TOP_LEVEL_INCLUDES", "CMAKE_PROJECT_vc_laser_quant_bridge_INCLUDE_BEFORE",
])
def test_bridge_cache_rejects_toolchain_and_project_hook_overrides(tmp_path, key):
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(f"{key}:FILEPATH=/unexpected/override\n", encoding="utf-8")
    with pytest.raises(ValueError, match=f"toolchain override {key}"):
        quant.check_cmake_cache(cache, {})


def test_bridge_environment_override_fails_before_runtime_or_compiler(tmp_path, monkeypatch):
    monkeypatch.setenv("CMAKE_TOOLCHAIN_FILE", "/unexpected/toolchain.cmake")
    manifest = tmp_path / "bridge.json"
    def unexpected(*args, **kwargs):
        pytest.fail("Compiler/runtime must not run with an undeclared toolchain")
    monkeypatch.setattr(quant, "discover_runtime", unexpected)
    with pytest.raises(ValueError, match="toolchain override CMAKE_TOOLCHAIN_FILE"):
        quant.build_bridge(tmp_path, tmp_path / "cann", tmp_path / "bridge", 4,
                           manifest, runner=unexpected)
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "failed"
