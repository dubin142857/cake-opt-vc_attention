"""CPU-only build isolation and provenance checks for the scale-reuse operator."""
import json
from pathlib import Path

import pytest

from scripts import build_mindie_laser_quant as base
from scripts import build_mindie_laser_quant_reuse as reuse


@pytest.fixture
def staged(tmp_path):
    arguments = (tmp_path / "reuse_source", tmp_path / "reference",
                 tmp_path / "source_manifest.json", tmp_path / "cann",
                 "ascend910_93", 4, tmp_path / "reuse_install", tmp_path / "reuse_bridge",
                 tmp_path / "results/reuse_build.json")
    calls = []

    def prepare(source, reference, manifest):
        calls.append("prepare")
        source.mkdir(parents=True, exist_ok=True)
        base.write_json(manifest, {"native_api": reuse.IDENTITY["native_api"]})

    def validate(source, manifest):
        calls.append("validate")
        return json.loads(manifest.read_text(encoding="utf-8"))

    def native(source, cann, soc, jobs, install, output, **kwargs):
        calls.append(("native", kwargs))
        report = {"status": "passed", **reuse.IDENTITY,
                  "vendor_root": str(install / "packages/vendors/vc_laser_quant_reuse"),
                  "host_libraries": [{"sha256": "host-proof"}],
                  "device_binaries": [{"sha256": "kernel-proof"}]}
        base.write_json(output, report)
        return report

    def bridge(source, cann, directory, jobs, output, **kwargs):
        calls.append("bridge")
        directory.mkdir(parents=True, exist_ok=True)
        binary = directory / reuse.BRIDGE_NAME
        binary.write_bytes(b"reuse bridge fixture")
        report = {"status": "passed", **reuse.IDENTITY, "bridge_path": str(binary),
                  "bridge_sha256": base.sha256_file(binary)}
        base.write_json(output, report)
        return report

    dependencies = dict(preparer=prepare, source_validator=validate,
                        operator_builder=native, bridge_builder=bridge)
    return arguments, dependencies, calls


def test_candidate_build_keeps_distinct_identity_and_audited_artifacts(staged):
    arguments, dependencies, calls = staged
    report = reuse.build(*arguments, **dependencies)
    assert report["status"] == "passed"
    assert all(report[key] == value for key, value in reuse.IDENTITY.items())
    assert report["marker_token"] == reuse.MARKER_TOKEN
    assert report["marker_environment"] == reuse.MARKER_ENVIRONMENT
    assert report["source_manifest"]["sha256"] == base.sha256_file(arguments[2])
    assert report["bridge_sha256"] == base.sha256_file(report["bridge_path"])
    assert report["bridge_manifest_sha256"] == base.sha256_file(report["bridge_manifest"])
    assert report["device_binaries"][0]["sha256"] == "kernel-proof"
    assert next(call[1] for call in calls if isinstance(call, tuple))["identity"] == reuse.IDENTITY
    assert base.IDENTITY["native_api"] == "aclnnVcLaserQuantAttention"
    assert base.BRIDGE_NAME == "libvc_laser_quant_bridge.so"


@pytest.mark.parametrize("which", ["operator_builder", "bridge_builder"])
def test_candidate_rejects_baseline_identity_before_publishing(staged, which):
    arguments, dependencies, calls = staged
    original = dependencies[which]

    def foreign(*args, **kwargs):
        report = original(*args, **kwargs)
        report.update(base.IDENTITY)
        return report

    dependencies[which] = foreign
    with pytest.raises(ValueError, match="identify|identity"):
        reuse.build(*arguments, **dependencies)
    assert json.loads(arguments[-1].read_text(encoding="utf-8"))["status"] == "failed"
    if which == "operator_builder":
        assert "bridge" not in calls


def test_generalized_composer_cannot_use_baseline_defaults_for_other_identity(tmp_path):
    arguments = (tmp_path / "source", tmp_path / "ref", tmp_path / "manifest.json",
                 tmp_path / "cann", "ascend910_93", 4, tmp_path / "install",
                 tmp_path / "bridge", tmp_path / "report.json")
    with pytest.raises(ValueError, match="requires explicit"):
        base.build(*arguments, identity=reuse.IDENTITY)
    assert not arguments[0].exists()


def test_candidate_bridge_configures_distinct_sources_and_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "copy_pinned_adapter", lambda *args: {"files": {}})
    runtime = {"torch_root": str(tmp_path / "torch"), "torch_npu_root": str(tmp_path / "torch_npu"),
               "torch_abi": 0, "python_executable": str(tmp_path / "python")}
    directory = tmp_path / "reuse_bridge"
    commands = []

    def runner(command, cwd, env, phase):
        commands.append(command)
        if phase == "quant bridge compile":
            directory.mkdir(parents=True, exist_ok=True)
            (directory / reuse.BRIDGE_NAME).write_bytes(b"candidate shared object")
        return 0

    report = reuse.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
                                tmp_path / "bridge.json", runtime=runtime, runner=runner)
    assert report["native_api"] == reuse.IDENTITY["native_api"]
    assert report["torch_namespace"] == "vc_laser_quant_reuse"
    assert commands[0][commands[0].index("-S") + 1] == str(reuse.BRIDGE_SOURCE.resolve())
    assert Path(report["bridge_path"]).name == reuse.BRIDGE_NAME
    assert {Path(row["path"]).parent for row in report["bridge_sources"]} == {reuse.BRIDGE_SOURCE}
    assert not (directory / base.BRIDGE_NAME).exists()


def test_candidate_cannot_reuse_baseline_cmake_cache(tmp_path, monkeypatch):
    directory = tmp_path / "existing_baseline_bridge"
    directory.mkdir()
    (directory / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={base.BRIDGE_SOURCE}\n", encoding="utf-8")
    runtime = {"torch_root": str(tmp_path / "torch"), "torch_npu_root": str(tmp_path / "torch_npu"),
               "torch_abi": 0, "python_executable": str(tmp_path / "python")}
    with pytest.raises(ValueError, match="mismatch for CMAKE_HOME_DIRECTORY"):
        reuse.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
                           tmp_path / "bridge.json", runtime=runtime,
                           runner=lambda *args: pytest.fail("Cannot compile into baseline cache"))


def test_candidate_bridge_dispatch_changes_identity_only():
    original = (base.BRIDGE_SOURCE / "register.cpp").read_bytes()
    expected = original.replace(b"vc_laser_quant", b"vc_laser_quant_reuse").replace(
        b"VcLaserQuantAttention", b"VcLaserQuantReuseAttention")
    assert (reuse.BRIDGE_SOURCE / "register.cpp").read_bytes() == expected
    cmake = (reuse.BRIDGE_SOURCE / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "-Wl,-Bsymbolic" in cmake


def test_cli_defaults_do_not_target_existing_baseline_caches():
    candidate = reuse.parser().parse_args(["--cann-root", "toolkit", "--soc", "ascend910_93"])
    baseline = base.parser().parse_args(["--cann-root", "toolkit", "--soc", "ascend910_93"])
    for key in ("source_dir", "source_manifest", "install_dir", "bridge_dir", "output"):
        assert getattr(candidate, key) != getattr(baseline, key)
        assert "quant_reuse" in getattr(candidate, key).name
    assert candidate.reference_dir == baseline.reference_dir
    assert candidate.jobs == 4
