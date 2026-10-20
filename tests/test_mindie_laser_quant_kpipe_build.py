"""CPU-only build isolation and provenance checks for the K double-buffer lookahead operator."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts import build_mindie_laser_quant as base
from scripts import build_mindie_laser_quant_kpipe as candidate
from scripts import build_mindie_laser_quant_cache as cache
from scripts import build_mindie_laser_quant_reuse as reuse
from scripts import prepare_mindie_laser_quant_cache_source as original
from scripts import prepare_mindie_laser_quant_kpipe_source as prepare
from scripts.prepare_cann_fas_source import RepositoryLock, git

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def staged(tmp_path):
    arguments = (tmp_path / "kpipe_source", tmp_path / "reference",
                 tmp_path / "source_manifest.json", tmp_path / "cann",
                 "ascend910_93", 4, tmp_path / "kpipe_install", tmp_path / "kpipe_bridge",
                 tmp_path / "results/kpipe_build.json")
    calls = []

    def prepare(source, reference, manifest):
        calls.append("prepare")
        source.mkdir(parents=True, exist_ok=True)
        base.write_json(manifest, {"native_api": candidate.IDENTITY["native_api"]})

    def validate(source, manifest):
        calls.append("validate")
        return json.loads(manifest.read_text(encoding="utf-8"))

    def native(source, cann, soc, jobs, install, output, **kwargs):
        calls.append(("native", kwargs))
        report = {"status": "passed", **candidate.IDENTITY,
                  "vendor_root": str(install / "packages/vendors/vc_laser_quant_kpipe"),
                  "host_libraries": [{"sha256": "host-proof"}],
                  "device_binaries": [{"sha256": "kernel-proof"}]}
        base.write_json(output, report)
        return report

    def bridge(source, cann, directory, jobs, output, **kwargs):
        calls.append("bridge")
        directory.mkdir(parents=True, exist_ok=True)
        binary = directory / candidate.BRIDGE_NAME
        binary.write_bytes(b"kpipe bridge fixture")
        report = {"status": "passed", **candidate.IDENTITY, "bridge_path": str(binary),
                  "bridge_sha256": base.sha256_file(binary)}
        base.write_json(output, report)
        return report

    dependencies = dict(preparer=prepare, source_validator=validate,
                        operator_builder=native, bridge_builder=bridge)
    return arguments, dependencies, calls


def test_candidate_build_keeps_distinct_identity_and_audited_artifacts(staged):
    arguments, dependencies, calls = staged
    report = candidate.build(*arguments, **dependencies)
    assert report["status"] == "passed"
    assert all(report[key] == value for key, value in candidate.IDENTITY.items())
    assert report["marker_token"] == candidate.MARKER_TOKEN
    assert report["marker_environment"] == candidate.MARKER_ENVIRONMENT
    assert report["source_manifest"]["sha256"] == base.sha256_file(arguments[2])
    assert report["bridge_sha256"] == base.sha256_file(report["bridge_path"])
    assert report["bridge_manifest_sha256"] == base.sha256_file(report["bridge_manifest"])
    assert report["device_binaries"][0]["sha256"] == "kernel-proof"
    assert next(call[1] for call in calls if isinstance(call, tuple))["identity"] == candidate.IDENTITY
    assert base.IDENTITY["native_api"] == "aclnnVcLaserQuantAttention"
    assert base.BRIDGE_NAME == "libvc_laser_quant_bridge.so"


@pytest.mark.parametrize("which", ["operator_builder", "bridge_builder"])
@pytest.mark.parametrize("foreign_identity", [base.IDENTITY, reuse.IDENTITY, cache.IDENTITY])
def test_candidate_rejects_existing_identity_before_publishing(staged, which, foreign_identity):
    arguments, dependencies, calls = staged
    original = dependencies[which]

    def foreign(*args, **kwargs):
        report = original(*args, **kwargs)
        report.update(foreign_identity)
        return report

    dependencies[which] = foreign
    with pytest.raises(ValueError, match="identify|identity"):
        candidate.build(*arguments, **dependencies)
    assert json.loads(arguments[-1].read_text(encoding="utf-8"))["status"] == "failed"
    if which == "operator_builder":
        assert "bridge" not in calls


def test_generalized_composer_cannot_use_baseline_defaults_for_other_identity(tmp_path):
    arguments = (tmp_path / "source", tmp_path / "ref", tmp_path / "manifest.json",
                 tmp_path / "cann", "ascend910_93", 4, tmp_path / "install",
                 tmp_path / "bridge", tmp_path / "report.json")
    with pytest.raises(ValueError, match="requires explicit"):
        base.build(*arguments, identity=candidate.IDENTITY)
    assert not arguments[0].exists()


def test_candidate_bridge_configures_distinct_sources_and_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "copy_pinned_adapter", lambda *args: {"files": {}})
    runtime = {"torch_root": str(tmp_path / "torch"), "torch_npu_root": str(tmp_path / "torch_npu"),
               "torch_abi": 0, "python_executable": str(tmp_path / "python")}
    directory = tmp_path / "kpipe_bridge"
    commands = []

    def runner(command, cwd, env, phase):
        commands.append(command)
        if phase == "quant bridge compile":
            directory.mkdir(parents=True, exist_ok=True)
            (directory / candidate.BRIDGE_NAME).write_bytes(b"candidate shared object")
        return 0

    report = candidate.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
                                tmp_path / "bridge.json", runtime=runtime, runner=runner)
    assert report["native_api"] == candidate.IDENTITY["native_api"]
    assert report["torch_namespace"] == "vc_laser_quant_kpipe"
    assert commands[0][commands[0].index("-S") + 1] == str(candidate.BRIDGE_SOURCE.resolve())
    assert Path(report["bridge_path"]).name == candidate.BRIDGE_NAME
    assert {Path(row["path"]).parent for row in report["bridge_sources"]} == {candidate.BRIDGE_SOURCE}
    assert not (directory / base.BRIDGE_NAME).exists()


@pytest.mark.parametrize("existing_source", [base.BRIDGE_SOURCE, reuse.BRIDGE_SOURCE, cache.BRIDGE_SOURCE])
def test_candidate_cannot_reuse_existing_cmake_cache(tmp_path, existing_source):
    directory = tmp_path / "existing_baseline_bridge"
    directory.mkdir()
    (directory / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={existing_source}\n", encoding="utf-8")
    runtime = {"torch_root": str(tmp_path / "torch"), "torch_npu_root": str(tmp_path / "torch_npu"),
               "torch_abi": 0, "python_executable": str(tmp_path / "python")}
    with pytest.raises(ValueError, match="mismatch for CMAKE_HOME_DIRECTORY"):
        candidate.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
                           tmp_path / "bridge.json", runtime=runtime,
                           runner=lambda *args: pytest.fail("Cannot compile into baseline cache"))


def test_candidate_bridge_dispatch_changes_identity_only():
    original = (base.BRIDGE_SOURCE / "register.cpp").read_bytes()
    expected = original.replace(b"vc_laser_quant", b"vc_laser_quant_kpipe").replace(
        b"VcLaserQuantAttention", b"VcLaserQuantKpipeAttention")
    assert (candidate.BRIDGE_SOURCE / "register.cpp").read_bytes() == expected
    cmake = (candidate.BRIDGE_SOURCE / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "-Wl,-Bsymbolic" in cmake


def test_cli_defaults_do_not_target_existing_baseline_caches():
    current = candidate.parser().parse_args(["--cann-root", "toolkit", "--soc", "ascend910_93"])
    baseline = base.parser().parse_args(["--cann-root", "toolkit", "--soc", "ascend910_93"])
    previous = reuse.parser().parse_args(["--cann-root", "toolkit", "--soc", "ascend910_93"])
    cached = cache.parser().parse_args(["--cann-root", "toolkit", "--soc", "ascend910_93"])
    for key in ("source_dir", "source_manifest", "install_dir", "bridge_dir", "output"):
        assert getattr(current, key) != getattr(baseline, key)
        assert getattr(current, key) != getattr(previous, key)
        assert getattr(current, key) != getattr(cached, key)
        assert "quant_kpipe" in getattr(current, key).name
    assert current.reference_dir == baseline.reference_dir
    assert current.jobs == 4


@pytest.fixture(scope="module")
def pinned():
    reference = ROOT / "build/mindie-sd-reference"
    if not (reference / ".git").exists():
        pytest.skip("Pinned MindIE source cache unavailable; tests never fetch")
    assert git(reference, "rev-parse", "HEAD").decode().strip() == prepare.SOURCE_LOCK.commit
    return reference


@pytest.fixture
def prepared(tmp_path, monkeypatch, pinned):
    # Only pinned operator files are needed for the isolated Git fixture. This
    # checks real patch anchors and stale-source handling without duplicating
    # the complete upstream repository for each destructive fixture edit.
    reference = tmp_path / "reference"
    reference.mkdir()

    def run(*args):
        return subprocess.run(["git", "-C", str(reference), *args],
                              capture_output=True, check=True).stdout

    run("init", "--quiet")
    run("config", "core.autocrlf", "false")
    run("config", "user.name", "K pipeline source test")
    run("config", "user.email", "k-pipeline@example.invalid")
    names = git(pinned, "ls-tree", "-r", "--name-only", "HEAD", "--", prepare.OP_ROOT).decode().splitlines()
    for name in [*names, prepare.PREPARE_SCRIPT]:
        path = reference / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(git(pinned, "show", f"HEAD:{name}"))
    run("add", ".")
    run("commit", "--quiet", "-m", "locked bytes fixture")
    lock = RepositoryLock("https://example.invalid/laser.git", run("rev-parse", "HEAD").decode().strip())
    monkeypatch.setattr(prepare, "SOURCE_LOCK", lock)
    source, manifest = tmp_path / "candidate", tmp_path / "candidate.json"
    report = prepare.prepare(source, reference, manifest)
    return reference, source, manifest, report


def test_source_delta_is_only_the_cube_pipeline_patch(pinned):
    old_files, old_records = original.expected_files(pinned)
    new_files, new_records = prepare.expected_files(pinned)
    changed = []
    for old_path, old_data in old_files.items():
        path = prepare.rename(old_path.encode()).decode()
        if new_files[path] != prepare.rename(old_data):
            changed.append(path)
        if old_path in old_records:
            assert new_records[path]["original_sha256"] == old_records[old_path]["original_sha256"]
            assert new_records[path]["scale_cache_prepared_sha256"] == prepare.sha256(old_data)
    assert changed == [f"{prepare.CUSTOM_OP_ROOT}/op_kernel/CubeForward.h"]
    marker = new_files[f"{prepare.CUSTOM_OP_ROOT}/op_host/vc_source_marker.h"]
    assert prepare.MARKER_TOKEN.encode() in marker and prepare.MARKER_ENV.encode() in marker
    assert original.MARKER_TOKEN.encode() not in marker
    assert all(not path.startswith(original.CUSTOM_OP_ROOT + "/") for path in new_files)
    assert prepare.IDENTITY == candidate.IDENTITY


def test_preparation_is_incremental_and_preserves_original_checkout(prepared):
    reference, source, manifest, report = prepared
    before = {name: (source / name).stat().st_mtime_ns for name in report["generated_files"]}
    manifest_before = manifest.stat().st_mtime_ns
    assert prepare.validate_prepared_source(source, manifest) == report
    assert prepare.prepare(source, reference, manifest) == report
    assert before == {name: (source / name).stat().st_mtime_ns for name in before}
    assert manifest.stat().st_mtime_ns == manifest_before
    assert git(source, "diff", "--name-only").decode().splitlines() == [prepare.PREPARE_SCRIPT]
    assert not git(reference, "status", "--porcelain")
    baseline_identity = original._identity(source)
    for key in ("qk_scale_contract", "scale_cache_contract", "baseline_scale_reuse_contract"):
        assert report[key] == baseline_identity[key]
    assert report["kpipe_contract"] == prepare.KPIPE_CONTRACT


def test_unknown_kernel_edits_fail_without_overwrite(prepared):
    reference, source, manifest, _ = prepared
    cube = source / prepare.CUSTOM_OP_ROOT / "op_kernel/CubeForward.h"
    data = cube.read_bytes() + b"// user edit\n"
    cube.write_bytes(data)
    prior = manifest.read_bytes()
    with pytest.raises(prepare.PreparationError):
        prepare.prepare(source, reference, manifest)
    with pytest.raises(prepare.PreparationError):
        prepare.validate_prepared_source(source, manifest)
    assert cube.read_bytes() == data and manifest.read_bytes() == prior


@pytest.mark.parametrize("field", ["variant", "native_api", "scale_cache_contract", "kpipe_contract", "qk_scale_contract", "baseline_scale_reuse_contract"])
def test_source_validation_rejects_stale_identity_or_cache_contract(prepared, field):
    _, source, manifest, report = prepared
    report[field] = "stale value"
    manifest.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(prepare.PreparationError, match="identity"):
        prepare.validate_prepared_source(source, manifest)


def test_preparation_is_local_only_even_when_offline_false(prepared, monkeypatch):
    reference, source, manifest, _ = prepared
    original_ensure = prepare.ensure_checkout
    seen = []

    def verify(path, lock, offline):
        seen.append(offline)
        assert offline is True
        assert os.environ["GIT_ALLOW_PROTOCOL"] == "file"
        assert os.environ["GIT_NO_LAZY_FETCH"] == "1"
        return original_ensure(path, lock, offline)

    monkeypatch.setattr(prepare, "ensure_checkout", verify)
    prepare.prepare(source, reference, manifest, offline=False)
    assert seen == [True]


def test_existing_cache_source_is_discovered_before_old_baselines(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    reuse_source = tmp_path / "custom reuse source"
    (tmp_path / "results/mindie_laser_quant_cache_build.json").write_text(
        json.dumps(dict(status="passed", source_dir=str(reuse_source))), encoding="utf-8")
    missing = tmp_path / "missing reference"
    assert prepare.cached_references(missing)[:2] == [missing, reuse_source]


def test_missing_cache_does_not_fetch_or_create_checkout(tmp_path, monkeypatch):
    source, missing = tmp_path / "candidate", tmp_path / "missing reference"
    monkeypatch.setattr(prepare, "cached_references", lambda reference: [reference])
    with pytest.raises(prepare.PreparationError, match="No local pinned"):
        prepare.prepare(source, missing, tmp_path / "candidate.json", offline=False)
    assert not source.exists()
