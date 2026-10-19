"""CPU-only build isolation and provenance checks for the score-prefetch operator."""
import json
from pathlib import Path
import subprocess

import pytest

from scripts import build_mindie_laser_quant as base
from scripts import build_mindie_laser_quant_prefetch as prefetch
from scripts import build_mindie_laser_quant_reuse as reuse
from scripts import build_mindie_laser_quant_gain as gain
from scripts.prepare_cann_fas_source import RepositoryLock, git

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def staged(tmp_path):
    arguments = (tmp_path / "prefetch_source", tmp_path / "reference",
                 tmp_path / "source_manifest.json", tmp_path / "cann",
                 "ascend910_93", 4, tmp_path / "prefetch_install", tmp_path / "prefetch_bridge",
                 tmp_path / "results/prefetch_build.json")
    calls = []

    def prepare(source, reference, manifest):
        calls.append("prepare")
        source.mkdir(parents=True, exist_ok=True)
        base.write_json(manifest, {"native_api": prefetch.IDENTITY["native_api"]})

    def validate(source, manifest):
        calls.append("validate")
        return json.loads(manifest.read_text(encoding="utf-8"))

    def native(source, cann, soc, jobs, install, output, **kwargs):
        calls.append(("native", kwargs))
        report = {"status": "passed", **prefetch.IDENTITY,
                  "vendor_root": str(install / "packages/vendors/vc_laser_quant_prefetch"),
                  "host_libraries": [{"sha256": "host-proof"}],
                  "device_binaries": [{"sha256": "kernel-proof"}]}
        base.write_json(output, report)
        return report

    def bridge(source, cann, directory, jobs, output, **kwargs):
        calls.append("bridge")
        directory.mkdir(parents=True, exist_ok=True)
        binary = directory / prefetch.BRIDGE_NAME
        binary.write_bytes(b"prefetch bridge fixture")
        report = {"status": "passed", **prefetch.IDENTITY, "bridge_path": str(binary),
                  "bridge_sha256": base.sha256_file(binary)}
        base.write_json(output, report)
        return report

    dependencies = dict(preparer=prepare, source_validator=validate,
                        operator_builder=native, bridge_builder=bridge)
    return arguments, dependencies, calls


def test_candidate_build_keeps_distinct_identity_and_audited_artifacts(staged):
    arguments, dependencies, calls = staged
    report = prefetch.build(*arguments, **dependencies)
    assert report["status"] == "passed"
    assert all(report[key] == value for key, value in prefetch.IDENTITY.items())
    assert report["marker_token"] == prefetch.MARKER_TOKEN
    assert report["marker_environment"] == prefetch.MARKER_ENVIRONMENT
    assert report["source_manifest"]["sha256"] == base.sha256_file(arguments[2])
    assert report["bridge_sha256"] == base.sha256_file(report["bridge_path"])
    assert report["bridge_manifest_sha256"] == base.sha256_file(report["bridge_manifest"])
    assert report["device_binaries"][0]["sha256"] == "kernel-proof"
    assert next(call[1] for call in calls if isinstance(call, tuple))["identity"] == prefetch.IDENTITY
    assert base.IDENTITY["native_api"] == "aclnnVcLaserQuantAttention"
    assert base.BRIDGE_NAME == "libvc_laser_quant_bridge.so"


@pytest.mark.parametrize("which", ["operator_builder", "bridge_builder"])
@pytest.mark.parametrize("foreign_identity", [base.IDENTITY, reuse.IDENTITY, gain.IDENTITY])
def test_candidate_rejects_existing_identity_before_publishing(staged, which, foreign_identity):
    arguments, dependencies, calls = staged
    original = dependencies[which]

    def foreign(*args, **kwargs):
        report = original(*args, **kwargs)
        report.update(foreign_identity)
        return report

    dependencies[which] = foreign
    with pytest.raises(ValueError, match="identify|identity"):
        prefetch.build(*arguments, **dependencies)
    assert json.loads(arguments[-1].read_text(encoding="utf-8"))["status"] == "failed"
    if which == "operator_builder":
        assert "bridge" not in calls


def test_generalized_composer_cannot_use_baseline_defaults_for_other_identity(tmp_path):
    arguments = (tmp_path / "source", tmp_path / "ref", tmp_path / "manifest.json",
                 tmp_path / "cann", "ascend910_93", 4, tmp_path / "install",
                 tmp_path / "bridge", tmp_path / "report.json")
    with pytest.raises(ValueError, match="requires explicit"):
        base.build(*arguments, identity=prefetch.IDENTITY)
    assert not arguments[0].exists()


def test_candidate_bridge_configures_distinct_sources_and_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "copy_pinned_adapter", lambda *args: {"files": {}})
    runtime = {"torch_root": str(tmp_path / "torch"), "torch_npu_root": str(tmp_path / "torch_npu"),
               "torch_abi": 0, "python_executable": str(tmp_path / "python")}
    directory = tmp_path / "prefetch_bridge"
    commands = []

    def runner(command, cwd, env, phase):
        commands.append(command)
        if phase == "quant bridge compile":
            directory.mkdir(parents=True, exist_ok=True)
            (directory / prefetch.BRIDGE_NAME).write_bytes(b"candidate shared object")
        return 0

    report = prefetch.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
                                tmp_path / "bridge.json", runtime=runtime, runner=runner)
    assert report["native_api"] == prefetch.IDENTITY["native_api"]
    assert report["torch_namespace"] == "vc_laser_quant_prefetch"
    assert commands[0][commands[0].index("-S") + 1] == str(prefetch.BRIDGE_SOURCE.resolve())
    assert Path(report["bridge_path"]).name == prefetch.BRIDGE_NAME
    assert {Path(row["path"]).parent for row in report["bridge_sources"]} == {prefetch.BRIDGE_SOURCE}
    assert not (directory / base.BRIDGE_NAME).exists()


@pytest.mark.parametrize("existing_source", [base.BRIDGE_SOURCE, reuse.BRIDGE_SOURCE, gain.BRIDGE_SOURCE])
def test_candidate_cannot_reuse_existing_cmake_cache(tmp_path, existing_source):
    directory = tmp_path / "existing_baseline_bridge"
    directory.mkdir()
    (directory / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={existing_source}\n", encoding="utf-8")
    runtime = {"torch_root": str(tmp_path / "torch"), "torch_npu_root": str(tmp_path / "torch_npu"),
               "torch_abi": 0, "python_executable": str(tmp_path / "python")}
    with pytest.raises(ValueError, match="mismatch for CMAKE_HOME_DIRECTORY"):
        prefetch.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
                           tmp_path / "bridge.json", runtime=runtime,
                           runner=lambda *args: pytest.fail("Cannot compile into baseline cache"))


def test_candidate_bridge_dispatch_changes_identity_only():
    original = (base.BRIDGE_SOURCE / "register.cpp").read_bytes()
    expected = original.replace(b"vc_laser_quant", b"vc_laser_quant_prefetch").replace(
        b"VcLaserQuantAttention", b"VcLaserQuantPrefetchAttention")
    assert (prefetch.BRIDGE_SOURCE / "register.cpp").read_bytes() == expected
    cmake = (prefetch.BRIDGE_SOURCE / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "-Wl,-Bsymbolic" in cmake


def test_cli_defaults_do_not_target_existing_baseline_caches():
    candidate = prefetch.parser().parse_args(["--cann-root", "toolkit", "--soc", "ascend910_93"])
    baseline = base.parser().parse_args(["--cann-root", "toolkit", "--soc", "ascend910_93"])
    previous = reuse.parser().parse_args(["--cann-root", "toolkit", "--soc", "ascend910_93"])
    previous_gain = gain.parser().parse_args(["--cann-root", "toolkit", "--soc", "ascend910_93"])
    for key in ("source_dir", "source_manifest", "install_dir", "bridge_dir", "output"):
        assert getattr(candidate, key) != getattr(baseline, key)
        assert getattr(candidate, key) != getattr(previous, key)
        assert getattr(candidate, key) != getattr(previous_gain, key)
        assert "quant_prefetch" in getattr(candidate, key).name
    assert candidate.reference_dir == baseline.reference_dir
    assert candidate.jobs == 4


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from scripts import prepare_mindie_laser_quant_prefetch_source as prepare
    pinned = ROOT / "build/mindie-sd-reference"
    if not (pinned / ".git").exists():
        pytest.skip("Pinned MindIE source cache unavailable; tests never fetch")
    assert git(pinned, "rev-parse", "HEAD").decode().strip() == prepare.SOURCE_LOCK.commit
    reference = tmp_path / "reference"
    reference.mkdir()

    def run(*args):
        return subprocess.run(["git", "-C", str(reference), *args], capture_output=True, check=True).stdout

    run("init", "--quiet")
    run("config", "core.autocrlf", "false")
    run("config", "user.name", "prefetch preparation source test")
    run("config", "user.email", "prefetch-preparation@example.invalid")
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
    report = prepare.prepare(source, reference, manifest, offline=True)
    return prepare, reference, source, manifest, report


def test_preparation_is_incremental_and_does_not_modify_reference(prepared):
    prepare, reference, source, manifest, report = prepared
    before = {name: (source / name).stat().st_mtime_ns for name in report["source_files"]}
    manifest_before = manifest.stat().st_mtime_ns
    assert prepare.IDENTITY == prefetch.IDENTITY
    assert prepare.validate_prepared_source(source, manifest) == report
    assert prepare.prepare(source, reference, manifest, offline=True) == report
    assert before == {name: (source / name).stat().st_mtime_ns for name in before}
    assert manifest.stat().st_mtime_ns == manifest_before
    assert git(source, "diff", "--name-only").decode().splitlines() == [prepare.PREPARE_SCRIPT]
    assert not git(reference, "status", "--porcelain")


def test_unknown_vector_edits_fail_without_overwrite(prepared):
    prepare, reference, source, manifest, _ = prepared
    vector = source / prepare.CUSTOM_OP_ROOT / "op_kernel/VectorForward.h"
    data = vector.read_bytes() + b"// user edit\n"
    vector.write_bytes(data)
    prior = manifest.read_bytes()
    with pytest.raises(prepare.PreparationError):
        prepare.prepare(source, reference, manifest)
    with pytest.raises(prepare.PreparationError):
        prepare.validate_prepared_source(source, manifest)
    assert vector.read_bytes() == data and manifest.read_bytes() == prior


def test_manifest_rejects_wrong_variant_api_or_prefetch_contract(prepared):
    prepare, _, source, manifest, report = prepared
    for key, value in (("variant", "quant_scale_reuse"), ("native_api", reuse.IDENTITY["native_api"]),
                       ("prefetch_contract", {"scope": "unrestricted"})):
        manifest.write_text(json.dumps({**report, key: value}), encoding="utf-8")
        with pytest.raises(prepare.PreparationError, match="identity"):
            prepare.validate_prepared_source(source, manifest)


def test_preparation_ignores_legacy_offline_false_and_never_fetches_remote(prepared, monkeypatch):
    prepare, reference, source, manifest, _ = prepared
    original = prepare.ensure_checkout
    seen = []

    def verify(path, lock, offline):
        seen.append(offline)
        assert offline is True
        return original(path, lock, offline)

    monkeypatch.setattr(prepare, "ensure_checkout", verify)
    prepare.prepare(source, reference, manifest, offline=False)
    assert seen == [True]


def test_existing_reuse_source_is_discovered_before_old_baselines(tmp_path, monkeypatch):
    from scripts import prepare_mindie_laser_quant_prefetch_source as prepare
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    reuse_source = tmp_path / "custom reuse source"
    (tmp_path / "results/mindie_laser_quant_reuse_build.json").write_text(
        json.dumps(dict(status="passed", source_dir=str(reuse_source))), encoding="utf-8")
    missing = tmp_path / "missing reference"
    assert prepare.cached_references(missing)[:2] == [missing, reuse_source]
