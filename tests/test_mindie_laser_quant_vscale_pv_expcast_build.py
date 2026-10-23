"""Isolation, offline provenance and cache checks for FP16 ExpCast on stable PV prefetch."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts import build_mindie_laser_quant as compositor
from scripts import build_mindie_laser_quant_vscale_pv_prefetch as previous_build
from scripts import build_mindie_laser_quant_vscale_pv_expcast as candidate
from scripts import prepare_mindie_laser_quant_vscale_pv_prefetch_source as previous
from scripts import prepare_mindie_laser_quant_vscale_pv_expcast_source as prepare
from scripts.prepare_cann_fas_source import RepositoryLock, git

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def staged(tmp_path):
    args = (tmp_path / "source", tmp_path / "reference", tmp_path / "source.json",
            tmp_path / "cann", "ascend910_93", 4, tmp_path / "install",
            tmp_path / "bridge", tmp_path / "build.json")

    def preparer(source, reference, manifest):
        source.mkdir(parents=True)
        compositor.write_json(manifest, {"native_api": candidate.IDENTITY["native_api"]})

    def validate(source, manifest):
        return json.loads(manifest.read_text(encoding="utf-8"))

    def native(source, cann, soc, jobs, install, output, **kwargs):
        assert kwargs["identity"] == candidate.IDENTITY
        report = {"status": "passed", **candidate.IDENTITY,
                  "vendor_root": str(install / "packages/vendors" / candidate.IDENTITY["vendor"]),
                  "host_libraries": [{"sha256": "host-proof"}],
                  "device_binaries": [{"sha256": "device-proof"}]}
        compositor.write_json(output, report)
        return report

    def bridge(source, cann, directory, jobs, output, **kwargs):
        directory.mkdir(parents=True)
        binary = directory / candidate.BRIDGE_NAME
        binary.write_bytes(b"independent candidate bridge")
        report = {"status": "passed", **candidate.IDENTITY, "bridge_path": str(binary),
                  "bridge_sha256": compositor.sha256_file(binary)}
        compositor.write_json(output, report)
        return report

    return args, dict(preparer=preparer, source_validator=validate,
                      operator_builder=native, bridge_builder=bridge)


def test_build_publishes_distinct_verified_artifacts(staged):
    args, dependencies = staged
    report = candidate.build(*args, **dependencies)
    assert report["status"] == "passed"
    assert all(report[key] == value for key, value in candidate.IDENTITY.items())
    assert report["marker_token"] == candidate.MARKER_TOKEN
    assert report["marker_environment"] == candidate.MARKER_ENVIRONMENT
    assert report["source_manifest"]["sha256"] == compositor.sha256_file(args[2])
    assert report["bridge_sha256"] == compositor.sha256_file(report["bridge_path"])
    assert report["bridge_manifest_sha256"] == compositor.sha256_file(report["bridge_manifest"])
    assert report["device_binaries"][0]["sha256"] == "device-proof"
    assert previous_build.IDENTITY["native_api"] == "aclnnVcLaserQuantVscalePvPrefetchAttention"


@pytest.mark.parametrize("which", ["operator_builder", "bridge_builder"])
def test_parent_build_cannot_be_mistaken_for_candidate(staged, which):
    args, dependencies = staged
    operation = dependencies[which]

    def old_identity(*args, **kwargs):
        return {**operation(*args, **kwargs), **previous_build.IDENTITY}

    dependencies[which] = old_identity
    with pytest.raises(ValueError, match="identify|identity"):
        candidate.build(*args, **dependencies)
    assert json.loads(args[-1].read_text(encoding="utf-8"))["status"] == "failed"


def test_bridge_only_renames_identity_and_enables_existing_expcast_argument():
    for name in ("register.cpp", "CMakeLists.txt"):
        expected = (previous_build.BRIDGE_SOURCE / name).read_bytes().replace(
            b"VcLaserQuantVscalePvPrefetchAttention", b"VcLaserQuantVscalePvExpcastAttention").replace(
            b"vc_laser_quant_vscale_pv_prefetch", b"vc_laser_quant_vscale_pv_expcast")
        if name == "register.cpp":
            guard = (b"    TORCH_CHECK(!expcast,\n"
                     b'                "Vector-scale Laser currently supports original-exp mode only (expcast=false)");\n')
            assert expected.count(guard) == 1
            expected = expected.replace(guard, b"", 1)
        assert (candidate.BRIDGE_SOURCE / name).read_bytes() == expected
    code = (candidate.BRIDGE_SOURCE / "register.cpp").read_text(encoding="utf-8")
    assert "TORCH_CHECK(!expcast," not in code
    assert "bool use_expcast = expcast;" in code
    assert '"float scale, int keys, bool expcast) -> Tensor"' in code


def test_bridge_uses_new_sources_and_rejects_parent_cmake_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(compositor, "copy_pinned_adapter", lambda *args: {"files": {}})
    runtime = {"torch_root": str(tmp_path / "torch"), "torch_npu_root": str(tmp_path / "torch_npu"),
               "torch_abi": 0, "python_executable": str(tmp_path / "python")}
    directory = tmp_path / "bridge"
    commands = []

    def runner(command, cwd, env, phase):
        commands.append(command)
        if phase == "quant bridge compile":
            directory.mkdir(parents=True, exist_ok=True)
            (directory / candidate.BRIDGE_NAME).write_bytes(b"pv prefetch bridge")
        return 0

    report = candidate.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
                                   tmp_path / "bridge.json", runtime=runtime, runner=runner)
    assert report["torch_namespace"] == candidate.TORCH_NAMESPACE
    assert commands[0][commands[0].index("-S") + 1] == str(candidate.BRIDGE_SOURCE.resolve())
    (directory / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={previous_build.BRIDGE_SOURCE}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch for CMAKE_HOME_DIRECTORY"):
        candidate.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
                               tmp_path / "bridge.json", runtime=runtime,
                               runner=lambda *args: pytest.fail("Must not compile into parent cache"))


def test_cli_caches_are_distinct_and_default_reference_is_existing_stable():
    arguments = ["--cann-root", "toolkit", "--soc", "ascend910_93"]
    current, parent = candidate.parser().parse_args(arguments), previous_build.parser().parse_args(arguments)
    for field in ("source_dir", "source_manifest", "install_dir", "bridge_dir", "output"):
        assert getattr(current, field) != getattr(parent, field)
        assert "quant_vscale_pv_expcast" in getattr(current, field).name
    assert current.reference_dir == parent.source_dir
    assert current.jobs == 4


@pytest.fixture(scope="module")
def pinned():
    reference = ROOT / "build/mindie-sd-reference"
    if not (reference / ".git").exists():
        pytest.skip("Pinned MindIE cache unavailable; tests never fetch")
    assert git(reference, "rev-parse", "HEAD").decode().strip() == prepare.SOURCE_LOCK.commit
    return reference


@pytest.fixture
def prepared(tmp_path, monkeypatch, pinned):
    reference = tmp_path / "reference"
    reference.mkdir()

    def run(*args):
        return subprocess.run(["git", "-C", str(reference), *args],
                              capture_output=True, check=True).stdout

    run("init", "--quiet")
    run("config", "core.autocrlf", "false")
    run("config", "user.name", "FP16 ExpCast fixture")
    run("config", "user.email", "fp16-expcast@example.invalid")
    names = git(pinned, "ls-tree", "-r", "--name-only", "HEAD", "--", prepare.OP_ROOT).decode().splitlines()
    for name in [*names, prepare.PREPARE_SCRIPT]:
        path = reference / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(git(pinned, "show", f"HEAD:{name}"))
    run("add", ".")
    run("commit", "--quiet", "-m", "pinned operator bytes")
    lock = RepositoryLock("https://example.invalid/laser.git", run("rev-parse", "HEAD").decode().strip())
    monkeypatch.setattr(prepare, "SOURCE_LOCK", lock)
    source, manifest = tmp_path / "candidate", tmp_path / "candidate.json"
    report = prepare.prepare(source, reference, manifest)
    return reference, source, manifest, report


def test_only_probability_helper_changes_and_parent_provenance_is_recorded(pinned):
    old_files, old_records = previous.expected_files(pinned)
    new_files, records = prepare.expected_files(pinned)
    changed = []
    for old_path, old_data in old_files.items():
        path = prepare.rename(old_path.encode()).decode()
        if new_files[path] != prepare.rename(old_data):
            changed.append(path)
        if old_path in old_records:
            assert records[path]["pv_prefetch_prepared_sha256"] == prepare.sha256(old_data)
            for field in ("original_sha256", "q_resident_prepared_sha256", "scale_cache_prepared_sha256"):
                assert records[path][field] == old_records[old_path][field]
    assert changed == [f"{prepare.CUSTOM_OP_ROOT}/op_kernel/vc_laser_probability.h"]
    marker = new_files[f"{prepare.CUSTOM_OP_ROOT}/op_host/vc_source_marker.h"]
    assert prepare.MARKER_TOKEN.encode() in marker and prepare.MARKER_ENV.encode() in marker
    assert previous.MARKER_TOKEN.encode() not in marker
    assert prepare.IDENTITY == candidate.IDENTITY


def test_cube_vector_calls_and_original_exp_branch_are_preserved(pinned):
    from scripts.mindie_laser_expcast_patch import BULK_VEXP, EXPERIMENTAL_PROBABILITY
    old_files, _ = previous.expected_files(pinned)
    new_files, _ = prepare.expected_files(pinned)
    for name in ("CubeForward.h", "VectorForward.h", "AddressMappingForwardOnline.h",
                 "AddressMappingVectorForwardOnline.h"):
        old_path = f"{previous.CUSTOM_OP_ROOT}/op_kernel/{name}"
        new_path = prepare.rename(old_path.encode()).decode()
        assert new_files[new_path] == prepare.rename(old_files[old_path])
    vector = new_files[f"{prepare.CUSTOM_OP_ROOT}/op_kernel/VectorForward.h"]
    assert vector.count(BULK_VEXP) == 1
    assert vector.count(EXPERIMENTAL_PROBABILITY) == 1
    assert b"if (use_expcast_)" in vector
    tiling = new_files[f"{prepare.CUSTOM_OP_ROOT}/op_host/{prepare.OP_NAME}_tiling.cpp"]
    assert b"if (useExpCast == nullptr) return ge::GRAPH_FAILED;" in tiling
    assert b"tilingData.set_useExpCast(*useExpCast);" in tiling
    assert b"!*useExpCast" not in tiling
    entry = new_files[f"{prepare.CUSTOM_OP_ROOT}/op_kernel/{prepare.OP_NAME}.cpp"]
    assert b"op.SetExpCast(tiling_data->useExpCast);" in entry
    assert all(b"CubeProbe" not in d and b"GetSystemCycle" not in d for d in new_files.values())


def test_new_probability_helper_is_required_and_not_silently_fallback(pinned, tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "PROBABILITY_HEADER", tmp_path / "missing_helper.h")
    with pytest.raises(FileNotFoundError):
        prepare.expected_files(pinned)


def test_incremental_offline_preparation_preserves_reference_and_contract(prepared, monkeypatch):
    reference, source, manifest, report = prepared
    before = {name: (source / name).stat().st_mtime_ns for name in report["generated_files"]}
    manifest_before = manifest.stat().st_mtime_ns
    original_ensure = prepare.ensure_checkout
    calls = []

    def local_ensure(path, lock, offline):
        calls.append(offline)
        assert offline is True
        assert os.environ["GIT_ALLOW_PROTOCOL"] == "file"
        assert os.environ["GIT_NO_LAZY_FETCH"] == "1"
        return original_ensure(path, lock, offline)

    monkeypatch.setattr(prepare, "ensure_checkout", local_ensure)
    assert prepare.prepare(source, reference, manifest, offline=False) == report
    assert prepare.validate_prepared_source(source, manifest) == report
    assert calls == [True, True]
    assert before == {name: (source / name).stat().st_mtime_ns for name in before}
    assert manifest.stat().st_mtime_ns == manifest_before
    assert git(source, "diff", "--name-only").decode().splitlines() == [prepare.PREPARE_SCRIPT]
    assert not git(reference, "status", "--porcelain")
    assert report["vector_scale_contract"] == previous.baseline.VSCALE_CONTRACT
    assert report["pv_prefetch_contract"] == previous.CONTRACT
    assert report["fp16_expcast_contract"] == prepare.CONTRACT
    assert report["original_exp_control"] is True
    assert report["probability_helper_sha256"] == prepare.sha256(prepare.PROBABILITY_HEADER.read_bytes())
    assert report["qk_scale_contract"] == previous.SCALE_CONTRACT
    assert report["bitwise_baseline_equivalent"] is False


def test_unknown_source_edits_are_not_overwritten(prepared):
    reference, source, manifest, _ = prepared
    vector = source / prepare.CUSTOM_OP_ROOT / "op_kernel/VectorForward.h"
    edited = vector.read_bytes() + b"// unknown user edit\n"
    vector.write_bytes(edited)
    prior = manifest.read_bytes()
    with pytest.raises(prepare.PreparationError):
        prepare.prepare(source, reference, manifest)
    with pytest.raises(prepare.PreparationError):
        prepare.validate_prepared_source(source, manifest)
    assert vector.read_bytes() == edited and manifest.read_bytes() == prior


@pytest.mark.parametrize("field", ["fp16_expcast_contract", "probability_helper_sha256", "original_exp_control",
                                  "pv_prefetch_contract", "vector_scale_contract",
                                  "qk_scale_contract", "op_type", "native_api",
                                  "marker_token", "marker_environment", "upstream"])
def test_manifest_cannot_claim_another_source_contract(prepared, field):
    _, source, manifest, report = prepared
    report[field] = "stale"
    manifest.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(prepare.PreparationError, match="identity"):
        prepare.validate_prepared_source(source, manifest)


def test_stable_cache_is_discovered_first_and_missing_objects_never_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    cached = tmp_path / "existing stable PV prefetch"
    (tmp_path / "results/mindie_laser_quant_vscale_pv_prefetch_build.json").write_text(
        json.dumps(dict(status="passed", source_dir=str(cached))), encoding="utf-8")
    missing = tmp_path / "missing"
    assert prepare.cached_references(missing)[:2] == [missing, cached]
    source = tmp_path / "candidate"
    with pytest.raises(prepare.PreparationError, match="No local pinned"):
        prepare.prepare(source, missing, tmp_path / "candidate.json", offline=False)
    assert not source.exists()
