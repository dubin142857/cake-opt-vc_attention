"""CPU-only build isolation, trace ABI and provenance checks for phase diagnostics."""
import json
from pathlib import Path
import subprocess

import pytest

from scripts import build_mindie_laser_quant as base
from scripts import build_mindie_laser_quant_probe as probe
from scripts import build_mindie_laser_quant_reuse as reuse
from scripts import build_mindie_laser_quant_gain as gain
from scripts.prepare_cann_fas_source import RepositoryLock, git

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def staged(tmp_path):
    arguments = (tmp_path / "probe_source", tmp_path / "reference",
                 tmp_path / "source_manifest.json", tmp_path / "cann",
                 "ascend910_93", 4, tmp_path / "probe_install", tmp_path / "probe_bridge",
                 tmp_path / "results/probe_build.json")
    calls = []

    def prepare(source, reference, manifest):
        calls.append("prepare")
        source.mkdir(parents=True, exist_ok=True)
        base.write_json(manifest, {"native_api": probe.IDENTITY["native_api"]})

    def validate(source, manifest):
        calls.append("validate")
        return json.loads(manifest.read_text(encoding="utf-8"))

    def native(source, cann, soc, jobs, install, output, **kwargs):
        calls.append(("native", kwargs))
        report = {"status": "passed", **probe.IDENTITY,
                  "vendor_root": str(install / "packages/vendors/vc_laser_quant_probe"),
                  "host_libraries": [{"sha256": "host-proof"}],
                  "device_binaries": [{"sha256": "kernel-proof"}]}
        base.write_json(output, report)
        return report

    def bridge(source, cann, directory, jobs, output, **kwargs):
        calls.append("bridge")
        directory.mkdir(parents=True, exist_ok=True)
        binary = directory / probe.BRIDGE_NAME
        binary.write_bytes(b"probe bridge fixture")
        report = {"status": "passed", **probe.IDENTITY, "bridge_path": str(binary),
                  "bridge_sha256": base.sha256_file(binary)}
        base.write_json(output, report)
        return report

    dependencies = dict(preparer=prepare, source_validator=validate,
                        operator_builder=native, bridge_builder=bridge)
    return arguments, dependencies, calls


def test_candidate_build_keeps_distinct_identity_and_audited_artifacts(staged):
    arguments, dependencies, calls = staged
    report = probe.build(*arguments, **dependencies)
    assert report["status"] == "passed"
    assert all(report[key] == value for key, value in probe.IDENTITY.items())
    assert report["marker_token"] == probe.MARKER_TOKEN
    assert report["marker_environment"] == probe.MARKER_ENVIRONMENT
    assert report["source_manifest"]["sha256"] == base.sha256_file(arguments[2])
    assert report["bridge_sha256"] == base.sha256_file(report["bridge_path"])
    assert report["bridge_manifest_sha256"] == base.sha256_file(report["bridge_manifest"])
    assert report["device_binaries"][0]["sha256"] == "kernel-proof"
    assert next(call[1] for call in calls if isinstance(call, tuple))["identity"] == probe.IDENTITY
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
        probe.build(*arguments, **dependencies)
    assert json.loads(arguments[-1].read_text(encoding="utf-8"))["status"] == "failed"
    if which == "operator_builder":
        assert "bridge" not in calls


def test_generalized_composer_cannot_use_baseline_defaults_for_other_identity(tmp_path):
    arguments = (tmp_path / "source", tmp_path / "ref", tmp_path / "manifest.json",
                 tmp_path / "cann", "ascend910_93", 4, tmp_path / "install",
                 tmp_path / "bridge", tmp_path / "report.json")
    with pytest.raises(ValueError, match="requires explicit"):
        base.build(*arguments, identity=probe.IDENTITY)
    assert not arguments[0].exists()


def test_candidate_bridge_configures_distinct_sources_and_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "copy_pinned_adapter", lambda *args: {"files": {}})
    runtime = {"torch_root": str(tmp_path / "torch"), "torch_npu_root": str(tmp_path / "torch_npu"),
               "torch_abi": 0, "python_executable": str(tmp_path / "python")}
    directory = tmp_path / "probe_bridge"
    commands = []

    def runner(command, cwd, env, phase):
        commands.append(command)
        if phase == "quant bridge compile":
            directory.mkdir(parents=True, exist_ok=True)
            (directory / probe.BRIDGE_NAME).write_bytes(b"candidate shared object")
        return 0

    report = probe.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
                                tmp_path / "bridge.json", runtime=runtime, runner=runner)
    assert report["native_api"] == probe.IDENTITY["native_api"]
    assert report["torch_namespace"] == "vc_laser_quant_probe"
    assert commands[0][commands[0].index("-S") + 1] == str(probe.BRIDGE_SOURCE.resolve())
    assert Path(report["bridge_path"]).name == probe.BRIDGE_NAME
    assert {Path(row["path"]).parent for row in report["bridge_sources"]} == {probe.BRIDGE_SOURCE}
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
        probe.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
                           tmp_path / "bridge.json", runtime=runtime,
                           runner=lambda *args: pytest.fail("Cannot compile into baseline cache"))


def test_probe_bridge_trace_abi_and_all_modes_zero_initialization():
    source = (probe.BRIDGE_SOURCE / "register.cpp").read_text(encoding="utf-8")
    assert 'std::tuple<at::Tensor, at::Tensor> QuantLaser' in source
    assert 'bool expcast, int64_t probe_mode)' in source
    assert 'TORCH_CHECK(!expcast' in source
    assert 'probe_mode >= 0 && probe_mode <= 2' in source
    assert 'auto statistics = at_npu::native::empty_with_format(' in source
    assert 'auto trace = at::zeros({4, 3, 8, 24}, query.options().dtype(at::kLong));' in source
    assert 'is_high_precision, use_expcast, phase_probe_mode, statistics, output, trace);' in source
    assert 'return std::make_tuple(output, trace);' in source
    assert 'int probe_mode) -> (Tensor, Tensor)' in source
    cmake = (probe.BRIDGE_SOURCE / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "-Wl,-Bsymbolic" in cmake


def test_cli_defaults_do_not_target_existing_baseline_caches():
    candidate = probe.parser().parse_args(["--cann-root", "toolkit", "--soc", "ascend910_93"])
    baseline = base.parser().parse_args(["--cann-root", "toolkit", "--soc", "ascend910_93"])
    previous = reuse.parser().parse_args(["--cann-root", "toolkit", "--soc", "ascend910_93"])
    previous_gain = gain.parser().parse_args(["--cann-root", "toolkit", "--soc", "ascend910_93"])
    for key in ("source_dir", "source_manifest", "install_dir", "bridge_dir", "output"):
        assert getattr(candidate, key) != getattr(baseline, key)
        assert getattr(candidate, key) != getattr(previous, key)
        assert getattr(candidate, key) != getattr(previous_gain, key)
        assert "quant_probe" in getattr(candidate, key).name
    assert candidate.reference_dir == baseline.reference_dir
    assert candidate.jobs == 4


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from scripts import prepare_mindie_laser_quant_probe_source as prepare
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
    run("config", "user.name", "probe preparation source test")
    run("config", "user.email", "probe-preparation@example.invalid")
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
    assert prepare.IDENTITY == probe.IDENTITY
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


def test_manifest_rejects_wrong_variant_api_or_probe_contract(prepared):
    prepare, _, source, manifest, report = prepared
    for key, value in (("variant", "quant_scale_reuse"), ("native_api", reuse.IDENTITY["native_api"]),
                       ("probe_contract", {"scope": "unrestricted"})):
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
    from scripts import prepare_mindie_laser_quant_probe_source as prepare
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    reuse_source = tmp_path / "custom reuse source"
    (tmp_path / "results/mindie_laser_quant_reuse_build.json").write_text(
        json.dumps(dict(status="passed", source_dir=str(reuse_source))), encoding="utf-8")
    missing = tmp_path / "missing reference"
    assert prepare.cached_references(missing)[:2] == [missing, reuse_source]


def test_output_and_attribute_abi_match_across_all_generated_components():
    from scripts import prepare_mindie_laser_quant_probe_source as prepare
    from scripts import prepare_mindie_laser_quant_reuse_source as source_base
    pinned = ROOT / "build/mindie-sd-reference"
    if not (pinned / ".git").exists():
        pytest.skip("Pinned source cache absent; ABI test never fetches")
    original, _ = source_base.expected_files(pinned)
    generated = {prepare.rename(name.encode()).decode(): prepare.patch_abi(
        prepare.rename(name.encode()).decode(), prepare.rename(data)) for name, data in original.items()}
    stem = prepare.CUSTOM_OP_ROOT
    definition = generated[f"{stem}/op_host/{prepare.OP_NAME}_def.cpp"].decode()
    assert definition.index('Output("softmax_log_max_sum")') < definition.index('Output("attention_out")')
    assert definition.index('Output("attention_out")') < definition.index('Output("phase_trace")')
    attrs = __import__("re").findall(r'this->Attr\("([^"]+)"\)', definition)
    assert attrs[7:] == ["use_expcast", "probe_mode"]
    assert 'DataType({ge::DT_INT64})' in definition
    proto = generated[f"{stem}/op_host/{prepare.OP_NAME}_proto.cpp"].decode()
    assert 'GetOutputShape(2)' in proto and 'SetOutputDataType(2, ge::DT_INT64)' in proto
    for index, dim in enumerate(prepare.TRACE_SHAPE):
        assert f'traceOut->SetDim({index}, {dim});' in proto
    tiling = generated[f"{stem}/op_host/{prepare.OP_NAME}_tiling.cpp"].decode()
    assert 'GetAttrPointer<int64_t>(8)' in tiling
    assert 'if (probeMode == nullptr || *probeMode < 0 || *probeMode > 2)' in tiling
    assert 'if (*useExpCast) return ge::GRAPH_FAILED;' in tiling
    assert f'if (tiling.Tiling4{prepare.OP_TYPE}(context) != ge::GRAPH_SUCCESS)' in tiling
    assert 'set_probeMode(static_cast<int32_t>(*probeMode))' in tiling
    assert b'TILING_DATA_FIELD_DEF(int32_t, probeMode);' in generated[f"{stem}/op_host/{prepare.OP_NAME}_tiling.h"]
    entry = generated[f"{stem}/op_kernel/{prepare.OP_NAME}.cpp"].decode()
    assert entry.index('softmax_log_max_sum_gm,') < entry.index('attention_out_gm,') < entry.index('phase_trace_gm,') < entry.index('workspace,')
    assert entry.count('op.SetupProbe(phase_trace_gm, tiling_data->probeMode);\n    op.Run();') == 2


def test_probe_header_is_in_source_manifest_and_tampering_is_rejected(prepared):
    prepare, reference, source, manifest, report = prepared
    name = f"{prepare.CUSTOM_OP_ROOT}/op_kernel/vc_laser_phase_probe.h"
    header = source / name
    assert report["diagnostic_instrumentation"] is True
    assert report["probe_contract"]["shape"] == [4, 3, 8, 24]
    assert report["generated_files"][name]["prepared_sha256"] == prepare.sha256(header.read_bytes())
    content = header.read_bytes() + b"// changed probe\n"
    header.write_bytes(content)
    with pytest.raises(prepare.PreparationError):
        prepare.validate_prepared_source(source, manifest)
    with pytest.raises(prepare.PreparationError):
        prepare.prepare(source, reference, manifest)
    assert header.read_bytes() == content
