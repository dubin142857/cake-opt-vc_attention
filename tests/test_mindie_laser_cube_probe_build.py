"""CPU-only checks for independent Cube providers, ABI consistency and local provenance."""
import json
from pathlib import Path
import subprocess

import pytest

from scripts import build_mindie_laser_quant as composer
from scripts import build_mindie_laser_cube_probe as build
from scripts import prepare_mindie_laser_cube_probe_source as prep
from scripts import prepare_mindie_laser_source as original
from scripts.prepare_cann_fas_source import RepositoryLock, PreparationError, git

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["int8", "fp16", "vscale"])
def kind(request):
    return request.param


def test_isolated_provider_build_preserves_identity_hashes_and_source_validator(tmp_path, kind):
    args = (tmp_path / "source", tmp_path / "reference", tmp_path / "source.json",
            tmp_path / "cann", "ascend910_93", 4, tmp_path / "install", tmp_path / "bridge", tmp_path / "build.json")
    calls = []
    identity = build.identity(kind)

    def prepare(source, reference, manifest):
        source.mkdir()
        composer.write_json(manifest, {"kind": kind})

    def validate(source, manifest):
        calls.append("validate")

    def native(source, cann, soc, jobs, install, output, **kwargs):
        assert kwargs["identity"] == identity
        assert kwargs["source_validator"] is validate
        report = {"status": "passed", **identity, "vendor_root": str(install / identity["vendor"])}
        composer.write_json(output, report)
        return report

    def bridge(source, cann, directory, jobs, output, **kwargs):
        directory.mkdir()
        binary = directory / build.bridge_name(kind)
        binary.write_bytes(b"independent-bridge-" + kind.encode())
        report = {"status": "passed", **identity, "bridge_path": str(binary),
                  "bridge_sha256": composer.sha256_file(binary)}
        composer.write_json(output, report)
        return report

    report = build.build(*args, kind=kind, preparer=prepare, source_validator=validate,
                         operator_builder=native, bridge_builder=bridge)
    assert calls == ["validate"]
    assert report["status"] == "passed"
    assert report["native_api"] == identity["native_api"]
    assert report["marker_token"] == build.marker_token(kind)
    assert report["marker_environment"] == build.marker_environment(kind)
    assert report["source_manifest"]["sha256"] == composer.sha256_file(args[2])
    assert report["bridge_manifest_sha256"] == composer.sha256_file(report["bridge_manifest"])
    assert report["bridge_sha256"] == composer.sha256_file(report["bridge_path"])


def test_bridge_metadata_schema_and_target_match_kind(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(composer, "copy_pinned_adapter", lambda *args: {"files": {}})
    directory = tmp_path / "bridge"
    runtime = {"torch_root": str(tmp_path / "torch"), "torch_npu_root": str(tmp_path / "torch_npu"),
               "torch_abi": 0, "python_executable": str(tmp_path / "python")}

    def run(command, cwd, env, phase):
        if phase == "quant bridge compile":
            directory.mkdir(parents=True, exist_ok=True)
            (directory / build.bridge_name(kind)).write_bytes(b"selected shared object")
        return 0

    report = build.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
                                tmp_path / "bridge.json", kind=kind, runner=run, runtime=runtime)
    assert report["torch_schema"] == build.SCHEMAS[kind]
    assert report["torch_namespace"] == prep.identity(kind)["vendor"]
    assert report["probe_kind"] == kind
    assert json.loads((tmp_path / "bridge.json").read_text(encoding="utf-8"))["torch_schema"] == build.SCHEMAS[kind]
    assert Path(report["bridge_path"]).name == build.bridge_name(kind)
    expected_sources = build.BRIDGE_SOURCE / "vscale" if kind == "vscale" else build.BRIDGE_SOURCE
    assert {Path(row["path"]).parent for row in report["bridge_sources"]} == {expected_sources.resolve()}


def test_default_paths_are_disjoint_and_kind_is_required():
    args = [build.resolve_defaults(build.parser().parse_args(["--kind", kind, "--cann-root", "cann", "--soc", "ascend910_93"]))
            for kind in prep.KINDS]
    for field in ("source_dir", "source_manifest", "install_dir", "bridge_dir", "output"):
        assert len({getattr(arg, field) for arg in args}) == len(prep.KINDS)
        assert "cube_int8_probe" in getattr(args[0], field).name
        assert "cube_fp16_probe" in getattr(args[1], field).name
        assert "cube_vscale_probe" in getattr(args[2], field).name
    with pytest.raises(SystemExit):
        build.parser().parse_args(["--cann-root", "cann", "--soc", "ascend910_93"])
    with pytest.raises(PreparationError):
        prep.identity("fp8")


@pytest.fixture
def prepared(tmp_path, monkeypatch, kind):
    pinned = ROOT / "build/mindie-sd-reference"
    if not (pinned / ".git").exists():
        pytest.skip("Pinned source cache is absent; test never downloads")
    reference = tmp_path / "reference"
    reference.mkdir()

    def run(*args):
        return subprocess.run(["git", "-C", str(reference), *args], capture_output=True, check=True).stdout

    run("init", "--quiet")
    run("config", "core.autocrlf", "false")
    run("config", "user.name", "Cube probe source test")
    run("config", "user.email", "cube-probe@example.invalid")
    names = git(pinned, "ls-tree", "-r", "--name-only", "HEAD", "--", prep.OP_ROOT).decode().splitlines()
    for name in [*names, prep.PREPARE_SCRIPT]:
        path = reference / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(git(pinned, "show", f"HEAD:{name}"))
    run("add", ".")
    run("commit", "--quiet", "-m", "pinned Laser fixture")
    lock = RepositoryLock("https://example.invalid/offline-laser.git", run("rev-parse", "HEAD").decode().strip())
    monkeypatch.setattr(prep, "SOURCE_LOCK", lock)
    source, manifest = tmp_path / "candidate", tmp_path / "candidate.json"
    report = prep.prepare(source, reference, manifest, kind=kind)
    return kind, reference, source, manifest, report


def test_original_vector_math_is_unchanged_and_trace_abi_agrees(prepared):
    kind, reference, source, manifest, report = prepared
    identity = prep.identity(kind)
    base, _ = prep.baseline_files(source, kind)
    original_name = {"int8": "vc_laser_quant_reuse_attention", "fp16": "vc_laser_attention",
                     "vscale": "vc_laser_quant_vscale_attention"}[kind]
    raw_vector = base[f"csrc/ops/{original_name}/op_kernel/VectorForward.h"]
    operator = source / "csrc/ops" / identity["op_name"]
    assert (operator / "op_kernel/VectorForward.h").read_bytes() == prep._rename(raw_vector, kind)
    entry = (operator / "op_kernel" / (identity["op_name"] + ".cpp")).read_text(encoding="utf-8")
    assert entry.count("SetupCubeProbe(cube_trace_gm, tiling_data->probeMode)") == 1
    assert entry.index("SetupCubeProbe") < entry.index("#elif __DAV_C220_VEC__")
    assert entry.index("attention_out_gm,") < entry.index("cube_trace_gm,") < entry.index("workspace,")
    host = operator / "op_host"
    proto = (host / (identity["op_name"] + "_proto.cpp")).read_text(encoding="utf-8")
    assert "GetOutputShape(2)" in proto and "SetOutputDataType(2, ge::DT_INT64)" in proto
    for dimension, size in enumerate(prep.TRACE_SHAPE):
        assert f"traceOut->SetDim({dimension}, {size});" in proto
    tiling = (host / (identity["op_name"] + "_tiling.cpp")).read_text(encoding="utf-8")
    assert f"GetAttrPointer<int64_t>({8 if prep.quantized(kind) else 7})" in tiling
    assert "*probeMode < 0 || *probeMode > 2" in tiling
    assert f"if (tiling.Tiling4{identity['op_type']}(context) != ge::GRAPH_SUCCESS)" in tiling
    definition = (host / (identity["op_name"] + "_def.cpp")).read_text(encoding="utf-8")
    if kind == "fp16":
        assert 'Input("q_scale")' not in definition and "use_expcast" not in definition
        assert "qk_scale_contract" not in report
        assert "SetQuantScales" not in entry
    else:
        assert 'Input("q_scale")' in definition and "use_expcast" in definition
        assert "if (*useExpCast) return ge::GRAPH_FAILED;" in tiling
    if kind == "vscale":
        assert report["vector_scale_contract"] == prep.vscale.VSCALE_CONTRACT
        assert report["qk_scale_contract"] == prep.vscale.SCALE_CONTRACT
        assert report["baseline_variant"] == "quant_vector_scale"
        assert "scale_reuse_contract" not in report
        assert "baseline_scale_reuse_contract" in report
        assert "constant 1/64 FIX DEQF16 transport" in report["algorithm_scope"]
        assert b"TryDenseTileRowMax" not in raw_vector
    assert report["probe_contract"]["shape"] == [4, 2, 3, 24]


def test_preparation_incremental_read_only_reference_and_wrong_kind_rejected(prepared):
    kind, reference, source, manifest, report = prepared
    before = {name: (source / name).stat().st_mtime_ns for name in report["source_files"]}
    manifest_mtime = manifest.stat().st_mtime_ns
    assert prep.validate_prepared_source(source, manifest, kind=kind) == report
    assert prep.prepare(source, reference, manifest, kind=kind, offline=False) == report
    assert before == {name: (source / name).stat().st_mtime_ns for name in before}
    assert manifest.stat().st_mtime_ns == manifest_mtime
    assert not git(reference, "status", "--porcelain")
    assert git(source, "diff", "--name-only").decode().splitlines() == [prep.PREPARE_SCRIPT]
    with pytest.raises(PreparationError, match="identity"):
        prep.validate_prepared_source(source, manifest, kind="fp16" if kind == "int8" else "int8")


def test_unknown_generated_mutation_is_preserved_and_rejected(prepared):
    kind, reference, source, manifest, report = prepared
    vector = source / "csrc/ops" / prep.identity(kind)["op_name"] / "op_kernel/VectorForward.h"
    edited = vector.read_bytes() + b"\n// unexpected change\n"
    vector.write_bytes(edited)
    with pytest.raises(PreparationError):
        prep.prepare(source, reference, manifest, kind=kind)
    assert vector.read_bytes() == edited


def test_bridge_trace_is_initialized_and_fp16_is_a_separate_schema():
    bridge = (build.BRIDGE_SOURCE / "register.cpp").read_text()
    assert "at::zeros({4, 2, 3, 24}, query.options().dtype(at::kLong))" in bridge
    assert "probe_mode >= 0 && probe_mode <= 2" in bridge
    assert "if (pre_tokens == 0) pre_tokens = 2147483647;" in bridge
    assert "is_high_precision, use_expcast, cube_probe_mode, statistics, output, trace);" in bridge
    assert "is_high_precision, cube_probe_mode, statistics, output, trace);" in bridge
    assert "TORCH_LIBRARY(vc_laser_cube_int8_probe" in bridge
    assert "TORCH_LIBRARY(vc_laser_cube_fp16_probe" in bridge
    assert "-Wl,-Bsymbolic" in (build.BRIDGE_SOURCE / "CMakeLists.txt").read_text()


def test_vscale_bridge_only_registers_current_quantized_schema_and_rejects_expcast():
    bridge = (build.BRIDGE_SOURCE / "vscale/register.cpp").read_text()
    cmake = (build.BRIDGE_SOURCE / "vscale/CMakeLists.txt").read_text()
    assert '"aclnnVcLaserCubeVscaleProbeAttention"' in bridge
    assert "TORCH_LIBRARY(vc_laser_cube_vscale_probe, module)" in bridge
    assert "TORCH_LIBRARY_IMPL(vc_laser_cube_vscale_probe, PrivateUse1, module)" in bridge
    assert 'TORCH_CHECK(!expcast,' in bridge
    assert "is_high_precision, use_expcast, cube_probe_mode, statistics, output, trace);" in bridge
    assert "bool use_expcast = false;" in bridge
    assert "VC_CUBE_PROBE_INT8" not in bridge
    assert "vc_laser_cube_int8_probe" not in bridge
    assert "vc_laser_cube_fp16_probe" not in bridge
    assert "set(target vc_laser_cube_vscale_probe_bridge)" in cmake
    assert 'add_library(${target} SHARED register.cpp' in cmake
    assert build.SCHEMAS["vscale"] == build.SCHEMAS["int8"]


def test_vscale_prefers_current_local_cache_and_leaves_legacy_cache_order_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    cached = tmp_path / "current-vscale"
    (results / "mindie_laser_quant_vscale_build.json").write_text(json.dumps(
        {"status": "passed", "source_dir": str(cached)}))
    reference = tmp_path / "reference"
    monkeypatch.setattr(prep.vscale, "cached_references", lambda x: [x, tmp_path / "qresident"])
    monkeypatch.setattr(prep.quant, "cached_references", lambda x: [x, tmp_path / "reuse"])
    assert prep.cached_references(reference, "vscale") == [
        cached, tmp_path / "build/mindie_laser_quant_vscale_source", reference, tmp_path / "qresident"]
    assert prep.cached_references(reference, "int8") == [reference, tmp_path / "reuse"]
    assert prep.cached_references(reference, "fp16") == [reference, tmp_path / "reuse"]
