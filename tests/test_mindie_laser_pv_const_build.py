"""Constant-entry PV diagnostic isolation: host checks, never launch a device."""
import json
import os
from pathlib import Path
import subprocess
import types

import pytest

from scripts import build_mindie_laser_cube_probe as build
from scripts import build_mindie_laser_quant as composer
from scripts import prepare_mindie_laser_cube_probe_source as prep
from scripts.prepare_cann_fas_source import PreparationError, git
from tests.test_mindie_laser_cube_probe_build import prepared
from tests.test_mindie_laser_cube_probe_build import (
    test_isolated_provider_build_preserves_identity_hashes_and_source_validator as _compose,
)

ROOT = Path(__file__).resolve().parents[1]
OLD_COMMIT = "b76f855f570c7b4789b480515ea5a4c4b8d01022"


@pytest.fixture
def kind():
    return "pv_const"


def _back_to_v2(data):
    new, old = prep.identity("pv_const"), prep.identity("pv")
    for source, target in (
        (new["op_type"], old["op_type"]), (new["op_name"], old["op_name"]),
        (prep.marker_token("pv_const"), prep.marker_token("pv")),
        (prep.marker_environment("pv_const"), prep.marker_environment("pv")),
    ):
        data = data.replace(source.encode(), target.encode())
    return data


def test_independent_constant_plain_provider_composition(tmp_path):
    _compose(tmp_path, "pv_const")


def test_v2_device_sources_differ_only_at_literal_entry_mode(prepared):
    _, _, source, _, report = prepared
    previous, _ = prep.expected_files(source, "pv")
    candidate, _ = prep.expected_files(source, "pv_const")
    normalized = {_back_to_v2(name.encode()).decode(): _back_to_v2(data)
                  for name, data in candidate.items()}
    assert normalized.keys() == previous.keys()
    changes = []
    for name, data in normalized.items():
        if name.endswith("/op_kernel/vc_laser_cube_pv_probe_attention.cpp"):
            replacement = b"op.SetupCubeProbe(cube_trace_gm, 0);"
            assert data.count(replacement) == 1
            data = data.replace(replacement, b"op.SetupCubeProbe(cube_trace_gm, tiling_data->probeMode);")
            changes.append(name)
        elif name.endswith("/op_host/vc_laser_cube_pv_probe_attention_tiling.cpp"):
            assert data.count(b"*probeMode != 0") == 1
            data = data.replace(b"*probeMode != 0", b"*probeMode < 0 || *probeMode > 2")
            changes.append(name)
        assert data == previous[name], name
    assert len(changes) == 2
    root = f"csrc/ops/{prep.identity('pv_const')['op_name']}/op_kernel/"
    cube = candidate[root + "CubeForward.h"]
    assert b"vc_laser_pv_probe::PvProbe pv_probe_;" in cube
    assert b"pv_probe_.BeginCall" in cube and b"pv_probe_.Start" in cube
    assert candidate[root + "vc_laser_pv_probe.h"] == (ROOT / "ascend/laser_cube_probe/vc_laser_pv_probe.h").read_bytes()
    assert report["diagnostic_instrumentation"] is True
    assert report["probe_contract"] == prep.PV_CONST_CONTRACT
    assert report["pv_prefetch_contract"] == prep.pv.CONTRACT
    assert report["vector_scale_contract"] == prep.vscale.VSCALE_CONTRACT


def test_trace_abi_and_zero_mode_guards(prepared):
    _, _, source, _, _ = prepared
    native = prep.identity("pv_const")
    assert native == dict(op_name="vc_laser_cube_pv_const_probe_attention",
                         op_type="VcLaserCubePvConstProbeAttention",
                         native_api="aclnnVcLaserCubePvConstProbeAttention",
                         vendor="vc_laser_cube_pv_const_probe")
    root = source / "csrc/ops" / native["op_name"]
    read = lambda suffix: (root / "op_host" / (native["op_name"] + suffix)).read_text(encoding="utf-8")
    proto, definition, tiling = read("_proto.cpp"), read("_def.cpp"), read("_tiling.cpp")
    assert definition.index('Output("attention_out")') < definition.index('Output("cube_trace")')
    assert definition.index('Attr("use_expcast")') < definition.index('Attr("probe_mode")')
    for dimension, size in enumerate([4, 3, 3, 64]):
        assert f"traceOut->SetDim({dimension}, {size});" in proto
    assert "GetOutputShape(2)" in proto and "SetOutputDataType(2, ge::DT_INT64)" in proto
    assert "GetAttrPointer<int64_t>(8)" in tiling
    assert "if (probeMode == nullptr || *probeMode != 0) return ge::GRAPH_FAILED;" in tiling
    assert "if (*useExpCast) return ge::GRAPH_FAILED;" in tiling
    bridge = (build.BRIDGE_SOURCE / "pv_const/register.cpp").read_text(encoding="utf-8")
    assert 'TORCH_CHECK(probe_mode == 0, "Constant-plain provider requires probe_mode=0")' in bridge
    assert 'TORCH_CHECK(!expcast,' in bridge
    assert "at::zeros({4, 3, 3, 64}, query.options().dtype(at::kLong))" in bridge
    assert "is_high_precision, use_expcast, cube_probe_mode, statistics, output, trace);" in bridge
    assert "TORCH_LIBRARY(vc_laser_cube_pv_const_probe, module)" in bridge
    assert '"aclnnVcLaserCubePvConstProbeAttention"' in bridge
    assert build.SCHEMAS["pv_const"] == build.SCHEMAS["pv"]


def test_local_objects_only_and_incremental_source(prepared, monkeypatch):
    _, reference, source, manifest, report = prepared
    before = {name: (source / name).stat().st_mtime_ns for name in report["generated_files"]}
    original = prep.ensure_checkout

    def offline(path, lock, offline):
        assert offline is True
        assert os.environ["GIT_ALLOW_PROTOCOL"] == "file"
        assert os.environ["GIT_NO_LAZY_FETCH"] == "1"
        return original(path, lock, offline)

    monkeypatch.setattr(prep, "ensure_checkout", offline)
    assert prep.prepare(source, reference, manifest, kind="pv_const", offline=False) == report
    assert prep.validate_prepared_source(source, manifest, kind="pv_const") == report
    assert before == {name: (source / name).stat().st_mtime_ns for name in before}
    assert not git(reference, "status", "--porcelain")
    for wrong in ("pv", "pv_abi"):
        with pytest.raises(PreparationError, match="identity"):
            prep.validate_prepared_source(source, manifest, kind=wrong)


@pytest.mark.parametrize("field", ["probe_contract", "pv_prefetch_contract", "diagnostic_instrumentation", "native_api"])
def test_stale_source_contract_is_rejected(prepared, field):
    _, _, source, manifest, report = prepared
    report[field] = "foreign"
    manifest.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(PreparationError, match="identity"):
        prep.validate_prepared_source(source, manifest, kind="pv_const")


def test_modified_instrumentation_is_not_silently_overwritten(prepared):
    _, reference, source, manifest, _ = prepared
    header = source / "csrc/ops" / prep.identity("pv_const")["op_name"] / "op_kernel/vc_laser_pv_probe.h"
    edited = header.read_bytes() + b"\n// unexpected mutation\n"
    header.write_bytes(edited)
    with pytest.raises(PreparationError, match="unknown|Unexpected|modified|changed"):
        prep.prepare(source, reference, manifest, kind="pv_const")
    assert header.read_bytes() == edited


def test_bridge_and_build_caches_are_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(composer, "copy_pinned_adapter", lambda *args: {"files": {}})
    runtime = {"torch_root": str(tmp_path / "torch"), "torch_npu_root": str(tmp_path / "torch_npu"),
               "torch_abi": 0, "python_executable": str(tmp_path / "python")}
    directory = tmp_path / "bridge"

    def run(command, cwd, env, phase):
        if phase == "quant bridge compile":
            directory.mkdir(parents=True, exist_ok=True)
            (directory / build.bridge_name("pv_const")).write_bytes(b"fake-host-build-only")
        return 0

    report = build.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
        tmp_path / "bridge.json", kind="pv_const", runner=run, runtime=runtime)
    assert report["probe_kind"] == "pv_const"
    assert {Path(row["path"]).parent for row in report["bridge_sources"]} == {(build.BRIDGE_SOURCE / "pv_const").resolve()}
    (directory / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={build.BRIDGE_SOURCE / 'pv_abi'}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CMAKE_HOME_DIRECTORY"):
        build.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
            tmp_path / "bridge.json", kind="pv_const", runtime=runtime,
            runner=lambda *args: pytest.fail("Must not overwrite an existing control bridge"))
    options = {kind: build.resolve_defaults(build.parser().parse_args(
        ["--kind", kind, "--cann-root", "cann", "--soc", "ascend910_93"])) for kind in prep.KINDS}
    for field in ("source_dir", "source_manifest", "install_dir", "bridge_dir", "output"):
        assert len({getattr(option, field) for option in options.values()}) == len(prep.KINDS)
        assert "cube_pv_const_probe" in getattr(options["pv_const"], field).name


def test_all_five_existing_sources_identities_bridges_and_defaults_unchanged():
    reference = ROOT / "build/mindie-sd-reference"
    if not (reference / ".git").exists():
        pytest.skip("No pinned local objects; never fetch")
    previous = {}
    for name in ("prepare_mindie_laser_cube_probe_source", "build_mindie_laser_cube_probe"):
        old = types.ModuleType("scripts.pv_const_previous_" + name)
        old.__file__, old.__package__ = str(ROOT / "scripts" / (name + ".py")), "scripts"
        code = git(ROOT, "show", f"{OLD_COMMIT}:scripts/{name}.py")
        exec(compile(code, old.__file__, "exec"), old.__dict__)
        previous[name] = old
    old = previous["prepare_mindie_laser_cube_probe_source"]
    old_build = previous["build_mindie_laser_cube_probe"]
    for kind in ("int8", "fp16", "vscale", "pv", "pv_abi"):
        assert prep.expected_files(reference, kind) == old.expected_files(reference, kind)
        assert prep._manifest_identity(reference, kind) == old._manifest_identity(reference, kind)
        assert prep.cached_references(reference, kind) == old.cached_references(reference, kind)
        args = ["--kind", kind, "--cann-root", "cann", "--soc", "ascend910_93"]
        assert vars(build.resolve_defaults(build.parser().parse_args(args))) == vars(
            old_build.resolve_defaults(old_build.parser().parse_args(args)))
        assert build.SCHEMAS[kind] == old_build.SCHEMAS[kind]
    old_bridges = ["ascend/laser_cube_probe/bridge/register.cpp", "ascend/laser_cube_probe/bridge/CMakeLists.txt"]
    old_bridges += [f"ascend/laser_cube_probe/bridge/{kind}/{name}" for kind in ("pv", "vscale", "pv_abi")
                    for name in ("register.cpp", "CMakeLists.txt")]
    result = subprocess.run(["git", "diff", OLD_COMMIT, "--", *old_bridges], cwd=ROOT,
                            capture_output=True, check=True)
    assert not result.stdout
