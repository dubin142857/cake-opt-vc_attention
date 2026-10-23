"""PV diagnostic ABI, offline provenance and isolation; no CCE/NPU claim."""
import json
import os
from pathlib import Path

import pytest

from scripts import build_mindie_laser_cube_probe as build
from scripts import build_mindie_laser_quant as composer
from scripts import prepare_mindie_laser_cube_probe_source as prep
from scripts import mindie_laser_pv_probe_patch as patch
from scripts.prepare_cann_fas_source import PreparationError
from tests.test_mindie_laser_cube_probe_build import prepared
from tests.test_mindie_laser_cube_probe_build import (
    test_isolated_provider_build_preserves_identity_hashes_and_source_validator as _compose,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def kind():
    return "pv"


def test_independent_provider_composition(tmp_path):
    _compose(tmp_path, "pv")


def test_pv_bridge_uses_dedicated_sources_and_rejects_old_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(composer, "copy_pinned_adapter", lambda *args: {"files": {}})
    runtime = {"torch_root": str(tmp_path / "torch"), "torch_npu_root": str(tmp_path / "torch_npu"),
               "torch_abi": 0, "python_executable": str(tmp_path / "python")}
    directory = tmp_path / "bridge"

    def runner(command, cwd, env, phase):
        if phase == "quant bridge compile":
            directory.mkdir(parents=True, exist_ok=True)
            (directory / build.bridge_name("pv")).write_bytes(b"isolated-pv-probe")
        return 0

    report = build.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
        tmp_path / "bridge.json", kind="pv", runner=runner, runtime=runtime)
    assert report["torch_namespace"] == "vc_laser_cube_pv_probe"
    assert report["torch_schema"] == build.SCHEMAS["vscale"]
    assert report["probe_kind"] == "pv"
    assert {Path(row["path"]).parent for row in report["bridge_sources"]} == {(build.BRIDGE_SOURCE / "pv").resolve()}
    (directory / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={build.BRIDGE_SOURCE / 'vscale'}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CMAKE_HOME_DIRECTORY"):
        build.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
            tmp_path / "bridge.json", kind="pv", runtime=runtime,
            runner=lambda *args: pytest.fail("Must not compile into existing vscale bridge"))


def test_source_contract_abi_and_v_only_parent(prepared):
    _, reference, source, manifest, report = prepared
    identity = prep.identity("pv")
    operator = source / "csrc/ops" / identity["op_name"]
    baseline, _ = prep.pv.expected_files(source)
    original_cube = baseline[f"{prep.pv.CUSTOM_OP_ROOT}/op_kernel/CubeForward.h"]
    original_vector = baseline[f"{prep.pv.CUSTOM_OP_ROOT}/op_kernel/VectorForward.h"]
    assert patch.strip_hooks((operator / "op_kernel/CubeForward.h").read_bytes()) == prep._rename(original_cube, "pv")
    assert (operator / "op_kernel/VectorForward.h").read_bytes() == prep._rename(original_vector, "pv")
    cube = (operator / "op_kernel/CubeForward.h").read_bytes()
    assert b"pv_v_prefetch" in cube and b"pv_p_prefetch" not in cube
    assert report["baseline_variant"] == "quant_vector_scale_pv_prefetch"
    assert report["pv_prefetch_contract"] == prep.pv.CONTRACT
    assert report["vector_scale_contract"] == prep.vscale.VSCALE_CONTRACT
    assert report["probe_contract"] == prep.PV_PROBE_CONTRACT
    assert report["probe_contract"]["schema_version"] == 2
    assert report["probe_contract"]["shape"] == [4, 3, 3, 64]
    assert "pv_p_prefetch_contract" not in report
    name = identity["op_name"]
    proto = (operator / "op_host" / (name + "_proto.cpp")).read_text(encoding="utf-8")
    for dim, length in enumerate(prep.PV_TRACE_SHAPE):
        assert f"traceOut->SetDim({dim}, {length});" in proto
    assert "GetOutputShape(2)" in proto and "SetOutputDataType(2, ge::DT_INT64)" in proto
    entry = (operator / "op_kernel" / (name + ".cpp")).read_text(encoding="utf-8")
    assert entry.count("SetupCubeProbe(cube_trace_gm, tiling_data->probeMode)") == 1
    assert entry.index("SetupCubeProbe") < entry.index("#elif __DAV_C220_VEC__")
    tiling = (operator / "op_host" / (name + "_tiling.cpp")).read_text(encoding="utf-8")
    assert "GetAttrPointer<int64_t>(8)" in tiling
    assert "*probeMode < 0 || *probeMode > 2" in tiling
    assert "if (*useExpCast) return ge::GRAPH_FAILED;" in tiling


def test_offline_incremental_reuse_and_wrong_family_rejection(prepared, monkeypatch):
    _, reference, source, manifest, report = prepared
    before = {name: (source / name).stat().st_mtime_ns for name in report["generated_files"]}
    stamp = manifest.stat().st_mtime_ns
    original = prep.ensure_checkout

    def offline(path, lock, offline):
        assert offline is True
        assert os.environ["GIT_ALLOW_PROTOCOL"] == "file"
        assert os.environ["GIT_NO_LAZY_FETCH"] == "1"
        return original(path, lock, offline)

    monkeypatch.setattr(prep, "ensure_checkout", offline)
    assert prep.prepare(source, reference, manifest, kind="pv", offline=False) == report
    assert prep.validate_prepared_source(source, manifest, kind="pv") == report
    assert before == {name: (source / name).stat().st_mtime_ns for name in before}
    assert manifest.stat().st_mtime_ns == stamp
    with pytest.raises(PreparationError, match="identity"):
        prep.validate_prepared_source(source, manifest, kind="vscale")


@pytest.mark.parametrize("field", ["probe_contract", "pv_prefetch_contract", "vector_scale_contract", "native_api"])
def test_stale_source_identity_rejected(prepared, field):
    _, _, source, manifest, report = prepared
    report[field] = "incorrect"
    manifest.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(PreparationError, match="identity"):
        prep.validate_prepared_source(source, manifest, kind="pv")


def test_unknown_probe_edits_preserved(prepared):
    _, reference, source, manifest, _ = prepared
    header = source / "csrc/ops" / prep.identity("pv")["op_name"] / "op_kernel/vc_laser_pv_probe.h"
    changed = header.read_bytes() + b"// unknown edit\n"
    header.write_bytes(changed)
    with pytest.raises(PreparationError):
        prep.prepare(source, reference, manifest, kind="pv")
    assert header.read_bytes() == changed


def test_bridge_schema_trace_allocation_and_separate_defaults():
    code = (build.BRIDGE_SOURCE / "pv/register.cpp").read_text(encoding="utf-8")
    assert "at::zeros({4, 3, 3, 64}, query.options().dtype(at::kLong))" in code
    assert 'TORCH_CHECK(!expcast,' in code
    assert '"aclnnVcLaserCubePvProbeAttention"' in code
    assert "probe_mode >= 0 && probe_mode <= 2" in code
    assert "TORCH_LIBRARY(vc_laser_cube_pv_probe, module)" in code
    arguments = [build.resolve_defaults(build.parser().parse_args(
        ["--kind", kind, "--cann-root", "cann", "--soc", "ascend910_93"])) for kind in prep.KINDS]
    for field in ("source_dir", "source_manifest", "install_dir", "bridge_dir", "output"):
        assert len({getattr(args, field) for args in arguments}) == len(prep.KINDS)


def test_v2_build_preserves_v1_evidence_and_other_provider_defaults():
    for kind in prep.KINDS:
        args = build.resolve_defaults(build.parser().parse_args(
            ["--kind", kind, "--cann-root", "cann", "--soc", "ascend910_93"]))
        prefix = f"mindie_laser_cube_{kind}_probe"
        cache = prefix + "_v2" if kind == "pv" else prefix
        for field, suffix in (("source_dir", "_source"), ("source_manifest", "_source_manifest.json"),
                              ("install_dir", "_install"), ("bridge_dir", "_bridge")):
            assert getattr(args, field) == ROOT / "build" / (cache + suffix)
        assert args.output == ROOT / "results" / (prefix + "_build.json")
