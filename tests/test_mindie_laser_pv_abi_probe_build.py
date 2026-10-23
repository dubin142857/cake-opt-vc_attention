"""Uninstrumented device bytes behind an independent trace ABI; host checks only."""
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
OLD_COMMIT = "f5adc1a"


@pytest.fixture
def kind():
    return "pv_abi"


def test_independent_abi_only_provider_composition(tmp_path):
    _compose(tmp_path, "pv_abi")


def test_device_bodies_are_exact_parent_and_only_entry_argument_is_added(prepared):
    _, _, source, _, report = prepared
    base, _ = prep.pv.expected_files(source)
    actual, _ = prep.expected_files(source, "pv_abi")
    native = prep.identity("pv_abi")
    target = f"csrc/ops/{native['op_name']}"
    for name, data in base.items():
        if "/op_kernel/" not in name:
            continue
        new_name = prep._rename(name.encode(), "pv_abi").decode()
        new_data = actual[new_name]
        if new_name.endswith(native["op_name"] + ".cpp"):
            added = b"    __gm__ uint8_t * __restrict__ cube_trace_gm,\n"
            assert new_data.count(added) == 1
            assert new_data.count(b"cube_trace_gm") == 1  # unused output pointer
            assert new_data.replace(added, b"") == prep._rename(data, "pv_abi")
        else:
            assert new_data == prep._rename(data, "pv_abi")
        assert b"PvProbe" not in new_data and b"SetupCubeProbe" not in new_data
        assert b"GetSystemCycle" not in new_data
    assert not any(name.endswith(("vc_laser_pv_probe.h", "vc_laser_cube_probe.h")) for name in actual)
    cube = actual[target + "/op_kernel/CubeForward.h"]
    assert b"pv_v_prefetch" in cube and b"pv_p_prefetch" not in cube
    assert report["diagnostic_instrumentation"] is False
    assert report["probe_contract"] == prep.PV_ABI_CONTRACT
    assert report["baseline_variant"] == "quant_vector_scale_pv_prefetch"
    assert report["pv_prefetch_contract"] == prep.pv.CONTRACT
    assert report["vector_scale_contract"] == prep.vscale.VSCALE_CONTRACT


def test_trace_abi_order_and_zero_mode_contract(prepared):
    _, _, source, _, _ = prepared
    native = prep.identity("pv_abi")
    assert native == dict(op_name="vc_laser_cube_pv_abi_probe_attention", op_type="VcLaserCubePvAbiProbeAttention",
                         native_api="aclnnVcLaserCubePvAbiProbeAttention", vendor="vc_laser_cube_pv_abi_probe")
    root = source / "csrc/ops" / native["op_name"]
    read = lambda suffix: (root / "op_host" / (native["op_name"] + suffix)).read_text(encoding="utf-8")
    proto, definition, tiling = read("_proto.cpp"), read("_def.cpp"), read("_tiling.cpp")
    assert 'Output("softmax_log_max_sum")' in definition
    assert definition.index('Output("attention_out")') < definition.index('Output("cube_trace")')
    assert definition.index('Attr("use_expcast")') < definition.index('Attr("probe_mode")')
    for index, size in enumerate([4, 3, 3, 64]):
        assert f"traceOut->SetDim({index}, {size});" in proto
    assert "GetOutputShape(2)" in proto and "SetOutputDataType(2, ge::DT_INT64)" in proto
    assert "GetAttrPointer<int64_t>(8)" in tiling
    assert "if (probeMode == nullptr || *probeMode != 0) return ge::GRAPH_FAILED;" in tiling
    assert "if (*useExpCast) return ge::GRAPH_FAILED;" in tiling
    entry = (root / "op_kernel" / (native["op_name"] + ".cpp")).read_text(encoding="utf-8")
    assert entry.index("attention_out_gm,") < entry.index("cube_trace_gm,") < entry.index("workspace,")
    assert "probeMode" not in entry and "SetupCubeProbe" not in entry
    bridge = (build.BRIDGE_SOURCE / "pv_abi/register.cpp").read_text(encoding="utf-8")
    assert 'TORCH_CHECK(probe_mode == 0, "ABI-only provider requires probe_mode=0")' in bridge
    assert "at::zeros({4, 3, 3, 64}, query.options().dtype(at::kLong))" in bridge
    assert "is_high_precision, use_expcast, cube_probe_mode, statistics, output, trace);" in bridge
    assert "TORCH_LIBRARY(vc_laser_cube_pv_abi_probe, module)" in bridge
    assert '"aclnnVcLaserCubePvAbiProbeAttention"' in bridge
    assert build.SCHEMAS["pv_abi"] == build.SCHEMAS["pv"]


def test_offline_incremental_source_and_immutable_parent(prepared, monkeypatch):
    _, reference, source, manifest, report = prepared
    before = {name: (source / name).stat().st_mtime_ns for name in report["generated_files"]}
    original = prep.ensure_checkout

    def offline(path, lock, offline):
        assert offline is True
        assert os.environ["GIT_ALLOW_PROTOCOL"] == "file"
        assert os.environ["GIT_NO_LAZY_FETCH"] == "1"
        return original(path, lock, offline)

    monkeypatch.setattr(prep, "ensure_checkout", offline)
    assert prep.prepare(source, reference, manifest, kind="pv_abi", offline=False) == report
    assert prep.validate_prepared_source(source, manifest, kind="pv_abi") == report
    assert before == {name: (source / name).stat().st_mtime_ns for name in before}
    assert not git(reference, "status", "--porcelain")
    with pytest.raises(PreparationError, match="identity"):
        prep.validate_prepared_source(source, manifest, kind="pv")


@pytest.mark.parametrize("field", ["probe_contract", "pv_prefetch_contract", "diagnostic_instrumentation", "native_api"])
def test_stale_or_foreign_source_identity_rejected(prepared, field):
    _, _, source, manifest, report = prepared
    report[field] = "foreign"
    manifest.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(PreparationError, match="identity"):
        prep.validate_prepared_source(source, manifest, kind="pv_abi")


def test_unexpected_probe_header_is_not_silently_overwritten(prepared):
    _, reference, source, manifest, _ = prepared
    header = source / "csrc/ops" / prep.identity("pv_abi")["op_name"] / "op_kernel/vc_laser_pv_probe.h"
    header.write_bytes(b"// unexpected instrumentation\n")
    with pytest.raises(PreparationError, match="unknown|Unexpected"):
        prep.prepare(source, reference, manifest, kind="pv_abi")
    assert header.read_bytes() == b"// unexpected instrumentation\n"


def test_independent_bridge_cache_and_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(composer, "copy_pinned_adapter", lambda *args: {"files": {}})
    runtime = {"torch_root": str(tmp_path / "torch"), "torch_npu_root": str(tmp_path / "torch_npu"),
               "torch_abi": 0, "python_executable": str(tmp_path / "python")}
    directory = tmp_path / "bridge"

    def run(command, cwd, env, phase):
        if phase == "quant bridge compile":
            directory.mkdir(parents=True, exist_ok=True)
            (directory / build.bridge_name("pv_abi")).write_bytes(b"fake-host-build-only")
        return 0

    report = build.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
        tmp_path / "bridge.json", kind="pv_abi", runner=run, runtime=runtime)
    assert report["probe_kind"] == "pv_abi"
    assert {Path(row["path"]).parent for row in report["bridge_sources"]} == {(build.BRIDGE_SOURCE / "pv_abi").resolve()}
    (directory / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={build.BRIDGE_SOURCE / 'pv'}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CMAKE_HOME_DIRECTORY"):
        build.build_bridge(tmp_path / "source", tmp_path / "cann", directory, 4,
            tmp_path / "bridge.json", kind="pv_abi", runtime=runtime,
            runner=lambda *args: pytest.fail("Must not overwrite the v2 diagnostic bridge"))
    options = [build.resolve_defaults(build.parser().parse_args(
        ["--kind", kind, "--cann-root", "cann", "--soc", "ascend910_93"])) for kind in prep.KINDS]
    for field in ("source_dir", "source_manifest", "install_dir", "bridge_dir", "output"):
        assert len({getattr(option, field) for option in options}) == len(prep.KINDS)
        assert "cube_pv_abi_probe" in getattr(options[-1], field).name
        assert "cube_pv_probe_v2" in getattr(options[-2], field).name or field == "output"


def test_all_four_existing_sources_identities_and_bridges_unchanged():
    reference = ROOT / "build/mindie-sd-reference"
    if not (reference / ".git").exists():
        pytest.skip("No pinned local objects; never fetch")
    old = types.ModuleType("scripts.pv_abi_previous_preparer")
    old.__file__, old.__package__ = prep.__file__, "scripts"
    code = git(ROOT, "show", f"{OLD_COMMIT}:scripts/prepare_mindie_laser_cube_probe_source.py")
    exec(compile(code, old.__file__, "exec"), old.__dict__)
    for kind in ("int8", "fp16", "vscale", "pv"):
        assert prep.expected_files(reference, kind) == old.expected_files(reference, kind)
        assert prep._manifest_identity(reference, kind) == old._manifest_identity(reference, kind)
        assert prep.cached_references(reference, kind) == old.cached_references(reference, kind)
    old_bridges = ["ascend/laser_cube_probe/bridge/register.cpp", "ascend/laser_cube_probe/bridge/CMakeLists.txt"]
    old_bridges += [f"ascend/laser_cube_probe/bridge/{kind}/{name}" for kind in ("pv", "vscale")
                    for name in ("register.cpp", "CMakeLists.txt")]
    result = subprocess.run(["git", "diff", OLD_COMMIT, "--", *old_bridges], cwd=ROOT,
                            capture_output=True, check=True)
    assert not result.stdout
