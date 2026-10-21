"""Pinned V-only hook boundaries and all original math/event sequence retention."""
from pathlib import Path
import re

import pytest

from scripts import prepare_mindie_laser_quant_vscale_pv_prefetch_source as parent
from scripts import mindie_laser_pv_probe_patch as patch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cube():
    reference = ROOT / "build/mindie-sd-reference"
    if not (reference / ".git").exists():
        pytest.skip("No pinned local cache; tests never fetch")
    files, _ = parent.expected_files(reference)
    return files[f"{parent.CUSTOM_OP_ROOT}/op_kernel/CubeForward.h"]


def test_additive_inverse_and_all_original_intrinsics(cube):
    new = patch.patch_cube(cube)
    assert patch.strip_hooks(new) == cube
    raw = rb"(?:set_flag|wait_flag|wait_flag_dev|ffts_cross_core_sync|pipe_barrier|set_quant_pre|mad|copy_[a-z_0-9]+|load_cbuf_[a-z_0-9]+)\([^;]*\);"
    assert re.findall(raw, new) == re.findall(raw, cube)
    assert b"PIPE_ALL" not in new
    assert b"pv_p_prefetch" not in new


def test_mad_fix_pair_and_history_are_only_passively_bracketed(cube):
    new = patch.patch_cube(cube)
    block = new[new.index(b"// VC_PV_PROBE_BEGIN MAD_FIX_BEGIN"):new.index(b"// VC_PV_PROBE_END MAD_FIX_END")]
    assert block.index(b"pv_probe_.Start(38)") < block.index(b"mad(")
    assert block.index(b"copy_matrix_cc_to_gm(") < block.index(b"pv_probe_.Stop(38)")
    inside = block[block.index(b"mad("):block.index(b"pv_probe_.Stop(38)")]
    assert b"pv_probe_" not in inside and b"PIPE_ALL" not in inside
    run = new[new.index(b"::Run() {"):]
    assert run.count(b"pv_probe_.EndCall();") == 3
    for logical in (b"roundId", b"roundId - 2", b"roundId + Z - 2"):
        assert b"pv_probe_.BeginCall(" + logical + b", Z);" in run
    header = (ROOT / "ascend/laser_cube_probe/vc_laser_pv_probe.h").read_bytes()
    assert header.count(b"AscendC::PipeBarrier<PIPE_ALL>();") == 1
    end = header[header.index(b"inline void EndCall()"):]
    assert end.index(b"issue_end = Tick()") < end.index(b"PipeBarrier<PIPE_ALL>") < end.index(b"record.SetValue")


def test_missing_and_repeated_pinned_hooks_fail_closed(cube):
    with pytest.raises(ValueError, match="already present"):
        patch.patch_cube(patch.patch_cube(cube))
    with pytest.raises(ValueError):
        patch.patch_cube(cube.replace(b"cube2_matmul_op(src, src_len, 1, roundId - 2);", b"changed();"))
    with pytest.raises(ValueError):
        patch.strip_hooks(patch.patch_cube(cube).replace(b"VC_PV_PROBE_END MAD_FIX_END", b"missing marker"))


def test_other_source_untouched():
    for name in ("VectorForward.h", "AddressMappingForwardOnline.h", "AddressMappingVectorForwardOnline.h"):
        assert patch.apply("csrc/op_kernel/" + name, b"unmodified") == b"unmodified"
