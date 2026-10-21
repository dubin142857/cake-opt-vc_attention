"""Pinned-byte invariance and real-header C++ checks, not CCE/NPU validation."""
from pathlib import Path
import re

import pytest

from scripts import mindie_laser_cube_probe_patch as patch
from scripts import prepare_mindie_laser_quant_reuse_source as reuse
from scripts import prepare_mindie_laser_quant_vscale_source as vscale
from scripts.prepare_cann_fas_source import git
from tests.test_mindie_laser_expcast_cpp import cpp_compiler, compile_syntax

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", params=["int8", "fp16", "vscale"])
def cube(request):
    reference = ROOT / "build/mindie-sd-reference"
    if not (reference / ".git").is_dir():
        pytest.skip("Locked source cache unavailable; tests do not download")
    if request.param in ("int8", "vscale"):
        parent = reuse if request.param == "int8" else vscale
        files, _ = parent.expected_files(reference)
        return files[f"{parent.CUSTOM_OP_ROOT}/op_kernel/CubeForward.h"]
    return git(reference, "show", "HEAD:csrc/ops/laser_attention/op_kernel/CubeForward.h")


def remove_hooks(data):
    data = data.replace(b'#include "vc_laser_cube_probe.h"\n', b"")
    data = data.replace(b"    vc_laser_cube_probe::CubeProbe cube_probe_;\n", b"")
    data = re.sub(rb"    __aicore__ inline void SetupCubeProbe\(__gm__ uint8_t \*trace, int32_t mode\) \{.*?\n    \}\n",
                  b"", data, flags=re.S)
    return b"".join(line for line in data.splitlines(keepends=True) if b"cube_probe_." not in line)


def test_inverse_is_byte_exact_including_all_arithmetic_and_flags(cube):
    changed = patch.patch_cube(cube)
    assert changed != cube
    assert remove_hooks(changed) == cube
    raw = rb"(?:set_flag|wait_flag|wait_flag_dev|ffts_cross_core_sync|pipe_barrier|mad|copy_matrix_cc_to_gm)\([^;]*\);"
    assert re.findall(raw, changed) == re.findall(raw, cube)


def test_no_probe_inside_any_matmul_or_diagonal_helper(cube):
    changed = patch.patch_cube(cube)
    run = changed.index(b"__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::Run()")
    declaration_end = changed.index(b"template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>\ntemplate <typename T_LEFT")
    assert b"cube_probe_" not in changed[declaration_end:run]
    assert b"Complete(" not in changed
    assert b"PIPE_ALL" not in changed


def test_scope_includes_mapping_whole_call_and_correct_logical_round(cube):
    changed = patch.patch_cube(cube)
    qk = re.findall(rb"cube_probe_\.Begin\(0, ([^,]+), Z\);\n(.*?)cube_probe_\.EndQK\(src, src_len\);", changed, re.S)
    assert len(qk) == 3
    for logical, body in qk:
        assert logical == b"roundId"
        assert body.index(b"address.addrMapping_cube1(") < body.index(b"cube1_matmul_op(")
        assert body.count(b";") == 2
    pv = re.findall(rb"cube_probe_\.Begin\(1, ([^,]+), Z\);\n(.*?)cube_probe_\.EndPV\(src, src_len\);", changed, re.S)
    assert [row[0] for row in pv] == [b"roundId", b"roundId - 2", b"roundId + Z - 2"]
    for logical, body in pv:
        assert body.index(b"address.addrMapping_cube2(") < body.index(b"cube2_matmul_op(")
        assert body.count(b";") == 2
        assert body.count(logical + b");") == 2


def test_second_application_and_missing_pinned_call_fail_closed(cube):
    with pytest.raises(ValueError, match="already present"):
        patch.patch_cube(patch.patch_cube(cube))
    with pytest.raises(ValueError, match="three pinned"):
        patch.patch_cube(cube.replace(b"cube1_matmul_op(src, src_len);", b"changed();", 1))
    with pytest.raises(ValueError):
        patch.patch_cube(cube.replace(b"cube2_matmul_op(src, src_len, 1, roundId - 2);", b"changed();", 1))


def test_no_vector_or_unrelated_source_changes():
    assert patch.apply("op_kernel/VectorForward.h", b"Vector stays unchanged") == b"Vector stays unchanged"
    assert patch.apply("op_kernel/AddressMappingForwardOnline.h", b"mapping stays unchanged") == b"mapping stays unchanged"


def test_header_contract_and_drain_only_after_issue_completion():
    header = patch.HEADER
    assert b"0x56434C4350524F42ULL" in header
    assert b"CORES = 4, STAGES = 2, POSITIONS = 3, FIELDS = 24" in header
    assert b"values_[1] = 1" in header
    assert b"mode_ < 1 || mode_ > 2" in header
    assert b"candidate = i == 0 ? 0 : (i == 1 ? total / 2 : total - 1)" in header
    assert b"if (candidate < total && candidate == round) { position = i; break; }" in header
    assert b"seen_ & (1U << slot)" in header
    assert header.count(b"AscendC::PipeBarrier<PIPE_ALL>();") == 1
    assert b"set_flag(" not in header and b"wait_flag(" not in header
    assert b"PIPE_S" not in header and b"HardEvent" not in header
    end = header[header.index(b"inline bool EndTiming()"):header.index(b"inline void Metadata(")]
    assert end.index(b"issue_end = Tick()") < end.index(b"Complete();") < end.index(b"            end = Tick()")
    assert b"values_[21] = mode_ == 2 ? 3 : 1" in end
    assert b"values_[17] = end - issue_end" in end
    for fn in (b"EndQK", b"EndPV"):
        body = header[header.index(b"inline void " + fn):]
        assert body.index(b"if (!EndTiming()) return;") < body.index(b"Metadata(src, count)") < body.index(b"Write();")


STUB = r"""
typedef unsigned char uint8_t; typedef unsigned int uint32_t;
typedef signed int int32_t; typedef signed long long int64_t;
typedef unsigned long long uint64_t;
#define __aicore__
#define __gm__
enum { PIPE_ALL };
namespace AscendC {
template<int Pipe> void PipeBarrier();
template<class T> class GlobalTensor {
public: void SetGlobalBuffer(T*,uint64_t); void SetValue(uint32_t,T);
GlobalTensor<T> operator[](uint32_t);
};
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
template<class T,CacheLine L,DcciDst D> void DataCacheCleanAndInvalid(const GlobalTensor<T>&);
uint64_t GetSystemCycle();
}
"""


@pytest.mark.parametrize("cube_build", [False, True])
def test_actual_header_instantiates_distinct_qk_and_pv_descriptors(cpp_compiler, tmp_path, cube_build):
    (tmp_path / "kernel_operator.h").write_text("// declaration-only host fixture\n", encoding="utf-8")
    source = STUB + ("#define __DAV_C220_CUBE__ 1\n" if cube_build else "") + patch.HEADER.decode() + r"""
struct QK { int32_t kx, ky; }; // deliberately no onStart/EndSection fields
struct PV { int32_t kx, ky; bool onStartSection, onEndSection; };
void invoke(uint8_t* output, int32_t mode, QK *qk, PV *pv) {
    vc_laser_cube_probe::CubeProbe probe;
    probe.Init(output, mode, 0);
    probe.Begin(0, 0, 3); probe.EndQK(qk, 1);
    probe.Begin(1, 0, 3); probe.EndPV(pv, 1);
}
"""
    code, diagnostic = compile_syntax(cpp_compiler, tmp_path, source)
    assert code == 0, diagnostic
