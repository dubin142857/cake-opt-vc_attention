"""Pinned-source and host-C++ checks; these do not replace CCE/NPU gates."""
from pathlib import Path
import re

import pytest

from scripts import mindie_laser_quant_phase_patch as patch
from scripts import prepare_mindie_laser_quant_reuse_source as reuse
from tests.test_mindie_laser_expcast_cpp import cpp_compiler, compile_syntax

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def kernels():
    reference = ROOT / "build/mindie-sd-reference"
    if not (reference / ".git").is_dir():
        pytest.skip("Pinned source cache unavailable; tests never fetch")
    files, _ = reuse.expected_files(reference)
    return {name: files[f"{reuse.CUSTOM_OP_ROOT}/op_kernel/{name}"]
            for name in ("CubeForward.h", "VectorForward.h")}


def remove_probe_additions(data):
    data = data.replace(b'#include "vc_laser_phase_probe.h"\n', b"")
    data = data.replace(b"    vc_laser_probe::PhaseProbe phase_probe_;\n", b"")
    data = re.sub(rb"    __aicore__ inline void SetupProbe\(__gm__ uint8_t \*trace, int32_t mode\) \{.*?\n    \}\n",
                  b"", data, flags=re.S)
    data = data.replace(b"                                section_block_nums, rowPerTime, lines);\n", b"")
    return b"".join(line for line in data.splitlines(keepends=True) if b"phase_probe_." not in line)


@pytest.mark.parametrize("name", ["CubeForward.h", "VectorForward.h"])
def test_only_diagnostic_additions_preserve_every_original_byte(kernels, name):
    original = kernels[name]
    changed = patch.apply("op_kernel/" + name, original)
    assert changed != original
    assert remove_probe_additions(changed) == original
    assert changed.count(b"wait_flag_dev(") == original.count(b"wait_flag_dev(")
    assert changed.count(b"ffts_cross_core_sync(") == original.count(b"ffts_cross_core_sync(")
    # Raw same-core event operations remain byte-for-byte in the original code;
    # only the companion header adds the explicitly audited adjacent V_S0 pair.
    raw = rb"(?:set_flag|wait_flag|pipe_barrier)\([^;]*\);"
    assert re.findall(raw, changed) == re.findall(raw, original)


@pytest.mark.parametrize("name", ["CubeForward.h", "VectorForward.h"])
def test_second_application_fails_closed(kernels, name):
    changed = patch.apply("op_kernel/" + name, kernels[name])
    with pytest.raises(ValueError):
        patch.apply("op_kernel/" + name, changed)


def test_cube_waits_use_probability_round_not_lookahead_qk(kernels):
    changed = patch.patch_cube(kernels["CubeForward.h"])
    rounds = re.findall(rb"phase_probe_\.WaitBegin\(([^,]+), Z, address\.is_running\(([^)]+)\)\);", changed)
    assert rounds == [(b"roundId", b"roundId"), (b"roundId - 2", b"roundId - 2"),
                      (b"roundId + Z - 2", b"roundId + Z - 2")]
    assert changed.count(b"phase_probe_.WaitEnd();") == 3


def test_phase_boundaries_follow_actual_data_dependencies(kernels):
    changed = patch.patch_vector(kernels["VectorForward.h"])
    max_start = changed.index(b"    IF_BF16, WORKSPACE_T>::process_line_phase_one_for_short_seq_max(")
    exp_start = changed.index(b"    IF_BF16, WORKSPACE_T>::process_line_phase_one_for_short_seq_exp_and_rowsum(")
    maximum = changed[max_start:exp_start]
    assert maximum.index(b"wait_flag(PIPE_MTE2, PIPE_V, event_id)") < maximum.index(b"Mark(16, 2)")
    assert maximum.index(b"Mark(16, 2)") < maximum.index(b"vmuls(")
    assert maximum.index(b"vexp(") < maximum.index(b"Mark(17, 3)")
    exp_end = changed.index(b" >::add_max_wrap(", exp_start)
    probability = changed[exp_start:exp_end]
    assert probability.index(b"vsub(") < probability.index(b"Mark(18, 4)") < probability.index(b"if (use_expcast_)")
    assert probability.index(b"vexp(") < probability.index(b"Mark(19, 5)") < probability.index(b"copy_ubuf_to_gm(")
    assert probability.index(b"copy_ubuf_to_gm(") < probability.index(b"StartSum()") < probability.index(b"process_calc_sum(")
    assert probability.index(b"process_calc_sum(") < probability.index(b"Mark(20, 6)") < probability.index(b"PhaseFinish()")
    # Every actual batch is considered: first observed class wins in the header,
    # so later tail sections are not silently omitted by an outer first-only guard.
    assert b"PhaseBegin(sectionLoop, is_head_section, is_tail_section," in changed


def test_unrelated_sources_untouched_and_header_contract():
    assert patch.apply("op_kernel/unrelated.h", b"input") == b"input"
    header = patch.HEADER
    assert b"0x56434C50524F4245ULL" in header
    assert b"CORES = 4, ROLES = 3, SLOTS = 8, FIELDS = 24" in header
    assert b"values_[1] = 1" in header
    assert header.count(b"set_flag(") == header.count(b"wait_flag(") == 1
    assert b"set_flag(PIPE_V, PIPE_S, EVENT_ID0)" in header
    assert b"PIPE_ALL" not in header
    assert b"PIPE_FIX" not in header
    assert b"rows != 4" in header and b"seen_classes_ & (1U << kind)" in header
    assert b"candidate < total && candidate == round" in header


STUB = r"""
typedef unsigned char uint8_t; typedef unsigned int uint32_t;
typedef signed int int32_t; typedef signed long long int64_t;
typedef unsigned long long uint64_t;
#define __aicore__
#define __gm__
enum { PIPE_V, PIPE_S, EVENT_ID0 };
void set_flag(int,int,int); void wait_flag(int,int,int);
namespace AscendC {
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


@pytest.mark.parametrize("vector", [False, True])
def test_real_header_has_valid_host_cpp_both_core_roles(cpp_compiler, tmp_path, vector):
    (tmp_path / "kernel_operator.h").write_text("// declaration-only host fixture\n", encoding="utf-8")
    source = STUB + ("#define __DAV_C220_VEC__ 1\n" if vector else "") + patch.HEADER.decode() + r"""
void invoke(uint8_t* output, int32_t mode) {
    vc_laser_probe::PhaseProbe probe;
    probe.Init(output,mode,0,1);
    probe.SetRound(0,3); probe.WaitBegin(0,3,true); probe.WaitEnd();
    probe.PhaseBegin(0,true,false,32,4,0);
    probe.Mark(16,2); probe.Mark(17,3); probe.Mark(18,4); probe.Mark(19,5);
    probe.StartSum(); probe.Mark(20,6); probe.PhaseFinish();
}
"""
    code, diagnostic = compile_syntax(cpp_compiler, tmp_path, source)
    assert code == 0, diagnostic


def test_missing_pinned_wait_or_phase_anchor_fails_closed(kernels):
    with pytest.raises(ValueError, match="three"):
        patch.patch_cube(kernels["CubeForward.h"].replace(b"wait_flag_dev(AIV2AICFLAGID);", b"changed_wait();", 1))
    with pytest.raises(ValueError):
        patch.patch_vector(kernels["VectorForward.h"].replace(b"    //  128/64=2", b"    // changed pinned code"))
