"""Pinned-source and host-semantic checks; never substitutes for CANN validation."""
from pathlib import Path

import pytest

from scripts import mindie_laser_quant_patch as patch
from scripts import prepare_mindie_laser_quant_source as prepare
from scripts.prepare_cann_fas_source import git
from tests.test_mindie_laser_expcast_cpp import cpp_compiler, compile_syntax


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def original():
    source = ROOT / "build/mindie-sd-reference"
    if not (source / ".git").exists():
        pytest.skip("Pinned source cache unavailable; no network in source tests")
    assert git(source, "rev-parse", "HEAD").decode().strip() == prepare.SOURCE_LOCK.commit
    return {name: git(source, "show", f"HEAD:{prepare.OP_ROOT}/op_kernel/{name}")
            for name in ("CubeForward.h", "VectorForward.h", "laser_attention.cpp")}


def test_qk_delta_preserves_pv_and_cross_core_schedule(original):
    old = original["CubeForward.h"]
    new = patch.patch_cube(old)
    # The complete PV, diagonal matmuls and history accumulation are unchanged.
    pv_start = b"__aicore__ __inline__ void  CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::cube2_matmul_op("
    qk_start = b"__aicore__ __inline__ void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::cube1_matmul_op("
    assert old[old.index(pv_start):old.index(qk_start)] == new[new.index(pv_start):new.index(qk_start)]
    run = b"__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::Run()"
    init = b"__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::Init("
    expected_run = old[old.index(run):old.index(init)].replace(
        b"PhyAddrForwardCube1Online<TYPE, TYPE, WORKSPACE_TYPE>",
        b"PhyAddrForwardCube1Online<int8_t, int8_t, WORKSPACE_TYPE>")
    assert expected_run == new[new.index(run):new.index(init)]
    assert new.count(b"copy_gm_to_cbuf_multi_nd2nz_b8(") == 2
    assert b"l1_a + m_offset * SIZE_32 + i * 512" in new
    assert b"k0_ * n0_ / 512" in new
    assert b"DEQF16" in new and b"0xffffe000u" in new
    with pytest.raises(ValueError):
        patch.patch_cube(new)


def test_vector_toggle_and_entry_do_not_pass_scales_as_mask(original):
    vector = patch.patch_vector(original["VectorForward.h"])
    assert patch.BULK_VEXP in vector
    assert patch.EXPERIMENTAL_PROBABILITY in vector
    assert b"if (use_expcast_)" in vector
    entry = patch.patch_entry(prepare.rename(original["laser_attention.cpp"]))
    assert b"op.SetQuantScales(q_scale_gm, k_scale_gm);" in entry
    assert b"op.Init(q_gm, k_gm, v_gm, nullptr, score_gm" in entry
    assert b"op.SetExpCast(tiling_data->useExpCast);" in entry


def test_all_pinned_patches_construct_and_abi_is_separate(original):
    files, records = prepare.expected_files(ROOT / "build/mindie-sd-reference")
    assert prepare.PREPARE_SCRIPT in files
    assert all("vc_laser_quant_attention" in name for name in records)
    definition = files[f"{prepare.CUSTOM_OP_ROOT}/op_host/vc_laser_quant_attention_def.cpp"]
    assert definition.count(b".DataType({ge::DT_INT8})") == 2
    assert definition.count(b".DataType({ge::DT_FLOAT16})") == 1
    assert b'Input("q_scale")\n                .ParamType(REQUIRED)' in definition
    assert b'Attr("use_expcast")' in definition
    assert b'AddConfig("ascend910")' not in definition


def test_qk_methods_host_cpp_types(cpp_compiler, original, tmp_path):
    """Instantiate the actual patched QK methods against typed declarations.

    No intrinsics execute. This catches C++ types/offset/signature mistakes,
    not support in bisheng, synchronization correctness or device performance.
    """
    data = patch.patch_cube(original["CubeForward.h"]).decode()
    start = data.index("template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>\ntemplate <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>\n__aicore__ __inline__ void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::cube1_matmul_op(")
    end = data.index("template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>\n__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::Run()", start)
    methods = data[start:end]
    preamble = r"""
typedef signed char int8_t;
typedef signed int int32_t;
typedef signed long long int64_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
struct half { unsigned short bits; };
#define __aicore__
#define __inline__ inline
#define __gm__
#define __cbuf__
#define __ca__
#define __cb__
#define __cc__
enum {SIZE_16=16, SIZE_32=32, SIZE_128=128, SIZE_256=256};
enum {PIPE_MTE1, PIPE_MTE2, PIPE_M, PIPE_FIX, inc, DEQF16};
void set_flag(int,int,int);
void wait_flag(int,int,int);
void set_quant_pre(uint64_t);
void pipe_barrier(int);
void copy_gm_to_cbuf_multi_nd2nz_b8(int8_t*, int8_t*, int,int,int,int,int,int,int,int,int);
void load_cbuf_to_ca(int8_t*,int8_t*,int,int,int,int,int,bool,int);
void load_cbuf_to_cb(int8_t*,int8_t*,int,int,int,int,int,bool,int);
void mad(int32_t*,int8_t*,int8_t*,int,int,int,int,int,int,int);
void copy_matrix_cc_to_gm(half*,int32_t*,int,int,int,int,int,int,int,int,bool,bool);
namespace Address {
template<class L, class R, class O> struct PhyAddrForwardCube1Online {
    int kx,ky,lineStride; bool upperRight,lowerLeft; L*left; R*right; O*out;
};
}
template<class TYPE, bool IF_BF16, class WORKSPACE_TYPE> class CubeForward {
public:
    template<class L,class R,class O> void cube1_matmul_op(const Address::PhyAddrForwardCube1Online<L,R,O>*,int64_t);
    void cube1_base_matmul(int8_t*,int8_t*,TYPE*,int32_t,int32_t,bool,int64_t,int64_t);
    float ReadQuantScale(float*);
    float *q_scales_,*k_scales_;
    int8_t *gm_Q,*gm_K;
    TYPE *l1_a_ping_,*l1_a_pong_,*l1_b_ping_,*l1_b_pong_;
    TYPE *l0_a_ping_,*l0_a_pong_,*l0_b_ping_,*l0_b_pong_;
    float *l0_c_ping_,*l0_c_pong_;
    int ping_pong_flag_l1_a_,ping_pong_flag_l1_b_,ping_pong_flag_l0_a_,ping_pong_flag_l0_b_,ping_pong_flag_l0_c_;
};
"""
    invoke = """
void instantiate(CubeForward<half,false,half>& op,
                 Address::PhyAddrForwardCube1Online<int8_t,int8_t,half>* src) {
    op.cube1_matmul_op(src,1);
}
"""
    code, log = compile_syntax(cpp_compiler, tmp_path, preamble + methods + invoke)
    assert code == 0, log
