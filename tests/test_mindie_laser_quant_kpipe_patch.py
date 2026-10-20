"""Execute the real transformed QK outer loop with a host ownership/event model."""
import ctypes
from pathlib import Path
import subprocess

import pytest

from scripts import mindie_laser_quant_kpipe_patch as patch
from scripts import prepare_mindie_laser_quant_cache_source as baseline
from scripts.prepare_cann_fas_source import git
from tests.test_mindie_laser_expcast_cpp import cpp_compiler


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def baseline_cube():
    reference = ROOT / "build/mindie-sd-reference"
    if not (reference / ".git").exists():
        pytest.skip("Pinned local MindIE source missing; tests never download it")
    assert git(reference, "rev-parse", "HEAD").decode().strip() == baseline.SOURCE_LOCK.commit
    files, _ = baseline.expected_files(reference)
    return files[f"{baseline.CUSTOM_OP_ROOT}/op_kernel/CubeForward.h"]


def test_strict_inverse_and_base_matmul_unchanged(baseline_cube):
    new = patch.patch_cube(baseline_cube)
    inverse = new.replace(patch.PROLOGUE + patch.N_LOOP, patch.N_LOOP)
    inverse = inverse.replace(patch.NEW_K_LOAD, patch.ORIGINAL_K_LOAD)
    assert inverse == baseline_cube
    start = b"__aicore__ __inline__ void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::cube1_base_matmul("
    assert new[new.index(start):] == baseline_cube[baseline_cube.index(start):]
    assert new[:new.index(patch.QK_BEGIN)] == baseline_cube[:baseline_cube.index(patch.QK_BEGIN)]
    assert patch.apply("op_kernel/VectorForward.h", baseline_cube) is baseline_cube
    assert b"n_index + 1 < n_loop" in new


@pytest.mark.parametrize("mutation", [
    lambda data: data.replace(patch.TILING, patch.TILING.replace(b"64, 2", b"64, 4")),
    lambda data: data.replace(b"q_scale_cache_line_ = ~uint64_t(0);", b"q_scale_cache_line_ = 0;"),
    lambda data: data.replace(patch.ORIGINAL_K_LOAD, patch.ORIGINAL_K_LOAD + patch.ORIGINAL_K_LOAD),
    lambda data: data.replace(b"gm_b + n_index * l1_n_size_ * k_", b"gm_b + changed_offset"),
    patch.patch_cube,
])
def test_drift_and_duplicate_application_fail_closed(baseline_cube, mutation):
    with pytest.raises(ValueError):
        patch.patch_cube(mutation(baseline_cube))


HOST_MODEL = r"""
typedef signed char int8_t; typedef unsigned char uint8_t;
typedef signed int int32_t; typedef unsigned int uint32_t;
typedef signed long long int64_t; typedef unsigned long long uint64_t;
struct half { unsigned short bits; };
#define __aicore__
#define __inline__ inline
#define __gm__
#define __cbuf__
#ifdef _MSC_VER
#define EXPORT extern "C" __declspec(dllexport)
extern "C" int _fltused = 0;
#else
#define EXPORT extern "C"
#endif
enum { PIPE_MTE1=1, PIPE_MTE2=2, SIZE_128=128 };
namespace Address {
template<class L,class R,class O> struct PhyAddrForwardCube1Online {
    int kx,ky,lineStride; bool upperRight,lowerLeft; L*left; R*right; O*out;
};
}
// Slots 0/1 are K/V L1, slots 2/3 are Q/P L1. The underlying bytes
// are never dereferenced: this model observes only ownership and addressing.
half l1_memory[4][32768];
int8_t q_memory[8*32768], k_memory[8*32*16384];
half score_memory[8*32*16384];
float q_scales[16], k_scales[256];
int free_event[4], ready_event[4], acquired[4], loaded[4], ready_waited[4], read_complete[4];
int tile_in_slot[4], expected_fault, active=-1, active_kx, active_ky, active_upper;
int loads, matmuls, overlap, start_k_slot, expected_next_descriptor;
int lengths[8], heights[8], upper_flags[8], actual_loads;

void fail(int code) { if(!expected_fault) expected_fault=code; }
int slot_of(half *ptr) {
    for(int i=0;i<4;++i) if(ptr==l1_memory[i]) return i;
    fail(1); return 0;
}
void wait_flag(int src,int dst,int id) {
    if(id<0 || id>=4) { fail(2); return; }
    if(src==PIPE_MTE1 && dst==PIPE_MTE2) {
        if(!free_event[id] || acquired[id] || loaded[id]) fail(3);
        free_event[id]=0; acquired[id]=1; read_complete[id]=0;
    } else if(src==PIPE_MTE2 && dst==PIPE_MTE1) {
        if(!ready_event[id] || !loaded[id] || ready_waited[id]) fail(4);
        ready_event[id]=0; ready_waited[id]=1;
    } else fail(5);
}
void set_flag(int src,int dst,int id) {
    if(id<0 || id>=4) { fail(6); return; }
    if(src==PIPE_MTE2 && dst==PIPE_MTE1) {
        if(!acquired[id] || !loaded[id] || ready_event[id]) fail(7);
        ready_event[id]=1;
    } else if(src==PIPE_MTE1 && dst==PIPE_MTE2) {
        if(!acquired[id] || !loaded[id] || !ready_waited[id] || !read_complete[id]) fail(8);
        if(free_event[id]) fail(9);
        acquired[id]=0; loaded[id]=0; ready_waited[id]=0; read_complete[id]=0; free_event[id]=1;
        if(id>=2 && active>=0) {
            if(loads!=active_kx || matmuls!=active_kx) fail(10);
            for(int k=0;k<2;++k)
                if(!free_event[k] || ready_event[k] || loaded[k] || acquired[k]) fail(11);
        }
    } else fail(12);
}
void copy_gm_to_cbuf_multi_nd2nz_b8(int8_t *dst,int8_t *src,int sid,int ndNum,
    int nValue,int dValue,int srcNdStride,int srcD,int dstC0,int dstN,int dstNd) {
    int slot=slot_of(reinterpret_cast<half *>(dst));
    if(!acquired[slot] || loaded[slot] || ready_waited[slot]) fail(13);
    if(sid!=0 || ndNum!=1 || srcNdStride!=0 || srcD!=128 || dstN!=1 || dstNd!=0) fail(14);
    if(slot>=2) {
        int which=int((src-q_memory)/32768);
        if(which<0 || which>=8 || src!=q_memory+which*32768) { fail(15); which=0; }
        if(which!=expected_next_descriptor) fail(16);
        ++expected_next_descriptor;
        active=which; active_kx=lengths[which]; active_ky=heights[which]; active_upper=upper_flags[which];
        loads=0; matmuls=0;
        if(nValue!=active_ky*128 || dValue!=128 || dstC0!=active_ky*128) fail(17);
        // This is the only Q copy; empty K descriptors still release Q later.
        read_complete[slot]=1;
    } else {
        if(active<0) { fail(18); return; }
        int tile=int((src-(k_memory+active*32*16384))/16384);
        if(src!=k_memory+(active*32+tile)*16384 || tile<0 || tile>=active_kx) fail(19);
        if(tile!=loads) fail(20);
        if(nValue!=128 || dValue!=128 || dstC0!=128) fail(21);
        if(loaded[1-slot]) ++overlap;
        tile_in_slot[slot]=tile; ++loads; ++actual_loads;
    }
    loaded[slot]=1;
}
template<class TYPE,bool IF_BF16,class WORKSPACE_TYPE> class CubeForward {
public:
    template<class L,class R,class O> void cube1_matmul_op(const Address::PhyAddrForwardCube1Online<L,R,O>*,int64_t);
    void cube1_base_matmul(int8_t *qa,int8_t *kb,TYPE *out,int ky,int stride,bool upper,float qs0,float qs1,float ks) {
        int a=slot_of(reinterpret_cast<half *>(qa)), b=slot_of(reinterpret_cast<half *>(kb));
        if(a<2 || b>=2 || !loaded[a] || !loaded[b] || !ready_waited[a] || !ready_waited[b]) fail(22);
        if(b!=int(ping_pong_flag_l1_b_) || a!=int(ping_pong_flag_l1_a_)+2) fail(23);
        if(tile_in_slot[b]!=matmuls || ky!=active_ky || stride!=32*16384) fail(24);
        if(upper!=(bool(active_upper) && matmuls==active_kx-1)) fail(25);
        if(out!=score_memory+(active*32+matmuls)*16384) fail(26);
        if(qs0!=q_scales[active*2] || qs1!=q_scales[active*2+(ky>1?1:0)] ||
           ks!=k_scales[active*32+matmuls]) fail(27);
        ++matmuls; read_complete[a]=1; read_complete[b]=1;
    }
    float ReadQuantScale(float *ptr,uint64_t &) { return *ptr; }
    float *q_scales_=q_scales,*k_scales_=k_scales;
    uint64_t q_scale_cache_line_=0,k_scale_cache_line_=0;
    int8_t *gm_Q=q_memory,*gm_K=k_memory;
    TYPE *l1_a_ping_=l1_memory[2],*l1_a_pong_=l1_memory[3];
    TYPE *l1_b_ping_=l1_memory[0],*l1_b_pong_=l1_memory[1];
    uint32_t ping_pong_flag_l1_a_=0,ping_pong_flag_l1_b_=0;
};
"""

SCENARIOS = r"""
void init_state() {
    expected_fault=0; active=-1; expected_next_descriptor=0; overlap=0; actual_loads=0;
    for(int i=0;i<4;++i) {
        free_event[i]=1; ready_event[i]=0; acquired[i]=0; loaded[i]=0; ready_waited[i]=0; read_complete[i]=0;
    }
    for(int i=0;i<16;++i) q_scales[i]=float(i+1)/32;
    for(int i=0;i<256;++i) k_scales[i]=float(i+1)/64;
}
void pv_turn(CubeForward<half,false,half>& op,int count) {
    // Ownership-only stand-in for unchanged PV, including the same free/ready
    // directions and slot toggle. No simulated PV numerical claim is made.
    active=-1;
    for(int i=0;i<count;++i) {
        int slot=int(op.ping_pong_flag_l1_b_);
        wait_flag(PIPE_MTE1,PIPE_MTE2,slot);
        if(loaded[slot]) fail(28);
        loaded[slot]=1;
        set_flag(PIPE_MTE2,PIPE_MTE1,slot);
        wait_flag(PIPE_MTE2,PIPE_MTE1,slot);
        read_complete[slot]=1;
        set_flag(PIPE_MTE1,PIPE_MTE2,slot);
        op.ping_pong_flag_l1_b_=1-op.ping_pong_flag_l1_b_;
    }
}
void run_descriptors(CubeForward<half,false,half>& op,int first,int count) {
    Address::PhyAddrForwardCube1Online<int8_t,int8_t,half> desc[8];
    int total=0, expected_overlap=0, before_slot=int(op.ping_pong_flag_l1_b_);
    for(int i=0;i<count;++i) {
        int x=first+i;
        desc[i].kx=lengths[x]; desc[i].ky=heights[x]; desc[i].lineStride=32*16384;
        desc[i].upperRight=upper_flags[x]!=0; desc[i].lowerLeft=false;
        desc[i].left=q_memory+x*32768; desc[i].right=k_memory+x*32*16384;
        desc[i].out=score_memory+x*32*16384;
        total+=lengths[x];
        expected_overlap+=lengths[x]>1?lengths[x]-1:0;
    }
    int before_loads=actual_loads, before_overlap=overlap;
    expected_next_descriptor=first;
    op.cube1_matmul_op(desc,count);
    if(actual_loads-before_loads!=total) fail(29);
    if(overlap-before_overlap!=expected_overlap) fail(30);
    if(op.ping_pong_flag_l1_b_!=uint32_t(before_slot^(total&1))) fail(31);
    for(int i=0;i<4;++i)
        if(!free_event[i] || ready_event[i] || acquired[i] || loaded[i] || ready_waited[i]) fail(32);
}
EXPORT int run_model() {
    // Cover all tail lengths, initial slot parities and both triangular skip
    // flags. Every K tile is still loaded when upperRight skips one Q sub-tile.
    for(int initial=0;initial<2;++initial)
    for(int n=0;n<=32;++n)
    for(int upper=0;upper<2;++upper)
    for(int ky=1;ky<=2;++ky) {
        init_state();
        CubeForward<half,false,half> op;
        op.ping_pong_flag_l1_b_=initial;
        op.ping_pong_flag_l1_a_=1-initial;
        lengths[0]=n; heights[0]=ky; upper_flags[0]=upper;
        run_descriptors(op,0,1);
        if(expected_fault) return expected_fault;
        pv_turn(op,3);
        lengths[1]=32-n; heights[1]=ky; upper_flags[1]=upper;
        lengths[2]=1; heights[2]=2; upper_flags[2]=0;
        lengths[3]=0; heights[3]=2; upper_flags[3]=0;
        run_descriptors(op,1,3);
        if(expected_fault) return expected_fault;
        pv_turn(op,2);
        run_descriptors(op,0,0);
        if(expected_fault) return expected_fault;
        // Drain each PresetFlag free token once: no outstanding ready token
        // or acquired slot may survive after QK/PV handoffs.
        for(int slot=0;slot<4;++slot) {
            wait_flag(PIPE_MTE1,PIPE_MTE2,slot);
            if(free_event[slot] || loaded[slot] || ready_event[slot]) return 33;
        }
    }
    return 0;
}
"""


def actual_outer_method(data):
    text = data.decode()
    template = "template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>\n"
    start = text.rfind(template, 0, text.index(patch.QK_BEGIN.decode()))
    end = text.index("::cube1_base_matmul(", start)
    end = text.rfind(template, start, end)
    return text[start:end]


def compile_and_run(compiler, directory, method):
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "kpipe_model.cpp"
    source.write_text(HOST_MODEL + method + SCENARIOS, encoding="utf-8")
    if compiler.name.lower() in ("cl", "cl.exe"):
        output = directory / "kpipe_model.dll"
        command = [str(compiler), "/nologo", "/std:c++17", "/utf-8", "/LD", "/Od", "/GS-", "/Zl", str(source),
                   "/link", "/NOENTRY", "/NODEFAULTLIB", f"/OUT:{output}"]
    else:
        output = directory / "kpipe_model.so"
        command = [str(compiler), "-std=c++17", "-shared", "-fPIC", "-O0", str(source), "-o", str(output)]
    result = subprocess.run(command, cwd=directory, capture_output=True, timeout=30)
    assert result.returncode == 0, (result.stdout + result.stderr).decode(errors="replace")
    library = ctypes.CDLL(str(output))
    library.run_model.argtypes = []
    library.run_model.restype = ctypes.c_int
    return library.run_model()


def test_real_transformed_loop_owns_slots_and_preserves_descriptor_and_pv_handoffs(
        cpp_compiler, baseline_cube, tmp_path):
    result = compile_and_run(cpp_compiler, tmp_path, actual_outer_method(patch.patch_cube(baseline_cube)))
    assert result == 0


@pytest.mark.parametrize("name,old,new", [
    ("empty_overread", "if (n_loop > 0)", "if (n_loop >= 0)"),
    ("overwrite", "const uint32_t next_k_slot = 1 - ping_pong_flag_l1_b_;",
     "const uint32_t next_k_slot = ping_pong_flag_l1_b_;"),
    ("missing_ready", "                wait_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_b_);", ""),
    ("missing_release", "                set_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_b_);", ""),
    ("tail_overread", "if (n_index + 1 < n_loop)", "if (n_index + 1 <= n_loop)"),
    ("wrong_parity", "ping_pong_flag_l1_b_ = 1 - ping_pong_flag_l1_b_;", "ping_pong_flag_l1_b_ = 0;"),
])
def test_model_detects_ownership_wait_drain_and_tail_regressions(
        cpp_compiler, baseline_cube, tmp_path, name, old, new):
    method = actual_outer_method(patch.patch_cube(baseline_cube))
    assert method.count(old) == 1
    assert compile_and_run(cpp_compiler, tmp_path / name, method.replace(old, new)) != 0
