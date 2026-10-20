"""Execute the real QK outer/base methods with L0A residency/event checks."""
import ctypes
from pathlib import Path
import re
import subprocess

import pytest

from scripts import mindie_laser_quant_qresident_patch as patch
from tests import test_mindie_laser_quant_kpipe_patch as prior
from tests.test_mindie_laser_expcast_cpp import cpp_compiler
from tests.test_mindie_laser_quant_kpipe_patch import baseline_cube


def restore(data):
    for old, new in ((patch.OLD_DECL, patch.NEW_DECL), (patch.OLD_CALL, patch.NEW_CALL),
                     (patch.OLD_DEF, patch.NEW_DEF), (patch.OLD_LOAD, patch.NEW_LOAD)):
        data = data.replace(new, old)
    return data


def test_exact_inverse_and_all_event_mad_fix_sites_unchanged(baseline_cube):
    new = patch.patch_cube(baseline_cube)
    assert restore(new) == baseline_cube
    raw = rb"(?:set_flag|wait_flag|wait_flag_dev|ffts_cross_core_sync|pipe_barrier|mad|copy_matrix_cc_to_gm|set_quant_pre)\([^;]*\);"
    assert re.findall(raw, new) == re.findall(raw, baseline_cube)
    assert new[new.index(patch.RUN_BEGIN):] == baseline_cube[baseline_cube.index(patch.RUN_BEGIN):]
    assert patch.apply("op_kernel/VectorForward.h", baseline_cube) is baseline_cube
    assert b"next_k_slot" not in new


@pytest.mark.parametrize("mutation", [
    lambda d: d.replace(patch.TILING, patch.TILING.replace(b"64, 2", b"64, 4")),
    lambda d: d.replace(patch.L0A_LAYOUT[1], patch.L0A_LAYOUT[1].replace(b"32 * 1024", b"16 * 1024")),
    lambda d: d.replace(patch.OLD_CALL, b"stale_call();"),
    lambda d: d.replace(patch.OLD_LOAD, patch.OLD_LOAD + patch.OLD_LOAD),
    lambda d: d.replace(b"q_scale_cache_line_ = ~uint64_t(0);", b"q_scale_cache_line_ = 0;"),
    lambda d: prior.patch.patch_cube(d),
    patch.patch_cube,
])
def test_drift_and_stacking_fail_closed(baseline_cube, mutation):
    with pytest.raises(ValueError):
        patch.patch_cube(mutation(baseline_cube))


def actual_methods(data):
    text = data.decode()
    template = "template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>\n"
    start = text.rfind(template, 0, text.index(patch.QK_BEGIN.decode()))
    end = text.rfind(template, start, text.index(patch.RUN_BEGIN.decode()))
    return text[start:end]


L0_MODEL = r"""
int8_t l0a_memory[65536], l0b_memory[65536];
int32_t l0c_memory[32768];
int l0_free[4], l0_ready[4], l0_acquired[4], l0_consumable[4];
int q_identity[2][8], k_identity[2], k_descriptor[2], c_identity[2];
int q_loads, last_loaded_q_row, last_mad_row, current_output_row;
int total_mads, total_fix, total_q_loads;
uint64_t operand_hash, event_hash, quant_value;

int l0a_slot(int8_t *ptr) { return int((ptr-l0a_memory)/32768); }
int l0b_slot(int8_t *ptr) { return int((ptr-l0b_memory)/32768); }
int l0c_slot(int32_t *ptr) { return int((ptr-l0c_memory)/16384); }
void poison_q() { for(int s=0;s<2;++s) for(int i=0;i<8;++i) q_identity[s][i]=-999; }
void mix(uint64_t &hash,uint64_t value) { hash=hash*1099511628211ull+value+1469598103934665603ull; }
void wait_flag(int src,int dst,int id) {
    mix(event_hash, src*100+dst*10+id);
    if(src==PIPE_M && dst==PIPE_MTE1) {
        if(id<0 || id>=4) { fail(41); return; }
        if(!l0_free[id] || l0_acquired[id] || l0_ready[id]) fail(42);
        l0_free[id]=0; l0_acquired[id]=1; l0_consumable[id]=0;
    } else if(src==PIPE_MTE1 && dst==PIPE_M) {
        if(id<0 || id>=4) { fail(43); return; }
        if(!l0_ready[id] || !l0_acquired[id]) fail(44);
        l0_ready[id]=0; l0_consumable[id]=1;
    } else l1_wait_flag(src,dst,id);
}
void set_flag(int src,int dst,int id) {
    mix(event_hash, 1000+src*100+dst*10+id);
    if(src==PIPE_MTE1 && dst==PIPE_M) {
        if(id<0 || id>=4) { fail(45); return; }
        if(!l0_acquired[id] || l0_ready[id]) fail(46);
        l0_ready[id]=1;
    } else if(src==PIPE_M && dst==PIPE_MTE1) {
        if(id<0 || id>=4) { fail(47); return; }
        if(!l0_acquired[id] || !l0_consumable[id] || l0_free[id]) fail(48);
        l0_acquired[id]=0; l0_consumable[id]=0; l0_free[id]=1;
    } else l1_set_flag(src,dst,id);
}
void load_cbuf_to_ca(int8_t *dst,int8_t *src,int base,int repeat,int stride,
                    int dstStride,int sid,bool transpose,int mode) {
    int slot=l0a_slot(dst);
    int offset=int(dst-l0a_memory-slot*32768), frag=offset/2048;
    if(slot<0 || slot>1 || offset!=frag*2048 || frag<0 || frag>=8) { fail(49); return; }
    if(!l0_acquired[slot] || l0_consumable[slot]) fail(50);
    int l1slot=-1, row=-1;
    for(int a=2;a<4;++a) for(int r=0;r<active_ky;++r)
        if(src==reinterpret_cast<int8_t *>(l1_memory[a])+r*4096+frag*512) { l1slot=a; row=r; }
    if(l1slot<2 || !loaded[l1slot] || !ready_waited[l1slot]) { fail(51); return; }
    if(base!=0 || repeat!=4 || stride!=active_ky*8 || dstStride!=0 ||
       sid!=0 || transpose || mode!=inc) fail(52);
    q_identity[slot][frag]=(active+1)*100+row;
    read_complete[l1slot]=1; ++q_loads; ++total_q_loads;
}
void load_cbuf_to_cb(int8_t *dst,int8_t *src,int base,int repeat,int stride,
                    int dstStride,int sid,bool transpose,int mode) {
    int slot=l0b_slot(dst), l1slot=slot_of(reinterpret_cast<half *>(src));
    if(slot<0 || slot>1 || dst!=l0b_memory+slot*32768) { fail(53); return; }
    if(!l0_acquired[slot+2] || l0_consumable[slot+2]) fail(54);
    if(l1slot>=2 || !loaded[l1slot] || !ready_waited[l1slot]) fail(55);
    if(tile_in_slot[l1slot]!=matmuls) fail(56);
    if(base!=0 || repeat!=32 || stride!=1 || dstStride!=0 || sid!=0 || transpose || mode!=inc) fail(57);
    k_identity[slot]=matmuls++; k_descriptor[slot]=active;
    read_complete[l1slot]=1; last_mad_row=-1;
}
void mad(int32_t *dst,int8_t *qa,int8_t *kb,int m,int k,int n,int unit,int align,int source,int init) {
    int a=l0a_slot(qa), b=l0b_slot(kb), c=l0c_slot(dst);
    if(a<0 || a>1 || b<0 || b>1 || c<0 || c>1) { fail(58); return; }
    if(qa!=l0a_memory+a*32768 || kb!=l0b_memory+b*32768 || dst!=l0c_memory+c*16384) fail(59);
    if(!l0_consumable[a] || !l0_consumable[b+2]) fail(60);
    int row=last_mad_row+1;
    if(active_upper && matmuls==active_kx && row==0) row=1;
    if(row>=active_ky) fail(61);
    for(int i=0;i<8;++i) if(q_identity[a][i]!=(active+1)*100+row) fail(62);
    if(k_identity[b]!=matmuls-1 || k_descriptor[b]!=active) fail(63);
    if(m!=128 || k!=128 || n!=128 || unit!=3 || align!=0 || source!=0 || init!=1) fail(64);
    c_identity[c]=(active+1)*10000+(matmuls-1)*10+row;
    current_output_row=row; last_mad_row=row;
    mix(operand_hash, c_identity[c]);
    mix(operand_hash, a*100+b*10+c);
    ++total_mads;
}
void set_quant_pre(uint64_t value) { quant_value=value; }
void pipe_barrier(int pipe) { if(pipe!=PIPE_FIX) fail(65); mix(event_hash,5000+pipe); }
void copy_matrix_cc_to_gm(half *out,int32_t *src,int sid,int n,int m,int dstStride,
                         int srcStride,int unit,int quant,int relu,bool channel,bool nz) {
    int c=l0c_slot(src), row=current_output_row;
    if(c_identity[c]!=(active+1)*10000+(matmuls-1)*10+row) fail(66);
    if(out!=score_memory+(active*32+matmuls-1)*16384+row*32*16384) fail(67);
    if(sid!=0 || n!=128 || m!=128 || dstStride!=128 || srcStride!=128 ||
       unit!=3 || quant!=DEQF16 || relu!=0 || channel || !nz) fail(68);
    float gain=q_scales[active*2+row]*k_scales[active*32+matmuls-1];
    uint32_t bits=*reinterpret_cast<uint32_t *>(&gain)&0xffffe000u;
    if(quant_value!=bits) fail(69);
    mix(operand_hash, quant_value); ++total_fix;
}
"""


CLASS = r"""
template<class TYPE,bool IF_BF16,class WORKSPACE_TYPE> class CubeForward {
public:
    template<class L,class R,class O> void cube1_matmul_op(const Address::PhyAddrForwardCube1Online<L,R,O>*,int64_t);
    void cube1_base_matmul(int8_t*,int8_t*,TYPE*,int32_t,int32_t,bool,float,float,float LOAD_ARG);
    float ReadQuantScale(float *ptr,uint64_t &) { return *ptr; }
    float *q_scales_=q_scales,*k_scales_=k_scales;
    uint64_t q_scale_cache_line_=0,k_scale_cache_line_=0;
    int8_t *gm_Q=q_memory,*gm_K=k_memory;
    TYPE *l1_a_ping_=l1_memory[2],*l1_a_pong_=l1_memory[3];
    TYPE *l1_b_ping_=l1_memory[0],*l1_b_pong_=l1_memory[1];
    TYPE *l0_a_ping_=reinterpret_cast<TYPE *>(l0a_memory);
    TYPE *l0_a_pong_=reinterpret_cast<TYPE *>(l0a_memory+32768);
    TYPE *l0_b_ping_=reinterpret_cast<TYPE *>(l0b_memory);
    TYPE *l0_b_pong_=reinterpret_cast<TYPE *>(l0b_memory+32768);
    TYPE *l0_c_ping_=reinterpret_cast<TYPE *>(l0c_memory);
    TYPE *l0_c_pong_=reinterpret_cast<TYPE *>(l0c_memory+16384);
    uint32_t ping_pong_flag_l1_a_=0,ping_pong_flag_l1_b_=0;
    uint32_t ping_pong_flag_l0_a_=0,ping_pong_flag_l0_b_=0,ping_pong_flag_l0_c_=0;
};
"""


SCENARIOS = r"""
void init_state() {
    expected_fault=0; active=-1; expected_next_descriptor=0; actual_loads=0;
    q_loads=0; total_q_loads=0; total_mads=0; total_fix=0;
    poison_q();
    for(int i=0;i<4;++i) {
        free_event[i]=1; ready_event[i]=0; acquired[i]=0; loaded[i]=0; ready_waited[i]=0; read_complete[i]=0;
        l0_free[i]=1; l0_ready[i]=0; l0_acquired[i]=0; l0_consumable[i]=0;
    }
    for(int i=0;i<16;++i) q_scales[i]=float(i+1)/32;
    for(int i=0;i<256;++i) k_scales[i]=float(i+1)/64;
}
void drained() {
    for(int i=0;i<4;++i) {
        if(!l0_free[i] || l0_ready[i] || l0_acquired[i] || l0_consumable[i]) fail(70);
        if(!free_event[i] || ready_event[i] || acquired[i] || loaded[i] || ready_waited[i]) fail(71);
    }
}
void pv_turn(CubeForward<half,false,half>& op,int count) {
    // Unchanged PV is modeled only as an intervening L0 writer. Its actual
    // math is not executed; all release tokens must exist before this handoff.
    drained(); poison_q(); active=-1;
    op.ping_pong_flag_l0_a_ ^= count&1;
    op.ping_pong_flag_l0_b_ ^= count&1;
    op.ping_pong_flag_l0_c_ ^= count&1;
    op.ping_pong_flag_l1_b_ ^= count&1;
}
void run_descriptors(CubeForward<half,false,half>& op,int first,int count) {
    Address::PhyAddrForwardCube1Online<int8_t,int8_t,half> desc[8];
    int expected_loads=0, expected_mads=0, total_k=0, total_q_toggles=0;
    for(int i=0;i<count;++i) {
        int x=first+i,n=lengths[x],ky=heights[x],u=upper_flags[x]&&n>0;
        desc[i].kx=n; desc[i].ky=ky; desc[i].lineStride=32*16384;
        desc[i].upperRight=upper_flags[x]!=0; desc[i].lowerLeft=false;
        desc[i].left=q_memory+x*32768; desc[i].right=k_memory+x*32*16384;
        desc[i].out=score_memory+x*32*16384;
        expected_mads+=n*ky-u;
        expected_loads+=8*((RESIDENT && ky==2)?(n>0?2-(u&&n==1):0):(n*ky-u));
        total_k+=n; total_q_toggles+=n*ky;
    }
    int before_q=total_q_loads,before_m=total_mads,before_f=total_fix;
    int before_a=op.ping_pong_flag_l0_a_,before_b=op.ping_pong_flag_l0_b_,before_c=op.ping_pong_flag_l0_c_;
    expected_next_descriptor=first;
    op.cube1_matmul_op(desc,count);
    if(total_q_loads-before_q!=expected_loads) fail(72);
    if(total_mads-before_m!=expected_mads || total_fix-before_f!=expected_mads) fail(73);
    if(op.ping_pong_flag_l0_a_!=(before_a^(total_q_toggles&1)) ||
       op.ping_pong_flag_l0_b_!=(before_b^(total_k&1)) ||
       op.ping_pong_flag_l0_c_!=(before_c^(total_q_toggles&1))) fail(74);
    drained();
}
EXPORT int run_model() {
    operand_hash=0; event_hash=0;
    for(int initial=0;initial<8;++initial)
    for(int n=0;n<=32;++n)
    for(int upper=0;upper<2;++upper)
    for(int ky=1;ky<=2;++ky) {
        init_state();
        CubeForward<half,false,half> op;
        op.ping_pong_flag_l0_a_=initial&1;
        op.ping_pong_flag_l0_b_=(initial>>1)&1;
        op.ping_pong_flag_l0_c_=(initial>>2)&1;
        op.ping_pong_flag_l1_a_=1-(initial&1);
        op.ping_pong_flag_l1_b_=(initial>>1)&1;
        lengths[0]=n; heights[0]=ky; upper_flags[0]=upper;
        run_descriptors(op,0,1);
        if(expected_fault) return expected_fault;
        pv_turn(op,3);
        lengths[1]=32-n; heights[1]=ky; upper_flags[1]=upper;
        lengths[2]=1; heights[2]=2; upper_flags[2]=1;
        lengths[3]=0; heights[3]=1; upper_flags[3]=0;
        lengths[4]=2; heights[4]=2; upper_flags[4]=0;
        run_descriptors(op,1,4);
        if(expected_fault) return expected_fault;
        pv_turn(op,2);
        run_descriptors(op,0,0);
        if(expected_fault) return expected_fault;
    }
    return 0;
}
EXPORT uint64_t get_operands() { return operand_hash; }
EXPORT uint64_t get_events() { return event_hash; }
"""


def host_source(methods, resident):
    # Reuse the already tested L1 ownership model, then execute both actual
    # source methods rather than replacing the base matmul with a stub.
    common = prior.HOST_MODEL[:prior.HOST_MODEL.index(
        "template<class TYPE,bool IF_BF16,class WORKSPACE_TYPE> class CubeForward")]
    common = common.replace("#define __cbuf__", "#define __ca__\n#define __cb__\n#define __cc__\n#define __cbuf__")
    common = common.replace("SIZE_128=128", "SIZE_128=128, SIZE_16=16, SIZE_32=32, PIPE_M=3, PIPE_FIX=4, inc=0, DEQF16=1")
    common = common.replace("half score_memory[8*32*16384];", "half score_memory[8*64*16384];")
    common = common.replace("void wait_flag(", "void l1_wait_flag(").replace("void set_flag(", "void l1_set_flag(")
    # Invalidate our logical residency tags at every Q descriptor load, even
    # when actual SRAM would happen to retain the previous descriptor's bytes.
    common = common.replace("void copy_gm_to_cbuf_multi_nd2nz_b8(",
                            "void poison_q();\nvoid copy_gm_to_cbuf_multi_nd2nz_b8(")
    common = common.replace("loads=0; matmuls=0;", "loads=0; matmuls=0; poison_q();")
    return (f"#define RESIDENT {int(resident)}\n" + common + L0_MODEL +
            CLASS.replace(" LOAD_ARG", ",bool" if resident else "") + methods + SCENARIOS)


def compile_model(compiler, directory, methods, resident):
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "qresident.cpp"
    source.write_text(host_source(methods, resident), encoding="utf-8")
    if compiler.name.lower() in ("cl", "cl.exe"):
        output = directory / "qresident.dll"
        command = [str(compiler), "/nologo", "/std:c++17", "/utf-8", "/LD", "/Od", "/GS-", "/Zl", str(source),
                   "/link", "/NOENTRY", "/NODEFAULTLIB", f"/OUT:{output}"]
    else:
        output = directory / "qresident.so"
        command = [str(compiler), "-std=c++17", "-shared", "-fPIC", "-O0", str(source), "-o", str(output)]
    result = subprocess.run(command, cwd=directory, capture_output=True, timeout=30)
    assert result.returncode == 0, (result.stdout + result.stderr).decode(errors="replace")
    library = ctypes.CDLL(str(output))
    library.run_model.restype = ctypes.c_int
    library.get_operands.restype = ctypes.c_uint64
    library.get_events.restype = ctypes.c_uint64
    status = library.run_model()
    return status, library.get_operands(), library.get_events()


def test_actual_outer_and_base_match_operands_events_and_reduce_q_loads(
        cpp_compiler, baseline_cube, tmp_path):
    old = compile_model(cpp_compiler, tmp_path / "old", actual_methods(baseline_cube), False)
    new = compile_model(cpp_compiler, tmp_path / "new", actual_methods(patch.patch_cube(baseline_cube)), True)
    assert old[0] == 0
    assert new[0] == 0
    assert old[1:] == new[1:]  # every MAD operand identity and every event/FIX order


@pytest.mark.parametrize("name,old,new", [
    ("missing_first_load", "n_index == 0 || ky != 2", "n_index != 0 || ky != 2"),
    ("ky1_wrong_reuse", "n_index == 0 || ky != 2", "n_index == 0"),
    ("wrong_slot", "ping_pong_flag_l0_a_ ? l0_a_pong_ : l0_a_ping_", "false ? l0_a_pong_ : l0_a_ping_"),
    ("wrong_q_fragment", "m_offset * SIZE_32 + i * 512", "0 * SIZE_32 + i * 512"),
    ("missing_ready", "            wait_flag(PIPE_MTE1, PIPE_M, ping_pong_flag_l0_a_);", ""),
    ("missing_release", "            set_flag(PIPE_M, PIPE_MTE1, ping_pong_flag_l0_a_);", ""),
    ("wrong_parity", "ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;", "ping_pong_flag_l0_a_ = 0;"),
])
def test_model_rejects_residency_slot_wait_release_and_mapping_regressions(
        cpp_compiler, baseline_cube, tmp_path, name, old, new):
    methods = actual_methods(patch.patch_cube(baseline_cube))
    assert methods.count(old) == 1
    assert compile_model(cpp_compiler, tmp_path / name, methods.replace(old, new), True)[0] != 0
