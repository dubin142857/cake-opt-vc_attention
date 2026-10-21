"""Execute the actual PV method against asynchronous ownership/operand models.

Host checks are not a CCE compiler, NPU timing, or hardware-memory-model proof.
The model makes queued transfers unreadable until their ready event is consumed,
and makes a slot unwritable until its previous L1 reader releases it.
"""
import ctypes
from pathlib import Path
import re

import pytest

from scripts import mindie_laser_quant_vscale_pv_prefetch_patch as patch
from scripts import prepare_mindie_laser_quant_vscale_source as baseline
from scripts.prepare_cann_fas_source import git
from tests.test_mindie_laser_expcast_cpp import cpp_compiler
from tests.test_mindie_laser_quant_vscale_patch import compile_library, function

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cube():
    reference = ROOT / "build/mindie-sd-reference"
    if not (reference / ".git").exists():
        pytest.skip("Pinned cached source unavailable; no network fetch")
    assert git(reference, "rev-parse", "HEAD").decode().strip() == baseline.SOURCE_LOCK.commit
    files, _ = baseline.expected_files(reference)
    return next(data for name, data in files.items() if name.endswith("/CubeForward.h"))


def restore(data):
    data = data.replace(patch.PRIME, b"").replace(b"\n" + patch.LOOKAHEAD, b"")
    for old in (patch.FREE_WAIT, patch.COPY + b"\n" + patch.READY_SET, patch.FREE_SET):
        data = data.replace(patch.guarded(old), old)
    return data


def test_only_v_transfer_schedule_changes(cube):
    new = patch.patch_cube(cube)
    assert restore(new) == cube
    assert patch.apply("op_kernel/VectorForward.h", cube) is cube
    before, after = cube.index(patch.PV_BEGIN), cube.index(patch.PV_END)
    assert new[:before] == cube[:before]
    assert new[new.index(patch.PV_END):] == cube[after:]
    math = rb"(?:mad|copy_matrix_cc_to_gm|diag_matmul|set_quant_pre)\([^;]*\);"
    assert re.findall(math, cube) == re.findall(math, new)
    assert b"l1_m == 128 && l1_k == 128 && l1_n == 128 && qk_triangle == 0" in new


@pytest.mark.parametrize("mutation", [
    lambda d: d.replace(patch.LAYOUT[3], patch.LAYOUT[3].replace(b"192", b"160")),
    lambda d: d.replace(patch.LAYOUT[4], patch.LAYOUT[4].replace(b"256", b"128")),
    lambda d: d.replace(patch.FIXED_QUANT_INIT, b"old_scale_owner"),
    lambda d: d.replace(patch.COPY, b"unknown_v_transfer();"),
    lambda d: d.replace(patch.READY_WAIT, patch.READY_WAIT + patch.READY_WAIT),
    patch.patch_cube,
])
def test_source_drift_and_duplicate_patch_fail_closed(cube, mutation):
    with pytest.raises(ValueError):
        patch.patch_cube(mutation(cube))


MODEL = r"""
typedef signed int int32_t;
typedef signed long long int64_t;
typedef unsigned short half;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#define __aicore__
#define __inline__ inline
#define __gm__
#ifdef _MSC_VER
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif
enum {PIPE_MTE1,PIPE_MTE2,PIPE_M,EVENT_ID7=7,NoQuant=0,inc=0};
enum {CUBE2_LENGTH_M=128,CUBE2_LENGTH_K=128,CUBE2_LENGTH_N=128,
      BASE_BLOCK_LENGTH=128,BLOCK_SIZE=16,BASE_BLOCK_SIZE=16384,MAX_SWITCH_TIME=4};
namespace Address {
template<class L,class R,class O> struct PhyAddrForwardCube2Online {
 L *left; R *right; O *out; int ky,kx,lineStride;
 bool upperRight,onStartSection,onEndSection;
};
}
half p_gm[4][64*16384],v_gm[4][32*16384],l1[4][32768];
half l0a[2][16384],l0b[2][16384];
float output_gm[4][32768],diag_gm[8*32768],l0c[2][16384];
int fault,descriptors,kx[4],ky[4],upper[4],history[4],tail[4],history_done[4],tail_done[4];
int active_desc,active_tile,active_row,total_mad,total_fix,total_v,total_p,prefetch_hits;
int v_loaded[4],v_consumed[4],last_v_tile[4];
uint64_t sequence_hash;
struct L1State { int free,owned,ready,readable,issued,read_count,desc,tile,rows; } s1[4];
struct L0State { int free,owned,ready,readable,frags,desc,tile,row; } s0[4];
void fail(int code) { if(!fault) fault=code; }
void mix(uint64_t x) { sequence_hash=sequence_hash*1099511628211ull+x+1469598103934665603ull; }
int slot1(half *p) {
 for(int i=0;i<4;i++) if(p>=l1[i] && p<l1[i]+32768) return i;
 fail(10); return 0;
}
int slot0(half *p,bool b) {
 for(int i=0;i<2;i++) if(b ? (p>=l0b[i]&&p<l0b[i]+16384) : (p>=l0a[i]&&p<l0a[i]+16384)) return i;
 fail(11); return 0;
}
void wait_flag(int from,int to,int id) {
 if(from==PIPE_MTE1 && to==PIPE_MTE2) {
   if(id<0||id>=4) {fail(12);return;}
   L1State &s=s1[id];
   if(!s.free||s.owned||s.ready) fail(13);
   s.free=0;s.owned=1;s.issued=0;s.readable=0;s.read_count=0;s.rows=0;
 } else if(from==PIPE_MTE2 && to==PIPE_MTE1) {
   if(id<0||id>=4) {fail(14);return;}
   L1State &s=s1[id];
   if(!s.owned||!s.ready||!s.issued||s.readable) fail(15);
   s.ready=0;s.readable=1;
 } else if(from==PIPE_M && to==PIPE_MTE1) {
   if(id<0||id>=4) {fail(16);return;}
   L0State &s=s0[id];
   if(!s.free||s.owned||s.ready) fail(17);
   s.free=0;s.owned=1;s.readable=0;s.frags=0;
 } else if(from==PIPE_MTE1 && to==PIPE_M) {
   if(id<0||id>=4) {fail(18);return;}
   L0State &s=s0[id];
   if(!s.owned||!s.ready||s.readable) fail(19);
   s.ready=0;s.readable=1;
 } else if(!(from==PIPE_M&&to==PIPE_MTE2&&id==EVENT_ID7)) fail(20);
}
void set_flag(int from,int to,int id) {
 if(from==PIPE_MTE2 && to==PIPE_MTE1) {
   if(id<0||id>=4) {fail(21);return;}
   L1State &s=s1[id];
   if(!s.owned||!s.issued||s.ready||s.readable) fail(22);
   s.ready=1;
 } else if(from==PIPE_MTE1 && to==PIPE_MTE2) {
   if(id<0||id>=4) {fail(23);return;}
   L1State &s=s1[id];
   if(!s.owned||!s.readable||s.ready||s.free) fail(24);
   // B may be reused only after all eight current L1 fragments were read.
   if(id<2 && s.read_count!=8) fail(25);
   s.free=1;s.owned=0;s.readable=0;s.issued=0;
 } else if(from==PIPE_MTE1 && to==PIPE_M) {
   if(id<0||id>=4) {fail(26);return;}
   L0State &s=s0[id];
   if(!s.owned||s.ready||s.readable) fail(27);
   s.ready=1;
 } else if(from==PIPE_M && to==PIPE_MTE1) {
   if(id<0||id>=4) {fail(28);return;}
   L0State &s=s0[id];
   if(!s.owned||!s.readable||s.ready||s.free) fail(29);
   s.free=1;s.owned=0;s.readable=0;
 } else if(!(from==PIPE_M&&to==PIPE_MTE2&&id==EVENT_ID7)) fail(30);
}
void copy_gm_to_cbuf_multi_nd2nz_b16(half *dst,half *src,int sid,int nd,int n,int d,
 int srcNd,int srcD,int dstC0,int dstN,int dstMatrix) {
 int slot=slot1(dst),desc=-1,tile=-1,row=0;
 bool is_v=slot<2;
 for(int i=0;i<descriptors;i++) {
   half *base=is_v?v_gm[i]:p_gm[i];
   int size=is_v?32*16384:64*16384;
   if(src>=base && src<base+size) {
     desc=i;int off=int(src-base);row=off/(32*16384);tile=(off%(32*16384))/16384;
     if(off%16384) fail(31);
   }
 }
 if(desc<0||tile<0||tile>=kx[desc]) {fail(32);return;}
 if(history[desc]&&!history_done[desc]) fail(33);
 L1State &s=s1[slot];
 if(!s.owned||s.free||s.ready||s.readable) fail(34);
 if(sid!=0||srcD!=128||dstC0!=128||dstN!=1||d!=128) fail(35);
 if(is_v) {
   if(dst!=l1[slot]||nd!=1||n!=128||srcNd!=0||dstMatrix!=0) fail(36);
   if(s.issued) fail(37);
   if(tile!=v_loaded[desc]) fail(38);
   // A future DMA is issued while the prior tile's MAD has not completed.
   if(tile>0&&active_desc==desc&&active_tile==tile-1&&active_row==0) ++prefetch_hits;
   ++v_loaded[desc];++total_v;
 } else {
   if(dst!=l1[slot]+row*16384 || (nd!=1&&nd!=2) || n!=128/nd ||
      srcNd!=128/nd*128*2 || dstMatrix!=128*16/nd) fail(39);
   if(s.issued && (s.desc!=desc||s.tile!=tile||(s.rows&(1<<row)))) fail(40);
   s.rows|=1<<row;++total_p;
   mix(4000+desc*100+tile*2+row);mix(slot*100+nd);
 }
 s.issued=1;s.desc=desc;s.tile=tile;
}
void load_cbuf_to_cb(half *dst,half *src,int base,int repeat,int stride,
 int dstStride,int sid,bool transpose,int mode) {
 int b=slot0(dst,true),slot=slot1(src),frag=int(dst-l0b[b])/2048;
 if(slot>=2||frag<0||frag>=8||dst!=l0b[b]+frag*2048||src!=l1[slot]+frag*256) {fail(41);return;}
 L1State &s=s1[slot];L0State &o=s0[b+2];
 if(!s.owned||!s.readable||s.ready||s.free) fail(42);
 if(!o.owned||o.readable||o.free||frag!=o.frags) fail(43);
 if(base||repeat!=8||stride!=8||dstStride||sid||!transpose||mode!=inc) fail(44);
 if(frag==0) {active_desc=s.desc;active_tile=s.tile;active_row=0;}
 o.desc=s.desc;o.tile=s.tile;++o.frags;++s.read_count;
 if(frag==7) {++v_consumed[s.desc];last_v_tile[s.desc]=s.tile;}
}
void load_cbuf_to_ca(half *dst,half *src,int base,int repeat,int stride,
 int dstStride,int sid,bool transpose,int mode) {
 int a=slot0(dst,false),slot=slot1(src),frag=int(dst-l0a[a])/2048;
 int row=int(src-l1[slot])/16384;
 if(slot<2||frag<0||frag>=8||dst!=l0a[a]+frag*2048||src!=l1[slot]+row*16384+frag*256) {fail(45);return;}
 L1State &s=s1[slot];L0State &o=s0[a];
 if(!s.owned||!s.readable||s.ready||s.free||!(s.rows&(1<<row))) fail(46);
 if(!o.owned||o.readable||o.free||frag!=o.frags) fail(47);
 if(base||repeat!=8||stride!=8||dstStride||sid||transpose||mode!=inc) fail(48);
 o.desc=s.desc;o.tile=s.tile;o.row=row;++o.frags;++s.read_count;
}
void mad(float *dst,half *pa,half *vb,int m,int k,int n,int unit,int align,int source,int init) {
 int a=slot0(pa,false),b=slot0(vb,true),j=active_row++;
 L0State &p=s0[a],&v=s0[b+2];
 if(!p.owned||!p.readable||!v.owned||!v.readable||v.frags!=8) fail(49);
 if(v.desc!=active_desc||v.tile!=active_tile) fail(50);
 bool skip=upper[active_desc]&&active_tile==kx[active_desc]-1&&j==0;
 if(!skip && (p.frags!=8||p.desc!=active_desc||p.tile!=active_tile||p.row!=j)) fail(51);
 bool last=(active_tile==kx[active_desc]-1&&(j!=0||!upper[active_desc])) ||
           (upper[active_desc]&&active_tile==kx[active_desc]-2&&j==0);
 if(dst!=(j?l0c[0]:l0c[1])||m!=128||k!=128||n!=128||unit!=(last?3:2)||
    align||source||init!=(active_tile==0&&!history[active_desc])) fail(52);
 mix(1000+active_desc*100+active_tile*2+j);mix(a*100+b*10+unit);
 mix(p.desc*1000+p.tile*10+p.row);mix(init);++total_mad;
}
void copy_matrix_cc_to_gm(float *dst,float *src,int sid,int n,int m,int ds,int ss,int unit,
 int quant,int relu,bool split,bool nz) {
 int j=active_row-1;
 if(dst!=output_gm[active_desc]+j*16384||src!=(j?l0c[0]:l0c[1])||sid||n!=128||m!=128||
    ds!=128||ss!=128||unit!=3||quant!=NoQuant||relu||split||!nz) fail(53);
 mix(2000+active_desc*100+active_tile*2+j);++total_fix;
}
void drained() {
 for(int i=0;i<4;i++) {
   if(!s1[i].free||s1[i].owned||s1[i].ready||s1[i].readable||s1[i].issued) fail(54);
   if(!s0[i].free||s0[i].owned||s0[i].ready||s0[i].readable) fail(55);
 }
}
template<class TYPE,bool IF_BF16,class WORKSPACE_TYPE> class CubeForward {
public:
 template<class L,class R,class O> void cube2_matmul_op(
   const Address::PhyAddrForwardCube2Online<L,R,O>*,int64_t,int64_t,int64_t);
 void diag_matmul(float *out,float *diag,int height,bool copy) {
   drained();int desc=-1;
   for(int i=0;i<descriptors;i++) if(out==output_gm[i]) desc=i;
   if(desc<0) {fail(56);return;}
   if(copy) {
     if(!tail[desc]||v_consumed[desc]!=kx[desc]) fail(57);
     tail_done[desc]=1;
   } else {
     if(!history[desc]||v_loaded[desc]!=0||height!=2) fail(58);
     history_done[desc]=1;
   }
   mix(3000+desc*2+copy);
   // Model the shared L0 writes of the diagonal helper. L1B is disjoint.
   for(int i=0;i<4;i++) {s0[i].desc=-100;s0[i].tile=-100;s0[i].row=-100;}
 }
 TYPE *l1_a_ping_=l1[2],*l1_a_pong_=l1[3],*l1_b_ping_=l1[0],*l1_b_pong_=l1[1];
 TYPE *l0_a_ping_=l0a[0],*l0_a_pong_=l0a[1],*l0_b_ping_=l0b[0],*l0_b_pong_=l0b[1];
 float *l0_c_ping_=l0c[0],*l0_c_pong_=l0c[1];
 float *rowmax_diag=diag_gm,*rowsum_diag=diag_gm;
 uint32_t ping_pong_flag_l1_a_=0,ping_pong_flag_l1_b_=0,ping_pong_flag_l0_a_=0,
          ping_pong_flag_l0_b_=0,ping_pong_flag_l0_c_=0;
 int D=128,sparseMode=0,qk_triangle=0,last_idx=0;
};
"""

SCENARIOS = r"""
void reset() {
 fault=0;descriptors=4;active_desc=-1;active_tile=-1;active_row=-1;
 total_mad=0;total_fix=0;total_v=0;total_p=0;prefetch_hits=0;
 for(int i=0;i<4;i++) {
   s1[i].free=1;s1[i].owned=0;s1[i].ready=0;s1[i].readable=0;s1[i].issued=0;
   s1[i].read_count=0;s1[i].rows=0;s1[i].desc=-100;s1[i].tile=-100;
   s0[i].free=1;s0[i].owned=0;s0[i].ready=0;s0[i].readable=0;s0[i].frags=0;
   s0[i].desc=-100;s0[i].tile=-100;s0[i].row=-100;
   history_done[i]=tail_done[i]=v_loaded[i]=v_consumed[i]=0;last_v_tile[i]=-1;
 }
}
void qk_handoff(CubeForward<half,false,half>& op) {
 drained();
 // An intervening unchanged QK owns/writes all shared L0 and L1 buffers;
 // then returns the same free-token state, with nontrivial slot parity.
 op.ping_pong_flag_l1_a_^=1;op.ping_pong_flag_l1_b_^=1;
 op.ping_pong_flag_l0_a_^=1;op.ping_pong_flag_l0_b_^=1;op.ping_pong_flag_l0_c_^=1;
 for(int i=0;i<4;i++) {s0[i].desc=-200;s0[i].tile=-200;s0[i].row=-200;}
}
void run_batch(CubeForward<half,false,half>& op,int count,int cores,int round) {
 Address::PhyAddrForwardCube2Online<half,half,float> desc[4];
 int num_tiles=0,num_rows=0;
 for(int i=0;i<count;i++) {
   desc[i].left=p_gm[i];desc[i].right=v_gm[i];desc[i].out=output_gm[i];
   desc[i].kx=kx[i];desc[i].ky=ky[i];desc[i].upperRight=upper[i]!=0;
   desc[i].onStartSection=!history[i];desc[i].onEndSection=tail[i]!=0;
   desc[i].lineStride=32*16384;
   num_tiles+=kx[i];num_rows+=kx[i]*ky[i];
 }
 int a1=op.ping_pong_flag_l1_a_,b1=op.ping_pong_flag_l1_b_,
     a0=op.ping_pong_flag_l0_a_,b0=op.ping_pong_flag_l0_b_,c0=op.ping_pong_flag_l0_c_;
 int mads=total_mad,loads=total_v;
 op.cube2_matmul_op(desc,count,cores,round);
 drained();
 if(total_mad-mads!=num_rows||total_v-loads!=num_tiles) fail(59);
 if(op.ping_pong_flag_l1_a_!=(a1^(num_tiles&1))||op.ping_pong_flag_l1_b_!=(b1^(num_tiles&1))||
    op.ping_pong_flag_l0_b_!=(b0^(num_tiles&1))||op.ping_pong_flag_l0_a_!=(a0^(num_rows&1))||
    op.ping_pong_flag_l0_c_!=(c0^(num_rows&1))) fail(60);
 for(int i=0;i<count;i++) {
   if(v_loaded[i]!=kx[i]||v_consumed[i]!=kx[i]||history_done[i]!=history[i]||tail_done[i]!=tail[i]) fail(61);
 }
}
EXPORT int run_model() {
 sequence_hash=0;
 int lengths[6]={0,1,2,3,8,32};
 for(int entry=0;entry<8;entry++) for(int len=0;len<6;len++) for(int guard=0;guard<6;guard++) {
   reset();CubeForward<half,false,half> op;
   op.ping_pong_flag_l1_a_=entry&1;op.ping_pong_flag_l1_b_=(entry>>1)&1;
   op.ping_pong_flag_l0_a_=(entry>>2)&1;op.ping_pong_flag_l0_b_=1-(entry&1);
   op.ping_pong_flag_l0_c_=(entry>>1)&1;
   op.D=guard==1?64:128;op.sparseMode=guard==2;op.qk_triangle=guard==3;
   int cores=guard==4?2:1;
   for(int i=0;i<4;i++) {
     kx[i]=lengths[(len+i)%6];ky[i]=guard==5?1:2;upper[i]=0;
     history[i]=i&1;tail[i]=(i>>1)&1;
   }
   run_batch(op,4,cores,entry%2);
   if(fault) return fault;
   if(CANDIDATE && guard==0 && prefetched_expected(kx,4)!=prefetch_hits) return 62;
   if((!CANDIDATE||guard!=0)&&prefetch_hits) return 63;
   qk_handoff(op);
   for(int i=0;i<4;i++) {history_done[i]=tail_done[i]=v_loaded[i]=v_consumed[i]=0;}
   active_desc=-1;active_tile=-1;active_row=-1;
   run_batch(op,4,cores,1-entry%2);
   if(fault) return fault;
   run_batch(op,0,cores,0);
   if(fault) return fault;
 }
 // Sparse/upper-right fallback has original dead j0 MAD on its last tile.
 for(int entry=0;entry<2;entry++) {
   reset();CubeForward<half,false,half> op;
   op.ping_pong_flag_l1_b_=entry;op.ping_pong_flag_l1_a_=entry;
   kx[0]=3;ky[0]=2;upper[0]=1;history[0]=0;tail[0]=1;
   run_batch(op,1,1,0);if(fault)return fault;if(prefetch_hits)return 64;
 }
 return 0;
}
EXPORT uint64_t get_sequence() { return sequence_hash; }
"""


def methods(data):
    return ("template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>\n"
            "template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>\n" +
            function(data.decode(), patch.PV_BEGIN.decode()))


def compile_model(compiler, directory, data, candidate):
    source = (f"#define CANDIDATE {int(candidate)}\n" + MODEL + methods(data) +
              "\nint prefetched_expected(int *ks,int n) {int c=0;for(int i=0;i<n;i++)if(ks[i]>0)c+=ks[i]-1;return c;}\n" + SCENARIOS)
    library = compile_library(compiler, directory, source)
    library.run_model.restype = ctypes.c_int
    library.get_sequence.restype = ctypes.c_uint64
    return library.run_model(), library.get_sequence()


def test_actual_pv_matches_mad_fix_history_and_all_slot_handoffs(cpp_compiler, cube, tmp_path):
    old = compile_model(cpp_compiler, tmp_path / "old", cube, False)
    new = compile_model(cpp_compiler, tmp_path / "new", patch.patch_cube(cube), True)
    assert old[0] == 0
    assert new[0] == 0
    assert old[1] == new[1]


@pytest.mark.parametrize("name,old,new", [
    ("missing_current_ready", patch.READY_WAIT, b""),
    ("missing_next_free", b"                        wait_flag(PIPE_MTE1, PIPE_MTE2, pv_next_v_slot);", b""),
    ("wrong_slot", b"const uint32_t pv_next_v_slot = 1 - ping_pong_flag_l1_b_;", b"const uint32_t pv_next_v_slot = ping_pong_flag_l1_b_;"),
    ("tail_overread", b"if (i + 1 < Kx)", b"if (i + 1 <= Kx)"),
    ("missing_ready_publish", b"                        set_flag(PIPE_MTE2, PIPE_MTE1, pv_next_v_slot);", b""),
    ("wrong_next_v", b"right_start_addr + (i + 1) * l1_k * n_", b"right_start_addr + i * l1_k * n_"),
    ("wrong_parity", b"ping_pong_flag_l1_b_ = 1 - ping_pong_flag_l1_b_;", b"ping_pong_flag_l1_b_ = 0;"),
])
def test_model_detects_lifetime_address_and_tail_faults(cpp_compiler, cube, tmp_path, name, old, new):
    data = patch.patch_cube(cube)
    begin, end = data.index(patch.PV_BEGIN), data.index(patch.PV_END)
    pv = data[begin:end]
    assert pv.count(old) == 1
    damaged = data[:begin] + pv.replace(old, new) + data[end:]
    assert compile_model(cpp_compiler, tmp_path / name, damaged, True)[0] != 0


def test_model_rejects_release_before_l1_read_complete(cpp_compiler, cube, tmp_path):
    data = patch.patch_cube(cube)
    release = b"                    set_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_b_);"
    assert data.count(release) == 1
    data = data.replace(release, b"")
    before_read = b"                for (int nn = 0; nn < k0_ / SIZE_16; nn++) {"
    assert data.count(before_read) == 1
    data = data.replace(before_read, b"                if (pv_v_prefetch) {" + release + b"}\n" + before_read)
    assert compile_model(cpp_compiler, tmp_path / "early_release", data, True)[0] != 0
