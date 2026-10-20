"""Execute actual vscale methods in host models, not a device compiler claim."""
import ctypes
from pathlib import Path
import re
import subprocess

import pytest

from scripts import mindie_laser_quant_vscale_patch as patch
from scripts import prepare_mindie_laser_quant_qresident_source as baseline
from scripts.prepare_cann_fas_source import git
from tests import test_mindie_laser_quant_qresident_patch as resident
from tests.test_mindie_laser_expcast_cpp import cpp_compiler


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source_files():
    reference = ROOT / "build/mindie-sd-reference"
    if not (reference / ".git").exists():
        pytest.skip("Pinned local MindIE source missing; no network fetch")
    assert git(reference, "rev-parse", "HEAD").decode().strip() == baseline.SOURCE_LOCK.commit
    files, _ = baseline.expected_files(reference)
    return {Path(name).name: data for name, data in files.items() if "/op_kernel/" in name}


def function(text, signature):
    start = text.index(signature)
    body = text.index("{", start)
    depth = 1
    end = body + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def test_only_score_scale_ownership_changes(source_files):
    old = source_files["CubeForward.h"]
    new = patch.patch_cube(old)
    assert b"ReadQuantScale" not in new and b"q_scales_" not in new
    assert new.count(b"set_quant_pre(") == 1
    assert new.count(b"pipe_barrier(PIPE_FIX)") == 1
    assert patch.FIXED_QUANT_INIT in new
    events = rb"(?:set_flag|wait_flag|wait_flag_dev|ffts_cross_core_sync|mad|copy_matrix_cc_to_gm)\([^;]*\);"
    # Store comment changes do not affect instruction parameters.
    clean = lambda d: re.sub(rb"//[^\n]*", b"", d)
    assert re.findall(events, clean(new)) == re.findall(events, clean(old))
    assert new[:new.index(patch.QK_BEGIN)].count(b"diag_matmul(") == old[:old.index(patch.QK_BEGIN)].count(b"diag_matmul(")
    vector = patch.patch_vector(source_files["VectorForward.h"])
    assert vector.count(patch.NEW_SCORE_SCALE) == 1 and patch.OLD_SCORE_SCALE not in vector
    assert b"rowPerTime = 4;" in vector
    assert patch.VECTOR_METHODS in vector
    assert patch.apply("unrelated.h", b"unchanged") == b"unchanged"


@pytest.mark.parametrize("filename", [
    "CubeForward.h", "VectorForward.h", "address_const.h",
    "AddressMappingVectorForwardOnline.h", "vc_laser_quant_qresident_attention.cpp",
])
def test_duplicate_patch_fails_closed(source_files, filename):
    new = patch.apply(filename, source_files[filename])
    with pytest.raises(ValueError):
        patch.apply(filename, new)


@pytest.mark.parametrize("filename,old,new", [
    ("CubeForward.h", b"n_index == 0 || ky != 2", b"changed_q_residency"),
    ("CubeForward.h", patch.TILING, patch.TILING.replace(b"64, 2", b"32, 2")),
    ("VectorForward.h", b"\n            rowPerTime = 4;", b"\n            rowPerTime = 8;"),
    ("VectorForward.h", patch.OLD_SCORE_SCALE, b"unknown_scale();"),
])
def test_source_drift_fails_closed(source_files, filename, old, new):
    with pytest.raises(ValueError):
        patch.apply(filename, source_files[filename].replace(old, new))


def compile_library(compiler, directory, source):
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory / "vscale.cpp"
    filename.write_text(source, encoding="utf-8")
    if compiler.name.lower() in ("cl", "cl.exe"):
        output = directory / "vscale.dll"
        command = [str(compiler), "/nologo", "/std:c++17", "/utf-8", "/LD", "/Od",
                   "/GS-", "/Zl", str(filename), "/link", "/NOENTRY", "/NODEFAULTLIB", f"/OUT:{output}"]
    else:
        output = directory / "vscale.so"
        command = [str(compiler), "-std=c++17", "-shared", "-fPIC", "-O0", str(filename), "-o", str(output)]
    result = subprocess.run(command, cwd=directory, capture_output=True, timeout=30)
    assert result.returncode == 0, (result.stdout + result.stderr).decode(errors="replace")
    library = ctypes.CDLL(str(output))
    library.run_model.restype = ctypes.c_int
    return library


def test_actual_qk_methods_preserve_residency_and_operand_events(cpp_compiler, source_files, tmp_path):
    new = patch.patch_cube(source_files["CubeForward.h"])
    source = resident.host_source(resident.actual_methods(new), True)
    source = source.replace("bool,float,float,float,bool", "bool,bool")
    old_gain_check = """    float gain=q_scales[active*2+row]*k_scales[active*32+matmuls-1];
    uint32_t bits=*reinterpret_cast<uint32_t *>(&gain)&0xffffe000u;"""
    assert old_gain_check in source
    source = source.replace(old_gain_check, "    uint32_t bits=0x3c800000u;")
    source = source.replace("void init_state() {", "void init_state() { quant_value=0x3c800000u;")
    assert compile_library(cpp_compiler, tmp_path, source).run_model() == 0


PREAMBLE = r"""
typedef signed char int8_t; typedef unsigned char uint8_t;
typedef signed int int32_t; typedef unsigned int uint32_t;
typedef signed long long int64_t; typedef unsigned long long uint64_t;
typedef unsigned long long uintptr_t; typedef unsigned long long size_t;
#define __aicore__
#define __inline__ inline
#define __gm__
#define __ubuf__
#define ROUND_UP_8(x) (((x)+7)/8*8)
#ifdef _MSC_VER
#define EXPORT extern "C" __declspec(dllexport)
extern "C" int _fltused=0;
#else
#define EXPORT extern "C"
#endif
int fault=0;
void fail(int code) { if(!fault) fault=code; }
uint32_t fbits(float f) { return *reinterpret_cast<uint32_t *>(&f); }
float frombits(uint32_t u) { return *reinterpret_cast<float *>(&u); }
struct half {
    unsigned short bits;
    half() {}
    explicit half(float f) {
        uint32_t u=fbits(f),sign=(u>>16)&0x8000u,m=u&0x7fffffu;
        int e=int((u>>23)&255)-127+15;
        if(((u>>23)&255)==255) { bits=(unsigned short)(sign|0x7c00u|(m?0x200u:0u)); return; }
        if(e<=0) {
            if(e < -10) { bits=(unsigned short)sign; return; }
            m|=0x800000u; int shift=14-e;
            uint32_t q=m>>shift,rem=m&((1u<<shift)-1),tie=1u<<(shift-1);
            if(rem>tie || (rem==tie && (q&1))) ++q;
            bits=(unsigned short)(sign|q); return;
        }
        m+=0xfffu+((m>>13)&1);
        if(m&0x800000u) { m=0; ++e; }
        bits=(unsigned short)(sign|(e>=31?0x7c00u:((uint32_t(e)<<10)|(m>>13))));
    }
    explicit operator float() const {
        uint32_t sign=(uint32_t(bits)&0x8000u)<<16,m=bits&0x3ffu,e=(bits>>10)&31;
        if(!e) {
            if(!m) return frombits(sign);
            int p=-14; while(!(m&0x400u)) { m<<=1; --p; }
            return frombits(sign|(uint32_t(p+127)<<23)|((m&0x3ffu)<<13));
        }
        return frombits(sign|(e==31?0x7f800000u:((e+112)<<23))|(m<<13));
    }
};
enum {PIPE_V,PIPE_FIX};
int vector_mask=128,vmul_calls=0,cast_calls=0,invalidations=0;
void set_vector_mask(uint64_t hi,uint64_t lo) {
    if(hi!=~uint64_t(0) || lo!=~uint64_t(0)) fail(1);
    vector_mask=128;
}
void pipe_barrier(int p) { if(p!=PIPE_V && p!=PIPE_FIX) fail(2); }
void vector_dup(half *dst,half x,int repeat,int a,int b,int c,int d) {
    if(repeat!=1||a!=1||b!=1||c!=8||d!=1||vector_mask!=64) fail(3);
    for(int i=0;i<vector_mask;++i) dst[i]=x;
}
void vmul(half*dst,half*a,half*b,int repeat,int db,int ab,int bb,int dr,int ar,int br) {
    if(db!=1||ab!=1||bb!=0||dr!=8||ar!=8||br!=1||vector_mask!=128) fail(4);
    ++vmul_calls;
    for(int r=0;r<repeat;++r) for(int lane=0;lane<128;++lane) {
        int dl=r*dr*16+(lane/16)*db*16+lane%16;
        int al=r*ar*16+(lane/16)*ab*16+lane%16;
        int bl=r*br*16+(lane/16)*bb*16+lane%16;
        dst[dl]=half(float(a[al])*float(b[bl]));
    }
}
void vconv_f162f32(float*d,half*s,int repeat,int db,int sb,int dr,int sr) {
    if(db!=1||sb!=1||dr!=8||sr!=4||repeat!=8) fail(5);
    ++cast_calls; for(int i=0;i<repeat*64;++i) d[i]=float(s[i]);
}
void vmuls(float*d,float*s,float x,int repeat,int db,int sb,int dr,int sr) {
    if(db!=1||sb!=1||dr!=8||sr!=8||repeat!=8) fail(6);
    for(int i=0;i<repeat*64;++i) d[i]=s[i]*x;
}
void vconv_f322f16(half*d,float*s,int repeat,int db,int sb,int dr,int sr) {
    if(db!=1||sb!=1||dr!=4||sr!=8||repeat!=8) fail(7);
    ++cast_calls; for(int i=0;i<repeat*64;++i) d[i]=half(s[i]);
}
alignas(64) float q_values[64],k_values[64];
float q_cache[64],k_cache[64];
namespace AscendC {
enum class CacheLine {SINGLE_CACHE_LINE}; enum class DcciDst {CACHELINE_OUT};
template<class T> struct GlobalTensor {
    T*ptr;
    void SetGlobalBuffer(T*p,int n) { ptr=p; if(n!=1) fail(8); }
    T GetValue(int i) {
        if(i!=0) fail(9);
        if(ptr>=q_values && ptr<q_values+64) return q_cache[ptr-q_values];
        if(ptr>=k_values && ptr<k_values+64) return k_cache[ptr-k_values];
        fail(10); return T(0);
    }
};
template<class T,CacheLine C,DcciDst D> void DataCacheCleanAndInvalid(GlobalTensor<T>g) {
    ++invalidations;
    T*base; T*cache;
    if(g.ptr>=q_values && g.ptr<q_values+64) { base=q_values;cache=q_cache; }
    else if(g.ptr>=k_values && g.ptr<k_values+64) { base=k_values;cache=k_cache; }
    else { fail(11);return; }
    int first=int(g.ptr-base)/16*16;
    for(int i=0;i<16;++i) cache[first+i]=base[first+i];
}
}
constexpr int32_t BASE_BLOCK_SIDE_LEN=128;
constexpr int32_t MAX_LENG_PER_UB_PROC=16384;
"""


VECTOR_SCENARIOS = r"""
alignas(32) half score_storage[16384+32], gain_storage[2048+32], scratch_storage[16384+32];
alignas(32) float coefficient_storage[32+8];
float original_scores[16384];
EXPORT int run_model() {
    fault=0;
    VectorForward<half,false,half> op;
    VectorForward<half,false,half>::UB_FOR_SHORT_LEN_ATTN_SCORE ub;
    op.allocate_ubuf_for_short_seq_attn_score(&ub);
    if((uintptr_t)ub.vscale_gain_fp16!=174848 ||
       (uintptr_t)ub.vscale_coeff_fp32!=178944) return 12;
    ub.vscale_gain_fp16=gain_storage;
    ub.vscale_coeff_fp32=coefficient_storage;
    ub.buf_for_cacl_rowmax_fp16=scratch_storage;
    for(int i=0;i<64;++i) q_values[i]=float(i+1)*0.125f;
    for(int count=1;count<=32;++count) {
        // Adjacent block scales force normal, zero, tiny and huge coefficients.
        for(int i=0;i<64;++i) {
            float cases[8]={0.0f,0.0000000001f,0.0000001f,0.0001f,0.03125f,1.0f,1000.0f,-0.003f};
            k_values[i]=cases[i%8];
        }
        for(int iteration=0;iteration<2;++iteration) {
            op.SCALE=0.08838835f;
            if(iteration) for(int i=0;i<64;++i) k_values[i]*=1.5f;
            op.SetQuantScales(reinterpret_cast<uint8_t *>(q_values),reinterpret_cast<uint8_t *>(k_values));
            int before=invalidations;
            op.PrepareQuantScaleSection(17,9,count,ub);
            if(invalidations<=before) return 13;
            uint32_t expected_slow=0;
            for(int j=0;j<count;++j) {
                float g=frombits(fbits(q_values[17]*k_values[9+j])&0xffffe000u);
                float coeff=(g*64.0f)*float(half(op.SCALE));
                if(fbits(coefficient_storage[j])!=fbits(coeff)) return 14;
                float a=coeff<0?-coeff:coeff;
                bool fast=coeff==0||(a>=0.00006103515625f&&a<=65504.0f);
                if(!fast) expected_slow|=1u<<j;
                for(int r=0;r<64;++r)
                    if(gain_storage[j*64+r].bits!=half(fast?coeff:1.0f).bits) return 15;
            }
            if(op.vscale_slow_mask_!=expected_slow) return 16;
            for(int chunk=0;chunk<16;++chunk) {
                for(int i=0;i<count*512;++i) {
                    float values[8]={-32768.0f,-64.0f,-0.015625f,0.0f,0.015625f,1.0f,100.0f,32768.0f};
                    score_storage[i]=half(values[(i+chunk)%8]);
                    original_scores[i]=float(score_storage[i]);
                }
                score_storage[count*512].bits=0x3555;
                scratch_storage[1024].bits=0x3555; // scratch FP32 block is 2048B.
                int before_vmul=vmul_calls;
                op.ApplyQuantScoreScale(score_storage,count,ub);
                if(vmul_calls!=before_vmul+1) return 17;
                for(int i=0;i<count*512;++i) {
                    int b=i/512; float c=coefficient_storage[b];
                    if(!(expected_slow&(1u<<b))) c=float(half(c));
                    half expected(original_scores[i]*c);
                    if(expected.bits!=score_storage[i].bits) return 18;
                }
                if(score_storage[count*512].bits!=0x3555||scratch_storage[1024].bits!=0x3555) return 19;
            }
        }
    }
    if(cast_calls==0) return 20;
    return fault;
}
"""


def vector_model(data):
    text = patch.patch_vector(data).decode()
    structure = function(text, "    struct UB_FOR_SHORT_LEN_ATTN_SCORE") + ";"
    allocate = function(text, "::allocate_ubuf_for_short_seq_attn_score(")
    allocate = allocate.replace("::allocate_", "void allocate_", 1)
    return PREAMBLE + """
template<class INPUT_T,bool IF_BF16,class WORKSPACE_T> struct VectorForward {
    int32_t rowPerTime=4, SIZE_FP16=2, SIZE_FP32=4;
    float SCALE=1.0f;
    uint32_t vscale_slow_mask_=0;
    void __set_mask(int n) { vector_mask=n; }
""" + structure + patch.VECTOR_SCALE_SETUP.decode() + patch.NEW_SCALE_BLOCK.decode() + \
        patch.VECTOR_METHODS.decode() + allocate + "\n};\n" + VECTOR_SCENARIOS


def test_actual_vector_prepare_scale_fallback_cache_and_ub_layout(cpp_compiler, source_files, tmp_path):
    source = vector_model(source_files["VectorForward.h"])
    status = compile_library(cpp_compiler, tmp_path, source).run_model()
    assert status == 0, status


MAPPING_SCENARIOS = r"""
EXPORT int run_model() {
    // Remote validation geometries: never skip a capacity violation here.
    // Runtime must report the actual Cube count; these are tested targets.
    int device_cores[]={24,32,64};
    int validation_keys[]={256,512,4352,8448,37888};
    for(int c=0;c<3;++c) for(int k=0;k<5;++k)
    for(int core=0;core<device_cores[c];++core) for(int vi=0;vi<2;++vi) {
        Address::AddressMappingVectorForwardOnline op;
        op.init(2,17,512,validation_keys[k],validation_keys[k],0,0,0,64,2);
        op.set_core_info(device_cores[c],core,vi); op.start();
        for(int round=0;round<op.get_total_round();++round) {
            if(!op.is_running(round)) continue;
            Address::ForWardAddrOnline descriptors[16]; int64_t n=0;
            op.forward_addrMapping_pre(descriptors,n,round);
            if(n>4) return 37;
            Address::FORWARD_SECTION_INFO s;
            op.get_section_info(round,s);
            if(s.sectionNum!=n*2) return 38;
            for(int i=0;i<n;++i) {
                auto d=descriptors[i];
                int64_t qb=(d.b*17+d.n)*4+d.iR;
                int64_t kb=(d.b*17+d.n)*(validation_keys[k]/128)+d.iC;
                for(int row=0;row<2;++row)
                    if(s.qScaleBlockIndex[2*i+row]!=qb+row ||
                       s.kScaleBlockIndex[2*i+row]!=kb) return 39;
            }
        }
    }
    int queries[]={256,512,768,4352};
    int keys[]={256,512,768,4096,4352,8192};
    for(int qi=0;qi<4;++qi) for(int ki=0;ki<6;++ki)
    for(int cores=1;cores<=4;++cores) for(int core=0;core<cores;++core)
    for(int vi=0;vi<2;++vi) {
        Address::AddressMappingVectorForwardOnline op;
        op.init(2,3,queries[qi],keys[ki],keys[ki],0,0,0,64,2);
        op.set_core_info(cores,core,vi); op.start();
        for(int round=0;round<op.get_total_round();++round) {
            if(!op.is_running(round)) continue;
            Address::ForWardAddrOnline descriptors[16];
            int64_t n=0; op.forward_addrMapping_pre(descriptors,n,round);
            // The inherited mapper stores at most four two-row descriptors.
            // Synthetic low-core tilings exceeding that capacity are outside
            // this unchanged scheduling contract; do not overflow its arrays.
            if(n>4) continue;
            Address::FORWARD_SECTION_INFO s;
            op.get_section_info(round,s);
            if(s.sectionNum!=n*2) return 31;
            for(int i=0;i<n;++i) {
                auto d=descriptors[i];
                int64_t qbase=(d.b*3+d.n)*(queries[qi]/128)+d.iR;
                int64_t kbase=(d.b*3+d.n)*(keys[ki]/128)+d.iC;
                for(int row=0;row<2;++row) {
                    int j=2*i+row;
                    if(s.qScaleBlockIndex[j]!=qbase+row || s.kScaleBlockIndex[j]!=kbase) return 32;
                    if(s.qScaleBlockIndex[j]!=s.rowmaxOffset[j]/128) return 33;
                    if(s.sectionBlockNums[j]!=d.kx || d.kx>32) return 34;
                    if(kbase+d.kx>(d.b*3+d.n+1)*(keys[ki]/128)) return 35;
                    // Workspace row address identifies the same Q128 half for either AIV.
                    if(s.sectionBlockOffset[j]%16384!=vi*64*128) return 36;
                }
            }
        }
    }
    return 0;
}
"""


def test_actual_dense_mapper_crosses_rounds_rows_heads_batches(cpp_compiler, source_files, tmp_path):
    const = patch.patch_address_const(source_files["address_const.h"]).decode()
    mapper = patch.patch_vector_address(source_files["AddressMappingVectorForwardOnline.h"]).decode()
    mapper = re.sub(r'#include[^\n]*\n', "", mapper)
    source = PREAMBLE + const + mapper + MAPPING_SCENARIOS
    status = compile_library(cpp_compiler, tmp_path, source).run_model()
    assert status == 0, status
