"""Cache-maintenance delta and host C++ stale-cache simulation; not an NPU test."""
import ctypes
from pathlib import Path
import subprocess

import pytest

from scripts import mindie_laser_quant_cache_patch as patch
from scripts import prepare_mindie_laser_quant_reuse_source as baseline
from scripts.prepare_cann_fas_source import git
from tests.test_mindie_laser_expcast_cpp import cpp_compiler


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def baseline_cube():
    reference = ROOT / "build/mindie-sd-reference"
    if not (reference / ".git").exists():
        pytest.skip("Pinned MindIE cache unavailable; these checks never download source")
    assert git(reference, "rev-parse", "HEAD").decode().strip() == baseline.SOURCE_LOCK.commit
    files, _ = baseline.expected_files(reference)
    return files[f"{baseline.CUSTOM_OP_ROOT}/op_kernel/CubeForward.h"]


def test_strict_inverse_proves_pipeline_and_math_unchanged(baseline_cube):
    new = patch.patch_cube(baseline_cube)
    restored = new.replace(patch.NEW_RESET, patch.OLD_RESET)
    restored = restored.replace(patch.NEW_SCALE_BLOCK, patch.OLD_SCALE_BLOCK)
    restored = restored.replace(patch.NEW_COMMENT, patch.OLD_COMMENT)
    for old, replacement in patch.READ_REPLACEMENTS:
        restored = restored.replace(replacement, old)
    assert restored == baseline_cube
    assert new.count(b"DataCacheCleanAndInvalid<") == baseline_cube.count(b"DataCacheCleanAndInvalid<")
    assert b"static " not in patch.NEW_SCALE_BLOCK
    assert b"SetGlobalBuffer(ptr, 1);" in patch.NEW_SCALE_BLOCK
    assert b"return scale.GetValue(0);" in patch.NEW_SCALE_BLOCK
    assert b"reinterpret_cast<__gm__ float *>(line)" not in new


@pytest.mark.parametrize("mutation", [
    lambda x: x + b"\nReadQuantScale(extra);\n",
    lambda x: x.replace(b"scale.SetGlobalBuffer(ptr, 1);", b"scale.SetGlobalBuffer(ptr, 16);"),
    lambda x: x.replace(patch.READ_REPLACEMENTS[2][0], b"ReadQuantScale(k_scales_ + changed_index)"),
    lambda x: x.replace(b"k_scales_ = reinterpret_cast<__gm__ float *>(k);", b"k_scales_ = nullptr;"),
    patch.patch_cube,
])
def test_stale_or_repeated_patch_fails_closed(baseline_cube, mutation):
    with pytest.raises(ValueError):
        patch.patch_cube(mutation(baseline_cube))


def test_other_files_are_byte_identical():
    for name in ("VectorForward.h", "vc_laser_quant_cache_attention.cpp", "AddressMappingForwardOnline.h"):
        raw = b"ReadQuantScale(value);\nmad(...);\nset_flag(...);"
        assert patch.apply("op_kernel/" + name, raw) is raw


HOST_MODEL = r"""
typedef unsigned char uint8_t;
typedef unsigned long long uint64_t;
#define __gm__
#define __aicore__
#ifdef _MSC_VER
#define EXPORT extern "C" __declspec(dllexport)
extern "C" int _fltused = 0;
#else
#define EXPORT extern "C"
#endif

// A deliberately simple data cache. DCCI invalidates the containing physical
// line; GetValue fetches the whole line on a miss. Only values read through
// GlobalTensor are checked against the registered tensor's actual bounds.
struct CacheEntry { uint64_t tag; int valid; float value[16]; };
CacheEntry cache[32];
uint64_t calls, reads, fault;
float *begin_q, *end_q, *begin_k, *end_k;
alignas(64) float q_memory[128];
alignas(64) float k_memory[128];

bool allowed(float *ptr) {
    uint64_t a = reinterpret_cast<uint64_t>(ptr);
    return (a >= reinterpret_cast<uint64_t>(begin_q) && a < reinterpret_cast<uint64_t>(end_q)) ||
           (a >= reinterpret_cast<uint64_t>(begin_k) && a < reinterpret_cast<uint64_t>(end_k));
}
void clear_cache() { for (int i=0;i<32;++i) cache[i].valid=0; }
float read_cached(float *ptr) {
    ++reads;
    if (!allowed(ptr)) { fault=1; return -1; }
    uint64_t address=reinterpret_cast<uint64_t>(ptr), tag=address & ~uint64_t(63);
    int slot=-1;
    for (int i=0;i<32;++i) if (cache[i].valid && cache[i].tag==tag) slot=i;
    if (slot<0) {
        for (int i=0;i<32;++i) if (!cache[i].valid) { slot=i; break; }
        if (slot<0) slot=0;
        cache[slot].tag=tag; cache[slot].valid=1;
        // The backing arrays are aligned and include these complete lines.
        // The production helper never rounds down the tensor pointer.
        float *physical=reinterpret_cast<float *>(tag);
        for (int i=0;i<16;++i) cache[slot].value[i]=physical[i];
    }
    return cache[slot].value[(address-tag)/4];
}
namespace AscendC {
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
template<class T> class GlobalTensor {
public:
    T *ptr;
    void SetGlobalBuffer(T *p,uint64_t count) { ptr=p; if(count!=1 || !allowed(p)) fault=2; }
    T GetValue(uint64_t offset) const { if(offset) fault=3; return read_cached(ptr+offset); }
};
template<class T, CacheLine, DcciDst> void DataCacheCleanAndInvalid(const GlobalTensor<T>& tensor) {
    if(!allowed(tensor.ptr)) { fault=4; return; }
    ++calls;
    uint64_t tag=reinterpret_cast<uint64_t>(tensor.ptr) & ~uint64_t(63);
    for(int i=0;i<32;++i) if(cache[i].valid && cache[i].tag==tag) cache[i].valid=0;
}
}
"""


SCENARIOS = r"""
void memory_setup(int offset_q, int offset_k) {
    calls=0; reads=0; fault=0; clear_cache();
    for (int i=0;i<128;++i) { q_memory[i]=100+i; k_memory[i]=200+i; }
    begin_q=q_memory+offset_q; end_q=q_memory+96;
    begin_k=k_memory+offset_k; end_k=k_memory+96;
}
EXPORT int run_model() {
    Context op;
    // Independent absolute addresses at the same logical scale index. Using
    // one tag for both arrays would introduce needless alternating DCCIs.
    memory_setup(0,0);
    op.SetQuantScales((uint8_t*)begin_q,(uint8_t*)begin_k);
    if(op.q(0)!=100 || op.k(0)!=200 || op.q(1)!=101 || op.k(1)!=201) return 10;
    if(calls!=2 || reads!=4 || fault) return 11;

    // Physical boundaries at ptr offsets 1, 15 and 16, rather than index/16.
    for(int which=0;which<3;++which) {
        int offset=which==0?1:(which==1?15:16);
        memory_setup(offset,offset);
        // Prime stale values, then let a hypothetical preceding DMA quantizer
        // update GM without touching the scalar cache.
        for(int i=0;i<32;++i) { read_cached(begin_q+i); read_cached(begin_k+i); }
        for(int i=0;i<32;++i) { begin_q[i]+=1000; begin_k[i]+=2000; }
        if(read_cached(begin_q)==begin_q[0]) return 20; // stale negative control
        calls=0; reads=0;
        op.SetQuantScales((uint8_t*)begin_q,(uint8_t*)begin_k);
        for(int i=0;i<32;++i) if(op.q(i)!=begin_q[i] || op.k(i)!=begin_k[i]) return 21;
        uint64_t expected=offset==16?4:6;
        if(calls!=expected || reads!=64 || fault) return 22;
    }

    // Return to a previously read line must invalidate again, not maintain an
    // unbounded visited-line set. Q/K transitions stay separate.
    memory_setup(1,15);
    op.SetQuantScales((uint8_t*)begin_q,(uint8_t*)begin_k);
    if(op.q(0)!=101 || op.k(0)!=215 || op.q(1)!=102 || op.k(1)!=216) return 30;
    if(calls!=3) return 31;
    if(op.q(14)!=115 || op.q(15)!=116 || calls!=4) return 32;
    if(op.q(0)!=101 || op.k(0)!=215 || calls!=6 || fault) return 33;

    // Same pointer storage is overwritten between native invocations; reset
    // must force fresh reads even if the first line and object are identical.
    begin_q[0]=777; begin_k[0]=888;
    if(read_cached(begin_q)==777 || read_cached(begin_k)==888) return 40;
    op.SetQuantScales((uint8_t*)begin_q,(uint8_t*)begin_k);
    if(op.q(0)!=777 || op.k(0)!=888 || calls!=8 || fault) return 41;
    // Cache eviction requires no special tag reset: immutable GM refills it.
    clear_cache();
    if(op.q(1)!=102 || op.k(0)!=888 || calls!=8 || fault) return 42;
    // New invocation/address is not confused with a matching logical index.
    begin_q=q_memory+16; begin_k=k_memory+16;
    op.SetQuantScales((uint8_t*)begin_q,(uint8_t*)begin_k);
    if(op.q(0)!=116 || op.k(0)!=216 || calls!=10 || fault) return 43;
    return 0;
}
"""


def test_actual_cpp_helper_handles_stale_cache_boundaries_and_invocation_reset(cpp_compiler, tmp_path):
    # Execute the actual new C++ helper, not a Python translation of it. This
    # models cache semantics; CCE address spaces and hardware remain unverified.
    setup = b"    void SetQuantScales(uint8_t *q, uint8_t *k) {\n" + patch.NEW_RESET + b"\n    }\n"
    source = HOST_MODEL + "\nclass Context { public:\n" + setup.decode() + patch.NEW_SCALE_BLOCK.decode() + r"""
    float q(int index) { return ReadQuantScale(q_scales_+index,q_scale_cache_line_); }
    float k(int index) { return ReadQuantScale(k_scales_+index,k_scale_cache_line_); }
};
""" + SCENARIOS
    filename=tmp_path / "scale_cache_model.cpp"
    filename.write_text(source,encoding="utf-8")
    if cpp_compiler.name.lower() in ("cl","cl.exe"):
        output=tmp_path / "scale_cache_model.dll"
        command=[str(cpp_compiler),"/nologo","/std:c++17","/utf-8","/LD","/Od","/GS-","/Zl",str(filename),
                 "/link","/NOENTRY","/NODEFAULTLIB",f"/OUT:{output}"]
    else:
        output=tmp_path / "scale_cache_model.so"
        command=[str(cpp_compiler),"-std=c++17","-shared","-fPIC","-O0",str(filename),"-o",str(output)]
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,timeout=30)
    assert result.returncode==0,(result.stdout+result.stderr).decode(errors="replace")
    library=ctypes.CDLL(str(output))
    library.run_model.argtypes=[]
    library.run_model.restype=ctypes.c_int
    assert library.run_model()==0
