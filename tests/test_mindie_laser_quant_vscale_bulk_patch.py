"""Execute the generated broadcast method with independent strides and canaries.

Host simulation validates values, bounds and our synchronization protocol, not
CCE lowering or actual asynchronous NPU scheduling.
"""
import pytest

from scripts import mindie_laser_quant_vscale_bulk_patch as patch
from scripts import mindie_laser_quant_vscale_patch as baseline
from tests import test_mindie_laser_quant_vscale_patch as prior
from tests.test_mindie_laser_expcast_cpp import cpp_compiler
from tests.test_mindie_laser_quant_vscale_patch import source_files


def test_candidate_only_changes_broadcast_and_disjoint_ub(source_files):
    original = baseline.patch_vector(source_files["VectorForward.h"])
    new = patch.patch_vector(original)
    assert prior.function(new.decode(), "    __aicore__ inline void ApplyQuantScoreScale(") == \
        prior.function(original.decode(), "    __aicore__ inline void ApplyQuantScoreScale(")
    # Cube, mapping, all existing source handoffs and score consumption stay
    # identical: only the old preparation span and allocation/field are touched.
    assert patch.apply("CubeForward.h", source_files["CubeForward.h"]) == source_files["CubeForward.h"]
    assert new.count(b"vbrcb(") == original.count(b"vbrcb(") + 1
    assert b"set_flag(PIPE_V, PIPE_S, EVENT_ID0)" in patch.NEW_PREPARE
    assert b"set_flag(PIPE_S, PIPE_V, EVENT_ID0)" in patch.NEW_PREPARE
    assert patch.CONTRACT["extra_ub_bytes"] == 64
    assert patch.CONTRACT["ub_end_bytes"] == 179136


@pytest.mark.parametrize("mutation", [
    lambda data: data,
    lambda data: data.replace(b"            rowPerTime = 4;", b"            rowPerTime = 8;"),
    lambda data: data.replace(b"static_cast<half>(packed_gain)", b"packed_gain"),
])
def test_wrong_or_duplicate_baseline_fails_closed(source_files, mutation):
    original = baseline.patch_vector(source_files["VectorForward.h"])
    mutated = mutation(original)
    if mutated == original:
        mutated = patch.patch_vector(original)
    with pytest.raises(ValueError):
        patch.patch_vector(mutated)


BRCP_MODEL = r"""
typedef unsigned short uint16_t;
enum {PIPE_S=2,EVENT_ID0=0};
int event_pending[3][3]={{0}};
int protocol=0,brcb_calls=0,dup_calls=0;
void set_flag(int from,int to,int event) {
    if(event!=EVENT_ID0||event_pending[from][to]) { fail(61);return; }
    if(from==PIPE_V && to==PIPE_S) {
        if(protocol!=0 && protocol!=3) fail(62);
        protocol=1;
    } else if(from==PIPE_S && to==PIPE_V) {
        if(protocol!=2) fail(63);
        protocol=4;
    } else fail(64);
    event_pending[from][to]=1;
}
void wait_flag(int from,int to,int event) {
    if(event!=EVENT_ID0||!event_pending[from][to]) { fail(65);return; }
    event_pending[from][to]=0;
    if(from==PIPE_V && to==PIPE_S) protocol=2;
    else if(from==PIPE_S && to==PIPE_V) protocol=3;
    else fail(66);
}
void vbrcb(uint16_t*dst,uint16_t*src,int block_stride,int repeat_stride,int repeats) {
    ++brcb_calls;
    if(protocol!=3||vector_mask!=128) fail(67);
    if(block_stride!=4||repeat_stride!=32||repeats<1||repeats>4) fail(68);
    // Brcb's eight input elements map to eight 32B output blocks per repeat.
    for(int r=0;r<repeats;++r) for(int b=0;b<8;++b) for(int e=0;e<16;++e)
        dst[r*repeat_stride*16+b*block_stride*16+e]=src[r*8+b];
}
"""


def model(raw):
    text = patch.patch_vector(baseline.patch_vector(raw)).decode()
    structure = prior.function(text, "    struct UB_FOR_SHORT_LEN_ATTN_SCORE") + ";"
    allocate = prior.function(text, "::allocate_ubuf_for_short_seq_attn_score(")
    allocate = allocate.replace("::allocate_", "void allocate_", 1)
    methods = "\n".join(prior.function(text, signature) for signature in (
        "    __aicore__ inline void PrepareQuantScaleSectionLegacy(",
        "    __aicore__ inline void PrepareQuantScaleSection(",
        "    __aicore__ inline void ApplyQuantScoreScale(",
    ))
    preamble = prior.PREAMBLE
    scenarios = prior.VECTOR_SCENARIOS.replace(
        "float original_scores[16384];",
        "alignas(32) half compact_storage[32+16];\nfloat original_scores[16384];")
    scenarios = scenarios.replace(
        "(uintptr_t)ub.vscale_coeff_fp32!=178944)",
        "(uintptr_t)ub.vscale_coeff_fp32!=178944 || "
        "(uintptr_t)ub.vscale_bulk_compact_fp16!=179072)")
    scenarios = scenarios.replace(
        "ub.vscale_coeff_fp32=coefficient_storage;",
        "ub.vscale_coeff_fp32=coefficient_storage;\n"
        "    ub.vscale_bulk_compact_fp16=compact_storage;")
    scenarios = scenarios.replace(
        "op.PrepareQuantScaleSection(17,9,count,ub);",
        """int old_brcb=brcb_calls;
            compact_storage[32].bits=0x3555;
            gain_storage[2048].bits=0x3555;
            op.PrepareQuantScaleSection(17,9,count,ub);
            if(brcb_calls-old_brcb!=(count>=8?4:0)) return 71;
            if(compact_storage[32].bits!=0x3555||gain_storage[2048].bits!=0x3555) return 72;
            if(count>=8) {
                for(int i=count;i<(count+7)/8*8;++i) {
                    if(compact_storage[i].bits!=0) return 73;
                    for(int row=0;row<64;++row)
                        if(gain_storage[i*64+row].bits!=0) return 74;
                }
            }""")
    # Each K start offset crosses a distinct scalar-cache-line alignment. The
    # expected coefficients are independently derived from the accessed index.
    scenarios = scenarios.replace(
        "for(int count=1;count<=32;++count) {",
        "for(int start=0;start<16;++start) for(int count=1;count<=32;++count) {")
    scenarios = scenarios.replace("PrepareQuantScaleSection(17,9,count,ub)",
                                  "PrepareQuantScaleSection(17,start,count,ub)")
    scenarios = scenarios.replace("k_values[9+j]", "k_values[start+j]")
    return preamble + BRCP_MODEL + """
template<class INPUT_T,bool IF_BF16,class WORKSPACE_T> struct VectorForward {
    int32_t rowPerTime=4, SIZE_FP16=2, SIZE_FP32=4;
    float SCALE=1.0f;
    uint32_t vscale_slow_mask_=0;
    void __set_mask(int n) { vector_mask=n; }
""" + structure + baseline.VECTOR_SCALE_SETUP.decode() + \
        baseline.NEW_SCALE_BLOCK.decode() + methods + allocate + "\n};\n" + scenarios


def test_actual_broadcast_values_fallback_boundaries_caches_and_canaries(
        cpp_compiler, source_files, tmp_path):
    code = model(source_files["VectorForward.h"])
    result = prior.compile_library(cpp_compiler, tmp_path, code).run_model()
    assert result == 0, result


@pytest.mark.parametrize("remove", [
    "        set_flag(PIPE_V, PIPE_S, EVENT_ID0);\n"
    "        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);",
    "        set_flag(PIPE_S, PIPE_V, EVENT_ID0);\n"
    "        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);",
])
def test_missing_scalar_vector_handoff_is_detected(
        cpp_compiler, source_files, tmp_path, remove):
    code = model(source_files["VectorForward.h"])
    assert code.count(remove) == 1
    result = prior.compile_library(cpp_compiler, tmp_path, code.replace(remove, "")).run_model()
    assert 61 <= result <= 68, result
