"""Run actual original/candidate row-max methods with an independent ISA model.

This is host validation of indexing, values, masking and dependency barriers;
it does not validate target CCE lowering or NPU instruction throughput.
"""
import pytest

from scripts import mindie_laser_quant_vscale_patch as baseline
from scripts import mindie_laser_quant_vscale_rowmax_patch as patch
from tests import test_mindie_laser_quant_vscale_patch as old
from tests.test_mindie_laser_expcast_cpp import cpp_compiler
from tests.test_mindie_laser_quant_vscale_patch import source_files


def test_only_the_actual_max_callsite_changes(source_files):
    original = baseline.patch_vector(source_files["VectorForward.h"])
    candidate = patch.patch_vector(original)
    begin = original.index(patch.START)
    end = original.index(patch.END, begin)
    new_begin = candidate.index(patch.CALL)
    new_end = candidate.index(patch.END, new_begin)
    reconstructed = candidate[:new_begin] + original[begin:end] + candidate[new_end:]
    assert reconstructed.replace(patch.METHOD, b"", 1) == original
    # Keep the existing max fallback literally, and all row sums untouched.
    for signature in ("::process_cacl_max(", "::cacl_max(", "::process_cacl_wrap(", "::cacl_wrap("):
        assert old.function(original.decode(), signature) == old.function(candidate.decode(), signature)
    assert patch.apply("CubeForward.h", source_files["CubeForward.h"]) == source_files["CubeForward.h"]


@pytest.mark.parametrize("mutation", [
    lambda x: patch.patch_vector(x),
    lambda x: x.replace(b"\n            rowPerTime = 4;", b"\n            rowPerTime = 8;"),
    lambda x: x.replace(patch.FINAL, b"changed_max();"),
    lambda x: x + b"\n// vscale_bulk_compact",
])
def test_source_drift_and_wrong_parent_fail_closed(source_files, mutation):
    data = mutation(baseline.patch_vector(source_files["VectorForward.h"]))
    with pytest.raises(ValueError):
        patch.patch_vector(data)


PREAMBLE = old.PREAMBLE[:old.PREAMBLE.index("enum {PIPE_V,PIPE_FIX};")] + r"""
enum {PIPE_V,PIPE_FIX}; enum Order_t {ORDER_ONLY_VALUE=2};
constexpr int BASE_BLOCK_SIDE_LEN=128,BLOCK_NUM_8=8;
constexpr int BLOCK_NUM_FOR_VMAX=16,MAX_BLOCK_PER_ONE_PROC=128;
constexpr float PADDING_FOR_MAX2=-65504.0f;
int vector_mask=128,vcmax_calls=0,vmax_calls=0;
struct Pending { half* ptr; half value; } pending[32768];
int pending_count=0;
void queue(half* ptr,half value) {
    if(pending_count>=32768) { fail(31);return; }
    pending[pending_count].ptr=ptr;pending[pending_count].value=value;++pending_count;
}
half read(half* ptr) {
    for(int i=0;i<pending_count;++i) if(pending[i].ptr==ptr) { fail(32);break; }
    return *ptr;
}
half max_value(half a,half b) {
    return float(a)>=float(b)?a:b;
}
void pipe_barrier(int p) {
    if(p!=PIPE_V) fail(33);
    for(int i=0;i<pending_count;++i) *pending[i].ptr=pending[i].value;
    pending_count=0;
}
void set_vector_mask(uint64_t hi,uint64_t lo) {
    if(hi!=~uint64_t(0)||lo!=~uint64_t(0)) fail(34);
    vector_mask=128;
}
void vcmax(half* dst,half* src,int repeats,int ds,int sb,int sr,Order_t order) {
    ++vcmax_calls;
    if((uintptr_t(src)%32)!=0||(uintptr_t(dst)%2)!=0||int(order)!=2||
       repeats<1||repeats>255||vector_mask<1||vector_mask>128) fail(35);
    // Source strides are 32-byte blocks; ONLY_VALUE destination stride is 2B.
    for(int r=0;r<repeats;++r) {
        half best=read(src+r*sr*16);
        for(int lane=1;lane<vector_mask;++lane)
            best=max_value(best,read(src+r*sr*16+(lane/16)*sb*16+lane%16));
        queue(dst+r*ds,best);
    }
}
void vmax(half* dst,half* a,half* b,int repeats,int db,int ab,int bb,int dr,int ar,int br) {
    ++vmax_calls;
    // Snapshot all reads before queued writes: same-instruction alias is allowed.
    for(int r=0;r<repeats;++r) for(int lane=0;lane<vector_mask;++lane) {
        int block=lane/16,element=lane%16;
        queue(dst+r*dr*16+block*db*16+element,
            max_value(read(a+r*ar*16+block*ab*16+element),
                      read(b+r*br*16+block*bb*16+element)));
    }
}
void vector_dup(half* dst,half value,int repeats,int db,int sb,int dr,int sr) {
    for(int r=0;r<repeats;++r) for(int lane=0;lane<vector_mask;++lane)
        queue(dst+r*dr*16+(lane/16)*db*16+lane%16,value);
}
void copy_ubuf_to_ubuf(half* dst,half* src,int sid,int burst,int len,int ss,int ds) {
    for(int b=0;b<burst;++b) for(int i=0;i<len*16;++i)
        queue(dst+b*(len+ds)*16+i,read(src+b*(len+ss)*16+i));
}
"""


SCENARIOS = r"""
alignas(32) half scores[16384+16], original_scores[16384];
alignas(32) half scratch[16384+16], final_buf[512+16], maxima[16], expected[16];
void fill_case(int blocks,int scenario) {
    for(int i=0;i<16384;++i) scores[i]=half(65504.0f);
    for(int k=0;k<blocks;++k) for(int row=0;row<4;++row) for(int lane=0;lane<128;++lane) {
        half value;
        if(scenario==0) value=half(-float((k*29+row*17+lane*7)%10000+1));
        else if(scenario==1) {
            value=half(-float(k+row+lane+100));
            if(k==(blocks-1-row%blocks) && lane==(127-row*31)) value=half(float(row+1));
        } else if(scenario==2) value=half(lane%17==0?7.0f:-55.0f);
        else if(scenario==3) { value.bits=((k+row+lane)%2)?0x8000:0; }
        else if(scenario==4) value=half((k+lane)%101==0?65504.0f:-65504.0f);
        else if(scenario==5) value.bits=0xfc00; // Padding must dominate all -infinity.
        else if(scenario==6) { value.bits=(unsigned short)(0x8001+(k*127+lane+row)%511); }
        else {
            value=half(-100.0f-float((k+row+lane)%23));
            // Original tail masking uses -60000; simulate its completed output.
            if(k==blocks-1 && lane>=17+row) value=half(-60000.0f);
        }
        scores[k*512+row*128+lane]=value;
    }
    for(int i=0;i<16384;++i) original_scores[i]=scores[i];
}
void poison_work() {
    for(int i=0;i<16384+16;++i) scratch[i].bits=0x7bff;
    for(int i=0;i<512+16;++i) final_buf[i].bits=0x3555;
    for(int i=0;i<16;++i) maxima[i].bits=0x3555;
    scores[16384].bits=0x3555;
}
EXPORT int run_model() {
    fault=0;pending_count=0;
    VectorModel<half> op;
    for(int blocks=1;blocks<=32;++blocks) for(int scenario=0;scenario<8;++scenario) {
        int pad=0;op.get_padding_info_for_row_max(blocks,&pad);
        fill_case(blocks,scenario);poison_work();
        half* pp[2]={scores,scores};
        op.padding_for_row_max_or_rowsum2(blocks,pad,0,0,pp);
        Params p{blocks,pad};
        vector_mask=128;
        op.old_reduce(p,scores,scratch,final_buf,maxima);
        for(int row=0;row<4;++row) expected[row]=maxima[row];
        if(fault) return fault;
        for(int i=0;i<16384;++i) scores[i]=original_scores[i];
        poison_work();op.padding_for_row_max_or_rowsum2(blocks,pad,0,0,pp);
        vector_mask=128;int before=vcmax_calls;
        op.new_reduce(p,scores,scratch,final_buf,maxima,0,false);
        if(fault) return fault;
        if(vector_mask!=128||pending_count!=0) return 41;
        if(vcmax_calls-before!=(blocks>=8?5:1)) return 42;
        for(int row=0;row<4;++row) {
            // Maxima signed-zero tie choices may differ; floating values must not.
            if(maxima[row].bits!=expected[row].bits &&
               !(float(maxima[row])==0.0f && float(expected[row])==0.0f)) return 43;
            half dense=half(-65504.0f);
            if(!pad) dense.bits=0xfc00;
            for(int k=0;k<blocks;++k) for(int lane=0;lane<128;++lane)
                dense=max_value(dense,original_scores[k*512+row*128+lane]);
            if(float(maxima[row])!=float(dense)) return 44;
        }
        if(maxima[4].bits!=0x3555||scores[16384].bits!=0x3555||
           scratch[16384].bits!=0x7bff||final_buf[512].bits!=0x3555) return 45;
        if(blocks>=8) {
            if(final_buf[0].bits!=0x3555||scratch[512].bits!=0x7bff) return 46;
            for(int row=0;row<4;++row) for(int k=blocks+pad;k<128;++k)
                if(scratch[row*128+k].bits!=0x7bff) return 47;
        }
    }
    // Unsupported shapes/features must do no work and leave mask and buffers alone.
    for(int mode=0;mode<7;++mode) {
        op.rowPerTime=mode==0?2:4;
        int blocks=mode==1?7:(mode==2?33:16);
        int pad=mode==3?-1:(mode==4?17:0);
        int before=vcmax_calls;vector_mask=128;
        if(op.TryDenseTileRowMax(maxima,scores,scratch,blocks,pad,mode==5,mode==6)) return 48;
        if(vcmax_calls!=before||vector_mask!=128||pending_count) return 49;
    }
    return fault;
}
"""


def model(raw):
    original = baseline.patch_vector(raw).decode()
    candidate = patch.patch_vector(original.encode()).decode()
    signatures = ("get_padding_info_for_row_max", "padding_for_row_max_or_rowsum2",
                  "cacl_max", "process_cacl_max")
    methods = "\n".join(old.function(original, "::" + name + "(").replace(
        "::" + name + "(", "void " + name + "(", 1) for name in signatures)
    begin = original.index(patch.START.decode())
    end = original.index(patch.END.decode(), begin)
    old_site = original[begin:end]
    begin = candidate.index(patch.CALL.decode())
    end = candidate.index(patch.END.decode(), begin)
    new_site = candidate[begin:end]
    return PREAMBLE + """
struct Params {int section_block_num,section_padding_block_num;};
template<class INPUT_T> struct VectorModel {
    int rowPerTime=4,data_num_per_blk=16;
    void __set_mask(int n) { vector_mask=n; }
""" + methods + patch.METHOD.decode() + """
    void old_reduce(Params param,half* cur_buf_for_attn_score_fp16,
        half*cur_buf_for_rowmax,half*cur_buf_for_final_rowmax,half*maxJVectorUbAddr) {
""" + old_site + """
    }
    void new_reduce(Params param,half*cur_buf_for_attn_score_fp16,
        half*cur_buf_for_rowmax,half*cur_buf_for_final_rowmax,half*maxJVectorUbAddr,
        int qk_triangle,bool sparse_flag) {
""" + new_site + "\n    }\n};\n" + SCENARIOS


def test_actual_old_and_new_maxima_padding_masks_and_scratch_lifecycle(
        cpp_compiler, source_files, tmp_path):
    status = old.compile_library(cpp_compiler, tmp_path, model(source_files["VectorForward.h"])).run_model()
    assert status == 0, status


@pytest.mark.parametrize("safe,unsafe", [
    ("reduced_blocks, 1, 1, 32, (Order_t)0b10", "reduced_blocks, 1, 1, 8, (Order_t)0b10"),
    ("        __set_mask(reduced_blocks);", "        __set_mask(128);"),
    ("        pipe_barrier(PIPE_V);\n        // Rows are spaced", "        // Rows are spaced"),
])
def test_wrong_row_stride_mask_or_missing_producer_barrier_is_detected(
        cpp_compiler, source_files, tmp_path, safe, unsafe):
    code = model(source_files["VectorForward.h"])
    assert code.count(safe) == 1
    status = old.compile_library(cpp_compiler, tmp_path, code.replace(safe, unsafe)).run_model()
    assert status != 0, status
