#include "candidate_selector.h"
extern "C" __global__ __aicore__ void vc_sparse_delay_selector_probe(GM_ADDR input,GM_ADDR geometry,GM_ADDR output,GM_ADDR indices,uint32_t rows,uint32_t valid) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;CandidateSelector select;select.Init(pipe,geometry);
    GlobalTensor<half> in,out;GlobalTensor<int32_t> ids;
    in.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));out.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));ids.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
    const uint32_t core=GetBlockIdx(),padded=(rows+63)/64*64;
    select.Load(in[core*8192],valid,1.f);
    select.Pair(out[core*64],ids[core*64],0,true);select.Pair(out[padded+core*64],ids[padded+core*64],0,false);
    select.Release(pipe);
}
