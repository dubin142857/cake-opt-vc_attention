#include "candidate_selector.h"
extern "C" __global__ __aicore__ void vc_sparse_delay_reduce_probe(GM_ADDR input,GM_ADDR geometry,GM_ADDR output,GM_ADDR indices,uint32_t valid0,uint32_t valid1,uint32_t base,float scale){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 TPipe pipe;CandidateSelector s;s.Init(pipe,geometry);
 GlobalTensor<half> in,out;GlobalTensor<int32_t> ids;
 in.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));out.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));ids.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
 s.ReadScorePair(in,in[8192]);s.TransformPair(valid0,valid1,scale);s.AccumulatePair(0,base);s.Flush(out,ids,2,64);s.Release(pipe);
}
