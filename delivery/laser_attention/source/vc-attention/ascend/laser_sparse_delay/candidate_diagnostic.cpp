#include "candidate_selector.h"
extern "C" __global__ __aicore__ void vc_sparse_delay_diagnostic(GM_ADDR input,GM_ADDR geometry,GM_ADDR output,uint32_t valid) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;CandidateSelector select;select.Init(pipe,geometry);
    GlobalTensor<half> in,out;GlobalTensor<int32_t> ids;
    in.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));out.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
    ids.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));
    select.Load(in,valid,1.f);
    const int vm3=pipe.FetchEventID(HardEvent::V_MTE3),m3v=pipe.FetchEventID(HardEvent::MTE3_V);
    SetFlag<HardEvent::V_MTE3>(vm3);WaitFlag<HardEvent::V_MTE3>(vm3);
    DataCopy(out,select.data,8192);DataCopy(out[8192],select.position,8192);DataCopy(out[2*8192],select.mask.ReinterpretCast<half>(),512);
    DataCopy(out[4*8192],select.pairOffset.ReinterpretCast<half>(),128);
    SetFlag<HardEvent::MTE3_V>(m3v);WaitFlag<HardEvent::MTE3_V>(m3v);
    WholeReduceMax(select.pair,select.data,128,64,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(vm3);WaitFlag<HardEvent::V_MTE3>(vm3);DataCopy(out[3*8192],select.pair,128);
    SetFlag<HardEvent::MTE3_V>(m3v);WaitFlag<HardEvent::MTE3_V>(m3v);
    select.Pair(out[5*8192],ids[6*8192/2],0,true);
    SetFlag<HardEvent::V_MTE3>(vm3);WaitFlag<HardEvent::V_MTE3>(vm3);
    DataCopy(out[7*8192],select.data,8192);DataCopy(out[8*8192],select.broadcast,8192);DataCopy(out[9*8192],select.mask.ReinterpretCast<half>(),512);
    SetFlag<HardEvent::MTE3_V>(m3v);WaitFlag<HardEvent::MTE3_V>(m3v);
    select.Release(pipe);
}
