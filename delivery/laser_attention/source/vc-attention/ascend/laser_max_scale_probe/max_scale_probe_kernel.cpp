#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void vc_max_scale_probe(GM_ADDR input,GM_ADDR mapped,GM_ADDR oldPair,GM_ADDR rawPair,GM_ADDR newMax,uint32_t rows,uint32_t cores,float scale) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 TPipe pipe;TBuf<TPosition::VECCALC> src_,scaled_,old_,raw_,off_,max_;
 pipe.InitBuffer(src_,16384);pipe.InitBuffer(scaled_,16384);pipe.InitBuffer(old_,256);pipe.InitBuffer(raw_,256);pipe.InitBuffer(off_,256);pipe.InitBuffer(max_,128);
 auto src=src_.Get<half>();auto scaled=scaled_.Get<half>();auto old=old_.Get<half>();auto raw=raw_.Get<half>();auto off=off_.Get<uint32_t>();auto maximum=max_.Get<half>();
 GlobalTensor<half> x,y,a,b,c;x.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));y.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(mapped));a.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(oldPair));b.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(rawPair));c.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(newMax));
 auto m2v=pipe.FetchEventID(HardEvent::MTE2_V);auto vm3=pipe.FetchEventID(HardEvent::V_MTE3);
 CreateVecIndex(off.ReinterpretCast<int32_t>(),int32_t(0),64);PipeBarrier<PIPE_V>();ShiftLeft(off,off,uint32_t(2),64);PipeBarrier<PIPE_V>();
 for(uint32_t tile=GetBlockIdx();tile<rows/64;tile+=cores) {
  DataCopy(src,x[tile*8192],8192);SetFlag<HardEvent::MTE2_V>(m2v);WaitFlag<HardEvent::MTE2_V>(m2v);
  Muls(scaled,src,half(scale),8192);PipeBarrier<PIPE_V>();
  WholeReduceMax(old,scaled,128,64,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);WholeReduceMax(raw,src,128,64,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
  Gather(maximum,raw,off,0,64);PipeBarrier<PIPE_V>();Muls(maximum,maximum,half(scale),64);PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::V_MTE3>(vm3);WaitFlag<HardEvent::V_MTE3>(vm3);
  DataCopy(y[tile*8192],scaled,8192);DataCopy(a[tile*128],old,128);DataCopy(b[tile*128],raw,128);DataCopy(c[tile*64],maximum,64);PipeBarrier<PIPE_ALL>();
 }
}
