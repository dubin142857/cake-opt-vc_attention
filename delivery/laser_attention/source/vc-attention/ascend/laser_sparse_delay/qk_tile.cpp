#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void vc_delay_qk_tile(GM_ADDR q,GM_ADDR k,GM_ADDR output) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
 TPipe pipe;TBuf<TPosition::A1>a1_;TBuf<TPosition::B1>b1_;TBuf<TPosition::A2>a2_;TBuf<TPosition::B2>b2_;TBuf<TPosition::CO1>c_;
 pipe.InitBuffer(a1_,32768);pipe.InitBuffer(b1_,32768);pipe.InitBuffer(a2_,32768);pipe.InitBuffer(b2_,32768);pipe.InitBuffer(c_,65536);
 auto a1=a1_.Get<half>();auto b1=b1_.Get<half>();auto a2=a2_.Get<half>();auto b2=b2_.Get<half>();auto acc=c_.Get<float>();
 GlobalTensor<half>qGm,kGm,out;qGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q));kGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k));out.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
 const auto ndQ=Nd2NzParams(1,128,128,0,128,128,1,0);const auto ndK=Nd2NzParams(1,128,128,0,128,128,1,0);
 const auto loadA=LoadData2DParams(0,8,8,0,0,false,0);const auto loadK=LoadData2DParams(0,64,1,0,0,false,0);
 DataCopy(a1,qGm,ndQ);DataCopy(b1,kGm,ndK);PipeBarrier<PIPE_ALL>();
 for(uint32_t m=0;m<8;++m)LoadData(a2[m*128*16],a1[m*16*16],loadA);
 LoadData(b2,b1,loadK);PipeBarrier<PIPE_ALL>();
 Mmad(acc,a2,b2,MmadParams(128,128,128,uint8_t(0),false,true));PipeBarrier<PIPE_ALL>();
 FixpipeParamsV220 copy(128,128,128,128,false);copy.quantPre=QuantMode_t::F322F16;Fixpipe(out,acc,copy);PipeBarrier<PIPE_ALL>();
}
