#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void vc_sparse_pair_math_probe(GM_ADDR values,GM_ADDR weights,GM_ADDR initial,GM_ADDR oldOut,GM_ADDR floatOut,GM_ADDR halfOut,uint32_t rows,uint32_t cores){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 TPipe p;TBuf<TPosition::VECCALC> x_,w_,old_,fp_,mixed_,wide_,broadcast_,row_,map_;
 p.InitBuffer(x_,4096);p.InitBuffer(w_,32);p.InitBuffer(old_,8192);p.InitBuffer(fp_,8192);p.InitBuffer(mixed_,8192);p.InitBuffer(wide_,8192);p.InitBuffer(broadcast_,512);p.InitBuffer(row_,64);p.InitBuffer(map_,1024);
 auto x=x_.Get<half>();auto w=w_.Get<half>();auto old=old_.Get<float>();auto fp=fp_.Get<float>();auto mixed=mixed_.Get<float>();auto wide=wide_.Get<float>();auto broadcast=broadcast_.Get<float>();auto row=row_.Get<float>();auto map=map_.Get<uint32_t>();
 GlobalTensor<half>xG,wG;GlobalTensor<float>aG,oG,fG,hG;xG.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(values));wG.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weights));aG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(initial));oG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(oldOut));fG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(floatOut));hG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(halfOut));
 auto m2v=p.FetchEventID(HardEvent::MTE2_V);auto vm3=p.FetchEventID(HardEvent::V_MTE3);
 CreateVecIndex(map.ReinterpretCast<int32_t>(),int32_t(0),256);PipeBarrier<PIPE_V>();ShiftRight(map,map,uint32_t(4),256);PipeBarrier<PIPE_V>();ShiftLeft(map,map,uint32_t(1),256);PipeBarrier<PIPE_V>();
 for(uint32_t tile=GetBlockIdx();tile<rows/16;tile+=cores){
  DataCopy(x,xG[tile*2048],2048);DataCopy(w,wG[tile*16],16);DataCopy(old,aG[tile*2048],2048);DataCopy(fp,aG[tile*2048],2048);DataCopy(mixed,aG[tile*2048],2048);
  SetFlag<HardEvent::MTE2_V>(m2v);WaitFlag<HardEvent::MTE2_V>(m2v);
  Cast(wide,x,RoundMode::CAST_NONE,2048);Cast(row,w,RoundMode::CAST_NONE,16);PipeBarrier<PIPE_V>();Brcb(broadcast,row,2,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
  MulAddDst(fp,wide,broadcast,uint64_t(64),uint8_t(16),BinaryRepeatParams(1,1,0,16,16,1));MulAddDst(fp[64],wide[64],broadcast,uint64_t(64),uint8_t(16),BinaryRepeatParams(1,1,0,16,16,1));PipeBarrier<PIPE_V>();
  Mul(wide,wide,broadcast,uint64_t(64),uint8_t(16),BinaryRepeatParams(1,1,0,16,16,1));Mul(wide[64],wide[64],broadcast,uint64_t(64),uint8_t(16),BinaryRepeatParams(1,1,0,16,16,1));PipeBarrier<PIPE_V>();Add(old,old,wide,2048);PipeBarrier<PIPE_V>();
  Gather(broadcast.ReinterpretCast<half>(),w,map,0,256);PipeBarrier<PIPE_V>();
  MulAddDst(mixed,x,broadcast.ReinterpretCast<half>(),uint64_t(64),uint8_t(16),BinaryRepeatParams(1,1,0,16,8,1));MulAddDst(mixed[64],x[64],broadcast.ReinterpretCast<half>(),uint64_t(64),uint8_t(16),BinaryRepeatParams(1,1,0,16,8,1));PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::V_MTE3>(vm3);WaitFlag<HardEvent::V_MTE3>(vm3);DataCopy(oG[tile*2048],old,2048);DataCopy(fG[tile*2048],fp,2048);DataCopy(hG[tile*2048],mixed,2048);PipeBarrier<PIPE_ALL>();
 }
}
