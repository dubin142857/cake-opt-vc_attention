#pragma once
#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t kFinalizeRows=16,kFinalizeRank=8,kFinalizeColumns=512,kFinalizeUbBytes=94560;
#define VC_FINALIZE_ARGS GM_ADDR value,GM_ADDR candidate,GM_ADDR score,GM_ADDR position,GM_ADDR geometry, \
 GM_ADDR output,GM_ADDR rowIds,GM_ADDR rowWeights,GM_ADDR packetIds,GM_ADDR packetWeights,GM_ADDR status, \
 uint32_t heads,uint32_t qwork,uint32_t columns,uint32_t keys,uint32_t dimension,uint32_t rank,uint32_t cores
#define VC_FINALIZE_PASS value,candidate,score,position,geometry,output,rowIds,rowWeights,packetIds,packetWeights,status,heads,qwork,columns,keys,dimension,rank,cores
class SparseFinalizeVector {
 TBuf<TPosition::VECCALC> candidate_,position_,geometry_,pick_,offset_,ids_,rowIds_,score_,prob_,norm_,rowWeights_,packetWeights_,values_,wide_,acc_,rowWeight_,broadcast_;
public:
 __aicore__ inline void Run(VC_FINALIZE_ARGS) {
  TPipe pipe;
  pipe.InitBuffer(candidate_,513*16*4);pipe.InitBuffer(position_,1056);pipe.InitBuffer(geometry_,1056*4);
  pipe.InitBuffer(pick_,256*4);pipe.InitBuffer(offset_,256*4);pipe.InitBuffer(ids_,256*4);pipe.InitBuffer(rowIds_,256*4);
  pipe.InitBuffer(score_,256*2);pipe.InitBuffer(prob_,256*4);pipe.InitBuffer(norm_,16*4);
  pipe.InitBuffer(rowWeights_,256*2);pipe.InitBuffer(packetWeights_,256*2);pipe.InitBuffer(values_,8*16*128*2);
  pipe.InitBuffer(wide_,16*128*4);pipe.InitBuffer(acc_,16*128*4);pipe.InitBuffer(rowWeight_,16*4);pipe.InitBuffer(broadcast_,16*8*4);
  auto cm=candidate_.Get<int32_t>();auto rawPos=position_.Get<int64_t>();auto geo=geometry_.Get<uint32_t>();
  auto picked=pick_.Get<int32_t>();auto offsets=offset_.Get<int32_t>();auto ids=ids_.Get<int32_t>();auto rid=rowIds_.Get<int32_t>();
  auto selected=score_.Get<half>();auto probs=prob_.Get<float>();auto norm=norm_.Get<float>();
  auto rw=rowWeights_.Get<half>();auto pw=packetWeights_.Get<half>();auto values=values_.Get<half>();
  auto wide=wide_.Get<float>();auto acc=acc_.Get<float>();auto rowWeight=rowWeight_.Get<float>();auto broadcast=broadcast_.Get<float>();
  GlobalTensor<half> vGm,sGm,rwGm,pwGm;GlobalTensor<int32_t> cGm,gGm,riGm,piGm;
  GlobalTensor<int64_t> pGm,statusGm;GlobalTensor<float> outGm;
  vGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(value));cGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(candidate));
  sGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(score));pGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(position));
  gGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(geometry));outGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
  riGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowIds));rwGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(rowWeights));
  piGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(packetIds));pwGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(packetWeights));
  statusGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(status));
  const int vm2=pipe.FetchEventID(HardEvent::V_MTE2),m2v=pipe.FetchEventID(HardEvent::MTE2_V),vs=pipe.FetchEventID(HardEvent::V_S),sv=pipe.FetchEventID(HardEvent::S_V);
  const int vm3=pipe.FetchEventID(HardEvent::V_MTE3),m3v=pipe.FetchEventID(HardEvent::MTE3_V);
  DataCopy(geo.ReinterpretCast<int32_t>(),gGm,1056);SetFlag<HardEvent::MTE2_V>(m2v);WaitFlag<HardEvent::MTE2_V>(m2v);
  const uint32_t core=GetBlockIdx(),groups=heads*qwork/16;uint64_t count[4]={};
  for(uint32_t group=core;group<groups;group+=cores) {
   const uint32_t start=group*16,head=start/qwork,qr=start%qwork;
   ++count[3];count[0]+=16;count[1]+=uint64_t(16)*rank;
   Duplicate(cm[columns*16],int32_t(0),16);Duplicate(rawPos.ReinterpretCast<int32_t>()[256],int32_t(columns),8);
   if(dimension==64)Duplicate(values,half(0),8*16*128);
   PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(vm2);WaitFlag<HardEvent::V_MTE2>(vm2);
   DataCopyPad(cm,cGm[(uint64_t(head)*columns)*qwork+qr],DataCopyExtParams{uint16_t(columns),64,(qwork-16)*4,0,0},DataCopyPadExtParams<int32_t>{false,0,0,0});
   const uint32_t posBlocks=(rank*8+31)/32;
   DataCopyPad(rawPos,pGm[uint64_t(start)*rank],DataCopyExtParams{16,rank*8,0,2-posBlocks,0},DataCopyPadExtParams<int64_t>{false,0,0,0});
   DataCopyPad(selected,sGm[uint64_t(start)*rank],DataCopyExtParams{16,rank*2,0,0,0},DataCopyPadExtParams<half>{true,0,uint8_t(16-rank),half(-__builtin_inff())});
   SetFlag<HardEvent::MTE2_V>(m2v);WaitFlag<HardEvent::MTE2_V>(m2v);
   Gather(picked,rawPos.ReinterpretCast<int32_t>(),geo,uint32_t(0),256);PipeBarrier<PIPE_V>();
   Mins(offsets,picked,int32_t(columns),256);PipeBarrier<PIPE_V>();Maxs(offsets,offsets,int32_t(0),256);PipeBarrier<PIPE_V>();
   Muls(offsets,offsets,int32_t(64),256);PipeBarrier<PIPE_V>();Add(offsets,offsets,geo[256].ReinterpretCast<int32_t>(),256);PipeBarrier<PIPE_V>();
   Gather(ids,cm,offsets.ReinterpretCast<uint32_t>(),uint32_t(0),256);PipeBarrier<PIPE_V>();
   Cast(probs,selected,RoundMode::CAST_NONE,256);PipeBarrier<PIPE_V>();
   Gather(norm,probs,geo[1024],uint32_t(0),16);PipeBarrier<PIPE_V>();
   Brcb(broadcast,norm,2,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
   Sub(probs,probs,broadcast,uint64_t(16),uint8_t(16),BinaryRepeatParams(1,1,0,2,2,1));PipeBarrier<PIPE_V>();
   Exp(probs,probs,256);PipeBarrier<PIPE_V>();WholeReduceSum(norm,probs,16,16,1,1,2);PipeBarrier<PIPE_V>();
   Brcb(broadcast,norm,2,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
   Div(probs,probs,broadcast,uint64_t(16),uint8_t(16),BinaryRepeatParams(1,1,0,2,2,1));PipeBarrier<PIPE_V>();
   Cast(rw,probs,RoundMode::CAST_RINT,256);PipeBarrier<PIPE_V>();
   Gather(pw,rw,geo[768],uint32_t(0),256);PipeBarrier<PIPE_V>();
   SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
   uint32_t bad=0;
   for(uint32_t key=0;key<rank;++key)for(uint32_t row=0;row<16;++row) {
    const int64_t raw=rawPos.GetValue(row*8+key);const int32_t id=ids.GetValue(key*16+row);
    const bool invalid=raw<0||uint64_t(raw)>=columns||id<0||uint32_t(id)>=keys;
    if(invalid){bad|=uint32_t(1)<<row;ids.SetValue(key*16+row,-1);}
    const uint32_t chosen=invalid?0:uint32_t(id);
    DataCopy(values[key*2048+row*128],vGm[uint64_t(chosen)*dimension],dimension);
   }
   SetFlag<HardEvent::S_V>(sv);WaitFlag<HardEvent::S_V>(sv);
   Gather(rid,ids,geo[512],uint32_t(0),256);PipeBarrier<PIPE_V>();
   SetFlag<HardEvent::MTE2_V>(m2v);WaitFlag<HardEvent::MTE2_V>(m2v);
   Duplicate(acc,0.0f,2048);PipeBarrier<PIPE_V>();
   for(uint32_t key=0;key<rank;++key) {
    Cast(wide,values[key*2048],RoundMode::CAST_NONE,2048);Cast(rowWeight,pw[key*16],RoundMode::CAST_NONE,16);PipeBarrier<PIPE_V>();
    Brcb(broadcast,rowWeight,2,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
    Mul(wide,wide,broadcast,uint64_t(64),uint8_t(16),BinaryRepeatParams(1,1,0,16,16,1));
    Mul(wide[64],wide[64],broadcast,uint64_t(64),uint8_t(16),BinaryRepeatParams(1,1,0,16,16,1));PipeBarrier<PIPE_V>();
    Add(acc,acc,wide,2048);PipeBarrier<PIPE_V>();
   }
   if(bad)for(uint32_t row=0;row<16;++row)if(bad&(uint32_t(1)<<row)){++count[2];Duplicate(acc[row*128],0.0f,128);PipeBarrier<PIPE_V>();}
   SetFlag<HardEvent::V_MTE3>(vm3);WaitFlag<HardEvent::V_MTE3>(vm3);
   DataCopy(outGm[uint64_t(start)*dimension],acc,DataCopyParams{16,uint16_t(dimension*4/32),uint16_t((128-dimension)*4/32),0});
   DataCopyPad(riGm[uint64_t(start)*rank],rid,DataCopyExtParams{16,rank*4,1,0,0});
   DataCopyPad(rwGm[uint64_t(start)*rank],rw,DataCopyExtParams{16,rank*2,0,0,0});
   DataCopy(piGm[uint64_t(group)*256],ids,256);DataCopy(pwGm[uint64_t(group)*256],pw,256);
   SetFlag<HardEvent::MTE3_V>(m3v);WaitFlag<HardEvent::MTE3_V>(m3v);
  }
  for(uint32_t i=0;i<64;++i)statusGm.SetValue(uint64_t(core)*64+i,i<4?int64_t(count[i]):0);
  DataCacheCleanAndInvalid<int64_t,CacheLine::ENTIRE_DATA_CACHE>(statusGm);PipeBarrier<PIPE_ALL>();
 }
};
