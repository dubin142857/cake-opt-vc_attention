#pragma once
// These buffers lease producer scratch only AFTER the final FoldWindow.
// Resident scores/IDs survive until both PV and evidence Flush complete.
class InlinePv {
 TBuf<TPosition::VECCALC> values_,packetMap_,transposeMap_,scoreMap_,packetIds_;
 LocalTensor<half> values;LocalTensor<uint32_t> packetMap,transposeMap,scoreMap;LocalTensor<int32_t> pid;
 TEventID ready_,free_;int vm2_,m2v_,vs_,sv_;
public:
 static constexpr uint32_t kExtraUbBytes=36864;
 uint64_t count[5]={};
 __aicore__ inline void Init(TPipe& p,OnlineTop8& t){
  p.InitBuffer(values_,8*16*128*2);p.InitBuffer(packetMap_,256*4);p.InitBuffer(transposeMap_,256*4);p.InitBuffer(scoreMap_,256*4);p.InitBuffer(packetIds_,256*4);
  values=values_.Get<half>();packetMap=packetMap_.Get<uint32_t>();transposeMap=transposeMap_.Get<uint32_t>();scoreMap=scoreMap_.Get<uint32_t>();pid=packetIds_.Get<int32_t>();
  auto a=t.labels;auto b=t.input.ReinterpretCast<uint32_t>();
  CreateVecIndex(packetMap.ReinterpretCast<int32_t>(),0,256);PipeBarrier<PIPE_V>();
  ShiftRight(a,packetMap,uint32_t(4),256);PipeBarrier<PIPE_V>();ShiftLeft(b,a,uint32_t(4),256);PipeBarrier<PIPE_V>();
  Sub(packetMap.ReinterpretCast<int32_t>(),packetMap.ReinterpretCast<int32_t>(),b.ReinterpretCast<int32_t>(),256);PipeBarrier<PIPE_V>();
  Mins(a.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),int32_t(8),256);PipeBarrier<PIPE_V>();ShiftLeft(a,a,uint32_t(6),256);PipeBarrier<PIPE_V>();
  Add(packetMap.ReinterpretCast<int32_t>(),packetMap.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),256);PipeBarrier<PIPE_V>();ShiftLeft(packetMap,packetMap,uint32_t(2),256);PipeBarrier<PIPE_V>();
  CreateVecIndex(transposeMap.ReinterpretCast<int32_t>(),0,256);PipeBarrier<PIPE_V>();
  ShiftRight(a,transposeMap,uint32_t(4),256);PipeBarrier<PIPE_V>();ShiftLeft(b,a,uint32_t(4),256);PipeBarrier<PIPE_V>();
  Sub(transposeMap.ReinterpretCast<int32_t>(),transposeMap.ReinterpretCast<int32_t>(),b.ReinterpretCast<int32_t>(),256);PipeBarrier<PIPE_V>();
  ShiftLeft(transposeMap,transposeMap,uint32_t(4),256);PipeBarrier<PIPE_V>();Add(transposeMap.ReinterpretCast<int32_t>(),transposeMap.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),256);PipeBarrier<PIPE_V>();
  ShiftLeft(transposeMap,transposeMap,uint32_t(2),256);PipeBarrier<PIPE_V>();Gather(scoreMap,packetMap,transposeMap,0,256);PipeBarrier<PIPE_V>();
  vm2_=p.FetchEventID(HardEvent::V_MTE2);m2v_=p.FetchEventID(HardEvent::MTE2_V);vs_=p.FetchEventID(HardEvent::V_S);sv_=p.FetchEventID(HardEvent::S_V);
  ready_=p.AllocEventID<HardEvent::V_MTE3>();free_=p.AllocEventID<HardEvent::MTE3_V>();
 }
 __aicore__ inline void Run(OnlineTop8& t,VC_CANDIDATE_ARGS,uint32_t currentHead,uint32_t qr,uint32_t qwork){
  GlobalTensor<half> vGm,wGm,pwGm;GlobalTensor<int32_t> iGm,piGm;GlobalTensor<float> outGm;
  vGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v));wGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weights));pwGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(packetWeights));
  iGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(selectedIds));piGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(packetIds));outGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
  auto probs=t.input;auto norm=t.input[256];auto broadcast=t.input[272];auto rowWeight=t.input[400];auto rid=t.labels.ReinterpretCast<int32_t>();
  auto rw=t.output;auto pw=t.output[256];auto acc=t.sorted;auto wide=t.sorted[2048];
  // Column8 is a safe sentinel for padded softmax columns and packet IDs.
  Duplicate(t.scores[512],-__builtin_inff(),64);Duplicate(t.ids[rank*64].ReinterpretCast<int32_t>(),int32_t(-1),(9-rank)*64);PipeBarrier<PIPE_V>();
  for(uint32_t local=0;local<64;local+=16){
   const uint64_t start=uint64_t(currentHead)*qwork+qr+local,group=start/16;count[0]+=16;count[1]+=16*rank;++count[3];
   Gather(probs,t.scores,scoreMap,local*4,256);Gather(pid,t.ids.ReinterpretCast<int32_t>(),packetMap,local*4,256);PipeBarrier<PIPE_V>();
   Gather(norm,probs,transposeMap,0,16);PipeBarrier<PIPE_V>();
   Brcb(broadcast,norm,2,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
   Sub(probs,probs,broadcast,uint64_t(16),uint8_t(16),BinaryRepeatParams(1,1,0,2,2,1));PipeBarrier<PIPE_V>();
   Exp(probs,probs,256);PipeBarrier<PIPE_V>();WholeReduceSum(norm,probs,16,16,1,1,2);PipeBarrier<PIPE_V>();
   Brcb(broadcast,norm,2,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
   Div(probs,probs,broadcast,uint64_t(16),uint8_t(16),BinaryRepeatParams(1,1,0,2,2,1));PipeBarrier<PIPE_V>();
   Cast(rw,probs,RoundMode::CAST_RINT,256);PipeBarrier<PIPE_V>();
   // Transpose offsets are in bytes for int32; halve them for half weights.
   ShiftRight(t.input[512].ReinterpretCast<uint32_t>(),transposeMap,uint32_t(1),256);PipeBarrier<PIPE_V>();
   Gather(pw,rw,t.input[512].ReinterpretCast<uint32_t>(),0,256);PipeBarrier<PIPE_V>();
   SetFlag<HardEvent::V_S>(vs_);WaitFlag<HardEvent::V_S>(vs_);SetFlag<HardEvent::V_MTE2>(vm2_);WaitFlag<HardEvent::V_MTE2>(vm2_);
   uint32_t bad=0;
   for(uint32_t key=0;key<rank;++key)for(uint32_t row=0;row<16;++row){
    const int32_t id=pid.GetValue(key*16+row);const bool invalid=id<0||uint64_t(id)>=heads*kpad||uint32_t(id)/kpad!=currentHead||uint32_t(id)%kpad>=keys;
    if(invalid){bad|=uint32_t(1)<<row;pid.SetValue(key*16+row,-1);}
    DataCopy(values[key*2048+row*128],vGm[uint64_t(invalid?0:uint32_t(id))*128],128);
   }
   SetFlag<HardEvent::S_V>(sv_);WaitFlag<HardEvent::S_V>(sv_);
   Gather(rid,pid,transposeMap,0,256);PipeBarrier<PIPE_V>();
   SetFlag<HardEvent::MTE2_V>(m2v_);WaitFlag<HardEvent::MTE2_V>(m2v_);
   Duplicate(acc,0.0f,2048);PipeBarrier<PIPE_V>();
   for(uint32_t key=0;key<rank;++key){
    Cast(wide,values[key*2048],RoundMode::CAST_NONE,2048);Cast(rowWeight,pw[key*16],RoundMode::CAST_NONE,16);PipeBarrier<PIPE_V>();
    Brcb(broadcast,rowWeight,2,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
    Mul(wide,wide,broadcast,uint64_t(64),uint8_t(16),BinaryRepeatParams(1,1,0,16,16,1));
    Mul(wide[64],wide[64],broadcast,uint64_t(64),uint8_t(16),BinaryRepeatParams(1,1,0,16,16,1));PipeBarrier<PIPE_V>();Add(acc,acc,wide,2048);PipeBarrier<PIPE_V>();
   }
   if(bad)for(uint32_t row=0;row<16;++row)if(bad&(uint32_t(1)<<row)){++count[2];Duplicate(acc[row*128],0.0f,128);PipeBarrier<PIPE_V>();}
   SetFlag<HardEvent::V_MTE3>(ready_);WaitFlag<HardEvent::V_MTE3>(ready_);
   DataCopy(outGm[start*128],acc,2048);
   DataCopyPad(iGm[start*rank],rid,DataCopyExtParams{16,rank*4,1,0,0});DataCopyPad(wGm[start*rank],rw,DataCopyExtParams{16,rank*2,0,0,0});
   DataCopy(piGm[group*256],pid,256);DataCopy(pwGm[group*256],pw,256);count[4]+=5;
   SetFlag<HardEvent::MTE3_V>(free_);WaitFlag<HardEvent::MTE3_V>(free_);
  }
 }
 __aicore__ inline void Release(TPipe& p){PipeBarrier<PIPE_ALL>();p.ReleaseEventID<HardEvent::V_MTE3>(ready_);p.ReleaseEventID<HardEvent::MTE3_V>(free_);}
};
