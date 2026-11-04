#pragma once
// Full float32/uint32 pairs: Sort32 layout checked by the E112 standalone probe.
class OnlineTop8 {
 TBuf<TPosition::VECCALC> scores_,ids_,input_,labels_,sorted_,map_,outMap_,flatMap_,half_;
 TBuf<TPosition::VECCALC> windowScores_,windowIds_,windowMap_;
 TEventID ready_,free_;
public:
 static constexpr uint32_t kExtraUbBytes=79104;
 LocalTensor<float> windowScores;LocalTensor<uint32_t> windowIds,windowMap;
 LocalTensor<float> scores,input,sorted;LocalTensor<uint32_t> ids,labels,map,outMap,flatMap;LocalTensor<half> output;
 __aicore__ inline void Init(TPipe& p){
  p.InitBuffer(scores_,1056*4);p.InitBuffer(ids_,1056*4);p.InitBuffer(input_,2048*4);p.InitBuffer(labels_,2048*4);p.InitBuffer(sorted_,4096*4);
  p.InitBuffer(map_,2048*4);p.InitBuffer(outMap_,512*4);p.InitBuffer(flatMap_,512*4);p.InitBuffer(half_,512*2);
  p.InitBuffer(windowScores_,2048*4);p.InitBuffer(windowIds_,2048*4);p.InitBuffer(windowMap_,2048*4);
  windowScores=windowScores_.Get<float>();windowIds=windowIds_.Get<uint32_t>();windowMap=windowMap_.Get<uint32_t>();
  scores=scores_.Get<float>();ids=ids_.Get<uint32_t>();input=input_.Get<float>();labels=labels_.Get<uint32_t>();sorted=sorted_.Get<float>();
  map=map_.Get<uint32_t>();outMap=outMap_.Get<uint32_t>();flatMap=flatMap_.Get<uint32_t>();output=half_.Get<half>();
  ready_=p.AllocEventID<HardEvent::V_MTE3>();free_=p.AllocEventID<HardEvent::MTE3_V>();
  auto a=labels;auto b=input.ReinterpretCast<uint32_t>();
  CreateVecIndex(map.ReinterpretCast<int32_t>(),0,2048);PipeBarrier<PIPE_V>();
  ShiftRight(a,map,uint32_t(5),2048);PipeBarrier<PIPE_V>();ShiftLeft(b,a,uint32_t(5),2048);PipeBarrier<PIPE_V>();
  Sub(map.ReinterpretCast<int32_t>(),map.ReinterpretCast<int32_t>(),b.ReinterpretCast<int32_t>(),2048);PipeBarrier<PIPE_V>();
  ShiftLeft(map,map,uint32_t(6),2048);PipeBarrier<PIPE_V>();Add(map.ReinterpretCast<int32_t>(),map.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),2048);PipeBarrier<PIPE_V>();
  Mins(map.ReinterpretCast<int32_t>(),map.ReinterpretCast<int32_t>(),int32_t(1024),2048);PipeBarrier<PIPE_V>();ShiftLeft(map,map,uint32_t(2),2048);PipeBarrier<PIPE_V>();
  CreateVecIndex(outMap.ReinterpretCast<int32_t>(),0,512);PipeBarrier<PIPE_V>();
  ShiftRight(a,outMap,uint32_t(6),512);PipeBarrier<PIPE_V>();ShiftLeft(b,a,uint32_t(6),512);PipeBarrier<PIPE_V>();
  Sub(outMap.ReinterpretCast<int32_t>(),outMap.ReinterpretCast<int32_t>(),b.ReinterpretCast<int32_t>(),512);PipeBarrier<PIPE_V>();
  ShiftLeft(outMap,outMap,uint32_t(8),512);ShiftLeft(a,a,uint32_t(3),512);PipeBarrier<PIPE_V>();
  Add(outMap.ReinterpretCast<int32_t>(),outMap.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),512);PipeBarrier<PIPE_V>();
  CreateVecIndex(flatMap.ReinterpretCast<int32_t>(),0,512);PipeBarrier<PIPE_V>();
  ShiftRight(a,flatMap,uint32_t(3),512);PipeBarrier<PIPE_V>();ShiftLeft(b,a,uint32_t(3),512);PipeBarrier<PIPE_V>();
  Sub(flatMap.ReinterpretCast<int32_t>(),flatMap.ReinterpretCast<int32_t>(),b.ReinterpretCast<int32_t>(),512);PipeBarrier<PIPE_V>();
  ShiftLeft(flatMap,flatMap,uint32_t(6),512);PipeBarrier<PIPE_V>();Add(flatMap.ReinterpretCast<int32_t>(),flatMap.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),512);PipeBarrier<PIPE_V>();
  ShiftLeft(flatMap,flatMap,uint32_t(2),512);PipeBarrier<PIPE_V>();
  CreateVecIndex(windowMap.ReinterpretCast<int32_t>(),0,2048);PipeBarrier<PIPE_V>();
  ShiftRight(a,windowMap,uint32_t(5),2048);PipeBarrier<PIPE_V>();ShiftLeft(b,a,uint32_t(5),2048);PipeBarrier<PIPE_V>();
  Sub(windowMap.ReinterpretCast<int32_t>(),windowMap.ReinterpretCast<int32_t>(),b.ReinterpretCast<int32_t>(),2048);PipeBarrier<PIPE_V>();
  ShiftLeft(windowMap,windowMap,uint32_t(6),2048);PipeBarrier<PIPE_V>();
  Add(windowMap.ReinterpretCast<int32_t>(),windowMap.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),2048);PipeBarrier<PIPE_V>();
  ShiftLeft(windowMap,windowMap,uint32_t(2),2048);PipeBarrier<PIPE_V>();
  Duplicate(scores[1024],-__builtin_inff(),32);Duplicate(ids[1024],uint32_t(0),32);PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void ResetWindow(){Duplicate(windowScores,-__builtin_inff(),2048);Duplicate(windowIds,uint32_t(0xffffffff),2048);PipeBarrier<PIPE_V>();}
 __aicore__ inline void Reset(){Duplicate(scores,-__builtin_inff(),512);Duplicate(ids,uint32_t(0),512);PipeBarrier<PIPE_V>();ResetWindow();}
 __aicore__ inline void Append(LocalTensor<half> incoming,LocalTensor<uint32_t> index,uint32_t slot,uint32_t count){
  Cast(windowScores[slot*512],incoming,RoundMode::CAST_NONE,count*64);
  Adds(windowIds[slot*512].ReinterpretCast<int32_t>(),index.ReinterpretCast<int32_t>(),int32_t(0),count*64);PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void FoldWindow(bool first){
  Gather(input,windowScores,windowMap,0,2048);Gather(labels,windowIds,windowMap,0,2048);PipeBarrier<PIPE_V>();
  Sort32(sorted,input,labels,64);PipeBarrier<PIPE_V>();
  const uint32_t start=first?0:512;
  Gather(scores[start],sorted,outMap,0,512);Gather(ids[start],sorted.ReinterpretCast<uint32_t>(),outMap,4,512);PipeBarrier<PIPE_V>();
  if(!first){
   Gather(input,scores,map,0,2048);Gather(labels,ids,map,0,2048);PipeBarrier<PIPE_V>();
   Sort32(sorted,input,labels,64);PipeBarrier<PIPE_V>();
   Gather(scores,sorted,outMap,0,512);Gather(ids,sorted.ReinterpretCast<uint32_t>(),outMap,4,512);PipeBarrier<PIPE_V>();
  }
  ResetWindow();
 }
 __aicore__ inline void Flush(GlobalTensor<half> out,GlobalTensor<int32_t> index,uint64_t head,uint32_t qr,uint32_t qwork){
  Gather(input,scores,flatMap,0,512);PipeBarrier<PIPE_V>();Cast(output,input,RoundMode::CAST_NONE,512);PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::V_MTE3>(ready_);WaitFlag<HardEvent::V_MTE3>(ready_);
  DataCopy(out[(head*qwork+qr)*8],output,512);
  DataCopyPad(index[head*32*qwork+qr],ids.ReinterpretCast<int32_t>(),DataCopyExtParams{8,256,0,qwork*4-256,0});
  SetFlag<HardEvent::MTE3_V>(free_);WaitFlag<HardEvent::MTE3_V>(free_);
  Duplicate(ids.ReinterpretCast<int32_t>(),int32_t(-1),1024);PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::V_MTE3>(ready_);WaitFlag<HardEvent::V_MTE3>(ready_);
  DataCopyPad(index[(head*32+8)*qwork+qr],ids.ReinterpretCast<int32_t>(),DataCopyExtParams{16,256,0,qwork*4-256,0});
  DataCopyPad(index[(head*32+24)*qwork+qr],ids.ReinterpretCast<int32_t>(),DataCopyExtParams{8,256,0,qwork*4-256,0});
  SetFlag<HardEvent::MTE3_V>(free_);WaitFlag<HardEvent::MTE3_V>(free_);
 }
 __aicore__ inline void Release(TPipe& p){PipeBarrier<PIPE_ALL>();p.ReleaseEventID<HardEvent::V_MTE3>(ready_);p.ReleaseEventID<HardEvent::MTE3_V>(free_);}
};
