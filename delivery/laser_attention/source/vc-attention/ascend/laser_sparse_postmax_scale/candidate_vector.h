#pragma once
#include "candidate_selector.h"
#include "online_top8.h"
#include "inline_pv.h"
class CandidateVector {
public:
 __aicore__ inline void Run(VC_CANDIDATE_ARGS){
  TPipe pipe;CandidateSelector select;select.Init(pipe,geometry);OnlineTop8 top;top.Init(pipe);InlinePv pv;pv.Init(pipe,top);
  GlobalTensor<half> scoreGm,outputGm;GlobalTensor<int32_t> indexGm;GlobalTensor<uint64_t> counterGm,pvCounterGm;
  scoreGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(score));outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(selectedScores));
  indexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(candidates));counterGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(counters));pvCounterGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(pvCounters));
  const uint32_t sub=GetSubBlockIdx(),core=GetBlockIdx()/GetSubBlockNum(),tiles=(queries+127)/128,qwork=tiles*128,blocks=kpad/128;
  uint64_t count[10]={};
  for(uint64_t task=core;task<heads*tiles;task+=cores){
   const uint64_t head=task/tiles;const uint32_t qr=task%tiles*128+sub*64;++count[0];top.Reset();
   for(uint32_t kb=0;kb<blocks;++kb){
    const uint32_t group=kb/8,slot=group&1,column=kb%8;if(column==0)CrossCoreWaitFlag(slot);
    const uint32_t valid=kb*128>=keys?0:(keys-kb*128<128?keys-kb*128:128);
    select.ReadScore(scoreGm[((uint64_t(core)*2+slot)*8+column)*128*128+sub*8192]);
    if(column==7||kb+1==blocks){CrossCoreSetFlag<2,PIPE_MTE2>(2+slot);++count[2];}
    select.Transform(valid,scale);select.Accumulate(column,head*kpad+kb*128);count[9]+=64;
    if(column==7||kb+1==blocks)top.Append(select.value,select.globalIndex,(kb%32)/8,column+1);
    if(kb%32==31||kb+1==blocks){const bool first=kb<32;top.FoldWindow(first);++count[4];count[8]+=first?1:2;}
    ++count[1];count[3]+=valid;
   }
   pv.Run(top,VC_CANDIDATE_PASS,head,qr,qwork);
   top.Flush(outputGm,indexGm,head,qr,qwork);count[5]+=4;count[6]+=512;count[7]+=1536;
  }
  for(uint32_t i=0;i<64;++i){counterGm.SetValue((uint64_t(cores)+core*2+sub)*64+i,i<10?count[i]:0);pvCounterGm.SetValue(uint64_t(core*2+sub)*64+i,i<5?pv.count[i]:0);}
  DataCacheCleanAndInvalid<uint64_t,CacheLine::ENTIRE_DATA_CACHE>(pvCounterGm);
  pv.Release(pipe);top.Release(pipe);select.Release(pipe);
 }
};
