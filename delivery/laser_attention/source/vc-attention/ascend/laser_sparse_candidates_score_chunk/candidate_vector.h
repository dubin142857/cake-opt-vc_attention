#pragma once
#include "candidate_selector.h"
class CandidateVector {
public:
    __aicore__ inline void Run(VC_CANDIDATE_ARGS) {
        TPipe pipe;CandidateSelector select;select.Init(pipe,v);
        GlobalTensor<half> scoreGm,outputGm;GlobalTensor<int32_t> indexGm;GlobalTensor<uint64_t> counterGm;
        scoreGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(score));outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
        indexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(flags));counterGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(counters));
        const uint32_t sub=GetSubBlockIdx(),core=GetBlockIdx()/GetSubBlockNum();
        const uint32_t tiles=(queries+127)/128,qwork=tiles*128,blocks=kpad/128;
        uint64_t count[6]={};
        for(uint64_t task=core;task<heads*tiles;task+=cores) {
            const uint64_t head=task/tiles;const uint32_t qr=task%tiles*128+sub*64;++count[0];
            for(uint32_t kb=0;kb<blocks;++kb) {
                const uint32_t group=kb/kScoreChunkBlocks,slot=group&1,column=kb%kScoreChunkBlocks;
                if(column==0)CrossCoreWaitFlag(slot);
                const uint32_t valid=kb*128>=keys?0:(keys-kb*128<128?keys-kb*128:128);
                select.ReadScore(scoreGm[((uint64_t(core)*2+slot)*kScoreChunkBlocks+column)*128*128+sub*8192]);
                if(column+1==kScoreChunkBlocks||kb+1==blocks){CrossCoreSetFlag<2,PIPE_MTE2>(2+slot);++count[2];}
                select.Transform(valid,scale);
                static_assert(kScoreChunkBlocks==kCandidateOutputColumns,"Score/output groups must match");
                select.Accumulate(column,head*kpad+kb*128);
                if(column+1==kCandidateOutputColumns||kb+1==blocks) {
                    const uint32_t n=column+1;
                    const uint64_t off=(head*flagpad+kb-column)*qwork+qr;
                    select.Flush(outputGm[off],indexGm[off],n,qwork);++count[4];count[5]+=2;
                }
                ++count[1];count[3]+=valid;
            }
        }
        for(uint32_t i=0;i<64;++i)counterGm.SetValue((uint64_t(cores)+core*2+sub)*64+i,i<6?count[i]:0);
        select.Release(pipe);
    }
};
