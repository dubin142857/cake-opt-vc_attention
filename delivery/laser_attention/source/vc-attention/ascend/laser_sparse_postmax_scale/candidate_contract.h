#pragma once
#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t kRows=128,kCounterStride=64,kScoreChunkBlocks=8;
#define VC_CANDIDATE_ARGS GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR geometry,GM_ADDR output,GM_ADDR score, \
 GM_ADDR candidates,GM_ADDR selectedScores,GM_ADDR selectedIds,GM_ADDR weights,GM_ADDR packetIds,GM_ADDR packetWeights, \
 GM_ADDR counters,GM_ADDR pvCounters,uint64_t heads,uint32_t qpad,uint32_t kpad,uint32_t queries,uint32_t keys, \
 uint32_t dimension,uint32_t cores,uint32_t columns,uint32_t rank,float scale
#define VC_CANDIDATE_PASS q,k,v,geometry,output,score,candidates,selectedScores,selectedIds,weights,packetIds,packetWeights,counters,pvCounters,heads,qpad,kpad,queries,keys,dimension,cores,columns,rank,scale
