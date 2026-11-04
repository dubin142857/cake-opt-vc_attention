#pragma once
#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t kRows=128,kCounterStride=64,kScoreChunkBlocks=8;
#define VC_CANDIDATE_ARGS GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output, GM_ADDR score, \
    GM_ADDR flags, GM_ADDR stats, GM_ADDR raw, GM_ADDR counters, uint64_t heads, \
    uint32_t qpad, uint32_t kpad, uint32_t queries, uint32_t keys, uint32_t dimension, \
    uint32_t cores, uint32_t flagpad, float scale, uint32_t skipZero
#define VC_CANDIDATE_PASS q,k,v,output,score,flags,stats,raw,counters,heads,qpad,kpad,queries,keys,dimension,cores,flagpad,scale,skipZero
