#pragma once
#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t kSparseRows=16,kSparseWidth=128,kSparseRank=16;
constexpr uint32_t kSparseElements=kSparseRows*kSparseWidth;
constexpr uint32_t kSparseUbBytes=84032;
static_assert(kSparseUbBytes<192*1024,"A2 UB budget");
class SparsePvVector {
    TBuf<TPosition::VECCALC> index_,weights_,values_,wide_,acc_,rowWeight_,broadcast_;
public:
    __aicore__ inline void Run(GM_ADDR value,GM_ADDR index,GM_ADDR weight,GM_ADDR output,GM_ADDR status,
        uint32_t rows,uint32_t keys,uint32_t dimension,uint32_t rank,uint32_t cores) {
        TPipe pipe;
        pipe.InitBuffer(index_,kSparseRows*kSparseRank*4);
        pipe.InitBuffer(weights_,kSparseRows*kSparseRank*2);
        pipe.InitBuffer(values_,kSparseRank*kSparseElements*2);
        pipe.InitBuffer(wide_,kSparseElements*4);
        pipe.InitBuffer(acc_,kSparseElements*4);
        pipe.InitBuffer(rowWeight_,kSparseRows*4);
        pipe.InitBuffer(broadcast_,kSparseRows*8*4);
        auto indices=index_.Get<int32_t>();auto weights=weights_.Get<half>();
        auto values=values_.Get<half>();auto wide=wide_.Get<float>();auto accumulated=acc_.Get<float>();
        auto rowWeight=rowWeight_.Get<float>();auto broadcast=broadcast_.Get<float>();
        GlobalTensor<half> vGm,wGm;GlobalTensor<int32_t> iGm;
        GlobalTensor<float> oGm;GlobalTensor<int64_t> sGm;
        vGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(value));
        wGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weight));
        iGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        oGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
        sGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(status));
        const int vm2=static_cast<int>(pipe.FetchEventID(HardEvent::V_MTE2));
        const int m2s=static_cast<int>(pipe.FetchEventID(HardEvent::MTE2_S));
        const int m2v=static_cast<int>(pipe.FetchEventID(HardEvent::MTE2_V));
        const int vm3=static_cast<int>(pipe.FetchEventID(HardEvent::V_MTE3));
        const int m3v=static_cast<int>(pipe.FetchEventID(HardEvent::MTE3_V));
        const uint32_t core=GetBlockIdx(),groups=(rows+kSparseRows-1)/kSparseRows;
        uint64_t count[4]={};
        for(uint32_t group=core;group<groups;group+=cores) {
            const uint32_t start=group*kSparseRows;
            const uint32_t valid=rows-start<kSparseRows?rows-start:kSparseRows;
            ++count[3];count[0]+=valid;count[1]+=uint64_t(valid)*rank;
            if(dimension!=kSparseWidth||valid!=kSparseRows) {
                Duplicate(values,half(0),kSparseRank*kSparseElements);PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE2>(vm2);WaitFlag<HardEvent::V_MTE2>(vm2);
            DataCopy(indices,iGm[uint64_t(group)*kSparseRows*kSparseRank],kSparseRows*kSparseRank);
            DataCopy(weights,wGm[uint64_t(group)*kSparseRows*kSparseRank],kSparseRows*kSparseRank);
            SetFlag<HardEvent::MTE2_S>(m2s);WaitFlag<HardEvent::MTE2_S>(m2s);
            uint32_t bad=0;
            for(uint32_t key=0;key<rank;++key)for(uint32_t row=0;row<valid;++row) {
                const int32_t raw=indices.GetValue(key*kSparseRows+row);
                const bool invalid=raw<0||uint32_t(raw)>=keys;
                if(invalid)bad|=uint32_t(1)<<row;
                const uint32_t selected=invalid?0:uint32_t(raw);
                DataCopy(values[key*kSparseElements+row*kSparseWidth],vGm[uint64_t(selected)*dimension],dimension);
            }
            SetFlag<HardEvent::MTE2_V>(m2v);WaitFlag<HardEvent::MTE2_V>(m2v);
            Duplicate(accumulated,0.0f,kSparseElements);PipeBarrier<PIPE_V>();
            for(uint32_t key=0;key<rank;++key) {
                Cast(wide,values[key*kSparseElements],RoundMode::CAST_NONE,kSparseElements);
                Cast(rowWeight,weights[key*kSparseRows],RoundMode::CAST_NONE,kSparseRows);
                PipeBarrier<PIPE_V>();
                Brcb(broadcast,rowWeight,kSparseRows/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                Mul(wide,wide,broadcast,uint64_t(64),uint8_t(kSparseRows),BinaryRepeatParams(1,1,0,16,16,1));
                Mul(wide[64],wide[64],broadcast,uint64_t(64),uint8_t(kSparseRows),BinaryRepeatParams(1,1,0,16,16,1));
                PipeBarrier<PIPE_V>();
                Add(accumulated,accumulated,wide,kSparseElements);PipeBarrier<PIPE_V>();
            }
            if(bad)for(uint32_t row=0;row<valid;++row)if(bad&(uint32_t(1)<<row)) {
                ++count[2];Duplicate(accumulated[row*kSparseWidth],0.0f,kSparseWidth);PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(vm3);WaitFlag<HardEvent::V_MTE3>(vm3);
            DataCopy(oGm[uint64_t(start)*dimension],accumulated,DataCopyParams{
                static_cast<uint16_t>(valid),static_cast<uint16_t>(dimension*4/32),
                static_cast<uint16_t>((kSparseWidth-dimension)*4/32),0});
            SetFlag<HardEvent::MTE3_V>(m3v);WaitFlag<HardEvent::MTE3_V>(m3v);
        }
        for(uint32_t i=0;i<64;++i)sGm.SetValue(uint64_t(core)*64+i,i<4?int64_t(count[i]):0);
    }
};
