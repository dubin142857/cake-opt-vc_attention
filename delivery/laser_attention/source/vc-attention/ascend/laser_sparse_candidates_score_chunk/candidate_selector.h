#pragma once
#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t kCandidateVectorRows=64,kCandidateElements=8192,kCandidateUbBytes=54656;
constexpr uint32_t kCandidateOutputColumns=8;
class CandidateSelector {
    TBuf<TPosition::VECCALC> data_,position_,broadcast_,mask_,pair_,value_,pairOffset_,rowOffset_,index_,temporary_,globalIndex_,indexHalf_;
    int vm2_,m2v_,vm3_[2],m3v_;
public:
    LocalTensor<half> data,position,broadcast,pair,value,indexHalf;
    LocalTensor<uint8_t> mask;
    LocalTensor<uint32_t> pairOffset,rowOffset,index,temporary,globalIndex;
    __aicore__ inline void Init(TPipe& pipe,GM_ADDR geometry) {
        pipe.InitBuffer(data_,8192*2);pipe.InitBuffer(position_,8192*2);pipe.InitBuffer(broadcast_,8192*2);
        pipe.InitBuffer(mask_,1024);pipe.InitBuffer(pair_,128*2);pipe.InitBuffer(value_,512*2);
        pipe.InitBuffer(pairOffset_,64*4);pipe.InitBuffer(rowOffset_,64*4);pipe.InitBuffer(index_,64*4);
        pipe.InitBuffer(temporary_,64*4);pipe.InitBuffer(globalIndex_,512*4);pipe.InitBuffer(indexHalf_,64*2);
        data=data_.Get<half>();position=position_.Get<half>();broadcast=broadcast_.Get<half>();mask=mask_.Get<uint8_t>();
        pair=pair_.Get<half>();value=value_.Get<half>();pairOffset=pairOffset_.Get<uint32_t>();rowOffset=rowOffset_.Get<uint32_t>();
        index=index_.Get<uint32_t>();temporary=temporary_.Get<uint32_t>();globalIndex=globalIndex_.Get<uint32_t>();indexHalf=indexHalf_.Get<half>();
        vm2_=pipe.FetchEventID(HardEvent::V_MTE2);m2v_=pipe.FetchEventID(HardEvent::MTE2_V);
        vm3_[0]=pipe.AllocEventID<HardEvent::V_MTE3>();vm3_[1]=pipe.AllocEventID<HardEvent::V_MTE3>();m3v_=pipe.FetchEventID(HardEvent::MTE3_V);
        GlobalTensor<half> g;g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(geometry));DataCopy(position,g,8192);
        SetFlag<HardEvent::MTE2_V>(m2v_);WaitFlag<HardEvent::MTE2_V>(m2v_);
        CreateVecIndex(pairOffset.ReinterpretCast<int32_t>(),int32_t(0),64);PipeBarrier<PIPE_V>();
        ShiftLeft(rowOffset,pairOffset,uint32_t(8),64);ShiftLeft(pairOffset,pairOffset,uint32_t(2),64);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Release(TPipe& pipe) {
        PipeBarrier<PIPE_ALL>();
        pipe.ReleaseEventID<HardEvent::V_MTE3>(vm3_[0]);pipe.ReleaseEventID<HardEvent::V_MTE3>(vm3_[1]);
    }
    __aicore__ inline void ReadScore(const GlobalTensor<half>& source) {
        SetFlag<HardEvent::V_MTE2>(vm2_);WaitFlag<HardEvent::V_MTE2>(vm2_);DataCopy(data,source,8192);
        SetFlag<HardEvent::MTE2_V>(m2v_);WaitFlag<HardEvent::MTE2_V>(m2v_);
    }
    __aicore__ inline void Transform(uint32_t valid,float scale) {
        Muls(data,data,half(scale),8192);PipeBarrier<PIPE_V>();
        if(valid<128) {
            auto words=position.ReinterpretCast<uint32_t>();
            Gather(mask.ReinterpretCast<uint32_t>(),words,words[1288],uint32_t(2048+valid*16),256);PipeBarrier<PIPE_V>();
            Select(data,mask,data,half(-__builtin_inff()),SELMODE::VSEL_TENSOR_SCALAR_MODE,8192);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Load(const GlobalTensor<half>& source,uint32_t valid,float scale) {
        ReadScore(source);Transform(valid,scale);
    }
    __aicore__ inline void Accumulate(uint32_t column,uint32_t base) {
        auto outValue=value[column*64];auto outIndex=globalIndex[column*64];
        WholeReduceMax(pair,data,128,64,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
        Gather(outValue,pair,pairOffset,0,64);
        ShiftRight(index,pair.ReinterpretCast<uint32_t>(),uint32_t(16),64);PipeBarrier<PIPE_V>();
        Adds(outIndex.ReinterpretCast<int32_t>(),index.ReinterpretCast<int32_t>(),int32_t(base),64);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Flush(const GlobalTensor<half>& output,const GlobalTensor<int32_t>& ids,uint32_t columns,uint32_t qwork) {
        SetFlag<HardEvent::V_MTE3>(vm3_[1]);WaitFlag<HardEvent::V_MTE3>(vm3_[1]);
        DataCopyPad(output,value,DataCopyExtParams{uint16_t(columns),128,0,qwork*2-128,0});
        DataCopyPad(ids,globalIndex.ReinterpretCast<int32_t>(),DataCopyExtParams{uint16_t(columns),256,0,qwork*4-256,0});
        SetFlag<HardEvent::MTE3_V>(m3v_);WaitFlag<HardEvent::MTE3_V>(m3v_);
    }
    __aicore__ inline void Pair(const GlobalTensor<half>& output,const GlobalTensor<int32_t>& ids,uint32_t base,bool exclude) {
        const uint32_t outputSlot=exclude?0:64;
        auto outValue=value[outputSlot];auto outIndex=globalIndex[outputSlot];
        WholeReduceMax(pair,data,128,64,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
        Gather(outValue,pair,pairOffset,0,64);
        ShiftRight(index,pair.ReinterpretCast<uint32_t>(),uint32_t(16),64);PipeBarrier<PIPE_V>();
        Adds(outIndex.ReinterpretCast<int32_t>(),index.ReinterpretCast<int32_t>(),int32_t(base),64);PipeBarrier<PIPE_V>();
        const uint32_t readySlot=exclude?0:1;
        SetFlag<HardEvent::V_MTE3>(vm3_[readySlot]);WaitFlag<HardEvent::V_MTE3>(vm3_[readySlot]);
        DataCopy(output,outValue,64);DataCopy(ids,outIndex.ReinterpretCast<int32_t>(),64);
        if(!exclude){SetFlag<HardEvent::MTE3_V>(m3v_);WaitFlag<HardEvent::MTE3_V>(m3v_);}
        if(exclude) {
            auto words=position.ReinterpretCast<uint32_t>();
            auto wide=broadcast.ReinterpretCast<uint32_t>();auto offsets=wide[256];
            Gather(wide,index,words[1032],0,256);PipeBarrier<PIPE_V>();
            ShiftLeft(offsets,wide,uint32_t(4),256);PipeBarrier<PIPE_V>();
            Add(offsets.ReinterpretCast<int32_t>(),offsets.ReinterpretCast<int32_t>(),words[1288].ReinterpretCast<int32_t>(),256);PipeBarrier<PIPE_V>();
            Gather(mask.ReinterpretCast<uint32_t>(),words,offsets,0,256);PipeBarrier<PIPE_V>();
            Select(data,mask,data,half(-__builtin_inff()),SELMODE::VSEL_TENSOR_SCALAR_MODE,8192);PipeBarrier<PIPE_V>();
        }
    }
};
