#pragma once
class CandidateCube {
    TBuf<TPosition::A1> a1_;TBuf<TPosition::B1> b1_;
    TBuf<TPosition::A2> a2_;TBuf<TPosition::B2> b2_;TBuf<TPosition::CO1> c_;
    TEventID l1Free_[2],l1Ready_[2],l0Free_[2],l0Ready_[2],cFree_[2],cReady_[2];
    __aicore__ inline void Allocate(TPipe& pipe) {
        for(uint32_t s=0;s<2;++s) {
            l1Free_[s]=pipe.AllocEventID<HardEvent::MTE1_MTE2>();
            l1Ready_[s]=pipe.AllocEventID<HardEvent::MTE2_MTE1>();
            l0Free_[s]=pipe.AllocEventID<HardEvent::M_MTE1>();
            l0Ready_[s]=pipe.AllocEventID<HardEvent::MTE1_M>();
            cFree_[s]=pipe.AllocEventID<HardEvent::FIX_M>();
            cReady_[s]=pipe.AllocEventID<HardEvent::M_FIX>();
        }
    }
    __aicore__ inline void Release(TPipe& pipe) {
        for(uint32_t s=0;s<2;++s) {
            pipe.ReleaseEventID<HardEvent::MTE1_MTE2>(l1Free_[s]);
            pipe.ReleaseEventID<HardEvent::MTE2_MTE1>(l1Ready_[s]);
            pipe.ReleaseEventID<HardEvent::M_MTE1>(l0Free_[s]);
            pipe.ReleaseEventID<HardEvent::MTE1_M>(l0Ready_[s]);
            pipe.ReleaseEventID<HardEvent::FIX_M>(cFree_[s]);
            pipe.ReleaseEventID<HardEvent::M_FIX>(cReady_[s]);
        }
    }
    __aicore__ inline void BeginOperands() {
        for(uint32_t s=0;s<2;++s) {
            SetFlag<HardEvent::MTE1_MTE2>(l1Free_[s]);
            SetFlag<HardEvent::M_MTE1>(l0Free_[s]);
        }
    }
    __aicore__ inline void EndOperands() {
        // Drain both used and unused initial permits; no quota survives a phase.
        for(uint32_t s=0;s<2;++s) {
            WaitFlag<HardEvent::MTE1_MTE2>(l1Free_[s]);
            WaitFlag<HardEvent::M_MTE1>(l0Free_[s]);
        }
    }
public:
    __aicore__ inline void Run(VC_CANDIDATE_ARGS) {
        TPipe pipe;pipe.InitBuffer(a1_,kRows*128*2);pipe.InitBuffer(b1_,2*128*128*2);
        pipe.InitBuffer(a2_,kRows*128*2);pipe.InitBuffer(b2_,2*128*128*2);pipe.InitBuffer(c_,2*kRows*128*4);
        Allocate(pipe);
        auto a1=a1_.Get<half>();auto b1=b1_.Get<half>();auto a2=a2_.Get<half>();auto b2=b2_.Get<half>();auto acc=c_.Get<float>();
        GlobalTensor<half> qGm,kGm,scoreGm;GlobalTensor<uint64_t> counterGm;
        qGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q));kGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k));
        scoreGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(score));counterGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(counters));
        const uint64_t core=GetBlockIdx(),tiles=(queries+kRows-1)/kRows,total=heads*tiles;
        const uint32_t blocks=kpad/128,groups=(blocks+kScoreChunkBlocks-1)/kScoreChunkBlocks;
        constexpr uint32_t bSlot=128*128,cSlot=kRows*128;
        const auto loadA=LoadData2DParams(0,8,kRows/16,0,0,false,0);
        const auto loadK=LoadData2DParams(0,64,1,0,0,false,0);
        const auto ndQ=Nd2NzParams(1,kRows,128,0,128,kRows,1,0);
        const auto ndK=Nd2NzParams(1,128,128,0,128,128,1,0);
        FixpipeParamsV220 scoreCopy(128,kRows,kRows,128,false);scoreCopy.quantPre=QuantMode_t::F322F16;
        uint64_t count[5]={};
        for(uint64_t task=core;task<total;task+=cores) {
            const uint64_t head=task/tiles;const uint32_t qr=task%tiles*kRows;++count[0];
            DataCopy(a1,qGm[(head*qpad+qr)*128],ndQ);PipeBarrier<PIPE_ALL>();
            for(uint32_t m=0;m<kRows/16;++m)LoadData(a2[m*128*16],a1[m*16*16],loadA);
            PipeBarrier<PIPE_ALL>();
            BeginOperands();
            for(uint32_t s=0;s<2;++s)SetFlag<HardEvent::FIX_M>(cFree_[s]);
            WaitFlag<HardEvent::MTE1_MTE2>(l1Free_[0]);
            DataCopy(b1,kGm[(head*kpad+uint64_t(0)*128)*128],ndK);
            SetFlag<HardEvent::MTE2_MTE1>(l1Ready_[0]);
            for(uint32_t kb=0;kb<blocks;++kb) {
                const uint32_t slot=kb&1,next=slot^1;
                const uint32_t group=kb/kScoreChunkBlocks,scoreSlot=group&1,column=kb%kScoreChunkBlocks;
                if(kb+1<blocks) {
                    WaitFlag<HardEvent::MTE1_MTE2>(l1Free_[next]);
                    DataCopy(b1[next*bSlot],kGm[(head*kpad+uint64_t(kb+1)*128)*128],ndK);
                    SetFlag<HardEvent::MTE2_MTE1>(l1Ready_[next]);
                }
                WaitFlag<HardEvent::MTE2_MTE1>(l1Ready_[slot]);
                WaitFlag<HardEvent::M_MTE1>(l0Free_[slot]);
                LoadData(b2[slot*bSlot],b1[slot*bSlot],loadK);
                SetFlag<HardEvent::MTE1_MTE2>(l1Free_[slot]);
                SetFlag<HardEvent::MTE1_M>(l0Ready_[slot]);
                WaitFlag<HardEvent::MTE1_M>(l0Ready_[slot]);
                WaitFlag<HardEvent::FIX_M>(cFree_[slot]);
                Mmad(acc[slot*cSlot],a2,b2[slot*bSlot],MmadParams(kRows,128,128,uint8_t(0),false,true));
                SetFlag<HardEvent::M_MTE1>(l0Free_[slot]);
                SetFlag<HardEvent::M_FIX>(cReady_[slot]);
                WaitFlag<HardEvent::M_FIX>(cReady_[slot]);
                if(column==0&&group>=2){CrossCoreWaitFlag(2+scoreSlot);++count[2];}
                Fixpipe(scoreGm[((core*2+scoreSlot)*kScoreChunkBlocks+column)*kRows*128],acc[slot*cSlot],scoreCopy);
                SetFlag<HardEvent::FIX_M>(cFree_[slot]);++count[1];
                if(column+1==kScoreChunkBlocks||kb+1==blocks){CrossCoreSetFlag<2,PIPE_FIX>(scoreSlot);++count[3];}
            }
            EndOperands();
            for(uint32_t s=0;s<2;++s)WaitFlag<HardEvent::FIX_M>(cFree_[s]);
            PipeBarrier<PIPE_ALL>();

            for(uint32_t g=groups>2?groups-2:0;g<groups;++g){CrossCoreWaitFlag(2+(g&1));++count[4];}
        }
        for(uint32_t i=0;i<64;++i)counterGm.SetValue(core*64+i,i<5?count[i]:0);
        DataCacheCleanAndInvalid<uint64_t,CacheLine::ENTIRE_DATA_CACHE>(counterGm);
        PipeBarrier<PIPE_ALL>();Release(pipe);
    }
};
