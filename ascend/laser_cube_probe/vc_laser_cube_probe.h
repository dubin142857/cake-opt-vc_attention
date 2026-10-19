// Sparse, independent diagnostics for the pinned Laser Cube pipeline.
// Raw GetSystemCycle ticks are NOT Cube clock cycles or production phase shares.
// Mode 1 brackets instruction issue; mode 2 also drains complete call boundaries.
// Neither mode changes arithmetic. PIPE_ALL intentionally perturbs overlap.
#ifndef VC_LASER_CUBE_PROBE_H
#define VC_LASER_CUBE_PROBE_H
#include "kernel_operator.h"

namespace vc_laser_cube_probe {
constexpr uint64_t MAGIC = 0x56434C4350524F42ULL;
constexpr uint32_t CORES = 4, STAGES = 2, POSITIONS = 3, FIELDS = 24;

class CubeProbe {
public:
    __aicore__ inline void Init(__gm__ uint8_t *base, int32_t mode, uint32_t core) {
        base_ = base; mode_ = mode; core_ = core; seen_ = 0; active_ = false;
    }
    __aicore__ inline void Begin(uint32_t stage, uint64_t round, uint64_t total) {
        active_ = false;
        if (mode_ < 1 || mode_ > 2 || core_ >= CORES || stage >= STAGES || total == 0) return;
        int32_t position = -1;
        // Duplicate first/middle/last rounds use the lowest position only.
        for (uint32_t i = 0; i < POSITIONS; ++i) {
            const uint64_t candidate = i == 0 ? 0 : (i == 1 ? total / 2 : total - 1);
            if (candidate < total && candidate == round) { position = i; break; }
        }
        if (position < 0) return;
        const uint32_t slot = stage * POSITIONS + static_cast<uint32_t>(position);
        if ((seen_ & (1U << slot)) != 0) return;
        seen_ |= 1U << slot;
        for (uint32_t i = 0; i < FIELDS; ++i) values_[i] = 0;
        values_[0] = MAGIC; values_[1] = 1; values_[2] = mode_; values_[3] = core_;
        values_[4] = stage; values_[5] = static_cast<uint32_t>(position);
        values_[6] = round; values_[7] = total;
        active_ = true;
        if (mode_ == 2) {
            const uint64_t start = Tick();
            Complete();
            values_[20] = Tick() - start;
            const uint64_t empty_start = Tick();
            Complete();
            values_[19] = Tick() - empty_start;
        }
        // Address mapping and the entire cube1/cube2 call follow this timestamp.
        values_[13] = Tick();
    }
    template<class Descriptor>
    __aicore__ inline void EndQK(const Descriptor *src, int64_t count) {
        if (!EndTiming()) return;
        Metadata(src, count);
        Write();
    }
    template<class Descriptor>
    __aicore__ inline void EndPV(const Descriptor *src, int64_t count) {
        if (!EndTiming()) return;
        Metadata(src, count);
        for (int64_t i = 0; i < count; ++i) {
            values_[11] += src[i].onStartSection ? 0 : 1;
            values_[12] += src[i].onEndSection ? 1 : 0;
        }
        Write();
    }
private:
    __aicore__ inline uint64_t Tick() { return static_cast<uint64_t>(AscendC::GetSystemCycle()); }
    __aicore__ inline void Complete() {
#ifdef __DAV_C220_CUBE__
        // Official GetSystemCycle timing pattern: PIPE_ALL -> scalar timestamp.
        // Called ONLY outside complete cube1/cube2 invocations, after all paired
        // MAD unitFlag/FIX-copy instructions have been issued. Never drain the
        // history diagonal MAD(copy_out=false) before its following PV/copy.
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
    }
    __aicore__ inline bool EndTiming() {
        if (!active_) return false;
        const uint64_t issue_end = Tick();
        uint64_t end = issue_end;
        if (mode_ == 2) {
            Complete();
            end = Tick();
        }
        values_[14] = issue_end; values_[15] = end;
        values_[16] = issue_end - values_[13];
        values_[17] = end - issue_end; values_[18] = end - values_[13];
        values_[21] = mode_ == 2 ? 3 : 1;
        active_ = false;
        return true;
    }
    template<class Descriptor>
    __aicore__ inline void Metadata(const Descriptor *src, int64_t count) {
        values_[8] = count;
        // Logical descriptor work, not an instruction count or FLOP count:
        // triangles may skip blocks and diagnostic runs currently use dense mode.
        for (int64_t i = 0; i < count; ++i) {
            values_[9] += src[i].kx;
            values_[10] += static_cast<uint64_t>(src[i].kx) * src[i].ky;
        }
    }
    __aicore__ inline void Write() {
        const uint64_t offset = ((uint64_t(core_) * STAGES + values_[4]) * POSITIONS + values_[5]) * FIELDS;
        AscendC::GlobalTensor<uint64_t> record;
        record.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(base_) + offset, FIELDS);
        // Metadata and scalar GM writes happen after the complete timing window.
        for (uint32_t i = 0; i < FIELDS; ++i) record.SetValue(i, values_[i]);
        for (uint32_t i = 0; i < FIELDS; i += 8) {
            AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                             AscendC::DcciDst::CACHELINE_OUT>(record[i]);
        }
    }
    __gm__ uint8_t *base_ = nullptr;
    int32_t mode_ = 0;
    uint32_t core_ = 0, seen_ = 0;
    uint64_t values_[FIELDS];
    bool active_ = false;
};
} // namespace vc_laser_cube_probe
#endif
