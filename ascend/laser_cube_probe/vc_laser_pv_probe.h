// Diagnostic-only scalar issue/dependency windows for the V-only PV schedule.
// GetSystemCycle ticks are not device-engine durations or phase percentages.
#ifndef VC_LASER_PV_PROBE_H
#define VC_LASER_PV_PROBE_H
#include "kernel_operator.h"

namespace vc_laser_pv_probe {
constexpr uint64_t MAGIC = 0x56434C5056505242ULL;
constexpr uint32_t CORES = 4, POSITIONS = 3, FIELDS = 64;

class PvProbe {
public:
    __aicore__ inline void Init(__gm__ uint8_t *base, int32_t mode, uint32_t core) {
        base_ = base; mode_ = mode; core_ = core; seen_ = 0;
        active_ = false; first_descriptor_ = false; tile_ = -1;
    }
    __aicore__ inline void BeginCall(uint64_t round, uint64_t total) {
        active_ = false; first_descriptor_ = false; tile_ = -1;
        if (mode_ < 1 || mode_ > 2 || core_ >= CORES || total == 0) return;
        const int32_t selected = Position(round, total);
        if (selected < 0 || (seen_ & (1U << selected))) return;
        seen_ |= 1U << selected;
        position_ = selected; round_ = round; total_ = total;
        for (uint32_t slot = 0; slot < POSITIONS; ++slot) {
            for (uint32_t f = 0; f < FIELDS; ++f) values_[slot][f] = 0;
        }
        active_ = true;
        // No leading completion fence: prior work remains part of natural overlap.
        call_begin_ = Tick();
    }
    __aicore__ inline void Descriptor(uint64_t index, uint64_t count, uint64_t kx,
                                     uint64_t ky, bool on_start, bool on_end) {
        first_descriptor_ = active_ && index == 0;
        tile_ = -1;
        if (!first_descriptor_ || kx == 0) return;
        for (uint32_t slot = 0; slot < POSITIONS; ++slot) {
            const uint64_t k = slot == 0 ? 0 : (slot == 1 ? kx / 2 : kx - 1);
            if (Position(k, kx) != static_cast<int32_t>(slot)) continue;
            auto *v = values_[slot];
            v[0] = MAGIC; v[1] = 2; v[2] = mode_; v[3] = core_;
            v[4] = position_; v[5] = slot; v[6] = round_; v[7] = total_;
            v[8] = index; v[9] = count; v[10] = kx; v[11] = ky; v[12] = k;
            v[13] = on_start ? 1 : 0; v[14] = on_end ? 1 : 0;
            v[16] = slot == 0 ? 1 : 0;
            v[51] = (slot == 0 ? 1U : 0U) | (mode_ == 2 ? 4U : 0U);
            if (slot == 0 && mode_ == 2) {
                v[51] |= (on_start ? 0U : 8U) | (on_end ? 16U : 0U);
                v[49] = on_start ? 0 : 1; v[50] = on_end ? 1 : 0;
            }
        }
    }
    __aicore__ inline void SetGuard(bool enabled) {
        if (!first_descriptor_) return;
        for (uint32_t slot = 0; slot < POSITIONS; ++slot) {
            auto *v = values_[slot];
            if (v[0] != MAGIC) continue;
            v[15] = enabled ? 1 : 0;
            if (mode_ == 2 && enabled && v[12] + 1 < v[10]) v[51] |= 32U;
        }
    }
    __aicore__ inline void BeginTile(uint64_t k) {
        tile_ = -1;
        if (!first_descriptor_ || mode_ != 2) return;
        for (uint32_t slot = 0; slot < POSITIONS; ++slot) {
            if (values_[slot][0] == MAGIC && values_[slot][12] == k) {
                tile_ = static_cast<int32_t>(slot); tile_begin_ = Tick(); return;
            }
        }
    }
    __aicore__ inline void EndTile() {
        if (tile_ >= 0) values_[tile_][25] = Tick() - tile_begin_;
        tile_ = -1;
    }
    __aicore__ inline void Start(uint32_t field) {
        if (tile_ >= 0 && field >= 26 && field <= 39) phase_begin_ = Tick();
    }
    __aicore__ inline void Stop(uint32_t field) {
        if (tile_ < 0 || field < 26 || field > 39) return;
        const uint64_t end = Tick();
        values_[tile_][field] += end - phase_begin_;
        values_[tile_][42] |= 1ULL << (field - 26);
    }
    __aicore__ inline void Count(uint32_t field, uint64_t count) {
        if (tile_ >= 0 && field >= 43 && field <= 48) values_[tile_][field] += count;
    }
    __aicore__ inline void StartGroup() {
        if (first_descriptor_ && mode_ == 2 && values_[0][0] == MAGIC) group_begin_ = Tick();
    }
    __aicore__ inline void StopGroup(uint32_t field) {
        if (first_descriptor_ && mode_ == 2 && values_[0][0] == MAGIC &&
            (field == 23 || field == 24 || field == 40 || field == 41)) {
            values_[0][field] += Tick() - group_begin_;
        }
    }
    __aicore__ inline void EndCall() {
        if (!active_) return;
        const uint64_t issue_end = Tick();
        // Version 2 is passive in BOTH modes. Do not insert an all-pipeline
        // drain into the parent's live event schedule. Reserved completion
        // fields 19/21/22 and validity bit 1 remain zero: these timestamps
        // observe scalar issue, never completion of queued DMA/MAD/FIX work.
        if (values_[0][0] == MAGIC) {
            auto *group = values_[0];
            group[17] = call_begin_; group[18] = issue_end;
            group[20] = issue_end - call_begin_;
        }
        active_ = false; first_descriptor_ = false; tile_ = -1;
        // All record stores and cache flushes are outside measured windows.
        for (uint32_t slot = 0; slot < POSITIONS; ++slot) {
            if (values_[slot][0] != MAGIC) continue;
            const uint64_t offset = ((uint64_t(core_) * POSITIONS + position_) * POSITIONS + slot) * FIELDS;
            AscendC::GlobalTensor<uint64_t> record;
            record.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(base_) + offset, FIELDS);
            for (uint32_t f = 0; f < FIELDS; ++f) record.SetValue(f, values_[slot][f]);
            for (uint32_t f = 0; f < FIELDS; f += 8) {
                AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                                 AscendC::DcciDst::CACHELINE_OUT>(record[f]);
            }
        }
    }
private:
    __aicore__ inline uint64_t Tick() { return static_cast<uint64_t>(AscendC::GetSystemCycle()); }
    __aicore__ inline int32_t Position(uint64_t value, uint64_t total) {
        for (uint32_t i = 0; i < POSITIONS; ++i) {
            const uint64_t candidate = i == 0 ? 0 : (i == 1 ? total / 2 : total - 1);
            if (candidate < total && candidate == value) return static_cast<int32_t>(i);
        }
        return -1;
    }
    __gm__ uint8_t *base_ = nullptr;
    int32_t mode_ = 0, position_ = -1, tile_ = -1;
    uint32_t core_ = 0, seen_ = 0;
    uint64_t round_ = 0, total_ = 0, call_begin_ = 0, tile_begin_ = 0;
    uint64_t phase_begin_ = 0, group_begin_ = 0;
    uint64_t values_[POSITIONS][FIELDS];
    bool active_ = false, first_descriptor_ = false;
};
} // namespace vc_laser_pv_probe
#endif
