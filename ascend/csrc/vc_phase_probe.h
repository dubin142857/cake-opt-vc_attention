#pragma once
#include "kernel_operator.h"
#include "vc_tiling.h"

namespace vc910 {
// Diagnostic-only samples. No timers, extra flags, record buffers or barriers
// survive the Enabled=false instantiation used by the normal attention APIs.
// GetSystemCycle is a system counter, not the Cube/Vector core clock.
template <bool Enabled>
class PhaseProbe {
public:
    __aicore__ inline PhaseProbe() {}
    __aicore__ inline void Init(GM_ADDR trace, uint32_t core, uint32_t role) {
        if constexpr (Enabled) {
            base_ = trace;
            core_ = core;
            role_ = role;
        }
    }

    __aicore__ inline void Begin(uint64_t task, uint32_t key, const Tiling& t) {
        if constexpr (Enabled) {
            active_ = false;
            if (task >= t.cores) return; // first task owned by this core only
            const uint32_t tile = key / N, last = t.kpad / N - 1;
            for (uint32_t s = 0; s < PROFILE_SAMPLES; ++s) {
                if (tile == uint64_t(last) * s / (PROFILE_SAMPLES - 1)) {
                    slot_ = s;
                    active_ = true;
                    break; // repeated positions in short sequences use one slot
                }
            }
            if (!active_) return;
            for (uint32_t i = 0; i < PROFILE_FIELDS; ++i) values_[i] = 0;
            values_[0] = 1;
            values_[1] = task;
            values_[2] = key;
            values_[3] = t.keys - key < N ? t.keys - key : N;
            AscendC::PipeBarrier<PIPE_ALL>();
            last_ = uint64_t(AscendC::GetSystemCycle());
            values_[12] = last_;
        }
    }

    __aicore__ inline void Mark(uint32_t phase) {
        if constexpr (Enabled) {
            if (!active_) return;
            const uint64_t now = uint64_t(AscendC::GetSystemCycle());
            values_[4 + phase] = now - last_;
            last_ = now;
        }
    }

    __aicore__ inline void MarkAll(uint32_t phase) {
        if constexpr (Enabled) {
            if (!active_) return;
            AscendC::PipeBarrier<PIPE_ALL>();
            Mark(phase);
        }
    }

    template <AscendC::HardEvent Event>
    __aicore__ inline void MarkEvent(uint32_t phase, AscendC::TPipe& pipe) {
        if constexpr (Enabled) {
            if (!active_) return;
            const int32_t id = static_cast<int32_t>(pipe.FetchEventID(Event));
            AscendC::SetFlag<Event>(id);
            AscendC::WaitFlag<Event>(id);
            Mark(phase);
        }
    }

    __aicore__ inline void Finish() {
        if constexpr (Enabled) {
            if (!active_) return;
            values_[13] = last_;
            // One 128-byte slot per role/sample: no Cube/AIV false sharing.
            const uint64_t offset = ((uint64_t(core_) * PROFILE_ROLES + role_)
                                      * PROFILE_SAMPLES + slot_) * PROFILE_FIELDS;
            AscendC::GlobalTensor<uint64_t> record;
            record.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(base_) + offset, PROFILE_FIELDS);
            for (uint32_t i = 0; i < PROFILE_FIELDS; ++i) record.SetValue(i, values_[i]);
            for (uint32_t i = 0; i < PROFILE_FIELDS; i += 8) {
                AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                                 AscendC::DcciDst::CACHELINE_OUT>(record[i]);
            }
        }
    }
private:
    GM_ADDR base_ = nullptr;
    uint32_t core_ = 0, role_ = 0, slot_ = 0;
    bool active_ = false;
    uint64_t last_ = 0, values_[PROFILE_FIELDS];
};

// Explicitly stateless production specialization: no sample array or member
// initialization can increase the normal kernel's stack/register requirements.
template <>
class PhaseProbe<false> {
public:
    __aicore__ inline PhaseProbe() {}
    __aicore__ inline void Init(GM_ADDR, uint32_t, uint32_t) {}
    __aicore__ inline void Begin(uint64_t, uint32_t, const Tiling&) {}
    __aicore__ inline void Mark(uint32_t) {}
    __aicore__ inline void MarkAll(uint32_t) {}
    template <AscendC::HardEvent Event>
    __aicore__ inline void MarkEvent(uint32_t, AscendC::TPipe&) {}
    __aicore__ inline void Finish() {}
};
} // namespace vc910
