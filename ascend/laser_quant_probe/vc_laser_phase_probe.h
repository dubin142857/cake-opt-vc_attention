// Sparse diagnostics for the pinned scale-reuse Laser kernel, not production.
// GetSystemCycle returns a system counter: these values are ticks, not AIC/AIV
// core cycles or calibrated nanoseconds. Mode 0 retains the compiled class and
// runtime branches; compare it separately with the unmodified production API.
#ifndef VC_LASER_PHASE_PROBE_H
#define VC_LASER_PHASE_PROBE_H
#include "kernel_operator.h"

namespace vc_laser_probe {
constexpr uint64_t MAGIC = 0x56434C50524F4245ULL;
constexpr uint32_t CORES = 4, ROLES = 3, SLOTS = 8, FIELDS = 24;

class PhaseProbe {
public:
    __aicore__ inline void Init(__gm__ uint8_t *base, int32_t mode,
                               uint32_t core, uint32_t role) {
        base_ = base; mode_ = mode; core_ = core; role_ = role;
    }
    __aicore__ inline void SetRound(uint64_t round, uint64_t total) {
        round_ = round; total_ = total;
    }
    // Wait slots choose rounds [0, 1, floor(total/2), total-1]. Repeated
    // positions use the lowest slot; inactive logical work is still recorded.
    __aicore__ inline void WaitBegin(uint64_t round, uint64_t total, bool running) {
        wait_active_ = false;
        if (mode_ < 1 || core_ >= CORES || role_ >= ROLES || total == 0) return;
        int32_t slot = -1;
        for (uint32_t i = 0; i < 4; ++i) {
            const uint64_t candidate = i == 0 ? 0 : (i == 1 ? 1 : (i == 2 ? total / 2 : total - 1));
            if (candidate < total && candidate == round) { slot = i; break; }
        }
        if (slot < 0) return;
        Reset(1, static_cast<uint32_t>(slot));
        values_[7] = round; values_[9] = running ? 1 : 0; values_[10] = total;
        wait_active_ = true;
        values_[21] = Tick();
    }
    __aicore__ inline void WaitEnd() {
        if (!wait_active_) return;
        const uint64_t end = Tick();
        values_[14] = end - values_[21]; values_[22] = end; values_[23] = 1;
        wait_active_ = false;
        Write(); // Scalar stores/cache flushes are outside the measured wait.
    }
    // First real four-row batch in each (head,tail) class, separately per AIV.
    // Slots 4..7 map class=(head?1:0)|(tail?2:0), and are never overwritten.
    __aicore__ inline void PhaseBegin(uint32_t section, bool head, bool tail,
                                     uint32_t key_blocks, uint32_t rows, uint32_t lines) {
        phase_active_ = false;
        if (mode_ != 2 || core_ >= CORES || role_ == 0 || role_ >= ROLES || rows != 4) return;
        const uint32_t kind = (head ? 1U : 0U) | (tail ? 2U : 0U);
        if ((seen_classes_ & (1U << kind)) != 0) return;
        seen_classes_ |= 1U << kind;
        Reset(2, 4 + kind);
        values_[7] = round_; values_[8] = section;
        values_[9] = 1U | (head ? 2U : 0U) | (tail ? 4U : 0U);
        values_[10] = total_; values_[11] = key_blocks; values_[12] = rows; values_[13] = lines;
        phase_active_ = true;
        VectorComplete();
        const uint64_t start = Tick();
        VectorComplete();
        const uint64_t end = Tick();
        values_[15] = end - start; values_[21] = start; values_[23] = 2;
        last_ = end;
    }
    __aicore__ inline void Mark(uint32_t field, uint32_t bit) {
        if (!phase_active_) return;
        VectorComplete();
        const uint64_t now = Tick();
        values_[field] = now - last_; values_[23] |= uint64_t(1) << bit;
        last_ = now;
    }
    __aicore__ inline void StartSum() {
        if (!phase_active_) return;
        // Exclude the V->MTE3 dependency and probability DMA issue from the
        // sum window. The DMA itself can overlap the following V reduction.
        VectorComplete();
        last_ = Tick();
    }
    __aicore__ inline void PhaseFinish() {
        if (!phase_active_) return;
        values_[22] = last_;
        phase_active_ = false;
        Write(); // No trace stores inside a completed phase interval.
    }
private:
    __aicore__ inline uint64_t Tick() { return static_cast<uint64_t>(AscendC::GetSystemCycle()); }
    __aicore__ inline void VectorComplete() {
#ifdef __DAV_C220_VEC__
        // The original inner helpers use V_S IDs 1..6. Original V_S0 is the
        // later Run-tail adjacent set/wait, consumed before another section.
        // Use this pair only within the sampled four-row helpers. PIPE_V alone
        // does not establish completion before a scalar timestamp read.
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
#endif
    }
    __aicore__ inline void Reset(uint32_t kind, uint32_t slot) {
        for (uint32_t i = 0; i < FIELDS; ++i) values_[i] = 0;
        values_[0] = MAGIC; values_[1] = 1; values_[2] = mode_;
        values_[3] = core_; values_[4] = role_; values_[5] = kind; values_[6] = slot;
    }
    __aicore__ inline void Write() {
        const uint64_t offset = ((uint64_t(core_) * ROLES + role_) * SLOTS + values_[6]) * FIELDS;
        AscendC::GlobalTensor<uint64_t> record;
        record.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(base_) + offset, FIELDS);
        for (uint32_t i = 0; i < FIELDS; ++i) record.SetValue(i, values_[i]);
        // 192-byte records occupy disjoint complete 64-byte cache lines.
        for (uint32_t i = 0; i < FIELDS; i += 8) {
            AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                             AscendC::DcciDst::CACHELINE_OUT>(record[i]);
        }
    }
    __gm__ uint8_t *base_ = nullptr;
    int32_t mode_ = 0;
    uint32_t core_ = 0, role_ = 0, seen_classes_ = 0;
    uint64_t round_ = 0, total_ = 0, last_ = 0, values_[FIELDS];
    bool wait_active_ = false, phase_active_ = false;
};
} // namespace vc_laser_probe
#endif
