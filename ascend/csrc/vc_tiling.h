#pragma once
#include <cstdint>

namespace vc910 {
constexpr uint32_t M = 32;
constexpr uint32_t Q_SCALE_BLOCK = 32;
constexpr uint32_t N = 128;
constexpr uint32_t ROWS = M / 2;
constexpr uint32_t MAX_D = 128;
constexpr uint16_t S_READY = 8, P_READY = 9, O_READY = 10, O_USED = 15;
constexpr float EXPCAST_GAIN = 11.541560173034668f;
constexpr uint32_t PROFILE_ROLES = 3, PROFILE_SAMPLES = 4, PROFILE_FIELDS = 16;

// Only scalar control data. Cube tilings are separately serialized by CANN.
struct Tiling {
    uint32_t batch, heads, queries, keys, qpad, kpad, dim, cores;
    uint32_t reduce_bytes, expcast;
    float scale;
    // Per-AIC GM stride, computed on host and shared by both AIV halves.
    uint32_t workspace_bytes;
    uint64_t tasks;
};
static_assert(sizeof(Tiling) == 56, "Host/device control layout must match");
}
