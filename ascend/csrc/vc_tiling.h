#pragma once
#include <cstdint>

namespace vc910 {
constexpr uint32_t M = 32;
constexpr uint32_t N = 128;
constexpr uint32_t ROWS = M / 2;
constexpr uint32_t MAX_D = 128;
constexpr uint16_t S_READY = 8, P_READY = 9, O_READY = 10, O_USED = 15;
constexpr float EXPCAST_GAIN = 11.541560173034668f;

// Only scalar control data. Cube tilings are separately serialized by CANN.
struct Tiling {
    uint32_t batch, heads, queries, keys, qpad, kpad, dim, cores;
    uint32_t reduce_bytes, expcast;
    float scale;
    uint32_t reserved;
    uint64_t tasks;
};

constexpr uint64_t WorkspaceBytes(uint32_t dim) {
    return uint64_t(M) * N * (sizeof(int32_t) + sizeof(uint16_t))
         + uint64_t(M) * dim * sizeof(float);
}
constexpr uint32_t VectorBytes(uint32_t reduce_bytes) {
    // int32 scores/indices, FP32 scores, FP16 P, accumulator, PV output,
    // channel scales, eight state/metadata vectors, LUT, reduction scratch.
    return ROWS * N * (4 + 4 + 2) + 2 * ROWS * MAX_D * 4
         + MAX_D * 4 + 8 * ROWS * 4 + 128 * 4 + reduce_bytes;
}
}
