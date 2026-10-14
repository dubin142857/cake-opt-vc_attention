#pragma once
#include "vc_tiling.h"

// Host-only sizing helpers. Ordinary constexpr functions are not callable from
// Ascend C device code with runtime arguments. Serialize workspace_bytes into
// Tiling instead; keep the device's shared header free of host functions.
namespace vc910 {
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
static_assert(WorkspaceBytes(64) == 32768 && WorkspaceBytes(128) == 40960,
              "Workspace must cover INT32 S, FP16 P and FP32 PV tiles");
}
