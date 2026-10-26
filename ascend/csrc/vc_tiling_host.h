#pragma once
#include "vc_tiling.h"

// Host-only sizing helpers. Ordinary constexpr functions are not callable from
// Ascend C device code with runtime arguments. Serialize workspace_bytes into
// Tiling instead; keep the device's shared header free of host functions.
namespace vc910 {
constexpr uint64_t WorkspaceBytes(uint32_t dim, uint32_t tile_m = M) {
    return uint64_t(tile_m) * N * (sizeof(int32_t) + sizeof(uint16_t))
         + uint64_t(tile_m) * dim * sizeof(float);
}
constexpr uint32_t VectorBytes(uint32_t reduce_bytes, uint32_t tile_m = M) {
    // int32 scores/indices, FP32 scores, FP16 P, accumulator, PV output,
    // channel scales, eight state/metadata vectors, LUT, reduction scratch.
    const uint32_t rows = tile_m / 2;
    // Two per-row Brcb broadcast buffers (rowmax, alpha) of ROWS blocks each.
    return rows * N * (4 + 4 + 2) + 2 * rows * MAX_D * 4
         + MAX_D * 4 + 8 * rows * 4 + 128 * 4 + 2 * rows * 8 * 4 + reduce_bytes;
}
static_assert(WorkspaceBytes(64) == 57344 && WorkspaceBytes(128) == 65536,
              "Workspace must cover INT32 S, FP16 P and FP32 PV tiles");
static_assert(WorkspaceBytes(64, 64) == 114688 && WorkspaceBytes(128, 64) == 131072,
              "M64 requires twice the scratch; Q quantization remains Q32");
}
