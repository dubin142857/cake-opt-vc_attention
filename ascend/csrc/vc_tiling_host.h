#pragma once
#include "vc_tiling.h"

// Host-only sizing helpers. Ordinary constexpr functions are not callable from
// Ascend C device code with runtime arguments. Serialize workspace_bytes into
// Tiling instead; keep the device's shared header free of host functions.
namespace vc910 {
constexpr uint64_t WorkspaceBytes(uint32_t dim, uint32_t tile_m = M) {
    //  H10: S 两个槽 (int32 x2) + P 一个槽 (fp16) + O 一个槽 (fp32)。
    //  只有 S 需要双缓冲 —— AIC 要在等本 tile 的 P_READY 之前就把下一 tile 的 QK 写出去,
    //  而 AIV 这时还在读本 tile 的 S。P 与 O 的单槽仍安全:
    //    P: AIV 要写 P(k+1) 必须先完成 tile k, 而那要等 O_READY(k), 发生在 AIC 的 PV(k) 读完 P 之后;
    //    O: AIC 的 PV(k+1) 之前有推迟的 wait O_USED(k) 把关 (h6a)。
    return uint64_t(tile_m) * N * (2 * sizeof(int32_t) + sizeof(uint16_t))
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
static_assert(WorkspaceBytes(64) == 90112 && WorkspaceBytes(128) == 98304,
              "Workspace must cover INT32 S, FP16 P and FP32 PV tiles");
static_assert(WorkspaceBytes(64, 64) == 180224 && WorkspaceBytes(128, 64) == 196608,
              "M64 requires twice the scratch; Q quantization remains Q32");
}
