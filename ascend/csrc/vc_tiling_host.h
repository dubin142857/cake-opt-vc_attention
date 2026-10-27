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
    //  H11: P 也改双槽 —— AIV 在等 O_READY(i) 之前先算第 i+1 块的概率, 那时 AIC 的 PV(i)
    //  可能仍在读 pg。安全性: AIV 要覆写 pgs[j&1] 发生在第 j+2 轮, 而第 j+1 轮已等过
    //  O_READY(j), 该标志在 PV(j) 之后置位, 故覆写时 PV(j) 必已读完。
    //  O 仍单槽: AIC 的 PV(i+1) 之前有推迟的 wait O_USED(i) 把关 (h6a)。
    return uint64_t(tile_m) * N * (2 * sizeof(int32_t) + 2 * sizeof(uint16_t))
         + uint64_t(tile_m) * dim * sizeof(float);
}
constexpr uint32_t VectorBytes(uint32_t reduce_bytes, uint32_t tile_m = M) {
    // int32 scores/indices, FP32 scores, FP16 P, accumulator, PV output,
    // channel scales, ten state/metadata vectors (H11 的两个 alpha 暂存槽在其中),
    // LUT, reduction scratch.
    const uint32_t rows = tile_m / 2;
    // Two per-row Brcb broadcast buffers (rowmax, alpha) of ROWS blocks each.
    //  H11 新增 rows * N * 4: S 的独立接收缓冲 (见 vc_kernel.cpp 的 sIntBuf 注释)。
    return rows * N * (4 + 4 + 2 + 4) + 2 * rows * MAX_D * 4
         + MAX_D * 4 + 10 * rows * 4 + 128 * 4 + 2 * rows * 8 * 4 + reduce_bytes;
}
static_assert(WorkspaceBytes(64) == 106496 && WorkspaceBytes(128) == 114688,
              "Workspace must cover two INT32 S slots, two FP16 P slots and the FP32 PV tile");
static_assert(WorkspaceBytes(64, 64) == 212992 && WorkspaceBytes(128, 64) == 229376,
              "M64 requires twice the scratch; Q quantization remains Q32");
}
