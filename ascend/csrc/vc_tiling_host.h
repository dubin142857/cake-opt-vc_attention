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
    //  H15: O 也改双槽 —— AIC 等 O_USED 的唯一原因是 og 只有一个槽。
    return uint64_t(tile_m) * N * (2 * sizeof(int32_t) + 2 * sizeof(uint16_t))
         + 2 * uint64_t(tile_m) * dim * sizeof(float);
}
//  ---- CubeV2 (按 chunk 分阶段) 的 workspace ---------------------------------
//  ★与 vc_kernel.cpp 里 CubeV2::RunCube 的布局是**同一个量的两处写法**, 改一处必须改另一处
//    (与上面 VectorBytes 和内核两次 InitBuffer 的关系同形, hangdiag C14 守的就是这种成对量)。
//  布局: S (int32, 2xCHUNK 槽) · P (half, 2xCHUNK 槽) · O (fp32, 2 槽)。
//  ★S/P 要**两个** chunk 组是因为错位一个 chunk: AIC 在算第 c 组的 QK 时,
//    AIV 还在读第 c-1 组的 S。不错位则两侧严格交替, 重叠比现状更差 (F189 的规则)。
constexpr uint64_t WorkspaceBytesV2(uint32_t dim, uint32_t tile_m, uint32_t nblk, uint32_t chunk) {
    return 2ull * chunk * tile_m * nblk * sizeof(int32_t)     // S
         + 2ull * chunk * tile_m * nblk * sizeof(uint16_t)    // P
         + 2ull * tile_m * dim * sizeof(float);               // O
}
//  本机几何 (M=64, N=128, CHUNK=4) 的取值钉死, 免得有人改了常量而没核预算。
static_assert(WorkspaceBytesV2(128, 64, 128, 4) == 458752, "D=128 时每核应是 448 KB");
static_assert(WorkspaceBytesV2(64, 64, 128, 4) == 425984, "D=64 时每核应是 416 KB");
//  ★一条刚被编译器否掉的直觉, 留痕: 我原本断言
//    WorkspaceBytesV2(128,64,128,4) > WorkspaceBytesV2(128,64,256,2), 它**不成立** ——
//    S/P 的字节只取决于 CHUNK x NBLK 的**乘积** (4x128 = 2x256), 与怎么切无关。
//    即"把 N 减半再把 chunk 加倍"对 workspace 是**等价**的, 收益全部来自 O 的搬出频次。
static_assert(WorkspaceBytesV2(128, 64, 128, 4) == WorkspaceBytesV2(128, 64, 256, 2),
              "S/P 的字节只取决于 CHUNK x NBLK 的乘积; 此条变假说明布局算错了");
//  与真实基线 (旧布局, S 两槽 + P 两槽 + O 两槽, N=256) 比才是"拿空间换流量"的那笔账。
static_assert(WorkspaceBytesV2(128, 64, 128, 4) > WorkspaceBytes(128, 64),
              "本设计用 448 KB/核 换 fixpipe 流量由 223.8 降到 186.3 GB; "
              "若此条变假说明 chunk 深度或块大小被改小了, 收益也就没了");

constexpr uint32_t VectorBytes(uint32_t reduce_bytes, uint32_t tile_m = M) {
    // int32 scores/indices, FP32 scores, FP16 P, accumulator, PV output,
    // channel scales, ten state/metadata vectors (H11 的两个 alpha 暂存槽在其中),
    // LUT, reduction scratch.
    const uint32_t rows = tile_m / 2;
    // Two per-row Brcb broadcast buffers (rowmax, alpha) of ROWS blocks each.
    //  H26: 位运算解码新增 subBuf (rows*N*4) 与 cmpBuf (rows*N/8) ——
    //  与 vc_kernel.cpp 的两次 InitBuffer 是同一个量的两处写法, 改一处必须改另一处
    //  (hangdiag 的 C14 会在给定点上比对两侧的数值)。
    return rows * N * (4 + 4 + 2 + 4) + rows * N / 8 + 2 * rows * MAX_D * 4
         + MAX_D * 4 + 10 * rows * 4 + 128 * 4 + 2 * rows * 8 * 4 + reduce_bytes;
}
static_assert(WorkspaceBytes(64) == 114688 && WorkspaceBytes(128) == 131072,
              "Workspace must cover two INT32 S slots, two FP16 P slots and the FP32 PV tile");
static_assert(WorkspaceBytes(64, 64) == 229376 && WorkspaceBytes(128, 64) == 262144,
              "M64 requires twice the scratch; Q quantization remains Q32");
}
