#pragma once
#include <cstdint>

namespace vc910 {
constexpr uint32_t M = 32;
constexpr uint32_t Q_SCALE_BLOCK = 32;
//  N 是 K 方向的 **matmul tile 宽度**; K_SCALE_BLOCK 是 K 的**量化块大小**。
//  两者此前共用同一个常量, 于是 tile 宽度被量化粒度钉死。拆开后 tile 可以跨多个量化块,
//  代价是 dequant scale 在 tile 内沿列方向分段, 由 AIV 按列半区分别施加。
constexpr uint32_t N = 256;
constexpr uint32_t K_SCALE_BLOCK = 128;
static_assert(N % K_SCALE_BLOCK == 0, "tile 宽度必须覆盖整数个 K 量化块");
constexpr uint32_t KSB_PER_TILE = N / K_SCALE_BLOCK;
constexpr uint32_t ROWS = M / 2;
constexpr uint32_t MAX_D = 128;
//  H16: PV 的 B (V) 由内核自管 L1 的两个槽。主机侧必须把这块从 PV matmul 的 L1
//  预算里扣掉, 否则两个 matmul 各占 l1/2 已把 L1 占满, 再加这 128 KB 就溢出。
//  这是**跨文件约定**: 内核的 InitBuffer 与主机的 SetBufferSpace 必须用同一个常量。
constexpr uint32_t VSCM_SLOT_BYTES = N * MAX_D * 2;   // 每槽一个 V tile (NZ, half)
constexpr uint32_t VSCM_SLOTS = 2;
constexpr uint32_t VSCM_BYTES = VSCM_SLOTS * VSCM_SLOT_BYTES;
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
