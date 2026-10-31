#pragma once
#include "vc_tiling.h"
//  CHUNK 与块几何住在地址表里 (它同时守着 L0 预算的 static_assert)。
#include "vc_l1l0_map.h"

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
//  布局: S (**half**, 2xCHUNK 槽) · P (half, 2xCHUNK 槽) · O (fp32, 2 槽)。
//  ★S 由 int32 改 half (E38): fixpipe 以 `DEQF16` 直接吐出已施加 ks·scale 的 fp16 分数。
//    硬件强加 —— 本机对 int32 源没有任何带 quant 的 fp32 出口 (F233)。
//  ★三段的字节数住在 **vc_l1l0_map.h** (设备两侧也要用), 这里只做加总。
//  ★S/P 要**两个** chunk 组是因为错位一个 chunk: AIC 在算第 c 组的 QK 时,
//    AIV 还在读第 c-1 组的 S。不错位则两侧严格交替, 重叠比现状更差 (F189 的规则)。
constexpr uint64_t WorkspaceBytesV2(uint32_t dim, uint32_t tile_m, uint32_t nblk, uint32_t chunk) {
    return vc910::map::WsSBytesV2(tile_m, nblk, chunk)
         + vc910::map::WsPBytesV2(tile_m, nblk, chunk)
         + vc910::map::WsOBytesV2(dim, tile_m);
}
//  本机几何 (M=64, N=128, CHUNK=4) 的取值钉死, 免得有人改了常量而没核预算。
static_assert(WorkspaceBytesV2(128, 64, 128, 4) == 327680, "D=128 时每核应是 320 KB");
static_assert(WorkspaceBytesV2(64, 64, 128, 4) == 294912, "D=64 时每核应是 288 KB");
//  ★一条刚被编译器否掉的直觉, 留痕: 我原本断言
//    WorkspaceBytesV2(128,64,128,4) > WorkspaceBytesV2(128,64,256,2), 它**不成立** ——
//    S/P 的字节只取决于 CHUNK x NBLK 的**乘积** (4x128 = 2x256), 与怎么切无关。
//    即"把 N 减半再把 chunk 加倍"对 workspace 是**等价**的, 收益全部来自 O 的搬出频次。
static_assert(WorkspaceBytesV2(128, 64, 128, 4) == WorkspaceBytesV2(128, 64, 256, 2),
              "S/P 的字节只取决于 CHUNK x NBLK 的乘积; 此条变假说明布局算错了");
//  与真实基线 (旧布局, S 两槽 + P 两槽 + O 两槽, N=256) 比才是"拿空间换流量"的那笔账。
static_assert(WorkspaceBytesV2(128, 64, 128, 4) > WorkspaceBytes(128, 64),
              "本设计用 320 KB/核 (S 转 fp16 之前是 448 KB) 换 fixpipe 流量; "
              "若此条变假说明 chunk 深度或块大小被改小了, 收益也就没了");

//  UB 预算 —— 与 vc_kernel.cpp 的 InitBuffer 列表是同一个量的两处写法, 逐项对齐。
//  ★2026-10-30 实撞 (F212): 旧式**漏了** scoreBuf 的 CHUNK 倍与 rawBuf 的多缓冲,
//    且 state 按 10 槽算而实际 12 —— 合计少算 65792 B。实际占用 160768 而它只报 94976,
//    于是 vc_torch.cpp 里那道 `TORCH_CHECK(VectorBytes(...) <= ub)` **形同虚设**:
//    真超了它也不会红。守卫的覆盖面必须随被守对象一起长 (同 F210 的 EvGroup)。
constexpr uint32_t VectorBytes(uint32_t reduce_bytes, uint32_t tile_m = M, uint32_t chunk = 4) {
    const uint32_t rows = tile_m / 2;
    return VEC_RAW_SLOTS * rows * N * 4   // rawBuf   (S 的多缓冲)
         + chunk * rows * N * 4           // scoreBuf (整个 chunk 的分数, 两趟法要留着)
         + rows * N * 2                   // pBuf
         + rows * MAX_D * 4               // accBuf
         + rows * MAX_D * 4               // pvBuf
         + MAX_D * 4                      // vsBuf
         + 12 * rows * 4                  // stateBuf (H11 的 alpha 两槽 + tmax + massAcc 在内)
         + 128 * 4                        // lutBuf
         + rows * N * 4                   // subBuf   (H26 位运算解码)
         + rows * N / 8                   // cmpBuf   (H26 的逐元素掩码)
         + 2 * rows * 8 * 4               // brcbBuf  (rowmax / alpha 的逐行广播)
         + reduce_bytes;                  // tmpBuf
}
//  本机几何 (M=64, N=128, MAX_D=128, CHUNK=4, VEC_RAW_SLOTS=2) 的取值钉死 ——
//  改任何一项都要回来核这个数, 否则又会悄悄与 InitBuffer 分叉。
static_assert(VectorBytes(0, 64, 4) == 160768, "★UB 账与内核的 InitBuffer 列表对不上");
static_assert(VectorBytes(0, 64, 4) <= 196608, "★静态部分已超 UB, 还没算 reduce_bytes");
//  ★旧布局的两条取值断言已删: N 由 256 改 128 后它们的数值不再成立, 而主路径也已
//    改用 WorkspaceBytesV2。WorkspaceBytes 本体保留 —— 上面那条"拿空间换流量"的
//    对照断言还在用它, 它是**基线的量**, 不是现役布局。
static_assert(WorkspaceBytesV2(128, 64, N, 4) == 327680, "现役布局 D=128 应是 320 KB/核");
}
