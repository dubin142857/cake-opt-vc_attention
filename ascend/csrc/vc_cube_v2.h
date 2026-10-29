//  CubeV2 —— Phase 0 的 cube 侧: 手写 mmad + 裸地址 + 128x128 块 + FP16 契约。
//
//  为什么另起一个类而不是改 Cube:
//    PV 的 B 是 BLK*dim half = 32KB, QK 的 B 同样 32KB, 两者加上双缓冲正好用满
//    64KB 的 L0B —— **前提是这个核里没有别的 L0 使用者**。旧 Cube 的两个
//    MatmulImpl 经 TPipe 自己占 L0, 与裸地址并存就是"同一个核里两套分配权威",
//    正是我们那个竞态的形态。故 V2 独占整核, 旧 Cube 原封不动留作对照。
//
//  与旧 Cube 的差别 (Phase 0 范围):
//    块 64x256 -> **128x128** (两个矩乘都成 128^3)
//    S 的 dtype int32 -> **half** —— 靠 copy_matrix_cc_to_gm 的 QuantPRE=F322F16
//      让 fixpipe **在写出的路上**收窄, 不需要额外一趟 (这是 T6 的真实机制)
//    MatmulImpl -> 手写 load/mad/fixpipe, 地址与事件号全部显式 (见 vc_l1l0_map.h)
//  **不在本相范围**: T8 的深度 2 软件流水 (Phase 2) · T1 的对角阵矩乘 (Phase 3) ·
//    INT8 (Phase 4, 且有硬闸: 必须先确认 cube 成为 binding pipe)。
#ifndef VC_CUBE_V2_H
#define VC_CUBE_V2_H

//  ★公开 C API 的入口。内建 (copy_matrix_cc_to_gm 等) 不需要 include, 所以此前
//    一直编得过; asc_* 这一套需要 —— 第一次改用公开 API 时漏了它, 报
//    "use of undeclared identifier 'asc_mmad_sync'"。
#include "c_api/asc_simd.h"
#include "vc_l1l0_map.h"
#include "vc_tiling.h"

namespace vc910 {

//  一次 128x128x128 的矩乘并把结果写回 GM。
//
//  ★用**公开 C API** (asc/include/c_api) 而不是 Laser 那套未声明的 ccec 内建。
//    2026-10-29 实撞: 逆向内建花了三次构建仍没对上 set_flag 的第三参类型, 后来才发现
//    cann-9.0.0/aarch64-linux/**asc/include/c_api** 下有一整套**有声明、按 dtype 重载**
//    的公开接口。此前搜公开头搜的是 ascendc/include —— 搜索根错了 (教训二)。
//    公开 API 的覆盖并不完整: L0C->GM **只有配置函数**没有搬运函数,
//    故那一步仍用内建 copy_matrix_cc_to_gm, 由 asc_set_l0c2gm_config 配置。
//
//  ★Phase 0 一律用 **_sync 变体**: 它们内嵌必要的流水同步, 构造性地避开手写
//    set/wait 的一整类错误 —— 而我们那个竞态正是同步问题。代价是少了重叠,
//    但本相的目标是"骨架做对, 不求快"。Phase 2 做软件流水时再换成手动同步。
template <bool UsePing, bool BTrans, typename OutT>
__aicore__ inline void mm128(__gm__ OutT *gm_out, __gm__ half *gm_a, __gm__ half *gm_b,
                             uint16_t a_rows_stride, uint16_t b_rows_stride,
                             uint16_t out_cols, uint64_t quant_pre) {
    using namespace map;
    constexpr uint32_t L1A = UsePing ? L1_A_PING : L1_A_PONG;
    constexpr uint32_t L1B = UsePing ? L1_B_PING : L1_B_PONG;
    constexpr uint32_t L0A = UsePing ? L0A_PING : L0A_PONG;
    constexpr uint32_t L0B = UsePing ? L0B_PING : L0B_PONG;
    constexpr uint32_t L0C = UsePing ? L0C_PING : L0C_PONG;

    auto l1a = reinterpret_cast<__cbuf__ half *>(static_cast<uintptr_t>(L1A));
    auto l1b = reinterpret_cast<__cbuf__ half *>(static_cast<uintptr_t>(L1B));
    auto l0a = reinterpret_cast<__ca__ half *>(static_cast<uintptr_t>(L0A));
    auto l0b = reinterpret_cast<__cb__ half *>(static_cast<uintptr_t>(L0B));
    auto l0c = reinterpret_cast<__cc__ float *>(static_cast<uintptr_t>(L0C));

    //  GM (ND) -> L1 (NZ)
    asc_copy_gm2l1_nd2nz_sync(l1a, gm_a, 1, BLK, BLK, 0, a_rows_stride, BLK, 1, 0);
    asc_copy_gm2l1_nd2nz_sync(l1b, gm_b, 1, BLK, BLK, 0, b_rows_stride, BLK, 1, 0);

    //  L1 -> L0A / L0B。repeat 以 512 B 的分形片为单位: 128*128*2 / 512 = 64。
    asc_copy_l12l0a_sync(l0a, l1a, 0, BLK * BLK * 2 / 512, 1, 0);
    asc_copy_l12l0b_sync(l0b, l1b, 0, BLK * BLK * 2 / 512, 1, 0);

    //  128^3。c_matrix_init_val=true 表示本次把 L0C 初始化为 0 (不累加)。
    asc_mmad_sync(l0c, l0a, l0b, BLK, BLK, BLK, 3, false, false, true);

    //  ★T6: fixpipe 在写出的路上做 dtype 转换与 NZ->ND, 不需要额外一趟。
    //    配置走公开 API, 发射仍用内建 (公开 API 没有 L0C->GM 的搬运函数)。
    asc_set_l0c2gm_config(0, quant_pre, true);
    copy_matrix_cc_to_gm(gm_out, l0c, 0, BLK, BLK, out_cols, BLK, 3, quant_pre, 0, false, true);
    (void)BTrans;
}

}  // namespace vc910

#endif  // VC_CUBE_V2_H
