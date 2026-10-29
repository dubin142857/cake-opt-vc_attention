//  片上缓存的**显式**地址表 —— Phase 0 (按 Laser 数据流重写) 的第一块。
//
//  为什么要有它:
//    此前用 TPipe/TBuf 分配, 两个 MatmulImpl 共用物理 L0A/L0B 而**源码里看不出来**,
//    于是竞态在 tile_m=32 上 100% 发作却查了整整一轮 (见 problem.yaml 的竞态条目)。
//    TPipe 的越界断言只在 ASCENDC_CPU_DEBUG 下存在, 真机上不设防。
//    ★手写地址的代价是失去分配器的守护, 收益是**重叠变得可见**。本文件把守护补回来:
//    所有区间用 constexpr 写死, 用 static_assert 让**编译器**拦住越界与重叠。
//
//  容量来源: 本机 /usr/local/Ascend/cann-9.0.0/aarch64-linux/data/platform_config/
//            Ascend910B2.ini 的 [AICoreSpec] 段 (l0_a_size / l0_b_size / l0_c_size /
//            l1_size / cube_m_size)。**不照抄任何外部实现**。
#ifndef VC_L1L0_MAP_H
#define VC_L1L0_MAP_H

#include <cstdint>

namespace vc910 {
namespace map {

//  ---- 片上容量 (Ascend910B2.ini [AICoreSpec]) ----
constexpr uint32_t L0A_SIZE = 65536;      // l0_a_size
constexpr uint32_t L0B_SIZE = 65536;      // l0_b_size
constexpr uint32_t L0C_SIZE = 131072;     // l0_c_size
constexpr uint32_t L1_SIZE  = 524288;     // l1_size
constexpr uint32_t FRACTAL  = 16;         // cube_m_size = cube_n_size = cube_k_size

//  ---- 基本块几何 ----
//  选 128x128 的理由不是"Laser 用它", 而是它在**本机容量上恰好贴合**:
//    128*128*2 (fp16) = 32768 = L0A/L0B 的 ping-pong 半区 (65536/2)
//    128*128*4 (fp32) = 65536 = L0C   的 ping-pong 半区 (131072/2)
//  即每级缓存正好各装一块, 双缓冲无碎片。
constexpr uint32_t BLK = 128;
constexpr uint32_t BLK_ELEMS = BLK * BLK;
constexpr uint32_t BLK_B16 = BLK_ELEMS * 2;      // 32768
constexpr uint32_t BLK_B32 = BLK_ELEMS * 4;      // 65536

static_assert(BLK % FRACTAL == 0, "基本块边长必须是 16x16 分形的整数倍");
static_assert(BLK_B16 * 2 == L0A_SIZE, "128x128 fp16 应恰好是 L0A 的一半 (双缓冲)");
static_assert(BLK_B16 * 2 == L0B_SIZE, "128x128 fp16 应恰好是 L0B 的一半 (双缓冲)");
static_assert(BLK_B32 * 2 == L0C_SIZE, "128x128 fp32 应恰好是 L0C 的一半 (双缓冲)");

//  ---- L0 布局 (各自独立编址, 从 0 起) ----
constexpr uint32_t L0A_PING = 0;
constexpr uint32_t L0A_PONG = BLK_B16;
constexpr uint32_t L0B_PING = 0;
constexpr uint32_t L0B_PONG = BLK_B16;
constexpr uint32_t L0C_PING = 0;
constexpr uint32_t L0C_PONG = BLK_B32;

static_assert(L0A_PONG + BLK_B16 == L0A_SIZE, "L0A 两个半区应正好填满");
static_assert(L0B_PONG + BLK_B16 == L0B_SIZE, "L0B 两个半区应正好填满");
static_assert(L0C_PONG + BLK_B32 == L0C_SIZE, "L0C 两个半区应正好填满");

//  ---- L1 布局: 512 KB 切成 8 个 64 KB 槽 ----
//  Phase 0 只用前四个 (A/B 各双缓冲); 后四个**预留**给 Phase 3 的对角阵与累加器驻留,
//  现在就把地址定下来, 免得那时再搬动前四个。
constexpr uint32_t L1_SLOT = 65536;
constexpr uint32_t L1_A_PING = 0 * L1_SLOT;
constexpr uint32_t L1_A_PONG = 1 * L1_SLOT;
constexpr uint32_t L1_B_PING = 2 * L1_SLOT;
constexpr uint32_t L1_B_PONG = 3 * L1_SLOT;
constexpr uint32_t L1_DIAG_PING = 4 * L1_SLOT;   // Phase 3 预留
constexpr uint32_t L1_DIAG_PONG = 5 * L1_SLOT;   // Phase 3 预留
constexpr uint32_t L1_ACC_PING  = 6 * L1_SLOT;   // Phase 3 预留
constexpr uint32_t L1_ACC_PONG  = 7 * L1_SLOT;   // Phase 3 预留

static_assert(L1_SLOT * 8 == L1_SIZE, "八个槽应正好填满 L1");
static_assert(BLK_B16 <= L1_SLOT, "一个 fp16 基本块必须装得进一个 L1 槽");
static_assert(BLK_B32 <= L1_SLOT, "一个 fp32 基本块必须装得进一个 L1 槽");

//  ---- 重叠检查: 把每个区间写成 (起点, 长度), 让编译器两两比 ----
//  ★这正是 TPipe 没有而我们付过代价的那一层。新增任何区间都要加进本表,
//    否则 _no_overlap 不会看它 —— 这一条靠下面的 _L1_REGIONS 计数守住。
struct Region { uint32_t off; uint32_t len; };
constexpr Region L1_REGIONS[] = {
    {L1_A_PING, L1_SLOT}, {L1_A_PONG, L1_SLOT},
    {L1_B_PING, L1_SLOT}, {L1_B_PONG, L1_SLOT},
    {L1_DIAG_PING, L1_SLOT}, {L1_DIAG_PONG, L1_SLOT},
    {L1_ACC_PING, L1_SLOT}, {L1_ACC_PONG, L1_SLOT},
};
constexpr uint32_t L1_REGION_N = sizeof(L1_REGIONS) / sizeof(Region);

constexpr bool _pair_ok(const Region &a, const Region &b) {
    return a.off + a.len <= b.off || b.off + b.len <= a.off;
}
constexpr bool _no_overlap(uint32_t i = 0, uint32_t j = 1) {
    return i >= L1_REGION_N ? true
         : j >= L1_REGION_N ? _no_overlap(i + 1, i + 2)
         : (_pair_ok(L1_REGIONS[i], L1_REGIONS[j]) && _no_overlap(i, j + 1));
}
constexpr bool _in_bounds(uint32_t i = 0) {
    return i >= L1_REGION_N ? true
         : (L1_REGIONS[i].off + L1_REGIONS[i].len <= L1_SIZE && _in_bounds(i + 1));
}
static_assert(_no_overlap(), "★L1 有两个区间重叠 —— 这正是此前竞态的成因形态");
static_assert(_in_bounds(), "★L1 有区间越界");
static_assert(L1_REGION_N == 8, "新增 L1 区间后请同步更新本计数, 否则重叠检查会漏看");

}  // namespace map
}  // namespace vc910

#endif  // VC_L1L0_MAP_H
