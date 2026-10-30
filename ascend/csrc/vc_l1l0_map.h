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

//  ---- chunk 深度 -----------------------------------------------------------
//  ★一个 chunk 内的全部 PV 在 L0C 上累加, 只在 chunk 末搬出一次 O。
//    取 4 是算出来的不是拍的 (见 problem.yaml F189 之后的 chunk 权衡表):
//      CHUNK=1 -> fixpipe 297.4 GB (= N 减半使 O 字节翻倍, 与 N=128 实测的 297.4 吻合)
//      CHUNK=2 -> 223.3 GB (≈ 基线 223.8, 无收益)
//      CHUNK=4 -> 186.3 GB, 向量侧 UB 141.1/192 KB  ★取它
//      CHUNK=8 -> 167.8 GB 但 UB 205.6 KB **超容**
//    上限来自向量侧: chunk 内要取**单一 rowmax**, 故整个 chunk 的 S 必须留在 UB。
constexpr uint32_t CHUNK = 4;
static_assert(CHUNK >= 2, "CHUNK=1 等于每个 key 块搬一次 O, 即单独改几何那个回退点");

//  ---- 逐相 L0 预算 -------------------------------------------------------
//  ★上面四条断言说的是 fp16 的情形。本算子两相的 dtype **不同**:
//      QK 相: int8 x int8 -> int32   (Q/K 在算子入口就是 int8, 是接口契约)
//      PV 相: half x half -> fp32
//  两相对 L0 的要求因此不同, 必须分别断言, 否则"128x128 正好贴合"只对其中一相成立。
//
//  ★更要紧的是**前提**: 下面按"同一时刻只有一相占 L0"来算。这要求 QK 与 PV
//    **按 chunk 分阶段** (先算完一个 chunk 的全部 QK, 向量做完 softmax, 再做该 chunk 的 PV),
//    而不是逐 tile 交替。前提不成立时要算的是**两相之和**, 下面 _INTERLEAVED_* 给出那个数,
//    它正是旧结构 (两个 MatmulImpl 逐 tile 交替) 装不下双缓冲的原因 —— 留在这里当反例。
constexpr uint32_t D_MAX = 128;      // 契约: D 只能是 64 或 128 (vc_torch.cpp 的 TORCH_CHECK)
constexpr uint32_t D_MIN = 64;

//  ---- 块形状: M=64 行 x N=128 键, **不是方块** ----
//  ★为什么不用 Laser 的 128x128 方块, 三条约束都是量出来的:
//    (1) UB: M=128 时 VectorBytes 要 184.5 KB, 而本机 ub_size=196608 (192 KB),
//        余量只有 7.5 KB 且还没算 reduce_bytes (运行时由 GetReduceSumMaxMinTmpSize 定)。
//        M=64 只要 92.8 KB。host 的 TORCH_CHECK(VectorBytes(...) <= ub) 会在 M=128 上擦边。
//    (2) Q 量化粒度: Q_SCALE_BLOCK=32, M=64 使每个 AIV 半区 ROWS=32 **恰好一个量化块**,
//        向量侧 qscale.GetValue(0) 的写法成立; M=128 则每半区两个块, scale 逻辑要重写。
//    (3) host 只接受 tile_m ∈ {32,64} (vc_torch.cpp:101)。
//  ★代价: 每 MAC 的 L1->L0 流量是方块的 1.5 倍 (MACs/载入字节 = M*N/(M+N): 42.7 vs 64)。
//    可接受的依据是实测 aic_mte1_ratio=0.215, 远不是 binding (binding 是 scalar 0.859)。
constexpr uint32_t MBLK = 64;    // 查询行
constexpr uint32_t NBLK = 128;   // 键
static_assert(MBLK % FRACTAL == 0 && NBLK % FRACTAL == 0, "块边长必须是 16x16 分形的整数倍");
static_assert(NBLK == BLK, "键方向仍用 BLK, 便于与 L1 槽和 K 量化块(128)对齐");
static_assert(NBLK % 128 == 0, "一个键块必须覆盖整数个 K 量化块 (K_SCALE_BLOCK=128)");

//  单缓冲一块需要多少字节 (按 D 参数化)
constexpr uint32_t qk_a(uint32_t d) { return MBLK * d * 1; }     // Q tile, int8
constexpr uint32_t qk_b(uint32_t d) { return NBLK * d * 1; }     // K tile, int8
constexpr uint32_t qk_c(uint32_t)   { return MBLK * NBLK * 4; }  // S tile, int32
constexpr uint32_t pv_a(uint32_t)   { return MBLK * NBLK * 2; }  // P tile, half
constexpr uint32_t pv_b(uint32_t d) { return NBLK * d * 2; }     // V tile, half
constexpr uint32_t pv_c(uint32_t d) { return MBLK * d * 4; }     // O tile, fp32

//  分时 + 双缓冲: 取两相的较大者再乘 2
constexpr uint32_t _max2(uint32_t a, uint32_t b) { return a > b ? a : b; }
constexpr uint32_t staged_l0a(uint32_t d) { return 2 * _max2(qk_a(d), pv_a(d)); }
constexpr uint32_t staged_l0b(uint32_t d) { return 2 * _max2(qk_b(d), pv_b(d)); }
constexpr uint32_t staged_l0c(uint32_t d) { return 2 * _max2(qk_c(d), pv_c(d)); }

static_assert(staged_l0a(D_MAX) <= L0A_SIZE, "D=128 分时双缓冲必须装得下 L0A");
static_assert(staged_l0b(D_MAX) <= L0B_SIZE, "D=128 分时双缓冲必须装得下 L0B");
static_assert(staged_l0c(D_MAX) <= L0C_SIZE, "D=128 分时双缓冲必须装得下 L0C");
static_assert(staged_l0a(D_MIN) <= L0A_SIZE, "D=64 分时双缓冲必须装得下 L0A");
static_assert(staged_l0b(D_MIN) <= L0B_SIZE, "D=64 分时双缓冲必须装得下 L0B");
static_assert(staged_l0c(D_MIN) <= L0C_SIZE, "D=64 分时双缓冲必须装得下 L0C");
//  ★D=128 时 L0B 正好占满 (PV 的 V 块 32 KB x2); L0A 与 L0C 各半空 ——
//    那半空是**余量不是浪费**, 留给后续把缓冲加深或让第二个查询块的 O 也驻留。
//    把它写成断言, 以免将来有人"顺手"把块加大而不知道 L0B 已经没有余地。
static_assert(staged_l0b(D_MAX) == L0B_SIZE, "★L0B 在 D=128 上已无余量, 加大键块会溢出");
static_assert(staged_l0a(D_MAX) * 2 == L0A_SIZE, "L0A 现用一半, 余量留痕");
static_assert(staged_l0c(D_MAX) * 2 == L0C_SIZE, "L0C 现用一半, 余量留痕");

//  ★反例 (留痕, 不是死代码): 逐 tile 交替时要算两相之和。旧结构 tile_m=64 / N=256 下
//    L0B 需 QK 32 KB + PV 64 KB = 96 KB > 64 KB, 两相**物理上不能共存** ——
//    这正是 vc_torch.cpp 写死 dbL0A/B/C=1 与内核里那道 PipeBarrier<PIPE_ALL>() 的根源
//    (F188: 无双缓冲 -> 流水不重叠 -> 标量卡在矩乘内部 flag 上空转, scalar 成为 binding pipe)。
//    本文件的几何之所以成立, **前提就是分阶段**; 谁把它改回逐 tile 交替, 下面这条会立刻变红。
constexpr uint32_t OLD_N = 256;   // 旧几何的键块宽度
constexpr uint32_t interleaved_l0b_old(uint32_t d) { return 2 * (OLD_N * d * 1 + OLD_N * d * 2); }
static_assert(interleaved_l0b_old(D_MAX) > L0B_SIZE,
              "★旧几何 (N=256, 逐 tile 交替) 的 L0B 需求应当超容 —— 这条是**反例对照**, "
              "它变绿说明这里的算法写错了, 而不是说明旧几何没问题");

//  ---- L1 布局: 512 KB 切成 8 个 64 KB 槽 ----
//  Phase 0 只用前四个 (A/B 各双缓冲); 后四个**预留**给 Phase 3 的对角阵与累加器驻留,
//  现在就把地址定下来, 免得那时再搬动前四个。
constexpr uint32_t L1_SLOT = 65536;
constexpr uint32_t L1_A_PING = 0 * L1_SLOT;
constexpr uint32_t L1_A_PONG = 1 * L1_SLOT;
constexpr uint32_t L1_B_PING = 2 * L1_SLOT;
constexpr uint32_t L1_B_PONG = 3 * L1_SLOT;
//  ★Q 常驻: Q 在一个任务的**所有键块上都相同**, 故每任务只装一次, 不随键块双缓冲。
//    省下的 GM->L1 流量 = Q 片 8192 B x (tiles-1) x 任务数; 37741^2/28 头上约 40 GB,
//    占 GM_to_L1 334.6 GB 的 12% (F196: mte2 是 binding pipe, 减输入流量直接减它)。
//    复用 A 的 PING 槽: A 不再需要双缓冲, 而 L1_A_PONG 正好留给 PV 的 P 片, 冲突反而少一处。
constexpr uint32_t L1_Q = L1_A_PING;
constexpr uint32_t L1_DIAG_PING = 4 * L1_SLOT;
constexpr uint32_t L1_DIAG_PONG = 5 * L1_SLOT;
constexpr uint32_t L1_ACC_PING  = 6 * L1_SLOT;
constexpr uint32_t L1_ACC_PONG  = 7 * L1_SLOT;

//  ---- PV 相的 P/V 双缓冲 (F209) -------------------------------------------
//  ★PV 相原先用 L1_A_PONG(P) 与 L1_B_PONG(V), 后者与 QK 的 K 槽**冲突**, 只能靠
//    chunk 末的全屏障挡着。改用后四个槽 (原 Phase 3 预留) 之后, PV 与 QK 的 L1
//    **完全不相交**, 那道交接少一项。
//  ★尺寸: P 片 MBLK*NBLK*2 = 16384 B · V 片 NBLK*D_MAX*2 = 32768 B, 均 < L1_SLOT。
constexpr uint32_t L1_PV_P_PING = L1_DIAG_PING;
constexpr uint32_t L1_PV_P_PONG = L1_DIAG_PONG;
constexpr uint32_t L1_PV_V_PING = L1_ACC_PING;
constexpr uint32_t L1_PV_V_PONG = L1_ACC_PONG;
static_assert(MBLK * NBLK * 2 <= L1_SLOT, "PV 的 P 片必须装得进一个 L1 槽");
static_assert(NBLK * D_MAX * 2 <= L1_SLOT, "PV 的 V 片必须装得进一个 L1 槽");

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


//  ---- 核内事件号 (按 **pipe 对** 分作用域) --------------------------------
//  ★为什么也要守: 此前那个竞态的另一半正是**事件号共用** —— 两个 MatmulImpl 的
//    M_MTE1 事件号由 TPipe 各自分配而源码里看不出来。CubeV2 独占整核, 事件号由我们自己管;
//    自己管就必须自己守, 否则只是把一个看不见的坑换成另一个。
//
//  ★2026-10-29 修正了一个**过约束**: 原来是 8 个号的扁平表 + 全局唯一断言。
//    实测 Laser 在 `MTE2->MTE1` 与 `MTE1->M` 上**都用了 {0,1}** (CubeForward.h),
//    即事件号按 **(from, to) pipe 对**分作用域, 同一对内才必须互异。
//    扁平唯一会让号不够用 —— 软件流水要六个 pipe 对各自的 ping/pong。
//  ★DAV-2201 的核内事件号是 3 位, 每个 pipe 对的合法范围都是 0..7。
constexpr uint8_t EVENT_MAX = 7;

//  每个 pipe 对内的号分配。命名按"谁等谁 + 守哪块缓冲", 不按用途 —— 用途会变, 流水关系不会。
//    MTE2->MTE1 / MTE1->MTE2 : L1 的 A/B 槽 (装好 / 可覆写)
//    MTE1->M    / M->MTE1    : L0 的 A/B 槽 (装好 / 可覆写)
//    M->FIX     / FIX->M     : L0C      (算完 / 可覆写)
constexpr uint8_t EV_L1A_PING = 0, EV_L1A_PONG = 1;   // 对: MTE2<->MTE1
constexpr uint8_t EV_L1B_PING = 2, EV_L1B_PONG = 3;   // 同上, 与 A 同对故须错开
constexpr uint8_t EV_L0A_PING = 0, EV_L0A_PONG = 1;   // 对: MTE1<->M
constexpr uint8_t EV_L0B_PING = 2, EV_L0B_PONG = 3;   // 同上, 与 L0A 同对故须错开
//  PV 的 P/V 双缓冲与 Q/K 同在 MTE2<->MTE1 对上, 故接着往下排, **正好填满 0..7**。
constexpr uint8_t EV_PVP_PING = 4, EV_PVP_PONG = 5;   // 对: MTE2<->MTE1
constexpr uint8_t EV_PVV_PING = 6, EV_PVV_PONG = 7;   // 同上
constexpr uint8_t EV_L0C_PING = 0, EV_L0C_PONG = 1;   // 对: M<->FIX

//  ★守的是**每个 pipe 对内**的互异与范围。跨对复用是合法的 (上面 L1A 与 L0A 都是 0),
//    所以不能再用一张扁平表做全局唯一 —— 那正是被修掉的过约束。
//  ★由四个号扩到八个: MTE2<->MTE1 这一对现在同时服务 Q/K 与 PV 的 P/V, 共八个号。
//    若仍只存四个, 唯一性检查就**看不见后面四个**, 撞号时它会静默放行。
struct EvGroup { const char *pair; uint8_t id[8]; uint8_t n; };
constexpr EvGroup EV_GROUPS[] = {
    {"MTE2<->MTE1 (L1)", {EV_L1A_PING, EV_L1A_PONG, EV_L1B_PING, EV_L1B_PONG,
                          EV_PVP_PING, EV_PVP_PONG, EV_PVV_PING, EV_PVV_PONG}, 8},
    {"MTE1<->M (L0)",    {EV_L0A_PING, EV_L0A_PONG, EV_L0B_PING, EV_L0B_PONG}, 4},
    {"M<->FIX (L0C)",    {EV_L0C_PING, EV_L0C_PONG}, 2},
};
constexpr uint32_t EV_GROUP_N = sizeof(EV_GROUPS) / sizeof(EvGroup);

constexpr uint8_t _ev_at(const EvGroup &g, uint8_t i) { return g.id[i]; }
constexpr bool _ev_group_unique(const EvGroup &g, uint8_t i = 0, uint8_t j = 1) {
    return i >= g.n ? true
         : j >= g.n ? _ev_group_unique(g, i + 1, i + 2)
         : (_ev_at(g, i) != _ev_at(g, j) && _ev_group_unique(g, i, j + 1));
}
constexpr bool _ev_group_in_range(const EvGroup &g, uint8_t i = 0) {
    return i >= g.n ? true : (_ev_at(g, i) <= EVENT_MAX && _ev_group_in_range(g, i + 1));
}
constexpr bool _ev_all_unique(uint32_t k = 0) {
    return k >= EV_GROUP_N ? true : (_ev_group_unique(EV_GROUPS[k]) && _ev_all_unique(k + 1));
}
constexpr bool _ev_all_in_range(uint32_t k = 0) {
    return k >= EV_GROUP_N ? true : (_ev_group_in_range(EV_GROUPS[k]) && _ev_all_in_range(k + 1));
}
static_assert(_ev_all_unique(), "★同一个 pipe 对里有两个事件号撞了 —— 这正是此前竞态的成因形态");
static_assert(_ev_all_in_range(), "★事件号超出 0..7");
static_assert(EV_GROUP_N == 3, "新增 pipe 对后请同步更新本计数, 否则唯一性检查会漏看它");

}  // namespace map
}  // namespace vc910

#endif  // VC_L1L0_MAP_H
