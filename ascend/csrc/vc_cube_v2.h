//  CubeV2 —— 手写 cube 的矩乘原语 (M=64 查询行 x N=128 键)。
//
//  为什么另起一个类而不是改 Cube:
//    旧 Cube 的两个 MatmulImpl 经 TPipe 自己占 L0, 与裸地址并存就是"同一个核里
//    两套分配权威", 正是此前那个竞态的形态。故 V2 独占整核, 旧 Cube 留作对照。
//
//  ★几何 M=64 x N=128 的依据见 vc_l1l0_map.h 的注释 (UB / Q 量化粒度 / host 的 tile_m 检查),
//    L0 预算由该文件的 static_assert 守住。**前提是两相按 chunk 分阶段**, 不是逐 tile 交替。
//
//  ★为什么要有数值探针 (vc_cube_probe_entry):
//    公开 API 有声明, 编译器能查**类型**, 但 repeat / src_stride / dst_gap /
//    dst_frac_stride 这几个参数的**语义**仍要确证。CANN 树里没有这几个函数的调用样例
//    (查过), 所以语义只能从内建推 —— 而"推"正是 F183 里烧掉三次构建的失败模式。
//    对策: 把原语单独跑在固定小输入上, 结果写 GM, 主机侧与 NumPy 参考比对。
//    猜错在**一次构建 + 一次运行**里暴露, 而不是污染整个内核后去 141 个用例里排查。
#ifndef VC_CUBE_V2_H
#define VC_CUBE_V2_H

//  ★公开 C API 的入口。内建 (copy_matrix_cc_to_gm 等) 不需要 include, 所以此前
//    一直编得过; asc_* 这一套需要 —— 第一次改用公开 API 时漏了它, 报
//    "use of undeclared identifier 'asc_mmad_sync'"。
#include "c_api/asc_simd.h"
#include "vc_l1l0_map.h"

namespace vc910 {
namespace cube2 {

using namespace map;

//  L1/L0 分区。QK 与 PV **分时**使用, 故可以各自占满;
//  PV 的累加器必须跨 chunk 内的多次 mad 存活, 所以它固定在 L0C_PING,
//  QK 的输出每块都搬走, 用 L0C_PONG。两者不同时存活, 但分开写地址更难出错。
constexpr uint32_t L0C_ACC = L0C_PING;    // PV 累加器 (chunk 内常驻)
constexpr uint32_t L0C_QK  = L0C_PONG;    // QK 的 S, 每块搬走

//  分形片大小: L1->L0 的 repeat 以 512 B 为单位 (与 mm128 初版一致, 待探针确证)。
constexpr uint32_t FRAC_BYTES = 512;   // 分形恒 512 B, 与 dtype 无关
constexpr uint16_t C0_INT8 = 32;       // 32 B / 1 B
constexpr uint16_t C0_HALF = 16;       // 32 B / 2 B

//  ---- unit flag: 本相一律**关闭** ------------------------------------------
//  ★2026-10-29 实撞, 代价一次 300 秒挂死: 初版给 mad 传 unit_flag=3 且
//    asc_set_l0c2gm_config 的第三参 enable_unit_flag 传 true, 而 PV 要跨 chunk 累加,
//    只在末尾搬一次 —— 于是"每次 mad 都宣告自己是最后一次"与"只有一次 fixpipe"
//    对不上, 内核在设备上挂住 (无 plog, 纯挂, 靠 timeout 124 才发现)。
//  ★CANN 自己的取值在 matmul_unit_flag.h:34 —— `isLast ? UNIT_FLAG_SET(3) : UNIT_FLAG_CHECK(2)`,
//    而关闭时 `return 0`。即 **0 是合法的"关"**, 且 Laser 的 cube2 也是
//    `last_k ? 3 : 2` —— 只有紧跟 fixpipe 的那次才用 3。
//  本相用 _sync 变体, 同步由它负责, 所以直接全关: 少一整类协议错配。
//  Phase 2 换手动同步、要靠 unit flag 让 fixpipe 提前起跑时, 再按"仅末次为 3"引入。
constexpr uint8_t UNIT_FLAG_OFF = 0;

//  ---- QK 的软件流水 ---------------------------------------------------------
//  ★为什么拆成四段: F191 实测 chunkv2 的四流水占比之和只有 **0.96** (平均连一条流水
//    都没占满, 即完全串行), 窗口/binding 由基线的 1.164 劣化到 **2.322** ——
//    而两条流量指标都已达预期 (fixpipe 186.0 GB vs 预测 186.3)。
//    **流量对了而时间没好 = 问题在重叠不在流量**, 故本步只动重叠。
//    成因是每次搬运都用 _sync 变体 (内嵌同步, 每次自等自己)。
//
//  ★同步用公开的 asc_sync_notify / asc_sync_wait (c_api/sync/sync.h 的宏,
//    直通 set_flag / wait_flag) —— **不再用未声明的内建**。
//  ★事件号的第三参类型: 取 **uint32_t 运行时值**。F183 当时试 uint8_t 与
//    constexpr AscendC::event_t 都报 "must be a type 'event_t'", 烧掉三次构建而未解;
//    答案在 Laser 的声明里 —— `uint32_t ping_pong_flag_l1_a_ = 0`, 且是运行时值。
//  ★事件号按 (from,to) pipe 对分作用域, 见 vc_l1l0_map.h 的 EV_GROUPS。
//
//  四段与它们守的缓冲:
//    gm2l1 : MTE2 写 L1   -> notify(MTE2,MTE1)      ; 先 wait(MTE1,MTE2) 确认槽可覆写
//    l12l0 : MTE1 写 L0   -> notify(MTE1,M)         ; 先 wait(MTE2,MTE1) 与 wait(M,MTE1)
//    mad   : M   写 L0C   -> notify(M,FIX)+notify(M,MTE1); 先 wait(MTE1,M) 与 wait(FIX,M)
//    fix   : FIX 读 L0C   -> notify(FIX,M)          ; 先 wait(M,FIX)
//  调用方把第 j+1 块的 gm2l1 发在第 j 块的 l12l0/mad/fix **之前**, 重叠即由此而来。

//  ★Q 与 K 分开装载: Q 每**任务**一次 (常驻 L1_Q), K 每**键块**一次 (双缓冲)。
//    原先两者合在一起, 于是 Q 跟着 K 被重复读了 tiles 次 —— 纯浪费的 GM->L1 流量。
//  nd_num=1 只搬一个矩阵; n_value/d_value 是本块行/列数; src_d_value 是 GM 相邻行的元素步长。
__aicore__ inline void qk_gm2l1_q(__gm__ int8_t *gm_q, uint16_t d, uint16_t q_row_stride) {
    auto l1a = reinterpret_cast<__cbuf__ int8_t *>(static_cast<uintptr_t>(L1_Q));
    asc_copy_gm2l1_nd2nz(l1a, gm_q, 1, MBLK, d, 0, q_row_stride, MBLK, 1, 0);
}

__aicore__ inline void qk_gm2l1_k(uint32_t ping, __gm__ int8_t *gm_k,
                                  uint16_t d, uint16_t k_row_stride) {
    auto l1b = reinterpret_cast<__cbuf__ int8_t *>(
        static_cast<uintptr_t>(ping ? L1_B_PING : L1_B_PONG));
    asc_copy_gm2l1_nd2nz(l1b, gm_k, 1, NBLK, d, 0, k_row_stride, NBLK, 1, 0);
}

__aicore__ inline void qk_l12l0(uint32_t ping, uint16_t d) {
    //  ★A 侧读的是**常驻**的 L1_Q, 与 ping 无关; 只有 B(K) 随键块双缓冲。
    auto l1a = reinterpret_cast<__cbuf__ int8_t *>(static_cast<uintptr_t>(L1_Q));
    auto l1b = reinterpret_cast<__cbuf__ int8_t *>(
        static_cast<uintptr_t>(ping ? L1_B_PING : L1_B_PONG));
    auto l0a = reinterpret_cast<__ca__ int8_t *>(
        static_cast<uintptr_t>(ping ? L0A_PING : L0A_PONG));
    auto l0b = reinterpret_cast<__cb__ int8_t *>(
        static_cast<uintptr_t>(ping ? L0B_PING : L0B_PONG));
    //  ★repeat 单位是**分形** (恒 512 B), 且 A 必须**按 m 方向分块多次调用** ——
    //    写成单次调用会使 mad 对多于 d 项求和 (实测 S 量级大十倍)。
    //    参数取自 CANN 自己的 load_to_l0a_load2dV2.h:132 (权威, 不是从 Laser 反推):
    //    mStep=madM/16 · kStep=madK/c0 · srcStride=aL1M/16; v1 的 API 一次只做一个 m 块。
    //  ★int8 的 c0 是 32 (分形 16x32); half 的 c0 是 16 (分形 16x16)。
    for (uint16_t mm = 0; mm < MBLK / 16; ++mm)
        asc_copy_l12l0a(l0a + mm * d * 16, l1a + mm * 16 * C0_INT8, 0,
                        uint8_t(d / C0_INT8), uint16_t(MBLK / 16), 0);
    //  ★B 用**普通** (非转置) 装载, 尽管 K 在逻辑上是转置操作数 —— 硬件标志与逻辑含义相反。
    //    两个独立来源一致: CANN 的 load_to_l0b_load2dV2.h:71 在 isBTranspose 为真时设
    //    ifTranspose=false; Laser 的 cube1 给 K 传的也是 transpose=false。
    asc_copy_l12l0b(l0b, l1b, 0, uint8_t(NBLK * d / (16 * C0_INT8)), 1, 0);
}

__aicore__ inline void qk_mad(uint32_t ping, uint16_t d) {
    auto l0a = reinterpret_cast<__ca__ int8_t *>(
        static_cast<uintptr_t>(ping ? L0A_PING : L0A_PONG));
    auto l0b = reinterpret_cast<__cb__ int8_t *>(
        static_cast<uintptr_t>(ping ? L0B_PING : L0B_PONG));
    auto l0c = reinterpret_cast<__cc__ int32_t *>(
        static_cast<uintptr_t>(ping ? L0C_PING : L0C_PONG));
    //  参数序: (left_height=m, n_dim=k, right_width=n)。本相 m=MBLK, k=d, n=NBLK。
    //  c_matrix_init_val=true: QK 的每块都是独立结果, 不累加。
    //  unit_flag 一律 0: 本内核不开 unit flag (开了而只在末尾 fixpipe 会挂死, 见 F190 之前那条)。
    asc_mmad(l0c, l0a, l0b, MBLK, d, NBLK, UNIT_FLAG_OFF, false, false, true);
}

__aicore__ inline void qk_fix(uint32_t ping, __gm__ int32_t *gm_s, uint16_t s_cols) {
    auto l0c = reinterpret_cast<__cc__ int32_t *>(
        static_cast<uintptr_t>(ping ? L0C_PING : L0C_PONG));
    //  第三参是 enable_unit_flag —— 必须 false。QuantPRE 用内建枚举 NoQuant 而非字面 0。
    asc_set_l0c2gm_config(0, NoQuant, false);
    copy_matrix_cc_to_gm(gm_s, l0c, 0, NBLK, MBLK, s_cols, MBLK,
                         UNIT_FLAG_OFF, NoQuant, 0, false, true);
}

//  ★给数值探针用的串行包装 —— 四段按序跑完, 段间用公开的 asc_sync() 保守隔开。
//    它不进主路径 (主路径走上面四段的流水), 只为让 vc910_cube_v2_probe 这条
//    已验证的回归检查继续可用: 改流水最容易出的是**间歇性竞态**, 而探针对精确
//    int32 参考零误差这一条, 是判断"搬运语义本身没被改坏"的独立证据。
__aicore__ inline void mm_qk(__gm__ int32_t *gm_s, __gm__ int8_t *gm_q, __gm__ int8_t *gm_k,
                             uint16_t d, uint16_t q_row_stride, uint16_t k_row_stride,
                             uint16_t s_cols) {
    qk_gm2l1_q(gm_q, d, q_row_stride);
    qk_gm2l1_k(1, gm_k, d, k_row_stride);
    asc_sync();
    qk_l12l0(1, d);
    asc_sync();
    qk_mad(1, d);
    asc_sync();
    qk_fix(1, gm_s, s_cols);
    asc_sync();
}

//  ---- PV: O[MBLK][d] fp32 (+)= P[MBLK][NBLK] · V[NBLK][d] --------------------
//  ★Init=false 时 c_matrix_init_val=false, 即**累加进 L0C 已有内容**。
//    这是 chunk 内跨 key 块累加的全部机制 —— O 因此只在 chunk 末搬出一次,
//    而不是每个 key 块一次 (F186/F189: 后者使 O 的 fixpipe 字节翻倍)。
//  ★V 不转置: V 在 GM 是 [keys][d] = [k][n], 正是 mmad 的 B 所要的布局。
template <bool Init>
__aicore__ inline void mm_pv_acc(__gm__ half *gm_p, __gm__ half *gm_v,
                                 uint16_t d, uint16_t p_row_stride, uint16_t v_row_stride) {
    auto l1a = reinterpret_cast<__cbuf__ half *>(static_cast<uintptr_t>(L1_A_PONG));
    auto l1b = reinterpret_cast<__cbuf__ half *>(static_cast<uintptr_t>(L1_B_PONG));
    auto l0a = reinterpret_cast<__ca__ half *>(static_cast<uintptr_t>(L0A_PING));
    auto l0b = reinterpret_cast<__cb__ half *>(static_cast<uintptr_t>(L0B_PING));
    auto l0c = reinterpret_cast<__cc__ float *>(static_cast<uintptr_t>(L0C_ACC));

    asc_copy_gm2l1_nd2nz_sync(l1a, gm_p, 1, MBLK, NBLK, 0, p_row_stride, MBLK, 1, 0);
    asc_copy_gm2l1_nd2nz_sync(l1b, gm_v, 1, NBLK, d, 0, v_row_stride, NBLK, 1, 0);

    //  A (P, half, c0=16): 与 QK 同法按 m 分块。
    for (uint16_t mm = 0; mm < MBLK / 16; ++mm)
        asc_copy_l12l0a_sync(l0a + mm * NBLK * 16, l1a + mm * 16 * C0_HALF, 0,
                             uint8_t(NBLK / C0_HALF), uint16_t(MBLK / 16), 0);
    //  ★B (V, half): V 在逻辑上**不**转置, 故硬件要 ifTranspose=true —— 与 K 那边相反。
    //    依据同样是两个独立来源: CANN 的 load_to_l0b_load2dV2.h:85 在 isBTranspose 为假时
    //    走的那一支要求 transpose; Laser 的 cube2 给 V 传的正是 transpose=true
    //    (CubeForward.h:275)。公开层 2D 的 asc_copy_l12l0b 把 transpose 硬编码为 false,
    //    做不到, 只能走 asc_copy_l12l0b_trans_sync。
    //    形参按 Laser cube2 的循环结构取: 每个 k 块一次调用, repeat = d/16, srcStride = NBLK/16。
    //  ★这一处用**内建**而不是公开 API, 是穷尽公开层之后的决定 (F183 定的次序):
    //    公开层对"2D 分形装载 + 转置"**没有覆盖** ——
    //      2D 的 asc_copy_l12l0b 把 transpose 硬编码为 false (asc_copy_l12l0b_2d_impl.h:41);
    //      3D 的 asc_copy_l12l0b 映射到 img2colv2_cbuf_to_cb, 是卷积 img2col 通路, 不适用;
    //      asc_copy_l12l0b_trans 映射到 load_cbuf_to_cb_**transpose**, 与这里要的
    //        load_cbuf_to_cb(..., transpose=true, ...) 是**两条不同指令**, 形参语义不同 ——
    //        照搬后者的参数给前者, 实测报
    //        "The read address of the mte load2d instruction is greater than the maximum
    //         address of the source (L1)" (aicore exception 507015)。
    //    而本内建正是公开 2D 包装自己调用的那一条, 只是 transpose 标志取 true;
    //    参数取自 Laser 的 cube2 (CubeForward.h:268, 同为 half、同为 128 键块) —— 有工作样例, 不是猜。
    for (uint16_t kk = 0; kk < NBLK / 16; ++kk)
        load_cbuf_to_cb(l0b + kk * d * 16, l1b + kk * 16 * C0_HALF, 0,
                        uint8_t(d / 16), uint16_t(NBLK / 16), 0, 0, true, inc);
    //  内建没有 _sync 变体, 公开层的 asc_sync_post_process 也只在 impl 层。
    //  本相用公开的 asc_sync() 保守补上; Phase 2 换手动细粒度同步时一并处理。
    asc_sync();

    //  m=MBLK, k=NBLK, n=d。
    asc_mmad_sync(l0c, l0a, l0b, MBLK, NBLK, d, UNIT_FLAG_OFF, false, false, Init);
}

//  把 chunk 内累好的 O 从 L0C 搬到 GM。公开层没有 L0C->GM 的搬运函数,
//  只有配置函数 asc_set_l0c2gm_config, 故发射仍用内建 (查过 c_api 全树)。
__aicore__ inline void pv_flush(__gm__ float *gm_o, uint16_t d, uint16_t o_cols) {
    auto l0c = reinterpret_cast<__cc__ float *>(static_cast<uintptr_t>(L0C_ACC));
    asc_set_l0c2gm_config(0, NoQuant, false);
    copy_matrix_cc_to_gm(gm_o, l0c, 0, d, MBLK, o_cols, MBLK,
                         UNIT_FLAG_OFF, NoQuant, 0, false, true);
}

}  // namespace cube2
}  // namespace vc910

#endif  // VC_CUBE_V2_H
