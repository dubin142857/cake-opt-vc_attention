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
constexpr uint32_t FRAC_BYTES = 512;

//  ---- QK: S[MBLK][NBLK] int32 = Q[MBLK][d] · K[NBLK][d]^T --------------------
//  ★B 需要转置: K 在 GM 是 [keys][d] = [n][k], 而 mmad 的 B 要 [k][n]。
//    公开层的 2D asc_copy_l12l0b **没有** transpose 参数; 转置走**另一个函数**
//    asc_copy_l12l0b_trans_sync, 它有 int8_t 重载 (cube_datamove.h:418 节)。
//    实现上它是内建 load_cbuf_to_cb_transpose 的薄包装 (参数 1:1 + asc_sync_post_process)。
__aicore__ inline void mm_qk(__gm__ int32_t *gm_s, __gm__ int8_t *gm_q, __gm__ int8_t *gm_k,
                             uint16_t d, uint16_t q_row_stride, uint16_t k_row_stride,
                             uint16_t s_cols) {
    auto l1a = reinterpret_cast<__cbuf__ int8_t *>(static_cast<uintptr_t>(L1_A_PING));
    auto l1b = reinterpret_cast<__cbuf__ int8_t *>(static_cast<uintptr_t>(L1_B_PING));
    auto l0a = reinterpret_cast<__ca__ int8_t *>(static_cast<uintptr_t>(L0A_PING));
    auto l0b = reinterpret_cast<__cb__ int8_t *>(static_cast<uintptr_t>(L0B_PING));
    auto l0c = reinterpret_cast<__cc__ int32_t *>(static_cast<uintptr_t>(L0C_QK));

    //  GM (ND) -> L1 (NZ)。nd_num=1 表示只搬一个矩阵;
    //  n_value/d_value 是本块的行/列数; src_d_value 是 GM 里相邻行的元素步长。
    asc_copy_gm2l1_nd2nz_sync(l1a, gm_q, 1, MBLK, d, 0, q_row_stride, MBLK, 1, 0);
    asc_copy_gm2l1_nd2nz_sync(l1b, gm_k, 1, NBLK, d, 0, k_row_stride, NBLK, 1, 0);

    asc_copy_l12l0a_sync(l0a, l1a, 0, uint8_t(MBLK * d / FRAC_BYTES), 1, 0);
    asc_copy_l12l0b_trans_sync(l0b, l1b, 0, uint8_t(NBLK * d / FRAC_BYTES), 1, 0, false,
                               uint64_t(0));

    //  参数序: (left_height=m, n_dim=k, right_width=n)。本相 m=MBLK, k=d, n=NBLK。
    //  c_matrix_init_val=true: 本次把 L0C 置零不累加 (QK 的每个块都是独立结果)。
    asc_mmad_sync(l0c, l0a, l0b, MBLK, d, NBLK, 3, false, false, true);

    asc_set_l0c2gm_config(0, 0, true);
    copy_matrix_cc_to_gm(gm_s, l0c, 0, NBLK, MBLK, s_cols, MBLK, 3, 0, 0, false, true);
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

    asc_copy_l12l0a_sync(l0a, l1a, 0, uint8_t(MBLK * NBLK * 2 / FRAC_BYTES), 1, 0);
    asc_copy_l12l0b_sync(l0b, l1b, 0, uint8_t(NBLK * d * 2 / FRAC_BYTES), 1, 0);

    //  m=MBLK, k=NBLK, n=d。
    asc_mmad_sync(l0c, l0a, l0b, MBLK, NBLK, d, 3, false, false, Init);
}

//  把 chunk 内累好的 O 从 L0C 搬到 GM。公开层没有 L0C->GM 的搬运函数,
//  只有配置函数 asc_set_l0c2gm_config, 故发射仍用内建 (查过 c_api 全树)。
__aicore__ inline void pv_flush(__gm__ float *gm_o, uint16_t d, uint16_t o_cols) {
    auto l0c = reinterpret_cast<__cc__ float *>(static_cast<uintptr_t>(L0C_ACC));
    asc_set_l0c2gm_config(0, 0, true);
    copy_matrix_cc_to_gm(gm_o, l0c, 0, d, MBLK, o_cols, MBLK, 3, 0, 0, false, true);
}

}  // namespace cube2
}  // namespace vc910

#endif  // VC_CUBE_V2_H
