//  Phase 0 去风险探针: 只为**验证本构建链认不认这组 CCE 内建函数**, 不参与计算。
//
//  背景: 手写 mmad 需要 load_cbuf_to_ca / load_cbuf_to_cb / mad / copy_matrix_cc_to_gm
//  与 set_flag / wait_flag。它们在 CANN 9.0.0 的**任何头文件里都没有声明** ——
//  全盘搜只在 TIK 的 Python 侧出现, 说明是 ccec 编译器内建 (类似 __builtin_*)。
//  MindIE Laser 在本机用这套编过 (F164), 但那是它自己的 cmake 链;
//  我们走 pip install ./ascend 的另一套, **不能假定等价** —— 故单独验一次。
//
//  ★本探针有对外入口是故意的: 未被引用的静态函数可能在代码生成前就被优化掉,
//    那样"编过了"什么也不证明。给它一个 __global__ 入口, 保证真的走完代码生成。
//    它永远不会被发射。
#ifndef VC_INTRINSIC_PROBE_H
#define VC_INTRINSIC_PROBE_H

#include "vc_l1l0_map.h"

__aicore__ inline void vc_intrinsic_probe_body(__gm__ half *out, __gm__ half *src) {
    using namespace vc910::map;
    //  按新地址表取裸指针 —— 这正是 T4 要替掉 TPipe 的那一步。
    auto l1_a = reinterpret_cast<__cbuf__ half *>(static_cast<uintptr_t>(L1_A_PING));
    auto l1_b = reinterpret_cast<__cbuf__ half *>(static_cast<uintptr_t>(L1_B_PING));
    auto l0_a = reinterpret_cast<__ca__ half *>(static_cast<uintptr_t>(L0A_PING));
    auto l0_b = reinterpret_cast<__cb__ half *>(static_cast<uintptr_t>(L0B_PING));
    auto l0_c = reinterpret_cast<__cc__ float *>(static_cast<uintptr_t>(L0C_PING));

    //  L1 -> L0A / L0B
    load_cbuf_to_ca(l0_a, l1_a, 0, BLK / 16, 1, 0, 0, false, inc);
    load_cbuf_to_cb(l0_b, l1_b, 0, BLK / 16, 1, 0, 0, false, inc);
    set_flag(PIPE_MTE1, PIPE_M, 0);
    wait_flag(PIPE_MTE1, PIPE_M, 0);
    //  128x128x128 的一次矩乘
    mad(l0_c, l0_a, l0_b, BLK, BLK, BLK, 3, 0, 0, 1);
    set_flag(PIPE_M, PIPE_FIX, 0);
    wait_flag(PIPE_M, PIPE_FIX, 0);
    //  ★T6 的机制: fixpipe 在写出的路上做 fp32->fp16 + NZ->ND, 不需要额外一趟。
    copy_matrix_cc_to_gm(out, l0_c, 0, BLK, BLK, BLK, BLK, 3, F322F16, 0, false, true);
    (void)src;
}

#endif  // VC_INTRINSIC_PROBE_H
