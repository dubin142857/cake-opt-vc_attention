#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CubeV2 手写原语的**数值**探针 —— 确证 stride 语义, 不是跑算子。

★为什么要它: 公开 C API 有声明, 编译器能查**类型**, 但 repeat / src_stride /
  dst_gap / dst_frac_stride 这几个参数的**语义**仍要确证 —— CANN 树里没有这几个
  函数的调用样例 (c_api 与 impl 两层都查过)。语义只能从内建推, 而"推"正是
  F183 里烧掉三次构建的失败模式。猜错在一次运行里暴露, 而不是去 141 个用例里排查。

三条判据:
  S: int8 矩乘在 int32 里**精确**, 故要求与 NumPy 参考**逐位相同**, 不设容差。
  O: PV 用同一对 P/V 连做两次, 结果必须是单次的**两倍** ——
     若 c_matrix_init_val=false 其实是覆写而非累加, 结果会等于单次, 一眼看出。
  R: fp32 对角阵乘 (O 的重标定)。取值全取 2 的幂次的整数倍, 故
     R1 = D @ O_base 与 R2 = D @ O_base + P @ V 在 fp32 里**精确**, 同样不设容差。
     R1 验三条 fp32 搬运的参数语义; R2 验"重标定打底 + PV 以 Init=false 累上去"
     这一**组合**, 即新内核的结构本身。
"""
import argparse, json, os, sys
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=1, help="物理卡号; 0 卡是 CI runner, 禁用")
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--output", default="results/cube_v2_probe.json")
    a = ap.parse_args()
    if a.device == 0:
        print("拒绝: 0 卡是管理员部署 CI runner 用的", file=sys.stderr)
        return 2

    import torch, torch_npu  # noqa: F401
    import vc_attention_ascend
    #  ★op 不是 import 时注册的, 要显式 load_native() —— 漏了它报的是
    #    "'_OpNamespace' 'vc_ascend' object has no attribute ...", 看着像没编进去。
    vc_attention_ascend.load_native()
    torch.npu.set_device(a.device)
    dev = f"npu:{a.device}"

    rng = np.random.default_rng(a.seed)
    MB, NB, D = 64, 128, 128
    #  取值刻意小: int8 矩乘在 int32 里精确, 而 half 的 P/V 用 2 的幂次保证
    #  P@V 在 fp32 里也精确 —— 这样"不逐位相同"只可能来自 stride 语义错, 不来自舍入。
    q = rng.integers(-8, 9, size=(MB, D), dtype=np.int8)
    k = rng.integers(-8, 9, size=(NB, D), dtype=np.int8)
    pm = (rng.integers(-4, 5, size=(MB, NB)).astype(np.float32) / 4.0).astype(np.float16)
    v = (rng.integers(-4, 5, size=(NB, D)).astype(np.float32) / 4.0).astype(np.float16)
    #  重标定: D 是对角阵 diag(alpha), alpha 取 2 的幂次; O_base 取 0.25 的整数倍。
    #  两者乘积与 P@V 同为 0.0625 的整数倍且 |.| < 256, fp32 可精确表示。
    alpha = (2.0 ** rng.integers(-2, 3, size=MB)).astype(np.float32)
    dmat = np.zeros((MB, MB), dtype=np.float32)
    dmat[np.arange(MB), np.arange(MB)] = alpha
    obase = (rng.integers(-16, 17, size=(MB, D)).astype(np.float32) / 4.0)

    tq = torch.from_numpy(q).to(dev)
    tk = torch.from_numpy(k).to(dev)
    tp = torch.from_numpy(pm).to(dev)
    tv = torch.from_numpy(v).to(dev)
    td = torch.from_numpy(dmat).to(dev)
    tob = torch.from_numpy(obase).to(dev)

    s_out, o_out, r1_out, r2_out = torch.ops.vc_ascend.cube_v2_probe(tq, tk, tp, tv, td, tob)
    s = s_out.cpu().numpy()
    o = o_out.cpu().numpy()
    r1 = r1_out.cpu().numpy()
    r2 = r2_out.cpu().numpy()

    s_ref = q.astype(np.int32) @ k.astype(np.int32).T          # [64,128]
    o_once = pm.astype(np.float32) @ v.astype(np.float32)      # [64,128]
    o_ref = 2.0 * o_once
    r1_ref = dmat @ obase                                      # [64,128]
    r2_ref = r1_ref + o_once

    #  ★把原始输入输出存盘: stride 语义的假设检验全在主机侧离线做,
    #    不必为每个假设再占一次卡 (一次运行 ~3 分钟, 假设有十几个)。
    raw = os.path.splitext(a.output)[0] + "_raw.npz"
    np.savez(raw, q=q, k=k, p=pm, v=v, s=s, o=o,
             dmat=dmat, obase=obase, r1=r1, r2=r2)
    print("原始数据已存:", raw)

    rep = {"device": a.device, "seed": a.seed, "shapes": {"M": MB, "N": NB, "D": D},
           "raw": raw}

    s_bad = int(np.count_nonzero(s != s_ref))
    rep["S"] = {"mismatched": s_bad, "total": int(s.size),
                "max_abs_diff": int(np.abs(s.astype(np.int64) - s_ref.astype(np.int64)).max()),
                "exact": s_bad == 0}

    o_bad = int(np.count_nonzero(o != o_ref))
    #  ★判"是累加还是覆写": 与单次比。两者都不符才是排布错。
    o_like_once = int(np.count_nonzero(o == o_once))
    rep["O"] = {"mismatched_vs_2x": o_bad, "total": int(o.size),
                "max_abs_diff_vs_2x": float(np.abs(o - o_ref).max()),
                "elems_equal_to_single_pass": o_like_once,
                "exact_2x": o_bad == 0,
                "verdict": ("累加正确" if o_bad == 0 else
                            "★c_matrix_init_val=false 是**覆写**不是累加"
                            if o_like_once > o.size // 2 else "排布错")}
    r1_bad = int(np.count_nonzero(r1 != r1_ref))
    r2_bad = int(np.count_nonzero(r2 != r2_ref))
    #  ★判"是搬运排布错还是组合错": r1 对而 r2 错 => 组合 (Init 语义) 的问题;
    #    r1 就错 => 三条 fp32 搬运的参数语义有一条不对。
    rep["R"] = {"r1_mismatched": r1_bad, "r2_mismatched": r2_bad, "total": int(r1.size),
                "r1_max_abs_diff": float(np.abs(r1 - r1_ref).max()),
                "r2_max_abs_diff": float(np.abs(r2 - r2_ref).max()),
                "r1_exact": r1_bad == 0, "r2_exact": r2_bad == 0,
                "verdict": ("fp32 对角阵乘与累加组合都正确" if r1_bad == 0 and r2_bad == 0 else
                            "★组合错: 重标定本身对, 但 PV 没累到它上面" if r1_bad == 0 else
                            "★三条 fp32 搬运的参数语义有错")}
    rep["status"] = ("passed" if rep["S"]["exact"] and rep["O"]["exact_2x"]
                     and rep["R"]["r1_exact"] and rep["R"]["r2_exact"] else "failed")

    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "w") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0 if rep["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
