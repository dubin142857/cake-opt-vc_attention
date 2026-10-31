#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`Axpy<float, half>` 形参语义的**数值**探针 —— F238 的实现前置, 不是跑算子。

★要确证什么: 本机 (dav_c220) 的 `Axpy` 支持 (dst=float, src=half), 语义
  `dst[i] = src[i] * scalar + dst[i]`, 且**标量类型跟 src**, 即 fp16
  (`impl/basic_api/dav_c220/kernel_operator_vec_ternary_scalar_impl.h:30` 的 SupportType)。
  它能把"读半精度 + 乘标量 + 累加到 fp32"融进**一条**指令 —— 这正是让
  "反量化的乘 + int->float 的转换 + 减行最大值"合成一遍的关键。

★为什么不能推 (F183 的次序): 混合宽度下 **repeat 的跨步单位**从声明里看不出来 ——
  dst 是 fp32 (一个 32B 块 8 个元素), src 是 fp16 (16 个元素), 两边的
  dstRepStride / srcRepStride 到底各按谁的块算, 只有数值能定。

三条判据 (都不设容差):
  C1 count 形与 NumPy 的 `base + f32(x) * f32(f16(a))` **逐位相同**;
  C2 level-0 形 (mask=64, repeat=64, dstRepStride=8, srcRepStride=4) 与 count 形**逐位相同**
     —— 它验的是"跨步各按自己的数据宽度算"这一假设;
  C3 本例**能分开**几种会静默走错的解释: base 逐行不同 (验 dst 真的被读了而不是被覆写),
     x 逐元素不同 (验标量没被当成逐元素), 且 a 不是 1.0 (验乘法真的发生)。

★取值刻意让 `base + x*a` 在 fp32 内**精确**: x 取小整数, a 取 2 的幂,
  base 取小整数 —— 于是"不逐位相同"只可能是形参或语义错, 不会是舍入。
"""
import argparse, json, os, sys
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=4, help="物理卡号; 0 卡是 CI runner, 禁用")
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--output", default="results/axpy_probe.json")
    a_ = ap.parse_args()
    if a_.device == 0:
        print("拒绝: 0 卡是管理员部署 CI runner 用的", file=sys.stderr)
        return 2

    import torch, torch_npu  # noqa: F401
    import vc_attention_ascend
    vc_attention_ascend.load_native()
    torch.npu.set_device(a_.device)
    dev = f"npu:{a_.device}"

    R, N = 32, 128
    rng = np.random.default_rng(a_.seed)
    #  小整数 + 2 的幂的标量: base + x*a 在 fp32 内精确, 舍入是恒等。
    x = rng.integers(-64, 65, size=(R, N)).astype(np.float16)
    base = rng.integers(-1000, 1001, size=(R, N)).astype(np.float32)
    a_scalar = np.float16(0.25)                 # 2 的幂, 且不是 1.0
    av = np.full(16, a_scalar, dtype=np.float16)

    tx = torch.from_numpy(x).to(dev)
    tb = torch.from_numpy(base).to(dev)
    ta = torch.from_numpy(av).to(dev)
    oc_t, ol_t = torch.ops.vc_ascend.axpy_probe(tx, tb, ta)
    oc = oc_t.cpu().numpy()
    ol = ol_t.cpu().numpy()

    model = base + x.astype(np.float32) * np.float32(a_scalar)
    #  几种会静默走错的解释, 本例必须能把它们与正解分开:
    alt = {
        "dst_被覆写而不是累加": (x.astype(np.float32) * np.float32(a_scalar)),
        "标量被当成逐元素(取x自身)": base + x.astype(np.float32) * x.astype(np.float32),
        "乘法没发生(a当成1)": base + x.astype(np.float32),
    }
    rep = {
        "C1_count_bitexact": bool(np.array_equal(oc, model)),
        "C2_level0_matches_count": bool(np.array_equal(ol, oc)),
        "C2_level0_bitexact_vs_model": bool(np.array_equal(ol, model)),
        "count_mismatch": int((oc != model).sum()),
        "level0_mismatch": int((ol != model).sum()),
        "sample_got_count": [float(v) for v in oc[0, :4]],
        "sample_got_level0": [float(v) for v in ol[0, :4]],
        "sample_model": [float(v) for v in model[0, :4]],
        #  C3: 每个替代解释都必须与正解**不同**, 否则本例分不开它们。
        "C3_case_separates": {k: (not bool(np.array_equal(model, v))) for k, v in alt.items()},
        "C3_got_matches_an_alt": {k: bool(np.array_equal(oc, v)) for k, v in alt.items()},
    }
    ok = (rep["C1_count_bitexact"] and rep["C2_level0_bitexact_vs_model"]
          and all(rep["C3_case_separates"].values())
          and not any(rep["C3_got_matches_an_alt"].values()))
    rep["all_pass"] = ok
    os.makedirs(os.path.dirname(a_.output) or ".", exist_ok=True)
    with open(a_.output, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"C1 count 形逐位相同: {rep['C1_count_bitexact']} (不符 {rep['count_mismatch']} 个)")
    print(f"C2 level-0 形逐位相同: {rep['C2_level0_bitexact_vs_model']} "
          f"(不符 {rep['level0_mismatch']} 个; 与 count 形一致: {rep['C2_level0_matches_count']})")
    print(f"C3 本例能分开的替代解释: {rep['C3_case_separates']}")
    if not ok:
        print(f"  实得 count  {rep['sample_got_count']}")
        print(f"  实得 level0 {rep['sample_got_level0']}")
        print(f"  模型        {rep['sample_model']}")
        print(f"  命中了哪个替代解释: {rep['C3_got_matches_an_alt']}")
    print("结论:", "三条判据全过" if ok else "**未过** —— 形参/语义假设被证伪")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
