#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`WholeReduceSum` 形参语义的**数值**探针 —— E33 的前置, 不是跑算子。

★为什么要它 (F183 定的次序): 形参 dstRepStride / srcBlkStride / srcRepStride 虽有本机
  权威样例 (CANN 自己的 reduce_sum_v220_impl.h::ReduceSumArCompute), 但那里的几何是
  248 个 repeat 而我们是 ROWS 个; 且 **dst 的排布**(连续? 按 32B 块跳?) 从声明里看不出来。
  猜错在 141 个用例里排查, 远贵于在这里一次测掉。

★取值刻意用小整数: fp32 内精确, **任何求和顺序都同果**。于是"不逐位相同"只可能是
  形参错, 不会是舍入 —— 这是本探针能下判决的全部依据。

两条判据 (都不设容差):
  old: 现行 `ReduceSum<float,AR,false>` 的结果, 必须与 NumPy 行和**逐位相同**
       (它是对照组; 它若不符, 说明探针本身或环境有问题, 而不是候选写法有问题)。
  new: `Add(折 128->64) + WholeReduceSum` 的结果, 必须与 NumPy 行和**逐位相同**。
"""
import argparse, json, os, sys
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=3, help="物理卡号; 0 卡是 CI runner, 禁用")
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--output", default="results/vec_reduce_probe.json")
    a = ap.parse_args()
    if a.device == 0:
        print("拒绝: 0 卡是管理员部署 CI runner 用的", file=sys.stderr)
        return 2

    import torch, torch_npu  # noqa: F401
    import vc_attention_ascend
    vc_attention_ascend.load_native()
    torch.npu.set_device(a.device)
    dev = f"npu:{a.device}"

    R, N = 32, 128
    rng = np.random.default_rng(a.seed)
    #  小整数: 行和最大 128*8 = 1024, fp32 内精确, 求和顺序无关。
    x = rng.integers(-8, 9, size=(R, N)).astype(np.float32)
    tx = torch.from_numpy(x).to(dev)

    out_new, out_old, out_after = torch.ops.vc_ascend.vec_reduce_probe(tx)
    new = out_new.cpu().numpy()
    old = out_old.cpu().numpy()
    after = out_after.cpu().numpy()
    ref = x.sum(axis=-1)

    rep = {"device": a.device, "seed": a.seed, "shape": [R, N]}
    for name, got in (("old_ReduceSum", old), ("new_WholeReduceSum", new)):
        bad = int(np.count_nonzero(got != ref))
        rep[name] = {"mismatched": bad, "total": int(ref.size),
                     "max_abs_diff": float(np.abs(got - ref).max()),
                     "exact": bad == 0}
        if bad:
            #  ★把前几个不符的位置打出来: dst 若按块跳, 症状是"只有每 8 个里的第 1 个对",
            #    一眼能从下标看出来, 不必再猜。
            idx = np.nonzero(got != ref)[0][:8].tolist()
            rep[name]["first_bad_rows"] = idx
            rep[name]["got_head"] = got[:8].tolist()
            rep[name]["ref_head"] = ref[:8].tolist()
    #  第三条: WholeReduceSum 之后的**全宽**操作有没有被残留的 mask 截断。
    #  ★底层 WholeReduceSum 与高层 ReduceSum 不同, 后者收尾有 SetMaskNorm()+ResetMask(),
    #    前者没有。这里全宽写 1.0 再求行和, 每行必须恰为 N。
    mask_bad = int(np.count_nonzero(after != float(N)))
    rep["mask_not_sticky"] = {
        "mismatched": mask_bad, "expect_each": float(N),
        "got_head": after[:8].tolist(), "exact": mask_bad == 0,
        "verdict": ("WholeReduceSum 没有把 mask 留下, 不必手动恢复" if mask_bad == 0
                    else "★mask 被留住了, 全宽操作被截断 —— 内核里必须在它后面补 SetMaskNorm()+ResetMask()")}
    rep["agree"] = bool(np.array_equal(new, old))
    rep["status"] = ("passed" if rep["old_ReduceSum"]["exact"]
                     and rep["new_WholeReduceSum"]["exact"]
                     and rep["mask_not_sticky"]["exact"] else "failed")
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "w") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0 if rep["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
