#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""确定性扫描 —— 同一 payload 在同一进程里连跑 N 次, 逐位比较。

★为什么要有它 (F194/F195): 流水版过了 141 例的小形状验收, 却在 37741² 的基准上
  第一次预热就不过, 且基准自己的 `repeat_all_rows` 显示**同样的行重跑结果不同**。
  用参考值判对错要先算 CPU 参考 (37741 键上很慢); 而"两次跑得不一样"**不需要参考**,
  一次就能判, 且它对竞态比对精度判据灵敏得多。
★与 validate 的分工: validate 判"对不对", 它判"稳不稳"。竞态可以稳定地错 (那次 141 例
  全过而大形状错就是), 也可以不稳定地对 —— 两条都要有。
★用法要先过 devguard: 污染卡上的任何判决无效 (F194)。
  python -m core.devguard --verify-artifact vc910_card_clean --card 3
"""
import argparse, hashlib, sys
import torch


def sha(t):
    return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--expcast", action="store_true")
    #  形状表: 逐步加大**每核的 chunk 数**与**任务数** —— 流水的竞态随这两者放大。
    ap.add_argument("--shapes", default="64x4096x2,64x16384x2,256x16384x2,1024x37741x2,64x37741x28")
    a = ap.parse_args()
    if a.device == 0:
        print("拒绝: 0 卡是管理员部署 CI runner 用的", file=sys.stderr)
        return 2
    import torch_npu  # noqa: F401
    import vc_attention_ascend as vc
    vc.load_native()
    torch.npu.set_device(a.device)
    dev = f"npu:{a.device}"
    bad = 0
    for spec in a.shapes.split(","):
        nq, nk, h = (int(x) for x in spec.split("x"))
        g = torch.Generator().manual_seed(91023)
        q, k, v = [torch.randn((1, h, n, a.dim), generator=g).half().to(dev)
                   for n in (nq, nk, nk)]
        payload = vc.prepare(q, k, v, layout="HND")
        #  CHUNK=4, NBLK=128 -> 每个任务的 chunk 数; MT=64 -> 任务数
        chunks = -(-(-(-nk // 128)) // 4)
        tasks = h * (-(-nq // 64))
        digests = [sha(vc.run_prepared(payload, expcast=a.expcast)) for _ in range(a.repeat)]
        uniq = sorted(set(digests))
        ok = len(uniq) == 1
        bad += not ok
        print(f"  {'过' if ok else '★不过'} q={nq} k={nk} h={h} (任务 {tasks}, 每任务 chunk {chunks}): "
              f"{a.repeat} 次得到 {len(uniq)} 种结果 {digests}")
    print(f"{len(a.shapes.split(','))} 个形状里 {bad} 个不确定")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
