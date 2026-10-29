#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""msOpProf 的被测进程 —— 只做"准备 payload 然后连launch 几次", 没有别的活动部件。

★为什么单独写一个而不是复用 benchmark: msprof 按 launch-count 截取前 N 个核,
  而 benchmark 会先跑预热与若干校验, 把要测的核挤到计数之外。

★选卡在脚本内部做 (torch.npu.set_device), **不设 ASCEND_RT_VISIBLE_DEVICES** ——
  msOpProf 路径上那个环境变量会让它看不到目标卡, 是本仓登记过的坑。
"""
import argparse, sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=1, help="物理卡号; 0 卡是 CI runner, 禁用")
    ap.add_argument("--queries", type=int, default=37741)
    ap.add_argument("--keys", type=int, default=37741)
    ap.add_argument("--heads", type=int, default=28)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument("--expcast", action="store_true")
    a = ap.parse_args()
    if a.device == 0:
        print("拒绝: 0 卡是管理员部署 CI runner 用的", file=sys.stderr)
        return 2

    import torch, torch_npu  # noqa: F401
    import vc_attention_ascend as vc
    vc.load_native()
    torch.npu.set_device(a.device)
    dev = f"npu:{a.device}"
    torch.manual_seed(20261029)
    q = torch.randn(1, a.heads, a.queries, a.dim, dtype=torch.bfloat16, device=dev)
    k = torch.randn(1, a.heads, a.keys, a.dim, dtype=torch.bfloat16, device=dev)
    v = torch.randn(1, a.heads, a.keys, a.dim, dtype=torch.bfloat16, device=dev)
    payload = vc.prepare(q, k, v, layout="HND")
    #  预热一次并同步, 使其后的 launch 是稳态 —— 但预热也会被 msprof 计数,
    #  故 --launch-count 要比 --launches 多留一个。
    vc.run_prepared(payload, expcast=a.expcast)
    torch.npu.synchronize()
    for _ in range(a.launches):
        vc.run_prepared(payload, expcast=a.expcast)
    torch.npu.synchronize()
    print(f"launched {a.launches} (+1 warmup) on device {a.device}, expcast={a.expcast}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
