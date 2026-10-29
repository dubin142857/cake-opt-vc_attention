#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单例烟测 —— 改同步之后**先跑它再跑全量 141**。

★为什么要有它 (F192): 一次挂死要占卡数分钟, 还可能把共用卡置成 Alarm。
  全量 141 例要跑到第 9 例才走到 n_in>=2 的分支, 而那正是预取分支第一次被执行的地方。
  本脚本直接打在那个形状上, 几十秒出结果。
默认 q=33 k=129 d=64 —— kpad=256 故 tiles=2, 是**第一个 n_in>=2** 的形状。
"""
import argparse, sys
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--queries", type=int, default=33)
    ap.add_argument("--keys", type=int, default=129)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--expcast", action="store_true")
    #  ★--repeat: 流水的缺陷是**间歇性**的, 单次通过不构成证据。
    #    同形状在同一进程里连跑 N 次, 用来区分"跨启动的残留状态"与"纯时序竞态"。
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    if a.device == 0:
        print("拒绝: 0 卡是管理员部署 CI runner 用的", file=sys.stderr)
        return 2

    import torch_npu  # noqa: F401
    import vc_attention_ascend as vc
    vc.load_native()
    from vc_attention_ascend.reference import payload_reference
    torch.npu.set_device(a.device)
    dev = f"npu:{a.device}"
    g = torch.Generator().manual_seed(91023)
    q, k, v = [torch.randn((1, a.heads, n, a.dim), generator=g).half().to(dev)
               for n in (a.queries, a.keys, a.keys)]
    payload = vc.prepare(q, k, v, layout="HND")
    cpu_p = {kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in payload.items()}
    ref = payload_reference(cpu_p, expcast=a.expcast).half()
    tiles = (a.keys + 127) // 128
    print(f"  形状 q={a.queries} k={a.keys} d={a.dim} expcast={a.expcast} -> kpad={tiles*128}, tiles={tiles}")
    bad = 0
    for it in range(a.repeat):
        out = vc.run_prepared(payload, expcast=a.expcast).cpu()
        rel = float(((out.double() - ref.double()).norm()
                     / ref.double().norm().clamp_min(1e-30)).item())
        ok = rel <= 1e-2
        bad += not ok
        print(f"    第 {it+1:>2} 次: RMSE {rel:.6e}  {'✓' if ok else '★不过'}")
    print(f"  {a.repeat} 次里 {bad} 次不过" + ("  ——**间歇**, 是竞态" if 0 < bad < a.repeat else ""))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
