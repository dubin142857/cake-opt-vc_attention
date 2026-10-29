#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""精确复现 validate_ascend910b.py 的某一个用例 —— 在**全新进程**里只跑它。

★为什么需要它: 第 9 例 (q=33 k=129) 在全量里 RMSE 1.057 而单独烟测 10/10 过且逐次
  完全相同 —— 不是时序竞态。要区分两种可能:
    (a) 数据不同 —— 验收对 nq==33 会改写 V (v[...,0]=0, v[...,1]=2), 烟测没复制;
        且共享生成器被前面的形状推进过。
    (b) 前置状态 —— 前 8 次启动在核上留下了什么。
  本脚本按验收的**原样顺序**推进生成器并施加同样的改写, 但在全新进程里只跑目标用例:
    它若**过** -> 是 (b) 跨启动状态; 它若**不过** -> 是 (a) 数据相关。
"""
import argparse, sys
import torch

SHAPES = [(1, 1), (17, 65), (31, 127), (32, 128), (33, 129), (65, 257), (97, 513)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--shape-index", type=int, default=4, help="SHAPES 的下标; 4 即 (33,129)")
    ap.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    ap.add_argument("--dim", type=int, default=64, choices=(64, 128))
    ap.add_argument("--expcast", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    #  ★--warmup K: 在同一进程里先按验收顺序跑前 K 个形状 (每个两种 cast), 再跑目标。
    #    用来复现"跨启动状态"并二分出是**哪一个**前置形状破坏了目标用例。
    ap.add_argument("--warmup", type=int, default=0)
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
    dtype = getattr(torch, a.dtype)

    #  ★按 validate 的原样顺序推进生成器: 它每个 (dtype, d, shape) 抽一次,
    #    cast 是更内层的循环故不额外消耗。这里只复现同一 (dtype, d) 下的前几个形状。
    generator = torch.Generator().manual_seed(91023)
    q = k = v = None
    for idx, (nq, nk) in enumerate(SHAPES):
        q, k, v = [torch.randn((1, 2, n, a.dim), generator=generator).to(dtype)
                   for n in (nq, nk, nk)]
        if nq == 33:
            v[..., 0] = 0.0
            v[..., 1] = 2.0
        if idx == a.shape_index:
            break
    #  先按验收顺序跑前 K 个形状 (两种 cast 各一次), 复现跨启动状态
    if a.warmup:
        wg = torch.Generator().manual_seed(91023)
        for widx in range(min(a.warmup, a.shape_index)):
            wnq, wnk = SHAPES[widx]
            wq, wk, wv = [torch.randn((1, 2, n, a.dim), generator=wg).to(dtype)
                          for n in (wnq, wnk, wnk)]
            if wnq == 33:
                wv[..., 0] = 0.0
                wv[..., 1] = 2.0
            wp = vc.prepare(*(x.to(dev) for x in (wq, wk, wv)), layout="HND")
            for wcast in (False, True):
                vc.run_prepared(wp, expcast=wcast)
            torch.npu.synchronize()
            del wp
        print(f"  已按验收顺序预跑前 {min(a.warmup, a.shape_index)} 个形状 (每个两种 cast)")

    nq, nk = SHAPES[a.shape_index]
    q, k, v = [x.to(dev) for x in (q, k, v)]
    payload = vc.prepare(q, k, v, layout="HND")
    cpu_p = {kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in payload.items()}
    ref = payload_reference(cpu_p, expcast=a.expcast).to(dtype)
    print(f"  复现 validate 第 {a.shape_index} 个形状: q={nq} k={nk} d={a.dim} "
          f"dtype={a.dtype} expcast={a.expcast}"
          + ("   ★含 nq==33 的 V 改写" if nq == 33 else ""))
    bad = 0
    for it in range(a.repeat):
        out = vc.run_prepared(payload, expcast=a.expcast).to(dtype).cpu()
        rel = float(((out.double() - ref.double()).norm()
                     / ref.double().norm().clamp_min(1e-30)).item())
        ok = rel <= 1e-2
        bad += not ok
        print(f"    第 {it+1:>2} 次: RMSE {rel:.6e}  {'✓' if ok else '★不过'}")
    print(f"  {a.repeat} 次里 {bad} 次不过")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
