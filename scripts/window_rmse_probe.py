#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在线归一化窗口对 dense RMSE 的影响 —— **离线**对比, 不碰内核也不碰交付物。

★用途 (F212 的 next): 把在线窗口由整个 chunk (ONLINE_TILE = K_TILE*CHUNK = 512)
  改回**单块** (K_TILE = 128) 可以把趟 1/趟 2 融成一趟 (AIV 的 t2_vec 占 46.8%),
  但它**改变数值契约**。先量出改完之后的 dense RMSE 是多少, 再决定要不要改内核。

★口径与 benchmark_ascend910b.py 一致: 同一 seed(0) 与 dtype 造输入, 取
  `torch.linspace(0, queries-1, check_queries)` 的查询行, 与 `dense_reference` 比。
★为什么用 payload_reference 代理内核输出: 两者的差是 decoded_reference 的量级
  (最大 9.5e-5), 远小于 1% 闸与 2.98% 的 dense 误差, 故不影响结论。
"""
import argparse, sys
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=3)
    ap.add_argument("--queries", type=int, default=37741)
    ap.add_argument("--keys", type=int, default=37741)
    ap.add_argument("--heads", type=int, default=28)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--check-queries", type=int, default=32)
    a = ap.parse_args()
    if a.device == 0:
        print("拒绝: 0 卡是管理员部署 CI runner 用的", file=sys.stderr)
        return 2
    import torch_npu  # noqa: F401
    import vc_attention_ascend as vc
    from vc_attention_ascend.reference import (payload_reference, dense_reference,
                                               K_TILE, CHUNK, ONLINE_TILE)
    torch.npu.set_device(a.device)
    dtype = getattr(torch, a.dtype)
    g = torch.Generator().manual_seed(0)      # 与 benchmark 同 seed
    original = [torch.randn((1, a.heads, n, a.dim), generator=g).to(dtype)
                for n in (a.queries, a.keys, a.keys)]
    q, k, v = [x.to(f"npu:{a.device}") for x in original]
    ids = torch.linspace(0, a.queries - 1, min(a.queries, a.check_queries)).long().unique()
    payload = vc.prepare(q, k, v, layout="HND")
    cpu_p = {kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in payload.items()}
    print(f"形状 q={a.queries} k={a.keys} h={a.heads} d={a.dim} dtype={a.dtype} "
          f"· 查询行 {len(ids)} · K_TILE={K_TILE} CHUNK={CHUNK} ONLINE_TILE={ONLINE_TILE}")
    dense = dense_reference(*original, query_indices=ids)

    def rmse(ref):
        d = (ref.float() - dense.float())
        return float(d.square().mean().sqrt() / dense.float().square().mean().sqrt())

    #  ★三组对照, 每组只动一个量 (F189):
    #    (1) 在线窗口的宽度  —— 现行 chunk 窗口 对 单块窗口
    #    (2) 分数暂存的精度  —— fp32 (现行) 对 fp16
    #    后者是 tile_m=128 的 UB 预算里唯一压得动的大头 (scoreBuf 131072 B),
    #    压它的代价必须先在这里量出来, 再决定动不动内核。
    cases = [("chunk 窗口 · 分数 fp32 (现行)", ONLINE_TILE, None),
             ("chunk 窗口 · 分数 fp16",        ONLINE_TILE, torch.float16),
             ("单块窗口 · 分数 fp32",          K_TILE,      None),
             ("单块窗口 · 分数 fp16",          K_TILE,      torch.float16)]
    base = {}
    for cast in (False, True):
        name = "expcast" if cast else "control"
        for label, w, sd in cases:
            ref = payload_reference(cpu_p, expcast=cast, query_indices=ids,
                                    tile_width=w, score_dtype=sd)
            r = 100 * rmse(ref)
            if sd is None and w == ONLINE_TILE:
                base[cast] = r
            delta = f"  (相对现行 {r - base[cast]:+.6f} 个百分点)" if cast in base else ""
            print(f"  {name:8s} {label:26s} tile_width={w:4d}  dense RMSE = {r:.6f} %{delta}")
    print("★判据: 交付闸是 dense RMSE 不劣于交付基线的 2.98%; "
          "逐例闸是 decoded_reference 的 1%, 后者不受本改动影响 (oracle 与内核同契约)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
