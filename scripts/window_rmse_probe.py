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

    #  ================= fixpipe 反量化契约的代价 (F233 的 next) =================
    #  ★这一段回答的是**另一个闸**: 141 例比的是「内核 对 oracle」。若内核改契约而
    #    oracle 不改, 两者的差就是下面这些数, 它要落在 1% 以内才允许不改 oracle。
    #  ★两项**分开量**再合起来量 (E4: 一次只动一个量), 否则看不出增量是谁贡献的:
    #      (1) S 转 fp16        —— F220 已量过, 这里复现一次作对照;
    #      (2) scale 低 13 位尾数被截断 —— F233 新测出的硬件行为, F220 的模型**没算**;
    #      (3) 两项一起        —— 真实的 fixpipe 契约。
    #  ★为什么是 13: F233 在 8 个候选模型里把它钉死 (trunc13 0/8192 不符,
    #    次优 trunc12 1992 个, 全精度 fp32 scale 3956 个)。
    print("\n=== fixpipe 反量化契约: 内核若改而 oracle 不改, 两者相差多少 (1% 闸) ===")
    contracts = [("S 转 fp16 (F220 复现)",            None, torch.float16),
                 ("scale 截 13 位尾数 (F233 新增)",    13,   None),
                 ("两项一起 = fixpipe 契约",           13,   torch.float16)]
    worst_overall = 0.0
    for cast in (False, True):
        name = "expcast" if cast else "control"
        base = payload_reference(cpu_p, expcast=cast, query_indices=ids,
                                 tile_width=ONLINE_TILE)
        bf = base.float()
        for label, tb, sd in contracts:
            ref = payload_reference(cpu_p, expcast=cast, query_indices=ids,
                                    tile_width=ONLINE_TILE, score_dtype=sd,
                                    deq_trunc_bits=tb).float()
            d = ref - bf
            glob = float(d.square().mean().sqrt() / bf.square().mean().sqrt())
            #  逐行: 把除最后一维 (dim) 外的所有维拍平, 每一行单独算相对 RMSE。
            dr = d.reshape(-1, d.shape[-1])
            br = bf.reshape(-1, bf.shape[-1])
            per_row = (dr.square().mean(-1).sqrt() / br.square().mean(-1).sqrt().clamp_min(1e-30))
            row_worst = float(per_row.max())
            flag = "  ★打穿 1% 闸" if row_worst > 0.01 else ""
            #  ★同时报 dense RMSE —— 那是**交付闸** (基线 2.98%), 与上面的 1% 闸是两回事:
            #    前者问"离真值多远", 后者问"内核与 oracle 差多远"。两个都要看。
            dn = 100 * float((ref - dense.float()).square().mean().sqrt()
                             / dense.float().square().mean().sqrt())
            if label.startswith("两项"):
                worst_overall = max(worst_overall, row_worst)
            print(f"  {name:8s} {label:28s} 对 oracle: 全局 {100*glob:.4f} % · "
                  f"逐行最坏 {100*row_worst:.4f} %{flag} | dense RMSE {dn:.6f} %")
    print(f"★结论依据: 1% 闸比的是 decoded_reference 的逐例相对误差; "
          f"两项一起时的逐行最坏 = {100*worst_overall:.4f} %。")
    print("★若 expcast 一侧打穿, 则 oracle 必须同步改契约 (传 deq_trunc_bits 与 score_dtype) ——"
          " 那是**契约变更, 不是闸放宽**, 必须留痕。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
