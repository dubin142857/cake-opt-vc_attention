#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""甲-5 / E48: 离线量"行和 `mass` 在 fp16 域求和"的数值代价 —— **不上机内核、不构建**。

★为什么是离线的: 这是**数值契约**问题, 不是性能问题。F254 已量清性能侧 ——
  保住 fp32 分母只得 −5.82% (低于 8% 门槛), 全 fp16 得 −15.51% 但分母也在 fp16。
  差的那 62% 能不能拿, 完全取决于 fp16 分母的误差是否在交付闸内。

★两道闸 (与交付一致):
  dense RMSE   不劣于交付基线 **2.98%** (现行 control 2.981115%, 口径是 benchmark);
  逐例         `decoded_reference` 的 **1%**。
★E7: 本脚本与 benchmark 是**两台仪器**, 同契约下相差约 0.0044 个百分点。
  **只有增量可以跨仪器搬, 绝对值不可以。**

★回归闸: `mass_dtype=None` 必须复现**当前契约**下的基线 control = **2.976722%**。
  ★这个数怎么来的 (2026-10-31 实撞, 记下免得下次又填错): 我最初填的是 2.975507%,
    那是 window_rmse_probe 在 **E38 契约改动之前**跑出来的 (那次日志紧接着就因为
    reference.py 还没有 `deq_trunc_bits` 参数而崩)。当前契约 (分数 fp16 + 截 13 位
    + 拆 qscale) 下是 2.976722%。
  ★交叉验证: benchmark 口径的 control 是 **2.981115%**, 两台仪器已知相差 0.0044 pp,
    2.981115 - 0.004393 = **2.976722** —— 逐位对上, 故不是回归 (E7: 增量可跨仪器搬,
    绝对值不可以; 这里正是用那个已知增量把两边对起来)。
"""
import argparse, sys
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=4)
    ap.add_argument("--queries", type=int, default=37741)
    ap.add_argument("--keys", type=int, default=37741)
    ap.add_argument("--heads", type=int, default=28)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--check-queries", type=int, default=32)
    a = ap.parse_args()
    if a.device == 0:
        print("拒绝: 0 卡是管理员部署 CI runner 用的", file=sys.stderr); return 2
    import torch_npu  # noqa: F401
    import vc_attention_ascend as vc
    from vc_attention_ascend.reference import payload_reference, dense_reference, ONLINE_TILE
    torch.npu.set_device(a.device)
    g = torch.Generator().manual_seed(0)      # 与 benchmark / window_rmse_probe 同 seed
    original = [torch.randn((1, a.heads, n, a.dim), generator=g).to(torch.bfloat16)
                for n in (a.queries, a.keys, a.keys)]
    q, k, v = [x.to(f"npu:{a.device}") for x in original]
    ids = torch.linspace(0, a.queries - 1, min(a.queries, a.check_queries)).long().unique()
    payload = vc.prepare(q, k, v, layout="HND")
    cpu_p = {kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in payload.items()}
    dense = dense_reference(*original, query_indices=ids)

    def rmse(ref):
        d = (ref.float() - dense.float())
        return 100 * float(d.square().mean().sqrt() / dense.float().square().mean().sqrt())

    print(f"形状 q={a.queries} k={a.keys} h={a.heads} d={a.dim} · 查询行 {len(ids)}")
    print(f"{'路径':10s}{'分母求和域':16s}{'dense RMSE %':>16s}{'相对现行 (pp)':>18s}")
    worst = 0.0
    for cast in (False, True):
        name = "expcast" if cast else "control"
        base = None
        for label, md in (("fp32 (现行)", None), ("**fp16**", torch.float16)):
            ref = payload_reference(cpu_p, expcast=cast, query_indices=ids,
                                    tile_width=ONLINE_TILE, mass_dtype=md)
            r = rmse(ref)
            if md is None:
                base = r; delta = ""
            else:
                delta = f"{r - base:+.6f}"
                worst = max(worst, r - base)
            print(f"{name:10s}{label:16s}{r:16.6f}{delta:>18s}")
    print()
    #  逐例闸: 内核改契约而 oracle 不改时, 两者相差多少
    print("=== 逐例闸 (1%): 内核走 fp16 分母而 oracle 仍 fp32, 两者的相对差 ===")
    worst_case = 0.0
    for cast in (False, True):
        name = "expcast" if cast else "control"
        b = payload_reference(cpu_p, expcast=cast, query_indices=ids, tile_width=ONLINE_TILE)
        n_ = payload_reference(cpu_p, expcast=cast, query_indices=ids, tile_width=ONLINE_TILE,
                               mass_dtype=torch.float16)
        bf = b.float()
        rel = float((n_.float() - bf).square().mean().sqrt() / bf.square().mean().sqrt())
        worst_case = max(worst_case, rel)
        print(f"  {name:10s} 相对 RMSE = {rel*100:.6f} %")
    print()
    print(f"★判据 (测前写定): dense RMSE 增量 <= **0.02 个百分点** 且逐例 < 1%")
    print(f"  实测最差增量 **{worst:+.6f} pp** -> {'过' if worst <= 0.02 else '**不过**'}")
    print(f"  实测最差逐例 **{worst_case*100:.6f} %** -> {'过' if worst_case < 0.01 else '**不过**'}")
    print(f"  ★回归闸: control/fp32 应为 **2.976722%** (本脚本口径, 当前契约);"
          f" 它 = benchmark 的 2.981115% 减去两台仪器已知的 0.0044 pp 差")
    return 0


if __name__ == "__main__":
    sys.exit(main())
