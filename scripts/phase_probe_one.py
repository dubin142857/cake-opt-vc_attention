#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小相位探针 —— 在**基准形状**上直接读 PhaseProbe 的每 chunk 分解。

★为什么单独写 (F205 的方法学缺口): profile_ascend910b.py 会先跑校验与 CPU 参考,
  在 37741^2 上跑不完超时, 于是我上一轮改用小形状 q=256 k=16384 h=2 ——
  那只有 8 个任务占 8 个核, 与 24 核满载的基准负载**不是同一回事**,
  等待比例不可外推。本脚本只做 prepare + 预热 + 一次 profile_prepared。

★相位含义见 vc_kernel.cpp 的 probe.Mark 注释; 全部是**不插屏障**的 Mark,
  量的是发射与等待的墙上时间。wait_o / wait_s 紧跟两处 CrossCoreWaitFlag,
  故"等多久"与"干多久"是分开的。
★采样位置: PhaseProbe::Begin 只对**每核第一个任务**生效, 且只在
  tile == last*s/(PROFILE_SAMPLES-1) 的若干位置置位; 取 tile>0 的那个作稳态。
"""
import argparse, statistics as st, sys

AIV_NAMES = ["t1_setup", "t1_vec", "wait_o", "t2_vec", "t2_publish",
             "chunk_tail", "pv_work", "wait_s"]
#  ★AIC 侧 (role 0) 自 F209 起也有标记, 与 AIV 同口径: 等待单列。
AIC_NAMES = ["qk_issue", "chunk_drain", "wait_p", "wait_o_used", "pv_issue",
             "-", "claim_l0c", "-"]
WAITS = {"wait_o", "wait_s", "wait_p", "wait_o_used", "claim_l0c"}
#  ★AIV 的这四个相位是**每块**的, 同一 chunk 内逐块覆写故只留末块 ——
#    直接求和会漏掉 n_in-1 块。AIC 的五个相位都是每 chunk 一次, 不必还原。
#  ★★2026-10-31 改正 (E41 判据 (a) 实测): 原来按 "末块值 x n_in" 还原, 并在注释里
#    写着"判据: 还原后应与每 chunk 的窗口对上" —— 而这条判据**从来没有被执行过**,
#    脚本只打印不断言。实测它不成立: 同一形状两次采样, x n_in 得到 span 的
#    **121.2%** 与 **102.1%** (末块比平均块贵, 且贵多少本身不稳)。
#    改用探针自带两端时戳的**恒等式**, 无假设:
#        span = values_[13] - values_[12]
#        每块相位的真实合计 = span - Σ(每 chunk 一次的相位)
#    四个每块相位之间怎么分, 仍只能按末块比例推 —— 那一步是假设, 输出里标出来。
#    AIC 侧 Σ == span 逐次吻合 (100.0% / 100.6%), 说明方法本身没问题, 问题只在还原假设。
PER_TILE = {"t1_setup", "t1_vec", "t2_vec", "t2_publish"}
NBLK, CHUNK = 128, 4
TICK_NS = 10.77


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=3)
    ap.add_argument("--queries", type=int, default=37741)
    ap.add_argument("--keys", type=int, default=37741)
    ap.add_argument("--heads", type=int, default=28)
    ap.add_argument("--dim", type=int, default=128)
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
    q, k, v = [torch.randn(1, a.heads, n, a.dim, dtype=torch.bfloat16, device=dev)
               for n in (a.queries, a.keys, a.keys)]
    payload = vc.prepare(q, k, v, layout="HND")
    vc.run_prepared(payload, expcast=a.expcast)   # 预热, 使探针跑在稳态
    torch.npu.synchronize()
    _, rec = vc.profile_prepared(payload, expcast=a.expcast)
    torch.npu.synchronize()
    rec = rec.cpu().tolist()

    by_key = {}          # (role_kind, key) -> [相位向量]
    by_rec = {}          # 同键 -> [整条记录], 恒等式分解要用 [12]/[13] 两端时戳
    for core in rec:
        for role_i, role in enumerate(core):
            kind = "AIC" if role_i == 0 else "AIV"
            for samp in role:
                if samp[0] != 1:
                    continue
                by_key.setdefault((kind, samp[2]), []).append(samp[4:12])
                by_rec.setdefault((kind, samp[2]), []).append(samp)
    if not by_key:
        print("★没有有效样本 —— 确认跑的是 PhaseProbe<true> 的 profile 内核", file=sys.stderr)
        return 1
    print(f"形状 q={a.queries} k={a.keys} h={a.heads} d={a.dim} expcast={a.expcast} "
          f"· 采样 {sorted(by_key)}")
    for kind, key in sorted(by_key):
        rows = by_key[(kind, key)]
        names = AIC_NAMES if kind == "AIC" else AIV_NAMES
        med = [st.median([r[i] for r in rows]) for i in range(8)]
        tiles = -(-a.keys // NBLK)
        n_in = min(CHUNK, max(1, tiles - key // NBLK))
        #  ---- 恒等式分解 (取代 "x n_in") ----
        span = st.median([r[13] - r[12] for r in by_rec[(kind, key)]])
        once = sum(v for n, v in zip(names, med) if n != "-" and n not in PER_TILE)
        tile_tot = span - once          # **精确**: 四块每块相位的真实合计
        tile_last = sum(v for n, v in zip(names, med) if n in PER_TILE)
        #  ★判据必须真的跑: 恒等式两边对不上 2% 就说明相位没覆盖整个 chunk,
        #    这组读数不能用来归因 (E23: 先修仪器)。
        if tile_tot < -0.02 * span:
            print(f"★{kind} key={key}: 每 chunk 一次的相位 Σ ({once:.0f}) 已超过 span "
                  f"({span:.0f}) —— 相位布点与 Begin/Finish 不自洽, 本组读数不可用",
                  file=sys.stderr)
            continue
        med = [(v / tile_last * tile_tot if tile_last else 0.0) if n in PER_TILE else v
               for n, v in zip(names, med)]
        tot = span
        if tot <= 0:
            continue
        wait = sum(v for n, v in zip(names, med) if n in WAITS)
        tag = "起始块" if key == 0 else "稳态"
        other = "AIV" if kind == "AIC" else "AIC"
        print(f"\n--- {kind} key={key} ({tag}, {len(rows)} 份样本, n_in={n_in}) "
              f"每 chunk span {span:.0f} ticks = {span*TICK_NS/1000:.2f} us ---")
        print(f"      [恒等式] 每 chunk 一次的相位 Σ {once:.0f} + 每块相位真实合计 "
              f"{tile_tot:.0f} = span {span:.0f}"
              f"   (旧式 x{n_in} 会给 {tile_last*n_in:.0f}, 即 span 的 "
              f"{100*tile_last*n_in/span:.0f}%)")
        for n, vv in sorted(zip(names, med), key=lambda kv: -kv[1]):
            if n == "-":
                continue
            mark = (f"  ← 等 {other}" if n in WAITS else
                    ("  (每块合计, 按末块比例分摊 —— 这一步是假设)" if n in PER_TILE else ""))
            print(f"      {n:12s} {vv:8.0f} ticks  {vv*TICK_NS/1000:7.3f} us  "
                  f"{vv/tot*100:5.1f}%{mark}")
        print(f"      ★等 {other} {wait/tot*100:.1f}% · {kind} 自身工作 "
              f"{(tot-wait)/tot*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
