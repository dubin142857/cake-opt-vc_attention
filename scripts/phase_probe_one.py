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
             "-", "-", "-"]
WAITS = {"wait_o", "wait_s", "wait_p", "wait_o_used"}
#  ★AIV 的这四个相位是**每块**的, 同一 chunk 内逐块覆写故只留末块 ——
#    直接求和会漏掉 n_in-1 块。还原时要乘 n_in (判据: 还原后应与每 chunk 的窗口对上)。
#    AIC 的五个相位都是每 chunk 一次, 不必还原。
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
    for core in rec:
        for role_i, role in enumerate(core):
            kind = "AIC" if role_i == 0 else "AIV"
            for samp in role:
                if samp[0] != 1:
                    continue
                by_key.setdefault((kind, samp[2]), []).append(samp[4:12])
    if not by_key:
        print("★没有有效样本 —— 确认跑的是 PhaseProbe<true> 的 profile 内核", file=sys.stderr)
        return 1
    print(f"形状 q={a.queries} k={a.keys} h={a.heads} d={a.dim} expcast={a.expcast} "
          f"· 采样 {sorted(by_key)}")
    for kind, key in sorted(by_key):
        rows = by_key[(kind, key)]
        names = AIC_NAMES if kind == "AIC" else AIV_NAMES
        med = [st.median([r[i] for r in rows]) for i in range(8)]
        #  还原每块相位: n_in = min(CHUNK, tiles - key/NBLK)
        tiles = -(-a.keys // NBLK)
        n_in = min(CHUNK, max(1, tiles - key // NBLK))
        med = [v * n_in if n in PER_TILE else v for n, v in zip(names, med)]
        tot = sum(v for n, v in zip(names, med) if n != "-")
        if tot <= 0:
            continue
        wait = sum(v for n, v in zip(names, med) if n in WAITS)
        tag = "起始块" if key == 0 else "稳态"
        other = "AIV" if kind == "AIC" else "AIC"
        print(f"\n--- {kind} key={key} ({tag}, {len(rows)} 份样本, n_in={n_in}) "
              f"每 chunk 合计 {tot:.0f} ticks = {tot*TICK_NS/1000:.2f} us ---")
        for n, vv in sorted(zip(names, med), key=lambda kv: -kv[1]):
            if n == "-":
                continue
            mark = f"  ← 等 {other}" if n in WAITS else ("  (每块 x%d)" % n_in if n in PER_TILE else "")
            print(f"      {n:12s} {vv:8.0f} ticks  {vv*TICK_NS/1000:7.3f} us  "
                  f"{vv/tot*100:5.1f}%{mark}")
        print(f"      ★等 {other} {wait/tot*100:.1f}% · {kind} 自身工作 "
              f"{(tot-wait)/tot*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
