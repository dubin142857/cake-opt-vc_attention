#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""卡的**稳定性**体检 —— 交替切换参数的多次启动, 看有没有偶发的整任务错。

★为什么不能用 smoke_one.py 顶替 (F218):
  `smoke_one.py` (devguard 的 vc910_card_clean) 只跑**一次** q=1 k=1, 核对一个解析已知的结论。
  2026-10-30 实测: 卡 7 通过了它, 却在本脚本下失败 6/40; 卡 3、卡 4 则 0/40。
  被漏掉的那类缺陷需要"交替切换 expcast 的连续启动 + 约 15% 的命中率"才显形,
  单次复现必然放行。**守卫说"可信"只等于"它查的那条过了"。**

★被查缺陷的形态 (卡 7):
  整个任务的输出被一个**逐行标量**乘错而方向保持 —— 与正确结果余弦 0.9994, 模比 266
  (或其倒数 0.00376)。坏 head 在不同启动之间漂移, 但命中时坏值**逐位相同**。
  `chunks=1` 的形状上 `l` 恰等于 `massAcc`, 故这一标量就是行和。

★形状取 q=65 k=257 d=64: 查询尾块只有 1 行有效 (第 64 行), 是实测命中率最高的一例。
  交替 expcast 是必要的 —— 同参数连跑 60 次**不**复现。

★判据: 失败 0 次才算过。这是**必要非充分**条件, 与 vc910_card_clean 同性质:
  它过了只说明这张卡在这一项上稳, 不说明内核对; 它不过则该卡上的一切数值判决无效。
"""
import argparse, sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=3, help="物理卡号; 0 卡是 CI runner, 禁用")
    ap.add_argument("--launches", type=int, default=60,
                    help="总启动次数, 一半 expcast=False 一半 True。"
                         "按实测 15%% 的单次命中率, 30 次 False 的漏检率约 0.8%%")
    ap.add_argument("--queries", type=int, default=65)
    ap.add_argument("--keys", type=int, default=257)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=91023)
    ap.add_argument("--tol", type=float, default=1e-3, help="逐行相对 RMSE 的判阈")
    a = ap.parse_args()
    if a.device == 0:
        print("拒绝: 0 卡是管理员部署 CI runner 用的", file=sys.stderr)
        return 2

    import torch, torch_npu  # noqa: F401
    import vc_attention_ascend as V
    V.load_native()
    torch.npu.set_device(a.device)
    dev = f"npu:{a.device}"
    from vc_attention_ascend import prepare, run_prepared
    from vc_attention_ascend.reference import payload_reference

    nq, nk, d = a.queries, a.keys, a.dim
    q, k, v = [torch.randn((1, 2, n, d), generator=torch.Generator().manual_seed(a.seed + i))
               .to(torch.float16) for i, n in enumerate((nq, nk, nk))]
    payload = prepare(*(x.to(dev) for x in (q, k, v)), layout="HND")
    cpu = {kk: (vv.cpu() if isinstance(vv, torch.Tensor) else vv) for kk, vv in payload.items()}
    refs = {c: payload_reference(cpu, expcast=c).float() for c in (False, True)}

    fails, worst, where = 0, 0.0, []
    for it in range(a.launches):
        cast = bool(it % 2)
        nat = run_prepared(payload, expcast=cast).float().cpu()
        ref = refs[cast]
        per = ((nat - ref).square().mean(-1).sqrt()
               / ref.square().mean(-1).sqrt().clamp_min(1e-12))[0]
        bad = sorted({(int(x[0]), int(x[1])) for x in (per > a.tol).nonzero()})
        if bad:
            fails += 1
            worst = max(worst, float(per.max()))
            if len(where) < 5:
                where.append({"launch": it, "expcast": cast, "bad_head_row": bad[:4]})
    print(f"卡 {a.device} 形状 q={nq} k={nk} d={d}: {a.launches} 次交替启动里"
          f" 不稳定 {fails} 次")
    if fails:
        print(f"  最坏逐行相对 RMSE {worst:.6g}; 前几处: {where}", file=sys.stderr)
        print("  ★本卡上的数值判决不可信。换卡 (实测干净: 3 · 4), 不要在此卡上判代码。",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
