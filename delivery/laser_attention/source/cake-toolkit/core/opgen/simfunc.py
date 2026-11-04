#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opgen.simfunc —— 功能层仿真 (PLAN 第 8.2 节的前半)。

它验什么
--------
**调度重写有没有改变数学语义。** 分块、在线 softmax、位掩码、因果尾段这些重写,
在 CPU 上就能判对错; 判错了再去写 AscendC 是白费一轮构建。

它不验什么
----------
不验硬件用法 (位置/布局/同步), 那是 M2/M3 的判据; 也不替代 opeval 的精度判定。
**按程序序执行的仿真器对漏了同步的核会愉快地给出正确结果** —— 所以它对同步问题
永远是绿的, 别拿它当同步验证。

参考实现: **基准自带的** tasks/level4/<op>/golden.py, 不自写 (见 semantics 的 SEM1)。

容差: 不拍阈值, 用参考自标定 (出处 ForgeMegakernel arXiv 2609.12379)
    G = fp64 精确算术; P = 基准 golden (工作精度); K = 待测调度
    B = δ(P, G)  参考本身离精确有多远
    C = δ(K, P)  待测离参考有多远
    判据 C < 2B —— 待测不得比"参考自己的误差"再差一倍以上。
这比固定阈值好在它随问题本身的病态程度自动缩放。
"""
from __future__ import annotations

import numpy as np

from .._refs import Ref

REFS = {
    "SIM1_tolerance_self_calibrated": Ref(
        what="功能层容差取 C < 2B (B=参考离精确的距离, C=待测离参考的距离), 不拍固定阈值",
        status="derived",
        evidence=[
            "出处 ForgeMegakernel (arXiv 2609.12379) 的中间态 oracle: 比的不是最终输出",
            "而是逐层中间量, 判据 C<2B。固定阈值在病态形状上要么放过要么误杀。",
            "★本仓尚未用它判过一次真实回归 —— 用过之后再升 measured。",
        ],
        verified="2026-09-22"),
    "SIM2_online_softmax_sentinel": Ref(
        what="分块在线 softmax 用有限哨兵 (-1e30) 时, **整行全遮蔽**的行会算成 v 的均值"
             "而不是 0; 而'首个分块全遮蔽'这一情形会自愈",
        status="measured",
        evidence=[
            "★我先写下的是'首块全遮蔽即出错', 被本模块的自检**证伪**: 首块死时 l 确实被",
            "  污染成块宽 (每个元素 exp(哨兵-哨兵)=exp(0)=1), 但下一个活块的修正因子",
            "  exp(-1e30 - 真最大值) = 0 会把污染整个抹掉 —— 它自愈。",
            "  真正不自愈的是**整行全死**: m 始终是哨兵, 修正因子恒 1, 每个元素权重恒 1,",
            "  结果成了 v 的均值; 而 golden 规定这种行输出 0 (torch.where(all_masked,0,..))。",
            "  且 l = S2 != 0, 所以'按 l==0 判死行'这个兜底也接不住。",
            "★现行 SFA 整行核靠 `m < kCubeNegBig*0.1f` 判死行, 改成分块形态时这条判断",
            "  必须一起搬过去, 且不能换成按 l 判。",
            "负例已构造成必然触发 (整行只选被因果剔除的列), 自检里断言它必须发散。",
        ],
        verified="2026-09-22"),
}

NEG = -np.inf


def delta(a, b):
    """两个数组的距离: 相对 Frobenius 范数。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.linalg.norm(a - b)
    n = np.linalg.norm(b)
    return d / n if n > 0 else d


def accept(kernel, golden, exact, name="", factor=2.0, floor=1e-12):
    """C < factor * B 的判定。返回 (通过?, 一句话)。"""
    B = delta(golden, exact)
    C = delta(kernel, golden)
    ok = C < max(factor * B, floor)
    return ok, (f"{name}  C={C:.3e}  B={B:.3e}  判据 C < {factor:g}B "
                f"({'过' if ok else '★未过'})")


# ------------------------------------------------------- SFA 的两种调度 (同一数学)

def _mask(rows, S2, si, causal):
    m = np.zeros((rows, S2), dtype=bool)
    r = np.repeat(np.arange(rows), si.shape[1])
    m[r, si.reshape(-1)] = True
    if causal:
        m &= np.arange(S2)[None, :] <= (np.arange(rows) + (S2 - rows))[:, None]
    return m


def sfa_rowwise(q, k, v, si, scale, causal, dtype=np.float64):
    """整行形态: 现行三段路径的数学形态。"""
    rows, S2 = q.shape[0], k.shape[0]
    mk = _mask(rows, S2, si, causal)
    s = (q.astype(dtype) @ k.astype(dtype).T) * dtype(scale)
    s = np.where(mk, s, NEG)
    m = s.max(axis=-1)
    dead = ~np.isfinite(m)
    e = np.exp(s - np.where(dead, 0.0, m)[:, None])
    e[~mk] = 0.0
    den = e.sum(axis=-1)
    p = np.where(dead[:, None], 0.0, e / np.where(den == 0, 1.0, den)[:, None])
    return p @ v.astype(dtype), p


def sfa_chunked_online(q, k, v, si, scale, causal, nT=512, mode="exact",
                       dtype=np.float64):
    """分块在线形态: 三段全融要用的数学形态。

    mode="exact"           正确形态: 真 -inf 掩码; 指数只在**活着**的列上做;
                           整块死的分块整块跳过, 不更新行最大值。
    mode="naive_sentinel"  **朴素移植**: 把现行整行核的写法原样搬到分块上 ——
                           加性有限哨兵 (-1e30) 掩码, 然后对**整块**求指数,
                           不区分死列死块。整行形态下它是对的 (行最大值只取一次),
                           分块形态下在"该行首个分块整块被遮蔽"时必错。
                           它是本模块的负例, **必须发散**。
    """
    rows, S2 = q.shape[0], k.shape[0]
    Dv = v.shape[1]
    mk = _mask(rows, S2, si, causal)
    naive = (mode == "naive_sentinel")
    neg = dtype(-1e30) if naive else NEG
    m = np.full(rows, neg, dtype=dtype)
    l = np.zeros(rows, dtype=dtype)
    acc = np.zeros((rows, Dv), dtype=dtype)
    for n0 in range(0, S2, nT):
        n1 = min(n0 + nT, S2)
        raw = (q.astype(dtype) @ k[n0:n1].astype(dtype).T) * dtype(scale)
        sub = mk[:, n0:n1]
        if naive:
            #  加性掩码, 与现行核的 Add(row, mk) 同形
            sc = raw + np.where(sub, dtype(0.0), neg)
            cm = sc.max(axis=-1)
            m_new = np.maximum(m, cm)
            corr = np.exp(m - m_new)
            e = np.exp(sc - m_new[:, None])      # ★对整块求指数, 不区分死列
        else:
            sc = np.where(sub, raw, neg)
            cm = np.where(sub.any(axis=-1), sc.max(axis=-1), neg)
            live = sub.any(axis=-1)
            m_new = np.where(live, np.maximum(m, cm), m)
            #  ★不要"先算出 NaN 再抹掉"。-inf - (-inf) = NaN, 而 NaN 一旦产生, 任何先
            #    碰到它的归约都会把整行毒掉; 硬件上 Select 与归约的先后顺序不像这里这样
            #    由我们完全掌握。站上 SFA 隐藏 44/45/61/62 与 MLA 隐藏 64 报的正是
            #    `NaN位置不匹配`。做法: 只在**活着**的位置上做指数, 死的位置直接置 0。
            d = np.where(m > NEG, m - m_new, NEG)
            corr = np.where(np.isfinite(d), np.exp(d), 0.0)
            arg = np.where(sub, raw - m_new[:, None], NEG)
            e = np.where(sub, np.exp(np.where(np.isfinite(arg), arg, NEG)), 0.0)
            e = np.where(live[:, None], e, 0.0)
        l = l * corr + e.sum(axis=-1)
        acc = acc * corr[:, None] + e @ v[n0:n1].astype(dtype)
        m = m_new
    dead = (l == 0)
    y = np.where(dead[:, None], 0.0, acc / np.where(dead, 1.0, l)[:, None])
    return y, None


#  ---- 负例控制 (门禁 G1) ----
def _nc_tolerance():
    """SIM1 的负例: 明显错的结果必须**不**通过 C<2B。"""
    g = np.array([1.0, 2.0, 3.0])
    ok, _ = accept(g * 1.5, g, g * (1 + 1e-9))
    return not ok


def _nc_sentinel():
    """SIM2 的负例: 朴素移植在整行全死时必须发散 (与 selftest 第 3 条同一构造)。"""
    q, k, v, si, sc, _ = _case(4, topK=8, causal=True)
    si = si.copy()
    si[0] = np.arange(k.shape[0] - 8, k.shape[0])
    y0, _ = sfa_rowwise(q, k, v, si, sc, True)
    y_bad, _ = sfa_chunked_online(q, k, v, si, sc, True, nT=32, mode="naive_sentinel")
    return bool(delta(y_bad, y0) > 1e-6)


NEGATIVE_CONTROLS = {
    "SIM1_tolerance_self_calibrated": _nc_tolerance,
    "SIM2_online_softmax_sentinel": _nc_sentinel,
}


# ------------------------------------------------------------------ 自检

def _case(seed, rows=16, S2=256, Dk=32, Dv=32, topK=24, causal=True):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((rows, Dk))
    k = rng.standard_normal((S2, Dk))
    v = k[:, :Dv]
    si = np.stack([rng.permutation(S2)[:topK] for _ in range(rows)])
    return q, k, v, si, 1.0 / np.sqrt(Dk), causal




def _bench_golden(q, k, v, si, scale, causal):
    """**基准自带的** golden, 不自写 (见 semantics 的 SEM1)。在 CPU 上跑, 不占卡。

    golden 的签名是 [B,N1,S1,Dk] 等四维, 这里把单 (b,n) 切片包成 B=N1=N2=1 喂进去。
    取不到 (没装 torch / 路径变了) 时返回 None, 由调用方标为"跳过"而不是"通过" ——
    "查不到"与"一致"长得一样是本仓反复强调的坑。
    """
    import importlib.util, os, sys                                # noqa: PLC0415
    from core import _benchpath                                   # noqa: PLC0415
    tasks = _benchpath.bench_tasks()          # 位置不写死 (2026-10-20 审计); 没有 = 跳过
    if not tasks:
        return None
    gp = os.path.join(tasks, "level4", "sparse_flash_attention", "golden.py")
    if not os.path.exists(gp):
        return None
    try:
        import torch                                              # noqa: PLC0415
    except Exception:                                             # noqa: BLE001
        return None
    spec = importlib.util.spec_from_file_location("sfa_golden", gp)
    gm = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("sfa_golden", gm)
    spec.loader.exec_module(gm)
    t = lambda x, d=torch.float32: torch.tensor(np.asarray(x), dtype=d)   # noqa: E731
    out = gm.sparse_flash_attention(
        t(q)[None, None], t(k)[None, None], t(v)[None, None],
        torch.tensor(np.asarray(si), dtype=torch.int32)[None, None],
        float(scale), "BNSD", bool(causal))
    return out[0, 0].double().numpy()


def selftest(verbose=True):
    bad = 0
    checks = []
    skipped = []

    #  1) 分块在线 == 整行 (多种分块宽度与形状)
    for seed in (0, 1, 2):
        for nT in (32, 64, 128, 256):
            q, k, v, si, sc, ca = _case(seed)
            y0, _ = sfa_rowwise(q, k, v, si, sc, ca)
            y1, _ = sfa_chunked_online(q, k, v, si, sc, ca, nT=nT)
            ok, msg = accept(y1, y0, y0, name=f"分块 nT={nT:<4} seed={seed}")
            checks.append((ok, msg))

    #  2) **整行全遮蔽**必须输出 0 (golden 的 torch.where(all_masked, 0, ...))
    q, k, v, si, sc, _ = _case(3, causal=True)
    rows, S2 = q.shape[0], k.shape[0]
    si_bad = si.copy()
    si_bad[0, :] = S2 - 1                       # 第 0 行只选最后一列, 因果下必被剔除
    y0, _ = sfa_rowwise(q, k, v, si_bad, sc, True)
    y1, _ = sfa_chunked_online(q, k, v, si_bad, sc, True, nT=64)
    ok = np.allclose(y0[0], 0.0) and np.allclose(y1[0], 0.0)
    checks.append((ok, f"整行全遮蔽两种形态都输出 0 ({'过' if ok else '★未过'})"))

    #  3) ★哨兵陷阱。★负例必须**构造成必然触发**, 不能靠碰运气 —— 也不能靠猜哪种情形
    #     才是陷阱: 我先猜"首块全遮蔽即出错", 被这里的对照证伪 (首块死会被下一个活块的
    #     修正因子 exp(-1e30-真最大值)=0 抹掉, 自愈)。真正不自愈的是**整行全死**。
    #  因果右下角对齐: 第 r 行允许列 j <= r + (S2 - rows)。rows=16, S2=256 时第 0 行
    #  允许到 240, 所以只有 241..255 这 15 列会被剔除 —— topK 必须 <= 15 才构造得出整行死。
    NT = 32
    q, k, v, si, sc, _ = _case(4, topK=8, causal=True)
    si = si.copy()
    si[0] = np.arange(k.shape[0] - 8, k.shape[0])             # 248..255, 全在因果之外
    y0, _ = sfa_rowwise(q, k, v, si, sc, True)
    assert np.allclose(y0[0], 0.0), "构造失败: 第 0 行没有整行全遮蔽"
    y_bad, _ = sfa_chunked_online(q, k, v, si, sc, True, nT=NT, mode="naive_sentinel")
    diverged = delta(y_bad, y0) > 1e-6
    checks.append((diverged,
                   f"负例: 朴素移植 (加性 -1e30 哨兵 + 整块求指数) -> 结果发散 "
                   f"δ={delta(y_bad, y0):.3e} ({'过 (抓住了)' if diverged else '★未过 (判据抓不住)'})"))

    #  4) 真 -inf + 跳过死块 在同一输入上必须**不**发散 (正例)
    y_ok, _ = sfa_chunked_online(q, k, v, si, sc, True, nT=NT)
    ok, msg = accept(y_ok, y0, y0, name="正例: 真 -inf + 只在活列求指数")
    checks.append((ok, msg))

    #  5) 与**基准自带的 golden** 对照 (算子级参考不自写)
    q, k, v, si, sc, ca = _case(7)
    g = _bench_golden(q, k, v, si, sc, ca)
    if g is None:
        #  ★不能按通过计: 首版 append((True, "跳过...")) —— 于是"没查成"在计数上等同
        #    "通过", 正是静默缺陷。改为单独计数并在汇总里显式列出。
        skipped.append("对照基准 golden (取不到 torch 或 tasks/.../golden.py)")
    else:
        y_exact, _ = sfa_rowwise(q, k, v, si, sc, ca)             # fp64, 当作精确算术 G
        y_chunk, _ = sfa_chunked_online(q, k, v, si, sc, ca, nT=64)
        ok, msg = accept(y_chunk, g, y_exact, name="分块在线 vs 基准 golden")
        checks.append((ok, msg))
        ok2, msg2 = accept(np.asarray(sfa_rowwise(q, k, v, si, sc, ca)[0]), g, y_exact,
                           name="整行     vs 基准 golden")
        checks.append((ok2, msg2))

    for ok, msg in checks:
        if not ok:
            bad += 1
        if verbose or not ok:
            print(f"  {'✓' if ok else '✗'} {msg}")
    if skipped:
        print(f"  ★没查成 {len(skipped)} 条 (既不算过也不算不过):")
        for x in skipped:
            print(f"      {x}")
    if verbose:
        print(f"\nsimfunc 自检: {len(checks)} 条, {bad} 条未过, {len(skipped)} 条没查成")
    return bad


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
