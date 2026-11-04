#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opgen.semantics —— 原语语义库 (PLAN 的 M1), 带**必要性闸 G9**。

一个原语 = 可执行的参考语义 + 形状规则 + dtype 规则 + 依赖的芯片能力 + **证人**。

为什么有闸 (2026-09-22 立, 由第一版自身触发)
--------------------------------------------
第一版随手写了 10 个函数, 其中 4 个不合格:
  scatter_mask  把"索引表->布尔掩码"(通用)与"因果上限"(attention 专属)焊死 ——
                别的算子要用前者就得拖一个它不需要的参数;
  sub_row / mul_row / div_row  是同一件事 (按行广播的二元运算) 的三个变体;
  sfa_reference 是**算子级**参考而非原语 —— 放进来意味着每接一个 cann-bench 算子
                就往语义库加一个函数, 这正是"积重难返"的形状。
更根本的一处: 算子级参考**根本不该自己写**。基准自带 golden
(tasks/level4/<op>/golden.py), 再写一份 numpy 参考就是又一处会发散的第二实现。
功能仿真应当直接拿那份 golden 当参考 —— 对每个 cann-bench 算子免费可用。

G9 必要性闸 (五条全过才准入)
---------------------------
  N1 证人 >= 2: 出现在两个**结构不同**的算子里。只有一个证人的记入 PENDING, 不进库。
  N2 不可由已有原语组合。能组合的是冗余; 若硬件上确是一条指令, 那属于**下降模板**。
  N3 形状/dtype 规则不依赖任何算子的参数名。
  N4 声明依赖的芯片能力 (可为空)。
  N5 一个正例 + 一个边界负例。

`python -m core.opgen gate` 会跑这五条。**PENDING 表必须留着** —— 删掉等于下次
再讨论一遍"要不要加 scan"。

命名跟 HIVM 走 (matmul / vselect / vgather / vcast / reduce), 不自造词。
"""
from __future__ import annotations

import glob
import os

import numpy as np

_CASES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "cases")

from .._refs import Ref

REFS = {
    "SEM1_single_source": Ref(
        what="原语语义只有一份定义; 算子级参考一律用基准自带的 golden, 不自写",
        status="derived",
        evidence=[
            "反面: 同一语义两处手写必然发散 (easyasc 的 a5 matmul sim != silicon 即此类)。",
            "本库只放原语; 算子级正确性由 tasks/level4/<op>/golden.py 定义, 不复制。",
            "★尚未与真机差分测试过 —— 升 measured 的条件见 PLAN 的 M5 验收。",
        ],
        verified="2026-09-22"),
    "SEM2_broadcast_row_is_the_hard_one": Ref(
        what="按行广播的二元运算在硬件上不是一条指令, 是整个分块形态的成本枢纽",
        status="measured",
        evidence=[
            "src1BlkStride=0 (repeat 内的块广播) 本会话实测不可用: 把 Brcb 换成逐键",
            "Duplicate 后误差逐位相同 (MERE 0.968644/0.726743/0.365384), 证明错在 blkStride。",
            "于是 x[r,j] -= v[r] 只能逐行发指令 (每条只覆盖一个分块宽度), 或整段留在",
            "寄存器里 (regbase SIMT)。分块宽度 <-> 指令效率 <-> UB 容量的三方耦合由此而来。",
        ],
        verified="2026-09-22"),
}


class Prim:
    """一个原语。字段即门禁的输入, 不是文档。"""
    __slots__ = ("name", "what", "ref", "shape_rule", "dtype_rule", "requires",
                 "witnesses", "witness_pat", "pos", "neg")

    def __init__(self, name, what, ref, shape_rule, dtype_rule, requires, witnesses,
                 witness_pat=None, pos=None, neg=None):
        self.name, self.what, self.ref = name, what, ref
        self.shape_rule, self.dtype_rule = shape_rule, dtype_rule
        self.requires = tuple(requires)          # 依赖的芯片能力 id (opgen.caps)
        self.witnesses = tuple(witnesses)        # 本仓里用到它的算子 (cases/ 下的目录名)
        #  ★证人必须**可核实**: 这条正则在该算子的 op_kernel 源码里必须命中。
        #    2026-09-22 建表当天实撞: 我把 cummin 列为 reduce 的证人, 一扫发现它的
        #    op_kernel 里 ReduceMax/ReduceSum/WholeReduce **一条都没有** (它用的是
        #    Min/Max 的扫描形态)。证人表是手写的断言, 断言会错 —— 所以 N1 要去源码里查。
        self.witness_pat = witness_pat
        self.pos, self.neg = pos, neg            # 正例 / 边界负例, 均为无参可调用


# ------------------------------------------------------------------ 参考语义

def _matmul(a, b, transa=False, transb=False):
    a = a.T if transa else a
    b = b.T if transb else b
    return np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)


_EW = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide,
       "max": np.maximum, "min": np.minimum, "exp": np.exp, "abs": np.abs,
       "neg": np.negative, "recip": np.reciprocal}


def _elementwise(kind, *xs):
    f = _EW[kind]
    return f(*[np.asarray(x, dtype=np.float64) for x in xs])


_RED = {"max": np.max, "min": np.min, "sum": np.sum}


def _reduce(kind, x, axis=-1):
    return _RED[kind](np.asarray(x, dtype=np.float64), axis=axis)


def _broadcast_row(kind, x, v, axis=-1):
    """x 的 axis 维与 v 的行对齐后做逐元素运算: x[..., r, j] op v[..., r]。

    ★硬件上这一步的代价见 REFS SEM2 —— 这里只定语义。
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return _EW[kind](x, np.expand_dims(v, axis))


def _select(cond, x, y):
    """cond 为真取 x, 否则取 y。x/y 可为标量。对应 HIVM 的 vselect。"""
    return np.where(np.asarray(cond, dtype=bool), x, y)


def _gather(src, idx, axis=-1):
    return np.take_along_axis(np.asarray(src), np.asarray(idx), axis=axis)


def _cast(x, dtype):
    return np.asarray(x).astype(dtype)


def _band_mask(rows, cols, lo=None, hi=None):
    """带状掩码: 第 r 行保留列 j 满足 lo(r) <= j <= hi(r)。

    因果 (右下角对齐) = hi(r) = r + (cols - rows), lo 为 None。
    三角 / 滑窗 / 局部注意力都是它的实例 —— 所以参数是 lo/hi 两条边界曲线,
    不是某个算子的 `is_causal`。
    """
    j = np.arange(cols)[None, :]
    m = np.ones((rows, cols), dtype=bool)
    if lo is not None:
        m &= j >= np.asarray(lo)[:, None]
    if hi is not None:
        m &= j <= np.asarray(hi)[:, None]
    return m


# ------------------------------------------------------------------ 正例 / 负例

def _pos_matmul():
    a = np.arange(6.0).reshape(2, 3)
    return np.allclose(_matmul(a, a, transb=True), a @ a.T)


def _neg_matmul():
    """收缩维不匹配必须抛错, 不能静默广播。"""
    try:
        _matmul(np.ones((2, 3)), np.ones((4, 5)))
    except ValueError:
        return True
    return False


def _pos_broadcast_row():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    return np.allclose(_broadcast_row("sub", x, np.array([1.0, 3.0])), [[0, 1], [0, 1]])


def _neg_broadcast_row():
    """行数不匹配必须抛错 —— 静默广播会把 [rows] 当成 [1, rows] 沿错轴展开。"""
    try:
        _broadcast_row("sub", np.ones((2, 4)), np.ones(4))
    except ValueError:
        return True
    return False


def _pos_reduce():
    return _reduce("max", np.array([[1.0, 5.0], [7.0, 2.0]])).tolist() == [5.0, 7.0]


def _neg_reduce():
    """空轴上的 max 必须抛错而不是返回 -inf —— 后者会被当成"整行遮蔽"误判。"""
    try:
        _reduce("max", np.zeros((2, 0)))
    except ValueError:
        return True
    return False


def _pos_select():
    return _select([True, False], np.array([1.0, 2.0]), -np.inf).tolist() == [1.0, -np.inf]


def _neg_select():
    """条件形状不匹配必须抛错。"""
    try:
        _select(np.ones((2, 3), dtype=bool), np.ones((2, 4)), 0.0)
    except ValueError:
        return True
    return False


def _pos_band():
    m = _band_mask(3, 5, hi=np.arange(3) + (5 - 3))
    return m[0].tolist() == [1, 1, 1, 0, 0] and m[2].tolist() == [1, 1, 1, 1, 1]


def _neg_band():
    """lo > hi 的行必须整行为假 (不是抛错, 也不是整行为真)。"""
    m = _band_mask(2, 4, lo=np.array([3, 0]), hi=np.array([1, 3]))
    return (not m[0].any()) and m[1].all()


def _pos_gather():
    src = np.array([[10.0, 20.0, 30.0]])
    return _gather(src, np.array([[2, 0]])).tolist() == [[30.0, 10.0]]


def _neg_gather():
    try:
        _gather(np.zeros((1, 3)), np.array([[5]]))
    except IndexError:
        return True
    return False


def _pos_ew():
    return _elementwise("exp", np.array([0.0])).tolist() == [1.0]


def _neg_ew():
    try:
        _elementwise("add", np.ones((2, 3)), np.ones((4, 5)))
    except ValueError:
        return True
    return False


def _pos_cast():
    return _cast(np.array([1.7]), np.float32).dtype == np.float32


def _neg_cast():
    """float -> int 必须是**截断**而不是四舍五入 (与 RoundMode 的默认不同, 写清楚)。"""
    return _cast(np.array([1.7, -1.7]), np.int32).tolist() == [1, -1]


# ------------------------------------------------------------------ 原语表

PRIMS = [
    Prim("matmul", "矩阵乘, 可选转置任一操作数", _matmul,
         "[M,K] x [K,N] -> [M,N] (转置后)", "累加按 fp32 及以上", (),
         ("sfa_bench", "mha_fast", "mla_fast"),
         r"\bMmad\(|::Matmul|MatmulImpl", _pos_matmul, _neg_matmul),
    Prim("elementwise", "逐元素一元/二元运算 (add/sub/mul/div/max/min/exp/abs/neg/recip)",
         _elementwise, "形状相同", "同 dtype", (),
         ("sfa_bench", "dynamic_quant", "cummin", "apply_rotary_pos_emb_bench"),
         r"\b(Muls|Adds|Add|Sub|Mul|Div|Exp|Abs)\(", _pos_ew, _neg_ew),
    Prim("reduce", "沿一个轴归约 (max/min/sum)", _reduce,
         "去掉该轴", "fp32 累加", (),
         #  ★cummin 曾被列为证人, 实扫发现它没有任何 Reduce* —— 已删。
         ("sfa_bench", "mha_fast", "dynamic_quant"),
         r"(ReduceMax|ReduceSum|ReduceMin|WholeReduce|BlockReduce)", _pos_reduce, _neg_reduce),
    Prim("broadcast_row", "x 与按行的一维量做逐元素运算 (x[..,r,j] op v[..,r])", _broadcast_row,
         "v 的长度 = x 在 axis 上的相邻维", "同 dtype", (),
         ("sfa_bench", "mha_fast", "dynamic_quant"),
         r"\b(Brcb|Adds|Muls)\(", _pos_broadcast_row, _neg_broadcast_row),
    Prim("select", "按条件在两者间取值 (HIVM vselect)", _select,
         "cond 与 x 同形; y 可为标量", "x/y 同 dtype", (),
         ("sfa_bench", "mha_fast"),
         r"\b(Select|Duplicate|Scatter)\(", _pos_select, _neg_select),
    Prim("band_mask", "带状掩码: 每行保留 lo(r)..hi(r) 的列 (因果/三角/滑窗都是它的实例)",
         _band_mask, "[rows, cols] 的 bool", "bool", (),
         ("sfa_bench", "mha_fast", "mla_fast"),
         r"causal|Duplicate\(", _pos_band, _neg_band),
    Prim("gather", "沿一个轴按下标取值 (HIVM vgather)", _gather,
         "idx 与输出同形", "idx 为整型", (),
         ("cummin", "apply_rotary_pos_emb_bench"),
         r"\bGather\(|Gatherb\(", _pos_gather, _neg_gather),
    Prim("cast", "dtype 转换 (HIVM vcast)", _cast, "形状不变", "目标 dtype 显式给出", (),
         ("sfa_bench", "dynamic_quant", "mla_fast"),
         r"\bCast\(", _pos_cast, _neg_cast),
]

#  ---- 只有一个证人, **不进库**。留着是为了不必下次再讨论一遍。
PENDING = [
    ("scatter", "按下标写入 (SFA 用它一条指令建 topK 掩码)", ["sfa_bench"],
     "第二个证人候选: moe_init_routing / roi_align 的反向散射 —— 未核实"),
    ("scan", "沿轴的前缀归约 (cummin/cumsum)", ["cummin"],
     "第二个证人候选: lstm 的时间步递推 —— 形态不同, 未必是同一原语"),
    ("interleave", "偶奇交错 / 反交错 (HIVM vinterleave)", ["apply_rotary_pos_emb_bench"],
     "第二个证人候选: 复数形态的 FFT 类算子 —— 本仓没有"),
    ("topk", "取前 K 大及其下标", ["moe_gating_top_k_softmax_bench"],
     "第二个证人候选: 稀疏 attention 的索引生成 —— 本仓的 SFA 索引是外部给的, 不算"),
]

PRIM_BY_NAME = {p.name: p for p in PRIMS}



def _case_kernel_dirs(case):
    return sorted(glob.glob(os.path.join(_CASES, case, "workspace", "csrc", "ops", "*", "op_kernel")))


def _check_witnesses(p):
    """去每个证人算子的 op_kernel 源码里核实这条原语确实被用到。

    返回 (核实不到的证人, **本机没有该 case 因而没查成的证人**)。
    ★两者必须分开且都要报出来: 前者是判据命中, 后者是**覆盖缺口**。
      首版把后者静默跳过 —— 既不报也不算失败, 于是"这条证人从没被查过"看起来和
      "查过且通过"一模一样。所有 action 必须有明确反馈信号。
    """
    if not p.witness_pat:
        return []
    import re as _re
    pat = _re.compile(p.witness_pat)
    bad, nocase, unreadable = [], [], []
    for w in sorted(set(p.witnesses)):
        dirs = _case_kernel_dirs(w)
        if not dirs:
            nocase.append(w)              # 本机没有这个 case: 记下来并报出, 不静默
            continue
        hit = False
        for d in dirs:
            for root, _, names in os.walk(d):
                for n in names:
                    if not n.endswith((".h", ".cpp", ".hpp")):
                        continue
                    try:
                        with open(os.path.join(root, n), errors="replace") as fh:
                            if pat.search(fh.read()):
                                hit = True
                                break
                    except OSError as e:
                        #  ★读不了的文件不能当"没命中": 那会让"查不成"伪装成"查过没有"。
                        unreadable.append(f"{os.path.join(root, n)} ({type(e).__name__})")
                if hit:
                    break
            if hit:
                break
        if not hit:
            bad.append(w)
    return bad, nocase + [f"读不了: {u}" for u in unreadable]


#  ---- 负例控制 (门禁 G1) ----
def _nc_single_source():
    """SEM1 的负例: 库里若混进算子级参考 (以 op 名之外的算子名出现), 必须被发现。
    这里用一个可机器检查的代理: 原语名不得含算子名 (sfa/mha/mla/...)。"""
    banned = ("sfa", "mha", "mla", "cummin", "roi", "moe", "lstm")
    return not any(b in p.name.lower() for p in PRIMS for b in banned)


def _nc_broadcast_row():
    """SEM2 的负例: 按行广播若退化成沿错轴展开, 形状不匹配必须抛错。"""
    return _neg_broadcast_row()


NEGATIVE_CONTROLS = {
    "SEM1_single_source": _nc_single_source,
    "SEM2_broadcast_row_is_the_hard_one": _nc_broadcast_row,
}


# ------------------------------------------------------------------ G9 门禁

def gate(verbose=True):
    """跑 N1..N5。返回未过条数。★覆盖缺口单独列, 不混进"通过"。"""
    bad, skipped = 0, []
    if verbose:
        print(f"=== G9 必要性闸: {len(PRIMS)} 个原语在库, {len(PENDING)} 个待定")
    for p in PRIMS:
        fail = []
        if len(set(p.witnesses)) < 2:
            fail.append(f"N1 证人只有 {len(set(p.witnesses))} 个: {p.witnesses}")
        else:
            unverified, nocase = _check_witnesses(p)
            if unverified:
                fail.append(f"N1 这些证人在源码里核实不到 {p.name}: {unverified}"
                            f" (证人表是手写断言, 断言会错 —— 见 Prim.witness_pat)")
            if nocase:
                skipped.append(f"{p.name}: 证人 {nocase} 本机没有该 case, **没查成**")
        if p.ref is None:
            fail.append("N2/N3 无参考语义")
        if not p.shape_rule or not p.dtype_rule:
            fail.append("N3 形状或 dtype 规则缺失")
        for cid in p.requires:
            from .caps import CAPS
            if cid not in {c.id for c in CAPS}:
                fail.append(f"N4 声明了不存在的能力 {cid}")
        for label, fn, want in (("正例", p.pos, True), ("负例", p.neg, True)):
            if fn is None:
                fail.append(f"N5 缺{label}")
                continue
            try:
                raw = fn()
            except Exception as e:                              # noqa: BLE001
                fail.append(f"N5 {label}抛异常 {type(e).__name__}: {e}")
                continue
            #  ★不能用 `is not True` 做身份比较: numpy 的归约返回 numpy.bool_,
            #    它 == True 但 `is True` 为假。门禁自己因此把一条过了的负例判成没过
            #    (2026-09-22 建库当天)。判据被自己的实现细节骗到, 与判据写错等价。
            if not isinstance(raw, (bool, np.bool_)):
                fail.append(f"N5 {label}返回的不是布尔 (得 {type(raw).__name__})")
                continue
            if bool(raw) != want:
                fail.append(f"N5 {label}未过 (得 {bool(raw)})")
        if fail:
            bad += 1
            print(f"  ✗ {p.name}")
            for f in fail:
                print(f"      {f}")
        elif verbose:
            print(f"  ✓ {p.name:<15} 证人 {', '.join(sorted(set(p.witnesses)))}")
    if verbose and PENDING:
        print("\n  待定 (证人不足 2, 不进库 —— 留着是为了不必下次再讨论一遍):")
        for name, what, wit, note in PENDING:
            print(f"    {name:<12} {what}")
            print(f"                 现有证人 {wit}; {note}")
    if skipped:
        print(f"\n  ★覆盖缺口 {len(skipped)} 处 (**没查成**, 既不算过也不算不过):")
        for x in skipped:
            print(f"      {x}")
    if verbose:
        print(f"\nG9: {len(PRIMS)} 个原语, {bad} 个未过, {len(skipped)} 处没查成")
        print("★算子级参考一律用基准自带的 tasks/level4/<op>/golden.py, 本库**不自写** —— "
              "再写一份就是又一处会发散的第二实现。")
    return bad
