#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opgen.graph —— 数学语义计算定义图 G_math 的节点与规范形 (PLAN 的 M1)。

范围 (刻意小)
-------------
**只做数学**: 一个节点回答"这个量由谁、经哪个原语、按什么属性算出来"。
位置 (loc/unit)、布局 (layout)、时刻 (t) **不在这里** —— 它们属于 G_move 与偏序,
是 M2/M3 的事。现在就加上这些字段等于加了没人检查的装饰, 而 PLAN 的 C6 禁止这个。

PLAN 4.1 的 Val 是 (math, layout, loc, unit, t, ...)。本文件实现的是其中的 `math`:
同一个 MathNode 可以有多份处在不同位置的实例, 那才是 Val, 由 M2 建立。
这条分离是整个模型的骨架: 数学错 (MathNode 不对) 与布局错 (math 对而 loc/layout 错)
是两类完全不同的故障, 混在一个对象里就分不开。

规范形 (对应宪法 C6 与门禁 G4)
------------------------------
节点 id = 规范形的哈希, **不是自增计数** —— 自增计数会让两次等价构造不相等,
于是同一个调度的两种写法变成两份数据, 那就是熵增。
    canon(canon(x)) == canon(x)   由 G4 机器检查
    等价构造 id 相同               由 G4 机器检查
"""
from __future__ import annotations

import hashlib

from .._refs import Ref
from .semantics import PRIM_BY_NAME

REFS = {
    "L1_unknown_prim": Ref(
        what="节点的算子必须是语义库里的原语 (进库要过 G9 必要性闸)",
        status="derived",
        evidence=["语义库 opgen.semantics 的 PRIMS; 不在表里的算子没有参考语义, "
                  "功能仿真无法执行, 代价也无从算起。"],
        verified="2026-09-22"),
    "L1_arity": Ref(
        what="非输入节点必须有父代",
        status="derived",
        evidence=["无父代的非输入节点是悬空的; 它在 G_math 上不可达, 却会进容量与流量账。"],
        verified="2026-09-22"),
    "L1_shape_required": Ref(
        what="每个节点必须有形状 —— 容量与流量判据全依赖它",
        status="measured",
        evidence=["SFA 2026-09-22: 只融 QK-softmax 的方案靠形状算出 UB 占 412.9% 才被否掉; "
                  "缺形状则这条判据整个失效, 方案会一路走到构建。"],
        verified="2026-09-22"),
    "G4_canon_idempotent": Ref(
        what="规范形幂等且等价构造 id 相同: canon(canon(x)) == canon(x)",
        status="derived",
        evidence=["id 取规范形的哈希而非自增计数 —— 自增计数会让同一调度的两种写法"
                  "变成两份数据 (熵增, 宪法 C6)。属性按键排序折叠, 故书写顺序无关。",
                  "自验: 两次等价构造 id 相同; 属性换序 id 相同; 见 gate 的 G4。"],
        verified="2026-09-22"),
}

INPUT = "input"          # 叶子: 算子的外部输入


class Finding:
    """判据命中的统一格式 (PLAN 第 9 节)。

    `crit` 必须在某个模块的 REFS 里有条目 —— 由门禁 G2 检查。不可证伪的判据不是判据。
    """
    __slots__ = ("layer", "crit", "level", "obj", "msg", "evidence")

    LEVELS = ("错误", "需确认")

    def __init__(self, layer, crit, level, obj, msg, evidence=""):
        assert layer in ("L1", "L2", "L3", "L4", "L5", "L6"), layer
        assert level in Finding.LEVELS, level
        self.layer, self.crit, self.level = layer, crit, level
        self.obj, self.msg, self.evidence = obj, msg, evidence

    def __str__(self):
        return f"[{self.layer} {self.crit}] {self.level}  {self.obj}: {self.msg}"


def _freeze(v):
    """把属性值折成可哈希且**与书写顺序无关**的形式。"""
    if isinstance(v, dict):
        return ("d",) + tuple(sorted((str(k), _freeze(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return ("s",) + tuple(_freeze(x) for x in v)
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return ("r", repr(v))


class MathNode:
    """G_math 的一个节点。

    op     : 原语名 (opgen.semantics 的 PRIMS) 或 INPUT
    args   : 父代节点 (tuple[MathNode])
    attrs  : 该原语的属性 (dict, 例如 {"kind": "exp"} / {"transb": True})
    shape  : 逻辑形状 (tuple[int])
    dtype  : 元素类型名 (str)
    name   : 显示标签，不进规范形。
    symbol : INPUT 的数学变量身份，进入规范形；默认取初始 name 兼容旧调用。
    """
    __slots__ = ("op", "args", "attrs", "shape", "dtype", "name", "symbol", "_canon", "_id")

    def __init__(self, op, args=(), attrs=None, shape=(), dtype="f32", name="", *, symbol=None):
        self.op = op
        self.args = tuple(args)
        self.attrs = dict(attrs or {})
        self.shape = tuple(int(x) for x in shape)
        self.dtype = str(dtype)
        self.name = name
        # Input symbols are mathematical identities; display names are not.
        self.symbol = (name if symbol is None else symbol) if op == INPUT else None
        self._canon = None
        self._id = None

    # ---- 规范形 ----
    def canon(self):
        if self._canon is None:
            if self.op == INPUT and (not isinstance(self.symbol, str) or not self.symbol):
                raise ValueError("Input requires a nonempty mathematical symbol")
            self._canon = (self.op,
                           tuple(a.id() for a in self.args),
                           _freeze(self.attrs),
                           self.shape,
                           self.dtype)
            if self.op == INPUT:
                self._canon += (("symbol", self.symbol),)
        return self._canon

    def id(self):
        if self._id is None:
            self._id = hashlib.sha256(repr(self.canon()).encode()).hexdigest()[:16]
        return self._id

    def why(self):
        if self.op == INPUT:
            return f"外部输入 {self.name or self.id()} {self.shape} {self.dtype}"
        src = ", ".join(a.name or a.id()[:8] for a in self.args)
        at = "".join(f" {k}={v}" for k, v in sorted(self.attrs.items()))
        return f"{self.name or self.id()[:8]} = {self.op}({src}){at} -> {self.shape} {self.dtype}"

    # ---- 自检 (L1 语义层) ----
    def check(self, ctx=None):
        out = []
        if self.op == INPUT:
            if not self.shape:
                out.append(Finding("L1", "L1_shape_required", "错误", self.id(),
                                   "外部输入没有形状", self.why()))
            return out
        p = PRIM_BY_NAME.get(self.op)
        if p is None:
            out.append(Finding("L1", "L1_unknown_prim", "错误", self.id(),
                               f"不是语义库里的原语: {self.op}",
                               "进库要过 G9 必要性闸 (N1 证人>=2 等), 见 opgen.semantics"))
            return out
        if not self.args:
            out.append(Finding("L1", "L1_arity", "错误", self.id(),
                               f"{self.op} 没有父代", self.why()))
        if not self.shape:
            out.append(Finding("L1", "L1_shape_required", "错误", self.id(),
                               "缺形状 —— 容量与流量判据全依赖它", self.why()))
        return out


#  ---- 负例控制 (门禁 G1): 判据 id -> 一个**已知违规**的最小输入, 断言它必须命中 ----
def _nc_unknown_prim():
    q = MathNode(INPUT, (), None, (4, 4), "f32", "q")
    return any(f.crit == "L1_unknown_prim"
               for f in MathNode("nosuch", (q,), {}, (4, 4)).check())


def _nc_arity():
    return any(f.crit == "L1_arity" for f in MathNode("matmul", (), {}, (4, 4)).check())


def _nc_shape():
    q = MathNode(INPUT, (), None, (4, 4), "f32", "q")
    return any(f.crit == "L1_shape_required"
               for f in MathNode("elementwise", (q,), {"kind": "exp"}, ()).check())


def _nc_canon():
    """规范形幂等的负例: 若 id 改用自增计数, 等价构造就会不相等。这里直接构造两个
    等价节点并断言它们相等 —— 反过来, 若实现退化成按对象身份取 id, 本条立刻红。"""
    q = MathNode(INPUT, (), None, (4, 4), "f32", "q")
    a = MathNode("elementwise", (q,), {"kind": "exp"}, (4, 4))
    b = MathNode("elementwise", (q,), {"kind": "exp"}, (4, 4))
    different_attr = MathNode("elementwise", (q,), {"kind": "abs"}, (4, 4))
    return a.id() == b.id() and a.id() != different_attr.id()


NEGATIVE_CONTROLS = {
    "L1_unknown_prim": _nc_unknown_prim,
    "L1_arity": _nc_arity,
    "L1_shape_required": _nc_shape,
    "G4_canon_idempotent": _nc_canon,
}


def leaf(name, shape, dtype="f32", *, symbol=None):
    return MathNode(INPUT, (), None, shape, dtype, name, symbol=symbol)


class Program:
    """一段 G_math。SSA: 每个节点只被定义一次, 由 id 去重。"""

    def __init__(self, name=""):
        self.name = name
        self._by_id = {}
        self._order = []

    def add(self, node: MathNode) -> MathNode:
        """加入并返回**规范形上唯一**的那个节点 (等价构造会折叠成同一个)。"""
        for a in node.args:
            self.add(a)
        k = node.id()
        if k in self._by_id:
            return self._by_id[k]
        self._by_id[k] = node
        self._order.append(node)
        return node

    def nodes(self):
        return list(self._order)

    def canon(self):
        """整段的规范形: 按 id 排序的节点规范形序列。与书写顺序无关。"""
        return tuple(sorted(n.canon() for n in self._order))

    def id(self):
        return hashlib.sha256(repr(self.canon()).encode()).hexdigest()[:16]

    def check(self, ctx=None):
        out = []
        for n in self._order:
            out += n.check(ctx)
        return out

    def why(self):
        return "\n".join(n.why() for n in self._order)
