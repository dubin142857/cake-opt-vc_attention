#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opgen.qty —— 带**来源**的量。让"返回一个没有来源的数"在类型上不可能。

为什么 (2026-09-22 立, 由本模块自己的两次静默缺陷逼出来)
--------------------------------------------------------
  (1) ir.lower_bounds 直接除以核数, **隐含假设完美均衡**, 没有任何波量化项 ——
      输出是一个看起来正常的数, 没有任何地方提示"这里做了一个乐观假设"。
  (2) semantics 的证人核实在本机没有该 case 时静默跳过; simfunc 取不到基准 golden 时
      按"通过"计数。两处都让**没查成**在计数上等同于**查过且通过**。

共同形状: **回落时不出信号**。对策不是"以后记得写注释", 是让每个推导量必须带来源,
而报告器对非 measured/derived 的来源必须显式打印。

来源取值 (与 core/_refs 的 status 对齐, 但这是**单次取值**的来源, 不是判据的状态):
    measured     本机实测 / 平台 ini 直读
    derived      由 measured 推导, 推导过程写在 note 里
    default      用了缺省值 —— **必须打印**, 并说明缺什么才能变成 measured
    unknown      暂时不知道 —— **必须打印**, 并说明要知道它需要做什么
    unavailable  这次拿不到 (环境缺件) —— **必须打印**, 说明缺什么
`default / unknown / unavailable` 三者合称**回落**; 回落的量不得参与"通过"的计数。
"""
from __future__ import annotations

MEASURED, DERIVED, DEFAULT, UNKNOWN, UNAVAILABLE = (
    "measured", "derived", "default", "unknown", "unavailable")
FALLBACKS = (DEFAULT, UNKNOWN, UNAVAILABLE)
_ALL = (MEASURED, DERIVED) + FALLBACKS


class Qty:
    """一个带来源的量。value 可以是 None (表示确实没有值)。"""
    __slots__ = ("value", "src", "unit", "note")

    def __init__(self, value, src, unit="", note=""):
        if src not in _ALL:
            raise ValueError(f"来源必须是 {_ALL} 之一, 收到 {src!r}")
        if src in FALLBACKS and not note:
            #  ★回落必须说清楚"缺什么"。没有 note 的回落就是静默回落,
            #    正是本模块要消灭的东西 —— 所以这里直接拒绝构造。
            raise ValueError(f"来源 {src} 必须给 note: 说明缺什么、怎样才能变成 measured")
        self.value, self.src, self.unit, self.note = value, src, unit, note

    @property
    def fallback(self):
        return self.src in FALLBACKS

    def __repr__(self):
        v = "无" if self.value is None else (f"{self.value:.4g}"
                                            if isinstance(self.value, float) else self.value)
        return f"Qty({v}{self.unit} {self.src})"

    def line(self, label, width=12):
        """一行报告。回落一律带 ★ 与 note, 不允许只印一个数。"""
        #  ★小数值要给足有效位: 波量化 0.9884 被印成 1.0 就等于把信号磨掉了,
        #    而"把信号磨掉"与"不给信号"在后果上一样 (2026-09-22 实撞)。
        if self.value is None:
            v = "—"
        elif isinstance(self.value, float):
            v = f"{self.value:.4f}" if abs(self.value) < 10 else f"{self.value:,.1f}"
        else:
            v = f"{self.value}"
        head = f"  {label:<{width}}{v:>12}{(' ' + self.unit) if self.unit else ''}"
        if self.fallback:
            return f"{head}   ★{self.src}: {self.note}"
        return f"{head}" + (f"   ({self.note})" if self.note else "")


def measured(v, unit="", note=""):
    return Qty(v, MEASURED, unit, note)


def derived(v, unit="", note=""):
    return Qty(v, DERIVED, unit, note)


def default(v, note, unit=""):
    return Qty(v, DEFAULT, unit, note)


def unknown(note, unit=""):
    return Qty(None, UNKNOWN, unit, note)


def unavailable(note, unit=""):
    return Qty(None, UNAVAILABLE, unit, note)


def report_all(d, order=None, width=12):
    """打印一组 Qty。返回回落的个数 —— 调用方必须把它计进"没查成"而不是"通过"。"""
    n, notqty = 0, []
    for k in (order or sorted(d)):
        q = d.get(k)
        if not isinstance(q, Qty):
            #  ★不是 Qty 的条目原来既不打印也不计数, 于是它从这份报告里整条消失,
            #    而返回值 (调用方据以判"没查成"几项) 也不含它 —— 一个装配错误
            #    被呈现成"这一项不存在" (2026-10-26, silentskip 扫出, 09 第 177 条)。
            notqty.append(k)
            continue
        print(q.line(k, width))
        n += 1 if q.fallback else 0
    if notqty:
        print(f"  ★{len(notqty)} 项**不是 Qty, 没打印也没查**: {', '.join(map(str, notqty[:8]))}"
              f" —— 按'没查成'计, 不是'不适用'")
        n += len(notqty)
    return n
