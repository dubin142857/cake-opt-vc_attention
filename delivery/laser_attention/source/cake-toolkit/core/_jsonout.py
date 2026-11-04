#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_jsonout — 诊断模块机读输出的统一契约 (下划线前缀: 共享设施, 不进 opdbg 清单).

动机 (2026-10-20 完备度审计 P7): 上层编排 (candrun 的 lint 步) 此前靠刮取下游
prose 的"汇总:"行 —— 措辞一变上层就静默退化成"(无汇总行)", 且判定随 rc 一起被
丢弃 (index 'verdict_discarded' 族)。机读口径一经声明只增字段不改语义,
上层按 schema 读, 不再与措辞耦合。

schema v1 (写进文件, 不混 stdout —— prose 输出保持原样, 两不相扰):
    {
      "module":  str,            # 产出方
      "schema":  1,
      "summary": {"total": int, "per_check": {check: n}, "hard_fail": bool},
      "hits":    [ {check, ...该判据的原样字段} ]   # 字段集合不统一, 读方一律 .get
    }
"""
from __future__ import annotations

import json

SCHEMA = 1


def dump(path: str, module: str, hits: list, hard_fail: bool, extra: dict | None = None) -> None:
    """把命中列表按契约落盘。hits 的字段原样保留 (各判据字段集不统一, 读方 .get)。"""
    per = {}
    for h in hits:
        c = str(h.get("check", "?"))
        per[c] = per.get(c, 0) + 1
    doc = {"module": module, "schema": SCHEMA,
           "summary": {"total": len(hits), "per_check": per, "hard_fail": bool(hard_fail)},
           "hits": hits}
    if extra:
        doc.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1, default=str)


def load(path: str) -> dict | None:
    """读一份契约文件; 读不到/不合 schema 返回 None, 由读方自己决定兜底 (不猜)。"""
    try:
        d = json.load(open(path, encoding="utf-8"))
        if isinstance(d, dict) and "summary" in d and "hits" in d:
            return d
    except Exception:
        pass
    return None
