#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_benchpath — cann-bench 安装位置的唯一取数口 (下划线前缀: 共享设施, 不进 opdbg 模块清单).

动机 (2026-10-20 审计实撞): cann-bench 根目录以裸默认值硬编码在 6 处
(envelope / precloc / opfuzz / opgen.simfunc / benchrun / opeval), 且写的是两台
不同机器的历史路径, 本机一个都不存在 —— 失效时报 FileNotFoundError /
ModuleNotFoundError, 而不是"该设什么"。与 chip_calib.json 同一条规则:
机器相关的量走配置口, 不写死在代码里。

解析次序: $CANN_BENCH_ROOT → 历史安装位置逐个探测 → required 时带设置说明报错,
否则返回 None (调用方自己决定没有 bench 时降级成什么)。
"""
from __future__ import annotations

import os

#  历史机器上的安装位置 (存在才会被用上; 新机器一律设 CANN_BENCH_ROOT)
KNOWN_ROOTS = (
    "/root/users/db00946854/cann-bench-950/_unpack/cann-bench/cann-bench-master",
    "/root/cann-bench-950/_unpack/cann-bench/cann-bench-master",
)

_HOW = ("未找到 cann-bench: 设 CANN_BENCH_ROOT=<cann-bench-master 目录> "
        "(即含 src/ 与 tasks/ 的那一层)")


def bench_root(required: bool = False):
    """cann-bench-master 根目录; 没有时 required 决定报错还是返回 None."""
    env = os.environ.get("CANN_BENCH_ROOT")
    if env:
        if required and not os.path.isdir(env):
            raise SystemExit(f"CANN_BENCH_ROOT={env} 不存在或不是目录")
        return env
    for p in KNOWN_ROOTS:
        if os.path.isdir(p):
            return p
    if required:
        raise SystemExit(_HOW)
    return None


def _sub(rel: str, required: bool):
    r = bench_root(required)
    return os.path.join(r, rel) if r else None


def bench_src(required: bool = False):
    """kernel_eval 所在的 src/ (供 sys.path 注入)."""
    return _sub("src", required)


def bench_tasks(required: bool = False):
    return _sub("tasks", required)
