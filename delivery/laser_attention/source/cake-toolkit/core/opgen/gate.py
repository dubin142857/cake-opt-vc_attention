#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opgen.gate —— 本模块自己的准入闸 (PLAN 第 10 节)。

一条命令跑完, 任一条不过即非零退出。改动 core/opgen/** 之前先过它。

现已实现: G5 跨芯片隔离 / G6 实测自校验 / G9 原语必要性 / G10 评测器隔离。
未实现的 G1-G4、G7、G8 在输出里**显式列出**, 不假装全绿 —— 一个不报告自己缺口的
门禁, 和没有门禁的区别只在心理上。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

_TODO = {}

#  ---- G8 的清单: 本模块该有哪些文件, 各自一句话职责 ----
#  没有这份清单就判不了"有没有新增平行文件", 于是 G8 一直挂着。清单本身就是
#  "一个文件一个概念"的约束: 加文件必须先在这里写下它**不能**塞进哪个已有文件。
FILE_MANIFEST = {
    "__init__.py": "包说明与导出",
    "__main__.py": "命令行入口 (caps / gate / selfcheck / refs)",
    "PLAN.md": "建设计划书: 宪法 / 数据模型 / 门禁 / 里程碑",
    "qty.py": "带来源的量: 让'返回一个没有来源的数'在类型上不可能",
    "caps.py": "芯片机制能力的可执行探测表",
    "graph.py": "G_math 的节点与规范形, 以及 Finding",
    "semantics.py": "原语语义库与 G9 必要性闸",
    "ir.py": "调度 IR 与四类判据 (容量/流量/同步边/下界)",
    "architecture.py": "P1不可变具体架构对象、严格JSON、区域/事件依赖及结构校验；旧ir只汇总阶段，不具备这些身份与依赖关系。",
    "architecture_test.py": "P1对象与数学输入身份的正反例；不运行设备，不替代旧IR回归。",
    "transforms.py": "P1带前提/证据失效/回执的架构操作；architecture定义静态对象，本模块定义版本之间的变换而非结构字段。",
    "transforms_test.py": "架构操作正反例及独立字节内存回放；验证可组合操作，不替代architecture对象或硬件调度测试。",
    "observation.py": "P1观测、时钟/launch、来源与coverage对象；观测不是静态架构或必要依赖，后接sim适配。",
    "campaign.py": "P1版本演化与开发决策图、联合验收、预算/预测对象；与执行事件图分开，后接P4策略。",
    "legacy.py": "旧IR全字段快照/恢复和显式迁移映射；不修改旧IR算法，也不猜不存在的事件/语义。",
    "research_test.py": "观测/方案/迁移的正反例与回滚行为；独立于架构事件检查和变换功能执行器。",
    "trace.py": "P2显式hash输入读取/Chrome与CANN事件归一/记录记账与独立时长对账；observation只定义对象，不读外部trace或report。",
    "trace_test.py": "trace适配的缺失/乱序/多launch/单位/读取边界/流候选和回执反例；不替代静态观测对象检查。",
    "schedule.py": "P3有限整数调度/资源窗口/连续租约地址证书与实现约束；ir是阶段汇总，trace只描述观测，均不解时间与地址布局。",
    "schedule_test.py": "独立时刻/地址穷举及实际物理偏移功能回放；覆盖预算、区间和证书误报，不使用设备。",
    "loops.py": "P3带迭代域/仿射视图/slot与边界seed的有限展开、接口初始化和输出覆盖；schedule只处理已具体化DAG，不猜循环历史。",
    "loops_test.py": "全局数组数学与真实slot池回放，覆盖首尾/尾部/历史seed/未初始化副本及陈旧覆盖；不替代调度/地址oracle。",
    "analysis.py": "P3显式观测/架构映射、覆盖/成本/依赖/资源/地址重放和假设场景增益区间；trace不评模型，schedule不绑定观测。",
    "analysis_test.py": "独立闭式流水时刻/地址与错配/遗漏/等待污染/容量和增益区间反例；不使用待验硬件假设制造数据。",
    "policy.py": "P4重放证据与完整场景驱动的组合分支/预算/里程碑建议及状态失效；campaign保留声明结构校验，analysis不作开发决策。",
    "policy_test.py": "中间退化/组合收益、错误乐观模型、重复冲突样本、预算与重开依赖失效反例。",
    "runner.py": "P5冻结argv/输入/工具身份与隔离子进程的阶段链/checkpoint；policy不执行，旧opeval是shell流程且不承接该对象证据。",
    "runner_test.py": "真实CPU namespace/故障/缓存身份/硬退出恢复/nonce/取消和外部witness反例；不冒充NPU守卫或性能。",
    "delivery.py": "P5真实RunReceipt指标/模型绑定驱动best保护与fresh原实现恢复；policy研究标签不代替交付证据。",
    "delivery_test.py": "真实CPU namespace晋升/排序反转/更快incumbent/故障后原artifact恢复与证据篡改反例。",
    "lowering.py": "P5真实变换/实施约束到固定native库与采样ABI存储实例化；不把源码标签当数学证明。",
    "lowering_test.py": "父链/布局/缺项/错卡/arena/真实别名/源码库与执行plan绑定负例，无NPU操作。",
    "ascend.py": "P5 Ascend物理节点/父域占用/合作lease和原三守卫适配；runner保留平台无关执行协议，不从隔离/proc假推空闲。",
    "ascend_cli.py": "独立CLI入口避免记录类型预导入后runpy二次执行；不替代ascend的验证/守卫逻辑。",
    "ascend_sandbox.py": "本机driver兼容策略的固定AArch64 seccomp：子userns保留cap数值但拒绝mount/namespace/设备创建及句柄绕过。",
    "ascend_sandbox_test.py": "无设备真实namespace验证限制与正常线程/进程；固定BPF决策/错误arch/缺失及篡改反例。",
    "ascend_test.py": "完整设备表/映射/未知/guard审计与旁路负例；单元测试不占NPU，实际适配另上机验收。",
    "simfunc.py": "功能层仿真: 调度重写有没有改变数学语义",
    "selfcheck.py": "用已实测算子反验判据本身",
    "gate.py": "本模块自己的准入闸",
    "equiv.py": "数学等价形式的枚举与排序 (改写轴 + 前提剪枝 + IR 判据排名)。"
                "不能塞进 ir.py: ir 评价**一个**被提出的调度, equiv 生成候选; "
                "也不能塞进 graph.py: graph 只求规范形与 id, 不含改写轴与降级。",
    "api.py": "AscendC **API 用法解释器**: 一条 API 的五层取证 (A1 存在/档位 -> A2 签名与"
              "操作数位置 -> A3 硬约束 -> A4 流水 -> A5 指令与编译器), 不确定就给可执行实验。"
              "不能塞进 caps.py: caps 的粒度是**机制** (有没有 regbase), 一颗芯片一张 9 条的表; "
              "本文件的粒度是**一条 API**, 且取证口多一个 —— 编译器自带的内建头 "
              "(vector_f32 = 64 x f32 这类事实在 CANN 的 asc/ 树里根本没有)。"
              "也不能塞进 semantics.py: 那里是硬件无关的数学原语参考实现, 不含任何 API 名。",
    "equiv_refuted.json": "被实测否掉的等价形式记录。G12 只认这张表里的条目, "
                          "prose 不算 —— 单独成文件是因为它是**数据**不是代码, "
                          "要能被 git blame 逐条追到是哪次测量否的。",
}


def g8_manifest(verbose=True):
    """目录里的文件必须与清单一一对应。多出来的要先说明它不能塞进哪个已有文件。"""
    import os as _os
    d = _os.path.dirname(_os.path.abspath(__file__))
    have = {n for n in _os.listdir(d)
            if _os.path.isfile(_os.path.join(d, n)) and not n.startswith((".", "__pycache__"))}
    want = set(FILE_MANIFEST)
    bad = 0
    for extra in sorted(have - want):
        bad += 1
        print(f"  ✗ G8 多出文件 {extra} —— 先在 FILE_MANIFEST 写清它为什么不能塞进已有文件")
    for miss in sorted(want - have):
        bad += 1
        print(f"  ✗ G8 清单里有而目录里没有: {miss}")
    if verbose and not bad:
        print(f"  ✓ G8 {len(want)} 个文件与清单一一对应")
    return bad


#  ---- G11 无静默回落 ----
_RISKY = (
    #  ★2026-09-22 采纳外部评审: 原来只写了 `or 1` / `or 1.0`, 于是 ir.py 里的
    #    `sp.get("vec_calc_B_per_cycle") or 128` 从来没被列出来 —— 抓不住的模式等于
    #    没有模式。改成任意数字字面量。
    (r"\bor\s+-?\d+(?:\.\d+)?\b", "用 `or <数字>` 兜底"),
    (r"except[^\n]*:\s*\n\s*pass", "空 except 吞掉异常"),
    (r"\.get\([^)]*,\s*(?:0|1|1\.0|True|False)\)", "`.get(k, 缺省值)` 静默取缺省"),
)


def g11_no_silent_fallback(verbose=True):
    """回落必须出信号。

    两段: (a) 结构约束 —— lower_bounds 的每个输出必须是带来源的 Qty, 回落必须带 note
    (由 Qty 构造器强制, 这里断言它确实生效); (b) 源码扫描 —— 列出"看起来像静默兜底"
    的写法作为**待确认清单**, 不下判定 (启发式判据不该当错误报)。
    """
    import os as _os
    import re as _re
    from .ir import Schedule, VEC
    from .qty import Qty
    from .. import socspec
    bad = 0
    sp = socspec.spec("ascend950pr_957b")
    s = Schedule("g11")
    s.stage("v", VEC, elem_ops=1000, instr=1000)          # 刻意不给 tasks
    lb = s.lower_bounds(sp)
    for k, q in lb.items():
        if not isinstance(q, Qty):
            bad += 1
            print(f"  ✗ G11 lower_bounds 的 {k} 不是带来源的 Qty (裸数字 = 没有信号)")
        elif q.fallback and not q.note:
            bad += 1
            print(f"  ✗ G11 {k} 是回落却没有 note")
    if any(isinstance(q, Qty) and q.fallback for q in lb.values()):
        if verbose:
            print("  ✓ G11 不给 tasks 时波量化如实报为 default 并说明缺什么")
    else:
        bad += 1
        print("  ✗ G11 不给 tasks 竟然没有产生回落 —— 说明某处默认了完美均衡")
    d = _os.path.dirname(_os.path.abspath(__file__))
    hits = []
    for n in sorted(FILE_MANIFEST):
        if not n.endswith(".py"):
            continue
        txt = open(_os.path.join(d, n), errors="replace").read()
        for pat, why in _RISKY:
            for m in _re.finditer(pat, txt):
                ln = txt.count("\n", 0, m.start()) + 1
                hits.append(f"{n}:{ln}  {why}")
    if hits and verbose:
        print(f"  · G11 疑似静默兜底 {len(hits)} 处 (**待确认清单**, 不是判定 —— "
              f"累加器初值 `d.get(k, 0)`、以及已由 Qty 报过回落之后的 `or 1.0` 都会命中, "
              f"那些是正常的; 要看的是**读不到就当没有**那一类):")
        for h in hits[:12]:
            print(f"        {h}")
        if len(hits) > 12:
            print(f"        ... 另有 {len(hits) - 12} 处")
    return bad


def _all_refs():
    #  ★清单是写死的, 所以**新增文件必须同时加到这里与 g1 的那份** —— 否则它的 REFS
    #    与负例既不进 G2 也不进 G1, 于是"判据存在"与"判据被看守"是两件事。
    #    2026-09-29 api.py 进来时实撞: G8 报绿 (文件登记了), 而 G1/G2 根本没看见它。
    from . import caps, ir, semantics, graph, api, architecture, transforms, observation, campaign, legacy, trace, schedule, loops, analysis, policy, runner, ascend
    out = {}
    for m in (caps, ir, semantics, graph, api, architecture, transforms, observation, campaign, legacy, trace, schedule, loops, analysis, policy, runner, ascend):
        out.update(getattr(m, "REFS", {}))
    try:
        from .. import mixaddr
        out.update(getattr(mixaddr, "REFS", {}))
    except Exception:                                              # noqa: BLE001
        pass
    return out


def _sample_programs():
    """G4 用的样本: 同一段 G_math 的两种等价写法 + 属性换序。"""
    from .graph import leaf, MathNode, Program
    q, k = leaf("q", (64, 128)), leaf("k", (8192, 128))
    def build(swap_attr_order, extra_first):
        p = Program("sfa_qk_softmax")
        at = {"transb": True, "transa": False}
        if swap_attr_order:
            at = {"transa": False, "transb": True}
        s = MathNode("matmul", (q, k), at, (64, 8192), "f32", name="s")
        e = MathNode("elementwise", (s,), {"kind": "exp"}, (64, 8192), "f32", name="e")
        r = MathNode("reduce", (e,), {"kind": "sum", "axis": -1}, (64,), "f32", name="l")
        #  ★刻意改变**加入顺序**: 规范形必须与书写顺序无关
        if extra_first:
            p.add(r); p.add(e); p.add(s)
        else:
            p.add(s); p.add(e); p.add(r)
        return p
    return [build(False, False), build(True, True)]


def g1_negative_controls(verbose=True):
    """每条判据都要有一个**已知违规**的最小输入, 断言它必须命中。

    抓不住错的判据只是断言 (宪法 C4)。没有声明 NEGATIVE_CONTROLS 的模块**如实报为
    未覆盖**, 不算通过 —— 报告自己的缺口是这套门禁的基本要求。
    """
    from . import caps, ir, semantics, graph, simfunc, equiv, api, architecture, transforms, observation, campaign, legacy, trace, schedule, loops, analysis, policy, runner, ascend
    from .equiv import NotApplicable as _NA
    bad, uncovered, na = 0, [], []
    for m in (caps, ir, semantics, graph, simfunc, equiv, api, architecture, transforms, observation, campaign, legacy, trace, schedule, loops, analysis, policy, runner, ascend):
        name = m.__name__.rsplit(".", 1)[-1]
        refs = getattr(m, "REFS", {})
        ncs = getattr(m, "NEGATIVE_CONTROLS", {})
        for cid in sorted(refs):
            fn = ncs.get(cid)
            if fn is None:
                uncovered.append(f"{name}.{cid}")
                continue
            try:
                ok = bool(fn())
            except _NA as e:
                #  ★三态: 不适用**既不算过也不算不过** (2026-09-22 按 dubin 的架构定调加)。
                #    本项目主场景是已有算子的单 case 极致优化; 打榜/benchmark 接入是增强
                #    模块, 不许绑死 —— 没有 benchmark 数据的环境里 core 门禁必须能全绿。
                #    计成未过会逼人伪造数据凑绿; 计成通过就是"没查成伪装成查过且通过"。
                na.append(f"{name}.{cid}: {e}")
                continue
            except Exception as e:                                 # noqa: BLE001
                ok, e_ = False, e
                print(f"  ✗ G1 {name}.{cid} 的负例抛异常 {type(e_).__name__}: {e_}")
            if not ok:
                bad += 1
                print(f"  ✗ G1 {name}.{cid} 的负例**没抓住** —— 这条判据只是断言")
            elif verbose:
                print(f"  ✓ G1 {name}.{cid}")
    if na:
        print(f"  · G1 **不适用**的判据 {len(na)} 条 (缺 benchmark 数据等; 既不算过也不算不过):")
        for c in na:
            print(f"        {c}")
    if uncovered:
        print(f"  · G1 还没有负例的判据 {len(uncovered)} 条 (如实列出, 不算通过也不算失败):")
        for c in uncovered:
            print(f"        {c}")
    return bad


def g2_refs_complete(verbose=True):
    """每个 Finding 的判据 id 必须在某个模块的 REFS 里有条目。"""
    from .graph import MathNode, Program, leaf
    refs = _all_refs()
    bad = 0
    #  构造一段**故意有问题**的程序, 收集它报出的所有判据 id
    q = leaf("q", (4, 4))
    p = Program("bad")
    p.add(MathNode("nosuch", (q,), {}, (4, 4)))
    p.add(MathNode("matmul", (), {}, (4, 4)))
    p.add(MathNode("elementwise", (q,), {"kind": "exp"}, ()))
    crits = {f.crit for f in p.check()}
    if not crits:
        bad += 1
        print("  ✗ G2 故意构造的坏程序没报出任何判据 —— 判据本身失效了")
    for c in sorted(crits):
        if c not in refs:
            bad += 1
            print(f"  ✗ G2 判据 {c} 在任何 REFS 里都没有条目 (不可证伪)")
        elif verbose:
            print(f"  ✓ G2 {c:<24} 有出处 ({refs[c].status})")
    return bad


def g3_interface(verbose=True):
    """节点类必须有 __slots__ / canon / id / why / check。"""
    from .graph import MathNode, Program
    bad = 0
    for cls in (MathNode, Program):
        for attr in ("canon", "id", "why", "check"):
            if not callable(getattr(cls, attr, None)):
                bad += 1
                print(f"  ✗ G3 {cls.__name__} 缺 {attr}()")
        if cls is MathNode and not getattr(cls, "__slots__", None):
            bad += 1
            print(f"  ✗ G3 {cls.__name__} 没有 __slots__ (随手挂属性 = 熵增)")
    if verbose and not bad:
        print("  ✓ G3 MathNode / Program 接口齐 (__slots__ / canon / id / why / check)")
    return bad


def g4_canon(verbose=True):
    """规范形幂等 + 等价构造 id 相同 + 与书写顺序无关。"""
    bad = 0
    a, b = _sample_programs()
    if a.canon() != b.canon():
        bad += 1
        print("  ✗ G4 两种等价写法的规范形不同 (属性换序 / 加入顺序不同)")
    if a.id() != b.id():
        bad += 1
        print("  ✗ G4 两种等价写法的 id 不同")
    for n in a.nodes():
        c1 = n.canon()
        if n.canon() != c1:
            bad += 1
            print(f"  ✗ G4 节点 {n.id()} 的 canon 不幂等")
    if verbose and not bad:
        print(f"  ✓ G4 规范形幂等, 等价构造同 id ({a.id()}), 与属性/加入顺序无关")
    return bad


def g5_cross_chip(verbose=True):
    """A5 独有的三项机制在 A2 / 310P 上必须为 False (不是 None, 不是 True)。"""
    from .caps import probe_all
    want_false = ("cv_direct_ub", "cv_dual_dst_split_m", "regbase_simd_vf")
    bad = 0
    for soc, expect in (("ascend950pr_957b", True), ("ascend910b2", False), ("ascend310p3", False)):
        got = {c.id: v for c, v, _ in probe_all(soc)}
        for cid in want_false:
            v = got.get(cid)
            ok = (v is True) if expect else (v is False)
            if not ok:
                bad += 1
                print(f"  ✗ G5 {soc} 的 {cid} = {v!r}, 期望 {'True' if expect else 'False'}")
            elif verbose:
                print(f"  ✓ G5 {soc:<18} {cid:<22} {v}")
    return bad


def g6_selfcheck(verbose=True):
    """实测自校验: qk / softmax 的 实测/下界 必须落在 [1.0, 1.4]。

    低于 1.0 = 模型算错 (下界比实测还大); 高于 1.4 = 模型漏了主导项。二者都要红。
    ★pv 不在判据里 —— 它实测是下界的 2.87 倍, 那是**模型正确地指出自己的盲区**,
      不是模型退化。给它加拟合项是宪法 C5 明令禁止的。
    """
    from . import selfcheck as sc
    from .. import socspec
    sp = socspec.spec(sc.SOC)
    l2 = sp["cap"]["L2"]
    dram = socspec.calib_value("dram_GBps", sp["soc"])
    s3 = sc.three_stage()
    idx = {m.name: m for m in s3.moves}
    per = {"qk": ("q_read", "k_read_qk", "sc_write"),
           "softmax": ("sc_read", "si_read", "pr_write")}
    bad = 0
    for st, names in per.items():
        dr = sum(idx[n].dram(l2)[0] for n in names)
        lb = dr / (dram * 1e9) * 1e6
        r = sc.MEASURED[st] / lb
        if not (1.0 <= r <= 1.4):
            bad += 1
            print(f"  ✗ G6 {st} 实测/下界 = {r:.2f}, 不在 [1.0, 1.4]")
        elif verbose:
            print(f"  ✓ G6 {st:<18} 实测/下界 {r:.2f}")
    return bad


def g9_prims(verbose=True):
    from .semantics import gate as sem_gate
    return sem_gate(verbose=verbose)


def g7_simfunc(verbose=True):
    """功能层: 调度重写有没有改变数学语义。"""
    from .simfunc import selftest
    return selftest(verbose=verbose)


def g10_eval_isolation(case_dirs=None, verbose=True):
    """评测器隔离: 每份测量记录必须带评测器指纹, 且跨指纹比较要被认出来。

    出处 ForgeMegakernel: 原文是"任何修改了评测器的轮次一律回退"。本仓的双环纪律
    要求 infra 可进化, 所以改成**可追溯**: 评测器可以改, 但改了就必须在每份测量记录
    里留痕, 跨版本比较能被机器认出来。

    ★同一次 git 提交同时改评测器与被测对象这条在本仓判不了: cases/ 被顶层 .gitignore
      排除 (客户算子源码 74 GB), 被测对象根本不进 git。所以判据放在**运行粒度**。
    """
    roots = case_dirs or sorted(glob.glob("cases/*/runs"))
    bad, seen = 0, 0
    miss_cases, miss_runs, ok_cases = 0, 0, 0
    for root in roots:
        runs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
        if not runs:
            continue   # ssok: 空 runs 目录里没有测量记录, 不在 seen 的口径里
        fps, missing, oldfmt, cards = {}, [], [], {}
        for r in runs:
            p = os.path.join(r, "evalfp.txt")
            if not os.path.exists(p):
                missing.append(os.path.basename(r))
                continue
            #  ★只按**影响测量**的部分判可比性: advisory 开头的行 (alignlint/hwfloor 这类
            #    只打印的判据模块) 仅供追溯, 变了不影响两轮能不能比。
            _raw = open(p, errors="replace").read()
            if "# fpver 4" not in _raw:
                #  ★旧格式的记录不能与新格式比 —— 取证口径本身变了。如实报成
                #    "无法判定"而不是"评测器换版了", 否则这条闸会为历史轮次永远红。
                #  fpver 3 (2026-09-23) 加了物理卡号; fpver 4 同日把"评测器"从
                #  "opeval 全文的 sha"换成"实际传给评测器的那组参数" —— 旧轮次的指纹里
                #  混着与测量无关的脚本改动, 无法与新口径对齐, 只能判"无法判定"。
                oldfmt.append(os.path.basename(r))
                continue
            #  ★卡号单独拆出来 (2026-09-23)。fpver 3 起 evalfp 把物理卡号记进可比性段,
            #    于是"同一评测器、不同卡"的两轮指纹也不同, 而本条原样会把它报成
            #    「存在 2 个不同的**评测器版本**」—— 判定(不可比)对, 措辞把人引去查工具版本。
            #    实撞: cases/sfa_bench 的 pad3(卡 4) 与 pad4(卡 5) 其余逐字节相同。
            #    两者性质也不同: 评测器换版是**缺陷**(跨版本比较不成立), 卡号不同是使用者
            #    比较时必须计入的**事实** —— 后者若也判失败, 这条闸会对所有跨卡的 case 永远红。
            _lines = [l for l in _raw.splitlines() if not l.startswith("advisory ")]
            _card = next((l.split(None, 1)[1].strip() for l in _lines
                          if l.startswith("card ")), None)
            _txt = "\n".join(l for l in _lines if not l.startswith("card "))
            fps.setdefault(_txt.strip(), []).append(os.path.basename(r))
            if _card is not None:
                cards.setdefault(_card, []).append(os.path.basename(r))
        seen += 1
        if oldfmt:
            print(f"  · G10 {root}: {len(oldfmt)} 个轮次用的是**旧版指纹格式** "
                  f"({', '.join(oldfmt[:6])}) —— 取证口径已变, 与新轮次**无法判定**可比性 "
                  f"(不回填: 回填的指纹是编的)")
        if missing:
            #  历史轮次没有指纹是既成事实, 报为**待补**而不是失败 —— 否则这条闸从第一天
            #  起就永远红, 而永远红的闸等于没有闸。逐 case 说同一句话没有信息量, 聚合成一行。
            miss_cases += 1
            miss_runs += len(missing)
        if len(fps) > 1:
            bad += 1
            print(f"  ✗ G10 {root}: 存在 {len(fps)} 个不同的评测器版本, 跨版本比较不成立:")
            for i, (_, names) in enumerate(sorted(fps.items(), key=lambda kv: -len(kv[1]))):
                print(f"        版本{i + 1}: {', '.join(names[:6])}"
                      + (f" 等 {len(names)} 个" if len(names) > 6 else ""))
        elif fps:
            ok_cases += 1
        if len(cards) > 1:
            #  不计入 bad: 这不是仓里的缺陷, 是比较时要计入的事实。点名是哪几轮在哪张卡上,
            #  否则使用者仍要自己去翻 evalfp。
            print(f"  · G10 {root}: 这些轮次**跨了 {len(cards)} 张卡**, 性能数字不可直接比 —— "
                  + "; ".join(f"卡 {c}: {', '.join(v[:6])}" for c, v in sorted(cards.items()))
                  + " (同一条对比线要锁定同一张卡)")
    if verbose:
        if miss_cases:
            print(f"  · G10 {miss_cases} 个 case 的 {miss_runs} 个历史轮次无评测器指纹 —— "
                  f"2026-09-22 之前的记录没有这一项, 与它们比较时**无法确认评测器同版**。"
                  f"新轮次起自动带上; 旧轮次不回填 (回填的指纹是编的)。")
        if ok_cases:
            print(f"  ✓ G10 {ok_cases} 个 case 的轮次都在同一评测器版本下")
    if seen == 0 and verbose:
        print("  · G10 没找到任何 runs 目录, 本条跳过")
    return bad


#  ---- G12 等价形式已处置 ----
_EQUIV_RATIO_GATE = 1.2      # 下界宽出 20% 以上就必须处置 (建它, 或用实测否掉它)


def g12_equiv_handled(verbose=True):
    """枚举器排在现状之前、且**成本项已全部量化**的等价形式, 必须被处置。

    为什么要这条闸 (2026-09-22 立)
    ------------------------------
    equiv 能把等价类摊开, 但摊开没人看就等于没有。本轮的实撞是: 六次调参、三次判负,
    而唯一的大赢是一次数学等价改写 —— 那次改写是用手找的, 因为**没有任何机制**
    要求我在调参之前先看一眼等价类。

    为什么它只管"已全部量化"的那些 (2026-09-22 codex 评审第三条, 采纳)
    ----------------------------------------------------------------
    首版对带 unquantified 的形式**照样报失败**, 只是提示"先测未知项"。那会产生一个
    反向激励: 模型漏算一项成本 -> 下界更低 -> 更容易超过 1.2 倍 -> 更容易制造"必须
    处置"的义务。**模型越不完整, 越能给自己派活。** 本模块原本是要用模型省真机试错,
    那样反而会被模型的遗漏项驱动去做低价值试错。
    而且下界比较本身推不出实现排序: L_候选 < L_现状 只说明候选**尚未被这些下界排除**,
    完全可能现状实际 100us 而候选实际 200us。
    所以权限分三档:
        已全部量化且更优 -> 必须处置 (建 / 实测反驳 / 记为有预算的延期)  [本闸管这档]
        含未量化项       -> **机会**, 进探索队列, 不产生义务, 不红          [只打印]
        装不下 / 机制不成立 -> 直接排除
    """
    import json as _json
    import os as _os
    from . import equiv as E
    d = _os.path.dirname(_os.path.abspath(__file__))
    doc = _json.load(open(_os.path.join(d, "equiv_refuted.json"), encoding="utf-8"))
    done = set()
    for r in doc.get("records", []):
        v = r.get("verdict", "refuted")
        if v == "refuted" and not str(r.get("measured", "")).strip():
            print(f"  ✗ G12 反驳记录缺 measured 字段 (只认带数字的实测): {r}")
            return 1
        if v == "deferred" and not (str(r.get("budget", "")).strip()
                                    and str(r.get("revive_when", "")).strip()):
            print(f"  ✗ G12 延期记录必须写 budget 与 revive_when (否则是无限期搁置): {r}")
            return 1
        done.add((r.get("op"), int(r.get("case", -1)), r.get("form")))
    from .. import socspec
    from . import caps as _caps
    sp = socspec.spec(E.SOC)
    cd = _caps.as_dict(E.SOC)
    dram = socspec.calib_value("dram_GBps", sp["soc"])
    pend, opp, bad = [], [], 0
    for op, ent in E.REGISTRY.items():
        for cid, dims in sorted(E.cases_of(op).items()):
            built = E.built_form(op, dims)
            if built is None:
                continue                  # 走本表未描述的回落路径, 如实跳过
            _d, rows, _cut = E.rank(op, cid, sp, cd, dram)
            usable = [r for r in rows if not r["over"] and r["capok"] is True]
            if not usable:
                continue
            top = usable[0]
            cur = next((r for r in usable if r["form"] == built), None)
            if cur is None or top["form"] == built:
                continue
            a = cur["lb"]["ideal_overlap_floor_us"].value
            b = top["lb"]["ideal_overlap_floor_us"].value
            if not (a and b) or a / b < _EQUIV_RATIO_GATE:
                continue
            if (op, cid, E.form_id(top["form"])) in done:
                if verbose:
                    print(f"  ✓ G12 {op} case {cid}: 已处置 (建 / 实测反驳 / 有预算的延期)")
                continue
            row = (op, cid, a / b, E.form_id(top["form"]), len(top["unq"]))
            (opp if top["unq"] else pend).append(row)
    for op, cid, ratio, fid, nunq in pend:
        bad += 1
        print(f"  ✗ G12 {op} case {cid}: 存在下界宽 {ratio:.1f}x 且**成本项已全部量化**的"
              f"等价形式 {fid} —— 既没建、也没被实测否掉、也没记为有预算的延期")
    if opp and verbose:
        print(f"  · G12 另有 {len(opp)} 个 case 的排名首位**含未量化成本项** —— 按**机会**"
              f"列出, 不产生义务 (下界更低可能只是因为模型漏算了那一项):")
        for op, cid, ratio, fid, nunq in opp[:6]:
            print(f"        case {cid}: {ratio:.1f}x  {fid}  ({nunq} 项未量化)")
        if len(opp) > 6:
            print(f"        ... 另有 {len(opp) - 6} 个")
    if not pend and verbose:
        print("  ✓ G12 没有'已全部量化且更优却未处置'的等价形式")
    return bad


def g13_precision_obligation(verbose=True):
    """非 bitwise 的等价改写, 在被**构建**之前必须有已验的精度义务。

    2026-09-22 采纳外部评审: 「数学公式等价」不等于「这个核可以安全替换基线」。
    support=gather 与 softmax=online 都是实数域恒等而浮点下求和的**项数与次序**都变了。
    这条闸把 equiv 的候选生成接到精度关上: 现状形式用到哪些非 bitwise 的轴取值,
    就必须有对应的 simfunc 检查跑过。
    """
    from . import equiv as E, simfunc
    bad = 0
    kinds = {a.name: a for a in E.AXES}
    #  非默认取值才引入义务: dense/two_pass 是与 golden 同形的基准取值。
    BASE = {"support": "dense", "softmax": "two_pass"}
    for op, ent in E.REGISTRY.items():
        need = set()
        for cid, dims in E.cases_of(op).items():
            b = E.built_form(op, dims)
            if b is None:
                continue
            for ax, val in b.items():
                a = kinds.get(ax)
                if a is None or a.equiv_kind == E.BITWISE:
                    continue
                if BASE.get(ax) == val:
                    continue
                need.add((ax, val))
        for ax, val in sorted(need):
            fn = getattr(simfunc, "main", None)
            if fn is None:
                bad += 1
                print(f"  ✗ G13 {op} 用了非 bitwise 取值 {ax}={val}, 但 simfunc 没有可调用的检查")
                continue
            if verbose:
                print(f"  ✓ G13 {op} {ax}={val}: 精度义务 -> simfunc")
        if not need and verbose:
            print(f"  ✓ G13 {op} 现状形式只用 bitwise 等价的取值, 无精度义务")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(prog="core.opgen gate", description="opgen 自己的准入闸")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    v = not a.quiet
    bad = 0
    #  ★每条闸单独兜异常并计为未过: 一条抛异常就消失的闸, 与静默回落同型
    #    (2026-09-22 实撞: G11 因为少导入一个名字直接崩掉, 而汇总行还在报"全绿")。
    #  ★sys.stdout 要 flush: 管道下 stdout 块缓冲而 stderr 直出, 回溯会排到前面去,
    #    看起来像"判据什么都没输出"。
    for name, fn in (("G1 负例控制", g1_negative_controls), ("G2 出处完备", g2_refs_complete), ("G3 接口一致", g3_interface),
                     ("G4 规范形幂等", g4_canon),
                     ("G5 跨芯片隔离", g5_cross_chip), ("G6 实测自校验", g6_selfcheck),
                     ("G7 功能仿真一致", g7_simfunc),
                     ("G8 文件清单", g8_manifest), ("G9 原语必要性", g9_prims),
                     ("G10 评测器隔离", g10_eval_isolation),
                     ("G11 无静默回落", g11_no_silent_fallback),
                     ("G12 等价形式已处置", g12_equiv_handled),
                     ("G13 精度义务已接", g13_precision_obligation)):
        print(f"\n=== {name}", flush=True)
        try:
            bad += fn(verbose=v)
        except Exception as e:                                     # noqa: BLE001
            import traceback
            bad += 1
            print(f"  ✗ {name} **抛异常**, 计为未过: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
    print("\n" + "=" * 70)
    print(f"已实现的闸: 13 条, {bad} 条未过")
    if _TODO:
        print("★尚未实现的闸 (显式列出, 不假装全绿):")
        for k, why in _TODO.items():
            print(f"    {k}  {why}")
    else:
        print("★G1-G13 全部已实现。新增判据必须同时补 REFS 条目与负例, 否则 G1/G2 会红。")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
