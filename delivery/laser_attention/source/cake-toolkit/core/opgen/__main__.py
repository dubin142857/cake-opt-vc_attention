#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opgen 的命令行入口。"""
from __future__ import annotations
import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="core.opgen",
                                 description="算子设计层: 芯片机制能力 + 调度 IR + 四类判据")
    sub = ap.add_subparsers(dest="cmd")
    p1 = sub.add_parser("caps", help="探测这颗芯片有哪些机制能力 (及其证据)")
    p1.add_argument("--soc", default=None)
    p2 = sub.add_parser("selfcheck", help="用已实测的算子反过来验这套 IR 的判据")
    p3 = sub.add_parser("gate", help="本模块自己的准入闸 (G5/G6/G9/G10, 未实现的显式列出)")
    p3.add_argument("--quiet", action="store_true")
    p4 = sub.add_parser("equiv", help="枚举一个算子的**数学等价形式**并按 IR 判据排序")
    p4.add_argument("--op", default="sparse_flash_attention")
    p4.add_argument("--case", type=int, default=None,
                    help="不给则扫全部 case, 只报第一名与现状的差距")
    p4.add_argument("--top", type=int, default=8)
    p4.add_argument("--spec", action="append", default=[],
                    help="接入算子谱的模块 (import 时调 equiv.register), 可多次给。"
                         "★benchmark 接入属**插件层**: core 不 import 插件, 由这里显式加载")
    p4.add_argument("--bench-report", default=None,
                    help="cann-bench 评测报告目录: 给了才排得出**先做哪个 case**")
    p5 = sub.add_parser("api", help="一条 AscendC API 的**五层取证** (A1 存在 -> A5 指令/编译器)")
    p5.add_argument("name", nargs="?", help="API 名, 如 DataCopyPad / Mmad / Fixpipe")
    p5.add_argument("--soc", default=None)
    p5.add_argument("--probe", action="store_true", help="跑 E1 编译探针 (秒级, 不占卡)")
    p5.add_argument("--raw", action="store_true", help="打印原始签名")
    p5.add_argument("--regwidth", action="store_true",
                    help="只看 A5 的地基: 向量寄存器字节宽, 以及它与平台 ini 的对账")
    p5.add_argument("--instr", default=None, help="查编译器内建指令表里的一条, 如 vadds")
    #  ★没有这两条转发时 api.selftest() 的负例控制从命令行**根本到不了**, 于是判据自检
    #    (opdbg --selftest-criteria) 也看不见它们 —— 沉淀了却没绑定, 等于没有。
    p5.add_argument("--selftest", action="store_true", help="负例控制自检")
    p5.add_argument("--refs", action="store_true", help="本层判据的实证出处")
    p6 = sub.add_parser("architecture", help="P1架构对象/具体场景结构校验；不证明数值或性能")
    source = p6.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="待校验的版本化JSON对象")
    source.add_argument("--example", choices=("matmul", "dense_attention", "sparse_attention"))
    source.add_argument("--selftest", action="store_true", help="对象/架构操作和数学图身份反例回归，无设备")
    source.add_argument("--verify-transform", help="重放已保存变换回执；不认证外部来源或硬件")
    source.add_argument("--observation", help="检查观测/来源/覆盖文档；不从时序推必需依赖")
    source.add_argument("--campaign", help="检查组合方案/开发依赖/预测预算文档")
    source.add_argument("--legacy", help="检查旧IR的版本化完整快照；不自动构造事件图")
    source.add_argument("--trace", help="读取一个Chrome/CANN JSON trace，生成带输入审计的观测回执")
    source.add_argument("--trace-plan", help="按已冻结的hash绑定TracePlan读取输入")
    source.add_argument("--verify-trace", help="从绑定输入重新生成并核对trace回执")
    p6.add_argument("--sim-report", help="显式模拟器独立时长JSON；不自动搜索父目录")
    p6.add_argument("--trace-unit", choices=("us","ns"), help="ts/dur存储单位；新建计划默认Chrome us")
    p6.add_argument("--trace-clock", help="输入时钟域；新建计划默认trace_time，不声明跨时钟对齐")
    p6.add_argument("--trace-time-encoding", choices=("number","decimal_ns"), help="显式支持CANN十进制字符串；decimal_ns要求精确整数ns")
    p6.add_argument("--trace-origin-ns", type=int, help="decimal_ns的显式时钟原点，不作为跨时钟标定")
    p6.add_argument("--single-launch", help="显式声明整份capture的单launch标签；默认保持未知")
    p6.add_argument("--expected-core", action="append", default=[], help="期望采集核，可重复")
    p6.add_argument("--relative-tolerance", type=float, help="独立时长对账的显式相对容差；默认只报残差")
    p6.add_argument("--absolute-tolerance-ns", type=float, help="独立时长对账的显式绝对容差ns")
    p6.add_argument("--transform", help="以--input为源应用版本化变换计划，输出回执")
    p6.add_argument("--output", help="显式新输出文件；拒绝覆盖旧证据")
    p7 = sub.add_parser("schedule", help="P3有限DAG条件调度、资源占用与连续地址证书")
    schedule_source=p7.add_mutually_exclusive_group(required=True)
    schedule_source.add_argument("--input", help="具体Architecture JSON")
    schedule_source.add_argument("--loop-receipt", help="重放LoopReceipt并调度其已核对的有限展开")
    schedule_source.add_argument("--verify", help="重放ScheduleReceipt，不认证硬件或数值")
    schedule_source.add_argument("--selftest", action="store_true", help="独立小图/地址oracle和功能回放")
    p7.add_argument("--plan", help="冻结的SchedulePlan；不接受同时覆盖参数")
    p7.add_argument("--strategy", choices=("list","exact"))
    p7.add_argument("--max-states", type=int, help="有限搜索状态预算，不是收益门槛")
    p7.add_argument("--horizon-ns", type=int, help="显式搜索范围；找不到不代表全局无解")
    p7.add_argument("--tick-ns", type=int, help="整数调度网格，默认1ns")
    p7.add_argument("--output", help="新建回执文件，拒绝覆盖")
    p8 = sub.add_parser("loop", help="P3显式有限循环/slot/seed/接口初始化与排空")
    loop_source=p8.add_mutually_exclusive_group(required=True)
    loop_source.add_argument("--input", help="带迭代距离的Architecture模板")
    loop_source.add_argument("--verify", help="重放LoopReceipt，不认证硬件/任意递推数学")
    loop_source.add_argument("--selftest", action="store_true", help="循环/尾块/初始化/物理回放反例，无设备")
    p8.add_argument("--plan", help="必须显式提供的LoopPlan")
    p8.add_argument("--output", help="仅新建输出回执")
    p9 = sub.add_parser("analysis", help="P3显式观测/模型约束重放与完整假设场景敏感性")
    analysis_source=p9.add_mutually_exclusive_group(required=True)
    analysis_source.add_argument("--input", help="Architecture或TimingHypothesis JSON；同时需要--plan和观测输入")
    analysis_source.add_argument("--study", help="完整场景StudyPlan，不自动拟合成本")
    analysis_source.add_argument("--verify", help="重放ReplayReceipt或StudyReceipt，保留父前提")
    analysis_source.add_argument("--selftest", action="store_true", help="独立流水/地址/观测负例与增益区间检查")
    p9.add_argument("--plan", help="绑定模型/观测身份的ReplayPlan")
    p9.add_argument("--trace-receipt", help="先重放受限输入读取的TraceReceipt父回执")
    p9.add_argument("--observation", help="ObservedExecution；仅清单来源，不认证文件读取")
    p9.add_argument("--loop-parent", help="可选LoopReceipt父前提，必须匹配Architecture")
    p9.add_argument("--output", help="只新建回执文件")
    p10 = sub.add_parser("policy", help="P4组合分支/预算/证据条件建议与不可变开发状态")
    policy_source=p10.add_mutually_exclusive_group(required=True)
    policy_source.add_argument("--input", help="Campaign JSON，与--plan组合")
    policy_source.add_argument("--verify", help="重放ReviewReceipt或StateReceipt及其父证据")
    policy_source.add_argument("--selftest", action="store_true", help="组合负收益/模型反证/预算/重开与失效回归")
    p10.add_argument("--plan", help="冻结的ReviewPlan或StatePlan")
    p10.add_argument("--evidence", help="ReviewEvidence；默认显式缺证据")
    p10.add_argument("--review", help="状态变化需使用的冻结ReviewReceipt")
    p10.add_argument("--output", help="只新建回执，建议不执行任务或发布候选")
    p11 = sub.add_parser("runner", help="P5隔离的身份绑定执行链首步；CPU假驱动不认证NPU")
    runner_source=p11.add_mutually_exclusive_group(required=True)
    runner_source.add_argument("--plan", help="冻结RunPlan；只能写其任务输出目录")
    runner_source.add_argument("--verify", help="只核验RunReceipt与父身份，不重跑命令")
    runner_source.add_argument("--selftest", action="store_true", help="真实namespace故障/缓存/恢复与超时反例")
    p11.add_argument("--resume", action="store_true", help="同计划、lease已释放且身份有效时从checkpoint恢复")
    p11.add_argument("--cancel", action="store_true", help="只请求取消该已登记计划")
    p11.add_argument("--output", help="可选新建导出回执；已存在则不启动执行")
    p12 = sub.add_parser("lowering", help="P5变换/实施约束到固定native代码及采样ABI存储绑定")
    lowering_source=p12.add_mutually_exclusive_group(required=True)
    lowering_source.add_argument("--plan", help="LoweringPlan JSON；只核验声明和身份，不启动设备")
    lowering_source.add_argument("--verify", help="重放采样物理约束LoweringReceipt")
    lowering_source.add_argument("--selftest", action="store_true", help="父链/布局/地址/设备/库与缺项负例，无设备")
    p12.add_argument("--output", help="只新建输出文件")
    sub.add_parser("refs", help="列出本模块判据的实证出处")
    pd=sub.add_parser('delivery',help='P5保护指标与模型绑定的best晋升/fresh恢复；不授权视频质量')
    ds=pd.add_mutually_exclusive_group(required=True);ds.add_argument('--plan');ds.add_argument('--verify');ds.add_argument('--selftest',action='store_true')
    pd.add_argument('--campaign');pd.add_argument('--output')
    a = ap.parse_args(argv)
    if a.cmd == "lowering":
        from .lowering import cli as lowering_cli
        return lowering_cli(a)
    if a.cmd == "runner":
        from .runner import cli as runner_cli
        return runner_cli(a)
    if a.cmd == "policy":
        from .policy import cli as policy_cli
        return policy_cli(a)
    if a.cmd == "analysis":
        from .analysis import cli as analysis_cli
        return analysis_cli(a)
    if a.cmd == "loop":
        from .loops import cli as loop_cli
        return loop_cli(a)
    if a.cmd == "schedule":
        from .schedule import cli as schedule_cli
        return schedule_cli(a)
    if a.cmd == "architecture":
        trace_action=any((a.trace,a.trace_plan,a.verify_trace))
        trace_options=any(x is not None for x in (a.sim_report,a.trace_unit,a.trace_clock,a.single_launch,a.relative_tolerance,a.absolute_tolerance_ns,a.trace_time_encoding,a.trace_origin_ns)) or bool(a.expected_core)
        if trace_options and not trace_action:
            ap.error('trace parameters require --trace, --trace-plan or --verify-trace')
        from .architecture import cli
        return cli(a)
    if a.cmd == 'delivery':
        from .delivery import cli
        return cli(a)
    if a.cmd == "caps":
        from .caps import show
        return show(a.soc)
    if a.cmd == "selfcheck":
        from .selfcheck import main as sc
        return sc()
    if a.cmd == "gate":
        from .gate import main as gm
        return gm(["--quiet"] if a.quiet else [])
    if a.cmd == "equiv":
        import importlib
        for m in a.spec:
            importlib.import_module(m)      # 模块在 import 时自己 register
        from .equiv import show, sweep
        return (show(a.op, a.case, a.top) if a.case
                else sweep(a.op, a.bench_report))
    if a.cmd == "api":
        from .api import main as am
        argv2 = ([a.name] if a.name else [])
        if a.soc:
            argv2 += ["--soc", a.soc]
        for f in ("probe", "raw", "regwidth", "selftest", "refs"):
            if getattr(a, f):
                argv2 += [f"--{f}"]
        if a.instr:
            argv2 += ["--instr", a.instr]
        return am(argv2)
    if a.cmd == "refs":
        from .._refs import print_refs
        from .caps import REFS as R1
        from .ir import REFS as R2
        from .semantics import REFS as R3
        from .equiv import REFS as R4
        from .api import REFS as R5
        from .architecture import REFS as R6
        from .transforms import REFS as R7
        from . import observation, campaign, legacy, trace, schedule, loops
        print_refs("opgen.caps", R1)
        print()
        print_refs("opgen.ir", R2)
        print()
        print_refs("opgen.semantics", R3)
        print()
        print_refs("opgen.equiv", R4)
        print()
        print_refs("opgen.api", R5)
        print()
        print_refs("opgen.architecture", R6)
        print()
        print_refs("opgen.transforms", R7)
        for module in (observation,campaign,legacy,trace,schedule,loops):
            print()
            print_refs(module.__name__,module.REFS)
        return 0
    ap.print_help()
    print("\n★先敲 `python -m core.opgen caps` 看这颗芯片给了什么机制, "
          "再敲 `selfcheck` 看这套判据被实测验到什么程度。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
