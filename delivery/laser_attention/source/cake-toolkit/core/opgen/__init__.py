"""opgen —— 算子**设计**层: 芯片机制能力探测 + 调度 IR + 四类判据。

与本仓其它模块的分工:
  socspec   这颗芯片有多大多快 (容量/核数/主频/MAC)      —— 规格
  opgen     这颗芯片让你怎么写, 以及某个调度成不成立    —— 设计
  mixaddr / detsync_lint / ub_budget / alignlint        —— 写完之后的静态判据
  opeval / diagnose / hwfloor                           —— 上板之后的实测与归因

P1 架构对象: 数学符号/实例/布局/区域/事件依赖/资源/来源的版本化对象，
用 architecture --input 校验具体场景，未知输出conditional；不替代数值或性能验证。

入口: python -m core.opgen  (或 ./core/opdbg opgen)
  python -m core.opgen architecture --example dense_attention --output model.json
  python -m core.opgen architecture --input model.json
  python -m core.opgen architecture --input model.json --transform plan.json --output receipt.json
  python -m core.opgen architecture --verify-transform receipt.json
  python -m core.opgen architecture --observation observation.json
  python -m core.opgen architecture --campaign campaign.json
  python -m core.opgen architecture --legacy legacy.json
  python -m core.opgen architecture --trace trace.json --sim-report sim_core_duration.json --output receipt.json
  python -m core.opgen architecture --verify-trace receipt.json
  python -m core.opgen schedule --input model.json --plan schedule-plan.json --output schedule-receipt.json
  python -m core.opgen schedule --verify schedule-receipt.json
  python -m core.opgen loop --input template.json --plan loop-plan.json --output loop-receipt.json
  python -m core.opgen schedule --loop-receipt loop-receipt.json --output schedule-receipt.json
  python -m core.opgen analysis --input model.json --plan replay-plan.json --trace-receipt trace-receipt.json
  python -m core.opgen analysis --study study-plan.json
  python -m core.opgen policy --input campaign.json --plan review-plan.json --evidence review-evidence.json
  python -m core.opgen runner --plan run-plan.json
  python -m core.opgen runner --verify run-receipt.json
  python -m core.opgen architecture --selftest
"""
from . import caps, ir, semantics, graph, equiv, api, architecture, transforms, observation, campaign, legacy, trace, schedule, loops, analysis, policy, runner, ascend, lowering, delivery

# 本仓自定: package-level discovery must expose real child criteria with their
# namespace intact. This registers provenance, not additional trust or coverage.
_REF_MODULES = (caps,ir,semantics,graph,equiv,api,architecture,transforms,observation,campaign,legacy,trace,schedule,loops,analysis,policy,runner,ascend,lowering,delivery)
REFS = {module.__name__.rsplit('.',1)[-1]+'.'+key:ref
        for module in _REF_MODULES for key,ref in getattr(module,'REFS',{}).items()}
