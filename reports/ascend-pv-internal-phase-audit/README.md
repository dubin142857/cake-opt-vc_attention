# V-only PV 内部诊断

本轮继续以已实测有效的 `vector_pv_prefetch` 为生产起点。P+V 预取的最新同进程配对结果没有新增收益，因此不将它带入探针。本次没有新的速度结果；目的是在可校准的条件下找出更值得修改的 PV 窗口。

## 来源与测量契约

固定上游 MindIE-SD commit `71098c20ff90838d1fd1ca0a40515c751a3869b0`。诊断为独立 kind `pv`、op `VcLaserCubePvProbeAttention`、vendor/namespace `vc_laser_cube_pv_probe`。生产 V-only、原版 FP16 和融合量化器的既有构建保持可用；准备源码只读取本地固定 Git objects。

运行 `bash scripts/run_mindie_laser_pv_phases.sh`。默认 BF16 BNSD `[1,28,37741,128]`。先逐模式同步 tiny smoke，再保留原有异步排队输出检查；通过后独立测量原版和 V-only 的 full/prepared/native，随后在相同 INT8 payload 上校准生产、plain、issue、detail，最后采样。没有将诊断版当优化候选，没有把 ticks 当生产百分比。

2026-10-23 远端 v1 的首个 FP16 B1/H2/Q1/K1/D128 检查发生 `507014 / aicore timeout`，设备报 FIXP timeout/trap，尚未进入计时。因三模式连续提交，现有日志不能唯一定位失败模式，更不能确认屏障或私有数组就是根因。已证实的主机缺陷是异常后的 `finally: save()` 再次 flush/读取 trace，覆盖原始异常。schema v2 修复首次错误保留与模式定位，同时移除新增全流水 drain。旧 probe 编译产物保留；新版本默认在独立 `*_v2_*` 缓存中从本地固定 Git 对象构建。NPU 修复效果待远端复验。

## 同步及观测边界

| 数据或工作 | Producer → consumer | 插桩约束 |
|---|---|---|
| 当前 P GM | Vector MTE3 发布 → Cube 原有跨核等待 → P GM→L1 | 保持发布、等待及 descriptor 映射顺序；不投机跨 descriptor |
| V 双缓冲 L1B | MTE2 搬入 → MTE1 装载 → 原位提前释放 | 保留 V-only 的 ready/free 及相反槽 lookahead；不改地址、事件或生命周期 |
| P L1A / P、V L0 | MTE1 装载 → MAD 消费 → 原事件释放 | 内部只插入被动计数器读取；不添加 Set/Wait 或 drain |
| history | diagonal MAD(copy_out=false) → 后续 PV 累加 | 未发完匹配的终端 MAD/FIX 前，禁止 PIPE_ALL |
| PV 终端输出 | MAD unitFlag=3 → FIX UnitFlagMode=3 | 保持指令配对；不在两者之间插完成屏障 |
| 末尾归一化 | diagonal MAD → 匹配输出 copy | 不添加完成屏障，不报告完成尾巴 |
| trace | 独占 core/round/K 槽 → 主机同步后解码 | 每次分配清零；有效位区分未采样与真实零值；记录写回在所测窗口外 |

`wait_flag` 是目标流水的同步命令，标量系统计数器不自动跟随该流水等待。因此内部记录是发射/依赖窗口，可能包含队列反压，不能直接称为硬件等待时长。v2 两种采样模式都没有诊断 drain，完成类字段 19/21/22 与 validity bit 1 保持零；decoder 拒绝 v1，summary 不把无效零值当成测得的零耗时。history/归一化、整次调用和单 K 块分别分组，不能相加或重复归属。

## CANNBot 工作流

采用本地 `build/cannbot-skills/ops/ascendc-perf-optimize/SKILL.md`、`ascendc-sync-audit/SKILL.md` 及 API 最佳实践。Step 1 保留现有 M/K/N=128、Ky=2，每 K 块 P=64KiB、V=32KiB 以及既有 L1 双槽、L0/UB 和核数；Step 2 无卡间通信而跳过；Step 3 保留 Cube/AIV 同步，不直接采样跨核等待，技能核间流程目录缺少细则，采用真实源码手工跟踪；Step 4 为被动标量窗口加全调用扰动校准。当前没有仿真流水图，也没有可以证明具体硬件瓶颈的完整证据。

`regenerate.py` 从固定对象重建 V-only 与诊断版，对同族源码运行两个 analyzer，保存原始输出、stderr、命令和哈希；另外核对原始 DMA/load/flag/MAD/FIX 调用序列及去插桩后的字节等价。CCE 编译和实际异步执行需要远端设备验证。

本地重建证明：Cube 去插桩后逐字节等于 V-only；匹配的原始调用序列 94/94 一致；Vector 的 96/96 条调用及两个地址映射文件逐字节不变。旧 `int8`、`vscale`、`fp16` 三种探针与提交 `67b9e75` 的生成源码、来源记录和 manifest identity 相同，避免本次功能扩展使已验证的构建失效。见 `source_comparison.json`、`legacy_equivalence.json`。

两个 analyzer 对 baseline/probe 分别扫描 13/14 个文件。v1 报告的一条 SYNC-09 性能提示对应新增 `PipeBarrier<PIPE_ALL>`；v2 删除该屏障而不换成另一种完成等待。更新后的原始输出见同目录 JSON。analyzer 未覆盖 Laser 原始 DMA/MAD/flag 的完整数据流，operations 为 0，因此无 findings 不构成同步证明，更不能证明这次设备超时已消失。去插桩字节等价也不能证明编译器生成的异步执行时序等价。

跨语言回环测试将实际 `PvProbe` C++ header 用主机 stub 编译执行，产生 trace，再经 JSON 交给实际 Python decoder，覆盖 plain/issue/detail、首中尾重合去重、inactive、完整长 shape 槽位、字段破损和分组不重复。v2 stub 将任何新增 PipeBarrier 视为失败，并检查旧 schema、伪造完成字段和 validity bit 1 均被拒绝；它不模拟真实 NPU 异步流水，也不替代远端输出门限与编译。

v1 曾通过 130 项本地测试但随后远端发生超时，表明 host 测试无法代替设备验证。v2 增加故障注入测试，检查 native launch、同步、trace CPU 读取和上层 finally 的失败路径，确保首错后不再读取设备、后续模式不再提交，同时保留原来的无同步 scale 更新门禁。当前 header 的 scalar GM 写与 DCCI 仅用于稀疏诊断并放在计时窗口之后；额外 footprint 与 trace 分配仍在 plain 校准中保留。未进行本轮 CCE 编译或 NPU 实测。

v2 本地验证：8 个 PV/Cube/vscale 测试文件共 **155 passed**、无 skip，包含真实 C++ writer→Python decoder 的 host 回环与异步错误路径故障注入。Python 编译、Bash 语法和 `git diff --check` 通过。`regenerate.py` 重建固定上游和探针同族源码，四次 analyzer 均退出 0；原始指令序列及旧三种探针源码/identity 保持一致。仍须远端对同一个 tiny 输入执行 CCE/NPU 验证。
