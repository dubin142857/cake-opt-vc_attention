# V-only PV 内部诊断

本轮继续以已实测有效的 `vector_pv_prefetch` 为生产起点。P+V 预取的最新同进程配对结果没有新增收益，因此不将它带入探针。本次没有新的速度结果；目的是在可校准的条件下找出更值得修改的 PV 窗口。

## 来源与测量契约

固定上游 MindIE-SD commit `71098c20ff90838d1fd1ca0a40515c751a3869b0`。诊断为独立 kind `pv`、op `VcLaserCubePvProbeAttention`、vendor/namespace `vc_laser_cube_pv_probe`。生产 V-only、原版 FP16 和融合量化器的既有构建保持可用；准备源码只读取本地固定 Git objects。

运行 `bash scripts/run_mindie_laser_pv_phases.sh`。默认 BF16 BNSD `[1,28,37741,128]`。先检查输出，再独立测量原版和 V-only 的 full/prepared/native，随后在相同 INT8 payload 上校准生产、plain、issue、detail，最后采样。没有将诊断版当优化候选，没有把 ticks 当生产百分比。

## 同步及观测边界

| 数据或工作 | Producer → consumer | 插桩约束 |
|---|---|---|
| 当前 P GM | Vector MTE3 发布 → Cube 原有跨核等待 → P GM→L1 | 保持发布、等待及 descriptor 映射顺序；不投机跨 descriptor |
| V 双缓冲 L1B | MTE2 搬入 → MTE1 装载 → 原位提前释放 | 保留 V-only 的 ready/free 及相反槽 lookahead；不改地址、事件或生命周期 |
| P L1A / P、V L0 | MTE1 装载 → MAD 消费 → 原事件释放 | 内部只插入被动计数器读取；不添加 Set/Wait 或 drain |
| history | diagonal MAD(copy_out=false) → 后续 PV 累加 | 未发完匹配的终端 MAD/FIX 前，禁止 PIPE_ALL |
| PV 终端输出 | MAD unitFlag=3 → FIX UnitFlagMode=3 | 保持指令配对；不在两者之间插完成屏障 |
| 末尾归一化 | diagonal MAD → 匹配输出 copy | 整次 PV 返回后才能观察完成尾巴 |
| trace | 独占 core/round/K 槽 → 主机同步后解码 | 每次分配清零；有效位区分未采样与真实零值；记录写回在所测窗口外 |

`wait_flag` 是目标流水的同步命令，标量系统计数器不自动跟随该流水等待。因此内部记录是发射/依赖窗口，可能包含队列反压，不能直接称为硬件等待时长。整次 PV 完成尾巴覆盖剩余排队工作，并非纯 FIX 或纯矩阵乘。history/归一化、整次调用和单 K 块分别分组，不能相加或重复归属。

## CANNBot 工作流

采用本地 `build/cannbot-skills/ops/ascendc-perf-optimize/SKILL.md`、`ascendc-sync-audit/SKILL.md` 及 API 最佳实践。Step 1 保留现有 M/K/N=128、Ky=2，每 K 块 P=64KiB、V=32KiB 以及既有 L1 双槽、L0/UB 和核数；Step 2 无卡间通信而跳过；Step 3 保留 Cube/AIV 同步，不直接采样跨核等待，技能核间流程目录缺少细则，采用真实源码手工跟踪；Step 4 为被动标量窗口加全调用扰动校准。当前没有仿真流水图，也没有可以证明具体硬件瓶颈的完整证据。

`regenerate.py` 从固定对象重建 V-only 与诊断版，对同族源码运行两个 analyzer，保存原始输出、stderr、命令和哈希；另外核对原始 DMA/load/flag/MAD/FIX 调用序列及去插桩后的字节等价。CCE 编译和实际异步执行需要远端设备验证。

本地重建证明：Cube 去插桩后逐字节等于 V-only；匹配的原始调用序列 94/94 一致；Vector 的 96/96 条调用及两个地址映射文件逐字节不变。旧 `int8`、`vscale`、`fp16` 三种探针与提交 `67b9e75` 的生成源码、来源记录和 manifest identity 相同，避免本次功能扩展使已验证的构建失效。见 `source_comparison.json`、`legacy_equivalence.json`。

两个 analyzer 对 baseline/probe 分别扫描 13/14 个文件，四次均退出 0。probe 保留一条性能提示，位于 `op_kernel/vc_laser_pv_probe.h:104`：

> SYNC-09，性能：PipeBarrier<PIPE_ALL> 粒度过粗
>
> 若仅需 V 排序可用 PipeBarrier<PIPE_V>；确认无跨 PIPE 依赖后再收窄

它对应 detail 的整次 PV 结束完成屏障，本实验有意保留并测量其扰动，不能替换成只等 Vector 的屏障来声称 Cube 已完成。没有红线或高级别候选。同步 analyzer 只识别了这个新增 barrier；数据流 analyzer 的 operations 均为 0，未覆盖 Laser 原始 DMA/MAD/flag 的完整数据流，因此该结果不构成同步证明。手工检查和去插桩字节检查补充了这一边界。

跨语言回环测试将实际 `PvProbe` C++ header 用主机 stub 编译执行，产生 trace，再经 JSON 交给实际 Python decoder，覆盖 plain/issue/detail、首中尾重合去重、inactive、完整长 shape 槽位、字段破损和分组不重复。stub 还检查完成屏障只能发生在整次调用结束；它不模拟真实 NPU 异步流水，也不替代远端输出门限与编译。

本轮本地共 130 项通过、无跳过：旧 Cube 源码/构建/入口回归及新 profile/回环合计 116 项，新 PV 来源/构建/插桩 14 项。其中真实 C++ writer→Python reader 回环为 35 项。Python 编译、Bash 语法和差异检查通过。当前 header 的 scalar GM 写与 DCCI 仅用于稀疏诊断并放在计时窗口之后，不能拿它当生产高性能写法；其额外 footprint 与 trace 分配仍在 plain 校准中保留。未进行本轮 CCE 编译或 NPU 实测。
