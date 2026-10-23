# V-only PV 内部诊断

本轮继续以已实测有效的 `vector_pv_prefetch` 为生产起点。P+V 预取的最新同进程配对结果没有新增收益，因此不将它带入探针。本次没有新的速度结果；目的是在可校准的条件下找出更值得修改的 PV 窗口。

## 来源与测量契约

固定上游 MindIE-SD commit `71098c20ff90838d1fd1ca0a40515c751a3869b0`。诊断为独立 kind `pv`、op `VcLaserCubePvProbeAttention`、vendor/namespace `vc_laser_cube_pv_probe`。生产 V-only、原版 FP16 和融合量化器的既有构建保持可用；准备源码只读取本地固定 Git objects。

原诊断入口为 `bash scripts/run_mindie_laser_pv_phases.sh`；**v2 已在远端失败，不重复运行旧探针。当前 ABI-only 六个小场景已通过，下一步为下述 constant-plain 隔离**。原计划默认 BF16 BNSD `[1,28,37741,128]`：先逐模式同步 tiny smoke，再保留原有异步排队输出检查；通过后独立测量原版和 V-only 的 full/prepared/native，随后在相同 INT8 payload 上校准生产、plain、issue、detail，最后采样。没有将诊断版当优化候选，没有把 ticks 当生产百分比。

2026-10-23 远端 v1 的首个 FP16 B1/H2/Q1/K1/D128 检查发生 `507014 / aicore timeout`，设备报 FIXP timeout/trap，尚未进入计时。因三模式连续提交，v1 日志不能唯一定位失败模式，更不能确认屏障或私有数组就是根因。已证实的主机缺陷是异常后的 `finally: save()` 再次 flush/读取 trace，覆盖原始异常。schema v2 修复首次错误保留与模式定位，同时移除新增全流水 drain。旧 probe 编译产物保留；v2 在独立 `*_v2_*` 缓存中从本地固定 Git 对象构建。后续 v2 仍失败，详见下节。

## v2 远端失败：先检查已有编译产物

2026-10-23 第二份远端日志已确定调用顺序：FP16 B1/H2/Q1/K1/D128 的 **V-only production 单独同步成功，随后 plain(mode 0) 单独同步超时**。issue/detail 尚未提交。错误为 `507014`，core 20，FIXP timeout/trap；没有取得本轮性能数据。移除新增 `PIPE_ALL` 未解决问题，不能再将它描述为已验证的修复。

plain 不执行系统计数器读取、trace 写回或 DCCI；但它仍与 issue/detail 共用一个包含动态分支和 `values_[3][64]` 私有数组（1536 字节）的编译内核。因此不能把 plain 当成与原核机器码相同的对照。源码复核未发现明确的 bridge 参数顺序、第三输出或 tiling attr 索引错误；代码生成、资源布局或原始调度被插桩扰动均是待验证假设。core 20 的故障也不能直接归属某个 PV 采样窗口。

```bash
git pull origin master
bash scripts/inspect_mindie_laser_pv_failure.sh
```

检查器只使用 Python 标准库和现有 CANN `llvm-objdump`，读取失败报告、worker 日志、production/probe 构建清单及其 device ELF，不加载 torch/NPU，不重新构建、不联网、不发射 kernel。按构建记录核对二进制 SHA256；不一致的对象停止反汇编。工具从清单记录的 CANN 路径寻找，找不到或不支持目标 ELF 时仍保存部分证据，不要求安装 NVIDIA `nvdisasm`。

结果摘要为 `results/mindie_laser_pv_failure.log`，结构化结果为 `results/mindie_laser_pv_failure.json`；原始工具输出保存在独立 artifact 目录。编译资源按实际导出的字段记录，缺少 stack/register 信息即为未知，不能凭源码数组大小断言寄存器溢出。日志中 `current_pc - start_pc = 0x50b4` 仅为直接相减值；运行时载入基址、MIX AIC/AIV 物理符号和对象身份尚未全部确认。检查器的符号内候选窗口是显式假设，不能作为已经锁定的故障指令，更不能从未知 FIXP 位域推算根因。

离线检查器本地回归 **40 passed**：覆盖构建/元数据哈希冲突、嵌套哈希不误判、工具缺失/超时、意外解析错误保留已采证据、不同 section 的同地址隔离、PC 边界及未知指令拒绝，并增加下述实际格式的不可用分类与有界原始输出预览。测试使用合成工具输出，不代表本地有 CANN/NPU 或已经确认故障指令。

## 反汇编不可用后：独立 ABI-only 隔离

2026-10-23 用户回传实际 `llvm-objdump --disassemble` 输出前 45 行：probe 和 baseline 的 ELF 格式均为 `elf64-hiipu`，回传片段的地址行均为 `<not available>`，stderr 为空。这些占位没有提供可解码的指令，不能靠放宽解析器得到指令，也不能据此断言对象损坏或寄存器溢出。检查器现在将只有这类占位、无实际指令的情形标为 `disassembly_unavailable`，保留原始输出；未知地址行不参与 PC 候选定位。不要求安装工具或重编已有内核来重试这个检查。

符号表给出的 probe/baseline AIC 函数大小分别为 43352/26680 字节，增加 16672 字节（约 62.5%）；AIV 均为 10036 字节。这证明两个 AIC 编译体的大小不同，不能把 plain 当成原版机器码。它不能证明大小增长就是 FIXP 超时根因；AIV 同大小也不等于机器码完全一致。资源字段仍未知，`0x50b4` 仍是未确认映射的 PC 相减值。

新增独立 kind `pv_abi`、op `VcLaserCubePvAbiProbeAttention`、vendor/namespace `vc_laser_cube_pv_abi_probe`。它保留 probe 的第三个 native 输出、`probe_mode` attr/tiling 字段与 PyTorch 二输出接口，但设备 Cube/Vector/mapping 使用 V-only 的原始源码，不引入探针头文件、成员数组或计时/写记录钩子。kernel 入口仅保留额外 trace 指针且不使用它；bridge 分配并清零 int64 `[4,3,3,64]` trace，只接受 mode 0、ExpCast 关闭。身份重命名外的设备主体逐字节等价由本地测试验证，**不声称编译后机器码等价**。

```bash
git pull origin master
bash scripts/run_mindie_laser_pv_abi.sh
```

脚本核验并复用现有 V-only 和融合量化器，只从本地 Git objects 构建新的 ABI-only provider；源码、安装和 bridge 使用独立 `mindie_laser_cube_pv_abi_probe_*` 路径。它不运行旧 v2，也不测长序列性能。独立进程依次检查 FP16/BF16 的 B1/H2/Q1/K1/D128、Q17/K129/D64、Q65/K257/D128；每个输入共用同一量化 payload，先运行并单独同步 V-only，再运行并单独同步 ABI-only，要求全部 padded native 输出精确一致、trace 全零。

结果保存到 `results/mindie_laser_pv_abi.json`、日志同名 `.log`。首次错误后只保存主机证据，不继续读取设备或发射后续用例。worker 默认 180 秒超时；超时会尝试终止主机进程组，但**不证明 NPU 任务已停止，也不自动继续其他设备测试**。

若通过，可缩小到“这些小输入上的扩展接口可工作”，随后再隔离插桩编译结构；若失败，保留生产/候选各自的 launch/sync 阶段和首个错误，继续检查扩展接口及其编译执行。两者均不能单独证明旧 v2 的超时根因。本地无 CCE/NPU，本轮没有性能或设备修复结论。

本轮相关本地回归共 **206 passed**：离线检查器、新 ABI 构建与既有 Cube/PV 回归 175 项，ABI-only 验证流程 31 项。覆盖设备主体源码等价、旧四族生成内容与身份不变、首次异常不被后续设备访问覆盖、超时保留证据、首调用 marker 及后续缓存复用。Python 编译、Bash 语法和 diff 检查通过；这些检查不替代远端 CCE 编译与 NPU 执行。

## ABI-only 远端通过：只改变编译期关闭条件

2026-10-23 用户回传 `results/mindie_laser_pv_abi.json` 的终端摘要：独立 bridge 编译成功，六个小输入全部通过；FP16/BF16 的 Q1/K1/D128、Q17/K129/D64、Q65/K257/D128 均为 B1/H2，全部 padded native 输出精确一致、trace 全零。包括此前 runtime plain0 超时的原 tiny 输入。`allow_internel_format=False` 提示后验证继续并通过，不是本次失败。

该结果证明扩展接口和未插桩设备主体在这些小输入上可以工作；不证明长序列正确性、速度、旧 v2 已修复，或动态数组就是根因。本地没有用户完整 JSON，以上是终端回传事实。结合旧 AIC 函数增大的证据，下一步检查运行时关闭模式的编译结构，而不是继续修改接口或放宽精度门限。

新增独立 kind `pv_const`、op `VcLaserCubePvConstProbeAttention`、vendor/namespace `vc_laser_cube_pv_const_probe`。它保留 v2 的原始 `PvProbe` header、成员数组及全部 hooks，只把 kernel 入口中的 `SetupCubeProbe(cube_trace_gm, tiling_data->probeMode)` 改为 `SetupCubeProbe(cube_trace_gm, 0)`。bridge 和 tiling 只接受 mode 0，ExpCast 关闭。`Init`、`SetupCubeProbe`、`Run` 和采样方法是 inline，常量实参可为编译器删除未执行分支提供条件；不能提前保证 CCE 已删除全部状态或生成了原版机器码。不能同时删数组或换空类，否则不再是这次对照。

```bash
git pull origin master
bash scripts/run_mindie_laser_pv_const.sh
```

入口先核验已有 ABI-only 通过报告与生产/ABI-only/融合量化器构建，只从本地 Git objects 构建新 provider，缓存独立为 `mindie_laser_cube_pv_const_probe_*`。随后读取新 provider 与 ABI-only 的 `llvm-objdump --syms` 输出，核对构建 SHA 并记录实际 AIC/AIV 大小。`--symbols-only` 不读旧故障日志、不尝试反汇编或 PC 映射；工具或资源字段缺失只留下部分辅助证据。符号大小相等不证明指令相等，大小减小不证明数组或全部分支已消除。

验证仍只跑原六个小输入，共用 production payload、逐调用单独同步、全部 padded 输出 exact、trace 全零；不启动旧 runtime0，不进入长 shape 或性能计时。沿用首错停止和 180 秒主机进程超时限制，主机终止不代表设备任务取消。输出为 `results/mindie_laser_pv_const.json`、`.log`；符号证据为 `results/mindie_laser_pv_const_symbols.json`、`.log`。

若通过，说明固定关闭条件的编译版本可以在这些输入执行，可进一步改用单窗口、少量具名标量的最小 active 探针；若失败，说明仅传常量不足以恢复执行，仍需考虑常量传播是否发生及其他编译差异。两种结果都不能单独指认寄存器溢出、编译器缺陷或某条 FIXP 指令。当前没有 constant-plain 的远端结果。

本次相关本地测试 **148 passed**：符号/离线检查 42 项、共享小输入验证流程 50 项、新 constant-plain 与 ABI-only 构建 22 项、既有 Cube/PV 构建和 patch 回归 34 项。测试确认五种旧 provider 的生成内容/身份/路径不变，新增源码在身份归一化后仅入口常量和 mode0 guard 与 v2 不同。Python、Bash 及内嵌 Python 语法检查通过。本地无 CCE/NPU，仍需远端验证。

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

v2 本地验证：8 个 PV/Cube/vscale 测试文件共 **155 passed**、无 skip，包含真实 C++ writer→Python decoder 的 host 回环与异步错误路径故障注入。Python 编译、Bash 语法和 `git diff --check` 通过。`regenerate.py` 重建固定上游和探针同族源码，四次 analyzer 均退出 0；原始指令序列及旧三种探针源码/identity 保持一致。这些 host 检查未覆盖随后远端出现的 plain0 超时。
