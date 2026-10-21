# 当前 Vector scale 的 Cube 诊断适配

本次将既有、远端使用过的 Cube 调用边界探针接到当前 `vector_scale`，并保留 pinned 原版 FP16 Laser 作独立诊断对照。旧 `int8` kind 仍明确表示 scale-reuse；新增 `vscale` 使用独立源码、算子、vendor、bridge、manifest 和缓存路径。没有采用已退化的 bulk/rowmax，也没有修改生产算术来制造新速度结果。

## 来源与数据路径

`regenerate.py` 从本地固定 MindIE-SD Git objects 重建 vscale/FP16 各自的 baseline/probe。固定提交为 `71098c20ff90838d1fd1ca0a40515c751a3869b0`。它对所有同族 C++ header/source 运行 CANNBot 同步和数据流扫描，保留未过滤的输出、stderr、工具哈希、命令及源码哈希。

独立字节检查要求：移除 Cube 的诊断 hook 后恢复自身 baseline；Vector 和两个地址映射文件不变；原事件、MAD、FIX copy 及 `set_quant_pre` 顺序不变。当前 vscale 的 Q 常驻和每次调用固定 `1/64` Cube transport 均保留，分块反量化仍由 Vector 完成。原版 FP16 从上游固定字节重建，不经过 INT8 patch。

本地重建已通过这些检查：vscale Cube 79/79、Vector 96/96 条匹配的 raw 调用序列一致；FP16 分别为 77/77、91/91。两种 Cube 去插桩后均与各自 parent 字节相同。详见 `source_comparison.json` 及两个 `*_CubeForward.h.diff`。

两种 analyzer 对 vscale baseline/probe 分别扫描 13/14 个文件，对 FP16 分别扫描 12/13 个文件，全部退出 0。两个 probe 的 sync analyzer 各保留一条相同的性能提示，位于 `op_kernel/vc_laser_cube_probe.h:71`：

> SYNC-09，性能：PipeBarrier<PIPE_ALL> 粒度过粗
>
> 若仅需 V 排序可用 PipeBarrier<PIPE_V>；确认无跨 PIPE 依赖后再收窄

该提示对应 completed 模式有意增加的完成屏障，属于需要校准的扰动。本轮不把屏障改成 PIPE_V，因为观测的是 Cube 多流水的完整调用。未产生红线或高级别候选。sync analyzer 只识别到这一条新增 barrier，flow analyzer 对所有树识别的 operations 都为 0；它们没有覆盖原 Laser raw DMA/MAD/flag 的完整数据流，零 findings 不构成同步正确性证明。

## CANNBot 流程和同步核对

- Tiling：保留当前 source 的核数、Q/K 分块、L1/L0/UB 地址和 ping-pong；本次只采样，不制定未经测量的新 tiling。
- 卡间流水：无通信，跳过。
- 核间流水：保留 Cube/AIV 的全部 handoff。技能该目录没有具体分析流程；本探针也未直接采样调用前的跨核等待。
- 核内流水：采集完整 QK/PV 组的 issue/completed 窗口，并先测插桩扰动。没有仿真流水图，不能直接宣布某硬件单元 bound。

| 数据 / owner | Producer → consumer | 诊断边界与生命周期 |
|---|---|---|
| 当前 Q/K L1/L0 及驻留 Q 槽 | 原 MTE2/MTE1 → MAD | 不在 matmul 内部插入新事件或 drain；原驻留复用条件不变 |
| QK L0C / score GM | terminal MAD → FIX copy → Vector load | 固定 transport setter 保留；完整 QK 调用返回后才可 drain；原 FIX 跨核通知位置不变 |
| Vector P、历史对角与输出 | 原跨核 wait → PV/history 调用 | Begin 位于既有 wait 之后；无等待时间分解或 Vector 阶段测量 |
| history 未落盘累积 | diagonal MAD(copy_out=false) → PV terminal MAD/FIX | 绝不在这对依赖之间排空；EndPV 在整个调用返回后 |
| 末尾归一化 | 原 diagonal MAD → 匹配 FIX copy | 完整 PV 组结束后才记录 completed；末尾分组明确记录 normalize 次数 |
| trace record | Cube(core,stage,position) → 主机读取 | 固定独占 `((core*2+stage)*3+position)*24` 个 INT64 字；每次清零，每记录 192B，三个 64B cache line 均 clean out；主机同步后解码 |

expected/actual score、P、output 的根和索引均为继承的地址映射；探针不产生这些张量，只追加独立 trace 写入。metadata 和写 trace 在计时窗口结束之后，但仍可能推迟后面的跨核通知，因此要校准整体扰动。诊断 mode 2 的 `PIPE_ALL` 仅在完整调用之前/之后，匹配的 unitFlag MAD/FIX 已发射；没有新增跨核事件、M→S 或 FIX→S 事件。

## 数据解读范围

每种算法先对自身生产版本进行输出验证和配对校准。plain 无 timestamp/write/drain，但仍有新 trace 分配和编译 footprint；issue 含内部依赖与队列背压，不是纯标量发射时间；completed 显式排空会改变原重叠。pre-drain/empty 样本另列，不相减来“校正”生产耗时。剩余完成窗口不是纯 FIX 或纯矩阵乘时间。

只采样核 0～3、first/middle/last 逻辑组；小任务中的重合位置取首次，非活动记录保持零。descriptor、K block、logical tile、history/normalize 次数在窗口外记录，按相同工作量分组。system ticks 不是 Cube 核时钟周期；各阶段不能相加、换算为生产百分比或外推完整 kernel 时间。跨 FP16/vscale 的校准在不同进程执行，不是配对性能比较。

本地 C++/来源测试只验证生成代码、API/字段一致性及控制流模型，不能替代 CANN 9.0.1、Ascend910_9382 的编译和真实异步执行。用户继续授权诊断适配，原有生产实现保持可用。

本轮本地验证：构建/源码/patch 测试 35 项通过，旧入口及新入口的 profile 测试 31 项通过，共 66 项；Python 编译检查、Bash 语法检查及 `git diff --check` 通过。旧 int8/FP16 生成字节、来源记录和 manifest identity 与修改前 HEAD 一致，公共 bridge 文件未改；新 vscale 离线源码 manifest 已验证。尚未进行本轮 CCE 编译和 NPU 实测。
