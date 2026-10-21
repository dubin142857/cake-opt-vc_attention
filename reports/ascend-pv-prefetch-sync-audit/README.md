# PV 下一块 V 搬运预取：独立候选与同步审计

本候选继承已实测的 `vector_scale`，只将 PV 内下一块 V 的 GM→L1 搬运提前。原版 FP16 Laser 是性能主基线，当前 Vector scale 是同算法基线。采用独立的 `VcLaserQuantVscalePvPrefetchAttention`，不叠加 bulk/rowmax，也不修改旧构建。用户已明确授权继续实现和远端测试，本轮沿用 CANNBot 分层性能分析、API 使用和同步审计流程。

## 依据与假设

2026-10-21 的当前版本 Cube 样本中，QK middle 完成窗口为 807 ticks，PV middle 为 1345.5 ticks；原版对应 985.5 和 1369.5。PV first 不含历史合并仍为 1352 ticks。这支持优先检查 PV 的数据路径，但不是 memory/MAD/FIX bound 证明，也不是整核阶段百分比。

原 PV 每个 K128 tile 搬两块 P 与一块 V 到 L1，再将 V 装入 L0B供两个 Q128 输出块的矩阵乘使用。本候选仅尝试将下一块 V 的 DMA 与当前计算重叠；现有代码本身已有异步重叠，因此净收益可能很小或退化，必须由同进程配对测试决定。

## 改动边界

- Tiling、核数、Q/K128、D128、Ky=2、L1/L0/UB 分配不变。公开 D64 输入仍由原预处理补齐到 native D128。
- 仅 dense 非因果（`sparseMode==0 && qk_triangle==0`）、非 upper-right、单 vcore、Ky=2、native D128、两字节 TYPE、M/K/N 块均为 128、Kx>0 的 descriptor 启用；其余按原路径执行。显式 N128 保证每个 K tile 只有一次 V 槽释放。
- 每个 descriptor 的 history diagonal 完整返回后预取 V0，不跨 descriptor 或 QK/PV 调用投机。
- 当前 V 装入 L0B 后，提前发出原有 L1B 释放事件；若 `i+1<Kx`，在原 Ky 矩阵乘循环之前搬运下一块 V 到另一 L1B 槽。
- 原 P 搬运与全部 MAD、FIX 输出、history/末尾归一化顺序保持；Vector 反量化、softmax、数值契约保持。
- 不增加 event ID、全流水屏障或缓存容量。精度要求仍为全部 padded native 输出精确一致，不用放宽门限换速度。

## 缓冲区与事件生命周期

| 数据 / expected 与 actual root | Producer → consumer | 同步与复用条件 |
|---|---|---|
| P 的 L1A ping/pong，起点 0 / 64 KiB | 原 GM→L1 两块 P → L1→L0A | 原 MTE1→MTE2 free、MTE2→MTE1 ready 和末尾释放不变 |
| V 的 L1B ping/pong，起点 128 / 192 KiB；每块实际 32 KiB | V GM→当前槽 → 8 条 L1→L0B 装载 | 每块 acquire 一次 `MTE1→MTE2, slot`，copy 后 publish `MTE2→MTE1, slot`，读前消费 ready；槽索引与指针统一使用同一 slot |
| 当前 V 的 L1B 释放 | 完整 L1→L0B → 后续该槽 GM→L1 | 释放 event 在 MTE1 完成该块读取后生效；后续 MAD 读的是 L0B，不再读 L1B |
| 下一块 V | 当前槽相反的 L1B → 下一轮 L0B | 先等另一槽 free，且只在 `i+1<Kx` 时发出；不以当前槽的 free 替代另一槽的 free |
| 当前 V 的 L0B ping/pong，起点 0 / 32 KiB | L1→L0B → 两个 Ky MAD | 原 M→MTE1 free 和 MTE1→M ready 均保留；下一块只写 L1，不提前覆盖当前 L0B |
| P 的 L0A 与输出 L0C | 原 P 装载 → 原 MAD/FIX | 全部指令、累加顺序及 unitFlag 保持；L1B 提前释放不表示 L0B/L0C 已可复用 |
| history diagonal | 原 GM FP32 数据 → diagonal MAD → PV 累加 | FP32 L1 缓冲位于 256～512 KiB，与 V L1B 分离；不在 diagonal 未落盘结果与 PV 之间插入完成屏障 |
| descriptor / QK-PV 交界 | 本 descriptor 的全部 V → 下一使用者 | 最后一块不预取越界数据；所有已加载 V 均消费并释放，ready 归零，free token 恢复，原 ping/pong toggle 不变 |

两条方向相反的 event 使用相同数字 slot，但属于不同方向的事件；L1B 的 0/1 与 L1A 的 2/3 区分。原 event 7 用于 diagonal / 输出相关同步，候选未更改。QK 同样复用 L1B 与事件 0/1，所以测试必须覆盖连续 descriptor 与交替调用，不能只验证单 tile。

## CANNBot 流程与能力边界

1. Tiling：保留已有内存布局和分块，验证实际 32 KiB V 块不越过 64 KiB 槽位，且不与 P 或 diagonal 区域重叠。
2. 卡间流水：无通信，跳过。
3. 核间流水：现有 Cube/AIV 的 GM slot、握手与 Run 调度保持；技能当前目录没有对应分析流程。本候选不声称优化跨核等待。
4. 核内流水：提出 V DMA 提前发射的单变量实验。没有真实流水仿真图，不将上一轮 raw ticks 当作硬件利用率。

`regenerate.py` 从本地固定 MindIE-SD `71098c20ff90838d1fd1ca0a40515c751a3869b0` 重建基线/候选，并扫描两者所有同族 C++ headers/source。保留两个 analyzer 的原始输出、stderr、工具哈希、命令及来源哈希。检查 QK、Run、Init、Preset/Clear、diagonal/FP32 helper 字节相同，PV 数学/输出调用 site 相同；Vector、mapper 等其他 kernel 文件在身份重命名后相同。

这些正则工具对 Laser raw intrinsic 的覆盖有限；findings 为空不能证明异步同步正确。实际生成方法的主机事件/地址模型用于检查事件生命周期、槽位、尾部和指令顺序，但不模拟真实执行时延或 CCE 调度，不能证明设备性能。

本轮本地验证共 **167 项通过**：构建/离线来源 18 项、基准与输出检查脚本 133 项、PV 补丁与 C++ 方法模型 16 项。方法模型执行实际生成的 PV 函数，覆盖 288 组入口槽位/长度/guard 组合、连续 descriptor、QK 交界和 upper-right 回退；基线与候选的 MAD/FIX/history/P-copy 序列一致。8 种故障注入（缺失等待/发布、错槽/地址/奇偶、末块越界和过早释放）均被检出。QK 交接和 FP32 diagonal 仅模拟共享缓冲区使用及事件状态，不执行它们的真实数学；这不是在设备上运行的精度测试。

最终来源审计为两个工具分别扫描基线/候选各 13 个文件：4 次返回码均为 0，findings 均为 0，但识别的 raw intrinsic event/operation 数也是 0，因此不把工具空报告当作同步证明。实际同步依据为缓冲区依赖审阅、方法模型和待执行的远端检查。

## 远端运行与验收

```bash
git pull origin master
bash scripts/run_mindie_laser_pv_prefetch.sh
```

默认 BF16 BNSD `[1,28,37741,128]`、设备 0、warmup 5、10 次/轮、9 轮。只构建新候选，复用已校验的原版 FP16 Laser、当前 Vector scale 与融合输入量化器。源码只读取本地固定 Git objects，不访问 GitHub。

主表同进程随机交错比较 `original_laser` / `vector_scale` / `vector_pv_prefetch`。full API 每次重新预处理；prepared/native 独立测量，profiler 在主计时之后另列，不相加、不混用。相对原版 FP16 的 dense 误差不同，不构成等精度或模型质量结论。

计时前保留当前 Vector-scale 自身数值参考、原址 scale 更新、全部 padded native 精确门限，并新增 K128/section 边界与多头映射、A/B/A 连续排队输入检查。公开预处理按 256 token 补齐，所以逻辑 K128 块数 1/3 并不意味着真实 Cube Kx=1/3；奇数和单迭代路径由主机方法模型单独覆盖，不冒充设备路径覆盖。

结果为 `results/mindie_laser_pv_prefetch.json`，日志为同名 `.log`。本地没有 CCE/NPU；首次编译、设备精确输出和性能验收仍需远端完成。
