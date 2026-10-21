# PV 内 P 预取：以实测 V 预取为基线

本次只测试将下一块 P 的 GM→L1 搬运提前，继承已实测的 V 预取调度。独立算子为 `VcLaserQuantVscalePvPPrefetchAttention`；主性能对照仍是固定源码复现的原版 FP16 MindIE Laser，同算法对照改为 `vector_pv_prefetch`。旧 V-only 候选和构建保留。本轮用户已明确授权实施，沿用 CANNBot 性能与同步审计流程。

## 依据、分块与改动范围

上一轮 V 预取相对 Vector scale 的 full/prepared/native 配对节省为 0.4356 / 0.4904 / 0.5541 ms，均为 9/9 更快；全部 padded native 输出精确一致。这支持沿 PV 数据路径继续做单变量实验，但不能证明 kernel 是 memory-bound，也不能保证同时预取 P/V 会更快。

保持原有核数、Q128/K128/D128、Ky=2 和 L1/L0/UB 布局。每轮 PV 读取两个 FP16 P128×128 块，共 64 KiB；两个 L1A 槽起点为 0 / 64 KiB。V 使用 128 / 192 KiB 的 L1B 槽，实际每块 32 KiB；FP32 diagonal scratch 从 256 KiB 开始。

P 预取仅在父 V 预取 guard 内且 Kx>1 时启用：dense noncausal、native D128、M/K/N 块均为 128、Ky=2、单 vcore、两字节 TYPE、非 upper-right。单块以及其他情况保持父版本路径。

1. 原 history helper 返回、V0 预取之后，获取当前 A 槽并搬入 P0 的两个行块，发布 ready。
2. 保留读当前 P 前的 ready wait。原循环内 P 的 acquire/copy/publish 仅在回退路径执行。
3. 在父版本 next-V 预取之后、当前 Ky MAD 循环之前，若 `i+1<Kx`，获取相反的 A 槽，搬入下一块 P 的两个行块并发布 ready。
4. **当前 A 槽的原释放位置不动**：在两个 Ky 行块完成 L1→L0A 读取之后才释放。下一 A 槽等待的是此前一轮已发出的释放事件，首轮使用 Preset token。
5. 最后一个 K tile 不额外预取。不跨 descriptor、QK/PV 或调用边界投机；保持 P 的地址/stride 参数、全部 MAD/FIX/history/normalize、Vector、QK、Run 和 ping/pong toggle。

无新增 buffer、event ID、全流水 barrier 或 intrinsic 签名；P 的两条 copy 逐项沿用父版本参数，实际 vcore=1 路径为 ndNum=1、nValue=128，不将它擅自改成两核形式。

## P 的生产与消费边界

Vector 的 `Run` 在当前轮的所有 section 处理结束后，用 `ffts_cross_core_sync(PIPE_MTE3, AIV2AICFLAGID 配置)` 发布；写回来自 `attention_score_short_double_line_one`。Cube 的 `Run` 明确令 `is_syn=true`，单轮、主循环及 drain 三条 PV 调用路径均在对应 `wait_flag_dev(AIV2AICFLAGID)` 后取当前轮 descriptor 并调用 `cube2_matmul_op`。候选不提前任何跨核 wait，也不访问下一轮的 P。

候选与父版本的 Vector、Cube Run、地址映射、Init/Preset/Clear 逐字节相同。P 的 GM root 仍为 mapper 给出的当前 `src[idx].left`，第二行块仍加原 `lineStride`。双缓冲 GM workspace 的轮次映射不变。源码对比和人工链路审查不是完整硬件内存模型证明，必须执行设备输出检查。

| 数据 / expected 与 actual root | Producer → consumer | 同步与复用条件 |
|---|---|---|
| 当前 descriptor 的 P GM；`src[idx].left` | Vector FP16 P 写回 → Cube GM→L1 | 保留当前轮 MTE3 跨核发布及 Cube AIV2AIC wait；不跨 descriptor 预取 |
| 当前 P 的 L1A；`ping_pong_flag_l1_a_` 对应槽 | 两次 P copy → 两个行块 L1→L0A | 读前消费 `MTE2→MTE1, slot+2` ready；写前获取该槽 `MTE1→MTE2, slot+2` free |
| 下一 P 的 L1A；`1-ping_pong_flag_l1_a_` 对应槽 | 下一块 P GM → 相反槽 → 下一轮 L0A | 地址与 event 均取同一 next slot；两个 copy 完成才 publish ready；当前槽仍由现有读取者持有 |
| P 的 A 槽释放 | 两行块的所有 L1→L0A 读取 → 后续复用 | 保留原循环末尾 `MTE1→MTE2` release；不将当前槽 token 当作下一槽 token |
| P 的 L0A | 原 L1→L0A → 原 MAD | M→MTE1 free 与 MTE1→M ready 保持；不提前覆盖当前 MAD 使用的 L0A |
| V 的 L1B/L0B | 父 V-only 预取 → 原 PV MAD | V 调度及 B 事件完整继承；A 的 event 2/3 与 B 的 0/1 区分 |
| 尾块与 descriptor/QK 交界 | 全部本轮 P/V → 后续使用者 | 不产生未消费的末块预取；free/ready 平衡、槽位奇偶保持；diagonal 的独立 L1 scratch 与共享 L0 生命周期保持 |

## CANNBot 流程与本地验证

1. Tiling：保持现有分块及内存分配，校验 P 的 64 KiB 与 V/diagonal 区域分离。
2. 卡间流水：无通信，跳过。
3. 核间流水：现有 Cube/Vector 交接不改；技能对应分析目录暂无内容，人工追踪 P 发布/消费链。
4. 核内流水：测试 P DMA 提前发射，保持矩阵计算顺序。没有新的生产阶段时间线，不将上一轮 raw ticks 解释为硬件利用率。

`regenerate.py` 从本地固定 Git objects 生成父 V-only 与新候选，扫描同族 C++ 文件，保留两个 analyzer 的 JSON、stderr、返回码、命令、工具与来源哈希。共用上次的审计协议，报告独立保存。若 regex 未识别 raw intrinsics，空 findings 不证明同步正确；人工数据流表和实际 PV 方法的主机事件/地址模型另行检查。

主机模型只验证指令/操作数序列与槽位生命周期；QK、diagonal 和硬件异步执行并非真实设备模拟。设备精度门限继续要求全部 padded native 输出 exact，并保留各自数值参考和连续输入检查。

本轮两工具各扫描父/候选 13 个文件，4 次退出码均为 0、findings 均为 0；识别的 raw event/operation 数同样为 0，不把空报告当作同步证明。源码比对确认两处 L1→L0 调用、四处 PV 数学/输出调用、原 A release 至方法末尾的代码相同；Vector、Run、mapper 及 QK/helper 同样保持。

内核补丁与 C++ 方法模型 19 项、离线来源/构建 19 项本地测试通过。模型覆盖 288 组入口槽位/长度/guard 组合、连续 descriptor、QK 交接、history/tail、空调用及 upper-right 回退；数学操作与 P-copy 各自的序列与父版本一致。针对 P 的缺失等待/发布、错槽/地址/ND/奇偶、尾部越界、过早释放和未发布 P 的故障注入均被检出。它们不是 NPU 精度或性能结果。

三方基准、旧候选兼容性及输出检查脚本合计 149 项唯一测试通过，本轮相关测试总计 **187 项通过**。远端保留原 38 个边界场景，再加 4 个 FP16/BF16 的 Q257、K385/K4097 定向用例；其相邻 Q128 行组偏好相反的 K128 块，CPU 解码后的概率质量检查确认这种差异，避免仅靠 V 花纹测试 P 搬运。

## 远端一键验证

```bash
git pull origin master
bash scripts/run_mindie_laser_pv_p_prefetch.sh
```

默认 BF16 BNSD `[1,28,37741,128]`、设备 0、warmup 5、10 次/轮、9 轮。仅编译新 P+V 候选，复用已验证的 V-only、原版 Laser 与输入量化器；源码从本地 Git objects 取得，不访问 GitHub。

比较 `original_laser` / `vector_pv_prefetch` / `vector_pv_p_prefetch`。full API 每次重新预处理，是主要指标；prepared/native 使用固定 payload，独立计时。profiler 在主计时之后另列。候选与 V-only 的精度检查覆盖全部 padded native 输出，原版与 INT8 的 dense 误差另报，不能视为等精度或模型质量验证。

报告为 `results/mindie_laser_pv_p_prefetch.json`，日志同名 `.log`。本地无 CCE/NPU，不声称已编译通过或获得性能提升，需远端验收。
