# MindIE Laser 的独立 ExpCast 实验

本实验以已复现的 MindIE Laser 为基线，只替换整块概率计算中的 `vexp`。目标是先量化这一局部修改的正确性与实际收益，再决定后续优化。当前已完成源码变换、CPU 数学检查、构造输入 oracle 和运行脚本；**ExpCast 的远端 CANN 编译、NPU 正确性及性能尚未验证**，不能提前称其更快。

固定上游为 [Ascend/MindIE-SD](https://github.com/Ascend/MindIE-SD/tree/71098c20ff90838d1fd1ca0a40515c751a3869b0)，提交 `71098c20ff90838d1fd1ca0a40515c751a3869b0`。默认比较 BF16、BNSD `[1, 28, 37741, 128]`，使用同一输入种子。已有 A3 / CANN 9.0.1 环境的用户实测如下，供后续复跑核对：

| 已复现版本 | full API / ms | prepared core / ms | dense FP32 相对 RMSE |
|---|---:|---:|---:|
| 环境已安装的 MindIE Laser | 86.3775 | 83.4513 | 0.190947% |
| 固定提交的本地源码 Laser | 86.4317 | 83.5724 | 0.190947% |

两者的输出抽样相对 RMSE 打印为 `0.000000%`。这说明该次抽样的一致性通过验收，**不证明完整输出逐位一致，也不证明安装包与源码构建的二进制相同**。上述 dense 参考覆盖 32 个 query 行和全部 37741 个有效 key，未计算整个长序列的完整 dense 输出。

在远端仓库目录依次执行：

```bash
git pull origin master
```

```bash
bash scripts/run_mindie_laser_expcast.sh
```

脚本默认构建或校验两个独立的源码变体：`baseline` 与 `expcast`；随后按独立进程组运行 installed、baseline、expcast。默认 warmup 5 次，每轮 10 次、共 3 轮同步 API wall 测量，并打印对照表。`full API` 包含输入转换、padding、native 调用和输出后处理；`prepared core` 复用预处理后的输入，仍包含输出转换、裁剪和 contiguous，**不是单个 device kernel 的计时**。各版本依次运行，速度比不属于同进程随机交错或成对试验。

主报告写入 `results/mindie_laser_expcast_compare.json`，终端记录在同名 `.log`。每个独立 worker 的报告、输出抽样和日志保存在 JSON 指向的 `artifact_dir`。准备、构建和 bridge 也分别保存 manifest 与日志；失败会保留已获得的证据及失败上下文。

实际修改位于 [概率变换 helper](ascend/laser_expcast/vc_laser_probability.h) 和 [精确源码补丁](scripts/mindie_laser_expcast_patch.py)。原 Laser 先将 BF16 输入转换为 FP16，Cube 的 QK 输出 score 为 FP16，Vector 中缩放和减去 online row max 后仍为 FP16。设这里的非正输入为 `x`，实验定义为：

```text
a = round_fp32(float32(x) * 11.541560173034668)
b = round_fp32(a + 119.6500015258789)       # FP32 表示的 119.65
c = round_to_nearest_even(clamp(b, 0, 120))
P = bitcast_fp16(uint16(c) << 7)
```

乘法、加法分别经过一次 FP32 舍入，未融合为 FMA；整数转换采用 RNE，恰好居中时取偶数。对于每一个整数编码 `c ∈ [0, 120]`，包含 E4M3 的 subnormal 编码，都有精确恒等式：

```text
bitcast_fp16(uint16(c) << 7) == decode_E4M3(c) / 256
```

因此无需将 FP8 张量交给 Cube，解码简化为整数左移和位重解释。结果 `P` 保持 FP16 且不超过 1，沿用原有 GM 交换、rowsum 和 PV 路径。实现借用 max reduction 已结束的 32 KiB UB scratch，最多按 8192 元素分块，随后原 rowsum 重建自身 scratch。

QK 和 PV 的 FP16 数据路径、历史最大值变化时的 `exp(old_max - new_max)`、online 更新、归一化和尾部 mask 都保留原实现。原 FP16 score 与段内 rowsum 路径也保留；分母代码读取新的 `P`，所以分母数值会随概率近似改变。V 平滑关闭，不做 V 重排，也没有引入 INT8 QK 或 FP8 Tensor Core/Cube 乘法。这是 Laser 上的局部 ExpCast 实验，不能视为完整 VC 论文量化流水线的复现。

Laser 原路径直接计算 FP16 exp，没有额外的 FP8 P 编码步骤可供合并；新路径则增加类型转换与整数编码。因此论文在量化流水线中的收益不能直接外推到本实验，是否有净收益须看正式版本的 API 测量，不能只按“去掉 exp”判断。

正确性验证有不同的职责：

- 未修改的源码 baseline 与 installed Laser 保留同算法抽样一致性门限 0.1%，长序列 dense 相对 RMSE 门限 1%。
- ExpCast 的 24 组构造输入 oracle 覆盖 FP16/BF16、D64/D128，以及 `(Q,K)=(17,129)、(256,256)、(33,385)、(65,513)、(1,1)、(17,4097)`。Q/K 仅一个特征非零，使 QK 为可精确表示的单乘；每个 128-key 块含零最大值；V 的非恒定通道采用 one-hot 编码，另外保留全零与常数 2 通道。这样可检出概率映射错误和 padding 未屏蔽的问题。K4097 用例跨越 4096-key section，并触发 helper 的两次 8192 元素 scratch 复用。
- 构造 oracle 显式模拟 FP16 QK、half scale、减最大值、ExpCast 解码和最终 half 输出转换；参考 sum/PV 用 FP32，并不声称逐指令模拟原 kernel 的 reduction 舍入。全部 query 行的非恒定通道采用 1% 实现门限，full/prepared 采用 0.1% 一致性门限，恒定通道另行检查。除 K=1 的退化情形外，测试还要求普通 softmax 明显不满足 ExpCast oracle，防止未修改的 kernel 误通过。
- ExpCast 对普通 dense FP32 的误差单独报告，包括小型构造输入和长序列随机输入的 32 行抽样。这里不套用同算法误差门限，也不据此声称模型质量通过；实际模型质量需要另行评估。

[CPU 数学测试](tests/test_mindie_laser_expcast_math.py) 检查编码/解码契约，[oracle 测试](tests/test_mindie_laser_expcast_validation.py) 包含拒绝普通 softmax、错误常量输出及未屏蔽 padding 的负例。源码补丁采用精确锚点，拒绝意外漂移，能恢复原字节；这些 CPU 检查不能代替远端编译与 NPU 执行。

baseline 保留原源码算法，manifest 中只有它可声明 `kernel_inverse_rename_equivalent=true`。ExpCast、可选 probe 各用独立 checkout、构建及安装目录，不能覆盖已有 baseline。各版本沿用唯一 native 名称 `aclnnVcLaserAttention` / `VcLaserAttention`，通过同一 bridge 接口调用，依靠独立进程隔离各自的库加载与缓存；本地源码 worker 不导入 `mindiesd`。运行时校验源码、bridge、host 库和打包 device binary 的哈希，检查 vendor 库归属以及新生成的成功 tiling marker。它们提供源码构建与 host dispatch 证据，单靠 host marker 不能完全证明实际执行了哪一个 device binary。

需要局部诊断时再显式启用：

```bash
LASER_PHASE_PROBE=1 bash scripts/run_mindie_laser_expcast.sh
```

这会额外构建 `baseline_probe` 与 `expcast_probe` 两个变体，增加两次构建任务；默认命令不构建它们。probe 只采样 core 0 的两个 AIV 子核，各取 head、steady、tail、head+tail 类别首次出现的首行。它只包围整块概率变换，不包括历史 alpha 指数、完整 softmax、P 的 GM 写回或 rowsum。

probe 使用该固定版本 Vector 中未使用的 `EVENT_ID7` 做局部 `PIPE_V → PIPE_S` 完成同步，保留原 Cube/Vector 握手、双缓冲事件和 tiling，不加入 `PIPE_ALL`。原 kernel 未写入的第一个返回张量在**诊断变体**中用作 1024 字节记录区；主 attention 输出仍按原接口返回，生产变体不包含探针代码。记录在计算循环结束后写回，解析器会拒绝缺失、损坏或不一致的记录。

计时采用 `AscendC::GetSystemCycle()`，它属于 `PIPE_S`，返回 **50 MHz 系统计数器**，不是 AIC/AIV 核心时钟周期；官方换算为 `ticks / 50` 微秒。接口与跨流水同步要求见 [CANN GetSystemCycle 官方文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0282.html)。报告分别保留 `probability_ticks` 和 `empty_bracket_ticks`，不将相减结果宣称为纯指令延迟。额外同步及记录状态会扰动流水重叠；报告同时给出 probe/生产版本的 API 时间比以观察扰动，不能把有限样本相加后换算成正常 kernel 的 softmax 百分比，或用 probe wall 宣称加速。

本实验已采用 CANNBot [ascendc-perf-optimize](build/cannbot-skills/ops/ascendc-perf-optimize/SKILL.md) 的分层分析方法：先审查既有 tiling、buffer 与同步，再采集局部流水证据。它是单卡计算算子，卡间通信步骤不适用；Cube/AIV 的核间依赖依然存在，因此本次保留 tiling 和全部核间握手。该技能当前核间优化参考尚未收录，不能据此把核间成本视为零；真实瓶颈及后续 tiling 修改仍等待远端 trace/profiling。当前 probe 提供单核概率区域的局部证据，并未完成整个流水的性能归因。

2026-10-15 本地检查：138 项 Laser 相关 CPU 测试通过；使用固定提交的真实 Git 对象准备 baseline、expcast、baseline_probe、expcast_probe，逐一重新核验来源并确认重复准备不改文件时间。Bash 编排测试覆盖隔离目录、离线复用、失败中止和可选探针。此次优先建立可区分算法的数值验收，再保留同一套 Laser 流水做性能对照；下一步以远端正式版本实测决定是否保留变换。以上没有替代 CANN 编译或真实 NPU 验证。

首次远端编译暴露了容量断言的 C++ 语义错误：原类的 `SIZE_FP16` 是普通对象成员，不能出现在 `static_assert` 常量表达式里。已改成 `sizeof(INPUT_T)`，并保留输入宽度和 scratch 容量检查，概率数值契约不变。增加普通 C++ 编译回归，确认旧片段失败、新片段通过；它使用声明桩，不替代 CANN 指令编译验证。

此修复可在旧实验目录上直接重跑同一条命令。准备阶段仅迁移与上一版完整源码及 manifest 均匹配的生成文件，手工修改仍会拒绝。上游 CMake 的源码复制和设备完成标记没有头文件依赖，因此构建前还会比较设备源码副本；发生变化时只清除该算子的复制/设备编译 `.done` 标记，让 make 更新副本并重编设备代码，保留 host 对象、CMake 缓存和其他算子产物。相关源码迁移、增量构建、数学与 C++ 检查共 74 项通过；实际 CCE 重编译仍由远端完成。

随后远端 CANN 9.0.1 在 `vmuls`、`vadds` 的第三个实参上报 `the 3rd parameter maybe need a type 'float'`。报错实参是 `constexpr float` 的 `GAIN`、`BIAS`；现改为调用处的 `static_cast<float>(GAIN)` 与 `static_cast<float>(BIAS)`，向 CCE intrinsic 提供明确的 `float` 值表达式。[官方接口原型](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/cceintrinsicapi/cceapi_0015.html) 的该参数为 `float`；const 左值导致此次拒绝的解释是根据诊断作出的推断，文档没有说明这一限制。常量数值、分开的 FP32 乘加、RNE 和概率解码均不变。这个修复针对实际诊断，不能由普通 C++ 声明桩证明 CCE 已兼容，仍须远端重编确认。旧容量断言版及其修正版的已知生成头文件均可经完整来源审计后升级，继续使用上面的两条命令，无需清空 build。
