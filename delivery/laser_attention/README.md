# Laser / 稀疏 Attention 优化进展

更新：2026-11-04。本文与父目录原 VC 交付分开维护。

**当前性能最佳为 E122 `sparse_delay`，核心 2.563 / 2.567 ms，比对应同条件 Laser 快 20.35% / 22.91%。全量数值质量尚未达标，视频生成效果尚未评审。** 本目录发布的是已测性能最佳的冻结源码与证据快照，状态为 `measured_contract_only`，不是质量发布版。

## 测试口径与结果

- 硬件：Ascend 910B2，24 AIC / 48 AIV；CANN 9.2.0。
- 输入：BF16、BHSD `[1, 1, 38139, 128]`、std=5；核心、预处理与完整调用分别记录。
- 两个主要测量均重新执行占用检查、原 clean/stable/determinism 三守卫、正确性检查和完整 capture；每轮 9 次配对 API trial 全部快于 E119。以下核心是命名 device task 时长，不是完整 API。

| 记录 | E122 核心 ms | 同轮 E119 核心 ms | 同轮 Laser 核心 ms | 相对 Laser 快 | 完整 device ms | 热态完整 API ms |
|---|---:|---:|---:|---:|---:|---:|
| primary D，seed0 | 2.563387 | 2.600607 | 3.218213 | 20.35% | 2.647387 | 2.719499 |
| primary E，seed0 | 2.567427 | 2.599307 | 3.330247 | 22.91% | 2.633407 | 2.709677 |
| 补充，seed2 | 2.561173 | 2.594767 | 3.183020 | 19.54% | 2.649427 | 2.711988 |

D/E 是预先规定的两次晋升重复，设备预处理分别 .062040 / .063333 ms。seed2 是补充输入，不混入 seed0 晋升，也不把最低单次读数替代主结果。来源见 [evidence/index.json](evidence/index.json)、各轮 [原始报告和 CSV](evidence/runs/)、[受保护交付回执](evidence/protected_delivery_receipt.json)。

**精度必须看完整 query：** 原 32 行抽样的 .383% / .792% 不能外推全输入。对全部 38139 query、全部 38139 key，E122 在已审计 seed0/2 上的相对 RMSE 为 **1.459117% / 1.499867%**，超过当前 1% 数值诊断目标；固定 Top8 在 std1 平坦输入也已知失效。通用输入质量和实际视频均未验收。参见 [全量审计 E123](evidence/full_query_quality_e123.json)。

## 去哪里看代码

**当前最佳的完整原生内核：[`source/vc-attention/ascend/laser_sparse_delay/`](source/vc-attention/ascend/laser_sparse_delay/)。** 这 14 个文件与 E122 实测构建清单逐字节相同。

| 内容 | 文件 |
|---|---|
| 原生入口与混合 AIC/AIV 调用 | [candidate_kernel.cpp](source/vc-attention/ascend/laser_sparse_delay/candidate_kernel.cpp) |
| QK 的 Cube、加载与交接 | [candidate_cube.h](source/vc-attention/ascend/laser_sparse_delay/candidate_cube.h) |
| AIV 主循环、双 K 块归约、在线选择与 PV | [candidate_vector.h](source/vc-attention/ascend/laser_sparse_delay/candidate_vector.h) |
| 块最大值、位置与 mask | [candidate_selector.h](source/vc-attention/ascend/laser_sparse_delay/candidate_selector.h) |
| window32 / resident Top8 状态 | [online_top8.h](source/vc-attention/ascend/laser_sparse_delay/online_top8.h) |
| FP32 中心化温度、softmax、half 权重 / FP32 vmla PV | [inline_pv.h](source/vc-attention/ascend/laser_sparse_delay/inline_pv.h) |
| Torch ABI、分配、元数据、来源诊断接口 | [candidate_torch.cpp](source/vc-attention/ascend/laser_sparse_delay/candidate_torch.cpp) |
| Python 适配与布局 | [mindie_laser_candidate_delay.py](source/vc-attention/scripts/mindie_laser_candidate_delay.py) |
| 构建驱动 | [build_mindie_laser_quant_preprocess.py](source/vc-attention/scripts/build_mindie_laser_quant_preprocess.py) |
| E122 来源 / 数值契约 | [checks.py](source/cake-toolkit/plugins/vc_attention_e122/checks.py) |
| 全 query 审计工具 | [audit.py](source/cake-toolkit/plugins/vc_attention_e123/audit.py) |

`source/vc-attention/ascend/` 其余目录是 E122 验证所用的历史参照或组件依赖，不代表新的最优方案。`source/cake-toolkit/` 提供相关评测代码；[source_origins.json](evidence/source_origins.json) 标注哪些文件属于 E122 冻结闭包，哪些是后续离线工具或静态导入支持。

## 当前算法

完整 half QK → 每个 K128 块保留一个 raw-half 最大值 → 每 32 块局部 Top8 与驻留 Top8 合并 → 对 raw Top8 减最大值后用 FP32 乘公共温度、softmax → half 权重与原 FP32 vmla PV → BF16 输出。

这是有显式数值与 tie 语义的稀疏近似。温度后移减少逐块缩放，同时避免部分 half 分数舍入；每块一个候选仍会丢失同块内强竞争项，不能声称与完整 dense attention 等价。显式 UB 声明 155520B，不是编译器私有空间或 native lowering 认证。

## 最近尝试、卡点与下一步

1. 已实施在线 Top8、融合 softmax/PV、双块归约、缓冲生命周期复用、延迟缩放，得到 E122 性能最佳。
2. 全量审计发现抽样漏掉坏行；原始输入一致性、加载前身份检查、逐行长尾与支撑集诊断已工具化。
3. E125 每块双候选已通过原生五阶段：同轮核心 **4.853987 ms**，E122 **2.595027 ms**，Laser **3.313480 ms**；全 query RMSE **.623448% / 1.459117% / .735184%**。E125 更低的平均误差伴随明显耗时回退，未晋升。其最差行误差14.28%也高于Laser6.79%，不能以全局 RMSE 决定视频质量。[性能证据](evidence/e125_negative_core.json)、[全量质量](evidence/full_query_quality_e125.json)。本包的原生发布源码仍以 E122 为准。
4. 直接双候选对约14.55亿个 half 元素做排除 mask；AIV vector busy 从1.616升到3.771ms，Cube MAC基本不变。原成本区间2.8–4.4ms被证伪，保留负结果并停止该直接实现的调参。
5. 下一方向是 KV 分桶与 Cube 选择性补算，仍未实现；路由、前缀、gather、DMA、merge、输出映射和同步必须计入实际成本。失败回执的原始产物身份收集也仍有待完善项。数学搜索、冻结预测、原语 / 来源 / PV 检查和离线复放继续由实战检验，不能用工具数量代替更快结果。

## 本地审阅与构建

从本目录执行，无需 NPU 的快照 / CSV 复核：

```bash
python3 tools/verify_snapshot.py
```

它验证本包文件 SHA256、实测原生源码身份，并复用冻结 CSV 解析器重新核对三轮记录。它不执行 NPU，不重算完整数值 oracle，也不代替硬件验收。大型 `.pt`、完整 raw trace tar、编译 `.so`、CANN 安装包与模型权重未打包；原回执中的绝对路径保留为历史来源，不能直接在另一机器原地回放。

在已配置的 **CANN 9.2.0 / torch 2.9.0 / torch_npu 2.9.0 / aarch64** 开发环境中，可单独构建原生最佳库：

```bash
# ASCEND_HOME_PATH 指向本机 CANN 9.2.0，先加载该安装的 set_env.sh。
python3 source/vc-attention/scripts/build_mindie_laser_quant_preprocess.py \
  --cann-root "$ASCEND_HOME_PATH" --soc Ascend910B2 \
  --operation sparse_delay --jobs 4 \
  --build-dir "$PWD/build/e122" \
  --output "$PWD/source/vc-attention/results/sparse_delay_cann920_build.json"
```

本发布快照已在本机 CANN 9.2.0 开发环境中按上述路径重新编译通过，14 个源文件 SHA256 与原实测清单一致；见 [publication_checks.json](evidence/publication_checks.json)。此编译检查不产生新的性能结果。

这只编译，不运行设备。原生 `torch.ops.vc_laser_sparse_delay.produce` 的输入是已按原适配器准备好的 contiguous half Q/K/V 与 packed lookup geometry，不能把任意原始 BF16 tensor 直接当作该 ABI 输入。完整原实验还依赖匹配的 Laser vendor、E119 raw 参照、PV 等构建以及重新绑定的 RunPlan / 卡守卫；应从源码重建并生成本机身份，不能把旧路径替换后冒充原回执。E122 适配器的 `VC_ATTENTION_INCUMBENT_MANIFEST` 指向 E119 `sparse_pair` raw 参照。

运行设备前必须重新核对占用并通过原 clean/stable/determinism 三守卫；物理 NPU0 禁用。本次发布仅复核与编译快照，没有新增性能测量，后续视频验证由同事协作评审。
