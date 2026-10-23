# 稳定 Laser 上的 FP16 ExpCast 实验

本实验从远端已跑通的 `vector_pv_prefetch` 派生，保留融合 INT8 Q/K 量化、Vector 反量化、V-only PV 预取和 FP16 PV，只给概率计算增加一个独立的 FP16 ExpCast 选项。原稳定入口仍为 `bash scripts/run_mindie_laser_stable.sh`。

在保留此前成功构建产物的远端仓库中执行：

```bash
git pull origin master
bash scripts/run_mindie_laser_fp16_expcast.sh
```

脚本读取稳定构建记录中的 CANN 路径和 SoC，只从本地 Git 对象准备、编译新的独立算子，不访问上游仓库。原版 Laser、稳定版 attention 和融合量化器复用已有产物。新算子使用独立的 `vc_laser_quant_vscale_pv_expcast` vendor、bridge、源码和安装目录；复跑会校验来源并复用有效缓存。缺少已有构建或校验失败时停止。

默认设备 0，BF16 输入、BNSD `[1,28,37741,128]`，warmup 5 次、每轮 10 次、9 轮。可通过 `NPU_DEVICE=1` 选择其他卡。验证通过后，在同一进程随机交错测量：

| 报告行 | QK | 概率计算 | V/PV |
| --- | --- | --- | --- |
| `original_laser` | 原版 FP16 | 原版 exp | FP16 |
| `vector_pv_prefetch` | 稳定版 INT8 | 原版 exp | FP16，V-only 预取 |
| `fp16_expcast_control` | 新 provider 的 INT8 | 关闭 ExpCast，原版 exp | 同稳定版 |
| `fp16_expcast` | 新 provider 的 INT8 | 直接 binary16 ExpCast | 同稳定版 |

新 provider 的关闭组用于观察重新编译及代码布局造成的变化，开启组用于测量近似映射的净影响。主基线仍是原版 Laser，同时报告与稳定版和关闭组的比较。full API 包含每次重新预处理；prepared 包含 native 和输出转换/裁剪；native 包含分配与分发，不能当作 kernel-only 时间。固定 INT8 对照共享同一份量化张量，full 则每次重新准备。各时间范围独立测量，不相减推导组件耗时。

报告为 `results/mindie_laser_fp16_expcast.json`，终端日志为同名 `.log`。默认不启动 profiler 或内部计时探针。

## 2026-10-23 远端结果

用户回传同进程随机交错 9 轮结果，BF16 BNSD `[1,28,37741,128]`：

| 版本 | full API ms | native API ms | dense RMSE % |
| --- | ---: | ---: | ---: |
| 原版 Laser | 85.5090 | 81.2776 | 0.190947 |
| 稳定 `vector_pv_prefetch` | 83.6454 | 79.1086 | 1.365174 |
| 新 provider 关闭 ExpCast | 83.5159 | 79.2287 | 1.365174 |
| 新 provider 开启 FP16 ExpCast | 123.1336 | 118.5415 | 2.282386 |

同 provider 的开启组 native 配对节省为 **-39.2932 ms，0/9 轮更快**；full 配对节省为 **-39.6128 ms，0/9 轮更快**。开启组 native 中位延迟比关闭组增加约 49.6%。当前适配没有性能收益，不替换稳定版本；误差也高于关闭组，这不是模型质量结论。原始报告在远端，以上数字来自用户回传，未在本机重测。

源码显示原 bulk FP16 `vexp` + Vector 屏障被替换为每个最多 8192 元素 chunk 的 7 道向量操作和 7 个屏障，包含 FP16/FP32/INT16 转换与 scratch 访问。额外执行、同步和 UB 访问是待验证的原因，API 时间不能确认各自占比或寄存器 spill。下一步用已有产物采集整核计数器，仍保留相同 provider 的开关对照：

```bash
LASER_MSOPPROF_EXPCAST=1 bash scripts/run_mindie_laser_msopprof.sh
cat results/mindie_laser_msopprof_summary.txt
```

采集不重新编译或加入内部探针；检查计数器与原版、稳定版、关闭组的差异，再决定是否值得继续改写该映射。

## 这次如何生成 FP16 概率

[VC 论文](https://arxiv.org/html/2609.15810v1#S3.SS3)的 ExpCast 使用仿射变换、整数舍入和浮点位编码近似指数。这里借用这一思路，适配 binary16，**不是论文 E4M3 编码的逐字复现**，也不需要 FP8 矩阵乘。

设 `x` 为原 Laser 完成减最大值后的 FP16 非正 score，映射为：

```text
t = round_fp32(float32(x) * 1477.3197021484375)
t = round_fp32(t + 27603.19921875)
c = RNE(clamp(t, 0, 27648))
P = round_fp16(bitcast_fp16(int16(c)) * 2^-12)
```

乘、加分别执行，未融合为 FMA。斜率为 FP32 的 `1024*log2(e)`，bias 为 FP32 的 `1024*(15+12-0.35/8)`；归一化的偏置修正由论文 FP8 公式迁移，并非另行拟合得到的最优 FP16 参数。额外 `+12` 先在放大后的范围编码，再乘 `2^-12`，以保留最终 FP16 的次正规概率。直接使用 bias 15 会过早截断这部分尾部。

编码位本身已经是 FP16，无需 E4M3 解码或 FP8 Cube 支持。`P(0)=0.97802734375`，分子 PV 与分母沿用同一份新 P。在线最大值更新、历史重缩放的原 exp、rowsum 和输出归一化代码保留。近似函数不满足精确指数的乘法恒等式，改变在线分段或最大值历史可能改变结果，不能把全局近似 softmax 当作任意输入的严格在线 oracle。

实现位于 [FP16 helper](ascend/laser_expcast_fp16/vc_laser_probability.h)，数值契约位于 [CPU 数学参考](scripts/mindie_laser_fp16_expcast_math.py)。复用已完成 max reduction 的 32 KiB UB scratch，下一步 rowsum 重新建立其 scratch；不增加 UB 分配或核间事件。同步检查和覆盖限制保存在 [审计记录](reports/ascend-fp16-expcast-audit/README.md)。

## 如何判断可用及是否更快

- 新 provider 关闭 ExpCast 时，要求与稳定版全部 padded native 输出精确一致。
- 开启时，FP16/BF16、D64/D128、零输入、尾块和 K4096 section 边界使用同最大值构造输入，对照新的 decoded reference。参考使用 FP32 sum/PV，不逐指令模拟原 reduction；非恒定通道采用 1% 实现 RMSE 门限。
- 信号构造要求原 exp 不能误过 ExpCast 门限；另以放大 V 的尾部构造检验小于 `2^-14` 的概率，避免全局误差掩盖次正规数被清零的问题。
- 最大值发生变化的输入单列有限性、full/prepared 一致性和 dense 误差诊断，不冒充严格在线等价验证。长序列 dense FP32 参考抽样 32 个 query 行、保留全部 key，单独报告算法误差，不作模型质量结论。

本地覆盖数学、源码生成、bridge、普通 C++ 声明兼容及脚本编排；**不能代替远端 CANN 编译与 NPU 执行**。尤其需要验证 FP16 `vmuls` 的次正规输出行为。新 helper 增加了类型转换、编码和缩放，原版 Laser 的单条 `vexp` 本来就有效率，去掉它不保证净加速。是否保留该版本，以正确性通过后的完整 API 成对测量为准。

2026-10-23 本地验证共 93 项通过，包含新实现的数学/源码/benchmark 检查，以及稳定入口的编排回归。完整生成源码族的 CANNBot 扫描已保存；该工具对 CCE raw intrinsic 的自动覆盖不足，另有手工 buffer 生命周期与同步核对，详见审计记录。
