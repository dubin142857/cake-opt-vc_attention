# SageAttention 主体 + VC 两项改动

当前包版本为 `2.2.0+vc22`，native kernel version 19。新候选在 vc16 上缩短完整 K 块的
整数 max 归约，并独立测试最大值不变时跳过历史输出的恒等缩放。FP8 编码、分母和两级
FP32 累加保留；本地交叉编译及 helper 检查通过，默认仍为 vc15。
H200 最新回传：`vc22_both` 完整 API **25.6150 ms**，对照 vc16 为 27.1694 ms、
原版 Sage 为 26.5558 ms，分别提速 **1.061× / 1.037×**。重复 CUPTI 三轮均确认收益方向，
全输出实现比较通过（打印 RMSE 0.000000%，不代表 dense FP32 零误差）。
后续实验以 `vc22_both` 为基线；22 ms 还需减少 3.615 ms。使用它须显式设置
`kernel_mode="vc22_both", v_smooth=False`，详见 [vc22 回传记录](results/h200_sage_vc22_reported_summary.json)。
执行 `bash scripts/validate_work_h200.sh`，看后面的 **Work-reduction candidates** 表。
说明和检查范围见 [vc22 说明](SAGE_VC22.md)。

上一轮 `2.2.0+vc21`（native version 18）的 D128 候选回到 vc16 的逐块顺序，
把每线程 28/64 个 FP32 历史输出值放入 14 KiB/CTA 共享内存，配合两个 N64 PV 片段，
以减少长期占用的寄存器。H200 低上限候选达到 128 寄存器、零 local、预测 4 块驻留；
目标 shape 的实现一致性门槛通过，但完整 API 仍慢于基线。
默认 `optimized` 仍为 vc15。见 [vc21 说明](SAGE_VC21.md)。
最新 H200 同场 vc16 ExpCast 核心 25.4766 ms、完整 API 27.2441 ms；原版 Sage 分别为 25.0419/26.5753 ms。
vc21 最快候选完整 API 28.1399 ms，比 vc16 慢 3.29%；重复 CUPTI 也确认退步。
共享存储改善了有 local 分配的旧低寄存器候选，但在 168 上限的同类对照中增加成本；
资源改善没有转化为相对 vc16 的净收益。vc17 至 vc21 均未改善完整 API。
该轮保留了 vc16 的 168 寄存器、N128 PV 作为实验基线；vc22 回传后的新基线见上文。
22 ms 尚未达到，当前无需重跑 vc21。
回传记录见 [vc21 结果](results/h200_sage_vc21_reported_summary.json)。
`bash scripts/validate_history_h200.sh` 保留用于离线复现，不要求 nvdisasm/ncu；
全输出精度、编译资源、核心/完整 API 和重复 CUPTI 分开报告。
[vc20 说明](SAGE_VC20.md) 保留连续提交实验的负结果。
[vc19 说明](SAGE_VC19.md) 记录先前双缓冲实验的负结果。
[vc18 说明](SAGE_VC18.md) 和 [vc17 说明](SAGE_VC17.md) 保留既往实验。下文保留历史开发记录。

`sageattention-local==2.2.0+vc9` 在已经同场测到 25.8055 ms 的本地 Sage 源码上，
新增独立 `sageattn_vc`。原版 `sageattn` 的数学与内核保留。
**25.8055 ms 是用户提供的原版源码构建结果，不是本次新入口的速度。**

这是 **Sage+VC 混合实验**：QK 保留 Sage per-thread INT8；新增 V 聚类/块平滑和 ExpCast。
没有改为完整 VC 的 FP8 QK/Hadamard，也尚未接入论文的去噪步数调度。

vc9 优化每次调用的聚类预处理。此前 H200 的 `cluster_sort` 约为 5.55 ms，
本轮将每 2048 行重复执行的平方、距离、argmin、scatter 等 ATen 操作改成 CUDA kernel：

- 每次加载一行 V，在寄存器中计算它到所有中心的距离，同时累积块内统计；再归约各块统计更新中心。
- 保留 FP32 linspace 初始化、两轮 Lloyd、空簇中心保留和组内稳定排序。
- 最后一轮只生成标签，省掉返回标签后不再使用的中心更新；每次调用仍重新聚类。
- 默认 `grouping_mode="fused"`，最多 8 个有效中心；更多中心走旧路径。
  `grouping_mode="legacy"` 明确选择 vc8 的 ATen 实现，attention 主体与量化配置不变。

FP32 求和顺序变化可能改变边界附近的分组，从而改变 QK/V 量化数据。
`--compare-grouping` 同时报告标签差异、完整 API 耗时及双方的 dense FP32 误差，
不把不同分组的输出套用“相同量化数据”的 0.1% 实现门槛。
当前 attention 内核保持 vc8 的实现：

- 在同一个 attention kernel 内将 PV 的 128 个输出通道分成两个 N64 WGMMA fragment，
  每次只保存一半临时结果。每个输出元素仍执行原来的四个 K32 累加。
- 加载一对 FP16 均值后完成两行的 FMA 补偿和 FP32 历史累加，缩短均值中间值的存活时间。
- 仅新 D128/channel 平滑入口设置 `__maxnreg__(168)`；其他配置及历史内核不设置该限制。
  需要 CUDA Toolkit >=12.4。寄存器限定语义见
  [NVIDIA CUDA 文档](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#maximum-number-of-registers-per-thread)。
- 最后一个 PV fragment 完成后才允许下一块 V 的 TMA 覆盖共享内存。
  量化、ExpCast、分母、均值补偿公式、共享 V scale 和两级 FP32 累加均保留。

这会增加 WGMMA 发射/等待次数，以较小的临时缓冲换取更高的驻留并发。
**编译资源改善不等于已确认提速，必须看同场 H200 对照。** 聚类预处理仍每次执行。

保留四种主内核模式，用相同安装包、相同量化数据作同场比较：

| kernel_mode | 概率路径 | V 恢复 |
|---|---|---|
| `optimized`（默认） | vc7 ExpCast | D128/channel 使用两个 N64 PV fragment + 配对均值恢复 |
| `vc7` | 与 optimized 相同 | 上一版完整 N128 PV 临时缓冲 |
| `vc6` | 上一版 ExpCast | 共享通道 scale 在输出末尾恢复 |
| `vc4` | 更早的概率实现 | 即使 scale 共享，也逐块恢复，作历史参考 |

`optimized` 和 `vc7` 在关闭 V 平滑、D64 或 block V scale 时使用同一个编译内核。
新路径保留 Sage 的两级 FP32 PV 累加，不把完整历史送回 FP8 WGMMA 累加器。
默认通道尺度对应 [VC 论文附录 B 的 8-bit 恢复方式](https://arxiv.org/html/2609.15810v1)，
QK 仍是 Sage INT8，**不是完整 VC 论文复现**。

`v_scale_mode="block"` 保留旧的逐块量化；与 channel 的 V 字节不同。
用 `v_scale_mode="block", kernel_mode="vc6"` 对照此前 vc5，
`v_scale_mode="block", kernel_mode="vc4"` 对照此前 vc4；均由当前工具链重新编译。
切换 `kernel_mode` 不改变准备好的 Q/K/V；切换 `v_scale_mode` 会改变 V 量化。

`sage_vc_build_info()` 中 `sage_vc_kernel_version=6`、`sage_vc_preprocess_version=1`、
包版本 `2.2.0+vc9` 表示新扩展已加载。

## 远端编译和测试

在 H200 的仓库根目录：

```bash
git pull --ff-only origin master
bash scripts/build_sage_local.sh
```

只更新独立包 `sageattention-local`，不用重新安装主包 `vc-attention`，
不覆盖已有 `sageattention`。需要 nvcc >=12.4、PyTorch >=2.4、Triton >=3.0。
新增 `_vc_sm90` 原生扩展，编译目标 `sm_90a`；旧安装需重新编译。

脚本先测原版，再运行 `validate_sage_vc.py`：

- V 编码、均值、尺度、尾块和稳定聚类检查；ExpCast 编码与独立 FP32-FMA 参考逐字节比较。
- 24 组聚类用独立 FP64 Lloyd 参考检查，并验证空簇、稳定 tie 和旧路径的已知分组。
- FP16/BF16、D64/128、Q/K 边界长度、四种开关及两种 V scale 粒度共 144 组 attention 对拍。
- 量化数据固定后，对比 CUDA 与独立 Python 参考，relative RMSE <=1% 是实现回归门槛。
- 双关闭入口与原版 Sage 比较，relative RMSE <=0.1%；另有非零均值已知答案和 HND/NHD 布局检查。
- 144 组还会在相同量化数据上对照 optimized/vc4、optimized/vc6 和 optimized/vc7，relative RMSE <=0.1%。
  另外记录是否逐位一致。这是实现回归门槛，不是模型质量门槛。
- FP32 dense 误差单独记录，不把上述实现门槛当成模型质量结论。测试失败即中止。

可以单独重跑：

```bash
python3 scripts/validate_sage_vc.py --output results/h200_sage_vc_validation.json
```

先测 1024 的完整误差和速度，再测你的 `[1,37741,28,128]`：

```bash
python3 scripts/benchmark_cuda.py --backends sage_local sage_vc_control sage_vc_expcast sage_vc_smooth sage_vc --baseline sage_local --queries 1024 --tokens 1024 --heads 2 --dim 128 --check-queries 0 --output results/h200_sage_vc_1024.json
bash scripts/benchmark_37741.sh --backends flash3 sage sage_local sage_vc_control sage_vc_expcast sage_vc_smooth sage_vc --baseline sage_local --output results/h200_sage_vc_37741.json
```

本轮先比较旧／新聚类和完整 API，不需要 `ncu`：

```bash
python3 scripts/profile_sage_vc.py --compare-grouping --trials 5 --repeats 10 --trace-dir results/h200_sage_vc9_traces --output results/h200_sage_vc9_profile.json
```

默认测试 BSHD `[1,37741,28,128]`，完整计算所有行，dense FP32 抽查 128 个 query、使用全部 keys。
`Grouping comparison` 的 `old/new > 1` 表示新路径更快，完整 API 每次重新聚类；
标签差异和 accuracy 使用同一次记录的分组，计时外计算。`--check-queries 0` 可检查全部输出行。
历史主内核仍可用 `--compare-kernels --kernel-baseline vc7` 做同量化数据对照。

| backend | 入口 | ExpCast | V 聚类 + 块平滑 |
|---|---|---|---|
| `sage_local` | 原版独立源码构建 | 关 | 关 |
| `sage_vc_control` | 新主体的控制组 | 关 | 关 |
| `sage_vc_expcast` | 新主体 | 开 | 关 |
| `sage_vc_smooth` | 新主体 | 关 | 开 |
| `sage_vc` | 新主体 | 开 | 开 |

时间包含每次 API 内部的聚类、K/V 重排、量化及融合 attention。
本版每次重新聚类，没有布局缓存；不能把预处理移出计时再称为端到端加速。
长序列误差默认抽样 128 个 query、使用所有 key；速度始终计算完整序列。
`speedup = sage_local 毫秒 / 当前行毫秒`，大于 1 才是快于源码 Sage 基线。

## Python 导入

```python
from sageattention_local import sageattn_vc

# q/k/v: [1, 37741, 28, 128]，CUDA BF16 或 FP16
out = sageattn_vc(q, k, v, tensor_layout="NHD", expcast=True, v_smooth=True)

# 相同量化方式，执行上一版 vc6 的概率路径
previous = sageattn_vc(q, k, v, tensor_layout="NHD", kernel_mode="vc6")

# 旧 vc5 的逐块量化 + optimized 内核
old_quantization = sageattn_vc(q, k, v, tensor_layout="NHD", v_scale_mode="block", kernel_mode="vc6")

# [batch, heads, tokens, dim] 则用 tensor_layout="HND"
```

当前支持 Hopper、D64/128、相同 Q/K/V 头数、dense noncausal 前向推理，
可用不同的 Q/K 长度。无反向、LSE、GQA、causal 或其他 mask。
只支持默认 CUDA stream 下 eager 执行：保留的 Sage 全序列 V 预处理使用默认 stream，
因此公开入口明确拒绝其他 stream。未注册 torch.compile/CUDA Graph 支持。
不支持的配置报错，无 CPU、SDPA 或原版 Sage 静默回退。

## 内核里改了什么

1. V 平滑开启时，每个 head 独立按 V 聚类；确定性等间距初始化，默认 8 类、2 次 Lloyd 迭代。
   C++ 调度 PyTorch CUDA 聚类运算，按标签稳定排序，同序重排 K/V，Q 不重排。
   随后仍用原版 Triton 做 K 去均值和 QK INT8 量化。
2. 新 CUDA 预处理先按块计算 FP32 均值及残差峰值，再归约得到每个 batch/head/channel
   在所有块上的最大残差幅值。共享尺度为 `amax/448`；所有块残差均为零时取 1，
   并以 `max(abs(block_mean))/65504` 为下界，避免 FP16 `mean/scale` 溢出。
   最后重新读取 V，融合残差量化与均值存储。三个 GPU pass 不写完整 FP32 残差张量，
   但比旧单 pass 预处理增加一次 V 读取和统计归约，full API 会计入这些成本。
   `vs` 为 `[B,H,D]`，`means` 仍为 `[B,H,ceil(N/128),D]`。
   均值不含补齐 token，补齐残差为 0，FP8 存储沿用 Sage 的转置及 token 排列。
3. Sage TMA、INT8 QK WGMMA、FP8 PV WGMMA 和两级 FP32 累加保留。
   默认每块执行 `history += FMA(rowmass, mean/channel_scale, PV_residual)`，
   随在线最大值调整历史分子/分母；末尾归一化后统一乘 `channel_scale`。
   旧 block 模式仍逐块恢复尺度。共享尺度上的 vc4 模式同样逐块恢复，作为同输入执行对照。
4. ExpCast 直接把减去在线 rowmax 的自然对数分数编码为 E4M3 字节：
   `RNE(clamp(FMA(x, 8*log2(e), 119.65), 0, 120))`。
   PV、分母和均值补偿共同使用解码后的权重；最大编码 120 对应 256，公共尺度在归一化抵消。
   关闭 ExpCast 时保留 Sage 的 `2^8.807` Softmax 权重尺度、普通 E4M3 P 舍入和未舍入的分母。

两项功能在主 attention CUDA 内核内部执行，没有外置完整 P 矩阵。
聚类、重排、QK/V 量化仍是预处理，整个 Python API 会调用多个 GPU kernel。
启用两项不能保证比 Sage 更快，也不能保证随机输入精度提高；四组实验用于测量实际影响。

已有 vc3 安装可以直接运行分阶段诊断，不用重新编译：
`python3 scripts/profile_sage_vc.py`。默认就是 `[1,37741,28,128]` 场景，
同时比较端到端、各预处理阶段和固定量化输入的主体耗时，见 [SAGE_VC_PROFILE.md](SAGE_VC_PROFILE.md)。

## 来源和本地验证范围

原始 Sage v2.2.0 commit `eb615cf6cf4d221338033340ee2de1c37fbdba4a`，Apache-2.0。
`csrc/vc/vc_sm90.cu` 是明确标注的衍生内核；原版文件与实验文件的来源/哈希分别记录于
`third_party/sageattention_local/UPSTREAM.json`，许可证和 NOTICE 保留。

本地 Windows / RTX 3060 可交叉编译 sm_90a，并执行 V 预处理与 ExpCast 辅助检查，
不能执行 Hopper WGMMA attention。完整 Linux CUDA 13 编译、H200 attention 数值和性能
仍需上面的远端脚本；本地检查不替代这些结果。

vc7 历史记录：本地 169 项测试通过，包括原有预处理、独立参考、脚本、来源检查，以及新旧
概率路径的逐位比较。64 块连续在线历史、极小/极大历史分母、极短尾块、所有 E4M3
非负概率字节对、RNE 边界相邻浮点数均已覆盖；原先保存的 16 组 helper 结果仍逐位一致。
Compute Sanitizer memcheck 为 0 errors。
56 种 Hopper 配置交叉编译、链接成功，无 spill，TMA/INT8 IGMMA/FP8 QGMMA 保留。
CUDA 12.4 的 D128/BF16 下，单 ExpCast 寄存器 167→168，全开 209→208；
没有明显降低寄存器占用。减少的截断/移位指令是否带来净收益，仍需 H200 CUDA 13 实测。
检查记录见 [results/sage_vc7_local_validation.json](results/sage_vc7_local_validation.json)。
该记录不包含 Hopper 执行；vc8 的本地范围和远端验证要求如下。

## vc8 本地验证与远端重点

本地 Windows / CUDA 12.4 / RTX 3060 能执行真实 CUDA 均值恢复及预处理 helper，
可以交叉编译 SM90a，但不能执行 Hopper WGMMA attention。
本地 182 项测试通过，Compute Sanitizer memcheck 为 0 errors。60 个 Hopper 配置编译链接成功，
旧版 56 个内核的 SASS 指令流与 vc7 记录相同。
新旧 helper 在极端 FP16 均值、尾块及 64 个连续在线累加块上相同；完整验证记录见
[results/sage_vc8_local_validation.json](results/sage_vc8_local_validation.json)。

本地 D128/BF16 平滑及全开寄存器均为 168，vc7 对照分别为 208、208。
平滑没有 spill；全开有 8 字节/线程 spill，SASS 中只在最后输出阶段出现，主循环没有。
这是 CUDA 12.4 的编译结果，CUDA 13 的寄存器、spill 和速度需要远端确认。

`--compare-kernels --kernel-baseline vc7` 比较同一份量化数据的全部输出，
relative RMSE <=0.1% 是实现回归门槛。`--trace-dir` 现在直接输出寄存器数、共享内存、
线程块大小和估计 occupancy；不需要手工解析 trace。occupancy 是估计值，不是利用率计数器。
先看新旧 core 是否提速，再看完整 API；vc8 没有减少聚类的数百次 kernel 下发。

profiling 开头还会打印 `compiled_d128_channel_resources`：通过 CUDA runtime 读取
当前安装二进制的寄存器、每线程 local memory、共享内存和预测驻留块数。
它覆盖当前输出 dtype 的 vc7/optimized 平滑及全开四个入口，不执行 attention，不计入计时。
`local_bytes_per_thread` 是静态局部存储量（含 stack/spill），不是动态流量。
此查询能区分远端 CUDA 13 是否生成了更多 spill，无需 ncu 或额外编译日志。

## vc9 本地验证范围

本地 RTX 3060 上 234 项测试通过，包含 50 项原生聚类测试和独立 FP64 Lloyd 对拍；
覆盖尾块、多个 batch/head、FP16/BF16/FP32、空簇、稳定排序、非默认 stream 和非法输入。
Compute Sanitizer memcheck 为 0 错误。60 个 Hopper attention 变体均编译、链接成功，
SASS 指令流全部与 vc8 相同。
新增聚类 kernel 的定向 racecheck 覆盖上述 50 项测试，0 错误、0 警告。

同一 BHND `[1,28,37741,128]` 的本地聚类含排序 smoke 测量约为旧版 34.4 ms、新版 6.27 ms。
这是 RTX 3060 的局部结果，按先旧后新顺序采样，不代表 H200 或完整 API 加速。
该随机输入中 1,056,748 个 token 有 11 个标签不同；量化后 attention 的误差需远端检查。
完整记录见 [results/sage_vc9_local_validation.json](results/sage_vc9_local_validation.json)。
远端 `--compare-grouping` 会交错新旧计时，同时检查 dense FP32 误差。

## vc9 H200 回传结果（2026-09-29）

用户回传的同场五轮聚类对照：

| 范围 | legacy | fused | old/new |
|---|---:|---:|---:|
| 聚类含稳定排序 | 5.4201 ms | 1.0217 ms | 5.305x |
| 仅平滑完整 API | 36.7985 ms | 33.2530 ms | 1.107x |
| 全开完整 API | 40.8505 ms | 36.4385 ms | 1.121x |

全开完整 API 节省 4.4120 ms，与聚类节省的 4.3984 ms 接近，支持收益来自聚类融合的判断。
该次分组检查的标签、稳定排列完全一致；抽查 128 个 query、使用全部 keys，
仅平滑和全开的 FP32 相对 RMSE 分别为 3.833999%、4.205950%，新旧相同。
这不等同于确认全部 attention 输出逐位一致或模型质量达标。

主计时中 Sage 完整 API 为 26.1890 ms、全开为 36.9610 ms，仍有约 10.77 ms 差距。
平滑/全开的完整调用在 trace 中均为 27 次 kernel 下发，此前 vc8 记录约 463 次。
固定量化输入下打开 ExpCast 仍额外增加约 3.68–3.75 ms，后续重点是主内核及 V 均值恢复。
trace 的 19% occupancy 是估计值，不能解读为 GPU 只有 19% 利用率。

记录见 [回传摘要](results/h200_sage_vc9_reported_summary.json)。该文件是控制台数值摘录，
不是远端原始 JSON；不同计时区间不可相加，完整 144 项远端验证结果未包含在此次回传中。

## vc10：只针对 ExpCast、关闭 V 平滑的主内核

版本 `2.2.0+vc10` / native kernel version 7。`kernel_mode="vc9"` 保留上一版；
`optimized` 只为 `expcast=True, v_smooth=False` 选择新内核。控制组、V 平滑、全开和
预处理复用 vc9 的内核。此次不改变 Q/K/V 量化、ExpCast FP32 FMA/RNE 字节和 PV 精度。

- 每个线程的每行 32 个 FP8 权重分成两条独立 FP32 累加链，先求本块的和，再加到历史分母。
  编码 0..120 的权重都是 `2^-9` 的整数倍，最大 256；这 32 项的和可由 FP32 精确表示。
  历史分母的加法顺序会变化，不承诺最终输出逐位一致。
- P 编码/解码及块内求和仍在 PV 前完成，历史输出缩放与 PV 异步计算重叠。
  PV 使用独立临时累加器，保留两级 FP32 累加；历史缩放明确舍入，编译器的 FMA 合并可能
  与旧版不同。完整输出仍按 <=0.1% relative RMSE 做实现回归检查。
- PV 提交后不会读取它的 P 寄存器。NVIDIA PTX 的 `wgmma.wait_group` 规则也限制
  输入 A 寄存器的访问，不能简单把 P 解码移到未完成的 PV 后面。

参考用户提供的 [BBuf CUDA 优化案例](https://github.com/BBuf/how-to-optim-algorithm-in-cuda)
采用局部测量、指令检查、同输入整核对照的工作方式，没有替换成案例中的独立 Softmax。
同步依据：[NVIDIA PTX WGMMA wait](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)。

本地 CUDA 12.4 的 64 个 Hopper 变体交叉编译通过；原有 60 个变体的 SASS 指令流相同。
新 D128 为 168 registers/thread，无 spill；D64 为 128，无 spill。局部 helper 检查包含
全部概率字节、尾块、不同 lane 的 QK scale，以及 300 个连续在线块。
H200 完整 attention 的正确性及提速仍需远端验证，不能由编译资源推出已经加速。

本地 RTX 3060 的独立概率 helper 测量约为旧版 5.03 ms、新版 5.08 ms，未显示提速。
该测量没有 QK/PV，不能测量本次的 PV 重叠收益；也不能把它当作 H200 性能结果。
远端优先跑 [SAGE_VC_PROFILE.md](SAGE_VC_PROFILE.md) 的 `--expcast-only` 同场对照。

本地共 246 项测试通过；85 项概率/累加 helper 的 Compute Sanitizer memcheck 为 0 errors。
完整记录见 [本地验证](results/sage_vc10_local_validation.json) 和
[局部微基准](results/sage_vc10_local_probability_micro.json)。这不包含 Hopper attention 执行。

## vc11：ExpCast 编码和解码交错调度

版本 `2.2.0+vc11` / native kernel version 8，默认 `optimized` 在只开 ExpCast 时
使用新的两组交错调度。`kernel_mode="vc10"` 保留上一版，`vc9/vc7/vc6/vc4` 继续可用。
V 平滑路径、控制组、所有 Q/K/V 量化、FP8 P 字节、FP32 块内累加顺序和历史缩放不变。
整数求和候选在 H200 上明显更慢，没有接入。

新实现先准备相邻两个 K fragment 的编码，发出 packed FP8 解码后再消费 FP32 值，
以改变独立指令的安排。PV 仍使用独立 FP32 临时缓冲；vc10 的 WGMMA 等待边界保持不变。
64 个历史 kernel 保留；新增四个 D64/D128 × FP16/BF16 specialization。

本地 CUDA helper 的新旧所有输出逐位一致，并覆盖 300 个块的连续历史。
完整 Hopper attention 只能交叉编译，不能在本地 3060 上执行；尚无 H200 提速结论。
升级和同输入验证命令见 [性能诊断](SAGE_VC_PROFILE.md#vc11两组编码与解码交错保留-vc10-同输入对照)。

## vc12：合并线性变换，保留量化分母

版本 `2.2.0+vc12` / native kernel version 9。默认 `optimized` 在只开 ExpCast 时
复用 FP32 gain/bias，把逐分数的缩放、减最大值和 ExpCast 编码变换合并成一次 FMA。
相邻两组交错调度、量化 P 构造分母和两级 FP32 PV 累加仍保留。

这次改变了 FP32 计算顺序，不能声称与 vc11 的 P 字节完全相同。
只有每个 warp 的行最大值处于 [-128,128] 且系数范围正常时走新路径；
其他范围在同一设备 kernel 内执行 vc11。V 平滑路径和控制内核没有变化。
`kernel_mode="vc11"` 可以完整回到上一版；其他历史模式继续保留。

本地 229,376 个编码样本中有 3 个与 vc11 不同，最大编码差为 1；
独立 FP64 模拟新公式的结果与 CUDA 逐位一致。这不是模型精度结论。
连续 64 块、四个随机种子、FP32/FP16/BF16 输出的解码 PV 模拟通过原有 0.1% 差异门槛，
但没有执行 Hopper WGMMA。远端安装检查和 profile 同样保留 0.1% 门槛。
完整验证步骤见 [性能诊断](SAGE_VC_PROFILE.md#vc12合并-expcast-的线性变换)。

## vc13：让当前块概率求和与 PV 重叠

版本 `2.2.0+vc13` / native kernel version 10。只修改 ExpCast 开启、V 平滑关闭的
路径；control、V 平滑、原版 `sageattn` 沿用原内核。vc12 的 H200 同场复测为
vc11 28.4311 ms、vc12 27.2819 ms，control 为 25.4781/25.5152 ms；这是用户提供的
核心 API 结果；vc13 后续 H200 结果和默认回退见下文。

vc12 在 PV 前完成 FP8 编码、解码和当前块求和。vc13 将同样的概率字节写入线程
私有的共享内存位置，再在 PV 执行期间从副本解码、FP32 求和和更新历史分母。
不能直接在 PV 期间读取 WGMMA 正在使用的概率寄存器；这是
[NVIDIA PTX 的异步寄存器访问约束](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)。

每 CTA 增加 8192 字节共享内存；D128 动态共享内存由 40960 增至 49152 字节。
各线程独占自己的副本，不增加跨线程同步。FP8 P 字节、FP32 tile sum 和两级
PV 累加保持 vc12 的数值约定，极端 logits 仍使用相同的旧编码回退。

编译器可能把 CUDA 算术移到等待之后。因此等待依赖两个非负 tile sum，串行
对照的 WGMMA fence 也依赖这两个和。每个和包含 32 个 [0,256] 的概率，范围
[0,8192]，所以整个 warpgroup 的条件一致为真，包括全掩码的零和。检查不成立
会 trap，不会跳过等待后读取未完成的矩阵乘结果。必须检查目标编译器的生成指令；
源码中的先后顺序不能单独证明发生了重叠。

三个模式均执行完整 attention 并返回输出：

| kernel_mode | 行为 | 对比目的 |
|---|---|---|
| `vc12` | 原版编码、解码求和调度，无共享副本 | 衡量最终净收益 |
| `vc13_serial` | 新共享副本，求和在 PV 开始前完成 | 检查副本与串行调度成本 |
| `vc13_overlap` | 新共享副本，求和与 PV 重叠 | 实验路径；vc13 发布时曾是默认 |

本地 SM86 可执行概率 helper 的数值/内存测试；SM90a 主内核仅交叉编译、检查
SASS，完整 H200 执行由远端验证。准确率门槛未放宽。具体命令见
[性能诊断](SAGE_VC_PROFILE.md#vc13完整内核中的概率求和调度对比)。

## vc13.1：H200 实测否决共享副本默认路径

用户回传的 9 轮同输入核心 API 中位数：

| 变体 | vc12 ms | serial ms | overlap ms |
|---|---:|---:|---:|
| control | 25.7339 | 25.6209 | 25.7394 |
| ExpCast | 27.4307 | 29.5502 | 30.7108 |

共享副本串行路径相对 vc12 增加 2.1195 ms，重叠路径再增加 1.1606 ms，总回退
3.2801 ms（约 11.96%）。control 接近不变，输出 RMSE 在打印精度下为 0.000000%。
这些是不同完整内核的净时间差，不等于孤立拷贝或求和的耗时；日志不能区分
共享内存吞吐、指令发射、同步依赖分别占了多少。

vc13.1 只恢复 Python 默认分发并修正诊断标签，native kernel version 仍为 10，
CUDA/C++ 源码未改。`optimized` 与 `vc12` 使用相同参数；`vc13_overlap` 显式选择
原 vc13 默认，`vc13_serial` 保留串行副本。资源报告也按实际默认路径重新标记，
默认不再申请这 8 KiB 副本空间。原版 Sage 仍更快，没有宣称达到加速目标。

当前安装的 vc13 无需先升级也可显式传 `kernel_mode="vc12"` 避开负优化。
升级默认行为：

```bash
git pull --ff-only origin master
python3 -m pip install ./third_party/sageattention_local --no-build-isolation --no-deps
```
