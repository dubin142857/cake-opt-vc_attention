# SageAttention 主体 + VC 两项改动

`sageattention-local==2.2.0+vc7` 在已经同场测到 25.8055 ms 的本地 Sage 源码上，
新增独立 `sageattn_vc`。原版 `sageattn` 的数学与内核保留。
**25.8055 ms 是用户提供的原版源码构建结果，不是本次新入口的速度。**

这是 **Sage+VC 混合实验**：QK 保留 Sage per-thread INT8；新增 V 聚类/块平滑和 ExpCast。
没有改为完整 VC 的 FP8 QK/Hadamard，也尚未接入论文的去噪步数调度。

vc7 集中优化 ExpCast 概率路径，默认继续使用 vc6 的通道共享 V scale：

- 使用 `cvt.rni.sat.u8.f32` 合并下界截断和整数 RNE 转换；移除独立 `max(0,u)`。
  attention 中 shifted <=0，FMA 结果 <=119.65，u8 的上界不会触发。
- 两个字节和两对字节均用 byte permutation 打包。
- 每四个概率解码后立即加入分母和块权重和，不再将整块解码概率写回 scores 数组。
  保留原来的左结合加法次序、FP32 FMA 常数、量化尺度及 mean 补偿。

饱和转换的语义见 [NVIDIA PTX cvt 文档](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)。
本地实际 helper 检查逐位一致，但 **减少指令不等于确认 H200 加速**。

保留三种主内核模式，用相同安装包、相同量化数据作同场比较：

| kernel_mode | 概率路径 | V 恢复 |
|---|---|---|
| `optimized`（默认） | vc7 饱和转换、字节打包、流式求和 | 与 vc6 相同 |
| `vc6` | 上一版 ExpCast | 共享通道 scale 在输出末尾恢复 |
| `vc4` | 更早的概率实现 | 即使 scale 共享，也逐块恢复，作历史参考 |

`optimized` 和 `vc6` 在 ExpCast 关闭时调用同一个内核，作为计时波动的控制组。
两者保留 Sage 的两级 FP32 PV 累加，不把完整历史送回 FP8 WGMMA 累加器。
默认通道尺度对应 [VC 论文附录 B 的 8-bit 恢复方式](https://arxiv.org/html/2609.15810v1)，
QK 仍是 Sage INT8，**不是完整 VC 论文复现**。

`v_scale_mode="block"` 保留旧的逐块量化；与 channel 的 V 字节不同。
用 `v_scale_mode="block", kernel_mode="vc6"` 对照此前 vc5，
`v_scale_mode="block", kernel_mode="vc4"` 对照此前 vc4；均由当前工具链重新编译。
切换 `kernel_mode` 不改变准备好的 Q/K/V；切换 `v_scale_mode` 会改变 V 量化。

`sage_vc_build_info()` 中 `sage_vc_kernel_version=5`、包版本 `2.2.0+vc7` 表示新扩展已加载。

## 远端编译和测试

在 H200 的仓库根目录：

```bash
git pull --ff-only origin master
bash scripts/build_sage_local.sh
```

只更新独立包 `sageattention-local`，不用重新安装主包 `vc-attention`，
不覆盖已有 `sageattention`。需要 nvcc >=12.3、PyTorch >=2.4、Triton >=3.0。
新增 `_vc_sm90` 原生扩展，编译目标 `sm_90a`；旧安装需重新编译。

脚本先测原版，再运行 `validate_sage_vc.py`：

- V 编码、均值、尺度、尾块和稳定聚类检查；ExpCast 编码与独立 FP32-FMA 参考逐字节比较。
- FP16/BF16、D64/128、Q/K 边界长度、四种开关及两种 V scale 粒度共 144 组 attention 对拍。
- 量化数据固定后，对比 CUDA 与独立 Python 参考，relative RMSE <=1% 是实现回归门槛。
- 双关闭入口与原版 Sage 比较，relative RMSE <=0.1%；另有非零均值已知答案和 HND/NHD 布局检查。
- 144 组还会在相同量化数据上对照 optimized/vc4 和 optimized/vc6，relative RMSE <=0.1%。
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

新旧内核同进程、同量化输入比较，不需要 `ncu`：

```bash
python3 scripts/profile_sage_vc.py --compare-kernels --kernel-baseline vc6 --trials 5 --repeats 10 --trace-dir results/h200_sage_vc7_traces --output results/h200_sage_vc7_profile.json
```

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

本次本地 169 项测试通过，包括原有预处理、独立参考、脚本、来源检查，以及新旧
概率路径的逐位比较。64 块连续在线历史、极小/极大历史分母、极短尾块、所有 E4M3
非负概率字节对、RNE 边界相邻浮点数均已覆盖；原先保存的 16 组 helper 结果仍逐位一致。
Compute Sanitizer memcheck 为 0 errors。
56 种 Hopper 配置交叉编译、链接成功，无 spill，TMA/INT8 IGMMA/FP8 QGMMA 保留。
CUDA 12.4 的 D128/BF16 下，单 ExpCast 寄存器 167→168，全开 209→208；
没有明显降低寄存器占用。减少的截断/移位指令是否带来净收益，仍需 H200 CUDA 13 实测。
检查记录见 [results/sage_vc7_local_validation.json](results/sage_vc7_local_validation.json)。
本地没有 Hopper GPU，144 组完整 attention 及 H200 性能仍待远端验证。
