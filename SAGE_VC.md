# SageAttention 主体 + VC 两项改动

`sageattention-local==2.2.0+vc6` 在已经同场测到 25.8055 ms 的本地 Sage 源码上，
新增独立 `sageattn_vc`。原版 `sageattn` 的数学与内核保留。
**25.8055 ms 是用户提供的原版源码构建结果，不是本次新入口的速度。**

这是 **Sage+VC 混合实验**：QK 保留 Sage per-thread INT8；新增 V 聚类/块平滑和 ExpCast。
没有改为完整 VC 的 FP8 QK/Hadamard，也尚未接入论文的去噪步数调度。

vc6 修正 V 量化尺度的粒度：默认 `v_scale_mode="channel"`，每个 batch/head/channel
共享一个尺度，块均值仍按 128 个有效 token 计算。主内核在统一尺度下累加残差及均值
补偿，归一化后只乘一次 V scale；不再在每个 K/V 块读取尺度并恢复缩放。
这对应 [VC 论文附录 B 的 8-bit 描述](https://arxiv.org/html/2609.15810v1)。
**这是尺度/恢复方式的对齐，不代表完整复现论文，也不预先承诺 H200 加速。**

保留 Sage 的两级 PV 累加：每块 WGMMA 的临时结果转入 FP32 历史缓冲。
Hopper FP8 硬件累加的精度限制使这个缓冲不能随意删除，见
[SageAttention2 论文](https://arxiv.org/abs/2411.10958)。ExpCast 本版没有新增数学改动。

两种对照含义不同：

- `v_scale_mode="block"` 保留 vc5 的逐块尺度量化。它与默认路径的 V 字节/均值会不同，
  必须分别对比 dense FP32 误差；不能把差异当成纯执行误差。
- `kernel_mode="vc4"` 保留逐块恢复的参考执行方式。对于新的共享尺度输入，每块都读取
  同一个通道尺度；与 optimized 使用完全相同的量化数据，检查移出缩放及 FMA 的实现误差。
  因此共享尺度上的 vc4 是参考调度，不是旧版包的原样端到端结果。
- `v_scale_mode="block", kernel_mode="optimized"` 对应此前 vc5 的路径；
  `v_scale_mode="block", kernel_mode="vc4"` 对应此前 vc4 的路径，均由当前工具链编译。

`sage_vc_build_info()` 中 `sage_vc_kernel_version=4`、包版本 `2.2.0+vc6` 表示新扩展已加载。
QK INT8、聚类和原版 `sageattn` 内核保持原有计算方式；原版与混合实验有独立入口。

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
- 144 组还会在相同量化数据上对照 optimized/vc4，relative RMSE <=0.1%。这仅是实现回归门槛。
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
python3 scripts/profile_sage_vc.py --compare-kernels --compare-v-scales --trace-dir results/h200_sage_vc6_traces --output results/h200_sage_vc6_profile.json
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

# 相同量化方式，用逐块恢复的参考调度作对照
old_schedule = sageattn_vc(q, k, v, tensor_layout="NHD", kernel_mode="vc4")

# 旧 vc5 的逐块量化 + optimized 内核
old_quantization = sageattn_vc(q, k, v, tensor_layout="NHD", v_scale_mode="block")

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

本次本地 131 项测试通过，包括 60 组新旧 V 预处理、额外长块归约/大均值用例、
ExpCast 编码、实际 CUDA helper、独立参考、脚本与来源检查；此前 16 组 helper
保存结果仍逐位一致。Compute Sanitizer memcheck 为 0 errors。
44 种 Hopper 配置交叉编译、链接成功，无 spill，SASS 含 TMA、INT8 IGMMA、FP8 QGMMA。
CUDA 12.4 下 D128/BF16 默认平滑和全开使用 208/209 个寄存器，高于旧 block 模式
的 180/181；因此不能只凭少了尺度读取就断言提速，远端 CUDA 13 的编译与计时为准。
记录见 [results/sage_vc6_local_validation.json](results/sage_vc6_local_validation.json)。
本地没有 Hopper GPU，144 组完整 attention 及 H200 性能仍待远端验证。
