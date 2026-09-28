# SageAttention 主体 + VC 两项改动

`sageattention-local==2.2.0+vc3` 在已经同场测到 25.8055 ms 的本地 Sage 源码上，
新增独立 `sageattn_vc`。原版 `sageattn` 的数学与内核保留。
**25.8055 ms 是用户提供的原版源码构建结果，不是本次新入口的速度。**

这是 **Sage+VC 混合实验**：QK 保留 Sage per-thread INT8；新增 V 聚类/块平滑和 ExpCast。
没有改为完整 VC 的 FP8 QK/Hadamard，也尚未接入论文的去噪步数调度。

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
- FP16/BF16、D64/128、Q/K 边界长度、四种开关共 96 组 attention 对拍。
- 量化数据固定后，对比 CUDA 与独立 Python 参考，relative RMSE <=1% 是实现回归门槛。
- 双关闭入口与原版 Sage 比较，relative RMSE <=0.1%；另有非零均值已知答案和 HND/NHD 布局检查。
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
2. 新 CUDA 预处理按 128 个有效 token、每通道计算 FP32 均值和残差。
   残差以 E4M3 保存，尺度为 `amax/448`；常量通道尺度取 1，并用 `abs(mean)/65504`
   作尺度下界以容纳 FP16 的 `mean/scale`。V 存储采用 Sage PV 的转置和 token 排列。
   均值不包括补齐 token，补齐残差为 0。
3. Sage TMA 搬运、INT8 QK WGMMA、FP8 PV WGMMA 主体保留。
   每块 PV 在寄存器里执行 `(PV_residual + rowmass * scaled_mean) * block_scale`，
   随在线最大值调整历史分子/分母。块 scale 不能沿用原版的最后统一乘一次。
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

本次本地结果：60 项 CPU/适配器/来源检查、30 组 CUDA V 预处理、65,540 个
ExpCast 字节检查、4 组实际 device helper 的寄存器打包/在线状态/均值恢复检查通过。
CUDA 12.4 交叉编译全部 16 个 sm_90a attention 变体并链接成功，ptxas 无寄存器溢出到本地内存；
SASS 确认有 TMA、INT8 IGMMA 与 FP8 QGMMA。上述两类本地 GPU 测试经
Compute Sanitizer memcheck 检查均为 0 errors。记录见 `results/sage_vc_local_validation.json`。
