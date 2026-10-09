# H200 CUDA 主线：vc28 最大值归约候选

2026-10-09。当前主线是 **INT8 QK + ExpCast + FP8 PV，V 聚类和平滑关闭**。
保留 vc28 已测实现，新候选不修改已安装包和默认 dispatch。

## 当前候选：balanced max

上一轮 split-mass 用户实测 CUPTI 为 **23.8129 ms**，同次编译的 vc28 为 **23.6389 ms**，
已安装 vc28 为 **23.5260 ms**，Sage 为 24.8834 ms、FA4 为 30.9834 ms。
split-mass 比 JIT 基线耗时多约 0.74%，没有加速证据，保留为可复测的历史候选。
该段数据只有 CUPTI 摘录，不推断完整 API 或模型端到端收益。

本轮 `vc28_balanced_max` 只改变每线程每行 32 个 INT32 分数的最大值归约树。
原实现包含较长的串行 max 链，新实现按三元组组成平衡树。
两者仍为每行 15 条三输入 max 和 1 条两输入 max，数学结果完全相同。
原 vc28 的编码、分母求和、跨线程最大值归约、K scale、QK/PV、尾块和 fallback 均保留。
新候选没有叠加 split-mass。

本地 CUDA 12.4 SM90a 的 SASS 检查确认，两行各自的 max 依赖链均从 **6 层降为 4 层**，
max 指令数量、完整 kernel 的静态指令数量相同；D128 仍为 168 寄存器、无 spill。
这是依赖结构证据，不是硬件等待计数，也不能换算为已实现的毫秒收益。

147 项本地检查通过，其中 34 项直接执行新 GPU helper：覆盖全部 32 个最大值位置、
INT32 极值、并列值、负值、1/7/296 块概率状态、正常/极端尺度和非默认 stream。
最大值、alpha、P 字节、局部分母均与 vc28 逐位一致。
完整 H200 attention 与性能仍需远端验证。

## 为什么回到 CUDA

用户最近一次 H200 CUPTI 数据（内核时间，不是完整 API 或模型端到端时间）：

| 实现 | 毫秒 |
|---|---:|
| vc28_folded | 23.4288 |
| Sage | 24.8400 |
| cuda_paper | 27.8298 |
| FlashAttention-4 | 30.1747 |
| TileLang 自然 V，s1 | 56.9391 |
| TileLang 自然 V，s2 | 57.0332 |
| TileLang PV 重叠，s1 | 61.6408 |
| TileLang PV 重叠，s2 | 62.2024 |

TileLang 重叠候选在该场景下退化，保留代码作为实验记录，暂不继续扩展。
vc28 在同轮达到 Sage 的约 1.060 倍内核速度，以它作为后续性能基线。

## 历史候选：split mass

vc28 对每组解码值先算 `x+y`，再累加至两个 FP32 部分和。
新候选分别累加四个分量，最后合并：

```text
vc28:       a += x0+y0; b += x1+y1;  mass = a+b
split mass: a += x0; b += y0; c += x1; d += y1; mass = (a+b)+(c+d)
```

目标是让解码后的加法链更独立；没有采用 FP16 分母、近似求和、整数查表或省略归一化。
这与历史 vc17 的四条“数对累加链”不同：新循环中去掉了内层 `x+y` 的依赖，
编码分组、流式 INT32 转换、FP8 解码均继承 vc28。

有效概率字节 0..120 的 E4M3 数值都是 `2^-9` 的整数倍，最大为 256。
每线程每行求和 32 个值，乘 512 后不超过 `2^22`，所以所有局部 FP32 加法均精确。
改变这段相加顺序不会改变局部分母。QK/PV、在线最大值与 alpha、历史分母更新、
超范围 fallback 和 masked tail 保持原路径。

本地 CUDA 12.4 生成的 D128 候选仍用 168 寄存器、无 spill。
静态完整 kernel 比基线多 4 条 FADD（末尾合并），解码指令数未减少。
因此这是**依赖调度候选**，不是已经证实减少总指令或达到 22 ms 的版本。
源码和编译信息无法替代 H200 完整内核实测。

## 运行

使用现有 Torch、`sageattention-local`、FA4 和 nvcc，无需安装 TileLang或重编译 Sage 包：

```bash
git pull --ff-only origin master
bash scripts/run_cuda_expcast_h200.sh
```

脚本默认选择 `--experiment balanced_max`，使用 `python3`；首次通过 PyTorch C++ 扩展
JIT 编译 vc28 基线和新候选，随后使用缓存。
默认 BF16、非因果 MHA，NHD 等价 `[1,37741,28,128]`，7 轮 API、3 轮 CUPTI。
代码、依赖已在机器上时无需联网；也不需要 ncu/nvdisasm。

```bash
# 仅跑小尺寸完整 attention 精度验证
bash scripts/run_cuda_expcast_h200.sh --validate-only
# 真实模型 QKV，文件包含 q/k/v 三个 Tensor
bash scripts/run_cuda_expcast_h200.sh --qkv-file captures/qkv.pt --qkv-layout NHD
# 需要复测上一轮 split-mass 时显式选择
bash scripts/run_cuda_expcast_h200.sh --experiment split_mass
```

默认输出 `results/h200_cuda_expcast_balanced_max.json`；trace 保存在
`results/cuda_expcast_balanced_max_traces/`。旧 split-mass/TileLang 结果不覆盖。
选择 `--experiment split_mass` 时仍使用旧版 split-mass 文件名。

## 对照组和判定

| 行名 | 用途 |
|---|---|
| vc28_jit | 同一源文件、工具链新编译的 vc28 基线 |
| vc28_balanced_max | 只改变局部 INT32 最大值归约树的当前候选 |
| vc28_installed | 已安装包的 vc28；检查 JIT 编译或封装差异 |
| sage_cpp | Sage 完整 API / 原 native core |
| flash4 | 已安装的 FlashAttention-4；无 prepared core |

full API 每次从原 Q/K/V 重新量化；core 复用同一份量化输入，含输出分配和 dispatch。
CUPTI 单独记录 GPU 活动，包含每次调用的所有 kernel，不与 API 时间混算。
所有实现交错、随机顺序计时。FA4 和 CUPTI 默认开启。

先验证 28 个小输入配置：FP16/BF16、D64/D128、对齐/尾块、单 token 和跨块不同 V 的
uniform attention，包含非默认 stream。随后主尺寸检查完整 API/core 一致，
并对全部输出行检查候选与 JIT 基线、已安装 vc28 的差异。
实现回归门槛仍为 0.1% 相对 RMSE，量化参考为 1%；同时记录逐位一致性。
这些是实现正确性检查，不是模型质量承诺。dense FP32 默认抽 128 个 query、全部 keys。

重点看 `Same-payload vc28 JIT baseline / vc28_balanced_max`：full/core 比值大于 1 表示候选更快。
然后核对独立 CUPTI 时间。若结果持平或退化，继续保留原 vc28，不自动切换默认实现。

vc28 使用折叠 affine 求值，边界舍入可与论文直接公式版本不同。
直接公式的 CUDA/TileLang 实验仍保留在 [原实验说明](SAGE_EXPCAST_TILELANG.md)，
本轮不把两种公式的误差差异归类为调度回归，也不声称完整复现了论文。

当前本地验证见 [balanced-max 验证](results/cuda_expcast_balanced_max_local_validation.json)，
历史记录见 [split-mass 验证](results/cuda_expcast_split_mass_local_validation.json)。
本地设备仅支持 SM86；完整 SM90a attention 的数值与性能需上述远端脚本验证。
可选独立 helper 测试：`python3 tests/cuda/run_balanced_max_probe.py`（需 pytest）。
