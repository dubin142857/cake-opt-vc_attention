# H200 CUDA 主线：ExpCast 调度候选

2026-10-09。当前主线是 **INT8 QK + ExpCast + FP8 PV，V 聚类和平滑关闭**。
保留 vc28 已测实现，新候选不修改已安装包和默认 dispatch。

## 当前候选：fused quad（以 balanced-max 为基线）

最新 H200 分段诊断中，balanced-max 的采样概率区段为 949 cycles，control 为 1362；
细探针编码 366、解码 153、FP32 求和 385 cycles。细探针相对实际 core 增加约 9.6% API
耗时，因此这些数据只用于选择方向，不能相加或折算成预期加速毫秒数。

本轮把 **四个概率的 INT32→FP32→ExpCast 编码、FP8→half2 解码、FP32 数对求和**
放在一个内联 PTX 小组中，先开始解码高两字节，再编码低两字节。
小组只返回 PV 所需的四字节 word 和两个 FP32 数对和，缩短中间值在源码中的存活范围。
仍按原来的两个累加链和原 K 顺序更新分母。第一对直接初始化累加链，去掉四次 `0 + pair_sum`。
该值域内 pair_sum 非负、有限，包含正零；这些中性加法的删除不改变位模式。

最大值树、FMA/RNE 编码、P 字节、FP32 数对相加和链更新顺序均保持。
half2 只承载精确解码值，**不做 FP16 求和或分母累加**。
范围判断、fallback、尾块、QK/PV 和历史累加保留原实现。
两条 JIT 路径都使用 balanced-max，避免把上轮归约树收益混到这次实验中。

```bash
git pull --ff-only origin master
bash scripts/run_fused_expcast_h200.sh
```

沿用已有 Torch、Sage local、FA4 和 nvcc，不安装依赖、不重编译包。
BF16、NHD `[1,37741,28,128]`、只开 ExpCast、7 轮 API 与 3 轮独立 CUPTI。
输出 `results/h200_cuda_expcast_fused_quad.json`，trace 在 `results/cuda_expcast_fused_quad_traces/`。
只验证可以追加 `--validate-only`；真实 QKV 可追加 `--qkv-file ... --qkv-layout NHD`。

| 行名 | 含义 |
|---|---|
| vc28_balanced_max | 同次编译的已测 balanced-max 基线 |
| vc28_fused_quad | 当前编码/解码/数对求和候选 |
| vc28_installed | 已安装 vc28，检查 JIT/封装差异 |
| sage_cpp | Sage 原 native core 与完整 API |
| flash4 | FA4 完整 API，无 prepared core |

测速前新增 **38 个概率状态逐位检查**：全部 121 个有效 E4M3 概率码、RNE 边界、
全零/全最大概率、随机/重复输入、极端/微小/交替尺度、1/7/296 连续 tile、非默认 stream。
P、最大值、alpha、局部分母必须逐位一致，分母另对照独立 FP64 解码求和；任一失败即停止。
然后继续原来的 28 个小 attention 配置、主尺寸全部输出及量化 reference 检查。

本地 CUDA 12.4 的 D128 两个路径均为 168 寄存器、无 stack/spill。
同次编译的候选完整 kernel 静态计数少 **4 FADD + 4 NOP**，FP8 解码数量相同。
这说明少量工作已被删除，不等于 H200 已获得某个百分比收益；调度变化仍可能退化。
235 项本地测试通过，其中新 38 项在 SM86 执行软件 FP8 解码路径。
原生 FP8 PTX 和完整 SM90a attention 已交叉编译，实际执行由上述 H200 脚本完成。
证据见 [本地验证](results/cuda_expcast_fused_quad_local_validation.json)。

`run_cuda_expcast_h200.sh` 默认仍保留 balanced-max 对 vc28 的旧实验，
也可显式 `--experiment fused_quad` 运行本候选。没有自动更改已安装包的默认内核。

## 已测候选：balanced max

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
用户随后回传的 H200 数据如下，全部输出与同编译 vc28 逐位一致：

| 实现 | full API ms | fixed core ms | CUPTI kernel ms |
|---|---:|---:|---:|
| vc28_jit | 25.4198 | 23.9726 | 23.6985 |
| vc28_balanced_max | 25.3386 | 23.8280 | 23.5850 |
| vc28_installed | 25.4918 | 23.9132 | 23.6553 |
| sage_cpp | 26.7730 | 25.2289 | 25.0843 |
| flash4 | 31.2107 | — | 30.7975 |

balanced-max 的 full/core/kernel 时间分别下降约 0.32% / 0.60% / 0.48%。
寄存器和共享内存没有变化。三种时钟方向一致，但收益小于 1%，
当前仅有用户粘贴的汇总，无原始 trial，不能宣布稳定显著收益。
摘录见 [H200 回传记录](results/h200_balanced_max_reported_summary.json)。

## 下一步：对当前 balanced-max 分段定位

```bash
git pull --ff-only origin master
bash scripts/diagnose_balanced_expcast_h200.sh
```

默认仍为 BF16、NHD `[1,37741,28,128]`、INT8 QK、仅开 ExpCast。
沿用已有环境，无需 pip 安装、ncu、nvdisasm 或 TileLang。
首次编译同源 vc28/balanced-max，以及当前 balanced-max 的粗/细两套探针。
7 轮正式 API、3 轮 CUPTI、5 轮 phase 校准，保留全部原始数据。

结果 `results/h200_expcast_balanced_profile.json`，trace 与原始 clocks 在同名前缀的
`results/h200_expcast_balanced_profile_traces/`；不覆盖上轮测速或旧 vc28 诊断报告。
也可以附加 `--qkv-file captures/qkv.pt --qkv-layout NHD`。

这里 `expcast` 明确表示 **balanced-max**；`vc28_jit`、`vc28_installed`、
`control`、Sage 和 FA4 保留作对照。分段预处理和 attention 探针使用同一个选定实现，
不会拿旧 vc28 的内部采样冒充当前候选。先验证小输入，再验证主尺寸全部输出，失败即停止。

诊断重点：

- 预处理：K 去均值、Q/K INT8 量化、V 补齐与 FP8 量化的 CUPTI 时长。
- 主循环：QK 发射/等待、概率计算、PV 发射、PV 期间历史更新和剩余等待。
- 概率内部：max/alpha、范围判断、bias、编码打包、FP8 解码、FP32 分母求和。
- 测量校准：同轮随机顺序测正式 core、未插桩诊断副本、粗探针、细探针及其未插桩副本。
  报告编译副本漂移和插桩影响，避免把两者误认作真实算子瓶颈。

phase 是采样线程的 elapsed cycles，含调度与插桩开销；各 phase 中位数不可相加，
也不能乘以正式 kernel 毫秒数来估计“省掉某段就能提速多少”。
PV 与历史更新有重叠，`pv_wait_remaining` 不是整个 PV 的计算量。
最终是否优化成功仍由无插桩完整 API 与独立 CUPTI 验证。

本轮增加诊断能力，未再更改 ExpCast 数值或已安装包的默认 dispatch。
可选 helper 对拍：`python3 tests/cuda/run_expcast_detail_probe.py --row-max balanced_max`。
本地 185 项测试通过，含 24 项新增 balanced 细探针 GPU 对拍；粗/细探针共 32 个完整
SM90a 实例交叉编译通过、无 stack/spill。D128 的未插桩和插桩副本均用 168 寄存器。
D64 插桩会增加寄存器，因此仍需看各自的校准结果，不能把编译成功当作零测量扰动。
证据见 [探针本地验证](results/balanced_expcast_profile_local_validation.json)。
本地 SM86 只执行概率 helper，完整 H200 attention 验证由远端脚本执行。

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
