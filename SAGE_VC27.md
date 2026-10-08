# vc27：FP8 QK 实验分支

**2026-10-08 状态调整：本页归为历史实验，H200 后续主路线回到 INT8 QK + FP8 PV。**
此前“论文 QK 统一用 FP8”的表述不准确；作者主流程为 INT8，B300 另有 FP8 变体。
主入口与验证命令见 [H200_INT8.md](H200_INT8.md)，不再默认运行本页的 FP8 QK 测速。
用户随后回传的 vc27.1 随机输入完整 API 为 26.5016 ms（FP8 ExpCast），
同场 INT8 vc22 为 25.6884 ms；FP8 核心虽稍快，完整调用并未提速。
下面保留当时的实现与验证记录；这些随机输入结果不是模型质量验收。

当前包版本 `sageattention-local==2.2.0+vc27.1`，native kernel version 25。
新增 `sageattn_vc_fp8`，在现有融合 CUDA attention 中让 QK 与 PV 都使用 E4M3 Tensor Core。
INT8 的 `sageattn_vc` 默认选择保持 vc15；实测候选 vc22 继续作为对照。

## vc27.1：修复全零 V 通道产生 NaN

首轮 H200 随机输入检查中，FP16/D64/Q193/K513 的实现相对 RMSE 为
0.3795%（普通概率）和 0.3055%（ExpCast），随后常量答案检查出现 12.5% NaN。
这是沿用的 V 量化在 `amax=0` 时计算 `448/0`，再执行 `0*Inf` 导致的。
测试中每八个 V 通道恰好一个全零，解释了 12.5% 的失败比例。

FP8 入口改用本地派生的 V 量化 kernel：零通道写入零 FP8 编码及零 scale，
非零通道保留原有计算和硬件转换。修复发生在原有量化 pass 内，没有新增 kernel 下发。
原版 Sage 对照源文件及 attention 主核不变；精度检查门槛不变。
验证脚本新增 60 组原生 V 量化检查，并将常量答案扩大到 48 组，覆盖全零 V 和尾块。
运行完整 H200 验证时还检查非零通道与原版的量化字节完全相同。

## 这一轮的依据

用户回传的 vc26 重复 CUPTI 核心耗时如下。测量时 profiler 开启，不能当作完整 API 或模型端到端耗时。

| 核心 | ms |
|---|---:|
| control_vc16 | 24.0596 |
| control_vc26_shared | 32.6539 |
| control_vc26_stagger | 30.0253 |
| expcast_vc22_both | 23.6912 |
| expcast_vc26_shared | 27.3786 |
| expcast_vc26_stagger | 27.1235 |
| sage_cpp | 24.2156 |

vc26 stagger 的 ExpCast 核心比 vc22 慢 14.5%，该方案不进入默认路径。
vc27 转向 FP8 QK，但速度是否改善仍需 H200 同场验证，没有已经达到 22 ms 的结论。

## 算法及实现

| 项目 | INT8 对照 | vc27 FP8 |
|---|---|---|
| Q/K | Sage per-thread INT8 | E4M3；每 Q64/K128 块一个 FP32 scale |
| K 去均值 | Sage 输入 dtype 的均值/减法 | FP32 分段统计与减法 |
| QK 指令 | INT8×INT8，INT32 累加 | FP8×FP8，FP32 累加 |
| QK 后处理 | 分数 INT32→FP32、缩放 | 直接使用浮点分数、缩放 |
| V | Sage per-channel FP8 | 同一量化方式及字节布局 |
| 概率 | 普通路径 / ExpCast | 普通路径 / ExpCast |
| V 平滑、Q/K Hadamard | 本轮关闭 | 尚未加入 |

Q/K 量化约定：`scale = max(RN(amax / 448), 1e-12)`，`code = E4M3_RNE_SATFINITE(RN(x / scale))`。
K 的 `x` 为原始 K 减去 FP32 全序列通道均值。最后不完整的块只统计有效 token。
外部 scale 仍必需；代码没有直接把 FP16/BF16 无缩放转换为 FP8。

新增 C++/CUDA 预处理：K 部分统计、统计归约、Q 分块量化、K 分块去均值并量化，共四次 kernel 下发。
这些中间统计不生成完整 FP32 K 张量。注意力复用单 warpgroup、Q64/K128、TMA 搬运、
在线归一化和两级 FP32 PV 累加。新增 shared/shared FP8 QK WGMMA 指令，移除逐分数整数转换；
ExpCast 保留 vc15 的配对字节编码、FP32 解码求和及恒等历史缩放跳过，最大值改用 FP32 归约。

普通路径的分母累加量化前的指数值，分子使用 FP8 P；ExpCast 的分母使用解码后的 FP8 P。
两者是不同概率算法，不能要求输出完全一致。

本轮同时改变 QK 格式、量化粒度和 K 统计舍入，因此 INT8/FP8 的结果是方案对比，不能归因为某条 FP8 指令。
这也不是完整论文复现：Hadamard、V 平滑、RoPE 融合和去噪步数调度均不在该入口内。

## H200 编译、验证和性能

远端已有 PyTorch、Triton、CUDA Toolkit >=12.4 即可，无需联网安装新依赖、ncu 或 nvdisasm：

```bash
git pull --ff-only origin master
bash scripts/validate_fp8_qk_h200.sh
```

脚本离线重新编译，先运行原有 Sage/VC 回归，再运行 vc27 验证，通过后才测试性能。
已经安装 vc27.1 时可加 `--skip-build`；旧 vc27 及更早二进制必须重新编译才能包含零通道修复。

默认形状为 NHD `[1,37741,28,128]`，BF16、全部 37741 个 key。
计时输入预先置为 HND 连续布局；不包括调用者做 NHD/HND 转换的时间。
对比 `int8_control_vc16`、`int8_expcast_vc22`、`fp8_control`、`fp8_expcast` 和 `sage_cpp`。

- `full API`：每次重新量化 Q/K/V，包含预处理、分配与 attention。
- `fixed core`：复用准备好的量化输入，包含输出分配及 native dispatch；不是纯 kernel 时间。
- `CUPTI`：单独一轮 profiler 开启的 GPU kernel duration，不能与 wall time 混算。
- `FP32 RMSE %`：128 个均匀抽样 query、全部 key，相对 dense FP32；没有模型质量通过门槛。
- `FP8 REF %`：相同 FP8 字节及 scale 下对照解码 FP32 oracle，检查实现；不能当成整体量化误差。

ExpCast 开关共享同一份 Q/K/V payload；INT8 与 FP8 只共享原始输入和 V payload。
完整验证另覆盖 D64/D128、FP16/BF16、批次/多头、1/63/64/65/127/128/129/257/513 等边界，
检查全部输出行、HND/NHD 公共入口、已知常量答案、非默认流核心及默认流公共入口约束。
FP8 解码 oracle 不模拟 Hopper 内部累加舍入，随机输入实现检查门槛为相对 RMSE 1%；
该门槛包含 QK/PV 累加和概率字节边界效应，不能用于判定模型质量。

结果文件：`results/h200_vc27_validation.json`、`results/h200_vc27_fp8.json`，
CUPTI 原始 trace 在 `results/h200_vc27_traces/`。没有 CUPTI 时，已完成的 wall-time 报告仍会保存。
独立运行不带 profiler 的比较：

```bash
python3 scripts/profile_fp8_qk.py --output results/h200_vc27_fp8.json
```

## Python 入口

```python
from sageattention_local import sageattn_vc_fp8

# CUDA FP16/BF16，非因果前向；NHD = [B,N,H,D]
out = sageattn_vc_fp8(q, k, v, tensor_layout="NHD", expcast=True)
control = sageattn_vc_fp8(q, k, v, tensor_layout="NHD", expcast=False)
```

只支持 Hopper、D64/128、同头数 Q/K/V。有限输入，无 backward、GQA、causal、LSE。
完整 API 仍要求默认 CUDA stream，因为沿用的 Sage V 预处理使用默认流；
`vc_fp8.forward_prepared` 使用当前流，拆分调用者需建立输入/输出的流依赖。

## 本地验证范围

CUDA 12.4 已交叉编译 8 个新 SM90a 实例：D64/D128 × FP16/BF16 × 两个概率开关。
D128 每线程 168 寄存器，动态 shared 40960 字节，静态 shared 128 字节；无 stack/spill/local 读写，
无 ptxas warning 或 WGMMA 序列化诊断。已有 292 个实例的反汇编指令文本保持相同。
新的 QK 和 PV 均生成 FP8 MMA；新核心还保留一个无关逐分数转换的初始化 I2F。

本地 SM86 执行了 36 组原生 QK 预处理检查：K 均值对照独立 FP64 统计，
FP8 字节与 scale 对照显式 RN 除法的 PyTorch oracle，并覆盖真实非默认流。
完整 Hopper attention 的精度和速度需远端执行，交叉编译不能替代运行验证。
详见 `results/sage_vc27_codegen_local.json` 与 `results/sage_vc27_local_validation.json`。

vc27.1 本地检查为 560 项通过，其中新 V 量化检查覆盖 60 组输入。
用生产量化主体禁用零通道保护，在本地 SM86 上复现了恰好 12.5% NaN；
启用保护后不再产生 NaN，这四组常量通道案例的非零编码与 scale 完全一致。
新增 V kernel 已交叉编译 SM90a：32 寄存器、260 字节静态 shared、无 stack/spill。
SM86 验证使用 CUDA 软件 FP8 编码；Hopper 硬件编码及完整 attention 仍需远端重跑。
记录见 `results/sage_vc27_zero_value_fix_validation.json`；修复不构成性能提升结论。
