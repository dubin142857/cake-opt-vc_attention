# 0.7.0：QK/PV 与 ExpCast 交叠

用户提供的 H200、BSHD `[1,37741,28,128]`、BF16 结果：0.6.0 API 中位耗时
93.9205 ms，0.5 通用核心为 129.6991 ms。另一次 profiler 的 3 次调用平均值为：

| GPU kernel 累计耗时 / call | 0.6 D128 | 0.5 通用 |
| --- | ---: | ---: |
| Attention 核心 | 77.8304 ms | 113.2724 ms |
| 其他 kernel | 14.8750 ms | 14.8740 ms |
| 合计 | 92.7054 ms | 128.1464 ms |

0.6 核心中完整 Q 分块为 76.5068 ms，Q 尾块为 1.3236 ms。本轮主要改完整分块。
以上是用户远端提供的 **0.6 实测**；0.7 的 H200 性能待同场验证。
GPU kernel 累计时间不等于完整 API 时间；CPU 同步等待不能再加进去。

## 调度与数值契约

新核心仍为每 CTA 64 Q 行、128 线程、每 KV 块 128 tokens，使用 FP8 E4M3 WGMMA。
它适用于 D=DV=128、Q/K 量化分组 128/256、KV 分块 128、FP16 均值和 ExpCast。
普通 Softmax、其他几何配置和 Q 尾块保留此前路径。

初始化 QK(0)、P(0) 后，每一轮执行：

```text
提交 QK(i+1) → 提交 PV(i) → wait_group 1（QK 已完成）
                              ↓
                       ExpCast(i+1) → 写入另一份 P 缓冲
                              ↓
                       wait_group 0（PV 已完成）
                              ↓
                       更新 O(i)，准备下一轮 K/V
```

两份 P 缓冲保证 PV(i) 的输入在完成前不会被 P(i+1) 覆盖。
K 的复用等待 QK 消费完成；V 的复用等待 PV 消费完成。每次发布/复用都有 CTA barrier；
异步拷贝仍使用 `cp.async` 和 `wait_group 0`。没有 TMA 或线程组分工。

WGMMA 的 issue/commit 与 wait 分开，累计寄存器通过 operand fence 限定访问顺序。
`--fmad=false` 和 ExpCast 显式 FMA/RNE 保持原样；Q/K/P/V 量化、逐块分母、
V 均值恢复与输出累加顺序不变。仅提前计算下一块概率，输出仍按 key 块顺序更新。
初始化和最后一块有独立处理，最后一次 PV 必须完成后才写输出。

共享内存从 48 KiB 增至 **56 KiB**（48 KiB 静态 + 8 KiB 动态），通过
`cudaFuncSetAttribute` 显式申请，使用调用者的 CUDA stream。原生 ExpCast 核心
本地 nvcc 12.4 编译仍为 255 registers/thread、0 spill；不宣称提高寄存器允许的驻留率。

寄存器暂存下一块 P 的试验版本被编译器提前等待，未保留完整 ExpCast 交叠，
因此最终采用 P 双缓冲。普通 Softmax 的试验版本出现寄存器 spill，生产分派继续选择 0.6。

依据：[NVIDIA WGMMA 等待语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)，
[CUTLASS warpgroup 同步接口](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_gmma.hpp)。

## 远端编译与 A/B

```bash
git pull --ff-only origin master
bash scripts/validate_h800.sh
bash scripts/benchmark_37741.sh --backends flash3 sage vc --vc-interleave-compare --output results/h200_interleaved_compare.json
```

- `vc`：新交叠调度，Q 尾块继续用 0.6。
- `vc_d128`：完整保留的 0.6 调度。

两者显示相同包版本 0.7.0，每次都重新进行聚类、量化和输入检查。
报告输出 `0.6_ms / new_ms` 加速比和最后一次计时输出的抽样对拍。
`--vc-specialize-compare` 仍可增加 0.5 通用核心，`--vc-fused-compare` 增加 0.4。
如果还指定 `--vc-ablation`，只有启用 ExpCast 的分组增加 `_d128` 对照；
普通 Softmax 分组本来就在运行 0.6。

Python API 显式使用 `Config(backend="cuda_fp8", cuda_schedule="fused", expcast=True)`。
默认 `cuda_fused_interleave=True`；设置为 False 保留 0.6。
`stats["cuda_fused_variant"]` 会报告 `interleaved` 或 `d128`。
Config 默认 schedule 仍是 `batched`，V-Smooth 是否启用仍由 `v_smooth` 决定。

```bash
python3 scripts/profile_cuda.py --batch 1 --queries 37741 --tokens 37741 --heads 28 --dim 128 --vc-interleave-compare --output-dir results/h200_interleaved_profile
python3 scripts/check_fused_binary.py
```

Profiler 现在直接打印每个后端的核心、其他 kernel 和总计耗时，并保存 `gpu_summary`。
二进制检查要求交叠核心具有原生 E4M3 QGMMA，以及两个等待点之间的 ExpCast 转换指令，
避免只在源码中存在交叠。它仍不能代替原生设备执行和性能计数器。

## 单个主体内核的硬件指标

如果已安装 Nsight Compute，用专门入口排除自检和预热，只捕获一次完整分块核心：

```bash
mkdir -p results
ncu --profile-from-start off --kernel-name regex:fused_attention_interleaved --launch-count 1 --set full -o results/h200_interleaved_ncu python3 scripts/profile_cuda_kernel.py
ncu --profile-from-start off --kernel-name regex:fused_attention_d128 --launch-count 1 --set full -o results/h200_d128_ncu python3 scripts/profile_cuda_kernel.py --variant d128
```

默认即完整目标形状、BF16、5 次预热。收集后查看 Tensor Core/SM 吞吐、寄存器与驻留率、
warp stall、内存吞吐等指标。`--set full` 会重放内核；这些时延不用于代替 API 测速。
详见 [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。

## 验证范围

本地使用 CUDA 12.4 编译 sm_86 和 sm_90a，在 RTX 3060 上运行数值诊断。
诊断入口故意在下一块 ExpCast 后才读取旧 P/V，帮助发现缓冲区过早覆盖，
但矩阵乘是标量模拟，不能证明原生 WGMMA 并发正确性。生产 API 不会选择模拟入口。

完整测试、同 0.6 的逐元素对拍、Compute Sanitizer 和实际安装二进制检查的结果记录在
[results/interleaved_local_validation.json](results/interleaved_local_validation.json)。
本地全套 305 项通过、148 项跳过；新增 20 组循环缓冲对拍逐元素一致。
模拟入口的 memcheck 114 项、racecheck 66 项通过，均无错误；二进制检查覆盖 9 个原生函数。
H200 原生并发、CUDA 13 编译产物和速度需要远端验证；本轮不预报加速比。
