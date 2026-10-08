# vc30：复用 WGMMA 共享内存描述符

包版本 `sageattention-local==2.2.0+vc30`，native kernel version 28。
新增 `kernel_mode="vc30_desc"`，以已测的 **vc28** 为基线，关闭 V 平滑。
Q/K 为 INT8，P/V 为 FP8；沿用 vc28 的 ExpCast、FP32 分母及两级 PV 累加。
这是待 H200 测量的性能候选，默认仍为支持 V 平滑的 vc15。

## 为什么改这里

上一轮 vc29 完整/核心 API 都比 vc28 慢约 1.2%，因此不采用两行编码交错。
同场 vc28 完整调用为 24.6613 ms，核心为 23.3914 ms。见
[vc29 回传](results/h200_vc29_reported_summary.json)。

本地检查 vc28 的 SM90a 指令发现，在各次 QK/PV MMA 前，存在重复的地址加法、
掩码、移位和描述符字段构建。单缓冲主循环中的 Q/K/V 共享内存地址固定，
各 K32 fragment 只在基地址上偏移 0、32、64 或 96 字节。

vc30 构建三个基地址描述符，片内通过设置低位偏移复用它们。
新增的 MMA wrapper 接收编码后的描述符，保留上游矩阵指令及所有操作数约束。
包括尾块在内，矩阵提交、等待、数据预取、scale 和输出累加顺序都不变。
旧路径和上游 `wgmma.cuh` 保留，改动只通过独立 schedule 37 启用。

## 等价条件

按照 [NVIDIA PTX 描述符格式](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor)，
起始地址编码为 `(address & 0x3ffff) >> 4`，位于描述符最低 14 位。
本内核 Q/K/V 基地址均 128 字节对齐，所以编码后的低三位为零。
0/32/64/96 字节对应编码偏移 0/2/4/6，直接 OR 与完整重建描述符逐位等价，
不会向 stride、base-offset 或 swizzle 字段进位。

该优化限定 Q64/K128、D64/128、INT8 QK、单 K/V 缓冲和未拆分的 PV。
不能把这个 OR 规则直接用于不满足对齐或偏移条件的其他内核。
目标是减少地址构建工作，不修改概率近似或降低累加精度。

## 本地检查与限制

- **600 项测试通过**，包含已有 CPU/CUDA helper、来源校验、调度与测速脚本检查。
- 新 GPU probe 在 256 个 128 字节对齐的共享内存基址上，覆盖 64/128 字节 stride、
  四个 K32 偏移和默认/非默认 stream；与上游完整构建的描述符逐位相同。
- CUDA 12.4 交叉编译 312 个 SM90a 实例，原有 308 个实例的机器指令文本保持一致。
- D128 静态指令数 **2856 → 2808**；D64 为 **2536 → 2512**。
  这是整个内核（包括尾块和回退路径）的静态统计，不是每轮执行数，也不是延迟降幅。
  D128 少了各 18 条地址加法、右移和掩码指令，另增加了少量 move/NOP。
- 新 D128 仍为 168 寄存器、D64 为 128；无 stack/spill 和 WGMMA 串行化诊断。
  新旧候选的矩阵、FP32 算术及概率转换指令数量相同。

本地 GPU 是 SM86，只执行 helper/预处理，不能运行完整 Hopper attention。
完整数值和性能还需 H200 验证；远端 CUDA 13 也可能生成不同指令调度。
证据见 [编译检查](results/sage_vc30_codegen_local.json) 和
[本地验证](results/sage_vc30_local_validation.json)。

## H200 执行

```bash
git pull --ff-only origin master
bash scripts/validate_int8_h200.sh
```

**本轮需重新编译，不能使用 `--skip-build`。** 全部使用 `python3`，沿用已有依赖，
无须 ncu/nvdisasm。脚本先验证，再比较四组 vc15 消融、vc22、vc28、vc29、vc30、Sage 和 FA4。
默认等价 NHD `[1,37741,28,128]`、BF16；各库原生连续布局的转换在计时外。

关注 `int8_expcast_vc30` 和末尾 `vc28/vc30`。完整 API 比值大于 1 才表示完整调用提速；
固定核心与全输出回归同时报告。0.1% 实现门槛不代表模型质量，FA4 的低误差也单独保留。

安装后如需增加轮次和独立 CUPTI：

```bash
python3 scripts/profile_int8_h200.py --warmup 10 --repeats 10 --trials 9 --kernel-trials 5 --trace-dir results/h200_vc30_traces --output results/h200_vc30_compare.json
```

新候选需在同场完整调用中超过 vc28 才采用；寄存器、静态指令和 helper 结果均不能代替 H200 实测。
