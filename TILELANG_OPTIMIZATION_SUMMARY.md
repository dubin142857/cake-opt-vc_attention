# TileLang Attention 优化总结

整理日期：2026-10-16。

本次将 Hopper 上的 INT8 QK / ExpCast / FP8 PV attention，从原始版本约 **56.9 ms** 优化到约 **38.4 ms**。最新一次同轮测量中，原始版本为 **56.8998 ms**，最低耗时方案为 **38.4041 ms**，节省 **18.4957 ms**，耗时减少 **32.51%**，加速 **1.482×**。

主要收益来自 **缓存量化 scale** 和 **调整线程分工及 K/V 搬运路径**。之后的 FP8 批量解码、异步 PV 重叠属于小幅优化，收益存在波动。

## 1. 测试口径

- 硬件：用户服务器上的 H200；TileLang 版本为 0.1.15。
- 以下时间使用用户提供的 **CUPTI GPU kernel 耗时**。优化比较复用已预处理的 Q/K/V，包含该调用内的全部 kernels；原始 QKV 预处理和 Python API 耗时另行统计，因此这些数值不是端到端耗时。
- 脚本默认输入为 `B=1、H=28、Q=K=37741、D=128、BF16`；完整远端运行 JSON 未取得，实际输入形状未独立核验。
- 优化保留 per-thread INT8 QK、FP8 PV、128-key ExpCast 分块、decoded-P 分母和原来的 FP32 在线更新顺序；V 平滑与分组关闭。

## 2. 用了什么方法，分别降低到多少

下面每行均使用**同一次运行中的对照值**。各行来自不同轮次，不能将各步节省的毫秒数或百分比直接相加。

| 步骤 | 方法与候选 | 同轮耗时变化（ms） | 节省（ms） | 耗时减少 | 数据来源 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | 缓存量化 scale：`baseline_s1 → cached_s1` | 56.9038 → 49.4891 | 7.4147 | 13.03% | [第一轮](results/h200_tilelang_opt_reported.json) |
| 2 | 关闭自动 warp specialization、改变 K/V 搬运：`cached_s1 → cached_no_ws_s1` | 49.4806 → 39.4224 | 10.0582 | 20.33% | [第二轮](results/h200_tilelang_no_ws_reported.json) |
| 3 | FP8 批量解码：`cached_no_ws_s1 → cached_no_ws_vec_decode_s1` | 39.3917 → 38.7156 | 0.6761 | 1.72% | [第四轮](results/h200_tilelang_no_ws_round4_reported.json) |
| 4 | 在批量解码基础上重叠异步 PV：`cached_no_ws_vec_decode_s1 → vec_overlap` | 38.7171 → 38.4041 | 0.3130 | 0.81% | [第五轮复测](results/h200_tilelang_no_ws_round5_repeat_reported.json) |

演进过程可以概括为：**约 56.9 ms → 49.5 ms → 39.4 ms → 38.4 ms**。这些约数用于展示优化过程，整体降幅采用最新同轮的 `56.8998 → 38.4041 ms` 计算。

### 方法一：缓存 Q/K 的量化 scale

将 Q 的 scale 读取移到 key 循环外，在当前 query tile 内复用；同时显式缓存当前 key tile 的 K scale，供 score 反量化使用。

第一轮从 **56.9038 ms 降到 49.4891 ms**。原始实现中的部分标量重复读取已经被编译器合并，所以不能把收益解释为消除了源码里看到的所有 scale load；实际改动主要包括 Q scale 跨 key tile 的复用和相应调度变化。

### 方法二：关闭自动 warp specialization，改用 cp.async 搬运

原先编译器生成一个 256 线程 CTA，其中 128 线程负责生产数据，另外 128 线程负责计算。关闭自动 warp specialization 后，保留 128 线程计算 CTA，K/V 搬运从 TMA 改为 `cp.async`，INT8 QK 与 FP8 PV 都继续使用 WGMMA。

第二轮相对缓存版本，从 **49.4806 ms 降到 39.4224 ms**，单步节省 **10.0582 ms**，是这次优化中贡献最大的一步。线程分工、搬运方式和同步调度同时改变，收益属于整组改动，尚未分离各项贡献。

### 方法三：用 FP8x2 批量解码概率 P

ExpCast 得到 FP8 概率后，需要将其解码为 FP32 来计算行和。使用 `T.copy(prob, logits, coalesced_width=2)`，让编译器生成 FP8x2 批量转换，保持 P 的原始编码及分母归约顺序。

离线默认主形状中，静态 FP8 解码指令数由 **128 降到 64**，每线程寄存器数由 **245 降到 222**。这是本地交叉编译结果，不能直接等同于服务器上的实际指令调度或加速幅度。

实测收益较小：

| 测量轮次 | noWS（ms） | 批量解码（ms） | 耗时减少 |
| --- | ---: | ---: | ---: |
| 第四轮 | 39.3917 | 38.7156 | 1.72% |
| 第五轮初次 | 38.8468 | 38.6095 | 0.61% |
| 第五轮复测 | 38.7870 | 38.7171 | 0.18% |

### 方法四：将异步 PV 与标量计算重叠

在批量解码版本上发出异步 FP8 PV 后，利用等待窗口计算 P 的行和、更新分母、将历史累加值乘以 alpha；等待 PV 完成并执行寄存器 fence 后，再将新 PV 加到历史结果。K/V/P 的 shared buffer 在相应等待和同步完成后才复用。

最新复测从 **38.7171 ms 降到 38.4041 ms**。但上一次同轮比较是 **38.6095 → 38.7002 ms**，反而增加 **0.23%** 耗时。两次排序反转，缺少 trial 级波动数据，**38.4041 ms 应作为目前收到的最好记录，尚不能认定异步组合稳定更快**。

## 3. 尝试过但没有取得收益的方法

| 方法 | 同轮对照（ms） | 候选耗时（ms） | 结果 |
| --- | ---: | ---: | --- |
| INT8 QK 改用 `mma.sync`，FP8 PV 保持 WGMMA | 原始 baseline：56.9038 | 137.2563 | 明显回退；候选包含额外的 K 寄存器复制和同步，尚无各部分开销的独立测量 |
| 单独重叠异步 PV，使用原来的标量 P 解码 | noWS：39.1710 | 40.0123 | 耗时增加 2.15% |
| 整数 mass：把 decoded-P 行和改成整数单位求和 | noWS：39.1710 | 47.3561 | 耗时增加 20.90% |
| reduce4：重排最大值和行和的局部归约 | noWS：39.3917 | 40.3754 | 耗时增加 2.50% |
| 缓存版本将流水线 stage 从 1 改为 2 | cached_s1：49.4891 | 50.0617 | 该轮耗时增加 1.16%，没有观察到收益 |

以上前三行分别见[第一轮](results/h200_tilelang_opt_reported.json)、[第三轮](results/h200_tilelang_no_ws_next_reported.json)，reduce4 见[第四轮](results/h200_tilelang_no_ws_round4_reported.json)。

另外，两级流水线等部分方案仅做了离线检查：D128 的 noWS/批量解码双阶段尾形状出现寄存器 spill；warp-specialized 批量解码双阶段出现 64 B stack 和热循环 local-memory 写入。因此未纳入默认 H200 候选，也没有实测加速结论。

## 4. 最终与 C++ 基线的差距

第五轮复测的同场结果如下：

| 实现 | GPU kernel 耗时（ms） |
| --- | ---: |
| 原始 TileLang：`baseline_s1` | 56.8998 |
| 缓存并调整搬运：`cached_no_ws_s1` | 38.7870 |
| 再加批量解码：`cached_no_ws_vec_decode_s1` | 38.7171 |
| 再加异步 PV：`vec_overlap` | **38.4041** |
| Sage C++：`sage_cpp` | 24.4413 |
| FlashAttention 4：`flash4` | 30.7025 |

以该轮 TileLang 的最好记录计算：

- 相对原始 TileLang，节省 **18.4957 ms**，耗时减少 **32.51%**，加速 **1.482×**。
- 相对 Sage C++，多用 **13.9628 ms**，耗时为其 **1.571×**，即多 **57.13%**。
- 相对 FA4，多用 **7.7016 ms**，耗时为其 **1.251×**，即多 **25.08%**。

目前这份实现已进入小幅优化的平台期。上述结果描述的是当前内核与该测试配置，尚不能证明 TileLang 本身存在固定的性能上限。

## 5. 尚未计入收益的方案与验证状态

`vec_q128` 将 query tile 从 64 行扩大到 128 行，使用两个计算 warpgroups 共用 K/V，以减少重复加载。量化仍保持 BLKQ64，并增加尾行 scale 读取保护；候选独立的 lowering 修正了自动寄存器 fence 的长度。

该方案已通过离线检查，但**尚未收到包含 `vec_q128` 的 H200 测速结果，不能写成已经实现的性能收益**。

仓库记录已有 105 项本地检查通过、62 份源码完成 SM90a 交叉编译。本地 GPU 为 SM86，未执行这些 SM90a 内核。服务器能够输出测速表，按脚本流程意味着已走过其验证门槛；完整远端 JSON、详细误差和 trial 波动尚未独立核验。本次总结仅整理已有记录，没有新增 H200 测量。

完整实验过程见 [TILELANG_OPTIMIZATION.md](TILELANG_OPTIMIZATION.md)，离线编译与检查记录见 [tilelang_opt_cross_compile.json](results/tilelang_opt_cross_compile.json)。实现与独立测试入口分别为 [tilelang_expcast_opt.py](scripts/tilelang_expcast_opt.py) 和 [benchmark_tilelang_opt.py](scripts/benchmark_tilelang_opt.py)。
