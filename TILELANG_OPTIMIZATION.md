# TileLang 独立优化实验

2026-10-10。新增 `scripts/tilelang_expcast_opt.py` 和独立入口
`scripts/benchmark_tilelang_opt.py`，针对现有 Hopper INT8 QK / ExpCast / FP8 PV。
原始 TileLang 实现继续作为对照。H200 同步修订后的测速摘要已返回：
`cached_no_ws_s1` 为当前最快的 TileLang 候选。Bash 入口自动运行本轮
no_ws、vec_decode、reduce4 三个方案，附加参数可覆盖指定选项；直接调用 Python
入口时，默认候选仍为
`cached_s1/s2` 和 `cached_no_ws_s1`。MMA 候选保留显式选择。

## 用户提供的 H200 结果

最近一次启动返回的是旧候选表，没有 vec_decode/reduce4 的测量结果。
该轮 no_ws 为 44.8870 ms，同轮 baseline_s1 为 58.7920 ms，CUPTI kernel
时间比为 1.310×。七行摘要保存在
`results/h200_tilelang_launch_failure_reported.json`，用户报告远端保存的是
`results/h200_tilelang_opt.json`；实际形状和完整 JSON 尚未核验。
随后参数行出现 Bash `command not found`。远端调用的原始字节尚未核实，
原因未确认，不能归因于 CRLF。该次输出不代表新两个候选已完成测量。

第三轮 no_ws 历史测速中，同轮 CUPTI 的全部 kernel 时间如下，没有混用 API 时间。

| 实现 | 时间（ms） |
| --- | ---: |
| `baseline_s1` | 57.2842 |
| `baseline_s2` | 57.3722 |
| `cached_no_ws_s1` | 39.1710 |
| `cached_no_ws_overlap_s1` | 40.0123 |
| `cached_no_ws_int_mass_s1` | 47.3561 |
| `sage_cpp` | 24.8608 |
| `flash4` | 30.8192 |

`cached_no_ws_s1` 仍是本轮最快的 TileLang 方案，相对 baseline_s1 加速
1.462×、耗时减少 31.62%。overlap 相对它增加 2.148% 耗时，integer mass
增加 20.896%；两个实验保留显式选择，均不进入默认候选。完整七行摘要与
同轮比较在 `results/h200_tilelang_no_ws_next_reported.json`。

这些时间来自用户提供的摘要。用户报告服务器保存了
`results/h200_tilelang_no_ws_next.json`，尚未读取其完整内容；实际形状、
详细精度指标和 trial 波动未独立核验。完成测速意味着脚本已走过校验门槛，
这是依据脚本流程和摘要作出的推断，不能当作独立验证结果。

前一轮 no_ws 历史结果如下：

| 实现 | 时间（ms） |
| --- | ---: |
| `baseline_s1` | 56.9145 |
| `baseline_s2` | 57.0217 |
| `cached_s1` | 49.4806 |
| `cached_s2` | 50.0607 |
| `cached_no_ws_s1` | 39.4224 |
| `sage_cpp` | 24.8074 |
| `flash4` | 30.2784 |

no_ws 相对同轮 baseline_s1 加速 1.444×、耗时减少 30.73%；相对 cached_s1
再加速 1.255×、耗时减少 20.33%。耗时仍为 Sage 的 1.589 倍、FA4 的 1.302 倍。
完整七行摘要及比较计算在 `results/h200_tilelang_no_ws_reported.json`，
上一轮保留在 `results/h200_tilelang_opt_reported.json`。两个摘要都来自用户提供的
终端输出，没有读取服务器的完整 JSON；实际输入形状、校验误差和 trial 波动尚未
独立核验。新候选完成测速意味着该轮脚本已经走过其校验门槛。

第一轮的历史结果如下：

| 实现 | 时间（ms） | 相对同 stage baseline |
| --- | ---: | ---: |
| `baseline_s1` | 56.9038 | 1.000× |
| `cached_s1` | 49.4891 | 1.150× 加速，耗时减少 13.03% |
| `cached_s2` | 50.0617 | 1.139× 加速，耗时减少 12.24% |
| `mma_qk_s1` | 137.2563 | 耗时为 baseline_s1 的 2.412 倍 |
| `sage_cpp` | 25.0549 | 外部对照 |
| `flash4` | 30.7743 | 外部对照 |

`cached_s1` 耗时仍为 Sage 的 1.975 倍、FA4 的 1.608 倍。这是单轮结果，
不能据此认定 stage 1 比 stage 2 稳定快。完整八行摘要和计算方式保存在
`results/h200_tilelang_opt_reported.json`。未取得完整运行 JSON，实际输入形状、
误差指标和每轮波动尚未独立核验；完成测速意味着脚本已走过其校验门槛。

## 候选

| 候选 | 改动 | 需要实测的问题 |
| --- | --- | --- |
| `cached_s1/s2` | 每个 query tile 缓存 Q scale，每个 key tile 缓存 K scale，显式广播到 score fragment | 编译器是否已消除原实现的重复读取，以及新调度是否受益 |
| `mma_qk_s1/s2` | 在缓存 scales 的基础上，K 从 shared 复制到 fragment，使 INT8 QK 使用 `mma.sync`；FP8 PV 继续使用 WGMMA | QK 等待和布局成本的变化能否抵消额外 K 寄存器复制 |
| `cached_no_ws_s1` | 在 cached 基础上禁用自动 warp specialization，实际 CTA 从 256 线程变为 128；K/V 搬运同时从 TMA 改为 `cp.async`，两个 GEMM 保持 WGMMA | 最新同轮 H200 39.1710 ms；前一轮相对 cached_s1 耗时减少 20.33%，不能单独归因于线程数 |
| `cached_no_ws_overlap_s1`（显式实验） | 异步 PV 与 decoded-P 行求和、denom 更新、历史 accum 乘 alpha 重叠；wait 后加 PV | 最新同轮 40.0123 ms，相对 no_ws 增时 2.148%，不默认 |
| `cached_no_ws_int_mass_s1`（显式实验） | 用整数单位计算每个 128-key tile 的 decoded-P 行求和，再精确转换为 FP32；PV 仍用 FP8 | 最新同轮 47.3561 ms，相对 no_ws 增时 20.896%，不默认 |
| `cached_no_ws_vec_decode_s1`（新显式实验） | 用 `T.copy(prob, logits, coalesced_width=2)` 批量将 FP8 解码到 FP32 | 本地主形状 PTX 的解码指令从 128 降到 64；H200 运行一致性和速度尚待验证 |
| `cached_no_ws_reduce4_s1`（新显式实验） | 每线程先归约 8 个值，再合并 4 个 partials，最后进行 4-lane 归约 | 缩短 max/sum 串行链，保持最大值与 decoded-P 分母语义；H200 尚待验证 |

所有候选保持 BM64/BN128、128-key ExpCast 分块、per-thread INT8 QK、自然顺序
`[B,H,D,Npad]` FP8 V、decoded-P 分母和逐块独立 FP32 PV 累加。
V 平滑/分组关闭；直接 ExpCast 的 FP32 运算顺序与原实现一致。
FP16/BF16、D64/128 均支持；cached/MMA 支持 stage 1/2，no_ws 只支持 stage 1。
只用于非因果 MHA 和 Hopper SM90a。

新两个显式实验均基于 no_ws 单阶段实现。vec_decode 保留 P 的原始 FP8 编码
和分母归约顺序，只尝试用 FP8x2 转换替代逐元素转换。reduce4 对 score 最大值
与 decoded-P 求和分别重排归约布局；输出仍使用原来的 online softmax 更新。
两个方案的 H200 运行一致性和速度均未测，离线生成和编译不能替代完整验证。
另一个 Q128 试验在静态审查中发现自动寄存器 fence 超出实际 fragment 的范围，
已撤回、不提供 CLI；vec_decode 和 reduce4 不受影响。

no_ws 的 Q65/K257 离线编译中，D64 使用 168 registers、D128 使用 247，
无 stack/spill；双阶段试验产生了 spill，因此不提供该配置。单阶段已经返回 H200
测速摘要。源码检查要求 `cp.async` 完成后再读 K/V，且 WGMMA 完成和线程同步后
才覆盖相应 shared buffer；这仍不能代替 GPU 运行一致性检查。
默认主形状 B1/H28/Q37741/K37741/D128/BF16 也已离线编译：单阶段为
245 registers、48 KiB 动态 shared、无 stack/spill，长循环的等待顺序同样通过检查。

overlap 在 `T.wait_wgmma(0)` 后显式设置每线程 32/64-register operand fence，
并用 V/P 的 pointer read 保持 shared buffer 生命周期，等待和线程同步后才覆盖 V。
每块 PV 保持 `clear_accum=True`；历史缩放和 PV 相加仍分别按原 FP32 RN 顺序执行。
与旧的 warp-specialized overlap 不同，此实验基于已经实测的 no_ws 搬运路径。

整数 mass 的适用范围是 ExpCast 的非负 code 0..120、每块最多 128 个值。
这些 FP8 值均为 2^-9 的整数倍，最大 units 为 131072，128 项之和不超过 2^24。
任意 FP32 加法树的中间值都可精确表示，因此 `float(intsum)*2^-9` 与原 decoded-P
FP32 行求和逐位一致。跨 tile 的在线 denom 更新保持原 FP32 操作。
Torch 实际 FP8 bit patterns 0..120 和 10,125 行边界/随机输入已经检查，
依据保存在 `results/tilelang_int_mass_proof.json`。这不替代完整 attention 的 H200 校验。

本地 D128/BF16/s1 的 PTX 确认：原实现 QS/KS 均在 key 循环中加载，
`cached` 把 QS 移到循环前。但 NVRTC 已把原实现的逐元素重复读取合并，
所以改善的是每线程 QS 随 key tile 重复加载，不能按源码估计大量 load 被消除。
最终 SASS 和运行时间仍需实测。

## H200 首轮失败与同步修订

提交 `2ba6299` 的 `mma_qk_s1` 在 FP16、D128、Q65/K129 上通过了相对误差门槛，
但非默认 stream 的输出与默认 stream 不逐位一致：115/16640 个元素不同，
最大绝对误差 0.00054931640625，最大差异位于 head 1 的 query 64。
这属于运行一致性问题，不能通过放宽精度门槛接受。

本地检查没有发现 benchmark 输入被释放或缺失跨流等待；源码和 PTX 中的
K barrier 也不足以确认根因。现采用更保守的 MMA 调度：将 K 的 empty-barrier
释放延到全部 QK MMA 和显式 128-consumer 同步之后，用指针读取依赖保持 K 的
生命周期，并设置编译器 memory fence。生产者不参与该同步，避免全 CTA 同步死锁。
修订后用户给出了包含两种 MMA stage 的完整测速摘要，说明本次脚本走过了
运行一致性检查。没有详细校验 JSON，仍不能认定已经确定初次失败的根因。

验证增加同一 prepared payload 在默认 stream、非默认 stream 各三次调用，
每次都要求与首次输出逐位一致；失败保存具体 head/query/channel 和行差异分布。
先针对本次尾块形状回归：

```bash
bash scripts/run_tilelang_opt_h200.sh --validate-only \
  --validation-case tail65_129 --variants mma_qk_s1 mma_qk_s2 \
  --output results/h200_tilelang_mma_tail_regression.json
```

该命令检查 FP16/BF16、D64/128、random/uniform 的八个 Q65/K129 用例，
并包含相同 stage 的原始 TileLang 对照。跳过案例仍按原顺序推进随机生成器，
保证选中案例的输入与完整运行相同。通过后仍需默认完整验证和测速。

## 在已有 H200 环境运行

已有 `.venv-tilelang-h200`、TileLang 0.1.15、Torch、Sage 量化组件和 CUDA 编译环境时：

```bash
bash scripts/run_tilelang_opt_h200.sh
```

Bash 启动会自动选择本轮三个方案：`cached_no_ws_s1`、
`cached_no_ws_vec_decode_s1`、`cached_no_ws_reduce4_s1`，并保留同场的
Sage attention、FlashAttention-4 和原始 TileLang stage 1/2 对照。
脚本先执行完整精度校验及默认/非默认 stream 各三次一致性检查，通过后测
完整 API、prepared 核心 API，并单独运行 CUPTI kernel 计时。

结果自动写到 `results/h200_tilelang_no_ws_round4.json`，生成源码和 traces
放到 `results/tilelang_no_ws_round4_traces/`。脚本复用已有环境，不安装依赖，
不构建项目的 C++ 扩展；启动本轮测试无需输入长参数。
后续迭代更新脚本内部的候选与结果路径，常规启动始终使用上述单行 Bash 命令。

脚本始终先设置本轮候选、JSON 和 trace 路径，再附加用户参数。只加
`--device`、`--validate-only` 或外部库开关时，本轮配置保持不变；显式给出
`--variants`、`--output`、`--trace-dir` 可分别覆盖对应项。启动时会显示
实际候选和结果文件，便于确认配置。直接调用 Python 入口的默认候选仍为
`cached_s1/s2` 和 `cached_no_ws_s1`。vec_decode/reduce4 只支持单阶段，H200 正确性和速度
待本轮验证；精度门槛、逐次一致性检查和 CUPTI 口径与前轮相同。
仅运行本轮 TileLang 候选时，可省略外部库对照：

```bash
bash scripts/run_tilelang_opt_h200.sh --skip-sage --skip-flash4
```

自定义 Python、GPU、捕获的真实 QKV 或候选范围：

```bash
TILELANG_PYTHON=/path/to/python3 bash scripts/run_tilelang_opt_h200.sh \
  --device cuda:1 --qkv-file /path/to/qkv.pt --qkv-layout NHD \
  --variants cached_s1 mma_qk_s1
```

QKV 文件需为 `torch.save(dict(q=q, k=k, v=v), path)`，FP16/BF16 MHA，
在 RoPE 后捕获。布局 `NHD` 表示 `[B,S,H,D]`，`HND` 表示 `[B,H,S,D]`。
不提供文件时默认随机输入为 B1/H28/Q37741/K37741/D128/BF16。

Python 入口的默认输出为 `results/h200_tilelang_opt.json`，生成源码和 CUPTI
traces 放在 `results/tilelang_opt_traces/`；Bash 入口默认使用上述 round4 路径，
附加参数会保留未覆盖的默认项。
并发的 C++ 实验应使用不同 GPU 或错开性能测量，
避免资源竞争使时间失真。可用 `--output`、`--trace-dir` 给每轮设置独立路径。

## 校验和计时

测速前执行 24 个小尺寸用例，覆盖两种 dtype、两个 D、单 token、Q/K 尾块、
多 key tile 和各块 V 不同的均匀 attention。独立解码参考检查所有 query 行；
V 的逻辑 FP8 字节和 scales 用另一条 Sage 预处理路径逐位核验。
候选与相同 stage 的原始 TileLang 全行比较，relative RMSE 必须 <=0.1%；
每个实现与解码参考必须 <=1%，并检查非默认 stream 的 prepared 调用逐位一致。

主尺寸同样检查 full/prepared 逐位一致、候选与 stage 对照的全部 query 行。
主尺寸的解码参考和 dense FP32 参考默认抽样 128 行、使用全部 keys。
dense FP32 误差仅作为算法精度诊断。失败会保存当前用例和生成源码，再停止测速。

完整调用每次从原始 QKV 重新预处理，核心调用复用 prepared tensors；
这两种 API 时间分别统计。各 trial 随机打乱测试顺序；编译和校验在计时外。
CUPTI attention / all-kernel / all-GPU-work 时间另报，不与 API 时间混用。
`--validate-only` 只跑校验；`--skip-cupti` 关闭 profiler；
`--include-overlap` 可额外重测此前的 overlap 调度。

## 本地已完成的检查

本地 GPU 是 RTX 3060 Laptop（SM86），没有执行 SM90a 内核。以下检查已经通过：

- 30 项 CPU 辅助测试：V inverse permutation 的全部 FP8 bit patterns、CLI、候选隔离、
  尾块差异定位、严格一致性失败记录和缩短验证时的随机输入复现。
  其中 3 项通过真实 Bash 启动检查无参数默认、附加参数保留本轮配置、
  显式覆盖及含空格路径。测试从项目外目录启动，仅用 stub 替代 Python，
  要求 benchmark 脚本实际存在，不调用 CUDA、无需 GPU。
- 36 项候选代码生成测试：cached/MMA 的两种 dtype、D64/128、stage 1/2，
  以及五个 no_ws 单阶段候选的四个组合；另有五项拒绝 no_ws 双阶段配置。
  检查低精度 GEMM 类型、Q scale 读取位置和 K/V buffer 释放时序。
- 原始 TileLang 的 8 项代码生成回归测试。
- 共 48 份源码经 CUDA 12.4 NVRTC 和 ptxas 交叉编译为 SM90a cubin，
  stack/spill 均为 0。原有 38 份源码 hash 保持不变，包括 baseline 和
  cached/MMA 的 24 份小形状、no_ws/overlap/integer mass 的 12 份小形状，
  以及 overlap/integer mass 的两份主形状。前 24 份为 240 registers；
  no_ws/overlap 小形状 D64/D128 为 168/247，integer mass 为 196/254。
  新增 vec_decode/reduce4 的 8 份小形状和两份主形状均已完成编译：
  vec_decode 小形状 D64/D128 为 168/221 registers，reduce4 为 168/247，
  两种 dtype 相同；BF16/D128 默认主形状分别为 222/246，原 no_ws 为 245。
  Windows NVRTC 检查使用标准整数头和 `cuda::std` 兼容包装；这不替代远端 NVCC 和运行校验。

CPU 辅助、候选代码生成、配置拒绝和原始回归共 79 项测试通过。
本次只调整启动入口和检查，kernel 源码未改，原有 48 份离线编译记录继续保留。
主形状 PTX 中，no_ws、vec_decode、reduce4 的 FP8 解码指令计数分别为
128、64、128；三者 shuffle 计数均为 16。vec_decode 确实减少了静态解码
指令，reduce4 没有减少 shuffle 数量；这些统计不能直接推导 H200 加速幅度。

记录在 `results/tilelang_opt_cross_compile.json`，本地源码/PTX/cubin 放在
`results/tilelang_opt_local_sources/`（生成文件不纳入 Git）。

测试命令（已有 TileLang 0.1.15 的 Python）：

```bash
python -m pytest tests/test_tilelang_opt_benchmark.py \
  tests/test_tilelang_opt_codegen.py tests/test_tilelang_expcast_codegen.py -q
```

另试过 `reduce8` 的归约布局，但 TileLang 0.1.15 生成了未初始化值；
该候选已撤掉，不在远端脚本中。最终速度与运行正确性以 H200 JSON 为准。
