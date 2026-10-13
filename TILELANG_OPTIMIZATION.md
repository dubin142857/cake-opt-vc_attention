# TileLang 独立优化实验

2026-10-10。新增 `scripts/tilelang_expcast_opt.py` 和独立入口
`scripts/benchmark_tilelang_opt.py`，针对现有 Hopper INT8 QK / ExpCast / FP8 PV。
原始 TileLang 实现继续作为对照。H200 同步修订后的测速摘要已返回：
最新同轮摘要中，`cached_no_ws_vec_decode_s1` 为已测 TileLang 的最低耗时方案，
后续实验以它作为已测对照；第六轮新增 `vec_q128`，尚未执行 H200 验证和测速。
Bash 入口运行 no_ws、vec_decode、`vec_q128` 三个方案，附加参数可覆盖指定选项；直接调用
Python 入口时，默认只比较已测的 no_ws 和 vec_decode。MMA 候选保留显式选择。

## 用户提供的 H200 结果

第五轮已返回以下同轮 CUPTI kernel 摘要；attention、all-kernel、all-GPU-work
三列逐行同值，没有混用 API 时间：

| 实现 | 时间（ms） |
| --- | ---: |
| `baseline_s1` | 56.8999 |
| `baseline_s2` | 57.0184 |
| `cached_no_ws_s1` | 38.8468 |
| `cached_no_ws_vec_decode_s1` | 38.6095 |
| `vec_overlap` | 38.7002 |
| `sage_cpp` | 24.4397 |
| `flash4` | 30.8387 |

vec_decode 相对同轮 no_ws 节省 0.2373 ms、耗时减少 0.611%；vec_overlap
相对 vec_decode 增加 0.0907 ms、耗时增加 0.235%，没有降低本轮对照时间。
vec_decode 仍是该表最低耗时的 TileLang 方案，相对 baseline_s1 加速
1.474×、耗时减少 32.145%；耗时比 Sage 多 57.979%、比 FA4 多 25.198%。
同轮比较记录在 `results/h200_tilelang_no_ws_round5_reported.json`。

用户报告远端结果保存为 `results/h200_tilelang_no_ws_round5.json`，完整文件
尚未读取，实际形状未核验。完成计时表明脚本已经经过验证门槛，这是流程推断，
详细精度指标和用例数量未提供。单轮摘要没有 trial 级波动和 raw trace，
不据此判断小差异的统计显著性。第五轮结果保留，当前启动脚本使用下面的第六轮配置。

第四轮历史 vec_decode/reduce4 的同轮 CUPTI kernel 摘要：

| 实现 | 时间（ms） |
| --- | ---: |
| `baseline_s1` | 56.9089 |
| `baseline_s2` | 57.0242 |
| `cached_no_ws_s1` | 39.3917 |
| `cached_no_ws_vec_decode_s1` | 38.7156 |
| `cached_no_ws_reduce4_s1` | 40.3754 |
| `sage_cpp` | 24.8511 |
| `flash4` | 30.5834 |

vec_decode 相对同轮 no_ws 加速 1.017×、节省 0.6761 ms，耗时减少 1.716%；
reduce4 相对 no_ws 增加 0.9837 ms、耗时增加 2.497%。vec_decode 相对
baseline_s1 加速 1.470×、耗时减少 31.97%，仍比 Sage 和 FA4 分别耗时多
55.79% 和 26.59%。vec_decode 是本轮已测最低耗时方案，作为后续对照；
这是一轮摘要，没有 trial 级波动数据，不宣称小差异具有统计显著性。

七行原值和同轮计算保存在 `results/h200_tilelang_no_ws_round4_reported.json`。
完整远端 JSON 尚未读取，形状只记录脚本默认值、尚未核验；完成测速表示
脚本已走过校验门槛的判断来自流程推断，详细精度指标和用例数量未提供。

此前一次启动返回的是旧候选表，没有 vec_decode/reduce4 的测量结果。
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

| 候选 | 改动 | 结果或待验证问题 |
| --- | --- | --- |
| `cached_s1/s2` | 每个 query tile 缓存 Q scale，每个 key tile 缓存 K scale，显式广播到 score fragment | 编译器是否已消除原实现的重复读取，以及新调度是否受益 |
| `mma_qk_s1/s2` | 在缓存 scales 的基础上，K 从 shared 复制到 fragment，使 INT8 QK 使用 `mma.sync`；FP8 PV 继续使用 WGMMA | QK 等待和布局成本的变化能否抵消额外 K 寄存器复制 |
| `cached_no_ws_s1` | 在 cached 基础上禁用自动 warp specialization，实际 CTA 从 256 线程变为 128；K/V 搬运同时从 TMA 改为 `cp.async`，两个 GEMM 保持 WGMMA | 第五轮 38.8468 ms；前轮相对 cached_s1 耗时减少 20.33%，不能单独归因于线程数 |
| `cached_no_ws_overlap_s1`（显式实验） | 异步 PV 与 decoded-P 行求和、denom 更新、历史 accum 乘 alpha 重叠；wait 后加 PV | 最新同轮 40.0123 ms，相对 no_ws 增时 2.148%，不默认 |
| `cached_no_ws_int_mass_s1`（显式实验） | 用整数单位计算每个 128-key tile 的 decoded-P 行求和，再精确转换为 FP32；PV 仍用 FP8 | 最新同轮 47.3561 ms，相对 no_ws 增时 20.896%，不默认 |
| `cached_no_ws_vec_decode_s1`（已测对照） | 用 `T.copy(prob, logits, coalesced_width=2)` 批量将 FP8 解码到 FP32 | 第五轮 38.6095 ms，相对同轮 no_ws 耗时减少 0.611%；本地主形状 PTX 解码指令从 128 降到 64 |
| `cached_no_ws_reduce4_s1`（显式实验） | 每线程先归约 8 个值，再合并 4 个 partials，最后进行 4-lane 归约 | 第四轮 40.3754 ms，相对同轮 no_ws 耗时增加 2.497%；保持最大值与 decoded-P 分母语义 |
| `vec_overlap`（显式实验） | 在 vec_decode 基础上，让异步 PV 与 decoded-P 行求和及 FP32 历史更新重叠，等待后加 PV | 第五轮 38.7002 ms，相对同轮 vec_decode 耗时增加 0.235%；详细一致性和误差指标未独立核验 |
| `vec_q128`（第六轮待测） | 在 vec_decode 基础上使用 BM128、256 线程、128-key tile 和候选独立的 BackendContext | 本地生成与交叉编译通过；H200 正确性和耗时尚未验证 |

除 `vec_q128` 使用 BM128 外，其余候选保持 BM64；所有候选保持 BN128、
128-key ExpCast 分块、per-thread INT8 QK、自然顺序
`[B,H,D,Npad]` FP8 V、decoded-P 分母和逐块独立 FP32 PV 累加。
V 平滑/分组关闭；直接 ExpCast 的 FP32 运算顺序与原实现一致。
FP16/BF16、D64/128 均支持；cached/MMA 支持 stage 1/2，no_ws 只支持 stage 1。
只用于非因果 MHA 和 Hopper SM90a。

新两个显式实验均基于 no_ws 单阶段实现。vec_decode 保留 P 的原始 FP8 编码
和分母归约顺序，只尝试用 FP8x2 转换替代逐元素转换。reduce4 对 score 最大值
与 decoded-P 求和分别重排归约布局；输出仍使用原来的 online softmax 更新。
两个方案已返回 H200 测速摘要，完整运行一致性和误差指标尚未独立核验；
离线生成和编译不能替代完整验证。
第五轮将 vec_decode 与安全的 PV 异步重叠组合成短名 `vec_overlap`，仅支持
单阶段。它保持原来的 FP8 P、解码分母和 FP32 历史累加语义，等待 PV 完成后
才读结果、复用 shared buffer。该组合已返回第五轮摘要，没有降低 vec_decode
对照的耗时；第六轮改测 `vec_q128`。此前较慢的 reduce4
保留显式选择，不进入本轮启动默认项。
vec_decode 的双阶段也未发布：离线 Q65/K257、D128、FP16/BF16 均为
255 registers、120 B stack，spill store/load 各 120 B。D64 双阶段虽能编译，
本轮也不提供该配置，所有 no_ws 方案与组合都继续使用单阶段。
另一个 Q128 试验在静态审查中发现自动寄存器 fence 超出实际 fragment 的范围，
该旧试验已撤回；vec_decode 和 reduce4 不受影响。

第六轮的短名 `vec_q128` 仅支持单阶段：BM128、256 线程、FullRow 布局，
key tile 仍为 128，继续使用 packed FP8 decode。QK 量化仍为原来的 BLKQ64，
QS 长度保持 `ceil(Q/64)*32`，读取 QS 时按真实 query 行判断尾部有效性。
候选使用独立的 BackendContext，在标准 lowering 后按物理 fragment 修正 fence，
不修改 TileLang 安装或全局 backend；原有候选源码保持不变。两个计算 warpgroups 完成等待并进行
CTA 同步后才复用 K/V shared buffer。本地静态检查不能证明 H200 输出正确
或性能改善，仍须完整运行验证和测速。

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
`cached_no_ws_vec_decode_s1`、`vec_q128`，并保留同场的
Sage attention、FlashAttention-4 和原始 TileLang stage 1/2 对照。
脚本先执行完整精度校验及默认/非默认 stream 各三次一致性检查，通过后测
完整 API、prepared 核心 API，并单独运行 CUPTI kernel 计时。

结果自动写到 `results/h200_tilelang_no_ws_round6.json`，生成源码和 traces
放到 `results/tilelang_no_ws_round6_traces/`。脚本复用已有环境，不安装依赖，
不构建项目的 C++ 扩展；启动本轮测试无需输入长参数。
后续迭代更新脚本内部的候选与结果路径，常规启动始终使用上述单行 Bash 命令。

脚本始终先设置本轮候选、JSON 和 trace 路径，再附加用户参数。只加
`--device`、`--validate-only` 或外部库开关时，本轮配置保持不变；显式给出
`--variants`、`--output`、`--trace-dir` 可分别覆盖对应项。启动时会显示
实际候选和结果文件，便于确认配置。直接调用 Python 入口的默认候选为
`cached_no_ws_s1` 和 `cached_no_ws_vec_decode_s1`；`vec_q128` 和 `vec_overlap` 需显式选择。
本轮三个方案只支持单阶段，精度门槛、逐次一致性检查和 CUPTI 口径与前轮相同。
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
traces 放在 `results/tilelang_opt_traces/`；Bash 入口默认使用上述 round6 路径，
附加参数会保留未覆盖的默认项。
并发的 C++ 实验应使用不同 GPU 或错开性能测量，
避免资源竞争使时间失真。可用 `--output`、`--trace-dir` 给每轮设置独立路径。

## 校验和计时

测速前执行 32 个小尺寸用例，覆盖两种 dtype、两个 D、单 token、Q/K 尾块、
多 key tile 和各块 V 不同的均匀 attention。独立解码参考检查所有 query 行；
V 的逻辑 FP8 字节和 scales 用另一条 Sage 预处理路径逐位核验。
原有 24 个用例保持原来的 RNG 输入，新增 random Q128/K257、Q129/K257 的
八个边界用例；新增用例使用独立 seed 3334，每个 dtype/D 组合重置生成器。
`tail65_129` 仍只选择原来的八个用例，输入与完整运行相同。
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

- 36 项 CPU 辅助测试：V inverse permutation 的全部 FP8 bit patterns、CLI、候选隔离、
  尾块差异定位、严格一致性失败记录和缩短验证时的随机输入复现。
  新增 Q128/Q129 边界用例及其独立 RNG 不影响原有输入的检查。
  其中 3 项通过真实 Bash 启动检查无参数默认、附加参数保留本轮配置、
  显式覆盖及含空格路径。测试从项目外目录启动，仅用 stub 替代 Python，
  要求 benchmark 脚本实际存在，不调用 CUDA、无需 GPU。
- 44 项候选代码生成测试：cached/MMA 的两种 dtype、D64/128、stage 1/2，
  以及七个基于 no_ws 的单阶段候选的四个组合；另有七项拒绝双阶段配置
  和四项 Q128 边界检查。
  检查低精度 GEMM 类型、Q scale 读取位置和 K/V buffer 释放时序。
- 原始 TileLang 的 8 项代码生成回归测试。
- Q128 独立 lowering 的 6 项物理 TIR 安全拒绝与幂等 metadata 测试。
- 共 62 份公开候选源码经 CUDA 12.4 NVRTC 和 ptxas 交叉编译为 SM90a cubin，
  stack/spill 均为 0。原有 38 份源码 hash 保持不变，包括 baseline 和
  cached/MMA 的 24 份小形状、no_ws/overlap/integer mass 的 12 份小形状，
  以及 overlap/integer mass 的两份主形状。前 24 份为 240 registers；
  no_ws/overlap 小形状 D64/D128 为 168/247，integer mass 为 196/254。
  新增 vec_decode/reduce4 的 8 份小形状和两份主形状均已完成编译：
  vec_decode 小形状 D64/D128 为 168/221 registers，reduce4 为 168/247，
  两种 dtype 相同；BF16/D128 默认主形状分别为 222/246，原 no_ws 为 245。
  第五轮 `vec_overlap` 新增的 4 份小形状和一份主形状全部编译通过：
  小形状 D64/D128 为 168/221，BF16/D128 主形状为 222，均无 stack/spill。
  第六轮新增 `vec_q128` 的 9 份源码，旧 53 份源码的 SHA 均未改变。
  D64 尾块为 167 registers；D128 的 Q65/Q128/Q1 为 239，
  Q129/Q257 为 233，BF16/D128 默认主形状为 238，全部无 stack/spill。
  Windows NVRTC 检查使用标准整数头和 `cuda::std` 兼容包装；这不替代远端 NVCC 和运行校验。

36 项 CPU 辅助、63 项代码生成/配置拒绝/原始回归和 6 项 helper 测试，
共 105 项通过。原有 53 份源码的 SHA 均未改变，离线编译记录保留；
加第六轮 Q128 的 9 份，共 62 份公开候选源码。
主形状 PTX 中，no_ws、vec_decode、reduce4 的 FP8 解码指令计数分别为
128、64、128；三者 shuffle 计数均为 16。vec_decode 确实减少了静态解码
指令，reduce4 没有减少 shuffle 数量；这些统计不能直接推导 H200 加速幅度。
`vec_overlap` 主形状继续保持 64 条解码指令和 16 条 shuffle，H200 第五轮摘要
已返回，详细运行一致性和误差指标尚未独立核验。

本轮两个 probe 未公开：BM64/256 的 FullCol 主形状为 148 registers，
没有提高寄存器容量限制下的驻留 warp 上限；warp-specialized `vec_s2`
热循环含 local-memory 写入、产生 64 B stack，因此也未加入候选入口。

记录在 `results/tilelang_opt_cross_compile.json`，本地源码/PTX/cubin 放在
`results/tilelang_opt_local_sources/`（生成文件不纳入 Git）。

测试命令（已有 TileLang 0.1.15 的 Python）：

```bash
python -m pytest tests/test_tilelang_opt_benchmark.py \
  tests/test_tilelang_opt_codegen.py tests/test_tilelang_expcast_codegen.py \
  tests/test_tilelang_q128_lowering.py -q
```

另试过 `reduce8` 的归约布局，但 TileLang 0.1.15 生成了未初始化值；
该候选已撤掉，不在远端脚本中。最终速度与运行正确性以 H200 JSON 为准。
