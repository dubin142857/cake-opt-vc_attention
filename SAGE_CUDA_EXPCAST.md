# H200 CUDA 主线：ExpCast 调度候选

2026-10-10。当前主线是 **INT8 QK + ExpCast + FP8 PV，V 聚类和平滑关闭**。
保留 vc28 已测实现，新候选不修改已安装包和默认 dispatch。

## 最新结论：paired QK 未获得收益，保留 K128 预取基线

本轮 H200 CUPTI 回传：

| 实现 | GPU 内核 ms |
|---|---:|
| vc28_scale_prefetch | 23.4789 |
| vc28_paired_qk | 24.5014 |
| vc28_installed | 23.6508 |
| sage_cpp | 24.8859 |
| flash4 | 30.5801 |

paired QK 比预取基线增加 **1.0225 ms（4.35%）**，不提升为默认实现。
继续保留 K128 预取路线，其同轮 Sage/FA4 内核加速比分别为 **1.060× / 1.302×**。
原始汇总见 [回传记录](results/h200_cuda_expcast_paired_qk_reported.json)。
随后回传的资源和配对结果进一步确认：

- 同输入输出 `exact_match=true`，RMSE/max-abs 均为 0。
- paired QK 在 full API 和 fixed core 中均 **0/9 轮更快**；配对耗时增加中位数分别
  为 **1.0978 ms / 1.1413 ms**，按各自 median 比值换算分别慢 **4.36% / 4.82%**。
  fixed core 排除了预处理，因此退步也存在于核心计算路径。
- 远端寄存器 **168 → 240**，动态共享内存 **40,960 → 57,344 B**，
  两者 static shared 均 128 B，`local_bytes` 均为 0。
- 预取 JIT 对 installed 的 full API 配对测试 **9/9 更快**，节省中位数 0.3941 ms；
  fixed core 为 7/9 更快，节省中位数 0.3886 ms。仅描述这次运行，不外推所有场景。

按 [NVIDIA Hopper 的每 SM 64K 个 32 位寄存器](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy)
及本内核每块 128 线程计算，寄存器预算允许的驻留上限由 **3 块降到 2 块**。
这是预算估算，不是实际 occupancy/utilization 计数；尚不能量化它对退步的贡献。
证据支持停止扩大 QK tile，后续以 K128 预取版为基线，优先检查概率处理的依赖链和
临时寄存器生命周期；新候选需先检查资源占用和逐位一致性，再进行完整内核测试。
这些 API/内核测量和精度门限均不代表完整模型推理收益或模型质量。

## 已测候选：两个 QK 块合并，ExpCast/PV 保留 K128 边界

最新 H200 CUPTI 回传：scale-prefetch **23.4322 ms**，mass-tail **23.4445 ms**，
installed **23.5268 ms**，Sage **24.8497 ms**，FA4 **30.9068 ms**。
mass-tail 慢 0.0123 ms，约 0.05%，没有显示收益，继续以 scale-prefetch 为基线。
[回传记录](results/h200_cuda_expcast_mass_tail_reported.json) 只包含用户提供的汇总数值，
不能据此推断完整 API 时间、统计显著性或模型质量。

此前局部求和、max 树和打包调度收益有限。这轮试验改变 QK 的计算块大小：

- 对两个连续 MAIN K128 块，一次发起 `m64n256k32.s32.s8.s8` QK 批次。
  两块的点积计算量不变，QK 提交/等待和 K 的 TMA 搬运次数按块对减少。
- 分数依次进入原有两次 K128 ExpCast 和 FP8 PV。每块仍用自己的 K scale，
  保留 online max、decoded-P FP32 分母、历史缩放和两级 PV 累加。
- 最后 1–2 个逻辑块使用原 N128 QK；最终块保留原来的浮点转换和 mask 路径。
  不把两个块合成一次 K256 softmax，以免改变概率量化和归一化语义。

主要代价是同时保留两块 QK 分数。本地 CUDA 12.4 SM90a 编译中，候选用
**240 寄存器/线程**，D128 动态共享内存 **57,344 B**；基线为 168 寄存器和
40,960 B。初版 cap224 有少量 spill，改为 cap240 后八个 attention 实例均无
stack/spill；这会减少可驻留线程块，**不保证更快**。
反汇编已确认生成真实 N256 INT8 MMA。编译和反汇编不能替代 H200 运行验证。

```bash
git pull --ff-only origin master
bash scripts/run_paired_qk_h200.sh
```

默认 BF16、NHD `[1,37741,28,128]`，INT8 QK + ExpCast + FP8 PV，V 聚类/平滑关闭。
使用现有 Torch、Sage local、FA4 和 nvcc；首次编译独立 JIT，不改已安装包，
不要求 ncu/nvdisasm。输出 `results/h200_cuda_expcast_paired_qk.json`。
可追加 `--validate-only`，或 `--qkv-file ... --qkv-layout NHD` 测真实输入。

| 行名 | 含义 |
|---|---|
| vc28_scale_prefetch | 同次 JIT 编译的预取基线 |
| vc28_paired_qk | N256 QK + 两次原 K128 ExpCast/PV |
| vc28_installed | 已安装 vc28 |
| sage_cpp / flash4 | 外部对照 |

先运行 6 个 N256 QK 的整数 CPU oracle 检查，独立确认累加寄存器与矩阵列的对应；
再跑 60 个分块/scale 检查和原有 28 个 attention 配置。覆盖 K 长度
1、127、128、129、255、256、257、383、384、385、511、512、513、769、37741，
FP16/BF16、D64/D128、多 batch/head 和非默认 stream，尾部填充 V 故意置为非零。
相对预取基线要求**所有输出逐位相等**，decoded reference 保留 1% 实现门限。
任一失败立即停止；成功后跑 9 轮 API 和 5 轮 CUPTI，二者分开报告。
精度门限验证实现一致性，不代表真实模型质量。

本地 37 项 Python/生成代码检查通过，完整 SM90a attention 与独立 QK 检查交叉编译通过。
本机只有 SM86，未在本地运行 Hopper attention；现已收到上方 H200 性能汇总。
详细本地编译资源见
[本地编译记录](results/cuda_expcast_paired_qk_local_validation.json)。

## 已测候选：保留一次 PV 提交，延后少量 FP32 分母计算

最新 H200 回传：预取基线 **23.4155 ms**，prefix-overlap **23.5372 ms**，
prefix-serial **23.6871 ms**，installed **23.6530 ms**，Sage **24.8849 ms**，
FA4 **30.8784 ms**。重叠版比串行版快 0.1499 ms，但仍比预取版慢 0.1217 ms。
因此保留预取版，不采用上轮 prefix 拆分。原始数值见
[用户回传记录](results/h200_cuda_expcast_pv_prefix_reported.json)，没有补造 API 或原始 trial 数据。

本地 CUDA 12.4 反汇编显示：prefix 串行/重叠版在 MAIN 区域比基线多出四次高字节右移，
重叠版还多一次 WGMMA fence，并把一次 PV 提交变成两次。
这是可确认的结构开销，不能单凭静态计数将 H200 的 0.1217 ms 全部归因于它们。

本轮回到预取基线，先完成所有 P 编码和 FP8 解码，只保留最后四对已解码的 `half2`。
这些值仅用作精确表示，**所有加法仍是 FP32**。提交四次 PV 后再完成最后几项
FP32 转换/求和，不访问正在被 WGMMA 使用的 P 或 PV 累加寄存器。
保持一次 PV 提交，无额外 fence，无共享内存概率副本。

局部分母的加法分组发生变化，但这一变化是精确的：每线程每行 32 个概率值均为
`2^-9` 的整数倍，最大值 256，任意部分和最多 `2^22` 个整数单位，FP32 可精确表示。
历史缩放、跨 tile 累加、P 字节和尾块处理保持原实现。fallback 用原算法算完整分母，
随后补的四对值为零。

```bash
git pull --ff-only origin master
bash scripts/run_mass_tail_h200.sh
```

默认 BF16、NHD `[1,37741,28,128]`，INT8 QK + ExpCast + FP8 PV，V 聚类/平滑关闭。
使用已有 Torch、Sage local、FA4 和 nvcc；首次运行编译独立 JIT 模块，不重新安装包，
不要求 ncu/nvdisasm。输出 `results/h200_cuda_expcast_mass_tail.json`，
9 轮 API、5 轮 CUPTI。可追加 `--validate-only` 或真实 QKV 参数。

| 行名 | 含义 |
|---|---|
| vc28_scale_prefetch | 上轮最快预取版，同次编译的基线 |
| vc28_mass_tail | 延后最后四对解码值的 FP32 转换/求和，PV 保持单批提交 |
| vc28_installed | 已安装 vc28 |
| sage_cpp / flash4 | 外部对照 |

仍先跑 38 个概率状态检查、28 个不同 scale 检查、8 个混合 warp fallback 检查和
28 个小尺寸 attention 配置。候选相对预取基线的完整输出须逐位一致；decoded reference
保留 1% 实现门限，主尺寸检查所有输出行。任一失败即停止计时。
full/core API、配对差值和 CUPTI 分别记录，不混算加速比。

本地 **139 项测试通过**，包含新增 39 项 GPU 概率检查；本机 SM86 使用软件 FP8 解码。
FP16/BF16 × D64/D128 × 两条路径共 8 个完整 SM90a kernel 交叉编译通过，
D128 为 168 寄存器，D64 为 127，均无 stack/spill。
预取基线的 D128 BF16 指令文本与上轮相同。新候选的八次 FP32 转换和八次加法
已被编译器排入 PV 指令之间，保留单次提交，未增加 fence/wait。
这仅证明静态调度发生变化；CUDA 13 的调度可能不同。现已收到上方 mass-tail GPU 时间汇总，
未显示收益；用户尚未提供该轮完整正确性报告或原始 trials。
证据见 [mass-tail 本地检查](results/cuda_expcast_mass_tail_local_validation.json)。

## 已测候选：最后一组概率解码/求和与前三次 PV 重叠

上轮 H200 CUPTI：scale-prefetch **23.4629 ms**，同次 JIT vc28 **23.6402 ms**，
installed vc28 **23.6394 ms**，Sage **24.8479 ms**，FA4 **30.6937 ms**。
预取候选节省 0.1773 ms，约 0.75%；这是小幅正向结果，目前只有 GPU 时间汇总，
没有据此推断 full API 收益或跨轮稳定性。

这次以 scale-prefetch 为基线，只调整 MAIN tile 中概率分母与 PV 的执行顺序。
一个 K128 tile 的 PV 分成四次 K32 WGMMA：

- 先编码全部 P 字节，完成前三组 P 的解码和 FP32 局部求和。
- 发起前三次 PV，再执行第四组 P 的解码和 FP32 求和。
- 所有 warp 执行寄存器 fence 后发起第四次 PV，保留后续历史缩放、下一块 scale 预取和最终等待。

待完成的前三组 P 和 PV 累加寄存器在等待前不被普通指令访问；中间只读取尚未发给 WGMMA
的第四组 P。fence 位于 fallback 分支之外，整个 warpgroup 都会执行。
这遵循 [NVIDIA PTX WGMMA 的寄存器访问与 fence 约束](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence)。
P 编码、分母加法顺序、两级 FP32 PV 累加及尾块处理保持原算法，未引入 P 的共享内存副本。

```bash
git pull --ff-only origin master
bash scripts/run_pv_prefix_h200.sh
```

默认 BF16、NHD `[1,37741,28,128]`，INT8 QK + ExpCast + FP8 PV，V 聚类/平滑关闭。
沿用已安装依赖，首次运行编译独立 JIT 模块；无需重新安装包，也无需 ncu/nvdisasm。
9 轮 API 和 5 轮 CUPTI 分开记录，输出 `results/h200_cuda_expcast_pv_prefix.json`。
可追加 `--validate-only` 或 `--qkv-file ... --qkv-layout NHD`。

| 行名 | 含义 |
|---|---|
| vc28_scale_prefetch | 上轮预取候选，同次 JIT 编译的基线 |
| vc28_prefix_serial | 使用拆分后的概率函数，但仍先完成全部求和，再发起 PV |
| vc28_prefix_overlap | 同一拆分函数，将最后一组解码/求和放在前三次 PV 发起后 |
| vc28_installed | 已安装 vc28，独立对照 |
| sage_cpp / flash4 | 外部性能及精度对照 |

三个 JIT 版本都保留 K scale 预取。串行版用于区分拆分函数本身的代价与重叠调度的收益。
JSON 的 `paired_comparisons` 增加串行版/重叠版的逐轮差值；full/core API 与 CUPTI
保持独立口径。仅当新版本在完整 attention 中变快，才有采用依据。

测速前执行 38 个概率状态检查、28 个变化 scale 检查、8 个不同 warp 混合普通/fallback
路径检查，以及原有 28 个 attention 配置。主尺寸也检查所有输出行。
候选和串行对照相对同次编译的预取基线必须**逐位一致**；decoded reference 仍保留 1%
实现一致性门限，任一失败即停止测速。这些门限不代表模型质量通过。

本地 **95 项测试通过**，其中 38 项概率状态检查在 SM86 GPU 上实际执行，使用软件 FP8 解码。
CUDA 12.4 已交叉编译 FP16/BF16 × D64/D128 × 三个版本，共 12 个完整 SM90a kernel；
D128 全部为 168 个寄存器，D64 为 127/128/128，均无 stack/spill。
SASS 确认前三次 PV 与第四次 PV 之间有解码、FP32 求和及 fence，没有插入等待前三次 PV 完成的指令。
本地未执行完整 Hopper attention；该候选的 H200 GPU 时间已回传，见本页最新结果，
完整验证详情和 API/trial 原始记录尚未提供。
证据见 [PV prefix 本地检查](results/cuda_expcast_pv_prefix_local_validation.json)。

## 已测候选：提前一轮读取 K scale

最新用户回传的 CUPTI 汇总：balanced-max **23.5873 ms**，fused-quad **23.7386 ms**，
installed vc28 **23.3888 ms**，Sage **25.0717 ms**，FA4 **31.0350 ms**。
fused-quad 比同编译基线慢约 0.64%，没有正收益证据，不替换已测 vc28。
这里只记录用户提供的 GPU 时间，不补造未提供的 full/core API、精度或原始 trial。

源码看上去在每轮 QK 前读取 K scale，但本地 SASS 显示编译器已将它放进部分 QK 执行区间。
因此本轮不重复简单的源码挪动：**当前块 PV 执行时读取下一块的 K scale，跨轮保留一个 FP32 标量**。
下一轮继续按原顺序计算 `q_scale * k_scale` 和 `original_sm_scale * dequant_scale`。
第一块单独读取；MAIN 循环只预取存在的下一块，最后一次预取供尾块使用，无越界预取。
不增加共享缓冲区，QK/PV、概率编码、最大值、分母求和和历史累加均保持 vc28 原方式。

```bash
git pull --ff-only origin master
bash scripts/run_scale_prefetch_h200.sh
```

沿用已有 Torch、Sage local、FA4 和 nvcc，无需安装或重编译包，也不需要 ncu/nvdisasm。
默认 BF16、NHD `[1,37741,28,128]`，9 轮 API、5 轮 CUPTI，结果保存在
`results/h200_cuda_expcast_scale_prefetch.json`。追加 `--validate-only` 可以只验证；
真实输入追加 `--qkv-file ... --qkv-layout NHD`。

| 行名 | 含义 |
|---|---|
| vc28_jit | 原生产 kernel 源码和模板签名，不增加实验布尔参数 |
| vc28_scale_prefetch | 仅提前下一块 scale 的读取 |
| vc28_installed | 已安装版本，独立检查编译和封装差异 |
| sage_cpp / flash4 | 保留的外部对照 |

先跑 28 个逐块/逐 lane 变化的 scale 检查，覆盖 1、128、129、256、257、513、37741 个 key，
FP16/BF16、D64/D128 和非默认 stream；随后跑已有 28 个 attention 配置和主尺寸检查。
候选与同编译 vc28 的全部输出必须**逐位一致**，量化 reference 仍用原 1% 实现一致性门限，
任何失败都停止测速。这不代表模型质量通过。

JSON 新增 `build_audit`：记录实际加载的 JIT/installed 动态库路径与 SHA256、JIT 编译参数、
原 kernel 源码摘要、启动配置以及 D128 的 installed 资源信息。
两者 launch 均为 Q64/K128、128 threads、D128 动态共享内存 40960 bytes、寄存器上限 168。
已安装包的版本号不能证明其二进制与当前源码完全对应；该记录不声称已经查明远端 JIT 时间差。

新增 `paired_comparisons`，在 full/core 两种口径分别记录 JIT/候选、installed/JIT、
installed/候选的逐轮耗时差、中位差和更快轮数；正差值表示后者更快。
配对使用同一 trial 编号，trial 内顺序随机；这是描述性统计，不是显著性检验。

本地 **50 项测试通过**，完整 SM90a kernel 的 FP16/BF16 × D64/D128 × 两路径共 8 个实例
已用 CUDA 12.4 交叉编译，无 stack/spill。D128 两路径均为 168 寄存器，
D64 原版 128、候选 127。SASS 确认下一块 scale 的加载位于 MAIN PV 指令之间，
移出了下一轮 QK 区间。H200 GPU 时间已回传并列在本页开头；本机 SM86 未执行这些完整 kernel。
本地证据见 [scale-prefetch 编译检查](results/cuda_expcast_scale_prefetch_local_validation.json)。

## 已测候选：fused quad（以 balanced-max 为基线）

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
