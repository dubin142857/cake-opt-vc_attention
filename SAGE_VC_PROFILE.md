# Sage+VC 性能诊断

用户提供的 vc3 / H200 / BF16 `[1,37741,28,128]` 诊断结果：

| 变体 | full API ms | fixed core API ms | trace 主 GPU kernel ms | 其他 kernel 总 ms | 总启动次数 |
|---|---:|---:|---:|---:|---:|
| sage_local | 25.6001 | — | — | — | — |
| sage_vc_control | 26.4194 | 23.8341 | 22.1712 | 1.8459 | 9 |
| sage_vc_expcast | 32.8908 | 31.3432 | 30.6065 | 1.8445 | 9 |
| sage_vc_smooth | 45.2558 | 38.0381 | 37.2852 | 7.4322 | 461 |
| sage_vc | 50.8135 | 42.1577 | 42.1488 | 7.4435 | 461 |

trace 每组只采集一次，带 profiler 开销，用于定位而非稳定速度结论；不能与独立 API
中位数相减来计算开销。ExpCast 和块均值/尺度补偿使主内核明显变慢，聚类还引入大量
短 kernel。优先优化主内核，再优化预处理。以上数据均为 vc3；后续 vc4 结果和 vc5
对照方法见文末，当前改动见 [SAGE_VC.md](SAGE_VC.md)。

## 直接运行，不用重新编译

诊断脚本兼容已经安装的 `sageattention-local==2.2.0+vc3` 及后续版本。
仅运行诊断不用重新编译；升级 vc7 则需要执行 `bash scripts/build_sage_local.sh`。
从仓库根目录执行：

```bash
git pull --ff-only origin master
python3 scripts/profile_sage_vc.py --output results/h200_sage_vc_profile.json
```

默认 BF16、BSHD `[1,37741,28,128]`，脚本内部统一为 BHND `[1,28,37741,128]`。
首次运行也可用 `--queries 1024 --tokens 1024 --heads 2` 检查环境。
默认 2 次预热、每轮 5 次、3 轮；计时顺序随机化，记录每轮均值及其中位数。

脚本输出三类结果：

1. **full API**：未插入阶段事件的公开入口同步 wall time，包含每次新做的聚类与所有预处理。
   与先前 benchmark 的含义一致，另保留原版 `sage_local` 对照。
2. **分阶段 CUDA-event 时间**：聚类/排序 `cluster_sort`、K/V 重排 `reorder_kv`、K 均值
   `k_mean`、含减均值的 QK INT8 量化 `qk_int8`、全局 V 补齐 `v_pad`、
   V 量化 `v_quantize`、融合主体 `attention`。阶段间不插入同步；结束后统一读取计时。
   平滑开启时，`v_quantize` 包含块均值、残差、尺度、FP8 编码及存储排列。
3. **fixed core API**：先准备好量化数据，然后只调用原生 attention。
   同一 V 模式下，普通 Softmax 和 ExpCast 共享同一份 Q/K/V 字节与尺度，仅切换 ExpCast 开关。
   此数值包含原生接口校验、输出分配、TMA 描述符及 kernel 下发，**不是纯 GPU kernel 时间**，
   更不能把它当成端到端加速。

分阶段代码使用安装包的原生量化函数，并在计时前与其 `vc.prepare` 比较全部实际使用的
量化字节/尺度。平滑组仅在这个对拍过程中固定排列，避免聚类归约次序干扰；
**真正的阶段计时每次都重新聚类**。对拍不一致会直接报错。
这项检查只保证诊断与安装包接线一致，不替代 attention 数值验证。

CUDA-event 区间可能包含 GPU 等待 CPU 下发的空闲时间；不要称为各 kernel 执行时间之和。
JSON 另外记录 `cpu_enqueue_ms`，它与 GPU 时间重叠，不能相加。
插入事件本身会影响调度，阶段之和也不要求等于独立测量的 full API。
默认诊断不计算 FP32 dense reference；vc6 的 `--compare-v-scales` 会额外计算抽样 FP32 参考，仍不输出模型质量结论。

## 进一步看实际 GPU 时间线

有 CUPTI 支持时追加：

```bash
python3 scripts/profile_sage_vc.py --trace-dir results/h200_sage_vc_traces --output results/h200_sage_vc_profile_trace.json
```

每组另采集一个完整调用，输出 Chrome/Perfetto 格式 trace。trace 标有阶段 CPU 范围，
包含 CUDA kernel 名称、启动次数及持续时间。报告将主 attention kernel 与其他 GPU kernel
分开统计，CPU runtime 等待不并入 GPU 总量。缺少 GPU/core 事件会报错，避免误报零耗时。
若环境不支持 CUPTI，先使用不带 `--trace-dir` 的命令。
Profiler 有额外开销，其时长用于定位，速度仍看 full API。

判断方法：

| 观察 | 下一步 |
|---|---|
| `cluster_sort` 明显占时 | 优化聚类/排序，并在真实去噪步骤中测合法的布局复用 |
| `reorder_kv`、`v_quantize` 明显占时 | 融合 gather、块统计和量化，减少大张量写回 |
| 固定数据后 ExpCast 仍慢 | 检查概率编码/打包/解码、归约、指令依赖及流水线 |
| event 时间大、GPU kernel 总时间小 | 检查 host 下发及大量短 kernel 造成的空隙 |

若已安装 Nsight Compute，可以各抓一个预热后的主内核；预处理在捕获区间外：

```bash
ncu --profile-from-start off --kernel-name regex:sage_vc_attention_kernel --launch-count 1 --set full -o results/sage_vc_control_ncu python3 scripts/profile_sage_vc.py --ncu-variant sage_vc_control
ncu --profile-from-start off --kernel-name regex:sage_vc_attention_kernel --launch-count 1 --set full -o results/sage_vc_expcast_ncu python3 scripts/profile_sage_vc.py --ncu-variant sage_vc_expcast
```

关注实际寄存器数、occupancy、Tensor Core 活跃度、指令组成和 warp 等待原因。
`--set full` 可能多次 replay；这些时长不能与常规 benchmark 直接比较。
选项含义见 [NVIDIA Nsight Compute CLI 文档](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。

## 为什么论文快，当前实现反而慢

[VC-Attention §3.4、§4.3](https://arxiv.org/html/2609.15810v1) 报告 H200 上相对
SageAttention2 的 attention 加速约为 **1.16 倍**。Figure 6 上排明确是 Wan2.2 的
稳态主 GPU kernel 计时，**排除了所有预处理**，不是 MiniMax 或完整生成端到端提速。
同图完整生成的 1.13 倍则是相对 BF16 FA4，基线不同。
这是作者在其数据与内核实现下的实验结果，不是任意 shape 或公式移植的速度保证。

论文将 gather、均值/尺度统计、旋转和量化等预处理融合，并在去噪步骤间复用分组；
分组/去均值集中在前 25% 步骤。当前 Sage+VC 每次重新聚类，K/V 重排及 QK 量化仍分开，
QK 还保留 Sage 的 INT8，所以不能视为已经复现这些性能条件。
预处理融合/复用影响完整 API 和模型端到端时间，不能解释主内核本身的速度差距。

V 平滑主要解决精度；ExpCast 是用来缩短概率处理阶段。只有省下的关键路径时间
大于新增编码、归约、补偿及预处理成本，整个 attention 才会变快。
用户 trace 里**仅 ExpCast 的主 GPU kernel 已比控制组慢约 8.44 ms**，
说明不能把全部问题都归因于聚类。

vc3 的本地 CUDA 12.4 交叉编译 D128 寄存器数为：控制组 168、ExpCast 167、平滑 170、
两项全开 205，无 spill。这提示全开模式可能存在额外寄存器压力，
但无法解释 ExpCast 单独变慢，也不能代替 H200/CUDA 13 的硬件计数器结果。
这仍然是待诊断的实现性能缺口，不应当推导成论文方法没有加速能力。

本地已运行阶段编排/字节对拍/时间单位/trace 汇总测试及实际 CUDA event 测试；
H200 profiling 由用户运行并提供结果，本地无法执行 Hopper attention。

## vc4 结果与 vc5 同场对照

用户 vc4 的完整 API 为：控制组 27.0946 ms、ExpCast 32.4597 ms、平滑 41.8509 ms、
全开 51.1880 ms；trace 主内核分别为 23.0391、29.7912、34.0194、42.7132 ms。
单平滑有所下降，全开没有提速；指令和寄存器数减少不等于运行更快。
两轮控制组时间也有变化，因此后续使用同进程对照。

升级 vc5 后运行：

```bash
git pull --ff-only origin master
bash scripts/build_sage_local.sh
python3 scripts/profile_sage_vc.py --compare-kernels --trace-dir results/h200_sage_vc5_traces --output results/h200_sage_vc5_profile.json
```

原有 full API、分阶段和 trace 表使用默认 optimized 模式。新增对照表将四种开关下的
`vc4` / `optimized` 放在同一进程中预热、随机交错计时。每对复用同一份量化数据及尺度。
`old/new > 1` 才表示新内核更快；两边计时均为固定数据的原生 API wall time，包含下发和
输出分配，排除全部预处理，不能当成完整 API 提速。vc4 模式是旧数学/调度在当前工具链下重编译。

计时前比较全部输出 query/heads/channels，FP64 分块计算误差，避免只看抽样；
新旧相对 RMSE 超过 0.1% 或存在非有限结果即报错。这不替代 FP32 dense 参考或模型质量验证。
`optimized` 的 FMA 舍入与 vc4 有意不同，具体改动见 [SAGE_VC.md](SAGE_VC.md)。

## vc6：共享 V scale 与末尾恢复

用户 vc5 同场数据：full API 为 Sage 25.4430 ms、全开 49.6464 ms；同量化输入
vc4/optimized 全开为 43.0974/41.8725 ms。控制组调用相同内核却也变化 2.6%，
所以 2.9% 的全开变化不能证明稳定提速。vc6 改动见 [SAGE_VC.md](SAGE_VC.md)。

重新编译后，先用较小 shape 查完整误差：

```bash
bash scripts/build_sage_local.sh
python3 scripts/profile_sage_vc.py --queries 1024 --tokens 1024 --heads 2 --check-queries 0 --compare-kernels --compare-v-scales --output results/h200_sage_vc6_small.json
```

再测 `[1,37741,28,128]`，默认检查 128 个 query，使用全部 key：

```bash
python3 scripts/profile_sage_vc.py --compare-kernels --compare-v-scales --trace-dir results/h200_sage_vc6_traces --output results/h200_sage_vc6_profile.json
```

- 主表 full API/stages/core 使用新的默认 channel V scale；原 Sage 基线不变。
- `--compare-kernels` 的两种执行方式共享全部量化输入，包括新 channel scale。
  vc4 参考每块恢复同一尺度；optimized 末尾恢复。两者保留两级 FP32 PV 累加。
  全输出 relative RMSE <=0.1% 是实现门槛。
- `--compare-v-scales` 用同一原始 QKV，固定聚类排列及 QK 量化数据，只重做旧 block V
  量化，比较两种 optimized core 的时间。另测两种公开 API（每次都重新预处理），
  并用同一个 dense FP32 参考分别报告误差。顺序随机化，精度计算不计时。
  V 量化改变导致的全输出差异单独报告，**不套用 0.1% 实现门槛**。
- block/channel 的量化粒度不同，不能要求 FP32 误差与 vc5 完全一致；需要同时看
  `full API`、`fixed core API` 和 `FP32 rel RMSE %`。不将 kernel-only 收益称为端到端收益。

共享尺度预处理为三个 GPU pass，主循环开销降低是否抵得上新增预处理，需要这次实测回答。

## vc7：ExpCast 同输入对照

用户 vc6 的同场结果：全开 block/channel API 为 49.3169/44.4655 ms，固定主体为
41.0172/36.1118 ms；FP32 relative RMSE 为 4.167050%/4.205950%。这是 vc6 的测量，
不能作为 vc7 的性能或精度结果。vc7 默认保留 channel V，只修改 ExpCast 编码与求和调度。

编译会自动运行 144 组完整 attention 正确性检查，并新增 optimized/vc6 对拍：

```bash
git pull --ff-only origin master
bash scripts/build_sage_local.sh
python3 scripts/profile_sage_vc.py --queries 1024 --tokens 1024 --heads 2 --compare-kernels --kernel-baseline vc6 --output results/h200_sage_vc7_small.json
```

实际长序列性能（默认 BSHD `[1,37741,28,128]`），增加轮数减少计时波动：

```bash
python3 scripts/profile_sage_vc.py --compare-kernels --kernel-baseline vc6 --trials 5 --repeats 10 --trace-dir results/h200_sage_vc7_traces --output results/h200_sage_vc7_profile.json
```

`--compare-kernels` 在 vc7+ 默认用 vc6，在旧安装上仍默认 vc4。显式指定
`--kernel-baseline vc6` 可拒绝旧安装；`--kernel-baseline vc4` 保留历史对照。
新表写明实际 baseline，JSON 记录 `baseline_mode`、全部轮次、全部输出误差及 exact_match。
vc6/optimized 共用同一对象的 Q/K/V、尺度、块均值与排列，量化不变。
关闭 ExpCast 的 control 和 smooth 两组调用同一个编译内核，用于观察计时波动。
表内 0.1% 是实现门槛；本地 helper 虽逐位一致，仍需看 H200 全输出对拍。

主表 full API 包含全部预处理；同量化数据表只测原生 API，不能称作端到端加速。
无需再次运行 `--compare-v-scales` 来定位本次改动；此选项仍可对照 V 量化粒度，
但两组默认都用 vc7 概率路径。

## vc8：D128 平滑路径的寄存器与 PV 临时缓冲

用户 vc7 trace（单次诊断）：control/ExpCast/smooth/full 主内核为
23.7116/27.5053/30.7571/32.9326 ms，寄存器为 168/168/208/209，
共享内存均为 41088 字节。估计 occupancy 为 19/19/13/13%。
这支持检查平滑路径的寄存器限制，但不证明所有额外耗时都来自 occupancy。

vc8 新路径见 [SAGE_VC.md](SAGE_VC.md)。在两个 N64 fragment 之间保持完整输出历史；
每个 fragment 的 PV 仍使用临时 FP32 累加器，等待 WGMMA 完成后再恢复均值。
关平滑、D64、block V scale 均复用 vc7 内核作为控制。

```bash
git pull --ff-only origin master
bash scripts/build_sage_local.sh
python3 scripts/profile_sage_vc.py --compare-kernels --kernel-baseline vc7 --trials 5 --repeats 10 --trace-dir results/h200_sage_vc8_traces --output results/h200_sage_vc8_profile.json
```

构建脚本的 144 组 attention 检查增加 optimized/vc7 的全输出对照。
profile 的比较表也检查全部输出，门槛为 0.1% relative RMSE；不改变模型质量标准。
默认比较基线在 vc8 上为 vc7，显式指定 vc7 可在开始长序列计算前拒绝旧安装。

trace 汇总现在保留并打印主内核的寄存器、共享内存、grid/block 和估计 occupancy。
缺失字段显示 None，不能当成零。估计 occupancy 不是硬件实测 GPU 利用率。
本地 CUDA 12.4 下新路径为 168 寄存器；全开末尾输出阶段存在 8 字节 spill，
主循环未见 spill。远端需同时观察新旧耗时与编译资源，不能仅凭 19% occupancy 判定成功。
本次没有更改聚类预处理，预处理时间仍计入完整 API。

此外，D128 运行的环境信息包含 `compiled_d128_channel_resources`，由 CUDA runtime
查询当前 dtype 的新旧平滑/全开入口，给出寄存器、local memory 和预测驻留块数。
这是静态资源查询，排除在计时外；local memory 包含 stack/spill，不能解释为实测内存流量。

## vc10：优先测只开启 ExpCast

vc10 只改 `expcast=True, v_smooth=False` 的主内核；其他路径保持 vc9。
新增 `--expcast-only` 跳过平滑及聚类，只测 `sage_local`、`sage_vc_control`、`sage_vc_expcast`。
默认 BSHD 仍是 `[1,37741,28,128]`。升级、完整正确性检查和同场性能对照：

```bash
git pull --ff-only origin master
bash scripts/build_sage_local.sh
python3 scripts/profile_sage_vc.py --expcast-only --compare-kernels --kernel-baseline vc9 --trials 5 --repeats 10 --trace-dir results/h200_sage_vc10_traces --output results/h200_sage_vc10_expcast.json
```

重点看两种比较，含义不同：

1. `sage_vc_expcast` 行的 `vc9 ms / optimized ms`：同一份量化 Q/K/V 的新旧实现，
   `old/new > 1` 才是改动收益；全部输出的 relative RMSE 超过 0.1% 会报错。
2. `control -> expcast` 的 `plain/cast`：是否真正比关闭 ExpCast 快；新版本比旧版本快，
   不等于已经超过原版 Softmax。完整 API 表和 trace 主 GPU 时间也分别保留。

资源信息增加 `expcast_vc9` / `expcast_optimized`，无需 ncu。
这里的 vc9 是旧数学和调度在同一工具链下重新编译，不是跨次运行比较日志。

如需独立检查概率处理部分，可选运行：

```bash
python3 scripts/benchmark_expcast.py --output results/h200_expcast_micro.json
```

该脚本用 nvcc JIT 编译实际 device helper，交错测量原 Sage 概率处理、vc9 ExpCast、
vc10 ExpCast。所有变体限制为 168 个寄存器，与 D128 生产入口一致。包含公共输入和
checksum 运算，但不包含 QK/PV、预处理或 PV 重叠，不能把它的比例当成 attention 加速。
SM86 只测两条 ExpCast 路径，自动跳过需要 SM89+ 的原 Sage FP8 转换。
本地 3060 的局部结果约 0.99x，没有证明提速；H200 整核对照才是此次性能验收依据。

## ExpCast 编码与解码求和：分段定位

H200 已测的完整概率 helper 为 Sage/vc9/vc10 = 0.094496/0.097600/0.099712 ms。
同次工具链下均为 168 寄存器、0 local bytes。它说明当前 ExpCast 的局部处理并未胜出，
但不能确定瓶颈在编码还是解码求和，也不能度量与 PV 的重叠。

新增 `--stages`，在完整 helper 表之外，独立编译、验证、计时两个部分。
不需要重装 Sage 包；首次运行会使用 nvcc JIT 编译新的诊断模块。

```bash
git pull --ff-only origin master
python3 scripts/benchmark_expcast.py --stages --output results/h200_expcast_stages.json
```

| 分组 | 候选 | 含义 |
| --- | --- | --- |
| encode | expcast_u8 | 当前 ExpCast 的 u8 饱和取整并打包，编码基线 |
| encode | expcast_u32 | u32 取整后打包，与 u8 编码逐位一致 |
| encode | exp2_fp8_scale256 | 普通指数再转 FP8，尺度为 256，输出不要求与 ExpCast 一致 |
| decode_sum | fp32_serial | 解码后顺序累加块内权重，仅为局部对照 |
| decode_sum | fp32_two_chains | 当前 vc10 的两条 FP32 累加链，求和基线 |
| decode_sum | integer_exact | 用整数精确累加块内权重，最后转一次 FP32 |

`baseline/new > 1` 表示该候选在本分组中快于当前方式。保留所有轮次、交错执行顺序、
实际寄存器和 local memory；168 是寄存器上限，不要求每个分段用满。
普通指数对照的尺度 256 与原 Sage 的 `2^8.807` 不同，不能把这行称为 Sage Softmax。

整数候选的依据：ExpCast 正 FP8 编码范围是 0..120，即权重 0..256；
所有权重都是 1/512 的整数倍。每线程一行的 32 个权重最多为 `2^22` 个单位，
所以整数累加再缩放与该块的精确 FP32 求和一致，包括 FP8 次正规数。
它仍保留量化权重构造分母的语义，不替换为未量化的指数和。

计时前，脚本检查 65,900 个编码输入（覆盖取整边界及相邻 FP32 值）、
18,858 行解码求和输入（包括全部 121×121 编码对、随机行和极值），并检查计时内核的
同输入校验和。失败会直接报错。整数候选只用于诊断，尚未接入生产 attention。

两个分段均处理每线程每轮 64 个值，包含各自公共的输入和校验工作；
它们独立编译，调度和寄存器用量不同，因此耗时不能相加、不能换算成完整 attention 的
时间占比。先用 H200 的这张表筛选候选，再接回完整内核做同 Q/K/V 的正确性与性能对照。

本地验证：14 项 CUDA 测试通过，Compute Sanitizer memcheck 零错误；CUDA 12.4
编译 SM90a 的六个分段内核均无 spill。3060 的原始测量保存在
`results/expcast_components_sm86.json`，其 FP8 解码使用软件路径，不能预测 H200 的收益。
