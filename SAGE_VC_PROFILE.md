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
短 kernel。优先优化主内核，再优化预处理。vc4 的主内核改动见 [SAGE_VC.md](SAGE_VC.md)，
其 H200 速度尚待实测，以上数据均为改动前 vc3。

## 直接运行，不用重新编译

诊断脚本兼容已经安装的 `sageattention-local==2.2.0+vc3` 及后续版本。
仅运行诊断不用重新编译；从 vc3 升级 vc4 则需要执行 `bash scripts/build_sage_local.sh`。
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
该诊断不计算 FP32 dense reference、不输出模型精度结论。

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
