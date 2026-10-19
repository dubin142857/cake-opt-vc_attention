# MiniMax-H3 / A5 VC-Attention 实验

**当前 Ascend 主线：原生 INT8 QK + 原 Laser softmax + FP16 V/PV。** 执行 `bash scripts/run_mindie_laser_quant.sh`，默认比较 `installed`、`baseline`、`qk_int8`、`qk_int8_fused` 四组。后两组分别使用原有多算子 Q/K 量化和新融合 Ascend C 预处理，采用相同的 attention 内核；V 保留 FP16 转换，不做重排或平滑。融合候选需远端 CANN 编译及 NPU 正确性、性能验证。E4M3 V 解码和 ExpCast 保留为实验；设置 `LASER_QUANT_EXPERIMENTS=1` 追加两组。A2/A3 实验中的 E4M3 V 会解码为 FP16 执行 PV，不宣称原生 FP8 加速。分阶段同步调用计时用于定位开销，不能相加或解读为内核耗时占比，详见 [Laser 量化说明](ascend/laser_quant/README.md)。原入口 `bash scripts/run_mindie_laser_local.sh` 继续提供[不改算法的源码基线](MINDIE_LASER_REPRO.md)，[仅替换概率计算的旧 ExpCast 实验](MINDIE_LASER_EXPCAST.md)和此前 CANN FAS 构建均保留。

已有成功 Ascend 量化构建时，执行 `bash scripts/run_mindie_laser_quant_profile.sh`，复用产物做同进程、固定输入的 9 轮配对诊断，随后单独采集 NPU profiler；不重新编译。报告为 `results/mindie_laser_quant_profile.json`。它区分相同 payload 的原生调用、独立 payload 的原生调用以及完整调用，避免把跨进程波动解释成核心退化。运行范围和设备指标限制见 [Laser 量化说明](ascend/laser_quant/README.md)。

scale 复用候选入口：`bash scripts/run_mindie_laser_scale_reuse.sh`。用户回传同轮配对结果：native 89.5797 → 87.5177 ms，full 93.0581 → 89.4787 ms，两项均 9/9 轮更快；这些是该次实验结果，不是所有形状的保证。

gain 提前准备版未显示收益：用户回传 native/full 配对节省中位数为 −0.0221/−0.0545 ms，仍保留 scale 复用版作为基线。

下一轮独立候选：`bash scripts/run_mindie_laser_score_prefetch.sh`，利用现有两块 Vector 分数缓冲区提前搬入下一批四行，尝试让 QK 分数传输与当前 max/exp/sum 重叠。只从本地缓存构建新候选，不联网、不重编旧基线；先验证全输出一致，再同进程配对比较，并补采原版 FP16 Laser 计数器。报告 `results/mindie_laser_score_prefetch.json`。新候选尚待远端 CANN 编译和 NPU 验证；[实验说明](ascend/laser_quant/README.md)和[同步审计材料](reports/ascend-score-prefetch-sync-audit/README.md)记录改动范围与限制。

**当前 H200 主路线（2026-10-13）：`sageattn_vc`，INT8 QK + FP8 PV。**
新候选 `bash scripts/run_tensor_mass_h200.sh`：针对概率分母求和，用无损解码后的 FP16 输入、FP32 累加的 Tensor Core 归约替代逐值升 FP32/相加。D128 编码/求和常规分支的本地静态指令从 325 降到 239，寄存器仍为 168；85 项检查通过，H200 整核性能待测。对比同次 scale-prefetch、installed、Sage、FA4，见 [CUDA 实验说明](SAGE_CUDA_EXPCAST.md)。
最新 scalar-prefetch 为 23.0945 ms，同轮 scale-prefetch 为 22.9710 ms，候选慢 0.1235 ms（0.54%），不采用，继续保留 K128 scale-prefetch 基线。随后回传确认两者编译资源相同、输出逐位一致，但 full/core 均仅 2/9 轮更快，配对节省中位数均为负。该轮基线相对 Sage/FA4 的 GPU 内核加速比为 1.077×/1.326×，不代表完整 API 或模型收益。`bash scripts/run_scalar_prefetch_h200.sh` 保留作实验对照，见 [CUDA 实验说明](SAGE_CUDA_EXPCAST.md)。
最新 paired-QK 实测 24.5014 ms，比同轮预取基线 23.4789 ms 慢 4.35%，不采用；继续保留 K128 预取路线。该轮基线对 Sage/FA4 的 GPU 内核加速比分别为 1.060×/1.302×，不是完整 API 收益，见 [CUDA 实验说明](SAGE_CUDA_EXPCAST.md)。`bash scripts/run_paired_qk_h200.sh` 保留为实验对照。
最新 mass-tail 为 23.4445 ms，预取基线为 23.4322 ms，未显示收益；保留为历史实验。
上轮 prefix-overlap 为 23.5372 ms，仍慢于预取基线 23.4155 ms，暂不采用；`bash scripts/run_pv_prefix_h200.sh` 保留作对照。
上轮 K scale 预取实测 23.4629 vs 同次 JIT 23.6402 ms，约节省 0.75% GPU 时间；原入口 `bash scripts/run_scale_prefetch_h200.sh` 保留。
上轮 fused-quad 未体现收益（23.7386 vs balanced-max 23.5873 ms，installed 23.3888 ms），保留为历史实验。
当前 H200 优化主线：`bash scripts/run_cuda_expcast_h200.sh`，以 vc28 为基线对比 INT32 最大值归约候选、Sage、FA4；见 [CUDA 实验说明](SAGE_CUDA_EXPCAST.md)。
当前候选分段诊断：`bash scripts/diagnose_balanced_expcast_h200.sh`，测 balanced-max 的 QK、概率编码/解码/求和、PV 与预处理，并校准插桩开销；保留 vc28、Sage、FA4 对照。
保留论文直接公式 CUDA/TileLang 对照：`bash scripts/run_expcast_tilelang_h200.sh`，
自动准备 TileLang、验证后对比 Sage/FA4；见 [运行说明](SAGE_EXPCAST_TILELANG.md)。
TileLang 当前使用自然顺序 FP8 V，去掉逐块 V 字节重排，只开 ExpCast；完整 API 包含各自预处理。
保留自然 V 基线，并新增异步 PV 与分母归约/历史缩放重叠的候选，脚本同时验证和测速。
当前先关闭 V 聚类/平滑，集中诊断 ExpCast 单开；vc27 FP8 QK 归为历史实验。
新入口 `bash scripts/diagnose_expcast_h200.sh` 对比 Sage、同内核 control/ExpCast 和 FA4，
拆分预处理，并通过独立探针观察 QK/概率/PV 区段，见 [ExpCast 专项诊断](SAGE_EXPCAST_FOCUS.md)。
加 `--probability-detail` 可继续细分 ExpCast 的 max、编码打包、FP8 解码和 FP32 分母求和，并同场测量插桩开销。
已有 `2.2.0+vc32` 无需重新安装包；首次用本机 nvcc 编译诊断模块，不要求 ncu/nvdisasm。
vc32 最新 H200 回传：双开最佳候选完整 API 30.3465 ms、固定输入核心 API 25.7254 ms，
全部输出与 vc31 双开相等；只开平滑的两个候选退步。该轮没有 CUPTI 内核时间。
平滑候选及原验证流程仍保留，见 [SAGE_VC32.md](SAGE_VC32.md)。
导入与测速见 [H200_INT8.md](H200_INT8.md)；随机输入、API 时间与论文纯 kernel 指标分别记录。
此前“VC 论文 QK 统一使用 FP8”的说法不准确，H200 路线不以此为依据。

以下保留独立 `vc_attention` 0.x / NPU 原型历史，**不是当前 H200 主入口**。

新增 H800 CUDA C++ 后端：`Config(backend="cuda_fp8")`。源码、`pip install .`
构建方式和远端验证步骤见 [CUDA_README.md](CUDA_README.md)。旧路径 QK/PV 使用真实
cuBLASLt FP8 矩阵乘；用户提供的 H200 日志中，旧版 0.1.0 已通过原生自检和 60 项测试。
0.2.0 增加论文数值对齐，见 [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md)。
旧路径尚不是融合内核。与 FlashAttention 2/3/4、SageAttention
的对比命令见 [BENCHMARK.md](BENCHMARK.md)。
另提供 SageAttention 2.2.0 的独立源码编译包 `sageattention_local`，可与已安装原版同场比较，见 [SAGE_LOCAL.md](SAGE_LOCAL.md)。
在此基础上新增 `sageattn_vc`：保留 Sage INT8 QK 和 Hopper TMA/WGMMA 主体，
加入 V 聚类/分块平滑与 ExpCast，四组开关对照见 [SAGE_VC.md](SAGE_VC.md)。
Sage+VC 的分阶段计时、固定量化输入对照和 profiler 用法见 [SAGE_VC_PROFILE.md](SAGE_VC_PROFILE.md)。
Q/K、P、V 分阶段精度诊断见 [ACCURACY_DIAGNOSTICS.md](ACCURACY_DIAGNOSTICS.md)，已有 0.2.0 安装可直接运行。
0.2.1 修正参考 scale 除法的舍入，并增加固定量化字节的 CUDA 对拍，见 [PAYLOAD_DIAGNOSTICS.md](PAYLOAD_DIAGNOSTICS.md)。
0.3.0 默认批量处理 head 和 Q 行、缓存 GEMM plan、合并状态更新及输入校验。
**须重新编译**；新旧路径与缓存复用对比、profiler 命令见 [CUDA_PERFORMANCE.md](CUDA_PERFORMANCE.md)。
BSHD `[1,37741,28,128]` 的完整长度性能场景：`bash scripts/benchmark_37741.sh`，见 [BENCHMARK.md](BENCHMARK.md)。
当前 **0.7.0** 为 D128 ExpCast 增加矩阵乘与概率计算交叠，测速默认 `--vc-schedule fused`；
Python 中显式选 `Config(backend="cuda_fp8", cuda_schedule="fused")`。需要 nvcc >= 12.4 并重新编译，
预处理仍未融合。与保留的 0.6.0 同场对比加 `--vc-interleave-compare`，见 [INTERLEAVED_CUDA.md](INTERLEAVED_CUDA.md)。
0.6.0 D128 特化与 0.5.0 通用核心对照见 [D128_CUDA.md](D128_CUDA.md)。
0.5.0 异步搬运流水线和 0.4.0 对照见 [PIPELINED_CUDA.md](PIPELINED_CUDA.md)。
0.4.0 融合主体的结构与历史验证见 [FUSED_CUDA.md](FUSED_CUDA.md)。
下文原有 A5 实验记录和两种后端的说明保留为历史背景；默认后端仍为 `reference`。

当前 Python 包源码位于 `src/vc_attention/`，C++/CUDA 源码位于 `csrc/`。
在本仓库运行测试前先按 CUDA 文档执行 `pip install .`；根目录不再是 Python 包。

这是可运行的**算法与接线原型**。已完成 CPU 数值测试；A5 自检、真实 H3 张量回放、视频/音频生成、融合内核性能均尚未验证。请先抓真实 Q/K/V 回放，再决定融合开发是否值得继续。

## 实现范围

| 文件 | 用途 |
| --- | --- |
| `src/vc_attention/core.py` | ExpCast、V 聚类重排、128-token 块去均值、在线均值恢复、步数调度 |
| `src/vc_attention/npu.py` | A5 原生 FP8 矩阵乘实验后端和设备自检 |
| `src/vc_attention/lightx2v.py` | LightX2V H3 注册表接入、受限张量抓取、生成入口 |
| `src/vc_attention/replay.py` | 同输入四组对照，以及单独的 MindIE BF16/FP8 基线 |
| `tests/` | 独立 dense oracle、编码、状态、适配器契约、A5 检查 |
| `results/` | 本机合成数据报告；不是 H3 实测结果 |

两个实验执行后端：

- `reference`：E4M3 编码后解码，用 FP32 矩阵乘进行数值模拟。当前在 CPU/CUDA 验证；新版 ExpCast 参考需要 FP64 中间运算，NPU 尚未重新验证。**不是 FP8 硬件测速。**
- `npu_fp8`：调用 `torch_npu.npu_add_quant_matmul_` 的 MX 接口，E8M0 scale 字节固定为 127（即 1），让 Cube 消费原始 E4M3 payload；普通量化 scale 在外部恢复。FP32 累加。此路径先跑设备自检，不支持时直接报错，无静默回退。

两者都以 Python 分块调度，尚未融合到 FIA。每个 tile 都会下发多个算子，**不应把原型时延解释成 ExpCast 的加速能力**。长序列完整 H3 推理可能非常慢；优先回放少量 query，保留完整 K/V。

## 数值契约与论文差异

按 [VC-Attention v1](https://arxiv.org/html/2609.15810v1) 的公式重建，非作者代码：

- ExpCast：`uint8(clamp(round(fma(S-rowmax,8*log2(e),119.65)),0,120))`，随后按位 `view(float8_e4m3fn)`，概率 scale 为 `1/256`。CUDA 使用单次 FP32 FMA，参考用加宽中间运算模拟；已增加边界字节测试。
- V-Smooth：每头独立聚类，K/V 同序重排，Q 保持原序；用 FP32 均值计算残差，按每块每通道 E4M3 量化；`mean / V_scale` 默认以 FP16 保存，补均值与 PV 残差共用 scale，随 running-max 一起 rescale。
- 普通 exp 路径沿公式 6，用 FP32 exp 的 rowmass 归一化/补均值，PV 使用舍入后的 FP8 P；ExpCast 路径用解码后的 P 同时计算 PV、rowmass 和归一化。
- 默认前 `ceil(0.25 * 实际去噪步数)` 步去均值，间隔 4 步更新 permutation；之后保留 permutation 并关闭去均值。只缓存排列/中心，每次重新算当前 V 的均值和残差。4 步模型只有第 0 步去均值。
- 聚类默认 8 类、2 次 Lloyd 迭代，确定性等间距初始化。这是可调实验选择，未声称复现作者的聚类内核。
- 默认先对 post-RoPE K 按通道去均值，再对 Q/K 做共同的归一化 Hadamard，之后分别按 128/256 tokens、128 channels 块量化。具体 Hadamard 约定和量化块是本实现选择，详见论文对齐记录。

`fp8_control / expcast / v_smooth / combined` 四组共享同一原型量化契约。**`fp8_control` 不等于现有 MindIE FP8 FIA**：后者 V/P 的块结构、概率处理和内部累加存在差异，报告中另列 `mindie_fp8`，不能把两者差值全部归因于 V-Smooth/ExpCast。

## 本地复现

进入 MindIE-SD 根目录，使用已有 PyTorch（需 E4M3 dtype）与 pytest：

```bash
export PYTHONPATH="$PWD/examples/minimax-h3:${PYTHONPATH:-}"
python -m pytest examples/minimax-h3/vc_attention/tests -q --noconftest
python -m vc_attention.replay --clustered --output vc-results/clustered.json
python -m vc_attention.replay --output vc-results/gaussian.json
```

`--noconftest` 隔离仓库其他需要 NPU 的 pytest 配置。运行时不会导入本地 MindIE 扩展，除非指定 `--mindie` 或启动原生 MindIE 基线。

## A5 执行顺序

沿用你已经验证可用的 CANN 9.1.0、torch_npu、MindIE-SD 环境，不用本机测试依赖覆盖设备环境。推荐将交付压缩包解到独立目录，避免未编译的 MindIE 源码遮蔽设备上已安装的库：

```bash
mkdir -p /path/to/vc-experiment
tar -xzf vc-attention-h3-experiment.tar.gz -C /path/to/vc-experiment
export PYTHONPATH="/path/to/vc-experiment:${PYTHONPATH:-}"
cd /path/to/vc-experiment
python -c 'import mindiesd; print(mindiesd.__file__)'
```

基线调用基于本次 MindIE `dev` 的 `quant_attention` 公共接口；若设备版本较旧且没有该入口，需要安装对应版本或接到你已验证的 FP8 wrapper，不能把接口不存在解释成 A5 不支持 FP8。下方 pytest 路径按解包方式使用。

先验证新增的 FP8 矩阵乘路径，及本机未跑的原生算子检查：

```bash
python -m vc_attention.npu --device npu:0
python -m pytest vc_attention/tests/test_npu.py -q --noconftest
```

检查包含真实 FP8 乘法、尾块补零、E4M3 reinterpret、RNE、四组结果与 CPU oracle 对拍。这里验证的是**新增路径**，不重复质疑已验证的普通 FP8 Attention。若环境缺 `npu_add_quant_matmul_`，仍可显式用 `reference` 检查算法误差。

### 抓 H3 的真实张量

接入基于 LightX2V 的 MiniMax-H3：在 RoPE 与 Ulysses All-to-All 之后，进入 attention 的位置抓取完整 K/V。默认仅 rank 0、第 0/4 步、第 0/12 层、每次 1 个头、最多 4 份、512 MiB 张量预算。基线输出仍走 MindIE 原生 FP8。

保持你已有的模型、提示词、seed、分辨率和步数；下例中 `H3_A5_CONFIG` 应指向已跑通的 A5 LightX2V 配置。示例 task 在 LightX2V 中拼作 `t2av`。

```bash
export H3_MODEL=/path/to/MiniMax-H3
export H3_A5_CONFIG=/path/to/known-working-a5-config.json
torchrun --nproc_per_node=4 --module vc_attention.lightx2v \
  --mode capture --output-dir vc-results/capture -- \
  --model_cls minimax_h3 --task t2av \
  --model_path "$H3_MODEL" --config_json "$H3_A5_CONFIG" \
  --prompt 'A red scarf moving in the wind by the sea.' --seed 1101 \
  --save_result_path vc-results/capture/baseline.mp4
```

入口生成配置副本，保留其他参数，注册独立 `mindie_vc_experiment`，文本 refiner 单独使用 `npu_flash_attn`。原配置不被覆盖。

当前约束：单序列、dense、非 causal、无 mask/bias/dropout、相同 Q/K/V 头数；不支持 packed、GQA、ring、head-parallel、CFG 或 compile。原配置中 `use_compile`、`warmup` 和 `parallel.seq_p_head_parallel` 必须关闭，所有对照组一致。显式关闭框架 warmup，防止把预热提示词的张量当作真实请求抓取；replay 有独立预热。未知参数显式拒绝，避免静默改变注意力语义。并发服务不在本适配器范围内；每次启动用于单一推理任务。

可以先 `--prepare-only` 检查生成的配置；它不导入 LightX2V、MindIE 或 torch_npu。

### 回放同一份张量

```bash
python -m vc_attention.replay \
  --capture vc-results/capture/captures/rank0_step0_layer0.pt \
  --device npu:0 --backend npu_fp8 --mindie \
  --q-start 0 --q-limit 128 --output vc-results/h3-layer0.json
```

`q-start` 必须位于 128-token 块边界。增加它以覆盖音频、视频、文本等不同位置；`q-limit=0` 表示完整 Q。始终保留全部 K/V，不把裁剪 context 的结果冒充完整 Attention。先跨层/跨步/跨头测 relative RMSE、最大绝对误差、cosine，再选参数；单次 replay 强制处于 smoothing 窗口，用于隔离算法，不能当整段去噪收益。

报告包含原张量 SHA256、形状、scale、运行时、完整采样时延、重复次数、参考定义。原生 MindIE BF16/FP8 与原型四组分列。计时含分组、量化、分块循环，交替反转顺序并同步设备；属于探索结果，无融合加速结论。

### 整段生成对照

数值和设备自检通过后，在小规模场景使用同一入口分别指定以下 `--mode`，每组使用独立 `--output-dir`、`--save_result_path`：

| mode | 作用 |
| --- | --- |
| `mindie_bf16` | 原生高精度生成参考 |
| `mindie_fp8` | 原生 FP8 生成基线，可选 `--fp8-mode C8V16_TILING512` |
| `fp8_control` | 原型不启用两种算法，排除后端量化契约变化 |
| `expcast` | 只开 ExpCast |
| `v_smooth` | 只开 V-Smooth，应用前 25% 步调度 |
| `combined` | 两者一起 |

审查 `audit.rank*.json` 中的 `calls`、`expcast_calls`、`smoothing_calls`、`regroup_calls`，确认开关确实命中。关闭本实验改回原来的 LightX2V 入口/配置即恢复基线。

同 prompt/seed/步数/模型权重比较视频帧、运动、细节及音画同步。可复用仓库 `evals/scripts/quality_compare.py`，对同帧索引的解码图片目录计算 SSIM/PSNR：

```bash
python evals/scripts/quality_compare.py --baseline bf16_frames \
  --config combined_frames --metric ssim --output vc-results/quality.json
```

音频质量须另查，不能只看视频帧得出 H3 整体无损结论。原型完整推理仅用于质量探索；要评估性能，需要把 ExpCast 写进 FIA vector Softmax，把均值恢复接到 PV 的在线累加，并融合 gather/quantize。相关来源位置记录在 `PROVENANCE.json`。
