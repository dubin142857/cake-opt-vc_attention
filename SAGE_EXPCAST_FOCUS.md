# H200：只开 ExpCast，先测清楚瓶颈

当前诊断路线固定为 **INT8 QK + FP8 PV，关闭 V 聚类和平滑**。
保留 Sage 的 K 去均值；V 仍做 per-channel FP8 量化。
`vc28_stream_i2f` 是当前多轮验证过的 ExpCast 单开路线，
也是 vc31/vc32 在关闭 V 平滑时转用的内核。本轮增加诊断入口，不修改生产内核或其默认参数。

## 运行

已有 `sageattention-local==2.2.0+vc32` 的 H200 环境直接执行：

```bash
git pull --ff-only origin master
bash scripts/diagnose_expcast_h200.sh
```

使用 `python3`。不需要重新安装包、不下载依赖，不依赖 ncu/nvdisasm。
首次运行会用本机 nvcc 和 PyTorch 扩展构建工具编译独立诊断模块并缓存，
因此仍需要此前编译算子时已有的 CUDA toolkit/C++ 编译器。
默认 BF16、非因果 attention，等价 NHD `[1,37741,28,128]`。
Sage/VC 输入为连续 HND，FA4 为连续 NHD，转换及首次 JIT 在计时外。

结果写入 `results/h200_expcast_focus.json`，
CUPTI trace 和原始采样时间戳在 `results/h200_expcast_focus_traces/`。
建议发回 JSON 或以下四张表：API、stage CUPTI、完整 kernel CUPTI、内部 phase。
失败时保留已经完成的结果及错误，不拿其他时钟代替缺失数据。

只测正式 API 和 CUPTI、不编译内部探针：

```bash
python3 scripts/profile_expcast_h200.py
```

可用 `--qkv-file captures/qkv.pt --qkv-layout NHD` 测实际模型的 Q/K/V。
文件为本地 `torch.save(dict(q=q,k=k,v=v))`，应在 RoPE 之后采集。
`--check-queries 0` 才是所有 query 对 FP32 reference 的检查。
只有明确不测 FA4 时才加 `--skip-flash4`；只有明确不采 CUPTI 时才加 `--skip-cupti`。

## 对照组与正确性

只比较四组，不运行历史 V 平滑分支：

| 名称 | 做法 |
|---|---|
| `sage_cpp` | 当前本地 Sage 原版 API / 原版 native core |
| `control` | `vc28_stream_i2f`，ExpCast 关闭，V 平滑关闭 |
| `expcast` | 同一模式，ExpCast 开启，V 平滑关闭 |
| `flash4` | 已安装的 FA4，接受原始 BF16/FP16 数据 |

control/ExpCast 共用完全相同的量化 Q/K/V 和 scale。完整调用每次重新量化，
fixed core 复用准备好的张量，仍包含输出分配和 dispatch。
报告逐轮配对的 `control_ms - expcast_ms`，正数表示节省时间，同时保留所有原始 trial。
这个差值是 ExpCast 及相关调度的整体净收益，不能直接叫作 softmax 的独立耗时。

测速前检查：

- 分段预处理与正式 `vc.prepare` 的所有有效量化字节/scale 一致。
- 每个正式 full API 与其 core 输出逐元素相等；control 对 Sage 全输出 RMSE ≤0.1%。
- 默认 128 个 query、全部 key 对照 dense FP32 和量化算法 reference。
  量化 reference 的 1% 实现门槛不用于判断实际模型质量。
- 内部探针另外验证 FP16/BF16、D64/D128、四种对齐/尾块长度和两个开关的 32 个配置，
  每个配置同时检查未插桩及插桩副本。
- 主场景再检查全部输出与安装版本的一致性，并检查非默认 stream。
  使用 0.1% 实现门槛，另报告是否逐元素相等。

## 各个时钟能说明什么

### 1. 正式 API

没有 profiler/内部插桩，用同步墙钟比较完整调用和固定输入 core。
FA4 没有 fixed core 数据。full/core 两个独立中位数的差不是实测的某一个预处理阶段。

### 2. 预处理与整个 attention kernel

CUDA event 表保留连续流水线，但含 CPU 发射造成的 stream 间隙。
CUPTI stage 表为了准确归属阶段，单独在各阶段边界同步：

| 阶段 | 包含什么 |
|---|---|
| `k_mean` | K 沿 token 维求均值 |
| `qk_int8` | 输入精度的 K 减均值、Q INT8 量化、K INT8 量化 |
| `v_pad` | 非 128 对齐的 V 补零 |
| `v_quantize` | V 转置/排列、FP8 量化和 per-channel scale |
| `attention` | 整个融合 QK → 概率 → PV 内核 |

`qk_int8` 下还列出真实 GPU kernel 名称及各自耗时，区分 Q 量化、K 量化和减均值。
这些同步后的 stage kernel 时长是诊断，不是正式流水线的可加分解。
随后另外用 CUPTI 多轮采集正常固定输入 attention，不在内部加入同步或计时。
API、event、CUPTI 不能相加。

### 3. 融合内核内部

独立 JIT 模块由当前仓库的生产循环生成，保留源文件许可证；
只对 vc28 control/ExpCast 插入时钟记录，既不移除计算，也不新增 barrier/WGMMA wait。
生成器检查每个插入位置的数量，源码变动导致位置不符就报错。
同时编译无插桩副本，检查编译资源和插桩耗时影响，不将探针当成性能候选。

每 64 个 CTA 抽一个，每 8 个 K tile 抽一个，最终 masked tile 必定记录。
采样 CTA 的 thread 0 记录 `clock64`。正常 tile 和最终 tile 分开统计：

| 区段 | 含义 |
|---|---|
| `k_scale_and_wait` | 读 K scale、循环准备、等待当前 K 搬运 |
| `qk_mma_issue_wait` | INT8 QK 矩阵乘提交和等待；不是 Q/K 量化 |
| `next_k_prefetch` | 下一块 K 的 TMA 提交，尾块没有下一块 |
| `probability` | 分数转换、max、普通 softmax 或 ExpCast、FP8 编码及分母工作 |
| `v_wait` | V 数据尚未就绪而暴露出来的等待 |
| `pv_issue` | FP8 PV 的 WGMMA 提交 |
| `history_during_pv` | PV 执行期间进行的历史输出缩放/分母更新，主要在 ExpCast 分支 |
| `pv_wait_remaining` | 上述重叠工作完成后，PV 仍未完成的等待 |
| `accumulate_next_v` | 本 tile 输出加回 FP32 历史，以及下一块 V 的搬运提交 |

普通 softmax 分支的历史缩放在 `probability` 内；ExpCast 将部分历史更新放在 PV 期间。
因此既要看 probability，也要看后续重叠区段和整个未插桩 kernel 的净收益。
目前没有把 max、编码、FP8 解码/求和再逐条插桩；先定位大区段，再决定是否值得细分。

**内部表是抽样线程经历的周期，包含调度、记录开销以及编译器调度影响，
不是 Tensor Core/SFU 利用率，也不是生产 kernel 各硬件单元的耗时百分比。**
PV 的完整硬件执行可能覆盖 `pv_issue`、历史更新和剩余等待；不能把 wait 等同 PV 总成本。
不能把内部比例乘以正式 kernel 的 ms。prologue/epilogue 不在采样循环内。
这一限制也见 [NVIDIA 对 clock64 的说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html)。

探针整体 native API 会额外分配并清零时间戳缓冲区，开销单独报告。
若探针与未插桩副本的耗时差超过 10%，脚本会提示该轮只作探索，建议降低采样密度复查：

```bash
bash scripts/diagnose_expcast_h200.sh --phase-cta-stride 256 --phase-tile-stride 32 --output results/h200_expcast_sparse_probe.json --trace-dir results/h200_expcast_sparse_traces
```

即使整体干扰小，也不能保证每个内部区段无偏；保留两种采样密度和正式 CUPTI 结果一起判断。

## 如何选择下一步

1. 先看 Q/K 量化等预处理的 GPU 实耗：若很小，就不把主要精力投入仅节省几十微秒的改动。
2. 看同 payload control/ExpCast 的正式内核差值，确认 ExpCast 的整体收益。
3. 看 probability 的内部诊断及其后续调度：若这里仍占很多采样周期，再拆 max、编码和分母归约。
4. 若 QK/PV 或搬运等待更突出，改矩阵乘与搬运的流水线；不会因为 ExpCast 少算 exp 就假定它一定更快。
5. 每次改动仍用相同输入检查正确性，再看未插桩完整 API/CUPTI 是否确有改善。

## 本地验证范围

本地只有 SM86，不能执行 H200 的完整 attention。
CPU 测试覆盖路径禁用 V 平滑、计时归属、缺失事件拒绝、正常/尾块采样位置、净收益符号及失败结果保存。
CUDA 12.4 SM90a 交叉编译了 16 个诊断实例（两种 dtype、两种 D、control/ExpCast、插桩开关），
全部无 spill、无 WGMMA 串行化诊断；D128 均为 168 寄存器。
未插桩 D64 的指令文本与生产构建一致；D128 只有少量初始化顺序不同，主要循环一致。
136 项相关 Python 测试通过；同一采样记录 helper 在 SM86 上跑过 30 组 grid/长度/stream 检查。
本地记录见 [验证摘要](results/expcast_focus_local_validation.json)。
这些是构建/诊断检查，**不是 H200 正确性通过或加速结论**；远端脚本会执行上述验证门槛。
