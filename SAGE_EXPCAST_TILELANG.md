# H200：直接公式 ExpCast 与 TileLang 对照

2026-10-09。本轮只研究 **INT8 QK + ExpCast + FP8 PV**，V 聚类/平滑关闭。
新增 CUDA 调度候选，以及 TileLang 编写的完整融合 attention。
目标是先确认数值契约，再测净收益；这里没有 H200 加速已达成的结论。

### 当前路线：TileLang 只开 ExpCast，去掉 V 字节重排

TileLang 从原始 FP16/BF16 V 直接生成按 token 顺序存储的 `[B,H,D,Npad]` FP8 V。
预处理复用 Sage 的 per-channel 量化内核，但绕过 `transpose_pad_permute_cuda`，
仅做 padding、普通维度转置和量化。没有聚类、V 平滑、Sage 的 16-token 字节排列或反排列。
attention 内核直接 `T.copy(V, v_shared)` 后计算 PV，删除 `v_packed` 和逐块重排循环。
K 去均值、INT8 QK、ExpCast 直接公式、FP8 PV 和两级 FP32 累加保持不变。

full API 从原始 Q/K/V 每次重新处理，包含上述 padding/转置/量化；fixed core 不做布局转换。
正式验证和主尺寸测速前，在计时外将两份独立预处理结果按**逻辑 token 顺序**比较，
要求 Q/K、scales 和所有 V 字节逐位相同。用于核对的 inverse permutation 只出现在校验中。
两种 prepared state 有显式布局标记，不能直接交换给另一个内核。

本地生成代码已确认原 D128 V 重排循环中的 64 次同步消失，V buffer 在 PV 完成后才释放。
新版本的 H200 精度和性能待实测；运行命令沿用下面的一键脚本。

### 历史：修复多块 V 缓冲区竞争

首轮 H200 实测中，单 token 校验通过，但 TileLang 的 `M=65,N=129,D=64`
相对量化参考误差约 100.33%，因此没有进入性能测试。
检查旧版生成的 CUDA 发现：生产线程直接把重排后的 V 写入 PV 使用的 shared buffer，
而自动流水线在 QK 完成后即释放该组缓冲区；下一块 V 可在当前 PV 完成前覆盖它。

`9a967ba` 用明确的 `T.copy` 将原始 V 搬入 `v_packed`，消费者等待后重排至独立的 `v`，
生产者只复用 `v_packed`。D64/D128、两种 stages 的生成代码检查已覆盖这条边界。
新增 uniform-attention 定向用例：128、129、257 个 key，各块 V 不同，并用解码 V 的
均值独立核验 oracle。精度门槛不变；失败前会保存当前 case、variant、误差及生成源码。

该版已由用户在 H200 跑完验证和测速：CUPTI 中 TileLang s1/s2 为 118.9744/118.6298 ms，
同轮 vc28 为 23.6273 ms、Sage 为 24.4361 ms；这些是内核时间。当前 natural-V 改动尚无测速结果。
无需重新安装依赖或重编译 Sage 包：

```bash
git pull --ff-only origin master
bash scripts/run_expcast_tilelang_h200.sh --skip-install
```

## 一键运行

远端已有本项目 `sageattention-local==2.2.0+vc32`、Torch、Sage、FA4 和 nvcc：

```bash
git pull --ff-only origin master
bash scripts/run_expcast_tilelang_h200.sh
```

脚本创建 `.venv-tilelang-h200`，使用 `--system-site-packages` 复用现有 Torch/Sage/FA4。
按 [TileLang 官方安装方式](https://github.com/tile-ai/tilelang/blob/v0.1.15/docs/get_started/Installation.md)
安装固定版本 `tilelang==0.1.15`；约束现有 Torch 版本，使用系统 nvcc，不安装另一套 CUDA。
首次需要网络下载 TileLang 及其依赖。Python 均通过 `python3` 或该虚拟环境启动。
已有安装后可以离线复测：

```bash
bash scripts/run_expcast_tilelang_h200.sh --skip-install
```

如果自行装好 TileLang，可指定 `TILELANG_PYTHON=/path/to/python3`。
默认创建环境需要 `python3 -m venv` 可用；缺少此组件时也可指定现有 Python。
首次编译两个 CUDA 调度和多个 TileLang 验证形状需要等待；JIT 全在计时外。
不需要重编译原 Sage 包，不依赖 ncu/nvdisasm。

默认输入为 BF16、非因果 MHA，NHD 等价 `[1,37741,28,128]`。
输出：`results/h200_paper_expcast_tilelang_natural_v.json`；trace/生成的 TileLang CUDA 源码：
`results/paper_expcast_natural_v_traces/`。失败保留已完成数据，校验失败不会继续测速。

参数示例：

```bash
# 已安装依赖，只做小尺寸正确性验证
bash scripts/run_expcast_tilelang_h200.sh --skip-install --validate-only
# 小场景试跑
bash scripts/run_expcast_tilelang_h200.sh --skip-install --heads 2 --queries 1024 --tokens 1024 --trials 3
# 实际模型 post-RoPE QKV，文件含 q/k/v 三个 Tensor
bash scripts/run_expcast_tilelang_h200.sh --skip-install --qkv-file captures/qkv.pt --qkv-layout NHD
```

默认测 FA4 和独立 CUPTI 时间。仅明确要省略时使用 `--skip-flash4` 或 `--skip-cupti`。

## 与论文对齐的内容

依据 [论文 §3.3、公式 7](https://arxiv.org/html/2609.15810v1#S3.SS3)，采用：

```text
z    = FP32(score - running_max)
u    = FMA_RN(z, FP32(8*log2(e)), FP32(119.65))
code = clip(RNE(u), 0, 120)
P8   = reinterpret_E4M3(code)
mass = sum(float32(P8))
```

最大概率的 E4M3 解码值为 256。分子和分母共同保留这个倍率，最终除法抵消。
历史缩放仍为 `exp(old_max-new_max)`；ExpCast 替代的是逐元素概率指数与格式转换。

本项目新增的**具体浮点求值契约**是：先将 INT32 点积分数按 FP32 缩放、减行最大值，
再做单次 FP32 FMA/RNE。这是对公开公式的明确实现，不声称与作者未核对的 H200 二进制逐位一致。
CUDA 和 TileLang 使用同一个标量算术头文件；注意公式中的 byte view 与数值 FP8 cast 不同。

现有 `vc28_folded` 把缩放和减最大值折进 gain/bias，边界字节可能不同。
它保留为历史性能对照；与新直接公式的差值不归为“仅调度收益”。
`cuda_paper` 与 `cuda_paper_interleaved` 才是相同算术的调度对照。

本轮继承 Sage 的 INT8 per-thread Q/K、K 去均值、per-channel FP8 V、128-key 在线块，
以及 host kernel 的缩放顺序/初始化。没有实现论文的 V-Smooth、分组摊销或完整模型融合预处理；
因此本轮是 **ExpCast 单开实验**，不是完整论文复现。

## 各组做法

| 行名 | 实现 |
|---|---|
| `cuda_paper` | Sage/VC 主循环，直接公式；每两个 K fragment 分组编码、解码、FP32 求和 |
| `cuda_paper_interleaved` | 每组字节生成后立即解码和求和，缩短中间值存活时间；同一 FP32 累加顺序 |
| `tilelang_paper_s1` | 自然顺序 FP8 V、无 V 重排的 TileLang 融合循环，一阶段流水线 |
| `tilelang_paper_s2` | 相同算法和自然 V，两阶段流水线 |
| `vc28_folded` | 已测过的 vc28 折叠公式路线 |
| `sage_cpp` | 本地 Sage 原版完整 API / 原版 native core |
| `flash4` | 已安装的 FlashAttention-4，接受原始 BF16/FP16 |

CUDA/vc28/Sage core 共用 Sage 物理布局，TileLang core 使用自然 token 顺序布局。
两条预处理路线的量化数值和 scales 逐位核对，物理 V 存储顺序不同。
TileLang 不再把 Sage 布局转换为自然布局，而是直接生成自然布局。
INT8 QK 和 FP8 PV 均编译为 Hopper WGMMA；每个 128-key 块使用新的 FP32 PV 临时累加，
再更新 FP32 历史输出。Python 不迭代 attention tiles，不生成 N×N 张量。

TileLang 的 INT8 QK 输出与 FP8 PV 寄存器输入布局不同，本版通过 shared P 转换布局。
编译器负责流水线和 warp specialization，因此生成的调度与手写 CUDA 不相同。
两种 stages 是候选配置，是否更快要实测；不能仅凭 Python 代码行数或使用 DSL 推断性能。
参考 API 来自 [TileLang 官网](https://tilelang.com/) 和
[v0.1.15 attention 示例](https://github.com/tile-ai/tilelang/blob/v0.1.15/examples/flash_attention/example_mha_fwd_bhsd.py)。
TileLang 示例的 MIT 许可随附于 `experiments/expcast/TILELANG_LICENSE`；
CUDA 生成器保留 Sage 源码 Apache-2.0 许可。

## 如何读结果

- **full API**：从原始 Q/K/V 重新预处理，每次分配输出；分别包含 CUDA/Sage 和 TileLang 各自的布局生成、量化开销。
- **fixed core**：复用量化输入，含输出分配/dispatch；FA4 此列为空。
- **Sage/API、FA4/API**：相应基线完整 API 除以本行完整 API；大于 1 表示更快。
- **CUPTI**：单独测固定输入 GPU kernel 时长，不与 API 时间混用。
- 比较 `cuda_paper_interleaved / cuda_paper` 看本轮 CUDA 调度效果；比较两者与 TileLang 看实现差异。
- `vc28_folded` 与直接公式可能有边界舍入差异，另看 FP32/量化参考误差，不能仅看速度。

正式测速前：FP16/BF16 × D64/D128 × 3 个随机/单 token 配置及 3 个定向跨块配置，共 24 个输入配置，
每种新实现均检查全部输出、量化 FP32 oracle 和非默认 stream。
主尺寸再检查完整 API 与固定 core 逐元素一致，并比较所有输出行。
编码验证覆盖 65,899 个输入，包括全部 120 个 RNE 半整数边界及左右相邻 FP32 值。

CUDA 两种调度使用 0.1% 全输出相对 RMSE 门槛；TileLang/native 和量化 oracle 使用 1%。
后者允许不同硬件累加/FP32 归约顺序；这些是**实现检查**，不是模型质量标准。
默认 dense FP32 误差检查抽 128 个 query、全部 keys；`--check-queries 0` 可检查全部 query。

## 本地验证的界限

本地设备为 SM86，完整 Hopper attention 不能运行。
CUDA helper 已在本地 GPU 验证直接公式边界、极端尺度、连续 1/7/296 块、默认/非默认 stream，
两种调度的字节和 FP32 状态逐位一致。
TileLang 生成的标量编码 kernel 也在 SM86 执行，对 65,899 个输入与独立 oracle 逐字节一致。
此项通过 CUDA driver 加载生成的 cubin 执行；本地 Windows 的 TileLang Python JIT 适配器
遇到编译错误输出编码问题，未完成运行验证。

原生完整候选和 TileLang 的代表性 D64/D128、FP16/BF16、长序列配置已做 CUDA 12.4
SM90a 交叉编译。具体资源、测试数量与限制见
[本地验证记录](results/paper_expcast_tilelang_local_validation.json)。
上述为首版记录；V 缓冲区修复的验证另见
[修复验证记录](results/tilelang_v_pipeline_fix_local_validation.json)。
去掉 V 重排的本地验证见 [natural-V 验证记录](results/tilelang_natural_v_local_validation.json)。
**H200 完整输出、Python JIT 调用和性能仍由远端脚本验证，不预设速度改善。**
