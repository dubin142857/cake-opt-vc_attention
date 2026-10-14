# VC Attention：Ascend A2 / A3（DAV-2201）实验移植

这是一版新的 **C++ / Ascend C 内核 + torch_npu 接口**，对应当前 H200 路线的 **INT8 QK + ExpCast，不启用 V 聚类/平滑**。它没有把 attention 偷换成 PyTorch SDPA。预处理使用设备上的 torch_npu 运算；QK、ExpCast、在线归一化和 PV 在自定义混合内核内执行。

**当前验收状态：0.1.1 修复跨 query 任务的 V scale 缓冲区复用竞争，等待远端重新编译验收；尚未得到有效性能结果。** 前版 A3 / CANN 9.0.1 已编译并通过小形状验收，但 BF16 `[1,37741,28,128]` 的 ExpCast 输出相对量化解码参考 RMSE 为 **1.342429%**（max abs `0.0074462890625`），超过原有 1% 实现门限。这是实现一致性失败，不是相对 dense FP32 的量化损失。

2026-10-14 的定向诊断进一步缩小了范围：Q32/K37741 的 ExpCast 最终 BF16 输出与参考逐位一致；Q1025/K129 的 control/ExpCast 却分别出现约 3.43%/6.48% 误差，同一量化输入重复调用也出现 4.61%/3.22% 误差。重新预处理的所有字节和 scale 一致；隔离原始出错 Q32 块后恢复准确和确定性。这些结果指向多任务执行，而非长 K 累加本身。

代码审查发现：任务尾部 `MTE3_V` 仅阻止下一次 Vector 写入，不能阻止下一任务 MTE2 提前覆盖 `vscale`，此时上一任务的 epilogue 可能还在读它。0.1.1 将这对事件改为通过 `FetchEventID` 分配的 `MTE3_S`，形成 `旧 V 读 scale → 输出 DMA → Scalar → 新任务 metadata DMA` 的先后关系。CrossCore 握手、量化方式和 1% decoded 门限保持原样。新增多任务、跨 head、固定输入重复执行的回归验收。本地没有 CANN、torch_npu 或 NPU，不能把代码修复当作实机通过。

**这次涉及原生内核，必须重新编译**。已有一键脚本会重建 0.1.1、运行包含新回归的小形状验收，再进入原始长形状 Laser 对比；任一步失败即停止，无需重新安装 MindIE：

```bash
git pull origin master
bash scripts/run_ascend910b_laser.sh
```

需要在重建后单独复查上述定向诊断时，运行：

```bash
bash scripts/run_ascend910b_accuracy.sh
```

结果在 `results/ascend910b_accuracy_diagnosis.json` 和同名 `.log`。固定 BF16、D128、H2，分别测试 Q32/K37741 的长 K 循环、Q1025/K129 的多 query 任务，以及两者组合；短对照为 K513/K4097。比较 native FP32 原始输出、最终 BF16 输出、相同量化输入重复运行和新鲜预处理。门限仍为 decoded 1%、重复/预处理一致性 0.1%；诊断完成后若有失败会非零退出，不输出速度结论。

定向诊断脚本只复用已安装的扩展，不能代替这次重建。长形状失败时，benchmark 会先保存 dense/decoded 误差、每个 head 的误差与最差行位置，再自动隔离最差 Q32 tile，保留其原始 Q/K/V 字节、量化 scale 和 K 中心化结果，比较 K 前缀、普通 exp/ExpCast、零 Q 和常量 V。原失败仍保留，诊断不会把失败改成通过。通过定向诊断也不等于原始长形状通过。

## 与 H200 路线的差异

| 环节 | 本版 A2/A3 实现 |
| --- | --- |
| 输入/输出 | FP16、BF16；NHD `[B,N,H,D]` 或 HND `[B,H,N,D]` |
| QK | FP32 K 中心化，Q 每 32 行、K 每 128 行对称 INT8 量化；Cube INT8×INT8→INT32 |
| ExpCast | Vector 计算 E4M3 编码、查表解码；分母累加解码后的 P |
| V | 默认每通道 E4M3 量化，再解码为 FP16；没有 token 重排、聚类或 V 平滑 |
| PV | **FP16×FP16→FP32 Cube 计算，不是原生 FP8 矩阵乘** |
| 分块 | Q32×K128；每个 AIC 配两路 AIV，每路处理 16 个 query |
| 中间存储 | 每个 AIC 一份 tile scratch；D128 时 40 KiB，不分配完整 N×N 矩阵 |
| 范围 | 非因果 dense MHA、D=64/128、推理；不支持 GQA、mask、dropout、反向 |

A2/A3 的 DAV-2201 不具备这里需要的原生 FP8 Cube 路径，因此不能把 H200 的 FP8 PV 指令直接搬过来。本版保存 ExpCast/E4M3 的数值方法，用 FP16 Cube 执行解码后的乘法。Q/K 的量化粒度和 FP32 运算顺序也与 Sage CUDA 不同；它不是逐位复现 H200 或论文完整优化实现，不能据此承诺论文加速比。

## 在 910B_23 机器上启动

使用已有且版本匹配的驱动、CANN Toolkit、PyTorch、torch_npu 环境，并准备 CMake >=3.20、C++ 编译器、setuptools 和 wheel。不要在 H200 环境执行此脚本。默认使用 `python3`，不会安装或升级这些基础依赖。

```bash
git pull origin master
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/run_ascend910b.sh
```

`set_env.sh` 路径以实际安装为准。脚本依次做：读取环境和 SoC → `pip install ./ascend` 编译 → 小形状验收 → 同卡性能对比。任一步失败就停止，日志和 JSON 位于 `results/ascend910b_*`。

**不把设备标签 `910B_23` 直接传给编译器，也不猜它等于 B2/B3。** 预检在选定设备后从 `aclrtGetSocName` 读取准确 SoC，并匹配 CANN platform ini 内的 `SoC_version` 字段。A2 的 `Ascend910B1~B4/B2C` 与 A3 的 `Ascend910_93xx` 同属 DAV-2201，但编译仍传入准确 SKU，不将 A3 改名为 910B。架构依据见 [CANNBot 产品映射表](https://github.com/hicann/cannbot-skills/blob/dbf4019cae5d757820f03c9cd21e0f0b277e1701/ops/npu-arch/references/npu-hardware-params.md)。

例如运行时为 `Ascend910_9382`、继承环境为 `SOC_VERSION=ascend910_9391` 时，脚本会记录警告并将本次构建目标改为 **`Ascend910_9382`**，无需手工清理环境变量。构建目录按 SKU 隔离，native 入口还会检查实际设备与编译目标一致、AIV 数量满足每 AIC 两路的要求。发现 Toolkit 配置与目标架构冲突则停止；没有暴露 ini 时记录警告，由 CANN 编译器继续验证准确目标。保留原脚本名和 `results/ascend910b_*` 路径以兼容已有命令。

验收通过后测原来的长形状：

```bash
NPU_DEVICE=0 bash scripts/run_ascend910b.sh --skip-build --long-shape -- --warmup 2 --repeats 3 --trials 5
```

此时 shape 为 **`[1,37741,28,128]`（NHD），BF16**。初始版每个 tile 会交换 GM 中间结果并同步，还没有双缓冲流水优化，长序列可能较慢。先跑默认 1024 长度确认可执行和精度。

单独查看环境：

```bash
python3 scripts/check_ascend910b.py --device 0
```

## Python 调用

```python
import torch
import torch_npu
from vc_attention_ascend import attention

q = torch.randn(1, 1024, 2, 128, device="npu:0", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
out = attention(q, k, v, layout="NHD", expcast=True)
torch.npu.synchronize()
```

安装包名为 `vc-attention-ascend`，独立于现有 CUDA 包。没有 CPU/CUDA 自动回退。

- `expcast=False`：同一量化 Q/K/V 上使用普通指数函数的 control；仍是量化 attention。
- `v_quantize=False`：显式切换为归一化 FP16 V，是另一种数值设置，不代表默认算法。
- `check_finite=True`：默认检查输入和 FP32 预处理范围，会同步设备。测速在对同一输入完成一次检查后显式关闭重复扫描，并记录这一点。
- `prepare(...)` / `run_prepared(...)`：分别用于新鲜预处理、固定量化输入的内核调用；正常 API 均要求 NPU tensors。

## 数值契约

令 `s = (Q_int8 @ K_int8.T) * (q_scale * k_scale / sqrt(D))`。每个 K tile 更新行最大值 `m_new=max(m_old,max(s))`，历史缩放系数为 `alpha=exp(m_old-m_new)`。

ExpCast 使用以下显式顺序：

```text
x = FP32(s - m_new)
y = FP32(FP32(x * FP32(8*log2(e))) + 120)
code = RNE(clamp(y, 0, 120))
p = decode_positive_E4M3(code)
l_new = alpha*l_old + sum(p)
o_new = alpha*o_old + FP32_matmul(FP16(p), FP16(V_decoded))
output = (o / l) * v_scale
```

`code=120` 解码为 256，这个共同倍率通过分子/分母归一化消去。查表中的实际 P 值都可由 FP16 精确表达。Muls、Adds 分别舍入，**没有宣称与 CUDA FMA 折叠路径逐位一致**。控制组的分母累加 FP32 exp，而 PV 使用转换后的 FP16 P；参考程序也保留这一差别。

独立参考在 CPU 上计算，包含所有 keys，避免用 NPU 的近似 FP32 matmul 作为唯一 oracle。内核对解码参考的 1% RMSE 门限检查实现一致性；同时报告相对 dense FP32 attention 的误差，后者用于观察量化损失，不表示模型质量通过。

## 计时与验收

验收覆盖 FP16/BF16、D64/128、完整块/尾块、跨多个 K tile、两路 AIV、零输入、常量 V、不同 B/H、非默认 stream。小测试比较全部输出；长序列基准默认检查 32 个 query 和全部 keys，可用 `--check-queries` 调整。

基准默认请求四组：`vc_expcast`、`vc_control`、`cann_fusion_attention`、`mindie_laser`。CANN 基准调用 `torch_npu.npu_fusion_attention`；MindIE Laser 使用下述显式接口。未安装的可选基准会打印 SKIP 并保存原因，`--require-laser` 可要求 Laser 必须可用。

- **full API**：新鲜 K 中心化和 Q/K/V 量化、attention、输出类型/布局转换。
- **fixed core**：复用量化输入，仍包括 host tiling、分配、dispatch、输出转换。
- 都是同步 API 墙钟毫秒，**不是纯 NPU kernel 时间**。只在同一张 NPU 上比较；不与 H200 的 CUPTI 数字混算。

## 与 MindIE Laser Attention 对比

**安装对象是 [MindIE-SD](https://github.com/Ascend/MindIE-SD)，Python 包名 `mindiesd`。** 在检查的 MindIE-LLM commit `9446ed407f817b2bd937ca1e4e63907b2f2eea7f` 中没有找到 Laser Attention；不能只安装 MindIE-LLM 然后期待这个算子存在。请按目标机 CANN / PyTorch / torch_npu 版本选择配套的 MindIE-SD，参见[官方安装说明](https://github.com/Ascend/MindIE-SD/blob/71098c20ff90838d1fd1ca0a40515c751a3869b0/docs/zh/installation.md)。此仓库的脚本不会替你升级基础环境或安装 MindIE。

安装好后，在 VC 仓库内运行：

```bash
git pull origin master
bash scripts/run_ascend910b_laser.sh
```

这个固定入口不需要参数：默认 `python3`、逻辑 NPU 0、BF16 `[1,37741,28,128]`，预热 2 次、每轮 3 次、共 5 轮，检查 32 个 query 和全部 keys。比较四个后端，结果写入 `results/ascend910b_vs_laser.json`。每次都会先检查环境、编译安装 VC、完成小形状精度验收，再进入长形状对比；失败即停止。MindIE-SD 仍需预先安装。

脚本按实际设备识别 A2/A3。`torch` 版本带 `+cpu` 不代表 NPU 运算没有启用，预检会通过 `torch_npu` 在所选设备上执行一次运算。测速仍应选择空闲逻辑设备；`npu-smi` 的板卡编号、芯片 Phy-ID 和可见设备映射后的逻辑编号不可混用。

若没有已激活的 CANN 环境，脚本会尝试系统或用户目录下的标准 `set_env.sh`。非标准安装仍需先激活对应环境。已有 `PYTHON_BIN` / `NPU_DEVICE` 环境变量时会保留选择，否则使用上述默认值。

需要自选参数时仍可用通用入口：`bash scripts/run_ascend910b.sh --with-laser` 首测 `[1,4096,2,128]`。VC 已编译时可加 `--skip-build`。

测原来的 BF16 `[1,37741,28,128]`：

```bash
bash scripts/run_ascend910b.sh --with-laser --skip-build --long-shape -- --trials 5 --output results/ascend910b_vs_laser.json
```

如果先安装好了 MindIE，还没编译 VC，可以独立测两个基准，不依赖我们的 native `.so`：

```bash
python3 scripts/benchmark_ascend910b.py --backends mindie_laser cann_fusion_attention --require-laser --queries 4096 --keys 4096 --heads 2 --dim 128 --output results/ascend910b_laser_baselines.json
```

对比口径如下：

| 项目 | MindIE Laser 行 |
| --- | --- |
| 实际调用 | `torch.ops.mindiesd.la.default`，检查原生 schema；取第二个返回 tensor |
| 预处理/后处理 | 使用安装版本的 `AscendLaserAttention.la_preprocess_input` / `la_postprocess_output`，再返回 contiguous HND |
| 固定设置 | dense、非因果、无 mask/dropout，`is_highPrecision=True` |
| 输入 | 同一份原始 Q/K/V；包装层将 BF16 转为 FP16，序列补齐到 256，D 补齐到 128 |
| 缩放和尾块 | 用原始 D 的 `1/sqrt(D)`；按上游约定传 K 的尾部 padding 长度 |
| full API | 每次重新转换/补齐 Q/K/V，原生 attention，输出类型转换/裁剪/连续化 |
| fixed core | 复用转换/补齐后的输入，包含原生分配/调用及输出后处理；没有声称和 VC 使用相同量化输入 |
| 比值 | `Laser/API = Laser full API / 当前行 full API`，大于 1 表示当前行更快 |

这里测的是 **Laser 后端及其预处理/后处理**，不包含模型、服务调度或 MindIE 通用 attention 的自动选型。通用 `attention_forward` 在短序列下即使指定 Laser，也可能回退为其他算子；环境变量也能覆盖选择。因此本适配器显式调用注册的 `mindiesd::la`，不把 Fused Attention 混报为 Laser。

基准在计时前运行额外的真实 NPU 检查：FP16/BF16、D64/128、完整与不完整 256 尾块、常量 V 通道、full/prepared 输出一致性。Laser 对 dense FP32 参考使用 1% 实现检查门限，full/prepared 使用 0.1%；这些不是模型质量标准。若已加载的 Laser 执行/精度检查失败，脚本直接报错，不静默替换。

JSON 记录安装版本、原生 op schema、已加载动态库、上游包装文件路径/SHA256、实际输入/补齐形状及精度设置。`inspected_commit` 只是编写适配器时检查的源码版本，**不代表远端安装包就是该提交**。当前没有本地 NPU/MindIE 运行或性能数据；新增 CPU 测试只验证适配器传参与失败处理。

## 同步审查与未完成的实机确认

已运行 CANNBot `sync_audit.py` 和 `ascendc_flow_analyzer.py`。结构化结果保存在 `results/ascend910b_local_validation.json`。

0.1.1 的本地复核记录见 `results/ascend_task_reuse_fix_local_validation.json`。新增修复对应的缓冲区生命周期如下；这条跨循环 WAR 竞争由人工审查发现，regex 数据流扫描未检出。

| 缓冲区 | 前一任务最后访问 | 后一任务覆盖 | 0.1.1 同步链 |
| --- | --- | --- | --- |
| `vsBuf / vscale` | Vector `Mul(acc, acc, vscale)` 读取 | MTE2 `DataCopy(vscale, vsg[head*D], D)` | V_MTE3 → 输出 DataCopy → MTE3_S → Scalar 才能发出下一 DMA |
| `accBuf / acc` | MTE3 写出结果时读取 | Vector `Duplicate(acc, 0)` | 输出 DataCopy → MTE3_S → Scalar 才能发出下一 Vector 写入 |

事件方向和参数依据 [CANN 9.0 SetFlag/WaitFlag 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0270.html)。本次没有调整跨核 flag 或删除 barrier。

目前仍有 **SYNC-07 红线候选：Matmul API 与自定义 CrossCore flag 的潜在冲突**，以及两条 SYNC-09 barrier 数量性能提示。此报告没有将红线候选标为通过。数据流扫描使用 regex 前端，未检出问题不构成硬件正确性证明。

代码选择本地 Cube `MatmulImpl`，没有 KFC 客户端或 `REGIST_MATMUL_OBJ`；显式 mode-2 flags 为 8/9/10/15，避开技能文档列出的 Matmul 0..7 和 SyncAll 11..14。AIC/AIV 的实际握手为：

```text
AIC: QK -> S_READY -> wait P_READY -> PV -> O_READY -> wait O_USED
AIV×2: wait S_READY -> P -> P_READY -> history update -> wait O_READY -> consume PV -> O_USED
```

这些是设计依据，尚不能关闭审查项。仍需按目标 CANN 的 `MatmulImpl` 头文件核对内部 flag 和资源分配，验证 mixed 1:2 派发、双路完成条件与非默认 stream。`PipeBarrier` 先按依赖保留，跑通后再依据实际 profiler 优化，不能只为了去掉提示而删除。

构建错误回传 `results/ascend910b_environment.json` 和 `results/ascend910b_build.log`；运行问题回传 `results/ascend910b_validation.json` / `.log`。源码没有依赖远端下载，预装好工具链后可离线编译。

## 来源

参考 [CANNBot skills](https://github.com/hicann/cannbot-skills/tree/dbf4019cae5d757820f03c9cd21e0f0b277e1701) 中的硬件、tiling、Ascend C API、直调扩展和同步审查技能；来源清单与官方 API 链接在 `ascend/SOURCES.json`，上游许可随包保留。移植目录与原 CUDA 代码分离。
