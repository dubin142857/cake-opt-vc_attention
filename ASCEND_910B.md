# VC Attention：Ascend 910B 实验移植

这是一版新的 **C++ / Ascend C 内核 + torch_npu 接口**，对应当前 H200 路线的 **INT8 QK + ExpCast，不启用 V 聚类/平滑**。它没有把 attention 偷换成 PyTorch SDPA。预处理使用设备上的 torch_npu 运算；QK、ExpCast、在线归一化和 PV 在自定义混合内核内执行。

**当前验收状态：本地 CPU 数值契约测试通过；尚未在 910B 上编译或运行，不是已验收的生产算子。** 本地没有 CANN、torch_npu 或 NPU。CANN 版本兼容、混合内核同步、实际精度和性能都需要目标机验证；同步审查的待核对项见下文。

## 与 H200 路线的差异

| 环节 | 本版 910B 实现 |
| --- | --- |
| 输入/输出 | FP16、BF16；NHD `[B,N,H,D]` 或 HND `[B,H,N,D]` |
| QK | FP32 K 中心化，Q 每 32 行、K 每 128 行对称 INT8 量化；Cube INT8×INT8→INT32 |
| ExpCast | Vector 计算 E4M3 编码、查表解码；分母累加解码后的 P |
| V | 默认每通道 E4M3 量化，再解码为 FP16；没有 token 重排、聚类或 V 平滑 |
| PV | **FP16×FP16→FP32 Cube 计算，不是原生 FP8 矩阵乘** |
| 分块 | Q32×K128；每个 AIC 配两路 AIV，每路处理 16 个 query |
| 中间存储 | 每个 AIC 一份 tile scratch；D128 时 40 KiB，不分配完整 N×N 矩阵 |
| 范围 | 非因果 dense MHA、D=64/128、推理；不支持 GQA、mask、dropout、反向 |

910B/A2 不具备这里需要的原生 FP8 Cube 路径，因此不能把 H200 的 FP8 PV 指令直接搬过来。本版保存 ExpCast/E4M3 的数值方法，用 FP16 Cube 执行解码后的乘法。Q/K 的量化粒度和 FP32 运算顺序也与 Sage CUDA 不同；它不是逐位复现 H200 或论文完整优化实现，不能据此承诺论文加速比。

## 在 910B_23 机器上启动

使用已有且版本匹配的驱动、CANN Toolkit、PyTorch、torch_npu 环境，并准备 CMake >=3.20、C++ 编译器、setuptools 和 wheel。不要在 H200 环境执行此脚本。默认使用 `python3`，不会安装或升级这些基础依赖。

```bash
git pull origin master
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/run_ascend910b.sh
```

`set_env.sh` 路径以实际安装为准。脚本依次做：读取环境和 SoC → `pip install ./ascend` 编译 → 小形状验收 → 同卡性能对比。任一步失败就停止，日志和 JSON 位于 `results/ascend910b_*`。

**不把设备标签 `910B_23` 直接传给编译器，也不猜它等于 B2/B3。** 预检从 `aclrtGetSocName` 读取 SoC，记录 CANN platform 配置证据，再传入 `SOC_VERSION`。若 Toolkit 编译配置采用不同名称，应依据该机器 Toolkit 的实际配置设置 `SOC_VERSION`；不能任意换成别的芯片。

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

基准输出三组：`vc_expcast`、`vc_control`、`cann_fusion_attention`。CANN 基准调用 `torch_npu.npu_fusion_attention`，如果当前版本/shape 不支持，会保存错误原因，不填造性能数字。

- **full API**：新鲜 K 中心化和 Q/K/V 量化、attention、输出类型/布局转换。
- **fixed core**：复用量化输入，仍包括 host tiling、分配、dispatch、输出转换。
- 都是同步 API 墙钟毫秒，**不是纯 NPU kernel 时间**。只在同一张 NPU 上比较；不与 H200 的 CUPTI 数字混算。

## 同步审查与未完成的实机确认

已运行 CANNBot `sync_audit.py` 和 `ascendc_flow_analyzer.py`。结构化结果保存在 `results/ascend910b_local_validation.json`。

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
