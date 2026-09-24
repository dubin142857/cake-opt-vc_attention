# 0.3.0：批量执行优化

大尺寸 BSHD `[1,37741,28,128]` 的五后端对比入口为
`bash scripts/benchmark_37741.sh`，形状映射和计时范围见 [BENCHMARK.md](BENCHMARK.md)。
下文的调用次数和命令默认对应原来的 1024 长度小场景。

0.2.1 在用户提供的 H200 `[1,2,1024,128]` BF16 消融中，基础 FP8 约 2.3925 ms、
完整 VC 约 3.0398 ms。本轮改变执行组织，保留 Q/K/V 量化、ExpCast、V 均值恢复及
KV tile 的在线 Softmax 顺序。尚无 0.3.0 的 H200 延迟结果，不能把调用次数下降当作加速比。

## 实际改动

- 默认 `Config(cuda_schedule="batched")`：将最多 1024 个 Q 行合成较大的矩阵乘，
  batch/head 通过 cuBLASLt strided batch 并行执行。继续逐个处理 KV tile，完整分数矩阵不会落盘。
- Softmax 同时处理各头和当前 Q chunk 的所有行。Q scale 仍按原来的 `q_block` 选取，
  K scale 仍按原来的 `k_quant_block` 选取，并未重新划分量化分组。
- 首个 KV tile 在计算时初始化状态，最后一个 KV tile 合并累加与输出归一化。
- 批量路径按线程、cuBLASLt handle、矩阵尺寸和 batch stride 缓存最多 32 个 GEMM plan，
  不缓存输入、输出或工作区，保留当前 CUDA stream 语义。
- Q/K/V 有限值检查合并为一个 CUDA kernel，保留 NaN/Inf 拒绝行为；主机等待从三次变成一次。
  支持 FP32/FP16/BF16 和非连续、广播 stride。没有通过关闭校验缩短计时。

默认形状的核心调用数（不含聚类、预处理、校验）：

| 项目 | 原串行路径 | 新批量路径 |
| --- | ---: | ---: |
| cuBLASLt GEMM 调用 | 256 | 16 |
| 状态/Softmax/累加/输出 kernel 启动 | 288 | 16 |

每个 cuBLASLt 调用自身可能启动多个 GPU kernel；该表不是 profiler 实测事件数。
新路径仍有独立的 QK、Softmax、PV 和累加步骤，`fully_fused_attention` 仍为 false。
GEMM 的 M 维和批量模式改变可能选择不同累加路径，因此不承诺与串行输出逐位一致。

暂存空间按 `B*H*query_chunk` 增长，`query_chunk <= 1024`，与当前 KV tile、V width 相乘；
长 Q 分多个 chunk 执行。相比串行路径会使用更多 scratch，以换取并行度。

## H200 验证与测速

```bash
git pull --ff-only origin master
bash scripts/validate_h800.sh
python3 scripts/benchmark_cuda.py --backends vc --vc-schedule-compare --vc-reuse-layout --warmup 10 --repeats 100 --trials 5 --output results/h200_optimized.json
```

| 行 | 范围 |
| --- | --- |
| `vc` | 新路径，每次重新聚类，量化/分配/验证均计时 |
| `vc_serial` | 保留的旧串行路径，包含原有限值检查和每次 GEMM plan 构建 |
| `vc_reuse` | 新路径，复用已建立的聚类排列，仍重算当前 Q/K/V、均值和量化 |

`Scheduling speedup vc` 是旧完整调用耗时 / 新完整调用耗时，包含批量调度、plan 缓存与
校验合并的共同收益。不是纯 GPU kernel 加速比。
`vc_reuse` 预先执行 step=0，计时 step=1、total_steps=20，V-Smooth 保持开启；
它仅代表已有缓存时的调用，不代表首步或真实多步轨迹的平均耗时。

加 `--vc-ablation` 可比较各数值配置；加其他后端可同时和 FA2/3/4、Sage 比较：

```bash
python3 scripts/benchmark_cuda.py --vc-schedule-compare --warmup 10 --repeats 100 --trials 5 --output results/h200_all_optimized.json
```

原 Python 调用自动使用新路径。需要回到旧路径时显式设置：

```python
from vc_attention import Config, attention
out, stats = attention(q, k, v, Config(backend="cuda_fp8", expcast=True, v_smooth=True, cuda_schedule="serial"))
```

原 `diagnose_cuda_accuracy.py` 随默认配置检查新批量路径。
`diagnose_cuda_payloads.py` 为了保持逐 tile GEMM 几何相同，明确使用串行路径；
其 `replay_vs_production_CUDA` 不再代表默认的批量调度。数值契约版本仍为 2，新增调度版本为 1。

## 定位剩余瓶颈

对于长时间没有输出的大场景，先用单次阶段诊断，默认 BSHD `[1,37741,28,128]`：

```bash
python3 scripts/diagnose_cuda_runtime.py --output results/h200_vc_runtime.json
```

它会先核验实际安装包与原生调度接口，然后同步测量一次完整调用各阶段。
旧 pip 安装不会自动随 `git pull` 更新；版本不匹配时需重跑 `bash scripts/validate_h800.sh`。
若停在原生 forward，才继续分析 CUDA GEMM、Softmax 和累加；若停在聚类，先定位聚类。
部分阶段报告会即时保存，但没有远端测量前不能仅凭代码确定实际阻塞位置。

```bash
python3 scripts/profile_cuda.py --output-dir results/h200_profile
```

分别导出三条路径的 `.trace.json`、CPU operator 表和 `summary.json`。
trace 可在 Perfetto 或 Chrome trace 查看器中打开。重点查看有限值检查引起的等待、
聚类相关 ATen 操作、GEMM、`batched_softmax` 和 `batched_accumulate`。
事件统计分别汇总 CUDA kernel 与 CUDA runtime，不能相加当作总延迟。
Profiler 自身有开销，实际加速比始终用未开启 profiler 的 benchmark 测量。
需要 PyTorch 的 CUDA/CUPTI profiling 支持，缺少 CUDA kernel 事件会明确报错。

## 验证边界

本地 Windows / RTX 3060 可编译 sm_86、sm_90，运行有限值检查、批量 Softmax、累加和归一化。
完整测试结果为 **178 passed, 38 skipped**；36 项 Hopper 和 2 项 NPU 因硬件条件跳过。
这些检查覆盖多个头、不同 Q scale、共享 K scale、尾块、非零 Q chunk 起点、首块 NaN 垃圾状态，
与旧内核逐项严格对拍。Hopper FP8 GEMM 在本机不能执行。
Hopper 测试额外覆盖多个 batch/head、长 Q 跨 chunk、不同 Q/V 宽度、有符号输入、两种 P 模式、
平滑开关、不同头的已知常量输出、plan 复用以及当前 stream；必须在 H200 执行验证脚本。
本地 PyTorch profiler 仅提供 CPU activity，CUDA trace 导出需要在远端验证；trace 汇总逻辑已做测试。

接口依据：[NVIDIA cuBLASLt layout/batch 文档](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublasltmatrixlayoutattribute-t)。
