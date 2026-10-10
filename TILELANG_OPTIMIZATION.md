# TileLang 独立优化实验

2026-10-10。新增 `scripts/tilelang_expcast_opt.py` 和独立入口
`scripts/benchmark_tilelang_opt.py`，针对现有 Hopper INT8 QK / ExpCast / FP8 PV。
原始 TileLang 实现继续作为对照。本轮候选尚未在 H200 执行，不能据此宣称加速。

## 候选

| 候选 | 改动 | 需要实测的问题 |
| --- | --- | --- |
| `cached_s1/s2` | 每个 query tile 缓存 Q scale，每个 key tile 缓存 K scale，显式广播到 score fragment | 编译器是否已消除原实现的重复读取，以及新调度是否受益 |
| `mma_qk_s1/s2` | 在缓存 scales 的基础上，K 从 shared 复制到 fragment，使 INT8 QK 使用 `mma.sync`；FP8 PV 继续使用 WGMMA | QK 等待和布局成本的变化能否抵消额外 K 寄存器复制 |

两组保持 BM64/BN128、128-key ExpCast 分块、per-thread INT8 QK、自然顺序
`[B,H,D,Npad]` FP8 V、decoded-P 分母和逐块独立 FP32 PV 累加。
V 平滑/分组关闭；直接 ExpCast 的 FP32 运算顺序与原实现一致。
FP16/BF16、D64/128、stage 1/2 均支持。只用于非因果 MHA 和 Hopper SM90a。

本地 D128/BF16/s1 的 PTX 确认：原实现 QS/KS 均在 key 循环中加载，
`cached` 把 QS 移到循环前。但 NVRTC 已把原实现的逐元素重复读取合并，
所以改善的是每线程 QS 随 key tile 重复加载，不能按源码估计大量 load 被消除。
最终 SASS 和运行时间仍需实测。

## 在已有 H200 环境运行

已有 `.venv-tilelang-h200`、TileLang 0.1.15、Torch、Sage 量化组件和 CUDA 编译环境时：

```bash
bash scripts/run_tilelang_opt_h200.sh
```

脚本复用现有环境，不安装依赖、不构建项目的 C++ 扩展。测试默认同时包含已安装的
Sage attention 和 FlashAttention-4 对照。只测 TileLang 时：

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

默认输出 `results/h200_tilelang_opt.json`，生成源码和 CUPTI traces 单独放在
`results/tilelang_opt_traces/`。并发的 C++ 实验应使用不同 GPU 或错开性能测量，
避免资源竞争使时间失真。可用 `--output`、`--trace-dir` 给每轮设置独立路径。

## 校验和计时

测速前执行 24 个小尺寸用例，覆盖两种 dtype、两个 D、单 token、Q/K 尾块、
多 key tile 和各块 V 不同的均匀 attention。独立解码参考检查所有 query 行；
V 的逻辑 FP8 字节和 scales 用另一条 Sage 预处理路径逐位核验。
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

- 12 项 CPU 辅助测试：V inverse permutation 的全部 FP8 bit patterns、CLI 和候选隔离。
- 16 项候选代码生成测试：两种 dtype、D64/128、stage 1/2、两个候选；
  检查低精度 GEMM 类型、Q scale 读取位置和 K/V buffer 释放时序。
- 原始 TileLang 的 8 项代码生成回归测试。
- 原始 baseline 和两个候选共 24 份源码，经 CUDA 12.4 NVRTC 和 ptxas
  交叉编译为 SM90a cubin，均为 240 registers，stack/spill 为 0。
  Windows NVRTC 检查使用标准整数头和 `cuda::std` 兼容包装；这不替代远端 NVCC 和运行校验。

记录在 `results/tilelang_opt_cross_compile.json`，本地源码/PTX/cubin 放在
`results/tilelang_opt_local_sources/`（生成文件不纳入 Git）。

测试命令（已有 TileLang 0.1.15 的 Python）：

```bash
python -m pytest tests/test_tilelang_opt_benchmark.py \
  tests/test_tilelang_opt_codegen.py tests/test_tilelang_expcast_codegen.py -q
```

另试过缩短 max/sum 依赖链的归约布局，但 TileLang 0.1.15 生成了未初始化值；
该候选已撤掉，不在远端脚本中。最终速度与运行正确性以 H200 JSON 为准。
