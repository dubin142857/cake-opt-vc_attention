# 0.4.0：原生 FP8 融合 Attention 主体

本文保留 0.4.0 实现记录。当前 0.5.0 默认融合核心增加异步搬运，
新旧核心的同场对比和编译验证步骤见 [PIPELINED_CUDA.md](PIPELINED_CUDA.md)。

新增显式 `Config(backend="cuda_fp8", cuda_schedule="fused")`。Attention 主体在一次
CUDA kernel 启动中完成所有 KV 块的 QK、Softmax/ExpCast、PV、V 均值恢复、在线累加和归一化。
QK/PV 使用 Hopper `wgmma.mma_async.m64n128k32.f32.e4m3.e4m3` Tensor Core 指令。
没有 cuBLASLt 调用，没有主机上的 KV tile 循环，也没有全局显存的 scores/P/PV/累加暂存。

这是第一版融合核心，不是论文作者内核，也还没有经过 H200 性能调优。
目前仍独立执行输入检查、V 聚类、KV gather、QK 变换与 Q/K/V 量化，输出最后转回输入 dtype。
`native_fused_attention=True` 表示 Attention 主体融合；`preprocessing_fused=False` 表示预处理仍独立。
`fully_fused_attention` 这个旧的全流程字段仍为 false。

## 远端编译与运行

要求 Hopper（H800/H100/H200）、已有 CUDA PyTorch、**nvcc >= 12.4**。
构建会包含 `sm_90a` 架构以启用 WGMMA；用户当前 CUDA 13.0 满足要求。

```bash
git pull --ff-only origin master
bash scripts/validate_h800.sh
bash scripts/benchmark_37741.sh --backends flash3 sage vc --warmup 1 --repeats 1 --trials 1 --output results/h200_fused_first.json
```

先用这条短测速验证完整 `[1,37741,28,128]` 场景能运行；稳定后移除三个计时参数覆盖，
按多个 trial 的中位数比较性能。计时仍包括每次调用的聚类、量化、检查、分配等，精度抽取
128 个 query、保留全部 key。没有把诊断模拟、预量化缓存或聚类缓存冒充完整 API 性能。

0.4.0 的 benchmark/大尺寸入口默认选择 **fused**，遇到旧扩展会报错，不会静默回退。
Python `Config` 默认仍为 batched，保留已有 API 的分块兼容性；代码里使用新核心需显式选择：

```python
from vc_attention import Config, attention
cfg = Config(backend="cuda_fp8", cuda_schedule="fused", expcast=True, v_smooth=True)
# VC 布局为 BHND，因此本例 q/k/v.shape = [1, 28, 37741, 128]。
out, stats = attention(q, k, v, cfg)
assert stats["native_fused_attention"]
```

需要逐阶段看耗时：

```bash
python3 scripts/diagnose_cuda_runtime.py --schedule fused --output results/h200_fused_runtime.json
```

旧批量核心可以用 `--vc-schedule batched` 测量。不要在大场景启用
`--vc-schedule-compare`，该选项会额外运行原来的数百万调用串行路径。

## 内核结构与边界

- 一个 CTA 含 4 个 warp，作为一个 warpgroup 协同执行 WGMMA；其输出片段每个 warp
  负责 16 个 Q 行，CTA 共 64 行。
- Q/KV/P 放入共享内存，score、行最大值、分母和输出累加保存在寄存器。
  K 和 V 分阶段复用共享内存，总静态 shared memory 32 KiB。
  使用 128B XOR swizzle，显式 async proxy fence、WGMMA commit/wait 保证共享内存可见性和结果完成。
- 每个 CTA 在 GPU 内依次处理全部 KV 块；CTA 之间并行处理不同 Q 块和 batch/head。
- 保留原有 Q/K 量化分组、每通道 V scale、scaled mean 存储精度、ExpCast FMA/RNE、
  掩掉尾部 padding 的规则。普通 E4M3 模式分母仍使用量化前概率，ExpCast 使用解码后概率。
- 当前要求 `kv_block=128`，head dimension 与 V dimension 各为 1..128；支持不整除的序列尾部、
  不同 Q/K 序列长度、非连续输入和不同 V 宽度。仅 dense、noncausal、forward MHA。
- 当前没有 TMA、warp specialization、跨 tile 异步多级流水线或融合聚类；不能承诺追平 Sage。
- FP8 MMA 的累加顺序与 cuBLASLt 不同，不要求逐位一致。实现一致性与模型生成质量分别验证。

## 验证方式

`tests/test_cuda_fused.py` 包含多 head、多 batch、有符号值、共享 K scale、Q/K/V 尾块、
最大值跨 KV 块变化、不同 mean dtype、非连续输入、当前 stream 和更新输入的检查。
Hopper 上会运行原生 FP8 路径与独立 Python 量化参考、旧批量核心的对照。

`debug_fused_emulated` 是**只供小张量测试**的另一个扩展入口：在同一内核中将 MMA
替换成 warp shuffle 加 FP32 标量乘加，以便在没有 FP8 Tensor Core 的开发 GPU 上检查
片段布局、共享内存通信与递推。公共 `attention` 和所有测速脚本都不调用它。
模拟通过不能证明真实 FP8 指令精度或速度；Hopper 原生测试必须在远端执行。

本地验证（Windows / RTX 3060 / PyTorch 2.3.1+cu121，使用独立 CUDA 12.4 编译工具）：

- 编译包含 `sm_86` 诊断和 `sm_90a` 原生目标的 0.4.0 wheel 并成功安装。
- 全套 **209 passed, 64 skipped**；Hopper 与 NPU 检查因硬件条件跳过。
- 最终诊断内核的 24 项测试通过 Compute Sanitizer memcheck，0 errors；
  6 项跨块最大值/均值恢复测试通过 racecheck，0 errors / 0 warnings。
- 两个原生 WGMMA 核心均为 32 KiB shared memory、255 registers/thread、0 spill loads/stores
  （本地 nvcc 12.4 编译结果，不代表 CUDA 13 编译后的资源数）。
- 原生 WGMMA 的共享内存描述符、异步代理语义、硬件累加精度及完整 H200 时延仍需远端验证。

可机器读取的记录见 [results/fused_local_validation.json](results/fused_local_validation.json)。

必须检查生成的机器码：Hopper 的 `mma.sync` FP8 写法可能被编译为 FP16 HMMA。
正式实现选择 WGMMA，并以反汇编中的 E4M3 QGMMA 验证，不能只凭 PTX 参数声称原生 FP8。
可运行 `python3 scripts/check_fused_binary.py` 检查实际安装的扩展（需要 Toolkit 的 cuobjdump）。

指令和片段布局依据：[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)。
