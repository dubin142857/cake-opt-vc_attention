# 0.5.0：融合内核的异步搬运流水线

0.5.0 保持 VC 的量化和在线累加规则，改动融合核心的数据搬运。
`cuda_schedule="fused"` 默认选择新路径；设置 `cuda_fused_pipeline=False`
可以在同一个安装包中运行保留的 0.4.0 融合核心。

用户提供的 H200 / CUDA 13.0 / BF16 / BSHD `[1,37741,28,128]` 日志中，
0.4.0 完整 API 单次测量为 243.9656 ms；另一次 profiler 的融合核心为
228.7975 ms/call，V 预处理为 3.9526 ms/call。两次测量不能严格相减。
这些数据用于选择优化方向，不是 0.5.0 的测速结果。

## 实现

- 保持 CTA 64 个 Q 行、128 个线程、KV tile 128，以及相同的 FP8 WGMMA 指令和乘加顺序。
- 将 K/V 共用的 16 KiB 存储改为各自独立的 16 KiB 缓冲区。加上 Q/P，总 shared memory
  从 32 KiB 增至 48 KiB。Q/K/V 用 16 字节 `cp.async` 直接从全局内存搬到 shared memory。
- 搬运当前 V 时执行 QK 与 Softmax/ExpCast；搬运下一块 K 时执行 PV 与输出更新。
  最后一块不发出越界预取；不足 128 的通道和 Q 尾行通过异步拷贝的 zero-fill 置零。
- 每次消费异步拷贝前执行 `cp.async.wait_group 0` 和 CTA barrier；复用 K/V/P 前保证所有
  使用者结束。WGMMA 的 proxy fence、commit/wait 保留，暂未交错多个未完成的 WGMMA group。
- 没有 TMA、warp specialization，也没有改变 Q/K 量化分组、V scale/mean、ExpCast 编码、
  概率分母或输出 dtype。聚类、校验、量化仍在完整 API 的计时范围内。

实现依据：[NVIDIA PTX cp.async 与等待语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)。

## 远端验证和同场 A/B 测速

```bash
git pull --ff-only origin master
bash scripts/validate_h800.sh
bash scripts/benchmark_37741.sh --backends flash3 sage vc --vc-fused-compare --output results/h200_pipeline_compare.json
```

这条测速使用完整 `[1,37741,28,128]`，默认 warmup=2、repeats=5、trials=3，
随机交错测试顺序。`vc` 是新流水线；`vc_fused_baseline` 是保留的 0.4.0 核心。
二者显示相同包版本 0.5.0，但 `config.cuda_fused_pipeline` 分别是 true/false。
二者都重新聚类和量化，没有复用预处理结果。额外打印 baseline/pipeline 加速比，
以及最后一个计时输出在同一组 query 样本上的直接误差。
默认精度检查抽取 128 个 query、保留全部 key；样本相同不能证明完整输出或模型质量。

同场查看两个内核的 GPU 耗时：

```bash
python3 scripts/profile_cuda.py --batch 1 --queries 37741 --tokens 37741 --heads 28 --dim 128 --dtype bfloat16 --vc-schedule fused --vc-fused-compare --warmup 3 --iterations 3 --output-dir results/h200_pipeline_profile
```

`fused_attention_pipeline` 是新核心，`fused_attention` 是对照核心。Profiler 用于定位，
正式延迟以 benchmark 为准。无需启用 `--vc-schedule-compare`，那个选项会运行很慢的旧串行路径。

Python 调用：

```python
from dataclasses import replace
from vc_attention import Config, attention

cfg = Config(backend="cuda_fp8", cuda_schedule="fused", expcast=True, v_smooth=True)
# q/k/v 的布局为 BHND，本例 [1, 28, 37741, 128]。
out, stats = attention(q, k, v, cfg)
control, _ = attention(q, k, v, replace(cfg, cuda_fused_pipeline=False))
assert stats["cuda_fused_pipeline"]
```

## 验证范围

本地只有 RTX 3060，可运行共享同一搬运/Softmax/更新代码的诊断入口，原生 WGMMA
需要 H200 复测。增加了新旧核心逐元素对拍，覆盖 1..769 个 key、整块/尾块、
不同 Q/K 量化分组、1..128 通道、三种 mean dtype 和四种平滑/ExpCast 组合。
原有独立 Python 量化参考、已知答案、非连续输入和 CUDA stream 测试继续保留。

本地全套测试 **236 passed, 90 skipped**（Hopper/NPU 条件跳过）；24 项新旧诊断
核心对拍逐元素一致。48 项诊断内核测试分别通过 Compute Sanitizer memcheck 与 racecheck，
0 errors / 0 warnings。记录见 [results/pipeline_local_validation.json](results/pipeline_local_validation.json)。

本地 nvcc 12.4 编译的新原生内核为 48 KiB shared、255 registers/thread、0 spill。
实际寄存器数没有下降，本轮不宣称提高了寄存器所允许的驻留块数。远端 CUDA 13
可能生成不同代码；性能和资源取舍以远端为准。

可用 Toolkit 的 cuobjdump 检查实际安装的二进制：

```bash
python3 scripts/check_fused_binary.py
```

检查同时验证两种 P 模式、两条融合路径的 E4M3 QGMMA，以及流水线中的 LDGSTS
异步拷贝指令，报告 local-memory load/store 指令数。原生内核没有 FP16 HMMA 替代。
