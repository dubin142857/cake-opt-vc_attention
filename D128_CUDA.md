# 0.6.0：D128 专用融合核心

本轮针对 BSHD `[1,37741,28,128]` 的主要耗时优化 Attention 核心。
用户提供的同场 H200 测速中，0.5.0 完整 API 为 **129.7160 ms**，
保留的 0.4.0 核心为 244.9036 ms。另一次三轮 profiler 中，0.5.0
核心约 113.18 ms/call，其余 GPU kernel 合计约 14.89 ms/call。
Profiler 和无插桩测速不能严格相减，也不能将 CPU 等待时间加到 GPU 时间上。

这些是上一版的数据；**0.6.0 的 H200 延迟仍需远端实测**。

## 实现与选择条件

`Config(backend="cuda_fp8", cuda_schedule="fused")` 默认自动选择专用核心，条件是：
Q/K 和 V 维度都为 128，`q_block=128`、`k_quant_block=256`、`kv_block=128`、
`mean_dtype="float16"`。输入可为 FP16、BF16 或 FP32；均值存储精度与输入精度是两回事。

- `csrc/fused_d128.cuh` 固定通道宽度、量化分组和均值类型，简化地址/scale 计算。
- 将完整 Q 分块和不足 64 行的 Q 尾块分为两个 kernel。主体去掉 Q 边界判断；
  KV 循环也分为完整 128-token 分块和最后的不足整块分支。
- 每次 Attention 核心启动 1 或 2 个 kernel；37741 长度使用 2 个。
  这是核心的提交次数，聚类、量化、转换等预处理另计。
- 保留 64 Q 行/CTA、128 线程、48 KiB shared memory、异步 K/V 拷贝和
  原生 E4M3 WGMMA。Q/K/P/V 量化、ExpCast、分母、均值恢复与 FP32 累加顺序保持不变。
- 不满足条件时选择 0.5.0 通用融合核心。`cuda_fused_specialize=False` 也可显式选择它；
  `cuda_fused_pipeline=False` 则选择 0.4.0 核心。返回的 `stats["cuda_fused_variant"]`
  分别为 `d128`、`pipeline`、`baseline`。Python Config 的默认 schedule 仍为 `batched`。

## 编译与同场对比

```bash
git pull --ff-only origin master
bash scripts/validate_h800.sh
bash scripts/benchmark_37741.sh --backends flash3 sage vc --vc-specialize-compare --output results/h200_d128_compare.json
```

需要 nvcc >= 12.4；验证脚本编译 `sm_90a`，随后执行自检和测试。
`git pull` 之后必须重新安装，旧 pip 包不会自动更新。

`vc` 是新 D128 核心，`vc_generic` 是保留的 0.5.0 通用核心，两者的包版本都是 0.6.0。
两行都包含重新聚类、量化、内存分配和输入检查；没有提前缓存这些工作。
脚本打印 `generic_ms / specialized_ms` 加速比，并对最后一次计时输出的抽样 query
做直接对拍。`exact sample match` 仅表示被检查的样本一致，不能代替完整精度验证。
如需同时对照 0.4.0，可再加 `--vc-fused-compare`。

核心耗时可用：

```bash
python3 scripts/profile_cuda.py --batch 1 --queries 37741 --tokens 37741 --heads 28 --dim 128 --vc-specialize-compare --output-dir results/h200_d128_profile
```

查看 `vc` 的 `fused_attention_d128` 完整行和尾行 kernel 的耗时之和，并与
`vc_generic` 的 `fused_attention_pipeline` 比较。端到端速度以无 profiler 的测速为准。

## 本地验证范围

本地 nvcc 12.4 / PyTorch 2.3.1 编译了 `sm_86` 和 `sm_90a`，安装 wheel 后运行测试。
RTX 3060 的诊断入口共享专用路径的异步搬运、Softmax 和在线更新代码，
但以标量运算替代 WGMMA；生产 API 不会选择这个诊断入口。

32 组专用/通用诊断核心对拍覆盖整块、尾块、多 KV 块和 scale 边界，逐元素一致。
另有独立 Python 量化参考、已知答案、非连续输入、不同 dtype、多个 head 和 CUDA stream 检查。
全套测试 282 passed / 128 skipped；93 项诊断测试通过 memcheck（0 errors），
45 项 D128 诊断测试通过 racecheck（0 errors / 0 warnings）。Windows 上常规 PyTorch
退出阶段会使 racecheck 卡住；最终检查在 GPU 同步和输出刷新后用 `os._exit` 退出，
获得了工具的完整汇总。检查范围及二进制验证结果见
[results/d128_local_validation.json](results/d128_local_validation.json)。

实际安装的原生二进制包含 E4M3 QGMMA 和 LDGSTS，无 FP16 HMMA 替换。
完整 Q 分块内核仍为 255 registers/thread、48 KiB shared、0 spill，
因此本轮不宣称提高寄存器所允许的驻留块数。CUDA 13 的资源分配可能不同。
原生 WGMMA 数值、与异步拷贝的交互及性能需要在 H200 上执行验证脚本和对比测速。
