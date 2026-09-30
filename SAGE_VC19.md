# vc19：跨 K 块的 TMA 搬运与计算重叠

包版本 `2.2.0+vc19`，native kernel version 16。本轮的目标是缩短内核等待 K/V 数据的时间。
vc18 回传中，ExpCast 最好的基线仍是 vc16：核心 API 25.2242 ms、完整 API 26.6750 ms；
同场 Sage 完整 API 26.1382 ms。更低的寄存器上限和 N64 PV 拆分都没有超过基线。
这些是历史实测，不是 vc19 的结果。**vc19 尚未在 H200 运行，不能宣称已加速或达到 22 ms。**

## 本次修改

沿用 vc16 的每线程 INT8 QK、FP8 V、概率计算、FP32 分母和两级 PV 累加。
保持 168 寄存器上限、K 块大小 128、D128 的 N128 PV；仅改变 K/V 的共享内存缓冲和搬运时机。
每个候选都有普通 softmax/control 和 ExpCast 两种模式，均要求 `v_smooth=False`。
支持 D64/128、FP16/BF16、dense noncausal MHA；比较脚本固定 D128 以对应资源查询。

| kernel_mode | K 缓冲数 | V 缓冲数 | 下一块何时搬运 | D128 动态共享内存 |
|---|---:|---:|---|---:|
| `vc16` | 1 | 1 | K 在 QK 后，V 在 PV 后 | 40 KiB |
| `vc19_k` | 2 | 1 | K 在当前 QK 前，V 保持原时机 | 56 KiB |
| `vc19_v` | 1 | 2 | V 在当前 QK 前，K 保持原时机 | 56 KiB |
| `vc19_kv` | 2 | 2 | K/V 都在当前 QK 前 | 72 KiB |
| `vc19_kv_late` | 2 | 2 | K 在 QK 后，V 在 PV 后 | 72 KiB |

`vc19_kv_late` 与 `vc19_kv` 使用相同缓冲数和 CTA 复用同步点，
用于检验提前搬运是否补偿了双缓冲的资源、指针和同步成本。比较仍是完整编译后内核，
编译调度可能不同，不能把差值直接当成纯 TMA 等待时间。

旧单缓冲已能把下一块 K 搬运与当前 softmax/PV 重叠；双缓冲是在此基础上进一步提前。
新候选不代表已确认瓶颈在 TMA。更多共享内存、地址计算和同步也可能抵消收益，
因此需要同时比较 control、ExpCast、vc16 与原版 Sage 的完整 API。

实现放在独立 `csrc/vc/tma_pipeline.cuh`，保留原内核作对照。
初始化 barrier 后发布到 async proxy；每个槽位独立推进 ready phase：
双缓冲时 tile 0/1 等待 phase 0，tile 2/3 等待 phase 1，tile 4/5 再等 phase 0。
每轮复用前，所有 warp 完成上轮 WGMMA 等待后经过 CTA 同步，才允许 thread 0 覆盖已释放槽位。
最后一块不发起下一块搬运，保留原来的尾部 mask 和反量化顺序。
同步依据见 [NVIDIA TMA 编程说明](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#using-tma-to-transfer-multi-dimensional-arrays)。

## 已完成的本地检查

- 402 项测试通过：Python 分派、同输入比较、全部输出行与非有限值门槛、资源配置校验、
  源码溯源，以及可在本地 SM86 执行的已有 helper 测试。
- CUDA 12.4 编译 SM90a 成功：212 个内核实例，其中 32 个新双缓冲实例。
  原有 180 个实例的 SASS 指令文本全部一致；不包含机器编码/control words 比较。
- 新候选的 D128 BF16 control/ExpCast 均为 168 寄存器，栈与 spill stores/loads 都为 0，
  ptxas 无警告。这是本地编译资源，远端 CUDA 13 的结果由脚本重新查询。
- 原生扩展链接和导入通过；sdist 中 61 个源码哈希与版本通过核对。

本地设备是 SM86，不能执行新 Hopper attention 内核；上述检查不代替 H200 正确性和性能验证。
原默认 `optimized` 仍走 vc15，新候选通过 `kernel_mode` 显式选择。
详情见 [编译报告](results/sage_vc19_codegen_local.json) 和 [本地验证记录](results/sage_vc19_local_validation.json)。

## 远端一次运行

在现有 CUDA/PyTorch/Triton 环境中同步代码后：

```bash
git pull --ff-only origin master
bash scripts/validate_pipeline_h200.sh
```

使用 `python3`，禁用 pip 索引且使用 `--no-build-isolation --no-deps`，不下载依赖，
也不要求 nvdisasm/ncu。构建后先跑正确性验证，成功后才测性能。

新增正确性检查覆盖单块、整块、尾块和两槽多轮复用：K 长度
1/127/128/129/255/256/257/383/384/385/511/512/513/1025/37741，Q 长度 65，
两种 dtype、D64/128、两种概率算法、四个候选，每组重复 3 次；
另有均匀 attention 对 decoded V 均值的独立检查。实现门槛为相对同算法 vc16 的 0.1%，
不是量化精度或模型质量门槛。目标大 shape 在性能比较前还会检查全部输出行。

性能 shape 为 BSHD `[1,37741,28,128]`、BF16，9 轮随机顺序比较：

- `core API`：同一组量化数据，每次新分配输出，包含原生分派；
- `full API`：每次重新预处理，不复用量化数据；
- CUPTI：每候选 3 轮、每轮 20 次 launch，单独报告纯 GPU kernel 时间；
- 实际编译资源：寄存器、local memory、共享内存和预测驻留块数。

开头的常规表仍测默认 vc15；新候选在后面的 `control_vc19_*` / `expcast_vc19_*` 表中。
`Sage/full > 1` 表示该候选完整 API 比同场 Sage 快；`late/early > 1` 表示提前搬运更快。
先看 `full API`，再用 core、CUPTI 和资源解释差异。22 ms 指完整 API，不能用 core 数字替代。

请回传 `results/h200_vc19_pipeline.json` 或候选耗时表、资源表和 CUPTI 汇总。
构建后的正确性记录在 `results/sage_vc_validation.json` 的 `pipeline` 字段。
