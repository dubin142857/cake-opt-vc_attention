# vc21：用共享内存减少长期占用的 FP32 输出寄存器

包版本 `sageattention-local==2.2.0+vc21`，native kernel version 18。
新增 D128 实验入口，默认 `optimized` 仍为 vc15，实验基线仍为 vc16。
**本地交叉编译和 helper 对拍通过，完整 H200 attention 正确性与速度尚未验证。**

vc20 连续提交当前 PV 和下一块 QK 后，同时存活的累加器增多，寄存器达到 206–224，
预测驻留从 3 块降至 2 块。同场最好的新 ExpCast 完整 API 为 28.0469 ms，慢于 vc16 的
27.0665 ms 和 Sage 的 26.5092 ms。vc21 回到 vc16 的逐块计算顺序，减少长期保存的输出寄存器。

## 实现

每线程原有 64 个 FP32 历史输出值。新实现把其中 28 个放入线程独占的共享内存槽，
36 个继续留在寄存器中。每 CTA 的暂存区为 14 KiB，使用对齐的 `float4` 向量读写；
这些槽没有跨线程交换，也不由 TMA 访问，不增加 CTA 同步。

PV 仍使用两级 FP32 累加，分两个 N64 片段计算。第一片段更新寄存器中的历史；
第二片段完成后，从共享内存读出旧值，先做 FP32 历史缩放，再加当前 PV 临时结果并写回。
乘法和加法显式分开，避免编译器合并成 FMA。最终才汇集完整输出、归一化并转换输出 dtype。
Q/K/V 量化、ExpCast 编码、decoded-P 分母、K 块大小 128、尾部 mask 都沿用 vc16。

这项改动有成本：每个 K 块每 CTA 增加 14 KiB 共享内存读取和 14 KiB 写入，以及相应指令。
**寄存器减少是否足以补偿这些开销，必须由完整 API 实测判断。**

| kernel_mode | 寄存器上限 | 历史暂存 | 用途 |
|---|---:|---:|---|
| `vc16` | 168 | 无 | 原 N128 PV 基线 |
| `vc18_pv64_r128` | 128 | 无 | 旧 N64 PV、低上限对照 |
| `vc18_pv64_r168` | 168 | 无 | 旧 N64 PV、不压低上限的对照 |
| `vc21_history_r128` | 128 | 14 KiB FP32 | 减少实际寄存器需求后尝试增加驻留 |
| `vc21_history_r168` | 168 | 14 KiB FP32 | 相同共享存储、不同寄存器上限的对照 |

新入口支持 FP16/BF16、D128、dense noncausal MHA，要求 `v_smooth=False`，使用默认 stream。
Python 和原生层均拒绝 D64；其他历史入口的能力不变。

## 编译资源与本地验证

本地 CUDA 12.4、SM90a 编译的新入口资源如下，FP16/BF16 相同：

| 模式 | control 寄存器 | ExpCast 寄存器 | 栈 / spill stores / spill loads |
|---|---:|---:|---|
| `vc21_history_r128` | 128 | 128 | 全部为 0 |
| `vc21_history_r168` | 154 | 168 | 全部为 0 |

动态共享内存为 55296 字节，静态部分 128 字节。预算计入 CUDA 每块额外保留的共享内存，
避免暂存区本身阻止 4 块驻留。Hopper 的容量与预留规则见
[NVIDIA Hopper 调优指南](https://docs.nvidia.com/cuda/archive/13.0.3/hopper-tuning-guide/index.html#unified-shared-memory-l1-texture-cache)。
远端脚本查询实际编译资源和 CUDA occupancy 预测；本地资源不等于远端 CUDA 13 的分配，
预测驻留也不是实际利用率或加速证明。

- 428 项本地测试通过，覆盖 API 分派、非法配置、同输入对照、全输出误差拦截和源码溯源。
- 新增 8 项实际 GPU helper 对拍：1/2/5/296 轮，独立的行/通道数据，多 CTA，缩放系数含 0 和 1；
  使用生产共享内存 helper，与独立 FP32 递推逐元素一致。它不执行 Hopper WGMMA。
- 244 个 SM90a 内核实例编译成功，其中 8 个是新候选，ptxas 无警告；新候选没有 local load/store。
  旧 236 个实例的 SASS 指令文本一致，比较不包含机器编码/control words。
- 检查新候选的矩阵指令分组：QK 完成后依次计算两个 N64 PV，没有 vc20 的跨块 QK/PV 联合提交。
- 原生扩展链接与导入通过；sdist 的版本及 64 个源码哈希核对通过。

本地设备为 SM86，不能运行完整 Hopper attention。详情见
[编译记录](results/sage_vc21_codegen_local.json) 和 [本地验证](results/sage_vc21_local_validation.json)。

## H200 离线运行

```bash
git pull --ff-only origin master
bash scripts/validate_history_h200.sh
```

脚本使用 `python3`、现有依赖，不下载、不需要 ncu/nvdisasm。先构建并验证，再测性能。
新增完整内核检查覆盖 120 个 dtype/长度/算法/候选组合，每个重复 3 次；长度包括单块、
128 边界、多块、尾块及 37741。另有 8 个均匀 attention 已知答案，使用不同通道值及 decoded V
排除 padding，并检查最终输出恢复。记录在 `results/sage_vc_validation.json` 的 `shared_history`。

性能仍为 BSHD `[1,37741,28,128]`、BF16。每个候选先检查全部输出行，再进行 9 轮随机顺序计时。
核心 API 使用相同量化数据；完整 API 每次重新预处理；CUPTI 单独跑 3 轮、每轮 20 次 launch。
普通 softmax/control 和 ExpCast 分别与同算法 vc16 比较，0.1% 是实现回归门槛，不是模型质量标准。

看后面的 `Shared-history candidates` 表，开头普通表仍测默认 vc15。只超过低寄存器的旧对照
不算成功，完整 API 需要同时超过 vc16 和同场 Sage。22 ms 目标仍指完整 API。
回传 `results/h200_vc21_history.json` 或候选表、资源表和重复 CUPTI 汇总即可。
