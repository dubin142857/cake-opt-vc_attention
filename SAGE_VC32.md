# vc32：完整 PV 与批量均值加载

包版本 `sageattention-local==2.2.0+vc32`，native kernel version **30**。
H200 主线保持 **INT8 QK + FP8 E4M3 PV、两级 FP32 累加**。
默认 API 仍使用原来的 `optimized`；vc31、vc28 与原版 Sage 都保留作为对照。

## 为什么改这里

用户回传的 vc31 结果，场景为 NHD `[1,37741,28,128]`、BF16：

| 实现 | Full API / ms | CUPTI kernel / ms |
|---|---:|---:|
| Sage | 26.3152 | 24.8521 |
| FlashAttention 4 | 30.6018 | 30.7961 |
| vc31 仅 ExpCast | 25.0893 | 23.6144 |
| vc31 仅 V-Smooth | 33.7321 | 30.5425 |
| vc31 双开 | 31.8464 | 28.4951 |

双开相对旧 vc15 的 Full API 从 37.6582 ms 降至 31.8464 ms，但仍慢于 Sage。
平滑分支的额外成本有相当一部分发生在 attention 内核里。因此本轮保持相同
分组和量化张量，对比 PV 分段和均值恢复代码；没有把聚类成本移出 Full API。

## 两个候选

| kernel_mode | D128 PV | 均值读取/恢复 |
|---|---|---|
| `vc31_unified` | 两段 N64 | 每个 half2 读取后做四个输出更新 |
| `vc32_full_pv` | 一段 N128 | 同 vc31 的逐组读取和 FP32 FMA/add |
| `vc32_packed_mean` | 一段 N128 | 批量读取 packed FP16，逐组解码和恢复 |

完整 N128 PV 减少分段 MMA 和等待。每个输出元素仍计算四个 K32 PV 累加，
不把跨 tile 历史放回 FP8 矩阵指令的累加器。

批量读取保留每对均值的 FP16 位模式，使用时才转 FP32。每个输出元素的公式
仍为 `history + fma(row_mass, mean, tile_pv)`，分母继续使用 lane 的局部 mass；
均值恢复使用四个 lane 的完整行 mass。归约只做一次，尾块与主循环都覆盖。

源码把批量加载放在 PV wait 前，但 **目标 D128 + ExpCast 的 CUDA 12.4 反汇编
显示实际加载在 wait 后**；其他分支有部分提前。因此该候选叫 packed mean，
不宣称目标分支已实现 PV 与均值加载重叠；目标是对比批量
读取/寻址和逐组读取的成本。远端 CUDA 13 的最终调度仍可能不同。

ExpCast 双开 D128 的两个新候选保持 **168 寄存器**、无 stack/spill。
只开平滑 D128 在 168 上限下出现 spill，所以这两个候选使用 **192 上限**，
交叉编译时无 spill；它可能降低驻留 CTA 数，性能必须实测。
因此 `vc31_smooth` 对新 smooth 行并非仅改变 PV 宽度，也改变了寄存器上限；
新 full 与 packed 两行彼此使用相同上限，可用于对比均值读取。
D64 无需拆分 PV；关闭平滑时，两个新模式均直接调用原 vc28。

## 远端执行

本次修改 CUDA，需要重新编译。使用本机现有依赖，不需要 ncu、nvdisasm 或联网下载。

```bash
git pull --ff-only origin master
bash scripts/validate_int8_h200.sh
```

脚本先完成原有检查，再分别验证 vc31、vc32 full、vc32 packed。每个模式包含
64 组随机完整流程和 64 组独立解析答案，覆盖 FP16/BF16、D64/128、四组开关、
整块/尾块、多 batch/head 及非默认 stream。任何一项失败都停止，不进入测速。
随机比较保留 0.1% 旧实现 / 1% 解码参考门槛，无平滑须逐位一致。
解析用例检查全部输出通道及 padding 掩码，门槛保持 rtol=.004/atol=.003。
vc32 使用 **同一份 prepared payload 上的 vc31** 作为旧实现，不重新量化后比较。

通过后脚本打印完整 API 表，新增四行：
`vc32_full_smooth`、`vc32_full_both`、`vc32_packed_smooth`、`vc32_packed_both`。
保留 Sage/FA4，以及旧 vc31 行。完整调用仍包含每次新做的预处理。

接着单独测 GPU kernel（下面是一整行）：

```bash
python3 scripts/profile_int8_h200.py --paper-kernels --kernel-backends sage_cpp vc31_expcast vc31_both vc32_full_both vc32_packed_both flash4 --kernel-trials 3 --kernel-warmup-seconds 20 --kernel-seconds 10 --trials 5 --trace-dir results/h200_vc32_traces --output results/h200_vc32_compare.json
```

该命令也会打印 API/精度表。六个 CUPTI 后端的预热/采集总计约 9 分钟，另加
检查、API 计时和导出。重点比较 vc31_both → vc32_full_both → vc32_packed_both，
以及各自与 Sage/FA4 的差距。API 和 CUPTI 各自使用自己的基线，不交叉相除。
默认 CUPTI 后端列表沿用 vc31，测 vc32 请保留上面的 `--kernel-backends`。

## 本地证据与边界

本地 GPU 为 SM86，不能执行 Hopper attention；可执行均值 helper、预处理测试，
以及 CUDA 12.4 SM90a 交叉编译/链接。

- 三种均值恢复 helper 均用真实 CUDA 执行，对照独立 FP64 外积、单次 FMA 舍入
  和独立 FP32 history add，覆盖多个 head/tile、D64/128、零 PV 和非默认 stream。
- 相关回归测试共 **751 项通过**，包含上述 24 组均值 GPU 检查、原有预处理/helper、
  新模式调度、独立解析答案、脚本失败中止及 CUPTI 归属逻辑。
- 332 个 SM90a 实例编译成功；规范化新增的 false 模板参数后，旧版 320 个实例
  的机器指令文本全部一致。
- 新候选无 stack/spill，无 ptxas warning 或 WGMMA 串行化诊断。
- 编译记录包含静态 MMA、等待点、实际均值加载位置；这些不是运行耗时。

详见 [编译记录](results/sage_vc32_codegen_local.json) 与
[本地验证记录](results/sage_vc32_local_validation.json)。H200 的完整正确性和提速
仍需上面的远端实测；不把交叉编译通过称为性能达标。
