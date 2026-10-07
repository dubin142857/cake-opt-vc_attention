# vc24：用 FP32 舍入生成 ExpCast 字节

包版本 `sageattention-local==2.2.0+vc24`，native kernel version 21。
实验基线为 `vc22_both`；新增 `kernel_mode="vc24_bitpack"`，仅用于关闭 V 平滑的路径。
默认仍为 vc15。H200 回传已确认 vc24 退步，不采用为新的性能基线。

## H200 回传结论

同场 `vc22_both` 的核心/完整 API 为 **23.9858 / 25.6664 ms**，
`vc24_bitpack` 为 **25.4895 / 26.9099 ms**；完整 API 慢 **4.85%**。
三轮 CUPTI 也全部慢于 vc22。两者都是 168 寄存器、local 0、共享内存 41088 字节，
不能将退步归因于已观察到的 spill 或驻留数变化。
替换转换指令增加了 FP32 算术和字节重排，这次没有得到净收益；计时本身不能定位具体硬件 stall。

全输出实现回归门槛通过，打印 RMSE 为 0.000000%，不代表模型质量或逐位相同。
原始 JSON、逐次 API 样本及独立形状验证日志未随回传提供。
结果归档于 [vc24 回传摘要](results/h200_sage_vc24_reported_summary.json)。
保留 `vc22_both` 作为下一轮基线，后续实验见 [vc25](SAGE_VC25.md)。以下为本版实现和复现说明。

## 改动与接受条件

vc22 对每个概率元素执行仿射 FMA，再用配对的浮点转整数指令生成字节。
vc24 在有范围保护的主循环中，改用 FP32 饱和 FMA、加法舍入和字节重排。
目的是测试将编码工作从转换指令转移到 FP32 算术能否获益；
它也增加了加法和字节重排，指令数量并未整体下降，不能预先认定更快。

整数 max、跳过恒等历史缩放来自 vc22。Q/K/V 量化、P 字节、FP8 解码、
FP32 分母、两级 FP32 PV 累加、TMA/WGMMA 顺序和尾块算法保持原来的规则。
小尺度或极端范围回退到 vc22；没有引入 vc23 的最大值状态或偏置缓存。

必须同时通过全输出实现回归检查，并在同场完整 API 中快于 `vc22_both`。
仅微基准、单次 trace 或比 vc16 快，不足以采用新候选。

## 为什么能得到相同字节

原来的正常范围路径计算 `u = RN(score * gain + bias)`，再将它按 RNE 舍入成无符号字节。
其中 gain/bias 已按 FP32 舍入；原有最大值范围保护保证有效编码不超过 120。

新路径先把这两个**已舍入的系数**各除以 128，计算：

```text
unit = clamp(RN(score * (gain / 128) + (bias / 128)), 0, 1)
bits = bitcast_uint32(RN(unit + 98304))
code = bits & 255
```

98304 等于 `1.5 * 2^16`，这个量级的 FP32 间隔恰好为 `1/128`，
其位表示低字节为零。上述加法的低位因此得到原 u 的最近偶数整数舍入。
缩放为二的幂，非下溢区间保留 FMA 舍入结果；接近下溢的概率也只会得到字节 0。
额外保护 `scale >= 2^-119`，防止 fast-math 将除以 128 后的 gain 冲成零。
主循环之外及不满足保护条件的输入继续使用旧编码。

实现显式使用 `fma.rn.sat.f32` 和独立 `__fadd_rn`，不能把这两次舍入合并。
PTX 的饱和 FMA 与舍入语义参见 [NVIDIA PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma)。
这是编码实现实验，不是更换 ExpCast 数学近似，也没有将分母改成 FP16。

## 本地验证

- 478 项测试通过，其中 29 个生产概率/历史 helper 场景在本地 SM86 GPU 上执行。
- 字节边界覆盖 0.5 到 119.5 每个半整数及左右各 8 个 FP32 相邻值，另有负无穷、零等特殊值。
  2,097,152 组随机仿射输入逐字节对照旧 CUDA 编码及独立整数舍入 oracle。
- 连续 1/2/7/296 个 tile、逐 lane scale、最大值变化、范围回退与小尺度边界均通过；
  测试中的 P 字节、最大值、alpha、块内和、分母及历史输出与基线零容差一致。
- CUDA 12.4 交叉编译 268 个 SM90a 内核实例；新增 4 个，历史 264 个 SASS 指令文本不变。
  新 D128 为 168 寄存器，D64 为 128，无 stack/spill/local load/store。
- 正常范围编码段出现 64 条饱和 FMA、64 条魔数加法，不含 F2IP；回退及尾块仍含旧转换。
  编译资源与指令检查不代表 H200 性能结果。
- native version 21 链接/导入及源码包检查通过。上游 36 个 CUDA kernel 文件保持原样。

记录：[编译检查](results/sage_vc24_codegen_local.json)、[本地验证](results/sage_vc24_local_validation.json)。
本地不能执行完整 Hopper attention；远端 CUDA 13 编译和完整输出验证仍需运行。

## H200 执行

```bash
git pull --ff-only origin master
bash scripts/validate_encoding_h200.sh
```

仅使用已安装依赖离线编译，脚本中的 Python 均为 `python3`，不需要 ncu/nvdisasm。
先验证再计时，目标仍为 BSHD `[1,37741,28,128]`、BF16、非因果。
新模式验证覆盖 120 个 dtype/维度/K 长度/算法组合，各重复三次；
另有 8 个均匀注意力已知答案和 32 个固定量化输入的最大值/尺度场景。
这些完整内核验证不计入本地 478 项通过结果。

看后面的 **Encoding candidates** 表；开头普通表仍是默认 vc15。
比较原版 Sage、vc16 control、vc16 ExpCast、vc22_both 和 vc24_bitpack。
核心 API 共用同一套量化数据；完整 API 每次重新预处理，随机顺序各测 9 轮。
全部输出行在计时外检查，0.1% 门槛仅用于实现回归，不代表模型质量。
另报告 3 轮 × 20 次 CUPTI 内核耗时，与 API 墙钟时间分开解释。

回传 `Encoding candidates`、资源表及 `Repeated CUPTI kernel durations` 即可。
结果保存在 `results/h200_vc24_encoding.json`；22 ms 是完整 attention API 的目标。
