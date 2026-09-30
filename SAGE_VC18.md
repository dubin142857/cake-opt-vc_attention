# vc18：寄存器上限与 PV 临时累加器

H200、BSHD `[1,37741,28,128]`、BF16、noncausal 的最近回传中，vc16 control 核心为
24.7514 ms，ExpCast 为 25.1619 ms，原版 Sage 为 24.7261 ms。vc16 ExpCast 完整 API 为
26.9367 ms，仍未达到 22 ms。vc17 direct 解码在独立测试中慢 2.56 倍，完整核心慢约 31%，
不继续使用它作为优化方向。分项测试不能相加或视为完整 attention 耗时占比。

本轮以 vc16 为基线，比较编译器寄存器上限，并减少 PV 阶段同时存活的临时累加器。
包版本 `2.2.0+vc18`，native kernel version 15。尚未在 H200 执行新候选或测量速度。

## 实现

| kernel_mode | 寄存器上限 | D128 PV 输出片段 | 用途 |
|---|---:|---|---|
| `vc16` | 168 | 一个 N128 | 同算法基线 |
| `vc18_r128` | 128 | 一个 N128 | 更低上限的资源/耗时对照 |
| `vc18_r144` | 144 | 一个 N128 | 寄存器压力对照 |
| `vc18_r192` | 192 | 一个 N128 | 放宽上限的对照 |
| `vc18_pv64_r128` | 128 | 两个 N64 | 缩小临时累加器后降低上限 |
| `vc18_pv64_r144` | 144 | 两个 N64 | 缩小临时累加器后降低上限 |
| `vc18_pv64_r168` | 168 | 两个 N64 | 单独检验分段的代价/收益 |

每个模式均支持 `expcast=False/True`，要求 `v_smooth=False`。API 支持 D64/128、FP16/BF16；
D64 没有拆分 PV，资源比较脚本专门测 D128。候选显式选择，默认 `optimized` 保持 vc15。

PV 分段将 128 个输出通道分成两个 64 通道片段，临时 FP32 累加器大小减半。
每个输出元素仍使用四个 K32 FP8 WGMMA，结果再由 FP32 加入历史输出。历史缩放与分母递推
仅在第一个片段更新一次；V 缓冲区在两个片段都完成读取后才允许覆盖。
Q/K/V 量化、ExpCast P 编码与解码求和、K 块大小 128 都保持原有定义。

`__maxnreg__` 影响编译器分配及调度，不保证更低的上限更快。
增加驻留线程块可能以额外 local memory 访问为代价。
参考 [NVIDIA 寄存器上限说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#maximum-number-of-registers-per-thread)。

## 本地验证与限制

- 391 项测试通过，包括 Python 分派、同输入比较、全部输出行误差/非有限值拦截、资源元数据校验，
  以及可在本地 SM86 运行的已有量化/分组 helper 测试。
- CUDA 12.4 对 SM90a 编译通过，共 180 个内核实例。原有 132 个实例在规范化新增模板参数后，
  SASS 指令文本全部一致；比较不包括机器编码/control words。
- 以下为本地编译器的 D128 BF16 ExpCast 静态栈/溢出数据，不是远端 CUDA 13.0 的结果或动态流量。

| 模式 | 实际寄存器 | 栈字节 | spill stores 字节 | spill loads 字节 |
|---|---:|---:|---:|---:|
| `vc18_r128` | 128 | 208 | 520 | 520 |
| `vc18_pv64_r128` | 128 | 120 | 180 | 184 |
| `vc18_r144` | 144 | 112 | 224 | 224 |
| `vc18_pv64_r144` | 144 | 16 | 16 | 16 |
| `vc18_pv64_r168` | 168 | 0 | 0 | 0 |

control 的 144 上限下，N128 栈为 72 字节，分段后为 0。
这些结果说明分段减轻了编译资源压力，但额外 WGMMA 提交/等待仍可能抵消收益。
编译记录见 [本地资源报告](results/sage_vc18_codegen_local.json)。本地没有 Hopper，
新完整内核的运行正确性和性能由下面的远端脚本验证；不以本地小测试代替 H200 结论。

## 远端运行

代码同步后，在已有 CUDA/PyTorch/Triton 和构建依赖的机器上执行：

```bash
bash scripts/validate_register_h200.sh
```

脚本对本地源码使用 `PIP_NO_INDEX=1` 构建，不访问包索引，不安装 nvdisasm/ncu。
先运行包含新候选的精度验证，再固定 shape `[1,37741,28,128]`，同时比较 control、ExpCast 和 Sage。
全输出行与相同算法的 vc16 比较，fresh full API 也与固定输入输出比较；0.1% 是实现回归门槛，
不是模型质量门槛。计时顺序随机，9 轮，每轮 10 次；CUPTI 独立执行 3 轮，每轮 20 次。

报告为 `results/h200_vc18_registers.json`，trace 在 `results/h200_vc18_traces/`。
最前面的普通 stage 表仍测默认 vc15；重点看后面的候选表，以及对应的资源表：

- `core API` / `full API`：固定量化输入与每次重新预处理的耗时。
- `cap` / `regs`：请求上限与远端实际编译寄存器数。
- `PV N`：128 或 64，便于按相同上限比较分段的净效果。
- `local B` / `blocks/SM`：本地存储分配和预测驻留块数，不是硬件流量或利用率计数。
- `Sage/full`：原版 Sage 完整 API 耗时除以该候选完整 API 耗时，大于 1 才更快。

如果 CUPTI 不可用，已保存的 API 计时有效。只需去掉 `--trace-dir` 重跑分析脚本，
不要求重编译。仅比较寄存器资源不需要 SASS 导出。

显式使用一个候选：

```python
from sageattention_local import sageattn_vc
out = sageattn_vc(q, k, v, tensor_layout="NHD", expcast=True,
                 v_smooth=False, kernel_mode="vc18_pv64_r144")
```

此示例选择待测候选，不表示已确认它最快。最终保留方案以 H200 完整 API 和精度对照为准。
