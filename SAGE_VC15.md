# vc15：ExpCast 四字节拼接与原版 Sage 核心对照

上一轮 H200 的 vc14 ExpCast fixed core 为 25.8497 ms，control 为 25.4165 ms，
仍慢 0.4332 ms。完整 API 分别为 ExpCast 27.3954 ms、control 27.0958 ms、
原版 Sage 26.1198 ms。下一步要同时解决 ExpCast 的剩余成本和 control 与原版的差距。
这些是 vc14 回传数据，**vc15 尚无 H200 性能结果**。

本轮 `sageattention-local==2.2.0+vc15`、native kernel version 12。
默认 `optimized` 对 ExpCast-only 使用新内核，`kernel_mode="vc14"` 保留上一版。

## 四字节拼接

vc14 每四个概率执行两条成对转换指令，再执行一条 PRMT 将两个字节对拼成矩阵乘法所需的四字节。
vc15 利用 `cvt.pack.sat.u8.s32.b32` 的第三个输入携带已经生成的高字节对：

```text
vc14: pair(a,b) + pair(c,d) + PRMT → [a,b,c,d]
vc15: pair(c,d) → append_pair(a,b,previous) → [a,b,c,d]
```

本地 CUDA 12.4 将新写法编译成两条 F2IP，省掉最后的 PRMT。
每线程每块 64 个概率的转换/拼接由 48 条指令变为 32 条；这是该局部的指令数，
不能换算成 attention 加速比。两条转换之间增加了数据依赖，实际调度收益需要 H200 测量。

P 字节舍入、极端值回退、FP8 解码、FP32 分母、在线历史更新及两级 PV 累加不变。
QK/PV、概率解码和归约仍然有成本。control 和 V 平滑路径仍选择原有编译内核。

## 定位 control 与原版 Sage 的差距

检查源码后，两者的未平滑预处理配置相同：K 去均值、per-thread INT8 QK、
per-channel FP8 V、相同的分块参数。control 使用本项目修改过的 attention 主体；
原版 Sage 使用上游主体。二者代码调度不同，不能把完整 API 的差距直接归因于 Python。

新增 `--compare-sage-core`，一次准备量化数据，五条路径共享全部 Q/K/V 字节和尺度：

| 输出行 | 调用路径 |
|---|---|
| `sage_cpp` | 原版 Sage pybind/C++ 核心 |
| `sage_registered` | 原版 Sage 的 Torch custom-op 注册入口 |
| `vc_control` | 本项目的普通 Softmax 核心 |
| `vc14_expcast` | 上一版 ExpCast 核心 |
| `vc_optimized_expcast` | 新版 ExpCast 核心，JSON 记录具体版本 |

五条路径每次都新分配输出，采用随机顺序、多轮同步 wall time；均排除预处理，
保留原生分配和下发开销。它们不是纯 GPU kernel 计时，也不是完整 API 加速比。
对原版与 control 的检查覆盖全部输出行；新旧 ExpCast 也单独检查全部输出行。
ExpCast 与原版的误差只报告，不使用同算法的 0.1% 实现回归门槛。

配合 `--trace-dir`，额外采集上述五个核心各一次 CUPTI trace，JSON 保留寄存器、
共享内存等元数据。用它区分 GPU 主体与接口差异；单次 trace 不能替代多轮正常计时。
不需要安装 ncu。

## 本地验证与限制

- 331 项测试通过，另有 16 组历史 helper golden 数据完全一致。
- 实际 CUDA helper 验证四字节顺序、舍入边界及相邻值、随机 FP32 位模式、
  次正规数、无穷值、遮罩、极端值回退和 300 个连续分块；与 vc14 保持精确一致。
- `sm_90a` 的 88 个内核编译通过；保留的 84 个历史内核机器指令文本完全一致。
- 新内核 D128 每线程 168 寄存器，D64 为 128；栈/溢出均为零，共享内存不增加。
- 149 项 CUDA helper 测试通过 Compute Sanitizer memcheck，报告 0 个错误。
- 源码包已生成并校验版本及全部 58 份受记录源码的哈希，上游 36 份内核主体未改动。
- Hopper 函数全部分支的静态 PRMT 数由 64 降到 0，F2IP 仍为 128。
  该统计包含主循环、尾块和回退分支，不是动态执行次数。

本地执行 GPU 为 RTX 3060 SM86，编译器为 CUDA 12.4。H200 的完整 WGMMA attention
需远端运行；CUDA 13 的生成代码也需核对。helper 正确性与实现回归检查不代表模型质量通过。
证据见 [本地验证摘要](results/sage_vc15_local_validation.json) 和
[Hopper 静态指令](results/sage_vc15_codegen_local.json)。

## 远端构建与对照

```bash
git pull --ff-only origin master
bash scripts/build_sage_local.sh
python3 scripts/profile_sage_vc.py --expcast-only --compare-kernels --kernel-baseline vc14 --compare-sage-core --warmup 3 --repeats 10 --trials 9 --trace-dir results/h200_vc15_traces --output results/h200_vc15_pack4.json
```

默认 BSHD `[1,37741,28,128]`、BF16。构建脚本包含完整 attention 正确性验证。
重点看新版 ExpCast 能否快于 `vc_control`，以及完整 API 能否快于 `sage_local`；
同时看同一份输入的 `sage_cpp` 与 `vc_control` 是否存在核心差距。
若 CUPTI 不可用，可去掉 `--trace-dir`；正常计时结果会先写入文件。

可选：确认当前 CUDA 13 编译器是否也消除了最后的 PRMT：

```bash
python3 scripts/inspect_expcast_codegen.py --output results/h200_vc15_codegen.json
```

Python 使用方式不变：

```python
from sageattention_local import sageattn_vc
output = sageattn_vc(q, k, v, tensor_layout="NHD", expcast=True, v_smooth=False)
previous = sageattn_vc(q, k, v, tensor_layout="NHD", expcast=True, v_smooth=False, kernel_mode="vc14")
```
