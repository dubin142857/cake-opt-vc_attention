# vc14：合并 ExpCast 成对转换与打包

这一版针对 `expcast=True, v_smooth=False`。基于最佳旧版 vc12，保留其 affine
计算、极端范围回退、decoded-P FP32 分母、在线历史更新和两级 PV 累加。
没有采用 vc13 的共享副本方案，没有改变 V 平滑路径，也没有达到已验证的 H200 提速结论。

原来每两个数先分别执行 FP32→u8 的 RNE 转换，再用 byte permutation 打包。
现在将两个转换和打包写在一起：

```ptx
cvt.rni.s32.f32 lo, a;
cvt.rni.s32.f32 hi, b;
cvt.pack.sat.u8.s32.b32 packed, hi, lo, 0;
```

两步仍按原规则舍入与饱和：负数和 `-inf` 遮罩映射为零；正数按 RNE 后截到字节范围。
中间没有转成 FP16，也没有减小分母精度。
`cvt.pack` 的饱和及字节顺序见 [NVIDIA PTX 文档](https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt-pack)。
本地 ptxas 将这个序列合并为一条 `F2IP.U8.F32.NTZ`；原写法生成两条 F2IP 和一条 PRMT。

对每个线程的一块 64 个概率值，**转换和成对打包部分**从 96 条指令降到 32 条。
生成四字节矩阵片段的 16 条 PRMT、FP8 解码、FP32 求和、QK/PV 等工作仍然存在。
这个指令比例不能换算为 attention 加速比。

本地验证使用 Windows、CUDA 12.4、RTX 3060（执行 helper）及 `sm_90a` 交叉编译：

- 323 项测试通过，另有 16 组历史 helper golden 结果完全一致。
- 新转换覆盖字节舍入边界及相邻浮点数、成对组合、随机 FP32 位模式、次正规数、正负零、溢出和无穷值。
- 实际概率 helper 覆盖极端回退、不同 K scale、遮罩与 300 个连续分块；与 vc12 的 P 字节、分母和历史输出完全一致。
- 84 个 Hopper 内核编译通过，保留的 80 个历史内核机器指令文本完全一致。
- 4 个新增 D64/D128、FP16/BF16 内核：D128 每线程 168 寄存器，D64 为 128；无栈及寄存器溢出，未增加共享内存。
- 147 项 CUDA helper 测试通过 Compute Sanitizer memcheck，报告 0 个错误。
- H200 完整 WGMMA attention 尚未在本地运行；本地 helper 正确性不能替代远端验证或模型质量评估。

本地 SM86 helper 小基准的 vc12/vc14 中位数为 3.75056/4.38637 ms（0.855x），
原始轮次有明显变化，没有稳定的提速证据。该测试没有 QK/PV，且 SM86 的 FP8 解码使用
不同实现，不能据此预测 H200。完整原始数据保存在
[本地小基准](results/sage_vc14_local_probability_micro.json)；
[验证摘要](results/sage_vc14_local_validation.json) 和
[Hopper 静态指令](results/sage_vc14_codegen_local.json) 单独保留。

远端更新、构建和验证：

```bash
git pull --ff-only origin master
bash scripts/build_sage_local.sh
python3 scripts/profile_sage_vc.py --expcast-only --compare-kernels --kernel-baseline vc12 --warmup 3 --repeats 10 --trials 9 --output results/h200_vc14_paired.json
```

默认形状为 BSHD `[1,37741,28,128]`、BF16。对照使用相同量化 Q/K/V、检查全部输出行，
计时顺序随机，保留每轮原始数据。0.1% 输出相对 RMSE 是实现回归门槛，不能解释为模型质量通过。
除 `vc12/optimized` 外，还要看 `sage_vc_control`：只有新 ExpCast 比 control 快，
才说明新增 ExpCast 已带来内核收益；完整 API 还需要与 `sage_local` 对比。

可以用当前工具链复核机器指令，无需 ncu：

```bash
python3 scripts/inspect_expcast_codegen.py --output results/h200_vc14_codegen.json
```

该脚本的数字统计整个函数里的主循环、尾块及回退分支，并非一次执行的动态指令数。
CUDA 13 的生成代码可能与本地 CUDA 12.4 不同，最终以同场 H200 实测为准。
`python3 scripts/benchmark_expcast.py` 也保留 vc12/vc14 helper 对照，但不测 QK/PV 或预处理。

Python 入口保持不变：

```python
from sageattention_local import sageattn_vc
out = sageattn_vc(q, k, v, tensor_layout="NHD", expcast=True, v_smooth=False)
old = sageattn_vc(q, k, v, tensor_layout="NHD", expcast=True, v_smooth=False, kernel_mode="vc12")
```

两次公开调用都重新做预处理；脚本里的 fixed core 对照则共用一次准备的量化输入。
