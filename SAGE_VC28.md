# vc28：INT8 分数转换与 ExpCast 编码调度，加入 FA4 同场对照

包版本 `sageattention-local==2.2.0+vc28`，native kernel version 26。
本轮新增 `kernel_mode="vc28_stream_i2f"`，基线为 `vc22_both`，仅用于 `v_smooth=False`。
**Q/K 继续使用 INT8，P/V 继续使用 FP8；默认内核仍为支持四组开关的 vc15。**
vc28 是待 H200 验证的性能候选，没有已经达到 22 ms 的结论。

## 依据与改动

上轮用户回传中，vc22 完整 API 为 25.2288 ms，原版 Sage 为 26.3233 ms，
相对提速约 1.043×。vc22 核心仍需 24.0906 ms，仅改善预处理无法达到 22 ms 完整 API。
这只能把重点收敛到主核，尚不能证明某一条指令是硬件瓶颈。
原始摘录及计时范围见 [H200 INT8 回传摘要](results/h200_int8_main_reported_summary.json)。

vc22 在完整 K 块的整数最大值归约之后，把每线程 64 个 INT32 分数转成 FP32，
再执行分组的仿射 FMA、最近偶数舍入、打包、FP8 解码和 FP32 求和。
新候选把 I2F 和 FMA/RNE 打包放进同一个短 PTX 片段，让转换后的浮点临时值只服务当前四个编码。
随后仍按原来的两个 K fragment 分组执行 FP8 解码和求和。

目的是缩短临时值存活范围、改变指令依赖与调度，不是减少 QK 矩阵乘或改用 FP8 QK。
每个正常 tile 仍需相同数量的 I2F；静态指令总数因正常/回退两个互斥分支分别包含转换而增加。
显式分组也可能限制编译器调度，是否有净收益必须比较实际完整 attention。

保留的数值和执行规则：

- INT8 QK 的整数最大值归约、每 lane scale 及跨 lane 最大值处理。
- INT8、D≤128 的点积绝对值不超过 `2^21`，转为 FP32 是精确的。
- 相同 FP32 gain/bias、FMA、RNE 和饱和字节打包；相同的 FP8 P 字节。
- 相同 FP8 解码、FP32 分母求和顺序及两级 FP32 PV 累加。
- vc22 的跳过恒等历史缩放；原有 TMA/WGMMA 提交等待顺序。
- 尾块保留原路径；正常范围保护不成立时使用原来的非折叠编码路径。

实现主要在 `csrc/vc/streamed_probability.cuh`，通过独立 schedule 35 接入。
`expcast=False` 选择此模式时执行原 vc16 control；不将它重复列为新的加速行。

## FA4 比较方法

`scripts/profile_int8_h200.py` 默认加入当前安装的 `flash-attn-4`，使用
[`flash_attn.cute.flash_attn_func`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py)。
安装版本、模块路径和调用参数保存在 JSON 中。不会升级 FA4 或下载依赖。

所有行使用相同原始 Q/K/V、相同 softmax scale、dense noncausal MHA。
Sage/VC 使用原生 HND 连续布局，FA4 使用原生 NHD 连续布局；转换在计时外。
FA4 接收原始 BF16/FP16，不使用 VC 的 INT8 量化数据。
首次 JIT、预热和精度检查在计时外；各行完整 API 随机交错测量并保留每轮结果。

FA4 只列完整 API，fixed core 显示 `--`。`FA4/API > 1` 表示该行完整调用更快；
`Sage/API > 1` 表示比原版 Sage 快。另打印 vc22/vc28 的完整 API、固定核心比值和全输出实现误差。
不能把不同量化算法的 dense FP32 误差当作同实现回归误差。

## 验证范围

本地 SM86 能执行独立概率/history helper，不能运行 Hopper 完整 attention。
测试复用生产 helper，覆盖随机/稳定/递增最大值、稀疏行更新、尾块、非默认 stream、
正常/回退范围切换、极小 scale、极端整数分数及最长 296 个 tile 的递推。
P 字节、最大值、alpha、tile mass、分母和模拟历史输出与旧 helper 逐项检查。

CUDA 12.4 SM90a 交叉编译检查保存在 [codegen](results/sage_vc28_codegen_local.json)。
它报告编译资源和与历史内核的指令文本比较；这些静态数据不是速度结论。
本地 582 项测试通过；304 个 SM90a 实例编译通过，原有 300 个实例的指令文本一致。
新 D128 候选为 168 寄存器、D64 为 128，均无 stack/spill 或 WGMMA 串行化诊断。
源码包中 77 份来源记录的哈希校验通过，包含新增头文件；本地检查记录见
[验证摘要](results/sage_vc28_local_validation.json)。

远端验证增加新候选的 D64/D128、FP16/BF16、15 种 K 长度及 Q 尾块检查，
每个配置重复三次；另有均匀注意力已知答案和固定输入的最大值/极端 scale 场景。
主场景在相同量化数据上比较 vc28/vc22 的全部输出行，0.1% 相对 RMSE 门槛用于实现回归。
只有正确性通过，且多轮完整 API 明显超过同场 vc22，才考虑采用新候选。

## H200 执行

```bash
git pull --ff-only origin master
bash scripts/validate_int8_h200.sh
```

**这次必须重新编译**，不要沿用上一轮的 `--skip-build`。
脚本使用 `python3` 和已安装依赖，先验证原版/VC，再比较四组算法开关、vc22、vc28、Sage 和 FA4。
默认等价 NHD shape 为 `[1,37741,28,128]`、BF16、seed 0，结果写入 `results/h200_int8_main.json`。

安装好本版后，如需更多轮次和独立 CUPTI 核心数据：

```bash
python3 scripts/profile_int8_h200.py --trials 9 --trace-dir results/h200_vc28_traces --output results/h200_vc28_fa4.json
```

不要求 ncu/nvdisasm。FA4 导入失败会明确报错；仅在主动省略 FA4 时才使用 `--skip-flash4`。
