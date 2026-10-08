# vc29：两行 ExpCast 编码交错调度

包版本 `sageattention-local==2.2.0+vc29`，native kernel version 27。
新增 `kernel_mode="vc29_interleave_rows"`，以已测的 vc28 为对照。
这是待 H200 验证的候选，尚无提速结论。Q/K 使用 INT8，P/V 使用 FP8，关闭 V 平滑。

## 依据与改动

vc28 复测完整 API 为 25.1828 ms，同场 vc22 为 25.5764 ms，原版 Sage 为
26.3211 ms，FA4 为 30.9672 ms。单独的五轮 CUPTI 中，vc28 在四轮快于 vc22，
五个轮次中位数的中位数分别为 23.7673 / 24.0296 ms。
这支持保留 vc28 的小幅收益，但不能据此定位某个硬件 stall，也未达到 22 ms。
详见 [回传摘要](results/h200_vc28_reported_summary.json)。

vc28 的短 PTX 编码片段一次处理一行的四个分数。
vc29 将两行、共八个分数放进同一片段，交错执行两行的独立转换链：

1. 转换两行的高位概率对，执行 FP32 FMA、RNE、配对字节打包。
2. 转换两行的低位概率对，执行相同步骤，将高位对追加进四字节结果。
3. 按原来的次序解码 FP8，累加 FP32 分母。

目的是为转换和算术调度提供更多独立指令，不是减少量化位数或 attention 计算量。
INT8 整数最大值、每 lane scale、FMA 系数、舍入、P 字节映射、分母加法顺序、
历史缩放、两级 FP32 PV 累加、TMA/WGMMA 顺序、尾块和极端范围回退均沿用 vc28。
寄存器分配与机器指令调度可能改变；是否有净收益必须测完整 attention。

实现位于 `csrc/vc/streamed_probability.cuh`，通过独立 schedule 36 接入。
vc28 的单行 helper 保留，旧默认仍是支持 V 平滑的 vc15；关闭 ExpCast 时复用 vc16 control。
没有将未经测量的新候选设为默认。

## 验证与复现

本地 SM86 执行生产概率 helper：随机/稳定/递增最大值、非默认 stream、尾块、
极小 scale、范围回退及 296 个 tile 递推。新增测试覆盖两行不同偏置、半整数舍入边界
及其左右 FP32 相邻值、262144 组随机整数输入，逐字节比较 vc28/vc29，并用独立
FP64 计算再舍入到 FP32 的 oracle 检查编码。完整 Hopper attention 需远端验证。

本地结果：**594 项测试通过**；CUDA 12.4 交叉编译 308 个 SM90a 实例，
原有 304 个实例的指令文本保持一致。新候选 D128 为 168 寄存器、D64 为 128，
无 stack/spill，也没有 WGMMA 串行化诊断。源码包 77 份来源记录哈希验证通过。
这些是正确性与编译检查，不能据此推断 H200 提速。记录见
[本地验证](results/sage_vc29_local_validation.json) 和 [编译检查](results/sage_vc29_codegen_local.json)。

```bash
git pull --ff-only origin master
bash scripts/validate_int8_h200.sh
```

**本次需重新编译，不能使用 `--skip-build`。** 脚本使用 `python3`，沿用已安装依赖，
不需要 ncu/nvdisasm。默认等价 NHD shape `[1,37741,28,128]`、BF16。
先做正确性验证，再同场比较四组 vc15 开关、vc22、vc28、vc29、原版 Sage 和 FA4。

重点看 `int8_expcast_vc29` 和末尾 `vc28/vc29`：完整 API 和固定核心比值大于 1
才表示新候选更快。全输出 0.1% 相对 RMSE 门槛仅用于实现回归，不是模型质量门槛。
FA4 仍接收同值的原始 BF16/FP16 数据，只列完整 API；布局转换和首次 JIT 在计时外。

安装后可增加轮次并单独采集 CUPTI：

```bash
python3 scripts/profile_int8_h200.py --warmup 10 --repeats 10 --trials 9 --kernel-trials 5 --trace-dir results/h200_vc29_traces --output results/h200_vc29_compare.json
```

比同场 vc28 更快且正确性通过才采用；比旧 vc15 快不构成本轮成功。
