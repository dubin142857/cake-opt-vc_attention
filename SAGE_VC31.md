# vc31：统一 ExpCast / V-Smooth，并独立测量 GPU kernel

包版本 `sageattention-local==2.2.0+vc31`，native kernel version 29。
H200 主线保持 **INT8 QK + FP8 E4M3 PV、两级 FP32 累加**。
vc28 是已测的性能基线；vc31 是需要远端验证的新候选，不预先宣称提速。

## 这次修正什么

此前 `int8_vc` 使用 vc15，vc28 只允许关闭 V-Smooth，不能把这两行解释为
“最新版内核上开关 V-Smooth”。vc31 在同一内核模板中接通 vc28 的整数最大值、
分组 I2F/FMA/打包、FP32 概率归约与 V 块均值恢复，支持四组功能开关。

无平滑时直接分派到原 vc28 内核，ExpCast 输出要求与 vc28 全部元素逐位相同。
有平滑时，归一化分母仍累加每个 lane 的局部概率质量；均值恢复使用四个 lane
归约后的整行质量。两者不能混用，否则均值项会少四倍，或分母多四倍。
均值仍按 `mu / V_scale` 存储，输出末尾统一乘通道 scale。

D128 平滑分支的 PV 拆为两个 N64 fragment，以降低临时累加器压力，
最后一个 fragment 完成后提前预取下一块 V。每个输出元素的四个 K32
矩阵乘、两级 FP32 累加仍保留。尾块沿用浮点掩码路径。
这不是重新量化 Q/K，也没有删除分母或降低累加精度。

## 远端执行

需要重新编译；所有命令使用 `python3`，不需要 ncu、nvdisasm 或下载新依赖。

```bash
git pull --ff-only origin master
bash scripts/validate_int8_h200.sh
```

脚本先做原有验证，再验证 vc31 的 FP16/BF16、D64/128、单 key、整块、尾块、
多 batch/head、均值通道和非默认 stream，最后运行完整 API 对比。
默认等价 NHD `[1,37741,28,128]`，BF16，非因果 MHA。
测试失败即停止，不进入性能测量。原版 Sage 和 FA4 继续作为对照。

随后运行单独的 CUPTI 测量（下面是一整行命令）：

```bash
python3 scripts/profile_int8_h200.py --paper-kernels --kernel-trials 3 --kernel-warmup-seconds 20 --kernel-seconds 10 --trials 5 --trace-dir results/h200_vc31_traces --output results/h200_vc31_compare.json
```

默认 CUPTI 比较 Sage、vc28、vc31 四组和 FA4。每个后端每轮预热 20 秒、
采集 10 秒；仅这部分就约 10.5 分钟，另有正确性、API 计时和导出开销。
日志会打印当前后端。可先用 `--kernel-trials 1` 做通路检查。

## 如何看输出

- 第一张表是原有同步 API wall time。full API 每次重新预处理，fixed core
  复用量化张量但仍含分配与调度；FA4 不虚构 prepared-core 时间。
- 第二张表来自 CUPTI GPU activity，预处理和首次 JIT 在计时外。
  每次调用同步隔离，只统计该调用内 GPU activity 的持续时间，排除 CPU 调度和流间空隙。
- `attention` 统计识别出的 attention kernel；`all kernels` 包括辅助 kernel；
  `all GPU work` 还包含 memcpy/memset。FA4 多 kernel 时不会漏掉辅助计算。
  第二张表的加速比使用 `all kernels`，不能与第一张表交叉相除。
- 每个窗口保存原始逐调用记录、kernel 名称、资源和 Chrome trace；汇总使用
  各窗口中位数的中位数。CUPTI 不可用或活动归属不完整时明确失败，保留已有 API/精度结果。
- `vc31_expcast` 应与 vc28 数值完全相同，可用于检查计时波动。
  `vc31_both` 对照旧 `int8_vc`，`vc31_smooth` 对照旧 `int8_v_smooth`。
  ExpCast 的系数折叠和加法顺序相对 vc15 有差异，不能要求平滑分支逐位相同；
  全输出使用 0.1% 旧实现回归门槛，另用 1% 解码参考门槛。这些不是模型质量标准。

## 真实输入

可以在模型内于 RoPE 后保存一次 Q/K/V，形状为 NHD 或 HND，三者 dtype 相同：

```python
torch.save({"q": q.detach().cpu(), "k": k.detach().cpu(), "v": v.detach().cpu()}, "qkv.pt")
```

```bash
python3 scripts/profile_int8_h200.py --qkv-file qkv.pt --qkv-layout NHD --paper-kernels --output results/h200_vc31_captured.json
```

脚本从文件取得实际 shape/dtype；搬运和连续布局转换在计时外，不会再次施加 RoPE。
文件只在本机读取，不上传。只支持非因果 MHA、D64/128。

## 与论文的边界

[论文附录 A](https://arxiv.org/html/2609.15810v1#A1) 使用真实 post-RoPE 输入、
20 秒预热、10 秒 CUPTI 窗口统计孤立 kernel。此处对齐这些测量要素，但默认仍是
随机输入且长度不同，聚合细节也明确记录，不能宣称复现论文 1.16×。
本版仍每次新做聚类/gather，没有实现模型去噪 step 的布局复用，也未融合 RoPE/Hadamard。
这些属于后续模型接入和预处理工作，不能通过修改计时范围掩盖其成本。

本地只有 SM86，可执行 helper/预处理测试及 SM90a 交叉编译；完整 Hopper 内核
正确性与性能须由上述远端命令验证。新的 V-Smooth 路径保持显式选择，默认 API 不变：

```python
from sageattention_local import sageattn_vc
out = sageattn_vc(q, k, v, tensor_layout="NHD", kernel_mode="vc31_unified",
                 expcast=True, v_smooth=True)
```

## 本地验证记录

- 622 项测试通过，包括真实 SM86 GPU 上的均值恢复、两段 PV 通道索引与 stream 测试。
- CUDA 12.4 编译/链接 320 个 SM90a 实例；对新增模板参数和辅助指针类型规范化后，
  原有 312 个实例的机器指令文本保持一致。
- 新双开分支 D128 使用 168 寄存器、D64 使用 128，无 stack/spill。
  只开平滑 D128 仍有 8 字节 stack、4 字节 spill store/load；这是候选的已知限制。
  没有 ptxas 警告或 WGMMA 串行化诊断；这些都不能代替性能实测。
- 源码包 78 项文件哈希验证通过。
- 本地 Windows PyTorch profiler 只有 CPU activity，无法执行真实 CUPTI 计时通路；
  新脚本对此明确报错，归属逻辑通过合成 trace 测试。远端此前可采集 CUPTI，仍须复核新通路。

详见 [编译记录](results/sage_vc31_codegen_local.json) 与
[本地验证记录](results/sage_vc31_local_validation.json)。

## 2026-10-09：修正常量通道验证

原验证脚本错误地要求四组开关都保持常量 V 原值。普通 Sage 路径关闭 ExpCast
和 V-Smooth 时，PV 分子使用 FP8(P)，分母使用未量化 P；即使 V 为常量，
输出仍可能偏移。它应与解码量化参考比较，不能直接要求等于稠密 softmax 的答案。
第一次修正保留了随机常量通道的逐元素参考检查；随后 BF16 的普通 Sage 分支
仍触发此附加检查，见下面的补充修正。原有 0.1% 旧实现门槛、1% 解码参考门槛
及无平滑的逐位对照均不变。

此次只修改验证脚本，没有修改 CUDA 内核或包版本，**不用重新编译**。
32 项 CPU 回归检查覆盖独立两种 logit 的解析答案、FP16/BF16、整块/尾块、
四组功能开关，以及错误分母/错误输出仍会被拒绝；不替代 H200 整体验证。
每个用例现在先打印 shape 和功能开关，失败时可直接定位。

```bash
git pull --ff-only origin master
python3 scripts/validate_sage_vc31.py
```

通过后再执行上面的 `profile_int8_h200.py --paper-kernels ...` 命令。

### 同日补充：随机误差检查与解析答案检查分开

远端 BF16、D64、Q65/K128、两个开关都关闭的用例，在通过 vc28 逐位一致和
全输出 1% 参考检查之后，触发新增的 `rtol=.004, atol=.003` 逐元素检查。
这说明该用例没有观察到 vc31 相对 vc28 的回归；并不证明原生结果与 FP32
解码参考逐元素相同。旧报错覆盖了 PyTorch 的数值详情，目前还不能确定具体
偏差来自 FP8 概率舍入、矩阵累加或最终 BF16 舍入。不能把推测当成已定位的原因。

当前验证分为两组，均必须通过：

- **64 组随机完整流程**：保留 0.1% 旧实现误差门槛、无平滑逐位一致、1% 全输出
  解码参考门槛及非默认 stream 检查。随机常量通道的附加逐元素检查改为诊断项，
  不再将这一未经硬件误差验证的阈值作为独立通过条件。
- **64 组固定量化输入**：直接构造 INT8 QK、精确可表示的 FP8 V/FP16 均值和
  两种 logit，用独立标量公式计算答案。检查全部输出通道，保留 `.004/.003`
  逐元素门槛；开启 ExpCast 或均值恢复时另检查常量保持。覆盖 FP16/BF16、
  D64/128、单 key、整块及多块尾部，padding 填入明显非零值以检查掩码。
  普通 Sage 的错误分母会产生约 3% 偏差，不能靠与同样错误的旧实现一致而通过。

每个随机用例记录最差元素坐标、实际值、参考值、旧版值和常量通道误差。
原逐元素阈值的结果仍保存在 `pointwise_reference_within_legacy_tolerance` 中。
失败也会保存 JSON（`status=failed`、已完成用例、当前用例、原始错误），退出码保持失败。
固定答案错误仍输出 PyTorch 的最大绝对/相对误差详情。

本次仍只改验证脚本和测试，**不用重编译**。104 项 CPU 回归测试通过，包含
BF16 相邻输出值触发旧附加检查的例子、全部解析输入的独立参考核对、原有门槛
仍拒绝错误，以及失败报告保存。连同相关参考实现、验证工具和 INT8 profiling
测试共 236 项通过。它们不代替 H200 的原生内核验证；按上述两条
`git pull` / `python3 scripts/validate_sage_vc31.py` 命令重跑即可。
