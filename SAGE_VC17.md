# vc17：概率处理的四个独立候选

用户回传 vc16 在 H200、BSHD `[1,37741,28,128]`、BF16、noncausal 上：

| 实现 | 核心 API ms | 完整 API ms | 重复 CUPTI kernel ms |
|---|---:|---:|---:|
| 原版 Sage | 24.6435 | 26.1921 | 24.5735 |
| vc16 control | 24.5392 | 26.2127 | 24.4804 |
| vc15 ExpCast | 25.4844 | 27.0211 | 25.4706 |
| vc16 ExpCast | 25.2107 | 26.7478 | 25.1577 |

vc16 消除了 control 与原版主体的大部分差距，ExpCast 仍比同主体 control 多约 0.67 ms。
该差值是整个内核的净差，不能直接归因于某个转换操作。
22 ms 是目标，**vc17 尚无 H200 完整 attention 性能结果**。

## 改动范围

包版本 `2.2.0+vc17`，native kernel version 14；默认 `optimized` 保持 vc15。
新候选以 vc16 的晚 V 预取、历史更新与 PV 重叠主体为基线，只有概率处理不同：

| 模式 | 每批 fragment | 每行 FP32 求和链 | 解码 |
|---|---:|---:|---|
| `vc16` | 2 | 2 | FP8 → half2 → FP32 |
| `vc17_g1` | 1 | 2 | 同上 |
| `vc17_g4` | 4 | 2 | 同上 |
| `vc17_c4` | 2 | 4 | 同上 |
| `vc17_direct` | 2 | 2 | 直接由 E4M3 位表示构造 FP32 |

新候选仅接受 `expcast=True, v_smooth=False`。Q/K/V 量化、FP8 P 编码及舍入、
FP32 历史递推、两级 PV 累加均保留；没有降低累加精度，也没有共享概率副本。
正 E4M3 编码 0..120 对应 0..256，都是 2^-9 的整数倍。每线程每行 32 个数的和
可在 FP32 中精确表示，因此更换局部加法树可以保持分块和不变。
直接解码单独处理编码 0..7（零及次正规数），正常数使用指数偏移。

候选未假设提速：增大批次可能增加寄存器压力，直接解码会增加整数位操作。
默认路径和所有历史 helper 保留，远端比较决定是否采用候选。

## 远端一次执行

```bash
git pull --ff-only origin master
bash scripts/validate_expcast_h200.sh
```

脚本依次完成：

1. 编译并安装 `sageattention-local`，执行原版与 VC 完整内核验证。
2. `benchmark_probability.py`：检查并测量当前 fused-quad 编码、独立解码求和、完整概率 helper。
3. `profile_sage_vc.py --compare-probability-candidates`：同场比较原版 Sage、vc16 control、
   vc16 ExpCast 与四个候选，包含完整 API、固定输入核心及 3 轮 × 20 次 CUPTI kernel。
4. `inspect_probability_codegen.py`：导出已安装内核的完整 SASS，统计 20 个 dtype/dimension/mode 组合。

全程使用 `python3`；无需 ncu。编译器使用 nvcc，机器码导出使用 CUDA Toolkit 的 cuobjdump。
如果 CUPTI 不可用，可手动运行下一节命令并去掉 `--trace-dir`。

结果文件：

- `results/h200_vc17_probability.json`：分项微基准，**各组不能相加或作为 attention 的耗时占比**。
- `results/h200_vc17_attention.json`：真正的 attention 对照，重点看 `expcast_vc17_*` 行。
- `results/h200_vc17_codegen.json` 及 `.sass.txt`：编译指令及原始机器码，不是动态利用率。

完整对照的基线为 vc16，输出列为 `vc16/core`；最前面的普通报表仍是默认 vc15。
JSON 的 `vs_same_algorithm_baseline` 与每行 `reference` 对应；全部输出行检查及非有限值拦截
先于计时，0.1% 门槛仅为实现回归，不代表模型质量。full API 每次重做预处理。

## 已构建时分别运行

```bash
python3 scripts/benchmark_probability.py --trials 9 --output results/h200_vc17_probability.json
python3 scripts/profile_sage_vc.py --expcast-only --compare-probability-candidates --warmup 3 --repeats 10 --trials 9 --kernel-samples 20 --kernel-trials 3 --trace-dir results/h200_vc17_traces --output results/h200_vc17_attention.json
python3 scripts/inspect_probability_codegen.py --output results/h200_vc17_codegen.json
```

候选的 Python 使用方式：

```python
from sageattention_local import sageattn_vc
out = sageattn_vc(q, k, v, tensor_layout="NHD", expcast=True,
                  v_smooth=False, kernel_mode="vc17_direct")
```

## 本地证据与限制

- 367 项测试通过，另有 16 组历史 helper golden 精确一致。
- 11 项新增 CUDA helper 测试通过 Compute Sanitizer memcheck，报告 0 个错误。
- 实际 CUDA helper 检查全部 121 个有效编码、所有编码对、RNE 边界及相邻值、
  mask、极端值回退及 300 个连续 tile。P 字节、分块和、最大值、alpha 均与 vc16 一致；
  分块和另与独立解码后的 FP64 求和核对。
- CUDA 12.4 编译全部 132 个 sm90a attention 内核，其中 16 个为新候选。
  116 个历史内核的机器指令文本保持一致（比较时规范化新增的值为零的模板参数）。
  新候选未报告栈或寄存器溢出。完整内核的 Hopper 执行待远端验证。
- 本地 RTX 3060 SM86 运行了微基准以验证流程；其 FP8 转换实现与 Hopper 不同，
  **本地微基准快慢不能外推 H200**。微基准的寄存器/局部内存也不能代表完整 attention。
- 原版 Sage 内核未修改；来源与许可证清单新增 `probability_variants.cuh`。

证据见 [本地验证](results/sage_vc17_local_validation.json)、[编译资源](results/sage_vc17_codegen_local.json)、
[本地微基准](results/sage_vc17_probability_sm86.json)。
