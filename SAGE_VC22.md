# vc22：整数最大值归约与跳过恒等缩放

版本 `sageattention-local==2.2.0+vc22`，native kernel version 19。
这是基于 vc16 的 ExpCast 无 V 平滑实验；H200 回传已测到组合候选收益，默认仍为 vc15。
目标 shape 为 BSHD `[1,37741,28,128]`，BF16、非因果注意力。

## H200 回传：完整 API 25.615 ms

用户回传了 9 轮 API 比较的中位数，以及 3 轮、每轮 20 次的 CUPTI kernel 数据。
原始 JSON、API 原始试次和逐次 kernel 样本未提供；shape/dtype 沿用约定脚本，摘录无环境头。

| 候选 | core API ms | full API ms | CUPTI ms |
|---|---:|---:|---:|
| `control_vc16` | 24.8623 | 26.5004 | 24.7247 |
| `expcast_vc16` | 25.5483 | 27.1694 | 25.3727 |
| `expcast_vc22_intmax` | 24.4487 | 26.0642 | 24.5493 |
| `expcast_vc22_skip` | 25.0444 | 26.5627 | 24.8647 |
| **`expcast_vc22_both`** | **24.0625** | **25.6150** | **24.1128** |
| 原版 `sage_cpp` | 24.9848 | 26.5558 | 24.8441 |

组合候选相对 vc16 完整 API 提速 **1.061×**，耗时降低 **5.72%**；
相对同场原版 Sage 提速 **1.037×**，耗时降低 **3.54%**、节省 **0.9408 ms**。
CUPTI 三轮中组合候选每轮都快于 vc16 和 Sage，汇总相对 Sage 提速 1.030×。
三种候选的寄存器、local、共享内存和预测驻留数完全相同：168 / 0 / 41088 字节 / 3 块。
本轮收益来自指令路径及其调度变化；预测驻留数没有提高。

整数 max 单独节省 1.1052 ms 完整 API，跳过恒等缩放单独节省 0.6067 ms，
一起节省 1.5544 ms，不能把两项收益直接相加。后续实验基线改为 `vc22_both`。
离 **22 ms 完整 API** 还差 **3.615 ms**，需要在当前基础上再降低约 **14.11%** 耗时。

目标 shape 的全输出实现比较通过，各候选打印的相对 vc16 RMSE 为 `0.000000%`。
打印值经过舍入，不能单凭这一列宣布逐位一致；原 JSON 的 `exact_match` 才能确认。
这不是相对 dense FP32 的误差，也不是下游模型质量结论。
此次摘录未包含单独的构建/shape 验证日志，不能据此宣称那些检查的具体执行数量。
记录见 [vc22 H200 汇总](results/h200_sage_vc22_reported_summary.json)。

使用已测候选时显式选择；输入按本次 BSHD shape 使用 `NHD`：

```python
from sageattention_local import sageattn_vc

out = sageattn_vc(q, k, v, tensor_layout="NHD", expcast=True,
                 v_smooth=False, kernel_mode="vc22_both")
```

默认 `optimized` 仍为 vc15。以上收益限于本次工作负载；V 平滑和完整 VC 论文方案未在本轮验证。

## 改动及数学依据

| `kernel_mode` | 完整 K 块的最大值 | 历史输出缩放 |
|---|---|---|
| `vc16` | FP32 max | 每块执行 |
| `vc22_intmax` | 三数整数 max | 每块执行 |
| `vc22_skip` | FP32 max | warp 内所有 alpha 等于 1 时跳过 |
| `vc22_both` | 三数整数 max | 同上 |

QK 的 WGMMA 累加器已经是 INT32。INT8、D≤128 时，点积绝对值至多为
`128*128*128 = 2^21`，转成 FP32 没有舍入误差。因此可以先对整数求 max，
再转换并乘当前 lane 的正数 scale，得到与 vc16 相同的局部最大值。
K scale 在不同 lane 上可以不同，所以缩放仍在跨 lane shuffle **之前**进行。
使用 NVIDIA 的 [`__vimax3_s32`](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-math-api/group__CUDA__MATH__INTRINSIC__SIMD.html)
缩短归约树。只修改完整 K 块；最后一块仍采用 vc16 的浮点归约、原缩放顺序和 `-inf` 掩码。
所有分数依然需要转成 FP32 编码 P，这项优化减少的是 max 指令，不是移除全部转换。

在线累加满足 `d = d*alpha + tile_mass`、`O = O*alpha + tile_PV`。
最大值不变时 `alpha=1`；若一个 warp 的所有行都满足这个条件，直接保留历史 O，
只给 d 加 tile_mass。任一行需要缩放时，该 warp 执行原来的逐项 FP32 乘法。
跳过的是恒等乘法，未改为 FP16 累加，也未把分母替换成近似和。
这段工作仍在 PV 提交后进行，只访问历史 O/d，不读取尚未完成的 WGMMA P 操作数或 PV 结果。

两项改动分别比较，并提供组合候选。`expcast=False` 选择这些模式时映射到原 vc16 控制内核；
测量表只显示一个控制行，避免把相同内核重复当作不同优化。
V 平滑仍不支持这些模式。Q/K/V 量化数据、ExpCast 编码、解码 P 的 FP32 分母、
两级 PV 累加、K128 tile、D128 的 N128 PV 和 168 寄存器上限均保留。

## 已完成的本地验证

- CUDA 12.4 交叉编译 256 个 SM90a 实例，12 个为新候选。
- 与 vc21 比较，原有 244 个实例的 SASS 指令文本一致；比较忽略新模板默认参数、编码及控制字。
- 新 D128 FP16/BF16 候选均为每线程 168 寄存器，D64 为 128；均无 stack/spill 或 local load/store。
- 整数 max 候选静态指令中有 32 条 `VIMNMX`，FP32 `FMNMX` 从 152 条减到 82 条；
  这是编译后的指令计数，不是动态耗时或加速比。
- D128 动态共享内存仍为 40960 字节，静态 128 字节；主循环及尾块仍为 QK、PV 分别提交等待。
- 本地测试 454 项通过。新增 17 个 GPU helper 场景覆盖随机、稳定、递增最大值、极端范围回退、
  单个 lane 组更新、非默认 stream，以及最长 296 次在线累加。
  FP8 字节、最大值、alpha、tile mass、分母和历史输出与原 helper 一致；
  另用 PyTorch 独立检查最大值、FP8 解码和 FP32 累加。

本地卡为 SM86，只执行 helper；DPX 在该卡上使用兼容实现。
Hopper 指令已交叉编译检查；完整 attention 的 H200 回传见本文开头，与本地验证分开记录。
CUDA 13 的寄存器分配可能不同，远端脚本会重新报告资源。
记录见 [codegen](results/sage_vc22_codegen_local.json) 和
[本地检查](results/sage_vc22_local_validation.json)。

## 远端执行

```bash
git pull --ff-only origin master
bash scripts/validate_work_h200.sh
```

脚本使用已安装依赖离线编译，不需要 ncu/nvdisasm；先验证再计时。
新增完整内核检查覆盖 D64/D128、FP16/BF16、15 种 K 长度、部分 Q/K 尾块，
360 个算法/模式/shape 组合各重复 3 次、24 个均匀注意力已知答案，
再检查 36 个固定量化数据的最大值变化场景（包括极端范围回退）。
这些是待远端执行的检查，不属于本地已通过的 454 项。

结果保存到 `results/h200_vc22_work.json`，观察 **Work-reduction candidates** 表：

- `expcast_vc16`：同场实验基线。
- `expcast_vc22_intmax`、`expcast_vc22_skip`、`expcast_vc22_both`：三项候选。
- `control_vc16`、`sage_cpp`：普通 softmax 控制和原版 Sage。

开头常规 `sage_vc_expcast` 表依然测默认 vc15，不代表新候选。
同一组预量化数据用于核心对照，完整 API 每次重新预处理；二者各随机顺序测 9 轮。
比较全部输出行，0.1% 仍只是实现回归门槛。随后 CUPTI 分开测 3 轮、每轮 20 次 kernel。
不把相对原版 Sage 的算法误差当作实现误差，也不把 helper 或静态指令数量当成 attention 提速。

同场完整 API 已测到快于 vc16 和 Sage，22 ms 目标尚未达到。
跳过恒等缩放的收益取决于输入中最大值更新频率，分支和 vote 本身也有成本。
此轮不继续 vc21 的共享历史路线；上轮资源改善但完整 API 退步的结论仍成立。
