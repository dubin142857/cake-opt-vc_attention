# vc20：把当前 PV 与下一块 QK 连续提交

版本 `sageattention-local==2.2.0+vc20`，native kernel version 17。
以 vc16 为实验基线，新增三个计算调度候选；公开默认 `optimized` 保持 vc15。
H200 回传的目标 shape 实现一致性门槛已通过，但**三个新调度均慢于 vc16，没有实现提速。**

## H200 回传：连续提交的收益不足以补偿成本

| ExpCast 模式 | 核心 API ms | 完整 API ms | 完整 API 比 vc16 增加 | 重复 CUPTI 中位数 ms |
|---|---:|---:|---:|---:|
| vc16 | 25.4491 | 27.0665 | 基线 | 25.3759 |
| vc20_pv_qk | 26.9492 | 28.5444 | 5.46% | 26.8780 |
| vc20_serial | 26.8977 | 28.5124 | 5.34% | 26.8962 |
| vc20_qk_pv | 26.5005 | 28.0469 | 3.62% | 26.5799 |

同场原版 Sage 核心 API 25.0432 ms、完整 API 26.5092 ms、重复 CUPTI 24.9067 ms。
vc16 control 对应 24.9024 / 26.4065 / 24.7149 ms，三个新 control 候选也全部退步。
最佳新 ExpCast 候选完整 API 比 Sage 慢 5.80%；vc16 ExpCast 比 Sage 慢 2.10%。
以 vc16 的完整 API 降到 22 ms，仍需减少约 18.72% 的耗时。

远端实际寄存器从 vc16 的 168 增至 206–224；预测驻留块数从 3 块/SM 降至 2 块/SM。
所有行的 local bytes 都为 0、共享内存都为 41088 字节，没有显示 local memory 分配。
同时保存下一块 QK 和当前 PV 临时结果增加了寄存器需求，驻留减少是明确的资源代价；
但预测驻留不是实测利用率，当前数据不能把全部减速归因于它，循环与指令调度也发生了变化。

ExpCast 的 QK→PV 联合提交相对匹配的串行候选，核心 API 从 26.8977 降至 26.5005 ms，
只有约 1.5% 收益，仍无法超过 vc16。PV→QK 联合提交没有改善串行候选的核心 API。
9 轮 API 比较与 3 轮、每轮 20 次 launch 的 CUPTI 汇总方向一致，不能把这次退步当成
Python 调用开销或单次 trace 的偶然波动。

同算法 vc16 对照的全输出实现门槛为 0.1%；新 control 打印 RMSE 为 0.000737%，
新 ExpCast 打印 0.000000%。这些是实现一致性结果，不是相对 dense FP32 的量化误差，
舍入后的 0 也不能证明逐位一致。附件未含独立 shape 验证日志、原始 JSON、wall-time
试次数组或每次 GPU launch 明细。见 [回传汇总](results/h200_sage_vc20_reported_summary.json)。

开头的常规 API 表和两条单次 trace 仍是默认 vc15，不代表 vc20。公开默认继续保留 vc15，
实验基线保留 vc16；无需再次编译或重复当前测试。后续若继续尝试计算重叠，应先解决
同时存活的中间结果所需寄存器，而不是继续增大寄存器上限或只调整提交顺序。
vc18 强制降低寄存器上限的实验也没有收益，不能把简单调低上限当成解决办法。

## 实验背景

vc19 的 TMA 双缓冲在 H200 上没有净收益。那次同场 vc16 ExpCast 完整 API 为 26.8888 ms，
Sage 为 26.3825 ms。这里沿用单 K/V 缓冲，改动矩阵指令的提交和等待时机。

## 改动与成本

vc16 每块先完成 QK，再处理概率，随后提交 PV 并等待它完成。
当前 P 已经生成后，当前块的 PV 与下一块 QK 不需要互相的计算结果。
因此 vc20 单独先算出第 0 块 QK，此后的主循环如下：

```text
处理 QK(i) 得到 P(i)
提交 PV(i) 和 QK(i+1)
更新独立的历史输出缩放与分母（ExpCast）
统一等待两个矩阵结果完成
累加 PV(i)，携带 QK(i+1) 进入下一轮
```

最后一块只计算自己的概率和 PV，不再提交下一块 QK。
保留 vc16 的 INT8 QK、FP8 P/V、K 块 128、D128 的 N128 PV、FP32 分母和两级 PV 累加，
包括最后一块先反量化再处理概率的原有顺序。普通 softmax/control 也有相同调度候选。
要求 `v_smooth=False`；支持 FP16/BF16、D64/128、dense noncausal MHA，默认 CUDA stream。

| kernel_mode | 主循环提交顺序 | 主循环 WGMMA 等待次数 | 寄存器上限 |
|---|---|---:|---:|
| `vc16` | 当前 QK 完成后处理 P，再提交当前 PV | 2 | 168 |
| `vc20_pv_qk` | 当前 PV → 下一块 QK → 统一等待 | 1 | 224 |
| `vc20_serial` | 当前 PV → 等待 → 下一块 QK → 等待 | 2 | 224 |
| `vc20_qk_pv` | 下一块 QK → 当前 PV → 统一等待 | 1 | 224 |

三个新模式使用相同的首块提前计算结构、相同的单缓冲和寄存器上限。
串行模式帮助判断连续提交的效果，但实际寄存器分配和编译器调度仍可能不同。
**只比串行候选快不够，完整 API 还必须比 vc16 和同场 Sage 快。**

同时保存下一块 QK 和当前 PV 临时结果会增加寄存器需求。本地 D128 BF16 编译结果：

| 模式 | control 寄存器 | ExpCast 寄存器 | 栈 / spill stores / spill loads |
|---|---:|---:|---|
| `vc20_pv_qk` | 221 | 224 | 全部为 0 |
| `vc20_serial` | 206 | 218 | 全部为 0 |
| `vc20_qk_pv` | 221 | 223 | 全部为 0 |

动态共享内存仍是 40960 字节；没有 vc19 新增的双缓冲或每轮 CTA 复用同步。
寄存器增加可能减少驻留块数，必须以远端编译资源与完整 API 耗时判断收益。
连续提交并不意味着两个矩阵乘法会在 Tensor Core 上同时执行。

每次提交前保留 WGMMA register fence；读取两个结果及复用 K/V 缓冲前等待相关组完成。
首块、单块和尾块单独处理；K(i+2) 只在 QK(i+1) 完成且确实还有后续块时搬入。
异步寄存器与等待规则参见 [NVIDIA PTX WGMMA 说明](https://docs.nvidia.com/cuda/pdf/ptx_isa_8.5.pdf)。

## 验证范围

- 411 项本地测试通过，包括新模式分派、同输入比较、全输出误差/非有限值拦截和资源配置验证。
- CUDA 12.4 对 SM90a 交叉编译、原生链接和导入通过。共 236 个内核实例，24 个是新候选。
  旧 212 个实例的 SASS 指令文本全部一致；比较不包含机器编码/control words。
- 检查所有新实例的生成指令：联合候选在主循环的 PV 与下一块 QK 之间没有 WGMMA 等待，
  串行候选仍有该等待。ptxas 无警告。该检查只能证明生成的提交结构，不能证明运行加速。
- sdist 的版本及 62 个源码哈希通过核对。

本地只有 SM86；可运行的已有 helper 测试不能替代 Hopper 完整内核检查。
本地编译资源也不等于远端 CUDA 13 的资源分配。
记录见 [编译报告](results/sage_vc20_codegen_local.json) 和 [本地验证](results/sage_vc20_local_validation.json)。

## H200 离线运行

```bash
git pull --ff-only origin master
bash scripts/validate_compute_h200.sh
```

使用 `python3`，禁用 pip 下载、不进行 build isolation，不要求 nvdisasm/ncu。
先构建和验证，再测 BSHD `[1,37741,28,128]`、BF16、noncausal。
新增 compute 验证共 360 个 shape/dtype/算法/候选组合，每个重复 3 次，并有 24 个均匀概率
已知答案检查；覆盖 K 长度 1/127/128/129/255/256/257/383/384/385/511/512/513/1025/37741。
实现误差与同算法 vc16 比较，0.1% 是实现回归门槛，不是模型质量标准。

性能脚本在计时前检查目标大 shape 的全部输出行，随后进行 9 轮随机顺序比较：
核心 API 使用相同量化数据，完整 API 每次重新预处理；CUPTI 另跑 3 轮，每轮 20 次 launch。
开头的常规表仍是默认 vc15；新增结果位于后面的 `Compute pipeline candidates` 表。
`Sage/full > 1` 表示完整 API 比同场 Sage 快；`serial/joint > 1` 仅表示连续提交比串行候选快。

回传 `results/h200_vc20_compute.json`，或其中的候选耗时表、资源表及重复 CUPTI 汇总即可。
完整正确性报告在 `results/sage_vc_validation.json` 的 `compute_pipeline` 字段。
