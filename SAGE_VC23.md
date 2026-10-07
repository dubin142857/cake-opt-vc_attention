# vc23：复用在线最大值状态

版本 `sageattention-local==2.2.0+vc23`，native kernel version 20。
本轮以 `vc22_both` 为实验基线。H200 回传未显示 vc23 的完整 API 收益：
同场 vc22 为 **25.6810 ms**，两个 vc23 候选为 **25.8416 / 25.8799 ms**。
继续保留 vc22 作为后续实验基线，默认仍为 vc15；22 ms 目标尚未达到。

## H200 回传结果

BSHD `[1,37741,28,128]`、BF16、非因果，API 随机顺序测 9 轮；
CUPTI 单独测 3 轮、每轮 20 次。shape 来自约定脚本，粘贴片段未包含环境头。

| 候选 | 核心 API ms | 完整 API ms | 重复 CUPTI ms |
|---|---:|---:|---:|
| 原版 `sage_cpp` | 25.0185 | 26.6324 | 24.8652 |
| `expcast_vc16` | 25.5495 | 27.2146 | 25.4331 |
| `vc22_both` | **24.0736** | **25.6810** | 24.1135 |
| `vc23_maxstate` | 24.1232 | 25.8416 | 23.9872 |
| `vc23_cached_bias` | 24.2670 | 25.8799 | 24.1250 |

maxstate 完整 API 比 vc22 多 0.1606 ms（0.625%），缓存偏置多 0.1989 ms（0.775%）。
maxstate 的 CUPTI 汇总稍快，但逐轮差值有正有负，核心/完整 API 中位数也未改善。
原始 API trial 数据未提供，不能据此判断微小差异的统计显著性；本轮没有达到接受条件。
各候选仍为 168 寄存器、local 0、共享内存 41088 字节、预测驻留 3 blocks/SM，
没有资源证据支持“新增 spill 或驻留下降”的解释。

一种待验证的解释是：省下的两个标量指数及偏置 FMA 对关键路径贡献有限，
提前加入的 vote、谓词和状态依赖抵消了收益。耗时与资源表不能直接证明这一机制。
后续保留 vc22，优先检查每元素概率编码、解码求和及其与主循环的依赖；
本轮不继续推广标量缓存方案，也未修改 attention 内核或默认路由。

vc22 的收益再次复现：完整 API 比同场原版 Sage 提速 **1.037×**，
达到 22 ms 还需减少 **3.6810 ms（14.33%）**。这里的完整 API 是带重新预处理的
attention 调用，不是模型端到端；当前候选关闭 V 平滑。

全输出同算法及 fresh/fixed 比较的 0.1% 实现门槛通过；打印 RMSE 均为 0.000000%，
不据此宣称逐位相同、dense FP32 零误差或模型质量达标。
附件未包含独立多 shape 验证日志，不能确认这些验证的执行数量。
开头普通分阶段表及单次 trace 对应默认 vc15，不是本表候选。
详见 [回传摘要](results/h200_sage_vc23_reported_summary.json)。

## 本轮改动

vc22 在每个完整 K 块中求出新的行最大值、计算 `alpha=exp(old_max-new_max)`，
编码 P 后再判断 alpha 是否全为 1，以决定是否缩放历史输出。
最大值不变时，它仍计算两次 `exp(0)`，以及两次 ExpCast 行偏置 FMA。

| `kernel_mode` | 更新判定 | 行偏置 |
|---|---|---|
| `vc22_both` | 概率编码后检查 alpha 是否为 1 | 每块重算 |
| `vc23_maxstate` | 求 max 后产生 warp 更新标志，编码后复用 | 每块重算 |
| `vc23_cached_bias` | 同上 | 仅 warp 内有最大值更新时重算 |

没有行更新时，alpha 直接取精确值 1，省去指数计算；同一标志用于 PV 提交后的
历史累加，省去后一次 alpha 比较与 vote。任一行更新时，整个 warp 执行原 FP32 缩放。
若最大值发生微小变化但 exp 舍入为 1，新候选可能多执行一次恒等缩放，数值仍相同。

ExpCast 偏置为 `fma(-max, 11.541560173034668, 119.6500015258789)`。
缓存版本在初始最大值处初始化，并在最大值更新时用相同 FMA 刷新；当前 tile 的
量化 scale/gain 仍每次读取和计算，不缓存。每个 warp 只用自己的更新标志，
不引入跨 warp 通信或新的共享内存。等待中的 P/WGMMA 累加器不被读取。

保留 Q/K/V 量化、整数 max、FP8 P 字节编码、解码 P 的 FP32 分母和两级 PV 累加。
最后一块保留 vc22 的浮点归约、scale 顺序、掩码和 alpha 检查，避免改变尾块行为。
极端范围仍进入原非折叠编码路径；进入/离开正常范围时缓存状态仍正确更新。
两个候选都只用于 `v_smooth=False`；关闭 ExpCast 时复用原 vc16 控制内核。
默认 `optimized` 保持 vc15。

## 本地检查

- 465 项测试通过，包含已有 CPU/CUDA 检查和扩展后的 21 个概率状态 helper 场景。
- helper 检查覆盖 1/2/7/296 个 tile，随机、稳定、递增及稀疏更新，逐 lane scale、
  极端范围回退、非默认 stream、最大值不变但 gain 变化，以及跨越正常范围边界。
- 新候选在这些测试中的 P 字节、最大值、alpha、tile mass、分母及历史输出与旧 helper 一致；
  独立 PyTorch 最大值、FP8 解码、FP32 累加检查也通过。
- CUDA 12.4 交叉编译 264 个 SM90a 实例；新增 8 个，历史 256 个 SASS 指令文本不变。
  比较不含编码/控制字。
- 新 D128 FP16/BF16 均为 168 寄存器，D64 为 128；全部无 stack/spill 和 local load/store。
  D128 动态共享内存仍为 40960 字节、静态 128 字节；QK/PV 的提交等待顺序不变。
- 反汇编确认：maxstate 候选的两条 `MUFU.EX2` 由更新谓词控制；缓存候选使用跳转，
  最大值不变时绕过两条 `MUFU.EX2` 和两条偏置 FMA。
  谓词与跳转的调度成本不同；这是控制流检查，不是 attention 加速证据。

本地是 SM86，只执行 helper；完整 Hopper attention 未在本地运行。
CUDA 13 的实际资源和耗时以远端结果为准。缓存寄存器、分支以及更新频率均可能影响收益。
检查记录：[codegen](results/sage_vc23_codegen_local.json)、
[本地验证](results/sage_vc23_local_validation.json)。

## 远端执行

```bash
git pull --ff-only origin master
bash scripts/validate_maxstate_h200.sh
```

使用现有依赖离线编译，不需要 ncu/nvdisasm。脚本先验证再测量，shape 仍为
BSHD `[1,37741,28,128]`、BF16、非因果。
新增完整内核验证包含 240 个 dtype/D64/D128/K 长度/算法/模式组合，每个重复三次，
16 个均匀注意力已知答案及 24 个固定量化数据的最大值变化场景。
这些不计入本地 465 项通过结果；本次粘贴片段也未包含它们的独立验证日志。

看后面的 **Max-state candidates** 表，开头普通分阶段表仍测安装包默认 vc15。
候选表包含 `control_vc16`、`expcast_vc16`、`expcast_vc22_both`、
`expcast_vc23_maxstate`、`expcast_vc23_cached_bias` 和原版 `sage_cpp`。

ExpCast 的精度和加速比基线为 **vc22_both**，控制行的基线为 vc16。
完整 API 每次重新预处理，核心 API 共用完全相同的量化数据，各随机顺序测 9 轮。
全部输出行的回归门槛仍为 0.1%，不代表对 dense FP32 的误差或模型质量。
另测三轮、每轮 20 次 CUPTI kernel duration，和 API 墙钟时间分开报告。
结果文件为 `results/h200_vc23_maxstate.json`。

本轮接受条件是同场完整 API 快于 vc22_both；仅比 vc16 快不够。
本次回传未达到该条件，保留 vc22 的实测收益记录和实验基线。
本次结果归档只修改文档和摘要，已完成测量的机器无需重新编译或重跑。
