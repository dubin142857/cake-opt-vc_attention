# vc23：复用在线最大值状态

版本 `sageattention-local==2.2.0+vc23`，native kernel version 20。
本轮以 H200 已测到完整 API **25.6150 ms** 的 `vc22_both` 为实验基线；
该数字属于 vc22，不是 vc23 的结果。vc23 尚未在 H200 上实测，22 ms 目标仍未达到。

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
这些是待 H200 执行的检查，不计入本地 465 项通过结果。

看后面的 **Max-state candidates** 表，开头普通分阶段表仍测安装包默认 vc15。
候选表包含 `control_vc16`、`expcast_vc16`、`expcast_vc22_both`、
`expcast_vc23_maxstate`、`expcast_vc23_cached_bias` 和原版 `sage_cpp`。

ExpCast 的精度和加速比基线为 **vc22_both**，控制行的基线为 vc16。
完整 API 每次重新预处理，核心 API 共用完全相同的量化数据，各随机顺序测 9 轮。
全部输出行的回归门槛仍为 0.1%，不代表对 dense FP32 的误差或模型质量。
另测三轮、每轮 20 次 CUPTI kernel duration，和 API 墙钟时间分开报告。
结果文件为 `results/h200_vc23_maxstate.json`。

本轮接受条件是同场完整 API 快于 vc22_both；仅比 vc16 快不够。
目前没有新性能承诺，vc22 的实测收益记录仍保留。
