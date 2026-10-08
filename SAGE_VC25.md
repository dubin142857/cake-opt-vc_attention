# vc25：把上一块概率分母求和移到下一块 QK 期间

包版本 `sageattention-local==2.2.0+vc25`，native kernel version 22。
实验基线为 `vc22_both`；默认仍为 vc15，两个新模式均显式选择、关闭 V 平滑。
22 ms 仍是完整 attention API 的目标。

2026-10-08 用户回传：同场 `vc22_both` 完整 API 25.5605 ms，
`vc25_mass_serial` 25.8718 ms，`vc25_mass_overlap` 25.8467 ms。
两种新候选均未改善完整 API，继续以 vc22 为实验基线。
CUPTI 的细小差异不足以说明稳定收益；详细记录见
[回传摘要](results/h200_sage_vc25_reported_summary.json)。下一轮实验见 [vc26](SAGE_VC26.md)。

## 为什么改这里

vc24 改变编码指令后，完整 API 从同场 vc22 的 25.6664 ms 增至 26.9099 ms。
本轮回到 vc22 的配对转换编码、INT32 max 和跳过恒等历史缩放。
要测试的是能否把解码 P、计算 FP32 行和的时间放进下一块 QK 的等待区间。
QK 和 PV 本身的矩阵规模不变，这一调度是否有收益必须实测。

新增两个同数据流的源码调度候选：

| kernel_mode | 上一块 P 的解码求和位置 |
|---|---|
| `vc25_mass_serial` | 下一块 QK 提交前 |
| `vc25_mass_overlap` | 下一块 QK 提交后、等待完成前 |

每线程将完整块的 16 个 packed P 寄存器及 2 个 alpha 留到下一次循环。
在 C++/PTX 顺序中，原块的 PV 已等待完成才读取 P；下一块 QK 读共享内存中的 Q/K，不使用这些 P。
不增加共享概率副本，不同时保留两块 QK 累加器，仍用单缓冲 K/V 和完整 N128 PV。

分母保持 `d = RN(RN(d * alpha) + mass)`，只是执行时刻推迟一块。
历史输出仍在当前 PV 期间乘当前 alpha；它不依赖 d。
最终尾块先补上一块 d，保留原有掩码和 P 编码规则，在最后一次 PV 完成及历史累加后
补齐最后一块 d。初始跨块状态设为 P=0、alpha=1，第一次补算不改变 d；
这样不需要在异步矩阵组内分支跳过第一块，也不会读未初始化状态。
FP8 P 字节、解码后的 FP32 求和顺序、
两级 FP32 PV 累加均保留，没有将分母改为 FP16。

寄存器参与异步 WGMMA 时必须遵循 [NVIDIA PTX 的顺序要求](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence)。
源码写在 commit/wait 之间不足以证明最终调度，需要检查生成指令。
本地 CUDA 12.4 将两种源码候选生成相同 SASS 指令文本，均将解码排入下一块 QK。
因此它们不是本地编译下的独立串行/重叠消融实验；不能把两者微小计时差解释为调度收益。
保留两个入口供远端 CUDA 13 编译观察，主要采用条件始终是相对 vc22 的完整 API 改善。

## 本地检查

- 497 项测试通过，包括 41 个生产概率/历史 helper 场景在 SM86 GPU 上执行。
  覆盖 1/2/7/296 块、稳定/变化最大值、范围回退、初始零概率块和非整块尾部。
  P 字节、最大值、alpha、mass、分母和历史输出与相同算法基线零容差一致；
  独立检查掩码字节为零，以及 FP8 解码求和。
- CUDA 12.4 交叉编译 276 个 SM90a 内核实例，其中新增 8 个。
  历史 268 个 SASS 指令文本不变（不包含编码及调度控制位比较）。
  新 D64/D128 均为 168 寄存器，stack/spill/local load/store 为零；无 WGMMA 串行化诊断。
- D128 主循环 QK 批次内可见 32 条 FP8 配对解码指令，主循环 PV 批次内没有这些解码。
  这是静态指令顺序证据，不是硬件利用率、延迟或 H200 正确性证明。
  尾块的独立指令可由编译器重新调度。
- native version 22 链接/导入、源码来源校验和源码包内 68 个文件哈希检查通过。
  上游 36 个 CUDA kernel 文件原样保留。

记录：[指令与资源](results/sage_vc25_codegen_local.json)、[本地验证](results/sage_vc25_local_validation.json)。
本地 GPU 不能执行完整 Hopper attention，以下远端验证仍需运行。

## H200 执行

```bash
git pull --ff-only origin master
bash scripts/validate_mass_h200.sh
```

只用已安装依赖离线编译，所有 Python 命令均为 `python3`，不需要 ncu/nvdisasm。
先验证完整内核再计时；验证失败会中止。
新模式覆盖 240 个 dtype/维度/K 长度/算法组合、每个重复三次，
另有 16 个均匀注意力已知答案及 24 个固定量化输入的最大值/尺度场景。
包括单块、两块、非整块尾部和 37741 长序列，全部输出行参与实现比较。

看后面的 **Deferred-mass candidates** 表；开头普通表仍是默认 vc15。
比较原版 Sage、vc16 control/ExpCast、vc22_both、两个 vc25 候选。
核心 API 共用量化数据，完整 API 每次重新预处理；随机顺序测 9 轮。
另报告 3 轮 × 20 次 CUPTI 内核耗时。

回传候选表、资源表和 **Repeated CUPTI kernel durations**。
结果保存在 `results/h200_vc25_mass.json`。
采用条件是全输出实现门槛通过，并且**同场完整 API 快于 vc22_both**。
两个源码候选可能被编译器合并，不能仅凭它们之间的计时差证明重叠收益；
比另一新候选快但比 vc22 慢仍不算优化成功。
0.1% 仅是实现回归门槛，不是 dense FP32 或模型质量门槛。
