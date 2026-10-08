# vc26：两个 Q64 warpgroup 共用 K/V

包版本 `sageattention-local==2.2.0+vc26`，native kernel version 23。
目标仍是 H200 上 `[B,S,H,D]=[1,37741,28,128]` 的完整 attention API。
这版已完成本地检查和 SM90a 交叉编译，尚未在 H200 执行，不能据此宣称提速。

## 实验依据

用户回传的 vc25 同场结果中，`vc22_both` 完整 API 为 **25.5605 ms**，
两种延迟分母候选分别为 25.8718 / 25.8467 ms，均未改善。
原版 Sage 同场为 26.4675 ms；vc22 的加速比为 1.0355×，距离 22 ms 还需减少约 13.9% 延迟。
这些是算子 API 耗时，包含预处理，不是模型端到端吞吐。
记录：[vc25 回传摘要](results/h200_sage_vc25_reported_summary.json)。

本轮改变 CTA 的工作分配：原来每个 128 线程 CTA 计算一个 Q64 块；
现在一个 256 线程 CTA 内两个 warpgroup 各算一个 Q64 块，共用同一份 K128/V128。
每个 warpgroup 仍使用原来的 Q64 INT8 量化尺度。没有改为 Q128 量化。

| 模式 | 调度 | 目的 |
|---|---|---|
| `vc26_shared` | 两组 QK 各自提交，完成后会合，再计算概率/PV | 测试共享 K/V 的影响 |
| `vc26_stagger` | WG0 完成 QK 后释放 WG1，WG0 继续概率/PV | 在相同共享布局上测试组间交错 |

两种模式都支持普通 softmax 和 ExpCast。ExpCast 基线是 `vc22_both`，普通 softmax 基线是 `vc16`。
保留 INT8 QK、当前 FP8 P 编码、解码 P 后的 FP32 分母、两级 FP32 PV 累加和尾块掩码。
关闭 V 平滑；默认 `optimized` 继续指向 vc15，候选必须显式选择。

## 同步与资源

两个组都完成当前 K 的读取后，才由 WG1 的线程 128 发起下一块 K 的 TMA 加载。
交错模式用 named barrier 1 将 K 从 WG0 交给 WG1；WG0 只 arrive，WG1 sync。
每轮 PV 完成后两组通过 CTA barrier 0 会合，再由线程 0 覆盖 V。
最终尾块也会合，避免一组退出时另一组仍使用共享内存。
Q 尾部不足一个 Q64 组时不提前退出，而是 TMA 填零、守护量化尺度读取和输出写入。
规则依据：[NVIDIA named barrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar)、
[WGMMA 完成等待](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)。

共享 K/V 减少的是两个 Q64 块重复发起的加载，不保证显存流量或运行时间减半。
D128 动态共享内存从 40960 增至 49152 字节，线程数从 128 增至 256。
本地编译 D128 每线程占 167–168 个寄存器；每 CTA 寄存器需求因此明显增加，可能减少驻留 CTA 数。
远端资源表用真实 256 线程计算 occupancy；它是静态预测，不是硬件利用率。

## 已完成的本地检查

- 536 项测试通过，包括 30 个双组同步/量化索引 GPU 测试及 2 个验证流程 stream 回归测试。
  覆盖 Q64/Q128 边界、奇数 Q64 块、不同 batch/head、37741 长度、296 轮复用和非默认 stream。
  同步探针用标量 shared-memory 操作模拟组内完成，再调用生产同步 helper；不验证 TMA/WGMMA 本身。
  stream 回归测试使用真实 CUDA stream 和公共 API wrapper，但用替身替代量化/attention 数学，
  验证预处理/公共 API 留在默认 stream，候选原生入口在侧 stream，并显式等待输入和输出。
- CUDA 12.4 交叉编译 292 个 SM90a 实例，新增 16 个；历史 276 个 SASS 指令文本不变。
  新实例无 stack/spill/local load/store，无 ptxas 或 WGMMA 串行化诊断。
  交错版保留 named-barrier arrive/sync 指令；每组 QK/PV 的矩阵指令数量保持原样。
- native version 23 链接和导入、源码来源校验及源码包检查通过。

记录：[代码生成](results/sage_vc26_codegen_local.json)、[本地验证](results/sage_vc26_local_validation.json)。
本地 GPU 为 SM86，完整 Hopper attention 的正确性和性能仍须远端确认；CUDA 13 也可能生成不同指令。

## H200 执行

```bash
git pull --ff-only origin master
bash scripts/validate_dual_query_h200.sh
```

如果已经成功安装 vc26，却遇到 `Sage VC currently requires the default CUDA stream`：
这是初版验证脚本把公共 API 和预处理放在侧 stream 的错误，不是编译失败。
修复后只有原生 `forward_prepared` 在侧 stream；预处理、基线和公共 API 均在默认 stream。
每次跨 stream 调用都有双向等待，并记录输出张量的消费 stream。
先前报错前的部分结果不足以证明完整 vc26 正确性，需要重新验证。
CUDA 内核及包版本不变，可复用已安装二进制：

```bash
git pull --ff-only origin master
bash scripts/validate_dual_query_h200.sh --skip-build
```

`--skip-build` 仍执行全部正确性检查，通过后才计时，不会跳过测试。

使用已安装依赖离线编译，Python 命令均为 `python3`，不需要 ncu/nvdisasm。
先跑完整验证再计时。新候选覆盖 352 个 dtype/维度/shape/算法/模式组合，每个重复三次，
另有 16 个独立均匀注意力答案及 32 个 HND/NHD 公共 API 检查。
已知答案特意将 V padding 设为非零，以检测错误尾块掩码。

看后面的 **Dual-query candidates** 表，开头的普通阶段表仍是默认 vc15。
候选表在同一轮中比较原版 Sage、vc16 control、vc22_both、两种双组模式各自的 control/ExpCast。
核心 API 共用量化输入；完整 API 每次重新预处理。随机顺序测 9 轮，另测 3 轮 × 20 次 CUPTI 内核耗时。
完整 API 和核心 API 都检查全部输出行，0.1% 是实现回归门槛，不代表模型质量。

结果：`results/h200_vc26_dual_query.json`。回传候选表、资源表和重复 CUPTI 表即可。
采用条件是正确性检查通过，并且同场完整 API 比 `vc22_both` 和原版 Sage 更快；
仅交错版快于共享版、或仅核心变快，都不足以宣称达到目标。

单独复核完整 Hopper 内核正确性：

```bash
python3 scripts/validate_sage_vc.py --dual-query-only --output results/h200_vc26_validation.json
```
