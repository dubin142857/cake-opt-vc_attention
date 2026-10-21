# Vector row-max：本地实现与检查记录

本候选基于已实测的 `vector_scale`，不叠加已经回退的 bulk 系数广播。只调整 dense 四行 score 的最大值归约；INT8 QK、原版指数、FP16 PV、输入量化及 Vector 反量化的数值契约不变。本文保留实现时的本地检查；本地没有 CCE/NPU。

**2026-10-21 远端结论：数值检查通过，性能明显退化，不采用。** 用户回传全部 padded native 输出精确一致；相对 Vector scale，full API 配对节省 −13.4784 ms，native 为 −13.4772 ms，均 0/9 更快。当前保留 Vector scale，详细三方对比和分析边界见[回测记录](../../ascend/laser_quant/README.md#行最大值候选回测正确但明确变慢)。本目录 manifest 只记录本地审计，不冒充远端完整报告。

## 优化假设

旧实现先用 `vmax` 折叠不同 K128 块的完整行向量，再做行内 `vcmax`。新实现先归约每个 K128 块的每行 128 个元素，再归约这些局部最大值，尝试减少宽中间向量的 UB 读写及依赖层级。最大值比较的数学工作量没有因此消失，归约指令吞吐和新调度可能抵消收益。

生成源码的 `VectorForward.h:199` 为 `TryDenseTileRowMax`，调用点在 `:1223`。只在 `rowPerTime=4`、非三角/非稀疏、实际块数 8～32、包含 padding 后不超过 32 块时启用。其余情况调用原来的完整归约代码。源码版本及八个同族文件哈希见 `manifest.json`；生成目录可由离线 preparer 重建。

输入布局是 `score[k][row][128]`。对四行分别发射 `vcmax`，重复次数为 actual+padding；源 repeat stride 为 32 个 32B 数据块，目标 repeat stride 为一个 half。局部最大值放在 `partials[row*128+k]`。第二级将 mask 设置为 actual+padding，源 repeat stride 为八个 32B 数据块，四次重复将结果连续写入四个 half。结束时恢复完整 mask。

原来的 score 缩放、尾部 mask 和 K-block padding 在调用前已经完成。padding 仍以 `-65504` 参与归约；尾元素的原 `-60000` mask 不变。没有改变求和顺序或在线历史更新。对有限值最大值相同；不声称 NaN 或正负零平局的位模式相同，最终 native 输出仍必须通过远端全元素精确一致门限。

## CANNBot 使用范围

按 `ascendc-perf-optimize` 检查分块、通信、核间及核内路径：不改 D128、Q/K128、Cube 分块、核数或双缓冲；没有多卡通信；保留 Cube↔AIV 所有 handoff；本轮只改变核内 Vector 归约。没有真实流水仿真，不能推断硬件利用率或阶段百分比。

使用 `ascendc-api-best-practices` 的归约规则，以及 `ascendc-sync-audit` 的 full-audit 流程。用户已经明确授权继续实现候选和测试，不增加重复确认。

两个 analyzer 均扫描生成后的全部八个 kernel 同族文件，退出码为 0，findings 为空。但 `sync_audit` 识别的 event 数为 0，flow analyzer 的 operations/依赖/同步边也为空：正则前端没有覆盖此处的 raw intrinsic。**零 findings 不是同步正确性证明。** 原始 JSON、stderr、命令和工具哈希均保存在本目录。

## 手工数据流和生命周期核对

| Buffer / 阶段 | Producer → consumer | 索引及同步 |
|---|---|---|
| 当前 ping/pong score | 原搬入、反量化、mask、padding → 第一级 vcmax | 使用原 `pp_buf_for_attn_score_fp16[ping_pong_flag]`；原搬入等待不变，调用前保留 PIPE_V barrier |
| rowmax scratch 前 1024B | 四条 vcmax → 第二级 vcmax | expected/actual root 都是 `ub_attn.buf_for_cacl_rowmax_fp16`；读写同为 `row*128+k`，仅读已写的 actual+padding 个值；两级之间 PIPE_V barrier |
| 本批四行 maxima | 第二级 vcmax → 原在线 max/history 和广播 | expected/actual root 都是原 `maxJVectorUbAddr + linesRound`；目标四个连续 half，结果后 PIPE_V barrier |
| scratch 复用 | 本批第二级读取 → 后续原 sum/下一批 scale fallback 写 | 候选返回前 PIPE_V barrier 完成 scratch 读取；后续既有同步及调用顺序保留 |
| Cube/AIV 与 GM slot | 原 producer → 原 consumer | kernel 入口、Cube、地址映射、slot/event 编号均未变；没有新增跨核或跨流水 event |

此表是手工关联，不是 analyzer 生成的 sync coverage。实际头文件中 scratch 起始地址为 512，原分配 32KiB；候选仅复用其前 1024B，与 score、maxima、持久系数区域分离。源行起点按 256B 对齐，score 源 repeat 步进 1024B。maxima 每批偏移四个 half，满足目标 4B 对齐。新 helper 不增加 UB，guard 在任何写入或 mask 改动之前返回。

## API 依据与本地边界

官方 [WholeReduceMax 文档](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0079.html)规定源 stride 的单位为 32B，ONLY_VALUE 的目标 repeat stride 为结果元素，half 的源/目标对齐要求分别为 32B/4B。固定 MindIE 源码 `csrc/ops/common/op_kernel/simd.h:76` 的 `cmax_v` 映射提供了参数顺序的交叉核对。最大 repeat 为 32；生成源码沿用已有 raw `vcmax(..., Order_t(2))` 调用形式。具体 CANN 9.0.1 的 CCE 降低和实际执行仍需远端确认。

`tests/test_mindie_laser_quant_vscale_rowmax_patch.py` 共 9 项通过。主机 C++ 模型执行实际原版/候选归约及 padding 方法，覆盖块数 1～32、全负数、不同 K 块和行的最大值、重复值、正负零、FP16 极值、负无穷、subnormal 和尾部 mask；检查最终最大值、mask 恢复、临时缓冲区边界及 guard 回退。故意修改源 stride、第二级 mask、删除两级间 barrier 的负例均被检测。模型 DLL 不运行 CRT 全局构造，padding 常量使用 constexpr float，再由实际方法做 half 转换。

另有 12 项构建/离线来源测试、82 项新旧基准脚本测试通过；Python 语法、Bash 语法和补丁格式检查通过。主机模型不能验证 CCE 优化或真实异步硬件行为。

## 远端验收

```bash
git pull origin master
bash scripts/run_mindie_laser_vector_rowmax.sh
```

固定 BF16 BNSD `[1,28,37741,128]`，同进程随机交错比较 `original_laser`、`vector_scale`、`vector_rowmax`。只构建新候选，已有来源和库先校验，源码复用本地 Git objects，不访问 GitHub。

计时前执行既有数值参考、同地址 scale 更新、104 个新增边界用例和长形状全部 padded native 输出精确一致检查。新增用例覆盖 K128、8/16/32 块及 K4096 section 边界。主要采用 fresh full API 对原版 Laser 的配对收益；prepared/native 和 profiler 另列，不混用或相加。原版与 INT8 的 dense FP32 误差不同，不能宣称等精度或模型效果等价。
