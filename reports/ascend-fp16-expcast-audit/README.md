# 稳定 PV 预取版的 FP16 ExpCast 同步审计

本实验以已跑通的 `vector_pv_prefetch` 为父版本，增加独立算子 `VcLaserQuantVscalePvExpcastAttention`。`expcast=false` 保留父版本普通指数，`expcast=true` 使用 FP16 位编码近似。仍是 INT8 QK、FP16 P/V 与原在线历史更新；不是论文 E4M3 编码，也没有原生 FP8 矩阵乘、V 重排或平滑。

## 扫描来源及结果

按照本地 CANNBot `ops/ascendc-sync-audit/SKILL.md` 与 `workflows/full-audit.md` 的完整源码族扫描流程执行。`regenerate.py` 只读取本地固定 MindIE-SD Git objects（`71098c20ff90838d1fd1ca0a40515c751a3869b0`），在 `build/ascend_fp16_expcast_audit` 分别重建父版本与候选。没有下载、CANN 编译或设备启动。

两个版本均扫描 13 个同族 C++ 源码/头文件，包括 Cube、Vector、两个地址映射头、常量、入口、tiling、host 定义及概率 helper。四次 analyzer 调用均退出 0，findings 为 0；原始 stdout、stderr、命令、工具和源码 SHA256 均保留在本目录。

**工具覆盖不足必须单列：** `sync_audit` 的 event 数为 0，`ascendc_flow_analyzer` 的 operation、buffer dependency 和 sync edge 数也为 0。它们使用正则前端，未识别本算子的 CCE raw intrinsic 数据流；没有展开 CANN SDK 头或模板。因此零 findings 不是同步正确性证明。未产生需要转述的红线、高级别或性能候选；不能宣称已被自动工具完整验证。

`source_comparison.json` 核对全部生成文件，在身份字符串归一化后只有 `op_kernel/vc_laser_probability.h` 发生变化。Cube、Vector 调用者、地址映射、tiling 及原 exp 分支逐字节不变。bridge 的唯一逻辑变化是移除拒绝 `expcast=true` 的条件，原接口、其他参数限制和流派发不变；tiling 的 attr7 指针非空检查保留。新概率 helper 的差异在 `vc_laser_probability.h.diff`。

## 手工 producer → consumer → sync 核对

以下是源码生命周期审阅，不是工具生成的数据流证明。位置指向本次生成的候选 `op_kernel/VectorForward.h` 和仓库中的 `ascend/laser_expcast_fp16/vc_laser_probability.h`。

| 数据与 expected/actual root | producer → consumer | 同步与生命周期依据 |
|---|---|---|
| score，`pp_buf_for_attn_score_fp16[ping_pong_flag]`，helper 实参和返回写入根相同 | 原 GM→UB → Vector 缩放、max、subtract → helper | 原 `MTE2→V` set/wait 保留（Vector 1147–1148）；subtract 后 `pipe_barrier(PIPE_V)` 保留。helper 没有新增 DMA，也不改变 ping/pong 索引 |
| scratch，`buf_for_cacl_rowmax_fp16`，以同根 `float*` 视图传入 | 原 max-reduction 使用结束 → helper FP32 affine 临时量 → 后续 rowsum scratch 重建 | max 最终结果已经转存/广播到独立缓冲，max及 subtract 完成后才调用 helper（Vector 1193–1333）。helper 各段之间均保留 V barrier；不改跨核事件 |
| helper 当前 chunk 的 score 输入 → scratch | `vconv_f162f32` → `vmuls/vadds/vmaxs/vmins` | 转换后及每个依赖操作后都有 `pipe_barrier(PIPE_V)`；scratch 本轮最多 8192 个 float，不越过 32 KiB |
| affine scratch → score 的 int16 视图 → FP16 probability | `vconv_f322s16r` → 同一 score 地址的 FP16 `vmuls(2^-12)` | RNE 写码后有 V barrier；int16 视图只改变解释方式，不增加缓冲。FP16乘法完成后有 V barrier，再复用 scratch 处理下一 chunk |
| FP16 P 的 UB→GM 与 rowsum | helper 返回 → 原 `copy_ubuf_to_gm`、`process_calc_sum` | helper 返回时恢复完整 vector mask；原 `V→MTE3` set/wait（Vector 1341–1342）不变。rowsum 保留原有归约 padding 和 scratch 重建，随后消费有效 score；后续原 buffer 释放/复用及 AIV→AIC 通知保持 |
| QK/PV 的 L1/L0、Cube/Vector round handoff | 父版本各 producer → consumer | `CubeForward.h`、`VectorForward.h` 与地址映射在身份归一化后字节相同；未加入 trace、计时数组、DCCI、PIPE_ALL 或任何新 event ID |

## 地址、长度与 mask

固定 dense 原生布局中 `MAX_LENG_PER_UB_PROC=16384`、`sizeof(INPUT_T)=2`、`rowPerTime=4`。rowmax scratch 是 UB 字节区间 `[512,33280)`；两个 score 槽为 `[66048,98816)`、`[98816,131584)`，与 scratch 分离。调用者保留静态检查，要求现有 scratch 至少容纳 `8192*sizeof(float)`。新代码没有增大 UB 终点。

helper 每次处理最多 8192 元素，FP32转换/算术以64元素重复，最后FP16乘法以128元素重复。count 来自 `section_block_num*rowPerTime*128`，原生允许路径下为128的倍数；8192分块及尾块都不会产生不足128的残块，repeat最大128，小于 uint8 上限。每个chunk仅读写自己的 `score+offset`，scratch从起点复用，结束barrier在下一轮之前。

转换与FP32阶段设置64位有效mask，FP16输出乘法和函数返回恢复完整128位mask。masked `-inf` 经下限截断编码为0。概率的次正规数是否按契约保留，仍必须由远端CCE编译和设备数值检查确认，不能用CPU oracle代替。

## 改动与验收边界

原父版本 helper 是未启用的 E4M3 解码实验；新 helper 改为 binary16 位编码。算术契约是 FP32 独立乘加、RNE生成FP16位码、FP16乘 `2^-12`，具体常量及版本写入源码manifest。原在线历史指数、分母更新与PV累加未变。由于近似函数不满足指数乘法恒等式，精度参考必须模拟相同在线分块，不能直接替换成一次全矩阵softmax。

当前审计没有发现需新增同步修复的源码差异；这不构成设备正确性或性能承诺。远端先做小形状、尾部、跨section、FP16/BF16的参考检查，并要求新算子关闭近似后与稳定父版本全部native输出一致。通过后才测同进程随机配对的完整调用/native时间；失败即停止，不启用此前挂起的PV内部计时探针。

复现本地静态证据：

```bash
python3 reports/ascend-fp16-expcast-audit/regenerate.py
```
