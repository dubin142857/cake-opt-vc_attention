# Vector-scale bulk broadcast：本地检查记录

本轮只改变 Vector 系数的展开方式，目标是在相同量化和 score 数值契约下减少重复指令发射。上一轮远端 Vector-scale 已有实测收益；本候选尚无 CCE 编译或 NPU 性能结果，不能把指令数下降当成已经提速。

## 优化假设和代价

长 section 有 32 个 K128 块。原版逐块计算系数并发射 32 次 `vector_dup`，每次写入四个相同的 32B row-gain 块。本候选保留逐块 FP32 运算、低 13 位截断、FP16 舍入及 fallback 判断，把结果先存入连续的 `half[32]`，然后发射四次 `vbrcb`。短 section（1～7 块）沿用原路径。

四次 `vbrcb` 分别负责一个 row-gain，参数是 `dstBlockStride=4`、`dstRepeatStride=32`、`repeat=ceil(block_count/8)`。这使输入的第 b 个系数写入 `gain[b*64 + row*16 : +16]`。不足八个系数的尾部补零；额外写入只在现有 gain 分配范围内，attention 不读取这些补齐块。

代价是新增 compact 标量写入、两个方向的同步，以及系数计算期间失去原先可能存在的标量/vector 重叠。收益必须以与原版 FP16 Laser 的完整 API 随机交错实测判断，不能预先断言。

## CANNBot 流程范围

- Step 1：保持现有 D128、Q/K128、Cube 每轮 64 块、ky=2、Vector 四行处理及双缓冲调度。只增加 64B UB，不改核数/L1/L0/GM 分块。
- Step 2：无多卡通信，跳过。
- Step 3：保留原有 Cube↔两个 AIV 的所有 round handoff，不修改核间事件；技能现有核间分析目录没有额外流程。
- Step 4：针对原先重复系数展开指令提出独立候选。没有仿真流水图，不推断新的硬件利用率或阶段占比。

查阅了 `ascendc-perf-optimize`、`ascendc-sync-audit` 的 full-audit、`ascendc-docs-search` 和 `ascendc-api-best-practices`。用户已授权继续实现优化，因此完成候选及验证材料，不另行请求确认。

## 数据流和同步

| Buffer / 生命周期 | Producer → consumer | 同步与索引 |
|---|---|---|
| compact `[179072,179136)`，32 个 half | 上 section `vbrcb` 读 → 下 section 标量写 | section 开头 `set_flag/wait_flag(PIPE_V,PIPE_S,EVENT_ID0)`，先完成旧读取 |
| compact 当前 section | 标量逐块写 → 四次 `vbrcb` 读 | 写入及补齐完成后 `set_flag/wait_flag(PIPE_S,PIPE_V,EVENT_ID0)` |
| gain `[174848,178944)` | 四次 `vbrcb` 写 → 原 `vmul` 读 | `pipe_barrier(PIPE_V)`；每个 K128 的四个 row 块均保持原索引 |
| coefficients `[178944,179072)` | 标量 coefficient 写 → 原 fallback 标量读 | 原数值、地址及 FP32 fallback 不变 |
| rowmax scratch `[512,2560)` | 原 FP32 fallback → 原 max reduction | 保留原 `PIPE_V` barriers；compact 不与该区域重叠 |

本候选两个方向都使用立即配对、立即等待的 event 0。原文件在 section 内的流水 V→S 事件使用数组 1～6；tail-diagonal 的 event 0 也在原调用点立即配对完成。既有 MTE、Cube 及跨核事件保持不变。两个 AIV 各自使用自己的 UB。

所有新增地址 32B 对齐，Brcb 输入输出不重叠。最大 `repeat=4`，最大写入索引 2047 half，恰好位于已有 4096B gain 区内。新增总 UB 结束地址 179136，小于 192KiB。

## API 来源与验证边界

官方 [Brcb 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0089.html) 明确：A2/A3 支持 uint16，每次迭代读取八个元素，各填一个 32B block，起始地址需要 32B 对齐。原始 raw `vbrcb(dst,src,blockStride,repeatStride,repeat)` 参数顺序以固定 MindIE 源码 `VectorForward.h` 中原有调用和 `common/op_kernel/simd.h` 的 `brcb_v` 映射交叉核对。未取得目标机器 CANN 9.0.1 的 raw intrinsic 头文件；本地不代替目标 CCE 编译。

[核内同步文档](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/intra_core_sync_overview.html) 列出 AIV 的 S→V / V→S 事件。新增原始调用形式与同一 pinned kernel 已使用的 `set_flag/wait_flag` 形式一致。

两个 CANNBot analyzer 扫描生成后的 8 个 kernel 同族文件，均退出 0，但识别到的 events / operations 都是 0，findings 也是 0。其正则前端没有覆盖这里的 raw intrinsic，**零 findings 不构成同步正确性证明**。原始输出、工具哈希、源码哈希在本目录保留。

## 本地验证

`tests/test_mindie_laser_quant_vscale_bulk_patch.py`：7 项通过。MSVC 编译执行实际生成的系数准备和应用方法，独立实现 Brcb stride 模型，检查：

- K block 数 1～32、K 起始 cache-line 内偏移 0～15、同一地址重复调用刷新；
- 零、tiny、normal、large、negative 系数及 FP32 fallback，FP16 输出 bit pattern；
- 每个 section 的 16 次四行 score 消费，gain / compact / scratch canaries；
- 补齐值只在有效分配内写入，长 section 发射四次广播；
- 分别删除 V→S 或 S→V handoff 的负例会被协议模型拒绝；
- 原 scale apply、Cube 和地址映射不变，错误或重复源版本 fail closed。

Host 模型不模拟 CCE 优化、真实异步执行或硬件舍入细节。远端还需完成原版 Laser / vector-scale / bulk 三组结构化精度、同 payload 全输出一致性、FP32 sampled reference 和随机交错 API 性能测试。
