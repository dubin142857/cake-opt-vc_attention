# Vector scale 独立候选：设计与同步审计

本候选从 pinned qresident 派生，将 QK 的动态反量化系数从 Cube FIX 路径移到 Vector 已有的 score 缩放位置。它改变舍入顺序，不能视为 qresident 的位级等价替换。初版 `9f9c5d0` 在远端 CANN 9.0.1 的 INFERCHANNEL 编译阶段失败，尚未进入 NPU 数值测试或性能测量。本次修复后的设备编译仍待远端确认。

2026-10-20 编译修复：`vector_dup` 的 gain 参数改为显式 `half` 值；FP32 回退路径先将 UB 数组元素读入普通 `float` 标量，再显式转换传给 `vmuls`。本次只修正这两个调用的标量类型表达形式，未改变数值公式、广播步长或同步位置。准备器仅允许识别出的旧版生成源码升级，任意其他内容修改仍拒绝覆盖；重跑原一键脚本即可增量构建，无需清理目录或访问 GitHub。

使用本地 `build/cannbot-skills/ops/ascendc-perf-optimize` 的四步分析框架、`ascendc-sync-audit` 的完整扫描流程及 `ascendc-api-best-practices` 的广播/精度/同步说明。用户已授权实现该独立候选；技能中的修复前确认步骤不构成新的审批要求。按本次明确要求保留已有的 DCCI + scalar GetValue 缓存一致性方案，未按 API 技能的一般性能建议另行扩大 scale DMA 改造范围。

## 四步性能分析及选择依据

1. **Tiling 建模。** 保留每 Cube 轮次 64 个 128×128 score 块、ky=2、每次处理四行的 Vector 布局、Q 的 L0A 驻留。每 section 最多 32 个 K128；快速路径每四行仍只执行一次向量乘法。新增 4096 字节 half gain 与 128 字节 FP32 系数，不扩大 score/P 的 GM 工作区。
2. **卡间流水。** 本算子不包含卡间通信，因此不适用；GM/L1/UB 搬运不归类为卡间通信。
3. **核间流水。** 保留已有 Cube/Vector 跨核通知、round 奇偶工作区和两轮调度。技能的核间专题资料为空；这里手工复核数据生产消费，未声称已有核间仿真或新的重叠改善测量。
4. **单核流水。** 本次仅改变 scale 所属流水与 QK transport。最新日志中 Q resident 将 AIC MTE1 导出活动时间降低约 4.63 ms，而 native 配对 wall 仅减少约 0.1013 ms；继续减少同类搬运不一定影响关键路径。INT8 与 FP16 的导出 FIX 活动差异、逐 tile `set_quant_pre`/FIX barrier 则提供了值得验证的不同方向。它们不是阶段延迟或瓶颈证明，不能相加推导收益。

待测假设：一次固定 FIX 配置能否减少 QK 结果路径的控制/串行成本，且收益大于 Vector scale 读取、gain 构造及可能的 FP32 回退开销。未改变分块的本次实验不能代替后续 INT8 专用分块与流水设计。运行时应报告实际设备及 Cube 数量，不把 64 当作设备事实。

## 数值与支持范围

令 `I` 为 D128 INT32 点积，`h` 为 FP16 舍入：

```text
transport = h(I / 64)
gain = bitclear_low13(float32(q_scale * k_scale))
coefficient = float32(float32(64 * gain) * float32(h(SCALE)))
fast = coefficient == 0 or 2^-14 <= abs(coefficient) <= 65504
fast score = h(transport * h(coefficient))
fallback score = h(float32(transport) * coefficient)
```

`dot(-128, -128)` 的 D128 极值为 2097152，除以 64 后为 32768，transport 对完整 INT8 范围安全。`1/32` 不能覆盖该极值。系数的 normal-half 判断避免把很小或很大的有效 FP32 系数提前转 half 造成额外下溢/溢出；回退块在统一 half 乘法中使用中性系数 1，然后按上式单独处理。

公开入口仍为八参数 torch bridge，限 dense MHA，原始 exp/softmax 和调用者提供的 FP16 V/PV。bridge 拒绝 `expcast=true`。保留的内部 ExpCast 代码不是本候选已验证的支持声明。没有新加 V 量化、平滑或 P/V INT8。

索引在原 dense mapper 已知 `b,n,ir,ic` 时写入：

```text
q_base = (b * H + n) * (Qpad / 128) + ir
k_base = (b * H + n) * (Kpad / 128) + ic
section 2*i     : q_base,   k_base
section 2*i + 1 : q_base+1, k_base
K block j       : k_base+j
```

两个 AIV 分别负责同一 Q128 的上下 64 行，因此共享该 Q128 的 scale。没有从 workspace 地址反推原 Q/K 索引。

## 工具扫描原始结果

对 qresident 和 vscale 的完整 `op_kernel` 同族分别扫描全部八个 `.h`/`.cpp`：两个 address mapper、`address_const.h`、`matmul_const.h`、`CubeForward.h`、`VectorForward.h`、`vc_laser_probability.h` 和对应入口。分析器和源文件哈希保存在 [manifest.json](manifest.json)；没有改动工具脚本。

| 扫描 | 文件数 | 退出码 | 识别事件/操作 | findings |
|---|---:|---:|---:|---:|
| baseline sync_audit | 8 | 0 | 0 events | 0 |
| candidate sync_audit | 8 | 0 | 0 events | 0 |
| baseline flow analyzer | 8 | 0 | 0 operations / 0 edges | 0 |
| candidate flow analyzer | 8 | 0 | 0 operations / 0 edges | 0 |

原始结果：[baseline sync](baseline-sync_audit.json)、[candidate sync](candidate-sync_audit.json)、[baseline flow](baseline-ascendc_flow_analyzer.json)、[candidate flow](candidate-ascendc_flow_analyzer.json)。stderr 文件也原样保留。

**这些零值表明 regex 前端未识别本代码使用的低阶 intrinsics，不能解释成同步正确性证明。** 没有工具生成的红线/高级别候选可列；人工数据流检查及主机模型提供补充证据，仍不验证真实设备异步流水。

## 人工 producer → consumer → sync 对照

| 数据及地址来源 | producer → consumer | 同步、生命周期与索引复核 |
|---|---|---|
| Q/K INT8 L1/L0 与 Q resident 两个 L0A 槽位 | MTE2 → MTE1 → MAD | 所有原 set/wait、MAD 参数、ping-pong 切换保留；实际 QK 方法的主机 ownership 模型通过。没有更改驻留有效期或跨 descriptor 复用策略。 |
| 固定 quant register | 每次 Cube `Run` → 后续 QK `DEQF16` | `CubeForward.h:668` 一次 `set_quant_pre(0x3c800000)`，随后一次 FIX barrier。全同族无其他 setter；PV 与 diagonal 输出使用 NoQuant。目标设备寄存器/UnitFlag 语义仍需 CCE 与 NPU 验证。 |
| score GM 奇偶缓冲 | QK FIX → Vector MTE2 → V | `ffts_cross_core_sync(PIPE_FIX,...)`、`wait_flag_dev` 及 MTE2→V 原事件全部保留。Cube mapper 的 round `%2` 与 Vector `attentionScoreOffset` 不变。 |
| Q/K scale GM 原始指针 | 前序同流量化 → AIV scalar | 每次入口重新设置两组缓存 tag 为无效；每个绝对 64 字节 cache line 首读/切换时 DCCI，使用界内原指针读取。实际方法主机缓存模型验证同一地址重写后刷新。 |
| gain UB `[174848,178944)` | `vector_dup` → section 内所有 `vmul` | prepare 开始/结束均有 PIPE_V barrier；数据跨 max/exp/rowsum 保持独立存活。每 128 score 元素对应一个 16-half gain 块；block stride 0、repeat stride 1。主机模型执行实际方法并逐元素验证广播。 |
| FP32 coefficient UB `[178944,179072)` | scalar store → 后续 scalar load 作为 `vmuls` 标量 | 仅 scalar 写/读；Vector 不把该内存作为矢量源。数组与 gain、score、softmax 状态分离；每 section 更新后读取。 |
| fallback scratch `[512,2560)` | f16→f32 cast → FP32 multiply → f32→f16 cast | 每步 PIPE_V barrier；只在当前 max reduction 之前复用原 rowmax scratch。每块 512 个 FP32，2048 字节；随后才允许 reduction 覆盖。主机模型 sentinel 验证写入范围。 |
| P GM 与 rowmax/rowsum diagonal | Vector V/MTE3 → Cube PV/diagonal | 原 V→MTE3、跨核 `PIPE_MTE3` 通知及 Cube wait 保留；P、输出历史与 diagonal 地址未改变。 |

这里的人工 expected/actual root 分别是同一组 Q/K 槽位、score 奇偶工作区、`ub_attn.vscale_gain_fp16`、`ub_attn.vscale_coeff_fp32` 和 `ub_attn.buf_for_cacl_rowmax_fp16`。脚本没有生成可引用的 `sync_coverages`，不能为这些人工链路补造工具证据。

## 具体源码变化

生成文件位于 `build/mindie_laser_quant_vscale_source/csrc/ops/vc_laser_quant_vscale_attention/op_kernel/`。以下行号对应 manifest 中记录的最终源码。

```diff
CubeForward.h:668
- 每 Q128/K128 读取动态 scale、设置 quant register 并 barrier
+ 每次 Run 一次设置 1/64；QK store 只输出固定缩放 transport

VectorForward.h:130 / :166 / :1150
- 对整个 score section 执行 (half)SCALE 的 vmuls
+ section 开始构造每块 gain；一次广播 vmul，必要时 FP32 scratch 回退

VectorForward.h:651
+ 在原 UB 分配末尾追加 4096B gain 和 128B coefficient

AddressMappingVectorForwardOnline.h:295 / address_const.h:164
+ 在 dense section 显式记录物理 Q128/K128 scale 索引

vc_laser_quant_vscale_attention.cpp:72
- Cube 接收 Q/K scale 指针
+ Vector 接收 Q/K scale 指针并重置 invocation-local cache tag
```

## 验证及边界

`tests/test_mindie_laser_quant_vscale_patch.py`：本次编译修复后 **15 passed**。使用本机已有 MSVC 编译并执行从实际补丁提取的方法，没有安装编译器。新增的两项回归通过严格主机桩要求显式标量值，检查两个旧调用形式不能回归；该桩不是 CCE 编译器，也不模拟地址空间类型规则。

- 实际 Cube QK outer/base 方法：不同 L0 ping-pong 初始值、K 块数 0–32、ky=1/2、原 upper-right skip、descriptor/PV 边界；数据身份和原事件 ownership 检查通过。
- 实际 Vector prepare/apply/UB allocation 方法：1–32 K 块、16 个四行 chunk、正常/零/很小/很大/负系数、完整 transport 极值、同地址 scale 改写、输出/scratch sentinel；快速与回退结果符合所声明公式。
- 实际 dense mapper：跨 row/head/batch/round、两个 AIV；另对 **24、32、64 个 Cube**、B=2/H=17/Qpad=512、Kpad=256/512/4352/8448/37888 逐配置执行容量检查，**不跳过**，全部通过。
- 探索性极少核配置揭示继承的边界：原 `FORWARD_SECTION_INFO` 仅能存八个 section，即四个两行 descriptor；某些低核数、短 K、大 B/H/Q 组合会超过容量。本候选没有扩容；探索模型明确跳过容量已不成立的原调度配置，不将其计为支持。真实设备需记录实际 Cube 数；远端验证不能在未知配置上默认安全。

尚未通过验证：修复后目标 CCE 对 intrinsic 参数及低阶地址空间的编译、真实 FIX register/UnitFlag 行为、跨流水执行顺序、设备浮点舍入和 FTZ、全算子精度与性能。初版远端的两个参数类型错误记录在本次验证记录中；主机执行模型只验证所建模的控制、寻址、广播及数值公式，不能替代上述设备验证。初版 137 项检查与本次实际重跑的检查分别列在 [local_validation.json](local_validation.json)，不将历史结果冒充本次重跑。
