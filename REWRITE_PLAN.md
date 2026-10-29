# 按 Laser 数据流重写 + 叠 INT8 —— 相位计划

2026-10-29 立。用户决定：按 Laser 的数据流重写，并在其上叠 INT8；(e) 的第三个设计点随重写一起做。
回退点：tag `delivery-h26-bitdecode` → `e2fc535`（产物 `3494c48d…`，带注解记录了实测状态与已知缺陷）。

## 为什么重写而不是继续优化

实测（同机同卡同形状，逐核均值 us）：

| | `aic_cube` | `aic_fixpipe` | `aiv_vec` | 窗口 | 下界 | 重叠度 |
|---|---:|---:|---:|---:|---|---:|
| **Laser** | 62731 | 34611 | 59172 | 87299 | **cube 62.7 ms** | 3.74 |
| 本交付 | 50767 | **192589** | 137343 | 280251 | **fixpipe 192.6 ms** | 1.64 |

量化砍 cube，而 cube 在我们这里只占窗口 18.1%——**交易做在了错的通路上**
（净账：cube 省 3.2 万 us，fixpipe + aiv_vec 亏 14.2 万 us）。
而 fixpipe 在当前数值契约下已无工程手段可降（两个 tile 轴都到 API 硬上限 F116；
收窄 dtype 过不了 1% 精度闸 H12）。
相对**自己下界**的效率我们 1.46 / Laser 1.39，**相当**——调度不是问题，下界才是。

★ **量化只在 cube 受限时付钱。Laser 的数据流正是 cube 受限的。**

## 从 Laser 源码读出的八条（已读，勿重读）

| | 技巧 | 出处 |
|---|---|---|
| T1 | softmax 行重缩放做成**对角阵矩乘**搬到 cube（AIV 只算 128 个 1/l_j） | `VectorForward.h:464-474` `form_diagmat` / `CubeForward.h:201,:353` `diag_matmul` |
| T2 | AIC/AIV 编成**两个独立程序** | `#ifdef __DAV_C220_CUBE__` / `__DAV_C220_VEC__` |
| T3 | 基本块 **128×128**，恰好填满各级缓存一半 | L0AB ping-pong 32KB / L0C 64KB / L1 512KB 切 8×64KB |
| T4 | **手写全部物理地址**，不用 `TPipe`/`TBuf`/`MatmulImpl` | `CubeForward.h:85-110` |
| T5 | 跨核握手手写且打在**最早点**（cube 在 fixpipe 写完即发） | `ffts_cross_core_sync(PIPE_FIX,…)` |
| T6 | workspace 分数矩阵是 **half（2 字节）** | `CubeForward<half,false,half>` |
| T7 | 地址计算与矩乘发射**解耦**（一次算 16 块再批量发） | `AddressMappingForwardOnline.h` `addrMapping_cube1` |
| **T8** | ★cube 侧**深度 2 的软件流水**：同一轮发射 `cube2(i−2)` 与 `cube1(i)` | `CubeForward.h` Run 的 `Z>1` 分支 |

**T8 形成三级重叠**：`QK(i) ‖ softmax(i−1) ‖ PV(i−2)`。
我们现在完全相反——`mmq` 与 `mmv` 背靠背且中间有 `PipeBarrier<PIPE_ALL>()`（竞态修复），**全串行**。
跨核握手也是**每轮一次**而不是每 tile 一次。

轮次模型（`is_running`）：块按二维网格切，每核每轮取 `kx×ky` 块，核间按行带交错。

## 相位

每相独立可验收，**上一相不过不进下一相**。

> ★**2026-10-29 修正（F185）**：本节原文"FP16 契约先不量化"与下方 Phase 4"叠 INT8 QK"
> 建立在一个**假前提**上——以为基线是 FP16。实际 `vc_torch.cpp:38` 写着
> `CheckTensor(q, q, at::kChar, "Q")`：**Q/K 在算子入口就是 int8**，INT8 QK 是接口契约，
> 不是我们的选择。验收闸也写错了参考对象，见下。原文保留以存证，以修正段为准。

### Phase 0（修正后）— 把 S 工作区由 int32 改 half，直接作用在 binding pipe 上
- **动因（数据而非类比）**：binding pipe 是 `aic_fixpipe`（192589 µs / 窗口 280251）。
  fixpipe 搬的是 L0C→GM 的字节。按契约算：S 每 (查询行, 键) 对 4 字节，O 每对 `4·D/N`=2 字节，
  **S 占 fixpipe 字节的 2/3**。把 S 由 int32 改 half 是作用在 binding pipe 上的最小改动。
- **不需要手写 cube**：`MatmulImpl` 自带 `SetQuantScalar(uint64_t)` /
  `SetQuantVector(GlobalTensor<uint64_t>)`（`asc/impl/adv_api/detail/matmul/matmul_impl_base.h:110-112`）。
  K 的 scale 沿**列**变化，正是 quant vector 能表达的方向；Q 的 scale 沿**行**变化，
  留给向量侧一次 `Muls`——比现在 `KSB_PER_TILE × K_SCALE_BLOCK/64` 次分段 `Muls` 还少。
- **连带收益**：向量侧少一次 `ROWS×N` 的 `Cast(int32→fp32)`；MTE2 读 S 的字节减半。
- **风险**：S 由 fp32 中间值改 fp16 会在 logit 阶段引入舍入。当前 `decoded_reference`
  RMSE 最大 7e-5、闸 1e-2，**余量 143 倍**；Laser 同样用 half 工作区且对 dense 只有 0.191%。
  有余量不等于够用，按闸实测。
- **验收**（★用仓库自己的闸，不是新编的）：
  `scripts/validate_ascend910b.py` → `status=passed`、**141 个用例**、
  `decoded_reference` 相对 RMSE ≤ 1%（`validate_ascend910b.py:173` 抛的就是这条）；
  `dense_fp32` 只记录**不设闸**（当前最大 4.63% / 中位 2.86%，即对甲方报的 2.98%）；
  固定 payload 8 次逐位一致。
- ★**冻结预测（E9，测前写定，不符时原文不改）**：若 fixpipe 时间与字节成正比，
  fixpipe 字节由 6 单位降到 4 单位（×0.667），192589 → 约 128400 µs，**fixpipe 仍 binding**
  （cube 50767），窗口 280251 → 约 216000 µs。但 Probe 1b 已部分否证按字节的比例关系
  （M 加倍时 fixpipe 只 ×1.487 而非 ×2），故这是**上界量级估计不是预测值**：
  实测落在 ×0.667 与 ×1.0 之间都不算意外，落在 ×0.667 以下才说明另有机制。
- ★**T6 必要但不充分**：S 与 O 都改 half 最多给 fixpipe 约 2 倍，而我们与 Laser 的
  fixpipe 差距是 **5.56 倍**。余下部分要另找机制，**不要用 T6 的结果解释全部差距**。

### Phase 0 原文（存证，已被上节取代）
- 新分支，新内核：AIC/AIV 两个程序、手写物理地址、128×128 块、**FP16 契约先不量化**。
- 验收：24 个用例过；对 dense FP32 参考的相对 RMSE 在 G1 闸（1%）内；逐位重复性 8 次一致。
- ★**最大风险在这一相**：手写地址没有 `TPipe` 的边界守护——而我们的竞态正是两个
  `MatmulImpl` 共用 L0 造成的。但 Laser 证明可行，且手写地址让重叠**可见**而非隐藏。
  对策：每块地址写进注释并用 `detsync_lint --share` 过一遍。

### Phase 1 — 逐通路对照
- 对 Phase 0 产物做 msOpProf，与 Laser 的 `62731 / 34611 / 59172 / 87299` 逐条比。
- 验收：**fixpipe 降到 Laser 量级**（T3+T6 生效的直接证据）。若没降，先查块大小与
  workspace dtype 是否真的生效，不要进 Phase 2。

### Phase 2 — 软件流水（T8）
- 做深度 2 错位：`QK(i) ‖ softmax(i−1) ‖ PV(i−2)`；跨核握手改成每轮一次。
- 验收：重叠度由 1.64 向 3.74 移动；窗口/下界比下降。

### Phase 3 — 对角阵矩乘（T1）
- softmax 行重缩放搬到 cube。
- 验收：`aiv_vec` 下降、`aic_cube` 上升，**净窗口下降**（搬运要以整个执行图衡量，
  不能只看单条 pipe，用户 2026-10-29 指出）。

### Phase 4 — ~~叠 INT8 QK~~ → **PV 侧量化**（F185 修正）
> ★QK **已经是 INT8**，这是接口契约（`vc_torch.cpp:38`），本相原本的目标不存在。
> 剩下可量化的只有 PV：V 在入口是 `half`，要自行量化就得占用**已经吃紧的向量单元**，
> 去省只占窗口 18.1% 的 cube。按已测数据这几乎必为负收益。下面的硬闸因此仍然成立，
> 且现在读作"大概率永远不做"。

- ★**前置条件（硬）**：Phase 1–3 之后必须确认 **cube 已成为 binding pipe**。
  若 cube 仍不是 binding，**不做本相** —— 那正是这条路线此前失败的全部原因。
- 验收：cube 忙时下降**且窗口下降**；精度按 G1/G2 闸。
- ★精度的 16 倍差距来自量化契约本身，与速度无关；本相是在"已经快了"的基础上
  用精度换更快，而不是反过来。

### Phase 5 — (e) 第三个设计点
- 此时块大小是自由变量，做一个 **M·N 乘积不同**的点，分离"tile 数"与"tile 形状"。
- 预测已按 E9 冻结于 `results/fixpipe_probe/phase1b_predictions.md`，**原样取用，不得改写**。

## 贯穿全程的纪律

- 每相的测量按 E9 先登记预测、E11 先核产物指纹（`expdesign pinned-pair`）。
- 不利结果照实写进交付文档。
- 任何下界引用必须连同三项条件（工作量分派 / tile 数 / 数值契约）一起写（P20）。
- `*_scalar_time` 含跨核等待，**不可填进 `max(各 pipe)`**（本次实测 42.7% 是 `wait_id9`）。
