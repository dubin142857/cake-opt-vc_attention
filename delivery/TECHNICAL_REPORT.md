# 技术说明 — Ascend 910B2 数据流优化：五处改动（V-Smooth 关闭路径）

本分支以甲方 `master` 的 `9e3a550d399c560cc7c7d9e25cbbc4157e82c0d5` 为基线，叠加五处
改动，长形状固定核心用时下降 **70.0%**。四处都只动数据流与同步时序或数值表示的粒度，
不改算子的对外契约。

## 语义层级（必读）

本分支实现的是甲方 910B 端口的**现有语义层级**，即 `ascend/vc_attention_ascend/api.py`
自己声明的那一级：

```python
if causal or v_smooth:
    raise NotImplementedError("This 910B port is dense noncausal, without V clustering/smoothing")
```

对照仓库自身的 `PAPER_ALIGNMENT.md`（对 arxiv 2609.15810 公式 4/6/7 与附录 B 的逐条记录），
该端口实现的是论文的一个子集：INT8 QK（含 K 每通道去均值）+ ExpCast + 每通道整序列单 scale
的 E4M3 V。论文中负责精度的两大机制——Q/K 共同归一化 Hadamard 旋转、V-Smooth 全套
（聚类、K/V 同序排列、128-token 块内去均值、FP16 `mu/V_scale` 在线恢复）——不在该端口内。

**因此本分支的 70.0% 是「该子集算子」的数据流收益，不等于「VC-Attention 快了 70%」。**
这五处改动都不碰数值路径或只改表示粒度，预期可平移到含 V-Smooth 的版本，但那需要重新实测。

## 五处改动

### 1. 推迟 `O_USED` 等待（`vc_kernel.cpp`，18 行）

原协议每个 `(task, key)` tile 是严格乒乓：`wait O_USED` 放在下一次 **QK 之前**，于是 AIC 的
下一次 QK 必须等 AIV 读完 `og` 并累加完毕。改为用一个布尔量跨 tile 与跨 task 携带这次待决
的等待，在**下一次 PV 之前**再 `Wait`，两层循环结束后补一次 drain。

`og` 的保护时点不变，仍在覆写它的 PV 之前；每个 `Set(O_READY)` 仍对应恰好一次 `Wait(O_USED)`，
标志计数守恒。

msprof 机理复核：`aic_scalar_wait_id15`（AIC 等 O_USED）284235 → **1.36 us**，被消除；
`aic_cube_time` 53020.79 两轮完全相同。

### 2. `Brcb` 广播替代标量回读（`vc_kernel.cpp`）

两处原本「把 ROWS 个行值逐个读回标量寄存器、再发 ROWS 次短向量调用」的写法，改为一次
`Brcb` 把每行的值展开成完整的 32 字节块，再用 `src1BlkStride=0` 的 `BinaryRepeatParams`
让一次调用覆盖所有行：

```cpp
Brcb(bmax, rowmax, ROWS / 8, BrcbRepeatParams(1, 8));
for (uint32_t off = 0; off < N; off += 64)
    Sub(score[off], score[off], bmax, uint64_t(64), ROWS, rowBrc);
```

`x + (-m)` 与 `x - m` 在所有有限值与两个带符号零上逐位相同，数值契约不变；随之消失的是
`V_S` 的往返。141 个验收用例与改动前**逐条比对，0 个数值项有差异**。

### 3. matmul tile 由 128 加宽到 256（`vc_tiling.h` 等）

`N` 由 128 改为 256，同时把 K 的**量化块**从 tile 宽度中解耦出来（新增 `K_SCALE_BLOCK = 128`
与 `KSB_PER_TILE`），使 K 仍按 128 量化而张量按 256 填充。scale 改为按列分段施加：

```cpp
const UnaryRepeatParams rp(1, 1, uint8_t(N / 8), uint8_t(N / 8));
for (uint32_t sb = 0; sb < KSB_PER_TILE; ++sb)
    for (uint32_t off = 0; off < K_SCALE_BLOCK; off += 64)
        Muls(score[sb * K_SCALE_BLOCK + off], score[sb * K_SCALE_BLOCK + off], seg_scale[sb], 64, ROWS, rp);
```

收益来自 `O` 的 GM 往返量与 `N` 成反比（`S` 与 `P` 的总量与 tile 尺寸无关）。

★**ExpCast 的数值契约与 tile 宽度绑定**：E4M3 编码是相对**当前 tile 的行最大值**定义的
（低于 `-120/EXPCAST_GAIN` 的分数一律压成索引 0），换一个宽度就是换一个契约。因此
`reference.py` 的 `payload_reference` 增加了 `tile_width` 参数并默认取 `K_TILE`，必须与
`vc_tiling.h` 的 `N` 一致；普通 `exp` 路径对宽度不变。实测该契约变化是**质量中性**的：
12 个可判别用例相对 dense 的误差变化在 −1.29% 至 +1.44% 之间，8/12 改善。

一处无符号下溢的加固：当 K 的填充粒度宽于 N 时，`key` 可能整块越过真实 key 数，
`t.keys - key` 会下溢成巨大值而绕过尾块掩码，故先判：

```cpp
const uint32_t remain = key >= t.keys ? 0u : t.keys - key;
const uint32_t valid = remain < N ? remain : N;
```

### 4. S 双缓冲（`vc_kernel.cpp`、`vc_tiling_host.h`）

AIC 在等本 tile 的 `P_READY` 之前就把下一 tile 的 QK 写出去，故 `S` 需要两个槽；`P` 与 `O`
的单槽仍安全（理由写在 `WorkspaceBytes` 的注释里）。AIC 侧改为**移位循环**——第 `i` 轮先发
QK(i)，再消费第 `i-1` 块：

```cpp
const uint32_t tiles = t.kpad / N;
for (uint32_t i = 0; i <= tiles; ++i) {
    if (i < tiles) { /* QK into sgs[i & 1]; set S_READY */ }
    if (i == 0) continue;
    CrossCoreWaitFlag(P_READY);
    /* 推迟的 O_USED; PV; set O_READY */
}
```

★设备侧**不能用 lambda 包装 QK 发射**：`auto issueQk = [&](...){ mmq.SetOrgShape(...); }` 会让
`__aicore__` 上下文里的设备重载对 lambda 体不可见，编译报 `SetOrgShape` / `IterateAll` /
`CrossCoreSetFlag` 全部 "no matching function"。移位后 QK 与 PV 的代码各只出现一次，无需复制。

msprof 机理复核：`aiv_scalar_wait_id8`（AIV 等 S_READY）69781.64 → **1.67 us**，被消除；
各条真实 pipe 的工作量未动（`aic_cube_time` 50766.88 逐位相同）。

### 5. AIV 推迟累加（`vc_kernel.cpp`、`vc_tiling_host.h`）

改动 1–4 之后，msprof 显示 AIV 有 54.2% 的时间空转，其主导等待是 `aiv_scalar_wait_id10`
（等 AIC 的 PV 输出），占 AIV 窗口的 85.1%。本改动让 AIV 在那段等待里去算**下一块**的概率：
第 `j` 轮先算第 `j` 块并发布 `P_READY`，再消费第 `j-1` 块的 PV。

**这一步此前不可行，是改动 4 使其成为可能**——AIC 现在预发 QK，故 `S(j)` 在 AIV
处理第 `j-1` 块期间就已就绪。

三处必须同时成立的条件：

```
P 改双槽   AIV 写 p(j) 时 AIC 的 PV(j-1) 可能仍在读 pg。安全性来自既有的握手顺序:
           AIV 覆写 pgs[j&1] 发生在第 j+2 轮, 而第 j+1 轮已等过 O_READY(j),
           该标志在 PV(j) 之后置位, 故覆写时 PV(j) 必已读完。
acc 推迟   acc 的 rescale 要等拿到对应的 pv 才能施加, 故 alpha 按 tile 存两槽;
           l 与 m 只依赖 rowmax 与 mass, 与 pv 无关, 仍立即更新。
补一条序   raw 身兼两职 —— MTE2 接收 S, 以及存 ExpCast 的 gather 字节偏移供向量
           单元读。两者在一个 tile 内生命周期并不相交 (S 在 Cast 之后即死, gather
           偏移到 ExpCast 段才出生), 冲突只发生在**跨轮**: 移位循环把上一轮的
           Gather 与本轮的 DataCopy(raw, ...) 拉近, 中间缺一条 V->MTE2 的序,
           在 expcast 且 tiles>=2 时触发 aicore exception
           (plog: fftsplus aivector error)。补上 HardEvent::V_MTE2 的 Set/Wait 对即解除。
           ★我们先用一块 ROWS*N*4 的独立缓冲绕开过 —— 那是拿 UB 换一条同步, 已改回。
           M64 的 VectorBytes 因此为 119040 B 而非 151808 B。
```

★**在线 softmax 的不变式先于编码在 CPU 上核对过**：推迟累加与即时累加**逐位相同**
（rRMSE 8.9e-16，浮点噪声量级），因为每一项的浮点运算顺序未变，只是施加时刻推迟。
实机输出与改动前的哈希逐位相同，与该预测一致。

## 测量

长形状 `[1, 37741, 28, 128]` BF16，固定 payload，`prepare()` 在计时之外，`tile_m=64`，
warmup 2 / 每次 5 遍取中位，整包交替配对 6 轮，每轮内两版本各在新进程跑一次，
每次记录所用 `.so` 的 sha256 并核对唯一性。

每一轮都把**基线**作为一个臂一起测，所以下表的「对基线」是**同轮直接配对**的结果，
不是把各级百分比累乘出来的。各级的绝对毫秒值来自不同轮次，轮间有漂移，故只给配对百分比。

| 级别 | commit | 对前一级（逐对中位） | 对基线（同轮直接配对） | 方向一致性 |
| --- | --- | ---: | ---: | --- |
| h6a 推迟 O_USED | `68ca21d` | −26.48% | −26.48% | 6/6 为负 |
| h8 Brcb 广播 | `69dfebd` | −16.28% | −38.44% | 6/6 为负 |
| h9 tile 256 | `4569c67` | −23.68% | −53.1% | 5/5 为负 |
| h10 S 双缓冲 | `485fb2f` | −12.49% | −59.03% | 6/6 为负 |
| h11 AIV 推迟累加 | 本分支顶端 | **−28.98%** | **−70.0%** | 8/8 为负 |

`expcast_m64` 口径：h8 对基线 −34.84%，h9 对基线 −48.10%，h10 对基线 −53.29%，
h11 对基线 **−71.2%**，均为全负。

绝对值参考（AB6 一轮内同时测的三个臂，产物哈希各自唯一
`505ad07d96c37c85` / `8785f746365ed1a6` / `47cd6f0eef6860e3`）：
`control_m64` 基线 1103.51 → h9 517.42 → h10 452.43 → h11 **331.37 ms**；
`expcast_m64` 基线 1230.62 → h9 634.35 → h10 574.59 → h11 **354.45 ms**。
轮间极差 0.57% / 0.69% / 0.85%，效应量约为噪声的 15 倍。

★**测量口径的已知局限**，建议贵方复验时注意：上表用的是 ABAB 交替臂序。我们随后用
**同一个产物挂两个标签**跑同一协议（真效应恒为 0）测出两个伪影——臂序偏置（后跑的一臂
系统性快约 0.16%）与两臂反相的周期 2 振荡（使效应估计随丢弃轮数摆动约 1.9 个百分点）。
上表每一级的效应都在 12%–27%，远高于这两个约 2% 的伪影，结论不受影响；但若要判定
小于 5% 的效应，应改用 ABBA 平衡臂序并先用空载跑标定分辨率下限。

## 正确性

- `scripts/validate_ascend910b.py`：141 个用例 `status=passed`，0 个超 1% 闸门。
- `scripts/profile_ascend910b.py` 长形状闸门：`accuracy_status=passed`。
  `decoded_reference` 相对 RMSE：control 0.000068 / expcast 0.000043。
  `dense_fp32` 相对 RMSE：control 0.029799 / expcast 0.043645。
- 改动 1、2、4、5 对各自前一级 **141 用例逐位一致**；改动 3 改变 ExpCast 的 tile 宽度契约，
  为质量中性（见上）。
- 改动 5 的四个用例输出哈希与改动 4 **完全相同**，并通过 12 个形状组合
  （`d` 64/128 × `nk` 129/257/513 × `tile_m` 32/64）。
- `m64_vs_m32_all_rows` 两路 `exact_match` 均为 true。
- 逐位重复性：6 组形状各连发 **8 次**互比，全部 8/8 相同。
  （不用 2 次——2 次正是漏掉同类竞态的原因。）

## 复现

```bash
git checkout <本分支>
source /usr/local/Ascend/ascend-toolkit/set_env.sh
NPU_DEVICE=<空闲卡> bash scripts/run_ascend910b.sh --long-shape --profile -- --warmup 2 --repeats 3 --trials 5
```

两条实撞过的坑，建议照做：

1. **不要直接 `pip install ./ascend`**——缺 `SOC_VERSION` 会以
   `RuntimeError: Set exact SOC_VERSION from preflight` 失败。走 `run_ascend910b.sh`，
   它从 preflight 的 `compiler_soc` 导出该变量。
2. **构建后先核对产物 sha256 再进入验收**。我们两次被误导过：一次是包装脚本里
   `pip install ... | tail -5` 后取 `$?`（拿到的是 `tail` 的退出码，构建失败仍报 0），
   一次是改动未提交、`git checkout` 把工作区改动带到了基线分支上。两次都靠哈希核对
   才发现，否则会拿旧二进制跑出一份看起来正常的验收。

## 本分支**不含**什么

- **V-Smooth**。见上「语义层级」。
- **已实现但未通过长形状闸门的两个候选**，故不在此分支：
  - QK 经 fixpipe 反量化直出 fp16（S 的回写字节减半）：141 用例过，长形状
    `decoded_reference` expcast 达 1.1089% 超 1% 闸门。对真实 dense 的质量未退
    （expcast 4.3627% 对 h10 的 4.3645%），劣化的是与同契约 oracle 的吻合度，
    原因尚未查明（已排除「舍入次数」这一解释）。
  - AIV 推迟累加（等 PV 的时间用来算下一块概率）：承自上一项，故同样未过。

## 环境

CANN 9.0.0，`Ascend910B2`（`npu-smi` 25.5.1），torch 2.9.0 + torch_npu 2.9.0，
Python 3.10，aarch64。贵方若在 910C/A3 上复验，跨核标志的实现与调度不保证一致，
请独立实测，不要据本侧数字推断加速比。
