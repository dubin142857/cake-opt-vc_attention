# VC-Attention / Ascend 910B2 — 交接文档

**写于** 2026-11-02 · **交接自** Claude Opus 5 会话 · **面向** 任意后继 agent（不限 Claude）

读这份文档不需要前一个会话的上下文。所有结论都给了出处与复现命令。
**凡是写"实测"的都有产物可查；凡是写"推算/未验"的都不要当事实用。**

---

## 0. 三十秒速览

| | |
|---|---|
| 做什么 | 把甲方的 VC-Attention 算子（量化注意力）在 Ascend 910B2 上做快，精度是前置硬闸 |
| 甲方交付基线 | **285.46 ms** |
| 当前 | **173.41 ms**（1.65 倍）|
| 参照 | CANN FA 139.5 ms · MindIE Laser 95.30 ms（dense RMSE 0.191%，比我们好 15 倍）|
| 当前代码 | `vc-attention` 仓 `r1-laser-dataflow` 分支 |
| **最要紧的未结** | 甲方点名的测试形状 **跑不出数**（见 §6 甲-11 / 甲-12）——交付级缺口，优先于性能 |
| **当前可用卡** | **3 · 5**；卡 4 本轮被污染，卡 0 禁用，卡 1 Alarm，卡 6 未过守卫 |

---

## 1. 目标与约束（不可协商的部分）

### 目标
来源：`handoff/problem.yaml` 的 `goals` 段。

- **primary**：在保持精度达标前提下显著提高 VC-Attention 算子性能。
- **final_acceptance**：目标视频模型固定 prompt/seed 的生成结果正常（端到端，当前阶段不做）。
- **current_stage**：先完成算子数值与性能闭环。

注意目标是"把**这个**算子做快"，不是"做一个最快的注意力"。Laser / CANN FA 在台账里是**参照点**，不是目标。

### 硬约束（违反会造成不可逆后果）

| id | 内容 |
|---|---|
| **C1** | **禁止修改 CANN 安装包源码** `/usr/local/Ascend/...`（与同事共用）。**读是允许的**，本会话多次读 CANN 源码定位问题。 |
| **C2** | 禁止重启主机 / Docker / 公共环境；不删他人镜像、容器、文件 |
| **C3** | 只用当时空闲的 NPU；每次运行前检查占用 |
| **C4** | 不把源码仓、构建产物、权重、海量 trace 复制进 PF_better |
| **C4b** | **Gitee 的 `vc-attention` 是甲方仓库，禁止推送任何改动**，包括候选分支 |
| **C6** | 任何性能或正确性结论**必须声明语义层级**（v1 / v2 / v3，见 problem.yaml 开头）|

### 用户给的长期指示

- **远程推送必须每次单独获得允许**（GitHub 也要问）
- 放宽任何闸门**必须留痕**，说明愿意与必要性
- 卡的使用不设限，可随时换卡
- **禁用 `pkill -f` / `pgrep -f`**
- 编辑源码**一律用唯一性断言的锚点**（防止误改到同形代码）

---

## 2. 代码在哪

```
/data0/z60106022/a-vc_attention_opt/
├── vc-attention/          算子仓（甲方的 fork）
│   ├── ascend/csrc/       内核源码 ← 主战场
│   │   ├── vc_kernel.cpp          AIC + AIV 两侧内核
│   │   ├── vc_tiling_host.h       host 侧 tiling 与 UB 预算
│   │   ├── vc_l1l0_map.h          L1/L0 布局（两侧唯一来源）
│   │   └── vc_cube_v2.h           cube 侧封装（fixpipe 量化等）
│   ├── ascend/vc_attention_ascend/reference.py   数值契约的 oracle
│   ├── scripts/           测量与探针（见 §4）
│   ├── results/           测量产物（已入库）
│   └── handoff/           ← 本目录
│       ├── HANDOFF.md     本文件
│       ├── problem.yaml   **问题台账（最重要，10464 行）**
│       └── arch_draft_v1.md
└── cake-toolkit/          方法论与工具仓（共用，多人在用）
    ├── core/              工具模块（见 §5）
    └── cases/vc_attention/problem.yaml   台账的**工作副本**（gitignore，机器本地）
```

### 远端

| 仓 | 远端 | 可否推 |
|---|---|---|
| vc-attention | `gh` = `github.com/dubin142857/cake-opt-vc_attention` | **要先问**，当前分支 `r1-laser-dataflow` |
| vc-attention | `origin` = `gitee.com/czf1234/vc-attention` | **绝对不推**（甲方）|
| vc-attention | `mine` = `gitee.com/dubin142857/...` | 未用，要推先问 |
| cake-toolkit | `origin` = `github.com/CAKE-math/cake-toolkit` | **要先问**；共用仓，推前必须先 merge |

> ★ `cake-toolkit/cases/` 被 gitignore，所以**台账不会随仓库同步**。
> `handoff/problem.yaml` 是本次交接时的快照。继续工作时请编辑
> `cake-toolkit/cases/vc_attention/problem.yaml`（工作副本），并在交接时重新快照。

---

## 3. 当前性能与精度（全部实测）

形状 `[1, 28, 37741, 128]` bf16，std=1，干净卡：

| 版本 | fixed core | dense RMSE (control) | 说明 |
|---|---:|---:|---|
| 甲方交付基线 h26 `e2fc535` | 285.46 ms | 2.98% | |
| 本会话起点 `334dc65` | 204.35 ms | 2.979952% | |
| `9bbe0cf` (E39) | 180.02 ms | 2.981115% | 趟 2 仍 fp32 |
| **当前 HEAD `f75f572`** (E51) | **173.41 ms** | **2.981832%** | 趟 2 控制路径 fp16 |
| CANN FA | 139.5–140.2 ms | — | |
| MindIE Laser | 95.30 ms | **0.191%** | dense fp16 |

**流水画像（msprof，当前 HEAD）**：窗口 171035 · `aiv_vec` **125570（约束流水）** · `aic_cube` 51055 · `aic_mte2` 97865 · `aiv_vec_ratio` 0.695

**算术下界**：我们 **43.3 ms**（int8 QK + fp16 PV）· Laser **57.8 ms**（全 fp16）
→ 我们的下界比 Laser **低 25%**，cube 已跑到自己下界的 **85%**。
→ 差距**全在向量**：125.6 ms 是 cube 的 2.5 倍，而 Laser 的向量藏在它的 cube 后面。
→ 若向量降到 cube 量级，窗口下界约 **60 ms**（比 Laser 还快）。**这是下界不是收益预测。**

**当前 HEAD 在干净卡上的五条确定性摘要**（变更后必须核对）：
```
46060e57fd6f4368 · 5586feb91be38621 · 71a80af6c887389e · c0ddb95adc79a3e7 · e7220e587db705aa
```

**交付闸**：141 例 `decoded_reference` < **1%**（当前最大 1.552e-03）· dense RMSE 对交付基线 2.98%
> ★ dense RMSE 已连续三跳在 2.98% 之上漂移（2.9800 → 2.9811 → 2.9824 → 2.9818）。这件事要向上反映，不该当常态。

---

## 4. 怎么跑（复制即用）

**先过守卫，再信任何数值判决**：
```bash
cd cake-toolkit
for t in vc910_card_clean vc910_card_stable vc910_card_determinism; do
  python3 -m core.devguard --card 3 --verify-artifact "$t"; done
```
三条全 rc=0 才能继续。任一未过 → **换卡**，不要在那张卡上判代码。

**构建**（改了 `ascend/` 之后）：
```bash
cd vc-attention
cd ../cake-toolkit && python3 -m core.goodstate --build-stamp vc_attention_ascend && cd ../vc-attention
SOC_VERSION=Ascend910B2 python3 -m pip install ./ascend --no-build-isolation --no-deps
cd ../cake-toolkit && python3 -m core.goodstate --build-check vc_attention_ascend
```
> ★ `setup.py build_ext --inplace` 是**静默空操作**（rc=0、零产物）。只用上面这条。
> `--build-check` 就是为了挡住"构建说成功了但产物没变"。

**确定性与摘要**：
```bash
python3 scripts/determinism_sweep.py --device 3
```

**141 例验收**：
```bash
python3 scripts/validate_ascend910b.py --device 3 --output results/validate_X.json
```

**性能（主形状，必须显式给形状——默认是 1024）**：
```bash
python3 scripts/benchmark_ascend910b.py --device 3 --backends vc_control vc_expcast \
  --queries 37741 --keys 37741 --heads 28 --dim 128 --dtype bfloat16 \
  --warmup 2 --repeats 3 --trials 3 --check-queries 32 --output results/bench_X.json
```
`--std`（2026-11-02 新增，默认 1.0）控制输入分布标准差。

**流水剖面**：
```bash
msprof op --application="python3 scripts/msprof_runner.py --device 3 --launches 3" \
  --kernel-name=vc910_forward_m64 --aic-metrics=PipeUtilization,ResourceConflictRatio \
  --output=results/msprof_X/pipe --launch-count=4
```
> ★ `msprof op` 遇到**不认识的参数照样返回 0 且零产出**（`--soc-version` / `--timeout` 是 simulator 专用）。
> **按产物数判**：`find <out> -name 'PipeUtilization_*.csv' | wc -l` 应为 4。

**两次剖面对比**（★用既有工具，不要自己写汇总脚本）：
```bash
cd cake-toolkit
python3 -m core.perf --compare <dirA>,<dirB>
```
口径已核实：`*_ratio` 的分母是 **Task Duration**（不是 `aic_time`），由 `python3 -m core.aprof --denom <dir>` 守住。

**相位分解**（AIV 逐段时间）：
```bash
python3 scripts/phase_probe_one.py --device 3
```

**离线数值契约探针**（零卡时、零构建）：
```bash
python3 scripts/probe_mass_dtype.py --device 3
```

---

## 5. 方法论基础设施（**本会话最重要的产出之一**）

问题不是"不知道该怎么做"，而是**知道了却执行不到**。本会话把三条写在文档里却从不被执行的纪律做成了**有执行体的闸**。

### `core/roundgate.py` — 开工闸（挂在 vc-attention 的 pre-commit 上）

```bash
cd cake-toolkit
python3 -m core.roundgate --check --proposed <速率|工作量|次序|仪器>   # 判红绿
python3 -m core.roundgate --brief                                      # 生成下一跳简报必带的那段
python3 -m core.roundgate --install-hook <repo>                        # 幂等安装钩子
```

**核心模型**：约束流水时间 = (元素数 × 每元素操作次数) / (速率 × 利用率)，恰好三个轴：

| 轴 | 含义 | 例 |
|---|---|---|
| **速率** | 每条指令处理多少元素 | fp32 64 / fp16 **128** / int8 256 元素每 repeat |
| **工作量** | 元素数 × 每元素操作次数 | 删掉一整遍 / 换算法 / 把活挪走 |
| **次序** | 利用率 | 调屏障 / 调缓冲 / 加重叠 |
| （仪器）| 本跳不改生产代码 | 量上界 / 修探针；不计入连跳，**也不能用来洗掉连跳** |

**判红条件**：连续 3 跳落在同一轴 · 有未结结构性待办而本跳还在磨上一轮那个轴 · 最新一跳没写 `move_class`。

**用法**：每跳在台账的冻结预测里写 `move_class:`；动内核源码提交时钩子自动跑。
`--brief` 会把未结待办**从台账生成**（不经过记忆），并自动跑 `opdbg whatfor` 提示仓里可能已有的工具。

### `core/expdesign.py` — 实验设计检查表（27 条）

```bash
python3 -m core.expdesign triggers          # 按触发点列
python3 -m core.expdesign checklist --id E42
python3 -m core.expdesign nextnum           # 下一个空闲 E 编号
```
本会话新增：**E40**（读分支只能排除"算得多"，排除不了"包装贵"）· **E42**（开工闸三轴）· **E43**（参考实现的画像差排在本实现的账目缺口之前）。

### 必须知道的既有工具（本会话吃过"重复造轮子"的亏三次）

| 想做什么 | 用它 | 不要自己写 |
|---|---|---|
| 两次 msprof 剖面对比 | `core.perf --compare A,B` | 我写过 `pipesum.py`，发现它早就在 |
| `*_ratio` 的分母是什么 | `core.aprof --denom <dir>` | F243 手工推了一遍，F253 又推了一遍 |
| 逐指令 / 流水空隙 | `core.sim`（`msprof op simulator`）| F225 写过重复模块已删 |
| "我想做 X，仓里有工具吗" | `./core/opdbg whatfor <两三个词>` | ★**动手前先问它** |
| 现象索引（挂死/零输出/精度）| `./core/opdbg symptom` | |

> ★ 本会话第三次撞同一模式：工具早在仓里，没人在需要的那一刻去搜。
> `roundgate --brief` 已经把 `whatfor` 挂到待办上，但**那只在待办里有反引号标识符时有效**。

---

## 6. 未结的事（按优先级）

跑 `python3 -m core.roundgate --brief` 取最新全文。截至交接：

### ★★ 交付级缺口（优先于一切性能工作）

**甲-11：`heads=1` 时输出非有限**
- 复现：`benchmark_ascend910b.py --queries 38139 --keys 38139 --heads 1 --dim 128 --std 1 --warmup 1 --repeats 1 --trials 1`
- 与 std 无关；**E50 之前同样失败**，是既有缺口不是本会话回归（已实测确认）
- 没验的方向：H=1 时任务切分退化（任务数 = ⌈S/64⌉×1）· `kpad`/workspace 尺寸假设 · 任务数少时的跨核旗标配对

**甲-12：`std=5` 时 decoded-reference 相对 RMSE 到 0.96（闸 1%）**
- 复现：同上但 `--heads 28 --std 5`
- **E50 之前是 0.9531，同样崩** —— 既有缺口
- 量级线索：分数幅值随 std 线性放大，QK 的 int8 对称量化很可能饱和；ExpCast 的 E4M3 值域也有限
- ★**先做离线判**（零卡时）：用 `reference.py` 的 oracle 在 std=1/2/5 上算，看是契约本身就崩还是内核与契约脱节

### ★ 路线前提

**甲-10：量化版 Laser 从来没测过** —— 它决定"量化在这颗芯片上值不值"
- 现状：源已备好（`prepare_mindie_laser_quant_source.py` 跑通，作用域与我们 v1 **同语义层级**）
- 卡在：TBE 设备核链接 `ld.lld: error: unknown emulation: aicorelinux`
- **八条假设已逐一证伪**（PATH 次序 · source setenv · 内核缓存 · cmake · 环境回归 · Python 解释器 · 陈旧构建目录 · 错误回放），详见台账 F261/F262
- 最怪的事实：在 PATH 首位放的日志包装器**一次都没被调用**，而错误照出 5 次 → 那个 `ld.lld` 不经本进程树的 PATH 解析
- 下一步（有界）：`cmake --build ... -- VERBOSE=1` 抓出失败的 opc/ccec **命令行原文**，手工单跑
- ★**若仍无解，建议改走对端**：台账开篇就记着"对端 A3/910_9382 已有稳定 Laser 结果"，通过 PF_better 要读数可能便宜一个数量级

### 性能侧

- **甲-7**：次序轴换抓手。窗口减 `aiv_vec` 还有 **32776 us** 向量空闲未定因（屏障已排除，34 道全去掉只值 0.42%）。先做**多次采样取分布**，不是消融——AIV 跨核等待在两次采样间从 23.4% 跳到 0.5%，单次读数不能下判。
- **甲-9**：把 oracle 对 fp16 趟 2 建准，恢复 `decoded_reference` 的回归检出能力（现 1.55e-3，E50 前 1.04e-4，差 15 倍）。**不许用放宽判据的办法"恢复"**——那是把探测器关掉。

---

## 7. 踩过的坑（省下重复踩的时间）

| 坑 | 症状 | 真相 |
|---|---|---|
| `setup.py build_ext --inplace` | rc=0，一行日志，零产物 | 包的 cmake 挂在 `build_py` 上。用 `pip install ./ascend`，并用 `goodstate --build-check` 守 |
| `msprof op` 参数错 | **返回 0** 且零产出 | 按产物数判。`--soc-version`/`--timeout` 是 simulator 专用 |
| `*_ratio` 的分母 | 以为是 `aic_time` | 是 **Task Duration**。`aprof --denom` 早就实现并守住了 |
| `Cast` 的 dst/src 重叠 | 结果错且不报错 | 实撞两次。`workh` 与 `work` 是同一段字节 |
| 缓冲区尺寸**三处镜像** | 漏一处则输出直接为零 | `vc_kernel.cpp` 的 `InitBuffer` · `vc_tiling_host.h` 的 UB 预算 · `static_assert(VectorBytes(0,64,4)==161920)` |
| 泛型 `ReduceMax/ReduceSum` | 以为只是包装 | `isReuseSource=false` 时**先整块暂存拷贝一遍**。换手写省了 5.99 ms |
| 自己写的探针判"坏" | 以为内核坏了 | 本会话三次是探针坏了。**先怀疑探针**（E23 反方向）|
| `rm -rf` 构建目录 | 审计拒绝 | 那是 git 跟踪的。删之前先看跟踪状态 |
| 推算的上界 | 当成上界用 | **推算的上界不是上界，是又一条待验的预测**。实撞：推 −33% 实测 −15.51%；模型漏算 fp16 相消 |
| 判据选错量 | 照它回退会丢掉好实现 | `decoded_reference` 是**建模保真度**，`dense_fp32` 才是**输出质量**。两者混了会误判 |

---

## 8. 工作方式（强烈建议沿用）

每一跳：

1. **跑开工闸**：`roundgate --check --proposed <轴>` + `--brief`，把 `--brief` 那段原样带进简报
2. **先读接口再动手**：以本机头文件为准（`/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc/impl/...`）
3. **写冻结预测**（台账 `kind: prediction`，含 `move_class:`）：
   - **测量之前写定，原文不改**
   - **每一档都配区间**，包括中间档
   - 写清"每种结果各自的含义"
4. **消融纪律**（F251 做对了的那套）：
   先关着构建核摘要 → 取**同一次构建**的 OFF 对照 → 开 → 测 → **立刻退回重建**核摘要
5. **等价性先判再看时间**：摘要不变才说明读数可用于归因
6. **落台账**（`kind: result`）：命中哪一档、哪条预测被证伪、过程中的错误也记
7. **沉淀**：撞墙的位置就是 infra 的缺口；能工具化就不要只写文档

台账的编号体系：`E<n>` = 冻结预测，`F<n>` = 结果/分析。订正**追记**不改原文（见 F246/F248/F257 的做法）。

---

## 9. 本会话做了什么（一句话索引，细节全在台账）

| 跳 | 做了什么 | 结果 |
|---|---|---|
| E38/F245 | 反量化拆成 fixpipe(ks) + AIV 的 Axpy(qs) | 190.32 → 182.38 ms |
| E39/F247 | 趟 1 行最大值改手写 `WholeReduceMax<half>` | 182.38 → 180.02 ms，逐位等价 |
| E41/F249 | 相位分解（并修了探针的还原假设）| `t2_vec` 占 AIV 自身时间 65.7% |
| — /F253 | 解 Laser 的"超峰值"矛盾 | 分母是 Task Duration；Laser cube 效率订正为 92% |
| E44/F251 | fp16 趟 2 的消融上界 | `aiv_vec` −15.51%（分母也 fp16，不可实现）|
| E45/F252 | mask 形调用的单价 | 19 周期/调用，五条占 `aiv_vec` 7.6% |
| E47/F254 | 保 fp32 分母的 fp16 趟 2 | 只 −5.82%，低于门槛 |
| E49/F257 | 补量 fp16 `exp` 的数值代价 | +0.000610 pp（分母那项只 −0.00001）|
| E50/F258 | 趟 2 控制路径搬 fp16（实现）| 180.02 → 174.33 ms，但健康探针被致盲 |
| E51/F259 | 行最大值改未乘 qs 的域 | **173.41 ms**，探针恢复 rc=0 |
| E48/F256 | 次序轴消融上界 | 34 道屏障只值 0.42% |
| E52/F261-262 | 量化版 Laser | 源已备好，构建卡在链接，八条假设证伪 |
| — /F263 | 甲方测试形状 | **两个既有缺口**，确认非本会话回归 |

**方法论侧**：`roundgate` 从无到有并挂上 pre-commit · `expdesign` 新增 E40/E42/E43 · `loopguard` 新增「开工闸」常设条款 · `phase_probe_one.py` 改为恒等式分解 · `probe_mass_dtype.py` 新建 · `benchmark_ascend910b.py` 加 `--std`。

---

## 10. 如果你只能做一件事

**做甲-11 或甲-12**。甲方点名的测试形状 `[1, head_num, 38139, 128] std=5` 现在跑不出任何数 —— 两个独立的既有缺口把它挡住了。

性能从 285 做到 173 很好，但**算子在客户要测的形状上不工作**，这比再快 3% 重要得多。

而且这两条都**不是**最近改动造成的（已实测排除），所以不要先去翻 git log 找回归——那个方向走过了。
