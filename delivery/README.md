# 交付说明 — Ascend 910B2 数据流优化

> **本目录是唯一入口。** 仓库顶层有甲方自己的 40 多个文档，本次交付的全部内容都在这里。
> 要跑起来只看本文的「启动」一节；要复核数字看「验证手段」一节。

| 文件 | 内容 |
| --- | --- |
| **`README.md`**（本文） | 当前进展、怎么启动、怎么验证、口径边界 |
| **`PERF_COMPARISON.md`** | **性能对照**：相对贵方基线的收益，以及与官方 MindIE Laser Attention 的同机直接对测 |
| **`TECHNICAL_REPORT.md`** | 九级改动的机理、证据、正确性 |
| **`REPRODUCE.md`** | 从基线逐级复现的完整步骤 |
| `OPTIONAL_FP16_S.md` | **可选级存档**：QK 直出 fp16。它在本验收判据下不通过，不在交付内 |

## 当前进展

交付分支 **`h26-bitdecode`**。**内核改动止于 `53ec7af`**，产物哈希对应的就是它；
分支顶端在其之上只有交付文档与复现脚本（不碰 `ascend/`，不影响产物）。
在甲方基线 `9e3a550d399c560cc7c7d9e25cbbc4157e82c0d5` 上叠加**九级**数据流、同步时序与
向量指令的改动，外加**一处正确性修复**（第十项，见下）。
长形状固定核心用时下降 **74.4%**（`control`）/ **75.8%**（`expcast`）。

> ### ★先读这一条：2026-10-28 发现并修复了一个竞态
>
> 此前交付的产物 `4d20b857d25f3541` 在 **`tile_m=32`**（即 `run_prepared()` 与
> `attention()` 的**默认值**）长形状上**不确定**：同一输入连调 24 次，23 次结果不同。
> 512×4 小形状上偶发 aicore 507015，`plog` 的 `errorStr` 为
> `L0B read/write conflict in the MTE (same address)`。
>
> **成因**：同一 AIC 上的两个 `MatmulImpl`（`mmq` 算 QK、`mmv` 算 PV）共用物理
> L0A/L0B，并共用 `M_MTE1` 事件号。这是**最初的 Ascend 移植（`da6026b`）就带的结构**，
> 两个实例的声明此后从未被改动过；本次九级优化把 AIV 侧提速，才把触发率从
> 9%（h15）推到 100%（h26b）。**这意味着 h11 时期交付的版本大概率同样带这个缺陷。**
>
> **修法**：在两个 matmul 的发射点之间串行一次。`tile_m=64` 输出与修复前**逐位相同**，
> 即只改时序不改值；代价 `tile_m=64` 上 +1.14%(control)/+0.41%(expcast)，
> `tile_m=32` 上无代价。
>
> **本文「收益」一节的数字全部在 `tile_m=64` 上测**，不受此缺陷影响；
> 受影响的是按默认 `tile_m=32` 调用的那条路径。

构建产物应为：

```
3494c48d2ce2b73ed7292dac3e26ce6e6d084fa83f77c2b777cae42ddf04cf55  libvc_ascend_ops.so
```

★**构建后先核对这个 sha256 再进入验收。** 我们两次被误导过：一次是 `pip install ... | tail -5`
后取 `$?`（拿到的是 `tail` 的退出码，构建失败仍报 0），一次是改动未提交、`git checkout`
把工作区改动带到了另一分支上。两次都靠哈希核对才发现，否则会拿旧二进制跑出一份看起来正常的验收。

## 启动

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
NPU_DEVICE=<空闲卡号> bash scripts/run_ascend910b.sh --long-shape
```

这一条命令依次做：preflight 环境检查 → 导出 `SOC_VERSION` → 构建 → 141 用例正确性验收 → 长形状基准。

三个要点：

- **不要直接 `pip install ./ascend`** —— 缺 `SOC_VERSION` 会以
  `RuntimeError: Set exact SOC_VERSION from preflight` 失败。上面的脚本会从 preflight 的
  `compiler_soc` 导出它（不同 SKU 的值不同，继承来的值可能指向另一颗芯片，所以以 preflight 为准）。
- **`--long-shape` 才是交付形状** `[1, 37741, 28, 128]`；不加是 1024×1024×2×128 的小形状，
  两者的数字不可互相引用。
- 卡号用 `NPU_DEVICE` 指定（默认 0）。

其它开关：`--skip-build` 跳过构建，`--profile` 改跑长形状的精度与分段画像。

## 收益

单一形状 `[B=1, N=37741, H=28, D=128]` BF16，自注意力，`tile_m=64`，
只计 `run_prepared()` 的原生核心调用（`prepare()` 在计时之外）：

| | `control_m64` | `expcast_m64` |
| --- | ---: | ---: |
| 基线 `9e3a550` | 1102.63 ms | 1234.05 ms |
| **本交付（内核 `0979605`）** | **283.38 ms（−74.36%）** | **298.23 ms（−75.79%）** |

两个独立测量会话给出 −74.30% / −74.42%（`control`）与 −75.83% / −75.75%（`expcast`），
跨会话极差 0.12% / 0.08%。基线为同次重新构建后实测，不是引用历史数字。

同机 CANN `npu_fusion_attention` 为 142 ms；差距由基线的 7.8 倍收窄到 **2.0 倍**。
★这是两个不同数值契约的算子，该比值只作背景参照，**不是加速比**。

### 与官方 MindIE Laser Attention 的对比（2026-10-28 实测）

甲方要求的对比已完成。**原版 Laser 从上游固定提交 `71098c20` 的源码在本机编译**
（不需华为账号；`mindiesd` 安装包仍需账号，但本对比不依赖它）。
同一台机器、同一张卡、同一形状 `[1, 28, 37741, 128]`，两者在数分钟内先后测得：

| | 核心耗时 | `dense_fp32` 相对 RMSE |
| --- | ---: | ---: |
| **原版 Laser**（不量化） | **95.30 ms** | **0.191 %** |
| 同机 CANN 融合算子 | — | 0.234 % |
| 本交付 `control`（`tile_m=64`） | 285.46 ms | 2.98 % |
| 本交付 `expcast`（`tile_m=64`） | 299.89 ms | 4.36 % |

**结论：在这台机器、这个形状上，原版 Laser 比本交付快 3.0 倍，且 `dense_fp32`
相对 RMSE 小约 16 倍。** 两项必须同时看：只报耗时比会误导，因为两条路线的
数值契约不同——Laser 走 FP16 不量化，本交付走 INT8 QK + E4M3 V 的量化路线。

口径边界：

- 两侧的精度是**同一个字段名、同一种口径**（对 FP32 稠密参考的 `relative_rmse`），
  单位已逐项核对（Laser 报表里的 `dense RMSE %` 即 json 中 `relative_rmse × 100`）。
- Laser 的精度采样覆盖至多 32 个 query 行（其脚本自述），本交付为长形状全量；
  但两者相差约一个量级以上，远超采样差异所能解释。
- 计时协议不同（Laser 为进程分组同步墙钟，本交付为 `bench_one.py` 的重复中位），
  两者都排除预处理。3.0 倍的差距同样远超协议差异。
- 未跑 `qkv_int8_e4m3_expcast` 等变体——那回答的是"把本路线搬进 Laser 框架会怎样"，
  与"本交付 vs Laser"不是同一个问题。

复现（`vendors` 隔离见下）：

```bash
python3 scripts/prepare_mindie_laser_source.py --reference-dir <含该提交的 MindIE-SD 仓库> --variant baseline
python3 scripts/build_mindie_laser_local.py --cann-root /usr/local/Ascend/cann-9.0.0 --soc ascend910b --jobs 1
python3 scripts/build_mindie_laser_bridge.py --source-dir build/mindie_laser_source --cann-root /usr/local/Ascend/cann-9.0.0 --jobs 16
ASCEND_OPP_PATH=<隔离的 opp> ASCEND_RT_VISIBLE_DEVICES=<非 0 卡> \
  python3 scripts/benchmark_mindie_laser_quant.py --variants baseline \
    --baseline-build results/mindie_laser_build.json \
    --bridge-manifest results/mindie_laser_bridge.json \
    --device 0 --warmup 2 --repeats 5 --trials 3
```

★**构建需要一次绕行**（2026-10-28 已用全新清理构建的受控对照查实）：
无论 `--jobs 16` 还是 `--jobs 1`，从干净状态起都会在 FATBIN 阶段报
`ld.lld: error: unknown emulation: aicorelinux`（两臂 `build_rc` 均为 1）。
**`--jobs 1` 并不足够**——此前文档里写的"串行可过"是错的，那次成功是复用了
手动编出的中间产物，现已更正。

已排除：并行、环境变量（把构建脚本算出的完整 44 个变量原样喂给生成的编译脚本，
手动跑 `rc=0` 且零错误）、ccache（`ENABLE_CCACHE=OFF`，`CMakeCache` 里
`COMPILER_LAUNCHER` 为空）、链接器本身（CANN 的两个 `ld.lld` 是同一个二进制，
实测接受 `-m aicorelinux`）。**在 make 下究竟差什么，尚未查出。**

可复现的绕行路径：

```bash
# 1) 先跑一次构建：它会在 configure 阶段生成 gen 脚本，随后在 FATBIN 阶段失败
python3 scripts/build_mindie_laser_local.py --cann-root /usr/local/Ascend/cann-9.0.0 --soc ascend910b --jobs 16
# 2) 用构建脚本算出的完整环境，手动跑那几条生成脚本（实测各 rc=0）
cd build/mindie_laser_source/build/build/binary/ascend910b/gen
for f in VcLaserAttention-vc_laser_attention-*.sh; do
    ASCEND_CUSTOM_OPP_PATH=$(cd ../../../custom && pwd) HI_PYTHON=python3 \
    TILINGKEY_PAR_COMPILE=1 BIN_FILENAME_HASHED=1 \
    bash "$f" ../src/vc_laser_attention/VcLaserAttention.py ../bin/vc_laser_attention
done
# 3) 再跑一次整体构建 —— 实测 rc=0，产物与 manifest 齐全
cd - && python3 scripts/build_mindie_laser_local.py --cann-root /usr/local/Ascend/cann-9.0.0 --soc ascend910b --jobs 16
```

★这是**绕行，不是修复**。

★`ASCEND_OPP_PATH`：该基准脚本对"进程里加载了非本次 vendor 的库"是 **fail closed** 的。
共用机器上装着他人的 vendor 时会直接拒跑。做法是用符号链接拼一棵隔离的 `opp`
（`built-in` 等软链回原树，只把 `vendors/` 换成真目录并将 `load_priority` 置空），
**不修改共用的 CANN 安装包**，也不降低该闸的标准。

## 验证手段

三件事各有各的脚本，**都在仓里，不需要另找**：

### 1. 正确性

```bash
python3 scripts/validate_ascend910b.py --device <卡号>
#  期望 status=passed，141 个用例，0 个超 1% 闸门

python3 scripts/profile_ascend910b.py --device <卡号> --warmup 2 --repeats 2 --trials 2
#  期望 accuracy_status=passed
#    decoded_reference  control 0.000068 / expcast 0.000043
#    dense_fp32         control 0.029799 / expcast 0.043645
```

★第 5 级之后的每一级对前一级都**逐位一致**，所以上面这组精度数字自第 5 级起**一字未变**。

### 2. 性能（复现上面那张收益表）

```bash
NPU_DEVICE=<空闲卡号> bash scripts/bench_pairwise.sh 9e3a550 h26-bitdecode
#  输出 results/bench_9e3a550_vs_h26-bitdecode/{session*.jsonl, summary.txt}
```

它会把两个 ref 各构建一次、快照**整个安装目录**、按 ABBA 臂序逐趟换包重测，
跑两个会话并判跨会话是否复现。任意两级之间都可以这样比，例如
`bash scripts/bench_pairwise.sh 9256a40 0979605` 复核最后一级。

脚本里的四条协议各挡一类我们实撞过的错，`scripts/bench_pairwise.sh` 的头部注释写明了是哪四类。
其中最要紧的一条：**必须整包切换而不是只换 `.so`** —— 不同级别的 Python 侧
（`reference.py` 的 `K_TILE`）可能不同，只换 `.so` 会让内核与 payload 的填充宽度对不上。

### 3. 逐位重复性

★**此前本节写的是「6 组形状各连发 8 次，全部 8/8 相同」。该结论当时不成立** ——
那 6 组形状都短，而竞态只在长形状 + `tile_m=32` 上暴露。连发次数从 2 次提到 8 次
不足以补救：**决定能否测到的不是次数，而是有没有跑在最高暴露的那个配置上。**

正确的判法是先找出最高暴露配置，再在那个配置上复测：

```bash
cd <cake-toolkit>
ASCEND_RT_VISIBLE_DEVICES=<卡号> python3 -m core.detrace rate \
    --repro cases/vc_attention/repro.py \
    --knobs "tile_m=32,64;expcast=0,1" --repeats 24 --tile 32 --lane-axis 1
#  期望: 四格触发率全为 0.0%
#  它同时打印损坏粒度与「触发率随旋钮的单调律」—— 修复前在
#  tile_m=32/expcast=0 这一格是 100%，另外三格是 0%。
```

短形状的那一组仍可跑，作为补充而非主判据：

```bash
python3 scripts/profile_ascend910b.py --device <卡号> --trials 8
```

## 九级改动 + 一处正确性修复

| # | 改动 | commit |
| --- | --- | --- |
| 1 | 推迟 `O_USED` 等待，让下一 tile 的 QK 与 AIV 尾段重叠 | `68ca21d` |
| 2 | `Brcb` 广播替代「标量回读 + ROWS 次短向量调用」 | `69dfebd` |
| 3 | matmul tile 128→256，并把 K 的量化块与 tile 宽度解耦 | `36bad76` |
| 4 | S 双缓冲，AIC 在等 `P_READY` 之前预发下一块 QK | `8608b21`、`e4156d5`、`593ae08` |
| 5 | AIV 推迟累加，等 PV 的时间用来算下一块的概率 | `3198686` |
| 6 | O 双缓冲 —— 去掉 AIC 等 `O_USED` 的结构性原因 | `539ccec` |
| 7 | V 预装进调用方自管的 L1（TSCM/NZ），并移到 `S_READY` 置位之后 | `b70354f`、`8568973` |
| 8 | 删掉 ExpCast 链里**恒不生效**的上界夹持 | `9256a40` |
| 9 | 用位运算直接构造 E4M3 的解码值，替掉查表 `Gather` | `0979605` |
| 10 | **正确性修复**：在 `mmq` 与 `mmv` 之间串行，去掉两个 Matmul 实例对共用 L0A/L0B 的重叠访问 | `53ec7af` |

机理与逐级证据见 `TECHNICAL_REPORT.md`。表中每个 commit 都在 `h26-bitdecode` 的祖先链上，
`git checkout <commit>` 可直接取到那一级。

## 四条口径边界（请勿略过）

1. **这是一个形状的结果，不是形状扫描。** 我们没有做过性能的形状扫描。
   仓库里那 141 个用例是**正确性**套件，不测时间。
2. **是核心调用时间，不是端到端。** `prepare()` 的预处理被排除在外，
   完整 API 的收益会被预处理摊薄。
3. **语义层级是现有端口的 v1 子集。** `ascend/vc_attention_ascend/api.py` 自己声明
   `causal` 与 `v_smooth` 未实现；对照仓库自身的 `PAPER_ALIGNMENT.md`，论文中负责精度的
   两大机制（Q/K 共同归一化 Hadamard 旋转、V-Smooth 全套）不在该端口内。
   **因此 74% 是「该子集算子」的收益，不等于「VC-Attention 快了 74%」。**
4. **第 8、9 级只作用在 `expcast` 路径上。** 它们改的是 ExpCast 的指令序，
   `control` 路径不走那条链，故第 6 级之后 `control` 的变化都在测量分辨率以下。

## 还剩多少

按真机 msprof 的逐通路分解，当前窗口 2.93 us/tile（`expcast`），其中最大的单条工作通路是
AIV 向量 2.03 us/tile。按「下界只能取 max(单项 pipe)」这条规则，
**任何跨核协议的改动都降不到 2.03 us/tile 以下**，即 `expcast` 臂约 207 ms
（`control` 臂的同类下界约 136 ms，其 AIV 向量为 1.34 us/tile）。

★**这个 207 ms 不是"本设计的硬下界"。** 本文件此前一版这样写过，是**过度推广**——
把"某一类改动（跨核协议）的下界"写成了"整个设计的下界"，现已更正。它成立的前提是
**当前的 AIV 指令条数**，而那不是硬件地板：

- **条数已经被压过一次。** 曾实测认定 `aiv_vec` 2.4481 us/tile 中逐元素指令的*单价*
  已接近向量单元吞吐上限（57/64 元素每周期）、省不出来；随后第九级改动用位运算替掉
  查表 `Gather`，`aiv_vec` 降到 2.03（**−17%，且未动跨核协议**）。余下构成已知：
  约 1.5 us/tile 曾在 `Gather` + 两次 `Reduce` + 两次 `Cast` 上，`Gather` 已去，其余仍在。
- **存在性证明指向相反方向。** 同机、同形状、同样 24 AIC / 48 AIV，CANN 的
  `FlashAttentionScore` 做**完整**注意力只要 **142.08 ms**，而且它的 cube 忙时
  82.2 ms 比我们的 50.8 ms **还多 38%** —— 做更多算术却快一倍。差距在重叠度：
  FA 的 AIC 重叠度 2.46，我们 1.51。**所以当前实现并未逼近硬件极限。**

★**一条已关的出路，和两组已作废的数字。** 「把 tile 做大以摊薄每 tile 固定开销」这条
曾被寄予期望，现已用主机侧 standalone tiling 探针**实测关闭**：`M=128` 即使把 S 由
int32 改 fp16 仍 `rc=-1`；`N=512` 被 PV 的 `K<=256` 挡住（API 把 `baseK` 定在 128、
`stepKa<=2`），**与 S 的 dtype 无关**。
更早还出现过两组更低的下界（fixpipe 受限 193.2 ms，及其杠杆 129 / 161 ms），
它们建立在「时间由搬运字节决定」的模型上，**该模型已被实测推翻**——
`tile_m=32` 与 `tile_m=64` 搬运完全相同的 fixpipe 字节（各 10.015 GB/核）而耗时差一倍。
那两组数字一并作废，不要再引用。

### 与 Laser 的差距落在哪

上面的不确定性**不影响**与 Laser 对比的结论，因为那条结论只依赖实测与精度：

- **耗时**：实测 285.46 ms vs 95.30 ms，3.0 倍。即便乐观地按 207 ms 算，仍是 2.2 倍。
- **精度**：`dense_fp32` 相对 RMSE 2.98% vs 0.191%，约 16 倍；这一项与 tile 切分、
  指令条数、重叠度**都无关**，它来自量化契约本身。
- **而能把耗时压低的那条杠杆恰好被精度闸拦着**：S 改 fp16 会使
  `decoded_reference` 达 1.1089%，超过 1% 闸门，甲方已否决过一次。

所以结论不是「还差多少毫秒」，而是：**量化换来的收益在这个形状上没有被测到，
而代价（耗时与精度）两项都被测到了。** 继续投入之前，应当先确认**这条路线要解决的
到底是什么问题**——若目标是显存或带宽，需要给出该维度的测量；若目标是速度或精度，
本轮数据不支持。

## 环境

CANN 9.0.0，`Ascend910B2`（`npu-smi` 25.5.1），torch 2.9.0 + torch_npu 2.9.0，
Python 3.10，aarch64。在 910C / A3 上跨核标志的实现与调度不保证一致，请独立实测，
不要据本侧数字推断加速比。
