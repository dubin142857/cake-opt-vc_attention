# 交付说明 — Ascend 910B2 数据流优化

> **本目录是唯一入口。** 仓库顶层有甲方自己的 40 多个文档，本次交付的全部内容都在这里。
> 要跑起来只看本文的「启动」一节；要复核数字看「验证手段」一节。

| 文件 | 内容 |
| --- | --- |
| **`README.md`**（本文） | 当前进展、怎么启动、怎么验证、口径边界 |
| **`TECHNICAL_REPORT.md`** | 九级改动的机理、证据、正确性 |
| **`REPRODUCE.md`** | 从基线逐级复现的完整步骤 |
| `OPTIONAL_FP16_S.md` | **可选级存档**：QK 直出 fp16。它在本验收判据下不通过，不在交付内 |

## 当前进展

交付分支 **`h26-bitdecode`**。**内核改动止于 `0979605`**，产物哈希对应的就是它；
分支顶端在其之上只有交付文档与复现脚本（不碰 `ascend/`，不影响产物）。
在甲方基线 `9e3a550d399c560cc7c7d9e25cbbc4157e82c0d5` 上叠加**九级**数据流、同步时序与
向量指令的改动。长形状固定核心用时下降 **74.4%**（`control`）/ **75.8%**（`expcast`）。

构建产物应为：

```
4d20b857d25f35414ed8090a5a441d07129796f3d9bb4f09b1ece1304eab602d  libvc_ascend_ops.so
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

```bash
python3 scripts/profile_ascend910b.py --device <卡号> --trials 8
#  6 组形状各连发 8 次，期望全部 8/8 相同
```

★用 **8 次而非 2 次** —— 2 次正是漏掉同类竞态的原因。

## 九级改动

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
AIV 向量 2.03 us/tile。**任何跨核协议的改动都降不到 2.03 以下**，即本设计的硬下界约 207 ms。
同机 CANN FA 做着 1.63 倍的 matmul 却只用 142 ms —— 它的 AIV 向量工作量不到我们的一半，
差距主要在 ExpCast 这条 E4M3 量化 softmax 的**数值契约**上，不在实现效率。
若要继续，这是需要双方先谈契约的部分。

## 环境

CANN 9.0.0，`Ascend910B2`（`npu-smi` 25.5.1），torch 2.9.0 + torch_npu 2.9.0，
Python 3.10，aarch64。在 910C / A3 上跨核标志的实现与调度不保证一致，请独立实测，
不要据本侧数字推断加速比。
