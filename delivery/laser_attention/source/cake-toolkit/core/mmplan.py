"""matmul tiling 配置的**前端证伪器** (算子无关): 在构建前排掉不可行/不可能有收益的候选.

定位 —— 这不是性能预测器. 实测证明 tiling 的绝对搬运量无法由静态模型准确预测 (见下方 CALIBRATION:
同一模型对基线偏 -10%, 对 stepM=2 偏 -31%), 原因是 matmul 库的内部循环序、尾块分布、L2 行为都不可
从源码推定. 但**必要条件**是可判的: 装不下就是装不下, 共享操作数不全 k 驻留就一定不复用. 本工具只做
这两类判定, 把 20 分钟一轮的构建搜索压缩成秒级筛选, 幸存者再交给真机扫描 (cases/*/sweep_tiling.py).

判据
  F1 容量可行性 (硬失败, 精确): L1 / L0A / L0B / L0C / 输出 dtype 字节数. **L1 上界是 511KB 不是 512KB**
  C1 CV 流水深度 (硬失败, 真机实证): parallNum >= 2·nBatch, 否则 CV 重叠塌掉 —— 每条流水都变快而端到端变慢
  F6 stepKa/stepKb 相容 (硬失败, 精确): 需 min(stepKa,stepKb) ∣ max(stepKa,stepKb). 违反 = 未定义行为,
     实测既出现 aicore 507015 崩溃, 也出现**静默算错且端到端快 16%** (少读 22% 字节). 17 点实测零反例.
     ★这是本工具最该先跑的一条: 它拦的错误在性能上表现为"优化".
  W_单缓冲 (警告, 不是失败): depthA1 = stepM·stepKa 时 A 单缓冲, 实测慢 17.7% 但 checksum 逐位相同
  F2 复用必要条件 (硬失败, 精确): 想靠 stepM 让 B 跨 M tile 复用, 必须 stepKb·baseK >= k
     (否则 B 每个 k 步即被换出, 每个 M tile 重新流一遍); 想靠 stepN 让 A 跨 N tile 复用, 同理需
     stepKa·baseK >= k. **这条判据若早存在, gmfr perf2b 的 stepM=2 就会在构建前被毙掉**
     (它 stepKb·baseK=256 < k=1536, 实测搬运量不降反升 28%).
  F3 搬运量下界 (参考值, 非预测): 按上面的复用条件算 GM→L1 的**理想下界**, 只用于给候选排序,
     并同时打印与 cube 算力地板的比值, 判断改这个旋钮是否可能触到瓶颈. 绝对值不可信, 相对序可用.

用法
  python -m core.mmplan --m 102400 --n 4096 --k 1536 --e 400 --enumerate
  python -m core.mmplan --m 102400 --n 4096 --k 1536 --e 400 \\
      --config baseM=128,baseN=256,baseK=256,stepM=2,stepKa=1,stepKb=1
"""
from __future__ import annotations
import argparse, itertools, json, sys

# 910B2 每核片上容量 (字节). L0C 为**实测可用值**: 128x256xint32 = 128KB 通过, 256x256xint32 = 256KB
# 被 GetTiling 拒绝, 故按 128KB 判 (标称 256KB 中另一半用于库内部/双缓冲, 不可直接支配).
#  ★L1 取 511KB 而非 512KB —— **实测**: host 侧 platform 查出的 l1Size 是 511KB (真机 tiling 日志
#    打 "L1 384KB/511KB"). 差这 1KB 会真出事: 一个请求 512KB 的配置 (depthA1=12 baseM=256 baseK=128
#    + depthB1=8 baseK=128 baseN=128) 正好越线, 而覆写钩子只打警告照跑, 实际生效的配置未知,
#    据它测出的数是无意义的 (gmfr 2026-08-10 的 s1 点就这么废掉了).
#  ★键名必须是**具体芯片型号**, 不是 CANN 的目录系列名。
#    `ascend910b` 在 CANN 里是二进制目录名 (binary/ascend910b/ 放 910B 全系列),
#    但作为芯片型号它指的是另一颗真实存在的芯片: Ascend910B 是 arch 1001 / AIC-C-100 /
#    30 个一体核 / L2 32MB / UB 256KB, 与本机的 Ascend910B2 (arch 2201 / 24 cube + 48 vector /
#    L2 192MB / UB 192KB) 是两代。本表原先用 `ascend910b` 当键却存着 910B2 的数,
#    名实不符 —— socspec --audit 报的"本机型号不在表里"正是这件事。2026-08-29 改名。
CAP = {"ascend910b2": {"L1": 511 * 1024, "L0A": 64 * 1024, "L0B": 64 * 1024, "L0C": 128 * 1024,
                      "UB": 192 * 1024, "L2": 192 * 1024 * 1024, "cores": 24,
                      "mac_per_cycle_int8": 8192, "ghz": 1.8,
                      # ★910B 芯片设计: cube 核与 vector 核是分立物理核, **没有 cube->vec 直接通路**,
                      #   所有 CV 数据必须经 GM 中转 (可被 L2 缓存). 故 matmul 的 C 矩阵不能落对侧 UB,
                      #   SetCType(TPosition::VECCALC) 在本 SoC 的分立核 MIX 场景不可用 (树内那些用例
                      #   属 arch32 一体核路径). CV 优化的唯一杠杆是让 staging 留在 L2 不落 DRAM.
                      "cv_direct_path": False,
                      # DRAM 可达带宽 (实测标定: 读3.34+写1.12=4.46GB / 4.16ms = 1.07TB/s 时已饱和)
                      "dram_GBps": 1070.0,
                      # 单核 MTE2 峰值 (实测达到过 132GB/s, 故 83.6GB/s 的基线不是 MTE2 能力上限)
                      "mte2_GBps_per_core": 132.0}}

# CALIBRATION —— 本模型的 F3 下界 vs 真机实测 (gmfr issue shape: m=102400 k=1536 n=4096 e=400 avg_m=256,
# group_list 随机). 保留在此是为了让使用者看见误差带, 而不是把下界当预测值用.
#   配置                                            F3下界    真机GM→L1   偏差
#   baseM128 baseN256 baseK128 stepM1 stepKa4 stepKb4  7.55GB    8.35GB   -10%
#   baseM128 baseN256 baseK256 stepM2 stepKa1 stepKb1  7.55GB   11.00GB   -31%
# 偏差来源 (已知不可静态推定): 随机 group_list 使 Σceil(m_i/M) 大于 m/M (尾块); matmul 库循环序未公开.
CALIB = [
    # gmfr issue shape 真机 8 点扫描 (cases/*/sweep_tiling.py, det_off, 空闲卡). 单位 MB/核 与 us.
    # baseK stepM stepKa stepKb | GM→L1  aic   mte2  cube   注
    #   128    1     4      4   | 348.4  4717  4186  1947   原厂基线, 全场最快
    #   256    2     1      1   | 446.5  5951  5782  2054   F2 判无效 (B 未全 k 驻留)
    #    64    2     1     24   | 346.5  5738  5532  2049   F2 过但 F4 不过 -> 搬运量与基线无差别
    #   128    2     1     12   | 346.5  5702  5493  2049   同上
    #    64    1     1     24   | 348.4  6146  5959  2024   单纯全 k 驻留, 无收益且更慢
    # 结论: 在 M1·N1 ≤ L0C/c_bytes = 32768 的约束下, 搬运量下界 (M1=N1≈181) 为 6.6GB, 基线 7.03GB
    # 已在地板 6% 以内 —— **tiling 空间已关闭**, 收益必须来自 tiling 之外 (见下方 staging/DRAM 记账).
    {"cfg": "baseK128 stepM1 stepKa4 stepKb4", "lower_GB": 7.03, "measured_GB": 8.35},
    {"cfg": "baseK256 stepM2 stepKa1 stepKb1", "lower_GB": 7.03, "measured_GB": 11.00},
    {"cfg": "baseK64 stepM2 stepKb24", "lower_GB": 7.03, "measured_GB": 8.32},
]
# ★第三轮 (2026-08-10, gmfr issue#2: m=256000 k=1536 n=4096 e=800 avg_m=320) —— **step/depth 空间已穷举**,
#   23 个合法点每点都过 det_bit + 输出 checksum 闸门, 结论: **出厂配置最优, 不必再扫这一层**.
#   (baseM,baseN)=(256,128) 族, det_off us, 相对出厂 ka4_kb4=12562:
#     ka4_kb2 -0.5%(噪声内)  ka4_kb4 0%(出厂)  ka6_kb3 +7.5%  ka6_kb2 +7.5%  ka2_kb4 +9.7%
#     ka2_kb2 +10.2%  ka4_kb1 +10.4%  ka2_kb1 +17.5%  单缓冲A +17.7%  ka6_kb1 +18.6%
#     ka3_kb3 +28.9%  ka1_kb4 +36.0%  ka3_kb1 +36.6%  ka1_kb1 +42.3%
#   (baseM,baseN)=(128,256) 族: 最好 ka4_kb2 +1.0%, 满连续比 ka12_kb1 +2.5% —— 整族都不如出厂.
# ★同轮否掉的一个假设 (省得再试): "激活 ND 行跨距 k, 每行只连续读 stepKa·baseK 字节, 拉长连续段能提交付率".
#   同族内锁住字节/块数/C侧流量/AIV工作量, 只把连续比从 0.33 提到 1.00 (stepKa 4->12):
#   GM→L1 853MB/核 不变, aic_mte2 12198->12076us, 交付率 40.7->41.1 B/cycle —— **只值 +1.0%**.
#   MTE2 交付率的跨配置波动 (实测 40.7~81.4 B/cycle) 几乎全部来自**系统竞争** (AIV 每单位时间施加的
#   访存压力), 而非请求形状: det_on 61.5 / 累加器缩到 40MB 66.9 / AIV 被拖慢 69% 时 70.8;
#   而 n 切片调度把原子写 L2 缺失降 62% 但**总 DRAM 不变**时, 交付率一动不动 (48.7->48.9).
#   => 想提交付率只能减 AIV 的访存总量; 而 GM 侧总流量 33.28GB 在参数层面是守恒的 (6 次尝试全部
#      被等量代价抵消). 详见 cases/grouped_matmul_finalize_routing/DEBUG_FINDINGS.md 2026-08-10 节.
# ★第二轮 (baseM/baseN 扫描, 7 个有效点) 的关键结论 —— 它限定了本工具的适用边界:
#   M256/N128/K128: GM→L1 350MB/核, aic 4253us   ← 比原厂基线 (348MB, 4681us) 快 9.1%
#   原厂基线       : GM→L1 348MB/核, aic 4681us
#   两者搬运字节数几乎相同而时间差 9%, 且赢家的 MTE2 指令数更多 (4709 vs 3957)、L2 命中率相同 (69.4%).
#   对 7 个点算相关系数: 指令数 r=+0.987 (被 M32/N512 离群点拉高, 且与赢家反向), 字节 r=+0.921,
#   L2 命中率 r=+0.705 (正相关 = 越高越慢, 显然是伴随而非因果).
#   => **没有任何可静态估的量能解释这 9%**. 故本工具只做硬约束证伪 (F1/F2/F4/F5), 判优必须交给
#      真机扫描 (cases/*/sweep_tiling.py). 不要试图给它加"性能模型".


# ---------------------------------------------------------------------------
# 判据出处表 —— 让每条判据可被证伪 (schema 见 core/_refs.py; opdbg --refs 横向聚合)
from core._refs import Ref, print_refs

REFS = {
    "F1_L0C_128KB": Ref(
        what="L0C 可支配上界按 128KB 判 (标称 256KB 的另一半不可直接支配)",
        status="measured",
        evidence="128x256xint32 通过 / 256x256xint32 被 GetTiling 拒绝 (gmfr, 见本文件 CAP 注释)",
        verified="2026-08-10"),
    "F1_L1_511KB": Ref(
        what="L1 上界 511KB 而非 512KB",
        status="measured",
        evidence="host 侧 platform 查出 l1Size=511KB, 真机 tiling 日志打 'L1 384KB/511KB'; "
                 "一个正好请求 512KB 的配置越线而覆写钩子只警告照跑 (gmfr 2026-08-10 的 s1 点作废)",
        verified="2026-08-10"),
    "F6_stepK_divides": Ref(
        what="min(stepKa,stepKb) 必须整除 max —— 违反是未定义行为, 既崩过也**静默算错且端到端快 16%**",
        status="measured",
        evidence="gmfr issue#2 17 点全覆盖零反例: 合法 13 点全 PASS, 非法 4 点全失败; "
                 "(4,6)+depthB1=12 -> 507015, (4,6)+depthB1=6 -> det_bit=False 且 99.99% 元素错",
        verified="2026-08-12"),
    "C1_CV_depth": Ref(
        what="parallNum >= 2*nBatch, 否则 CV 重叠塌掉 —— 每条流水都变快而端到端 +17%",
        status="measured",
        evidence="gmfr 真机: 逐 pipe 时间全部下降而 aicore_time 上升 17% 的那一组构建",
        verified="2026-08-14"),
    "C2_UB_row_align": Ref(
        what="vecBaseM = ubCalSize/(baseN*nBatch) 须 32B 对齐, 违反 = 真机 507015",
        status="measured",
        evidence="16 点实测全中; '整除' 与 '2 的幂' 两种更弱的写法均被同批数据证伪",
        verified="2026-08-16"),
    "F3_transfer_lower_bound": Ref(
        what="F3 给的是 GM->L1 理想下界, **只可用于排序, 绝对值不可信**",
        status="disputed",
        evidence="同一模型对基线偏 -10%, 对 stepM=2 偏 -31% (本文件 CALIBRATION 两行); "
                 "偏差源 (随机 group_list 尾块 / 库内循环序未公开) 不可静态推定",
        verified="2026-08-12"),
    "W7_wave_quantization": Ref(
        what="tile 总数与 aic 核数不成整数倍时, 每核 cube 时间 max/mean = ceil(tiles/cores)*cores/tiles",
        status="measured",
        evidence="cases/gemm_v3/runs/prof_fused_*_r0 共 8 点零反例, 偏差 <=3.2%: 建立时四点 "
                 "(1024^3 1.50/1.55, 1536 1.00/1.015, 3072 1.00/1.015, 640x1280 1.92/1.96) + "
                 "**样本外四点** (768 1.00/1.00, 896 1.71/1.74, 1408 1.09/1.11, 2048 1.12/1.14, "
                 "覆盖建立时空缺的中间段 0.583/0.889/0.917); 同期端到端 fused/split 从 0.83 (整波) "
                 "到 1.32 (最坏), 见 cases/gemm_v3/PREDICTION_wave.md",
        verified="2026-08-27"),
    "W7_workload_not_tilecount": Ref(
        what="核间均衡要按**每核实际工作量**算, 不能只数 tile 个数 —— 尾 tile 更小时按 tile 数会低估",
        status="measured",
        evidence="cases/gemm_v3 wave1 消融 2026-08-27: 1024^3 配 m0=96 时按 tile 数报 1.09、按工作量报 1.12、"
                 "实测 1.14; 896 配 m0=80 按 tile 数 1.00 / 按工作量 1.07 / 实测 1.07. "
                 "更极端的是 wave2 的 m0=160 (1024 除不尽, 尾 tile 只有 64 行): 按 tile 数 1.14 而实测 1.41. "
                 "建立判据时的八点 m/n 恰好都被 m0/n0 整除, 掩盖了这一项",
        verified="2026-08-27"),
    "W7_n0_row_alignment": Ref(
        what="n0 必须让一行凑满 512B —— **仅当用 Nd2NzParams 逐 tile 大块搬运时**; 换搬运方式则不成立",
        status="disputed",
        evidence="cases/gemm_v3 wave2 消融 2026-08-27, 1024^3 同时段三档: n0 256->176 (行长 512B->352B) "
                 "使 GM->L1 只多 8% (1040->1125 KB/核) 而 aic_mte2_time 涨 161% (9.24->24.13us), "
                 "aic_time 20.12->40.27; l1d3 的 k0 256->224 (512B->448B) 同向, mte2 +68%。"
                 "★2026-08-28 被 AscendC 库推翻其普适性: 库对同一 tile 选 baseK=64, A 行长只有 128B "
                 "根本不对齐, 却更快 —— 这条是**手写搬运方式的约束, 不是硬件的**。"
                 "适用范围收窄到: 用 Nd2NzParams 单条大指令逐 tile 搬运时。",
        verified="2026-08-28"),
    "W7_shrink_tile_does_not_pay": Ref(
        what="**缩 tile 换均衡在保守口径 (msprof 单次, L2 较冷) 下不成立** —— 两条路都被实测否掉",
        status="measured",
        evidence="cases/gemm_v3 2026-08-27: (a) 只缩 m0 (wave1): 均衡 1.53->1.14 达标, 但 B 多流几遍使 "
                 "GM->L1 +25%、mte2 +29%, msprof 端到端 640/896 两 shape **倒亏 7.9%/4.5%**; "
                 "(b) 改用非 512B 对齐的 n0 换取搬运量只 +8% (wave2): mte2 时间反而 +161%, aic_time 翻倍。"
                 "连续下发口径下 wave1 快 8~12%, 那是 A/B 全驻留 L2 时多搬的字节近乎免费, 不能作为结论",
        verified="2026-08-27"),
    "W9_n_row_stride_align": Ref(
        what="ND 布局下 M/K/N 三维里, **进了行跨距的那些维**须让跨距为 256B 的整数倍, "
             "否则每行起始地址逐行漂移, Nd2Nz 转换吞吐掉到 40%; 归属随转置翻转, "
             "与 tiling 无关, 补齐是唯一手段",
        status="measured",
        evidence="cases/gemm_v3 2026-08-29。密集扫 N (992~1152, 16 点): N mod 128 == 0 时 27.4us, "
                 "== 64 时 30.4us, 其余 39~52us。同一 GM->L1 字节数 (1040.00KB 逐位相同) 下 "
                 "aic_mte2 9.35 -> 36.46us, aic_l1_write_bw 35.85 -> 14.24 GB/s, 而 L2 命中率反而"
                 "更高 (82.98 -> 90.49) —— 不是缓存问题是访问粒度。跨 M/K/dtype 复现 "
                 "(1024x1024 +76%, 2048x512 +102%, 512x2048 +71%, 4096x1024 +117%, bf16 +86%)。"
                 "**排除 tiling**: GEMM_N0 扫过九种 n0 (128/144/168/192/240/252/256/336, 含能整除 N "
                 "的、尾块为满的) 全部 44~51us 无改善。补齐后 N +32%~49% / K +16%~17%, "
                 "checksum 逐位不变 (GEMM_PADN / GEMM_PADK)。"
                 "★归属随转置翻转 (A 是 [K,M] 时行跨距变 M, B 是 [N,K] 时变 K, C/Y 恒为 N), "
                 "四种组合 12 个真机点: tA0/tB0 N+79% K+18% M-4%; tA0/tB1 N+21% K+70% M-2%; "
                 "tA1/tB0 N+84% K-6% M+53%; tA1/tB1 N+32% K+59% M+61%。"
                 "规律只看**进了几条行跨距**: 0 条 -6%~-2% 不塌, 1 条 +18%~+61%, 2 条 +70%~+84%。"
                 "因此'M 不必补齐'只在 transA=0 时成立。"
                 "★跨算子与跨 dtype 验证 (dual-loop 的 cross-artifact): fp32 输入**不走 GemmV3** "
                 "(aclnnAddmm 拆成 MatMulV2 + Add, ws=0 是判别特征), 正好用来验另一个算子 —— "
                 "MatMulV2 上 N=1012 对 1024 的 matmul 段 29.44 -> 43.29us (**+47%**), "
                 "而同一次里 Add 段 6.52 -> 6.56us 纹丝不动 (逐元素算子不碰行跨距)。"
                 "dtype 周期同步验证: 256B 是硬件粒度与 dtype 无关, fp32 (4B) 的周期是 mod **64** "
                 "而非 fp16 的 mod 128, 实测 mod64==0 的三点 35.7/38.5/39.0us 快过其余 40.1~50.5us。"
                 "int8 (1B, 周期应为 mod 256) 尚未验证。"
                 "★两条绕过手段实测 (都不多算, 优于补齐): (1) NZ 输入 —— NZ 是分形存储没有行跨距, "
                 "本判据不适用; N=1023 上 aic_mte2 36.46 -> 8.84us, 端到端 50.2 -> 35.5 (+29%); "
                 "GemmV3 的 NZ 分支写好了却从没被实例化 (gemm_v3.cpp 的实例都用默认 ND)。"
                 "(2) 算子内部自己造的非对齐写出 —— L0C->GM 落 workspace 用 dstStride=nTileActual "
                 "这种紧凑跨距, N 不整除 n0 时最后一个 tile 就非对齐, fixpipe 5.28 -> 3.47us; "
                 "改按标称 nTile_ 写 (读侧行偏移与 srcGap 成对改) 是纯内部改动。"
                 "两条叠加 28%~40%, checksum 逐位不变。"
                 "★泛化性 (40 个 shape, 覆盖 m>=512/n>=512/k>256): **wsstride 在整个范围内 0 劣化** "
                 "(真收益 12 / 持平 27 / 劣化 0, 收益全部落在 N 不对齐的 shape 上, 与'对齐时是恒等变换'"
                 "的机理吻合), 可无条件合入; 而 **NZ 是条件性的** —— 不对齐 +26.7%~49.2%, "
                 "对齐 **-5%~-11%** (逐 pipe 看是 fixpipe 涨 79%, 而 mte2 确实更好 9.35->8.93, "
                 "成因未定位)。只在几个点上测会得出'NZ 无条件更优'的错误结论, "
                 "因为那几个点恰好全是不对齐的",
        verified="2026-08-29"),
    "W8_splitk_upper_bound": Ref(
        what="沿 K 切分是唯一**不增加 A/B 搬入量**的填核手段, 代价改为出现在输出侧 (L0C->GM 写 x s)",
        status="measured",
        evidence="910B 既有实现 mat_mul_v3/op_kernel/mat_mul_multi_core_splitk_kernel.h (全核屏障 + "
                 "GetTensorC(enAtomic=1) 原子累加); 实测 aclnnGemm M=256 K=16384 N=256 走 MatMulV3: "
                 "2 个输出 tile 却用满 BlockDim=24 (K 切 12 份), 核间不均衡 1.13, "
                 "L0C->GM 写 459KB/核 而输出本身只有 256KB, fixpipe 4.10us / mte3 2.79us",
        verified="2026-08-27"),
    "W8_uniform_splitk_breaks_cv_pipeline": Ref(
        what="均匀 split-k 会拆掉 core-local 的 CV pingpong 流水 (vector 必须等所有分片写完), "
             "这一项代价本模型算不出来 —— 只切尾波的变体才保留主流水",
        status="derived",
        evidence="由 GemmV3 的结构推出: cube 每算完一个 tile 就用 CrossCoreSetFlag 交给**本核** vector "
                 "做 epilogue (gemm_v3_base_kernel.h RunCube/RunVector 的 cvPingPongFlag); split-k 后同一 "
                 "输出 tile 的分片落在不同核上, 必须全核屏障后才能归约, 于是 cube 与 vector 从并行变串行。"
                 "base 实测 aic_time 20.12 与 aiv_time 20.84 基本相等 = 两者现在是并行的",
        verified="2026-08-27"),
    "PIPE_three_pipe_serial": Ref(
        what="mte2+cube+mte1 三者活跃时间之和 ≈ aic_time 时是**全链串行**, 只修其中一道缝不会有整体收益",
        status="measured",
        evidence="GemmV3 1024^3 2026-08-27: 9.58+6.51+4.28=20.37 vs aic_time 20.51 (吻合 99.3%); "
                 "照 H10 把 MTE1↔M 那道缝改成软件流水后, 重叠度 1.22->1.23、aic_time -2.6% (噪声内) —— "
                 "因为 MTE2 与 cube 之间那道更宽的缝没动 (L1 只有两组 pingpong buffer, 且 MTE1_MTE2 "
                 "释放 flag 只在 fragment 循环最后一片才置位, 于是 MTE2(k+1) 必须等 MTE1(k) 整个 tile 走完)。"
                 "同 shape 的 MatMulV2 这个比值是 2.00",
        verified="2026-08-27"),
    "PIPE_overlap_ratio": Ref(
        what="pipe 重叠度 = Σ各 pipe 活跃时间 / aic_time; 接近 1 = 各 pipe 串行, 该查指令序而不是加带宽",
        status="measured",
        evidence="1024^3 fp16 同 shape 对照: AscendC matmul 库 (MatMulV2) 2.25 (aic_time 14.49, Σpipe 32.6) "
                 "vs 手写 pp_matmul 循环 (GemmV3) 1.23 (aic_time 20.12, Σpipe 25.0), 而两者 cube 总工作量 "
                 "相当 (152 vs 159 核·us); GemmV3 各 pipe 的 wait_ratio 又都不高 (cube 0.38/mte1 0.36/"
                 "mte2 0.13), 排除'某条 pipe 卡住'。已固化为 core/perf --signals 的派生量",
        verified="2026-08-27"),
    "W7_no_benefit_prediction": Ref(
        what="W7 只给均衡上界, **不预测端到端收益** —— 改分块这一维总会同时改动访问模式那一维",
        status="measured",
        evidence="cases/gemm_v3 2026-08-28 直接实测: 1024^3 上默认 m0=128 n0=256 的 wave_eff 只有 0.667 "
                 "(理论 max/mean 1.50, 实测 cube 1.55, 判据本身是准的), 而 128x192 与 192x128 都能做到 "
                 "wave_eff=1.000。改用 wave_eff 最优的 128x192 后 **8192x1024x1024 上 129.8us 对默认的 "
                 "75.0us, 慢 73%** (三 shape 全部 checksum 逐位正确, 排除实现错误): n0=192 时 B 每行读 "
                 "384 字节, 既非 512 的倍数也非 2 的幂, nTileRound/16=12 个分形不能对齐到 2 的幂边界, "
                 "访问模式的劣化完全淹没均衡收益。同期另一次同型证伪见 W8/gmfr 记录: 按搬运量 "
                 "k*(m*nGrid+n*mGrid) 选二维分核网格, 算得 GM->L1 从 50.2MB 降到 20.2MB, 实测反而 "
                 "58.0us 对 36.2us。故 wave_pareto 输出前沿而非排名, 净收益一律实测",
        verified="2026-08-28"),
}
# ---------------------------------------------------------------------------


#  ---- 规格取数: 走 socspec 这个唯一取数口, 不再让本文件的内置表当事实 ------------------
#  ★2026-08-29 补。本模块此前 `check(..., soc="ascend910b")` 把型号**硬编码成默认参数**,
#    表里恰好有这个键所以永不报错 —— 它根本不按本机探测。910B2 的 L0C 恰好等于表里 910B 的
#    131072, 所以此前的结论数值上是对的, 但那是**巧合正确**: 换一颗芯片会默默用 910B 的数
#    给出错误结论。socspec --audit 把这 9 项标为"本机无标定", 正是这个问题。
#
#  一处必须保留的偏差: L1 取 **ini 的 l1_size 减去 bt_size**, 不是 l1_size 本身。
#  tiling 运行时从 GetCoreMemSize(L1) 拿到的是扣掉 bias table 之后的值 (真机日志 511KB),
#  判据必须对齐 tiling 实际比较的那个数, 而不是数据手册的 512KB。
#  socspec 的 cap 里有 BT=1024, 524288-1024=523264=511KB, 与内置表逐位吻合 —— 可以精确算出来。
def cap_for(soc: str | None = None) -> tuple:
    """返回 (规格 dict, 出处字符串)。socspec 取不到时回落内置表, **且必须说话**。

    静默回落是本仓明令禁止的形态 (ub_budget 的 .get(soc, 192KB) 就是反面教材):
    用别的芯片的数算出一个"通过", 比报错危险得多。
    """
    try:
        from core import socspec
        sp = socspec.spec(soc)
        cap, mac = sp["cap"], sp["mac_per_cycle"]
        #  ★非 ini 项 (cv_direct_path / 实测可达带宽) **按本芯片取**, 不再沿用 910B 的。
        #    2026-09-11 实撞: 这里原来是 `dict(CAP["ascend910b2"])`, 于是在 950PR 上
        #    cv_direct_path 被当成 False (而本机 arch3510 有 cube->UB 直通, FIA 内核正在用,
        #    msprof 的 L0C_to_GM_datas = 0 就是证据), dram_GBps 用的是 910B 的 1070。
        #    ini 事实自动跟着芯片走, 实测标定却悄悄借用别的芯片 —— 正是本仓禁止的静默回落。
        base = {k: socspec.calib_value(k, sp["soc"])
                for k in ("cv_direct_path", "dram_GBps", "mte2_GBps_per_core")}
        missing = [k for k in ("L1", "L0A", "L0B", "L0C", "UB", "L2") if cap.get(k) is None]
        if missing:
            #  socspec 读得到这颗芯片的 ini, 但缺本模块要用的字段 (老架构的 ini 段名不同)。
            #  这时既不能拿 None 去算, 也不能悄悄换成别的芯片的数 —— 直接说清楚缺哪几项。
            raise KeyError(f"{sp['soc']} 的 ini 里缺 {missing}, 本模块的容量判据无法成立")
        base.update({
            "L1": cap["L1"] - (cap.get("BT") or 0),  # ★见上: 对齐 tiling 实际比较的值
            "L0A": cap["L0A"], "L0B": cap["L0B"], "L0C": cap["L0C"],
            "UB": cap["UB"], "L2": cap["L2"],
            "cores": sp["cube_cores"], "ghz": sp["freq_ghz"],
            "mac_per_cycle_int8": mac["int8"],
        })
        return base, f"socspec/{sp['soc']}"
    except Exception as e:
        #  ★不再把某颗芯片当默认: 没给 --soc 又取不到 socspec 时, 只能按"无型号"处理。
        key = (soc or "").lower()
        if key in CAP:
            print(f"  ★规格回落到内置表 [{key}] —— socspec 取数失败: {e}\n"
                  f"    内置表标定的是 910B 一档; 若本机不是它, 下面的容量判据全部不作数。",
                  file=__import__("sys").stderr)
            return CAP[key], f"内置表/{key}(回落)"
        raise KeyError(f"{key} 无标定, 且 socspec 不可用: {e}")


def _legal_stepkb(sKa: int, bK: int, k: int) -> list:
    """与 stepKa 相容 (min∣max) 且能整分 k 的 stepKb 取值."""
    return sorted(s for s in range(1, k // bK + 1)
                  if k % (s * bK) == 0 and (sKa % s == 0 or s % sKa == 0))


def check(cfg: dict, prob: dict, soc: str | None = None) -> dict:
    c, _src = cap_for(soc)
    bM, bN, bK = cfg["baseM"], cfg["baseN"], cfg["baseK"]
    sM, sN = cfg.get("stepM", 1), cfg.get("stepN", 1)
    sKa, sKb = cfg.get("stepKa", 1), cfg.get("stepKb", 1)
    dA = cfg.get("depthA1") or sM * sKa * 2
    dB = cfg.get("depthB1") or sN * sKb * 2
    m, n, k, e = prob["m"], prob["n"], prob["k"], prob["e"]
    avg_m = prob.get("avg_m") or (m // e if e else m)
    ab, bb = prob.get("a_bytes", 1), prob.get("b_bytes", 1)      # int8 -> 1
    cb = prob.get("c_bytes", 4)                                   # 累加输出 int32 -> 4

    fails, warns = [], []
    # --- F1 容量 ---
    l1 = dA * bM * bK * ab + dB * bK * bN * bb
    if l1 > c["L1"]:
        fails.append(f"F1_L1: depthA1·baseM·baseK + depthB1·baseK·baseN = {l1//1024}KB > L1 {c['L1']//1024}KB")
    if bM * bK * ab > c["L0A"]:
        fails.append(f"F1_L0A: baseM·baseK = {bM*bK*ab//1024}KB > {c['L0A']//1024}KB")
    if bK * bN * bb > c["L0B"]:
        fails.append(f"F1_L0B: baseK·baseN = {bK*bN*bb//1024}KB > {c['L0B']//1024}KB")
    if bM * bN * cb > c["L0C"]:
        fails.append(f"F1_L0C: baseM·baseN·{cb}B = {bM*bN*cb//1024}KB > 可用 L0C {c['L0C']//1024}KB "
                     f"(实测上界, 非标称值)")
    if k % bK != 0:
        warns.append(f"k={k} 不被 baseK={bK} 整除, 尾块单独处理会吃掉收益")

    # --- F6 stepKa/stepKb 的 k 循环相容 (硬失败; 违反 = **未定义行为**, 不是变慢) ---
    #   A 与 B 共用同一个 k 循环, 各自按 stepKa·baseK / stepKb·baseK 的粒度重填 L1. 两个粒度必须
    #   互为整数倍, 否则重填边界对不齐, 库产生的地址越界或漏算.
    #   ★真机实测 17 点全覆盖 (gmfr issue#2, baseM256/baseN128/baseK128, 双缓冲, 每点都过
    #     det_bit + 输出 checksum 闸门), 规则 min∣max 与实测**零不一致**:
    #       合法且 PASS: (1,1)(1,4)(2,1)(2,2)(2,4)(3,1)(3,3)(4,1)(4,2)(4,4)(6,1)(6,2)(6,3)
    #       非法且失败:  (3,4)(4,3)(4,6)(6,4)
    #   ★两种失败表现都出现过, 取决于 depth 设置 —— 所以必须在构建前拦, 不能靠"跑一下看崩不崩":
    #       (4,6) depthB1=12 -> aicore exception 507015 (崩, 响亮)
    #       (4,6) depthB1=6  -> **静默算错**: det_bit=False, 双万分之一 9.999e-01 (99.99% 元素错),
    #                           |y|max 比值 0.87 被截顶, 而 GM→L1 从 905 掉到 705MB/核、端到端**快 16%**
    #   最后那条是本判据存在的真正理由: 违规的"收益"是少读 22% 字节买来的, 只看性能会当成优化上报.
    if sKa % sKb and sKb % sKa:
        fails.append(f"F6_stepK不相容: stepKa={sKa} 与 stepKb={sKb} 互不整除 (需 min∣max) -> A/B 的 L1 "
                     f"重填粒度 {sKa*bK} 与 {sKb*bK} 边界对不齐, 未定义行为. 实测表现为 aicore 507015 "
                     f"或**静默算错且看起来更快** (少读字节). 同时满足 min∣max 与 k%(stepKb·baseK)==0 "
                     f"的合法取值: {_legal_stepkb(sKa, bK, k)}")

    # --- F4 L0C 存活累加器 (复用的**充分性**条件, 由真机扫描反证得来) ---
    #   for k: for m: 的循环序 (B 载入一次给 stepM 个 M tile 共用) 要求这 stepM 个累加器在整个 k 循环
    #   期间同时存活于 L0C. 装不下则库退化成 for m: for k:, B 每个 M tile 重载一次 —— 此时 stepM 完全
    #   无效, 即使 B 已全 k 驻留 L1.
    #   真机反证 (gmfr, 8 点扫描): baseK=64 stepM=2 stepKb=24 满足 F2 (24x64=1536=k) 却仍无收益,
    #   GM→L1 346.5MB/核 与基线 348.4 无差别 —— 因为 2x128x256x4 = 256KB > 可用 L0C 128KB.
    live_l0c = sM * sN * bM * bN * cb
    if (sM > 1 or sN > 1) and live_l0c > c["L0C"]:
        fails.append(f"F4_L0C存活累加器: stepM·stepN·baseM·baseN·{cb}B = {live_l0c//1024}KB > 可用 L0C "
                     f"{c['L0C']//1024}KB -> 这 {sM*sN} 个累加器无法同时存活, 库退化为 for m: for k:, "
                     f"stepM/stepN 不产生复用 (即使 F2 通过). 需 stepM·stepN ≤ {c['L0C']//(bM*bN*cb)}")

    # --- C1 CV 流水深度 vs 攒批粒度 (硬失败: 违反不会算错, 但会把所有流水收益吃光还倒亏) ---
    #   gmfr 的 CV 协议: AIC 每算完一块发一个 SYNC_AIC_TO_AIV, 并在 cubeCount>=parallNum 时等一个
    #   SYNC_AIV_TO_AIC; AIV 攒够 nBatch 块才消费并回同样多的 flag.
    #   parallNum == nBatch 时两边轮流干活 —— **真机实测 (2026-08-12, issue#1, nBatch=4/parallNum=4)**:
    #     aiv_scalar -62%  aiv_mte3 -63%  aiv_mte2 -52%  aic_mte2 -20%  aic_fixpipe -16%  (每条都降)
    #     而 aic_time 4252 -> 4985 (**+17%**), AIC 忙时重叠比 1.82x -> 1.40x.
    #   这是本工具里第一条"所有分项都改善而总量变差"的判据, 也是它存在的理由: 逐 pipe 看只会得出
    #   "改动成功"的错误结论. 需 parallNum >= 2·nBatch (一批被消费的同时下一批能发满 = 批级双缓冲).
    nb = cfg.get("nBatch", 1)
    pn = cfg.get("parallNum", 4)
    if nb > 1 and pn < 2 * nb:
        fails.append(f"C1_CV流水深度: parallNum={pn} < 2·nBatch={2*nb} -> AIC 发满即停等, AIV 攒齐才动, "
                     f"两边轮流干活. 真机实测 nBatch=4/parallNum=4 时每条流水忙时都降而端到端 +17% "
                     f"(重叠比 1.82x->1.40x). 需 parallNum >= {2*nb}, staging 随之涨到 "
                     f"{2*nb}·coreNum·singleM·baseN·{cb}B")

    # --- C2 UB 宽 tile 的行数必须 32B 对齐 (硬失败: 违反 = 真机 aicore exception 507015) ---
    #   攒批把 UB tile 从 vecBaseM x baseN 改成 (ubCalSize/(baseN*nBatch)) x (baseN*nBatch).
    #   主循环 offsetM 每次步进 vecBaseM, 而**逐行标量**(gmfr 里是 perTokenScale)在 UB 里被按
    #   offsetM 索引后喂给 BroadCast. vecBaseM 不是 32B/elem 的倍数时该源地址不再 32B 对齐 -> 507015.
    #   ★判据来之不易, 两条更"自然"的写法都被实测证伪 (gmfr 2026-08-13, 16 个测点):
    #     "ubCalSize 整除 baseN*nBatch"  -> (ubCalSize,nBatch)=(3072,2)/(3072,4) 整除却挂
    #     "baseN*nBatch 是 2 的幂"       -> (3072,3) 非 2 的幂却通过
    #   只有 "vecBaseM % (32/elem_bytes) == 0" 对上全部 16 点 (含 8 个样本外预测).
    #   ★elem_bytes 取的是**逐行标量那条向量**的元素字节 (gmfr 是 float=4 -> 8 个/块), 不是输出 dtype.
    #  ★C2 的**成因是那条 BroadCast**, 不是宽 tile 行数本身 (gmfr 2026-08-24 核实):
    #    逐行标量被按 offsetM 索引后喂给 BroadCast, 而向量指令的源要 32B 对齐; 理论上若实现改成
    #    "每行一条 Muls 吃 GetValue 读出的标量", 标量读没有对齐要求, C2 就不再成立。
    #  ★但**不提供跳过 C2 的开关** (2026-08-25 撤回了曾加过的 row_scalar_broadcast):
    #    ① 加它是为了放宽 P3 好让写出粒度翻倍, 而那条路真机实测只值 +40.4us (等 staging 下
    #       写次数减半 ≈ 零收益), 即这个开关服务的方向已被证伪, 无人调用;
    #    ② 它是**正确性判据**上的逃生门 —— 使用者以为自己删干净了 BroadCast, 漏掉别处一个同类
    #       索引, 检查器就会对一个真会 507015 的配置说"通过"。判据宽于实现只损失性能,
    #       窄于实现会放过真崩的配置, 这道门开错方向。
    #    将来真把 BroadCast 删了, 拿实测数据重新加, 不要凭推理开门。
    ub_cal = cfg.get("ubCalSize", 0)
    row_elem_bytes = cfg.get("row_scalar_bytes", 4)   # 逐行标量的元素字节; gmfr: perTokenScale 是 float
    if nb > 1 and ub_cal:
        blk = 32 // row_elem_bytes
        wide = bN * nb
        vbm = ub_cal // wide if wide else 0
        if vbm == 0 or vbm % blk != 0:
            fails.append(f"C2_UB行对齐: vecBaseM=ubCalSize/(baseN*nBatch)={ub_cal}/{wide}={vbm} "
                         f"不是 {blk} 的倍数 -> 逐行标量在 UB 里的索引地址非 32B 对齐, 真机 507015 "
                         f"(gmfr 实测: ubCalSize=4096/nBatch=3 -> vecBaseM=10 挂; =4 -> 8 通过). "
                         f"可选 nBatch: {[d for d in range(1, nb + 5) if wide and (ub_cal // (bN * d)) and (ub_cal // (bN * d)) % blk == 0]}")

    # --- F5 C 矩阵位置 ---
    if c.get("cv_direct_path") is None:
        #  没有本芯片的标定就不判这一条, 而不是按 False 判死 (那等于用别的芯片的拓扑下结论)
        return {"check": "F5_c_position", "ok": None, "level": "无法判定",
                "msg": (f"本芯片没有 cv_direct_path 的标定 —— 它是**实测/拓扑事实**, ini 里没有。"
                        f" 补法: 在 core/socspec.CALIB 里加本芯片一项 (证据: 内核里 Fixpipe 到 UB 的"
                        f"那条通路能否跑通, 或 msprof 的 L0C_to_GM_datas 是否为 0)。")}
    if cfg.get("c_pos", "gm").lower() in ("ub", "veccalc") and not c.get("cv_direct_path", False):
        fails.append(f"F5_无CV直连: 本 SoC 的 cube 核与 vector 核分立, 没有 cube->vec 直接通路, matmul 的 C "
                     f"不能落对侧 UB (SetCType(VECCALC) 仅适用一体核路径). CV 必须经 GM 中转; 可优化的是"
                     f"让 staging 留在 L2 (SetL2CacheHint), 而非消除中转.")

    # --- W_单缓冲 (只警告: 正确性不受影响, 但实测慢) ---
    #   depthA1 = stepM·stepKa 即 A 面板单缓冲, MTE2 必须等 MTE1 把上一块面板排空才能载下一块.
    #   ★实测 (gmfr issue#2, stepKa=stepKb=4, depthA1 8->4, 其余不变): 输出 checksum 与双缓冲**逐位
    #     相同**、det_bit 仍 True, 即**不致错**; 但 det_off 12562->14791us (+17.7%).
    #   曾据"单缓冲非法"把两个有效实测点 (stepKa6/stepKb2 与 stepKa12/stepKb1) 误判作废, 教训是
    #   "慢" 与 "错" 必须分开判 —— 故此处是 WARN 不是 FAIL.
    if dA <= sM * sKa:
        warns.append(f"depthA1={dA} = stepM·stepKa ({sM}·{sKa}) 即 A 单缓冲: MTE2 要等 MTE1 排空面板, "
                     f"实测慢 17.7% (正确性不受影响, checksum 逐位相同). 双缓冲需 depthA1={2*sM*sKa} "
                     f"(L1 +{sM*sKa*bM*bK*ab//1024}KB)")
    if dB <= sN * sKb:
        warns.append(f"depthB1={dB} = stepN·stepKb ({sN}·{sKb}) 即 B 单缓冲, 同上 (只影响性能不影响正确性)")

    # --- F2 复用必要条件 ---
    b_resident = sKb * bK >= k        # B 全 k 驻留 L1 -> 才可能跨 M tile 复用
    a_resident = sKa * bK >= k        # A 全 k 驻留 L1 -> 才可能跨 N tile 复用
    if sM > 1 and not b_resident:
        fails.append(f"F2_stepM无效: stepM={sM} 想让 B 跨 M tile 复用, 但 stepKb·baseK = {sKb*bK} < k = {k}, "
                     f"B 每个 k 步即被换出 -> 每个 M tile 重新流一遍, 搬运量不降. 需 stepKb >= {-(-k//bK)}")
    if sN > 1 and not a_resident:
        fails.append(f"F2_stepN无效: stepN={sN} 想让 A 跨 N tile 复用, 但 stepKa·baseK = {sKa*bK} < k = {k}. "
                     f"需 stepKa >= {-(-k//bK)}")

    # --- F3 搬运量下界 (排序用, 非预测) ---
    M1 = bM * sM if b_resident else bM     # 权重实际能摊到的 M 行数
    N1 = bN * sN if a_resident else bN     # 激活实际能摊到的 N 列数
    b_bytes = e * k * n * bb * max(1, -(-avg_m // M1))     # 权重: 每 M 面板读一遍
    a_bytes = m * k * ab * max(1, -(-n // N1))             # 激活: 每 N 面板读一遍
    total = a_bytes + b_bytes
    # ★CV staging: 910B 无 cube->vec 直连, matmul 的 int32 中间结果必须写 GM 再由 vector 核读回.
    #   字节数与 tiling 无关 (每个输出元素恰好写一次读一次), 故它是 tiling 空间之外的独立开销.
    #   常驻占用 = parallNum·coreNum·singleM·singleN·c_bytes, 远小于 L2, 理论上可全命中 L2;
    #   实测 (gmfr) 只有 32% 被 L2 吸收, 68% 落 DRAM -> SetL2CacheHint 是这里的杠杆.
    stg_once = m * n * cb
    # ★L2 命中率取 0.98 —— **实测值**: msprof L2Cache 表 aic_write_hit_rate=99.2%, aiv_read_hit_rate=97.5%,
    #   即 CV staging 往返几乎完全在 L2 内完成, 基本不落 DRAM.
    #   (此处曾默认 0.32 —— 那是我未经测量的假设, 据它算出的"DRAM 带宽地板 4631us"与实测 wall 4717us
    #    偶然吻合, 险些把"staging 落 DRAM 是瓶颈"这个错结论坐实. 凡默认值必须标明是实测还是假设.)
    l2_hit = prob.get("staging_l2_hit", 0.98)
    stg_dram = stg_once * 2 * (1.0 - l2_hit)           # 写一次 + 读一次
    # DRAM 侧: weight/x 的唯一字节 (重复读多半被 L2 吸收) + staging 落盘部分 + 输出
    uniq = e * k * n * bb + m * k * ab
    dram = uniq + stg_dram
    cube_us = (m * n * k) / (c["cores"] * c["mac_per_cycle_int8"] * c["ghz"] * 1e9) * 1e6
    if not c.get("dram_GBps"):
        raise SystemExit(
            "本芯片没有 dram_GBps 的**饱和实测**标定, 拒绝用模型值或别的芯片的数算 DRAM 下界。\n"
            "  ini 的 ddr_rate 是编译器成本模型参数 (本机算出 1.25TB/s), 不是可达带宽;\n"
            "  要补标定请专门跑一个纯流式负载打满 DRAM, 再写进 core/socspec.CALIB。")
    dram_us = dram / (c["dram_GBps"] * 1e9) * 1e6
    mte2_us = total / (c["cores"] * 83.6e9) * 1e6      # 83.6GB/s/核 = 基线实测值 (非峰值 132)
    return {"cfg": cfg, "fails": fails, "warns": warns, "feasible": not fails,
            "L1_KB": l1 // 1024, "M1": M1, "N1": N1,
            "b_resident": b_resident, "a_resident": a_resident,
            "a_GB": a_bytes / 2**30, "b_GB": b_bytes / 2**30, "total_GB": total / 2**30,
            "live_l0c_KB": live_l0c // 1024,
            "staging_GB": stg_once * 2 / 2**30, "staging_dram_GB": stg_dram / 2**30,
            "dram_GB": dram / 2**30, "cube_floor_us": cube_us,
            "dram_floor_us": dram_us, "mte2_est_us": mte2_us}



# ============================================================================
# W7 波量化 (核间负载均衡) —— 与 step/depth 无关的独立判据, 适用于任何"按 tile 切核"的算子.
#
# 判什么: tile 总数与 aic 核数不成整数倍时, 最后一波只有部分核在跑, 其余核空转到 kernel 结束.
#   有效并行度 wave_eff = tiles / (ceil(tiles/cores) * cores);
#   理论 `每核 cube 时间 max/mean` = 1 / wave_eff (完美均衡时 = 1.0).
#
# ★实证 (2026-08-27, GemmV3 @ 910B2, cases/gemm_v3/runs/prof_fused_*_r0), 四点零反例:
#     shape (m,k,n)        tiles  波  预测 max/mean  实测 max/mean
#     1024,1024,1024        32    2      1.50           1.55
#     1536,1024,1024        48    2      1.00           1.015
#     3072,1024,1024        96    4      1.00           1.015
#     640,1024,1280         25    2      1.92           1.96
#   同期端到端: 波满时 GemmV3 比 MatMulV2+Add 快 17%, 波最差时慢 32% —— 同一个 kernel, 只有 shape 不同.
#
# 为什么会缺这条: pp_matmul 选 m0/n0 的 CostFunc 只含 A/B 搬入量 (1/(a·n0)+1/(b·m0)), 没有均衡项;
#   SetBaseOp 只做 blockDim=min(coreLoop,coreNum), 不回头调 tile 尺寸.
#   MatMulV3 有对应的一段 (matmul_v3_base_tiling.cpp:OptimizeLoadBalanceBasicKernel), pp_matmul 没有.
#
# 局限 (必须一起读): 本判据只给**均衡上界**, 不预测端到端收益 —— 缩小 m0 改善均衡的同时会让
#   B 矩阵多流几遍 (mLoop 变大), 而 mte2 往往已经是最大单项. 两者的净和不可静态推定, 必须实测.
#   故 wave_pareto() 输出的是**帕累托前沿**而不是排名, 前沿上选谁由真机消融定 (core/ablate).

def wave_check(m: int, n: int, m0: int, n0: int, k: int = 0, cores: int = 24,
               ab_bytes: int = 2) -> dict:
    """给定 tile 尺寸, 报核间均衡与 GM->L1 搬入量. 不判可行性 (那是 F1 的事).

    ★两个均衡口径, 默认用后者:
      by_tiles  = ceil(tiles/cores)*cores/tiles —— 只数 tile 个数, **假设所有 tile 等大**
      by_work   = 按每核实际工作量 (尾 tile 更小) 模拟分配后取 max/mean
    m 或 n 不被 m0/n0 整除时两者会分开: 1024 配 m0=160 时最后一行 tile 只有 64 行 (满 tile 的 40%),
    by_tiles 报 1.14 而实测 1.41, by_work 报 1.29 —— 靠 tile 数会**低估**不均衡。
    (by_work 仍忽略 swizzle 的分配顺序, 故对不整除的点还有个位数偏差; 整除时两者相同。)
    """
    mLoop = -(-m // m0)
    nLoop = -(-n // n0)
    tiles = mLoop * nLoop
    waves = -(-tiles // cores)
    used = min(tiles, cores)
    eff_tiles = tiles / (waves * cores)
    #  按工作量: 尾 tile 的实际行/列数更少
    mAct = [min(m0, m - i * m0) for i in range(mLoop)]
    nAct = [min(n0, n - j * n0) for j in range(nLoop)]
    load = [0.0] * cores
    for idx in range(tiles):
        i, j = divmod(idx, nLoop)
        load[idx % cores] += mAct[i] * nAct[j]
    mean = sum(load) / cores if cores else 0.0
    imb_work = (max(load) / mean) if mean else 1.0
    gm_l1 = (m * k * nLoop + k * n * mLoop) * ab_bytes if k else 0
    return {"m0": m0, "n0": n0, "mLoop": mLoop, "nLoop": nLoop, "tiles": tiles,
            "waves": waves, "used_cores": used,
            "wave_eff": eff_tiles, "cube_max_over_mean": imb_work,
            "imbalance_by_tiles": 1.0 / eff_tiles if eff_tiles else 0.0,
            "imbalance_by_work": imb_work,
            "core_util": used / cores, "gm_to_l1_MB": gm_l1 / 2**20}


def wave_pareto(m: int, n: int, k: int, cores: int = 24, l0c: int = 128 * 1024,
                c_bytes: int = 4, ab_bytes: int = 2, axis_max: int = 256,
                align: int = 16, row_bytes: int = 512) -> list:
    """枚举候选 (m0,n0), 返回 (均衡好, GM->L1 小) 两目标下的**非支配解**。

    ★n0 必须让输出/B 的一行凑满 row_bytes (默认 512B) —— **实测**: 1024^3 上把 n0 从 256 改成 176
      (行长 512B -> 352B), GM->L1 字节只多 8%, 而 mte2 时间涨 161% (9.24 -> 24.13us), 端到端翻倍。
      按字节数排候选会把这种点排在前面, 所以对齐是硬筛不是打分项。
      (这条目前只有 176 vs 256 一个对照点, n0=128 未测 —— 见 REFS W7_n0_row_alignment。)
    """
    n0_step = max(align, row_bytes // ab_bytes)
    cand = []
    for m0 in range(align, axis_max + 1, align):
        for n0 in range(n0_step, axis_max + 1, n0_step):
            if m0 * n0 * c_bytes > l0c:          # F1_L0C
                continue
            cand.append(wave_check(m, n, m0, n0, k, cores, ab_bytes))
    front = []
    for r in cand:
        dominated = any(o["imbalance_by_work"] <= r["imbalance_by_work"]
                        and o["gm_to_l1_MB"] <= r["gm_to_l1_MB"]
                        and (o["imbalance_by_work"] < r["imbalance_by_work"]
                             or o["gm_to_l1_MB"] < r["gm_to_l1_MB"])
                        for o in cand)
        if not dominated:
            front.append(r)
    front.sort(key=lambda r: (r["imbalance_by_work"], r["gm_to_l1_MB"]))
    return front



# ---------------------------------------------------------------------------
# W8 split-k 的收益上界与代价 —— 与 W7 成对: W7 说"最后一波空了多少核", W8 说"用 K 维去填值不值"
#
# 为什么 split-k 是这类问题的唯一无损手段: 缩 tile 换均衡会让共享操作数多流几遍
#   (m0 缩小 -> B 多读, n0 缩小 -> A 多读), 而**沿 K 切不改变 A/B 的总搬入量** ——
#   每个分片只读 K/s 的行/列, s 份合起来正好是原来那一份。代价改为出现在输出侧:
#   每个输出 tile 要写 s 份 fp32 部分和再归约, 即 L0C->GM 写流量 x s。
#
# ★910B 上 split-k 的既有写法 (peerref): mat_mul_v3/op_kernel/mat_mul_multi_core_splitk_kernel.h
#   —— 各核按 blockIdx 解出 (mIdx,nIdx,kIdx), 算完用 CrossCoreSetFlag<0,PIPE_MTE3> 做**全核屏障**,
#   再 GetTensorC(..., enAtomic=1) 原子累加。实测 (M=256 K=16384 N=256, aclnnGemm 走 MatMulV3):
#   BlockDim=24 (2 个输出 tile 却用满 24 核 -> K 切 12 份), 核间不均衡降到 1.13,
#   L0C->GM 写 459KB/核 (输出本身只有 256KB), fixpipe 4.10us / mte3 2.79us。
#
# 本函数只给**上界与代价**, 不预测端到端 —— 归约写与 cube 的重叠程度不可静态推定。
# 尤其: 均匀 split-k 会把 core-local 的 CV pingpong 流水拆掉 (vector 必须等所有分片写完),
#   那一项的代价本模型算不出来, 必须实测。只切尾波的变体保留主流水, 见 tail_only。

def splitk_plan(m: int, n: int, k: int, m0: int, n0: int, cores: int = 24,
                s_max: int = 8, k_align: int = 16, c_bytes: int = 4) -> dict:
    """枚举 split 份数 s, 报 cube 有效波数与归约写代价。tail_only 只切最后一波。"""
    mLoop, nLoop = -(-m // m0), -(-n // n0)
    tiles = mLoop * nLoop
    base_waves = -(-tiles // cores)
    out = {"tiles": tiles, "base_waves": base_waves,
           "base_cube_units": base_waves * 1.0,       # 以"一个满 tile 的 cube 时间"为单位
           "l0c_write_MB": tiles * m0 * n0 * c_bytes / 2**20, "uniform": [], "tail_only": []}
    tail = tiles - (base_waves - 1) * cores          # 最后一波真正在跑的核数
    for s in range(2, s_max + 1):
        if k // s < k_align:                          # 每份至少一个对齐粒度
            continue
        # 均匀切: 单元数 tiles*s, 每个单元的 cube 时间是满 tile 的 1/s
        u_waves = -(-(tiles * s) // cores)
        out["uniform"].append({
            "s": s, "units": tiles * s, "waves": u_waves,
            "cube_units": u_waves / s,                # 有效 cube 时间 (单位同上)
            "gain_pct": (1 - (u_waves / s) / base_waves) * 100,
            "extra_l0c_write_MB": tiles * (s - 1) * m0 * n0 * c_bytes / 2**20,
            "k_per_split": (k // s) // k_align * k_align,
        })
        # 只切尾波: 前面几波照旧, 最后一波的 tail 个 tile 各切 s 份
        t_units = -(-(tail * s) // cores)             # 尾波拆开后占几波
        out["tail_only"].append({
            "s": s, "tail_tiles": tail, "tail_waves": t_units,
            "cube_units": (base_waves - 1) + t_units / s,
            "gain_pct": (1 - ((base_waves - 1) + t_units / s) / base_waves) * 100,
            "extra_l0c_write_MB": tail * (s - 1) * m0 * n0 * c_bytes / 2**20,
            "k_per_split": (k // s) // k_align * k_align,
        })
    return out


def splitk_report(m: int, n: int, k: int, m0: int, n0: int, cores: int = 24) -> None:
    r = splitk_plan(m, n, k, m0, n0, cores)
    print(f"m={m} n={n} k={k} | m0={m0} n0={n0} cores={cores}")
    print(f"  不切: tiles={r['tiles']}, 波={r['base_waves']}, 尾波在跑的核="
          f"{r['tiles'] - (r['base_waves']-1)*cores}/{cores}, L0C->GM 写 {r['l0c_write_MB']:.1f} MB")
    for name, rows in (("均匀切 (会拆掉 CV 流水)", r["uniform"]), ("只切尾波 (保留主流水)", r["tail_only"])):
        print(f"  {name}:")
        print(f"    {'s':>2s}{'每份K':>7s}{'有效cube波':>11s}{'cube上界':>10s}{'额外L0C->GM写':>15s}")
        for x in rows:
            print(f"    {x['s']:>2d}{x['k_per_split']:>7d}{x['cube_units']:>11.2f}"
                  f"{x['gain_pct']:>9.1f}%{x['extra_l0c_write_MB']:>13.1f} MB")
    print("  ★cube 上界不是端到端收益: cube 只占 aic_time 的三成左右, 且归约写与 cube 的重叠程度"
          "不可静态推定。均匀切还要额外付'CV 流水被拆掉'的代价, 本模型算不出来。")


#  ---- W9: 输出宽度 N 的行跨距对齐 (ND 格式下 B/C/Y 的每行起始地址) --------------------
#  ND 布局里 B 是 K x N, 行跨距 = N * sizeof(dtype) 字节。硬件按 256B 的粒度取,
#  跨距不是 256B 的整数倍时**每一行的起始地址逐行漂移**, Nd2Nz 的转换吞吐随之塌掉:
#  实测 aic_l1_write_bw 从 35.85 掉到 14.24 GB/s (40%), 而 GM->L1 的字节数一个不差。
#  注意这与 tiling 无关 —— 换过九种 n0 (含能整除 N 的) 全部无改善, 是数据布局决定的。
#  ★256B 是**硬件访问粒度**, 与 dtype 无关; 变的是"一行要多少个元素才凑够 256B":
#      fp16/bf16 (2B) -> 128 个元素   fp32 (4B) -> 64 个   int8 (1B) -> 256 个
#    fp32 已验证 (见 REFS): 周期确实是 mod 64。int8 未验证。
_ALIGN_BYTES = 256


def align_check(n: int, dtype_bytes: int = 2) -> dict:
    """判 N 的行跨距对齐, 并给出补齐建议。"""
    row = n * dtype_bytes
    rem = row % _ALIGN_BYTES
    step = _ALIGN_BYTES // dtype_bytes          # fp16 -> 128 列
    half = _ALIGN_BYTES // 2
    n_pad = ((n + step - 1) // step) * step
    #  余数落在半线 (128B) 上时只掉一档, 其余掉两档 —— 实测 1088(余128) 30.4us 对
    #  1024(余0) 27.4us, 而 1008(余224) 是 47.8us。
    grade = "对齐" if rem == 0 else ("半对齐" if rem == half else "未对齐")
    return {"n": n, "dtype_bytes": dtype_bytes, "row_bytes": row, "rem": rem,
            "grade": grade, "n_pad": n_pad, "extra_cols": n_pad - n,
            "extra_compute_pct": (n_pad / n - 1) * 100 if n else 0.0}


#  ★哪个维度进哪条行跨距 —— **随转置翻转**, 这是 W9 的全部依据:
#      A 是 [M,K] 或 (transA) [K,M]  -> 行跨距 = K 或 **M**
#      B 是 [K,N] 或 (transB) [N,K]  -> 行跨距 = N 或 **K**
#      C/Y 恒为 [M,N]                -> 行跨距恒为 N
#    一个维度塌不塌, 只看它**进了几条行跨距**。四种转置组合的实测 (1024^3 fp16,
#    把某一维从 1024 改成 1000):
#
#      tA/tB   改N     改K     改M      M/K/N 各进几条
#      0/0    +79%    +18%     -4%     M:0  K:1(A)    N:2(B,C/Y)
#      0/1    +21%    +70%     -2%     M:0  K:2(A,B)  N:1(C/Y)
#      1/0    +84%     -6%    +53%     M:1(A)  K:0    N:2(B,C/Y)
#      1/1    +32%    +59%    +61%     M:1(A)  K:1(B) N:1(C/Y)
#
#    规律非常干净: **0 条 -> 不塌 (-6%~-2%); 1 条 -> +18%~+61%; 2 条 -> +70%~+84%**。
#    ★2026-08-29 修正: 本判据此前硬编码"M 不需要补齐", 那句话**只在 transA=0 时成立** ——
#      tA=1 时 M 进了 A 的行跨距, 实测塌 +53%~+61%。判据说"某某永远不必管"之前,
#      先问一遍它在别的配置下是不是也这样。
_STRIDE_DEGRADE = {0: ("不塌", "-6%~-2%, 噪声内"),
                   1: ("会塌", "+18%~+61%"),
                   2: ("塌得最重", "+70%~+84%")}


def stride_owners(trans_a: bool = False, trans_b: bool = False) -> dict:
    """返回 {维度: [它进了哪些张量的行跨距]}。"""
    owners = {"M": [], "K": [], "N": []}
    owners["M" if trans_a else "K"].append("A")
    owners["K" if trans_b else "N"].append("B")
    owners["N"].append("C/Y")
    return owners


def align_report(n: int = 0, dtype_bytes: int = 2, k: int = 0, m: int = 0,
                 trans_a: bool = False, trans_b: bool = False) -> int:
    """按转置组合判 M/K/N 三维的行跨距对齐。"""
    owners = stride_owners(trans_a, trans_b)
    vals = {"M": m, "K": k, "N": n}
    print(f"转置: transA={int(trans_a)} transB={int(trans_b)}  -> "
          f"A 的行跨距 = {'M' if trans_a else 'K'}, B 的行跨距 = {'K' if trans_b else 'N'}, "
          f"C/Y 的行跨距 = N")
    hit = 0
    for dim in ("M", "K", "N"):
        val = vals[dim]
        if not val:
            continue   # ssok: 调用方没给这一维 (值为 0), 它不在本次判的三维里
        own = owners[dim]
        cnt = min(len(own), 2)
        verdict, degrade = _STRIDE_DEGRADE[cnt]
        if not own:
            print(f"{dim}={val} -> **不进任何行跨距, 不需要补齐** (实测 {degrade})")
            continue
        r = align_check(val, dtype_bytes)
        print(f"{dim}={val} ({dtype_bytes}B/元素) -> 行跨距 {r['row_bytes']}B, "
              f"mod {_ALIGN_BYTES} = {r['rem']}  [{r['grade']}]   "
              f"(进 {len(own)} 条: {'/'.join(own)}, {verdict})")
        if r["grade"] == "对齐":
            print("  ✓ 每行起始都落在 256B 边界上, 无需补齐。")
            continue
        hit += 1
        #  ★幅度的标定来源要说清楚 —— 它来自 fp16 的 GemmV3, 换算子/换 dtype 幅度会变。
        #    把一个算子的数字当成通用值报出去, 是判据最容易犯的越权。
        if dtype_bytes == 2:
            print(f"  ★实测劣化 {degrade} (fp16 GemmV3 标定: 四种转置组合共 12 个点)")
        else:
            print(f"  ★方向确定会塌, 但**幅度未在本 dtype 上标定**。"
                  f"括号里的 {degrade} 来自 fp16 的 GemmV3, 仅供量级参考。")
            print(f"    已知的另一个点: fp32 输入不走 GemmV3 (aclnnAddmm 拆成 MatMulV2 + Add, "
                  f"ws=0 是判别特征), 在 MatMulV2 上实测 **+47%** (29.44 -> 43.29us), "
                  f"而同一次里 Add 段纹丝不动。")
        print(f"  成因: 行跨距非 {_ALIGN_BYTES}B 倍数 -> 每行起始地址逐行漂移 -> Nd2Nz 转换吞吐"
              f"掉到 40% (aic_l1_write_bw 35.85 -> 14.24 GB/s, 而 GM->L1 字节数逐位不变)。")
        print(f"  ★不是 tiling 问题: 换过九种 n0 (含能整除的、尾块为满的) 全部无改善。")
        print(f"  建议: {dim} 补到 {r['n_pad']} (多 {r['extra_cols']} 个 = "
              f"{r['extra_compute_pct']:.1f}% 计算)")
        if dim == "N" and not trans_b:
            print(f"        实测净收益 +32%~49% (checksum 逐位不变); 多算 11.6% 的那档仍净赚 32%。")
        elif dim == "K" and not trans_b:
            print(f"        实测净收益 +16%~17% (四点); 补法: A 的多出列与 B 的多出行都补 0。")
        else:
            print(f"        该组合下的补齐收益未单独实测, 按劣化幅度估上界; 补前先跑一次对照。")
    if hit:
        print("")
        print("★补齐不是唯一手段, 而且往往不是最好的那个 (它要多算):")
        print("  1. **改用 NZ 格式的输入** —— NZ 是分形存储, 没有行跨距这个概念, 本判据整个不适用。")
        print("     gemm_v3 实测: N=1023 上 aic_mte2 从 36.46 掉回 8.84us (对齐时是 9.35),")
        print("     端到端 50.2 -> 35.5 (+29%), 而**计算量一个字节都不多**。")
        print("     ★但它是**条件性**优化, 不能无条件开: 40 个 shape 的泛化性扫描显示")
        print("       N 不对齐时 +26.7%~49.2%, 而 **N 对齐时 -5%~-11%** (fixpipe 涨 79%, 成因未定位)。")
        print("       正确用法是按 N 是否对齐来决定用不用 NZ —— 而格式由调用方给, 所以判断也在调用方。")
        print("     代价: kernel 要有 formatB=NZ 的模板实例化 (GemmV3 的 NZ 分支写好了但从没被实例化,")
        print("     所有实例都用默认的 ND), 调用方要传 NZ 张量 —— 权重可离线转一次, 之后零成本。")
        print("  2. **查一遍算子内部有没有自己造出来的非对齐写出**: L0C->GM 落 workspace 时若用")
        print("     `dstStride = nTileActual` 这种紧凑跨距, N 不整除 n0 的最后一个 tile 就非对齐了。")
        print("     gemm_v3 实测 fixpipe 5.28 -> 3.47us; 改成按标称 nTile_ 写 (读侧的行偏移与 srcGap")
        print("     必须成对改), 是**纯内部改动, 不需要调用方配合**。")
        print("  两条叠加在**不对齐 shape** 上实测 28%~40% (checksum 逐位不变), 优于补齐的 32%~49%")
        print("  (因为不多算); 但对齐 shape 上 NZ 是负收益, 只该开 wsstride 那一条。")
    if not (n or k or m):
        print("用法: --align --n <N> [--k <K>] [--m <M>] [--trans-a] [--trans-b]")
        return 0
    return 1 if hit else 0

def align_selftest() -> int:
    """W9 负例自检: 已知会塌的 N 必须被判未对齐, 已知快的不许误报。

    取值全部来自 2026-08-29 的密集扫描, 每条后面是实测耗时 (M=K=1024, fp16)。
    """
    cases = [  # (N, 期望等级, 实测us, 备注)
        (1024, "对齐", 27.4, "基准"),
        (1152, "对齐", 27.6, "非 2 的幂但 mod 128 == 0"),
        (896, "对齐", 26.1, ""),
        (1408, "对齐", 27.7, ""),
        (1088, "半对齐", 30.4, "余 128B, 只掉一档"),
        (960, "半对齐", 30.9, ""),
        (1008, "未对齐", 47.8, "余 224B"),
        (1023, "未对齐", 49.2, "余 254B, 离对齐只差一列"),
        (1040, "未对齐", 41.1, "余 32B"),
        (1012, "未对齐", 51.6, "余 232B, 最慢"),
    ]
    bad = 0
    print(f"=== W9 行跨距对齐 (10 条, 取值来自真机扫描) ===")
    for n, want, us, note in cases:
        got = align_check(n)["grade"]
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} N={n:<5} 期望 {want:<6} 实际 {got:<6} "
              f"(实测 {us}us){' ' + note if note else ''}")
    #  ★归属随转置翻转 —— 12 个真机点, 每个都断言"进几条跨距"与实测劣化档位一致。
    #    判据若退回"M 永远不必补齐"这种无条件表述, 下面 tA=1 的两行会直接报未过。
    #    (期望档位: 0 条 = 不塌; 1 条 = +18%~+61%; 2 条 = +70%~+84%)
    trans_cases = [
        # (tA, tB, 维度, 期望进几条, 实测劣化%)
        (0, 0, "N", 2, +79), (0, 0, "K", 1, +18), (0, 0, "M", 0, -4),
        (0, 1, "N", 1, +21), (0, 1, "K", 2, +70), (0, 1, "M", 0, -2),
        (1, 0, "N", 2, +84), (1, 0, "K", 0, -6), (1, 0, "M", 1, +53),
        (1, 1, "N", 1, +32), (1, 1, "K", 1, +59), (1, 1, "M", 1, +61),
    ]
    print("  --- 转置下的归属 (12 个真机点) ---")
    for ta, tb, dim, want_cnt, meas in trans_cases:
        got = len(stride_owners(bool(ta), bool(tb))[dim])
        cnt_ok = got == want_cnt
        #  劣化档位必须与条数相符: 0 条不许塌, >=1 条必须塌
        band_ok = (meas < 10) if want_cnt == 0 else (meas >= 10)
        ok = cnt_ok and band_ok
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} tA={ta} tB={tb} {dim}: 进 {got} 条"
              f"(期望 {want_cnt}), 实测 {meas:+d}%")

    #  ★dtype 周期: 256B 是硬件粒度, 一行要多少元素才凑够它随 dtype 变
    for db, period, name in ((2, 128, "fp16/bf16"), (4, 64, "fp32"), (1, 256, "int8")):
        r = align_check(period, db)
        r2 = align_check(period // 2, db)
        ok = r["grade"] == "对齐" and r2["grade"] == "半对齐"
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} {name} ({db}B): 周期应为 mod {period} "
              f"—— N={period} 判 {r['grade']}, N={period//2} 判 {r2['grade']}")

    #  补齐建议必须落在下一个对齐点上, 且不能倒退
    for n in (1000, 1008, 1023, 1032, 1040, 1056):
        r = align_check(n)
        if r["n_pad"] % 128 != 0 or r["n_pad"] < n:
            print(f"  **未过** N={n} 的补齐建议 {r['n_pad']} 不是对齐点或小于 N")
            bad += 1
    print(f"W9 自检: {len(cases) + 6 + len(trans_cases) + 3} 条, {bad} 条未过")
    return 1 if bad else 0


def wave_selftest() -> int:
    """判据负例自检: 8 个 shape (4 个建立点 + 4 个样本外点) 预测与实测偏差须 <=5%."""
    # (m, k, n, m0, n0, 实测 cube max/mean) —— 出处见上方实证表
    #  建立判据时用的四点 (两端: 0.52 / 0.667 / 1.0 / 1.0)
    pts = [(1024, 1024, 1024, 128, 256, 1.55),
           (1536, 1024, 1024, 128, 256, 1.015),
           (3072, 1024, 1024, 128, 256, 1.015),
           (640, 1024, 1280, 128, 256, 1.96),
           #  ★样本外四点 (中间段 0.583 / 0.889 / 0.917, 建立判据时没有的区间) —— 判据不能只用
           #    建立它的数据自证。全部固定 n=k=1024 只动 m, 单变量。
           (768, 1024, 1024, 128, 256, 1.00),
           (896, 1024, 1024, 128, 256, 1.74),
           (1408, 1024, 1024, 128, 256, 1.108),
           (2048, 1024, 1024, 128, 256, 1.140),
           #  ★改 tile 尺寸后的实测点 (2026-08-27 wave1/wave2 消融). 前八点 m/n 恰好都被 m0/n0 整除,
           #    掩盖了"尾 tile 更小"这件事; 下面三点里 wave2 那个不整除, 按 tile 数会低估到 1.14。
           (1024, 1024, 1024, 96, 256, 1.14),    # wave1
           (896, 1024, 1024, 80, 256, 1.07),     # wave1
           (640, 1024, 1280, 80, 256, 1.20)]     # wave1
    bad = 0
    for m, k, n, m0, n0, meas in pts:
        r = wave_check(m, n, m0, n0, k)
        pred = r["imbalance_by_work"]
        dev = abs(pred - meas) / meas
        ok = dev <= 0.05
        bad += 0 if ok else 1
        print(f"  m={m:5d} n={n:5d} m0={m0:3d} n0={n0}: tiles={r['tiles']:3d} 波={r['waves']} "
              f"预测(按工作量)={pred:.2f} 实测={meas:.2f} 偏差={dev*100:4.1f}% "
              f"{'ok' if ok else '**MISS**'}" +
              (f"  [按tile数只有 {r['imbalance_by_tiles']:.2f}]"
               if abs(r["imbalance_by_tiles"] - pred) > 0.02 else ""))
    print(f"W7 自检: {len(pts)-bad}/{len(pts)} 通过")
    return 1 if bad else 0


def enumerate_space(prob, soc=None, topn=12):
    out = []
    for bM, bN, bK in itertools.product((64, 128, 256), (128, 256, 512), (64, 128, 256, 512)):
        for sM, sN in itertools.product((1, 2, 4), (1, 2, 4)):
            kst = -(-prob["k"] // bK)
            for sKa, sKb in itertools.product({1, 2, 4, kst}, {1, 2, 4, kst}):
                if sKa % sKb and sKb % sKa:      # F6: 互不整除即未定义行为, 不进候选
                    continue
                for dbA, dbB in ((2, 2),):       # 只列双缓冲: 单缓冲实测慢 17.7% (见 W_单缓冲), 无需枚举
                    cfg = {"baseM": bM, "baseN": bN, "baseK": bK, "stepM": sM, "stepN": sN,
                           "stepKa": sKa, "stepKb": sKb,
                           "depthA1": sM * sKa * dbA, "depthB1": sN * sKb * dbB}
                    r = check(cfg, prob, soc)
                    if r["feasible"] and not r["warns"]:
                        out.append(r)
    out.sort(key=lambda x: x["total_GB"])
    # 同一 (M1,N1) 只留搬运量最小的一个代表, 免得刷屏
    seen, uniq = set(), []
    for r in out:
        key = (r["M1"], r["N1"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq[:topn]


def fmt(r):
    c = r["cfg"]
    return (f"baseM={c['baseM']:<4d} baseN={c['baseN']:<4d} baseK={c['baseK']:<4d} "
            f"stepM={c['stepM']} stepN={c['stepN']} stepKa={c['stepKa']:<2d} stepKb={c['stepKb']:<2d} "
            f"dA1={c['depthA1']:<2d} dB1={c['depthB1']:<2d} | L1 {r['L1_KB']:3d}KB | "
            f"M1={r['M1']:<4d} N1={r['N1']:<4d} L0C活{r['live_l0c_KB']:3d}KB | "
            f"A {r['a_GB']:5.2f} + B {r['b_GB']:5.2f} = {r['total_GB']:5.2f} GB")


#  ★判据的负例自检 —— 与 --refs 是一套: --refs 说"这条判据的实证是什么",
#    本函数说"这条判据现在还抓得住吗"。两者合起来判据才真的可被证伪 (dual-loop: 方法必须押注)。
#    每条给一个**已知会挂**的配置, 断言对应判据必须命中。工具改坏了这里会立刻红。
SELFTEST_CASES = [
    ("F1 容量: baseM*baseN*4 超 L0C",
     dict(baseM=256, baseN=256, baseK=128), "L0C"),
    ("C2 UB 行对齐: vecBaseM=4096/(256*3)=5 不是 8 的倍数",
     dict(baseM=128, baseN=256, baseK=128, nBatch=3, ubCalSize=4096), "C2"),
    ("C1 CV 深度: parallNum < 2*nBatch",
     dict(baseM=128, baseN=256, baseK=128, nBatch=4, ubCalSize=4096, parallNum=4), "C1"),
]


def selftest(prob=None, soc=None):
    #  ★自检必须是**确定性**的: 用内置表的标定值, 不走本机探测。
    #    否则同一份代码在 910B2 上过、在别的机器上挂, 分不清是判据坏了还是机器不同。
    #    (2026-08-29 实撞: 把 selftest 接到 socspec 后, 传的旧键名被解析成另一颗真实芯片
    #     Ascend910B, 它的 ini 缺本模块要的字段, 三条自检全挂。)
    #  ★自检跑的是**这几个实测点当初测出来的那颗芯片**, 不是本机 —— 换机器后自检仍应同号,
    #    否则分不清"判据坏了"与"机器不同"。型号取自内置标定表的唯一键, 不在代码里写死名字。
    soc = soc or next(iter(CAP))
    prob = prob or {"m": 102400, "n": 4096, "k": 1536, "e": 400, "avg_m": 256,
                    "c_bytes": 4, "staging_l2_hit": 0.98}
    bad = 0
    print(f"=== mmplan 判据负例自检 ({len(SELFTEST_CASES)} 条) ===")
    for name, cfg, want in SELFTEST_CASES:
        c = dict(cfg)
        c.setdefault("stepM", 1); c.setdefault("stepN", 1)
        c.setdefault("stepKa", 1); c.setdefault("stepKb", 1)
        c.setdefault("depthA1", c["stepM"] * c["stepKa"] * 2)
        c.setdefault("depthB1", c["stepN"] * c["stepKb"] * 2)
        c["c_pos"] = "gm"
        try:
            r = check(c, prob, soc)
        except Exception as e:
            print(f"  ★挂了 {name}: {type(e).__name__}: {e}"); bad += 1; continue
        txt = " ".join(str(x) for x in (r.get("fails") or []) + (r.get("warns") or []))
        ok = (want in txt) and not r.get("feasible", True)
        print(f"  {'过' if ok else '★未命中'} {name}   (期望命中 {want!r})")
        if not ok:
            print(f"      实际: feasible={r.get('feasible')} fails={r.get('fails')}")
            bad += 1
    print(f"mmplan 自检: {len(SELFTEST_CASES)} 条, {bad} 条未过")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int)
    ap.add_argument("--n", type=int)
    ap.add_argument("--k", type=int)
    ap.add_argument("--e", type=int, default=1)
    ap.add_argument("--avg-m", type=int)
    ap.add_argument("--c-bytes", type=int, default=4, help="matmul 输出元素字节 (int32 累加=4, fp16=2)")
    ap.add_argument("--config", help="逗号分隔 k=v, 检查单个配置")
    ap.add_argument("--measured-us", dest="measured_us", type=float, default=None,
                    help="实测微秒; 给了才报**当前是地板的几倍** (判词走 core.hwfloor)")
    ap.add_argument("--enumerate", action="store_true", help="枚举可行空间, 按搬运量下界排序")
    ap.add_argument("--soc", default=None,
                    help="芯片**型号** (默认按本机探测)。注意 ascend910b 是 CANN 的目录系列名, 作型号它指另一颗芯片 (arch 1001, 30 个一体核)")
    ap.add_argument("--c-pos", default="gm", choices=["gm", "ub"],
                    help="matmul C 矩阵位置; ub 需要 SoC 有 cube->vec 直连 (910B 没有)")
    ap.add_argument("--staging-l2-hit", type=float, default=0.98,
                    help="CV staging 被 L2 吸收的比例 (默认 0.98 = gmfr 真机实测 aic_write_hit 99.2%%/aiv_read_hit 97.5%%)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--refs", action="store_true", help="判据出处表: 每条判据的实证与最后验证日期")
    ap.add_argument("--selftest", action="store_true", help="判据负例自检: 已知会挂的配置必须被抓住")
    ap.add_argument("--align", action="store_true",
                    help="W9: 判输出宽度 N 的行跨距对齐, 给出补齐建议 (配 --n, 可选 --dtype-bytes)")
    ap.add_argument("--dtype-bytes", type=int, default=2, help="元素字节数 (fp16/bf16=2, fp32=4)")
    ap.add_argument("--trans-a", action="store_true", help="A 是 [K,M] (行跨距变 M)")
    ap.add_argument("--trans-b", action="store_true", help="B 是 [N,K] (行跨距变 K)")
    ap.add_argument("--wave", action="store_true",
                    help="W7 波量化: 判 tile 数与核数是否成整数倍 (核间负载均衡), 并给 (m0,n0) 帕累托前沿")
    ap.add_argument("--m0", type=int, help="--wave/--splitk: 只看这一个 tile 尺寸")
    ap.add_argument("--splitk", action="store_true",
                    help="W8: 沿 K 切分的收益上界与归约代价 (与 --wave 成对看)")
    ap.add_argument("--n0", type=int)
    ap.add_argument("--cores", type=int, default=24, help="aic 核数")
    a = ap.parse_args()
    if a.refs:
        hot = print_refs("mmplan", REFS)
        print("\n★F3 的搬运量绝对值是 disputed —— 只用它排序, 判收益一律实测 (core/ablate)。")
        return 0
    if a.selftest:
        rc = selftest(soc=a.soc)
        rc = align_selftest() or rc
        print("\n--- W7 波量化 ---")
        return rc or wave_selftest()
    if a.splitk:
        splitk_report(a.m, a.n, a.k, a.m0 or 128, a.n0 or 256, a.cores)
        return 0
    if a.align:
        return align_report(a.n, a.dtype_bytes, a.k, a.m, a.trans_a, a.trans_b)
    if a.wave:
        cb = a.c_bytes if a.c_bytes else 4
        if a.m0 and a.n0:
            r = wave_check(a.m, a.n, a.m0, a.n0, a.k, a.cores)
            print(f"m={a.m} n={a.n} k={a.k} | m0={r['m0']} n0={r['n0']} -> "
                  f"tiles={r['tiles']} ({r['mLoop']}x{r['nLoop']}), 波={r['waves']}, "
                  f"用核={r['used_cores']}/{a.cores}")
            print(f"  有效并行度 wave_eff = {r['wave_eff']:.3f}  ->  理论每核 cube max/mean = {1/r['wave_eff']:.2f}")
            print(f"  GM->L1 搬入 {r['gm_to_l1_MB']:.1f} MB")
            if r["wave_eff"] < 0.85:
                print(f"  [W7] 波量化损失 {(1-r['wave_eff'])*100:.0f}% —— 最后一波只有 "
                      f"{r['tiles'] - (r['waves']-1)*a.cores}/{a.cores} 个核在跑, 其余空转到 kernel 结束")
                #  ★这一句必须跟着上面那个数字一起出去。W7 量的是**均衡这一维的上界**,
                #    而换 m0/n0 会同时动到访问模式与 kFrag 整除性, 那两维本模块看不见。
                print(f"       ★这是均衡那一维的上界, **不是净收益**: 换 m0/n0 会同时改 B 每行的"
                      f"字节数与 kFrag 对 k0 的整除性。")
                print(f"       gemm_v3 实测: wave_eff 0.667->1.000 的 128x192 慢 73%; "
                      f"分形系数最小的 176x176 慢 155%; 十个 tile 点扫下来默认的 128x256 最优。")
                print(f"       正确用法是**排除** —— 上界都不够大的候选不必做; 够大的一律实测。")
            if a.json:
                print(json.dumps(r, ensure_ascii=False))
            return 0
        front = wave_pareto(a.m, a.n, a.k, a.cores, c_bytes=cb)
        print(f"m={a.m} n={a.n} k={a.k} cores={a.cores} —— (wave_eff 高, GM->L1 小) 帕累托前沿:\n")
        print(f"{'m0':>5s}{'n0':>5s}{'tiles':>7s}{'波':>4s}{'用核':>5s}{'wave_eff':>10s}"
              f"{'max/mean':>10s}{'GM->L1(MB)':>12s}")
        for r in front:
            print(f"{r['m0']:>5d}{r['n0']:>5d}{r['tiles']:>7d}{r['waves']:>4d}{r['used_cores']:>5d}"
                  f"{r['wave_eff']:>10.3f}{1/r['wave_eff']:>10.2f}{r['gm_to_l1_MB']:>12.1f}")
        print("\n★前沿不是排名: 缩小 m0 改善均衡的同时 B 会多流几遍 (mLoop 变大), 而 mte2 常已是最大单项.")
        print("  两者净和不可静态推定 —— 前沿上选谁**必须**用 core/ablate 真机消融定.")
        if a.json:
            print(json.dumps(front, ensure_ascii=False))
        return 0
    prob = {"m": a.m, "n": a.n, "k": a.k, "e": a.e, "avg_m": a.avg_m or (a.m // a.e if a.e else a.m),
            "c_bytes": a.c_bytes, "staging_l2_hit": a.staging_l2_hit}

    if a.config:
        cfg = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.config.split(",")}
        cfg.setdefault("stepM", 1); cfg.setdefault("stepN", 1)
        cfg.setdefault("stepKa", 1); cfg.setdefault("stepKb", 1)
        # depth 未给时按惯例推 (stepM·stepKa·双缓冲 / stepN·stepKb·双缓冲), 与 host 侧默认一致
        cfg.setdefault("depthA1", cfg["stepM"] * cfg["stepKa"] * 2)
        cfg.setdefault("depthB1", cfg["stepN"] * cfg["stepKb"] * 2)
        cfg["c_pos"] = a.c_pos
        r = check(cfg, prob, a.soc)
        print(fmt(r))
        #  ★判词走 core.hwfloor.verdict_lines (全仓唯一出处, 2026-09-18 合并):
        #    本模块只负责算 cube_floor_us / dram_floor_us 这两个领域量。
        from .hwfloor import verdict_lines
        _fl = max(r["cube_floor_us"], r.get("dram_floor_us") or 0) / 1e3
        for ln in verdict_lines(_fl, (a.measured_us / 1e3) if getattr(a, "measured_us", None) else None,
                                kind="capability",
                                bound_by=("cube" if r["cube_floor_us"] >= (r.get("dram_floor_us") or 0)
                                          else "DRAM")):
            print(ln)
        print(f"  cube 算力地板 {r['cube_floor_us']:.0f} us | MTE2 估 {r['mte2_est_us']:.0f} us "
              f"({r['mte2_est_us']/r['cube_floor_us']:.2f}x 算力地板)")
        print(f"  CV staging (910B 无直连, 必经 GM): {r['staging_GB']:.2f} GB 往返, 其中 "
              f"{r['staging_dram_GB']:.2f} GB 落 DRAM | DRAM 合计 {r['dram_GB']:.2f} GB "
              f"-> 带宽地板 {r['dram_floor_us']:.0f} us")
        for f in r["fails"]:
            print(f"  [FAIL] {f}")
        for w in r["warns"]:
            print(f"  [WARN] {w}")
        if r["feasible"]:
            print("  可行 —— 但搬运量绝对值不可信 (见 CALIBRATION 误差带), 须用 sweep_tiling.py 真机验证")
        if a.json:
            print(json.dumps(r, ensure_ascii=False, default=str))
        return 0 if r["feasible"] else 1

    if a.enumerate:
        rows = enumerate_space(prob, a.soc)
        print(f"问题规模 m={a.m} n={a.n} k={a.k} e={a.e} avg_m={prob['avg_m']}, "
              f"输出 {a.c_bytes}B/元素; cube 算力地板 "
              f"{(a.m*a.n*a.k)/(24*8192*1.8e9)*1e6:.0f} us")
        print("可行配置 (按 GM→L1 下界排序; 同一 (M1,N1) 只列代表):\n")
        for r in rows:
            print("  " + fmt(r))
        print(f"\n误差带 (务必看): 本列的下界对已实测的两个配置分别偏 -10% 与 -31% —— 只可用于**排序与筛除**, "
              f"不可当预测. 选头部几个交 sweep_tiling.py 真机测.")
        return 0
    ap.error("需 --config 或 --enumerate")


if __name__ == "__main__":
    # ★必须把返回码交给 sys.exit: 本工具是构建前闸门, 判 FAIL 却退出 0 等于没有闸门
    #   (2026-08-10 发现: --config 对违规配置打了 [FAIL] 但退出码仍是 0)
    sys.exit(main())
