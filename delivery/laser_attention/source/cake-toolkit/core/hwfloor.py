"""硬件下界 —— 优化开工之前必须先算出来的那个数。

沉淀动机 (2026-09-18 dubin 当面指出, sfag 复盘):
  我给 sfag 圈定的可动空间是 `原厂 det=1 37.544 - det=0非优化 31.482 = 6.325 ms`。
  **这个推导里没有一个字是硬件** —— 一个实测数减另一个实测数, 而两个数出自同一族
  有同样缺陷的实现。于是"回收了 46%"看起来是好成绩, 真实情况是交付版离硬件下界 21.2 倍。
  ★代价具体化: 逐 (q,j) 做 fp32 原子加的写流量是 9.13 GB = 8.589 ms, 而每行写一次
    只要 285 MB = 0.268 ms —— **32 倍**。光这一项的超额就大于我以为存在的全部空间。
    这个数十行算术、零次测量就能得到, 它直接指向"改分核让每行单 owner"。
  ★同一份计算还指出 gather 的 4.56 GB 对 192 MB 的 L2 有 16 倍复用潜力 -> 指向打包与分块。
  **两个最大的优化点都是算出来的, 不是测出来的。** 我一年也测不出它们, 因为
  profiler 只报"时间花在哪", 从不报"这件事本来要花多少"。

★★★ 本模块是**初版, 仍在迭代中。它给出的一切数字仅供参考, 不可作为优化终点。** ★★★

  为什么必须把这句话写在最前面 (2026-09-18 dubin 当面指出, 而且我当天就撞上了实例):
    本模块第一版的「编排下界 26.794 ms」是拿 socspec 里 **`dram_GBps` = None / status
    `uncalibrated`** 的那个量, 用 ini 的成本模型参数乘一个我自己编的 0.77 系数算出来的。
    仓里 `mmplan` 对同一个问题的态度是**直接拒绝**:
      「本芯片没有 dram_GBps 的**饱和实测**标定, 拒绝用模型值或别的芯片的数算 DRAM 下界。
        ini 的 ddr_rate 是编译器成本模型参数, 不是可达带宽。」
    而 CALIB 里记着这个量被撤回过**两次** (1070 错在负载类型, 1690 错在工作集全命中 L2),
    两次都是"把一次特定条件下的观测当成硬件常数"。
    **于是我造出了一个假下限, 而 loop 已经挂上, 下一轮就会拿它当靶子去优化。**

  所以本模块的每个数都必须带**出处标记**, 报告首尾都要有横幅:
    [实测标定] 本机饱和实测, 可用
    [模型值]   ini 成本模型参数 —— **不是可达带宽**, 只能排序不能定量
    [未标定]   该量在 CALIB 里是 None —— 任何据它算出的下界都是**假下界**
    [外推]     超出标定曲线的取值范围, 误差未知

  用它的人要记住三条:
    1. **下界为真, 不等于下界可达**。它是对偶的上界, 用来证伪"没空间了", 不用来定目标。
    2. 下界**比实测还大**时不是"贴住了", 是**模型错了** —— 先查带宽标定与流量口径。
    3. 判据本身会过期。`--refs` 报每个量的状态与最后验证日期; 状态是 uncalibrated 的
       就别拿它下结论, 去补标定。

它回答什么 / 不回答什么
  回答: (1) 这个算子在这颗芯片上最少要多久; (2) 当前实现是它的几倍;
        (3) **哪一项流量离强制量最远** —— 这一条是候选来源, 比 profiler 的分解表更靠前。
  不回答: 怎么改。下界只说"还差多远", 不说"往哪走" —— 但超额表会把量级排序, 那就是线索。
  ★下界不是可达目标。它是**对偶的上界**: 任何实现都不可能比它快, 所以倍数是一个
    有硬件证据的、与现有实现无关的标尺。这正是"不让既有方案约束思维"的手段。

用法
  python -m core.hwfloor <spec.json>                  # 算下界与超额表
  python -m core.hwfloor <spec.json> --measured 34.9  # 附上实测, 报倍数
  python -m core.hwfloor --template                   # 打印 spec 骨架
  python -m core.hwfloor --selftest
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys

from ._refs import Ref, print_refs

REFS = {
    "HF7_which_execution_unit": Ref(
        what="慢一两个数量级时, 先看**哪个执行单元在干活**, 再看离哪个天花板多远",
        status="measured",
        evidence=[
            "SFA 2026-09-21: 源码里 Mmad/Matmul/Fixpipe **各 0 条**, 一个以矩阵乘为主的算子",
            "全部算术在向量单元。两个天花板: cube fp16 189.2 T MAC/s, vector fp32 2.96 T",
            "(56 核 x 128 B/cy / 4 B x 1.65 GHz), 相差 64 倍。",
            "★但仅凭'cube 空转'不能下结论: case 3 实测 17.18 G MAC / 393765 us = 0.0436 T MAC/s,",
            "  只有向量天花板的 **1.5%** —— 连向量算力都没摸到, 所以主导项是每对 (head,key)",
            "  一次 PIPE_ALL 与一次标量回读, 不是算力。此时搬 cube 是在不承压的一侧用力。",
            "★这条判据当天写出来之前, 我对同一个现象先后给出两个相反结论, 都没有算术支撑。",
            "  取数: core.socspec 的 vec_calc_B_per_cycle (平台 ini [VectorCoreSpec] vec_calc_size)。",
            "★★判据首版自身有错, 同日由 msprof 纠正: 它把**吞吐效率**当成了**流水占用**,",
            "  据此给出'不到向量天花板一成就别搬 cube'。同一次测量里吞吐效率 6.5% 而",
            "  aiv_vec_ratio 78.4%、aic 零占用 —— 向量流水忙到 78% 正说明它就是瓶颈,",
            "  效率低只说明它在发小指令。现在两个量都要, 缺 --msprof 就不给换单元的结论。",
            "★同一次还证伪了一条: 减半 gather 条数 (两遍改单遍) 反而慢 8%, 与 aiv_mte2",
            "  只占 10.9% 一致。跨 case 除出来的'每行 246~473 周期'是相关性不是因果。",
        ],
        verified="2026-09-21"),
    "HF1_floor_must_be_derived_not_measured": Ref(
        what="优化范围的下界必须由硬件能力算出, **禁止拿另一个构建的实测值当地板**",
        status="measured",
        evidence=[
            "sfag 2026-09-18 复盘: 我用 `det=0 非优化 31.482 ms` 当这条路径的地板, 推出可动空间",
            "6.325 ms, 抢回 2.898 后自评'回收 46%'。真实硬件下界 1.651 ms (cube 受限),",
            "交付版是它的 **21.2 倍**, 对手 14.5 倍, 原始目标 18.3 ms 也才 11.1 倍。",
            "★地板取自同一族实现, 就会把那一族共有的缺陷一起继承成'物理定律'。",
            "  实测表明这不是抽象风险: det=0 与 det=1 两条路径**都**有紧邻 Set/Wait (H10 错误级,",
            "  L1421/1425/1473 在非确定性路径), 于是'差值'这个度量把它们精确抵消掉了。",
            "★与 docs/perf-ceiling-estimation.md 的关系: 那份文档列了五种**已证伪**的估收益做法,",
            "  但没有任何肯定方法 —— 只有否定条目的文档会让人退回到'拿实测当地板'。本模块补上正面方法。",
        ],
        verified="2026-09-18"),
    "HF4_excess_must_say_what_reaching_compulsory_requires": Ref(
        what="超额表报了倍数就必须报**要达到强制量需要什么** —— 只报倍数等于把最后一步留给灵感",
        status="measured",
        evidence=[
            "sfag 2026-09-18 (dubin 直接指出 'infra 不行'): 超额表当时报",
            "  `dK 累加器 原子写 16.0x` / `读-改-写 ∞`, 我据此排候选、撞墙、宣布候选池穷尽。",
            "  参考实现 (PR 12104) 同 shape 逐位一致、**22.485 vs 我的 34.422 ms, 快 53%**,",
            "  做法是「按输出下标遍历 + 单 owner 本地累加」。",
            "★决定性的一点: **那个做法可以从超额表自己的算术推出来**, 而本模块从没推:",
            "    超额 16 = T1*sel / T2 = 每个输出行平均被写 16 次;",
            "    要回到每行写一次 -> 同一行的贡献必须在处理序里**连续**;",
            "    当前按输入下标遍历 -> 贡献是散的 -> 必须按**输出**下标遍历 (要倒排);",
            "    按输出遍历时累加器放本地 -> **块宽 = 片上容量 / 每行字节**。",
            "  四步全是机械推导。工具报了 16.0x 却不说这四步, 我就只能靠想到。",
            "★接上之后本模块在真 spec 上直接推出: 「85 行可常驻 UB, 取 2 的幂并留余量则为 **64 行**」",
            "  —— 而参考实现的 KeyOwnedBucketWidth 正是 **64**。",
            "★仍然是给线索不给处方: 它给的是**结构要求** (遍历顺序 / 块宽上界), 不是怎么写代码。",
            "★误报控制: 只对显式标了 `pattern: scatter_accumulate` 的流出这一段; 流式/gather 不出。",
        ],
        verified="2026-09-18"),
    "HF3_roundtrip_removability_is_soc_dependent": Ref(
        what="「两端都在片上 -> 这趟 GM 往返可以去掉」**只在有 CV 直通路径的芯片上成立**",
        status="measured",
        evidence=[
            "sfag 2026-09-18: 本模块把 mm4/mm5 的 cube<->vector 中转标为 ★GM 纯中转,",
            "  提示语是「数据没有离开芯片的必要却走了一趟 GM」, 并把它排在超额表**第一位**。",
            "  我据此在 loop 提示词里写下「C1 = **去掉**这趟中转, 理由是结构性质不依赖带宽标定」。",
            "★动手前查 CALIB 才发现: **910B2 的 cv_direct_path = False** ——",
            "  cube 与 vector 是**分立物理核**, CV 数据必须经 GM 中转;",
            "  SetCType(TPosition::VECCALC) 在本 SoC 的分立核 MIX 场景不可用。",
            "  (950PR 才是 True 且 status=measured: FIA arch35 的 Fixpipe<..., PFA_CFG_ROW_MAJOR_UB>",
            "   直接把 L0C 写进对侧 UB, msprof 的 L0C_to_GM_datas = 0 佐证。)",
            "=> **那趟往返在本芯片上去不掉**。本模块把一个**物理上不可能**的改法排在第一位,",
            "   我差一步就为它烧一轮 40 分钟的构建。",
            "★判据给线索不给处方, 但**不能给一条硬件上不存在的处方**。",
            "★同时修掉一处同屏自相矛盾: 「这趟往返去不掉」紧接着「强制量为 0 -> 整笔都不必要」。",
            "  措辞必须分开 **GM** 与 **DRAM**: 强制的 DRAM 字节为 0 = 这笔**可以完全由 L2 供给**,",
            "  不等于'整笔不必要'。二测批评过的正是这类两条判据同屏给出相反方向。",
            "★于是 C1 的形态变了: 从「消除这趟往返」变成「让它的工作集进 L2」——",
            "  **两个完全不同的工程问题**, 后者受 428MB vs L2 201MB 的硬约束。",
        ],
        verified="2026-09-18"),
    "HF2_excess_table_is_the_candidate_source": Ref(
        what="超额表 (实际流量 / 强制流量) 是优化候选的来源, 它比 profiler 的时间分解表**更靠前**",
        status="measured",
        evidence=[
            "sfag 2026-09-18: 同一份十行算术同时给出两个最大的优化点, 而我看了一整轮 msprof 分解表",
            "一个都没看出来:",
            "  scatter 写: 强制 285 MB vs 逐(q,j)原子加 9.13 GB = **32x** -> 改分核让每行单 owner",
            "  gather 读: 4.56 GB vs K+V 全量 285 MB, L2 192 MB  = **16x 复用潜力** -> 打包 + 分块",
            "★为什么分解表看不出来: 它报的是'时间花在哪', 不报'这件事本来要花多少'。",
            "  一项花了 8.6 ms 且本该花 0.27 ms, 与一项花了 8.6 ms 且本就该花 8 ms, 在分解表上",
            "  **长得一模一样**。少的那个参照量只能算, 不能测。",
        ],
        verified="2026-09-18"),
}

#  ---- 判词的唯一出处 ------------------------------------------------------------
#  2026-09-18 合并: 仓里有**六处**各自实现"地板 + 倍数 + 结论"的判词
#  (attnplan 70 处提及 / mmplan 20 / sortplan 7 / scalarplan 4 / pipeplan 3 / perf 2)。
#  判词散开的后果不是重复劳动, 是**判读不一致** —— 同一个 3 倍差距, 一处写"还有空间",
#  另一处写"已贴住地板", 而读的人不知道两者用的是不同种类的地板。
#  ★三种地板必须分清, 合并的是**判词**不是**定义**:
#    capability  硬件能力下界 (字节/带宽, 运算/吞吐) —— 与实现无关, 任何实现都不可能更快
#    critpath    并行关键路径下界 (工作量/核数, 或由 binding pipe 定) —— 与分核/编排有关
#    empirical   经验地板 (实测拟合) —— **最弱**, 会被新设计推翻 (scalarplan 的那条已被 S17b 推翻)
#  把种类标出来是必须的: 拿 critpath 的地板去论证"没空间了", 等于拿当前分核方案当物理定律,
#  而那正是 sfag 2026-09-18 栽的那一跤 (见 REFS HF1)。
FLOOR_KIND = {
    "capability": ("硬件能力下界", "与实现无关 —— 任何实现都不可能比它快"),
    "critpath":   ("并行关键路径下界", "**与当前分核/编排有关** —— 换编排它会变, 别拿它论证'没空间'"),
    "empirical":  ("经验地板 (实测拟合)", "**最弱的一种** —— 换设计就会被推翻, 只在同族设计内可用"),
}


#  ---- 片上层级端口 (2026-09-18 扩; dubin: "内部层级硬件交接、数据搬运、计算流水之间的预估建模还不够全面")
#  只看 DRAM 的下界会漏掉真正的墙: 一笔数据从 GM 进 L1、再进 L0A/L0B、算完从 L0C 出来,
#  每一跳都有自己的端口速率, 任何一跳都可能先饱和。
#  速率来自 socspec 的 rate_B_per_cycle (平台 ini), **不猜**: 表里没有的端口显式报"无参照"。
#  ★两个下界必须分开 (这是本模块最重要的一次结构改动):
#    capability floor = 按**强制流量**算 —— 任何实现都不可能更快, 与当前实现无关。
#    orchestration floor = 按**本实现真实搬的流量**算, 且假定各端口完美重叠 ——
#                          这是**当前这套数据流**的下限。
#    于是差距被分成两段, 各对应完全不同的动作:
#      measured - orchestration = 重叠没做好 (软件流水/屏障/发射次序)  -> 改编排
#      orchestration - capability = 搬了不该搬的 (多余往返/复用不足)   -> 改数据流
#    此前我把这两段混在一起, 于是"还差 21 倍"既不知道该改编排还是改数据流。
PORTS = {
    #  端口名: (socspec 的 rate 键, 按哪种核数聚合, 人读的说明)
    "dram":     ("ddr_read_rate",   "cube",   "GM <-> 片外"),
    "l2":       ("l2_read_rate",    "cube",   "L2 读"),
    "l2_write": ("l2_write_rate",   "cube",   "L2 写"),
    "l1_l0a":   ("l1_to_l0_a_rate", "cube",   "L1 -> L0A (左矩阵)"),
    "l1_l0b":   ("l1_to_l0_b_rate", "cube",   "L1 -> L0B (右矩阵)"),
    "l0c_ub":   ("l0_c_to_ub_rate", "cube",   "L0C -> UB (fixpipe 出口)"),
    "l1_ub":    ("l1_to_ub_rate",   "cube",   "L1 -> UB"),
    "ub_ddr":   ("ub_to_ddr_rate",  "vector", "UB -> 片外 (向量侧写出)"),
    "ub_l2":    ("ub_to_l2_rate",   "vector", "UB -> L2"),
    "ub_l1":    ("ub_to_l1_rate",   "vector", "UB -> L1"),
}


PORT_NOTES: list = []     # port_bandwidths 的注解 (为什么某个端口用了这个带宽)


def port_bandwidths(soc: dict) -> tuple:
    """每个端口的聚合带宽 (B/s)。取不到的**列出来说取不到**, 不填默认值。"""
    rates = soc.get("rate_B_per_cycle") or {}
    freq = float(soc.get("freq_ghz") or 0) * 1e9
    nc = {"cube": float(soc.get("cube_cores") or 0), "vector": float(soc.get("vector_cores") or 0)}
    #  ★片外端口要打实测折扣, 片上不打 —— **但只在没有饱和实测标定时**。
    #    OFFCHIP_REALISM 存在的理由是: ini 的 DRAM 式给的是**峰值**, socspec 自己标注它在
    #    910B2 上高估实测 23%。所以这个系数是"把峰值折算成可达值"的补丁。
    #    ★2026-09-18: 本机做了饱和标定 (dram_GBps=1428.9 measured, runs/dramcalib_20260918),
    #      那个值**本身就是可达值**。再乘 0.77 是**折两次** —— 带宽被低估 23%
    #      => DRAM 端口的能力时间被高估 => **超额被算小**, 正是 6.77ms 对 11.94ms
    #      低估 1.76 倍的那个方向。标定落地时必须同时撤掉为"没有标定"而设的补偿, 否则两处相乘。
    #    同族教训见 memory「等价变换有作用域」: 前移一个因子必须撤掉每条通路上的补偿点。
    OFFCHIP_REALISM = 0.77
    OFFCHIP = {"dram", "ub_ddr"}
    cal = ((soc.get("_calib") or {}).get("dram_GBps") or {})
    measured_dram = cal.get("value") if cal.get("status") == "measured" else None
    bw, missing = {}, []
    #  ★注解走模块级, **不塞进 bw** —— bw 是 {端口: 浮点带宽}, 混一个列表值进去,
    #    下一个写 `for p, v in bw.items()` 的人就当场炸。治形态不治症状。
    PORT_NOTES.clear()
    for port, (key, kind, _desc) in PORTS.items():
        r = float(rates.get(key) or 0)
        n = nc.get(kind, 0)
        if port == "dram" and measured_dram:
            bw[port] = float(measured_dram) * 1e9
            PORT_NOTES.append(f"dram 用**饱和实测标定** {measured_dram} GB/s, "
                         f"**不再乘 OFFCHIP_REALISM {OFFCHIP_REALISM}** (实测值本身就是可达值, "
                         f"再乘是折两次); ini 峰值式给的是 {r*n*freq/1e9:.0f} GB/s")
            continue
        if r > 0 and n > 0 and freq > 0:
            bw[port] = r * n * freq * (OFFCHIP_REALISM if port in OFFCHIP else 1.0)
        else:
            missing.append((port, key))
    return bw, missing


def verdict_lines(floor_ms: float, measured_ms: float | None = None, *,
                  kind: str = "capability", bound_by: str | None = None,
                  excess_top: dict | None = None, indent: str = "  ") -> list:
    """地板 + 倍数 + 该怎么读 —— **全仓判词只此一处**。

    调用方只负责算出 floor_ms 并说明它是哪一种; 措辞、阈值、告诫都在这里。
    """
    label, caveat = FLOOR_KIND.get(kind, FLOOR_KIND["capability"])
    out = [f"{indent}{label} {floor_ms:.3f} ms" + (f" (受限于 {bound_by})" if bound_by else "")]
    out.append(f"{indent}  {caveat}")
    #  ★地板算出来是 0 或小得离谱时, **拒绝给倍数** —— 那说明输入错了, 不是"还能快一百万倍"。
    #    2026-09-18 合并 attnplan 时当场撞见: `--selected` 被当成全局块数, 每 q 块选中数
    #    向下取整成 0, cube 地板 = 0 us, 判词照样打出 "2945446.9x"。
    #    一个**自信的荒谬数字**比报错危险得多 —— 它看起来像结论。守卫放在判词唯一出处,
    #    所有调用方一起受益 (这正是合并的收益, 不只是少写几行)。
    if measured_ms and (floor_ms <= 0 or measured_ms / floor_ms > 1000):
        out.append(f"{indent}★**拒绝给倍数**: 地板算出来是 {floor_ms:.6f} ms, "
                   f"对实测 {measured_ms:.3f} ms 意味着 1000 倍以上的差距 —— "
                   f"这几乎一定是**输入错了** (形状/选中数/单位), 不是真有这么大空间。"
                   f"先核对各项再谈结论。")
        return out
    if measured_ms:
        mult = measured_ms / floor_ms if floor_ms > 0 else float("inf")
        out.append(f"{indent}实测 {measured_ms:.3f} ms = 它的 **{mult:.1f}x**"
                   f" (相当于最多还能快 {1 - floor_ms / measured_ms:.0%})")
        if kind == "capability" and mult >= 2:
            out.append(f"{indent}★倍数 >= 2 -> 默认动作是**结构性**改动, 不是增量调参。"
                       f"在 {mult:.0f}x 的差距上, 1%~3% 的增量收益是噪声,"
                       f" 而**持续的小收益恰恰会让错误的框定显得在正常工作**。")
        elif kind == "capability" and mult < 1.3:
            out.append(f"{indent}★已贴住能力下界 —— 再往下只能减**工作量**(改算法/改 dtype/改块数),"
                       f" 那是改语义不是改编排。")
        if kind != "capability":
            out.append(f"{indent}★这不是能力下界: 它随分核/编排变。"
                       f"要判'还剩多少物理空间'请另算 capability 那一种"
                       f" (python -m core.hwfloor <spec.json>)。")
    if excess_top and excess_top.get("ratio"):
        out.append(f"{indent}[超额] 最大一项 `{excess_top['name']}` = 强制量的 {excess_top['ratio']:.1f}x"
                   + (f", 多花 {excess_top['excess_s']*1e3:.2f} ms" if excess_top.get("excess_s") else "")
                   + "  <- 下一个候选该去的地方")
    return out


def capability_floor(compute_macs: float, compulsory_bytes: float,
                     cube_flops: float, dram_bytes_per_s: float) -> dict:
    """kind=capability 的最小算子: 给 MAC 数与强制字节, 出两个地板与受限侧。

    ★领域模块 (attnplan / mmplan / ...) 各自负责**算出这两个量** —— 那是领域知识, 不该合并;
      合并的是这里的取大者、单位换算与下面 verdict_lines 的判词。
    """
    cube_s = 2 * compute_macs / cube_flops if cube_flops > 0 else 0.0
    mem_s = compulsory_bytes / dram_bytes_per_s if dram_bytes_per_s > 0 else 0.0
    return {"cube_ms": cube_s * 1e3, "mem_ms": mem_s * 1e3,
            "floor_ms": max(cube_s, mem_s) * 1e3,
            "bound_by": "cube" if cube_s >= mem_s else "memory"}


_SAFE = {"min": min, "max": max, "ceil": math.ceil, "floor": math.floor, "log2": math.log2}


def _ev(expr, vars_):
    if isinstance(expr, (int, float)):
        return float(expr)
    try:
        return float(eval(str(expr), {"__builtins__": {}}, dict(_SAFE, **vars_)))
    except Exception as e:
        raise SystemExit(f"表达式求值失败: {expr!r} -> {type(e).__name__}: {e}\n"
                         f"  可用变量: {', '.join(sorted(vars_))}")


TEMPLATE = {
    "name": "<算子名>",
    "soc": {"cube_flops": 353.9e12, "dram_bytes_per_s": 1.06e12,
            "_note": "cube_flops/dram 从 core.socspec 取; DRAM 用**实测可达**不用模型值"},
    "vars": {"T1": 1024, "T2": 131072, "N1": 64, "D": 512, "_": "形状量, 下面的表达式可以引用"},
    "compute": [{"name": "dQ = dS @ K", "macs": "T1*N1*sel*Dk"}],
    "traffic": [
        {"name": "K gather",
         "compulsory": "T2*Dk*2",
         "actual": "T1*sel*Dk*2",
         "note": "强制 = 每个 K token 读一次; 实际 = 每个 (query, 选中) 各读一次"}
    ],
}


def analyze(spec: dict, measured_ms: float | None = None) -> dict:
    v = {k: val for k, val in spec.get("vars", {}).items() if isinstance(val, (int, float))}
    #  ★硬件参数从 socspec 直接取, spec 里写的值**覆盖**它 —— 不让人手抄。
    #    手抄硬件常量是本仓反复栽的那类错 (U7 那次我手传 dimDAlign=512 而真值 576);
    #    socspec 的定位本来就是"芯片规格唯一取数口", 让 spec 再抄一遍等于制造第二个真源。
    soc = {}
    try:
        from . import socspec as _ss
        sp = _ss.spec(_ss.default_soc())
        soc.update({"rate_B_per_cycle": sp.get("rate_B_per_cycle") or {},
                    "freq_ghz": sp.get("freq_ghz"),
                    "cube_cores": sp.get("cube_cores"), "vector_cores": sp.get("vector_cores"),
                    "l2_bytes": float((sp.get("cap") or {}).get("L2") or 0),
                    "cube_flops": float((sp.get("derived") or {}).get("peak_fp16_TFLOPS") or 0) * 1e12,
                    #  ★DRAM 带宽的出处必须跟着数一起传下去 (2026-09-18):
                    #    优先用 CALIB 的**饱和实测**标定; 没有标定时退回 ini 模型值,
                    #    但**标记成 uncalibrated** —— 据它算出的下界是**假下界**。
                    #    此前这里直接写 `模型值 * 0.77`, 那个 0.77 是我自己编的系数,
                    #    而 CALIB 里 dram_GBps 明写 status=uncalibrated 且被撤回过两次。
                    "dram_bytes_per_s": 0.0, "dram_provenance": "未知"})
        _cal = None
        try:
            _cd, _ = _ss.calib(None)
            _cal = (_cd or {}).get("dram_GBps")
        except Exception:
            _cal = None
        _val = (_cal or {}).get("value") if isinstance(_cal, dict) else None
        soc["_calib"] = {"dram_GBps": _cal if isinstance(_cal, dict) else None}
        if _val:
            soc["dram_bytes_per_s"] = float(_val) * 1e9
            soc["dram_provenance"] = f"[实测标定] CALIB dram_GBps={_val} ({(_cal or {}).get('status')})"
        else:
            soc["dram_bytes_per_s"] = float((sp.get("derived") or {}).get("dram_model_TBps") or 0) * 1e12
            soc["dram_provenance"] = (
                "[未标定] CALIB 里 dram_GBps 是 None(uncalibrated), 这里退回 ini 的**成本模型参数** —— "
                "它不是可达带宽。据它算出的内存侧下界是**假下界**, 只能排序不能定量。"
                "补标定需专门跑一个工作集**远大于 L2** 的纯流式负载 (CALIB 里记着两次错误标定的教训)。")
    except Exception:
        pass
    soc.update(spec.get("soc", {}))
    cube_flops = float(soc.get("cube_flops") or 0)
    dram = float(soc.get("dram_bytes_per_s") or 0)
    #  ★2026-09-18 实测推翻: 把"多花的字节"按**顺序流饱和平台**折算成时间, 是乐观的。
    #    runs/bytes_pair_20260918: 本算子真形状上, g15 与 ref12104 的**达成** DRAM 带宽
    #    分别是 715 / 675 GB/s = 标定平台 1428.9 的 50% / 47% —— 它的访问形态远非顺序流。
    #    按平台折算**系统性低估每 GB 的时间代价约 2 倍**。
    #    分工 (这是语义上该分开的两件事, 不是打补丁):
    #      能力下界 = 强制字节 / **平台**      -> 任何实现最好能到哪 (顺带也要修访问形态)
    #      编排下界 / 多花 = 实际字节 / **达成** -> 本实现的流量按它真实的效率要多久
    #    spec 里没给达成带宽时退回平台, 并**说出来** —— 静默用平台就是静默乐观。
    dram_ach = float(spec.get("dram_achieved_GBps") or 0) * 1e9
    if dram_ach <= 0:
        dram_ach = dram
        spec.setdefault("_dram_ach_note",
                        "spec 没给 dram_achieved_GBps -> 编排/多花退回用**平台**带宽折算, "
                        "这是**乐观**的: 散射访问形态上达成带宽可能只有平台的一半。"
                        "测法见 core/probe/bytes_report.py (L2 miss_allocate x 反推颗粒)。")
    if not cube_flops or not dram:
        raise SystemExit("soc.cube_flops 与 soc.dram_bytes_per_s 都必须给 —— "
                         "不猜默认值, 猜错会把下界算低而让倍数显得正常。")

    macs = 0.0
    comp_rows = []
    for c in spec.get("compute", []):
        m = _ev(c["macs"], v)
        macs += m
        comp_rows.append({"name": c["name"], "macs": m})
    cube_s = 2 * macs / cube_flops

    #  ★流量项的四个维度 (2026-09-18 扩, dubin: "观察粒度要细致全面, Q/dY/dQ 太特殊了"):
    #    compulsory  最少要**穿过 DRAM** 多少字节。中间量若能全程留在片上, 这里就是 0 ——
    #                此时 actual>0 意味着"这笔流量在原理上完全不必要", 是最强的一类候选。
    #    actual      本实现真的搬了多少。
    #    footprint   工作集字节。<= L2 则这笔流量由 L2 供给, **不占 DRAM** ——
    #                盲测审计里最决定性的一步就是这个分流, 而它当时只能手算 (工具缺口 #1)。
    #    producer/consumer  谁写谁读。两端都是片上单元 (cube/vector) 的中间量 = **GM 纯中转**,
    #                单独报出来: 这趟往返本身就是候选 (盲测里排第一的那笔就是这种)。
    l2 = float(soc.get("l2_bytes") or 0)
    rows, comp_tot, act_tot, dram_act, l2_act = [], 0.0, 0.0, 0.0, 0.0
    ONCHIP = {"cube", "vector", "aic", "aiv"}
    for t in spec.get("traffic", []):
        cb = _ev(t["compulsory"], v)
        ab = _ev(t["actual"], v) if "actual" in t else None
        fp = _ev(t["footprint"], v) if "footprint" in t else None
        #  显式 resident 覆盖自动判定 (有些流的工作集不是一个静态数)
        res = t.get("resident")
        if res is None and fp is not None and l2 > 0:
            res = fp <= l2
        comp_tot += cb
        if ab is not None:
            act_tot += ab
            if res:
                l2_act += ab
            else:
                dram_act += ab
        prod, cons = t.get("producer"), t.get("consumer")
        roundtrip = bool(prod in ONCHIP and cons in ONCHIP)
        #  compulsory==0 的比值是无穷 —— 那不是"倍数很大", 是**整笔都不必要**, 要分开表达
        ratio = (ab / cb) if (ab is not None and cb > 0) else (float("inf") if ab else None)
        rows.append({"name": t["name"], "compulsory": cb, "actual": ab,
                     "port": t.get("port", "dram"),
                     "pattern": t.get("pattern"),
                     "out_rows": _ev(t["out_rows"], v) if "out_rows" in t else None,
                     "row_bytes": _ev(t["row_bytes"], v) if "row_bytes" in t else None,
                     "contributions": _ev(t["contributions"], v) if "contributions" in t else None,
                     "footprint": fp, "resident": res,
                     "producer": prod, "consumer": cons, "roundtrip": roundtrip,
                     "ratio": ratio,
                     #  超额只在**落 DRAM** 时才折算成时间; 由 L2 供给的那部分不占 DRAM 时间
                     "excess_s": (((ab - cb) / dram_ach) if (ab is not None and not res) else
                                  (0.0 if ab is not None else None)),
                     "note": t.get("note", "")})
    mem_s = comp_tot / dram
    floor_s = max(cube_s, mem_s)
    #  ★超额按**秒**排序而不是按倍数: 一个 100 倍但只有 1 MB 的项不值得动,
    #    而 3 倍的 9 GB 项值得。倍数用来说明性质, 秒数用来定优先级。
    #  ★**analyze 内部不排序** —— rows 全程与 spec["traffic"] 同序。
    #    2026-09-18: 端口聚合曾因"先排序再 zip 原始 traffic"而错位配对 (第二次盲测抓到);
    #    我修了那一处, 两跳后在并发工作集去重时**又写了一遍同样的 zip**。
    #    => 只治症状不治形态就会复发。排序挪到 report 里对副本做, 从根上消掉这一类。
    #  ★并发工作集之和 vs L2 —— 单条流的 footprint <= L2 **不等于**它能常驻。
    #    2026-09-18 实撞: 我据"把 mm4/mm5 从 428MB 压到 107MB 就进 L2"排了一整轮候选,
    #    而分块后并发工作集是 107(mm4/mm5) + 302(dK) + 151(K) = **560 MB = L2 的 2.8 倍** ——
    #    压小任何单一流都不足以让它常驻, 别的流照样把它挤出去。
    #    本模块此前逐条判 `footprint <= l2` 就标 resident, **完全不看竞争**, 于是
    #    "把它压进 L2"这类候选会被系统性高估。
    #  ★按**张量集合**去重 —— 同一块缓冲的读与写各是一条流, 但工作集只有一份。
    #    不去重会把比值虚高一倍以上 (sfag: 2038 MB 实为 ~950 MB), 而虚高的判据
    #    即使方向对也会让人不信它。spec 里的 `tensors` 声明正好用来去重。
    _seen, concurrent = set(), 0.0
    for _t, _row in zip(spec.get("traffic", []), rows):
        fp = _row.get("footprint")
        if not fp:
            continue
        key = tuple(sorted(_t.get("tensors") or [])) or _row["name"]
        if key in _seen:
            continue
        _seen.add(key)
        concurrent += fp
    contended = bool(l2 and concurrent > l2)

    #  ★按端口聚合 -> 每个端口各自的时间下限。任何一跳都可能先饱和, 只看 DRAM 会漏掉真正的墙。
    pbw, pmissing = port_bandwidths(soc)
    #  ★两个真 bug, 2026-09-18 由第二次盲测审计揪出 (它们让本模块给出**方向相反**的结论):
    #    (1) rows 在上面已按超额排过序, 这里再 zip(spec.traffic, rows) 就是**错位配对**。
    #        本 spec 恰好全是 dram 端口所以没暴露 —— 一旦出现第二个端口就会把流量记错端口。
    #        修法: port 在建 row 时就存进 row, 这里只遍历 rows, 不再 zip。
    #    (2) 累加 actual 时**完全不看 resident**, 于是由 L2 供给的流量被记在 DRAM 端口上,
    #        编排下界因此虚高, 打印出"编排下界已超过实测 -> 重叠不是问题, 再调流水拿不到东西"
    #        —— 而真实情况是**确实有 8 ms 在重叠与等待上**。一条算错的判据比没有判据危险。
    #    口径: resident 行的 **actual** 记到 l2 端口; **compulsory** 仍记在它声明的端口
    #    (compulsory 的含义是"最少要穿过这个边界多少", 与它被谁供给无关)。
    pagg = {}
    for row in rows:
        dport = row.get("port") or "dram"
        a = pagg.setdefault(dport, {"compulsory": 0.0, "actual": 0.0, "items": []})
        a["compulsory"] += row["compulsory"]
        if row["actual"] is not None:
            aport = "l2" if row.get("resident") else dport
            b = pagg.setdefault(aport, {"compulsory": 0.0, "actual": 0.0, "items": []})
            b["actual"] += row["actual"]
        a["items"].append(row["name"])
    ports = []
    for port, a in pagg.items():
        b = pbw.get(port)
        ports.append({"port": port, "desc": PORTS.get(port, ("", "", port))[2],
                      "bw": b, "compulsory": a["compulsory"], "actual": a["actual"],
                      #  ★能力用平台, 编排用达成 (见 dram_ach 处的注释): 二者不是同一个问题。
                      "bw_achieved": (dram_ach if port == "dram" else b),
                      "cap_ms": (a["compulsory"] / b * 1e3) if b else None,
                      "orch_ms": (a["actual"] / (dram_ach if port == "dram" else b) * 1e3)
                                 if (dram_ach if port == "dram" else b) else None,
                      "items": a["items"]})
    ports.sort(key=lambda x: -(x["orch_ms"] or 0))
    #  编排下界 = 各端口与 cube 里最慢的那一个 (假定完美重叠)
    orch_ms = max([cube_s * 1e3] + [p["orch_ms"] for p in ports if p["orch_ms"]])
    #  能力下界同理, 但用强制流量
    cap_mem_ms = max([p["cap_ms"] for p in ports if p["cap_ms"]] or [0.0])

    #  ★完备性自检 —— **spec 漏了流量时必须自己说出来**。
    #    2026-09-18 实撞: 我手列了 gather/scatter/Q/dY/dQ/索引 六项就开始下结论,
    #    完全漏掉 mm4/mm5 的 cube<->vector 交接往返 (18.25 GB), 而那是最大的一笔。
    #    于是"超额表第一行"指错了地方, 报出的下界与倍数也都是错的。
    #    判法不需要任何额外信息: 实测时间 x DRAM 带宽 = 这段时间内**至多**能搬的字节。
    #    若它远大于 spec 里落 DRAM 的字节, 说明 spec 没把流量列全 (或带宽标定偏低)。
    #    这是一条**只会在真漏时响**的判据 —— 列全了就自然不响。
    cover = None
    if measured_ms and dram_act > 0:
        budget = measured_ms / 1e3 * dram
        cover = {"dram_actual": dram_act, "time_budget_bytes": budget,
                 "ratio": dram_act / budget}
    return {
        "name": spec.get("name", "?"),
        "cube_floor_ms": cube_s * 1e3, "mem_floor_ms": mem_s * 1e3,
        "floor_ms": floor_s * 1e3,
        "bound_by": "cube" if cube_s >= mem_s else "memory",
        "compulsory_bytes": comp_tot, "actual_bytes": act_tot,
        "dram_actual_bytes": dram_act, "l2_actual_bytes": l2_act,
        "l2_bytes": l2,
        "concurrent_footprint": concurrent, "l2_contended": contended,
        "dram_provenance": soc.get("dram_provenance", "未知"),
        "compute": comp_rows, "traffic": rows,
        "ports": ports, "ports_missing_bw": pmissing,
        "orchestration_floor_ms": orch_ms,
        "port_cap_floor_ms": cap_mem_ms,
        "measured_ms": measured_ms,
        "multiple": (measured_ms / (floor_s * 1e3)) if measured_ms else None,
        "coverage": cover,
    }


def _cv_direct():
    """本芯片有没有 cube->vector 的片上直通路径。取不到返回 None (不猜)。"""
    try:
        from . import socspec as _ss
        c, _ = _ss.calib(None)
        e = (c or {}).get("cv_direct_path")
        return e.get("value") if isinstance(e, dict) else None
    except Exception:
        return None


def workset_curve_estimate(r: dict, soc_name: str | None = None) -> dict | None:
    """用**本机实测的工作集曲线**估时间 —— 比模型带宽可信得多的一条路。

    2026-09-18: CALIB 的 `dram_GBps` 是 None(uncalibrated), 所以按模型带宽算出的内存
    下界是假下界。但同一份 CALIB 里有一条 **status=measured** 的曲线, 而且就测在这个
    算子这个 shape 上: 逻辑流量固定 28.44 GB, 只改 kv 工作集 ->
        4.5MB 11.78 | 18MB 11.91 | 72MB 15.24 | 144MB 18.43 ms
    它说明一件模型带宽表达不出来的事: **同样的字节, 工作集不同, 时间差 1.56 倍**。
    于是"减流量"与"压工作集"是两条独立的路, 而后者不用动一个字节。

    ★引用前必须核对 held_fixed —— 这条曲线自带一个**方向相反**的反例:
      二维分核把工作集从 144MB 压到 7.1MB, 照曲线该 ~13ms, 实测 44.59ms。
      原因是它在压工作集的同时把行切换次数乘了 8, 而那正是标定时 held_fixed 的量。
    所以本函数只报"曲线在哪个位置、压到 L2 内按曲线值多少", **不预测你的改动能拿到多少**。
    """
    try:
        from . import socspec as _ss
        _d, _src = _ss.calib(soc_name)
        c = (_d or {}).get("l2_working_set_curve")
    except Exception:
        return None
    if not isinstance(c, dict) or c.get("status") != "measured" or not c.get("value"):
        return None
    pts = sorted((float(k.replace("MB", "")) * 1e6, float(v)) for k, v in c["value"].items())
    #  本 spec 里**落 DRAM** 的流中最大的那个工作集 = 决定曲线位置的那个
    ws = max([t.get("footprint") or 0 for t in r.get("traffic", []) if not t.get("resident")] or [0])
    inside = bool(pts) and pts[0][0] <= ws <= pts[-1][0]
    return {"points": pts, "workset": ws, "inside": inside,
            "best_ms": pts[0][1], "worst_ms": pts[-1][1],
            "unit": c.get("unit", ""), "held_fixed": c.get("held_fixed", []),
            "verified": c.get("verified", "?"), "superseded": len(c.get("superseded", [])),
            "counterexample": c.get("counterexample_scheduling", "")}


def verify_against_msprof(r: dict, prof_dir: str, cores: int | None = None) -> dict:
    """拿 msprof 实测的 DRAM 字节去**验证 spec 预测的流量** —— 让 spec 可证伪。

    2026-09-18 沉淀动机: 整个超额表的排序都压在"本实现搬了多少字节"这个预测上, 而它
    是逐函数读源码算的 —— v1 漏 62%, v2 漏 48%, v3 才到 77%。**纯机械但最容易算错**,
    而且错了之后超额表会把人指向错误的地方 (v1 就把第一行指错了)。
    msprof 的 read/write_main_memory_datas 是同一件事的**实测**, 两者一比就知道 spec 准不准。
    ★这是"预测 vs 实测"的闭环, 也是本模块从"看起来有道理"变成"可证伪"的那一步。
    ★注意口径: 这两个信号是**每核**的量, 要乘核数; 且它们量的是**落 DRAM** 的字节,
      所以要跟 spec 的 dram_actual_bytes 比, 不是跟 actual_bytes 比。
    """
    try:
        from . import perf as _perf
        sig = _perf.extract_signals(prof_dir)
    except Exception as e:
        return {"ok": False, "err": f"读不出 msprof 产物 {prof_dir}: {type(e).__name__}: {e}"}
    if not sig:
        return {"ok": False, "err": f"{prof_dir} 里没有可解析的信号 —— 采集可能没跑起来 "
                                    f"(空目录与'对不上'长得一样, 所以这里显式报出来)"}
    def g(k):
        v = sig.get(k)
        return (v or {}).get("mean") if isinstance(v, dict) else v
    nc = cores or 24
    rd, wr = g("read_main_memory_datas(KB)"), g("write_main_memory_datas(KB)")
    if rd is not None or wr is not None:
        meas = ((rd or 0) + (wr or 0)) * 1024 * nc
        src = "字节计数器 read/write_main_memory_datas"
    else:
        #  ★没有字节计数器时退回 带宽 x 时间。但**必须先查这些计数器可不可信** ——
        #  2026-09-18 实撞 (CANN 9.1.0 / 910B2): Memory 档根本没有字节计数器, 而
        #  `aic_main_mem_read_bw` 报 0.263 GB/s/核 (x24 = 6.3 GB/s, 34ms 只能搬 0.21 GB,
        #  而算子要搬约 28 GB —— 低了两个数量级), `aic_l2_read_bw` 直接报 0.0。
        #  **计数器不可信时绝不能据它判 spec 错** —— 那会把一个好 spec 判死。
        #  这正是"每个数要带参照量"的价值: 没有参照时 0.263 只是"一个数"。
        #  ★分母必须是 **Task Duration**, 不是 <kind>_time —— 2026-10-19 机械挖掘实证
        #    (docs/msprof-re/08 F3, docs/msprof-re/tools/relmine.py):
        #    五个带宽列在除以 TaskDur 时精确到 1e-7, 除以 <kind>_time 则 AIC 侧差到 **4.86%**。
        #    成因: `<kind>_time` 是**该核的活跃时长**(F1/F2), 而带宽列是**任务窗口均值**。
        #    4.86% 正好等于 aic_time/TaskDur 的最大偏离 —— 两者对得上, 不是巧合。
        #    退回 <kind>_time 只是为了产物里没有 Task Duration 列的老格式, 并显式标注口径。
        t_us = g("Task Duration(us)")
        _t_src = "Task Duration(us)"
        if not t_us:
            t_us = g("aiv_time(us)") or g("aic_time(us)")
            _t_src = "<kind>_time(us) (**口径不对**, 产物里没有 Task Duration; AIC 侧可低估约 5%)"
        bws = {k: g(k) for k in ("aic_main_mem_read_bw(GB/s)", "aic_main_mem_write_bw(GB/s)",
                                 "aiv_main_mem_read_bw(GB/s)", "aiv_main_mem_write_bw(GB/s)")}
        have = {k: v for k, v in bws.items() if v}
        if not have or not t_us:
            return {"ok": False, "err": (
                "产物里**既没有字节计数器, 也没有可用的 main_mem 带宽** —— 本次没验成。\n"
                "    (CANN 9.1.0 / 910B2 实撞: Memory 档不产出 read/write_main_memory_datas。)")}
        #  ★单位: msprof 的 `*_bw(GB/s)` 列名写 GB/s, 除的却是 `8589934592.0` = 8*2^30,
        #    即它实际是 **GiB/s**。用 1e9 还原字节会系统性少算 7.37%。
        #    2026-10-19 实证 (runs/sweep_64_512_128_0_4_4 的 Memory 档, block 0):
        #      read_main_memory_datas 556313.125 KB ÷ aic_time 5069.598 us
        #        = 112.369 GB/s (十进制) = 104.652 GiB/s
        #      而同一行的 aic_main_mem_read_bw 报 104.497 —— 对上的是 GiB/s, 差 0.15%。
        #    ★证据是**实测自洽**, 不是源码: 本函数读的 OPPROF csv 由 msopprof 这个闭源 C++
        #      二进制生成 (它自带 204 个 aic_/aiv_ 列名, 不引用 CANN 的 Python 分析包),
        #      与 analysis/msconfig/ai_core_config.py 那套公式**不是同一条路径**。
        #      引那段 Python 源码来解释这里会引错对象 —— 详见 docs/msprof-re/06 的 V9。
        #    上面那条主路径 (KB×1024) 不受影响, 只有这条退路要用 2^30。
        meas = sum(have.values()) * (2 ** 30) * (t_us / 1e6) * nc
        src = f"带宽 x 时间 (无字节计数器时的退路; 时间口径 = {_t_src})"
        #  可信度闸: 隐含的聚合带宽若低到不可能 (< 50 GB/s 全卡), 说明计数器坏了
        agg = sum(have.values()) * nc
        if agg < 50:
            return {"ok": False, "err": (
                f"main_mem 带宽计数器**不可信, 拒绝据它下结论**: 全卡合计仅 {agg:.1f} GB/s "
                f"({', '.join(f'{k.split(chr(40))[0]}={v:.3f}' for k, v in have.items())} x {nc} 核)。\n"
                f"    算子跑了 {t_us/1e3:.1f} ms, 按这个带宽只能搬 {meas/1e9:.2f} GB, "
                f"而 spec 预测 {(r.get('dram_actual_bytes') or 0)/1e9:.1f} GB —— 差两个数量级。\n"
                f"    同一份产物里 aic_l2_read_bw 也报 0.0, 更说明是**计数器坏了不是 spec 错了**。\n"
                f"    ★不要据此判 spec —— 一个坏掉的测量会把好 spec 判死。\n"
                f"    可用的替代: 该档的 L1/UB 侧带宽量级正常 (aic_l1_read_bw / aiv_ub_read_bw),"
                f" 但它们量的是**片上端口**不是 DRAM, 不能直接当 DRAM 流量。")}
    pred = r.get("dram_actual_bytes") or 0
    return {"ok": True, "measured_bytes": meas, "predicted_bytes": pred, "source": src,
            "read_KB_per_core": rd, "write_KB_per_core": wr, "cores": nc,
            "ratio": (pred / meas) if meas else None}


#  ---- 超额表的"怎么才能到达强制量" ------------------------------------------------
#  沉淀动机 (2026-09-18, dubin 指出 "infra 不行"):
#    超额表当时报 `dK 累加器 16.0x / 读-改-写 ∞`, 我据此排了候选、撞墙、宣布穷尽。
#    参考实现同 shape 逐位一致快 53%, 做法是"按输出下标遍历 + 单 owner 本地累加"。
#    ★关键: 那个做法**可以从超额表自己的算术推出来**, 而本模块从没推:
#       超额倍数 16 = T1*sel / T2 = **每个输出行平均被写 16 次**;
#       要回到"每行写一次", 同一输出行的全部贡献必须在处理序里**连续**;
#       现在按 query 遍历 -> 贡献是散的 -> **必须按输出下标遍历** (要倒排);
#       按输出下标遍历时累加器要放本地 -> **块宽 = 片上容量 / 每行字节**。
#    四步全是机械推导。**报了倍数却不报"要达到它需要什么", 等于把最后一步留给灵感。**
#  ★仍然是"给线索不给处方": 它说的是**结构要求**(遍历顺序/块宽上界), 不是怎么写代码。
def reorder_requirement(t: dict, l2_bytes: float = 0, ub_bytes: float = 192 * 1024) -> list:
    """一条超额流要回到强制量, 遍历顺序与本地容量必须满足什么。"""
    out = []
    pat = t.get("pattern")
    if pat == "gather_reuse":
        #  ★与 scatter_accumulate **同源**: 超额倍数都是"每个元素被碰几次",
        #    要回到一次都得让"碰它的那些"在遍历序里连续 —— 也就是同一个倒排。
        #    散射侧是"多个写者写同一行", gather 侧是"多个读者读同一行", 对偶。
        rows_, contrib_ = t.get("out_rows"), t.get("contributions")
        out.append("★**要达到强制量, 遍历顺序必须满足**:")
        if rows_ and contrib_:
            out.append(f"    每个源行平均被读 **{contrib_/rows_:.1f} 次** "
                       f"(取数 {contrib_:,.0f} / 源行 {rows_:,.0f})。")
            out.append(f"    这 {contrib_/rows_:.1f} 倍不是搬多了, 是同一行被**反复取** —— "
                       f"要回到一次, 读它的那些消费者必须在处理序里**连续**。")
        out.append("    当前按**消费者**(query)下标遍历 -> 同一源行的消费者散落在整趟里;")
        out.append("    改成按**源行**下标遍历 -> 取一次、喂给它的全部消费者。")
        out.append("    ★它与散射累加要的是**同一个倒排** (源行 -> 碰它的那些), "
                   "所以两条流由**同一个改动**一起解决。")
        if t.get("row_bytes"):
            n_ = int(ub_bytes // t["row_bytes"])
            out.append(f"    ★一次能驻留的源行数上界: {ub_bytes/1024:.0f} KB / {t['row_bytes']} B "
                       f"= **{n_} 行**。")
        return out
    if pat != "scatter_accumulate":
        #  ★没标 pattern 就沉默 = 又回到"工具不被触发": 我得**先知道它重要**才会去标。
        #    所以超额明显而没标时要主动**提示补标**, 把沉默变成一句话。
        r_ = t.get("ratio")
        if r_ and r_ != float("inf") and r_ >= 2 and (t.get("actual") or 0) > (t.get("compulsory") or 0):
            out.append(f"★本条超额 {r_:.1f}x 但**没标 `pattern`** —— 本模块因此推不出"
                       f"「要达到强制量需要什么」。")
            out.append(f"    若它是**散射累加** (同一输出行被多个输入反复读-改-写), 在 spec 里补"
                       f" `pattern: scatter_accumulate` 与 out_rows / row_bytes / contributions,")
            out.append(f"    就能直接推出遍历顺序要求与本地累加器的块宽上界。"
                       f"(2026-09-18 实撞: 没推这一步, 我在 16x 的那条流上撞墙并宣布穷尽。)")
        return out
    ratio, act, comp = t.get("ratio"), t.get("actual"), t.get("compulsory") or 0
    rows = t.get("out_rows")
    row_b = t.get("row_bytes")
    contrib = t.get("contributions")
    out.append("★**要达到强制量, 遍历顺序必须满足**:")
    if rows and contrib:
        per = contrib / rows
        out.append(f"    每个输出行平均被写 **{per:.1f} 次** (贡献 {contrib:,.0f} / 输出行 {rows:,.0f})。")
        out.append(f"    这 {per:.1f} 倍**不是搬多了**, 是同一行被反复读-改-写 —— "
                   f"要回到一次, 同一行的全部贡献必须在处理序里**连续**。")
    out.append("    当前按**输入**下标遍历 -> 同一输出行的贡献散落在整趟里 -> 只能靠原子累加;")
    out.append("    改成按**输出**下标遍历 -> 同一行的贡献连续 -> **本地累加, 写一次**,")
    out.append("      原子加与为保序而设的跨核汇合**一起消失**, 确定性变成结构性的。")
    out.append("    ★代价: 输入是 (输入行 -> 选中的输出下标) 这种关系, 按输出遍历要先**倒排** ——")
    out.append("      一个预处理 pass (排序 / 分桶 / 建 header)。这一步是机械的, 不是创造性的。")
    if row_b:
        n_ub = int(ub_bytes // row_b)
        out.append(f"    ★本地累加器的**块宽上界**: 片上 {ub_bytes/1024:.0f} KB / 每行 {row_b} B "
                   f"= **{n_ub} 行**可常驻 UB。")
        out.append(f"      分块尺寸就是这么反推的 —— 先问'要让累加器进 UB 一次能装多少', "
                   f"再把那个数定成块宽 (取 2 的幂并留余量则为 {1 << (n_ub.bit_length() - 1)} 行)。")
    if rows and row_b and l2_bytes:
        tot = rows * row_b
        out.append(f"    ★注意整个输出 {tot/1e6:.0f} MB 仍 > L2 {l2_bytes/1e6:.0f} MB —— "
                   f"但这不再重要: 每行只写一次时, 它是**流式写**而不是反复命中。")
    return out


def shared_remedy_groups(r: dict) -> list:
    """把**需要同一个改法**的超额流归到一组, 并把它们多花的时间**加起来**。

    沉淀动机 (2026-09-18): 超额表逐条排序, 于是 dK 写 3.28ms / dK 读-改-写 4.54ms /
    gather 读 2.14ms 看起来是三件事、各自都不够大。
    **而它们要的是同一个倒排** —— 一个改动同时解决, 收益是它们之和。
    (sfag 实测: 参考实现用一个改动同时解决, 34.422 -> 22.485 ms, 差 11.94 ms;
     本模型算出的合计超额 6.77 ms 偏小, 因为内存侧带宽未标定 + gather 那条被记成 L2 供给。)
    ★逐条排序会系统性**低估**那些需要大改动的方向: 大改动往往同时解决多条流,
      而表把它的收益拆散了。这正是我把候选池排成"没有大项"的原因之一。
    """
    KEY = {"scatter_accumulate": "按**输出/源行**下标遍历 (需要倒排: 源行 -> 碰它的那些)",
           "gather_reuse": "按**输出/源行**下标遍历 (需要倒排: 源行 -> 碰它的那些)"}
    g = {}
    for t in r.get("traffic", []):
        k = KEY.get(t.get("pattern"))
        if not k:
            continue
        e = t.get("excess_s") or 0
        a = g.setdefault(k, {"rows": [], "excess_s": 0.0})
        a["rows"].append(t["name"])
        a["excess_s"] += e
    return sorted(({"remedy": k, **v} for k, v in g.items()), key=lambda x: -x["excess_s"])


def sorted_traffic(r: dict) -> list:
    """展示用的排序: 按**落 DRAM 多花的秒数**降序。

    ★单独抽出来是为了让 analyze 的输出保持与 spec 同序 (见 analyze 里的注释),
      同时让自检能测"排序规则"本身而不必依赖 analyze 顺带排好。
    """
    return sorted(r.get("traffic", []), key=lambda x: -(x.get("excess_s") or 0))


def report(r: dict) -> int:
    prov = r.get("dram_provenance", "未知")
    shaky = prov.startswith("[未标定]") or prov == "未知"
    print("=" * 78)
    print("★ core/hwfloor 是**初版, 仍在迭代**。下面的数字**仅供参考, 不可作为优化终点**。")
    if shaky:
        print("★★ 本次内存侧的带宽**没有实测标定**, 据它算出的内存下界是**假下界** ——")
        print(f"   {prov}")
        print("   => 只能用来**排序**(哪一项流量最离谱), **不许**用来定目标或判'还剩多少空间'。")
    else:
        print(f"   DRAM 带宽出处: {prov}")
    print("=" * 78)
    print(f"=== {r['name']} 硬件下界 ===")
    print(f"  cube  : {r['cube_floor_ms']:8.3f} ms")
    print(f"  内存  : {r['mem_floor_ms']:8.3f} ms  (强制流量 {r['compulsory_bytes']/1e6:.0f} MB)")
    for ln in verdict_lines(r["floor_ms"], r.get("measured_ms"), kind="capability",
                            bound_by=r["bound_by"]):
        print(ln)
    #  排序只在展示时做, 且对副本做 (见 analyze 里的注释)
    r = dict(r, traffic=sorted_traffic(r))
    ports = r.get("ports") or []
    if ports:
        print(f"\n=== 逐端口下界 (每一跳都可能先饱和; 只看 DRAM 会漏掉真正的墙) ===")
        print(f"  {'端口':<10}{'说明':<20}{'带宽GB/s':>10}{'强制':>10}{'实际':>10}"
              f"{'能力ms':>9}{'编排ms':>9}")
        for p_ in ports:
            b = f"{p_['bw']/1e9:.0f}" if p_["bw"] else "无参照"
            print(f"  {p_['port']:<10}{p_['desc']:<20}{b:>10}"
                  f"{p_['compulsory']/1e6:>8.0f}MB{p_['actual']/1e6:>8.0f}MB"
                  f"{(f'{p_[chr(99)+chr(97)+chr(112)+chr(95)+chr(109)+chr(115)]:.3f}' if p_['cap_ms'] else '—'):>9}"
                  f"{(f'{p_[chr(111)+chr(114)+chr(99)+chr(104)+chr(95)+chr(109)+chr(115)]:.3f}' if p_['orch_ms'] else '—'):>9}")
        _ach = [p_ for p_ in ports if p_.get("bw_achieved") and p_["bw"]
                and abs(p_["bw_achieved"] - p_["bw"]) / p_["bw"] > 0.02]
        for p_ in _ach:
            print(f"  ★{p_['port']} 的**编排ms 用达成带宽 {p_['bw_achieved']/1e9:.0f} GB/s** "
                  f"(平台 {p_['bw']/1e9:.0f}, 即平台的 {p_['bw_achieved']/p_['bw']*100:.0f}%); "
                  f"能力ms 仍用平台 —— 能力问'任何实现最好到哪'(顺带修访问形态), "
                  f"编排问'本实现的流量按它真实效率要多久'。")
        for _n in PORT_NOTES:
            print(f"  ★{_n}")
        if r.get("ports_missing_bw"):
            print(f"  ★以下端口**取不到带宽**, 本表查不了 (不是查过没问题): "
                  + ", ".join(f"{x[0]}({x[1]} 缺)" for x in r["ports_missing_bw"]))
    of = r.get("orchestration_floor_ms")
    if of and r.get("measured_ms"):
        cap = r["floor_ms"]; meas = r["measured_ms"]
        print(f"\n=== 差距分成两段 (这是本模块最要紧的一张表) ===")
        print(f"  能力下界      {cap:8.3f} ms   <- 强制流量 / 端口能力; 任何实现都不可能更快")
        print(f"  编排下界      {of:8.3f} ms   <- **本实现真实搬的流量** / 端口能力, 且假定完美重叠")
        print(f"  实测          {meas:8.3f} ms")
        if of >= meas:
            #  ★编排下界 >= 实测: 本实现已经贴住它自己的流量下界。
            #    这不是"算错了"就完事 —— 它是一个很强的结论: **重叠已经不是问题**,
            #    再怎么调流水/屏障/发射次序都拿不到东西, 唯一的方向是减流量。
            #    同时它给出一条自检: 差得太多 (>20%) 时要回头查流量估计或带宽标定。
            over = (of - meas) / meas
            print(f"  ── 实测 - 编排下界 = {meas - of:7.3f} ms  **编排下界已经超过实测**")
            print(f"     => 本实现**已经贴住它自己的流量下界**, 重叠不是问题。"
                  f"再调流水/屏障/发射次序拿不到东西 —— **唯一方向是减流量**。")
            #  ★下界反超实测 = 两个输入至少有一个错, 而**光说"回头查"没人查得动**。
            #    把它算成一个数: 要与实测自洽, 实际字节最多只能是多少, spec 给的高出多少。
            #    2026-09-18 实撞: spec 给 dram 实际 28.52GB, 而 L2 miss 计数器实测 24.28GB
            #    (runs/bytes_pair_20260918), 高出 17% —— 正好是这条反超的来源。
            dport = next((p_ for p_ in ports if p_["port"] == "dram"), None)
            if dport and dport.get("bw_achieved") and dport.get("actual"):
                budget = meas / 1e3 * dport["bw_achieved"]
                print(f"     ★**下界不可能超过实测 —— 两个输入必有一个错**, 算给你看:")
                print(f"        按达成带宽 {dport['bw_achieved']/1e9:.0f} GB/s, "
                      f"{meas:.3f} ms 最多只搬得动 **{budget/1e9:.2f} GB**, "
                      f"而 spec 说实际搬了 {dport['actual']/1e9:.2f} GB "
                      f"(**高出 {dport['actual']/budget-1:+.0%}**)。")
                print(f"        所以要么 spec 某条流的字节估高了, 要么达成带宽比这个数高。"
                      f"两者都可实测: 字节用 core/probe/bytes_report.py (L2 miss x 颗粒), "
                      f"带宽用同一次采集的 字节/时长。")
            elif over > 0.20:
                print(f"     ★但超出 {over:.0%} 偏多, 回头查两处: (a) 某条流的字节数估高了;"
                      f" (b) 实际达成带宽高于标定值。")
        else:
            print(f"  ── 实测 - 编排下界 = {meas - of:7.3f} ms  ({(meas-of)/meas:.0%})"
                  f"  **重叠没做好** -> 改编排 (软件流水/屏障/发射次序)")
        print(f"  ── 编排 - 能力下界 = {of - cap:7.3f} ms  ({(of-cap)/meas:.0%})"
              f"  **搬了不该搬的** -> 改数据流 (多余往返/复用不足/粒度)")
        print(f"  ★两段对应完全不同的动作。把它们混在一起看, '还差 N 倍'这句话就无法指导下一步 ——"
              f"\n    那正是 sfag 2026-09-18 之前的状态。")
    if r.get("l2_contended"):
        print(f"\n=== ★L2 竞争 ===")
        print(f"  各流工作集之和 = **{r['concurrent_footprint']/1e6:.0f} MB**, "
              f"而 L2 只有 {r['l2_bytes']/1e6:.0f} MB (**{r['concurrent_footprint']/r['l2_bytes']:.1f} 倍**)")
        print(f"  ★**单条流的 footprint <= L2 不等于它能常驻** —— 别的流会把它挤出去。")
        print(f"    所以下面标 'L2' 的那几条是**乐观口径**, 而「把某条流压进 L2」这类候选")
        print(f"    在竞争没解决之前**拿不到它承诺的收益**。")
        print(f"    sfag 2026-09-18 实撞: 据'把 mm4/mm5 从 428MB 压到 107MB 就进 L2'排了一整轮候选,")
        print(f"    而分块后三条流合计仍是 L2 的 2.8 倍。")
    if r.get("l2_bytes"):
        print(f"\n=== 流量分流 (工作集 <= L2 {r['l2_bytes']/1e6:.0f} MB 的由 L2 供给, 不占 DRAM) ===")
        print(f"  落 DRAM {r['dram_actual_bytes']/1e9:.2f} GB | 由 L2 供给 "
              f"{r['l2_actual_bytes']/1e9:.2f} GB | 合计 {r['actual_bytes']/1e9:.2f} GB")
    print(f"\n=== 超额表 (实际 / 强制, 按**落 DRAM** 多花的秒数排序) ===")
    #  ★2026-09-20 实测反例, 必须打在表头: 「多花 N ms」是**字节除以带宽**得到的,
    #    它隐含"这条流独占带宽"。真机上有流水重叠, 去掉一条流的搬入, 时间被别的流填回去。
    #    sfag: 本表说「mm4/mm5 读回」多花 12.76ms, 而定位探针 p7 直接跳过那整条流,
    #    端到端只省 **0.88ms** —— **差 14 倍** (runs/pipe_p7 vs runs/pipe_ref12104)。
    #    不打这句, 读表的人 (包括我) 会把这一列当成"去掉它能省多少"的预算去挑方向。
    print("  ★**「多花」是字节/带宽算的, 隐含'这条流独占带宽'** —— 真机有流水重叠。")
    print("     实测反例 (sfag 2026-09-20): 本表说「mm4/mm5 读回」多花 12.76ms,")
    print("     而探针直接去掉整条流, 端到端只省 **0.88ms**, **差 14 倍**。")
    print("     ⇒ 这一列**只能排序**, 不是'去掉它能省多少'。那个数只能实测 (删该流的定位探针)。")
    print(f"  {'项':<22}{'强制':>10}{'实际':>10}{'工作集':>10}{'供给':>6}{'倍数':>8}{'多花':>10}")
    any_row = False
    for t in r["traffic"]:
        a = f"{t['actual']/1e6:.0f}MB" if t["actual"] is not None else "—"
        fp = f"{t['footprint']/1e6:.0f}MB" if t.get("footprint") else "—"
        src = ("L2" if t.get("resident") else "DRAM") if t.get("resident") is not None else "?"
        rt = ("∞" if t["ratio"] == float("inf") else
              (f"{t['ratio']:.1f}x" if t["ratio"] else "—"))
        ex = f"{t['excess_s']*1e3:.2f}ms" if t.get("excess_s") else "—"
        print(f"  {t['name']:<22}{t['compulsory']/1e6:>7.0f}MB{a:>10}{fp:>10}{src:>6}{rt:>8}{ex:>10}")
        if t.get("roundtrip"):
            #  ★"两端都在片上 -> 这趟往返可以去掉" **只在有 CV 直通路径的芯片上成立**。
            #    2026-09-18 实撞: 910B2 的 CALIB 明写 cv_direct_path=False —— cube 与 vector
            #    是分立物理核, CV 数据**必须**经 GM 中转 (950PR 才有直通, 且是 measured)。
            #    而本模块此前一律打"数据没有离开芯片的必要", 把一个**物理上不可能**的改法
            #    排在超额表第一位 —— 我差一步就为它烧一轮 40 分钟的构建。
            #    ★判据给线索不给处方, 但**不能给一条硬件上不存在的处方**。
            if _cv_direct() is False:
                print(f"      ★**CV 中转**: {t['producer']} 写 -> {t['consumer']} 读。"
                      f"本芯片 **cv_direct_path=False** (cube/vector 分立物理核), "
                      f"这趟往返**去不掉** —— 硬件要求 CV 数据经 GM。")
                print(f"        能做的是让它的**工作集进 L2**, 从而不落 DRAM; "
                      f"「消除」与「让它由 L2 供给」是两个完全不同的工程问题, 别混。")
            else:
                print(f"      ★**GM 纯中转**: {t['producer']} 写 -> {t['consumer']} 读, 两端都在片上。"
                      f"这趟往返本身就是候选 —— 本芯片有 CV 直通路径, 数据不必走 GM。")
        if t["ratio"] == float("inf"):
            #  ★measure 的是 **DRAM** 强制量。当这笔是本芯片去不掉的 CV 中转时,
            #    "整笔都不必要"会与上一行的"去不掉"在**同一屏**上自相矛盾 ——
            #    二测批评过的正是这类 (两条判据同屏给出相反方向)。措辞必须分开 GM 与 DRAM。
            if t.get("roundtrip") and _cv_direct() is False:
                print(f"      ★强制的 **DRAM** 字节为 0 —— 意思是这笔**可以完全由 L2 供给**, "
                      f"不是'整笔不必要'。它必须经过 GM (见上一行)。")
            else:
                print(f"      ★强制量为 0 -> **整笔都不必要**, 不是'倍数很大'。")
        for ln in reorder_requirement(t, r.get("l2_bytes") or 0):
            print(f"      {ln}")
        if t["note"]:
            print(f"      {t['note']}")
        any_row = True
    if not any_row:
        print("  (spec 里没有 traffic 项 —— 那么这份报告只判了 cube, 内存侧没查)")
    wc = workset_curve_estimate(r)
    if wc:
        print(f"\n=== 工作集曲线 [实测标定] ===")
        print(f"  标定条件: {wc['unit']}")
        print(f"  测于 {wc['verified']}" + (f", 此前已被推翻过 {wc['superseded']} 次" if wc["superseded"] else ""))
        #  ★2026-09-18 实撞: 我把一条**测于 2026-09-15 旧构建**的曲线标成"比模型带宽可信得多"
        #    接进本模块; 当天在 g15 上按同样方法重测, 同一工作集下**绝对值差近 2 倍**
        #    (71.3MB: 15.24 -> 29.375)。原因是它 held_fixed 里的「调度/分核/核数」
        #    已被 g3~g15 一系列改动动过。
        #    => 标定的**绝对值**只在测它的那个构建上成立; 跨档的**比值**才是可移植的那一半。
        print(f"  ★★**绝对值只在测它的那个构建上成立** —— 换了实现必须重测。")
        print(f"     2026-09-18 实撞: 一条测于三天前旧构建的曲线, 同一工作集下绝对值差近 2 倍。")
        print(f"     可移植的是**跨档的比值** ({wc['worst_ms']/wc['best_ms']:.2f} 倍),"
              f" 不是那一组毫秒数。")
        for b, ms in wc["points"]:
            print(f"    工作集 {b/1e6:7.1f} MB -> {ms:6.2f} ms")
        print(f"  本实现落 DRAM 的最大工作集 = **{wc['workset']/1e6:.0f} MB**"
              + ("" if wc["inside"] else "  ★**超出标定范围, 只能外推, 误差未知**"))
        print(f"  ★曲线说明一件模型带宽表达不出来的事: **同样的字节, 工作集不同, 时间差 "
              f"{wc['worst_ms']/wc['best_ms']:.2f} 倍** ({wc['worst_ms']:.2f} -> {wc['best_ms']:.2f} ms)。")
        print(f"    => 「减流量」与「压工作集」是**两条独立的路**, 后者**不用动一个字节**。")
        print(f"  ★★**不要把本曲线的结论套到下面这些被固定住的量上** (expdesign E7):")
        print(f"     结论只适用于扫描**真正变动的那个自变量**; held_fixed 的量在结论之外。")
        print(f"     sfag 2026-09-18 实撞: 据本曲线的 12.0 ms 去论证 mm4/mm5 交接 workspace 的收益,")
        print(f"     而那个 workspace 的大小与索引分布无关 —— 它在五档扫描里是**恒定**的。")
        print(f"  ★引用前必须逐条核对 held_fixed (改了其中任何一条, 曲线不适用):")
        for h in wc["held_fixed"]:
            print(f"      {'/'.join(h.get('keys', []))}: {h.get('note','')[:80]}")
        if wc["counterexample"]:
            print(f"  ★★该曲线自带一个**方向相反**的反例, 必读:")
            print(f"      {wc['counterexample'][:230]}")
    grp = shared_remedy_groups(r)
    if grp and any(len(x["rows"]) > 1 for x in grp):
        print(f"\n=== ★共用同一个改法的流 (它们的超额要**加起来**看) ===")
        for x in grp:
            if len(x["rows"]) < 2:
                continue
            print(f"  改法: {x['remedy']}")
            print(f"    覆盖 {len(x['rows'])} 条流, 合计多花 **{x['excess_s']*1e3:.2f} ms**:")
            for nm in x["rows"]:
                print(f"      - {nm}")
        print("  ★逐条排序会**系统性低估**需要大改动的方向 —— 大改动往往同时解决多条流,")
        print("    而表把它的收益拆散了。**这正是'候选池里没有大项'这种错觉的来源之一。**")
        print("    2026-09-18 实撞 (sfag): 这几条各自只有 2~4 ms, 看着都不够大;")
        print("    而参考实现用**一个**改动同时解决了它们, 实测 34.422 -> 22.485 ms (差 11.94 ms)。")
    cov = r.get("coverage")
    if cov:
        print(f"\n=== spec 完备性 ===")
        print(f"  实测 {r['measured_ms']:.2f} ms 的时间预算 = 这段时间**至多**能搬 "
              f"{cov['time_budget_bytes']/1e9:.1f} GB; spec 里落 DRAM 的只有 "
              f"{cov['dram_actual']/1e9:.1f} GB ({cov['ratio']:.0%})")
        if cov["ratio"] < 0.5:
            print(f"  ★**spec 很可能漏了流量** —— 只覆盖了时间预算的 {cov['ratio']:.0%}。")
            print(f"    2026-09-18 实撞: 手列六项就下结论, 漏掉 cube<->vector 交接往返 18.25 GB "
                  f"(最大的一笔), 于是超额表第一行指错了地方, 下界与倍数也都是错的。")
            print(f"    逐函数过一遍所有 GlobalTensor / workspace, 尤其是**中间量** —— "
                  f"算子接口上的输入输出最容易列全, 也最不容易是大头。")
        elif cov["ratio"] > 1.2:
            print(f"  ★spec 的流量**超过**了时间预算 {cov['ratio']:.0%} —— 要么重复计了某一项, "
                  f"要么某条流其实由 L2 供给 (给它补 footprint), 要么带宽标定偏低。")
    print("\n★下界是**对偶的上界**: 没有实现能比它快。它与现有实现无关, 所以它是"
          "\n  唯一不会被既有方案框住的标尺。它不预测收益, 也不给改法 —— "
          "\n  但超额表最上面那一行就是下一个候选该去的地方。")
    #  ★尾横幅: 首横幅在屏幕外时人只看得到结论。一个假下限被当成目标, 后果是整轮跑偏 ——
    #    2026-09-18 我自己就把 26.794 ms 这个由未标定带宽算出的数报了出去。
    print("\n" + "=" * 78)
    print("★ 再说一次: core/hwfloor **初版**, 上面每个数**仅供参考**。")
    if shaky:
        print("★★ 本次的内存侧下界建立在**未标定**的带宽上 —— 它是**假下界**。")
        print("   ★**不要拿它当优化目标**, 也不要据它说'还剩多少空间'。它只能排序。")
        print("   要拿到真下界: 跑一个工作集**远大于 L2** 的纯流式负载标定 dram_GBps,")
        print("   写进 core/chip_calib.json (CALIB 里记着两次错误标定的教训, 先读)。")
    print("★ 可信度排序: [实测标定] 工作集曲线 > [模型值] ini 端口速率 > [未标定] 任何据它算的数。")
    _print_trust_tail()


def _print_trust_tail() -> None:
    """把本模块各判据**当前的信任等级**与未修的对账落空直接打在输出里。

    ★2026-09-18 立: 光写"仅供参考"不够 —— 它对每次输出都一样, 于是读的人会脱敏。
      真正让人停手的是**具体的数**: "这条判据上次对账偏 1.76 倍, 还没修"。
      等级与落空记录都在 core/_trust_ledger.jsonl, 由 core.trust 算, 这里只负责**把它打出来**。
    """
    #  ★2026-09-18 改成调 trust.verdict_banner 这一个共用入口 —— 此前 hwfloor 自己写了一份,
    #    于是 `trust adoption` 把它算成"没接入"。同一件事两份实现, 第二份必然先过期。
    try:
        from core import trust
        for _ln in trust.verdict_banner("hwfloor", refusing=False, prefix=" "):
            print(_ln)
        led = trust.read_ledger()
        miss = [r for r in led if r.get("op") == "resolve"
                and r.get("verdict") in ("miss", "violated", "loose")
                and (r.get("criterion") or "").startswith("hwfloor.")]
    except Exception as e:
        print(f"★ 信任等级取不到 ({type(e).__name__}) —— **按最低信任处理**, 不要当默许。")
        return
    for m in miss:
        print(f"★★ 未修的对账落空: {m['criterion'].split('.', 1)[1]} 预测 {m['predicted']} → "
              f"实测 {m['actual']} {m.get('unit','')} = **{m['ratio']}x**")
        print(f"    偏差方向是{'低估' if m['ratio'] > 1 else '高估'} —— "
              f"{'即本模块把真实可省空间算小了, 结构性大改会被它系统性筛掉' if m['ratio'] > 1 else '即本模块高估了收益'}。")
    print("★ infra 本身仍在迭代; 发现它算错了, **先修工具再继续优化** —— "
          "一条算错的判据比没有判据危险, 它看起来像结论。")
    print("=" * 78)
    return 0


def check_no_divergence(core_dir: str | None = None) -> list:
    """判词有没有又散回去 —— **合并之后必须有东西守住它**, 否则半年后又是六处。

    2026-09-18 合并时的状态: 判词集中到 verdict_lines / FLOOR_KIND 两处,
    attnplan / mmplan 调用它算能力下界, sortplan / pipeplan 调用它标 critpath 种类,
    scalarplan 的那条经验地板在 REFS 里挂了种类。
    ★守法: 除本模块外, 任何模块里出现判词的**特征短语**就报出来。
      特征短语要选**只会出现在判词里**的 (不是 "地板" 这种到处都有的词), 否则天天误报。
    """
    import glob
    core_dir = core_dir or os.path.dirname(os.path.abspath(__file__))
    #  这些短语是判词独有的; 领域模块算自己的地板不会用到它们
    PHRASES = ["默认动作是**结构性**", "最大可得加速比", "相当于最多还能快", "拒绝给倍数"]
    out = []
    for f in sorted(glob.glob(os.path.join(core_dir, "*.py"))):
        base = os.path.basename(f)
        if base in ("hwfloor.py",) or base.startswith("_"):
            continue
        try:
            txt = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for ph in PHRASES:
            if ph in txt:
                out.append({"module": base, "phrase": ph,
                            "err": (f"{base} 里出现了判词特征短语 {ph!r} —— 判词的唯一出处是 "
                                    f"core.hwfloor.verdict_lines。领域模块只负责算出地板值与"
                                    f"声明它是哪一种 (FLOOR_KIND), 措辞与阈值不在领域模块里。"
                                    f"判词散开的后果是**判读不一致**: 同一个 3 倍差距, 一处写"
                                    f"'还有空间'另一处写'已贴住地板', 而读的人不知道两者用的"
                                    f"是不同种类的地板。")})
    return out


def selftest() -> int:
    bad, n = 0, 0

    def ok(msg):
        nonlocal n
        n += 1
        print("  过 " + msg)

    spec = {"name": "t", "soc": {"cube_flops": 1e12, "dram_bytes_per_s": 1e9},
            "vars": {"A": 1000}, "compute": [{"name": "mm", "macs": "A*A"}],
            "traffic": [{"name": "x", "compulsory": "A", "actual": "A*10"},
                        {"name": "y", "compulsory": "A*1000", "actual": "A*1000"}]}
    r = analyze(spec, measured_ms=100.0)
    #  cube = 2*1e6/1e12 = 2e-6 s; mem = (1000 + 1e6)/1e9 = 1.001e-3 s -> 内存受限
    if r["bound_by"] != "memory":
        print(f"自检失败: 该例内存受限, 实得 {r['bound_by']}"); bad += 1; n += 1
    else:
        ok("受限侧判定: 取 cube 与内存的大者")
    if abs(r["multiple"] - 100.0 / (1.001)) > 1:
        print(f"自检失败: 倍数算错 {r['multiple']}"); bad += 1; n += 1
    else:
        ok("倍数 = 实测 / 下界")
    #  ★排序按秒不按倍数: x 是 10 倍但只多花 9000 B, y 是 1 倍。x 该在前, 但若有更大的项则让位
    spec2 = dict(spec, traffic=[{"name": "小而倍数高", "compulsory": 1, "actual": 1000},
                                {"name": "大而倍数低", "compulsory": 1e9, "actual": 2e9}])
    r2 = analyze(spec2)
    if sorted_traffic(r2)[0]["name"] != "大而倍数低":
        print(f"自检失败: 超额应按**多花的秒数**排序, 不按倍数; 实得 {r2['traffic'][0]['name']}")
        bad += 1; n += 1
    else:
        ok("排序按多花的秒数 (倍数高但量小的不该排在前面)")
    #  地板为 0 / 倍数离谱时必须拒绝给结论 (合并 attnplan 时当场撞见)
    ls = verdict_lines(0.0, 34.909)
    if not any("拒绝给倍数" in x for x in ls):
        print("自检失败: 地板为 0 时竟然照样给倍数 —— 自信的荒谬数字比报错危险"); bad += 1; n += 1
    else:
        ok("地板为 0 时拒绝给倍数 (输入错了, 不是真有这么大空间)")
    #  ★用**真实**的地板值 1.6505 而不是四舍五入后的 1.651 —— 后者给 21.1x。
    #    判据的输入自己就不该是二手的近似值 (与 U7 那次手传 dimDAlign=512 同一形态)。
    ls = verdict_lines(1.650510178739757, 34.909, bound_by="cube")
    if not any("21.2x" in x for x in ls) or not any("结构性" in x for x in ls):
        print(f"自检失败: 正常倍数应给出 21.2x 与结构性提示, 实得 {ls}"); bad += 1; n += 1
    else:
        ok("正常倍数: 21.2x 且 >=2 时提示走结构性改动")
    ls = verdict_lines(1.0, 1.1, kind="critpath")
    if not any("不是能力下界" in x for x in ls):
        print("自检失败: critpath 种类必须声明它随分核/编排变"); bad += 1; n += 1
    else:
        ok("非能力下界要声明它随分核/编排变 (别拿它论证没空间)")
    #  --- 2026-09-18 扩的四个维度各配正负例 ---
    SOC = {"cube_flops": 1e12, "dram_bytes_per_s": 1e9, "l2_bytes": 100.0,
           "rate_B_per_cycle": {"ddr_read_rate": 1}, "freq_ghz": 1.0, "cube_cores": 1}
    #  footprint <= L2 -> 由 L2 供给, 不该计进 DRAM 超额时间
    r = analyze({"name": "t", "soc": SOC, "vars": {}, "compute": [],
                 "traffic": [{"name": "小工作集", "compulsory": 1, "actual": 1000, "footprint": 50},
                             {"name": "大工作集", "compulsory": 1, "actual": 1000, "footprint": 500}]})
    res = {t["name"]: t for t in r["traffic"]}
    if not res["小工作集"]["resident"] or res["大工作集"]["resident"]:
        print(f"自检失败: footprint 与 L2 比较判 resident 出错"); bad += 1; n += 1
    elif res["小工作集"]["excess_s"] != 0.0 or not res["大工作集"]["excess_s"]:
        print(f"自检失败: 由 L2 供给的流量不该计进 DRAM 超额时间"); bad += 1; n += 1
    else:
        ok("工作集 <= L2 的流量由 L2 供给, 不计进 DRAM 超额")
    #  compulsory==0 -> 整笔不必要, 不是"倍数很大"
    r = analyze({"name": "t", "soc": SOC, "vars": {}, "compute": [],
                 "traffic": [{"name": "中转", "compulsory": 0, "actual": 1000,
                              "producer": "cube", "consumer": "vector"}]})
    t0 = r["traffic"][0]
    if t0["ratio"] != float("inf"):
        print(f"自检失败: 强制量为 0 时应报 inf (整笔不必要), 实得 {t0['ratio']}"); bad += 1; n += 1
    elif not t0["roundtrip"]:
        print("自检失败: 两端都在片上的中间量应标 roundtrip (GM 纯中转)"); bad += 1; n += 1
    else:
        ok("强制量为 0 记作'整笔不必要'; 两端都在片上标为 GM 纯中转")
    #  完备性: spec 里落 DRAM 的字节远小于时间预算 -> 必须报"可能漏了流量"
    r = analyze({"name": "t", "soc": SOC, "vars": {}, "compute": [],
                 "traffic": [{"name": "一点点", "compulsory": 1, "actual": 100}]}, measured_ms=1000.0)
    if not r["coverage"] or r["coverage"]["ratio"] > 0.5:
        print(f"自检失败: 只覆盖极小比例时应判为'可能漏了流量', 实得 {r.get('coverage')}")
        bad += 1; n += 1
    else:
        ok("完备性: 落 DRAM 字节远小于时间预算 -> 判为 spec 可能漏了流量")
    #  端口取不到带宽必须列出来, 不许静默当没有
    #  ★socspec 现在会自动注入速率表, 所以要显式清空才能造出"取不到"的场景。
    #    这条自检本身也因此变了含义: 从"没人填"变成"连 socspec 都给不出"。
    r = analyze({"name": "t", "soc": {"cube_flops": 1e12, "dram_bytes_per_s": 1e9,
                                      "rate_B_per_cycle": {}},
                 "vars": {}, "compute": [], "traffic": [{"name": "x", "compulsory": 1, "actual": 2}]})
    if not r["ports_missing_bw"]:
        print("自检失败: 拿不到端口速率时必须列出来 (没查成 != 查过没问题)"); bad += 1; n += 1
    else:
        ok("端口取不到带宽要显式列出, 不静默略过")
    #  --- HF4: 超额表要说"怎么才能到达强制量" ---
    t_sc = {"name": "散射累加", "pattern": "scatter_accumulate", "ratio": 16.0,
            "actual": 4.8e9, "compulsory": 3e8,
            "out_rows": 131072, "row_bytes": 2304, "contributions": 2097152}
    lines = reorder_requirement(t_sc, l2_bytes=201e6)
    txt = " ".join(lines)
    if not lines:
        print("自检失败: scatter_accumulate 的超额流必须给出遍历顺序要求"); bad += 1; n += 1
    elif "按**输出**下标遍历" not in txt or "倒排" not in txt:
        print(f"自检失败: 必须点明'按输出下标遍历'与'倒排'这两件事"); bad += 1; n += 1
    elif "85 行" not in txt or "64 行" not in txt:
        print(f"自检失败: 必须由 片上容量/每行字节 推出块宽上界与 2 的幂取值"); bad += 1; n += 1
    else:
        ok("超额表给出遍历顺序要求 + 由容量反推块宽 (真 spec 上推出 64, 与参考实现一致)")
    #  负例: 非散射累加的流不许出这一段 (判据过宽会天天响)
    ls_ = reorder_requirement({"name": "流式", "ratio": 1.0, "actual": 1, "compulsory": 1})
    if ls_:
        print(f"自检失败: 不超额且没标 pattern 的流应沉默, 实得 {ls_}"); bad += 1; n += 1
    else:
        ok("不超额的流沉默, 不过宽")
    #  ★超额明显却没标 pattern 时必须**主动提示补标** —— 否则又回到"工具不被触发"
    ls2 = reorder_requirement({"name": "未标", "ratio": 16.0, "actual": 10, "compulsory": 1})
    if not ls2 or "没标" not in " ".join(ls2):
        print("自检失败: 超额 16x 却没标 pattern 时, 应提示补标而不是沉默"); bad += 1; n += 1
    else:
        ok("超额而未标 pattern 时主动提示补标 (沉默=工具不被触发)")

    #  --- gather 侧与散射侧同源, 且共用改法的流要能归组 ---
    lg = reorder_requirement({"name": "g", "pattern": "gather_reuse", "ratio": 16.0,
                              "actual": 10, "compulsory": 1, "out_rows": 131072,
                              "row_bytes": 1152, "contributions": 2097152})
    if not lg or "同一个倒排" not in " ".join(lg):
        print("自检失败: gather_reuse 要指出它与散射累加是同一个倒排"); bad += 1; n += 1
    else:
        ok("gather 侧与散射侧同源: 指出两者要的是同一个倒排")
    rr = {"traffic": [
        {"name": "写", "pattern": "scatter_accumulate", "excess_s": 0.003},
        {"name": "读改写", "pattern": "scatter_accumulate", "excess_s": 0.004},
        {"name": "取数", "pattern": "gather_reuse", "excess_s": 0.002},
        {"name": "无关", "excess_s": 0.010}]}
    gg = shared_remedy_groups(rr)
    if not gg or len(gg[0]["rows"]) != 3:
        print(f"自检失败: 三条同改法的流应归成一组, 实得 {gg}"); bad += 1; n += 1
    elif abs(gg[0]["excess_s"] - 0.009) > 1e-9:
        print(f"自检失败: 组内超额要**相加** (0.003+0.004+0.002), 实得 {gg[0]['excess_s']}")
        bad += 1; n += 1
    elif any("无关" in x["rows"] for x in gg):
        print("自检失败: 没标 pattern 的流不该被归进任何组"); bad += 1; n += 1
    else:
        ok("共用改法的流归一组且超额相加 (逐条排序会低估需要大改动的方向)")

    #  ★端口聚合两个真 bug 的回归用例 (2026-09-18 第二次盲测审计揪出, 它们让本模块
    #    给出**方向相反**的结论: 编排下界虚高 -> 打印"重叠不是问题, 再调流水拿不到东西")
    SOC2 = {"cube_flops": 1e12, "dram_bytes_per_s": 1e9, "l2_bytes": 100.0,
            "rate_B_per_cycle": {"ddr_read_rate": 1, "l2_read_rate": 10},
            "freq_ghz": 1.0, "cube_cores": 1, "vector_cores": 1}
    #  (1) resident 的流量不许记到 DRAM 端口上
    r = analyze({"name": "t", "soc": SOC2, "vars": {}, "compute": [],
                 "traffic": [{"name": "常驻L2", "compulsory": 1, "actual": 1e9, "footprint": 10},
                             {"name": "落DRAM", "compulsory": 1, "actual": 10, "footprint": 1e9}]})
    pd = {p_["port"]: p_ for p_ in r["ports"]}
    if pd.get("dram", {}).get("actual", 0) > 100:
        print(f"自检失败: resident 的流量被记进了 DRAM 端口 -> 编排下界虚高, 结论会反")
        bad += 1; n += 1
    elif pd.get("l2", {}).get("actual", 0) < 1e9:
        print("自检失败: resident 的流量应记到 l2 端口"); bad += 1; n += 1
    else:
        ok("由 L2 供给的流量记到 l2 端口, 不抬高 DRAM 的编排下界")
    #  (2) 端口配对不许受超额排序影响 (rows 排过序后再 zip 原始 traffic = 错位)
    r = analyze({"name": "t", "soc": SOC2, "vars": {}, "compute": [],
                 "traffic": [{"name": "小", "port": "dram", "compulsory": 1, "actual": 2},
                             {"name": "大", "port": "l2", "compulsory": 1, "actual": 1e8}]})
    pd = {p_["port"]: p_ for p_ in r["ports"]}
    if pd.get("l2", {}).get("actual", 0) != 1e8 or pd.get("dram", {}).get("actual", 0) != 2:
        print(f"自检失败: 端口配对错位 (rows 排过序) -> {[(x['port'], x['actual']) for x in r['ports']]}")
        bad += 1; n += 1
    else:
        ok("端口配对不受超额排序影响 (每行自带 port, 不再 zip)")
    #  判词不许散回领域模块 (合并之后要有东西守住)
    div = check_no_divergence()
    if div:
        for d in div:
            print(f"判词散落: {d['err']}")
        bad += len(div); n += 1
    else:
        ok("判词未散落: 全仓只有 core.hwfloor 写判词")
    #  ★负例 (2026-10-30 补): 上面那条**只走得到通过路径** —— 全仓当前确实没散落,
    #    于是它恒返回 []，判据逻辑整个坏掉 (比如 PHRASES 打错字) 自检照样绿。
    #    `opdbg --mutation --traced` 把这种情形单列成"检出路径未覆盖", 它就是这么找出来的。
    #    造一个**真的散落**喂进去, 检出路径才算被走过。
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _d:
        open(os.path.join(_d, "fake_domain.py"), "w", encoding="utf-8").write(
            "#  默认动作是**结构性**改动\n")
        open(os.path.join(_d, "hwfloor.py"), "w", encoding="utf-8").write(
            "#  默认动作是**结构性**改动\n")        # 本模块自己不算散落
        open(os.path.join(_d, "_priv.py"), "w", encoding="utf-8").write(
            "#  默认动作是**结构性**改动\n")        # 下划线开头的内部文件也跳过
        hits = check_no_divergence(_d)
        got = sorted(h["module"] for h in hits)
        if got != ["fake_domain.py"]:
            print(f"自检失败: 散落检出应恰为 ['fake_domain.py'], 实得 {got}"); bad += 1; n += 1
        else:
            ok("★负例: 领域模块里出现判词特征短语必须报出 (hwfloor.py 与 _ 开头的不算)")
    #  缺 soc 必须拒绝, 不许猜默认值
    try:
        #  socspec 可用时会自动补齐, 所以"缺"要写成**显式给 0** (等价于 socspec 也拿不到)。
        #  不猜默认值这条纪律没变, 变的是"谁来填" —— 现在是 socspec 填, 不是人手抄。
        analyze({"name": "t", "soc": {"cube_flops": 0, "dram_bytes_per_s": 0},
                 "vars": {}, "compute": [], "traffic": []})
        print("自检失败: 拿不到 soc 参数竟然放行 —— 猜默认值会把下界算低"); bad += 1; n += 1
    except SystemExit:
        ok("缺 soc 参数要拒, 不猜默认值")
    #  表达式引用未定义变量要明确报错, 不能静默当 0
    try:
        analyze({"name": "t", "soc": {"cube_flops": 1, "dram_bytes_per_s": 1},
                 "vars": {}, "compute": [{"name": "m", "macs": "UNDEFINED*2"}], "traffic": []})
        print("自检失败: 未定义变量竟然放行"); bad += 1; n += 1
    except SystemExit:
        ok("表达式引用未定义变量要报错, 不静默当 0")
    print(f"hwfloor 自检: {n} 条, {bad} 条未过")
    return bad


#  ======================================================================
#  --bench-report: 直接从 cann-bench 的评测报告出「逐 case 离访存下界几倍」
#  ======================================================================
#  沉淀动机 (cummin 2026-09-20, dubin 当面纠正「方向选错, 努力白费」):
#    本模块的 verdict_lines 早就写着「倍数 >= 2 -> 默认动作是结构性改动, 不是增量调参;
#    持续的小收益恰恰会让错误的框定显得在正常工作」。而 cummin 这一轮我在 **108x** 的
#    case 上调了三轮 tiling, 拿到 38% 的累计收益 —— 判词一个字都不用改, 我只是从没跑过它。
#    ★缺口不在规则, 在**绑定**: hwfloor 不在 CLAUDE.md 的「工具使用次序」里, 没有任何
#      环节强制在调参前算这个倍数。我把 109x / 51x 手算进了 problem.yaml 的注释, 然后
#      忽略了它蕴含的动作 —— 数字在手而决策没变, 比没算还危险。
#    所以这里不新增判据, 只把**已有判词**接到一个零成本的入口上: 一条命令, 任意算子,
#    不需要手写 spec。spec 那条路要求先把流量模型写对, 门槛高到我每次都跳过它。
#
#  流量口径怎么来的 (只用报告自身, 不手抄形状):
#    输入字节 = t_hw_us x 1.6e6   —— 官方 hardware_limit_us 的定义就是 输入字节/1.6TB/s,
#                                    所以可以反解。cummin case 1 反解 2.097 MB = 1048576 x 2 ✓
#    输出字节 = sum(每个输出的 total_count x dtype 字节)
#    ★官方 hardware_limit_us **只算输入**, 恒不可达, 不要拿它当目标。
#      cummin f16: 输出的 indices 是 int64, 独占总流量的 8/12 = 67%。
#
#  已知边界 (不夸大):
#    - 精度检查跳过的输出 (total_count == 0) 拿不到元素数, 这里用同 case 其它输出的
#      最大 total_count 顶上, 并在表里标 ~。顶错了流量会偏小 -> 倍数偏大。
#    - 分母用的是 1.6 TB/s 这个**官方口径常数**, 不是本机实测可达带宽
#      (chip_calib 里 ascend950pr 的 dram_GBps 状态是 uncalibrated)。所以倍数是
#      「离官方口径下界几倍」, 跨 case 比较有效, **绝对值不可当"还能快这么多"**。

_DT_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8, "int64": 8,
             "int32": 4, "int16": 2, "int8": 1, "uint8": 1, "bool": 1}


#  ---------------------------------------------------------------- 执行单元占用
#
#  为什么要有这一轴 (2026-09-21 SFA 实撞, dubin 当面指出两次):
#    SFA 的实现里 Mmad / Matmul / Fixpipe **一条都没有** —— 一个以矩阵乘为主的算子,
#    全部算术跑在向量单元上, 矩阵单元占用 0。而本模块此前只报访存下界, 报出的
#    「离下界 4950 倍」既不说明是哪个单元在干活, 也不说明该往算力还是往访存查。
#    我因此先后给出两个**相反**的结论, 两个都没有算术支撑:
#      先说「去掉屏障就行」(只在向量单元内部省指令, 矩阵单元照样 0);
#      再说「cube 空转是决定性的」(也没算 —— 算完发现实测只有向量天花板的 1.5%,
#      开销才是主导, 搬 cube 会把力气花在不承压的一侧)。
#    结论: 这个判断不能靠人记得问。它需要的三个数 (实测吞吐 / 向量天花板 /
#    矩阵天花板) 都是可算的, 所以做成判据。
#
#  本轴只回答"算力有没有用上、用的是哪个单元", 不回答怎么改。

CUBE_APIS = ("Mmad", "LoadData", "LoadDataWithTranspose", "Fixpipe", "FixpipeParams",
             "MatmulImpl", "Matmul<", "SetFmatrix", "InitConstValue", "LoadImageToLocal")
#  ★全流水屏障与标量往返: 它们出现在**按元素/按对**的循环里时, 是"离两个天花板都很远"
#    这一类的直接成因。静态计数不知道循环次数, 所以只作线索, 与实测吞吐合起来判。
OVERHEAD_APIS = ("PipeBarrier<PIPE_ALL>", "GetValue(", "SetValue(", "SyncAll")


def unit_usage(src: str) -> dict:
    """数一份 kernel 源码里各执行单元的接口出现次数。"""
    import glob as _glob
    files = []
    for ext in ("*.h", "*.hpp", "*.cpp", "*.cce"):
        files += _glob.glob(os.path.join(src, "**", ext), recursive=True)
    text = ""
    for f in files:
        try:
            text += open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            pass
    cube = {a: text.count(a) for a in CUBE_APIS if text.count(a)}
    over = {a: text.count(a) for a in OVERHEAD_APIS if text.count(a)}
    return {"files": len(files), "cube": cube, "cube_total": sum(cube.values()),
            "overhead": over, "overhead_total": sum(over.values())}


def unit_ceilings() -> dict:
    """两个执行单元的算力天花板 (T MAC/s)。取数只走 core.socspec。"""
    from . import socspec as _ss                                   # noqa: PLC0415
    d = _ss.spec().get("derived") or {}
    return {"cube_fp16": d.get("peak_cube_fp16_TMACs"),
            "vector_fp32": d.get("peak_vector_fp32_TMACs"),
            "vector_fp16": d.get("peak_vector_fp16_TMACs")}


def msprof_occupancy(prof_dir: str) -> dict:
    """从 msprof 的 PipeUtilization_*.csv 读**流水占用率** (活跃时间 / 总时间)。

    ★占用率与吞吐效率是两件事, 混为一谈会得出相反的结论 (2026-09-21 SFA 实撞,
      见 unit_verdict 的注释)。这里只报占用, 效率由 macs 那一路算。
    """
    import csv as _csv                                            # noqa: PLC0415
    import glob as _glob                                          # noqa: PLC0415
    out: dict = {}
    files = _glob.glob(os.path.join(prof_dir, "**", "PipeUtilization_*.csv"), recursive=True)
    for f in files:
        try:
            rows = list(_csv.DictReader(open(f, errors="ignore")))
        except OSError:
            continue
        for r in rows:
            for k, v in r.items():
                if not k.endswith("_ratio") or v in ("NA", "", None):
                    continue
                try:
                    out.setdefault(k, []).append(float(v))
                except ValueError:
                    pass
    return {k: sum(v) / len(v) for k, v in out.items() if v}


def unit_verdict(usage: dict, measured_us: float | None = None,
                 macs: float | None = None, occ: dict | None = None) -> list:
    """把"单元占用 + 流水占用率 + 实测吞吐 + 两个天花板"合成判词。

    ★这个函数的第一版有一条**错判**, 2026-09-21 SFA 实撞:
      原判词是「实测不到向量天花板的一成 -> 瓶颈不是算力 -> 此时搬 cube 是在不承压的
      一侧用力」。它把**吞吐效率**当成了**流水占用**。同一次测量里:
        吞吐效率 = 0.192 / 2.96 = 6.5%   (每周期搬的数据少)
        流水占用 = msprof aiv_vec_ratio = 78.4%   (向量流水忙了 78% 的墙上时间)
        cube 占用 = 0 (aic 各列全 NA)
      向量流水忙到 78% 就说明它**正是**瓶颈; 效率低只说明它在发一堆小指令。
      此时两条路都对: 加大单条指令的工作量, 或者把矩阵乘搬到空闲的 cube。
      原判词把后一条挡掉了, 而那恰恰是这个算子的出路 (P@V 本身就是矩阵乘)。
    所以: 没有 occ 时不给"该不该换单元"的结论, 只说缺什么。
    """
    c = unit_ceilings()
    out: list[str] = []
    cube_n = usage.get("cube_total", 0)
    ratio = (c["cube_fp16"] / c["vector_fp32"]) if (c["cube_fp16"] and c["vector_fp32"]) else None
    out.append(f"算力天花板: cube fp16 {c['cube_fp16']:.1f} T MAC/s  |  "
               f"vector fp32 {c['vector_fp32']:.2f} T MAC/s"
               + (f"  (相差 {ratio:.0f} 倍)" if ratio else ""))
    if cube_n == 0:
        out.append("★源码里矩阵单元接口 **0 条** (Mmad/Matmul/Fixpipe 都没有) —— "
                   "全部算术在向量单元, 上限是上面那个小的数。")
    else:
        out.append(f"源码用到矩阵单元接口 {cube_n} 条: "
                   + ", ".join(f"{k}x{v}" for k, v in sorted(usage['cube'].items())))
    if usage.get("overhead_total"):
        out.append("同步/标量往返: "
                   + ", ".join(f"{k.rstrip('(')}x{v}"
                               for k, v in sorted(usage["overhead"].items()))
                   + "  —— 若它们在按元素/按对的循环里, 实测会同时远低于两个天花板。")
    fv = None
    if measured_us and macs:
        got = macs / (measured_us * 1e-6) / 1e12
        fv = got / c["vector_fp32"] if c["vector_fp32"] else None
        fc = got / c["cube_fp16"] if c["cube_fp16"] else None
        out.append(f"实测吞吐 {got:.4f} T MAC/s = 向量天花板的 {fv * 100:.1f}% (吞吐效率)"
                   f" / 矩阵天花板的 {fc * 100:.3f}%")
    vocc = (occ or {}).get("aiv_vec_ratio")
    socc = (occ or {}).get("aiv_scalar_ratio")
    mocc = (occ or {}).get("aiv_mte2_ratio")
    cocc = (occ or {}).get("aic_cube_ratio")
    if occ:
        parts = [f"{k.replace('_ratio', '')} {v * 100:.1f}%"
                 for k, v in sorted(occ.items(), key=lambda kv: -kv[1])[:5]]
        out.append("流水占用 (msprof): " + "  ".join(parts))

    if fv is None or vocc is None:
        miss = []
        if fv is None:
            miss.append("--macs <总乘加数> (算吞吐效率)")
        if vocc is None:
            miss.append("--msprof <msprof 产物目录> (读流水占用)")
        out.append("(缺 " + " 与 ".join(miss) + " —— **吞吐效率与流水占用是两件事**, "
                   "只有一个就下'该不该换单元'的结论必错, 见本函数注释)")
        return out

    busy = vocc >= 0.60
    if busy and fv < 0.30:
        out.append(f"→ **向量流水忙 {vocc * 100:.0f}% 而吞吐效率只有 {fv * 100:.1f}%**: "
                   "瓶颈就是向量流水, 但它在发一堆小指令。两条路都对 —— "
                   "①加大单条指令的工作量 (合并 repeat、成批处理); "
                   f"②把矩阵乘搬到{'空闲的 ' if not cocc else ''}矩阵单元"
                   f"{'(aic 占用 %.0f%%)' % (cocc * 100) if cocc else '(aic 零占用)'}。")
        if socc and socc >= 0.40:
            out.append(f"   标量占用 {socc * 100:.0f}% 也高: 内层循环里的 GetValue/SetValue "
                       "与地址算式同样要一起消掉, 只改向量侧会被标量挡住。")
        if mocc is not None and mocc < 0.25:
            out.append(f"   搬运占用只有 {mocc * 100:.0f}%: 访存不是瓶颈, "
                       "减少搬运次数的改动不会有整体收益 (实证: 减半 gather 反而更慢)。")
    elif busy and fv >= 0.30:
        out.append(f"→ 向量流水忙 {vocc * 100:.0f}% 且吞吐效率 {fv * 100:.1f}%: "
                   "向量算力接近打满, 结构性动作是换到矩阵单元。")
    else:
        out.append(f"→ 向量流水只忙 {vocc * 100:.0f}%: 既不算也不忙, "
                   "时间花在等待上 —— 查同步、串行依赖与启动开销, 先看 msprof 的 wait 列。")
    return out


def _better_run_exists(report_dir: str) -> str:
    """同一个 case 目录里有没有**更好**的轮次。有就返回一句提醒, 没有则空串。

    ★为什么要这条 (2026-09-22 实撞, 本会话第四次同型): 我拿 `runs/colseg_off` 当基线做了
      整轮分析, 而那是一个 **A/B 对照组**(把 colseg 关掉的那一支), 7362 us / 性能分 19.70;
      同目录下 `rep3` 是 7239 us / 20.11。**「最近改动的一轮」不等于「当前最好的一轮」** ——
      轮次标签里编码着变体 (xxx_off / xxx-ungated / bis-* 都是对照组), 而人往往
      按时间戳挑最新的那个。
    ★只提醒不改判: 有时你**就是**想看对照组。
    """
    import glob
    import json
    d = os.path.dirname(os.path.abspath(report_dir))          # runs/
    if os.path.basename(d) != "runs":
        return ""
    cur = os.path.basename(os.path.abspath(report_dir))
    best, best_tag = None, ""
    for sub in glob.glob(os.path.join(d, "*")):
        fs = sorted(glob.glob(os.path.join(sub, "cann_performance_eval_*.json")))
        if not fs:
            continue
        try:
            doc = json.load(open(fs[-1], encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            continue
        cs = []

        def _w(o):
            if isinstance(o, dict):
                if "case_id" in o:
                    cs.append(o)
                for v in o.values():
                    _w(v)
            elif isinstance(o, list):
                for v in o:
                    _w(v)
        _w(doc)
        if len(cs) < 2:
            continue                                           # 单 case 的探针轮次不参与
        sc = sum(c.get("perf_score") or 0 for c in cs) / len(cs)
        if best is None or sc > best:
            best, best_tag = sc, os.path.basename(sub)
    if not best_tag or best_tag == cur:
        return ""
    fs = sorted(glob.glob(os.path.join(d, cur, "cann_performance_eval_*.json")))
    if not fs:
        return ""
    doc = json.load(open(fs[-1], encoding="utf-8"))
    cs = []

    def _w2(o):
        if isinstance(o, dict):
            if "case_id" in o:
                cs.append(o)
            for v in o.values():
                _w2(v)
        elif isinstance(o, list):
            for v in o:
                _w2(v)
    _w2(doc)
    if len(cs) < 2:
        return ""
    mine = sum(c.get("perf_score") or 0 for c in cs) / len(cs)
    if best <= mine + 1e-9:
        return ""
    return (f"你给的是 `{cur}` (均 perf_score {mine:.4f}), 而同目录的 `{best_tag}` 更好 "
            f"({best:.4f}, 高 {(best / max(mine, 1e-9) - 1) * 100:.1f}%) —— "
            f"**「最近的一轮」不等于「最好的一轮」**, 带 _off / -ungated / bis- 之类后缀的"
            f"多半是 A/B 对照组。确认你要的就是这一轮再往下读")


def _inflation_records(report_dir: str) -> str:
    """本仓有没有记录过该算子的**工作量膨胀**实验结果。返回一句可读的出处, 没有则空串。

    找三处 (都是本仓实际记录的地方):
      - `cases/<op>/problem.yaml` 里出现「膨胀」/「工作量翻倍」;
      - 内核源码注释里出现同样的词 (cummin 的 ROW 路径就写在那里);
      - `cases/<op>/runs/` 下名字像膨胀轮次的目录 (wm1 / wm2 / inflate*)。
    只报出处, 不替人下结论 —— 记录可能只覆盖部分 case。
    """
    import glob
    import re                                          # noqa: PLC0415
    d = os.path.abspath(report_dir)
    for _ in range(4):                       # runs/<tag> -> runs -> cases/<op>
        if os.path.basename(d) == "runs":
            d = os.path.dirname(d)
            break
        d = os.path.dirname(d)
    if not os.path.isdir(d):
        return ""
    hits = []
    pat = re.compile(r"膨胀|工作量翻倍|向量工作量")
    for f in [os.path.join(d, "problem.yaml")] + glob.glob(
            os.path.join(d, "workspace", "csrc", "ops", "*", "op_kernel", "*.h")):
        try:
            if pat.search(open(f, encoding="utf-8", errors="replace").read()):
                hits.append(os.path.relpath(f))
        except OSError:
            pass
    wm = [os.path.basename(x) for x in glob.glob(os.path.join(d, "runs", "*"))
          if re.fullmatch(r"wm\d*|inflate\w*", os.path.basename(x))]
    if wm:
        hits.append("runs/" + ",".join(sorted(wm)))
    return "; ".join(hits[:4])


def bench_report_floors(report_dir: str) -> list:
    """读 cann-bench 的 cann_performance_eval_*.json, 每 case 出 (耗时, 下界, 倍数)。"""
    import glob as _glob
    fs = sorted(_glob.glob(os.path.join(report_dir, "cann_performance_eval_*.json")))
    if not fs:
        raise SystemExit(f"没有 cann_performance_eval_*.json: {report_dir}")
    doc = json.load(open(fs[-1], encoding="utf-8"))
    #  ★case 集完整性守卫 (cummin 2026-09-20 实撞).
    #    一个 case 崩设备 (507035) 会**级联**打挂后续 case, 性能阶段只跑"精度通过"的子集,
    #    于是 20 个 case 剩 12 个, 报表打出「12/12 通过」—— **部分失败伪装成满分**,
    #    而总耗时、均值 speedup 全部基于缩小的样本, 与上一轮不可比。
    #    精度阶段的报告里 case 数是全的, 拿它当分母核对。
    cfs = sorted(_glob.glob(os.path.join(report_dir, "cann_correctness_eval_*.json")))
    if cfs:
        cdoc = json.load(open(cfs[-1], encoding="utf-8"))
        for cop, pop in zip(cdoc.get("operators", []), doc.get("operators", [])):
            n_all, n_perf = cop.get("total_cases") or 0, len(pop.get("cases") or [])
            if n_all and n_perf < n_all:
                bad = [str(c.get("case_id", "")).split("_")[-1]
                       for c in (cop.get("cases") or []) if c.get("status") != "success"]
                print(f"★★**样本不完整**: {pop.get('operator')} 精度阶段 {n_all} 个 case, "
                      f"性能阶段只有 {n_perf} 个 —— 未通过的: {', '.join(bad) or '?'}")
                print("   性能阶段只跑精度通过的 case, 所以下表的总耗时与均值**与完整样本不可比**。")
                print("   先修掉崩的 case (507035 会级联打挂后续 case), 不要在缩小的样本上比较收益。\n")
    rows = []
    for op in doc.get("operators", []):
        for c in op.get("cases", []):
            t_hw = c.get("t_hw_us") or 0.0
            outs = (c.get("accuracy") or {}).get("output_results") or []
            counts = [o.get("total_count") or 0 for o in outs]
            nmax = max(counts) if counts else 0
            approx = False
            obytes = 0
            for o in outs:
                n = o.get("total_count") or 0
                if n == 0 and nmax:
                    n, approx = nmax, True
                obytes += n * _DT_BYTES.get(o.get("dtype", ""), 0)
            ibytes = t_hw * 1.6e6
            traf = ibytes + obytes
            us = c.get("elapsed_us") or 0.0
            floor_us = traf / 1.6e12 * 1e6
            rows.append(dict(op=op.get("operator"), case=str(c.get("case_id", "")).split("_")[-1],
                             us=us, sp=c.get("speedup"), bytes=traf, approx=approx,
                             floor_us=floor_us,
                             mult=(us / floor_us) if floor_us > 0 else None,
                             gbps=(traf / (us * 1e-6) / 1e9) if us > 0 else None))
    return rows


def report_bench(rows: list, report_dir: str = "") -> int:
    print("=== 逐 case 离访存下界几倍 (分母是官方口径 1.6 TB/s, 不是本机实测可达带宽) ===")
    print(f"{'算子':<20}{'case':>5}{'耗时us':>10}{'sp':>7}{'下界us':>9}{'倍数':>7}{'GB/s':>7}")
    tot = rec = 0.0
    for r in sorted(rows, key=lambda r: -(r["us"] or 0)):
        tot += r["us"]
        rec += max(0.0, r["us"] - r["floor_us"])
        m = f"{r['mult']:.1f}x" if r["mult"] else "  -"
        print(f"{r['op'] or '?':<20}{r['case']:>5}{r['us']:>10.2f}"
              f"{(r['sp'] or 0):>7.3f}{r['floor_us']:>9.2f}{m:>7}"
              f"{(r['gbps'] or 0):>7.0f}" + ("  ~" if r["approx"] else ""))
    print(f"\n总耗时 {tot:.0f} us, 其中 {rec:.0f} us ({rec/tot:.0%}) 在下界之上")
    print("\n=== 按「可回收时间」排序 —— 下一步该去哪 ===")
    print("  ★★**下界只是对偶的上界, 不是可达目标**。倍数小**不等于**贴住了硬件:")
    print("    cummin 2026-09-20 实测, case 5 在 1.64x / 979 GB/s 处看着像撞了带宽墙,")
    print("    把向量工作量翻倍 (访存量一字节不变) 时间涨了 48% —— 它是**计算受限**,")
    print("    979 GB/s 只是当时的计算速率允许的搬运量。同批 case 4/6 倍数 1.67/1.72,")
    print("    翻倍后 +79%/+77%。**据倍数小就判「没空间」会漏掉最大的一类收益。**")
    print("  ★分辨手段 (零风险, 不破坏正确性): **幂等工作量膨胀**。")
    print("    prefix-Min/Max 扫描幂等 (prefix 非增 -> min(prefix[i],prefix[i-d])==prefix[i]),")
    print("    重复 N 遍则输出逐位不变、访存量不变, 只有向量趟数线性增加。")
    print("    时间随 N 线性涨 = 计算受限; 几乎不动 = 访存受限。一次构建即可判。")
    print("    同类可幂等的运算: min/max/and/or/按位运算、幂等投影。加法类不可 —— 改用")
    print("    「插入可被优化掉之外的冗余读」或对比两个只差趟数的等价实现。")
    print("  ★★**这张表排的是墙上时间, 不是榜单分**。cann-bench 的分是逐 case 的")
    print("    perf_score 求平均, 单个 case 最多值 100*0.5/n 分 (n=20 时 2.5 分),")
    print("    而 perf_score = (T_base-T_hw)/((T_cand-T_hw)+(T_base-T_hw)) —— T_base 小的")
    print("    case 再快也换不到多少分。2026-09-21 Softmax case 13 实撞: 6776 -> 1021.87 us")
    print("    (-85%), 榜单分只从 60.23 涨到 60.25, 因为它的 T_base 只有 20.3 us。")
    print("    要按**分**排序看 `core/opreport` 的「可回收」列 (= 100*0.5/n*(1-perf_score))。")
    print("  ★排序键是 (耗时 - 下界) 的**绝对值**, 不是倍数。倍数大而耗时小的 case")
    print("    看着吓人但回收不了多少总时间; 这是 cummin 那轮把力气放错地方的原因之一。")
    #  ★先查本仓**有没有已经做过**这个实验 —— 不查就会让人重做一个答案已经在的实验。
    #    2026-09-22 实撞: 上面那段通用提示引用的正是 cummin case 4/5/6 的膨胀结果
    #    (+48% / +79% / +77%), 而下面的逐 case 建议照样对这三个 case 说"先做膨胀实验"。
    #    照表执行就会白烧一轮构建。答案记在 `cases/<op>/problem.yaml` 与内核源码注释里
    #    (cummin 的 ROW 路径写着"wm2 实测 向量工作量翻倍 -> 时间 +75~79%, 按趟数计价"),
    #    还有 `runs/wm1` / `runs/wm2` 两个轮次目录。
    #  ★这里**不吞异常**: 判据自己崩了要看得见。上一版忘了 import re, 异常被上层捕获,
    #    表现成"逐 case 建议整段消失"而退出码仍是 0 —— 正是本仓那条
    #    「失败路径上的值不得当真值用」的现场。
    try:
        done = _inflation_records(report_dir) if report_dir else ""
        better = _better_run_exists(report_dir) if report_dir else ""
        if better:
            print(f"  ★**基线选择存疑**: {better}")
    except Exception as _e:                            # noqa: BLE001
        done = ""
        print(f"  ★「已做过膨胀实验」的交叉引用**算不出**({type(_e).__name__}: {_e}) —— "
              f"下面的建议里那一条按「没查过」读")
    if done:
        print(f"  ★**本仓已记录过该算子的工作量膨胀结果**, 先读再决定要不要重做: {done}")
    for r in sorted(rows, key=lambda r: -(r["us"] - r["floor_us"]))[:8]:
        gain = r["us"] - r["floor_us"]
        band = ("访存下界附近, 访存侧无空间" if r["mult"] and r["mult"] < 1.3 else
                ("★倍数判不出来, 但**本仓已做过膨胀实验**(见上), 别重做" if done else
                 "★**倍数判不出来**: 访存受限还是计算受限都可能 -> 先做工作量膨胀实验")
                if r["mult"] and r["mult"] < 2 else
                "★计算层或占用率: 数每元素的向量指令趟数, 不是调 tiling" if r["mult"] and r["mult"] < 5 else
                "★★结构层: 占用率塌陷 -> 调 tiling 无解, 必须换分解方式")
        print(f"  case {r['case']:>3}  可回收 {gain:>8.0f} us  ({r['mult']:.1f}x)  {band}")
    worst = max((r for r in rows if r["mult"]), key=lambda r: r["mult"], default=None)
    if worst and worst["mult"] >= 2:
        print()
        for line in verdict_lines(worst["floor_us"] / 1e3, worst["us"] / 1e3,
                                  kind="capability", indent="  "):
            print(line)
    return 0
def slack_table(r: dict, spec: dict, measured_ms=None) -> list:
    """**余量表**: 每种资源还剩多少自由度, 以及用得上它的条件。

    ★2026-09-18 立, 起因是 dubin 的一句话: "你要展示优化自由度在哪? 优化空间在哪?
      老是预估+否定干啥"。当时仓里 47 个模块全在回答两类问题 ——
      「这条流花了多少」(成本表) 与「这个方案违不违规」(合规判据), **没有一个回答
      「还有哪些自由度没被占用」**。于是候选生成退化成"猜一个 -> 判据说不行"的拒绝采样,
      而可行域从来没被看见过。当天连毙三个候选, 每个都是提出之后才被拦下。

    ★代价最直接的证据: 标定 DRAM 之后才第一次露出「达成带宽只有平台的 47%」
      = **2.12 倍的带宽效率余量**, 而在此之前所有候选没有一个指向它 ——
      因为没有任何工具把它当成一个自由度报出来过。

    ★纪律: 取不到的**说取不到**, 不填默认值。余量表最怕的是把"没测"显示成"没余量"。
    """
    out, sp = [], spec
    cube_ms = float(r.get("cube_floor_ms") or 0)
    if measured_ms and cube_ms > 0:
        out.append(("cube 时间", f"{measured_ms:.2f} ms 可用", f"{cube_ms:.2f} ms 强制计算",
                    measured_ms / cube_ms, "cube 空着的那部分 —— 把 vector 的活挪过去"))
    dports = [p_ for p_ in (r.get("ports") or []) if p_["port"] == "dram"]
    if dports and dports[0].get("compulsory") and dports[0].get("actual"):
        d = dports[0]
        out.append(("DRAM 字节", f"强制 {d['compulsory']/1e9:.2f} GB",
                    f"实际 {d['actual']/1e9:.2f} GB", d["actual"] / d["compulsory"],
                    "改数据流: 多余往返 / 复用不足 / 散射累加"))
        ach, plat = d.get("bw_achieved"), d.get("bw")
        #  ★参照量必须是**同形态的可达带宽**, 不是平台峰值 (2026-09-18 实撞, 差 2.4 倍)。
        #    平台 1428.9 是 128KB 大块顺序纯读测出来的; 而本算子是 1KB 随机块 + 读写 2.2:1 混合。
        #    同形态合成负载的天花板实测只有 845 GB/s (runs/loadmix2_20260918):
        #      1429 -(粒度 1KB+随机 -15%)-> 1219 -(读写混合 -27%)-> 845 -(算子)-> 675
        #    据平台算余量得 2.0x, 而真实可争取的是 845/675 = **1.25x** —— 前者把"这条负载
        #    本来就到不了的部分"也算成了空间, 会把人推向一个不存在的目标。
        same = float(sp.get("dram_sameshape_GBps") or 0) * 1e9
        ref_bw, ref_name = (same, "同形态可达") if same else (plat, "平台")
        if ach and ref_bw and abs(ach - ref_bw) / ref_bw > 0.02:
            note = ("★**同形态天花板**下的余量 —— 这部分才是真能争取的"
                    if same else
                    "★参照量是**平台峰值**, 对非顺序/混合负载**偏乐观** —— "
                    "测同形态可达带宽 (core/probe 调到同粒度同读写比) 再看这一行")
            out.append(("DRAM 带宽效率", f"{ref_name} {ref_bw/1e9:.0f} GB/s",
                        f"达成 {ach/1e9:.0f} GB/s", ref_bw / ach, note))
            if same and plat:
                out.append(("  └ 其中形态本身的损失", f"平台 {plat/1e9:.0f} GB/s",
                            f"同形态上限 {same/1e9:.0f} GB/s", None,
                            "粒度/随机/读写混合造成, **不是可优化的空间** (除非改数据布局)"))
        elif not sp.get("dram_achieved_GBps"):
            out.append(("DRAM 带宽效率", "平台已知", "**达成带宽没测**", None,
                        "测法: core/probe/bytes_report.py (L2 miss x 颗粒) / 时长"))
    #  ★pipe 占用是余量表**此前整类缺失**的维度 (2026-09-18 补)。
    #    没有它时, "cube 时间 13.5x" 看起来是最大的机会 —— 而实测 ref12104 的
    #    aiv_vec (真正的算术) 只有 3.73ms, 就算全给 cube 也就这么多, 且 910B2 的
    #    cv_direct_path=False, 搬过去要经 GM 往返。真正的墙是 aiv_mte2 与 **aiv_scalar**
    #    (各 55.1% = 12.23ms), 后者是**地址计算**, 既不是算术也不是带宽 ——
    #    没有这一行, 这个自由度在表上根本不存在。
    pipes = sp.get("_measured_pipes") or {}
    if pipes:
        wall = float(pipes.get("wall_ms") or 0)
        busy = {k: float(v) for k, v in pipes.items() if k != "wall_ms"}
        if wall > 0 and busy:
            longest = max(busy, key=lambda k: busy[k])
            for k in sorted(busy, key=lambda x: -busy[x]):
                mark = "  ★**最长的那条 = 墙**" if k == longest else ""
                out.append((f"pipe {k}", f"{wall:.2f} ms 全程", f"忙 {busy[k]:.2f} ms",
                            wall / busy[k] if busy[k] else None,
                            ("缩短它才动端到端" if abs(busy[k] - busy[longest]) < 1e-9
                             else "不是墙, 缩短它端到端不变")
                            + ("  ★**最长的那条 = 墙**" if abs(busy[k] - busy[longest]) < 1e-9 else "")))
    else:
        out.append(("pipe 占用", "—", "**没测**", None,
                    "测法: prof.py --metric PipeUtilization, 看哪条 pipe 是墙 (占比之和>1 说明有重叠)"))

    hit = sp.get("_measured_l2_read_hit")
    if hit is not None:
        out.append(("L2 读命中", "100%", f"{hit*100:.1f}%", 1 / hit if hit else None,
                    "提高复用距离: 分块 / 遍历序 / 预处理重排"))
    else:
        out.append(("L2 读命中", "—", "**没测**", None,
                    "测法: L2Cache 档 read_cache_hit / (hit+miss_allocate)"))
    return out


def print_slack(r: dict, spec: dict, measured_ms=None) -> None:
    rows = slack_table(r, spec, measured_ms)
    #  ★两类行的倍数**含义相反**, 混排会误导 (2026-09-18 自己写出来的坑):
    #    「可回收的浪费」倍数越大 = 机会越大 (用了 20 倍于强制的字节);
    #    「pipe 占用」倍数越小 = 它越是墙 (全程 22.2ms 而它忙 12.2ms)。
    #    按同一方向排序会把最该忽略的 aic_mte1 (9x "余量") 排到最前面。
    pipe = [x for x in rows if str(x[0]).startswith("pipe ")]
    waste = [x for x in rows if not str(x[0]).startswith("pipe ")]
    print("\n=== 余量表 (还有哪些自由度没被占用) ===")
    print("  ★这张表回答的是'空间在哪', 与超额表(成本在哪)、判据(违不违规)是三件不同的事。")
    print("\n【可回收的浪费】**倍数越大机会越大**")
    for name, cap, used, ratio, how in sorted([x for x in waste if x[3]], key=lambda x: -x[3]):
        print(f"  {name:16s} {cap:22s} {used:24s} **{ratio:5.2f}x**  {how}")
    for name, cap, used, ratio, how in [x for x in waste if not x[3]]:
        print(f"  {name:16s} {cap:22s} {used:24s} {'—':>7}  {how}")
    if pipe:
        print("\n【pipe 占用】**倍数越小它越是墙** —— 只有墙缩短了端到端才动")
        for name, cap, used, ratio, how in sorted([x for x in pipe if x[3]], key=lambda x: x[3]):
            print(f"  {name:16s} {cap:22s} {used:24s} **{ratio:5.2f}x**  {how}")
    print("  ★**没测的不等于没余量** —— 上面标'没测'的行是**未知**, 不是 0。")
    try:
        from core import trust
        for ln in trust.verdict_banner("hwfloor", refusing=False, prefix="  "):
            print(ln)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", help="算子的 spec json")
    ap.add_argument("--measured", type=float, default=None, help="实测毫秒, 给了才报倍数")
    ap.add_argument("--verify", metavar="MSPROF_DIR", default=None,
                    help="拿 msprof (--metric Memory) 实测的 DRAM 字节验证 spec 的流量预测 —— "
                         "让 spec **可证伪**。预测对不上就说明超额表的排序不可信")
    ap.add_argument("--cores", type=int, default=24, help="--verify 用: 每核信号乘几倍 (默认 cube 24)")
    ap.add_argument("--bench-report", metavar="REPORTS_DIR", default=None,
                    help="直接读 cann-bench 评测报告目录, 出逐 case 的下界倍数与可回收时间 "
                         "(不需要手写 spec —— spec 那条路门槛高到每次都被跳过)")
    ap.add_argument("--src", metavar="KERNEL_SRC_DIR", default=None,
                    help="**执行单元占用**: 数这份源码用了哪个单元, 并排给出两个算力天花板。"
                         "配 --measured-us 与 --macs 可直接判'是不是卡在算力上'")
    ap.add_argument("--measured-us", type=float, default=None, help="--src 用: 实测微秒")
    ap.add_argument("--macs", type=float, default=None, help="--src 用: 该次测量的总乘加数")
    ap.add_argument("--msprof", metavar="OPPROF_DIR", default=None,
                    help="--src 用: msprof 产物目录, 读逐流水占用率。**吞吐效率与流水占用"
                         "是两件事**, 缺它不给换单元的结论")
    ap.add_argument("--template", action="store_true", help="打印 spec 骨架")
    ap.add_argument("--json", action="store_true", help="机读输出 (给闸门用)")
    ap.add_argument("--slack", action="store_true",
                    help="**余量表**: 还有哪些自由度没被占用 (与超额表互补: 那张是成本, 这张是空间)")
    ap.add_argument("--refs", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.refs:
        print_refs("hwfloor", REFS)
        return 0
    if a.template:
        print(json.dumps(TEMPLATE, indent=2, ensure_ascii=False))
        return 0
    if a.src:
        u = unit_usage(a.src)
        occ = msprof_occupancy(a.msprof) if a.msprof else None
        print(f"=== 执行单元占用 ({u['files']} 个源文件) ===")
        for ln in unit_verdict(u, a.measured_us, a.macs, occ):
            print("  " + ln)
        return 0
    if a.bench_report:
        rows = bench_report_floors(a.bench_report)
        if a.json:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
            return 0
        return report_bench(rows, a.bench_report)
    if not a.spec:
        ap.error("要给 spec 文件 (先看 --template)")
    _spec = json.load(open(a.spec, encoding="utf-8"))
    r = analyze(_spec, a.measured)
    if a.verify:
        v = verify_against_msprof(r, a.verify, a.cores)
        print("=== spec 流量预测 vs msprof 实测 ===")
        if not v["ok"]:
            print(f"  ★**没验成** (不是验过没问题): {v['err']}")
            return 2
        rk, wk = v.get("read_KB_per_core"), v.get("write_KB_per_core")
        det = (f"  (读 {rk:.0f} + 写 {wk:.0f} KB/核 x {v['cores']} 核)" if rk is not None else
               f"  (来源: {v.get('source')})")
        print(f"  msprof 实测落 DRAM: {v['measured_bytes']/1e9:.2f} GB{det}")
        print(f"  spec 预测落 DRAM  : {v['predicted_bytes']/1e9:.2f} GB")
        rt = v["ratio"]
        print(f"  预测/实测 = **{rt:.2f}**")
        if rt is None:
            pass
        elif 0.85 <= rt <= 1.15:
            print("  ★两者吻合 -> **超额表的排序可信**, 可以据它定优先级。")
        elif rt < 0.85:
            print(f"  ★**spec 漏了流量** (只覆盖实测的 {rt:.0%})。逐函数过一遍所有 GlobalTensor")
            print("    与 workspace, 尤其是**中间量** —— 算子接口上的输入输出最容易列全, 也最不容易是大头。")
            print("    ★漏了之后超额表会把人**指向错误的地方** (本模块 v1 的第一行就指错了)。")
        else:
            print(f"  ★**spec 多算了流量** (是实测的 {rt:.1f} 倍)。查三处: (a) 某条流其实由 L2 供给,")
            print("    给它补 footprint; (b) 重复计了同一笔; (c) 该流实际次数比模型少 (有提前退出?)。")
        return 0
    if a.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return 0
    if a.slack:
        return print_slack(r, _spec, a.measured) or 0
    return report(r)


if __name__ == "__main__":
    sys.exit(main())
