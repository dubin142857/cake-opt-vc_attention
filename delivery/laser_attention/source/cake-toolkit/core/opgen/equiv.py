#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opgen.equiv —— 枚举一个算子的**数学等价形式**, 逐条过 IR 判据后排序。

为什么必须有这个模块 (必要性门禁, 2026-09-22 立)
------------------------------------------------
本模块的必要性不是推理出来的, 是一次代价明确的实撞:

  sparse_flash_attention 本轮做的七个改动里, 六个是**在既定形式内调参**
  (kBlk / kStep / 参数外提 / 缓冲深度 / VF 改写 / 三段全融), 其中三个被实测判负、
  两个零效果。唯一的大赢是第七个 —— 把 M 维从 s1 改成"共用同一份 K 的 G 个头",
  110266 us -> 60838 us (-45%)。那不是调参, 是换了一个**数学上等价的写法**。

  而那一个是我**用手找出来的**, 在四轮无效调参之后。

infra 当时的状态解释了为什么: `opgen.ir` 能对**一个**被提出的调度算容量/流量/双侧界,
`opgen.graph` 能对数学图求规范形与 id —— 但没有任何东西**生成候选**。工具只会评价
你交给它的东西; 没有枚举, 剩下的动作就只有拧旋钮。

所以本模块只做一件生成式 IR 也做、而本仓一直缺的事: **把等价类摊开**。

它不做等价饱和 (e-graph)
------------------------
PLAN 的反熵条款: 等价类由**有限条具名代数恒等式**生成, 每条是一个独立的**改写轴**,
枚举 = 各轴取值的笛卡尔积 + 前提剪枝 + 规范形去重。理由与 PLAN 第 6 节一致:
饱和搜索的产物无法归因 (说不清"为什么是这一支"), 而本仓要的恰恰是归因。
轴是受控词表, 加一条要走门禁, 这样枚举表的规模是**可预期**的, 不会随迭代熵增。

诚实性条款 (与 qty 的回落纪律同源)
----------------------------------
排序按 IR 的下界。但一个新形式往往带来 IR **还没有标定**的代价项 (典型: gather 的
逐行搬运单价)。这类项一律进 `unquantified`, 并且:
  - 表里单列一栏, 不允许只出现在脚注;
  - **排第一但带未标定项时, 本模块拒绝宣布赢家**, 改为输出"要先做哪一次测量"。
否则枚举器就是个幻想发生器: 它总能排出一个纸面上更快的形式。
"""
from __future__ import annotations

import csv
import itertools
import json
import os

from .._refs import Ref
from .ir import Schedule, GM, L1, L0A, L0B, L0C, UB, CUBE, VEC
from .qty import Qty

#  ================================================================= 改写轴
#  每条轴 = 一条代数恒等式给出的若干等价取值。identity 是**为什么这些取值等价**;
#  precond 是**这个取值在这个形状下成不成立**。两者分开: 等价性与形状无关,
#  可行性与形状有关 —— 混在一起就没法说清"这条路被排除是因为不等价还是因为装不下"。

MATH = "G_math"        # 改写数学语义图 (什么被算出来)
MOVE = "G_move"        # 改写硬件流动图 (算在哪、经过谁) —— 数学语义不变


#  一条"前提"其实有四种截然不同的东西, 混在一起会把等价类提前剪掉:
#    equivalence  数学上这个取值成不成立           -> 不成立则剪 (它根本不是等价形式)
#    lowerable    当前降级模板能不能把它落成代码   -> 不能则剪, 但**要报出来**
#    capability   这颗芯片有没有需要的机制         -> 由 Schedule.requires + caps 判, 不在这里
#    cost         能做, 但明显不划算               -> **不剪**, 只标注, 让判据去排
#  ★2026-09-22 采纳外部评审: 首版 _pc_support 把"按 s1 分组时 gather 收益归零"写成
#    返回 False, 那是**代价判断伪装成可行性前提**。本模块存在的理由正是"不要因为今天的
#    实现习惯而漏掉明天可能的大赢结构", 把代价写进前提就自己违反了这一条。
EQUIVALENCE, LOWERABLE, COST = "equivalence", "lowerable", "cost"
PRUNING = (EQUIVALENCE, LOWERABLE)

#  等价的**种类**。区分这三档是给正确性用的, 不是修辞:
#    real_exact   实数域代数恒等, 但浮点下不逐位相同 (分块归约 / 重结合 / 在线 softmax)
#    numeric      浮点下误差在容差内 (需要一次精度验证才能这么说)
#    bitwise      加法次序等完全一致, 逐位相同
#  ★real_exact 的形式**不能**因为"数学上等价"就当作可以安全替换基线: 它带一个
#    **精度义务**, 必须由 simfunc 对着 golden 验过才准进构建。G13 就是这条闸。
REAL_EXACT, NUMERIC, BITWISE = "real_exact", "numeric", "bitwise"


class Axis:
    __slots__ = ("name", "layer", "identity", "options", "conds",
                 "equiv_kind", "precision_obligation")

    def __init__(self, name, layer, identity, options, conds=(),
                 equiv_kind=BITWISE, precision_obligation=""):
        self.name, self.layer = name, layer
        self.identity = identity                  # 代数恒等式 (为什么等价)
        self.options = dict(options)              # 取值 -> 一句话说明
        #  conds: ((种类, fn(dims, form) -> (ok, 理由)), ...)
        self.conds = tuple(conds)
        self.equiv_kind = equiv_kind
        #  非 bitwise 的轴必须写清"要验什么才敢用", 否则构造时就拒绝。
        if equiv_kind != BITWISE and not precision_obligation:
            raise ValueError(f"轴 {name} 的等价种类是 {equiv_kind}, 必须写 precision_obligation")
        self.precision_obligation = precision_obligation


def _eq_gather_needs_indices(d, form):
    """等价条件: 收缩到支撑集要求支撑集是**已知**的 (有索引张量)。"""
    if form["support"] != "gather":
        return (True, "")
    return (d.get("topK", 0) > 0, "没有索引张量, 支撑集未知 -> 收缩不成立")


def _cost_gather_shared_index(d, form):
    """代价标注 (**不剪枝**): 分块内各行索引行不同时, 一个分块要 mT 次 gather。

    等价性照样成立, 只是右操作数不再共享 -> 搬运按行数放大。让降级去如实算这个代价,
    再由判据排序决定 —— 首版在这里 return False, 把 case 1/3 的整条 gather 分支
    直接抹掉了, 而那两个 case 正是我们比厂商基线慢 10~17 倍的地方。
    """
    if form["support"] != "gather":
        return (True, "")
    if form["grouping"] != "head":
        return (False, "分块内各行索引行不同 -> gather 次数按行数放大 (代价, 非不可行)")
    if d["topK"] >= d["S2"]:
        return (False, f"topK={d['topK']} >= S2={d['S2']}, 没有可省的列 (代价, 非不可行)")
    return (True, "")


AXES = (
    Axis("grouping", MATH,
         "若 B 与批轴 b 无关, 则 matmul(A[b,m,k], B[k,n]) = matmul(A[(b*m),k], B) —— "
         "任意一组**共用同一个右操作数**的独立行都可以并进 M 维",
         {"s1": "M 维由同一个查询头的连续 s1 行组成",
          "head": "M 维由同一个 (b,n2,s1) 的 G 个查询头组成 (它们共用 K, 也共用索引行)",
          "s1_head": "M 维同时并 s1 与头"},
         equiv_kind=BITWISE),
    Axis("support", MATH,
         "若 A[m,k]=0 对一切 k∉I(m), 则 Σ_k A[m,k]·B[k,n] = Σ_{k∈I(m)} A[m,k]·B[k,n] —— "
         "收缩维可以缩到左操作数的**支撑集**上, 代价是把右操作数按 I 取行",
         {"dense": "在整个 S2 上收缩, 非支撑集位置靠掩码置零",
          "gather": "先按索引取出 topK 行, 收缩维缩到 topK"},
         conds=((EQUIVALENCE, _eq_gather_needs_indices),
                (COST, _cost_gather_shared_index)),
         equiv_kind=REAL_EXACT,
         precision_obligation=(
             "稠密路径把非支撑集位置置 -inf 后过 exp 得 0, 再参与行求和; gather 路径"
             "根本不产生那些项。实数域上二者相同, 浮点下**求和的项数与次序都变了**。"
             "另外稠密路径的行最大值取自含 -inf 的整行, gather 路径取自 topK 行 —— "
             "两者在'整行全被掩掉'时行为不同 (稠密得 0/0, gather 得空集)。"
             "验法: core/opgen/simfunc 的 sfa_rowwise vs gather 版对着 golden 比,"
             "必须含'整行被掩掉'的负例 (已建: 因果掩码 + topK 全在因果界之外)")),
    Axis("softmax", MATH,
         "max 与 sum 可分块求, 块间用修正因子合并 (在线 softmax); 实数域恒等",
         {"two_pass": "先扫一遍求行最大与行和, 再扫一遍归一化",
          "online": "单遍流式, 每块更新修正因子"},
         equiv_kind=REAL_EXACT,
         precision_obligation=(
             "在线归约改变了加法的**结合次序**, 且引入逐块的 exp 修正因子相乘。"
             "fp32 累加下相对误差随块数增长。验法: simfunc 的 "
             "sfa_chunked_online(mode='exact') vs sfa_rowwise, 容差自校准于 golden.py")),
    Axis("residency", MOVE,
         "中间量存哪一级不改变数学语义 (G_move 层的取值)",
         {"gm": "分数与概率经 GM 交接 (三次核发射)",
          "on_chip": "L0C 直写对侧 UB, 概率写回 L1 (需 cv_direct_ub 等机制)"},
         equiv_kind=BITWISE),
)
#  ★residency=on_chip 与 softmax=two_pass 的冲突**不再写成前提**: 两遍法要整块分数
#    驻留 UB, 那是一条**容量**约束, 由 lower_sfa 如实开出 mT*Keff*4 的缓冲, 让 IR 的
#    容量判据去拒。用判据拒 > 用决条拒: 前者换个形状会自动变, 后者是写死的。


def enumerate_forms(dims):
    """各轴取值的笛卡尔积 + **只按 equivalence/lowerable** 剪枝。

    返回 (可行形式, 被剪枝的 (形式, 种类, 理由), 代价标注 {form_id: [理由]})。
    ★cost 类条件不剪枝 —— 它们只标注, 由 IR 判据去排。
    """
    keys = [a.name for a in AXES]
    ok, cut, noted = [], [], {}
    for combo in itertools.product(*[list(a.options) for a in AXES]):
        form = dict(zip(keys, combo))
        why = None
        notes = []
        for a in AXES:
            for kind, fn in a.conds:
                good, reason = fn(dims, form)
                if good:
                    continue
                if kind in PRUNING:
                    why = (kind, f"{a.name}={form[a.name]}: {reason}")
                    break
                notes.append(f"{a.name}={form[a.name]}: {reason}")
            if why:
                break
        if why:
            cut.append((form, why[0], why[1]))
        else:
            ok.append(form)
            if notes:
                noted[form_id(form)] = notes
    return ok, cut, noted


def form_id(form):
    return "+".join(f"{k}={form[k]}" for k in sorted(form))


#  ================================================================= 算子谱 (spec)
#  ★2026-09-22 按 dubin 的架构定调重整: **本项目的主场景是已有算子的单 case 极致性能
#    优化; 打榜/泛化是增强模块, 不许绑死。** 重整前我把 cann-bench 的 case 表路径、
#    我方插件的形状路由、以及站上报告里的厂商基线, 全写在了 core 里, 而 `opgen gate`
#    又刚进了 CRITERIA_SELFTEST —— 于是**没有 cann-bench 数据的环境里 core 的门禁全红**。
#    实测: `CANN_BENCH_TASKS=/nonexistent` 下 EQ1~EQ5 五条负例全抛 FileNotFoundError。
#    现在 core 只留算子无关的机制 + 一份**合成算子谱**用于自证; 具体 benchmark 的
#    接入放插件 (插件 import core, core 不 import 插件), 缺席则如实报 NOT_APPLICABLE。
#
#  一份 spec 是这样一个 dict:
#      cases_fn   () -> {case_id: dims}            必需
#      lower      (dims, form) -> Lowered          必需
#      built      (dims) -> form | None            可选; 没有则 EQ2/G12 对它 NOT_APPLICABLE
#      baseline_fn() -> {case_id: 实测us}          可选; 没有则 EQ4 对它 NOT_APPLICABLE
#      note       一句话                           可选


class NotApplicable(Exception):
    """这条判据在当前环境下**不适用** (缺数据/缺依赖), 既不算过也不算不过。

    ★与"未过"严格区分。把不适用计成未过, 会逼着人为了让门禁变绿去伪造数据;
      把它计成通过, 就是本仓一直在消灭的"没查成伪装成查过且通过"。
    """


REGISTRY = {}


def register(name, spec):
    """接入一个算子谱。插件在 import 时调用它; core 自己只注册下面那份合成谱。"""
    missing = [k for k in ("cases_fn", "lower") if k not in spec]
    if missing:
        raise ValueError(f"spec {name} 缺必需字段 {missing}")
    REGISTRY[name] = spec
    return spec


def spec_of(name):
    if name not in REGISTRY:
        raise NotApplicable(
            f"算子谱 {name} 没有注册。已注册: {', '.join(sorted(REGISTRY)) or '(无)'}。"
            f"接入 benchmark 用 `--spec <模块>` 让它在 import 时调 equiv.register()")
    return REGISTRY[name]


def cases_of(name):
    return spec_of(name)["cases_fn"]()


def built_form(name, d):
    """现状形式。**可能为 None** —— 该形状走的是本表没有描述的回落路径, 如实报空。"""
    b = spec_of(name).get("built")
    return b(d) if callable(b) else b


#  ================================================================= 降级 (SFA)
#  一个算子一个 lower(): 把 (维度, 形式) 变成一个 ir.Schedule + 未标定项清单。
#  这一段是**算子专属**的 —— 上面的轴是通用的, 这里的映射不是, 两者刻意分开放。

#  一条向量指令的最大覆盖: repeat 上限 255, 每 repeat 256 B -> 65280 B。
#  ★首版把 instr 按**行数**算 (rows*passes*nchunk), 等于假设每行各发一条。真实核把
#    一个 [mT, nT] 的分块连续放在 UB 上, 一条指令覆盖整块 —— 首版因此把在线归约形式
#    的向量下界抬高了约 30 倍 (case 1: 1,441,792 条 vs 实际约 47,000 条), 于是
#    "中间量不落 GM 的在线形式"被系统性地排到稠密落盘形式后面。
#    是与**厂商基线实测**对照才发现的: 模型给的下界比一个真实跑着的实现还高 7.3 倍。
VEC_INSTR_MAX_BYTES = 255 * 256


def _instr(elem_f32, rows_in_tile, n_elem_pass, n_reduce_pass):
    """一个分块一轮要发几条向量指令。

    elementwise 类按**连续字节数**切 (一条指令覆盖 65280 B);
    行归约类 (ReduceMax/ReduceSum) 每 repeat 处理一行, 一条指令最多 255 行。
    """
    import math as _m
    per_elem = max(1, _m.ceil(elem_f32 * 4 / VEC_INSTR_MAX_BYTES))
    per_red = max(1, _m.ceil(rows_in_tile / 255))
    return per_elem * n_elem_pass + per_red * n_reduce_pass


def _ceil_div(a, b):
    """★2026-09-22 采纳外部评审: 首版 tiles/nchunk 用了 floor 除。公开集 20 个 case 的
    rows/mT 与 Keff/nT 恰好整除 (维度全是 2 的幂), 所以排名不受影响 —— 但这正是最坏的
    一类缺陷: 今天答案对, 换个形状静默少算一个尾块, 而尾块要继续进 cube task 数、
    在线归约的指令条数与 rescale 条数。隐藏集已知有 Dk=640 / topK=2048 这种非公开形状。
    """
    return -(-int(a) // int(b))


class Lowered:
    __slots__ = ("sch", "unquantified")

    def __init__(self, sch, unquantified=()):
        self.sch = sch
        #  (名字, 为什么没算, 做哪一次测量能把它变成 measured)
        self.unquantified = list(unquantified)


def lower_sparse_attention(d, form):
    #  ★本降级依赖 SFA 的一条**算子契约**: value 与 key 同源, Dk==Dv 时 value==key,
    #    Dk>Dv 时 value==key[..., :Dv] (见 tasks/level4/sparse_flash_attention/golden.py
    #    的"语义约束"段)。于是按索引取 K 的 topK 行**自带**了 V 的那些行, gather 只需
    #    一次而不是 K/V 各一次。
    #    2026-09-22 外部评审据"SFA 有独立的 K 和 V 输入"判定 gather 代价被低估 —— 对
    #    一般 attention 成立, 对本算子不成立。但那次质疑指出的根子是对的: 这条契约此前
    #    只存在于我脑子里。现在它是断言: 契约不成立就**直接报错**, 而不是静默少算一半
    #    搬运。换一个 V 独立的算子接进来时, 这里会立刻炸而不是悄悄给出乐观的表。
    assert d["Dv"] <= d["Dk"], (
        f"case {d['case']}: Dv={d['Dv']} > Dk={d['Dk']} —— V 不是 K 的前缀, "
        f"gather 分支必须另开一条 V 的搬运, 当前降级会低估搬运量")
    ESZ, FSZ = d["ESZ"], d["FSZ"]
    Dk, Dv, S2, topK = d["Dk"], d["Dv"], d["S2"], d["topK"]
    rows, groups, G = d["rows"], d["groups"], d["G"]
    gather = form["support"] == "gather"
    Keff = topK if gather else S2
    onchip = form["residency"] == "on_chip"

    mT = {"s1": min(d["S1"], 64), "head": min(G, 64),
          "s1_head": min(d["S1"] * G, 64)}[form["grouping"]]
    mT = max(mT, 1)
    #  ★不剪枝就必须如实算代价 (2026-09-22 采纳外部评审后补): gather 把右操作数换成了
    #    **按本行索引取出的** topK 行, 所以只有分块内各行共用同一行索引时, 这一块才还是
    #    一个共享右操作数的 GEMM。按 s1 / s1_head 分组时各行索引不同 -> 退化成 mT 个
    #    M=1 的独立小矩阵乘, 每个仍要占满 cube 的 16 行分形。
    #    把这条漏掉的后果是可测的: 漏掉时 gather 在全部 20 个 case 上胜出、可回收上界
    #    报到 51228 us; 补上之后 GQA 一组自己退回稠密。**代价条件从前提挪进模型, 而不是
    #    消失** —— 这正是拆分前提种类的义务所在。
    shared_index = (form["grouping"] == "head")
    if gather and not shared_index:
        mT = 1
    #  cube 的 M 分形是 16: 不足 16 行也要占满一个分形, 浪费按补齐后的行数计。
    mPad = ((mT + 15) // 16) * 16
    #  键分块 nT 同时受 **两级** 约束, 少算一级就会把正在跑的形式判成装不下:
    #    L0C: mT*nT*4 <= 256 KB
    #    L1 : q 常驻 + K 乒乓 <= 512 KB   ★首版只算了 L0C, 于是把现状形式判成不可行 ——
    #         是"现状形式必须可行"这条自检把它抓出来的 (2026-09-22)。
    l1_q_bytes = mPad * Dk * ESZ
    nT = min(Keff, 512)
    while nT > 16 and (mPad * nT * FSZ > 256 * 1024 or
                       l1_q_bytes + 2 * nT * Dk * ESZ > 512 * 1024):
        nT //= 2
    tiles = max(_ceil_div(rows, mT), 1)

    req = ("cv_direct_ub", "cv_dual_dst_split_m", "crosscore_flag",
           "mix_aic_to_aiv_ratio") if onchip else ()
    s = Schedule(f"case{d['case']} / {form_id(form)}", SOC,
                 "", requires=req)
    unq = []

    s.buf("l1_q", L1, mPad * Dk * ESZ)
    s.buf("l1_k", L1, nT * Dk * ESZ, double=True)
    s.buf("l0c", L0C, mPad * nT * FSZ)
    #  softmax 的 UB: 两遍法要整行驻留, 在线法只要一块
    #  两遍法要把整块分数留到第二遍: 落 GM 时逐行处理 (一行 Keff), 片上驻留时整块
    #  (mT 行 x Keff) 都得在 UB 上 —— 后者往往放不下, 让**容量判据**去拒, 不写成决条。
    ub_rows = mT if (form["softmax"] == "two_pass" and onchip) else 1
    s.buf("ub_row", UB, ub_rows * (Keff if form["softmax"] == "two_pass" else nT) * FSZ * 3)
    if not onchip:
        s.buf("gm_sc", GM, rows * Keff * FSZ)
        s.buf("gm_pr", GM, rows * Keff * ESZ)

    #  ---- 搬运
    s.move("q_read", GM, L1, rows * Dk * ESZ, 1)
    k_tensor = d["B"] * d["N2"] * S2 * Dk * ESZ
    #  索引: 按头分组时一个分块只读一行索引; 按 s1 分组时每行各读一次
    si_times_unique = groups * topK * FSZ
    si_req = si_times_unique if form["grouping"] == "head" else rows * topK * FSZ
    s.move("si_read", GM, UB, si_req, 1, unique=si_times_unique,
           reuse_window=rows * Keff * FSZ)

    if gather:
        gathered = groups * topK * Dk * ESZ
        #  逐行散读**直接进 L1**, 不经 GM 暂存。★首版写了 GM 暂存的一写两读, 平白
        #    给 case 1 加了 537 MB 请求量, 把 gather 形式压到稠密形式之后。真实核按
        #    kStep 分块 gather 进 L1 就地用, 从不落盘 —— 这不是优化, 是本来的写法。
        #  residency=gm 时 QK 与 PV 是两个核, 各自要 gather 一遍; 融合时只一遍。
        ntimes = 1 if onchip else 2
        s.move("k_gather", GM, L1, gathered, ntimes, unique=min(gathered, k_tensor),
               reuse_window=S2 * Dk * ESZ,
               note=f"逐行散读 K 的 topK 行直接进 L1 ({ntimes} 遍: "
                    + ("QK/PV 同核复用" if onchip else "QK 与 PV 各一遍") + ")")
        s.stage("gather", CUBE, reads=(), writes=("l1_k",), tasks=groups,
                scalar_ops=groups * topK * ntimes,
                note="每行一条搬运发射")
        unq.append((
            "逐行散读单价",
            f"{groups*topK:,} 次 {Dk*ESZ} 字节的非连续 GM 读; 本仓只标定过**向量**指令的"
            f"边际开销 (40.8 周期/条), MTE2 的逐行搬运单价没有标定",
            f"用 core/acl_driver 起一个只做 gather 的核, 行宽固定 {Dk*ESZ} B, "
            f"行数取两档解出 (固定开销, 每字节代价)"))
    else:
        kwin = S2 * Dk * ESZ + mT * S2 * FSZ
        s.move("k_read_qk", GM, L1, S2 * Dk * ESZ, tiles,
               unique=k_tensor, reuse_window=kwin)
        s.move("k_read_pv", GM, L1, S2 * Dv * ESZ, tiles,
               unique=d["B"] * d["N2"] * S2 * Dv * ESZ, reuse_window=kwin)

    if not onchip:
        s.move("sc_write", L0C, GM, rows * Keff * FSZ, 1)
        s.move("sc_read", GM, UB, rows * Keff * FSZ, 1)
        s.move("pr_write", UB, GM, rows * Keff * ESZ, 1)
        s.move("pr_read", GM, L1, rows * Keff * ESZ, 1)
    s.move("y_write", GM if not onchip else UB, GM, rows * Dv * ESZ, 1)

    #  ---- 计算
    nchunk = max(_ceil_div(Keff, nT), 1)
    #  标量操作数按**发射次数**数, 不是猜: 每个 (分块 x 键块) 要发 A 搬运 / B 搬运 /
    #  Mmad 各一条, 外加循环控制与地址计算 —— 这里按每轮 4 条计, 并在 note 里写明口径。
    #  ★msprof 实测现状 PV 核标量忙时比 96.4%、矩阵 1.8%, 即这一项才是该核的瓶颈资源。
    #  单价 (scalar_op_cycles) 本机没有标定, 所以 scalar_us 会报 unknown 并进 bound_missing。
    #  21 条不是估的: 2026-09-22 做"参数结构体提到循环外"那次改动时, 逐条数过 PV 内层
    #  循环体里的标量存数 —— 21 条, 其中只有 6 条随轮次变化 (那次改动把 15 条常量外提,
    #  耗时一字未动, 正是这个数字让"开销在**发射**而非参数赋值"成为可判的结论)。
    SCALAR_PER_INNER = 21
    s.stage("qk", CUBE, reads=("l1_q", "l1_k"), writes=("l0c",),
            macs=tiles * mPad * Keff * Dk, tasks=tiles * nchunk,
            scalar_ops=tiles * nchunk * SCALAR_PER_INNER,
            note="scalar_ops 口径: 每 (m分块 x 键块) 21 条标量存数 (实读内层循环体)")
    #  稠密: 掩码要先 Duplicate 整行再 Scatter topK 个位置, 共约 9 遍全宽;
    #  gather: 取出来的列就是被选中的列, 掩码构造整段消失, 只剩 6 遍 (+因果比较 2 遍)。
    #  稠密: 掩码要 Duplicate 整行 + Scatter topK 个位置, 共 7 遍逐元素 + 2 遍行归约;
    #  gather: 取出的列就是被选中的列, 掩码构造整段消失 -> 4 遍逐元素 + 2 遍行归约
    #          (因果时再加 2 遍逐元素做索引比较)。
    n_ep = (7 if not gather else (4 + (2 if d["causal"] else 0)))
    n_rp = 2
    passes = n_ep + n_rp
    if form["softmax"] == "online":
        blk_elem, blk_rows, reps = mT * nT, mT, tiles * nchunk
    else:
        blk_elem, blk_rows, reps = mT * Keff, mT, tiles
        if blk_elem * FSZ > 200 * 1024:       # UB 放不下整块 -> 退回逐行
            blk_elem, blk_rows, reps = Keff, 1, rows
    s.stage("softmax", VEC, reads=("l0c",), writes=("gm_pr" if not onchip else "l1_q",),
            elem_ops=rows * Keff * passes,
            instr=reps * _instr(blk_elem, blk_rows, n_ep, n_rp),
            tile_bytes=nT * FSZ, tasks=rows)
    s.stage("pv", CUBE, reads=("l1_k",), writes=("l0c",),
            macs=tiles * mPad * Keff * Dv, tasks=tiles,
            scalar_ops=tiles * nchunk * SCALAR_PER_INNER,
            note="同上口径")
    if form["softmax"] == "online":
        s.stage("rescale", VEC, reads=("l0c",), writes=("l0c",),
                elem_ops=rows * Dv * nchunk * 2,
                instr=tiles * nchunk * _instr(mT * Dv, mT, 2, 0),
                tile_bytes=Dv * FSZ, tasks=rows)
    #  ★标量单价没有标定时, 它是一项**未计入任何参考线**的代价 —— 必须进 unquantified,
    #    否则表面上排第一的形式就是在"标量免费"的假设下赢的, 而实测说标量才是瓶颈。
    from .. import socspec as _ss
    if not _ss.calib_value("scalar_op_cycles", SOC):
        tot_sc = sum(st.scalar_ops * s.stage_times[st.name] for st in s.stages)
        if tot_sc:
            unq.append((
                "标量发射单价",
                f"本形式共 {tot_sc:,} 条标量发射, 而本芯片没有 scalar_op_cycles 标定 —— "
                f"这一项**没有计入**上面的参考线。msprof 实测现状 PV 核标量忙时比 96.4%、"
                f"矩阵 1.8%, 即它是该核的瓶颈资源, 按 0 计会系统性高估所有形式",
                "对同一个核做两档标量操作数 (内层轮数减半) 各跑一次 "
                "`python -m core.perf --pipes`, 由 aic_scalar_ratio x 耗时的差解出每条周期数"))
    return Lowered(s, unq)


#  本模块判据所依据的芯片。换机器给 --soc, 不要让它散落在各函数里。
SOC = os.environ.get("CANN_BENCH_SOC", "ascend950pr_957b")

#  ---- core 自带的**合成算子谱**: 让本模块的判据在**没有任何 benchmark 数据**时也能自证 ----
#  两个形状刻意选成一大一小、一个能按头分组一个不能, 覆盖四条轴的主要分支。
#  它不对应任何真实提交, 所以没有 baseline_fn; EQ4 对它如实报 NOT_APPLICABLE。
_SYN_CASES = {
    1: {"case": 1, "B": 16, "N1": 64, "S1": 4, "Dk": 576, "N2": 1, "S2": 8192,
        "Dv": 512, "topK": 512, "G": 64, "ESZ": 2, "FSZ": 4, "causal": True,
        "layout": "BNSD", "dtype": "float16", "rows": 16 * 64 * 4, "groups": 16 * 1 * 4},
    2: {"case": 2, "B": 2, "N1": 32, "S1": 1024, "Dk": 128, "N2": 8, "S2": 8192,
        "Dv": 128, "topK": 1024, "G": 4, "ESZ": 2, "FSZ": 4, "causal": True,
        "layout": "BNSD", "dtype": "bfloat16", "rows": 2 * 32 * 1024, "groups": 2 * 8 * 1024},
}


def _syn_built(d):
    """合成谱的"现状形式": 稠密 + 落 GM + 两遍法, 按 S1 或 G 选分组。"""
    if d["S1"] >= 64:
        return {"grouping": "s1", "support": "dense", "softmax": "two_pass", "residency": "gm"}
    if d["G"] >= 64:
        return {"grouping": "head", "support": "dense", "softmax": "two_pass", "residency": "gm"}
    return None


register("_synthetic_attention", {
    "cases_fn": lambda: _SYN_CASES,
    "lower": lower_sparse_attention,
    "built": _syn_built,
    "note": "core 自带的合成谱, 只用于让判据在无 benchmark 数据时自证; 不对应任何真实提交",
})


#  ================================================================= 排序
def rank(op, case_id, sp=None, caps_dict=None, dram_GBps=None):
    """返回 (维度, 排好序的行, 被剪枝的)。行 = dict(form, lb, unq, over, capok)。"""
    from .. import socspec
    from . import caps as _caps
    ent = spec_of(op)
    if sp is None:
        sp = socspec.spec(SOC)
    if caps_dict is None:
        caps_dict = _caps.as_dict(SOC)
    if dram_GBps is None:
        dram_GBps = socspec.calib_value("dram_GBps", sp["soc"])
    d = cases_of(op)[case_id]
    forms, cut, noted = enumerate_forms(d)
    out = []
    for form in forms:
        lw = ent["lower"](d, form)
        lb = lw.sch.lower_bounds(sp, dram_GBps)
        over = [sp_ for sp_, u, c, r, _e in lw.sch.capacity(sp["cap"])
                if r is not None and r > 1]
        capok, _rows = (lw.sch.check_caps(caps_dict) if lw.sch.requires else (True, []))
        out.append({"form": form, "sch": lw.sch, "lb": lb, "unq": lw.unquantified,
                    "over": over, "capok": capok,
                    "cost_notes": noted.get(form_id(form), [])})
    #  排序: 装不下 / 机制不成立的沉底; 其余按完美重叠下界升序
    def key(r):
        bad = 0 if (not r["over"] and r["capok"] is True) else 1
        v = r["lb"].get("ideal_overlap_floor_us")
        return (bad, v.value if isinstance(v, Qty) and v.value is not None else 1e18)
    out.sort(key=key)
    return d, out, cut


def show(op, case_id, top=8):
    d, rows, cut = rank(op, case_id)
    print(f"=== {op} case {d['case']}  "
          f"B{d['B']} N1{d['N1']} S1{d['S1']} Dk{d['Dk']} | N2{d['N2']} S2{d['S2']} "
          f"Dv{d['Dv']} | topK{d['topK']} G{d['G']} {d['dtype']}"
          f"{' causal' if d['causal'] else ''}")
    print(f"    等价形式 {len(rows)+len(cut)} 种 = {len(rows)} 可行 + {len(cut)} 被前提剪枝\n")
    hdr = (f"{'形式':<46}{'理想重叠':>10}{'服务和':>10}{'MAC':>12}"
           f"{'DRAM(MB)':>10}{'向量条':>11}  状态")
    print(hdr)
    print("-" * len(hdr.encode('utf-8')) if False else "-" * 108)
    built = built_form(op, d)
    for r in rows[:top]:
        f = r["form"]
        mark = " ←现状" if built and f == built else ""
        mx = r["lb"].get("ideal_overlap_floor_us")
        sr = r["lb"].get("service_sum_us")
        mac = sum(st.macs * r["sch"].stage_times[st.name]
                  for st in r["sch"].stages if st.unit == CUBE)
        dram = r["lb"]["dram_bytes"].value
        vi = r["lb"]["vector_instr"].value
        st = ("★装不下 " + ",".join(r["over"])) if r["over"] else (
            "机制探不到, 拒绝下结论" if r["capok"] is None else (
                "机制不成立" if r["capok"] is False else (
                    f"★含 {len(r['unq'])} 项未标定" if r["unq"] else "可比")))
        print(f"{form_id(f):<46}{mx.value:>10,.0f}{sr.value:>10,.0f}{mac:>12,.3g}"
              f"{dram/1e6:>10,.1f}{vi:>11,.0f}  {st}{mark}")
    #  ---- 未标定项: 必须单列, 不许只当脚注
    top1 = rows[0]
    if top1["unq"]:
        print(f"\n★第一名 {form_id(top1['form'])} 含**未标定代价项**, "
              f"本模块**拒绝宣布它更快**。要先做的测量:")
        for nm, why, how in top1["unq"]:
            print(f"  - {nm}: {why}")
            print(f"    做法: {how}")
    if built:
        cur = next((r for r in rows if r["form"] == built), None)
        if cur is not None and cur is not top1:
            a = cur["lb"]["ideal_overlap_floor_us"].value
            b = top1["lb"]["ideal_overlap_floor_us"].value
            print(f"\n⇒ 现状形式 {form_id(built)} 的下界 {a:,.0f} us; "
                  f"表中第一名 {b:,.0f} us, 相差 {a/b:.1f}x。"
                  f"\n  下界之差**不是**收益预测 (本仓已证伪五种按量估收益的做法), "
                  f"它只说明: 在这个形状上还存在一个**数学等价**的写法, "
                  f"其硬件约束比现状宽 {a/b:.1f} 倍。")
    if cut:
        print(f"\n被剪枝的 {len(cut)} 种 (只按 equivalence/lowerable 剪; cost 不剪):")
        seen = set()
        for form, kind, why in cut:
            k = (kind, why.split(":")[0])
            if k in seen:
                continue
            seen.add(k)
            print(f"  [{kind}] {why}")
    notes = top1.get("cost_notes") or []
    if notes:
        print(f"\n第一名带 {len(notes)} 条**代价标注** (不剪枝, 已按代价进判据):")
        for n in notes:
            print(f"  {n}")
    return 0


#  ---- 负例控制 (门禁 G1 要覆盖) ----
SYN = "_synthetic_attention"


def _nc_cost_is_not_a_precondition():
    """代价条件**不得**剪枝: 它只标注, 形式必须仍在表里。

    ★这条是 2026-09-22 外部评审的直接产物。首版把"按 s1 分组时 gather 收益归零"写成
      返回 False, 于是 case 1/3/5/20 (正是我们比厂商基线慢 10~17 倍的那一组) 的整条
      gather 分支被抹掉, 表里报"无更优等价形式"。造这个模块的目的恰恰是不要因为今天的
      实现习惯而漏掉明天的结构 —— 把代价写进前提就自己违反了这一条。
    """
    d = cases_of(SYN)[2]                       # 合成谱的 G=4, S1=1024 那一档
    forms, cut, noted = enumerate_forms(d)
    gs = [f for f in forms if f["support"] == "gather" and f["grouping"] == "s1"]
    assert gs, "gather+s1 被剪掉了 —— 代价条件又变成了剪枝条件"
    assert any("索引行不同" in n for f in gs for n in noted.get(form_id(f), [])), \
        "gather+s1 没有带上代价标注 —— 标注丢了等于没说"
    #  被剪掉的只能是 equivalence/lowerable 两类
    assert all(k in PRUNING for _f, k, _w in cut), \
        f"有非 equivalence/lowerable 的剪枝: {[k for _f, k, _w in cut]}"
    return True


def _nc_precision_obligation_declared():
    """非 bitwise 的轴必须带精度义务, 且义务里要点名验法。

    「数学公式等价」不等于「这个核可以安全替换基线」。support=gather 与 softmax=online
    都是实数域恒等而浮点下求和项数/次序都变了, 它们各自带一条要 simfunc 验的义务。
    """
    import re as _re
    for a in AXES:
        if a.equiv_kind == BITWISE:
            continue
        assert a.precision_obligation, a.name
        assert _re.search(r"验法|simfunc", a.precision_obligation), \
            f"轴 {a.name} 的精度义务没写**怎么验** —— 写不出验法的义务等于没有义务"
    #  Axis 构造器必须拒绝"非 bitwise 但不写义务"
    try:
        Axis("x", MATH, "id", {"a": "A"}, equiv_kind=REAL_EXACT)
    except ValueError:
        return True
    raise AssertionError("Axis 允许了 real_exact 而不写精度义务")


def _nc_onchip_two_pass_rejected_by_capacity():
    """on_chip + two_pass 必须被**容量判据**拒, 而不是被写死的前提拒。

    用判据拒 > 用决条拒: 换个形状 (Keff 小到整块放得下 UB) 判据会自动放行, 决条不会。
    """
    from .. import socspec
    sp = socspec.spec(SOC)
    d = cases_of(SYN)[1]
    lw = lower_sparse_attention(d, {"grouping": "head", "support": "dense",
                       "softmax": "two_pass", "residency": "on_chip"})
    over = [x[0] for x in lw.sch.capacity(sp["cap"]) if x[3] is not None and x[3] > 1]
    assert "UB" in over, f"on_chip+two_pass 没被容量判据拒: {over}"
    return True


def _nc_ceil_not_floor():
    """分块数必须 ceil: 尾块不能被静默丢掉。"""
    assert _ceil_div(1000, 512) == 2 and _ceil_div(1024, 512) == 2 and _ceil_div(1, 512) == 1
    return True


def _nc_v_is_prefix_of_k():
    """V 不是 K 前缀时降级必须**报错**, 不得静默按一次 gather 计。"""
    d = dict(cases_of(SYN)[1])
    d["Dv"] = d["Dk"] + 64
    try:
        lower_sparse_attention(d, {"grouping": "head", "support": "gather",
                      "softmax": "two_pass", "residency": "gm"})
    except AssertionError:
        return True
    raise AssertionError("Dv>Dk 时降级没有报错 —— gather 搬运量被静默低估一半")


def _nc_unquantified_blocks_winner():
    """第一名带未标定项时, 必须走"拒绝宣布赢家"的分支 —— 不能静默排个名就算数。"""
    d = cases_of(SYN)[1]
    lw = lower_sparse_attention(d, {"grouping": "head", "support": "gather",
                       "softmax": "two_pass", "residency": "gm"})
    assert lw.unquantified, "gather 形式必须自带未标定项 (逐行散读单价)"
    return True


def _nc_built_form_must_be_feasible():
    """**最重要的一条**: 已经在真机上跑着的形式, 必须被本模型判为可行。

    判它不可行只可能有两种解释 —— 硬件在跑一个装不下的东西 (不可能), 或者降级模型
    算错了。所以这条自检的失败一律归责给 lower_*, 不归责给形式。
    2026-09-22 首次运行时它就抓到一次: nT 的选取只算了 L0C 没算 L1, 于是全部 9 种
    形式 (含正在跑的那一种) 都被判成"装不下 L1"。没有这条自检, 那张表会被当成
    "所有路都堵死"读, 而真实情况是模型少了一个约束。
    ★遍历**所有已注册的谱**; 一个 built 都没有时报 NotApplicable, 不假装通过。
    """
    from .. import socspec
    sp = socspec.spec(SOC)
    bad, checked = [], 0
    for name, ent in sorted(REGISTRY.items()):
        if not ent.get("built"):
            continue
        for cid, d in cases_of(name).items():
            built = built_form(name, d)
            if built is None:
                continue              # 该形状走回落路径, 不在本表覆盖内
            checked += 1
            lw = ent["lower"](d, built)
            over = [s_ for s_, u, c, r, _e in lw.sch.capacity(sp["cap"])
                    if r is not None and r > 1]
            if over:
                bad.append((name, cid, over))
    if not checked:
        raise NotApplicable("没有任何已注册的谱声明了 built (现状形式) —— 这条锚无处可施")
    assert not bad, f"现状形式在 {bad} 上被判装不下 —— 错在 lower_*, 不在形式"
    return True


def _nc_bound_below_vendor_baseline():
    """**外部真值锚**: 下界必须低于同硬件同形状上一个**真实实现**的实测耗时。

    下界高于它 = 模型被证伪, 没有解释余地。2026-09-22 这条一上来就判死 7/20 个 case,
    并直接定位出三处建模缺陷: 指令条数按行数算 (虚高约 30 倍)、gather 强制经 GM 暂存、
    L2 命中流量零代价。在此之前本模块唯一的锚是"现状形式必须可行", 那只能查容量。
    ★基线来自具体 benchmark 的报告, 属**插件层**数据。没有任何谱提供 baseline_fn 时
      报 NotApplicable —— 不适用不等于通过, 也不等于未过。
    """
    from .. import socspec
    from . import caps as _caps
    sp = socspec.spec(SOC)
    cd = _caps.as_dict(SOC)
    dram = socspec.calib_value("dram_GBps", sp["soc"])
    bad, checked = [], 0
    for name, ent in sorted(REGISTRY.items()):
        bf = ent.get("baseline_fn")
        if not callable(bf):
            continue
        base = bf() or {}
        for cid, bl in sorted(base.items()):
            _d, rows, _c = rank(name, cid, sp, cd, dram)
            ok = [r for r in rows if not r["over"] and r["capok"] is True]
            if not ok or not bl:
                continue
            checked += 1
            v = ok[0]["lb"]["ideal_overlap_floor_us"].value
            if v and v > bl:
                bad.append((name, cid, round(v), bl, round(v / bl, 2)))
    if not checked:
        raise NotApplicable(
            "没有任何已注册的谱提供 baseline_fn (真实实现的实测耗时) —— 这条外部锚"
            "在本环境不适用。接入 benchmark 的插件应提供它")
    assert not bad, (f"下界高于真实实现实测的 case (谱, case, 下界, 基线, 倍数): {bad} —— "
                     f"模型被证伪, 先修模型再看排名")
    return True


def _nc_unquantified_is_opportunity_not_obligation():
    """EQ5: 从模型里**删掉**一项已知成本, 不得因此自动提高该候选的强制优先级。

    2026-09-22 codex 评审第三条的验收条件。反向激励是这样成立的: 漏算一项成本 ->
    下界更低 -> 更容易超过 1.2 倍 -> 更容易触发"必须处置"。于是模型越不完整,
    越能给自己派活, 而本模块原本是要用模型**省**真机试错。
    本负例沿真实入口 g12_equiv_handled 跑, 只喂一个 case:
      (A) 首位形式带未量化项 -> 只算机会, 门禁不得红
      (B) 把未量化项抹掉 (假装已标定) -> 同一个形式立刻产生处置义务
    两者的下界完全一样, 差别只在"成本项是否已全部量化" —— 权限必须挂在后者上。
    """
    import contextlib as _ctx
    import io as _io
    from . import gate as _gate
    #  ★把被测门禁的输出吞掉: 它在这里是**被测对象**, 它打出的 ✗ 不是本次门禁的结论。
    #    不吞掉会出现"汇总说 0 条未过, 屏幕上却有 ✗"——正是本仓一直在消灭的
    #    "输出与裁决不一致"。
    _sink = _io.StringIO()
    ent = REGISTRY[SYN]
    one = {1: cases_of(SYN)[1]}
    real_cases, real_lower = ent["cases_fn"], ent["lower"]
    saved = dict(REGISTRY)
    try:
        REGISTRY.clear()
        REGISTRY[SYN] = dict(ent, cases_fn=lambda: one)
        with _ctx.redirect_stdout(_sink):
            bad_with = _gate.g12_equiv_handled(verbose=False)

        def _no_unq(d, form):
            lw = real_lower(d, form)
            lw.unquantified = []           # 假装两项成本都已标定
            return lw
        REGISTRY[SYN] = dict(ent, cases_fn=lambda: one, lower=_no_unq)
        with _ctx.redirect_stdout(_sink):
            bad_without = _gate.g12_equiv_handled(verbose=False)
    finally:
        REGISTRY.clear()
        REGISTRY.update(saved)
    assert bad_with == 0, (
        f"带未量化项的形式产生了 {bad_with} 条处置义务 —— 模型漏算成本就能给自己派活")
    assert bad_without > 0, (
        "成本项全部量化之后反而不产生义务 —— 那这条闸什么也没管住")
    return True


def _nc_enumerate_finds_nonbuilt_optimum():
    """EQ1 的负例: 在一个**已知**存在更优等价写法的形状上, 枚举器不得把现状排第一。

    case 17 是 2026-09-22 的实证锚: 它的收缩维可以从 S2=8192 缩到 topK=512 (16 倍),
    而现状是稠密。枚举器若把现状排第一, 说明轴集合漏了那条改写 —— 那正是本模块
    存在的理由被抽掉。
    """
    assert _nc_cost_is_not_a_precondition()
    assert _nc_precision_obligation_declared()
    assert _nc_onchip_two_pass_rejected_by_capacity()
    assert _nc_ceil_not_floor()
    assert _nc_v_is_prefix_of_k()
    _d, rows, _cut = rank(SYN, 1)
    usable = [r for r in rows if not r["over"] and r["capok"] is True]
    assert usable, "case 17 一种可行形式都没有 —— 降级模型有问题"
    built = built_form(SYN, _d)
    assert usable[0]["form"] != built, "枚举器把现状排第一 —— 轴集合漏了 support=gather"
    assert usable[0]["form"]["support"] == "gather", usable[0]["form"]
    return True


NEGATIVE_CONTROLS = {
    "EQ1_enumerate_beats_tuning": _nc_enumerate_finds_nonbuilt_optimum,
    "EQ2_built_form_is_the_ground_truth": _nc_built_form_must_be_feasible,
    "EQ3_unquantified_blocks_the_winner": _nc_unquantified_blocks_winner,
    "EQ4_vendor_baseline_is_the_ceiling": _nc_bound_below_vendor_baseline,
    "EQ5_unquantified_is_opportunity_not_obligation":
        _nc_unquantified_is_opportunity_not_obligation,
}


#  ================================================================= 全 case 扫描
def sweep(op, bench_report=None, top=1):
    """扫全部 case: 现状形式的下界 vs 等价类里最好的那一支, 报倍数。

    给了 --bench-report 就再乘上实测耗时, 出**模型允许的最大可回收时间**并据此排序。
    ★这是**上界**不是收益预测 —— 本仓已证伪五种按量估收益的做法 (docs/perf-ceiling-estimation.md),
      这里出的数只回答"值不值得去测", 不回答"能省多少"。
    """
    from .. import socspec
    from . import caps as _caps
    if op not in REGISTRY:
        #  ★如实报"没登记", 不静默跳过: 一个算子没有等价形式表, 与"它没有更优等价形式"
        #    是两回事, 混起来正是本模块要消灭的东西。
        print(f"=== 等价形式表: **{op} 尚未登记** —— 本轮不输出等价类排名。")
        print(f"    已登记: {', '.join(sorted(REGISTRY)) or '(无)'}")
        print( "    登记方法: 写一个模块在 import 时调 equiv.register(名字, spec), "
               "再用 `--spec <模块>` 加载它 (benchmark 接入属**插件层**, core 不 import 插件)。")
        return 0
    sp = socspec.spec(SOC)
    cd = _caps.as_dict(SOC)
    dram = socspec.calib_value("dram_GBps", sp["soc"])
    ent = spec_of(op)
    cases = cases_of(op)
    meas = {}
    if bench_report:
        from ..hwfloor import bench_report_floors
        for r in bench_report_floors(bench_report):
            try:
                meas[int(r["case"])] = r["us"]
            except (TypeError, ValueError):
                pass
    rows = []
    for cid in sorted(cases):
        d = cases[cid]
        built = built_form(op, d)
        forms, _cut, _noted = enumerate_forms(d)
        best = cur = None
        for form in forms:
            lw = ent["lower"](d, form)
            if [x for x in lw.sch.capacity(sp["cap"]) if x[3] is not None and x[3] > 1]:
                continue
            capok = lw.sch.check_caps(cd)[0] if lw.sch.requires else True
            if capok is not True:
                continue           # 机制不成立 / 探不到 -> 不参与排名 (探不到不折叠成有)
            v = lw.sch.lower_bounds(sp, dram).get("ideal_overlap_floor_us")
            v = v.value if isinstance(v, Qty) and v.value is not None else None
            if v is None:
                continue
            if best is None or v < best[1]:
                best = (form, v, lw.unquantified)   # unq 非空时下面一律改口, 不称"最优"
            if built and form == built:
                cur = (form, v)
        if best is None:
            continue
        ratio = (cur[1] / best[1]) if (cur and best[1]) else None
        us = meas.get(cid)
        head = us * (1 - 1 / ratio) if (us and ratio and ratio > 1) else None
        rows.append(dict(case=cid, cur=(cur[1] if cur else None), best=best[1],
                         form=best[0], ratio=ratio, us=us, head=head,
                         unq=len(best[2])))
    rows.sort(key=lambda r: -(r["head"] if r["head"] is not None
                              else (r["ratio"] or 0)))
    print(f"=== {op}: 每个 case 的**等价类排名首位** vs 现状形式")
    if ent.get("note"):
        print(f"    现状形式: {ent['note']}")
    if not meas:
        print("    ★没给 --bench-report: 只能按倍数排, **排不出先做哪个** —— "
              "绝对可回收时间要实测耗时才算得出 (本仓硬规则: 倍数定手段, 绝对值定次序)。")
    hdr = (f"{'case':>5}{'实测us':>10}{'现状下界':>10}{'排名首位':>10}{'倍数':>7}"
           f"{'可回收上界us':>13}  排名首位的等价形式 (★ = 带未标定项, **不称最优**)")
    print(hdr)
    print("-" * 104)
    tot = 0.0
    for r in rows:
        tot += r["head"] or 0.0
        c_us = f"{r['us']:,.0f}" if r["us"] else "—"
        c_cur = f"{r['cur']:,.0f}" if r["cur"] else "—"
        c_rat = f"{r['ratio']:.1f}x" if r["ratio"] else "—"
        c_hd = f"{r['head']:,.0f}" if r["head"] else "—"
        #  ★与 show() 的诚实性纪律对齐: 带未标定代价项的形式只叫"排名首位", 不叫"最优",
        #    它的可回收上界也随之不可信 (2026-09-22 采纳外部评审的语义冲突指正)。
        tail = f"  ★{r['unq']}项未标定, 该行倍数与可回收上界均待定" if r["unq"] else ""
        print(f"{r['case']:>5}{c_us:>10}{c_cur:>10}{r['best']:>10,.0f}"
              f"{c_rat:>7}{c_hd:>13}  {form_id(r['form'])}{tail}")
    if meas:
        print(f"\n合计可回收上界 {tot:,.0f} us。★这是**模型允许的最大值**, 不是收益预测。")
    return 0


REFS = {
    "EQ1_enumerate_beats_tuning": Ref(
        what="等价形式必须被**枚举**而不是靠人想起来; 没有枚举时优化动作会退化成调参",
        status="measured",
        evidence=[
            "SFA 2026-09-22: 本轮七个改动里六个是既定形式内的调参 (kBlk / kStep / 参数外提 /",
            "缓冲深度 / VF 改写 / 三段全融), 三个被实测判负、两个零效果;",
            "唯一的大赢是换了一种**数学等价写法** (M 维由 s1 换成共用同一份 K 的 G 个头),",
            "110266 -> 60838 us (-45%), 而它是在四轮无效调参之后用手找出来的。",
            "本模块把这次手工发现变成一条可枚举的轴 (grouping), 并由 support 轴的前提",
            "自动推出'只有按头分组才能 gather' —— 即那次发现的结构原因现在是机器可判的。",
        ],
        verified="2026-09-22"),
    "EQ2_built_form_is_the_ground_truth": Ref(
        what="已在真机跑着的形式必须被降级模型判为可行; 判不可行一律归责给模型",
        status="measured",
        evidence=[
            "2026-09-22 首次运行: 9 种可行形式全被判'装不下 L1', 含正在跑的那一种。",
            "原因是 nT 的选取只算了 L0C 没算 L1。没有这条自检, 那张表会被读成'所有路都堵死'。",
            "★这条是本模块**唯一的外部真值锚**: 枚举与排序本身无法自证, 它靠现状可行性证伪。",
        ],
        verified="2026-09-22"),
    "EQ5_unquantified_is_opportunity_not_obligation": Ref(
        what="下界更低不等于该建它; 只有**成本项已全部量化**的更优形式才产生处置义务",
        status="measured",
        evidence=[
            "2026-09-22 codex 评审第三条, 内存替身复现: 现状下界 100us、候选下界 50us 且",
            "含未标定项时, 首版 G12 返回 1 (失败)。反向激励因此成立 —— 模型漏算一项成本,",
            "下界就更低, 更容易超过 1.2 倍, 更容易制造'必须处置'的义务。",
            "★而下界比较本身推不出实现排序: L_候选 < L_现状 只说明候选**尚未被这些下界",
            "排除**, 完全可能现状实际 100us 而候选实际 200us。",
            "现权限分三档: 已全部量化且更优 -> 义务; 含未量化项 -> 机会(不红); 装不下 -> 排除。",
            "本机实况: 16 个 case 的首位形式都含 2 项未量化 (逐行散读单价 / 标量发射单价),",
            "按新规全部降为机会 —— 该做的是那两次标定, 不是先建 16 个候选。",
        ],
        verified="2026-09-22"),
    "EQ4_vendor_baseline_is_the_ceiling": Ref(
        what="下界必须低于**厂商基线**在同硬件同形状上的实测耗时; 高于它即模型被证伪",
        status="measured",
        evidence=[
            "2026-09-22: 本模块建成当天, dubin 指出'榜一 80 分说明的确存在更优算法实现,",
            "没效果说明工具建设质量不高'。据此把评测报告里的 baseline_perf_us 接成外部锚,",
            "一上来判死 7/20 个 case —— case 1 下界 767 us 而基线实测 104.9 us (高 7.3 倍)。",
            "由它直接定位出三处建模缺陷: (1) 向量指令条数按**行数**算而非按分块几何,",
            "虚高约 30 倍, 系统性地把'在线归约+片上驻留'排到'稠密+落盘'之后;",
            "(2) gather 分支强制经 GM 暂存的一写两读, 给 case 1 平白加 537 MB 请求量;",
            "(3) 命中 L2 的请求量在下界里零代价 (SFA gather 形式请求 4.29 GB 而下 DRAM",
            "只有 33.5 MB), 已补 IR7 的 l2_us 项并在缺标定时报 unknown。",
            "修完 0/20 被证伪, 且 20 个 case **一致**指向同一形式 (按头分组+片上驻留+",
            "在线归约+gather), 而现状形式在四条轴里有三条取反。",
            "★此前本模块唯一的锚是'现状形式必须可行', 那只能查容量, 查不出下界虚高。",
        ],
        verified="2026-09-22"),
    "EQ3_unquantified_blocks_the_winner": Ref(
        what="第一名带未标定代价项时必须拒绝宣布赢家, 改为输出要先做哪一次测量",
        status="measured",
        evidence=[
            "SFA case 17 的等价类最优是 gather 形式, MAC 降 16 倍、DRAM 降 3.9 倍,",
            "但它引入 32768 次 1152 字节的非连续 GM 读, 而本仓只标定过**向量**指令的",
            "边际开销 (40.8 周期/条), MTE2 逐行搬运单价没有标定。",
            "★没有这条, 枚举器就是幻想发生器: 它总能排出一个纸面上更快的形式。",
        ],
        verified="2026-09-22"),
}
