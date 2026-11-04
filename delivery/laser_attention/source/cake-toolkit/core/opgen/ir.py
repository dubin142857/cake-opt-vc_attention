#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opgen.ir —— 算子**调度**的中间表示, 以及从它推出来的四类判据。

为什么要一个 IR (不是为了好看)
------------------------------
这个会话里 sparse_flash_attention 的每一次结构决策, 拆开看都是同四个问题:

  1. 这样分块, 各级存储装得下吗          (UB 装不下整行 -> 只融 QK-softmax 不成立)
  2. 这样排循环, GM 要搬多少遍           (m 外层两遍让 K 读取翻倍, 恰好吃掉省下的分数往返)
  3. 谁产谁消, 中间量落不落 GM           (有 C-V 直通就不落, 流量 10.7 GB -> 2.2 GB)
  4. 分块做小之后, 向量指令还剩多少效率  (nT=512 的指令只覆盖 2 KB, 固定开销占比升高)

这四问与算子无关, 但每换一个算子就得重推一遍, 而推的过程全在脑子里、结论散在 prose 里。
IR 的作用就是把这四问变成**可执行的算术**: 描述一次调度, 四类判据自动出数, 方案之间
可以并排比较。它不生成最优解 —— 它让人写下的方案**立刻被容量与流量证伪**, 省掉一轮构建。

这个 IR 的定位: **决策与证伪**, 不是代码生成
--------------------------------------------
调研过的三条现成路线都是**生成式**的 (2026-09-22):
  AscendNPU-IR (MLIR, 2025-09 开源): HFusion 硬件无关层 + HIVM 硬件相关层, 算子表与本仓
    手写的东西一一对应 (hivm.hir.fixpipe / mix_matmul / sync_block_set / set_flag /
    pipe_barrier, 空间 GM/UB/L1/L0), 覆盖 A2/A3 到 950 的 SIMD/SIMT。
  TileLang (arXiv 2504.17577) + Ascend 适配: 把数据流 (GEMM/COPY/REDUCE/ATOMIC 的 tile
    算子) 与调度空间 (线程绑定/布局/张量化/流水) 解耦, 布局当类型属性用代数约束推断;
    Ascend 侧 2025-12 加了 T.Pipelined 的 CV 自动流水。
  Exo / Exo-2 / Exo-GPU (PLDI'22 / ASPLOS'25): exocompilation —— 硬件指令与专用内存
    **定义在用户库而非编译器里**, 调度是可组合、带检查的重写规则。

它们的产物是代码。本仓不需要生成代码 (AscendC 手写, 生成侧厂商已有栈), 反复付代价的是
另一件事: **一个被提出的调度, 在构建之前能不能被证伪**。所以本 IR 只做四件生成式 IR
不做的事:
  (a) 能力即类型: 调度声明它依赖哪些芯片机制, 对 opgen.caps 探到的集合做检查;
      探不到 (None) 不折叠成"没有", 而是拒绝对依赖它的方案下结论。
  (b) 流量三个数: 请求量 / 下 DRAM 量 / 复用窗口。多级 roofline 是对成品核的分析,
      而复用窗口必须在**提出方案时**就写下来, 否则带宽反推会得出不可能的数 (见 Move)。
  (c) 每个推导量带标定状态; 本机没标定的报区间, 并写出能把区间收成一点的那次测量。
  (d) 同步边由数据流派生: HIVM 把 sync_block_set/wait 做成你去写的算子, 本 IR 反过来
      从"跨单元的生产-消费边 x 1:N 配比"推出**需要几个 flag**, 再由 core/mixaddr 去核对
      手写代码发够没有。

术语与 HIVM 对齐 (空间名 GM/UB/L1/L0A/L0B/L0C, 搬运动词 load/store/copy/fixpipe),
不自造词 —— 将来要对接或对照厂商栈时不必翻译。

不做什么
--------
不做符号推导。IR 是**给定形状的具体实例**: 分块大小、循环次数全是数, 不是表达式。
理由是本仓的硬规则: 禁用小形状的成本分布去定大形状的优先级 —— 一个能对"任意形状"
给出答案的模型, 一定是在某处把形状相关的取舍藏起来了。
"""
from __future__ import annotations

import math

from .._refs import Ref
from .qty import (Qty, measured, derived, default, unknown, unavailable,
                  report_all)

REFS = {
    "IR1_capacity_before_build": Ref(
        what="调度的各级存储占用必须先于构建算清; 超了就是编译白跑",
        status="measured",
        evidence=[
            "SFA 2026-09-22: PV 的 L0B 在 Dv=512 时 2*32*512*2 = 64 KB 整, Dv=576 溢出;",
            "首版为此把 kStep 对所有 Dv 一刀切成 16, 于是 Dv=128 的 case 要发 512 条 Mmad,",
            "PV 占该 case 耗时 72%。按容量算出每个 Dv 各自的上界后降到 35%。",
            "即容量既是**可行性**约束也是**性能**约束, 拍一个保守值两头都输。",
        ],
        verified="2026-09-22"),
    "IR2_traffic_counts_reuse": Ref(
        what="GM 流量要按循环次序数重复读取次数, 不是按张量大小求和",
        status="measured",
        evidence=[
            "SFA case 3: K 张量本身只有 2 MB/(b,n2), 但 m 外层下每个 m 分块都要重读一遍,",
            "1024 个 m 分块 => 2.15 GB。实测 QK 段 2123 us 搬 4.3 GB = 2.03 TB/s,",
            "与'按张量大小求和'得到的 36 MB 差两个数量级 —— 后者会把带宽瓶颈完全看漏。",
        ],
        verified="2026-09-22"),
    "IR7_l2_hits_cost_time": Ref(
        what="命中 L2 的请求量也要计时; 判成'不下 DRAM'不等于零代价",
        status="disputed",
        evidence=[
            "2026-09-22: 判据 IR2 把复用窗口小于 L2 的重复读判成不下 DRAM, 之后这部分",
            "请求量在时间上就**完全免费**了。SFA 的 gather 形式请求 4.29 GB 而下 DRAM 只有",
            "33.5 MB —— 差的 4.26 GB 无代价, 下界因此可以低到任意程度。",
            "★状态是 disputed 而非 measured: 本机**没有** l2_GBps 标定, 所以这一项目前只能",
            "报 unknown 并进 bound_missing。910B 的标定曲线显示 L2 驻留段 4858 GB/s 与",
            "DRAM 饱和段 1429 GB/s 差 3.4 倍, 即这一项的量级不可忽略。",
            "收敛它: 用 core/probe 的带宽臂把工作集压到 L2 容量以内重测一条曲线。",
        ],
        verified="2026-09-22"),
    "IR8_scalar_is_a_resource": Ref(
        what="标量发射是一类独立资源; 模型里没有它, 就无法解释以标量为瓶颈的核",
        status="measured",
        evidence=[
            "2026-09-22 msprof 实测 SFA 的 PV 核 (_Z17cann_bench_sfa_pv, blockDim 28):",
            "忙时比 标量 96.4% / MTE2 3.8% / MTE1 1.8% / 矩阵 1.8%, 等待全部 <= 4%。",
            "而本模型此前只有 cube / dram / l2 / vector 四类资源, **没有标量**。",
            "后果是可测的: PV 实测比当时的下界高 21 倍, 我把它读成'还有 21 倍余量'并据此",
            "调了三轮参数 (kBlk 放大 +5~8%、kStep 翻倍 +15~24%、参数外提 0), 全部失败。",
            "IR6 早就写着'实测高于服务时间之和 = 模型缺项, 不是优化空间' —— 缺的就是这一项。",
            "★本机没有 scalar_op_cycles 标定, 故 scalar_us 现报 unknown 并进 bound_missing:",
            "  宁可显式说'标量没算进来', 也不让它continue悄悄按 0 计。",
        ],
        verified="2026-09-22"),
    "IR6_two_sided_bounds": Ref(
        what="下界必须成对给: max(完美重叠) 与 sum(全串行)。只给 max 的模型不可证伪",
        status="measured",
        evidence=[
            "2026-09-22 SFA: PV 段实测 4299us, 而 max 下界 (cube 91 / DRAM 203) 只有 203us,",
            "工具只会说'还有 21 倍空间'。但全串行上界 sum ≈ 547us —— 实测比**全串行还高 7.9 倍**,",
            "那不是优化空间, 是**模型里缺项**。缺的项由 msprof 一跑就现形: 矩阵 84% 在等、",
            "MTE1 83% 在等, 真正忙的是 MTE2 (69%) 与标量 (47%) —— 而模型里根本没有'等'这个量。",
            "★单侧的界只能说'还有空间', 永远不能说'你的模型错了'。成对给出之后,",
            "  measured > serial 这一条就成了模型自身的证伪判据。",
        ],
        verified="2026-09-22"),
    "IR5_wave_quantization": Ref(
        what="算下界时必须带核间波量化因子, 不能默认完美均衡",
        status="measured",
        evidence=[
            "判据本体 mmplan 的 W7 (GemmV3 @910B2 2026-08-27 四点):",
            "tiles 32/48/96/25, 预测 max/mean 1.50/1.00/1.00/1.92, 实测 1.55/1.015/1.015/1.96,",
            "偏差 <= 4%; 同期端到端波满比波最差快 17% vs 慢 32%, 同一 kernel 只有 shape 不同。",
            "★本模块首版 lower_bounds 直接除以核数, **隐含假设完美均衡**, 是静默的错。",
            "★按 tile 数算会低估 (mmplan 实证: 按数 1.14 而实测 1.41), 故本项给的是乐观上界。",
            "SFA case3 实算: QK 0.9998 / PV 0.988 / softmax 0.9994 —— 这个算子上不是主导项,",
            "但小形状上会咬人 (rows=512 时 cube 侧 wave_eff 只有 0.286)。",
        ],
        verified="2026-09-22"),
    "IR4_instr_marginal_cost": Ref(
        what="把一条宽向量指令拆成多条时, 每多一条的边际代价 = 40.8 周期/条 (单核)",
        status="measured",
        evidence=[
            "sparse_flash_attention 2026-09-22 的七点扫描 (算子自身的两个变体, 非合成负载):",
            "softmax 核里 6 条逐元素全宽指令按宽度 W 分段, 算术与输出逐位不变。",
            "W=8192..128 对应 393K..25.2M 条指令, 耗时 8531.1..19467.7 us;",
            "线性拟合 t = 8357.5 us + 4.4148e-4 us/条 * 条数, 七点残差 <= 1.4%;",
            "折单核 4.4148e-4 us * 56 * 1.65 GHz = 40.8 周期/条。存于 chip_calib 的",
            "vec_instr_overhead_cycles, 含分块循环与地址计算的标量开销, 是边际代价不是发射延迟。",
            "★此前本条状态是 disputed 并只给 4~24 周期的区间 —— 实测比区间上界还高 1.7 倍,",
            "  说明那个区间是**猜**的。这正是本仓'没标定就给区间并写出收敛它的实验'的用处。",
        ],
        verified="2026-09-22"),
}

def wave_eff(tasks, cores):
    """有效并行度 = tasks / (ceil(tasks/cores) * cores)。完美均衡时 1.0。

    判据本体是 mmplan 的 W7 (GemmV3 四点实测, 预测与实测偏差 <= 4%)。这里是**唯一出处**
    的一次归拢: 同一算术此前在 mmplan / attnplan / sortplan 三处各写了一遍。
    ★mmplan 的实证里有一条反例: 按 tile **数**算会低估 (按数 1.14 而实测 1.41),
      因为尾块的**工作量**可能比整块小。本函数按数算, 所以它给的是**乐观**上界;
      尾块工作量不足整块时真实惩罚更大。
    """
    import math as _m
    if not tasks or not cores:
        return None
    return tasks / (_m.ceil(tasks / cores) * cores)


GM, L1, L0A, L0B, L0C, UB, REG = "GM", "L1", "L0A", "L0B", "L0C", "UB", "REG"
CUBE, VEC = "cube", "vector"
_ON_CHIP = (L1, L0A, L0B, L0C, UB)


class Buf:
    """一块缓冲。bytes 是**单份**字节数; double=True 表示乒乓 (占两份)。

    per_core: UB/L0*/L1 是每核私有的, 容量判据按单核算; GM 是全局的。
    """
    __slots__ = ("name", "space", "bytes", "double", "lifetime", "note")

    def __init__(self, name, space, nbytes, double=False, lifetime=None, note=""):
        self.name, self.space = name, space
        self.bytes, self.double, self.note = int(nbytes), bool(double), note
        #  lifetime: 该缓冲活着的 stage 名集合。不给则视为**全程存活**。
        #  ★2026-09-22 采纳外部评审第五条: 容量判据原先把所有缓冲无条件相加, 于是
        #    "和 > 容量"被当成"放不下"。生命期不相交的两块可以复用同一片 UB, 所以那个
        #    方向**不成立**: 和小于容量能证明放得下, 和大于容量证明不了放不下。
        #    给了 lifetime 就按逐时刻峰值算, 没给则如实声明用的是"全程存活"假设。
        self.lifetime = frozenset(lifetime) if lifetime else None

    def live_bytes(self):
        return self.bytes * (2 if self.double else 1)


class Stage:
    """一段计算。

    unit         : CUBE 或 VEC
    reads/writes : 缓冲名
    macs         : 本 stage **一次执行**的乘加数 (cube 用)
    elem_ops     : 本 stage 一次执行处理的"元素 x 遍数" (vector 用), 按 fp32 计
    instr        : 一次执行发出的向量指令条数 (vector 用, 判指令效率)
    tile_bytes   : 单条向量指令覆盖的字节数 (vector 用)
    """
    __slots__ = ("name", "unit", "reads", "writes", "macs", "elem_ops", "instr",
                 "tile_bytes", "tasks", "scalar_ops", "note")

    def __init__(self, name, unit, reads=(), writes=(), macs=0, elem_ops=0,
                 instr=0, tile_bytes=0, tasks=0, scalar_ops=0, note=""):
        self.name, self.unit = name, unit
        self.reads, self.writes = tuple(reads), tuple(writes)
        self.macs, self.elem_ops = int(macs), int(elem_ops)
        self.instr, self.tile_bytes = int(instr), int(tile_bytes)
        #  tasks: 本 stage 分到核上的**独立任务数** (tile 数)。给了才算波量化;
        #  不给则**如实报"未给"**而不是默认完美均衡 —— 静默的乐观假设是本模块首版的错。
        self.tasks = int(tasks)
        #  scalar_ops: 本 stage 一次执行发出的**标量**操作条数 (地址计算 / 循环控制 /
        #  参数装填 / 搬运与 Mmad 的发射)。不给则按 0 计, 而 lower_bounds 会明说
        #  "按 0 计 = 没描述, 不是没有" —— SFA 的 PV 核实测标量忙时比 96.4%。
        self.scalar_ops = int(scalar_ops)
        self.note = note


class Move:
    """一次数据搬运。

    bytes x times = **请求量** (核发出的读写字节数)。
    unique        = 这些请求触及的**不同**字节数 (张量自身的大小)。
    reuse_window  = 同一个字节两次被用之间, 中途还会触及多少别的字节。

    为什么要后两项 (2026-09-22 由实测逼出来的)
    ------------------------------------------
    SFA case 3 的 QK 段: K 被 1024 个 m 分块各重读一遍 = 2.15 GB 请求量, 但 K 本身
    每个 (b,n2) 只有 2 MB。把 2.15 GB 当成 DRAM 流量, 反推出的带宽是 2.03 TB/s ——
    超过本机标定 (1116 GB/s) 近一倍, 而那不是芯片跑超了, 是**这部分根本没下到 DRAM**,
    它命中 L2 (114.7 MB)。真正下 DRAM 的是那 2.15 GB 分数矩阵, 反推 1.04 TB/s,
    与标定吻合。

    ★把请求量当 DRAM 流量, 会把瓶颈指到错误的一段上 (这里会指向 K 的重读, 而真正
      该消灭的是分数矩阵的落盘)。判据: reuse_window <= L2 则按 unique 计 DRAM,
      否则按请求量计。两个都报, 并说明用的是哪一条。
    """
    __slots__ = ("name", "src", "dst", "bytes", "times", "unique", "reuse_window", "note")

    def __init__(self, name, src, dst, nbytes, times, unique=None, reuse_window=None, note=""):
        self.name, self.src, self.dst = name, src, dst
        self.bytes, self.times, self.note = int(nbytes), int(times), note
        self.unique = int(unique) if unique is not None else int(nbytes) * int(times)
        self.reuse_window = None if reuse_window is None else int(reuse_window)

    def requested(self):
        return self.bytes * self.times

    def dram(self, l2_bytes):
        """(下到 DRAM 的字节数, 依据)。"""
        #  ★判据是"请求量是否大于不同字节数", 不是"times 是否大于 1": 一次 move 内部也可能
        #    重复读 (SFA 的 si 被同一组的 G 个查询头各读一遍, times 仍是 1)。首版按 times 判,
        #    于是把 268 MB 的稀疏索引读算成 67 MB, 少了 200 MB。
        if self.unique >= self.requested():
            return self.unique, "无重复读取"
        if self.reuse_window is None:
            return self.requested(), "未给 reuse_window —— 保守按请求量计 (可能高估)"
        if l2_bytes and self.reuse_window <= l2_bytes:
            return self.unique, f"复用窗口 {self.reuse_window/1e6:.1f} MB <= L2 {l2_bytes/1e6:.1f} MB, 命中 L2"
        return self.requested(), (f"复用窗口 {self.reuse_window/1e6:.1f} MB > L2 "
                                  f"{l2_bytes/1e6:.1f} MB, 每次都下 DRAM")


class Schedule:
    """一个具体形状下的一种调度方案。"""

    def __init__(self, name, soc=None, note="", requires=()):
        self.name, self.soc, self.note = name, soc, note
        #  requires: 本调度依赖的芯片机制 (opgen.caps 的能力 id)。缺一条这个调度就不成立。
        self.requires = tuple(requires)
        self.bufs, self.stages, self.moves = {}, [], []
        self.stage_times = {}          # stage 名 -> 该 stage 在整个调度里执行几次

    def check_caps(self, caps_dict):
        """能力即类型。返回 (可行?, 逐条说明)。

        三种结果分开处理, **不合并**:
          有     -> 通过
          无     -> 该调度在这颗芯片上不成立
          探不到 -> **拒绝下结论**。把它当"无"会藏掉一条真实存在的路 (mmplan 2026-09-11
                   实撞), 当"有"则会让人据一个没根据的前提去构建。
        """
        rows, ok = [], True
        for cid in self.requires:
            ent = caps_dict.get(cid)
            if ent is None:
                rows.append((cid, "缺表", f"能力表里没有 {cid} 这一条"))
                ok = False
                continue
            v = ent.get("value")
            if v is False:
                rows.append((cid, "无", ent.get("evidence", "")))
                ok = False
            elif v is None:
                rows.append((cid, "探不到", ent.get("evidence", "")))
                ok = None if ok is not False else False
            else:
                rows.append((cid, f"有({v})" if v is not True else "有", ent.get("evidence", "")))
        return ok, rows

    def buf(self, *a, **kw):
        b = Buf(*a, **kw)
        self.bufs[b.name] = b
        return b

    def stage(self, *a, times=1, **kw):
        st = Stage(*a, **kw)
        self.stages.append(st)
        self.stage_times[st.name] = int(times)
        return st

    def move(self, *a, **kw):
        m = Move(*a, **kw)
        self.moves.append(m)
        return m

    # ---------------------------------------------------------------- 判据

    def capacity(self, caps_bytes):
        """判据 1: 各级存储占用 = **逐时刻峰值**, 不是无条件求和。

        返回行里第五项是 `exact`: True 表示所有缓冲都声明了 lifetime, 峰值是真的;
        False 表示有缓冲没声明, 按"全程存活"算, 于是**超了也证明不了放不下**
        (只是在这个假设下放不下)。调用方必须把这两种情况分开报。
        """
        used, rows = {}, []
        names = [st.name for st in self.stages] or ["*"]
        exact = all(b.lifetime for b in self.bufs.values() if b.space in _ON_CHIP)
        for t in names:
            cur = {}
            for b in self.bufs.values():
                if b.space not in _ON_CHIP:
                    continue
                if b.lifetime is not None and t not in b.lifetime:
                    continue
                cur[b.space] = cur.get(b.space, 0) + b.live_bytes()
            for sp_, v in cur.items():
                used[sp_] = max(used.get(sp_, 0), v)
        for space, u in sorted(used.items()):
            cap = caps_bytes.get(space)
            rows.append((space, u, cap,
                         None if cap in (None, 0) else u / cap, exact))
        return rows

    def gm_traffic(self, l2_bytes=None):
        """判据 2: 请求量与 DRAM 流量**分开**报。返回 (dram_total, requested_total, rows)。"""
        dram_tot = req_tot = 0
        rows = []
        for m in self.moves:
            if GM not in (m.src, m.dst):
                continue
            req = m.requested()
            dr, why = m.dram(l2_bytes)
            dram_tot += dr
            req_tot += req
            rows.append((m.name, m.src, m.dst, dr, req, why or m.note))
        rows.sort(key=lambda r: -r[3])
        return dram_tot, req_tot, rows

    def sync_edges(self, mix_ratio):
        """判据 3: 跨单元的生产-消费边 -> 需要的跨核旗标对数。

        一条边 = 一个 CUBE stage 写、一个 VEC stage 读 (或反向) 的**非 GM** 缓冲。
        每条边要一对 (发/等), 且矩阵侧每次要发 mix_ratio 个 flag。
        另需一条反向边 (消费方告诉生产方"可以覆写"), 否则第二轮就挂住。
        """
        edges = []
        for producer in self.stages:
            for consumer in self.stages:
                if producer.unit == consumer.unit:
                    continue
                shared = set(producer.writes) & set(consumer.reads)
                for nm in sorted(shared):
                    b = self.bufs.get(nm)
                    if b is None or b.space == GM:
                        continue
                    edges.append((producer.name, consumer.name, nm, b.space))
        n = len(edges)
        return edges, {
            #  ★mix_ratio 取不到时**不当 1**: 报 None 让调用方知道这个数没算出来。
            "forward_flags": (n * mix_ratio) if mix_ratio else None,
            "backward_flags": (n * mix_ratio) if mix_ratio else None,
            "ratio_known": mix_ratio is not None,
            "total_setflag_sites": 2 * n,
        }

    def lower_bounds(self, sp, dram_GBps=None, issue_cycles_range=None):
        """判据 4: 四条下界, 取最大者即该调度的下界。

        **每个量都带来源** (opgen.qty)。回落 (default/unknown/unavailable) 必须带
        "缺什么才能变成 measured", 由 Qty 的构造器强制 —— 这样"悄悄用了个缺省值"
        在类型上就不可能。
        """
        ghz = sp["freq_ghz"]
        cube_cores, vec_cores = sp["cube_cores"], sp["vector_cores"]
        #  ★2026-09-22 采纳外部评审: 这里原是 `sp.get(...) or 128` —— 取不到就悄悄用 128,
        #    而下游照样标成 derived, 于是"没这个字段"被洗成"推导所得"。本模块的整条
        #    provenance 纪律在这一行上破功。现在取不到就让向量各项直接变 unknown。
        vec_B_cy = sp.get("vec_calc_B_per_cycle")
        macs = sum(st.macs * self.stage_times[st.name]
                   for st in self.stages if st.unit == CUBE)
        mac_cy = (sp.get("mac_per_cycle") or {}).get("fp16")
        out = {}

        #  ---- 波量化 (IR5). **逐 stage 局部**, 不是全局取最差。
        #  ★2026-09-22 采纳外部评审: 原实现取 min(各 stage 的 wave_eff) 再去除**所有**
        #    stage 的工作量之和。一个 1us / 效率 0.25 的小 stage 会把一个 100us / 效率 1.0
        #    的大 stage 一起罚成 0.25 —— (1+100)/0.25 = 404us, 而逐 stage 应为
        #    1/0.25 + 100/1 = 104us。这是算术错误, 不是保守估计。
        def _eff(st, cores):
            if not st.tasks:
                return None
            return wave_eff(st.tasks, cores)

        def _report_eff(label, unit, cores):
            es = [(st.name, _eff(st, cores)) for st in self.stages if st.unit == unit]
            known = [(n, e) for n, e in es if e is not None]
            if not es:
                return
            if not known:
                out[label] = default(
                    None, f"{unit} 的 stage 都没给 tasks (分到核上的独立任务数) —— "
                          f"**不默认完美均衡**, 该单元的时间按无波量化算并在此标记")
            else:
                worst = min(known, key=lambda x: x[1])
                out[label] = derived(
                    worst[1], note=f"逐 stage 各自折算; 此处只报最差的一个 ({worst[0]}) 供参考, "
                                   f"计时用的是各 stage 自己的值。按 tile **数**算, "
                                   f"尾块工作量不足时真实惩罚更大 (mmplan W7)")
            miss = [n for n, e in es if e is None]
            if miss and known:
                out[label + "_missing"] = default(
                    len(miss), f"{unit} 有 {len(miss)} 个 stage 没给 tasks ({', '.join(miss)}), "
                               f"它们按无波量化计入 —— 即这部分**偏乐观**")

        _report_eff("wave_eff_cube", CUBE, cube_cores)
        _report_eff("wave_eff_vec", VEC, vec_cores)

        #  ---- 矩阵算力
        if mac_cy:
            peak = cube_cores * mac_cy * ghz * 1e9
            tot = 0.0
            for st in self.stages:
                if st.unit != CUBE or not st.macs:
                    continue
                e = _eff(st, cube_cores) or 1.0
                tot += st.macs * self.stage_times[st.name] / peak * 1e6 / e
            out["cube_us"] = derived(tot, "us", f"{macs:,} MAC, **逐 stage** 各按自己的 "
                                                f"wave_eff 折算后求和 ({cube_cores}核 x {mac_cy} "
                                                f"MAC每周期 x {ghz}GHz)")
        else:
            out["cube_us"] = unknown("ini 里取不到本芯片 fp16 的 mac_per_cycle", "us")

        #  ---- 流量
        l2 = (sp.get("cap") or {}).get("L2")
        dram_tot, req_tot, _ = self.gm_traffic(l2)
        out["dram_bytes"] = derived(dram_tot, "B", "按各 Move 的 unique/reuse_window 判 L2 命中后求和")
        out["requested_bytes"] = derived(req_tot, "B", "核发出的请求字节, 含命中 L2 的重复读")
        if dram_GBps:
            out["dram_us"] = derived(dram_tot / (dram_GBps * 1e9) * 1e6, "us",
                                     f"按标定 {dram_GBps} GB/s (calibrated_lower_bound)")
        else:
            out["dram_us"] = unavailable("没给 dram_GBps, 取不到本芯片的带宽标定", "us")
        #  ---- 命中 L2 的那部分也要花时间 (IR7)
        #  ★2026-09-22 由与**厂商基线实测**对照逼出来: 判"命中 L2 -> 不计入 DRAM"之后,
        #    这部分请求量就再没有任何时间代价了, 等于把 L2 当成零延迟无限带宽。
        #    SFA 的 gather 形式请求量 4.29 GB 而下 DRAM 只有 33.5 MB —— 差的 4.26 GB
        #    全部无代价, 下界因此可以低到任意程度。缺项必须显形, 不能当零。
        l2_hit = max(req_tot - dram_tot, 0)
        out["l2_hit_bytes"] = derived(l2_hit, "B", "请求量减去下 DRAM 的部分, 走 L2")
        from .. import socspec as _ss
        l2_bw = _ss.calib_value("l2_GBps", sp.get("soc"))
        if l2_bw:
            out["l2_us"] = measured(l2_hit / (l2_bw * 1e9) * 1e6, "us",
                                    f"按标定 {l2_bw} GB/s")
        elif l2_hit > 0:
            out["l2_us"] = unknown(
                f"本芯片没有 l2_GBps 标定, 而本调度有 {l2_hit/1e9:.2f} GB 请求命中 L2 —— "
                f"**这部分在下界里被当成零代价**。要变成 measured: 用 core/probe 的带宽臂"
                f"把工作集压到 L2 容量以内 (< {(sp.get('cap') or {}).get('L2', 0)/1e6:.0f} MB) "
                f"重测一条曲线, 取平台段的值", "us")
        else:
            out["l2_us"] = derived(0.0, "us", "本调度没有命中 L2 的重复请求")

        #  ---- 向量 (逐 stage 折算波量化)
        elem = sum(st.elem_ops * self.stage_times[st.name]
                   for st in self.stages if st.unit == VEC)
        instr = sum(st.instr * self.stage_times[st.name]
                    for st in self.stages if st.unit == VEC)
        out["vector_instr"] = derived(instr, "条", "各 VEC stage 的 instr x 执行次数")

        def _vec_us(cy_per_instr):
            """Σ_stage (算术周期 + 指令条数 x 单价) / 核数 / 频率 / 该 stage 的 wave_eff。"""
            tot = 0.0
            for st in self.stages:
                if st.unit != VEC:
                    continue
                n = self.stage_times[st.name]
                cy = st.elem_ops * n * 4 / vec_B_cy + st.instr * n * (cy_per_instr or 0)
                e = _eff(st, vec_cores) or 1.0
                tot += cy / vec_cores / (ghz * 1e9) * 1e6 / e
            return tot

        if not vec_B_cy:
            #  ★取不到 vec_calc_B_per_cycle 就**不给数**, 不用 128 兜底 (见上面那条注释)。
            why = ("平台 ini 里取不到 vec_calc_B_per_cycle, 向量算术时间无法计算。"
                   "要变成 measured: 从本 SoC 的 ini 读 VectorCoreSpec 的计算带宽, "
                   "或用 core/probe 的向量臂在一个纯算术核上实测每周期字节数")
            out["vector_us_throughput_only"] = unknown(why, "us")
            out["vector_us_with_issue"] = unknown(why, "us")
        else:
            out["vector_us_throughput_only"] = derived(
                _vec_us(0), "us",
                f"{elem:,} 元素 x 4B / {vec_B_cy}B每周期 / {vec_cores}核, "
                f"**逐 stage** 各按自己的 wave_eff 折算; 只算算术")
            if issue_cycles_range is None:
                from .. import socspec
                c = socspec.calib_value("vec_instr_overhead_cycles", sp.get("soc"))
                if c:
                    out["vector_us_with_issue"] = measured(
                        _vec_us(c), "us",
                        f"{instr:,} 条 x {c} 周期/条 (chip_calib 实测 2026-09-22) + 算术")
                else:
                    lo, hi = 4, 24
                    out["vector_us_with_issue"] = default(
                        _vec_us(hi), f"取不到本芯片的 vec_instr_overhead_cycles 标定, 用 "
                                     f"{hi} 周期/条的猜测上界; 要变成 measured 需在两种分块"
                                     f"宽度下各测一次解出它", "us")
            else:
                lo, hi = issue_cycles_range
                out["vector_us_with_issue"] = default(
                    _vec_us(hi), f"调用方显式给了区间 {lo}~{hi} 周期/条, 这里取上界", "us")

        #  ---- 标量 (IR8). ★2026-09-22 采纳外部评审第九条并由实测坐实:
        #  本模型此前**根本没有标量这一项**, 而 msprof 实测 SFA 的 PV 核标量忙时比 96.4%、
        #  矩阵只有 1.8%。也就是说: 该算子的实际瓶颈资源在这套下界里没有对应的量, 于是
        #  "实测比下界高 21 倍"从来不是余量, 而是**缺项**——IR6 早写着"实测高于 sum = 模型
        #  缺项", 缺的就是这一项。现在它显形: 调度声明 scalar_ops, 取不到单价就报 unknown
        #  并进 bound_missing, 让每一条界都明说"标量没算进来"。
        sc_ops = sum(st.scalar_ops * self.stage_times[st.name] for st in self.stages)
        out["scalar_ops"] = derived(sc_ops, "条", "各 stage 的 scalar_ops x 执行次数")
        if sc_ops:
            from .. import socspec as _ss2
            sc_cy = _ss2.calib_value("scalar_op_cycles", sp.get("soc"))
            if sc_cy:
                out["scalar_us"] = measured(
                    sc_ops * sc_cy / (ghz * 1e9) * 1e6, "us", f"{sc_cy} 周期/条")
            else:
                out["scalar_us"] = unknown(
                    f"本调度有 {sc_ops:,} 条标量操作, 而本芯片没有 scalar_op_cycles 标定 —— "
                    f"**这一项没有计入下面两条界**。要变成 measured: 对同一个核做两档标量"
                    f"操作数 (例如内层循环轮数减半) 各跑一次 msprof, 由 aic_scalar_ratio x "
                    f"耗时的差解出每条的周期数", "us")
        else:
            out["scalar_us"] = default(
                0.0, "调度没有声明 scalar_ops —— 按 0 计。★这不是'没有标量开销', 是"
                     "'没描述'; SFA 的 PV 核实测标量忙时比 96.4%, 不声明就等于漏掉主项")

        #  ---- 两条参考线。★2026-09-22 采纳外部评审第三、四条改名:
        #  它们**都不是数学意义上的界**。max 那条依赖若干乐观假设 (工作集小于 L2 就判
        #  驻留 L2、40.8 周期/条是标定参数而非物理常数), 所以叫"理想重叠地板"而不是下界;
        #  sum 那条只是各资源服务时间之和, 而真实耗时还要加依赖阻塞 / 资源冲突 / 流水气泡
        #  / 发射延迟, 所以实测超过它**不违反任何数学上界**, 它只说明这套资源服务模型
        #  覆盖不住实测。原来的名字 bound_serial_us ("全串行上界") 是错的。
        _KEYS = ("cube_us", "dram_us", "l2_us", "vector_us_with_issue", "scalar_us")
        _terms = {k: out[k].value for k in _KEYS
                  if isinstance(out.get(k), Qty) and isinstance(out[k].value, (int, float))}
        if _terms:
            out["ideal_overlap_floor_us"] = derived(
                max(_terms.values()), "us",
                "理想重叠地板 (estimate, 非 hard bound): 各资源互不阻塞时的最大单项")
            out["service_sum_us"] = derived(
                sum(_terms.values()), "us",
                "资源服务时间之和 (estimate): 各资源都不重叠时的服务总量。"
                "★实测高于它 = 本资源服务模型有**未建模残差** (依赖阻塞/冲突/气泡/发射延迟), "
                "不是数学上界被突破, 也不是可优化余量")
            _miss = [k for k in _KEYS
                     if isinstance(out.get(k), Qty) and out[k].value is None]
            if _miss:
                #  ★取不到值的项**不会**进 max/sum, 于是双侧界会静默偏低。必须点名。
                out["bound_missing"] = unknown(
                    "这些项取不到值, 因此**没有**计入上面两条界, 两条界都偏低: "
                    + ", ".join(_miss), "项")
        return out

    def verdict(self, measured_us, sp, dram_GBps=None):
        """把一次实测放进双侧界里判读。**这是模型自己的证伪判据**。"""
        lb = self.lower_bounds(sp, dram_GBps)
        mx = lb.get("ideal_overlap_floor_us")
        sr = lb.get("service_sum_us")
        if not (mx and sr and mx.value and sr.value):
            return "取不到双侧界 —— 本判读不适用"
        if measured_us < mx.value * 0.95:
            return (f"实测 {measured_us:.0f}us **低于**理想重叠地板 {mx.value:.0f}us "
                    f"—— 这条线是 estimate 不是 hard bound, 所以低于它不是奇迹, "
                    f"是模型高估了某一项; 先查模型再谈优化")
        if measured_us <= sr.value:
            r = measured_us / mx.value
            return (f"实测 {measured_us:.0f}us 落在 [{mx.value:.0f}, {sr.value:.0f}] 之间 "
                    f"(离理想重叠 {r:.2f}x) —— 资源服务模型覆盖住了, 差距是**重叠质量**, 可优化")
        return (f"★实测 {measured_us:.0f}us **高于资源服务时间之和** {sr.value:.0f}us "
                f"({measured_us / sr.value:.1f}x) —— 注意这**不是**突破了什么上界 "
                f"(真实耗时 = 服务时间 + 依赖阻塞 + 资源冲突 + 流水气泡 + 发射延迟), "
                f"而是本资源服务模型有未建模残差。这不是优化空间, 是**模型里缺项**。"
                f"先用 msprof 的 PipeUtilization + ResourceConflictRatio 找出缺的量 "
                f"(等待比往往就是缺的那一项), 再谈改哪里")


#  ---- 负例控制 (门禁 G1) ----
#  ★这一段 2026-09-22 被我自己整体替换 lower_bounds 时误删过一次 (切片一直切到 def report,
#    把夹在中间的负例函数连带删了), 而我当时没有复跑门禁。是 G1 把它们报成"还没有负例"
#    才发现的 —— 门禁如实报覆盖缺口而不是静默通过, 这条价值在这里兑现了一次。
def _nc_capacity():
    """IR1 的负例: 明显放不下的调度必须被报出 ratio > 1; 且生命期必须被尊重。"""
    assert _nc_capacity_needs_lifetime()
    s = Schedule("nc_cap")
    s.buf("too_big", UB, 1024 * 1024)               # 1 MB 进 248 KB 的 UB
    rows = s.capacity({UB: 253952})
    return any(sp == UB and r is not None and r > 1 for sp, _u, _c, r, _e in rows)


def _nc_traffic_reuse():
    """IR2 的负例: 按张量大小求和会把重复读取整个漏掉, 两个口径必须给出不同的数。"""
    s = Schedule("nc_traffic")
    s.move("k", GM, L1, 2 * 1024 * 1024, 1024,
           unique=2 * 1024 * 1024, reuse_window=4 * 1024 * 1024)
    dram, req, _ = s.gm_traffic(l2_bytes=117 * 1024 * 1024)
    return req == 1024 * dram and dram == 2 * 1024 * 1024


def _nc_instr_marginal_cost():
    """IR4 的负例: 有本机标定时下界必须用它, 而不是退回猜的区间。"""
    from .. import socspec
    sp = socspec.spec("ascend950pr_957b")
    s = Schedule("nc_instr")
    s.stage("v", VEC, elem_ops=1000, instr=1000)
    q = s.lower_bounds(sp)["vector_us_with_issue"]
    return q.src == "measured" and not q.fallback


def _nc_wave_quantization_outer():
    assert _nc_wave_eff_is_stage_local()
    return _nc_wave_quantization()


def _nc_wave_quantization():
    """IR5 的负例: 任务数不是核数整数倍时必须报出 wave_eff < 1, 不得默认 1.0。"""
    e = wave_eff(8, 28)
    return e is not None and abs(e - 8 / 28) < 1e-12 and wave_eff(56, 28) == 1.0


def _nc_two_sided():
    """IR6 的负例: 高于全串行上界的实测必须判成"模型缺项", 不能报成"还有空间"。"""
    from .. import socspec
    sp = socspec.spec("ascend950pr_957b")
    s = Schedule("nc_two")
    s.stage("c", CUBE, macs=10 ** 9, tasks=28)
    s.stage("v", VEC, elem_ops=10 ** 6, instr=10 ** 4, tasks=56)
    sr = s.lower_bounds(sp, dram_GBps=1116)["service_sum_us"].value
    return ("模型里缺项" in s.verdict(sr * 10, sp, 1116)
            and "重叠质量" in s.verdict(sr * 0.9, sp, 1116))


def _nc_l2_hits_cost_time():
    """IR7 的负例: 一个请求量远大于 DRAM 流量的调度, 必须报出 L2 命中量且不得当零代价。"""
    s = Schedule("nc_l2")
    s.move("k", GM, L1, 2 * 1024 * 1024, 1024,
           unique=2 * 1024 * 1024, reuse_window=4 * 1024 * 1024)
    from .. import socspec
    sp = socspec.spec("ascend950pr_957b")
    lb = s.lower_bounds(sp, dram_GBps=1116)
    hit = lb["l2_hit_bytes"].value
    assert hit > 2e9, f"L2 命中量没报出来: {hit}"
    q = lb["l2_us"]
    #  没标定时必须是 unknown 且被 bound_missing 点名, 不能是 0 也不能悄悄不出现
    assert q.value is None and q.fallback, f"l2_us 应为回落, 实为 {q}"
    assert "bound_missing" in lb and "l2_us" in lb["bound_missing"].note
    return True


def _nc_scalar_is_a_resource():
    """IR8 的负例: 声明了 scalar_ops 而没有单价标定时, 必须报 unknown 并点名进缺项。"""
    from .. import socspec
    sp = socspec.spec("ascend950pr_957b")
    s = Schedule("nc_scalar")
    s.stage("pv", CUBE, macs=10**9, tasks=28, scalar_ops=10**7)
    lb = s.lower_bounds(sp, dram_GBps=1116)
    assert lb["scalar_ops"].value == 10**7
    assert lb["scalar_us"].value is None and lb["scalar_us"].fallback
    assert "scalar_us" in lb["bound_missing"].note, lb["bound_missing"].note
    #  不声明 scalar_ops 时按 0 计, 但必须明说"没描述 != 没有"
    s2 = Schedule("nc_scalar0")
    s2.stage("pv", CUBE, macs=10**9, tasks=28)
    q = s2.lower_bounds(sp, dram_GBps=1116)["scalar_us"]
    assert q.fallback and "没描述" in q.note
    return True


def _nc_wave_eff_is_stage_local():
    """IR5 的补充负例: 一个小而低效的 stage 不得把大而高效的 stage 一起罚。

    两个 cube stage: A 工作量 1 单位 / 任务数 29 (效率 0.518), B 工作量 100 单位 /
    任务数 28 (效率 1.0)。全局取最差的做法会给出约 195 单位, 逐 stage 应约 102。
    """
    from .. import socspec
    sp = socspec.spec("ascend950pr_957b")
    s = Schedule("nc_wave")
    s.stage("small", CUBE, macs=10**9, tasks=29)
    s.stage("big", CUBE, macs=100 * 10**9, tasks=28)
    v = s.lower_bounds(sp, dram_GBps=1116)["cube_us"].value
    per_stage = (10**9 / wave_eff(29, 28) + 100 * 10**9 / wave_eff(28, 28))
    worst = (101 * 10**9) / wave_eff(29, 28)
    peak = sp["cube_cores"] * sp["mac_per_cycle"]["fp16"] * sp["freq_ghz"] * 1e9
    assert abs(v - per_stage / peak * 1e6) < 1e-6, (v, per_stage / peak * 1e6)
    assert v < worst / peak * 1e6 * 0.9, "还在用全局最差效率罚所有 stage"
    return True


def _nc_capacity_needs_lifetime():
    """IR1 的补充负例: 生命期不相交的两块缓冲不得被无条件相加。"""
    s = Schedule("nc_life")
    s.stage("a", VEC, elem_ops=1, tasks=1)
    s.stage("b", VEC, elem_ops=1, tasks=1)
    s.buf("x", UB, 200 * 1024, lifetime=("a",))
    s.buf("y", UB, 200 * 1024, lifetime=("b",))
    rows = s.capacity({UB: 253952})
    peak = [r for r in rows if r[0] == UB][0]
    assert peak[1] == 200 * 1024, f"逐时刻峰值算错: {peak[1]}"
    assert peak[4] is True, "全部声明了 lifetime 却没标成 exact"
    #  不声明 lifetime 时按全程存活, 且必须标成非 exact
    s2 = Schedule("nc_life2")
    s2.stage("a", VEC, elem_ops=1, tasks=1)
    s2.buf("x", UB, 200 * 1024)
    s2.buf("y", UB, 200 * 1024)
    r2 = [r for r in s2.capacity({UB: 253952}) if r[0] == UB][0]
    assert r2[1] == 400 * 1024 and r2[4] is False
    return True


NEGATIVE_CONTROLS = {
    "IR7_l2_hits_cost_time": _nc_l2_hits_cost_time,
    "IR8_scalar_is_a_resource": _nc_scalar_is_a_resource,
    "IR1_capacity_before_build": _nc_capacity,
    "IR2_traffic_counts_reuse": _nc_traffic_reuse,
    "IR4_instr_marginal_cost": _nc_instr_marginal_cost,
    "IR5_wave_quantization": _nc_wave_quantization_outer,
    "IR6_two_sided_bounds": _nc_two_sided,
}


def report(sch, sp, caps_dict, dram_GBps=None):
    """把一个调度的四类判据打成人能读的表。"""
    print(f"=== 调度 {sch.name}" + (f" —— {sch.note}" if sch.note else ""))
    cap = sp["cap"]
    if sch.requires:
        ok, rows = sch.check_caps(caps_dict)
        print("\n[0] 依赖的芯片机制 (能力即类型)")
        for cid, mark, ev in rows:
            print(f"  [{mark:>6}] {cid:<24} {ev[:96]}")
        if ok is False:
            print("  ⇒ ★该调度在这颗芯片上**不成立** —— 缺的是机制, 不是调参能补的。")
        elif ok is None:
            print("  ⇒ ★有能力探不到: 本模块**拒绝**对该调度下结论 (既不算通过也不算不通过)。")
        else:
            print("  ⇒ 机制齐备。")
    print("\n[1] 各级存储 (每核)")
    over = []
    for space, used, capb, ratio, _exact in sch.capacity(cap):
        if capb in (None, 0):
            print(f"  {space:<4} 用 {used:>9,} B   容量未知 —— 判不了")
            continue
        flag = "★超" if ratio > 1 else ("紧" if ratio > 0.9 else "")
        print(f"  {space:<4} 用 {used:>9,} B / {capb:>9,} B = {ratio*100:5.1f}% {flag}")
        if ratio > 1:
            over.append(space)
    l2 = cap.get("L2")
    dram_tot, req_tot, rows = sch.gm_traffic(l2)
    print(f"\n[2] 流量: 下 DRAM {dram_tot/1e9:.3f} GB   核发出的请求量 {req_tot/1e9:.3f} GB"
          f"   (差额命中 L2 {l2/1e6:.0f} MB)")
    print(f"  {'搬运':<20} {'':7} {'DRAM':>9} {'请求':>9}   依据")
    for name, s_, d_, dr, req, why in rows[:9]:
        print(f"  {name:<20} {s_}->{d_:<3} {dr/1e9:8.3f} {req/1e9:8.3f} GB   {why}")
    ratio = caps_dict.get("mix_aic_to_aiv_ratio", {}).get("value")
    edges, need = sch.sync_edges(ratio if isinstance(ratio, int) else 1)
    print(f"\n[3] 跨单元交接 {len(edges)} 条边"
          f" (1 矩阵核配 {ratio} 向量核 -> 每条边每方向发 {ratio} 个 flag)")
    for p, c, nm, space in edges:
        print(f"  {p} -> {c}  经 {nm} ({space})")
    if edges:
        print(f"  需要: 正向 {need['forward_flags']} 个 flag + 反向 {need['backward_flags']} 个 "
              f"(反向是'缓冲可覆写'的额度, 漏了第二轮就挂住)")
        if any(self_space == GM for *_x, self_space in edges):
            pass
    lb = sch.lower_bounds(sp, dram_GBps)
    print("\n[4] 资源服务时间 (estimate, 非 hard bound) —— 每个量带来源, 回落一律标 ★")
    nfb = report_all(lb, order=["wave_eff_cube", "wave_eff_vec", "cube_us", "dram_us",
                                "l2_hit_bytes", "l2_us",
                                "vector_us_throughput_only", "vector_instr",
                                "vector_us_with_issue", "scalar_ops", "scalar_us",
                                "ideal_overlap_floor_us", "service_sum_us",
                                "bound_missing"], width=26)
    cand = [lb[k].value for k in ("cube_us", "dram_us", "vector_us_throughput_only")
            if lb.get(k) is not None and lb[k].value is not None]
    dom = max(cand, default=0)
    print(f"  {'⇒ 不含指令开销的下界':<26}{dom:>12,.1f} us")
    if nfb:
        print(f"  ★本表有 {nfb} 个量是**回落**得来的 —— 它们不是测出来的, 上面已逐条说明缺什么。")
    if over:
        _ex = all(r[4] for r in sch.capacity(cap))
        if _ex:
            print(f"\n★该调度在 {', '.join(over)} 上**放不下** —— 逐时刻峰值已超, 不必构建。")
        else:
            print(f"\n★该调度在 {', '.join(over)} 上, **按'所有缓冲全程存活'的假设**放不下。"
                  f"\n  注意方向性: 和小于容量能证明放得下, 和大于容量**证明不了放不下** ——"
                  f"生命期不相交的缓冲可以复用同一片空间。要把这条变成真判据,"
                  f"给 Buf 声明 lifetime。")
    return 1 if over else 0
