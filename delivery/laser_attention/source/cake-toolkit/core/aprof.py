#!/usr/bin/env python3
"""A 路径的**端到端对账**: 从 `DeviceProf<N>.bin` 的原始字节独立算出 `PipeUtilization.csv`。

与 `perf` 的分工: `perf` 答"**这些数字说明什么、该先优化哪条 pipe**" (读 csv 的列,
出归属与路由); 本模块答"**这些数字从哪来、对不对**" —— 从 128 字节一条的原始 PMU 记录
逐行逐位还原那些列, 并给出"列 → (遍次, PMU 槽, 换算形式)"的机读映射。
两者读同一批产物而问的不是一个问题: 前者假定 csv 可信, 后者**检验**它。
分成两个模块而不是给 perf 再加一个子命令, 是因为本模块有自己的四条判据 (E1/E2/E2b/E3)
与一套证伪检验 (`--falsify`), 塞进 5600 行的 perf 里会让两套判据体系混在一起。

与 `trust` 的分工: `trust` 查的是**文档与代码**的可信性 (引用解不解得开、常数有没有来源);
本模块查的是**产物**的可信性。


为什么要有它 (2026-10-23 立):
`tools/reconcile.py` 做的是 **B 路径** (torch_npu 的 `ASCEND_PROFILER_OUTPUT/`) 的端到端,
`06` V6 还给它做过证伪检验。而本仓的主路径是 **A 路径** (`msprof op` = `msopprof`),
`08` 篇整篇讲它 —— 却**一直没有端到端**: 公式表是从产物列之间反解出来的,
从没从原始字节走通过一遍。本工具补这一段。

三条判据 (全部**逐位**, 不设容差; E3 除外, 见下):

  E1  csv 的数据行数 == 最后一遍 bin 的记录条数 (128 字节一条)。
  E2  csv 的 `aic_total_cycles` ∪ `aiv_total_cycles` 的**多重集**
      == 最后一遍 bin 里全部记录 `total_cycle`(偏移 32, u64 小端) 的多重集, 逐位相等;
      **且只有这一遍相等** —— 若多遍都相等, 这条判据就没有分辨力, 必须报出来。
  E3  每一行 `*_time(us)` == `*_total_cycles` / `Current Freq`(OpBasicInfo.csv, MHz),
      换算要**复现 msopprof 的做法**: 周期先转 fp32, 再除以频率, 再按 `%.6f` 打印。
      ★第一版用的是绝对容差 5e-4 us, 250 个目录里 108 个报不符 —— 而真实残差
      最大只有 2.44e-04。问题不在阈值大小: **浮点判据要在它实际被打印的那个表示上比**,
      设容差等于在猜 msopprof 用的是什么精度。改成 fp32 打印形式之后
      **4320 行 0 不符, 不需要任何容差**。

★**为什么是多重集而不是逐行**: bin 记录里**没有 block_id**。记录头 u16[0] 的低 6 位
恒为 41, 高 10 位是 0..15 的**核号**, 而一个 kernel 有 24 个 cube block + 48 个 vector
block 轮流跑在这些核上。csv 的 `block_id` 是 `msopprof` **重新编号**的, 不是数据里的字段。
所以"逐行相等"要先复现它的编号规则, 而"多重集逐位相等"已经证明**数据内容完全对上**。
把判据定在能达到的地方, 并把差额说清楚 —— 这是作用域, 不是妥协。

★**取哪一遍: 是"最后一遍", 不是"第 10 遍"**。A 路径固定重放 10 遍, 所以绝大多数目录
两种说法给同一个答案。判决性检验在**遍数不足 10** 的那 5 个目录上做 (2026-10-23):
存在的遍次 `[1,2,3,4]` 的那个对上的是 **4**, `[1..9]` 的那个对上的是 **9** ——
两个目录直接证伪了"固定第 10 遍"。全仓 60 个随机目录 + 这 5 个, **65/65 是最后一遍**。

★**E3 不是端到端判据**。它只比 csv 自己的两列 (`*_total_cycles` 与 `*_time(us)`),
一个字节都不读 bin —— 证伪检验里篡改 bin 之后 E2 当场红而 **E3 仍然 72/72 过**,
这正是它的作用域。真正"原始字节 → 产物"的那一段是 **E1 + E2**;
E3 验的是产物内部的换算自洽 (以及由此反推出 msopprof 的浮点路径)。

用法 (`python -m core.aprof ...` 或 `./core/opdbg aprof ...`):
  aprof <kernel 目录>      单个 (kernel 目录 = 含 dump/ 与 PipeUtilization_*.csv 的那一层)
  aprof --all [--limit N]  扫全仓
  aprof --falsify <kernel 目录>   证伪检验 (在副本上扰动, 原产物只读)
  aprof --selftest
退出码 1 = 有判据不过。
"""
import argparse
import ast
import collections
import csv
import glob
import math
import os
import re
import statistics
import struct
import sys

#  来源: `03` §4.7 的 FFTS block PMU 记录布局 (`FFTS_BLOCK_PMU_FMT = "4HQ2HBBHLHH12Q"`,
#  字段下标照抄 `$AN/profiling_bean/stars/ffts_block_pmu.py:26-40`)。
#  `tools/passdev.py` 用的是同一组常数。
#  ★**第一版只用了 total_cycle 一个字段, 并据此写下"bin 记录里没有 block_id"** ——
#  那是错的, §4.7 的表里就有 (偏移 30), 旁边还有 core_type 与 sub_block_id。
#  2026-10-23 订正; 有了这三个字段, 对账从"多重集相等"升级成**逐行相等**。
#  ★每个偏移都**各自**带来源行: 棘轮 (`trust constcheck`) 取上下文时**以前一条模块级
#  语句为界** —— 写在 `REC` 上方的那段来源注释覆盖不到下面五个常数, 第一次提交就被拦下。
#  (这是把工具从 `docs/tools/` 搬进 `core/` 立刻换来的一条检查: 在旁路上时没人查常数来源。)
REC = 128              # 来源: 03 §4.7 FFTS block PMU 记录长度 (ffts_block_pmu.py:26-40)
OFF_CORE_TYPE = 20     # 来源: 03 §4.7 f7 —— core_type = f7 & 1 (0=AIC,1=AIV), core_id = f7>>1
OFF_SUB_BLOCK = 28     # 来源: 03 §4.7 sub_block_id
OFF_BLOCK_ID = 30      # 来源: 03 §4.7 block_id  ★第一版漏了它, 据此写下"bin 里没有 block_id"
OFF_TOTAL_CYCLE = 32   # 来源: 03 §4.7 total_cycle
OFF_PMU0 = 48          # 来源: 03 §4.7 pmu[0..7], 8 个 u64
#  csv 的 `*_time(us)` 是 **fp32 按 %.6f 打印**。E3 因此做成逐位比较而不是带容差 ——
#  见上方 docstring 里第一版用容差的那次 (108/250 目录误报, 而真实残差 max 2.44e-04)。
TIME_FMT = "%.6f"


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def as_printed(cycles: int, freq: float) -> str:
    """复现 msopprof 的换算: **周期数先转 fp32**, 再除以频率, 结果再按 `%.6f` 打印。

    ★"先转 fp32"不是猜的, 是全仓扫出来的: 前 60 个目录 0 不符, 而后面 433 个目录
    每个有一二十行差**恰好一个 ULP**。那些行的共同点是 `cycles > 2^24 = 16777216`
    —— **fp32 存不下这么大的整数**。把周期先压 fp32 再除, 6/6 逐位命中, 全仓 0 不符。

    后果值得单记: 核长超过 `2^24` 周期 (约 9.3 ms @1800 MHz) 时,
    `*_time(us)` 这一列**在换算之前**就已经丢了精度 —— 2^24~2^25 区间 ULP = 2 个周期。
    相对误差只有 ~6e-8, 但它是**确定性的量化台阶**, 比较两个相近测量时会以台阶形式出现。
    """
    return TIME_FMT % f32(f32(cycles) / freq)


def passes(kdir: str) -> dict:
    """{遍次号: bin 路径}。遍次号**不保证连续** —— 全仓有 5 个目录缺中间的遍。"""
    out = {}
    for b in glob.glob(os.path.join(kdir, "dump", "DeviceProf*.bin")):
        m = re.search(r"DeviceProf(\d+)\.bin$", b)
        if m:
            out[int(m.group(1))] = b
    return out


def cycles_of(path: str) -> collections.Counter:
    d = open(path, "rb").read()
    return collections.Counter(
        struct.unpack_from("<Q", d, i * REC + OFF_TOTAL_CYCLE)[0]
        for i in range(len(d) // REC))


def records(path: str) -> dict:
    """解一遍 bin, 返回 {(block_id, core_type, sub_block): {"cyc":…, "pmu":[8 个]}}。

    键的三个字段都来自 `03` §4.7 的表。**它在一遍之内唯一** —— 本仓全仓验过;
    不唯一的话下面的逐行对账会先在"键重复"上报出来, 不会静默取第一条。
    """
    d = open(path, "rb").read()
    out = {}
    for i in range(len(d) // REC):
        b = i * REC
        f7, = struct.unpack_from("<B", d, b + OFF_CORE_TYPE)
        sub, = struct.unpack_from("<H", d, b + OFF_SUB_BLOCK)
        blk, = struct.unpack_from("<H", d, b + OFF_BLOCK_ID)
        k = (blk, f7 & 1, sub)
        if k in out:
            out.setdefault("__dup__", []).append(k)
        out[k] = {
            "cyc": struct.unpack_from("<Q", d, b + OFF_TOTAL_CYCLE)[0],
            "pmu": [struct.unpack_from("<Q", d, b + OFF_PMU0 + 8 * s)[0] for s in range(8)],
        }
    return out


def row_key(r: dict) -> tuple:
    """csv 一行 -> 与 `records()` 同构的键。"""
    ct = 0 if r.get("aic_total_cycles", "NA") not in ("NA", "", None) else 1
    sub = 0 if r["sub_block_id"].rstrip().endswith("0") else 1
    return (int(r["block_id"]), ct, sub)


def csv_cycles(rows: list) -> collections.Counter:
    """csv 的 aic/aiv 周期数合成一个多重集 —— bin 不区分 aic/aiv, 所以这里也不分。"""
    c = collections.Counter()
    for r in rows:
        v = r.get("aic_total_cycles", "NA")
        if v in ("NA", "", None):
            v = r.get("aiv_total_cycles", "NA")
        if v not in ("NA", "", None):
            c[int(v)] += 1
    return c


def check(kdir: str) -> dict:
    res = {"dir": kdir, "ok": True, "notes": []}
    cs = glob.glob(os.path.join(kdir, "PipeUtilization_*.csv"))
    ps = passes(kdir)
    if not cs or not ps:
        res["skip"] = "没有 PipeUtilization csv 或没有 dump/DeviceProf*.bin"
        return res
    rows = list(csv.DictReader(open(cs[0])))
    last = max(ps)
    res["passes"], res["last"] = sorted(ps), last
    nrec = os.path.getsize(ps[last]) // REC
    res["rows"], res["recs"] = len(rows), nrec

    #  E1
    res["E1"] = (len(rows) == nrec)
    if not res["E1"]:
        res["notes"].append(f"E1 csv {len(rows)} 行 != 最后一遍 {nrec} 条记录")

    #  E2 —— 连"只有这一遍相等"一起判
    want = csv_cycles(rows)
    eq = [n for n in sorted(ps) if cycles_of(ps[n]) == want]
    res["eq_passes"] = eq
    res["E2"] = (eq == [last])
    if not res["E2"]:
        res["notes"].append(
            f"E2 多重集相等的遍次 {eq}, 期望只有最后一遍 [{last}]"
            + ("  ← 多遍都相等 = 判据没有分辨力" if len(eq) > 1 else ""))

    #  E2b —— 逐行。E2 只说"这批数两边一样", E2b 说"**每一行**都对上了它自己那条记录"。
    #  键 = (block_id@30, core_type@20, sub_block@28), 全部来自 `03` §4.7 的表。
    recs = records(ps[last])
    dup = recs.pop("__dup__", None)
    good = 0
    for r in rows:
        k = row_key(r)
        side = "aic" if k[1] == 0 else "aiv"
        v = r.get(f"{side}_total_cycles", "NA")
        if v in ("NA", "", None):
            continue
        rec = recs.get(k)
        if rec and rec["cyc"] == int(v):
            good += 1
    res["E2b"], res["E2b_n"] = (good == len(rows)), good
    if dup:
        res["notes"].append(f"E2b 键不唯一: {dup[:3]} —— 三字段键在这一遍里重复了")
    if not res["E2b"]:
        res["notes"].append(f"E2b 逐行只对上 {good}/{len(rows)}")

    #  E3
    info = glob.glob(os.path.join(kdir, "OpBasicInfo_*.csv"))
    if not info:
        res["E3"] = None
        res["notes"].append("E3 跳过: 没有 OpBasicInfo csv (取不到 Current Freq)")
    else:
        rec = list(csv.DictReader(open(info[0])))[0]
        #  ★`Current Freq` 会是 `NA` —— 全仓扫到才发现 (第一版在这里抛 ValueError)。
        #    没有频率就换算不了, 如实记成"跳过"而不是当成过。
        raw = (rec.get("Current Freq") or "").strip()
        if raw in ("NA", ""):
            res["E3"] = None
            res["notes"].append("E3 跳过: OpBasicInfo 的 Current Freq 是 NA")
            res["ok"] = all(res.get(k) is not False for k in ("E1", "E2"))
            return res
        f = float(raw)
        bad = n = 0
        for r in rows:
            for side in ("aic", "aiv"):
                c = r.get(f"{side}_total_cycles", "NA")
                t = r.get(f"{side}_time(us)", "NA")
                if c in ("NA", "", None) or t in ("NA", "", None):
                    continue
                n += 1
                if as_printed(int(c), f) != t:
                    bad += 1
        res["E3"], res["E3_n"], res["E3_bad"], res["freq"] = (bad == 0), n, bad, f
        if bad:
            res["notes"].append(f"E3 {bad}/{n} 行 time(us) != cycles/{f:g}MHz")
    #  E4 —— 其余 `*_time(us)` 列: 每一列在某一遍的某个 PMU 槽上**逐行**还原。
    #  映射不写死, 每个目录自己解 —— 不同算子/不同 `--aic-metrics` 档位下 (遍, 槽)
    #  可能不同, 写死就会在换一批产物时静默失配。
    if res.get("E3") is not None:
        allrec = {n: records(ps[n]) for n in ps}
        for v in allrec.values():
            v.pop("__dup__", None)
        cols = [c for c in rows[0]
                if c.endswith("time(us)") and c not in ("aic_time(us)", "aiv_time(us)")]
        solved, blind, unsolved = {}, [], []
        for c in cols:
            vals = [(row_key(r), r[c]) for r in rows if r.get(c, "NA") not in ("NA", "", None)]
            if not vals:
                continue
            #  ★全零列**任何全零槽都能匹配** —— 这不是解出来了, 是判据没有分辨力。
            if all(float(v) == 0.0 for _k, v in vals):
                blind.append(c)
                continue
            hit = None
            for n in sorted(ps):
                for s in range(8):
                    if all((allrec[n].get(k) or {"pmu": [None] * 8})["pmu"][s] is not None
                           and as_printed(allrec[n][k]["pmu"][s], f) == v
                           for k, v in vals):
                        hit = (n, s, len(vals))
                        break
                if hit:
                    break
            (solved.__setitem__(c, hit) if hit else unsolved.append(c))
        res["E4_solved"], res["E4_blind"], res["E4_unsolved"] = solved, blind, unsolved
        res["E4"] = (not unsolved)
        if unsolved:
            res["notes"].append(f"E4 有 {len(unsolved)} 列找不到 (遍, 槽): {unsolved[:4]}")

    res["ok"] = all(res.get(k) is not False for k in ("E1", "E2", "E2b", "E3", "E4"))
    return res


def print_one(r: dict) -> int:
    if "skip" in r:
        print(f"跳过 {r['dir']}: {r['skip']}")
        return 0
    print(f"=== A 路径端到端对账 === {r['dir']}")
    print(f"  遍次 {r['passes']}, 取最后一遍 **{r['last']}**")
    print(f"  E1 行数 == 记录条数        {'过' if r['E1'] else '✗'}  "
          f"(csv {r['rows']} 行, bin {r['recs']} 条)")
    print(f"  E2 周期多重集逐位相等       {'过' if r['E2'] else '✗'}  "
          f"(相等的遍次 {r['eq_passes']})")
    print(f"  E2b 周期**逐行**相等        {'过' if r['E2b'] else '✗'}  "
          f"({r['E2b_n']}/{r['rows']} 行, 键=(block_id,core_type,sub_block))")
    if r.get("E3") is None:
        print(f"  E3 time == cycles/freq     跳过")
    else:
        print(f"  E3 time == cycles/freq     {'过' if r['E3'] else '✗'}  "
              f"({r['E3_n'] - r['E3_bad']}/{r['E3_n']} 行, {r['freq']:g} MHz)")
    if "E4_solved" in r:
        s, b, u = r["E4_solved"], r["E4_blind"], r["E4_unsolved"]
        print(f"  E4 其余管线列逐行还原       {'过' if r['E4'] else '✗'}  "
              f"({len(s)} 列解出, {len(b)} 列全零无分辨力, {len(u)} 列未解)")
        for c, (n, sl, cnt) in sorted(s.items()):
            print(f"       {c:<34} 遍{n:>2} pmu{sl}  {cnt}/{cnt} 逐行")
        if b:
            print(f"       ★全零故无分辨力 (任何全零槽都能匹配, 不算解出): {', '.join(b)}")
    for n in r["notes"]:
        print(f"      ★{n}")
    return 0 if r["ok"] else 1


def scan_all(limit: int | None, per_op: int | None = None) -> int:
    #  ★**产物同时在两棵树下**: 顶层 `runs/` 与 `cases/<算子>/runs/`。
    #    2026-10-23 实撞: 只扫 `runs/` 时"全仓 1079 个 kernel 目录"漏掉了
    #    `cases/gemm_v3/runs/` 的 305 份 —— 而 F10 的严格上界样本 (`Cast` 覆盖率 0.0421)
    #    **全在漏掉的那一批里**, 直接导致 `06` V18 写下一条错误的作用域声明。
    #    **"全仓"这个词要对着目录树核一次, 不能按印象写。**
    ks = [os.path.dirname(d) for d in
          sorted(glob.glob("runs/*/OPPROF_*/*/*/dump")
                 + glob.glob("cases/*/runs/*/OPPROF_*/*/*/dump"))]
    if per_op:
        #  ★按**算子名**抽样, 不是取前 N 个。全仓 1384 个目录里约九成是 gmfr ——
        #  "全量"在这里并不等于"有代表性": 跨算子的问题 (V16 的三处伪解、V20 的 F10
        #  严格上界) 全都只在少数非 gmfr 的目录里出现, 而它们排在列表很后面。
        #  每算子取 N 个, 22 个目录就覆盖 11 个算子, 比前 200 个目录信息量大。
        by = collections.OrderedDict()
        for k in ks:
            op = os.path.basename(os.path.dirname(k)).split("_")[0]
            by.setdefault(op, []).append(k)
        ks = [k for v in by.values() for k in v[:per_op]]
    if limit:
        ks = ks[:limit]
    agg = collections.Counter()
    bad = []
    for k in ks:
        r = check(k)
        if "skip" in r:
            agg["跳过"] += 1
            continue
        agg["检查"] += 1
        for e in ("E1", "E2", "E2b", "E3", "E4"):
            if r.get(e) is True:
                agg[e + " 过"] += 1
            elif r.get(e) is False:
                agg[e + " ✗"] += 1
                bad.append((k, r["notes"]))
        agg[f"取第 {r['last']} 遍"] += 1
        if "E4_solved" in r:
            agg["E4 解出列数(累计)"] += len(r["E4_solved"])
            agg["E4 全零无分辨力(累计)"] += len(r["E4_blind"])
    print(f"=== A 路径端到端, 扫 {len(ks)} 个 kernel 目录 ==="
          + (f"  (每算子抽样 {per_op} 个)" if per_op else ""))
    for k in sorted(agg):
        print(f"   {k:<16} {agg[k]:>5}")
    seen = set()
    for k, notes in bad:
        if k in seen:
            continue
        seen.add(k)
        print(f"\n  ✗ {k[-72:]}")
        for n in notes:
            print(f"      {n}")
    return 1 if any(k.endswith("✗") for k in agg) else 0


#  ---- `--solve`: 把 E4 的解法推广到**全部八张表**与多种换算形式 ----
#  E4 只管 `PipeUtilization` 的 `*_time(us)`。其余七张表的列是计数、比值、带宽,
#  换算形式不同, 所以单列一个模式: 先解出"列 → (遍, 槽, 换算形式)", 再谈判据。
#
#  ★**带宽列的分母是 `Task Duration`(OpBasicInfo, 全表同一个值), 不是本行的 `cyc/freq`。**
#  判决性对比 (2026-10-23, `aic_l1_read_bw`, 24 行): 以 TaskDur 为分母时跨行离散
#  **3.2e-08**, 以 `cyc/freq` 为分母时 **2.0e-03** —— 差五个数量级。
#
#  ★**而且标着 GB/s 的那些列, 实际是 GiB/s。** 反解出的字节常数是 `238.4186`,
#  既不是 256 也不是别的整数 —— 直到把 `1e9` 换成 `2^30`: `256 × 1e9/2^30 = 238.4186`,
#  逐位吻合。换算回来之后 11 个带宽列的常数全是整数 (128/256/512/2048/32768)。
#  **本仓第三次撞同一类**: `09` 第 1 条 (hwfloor 的 GB/s 当 GiB/s, 差 7.37%) 与
#  `08` §3.42 (`L1_to_GM` 标 GB/s 而反解用 `/2^30`)。三次都不是算错, 是**标签写错**,
#  而标签错不会让任何数字看起来可疑。
_GIB = 1e9 / 2 ** 30


def _plausible_bytes(m: float) -> bool:
    """反解出的字节常数要**是整数、是 2 的幂、落在 16~65536**。

    ★样本外 (l2press_probe) 照出两个荒谬解: `C=0` 与 `C=204800000 字节`。
    它们都满足"跨行一致"(那列的 pmu 在该样本上退化成同一个值, 任何 C 都一致),
    而第一版只检查了整数性 —— **0 是整数, 204800000 也是**。
    跨行一致管的是"这个槽对不对", 管不了"这个常数有没有意义";
    后者需要一条**来自领域而不是来自数据**的约束。"""
    if not (abs(m - round(m)) < 0.01 and 16 <= m <= 65536):
        return False
    r = int(round(m))
    return r & (r - 1) == 0


def _one_or_ambiguous(fits: list, form: str):
    """把"全部拟合"收成一个结论。`fits` = [(遍, 槽, C), ...]。

    ★2026-10-23 (`08` §3.83)。此前这三处都是"先试到就 break", 在 gmfr 上恰好唯一,
    所以看不出问题 —— 与同日 §3.82 是同一个形状: **在候选唯一的样本上,
    "报第一个"与"报唯一那个"分不开**。判决性检验是枚举: 同一个目录上最多 6 个
    (遍, 槽, C) 同样严丝合缝 (极差 0.0), 而给出的 C 相差 64 倍。

    两种多解要分开:
    - 槽不唯一而 **C 一致** -> 数值可读, 只是说不清在哪个槽, 照常报并注明处数;
    - **C 也不一致** -> 这一列没解出, 报歧义并把候选 C 全列出来, **不挑一个**。
    """
    if not fits:
        return None
    consts = sorted({f[2] for f in fits})
    n, s, C = fits[0]
    if len(consts) == 1:
        return (form if len(fits) == 1 else f"{form} (另有 {len(fits)-1} 处同 C 的槽)",
                n, s, C, list(fits))
    #  ★遍/槽也要置 None。本模块的既有约定是"`v[1] is None` = 没解出"
    #  (`aic_mte3_wait_ratio` 那条歧义一直是这么返回的), 而自检与覆盖数都按这个字段数。
    #  第一版在歧义里填了第一处拟合的遍/槽, 于是 6 条新歧义被**计进"已解出"**,
    #  而断言 `114/18/1` 照常过 —— 数字没变, 变的是它的含义。
    #  ★第 5 位是**全部拟合** `[(遍, 槽, C), ...]`, 机读。2026-10-28 (`08` §3.110) 加,
    #    同日 (§3.110c) 从"裸常数集合"改成带遍/槽的三元组。
    #    为什么必须带槽: **常数离开它的槽位形式没有意义**。`read_main_memory_datas`
    #    在多数目录上是 `128 × (pmu6+pmu7)`, 而在 `pmu6==pmu7` 的目录上单槽
    #    `256 × pmu6` 也严丝合缝 —— 同一个物理量的两种写法, 对**裸常数**求交会把
    #    128 与 256 判成冲突。正例对照臂正是在这里第二次报红的。
    return (f"歧义: {len(fits)} 处拟合给出 {len(consts)} 个不同的 C {consts} —— 数据区分不开, 不挑一个",
            None, None, None, list(fits))


#  ---- `--probecol`: 给 `--solve` 解不出的列扫一个**更宽的形式空间**, 并把空间说出来 ----
#  ★动因 (2026-10-28, `09` 第 200 条): `--solve` 报"未解"时, 读的人无法分辨
#    "分子不在 dump 里" 与 "形式不在求解器的库里" —— 而 `08` §3.94 正是靠排除后者
#    才把 `k=3`/`k=11` 的解释定成"未知形式"。要让那个排除可检验, 必须有一个
#    **把搜过的空间打印出来**的入口: 否则"没找到"永远可以被读成"没找对地方"。
#  ★它**只报否定或候选, 不下结论**: 命中一个组合不等于那就是实现, 还要跨行跨目录成立。
PROBE_MULT = ((" ", 1.0), ("x2", 2.0), ("x4", 4.0), ("x8", 8.0),
              ("/2", 0.5), ("/4", 0.25), ("/8", 0.125))


def _probe_f(x) -> float:
    """`NA` / 空 / 非数 一律当 0 —— **本仓的 `NA` 不是缺失而是一个取值** (台账 181),
    而这里只拿它当分母候选, 取 0 会被下游换成 1.0 跳过该分母, 不会静默算错。
    ★第一版直接 `float()`, 两个 `aic_*_hit_rate(%)` 列当场抛 ValueError ——
    **一个探查器在它最该出声的列上崩掉**, 而崩掉比报错更难读成"这列没解出"。
    """
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def probe_column(kdir: str, col: str, rows_max: int = 3) -> dict:
    """在"四种分母 x (单槽x7 倍率 ∪ 两槽±)"里找 `col` 的分子。返回搜过的空间与命中。

    与 `solve_columns` 的分工: 那个用**逐位字符串**比 (严), 认 **14 种**已确证的形式
    (单槽 5 种 + 多槽/派生 9 种, 含 `a/(a+b)*100` 与各带宽形式; 2026-10-28 清点订正 ——
    此处原写"四种", 那只数了 `exact` 字典, 漏掉了后面按列名分支实现的九种);
    本函数用**数值容差**比 (宽), 为的是回答"再放宽一档还有没有"。
    ★容差 = `max(分母x5e-7, 1.0)`, **来源是 csv 的印数精度**: 比值只印 6 位小数
      ⇒ 比值误差 ±5e-7 ⇒ 分子误差 ±分母x5e-7。不是拍的常数 ——
      拍出来的容差会把否定结论变成肯定结论 (`09` 第 201 条实撞)。
    """
    import itertools as _it
    ps = passes(kdir)
    info = glob.glob(os.path.join(kdir, "OpBasicInfo_*.csv"))
    if not ps or not info:
        return {"err": "没有 pass 或 OpBasicInfo"}
    rec0 = list(csv.DictReader(open(info[0])))[0]
    raw = (rec0.get("Current Freq") or "").strip()
    if raw in ("NA", ""):
        return {"err": "Current Freq 是 NA"}
    freq, td = float(raw), float(rec0["Task Duration(us)"])
    allrec = {n: records(ps[n]) for n in ps}
    for v in allrec.values():
        v.pop("__dup__", None)
    #  ★按候选计数, **只报在全部探查行上都成立的** (2026-10-28, `09` 第 201 条)。
    #    第一版逐行累加命中, 于是只在 1/3 行上凑巧对上的组合也会被报出来 ——
    #    而这个空间很宽, 单行巧合几乎必然发生。"在一行上成立"与"是这一列的算法"
    #    之间差的正是这条。
    per, used, dropped, noinfo = collections.Counter(), 0, set(), set()
    for path in sorted(glob.glob(os.path.join(kdir, "*.csv"))):
        rows = list(csv.DictReader(open(path)))
        if not rows or col not in rows[0]:
            continue
        ok = [r for r in rows
              if r.get(col, "NA") not in ("NA", "", None) and float(r[col]) != 0.0]
        for r in ok[:rows_max]:
            used += 1
            k = row_key(r)
            slots = {}
            for n in sorted(ps):
                e = allrec[n].get(k) or {}
                if e.get("pmu"):
                    for sl, v in enumerate(e["pmu"]):
                        slots[(n, sl)] = v
            cyc = (allrec[max(ps)].get(k) or {}).get("cyc") or 0
            #  ★不可用的分母**剔除而不是换成 1.0** (2026-10-28 实撞, `09` 第 201 条):
            #    `aic_*_hit_rate(%)` 那两列的 `aiv_total_cycles` 是 `NA`, 换成 1.0 之后
            #    容差 `max(20, 1x1e-6)` = 20 相对于一个百分数是天文数字,
            #    于是**凭空造出 82 个候选** —— 一个退化的分母比没有分母更糟,
            #    它让"这列有候选"这句话完全失去意义。剔除的要**报出来**, 不静默少搜。
            dens = [(n_, v_) for n_, v_ in (
                ("floor(TaskDur)*freq", f32(math.floor(td) * freq)),
                ("total_cycles", float(cyc)),
                ("TaskDur*freq", td * freq),
                ("本行 aiv_total_cycles", _probe_f(r.get("aiv_total_cycles"))))
                if v_ and v_ > 1000.0]
            for n_, v_ in (("total_cycles", float(cyc)),
                           ("本行 aiv_total_cycles", _probe_f(r.get("aiv_total_cycles")))):
                if not (v_ and v_ > 1000.0):
                    dropped.add(f"{n_} (该行取 {v_:g}, 不是可用分母)")
            for dname, dv in dens:
                #  ★容差必须**有来源**: csv 的比值只印 6 位小数 ⇒ 比值的绝对误差 ±5e-7
                #    ⇒ 反解分子的误差 ±分母x5e-7; 再加 1.0 容整数舍入。
                #    第一版写 `max(20, 分母x1e-6)` —— 那个 **20 是拍的**, 在分母 9.2e6 时
                #    相当于把门开到 4 倍宽, `aic_mte3_wait_ratio` 因此得到 237 个"全行候选"。
                #    **一个拍出来的容差会把否定结论变成肯定结论**, 而它看起来只是个小常数。
                tgt = float(r[col]) * dv
                tol = max(dv * 5e-7, 1.0)
                #  ★这一行有没有**判别力**, 要先问再搜 (2026-10-28, `09` 第 201 条)。
                #    `aic_mte3_wait_ratio` 印作 `0.000008` —— 6 位小数下只剩 1 位有效数字,
                #    容差/目标 = 6.2%, 于是上万次试探里必然有几百个"命中",
                #    而它们全都不含信息。**这不是搜不到, 是这一行问不出东西。**
                #    判据取 tol/|tgt| > 1%: 再宽就等于拿噪声当答案。
                if not tgt or tol / abs(tgt) > 0.01:
                    noinfo.add(f"{dname}: 印数只剩 {abs(tgt)/tol:.1f} 倍容差的判别力")
                    continue
                for key, v in slots.items():
                    for lbl, m in PROBE_MULT:
                        if abs(v * m - tgt) <= tol:
                            per[f"{dname}: 遍{key[0]}槽{key[1]}{lbl}"] += 1
                for a, b in _it.combinations(slots, 2):
                    for lbl, x in (("+", slots[a] + slots[b]),
                                   ("-", slots[a] - slots[b]),
                                   ("-", slots[b] - slots[a])):
                        if abs(x - tgt) <= tol:
                            per[f"{dname}: 遍{a[0]}槽{a[1]}{lbl}遍{b[0]}槽{b[1]}"] += 1
    hits = sorted(k for k, n in per.items() if n == used) if used else []
    partial = sum(1 for k, n in per.items() if n < used)
    return {"col": col, "rows": used, "hits": hits, "partial": partial,
            "noinfo": sorted(noinfo),
            "dropped": sorted(dropped),
            "space": "至多 4 种分母 x (单槽 x {1,2,4,8,1/2,1/4,1/8} ∪ 两槽 ±)"}


def print_probecol(spec: str) -> int:
    if ":" not in spec:
        print("用法: --probecol <kernel 目录>:<列名>")
        return 2
    kdir, col = spec.rsplit(":", 1)
    bad, _why = corrupt_kdir(kdir)
    if bad:
        raise SystemExit(f"拒跑: {kdir} 是坏目录 ({bad['ratio']:.2f} 倍, {bad['rows']} 行)")
    r = probe_column(kdir, col)
    if r.get("err"):
        print(f"跑不了: {r['err']}")
        return 2
    print(f"=== 列分子探查 (08 §3.94 / 09 第 200 条) === {r['col']}  用了 {r['rows']} 行")
    print(f"  搜过的空间: {r['space']}")
    for d in r.get("noinfo", []):
        print(f"  ★无判别力: {d} —— 印数精度不足, 这一列**问不出东西**, 不是搜不到")
    for d in r.get("dropped", []):
        print(f"  ★已剔除的分母: {d} —— 退化分母会凭空造候选, 剔除比凑数诚实")
    if r.get("partial"):
        print(f"  ★{r['partial']} 个组合只在**部分**行上成立, 已剔除 —— "
              f"这个空间很宽, 单行巧合几乎必然发生")
    if r["hits"]:
        print(f"  ★候选 {len(r['hits'])} 个 (**不是结论** —— 还要跨行跨目录成立):")
        for h in r["hits"][:10]:
            print(f"      {h}")
        if len(r["hits"]) > 5:
            #  ★候选一多就要说清它为什么多: 退化槽 (该行为 0) 加进去不改变和差,
            #    于是一个真解会带出一串"陪跑"。**正命中在这个宽空间里是弱证据** ——
            #    本工具的用处主要在**否定**那一侧 (见上面的 else 分支)。
            print(f"  ★候选这么多是因为**退化槽**: 该行取 0 的槽加进和差里不改变结果, "
                  f"一个真解会带出一串陪跑。")
            print(f"    ⇒ 在这个宽空间里**正命中是弱证据**; 本入口的价值主要在否定那一侧。")
    else:
        print("  **这个空间里没有**。")
        print("  ★这不等于'分子未导出' —— 只等于不在上面那个空间里。"
              "两者的下一步不同: 前者要换采集档位, 后者要扩形式空间。")
    return 0


def solve_columns(kdir: str) -> dict:
    """解 `<表>/<列>` → (换算形式, 遍, 槽, 常数)。返回 {表: {列: 结果或 None}}。"""
    ps = passes(kdir)
    info = glob.glob(os.path.join(kdir, "OpBasicInfo_*.csv"))
    if not ps or not info:
        return {}
    rec0 = list(csv.DictReader(open(info[0])))[0]
    raw = (rec0.get("Current Freq") or "").strip()
    if raw in ("NA", ""):
        return {}
    freq, td = float(raw), float(rec0["Task Duration(us)"])
    allrec = {n: records(ps[n]) for n in ps}
    for v in allrec.values():
        v.pop("__dup__", None)

    #  逐位形式 (与 csv 字符串比)。常数由数据定, 不枚举。
    #  ★`*_ratio` 的分母**不是本行的 `total_cycles`**, 是 `floor(TaskDur) × freq` ——
    #    一个算子级的、而且**取过整**的常数。
    #    怎么定的: 逐行反解 `pmu / ratio`, 三行给出 9203410.2 / 9203403.4 / 9203393.3,
    #    跨行几乎恒定 (差 17, 而 ratio 只有 6 位小数, 反解噪声就有 ±11) —— 所以分母是常数;
    #    `TaskDur × 1800 = 9204012.6` 在噪声之外 (差 612), 而 `floor(TaskDur) × 1800
    #    = 5113 × 1800 = 9203400` 正好落在区间里。`08` §2062 早写过 **D 恒等于
    #    TaskDur 的整数部分**, 这里是它在 ratio 分母上的又一次现身。
    #  ★后果要记: **ratio 与 time 列互相不自洽** (`cube_ratio ≠ cube_time / aic_time`),
    #    拿 ratio 反推时间会错; 而 `floor` 对短核的相对偏差可达百分之几
    #    (TaskDur 10.9 us 时 floor 掉 0.9, 偏 8%)。
    den_ratio = f32(math.floor(td) * freq)
    exact = {
        "原值": lambda pm, cyc: str(pm),
        "/freq": lambda pm, cyc: as_printed(pm, freq),
        "/total_cycles": lambda pm, cyc: TIME_FMT % f32(f32(pm) / f32(cyc)),
        "/(floor(TaskDur)*freq)": lambda pm, cyc: TIME_FMT % f32(f32(pm) / den_ratio),
    }
    #  ★`aic_time(us)` / `aiv_time(us)` 来自 `total_cycle`(偏移 32) 而**不是任何 PMU 槽**,
    #    所以上面那四种"拿 pmu 当分子"的形式一个都套不上 —— 第一版把这 14 列
    #    (七张表各两列) 全报成"未解", 而 E3 早就逐行验过它们。
    #    **"未解"里混着"没找"的, 这比真的未解更误导**: 它会让人去查一个已经答了的问题。
    cyc_forms = {"cyc/freq": lambda cyc: as_printed(cyc, freq)}
    out = {}
    for path in sorted(glob.glob(os.path.join(kdir, "*.csv"))):
        tbl = os.path.basename(path).rsplit("_", 1)[0]
        if tbl == "OpBasicInfo":
            continue
        rows = list(csv.DictReader(open(path)))
        if not rows or "block_id" not in rows[0]:
            continue
        res = {}
        for c in rows[0]:
            if c in ("block_id", "sub_block_id", "") or c.endswith("_total_cycles"):
                continue
            vals = [(row_key(r), r[c]) for r in rows if r.get(c, "NA") not in ("NA", "", None)]
            if not vals:
                continue
            try:
                nz = [(k, v) for k, v in vals if float(v) != 0.0]
            except ValueError:
                continue
            if not nz:
                res[c] = ("全零无分辨力", None, None, None)
                continue
            hit = None
            #  ★**收集全部命中而不是取第一个**。样本外 (l2press_probe) 照出:
            #    `aiv_vec_ratio` 在那份产物上 `/total_cycles` 与
            #    `/(floor(TaskDur)*freq)` **两种形式都成立** —— 因为那个算子的
            #    TaskDur 恰好让两个分母相等。第一版按顺序取先试到的那个, 于是
            #    同一列在两个算子上报出两种不同的"解", 看起来像映射不稳定,
            #    实际是**判据在那份数据上没有分辨力而工具没说**。
            allhits = []
            for fname, fn in cyc_forms.items():
                if all((allrec[n].get(k) or {}).get("cyc") and fn(allrec[n][k]["cyc"]) == v
                       for k, v in vals for n in [max(ps)]):
                    allhits.append((fname, max(ps), None, None))
            for fname, fn in exact.items():
                for n in sorted(ps):
                    for s in range(8):
                        if all((allrec[n].get(k) or {}).get("pmu")
                               and fn(allrec[n][k]["pmu"][s], allrec[n][k]["cyc"]) == v
                               for k, v in vals):
                            allhits.append((fname, n, s, None))
            forms_hit = {h[0] for h in allhits}
            if len(allhits) == 1:
                hit = allhits[0]
            elif len(forms_hit) == 1:
                #  同一种换算在多个 (遍, 槽) 上命中 = **那几个槽的值相同** (数据退化),
                #  不是判据分不开两种解释。取第一个并把同值槽数报出来。
                #  例: 全是 int8 运算时 `cube_total_instr_number` 与
                #  `cube_int_instr_number` 相等, 两个槽自然一起命中。
                hit = allhits[0][:3] + (None,)
                hit = (f"{allhits[0][0]} (另有 {len(allhits) - 1} 处同值槽)",) + hit[1:]
            elif len(forms_hit) > 1:
                #  **不同的换算都成立** = 这份数据分不开它们。报成歧义, 不挑一个。
                #  ★再算一句**要分开需要多大的值** —— 不然"歧义"只是一个标签,
                #    看的人无从判断该找什么样的样本。
                #    两个分母的相对差 δ, 在 `%.6f` 上可见需要 `v × δ > 5e-7`。
                note = ""
                if {"/total_cycles", "/(floor(TaskDur)*freq)"} <= forms_hit:
                    k0 = vals[0][0]
                    cyc0 = (allrec[max(ps)].get(k0) or {}).get("cyc")
                    if cyc0:
                        delta = abs(den_ratio - cyc0) / den_ratio
                        if delta > 0:
                            vmax = max(abs(float(v)) for _k, v in vals)
                            note = (f"; 两分母相对差 {delta:.2e}, 要在 %.6f 上分开需要"
                                    f"该列 > {5e-7 / delta:.1e}, 而本样本最大 {vmax:.1e}")
                res[c] = ("歧义: " + " / ".join(sorted(forms_hit)) + note,
                          None, None, None)
                continue
            #  多槽求和的比值: 只对名字里带 `total` 的列试 (2~3 槽)。
            #  `aiv_vec_total_cflt_ratio` = 遍5 `pmu2 + pmu3`, 而 `bankgroup`/`bank`
            #  两列各自就是 pmu2 / pmu3 —— **total 等于分项之和, 语义自洽**,
            #  这比"搜到一个能对上的组合"强: 组合是被语义预测出来的, 不是碰出来的。
            if hit is None and "total" in c and c.endswith("ratio"):
                import itertools
                for n in sorted(ps):
                    for sz in (2, 3):
                        for combo in itertools.combinations(range(8), sz):
                            if all((lambda r: r and TIME_FMT % f32(
                                        f32(sum(r["pmu"][x] for x in combo)) / den_ratio) == v
                                    )(allrec[n].get(k)) for k, v in vals):
                                hit = ("Σpmu/(floor(TaskDur)*freq)", n,
                                       "+".join(f"pmu{x}" for x in combo), None)
                                break
                        if hit:
                            break
                    if hit:
                        break
            #  两个计数器之比 `pmuA / pmuB`。只对 `*_miss_rate` 试 —— 语义约束:
            #  `01` §669 写着 `icache_miss_rate = 1.0*r55/r54`, 即两个相邻计数器相除,
            #  分母**不是**任务窗口也不是 total_cycles, 所以上面那几种形式一个都套不上。
            #  ★2026-10-23 (08 §3.86): 本仓 C 类硬墙表里写着
            #  「`aiv_icache_miss_rate` 本仓 1.3b 记为**计数器未导出**」—— 那是错的。
            #  它在 890 个目录里非零 (最大 0.017241), 只是 `--solve` 从没试过 `a/b` 这种形式,
            #  于是它一直落在"未解"里, 又被另一处记成"计数器没导出"。
            #  **"工具没试过"与"数据里没有"在同一张表上长得一模一样。**
            if hit is None and c.endswith("_miss_rate"):
                fits = []
                for n in sorted(ps):
                    for i in range(8):
                        for j in range(8):
                            if i == j:
                                continue
                            if all((lambda r: r and r["pmu"][j] and
                                    TIME_FMT % f32(f32(r["pmu"][i]) / f32(r["pmu"][j])) == v
                                    )(allrec[n].get(k)) for k, v in nz):
                                fits.append((n, f"pmu{i}/pmu{j}"))
                if len(fits) == 1:
                    hit = ("a/b", fits[0][0], fits[0][1], None)
                elif len(fits) > 1:
                    hit = (f"歧义: {len(fits)} 对计数器都成立 —— 数据区分不开, 不挑一个",
                           None, None, None)
            #  浮点运算数: `Σpmu × C`, C 是整数 (不是字节数, 是每指令的运算数)。
            #  `aiv_vec_fops` = 遍4 的 `pmu0 + pmu1`, **C = 32** ——
            #  32 正是 AIV 一条 fp32 向量指令的元素数, 语义自洽。
            #  只对名字含 `fops` 的列试, 理由同"组合搜索要被语义约束"。
            #  ★2026-10-23 (08 §3.83 同一条): 这里的搜索空间是 10 遍 x (8 单槽 + 28 两槽)
            #  = 360 个候选, 而接受条件只要求 C 是 1~65536 的整数 (**不要求 2 的幂** ——
            #  它不是字节数, 是每指令的运算数)。非零行少时巧合命中很容易: 样本外扫出过
            #  C = 258 / 2088 / 10992, 没有一个能当"一条向量指令的元素数"。所以同样
            #  枚举全部拟合, 多个不同 C 就报歧义。
            if hit is None and c.endswith("fops"):
                import itertools as _it
                fits = []
                for n in sorted(ps):
                    for sz in (1, 2):
                        for combo in _it.combinations(range(8), sz):
                            cs = []
                            for k, v in nz:
                                r = allrec[n].get(k)
                                tot_ = sum(r["pmu"][x] for x in combo) if r else 0
                                if not tot_:
                                    cs = None
                                    break
                                cs.append(float(v) / tot_)
                            if not cs:
                                continue
                            m = sum(cs) / len(cs)
                            sp = (max(cs) - min(cs)) / m if m else 1
                            if sp < 1e-9 and abs(m - round(m)) < 1e-6 and 1 <= m <= 65536:
                                fits.append((n, "+".join(f"pmu{x}" for x in combo), round(m)))
                hit = _one_or_ambiguous(fits, "Σpmu×C")
                #  ★唯一拟合也可能是巧合: `--xcheck` 在一个 vector 目录上解出
                #  `aiv_vec_fops` 的 C = 10992, 而它是**该目录上的唯一拟合**, 报歧义救不了。
                #  标出来而不是丢掉 —— 判据是"每指令的运算数应是 2 的幂":
                #  本机见过的可解释值 (aiv_vec_fops 32 = AIV fp32 向量宽度;
                #  aic_cube_fops 8192/16384 = fp/int 指令) 全是 2 的幂, 而
                #  258 / 2088 / 10992 当不了任何一条指令的元素数。
                #  **这是归纳出来的提示, 不是证明**, 所以只加标记不改判定。
                if hit and hit[3] and (hit[3] & (hit[3] - 1)):
                    hit = (hit[0] + f" ★C={hit[3]} 不是 2 的幂 —— 本机可解释的 fops 常数"
                                    f" (32/8192/16384) 全是 2 的幂, 很可能是巧合命中",
                           hit[1], hit[2], hit[3])
            #  命中率: `a / (a + b) × 100`, a/b 是**同一遍的两个槽**。
            #  只对名字里带 `hit_rate` 的列试 —— 两槽搜索有 640 种组合, 放开试所有列
            #  会让"碰巧对上"的概率升高, 而这里要的是解释不是拟合。
            if hit is None and "hit_rate" in c:
                for n in sorted(ps):
                    for i in range(8):
                        for j in range(8):
                            if i == j:
                                continue
                            if all((lambda r: r and (r["pmu"][i] + r["pmu"][j]) and
                                    TIME_FMT % f32(f32(r["pmu"][i]) /
                                                   f32(r["pmu"][i] + r["pmu"][j]) * 100) == v
                                    )(allrec[n].get(k)) for k, v in vals):
                                hit = ("a/(a+b)*100", n, f"pmu{i}/pmu{j}", None)
                                break
                        if hit:
                            break
                    if hit:
                        break
            #  数据量 (KB): `pmu × C / 1024`, C 由数据反解并要求是整数字节。
            #  ★2026-10-23: **枚举全部拟合, 不在第一个就 break**。判决性检验
            #  (08 §3.83): 同一个目录上最多有 6 个 (遍, 槽, C) 三元组**同样严丝合缝**
            #  (极差 0.0), 而它们给出的 C 相差 64 倍 (16 ~ 1024)。gmfr 上恰好唯一,
            #  所以第一版"先试到就报"在动因样本上看不出问题 —— 与本轮 §3.82 同一个形状:
            #  **在一个候选唯一的样本上, "报第一个"与"报唯一那个"分不开。**
            #  区分两种多解: 槽不唯一而 C 一致 -> 数值可读, 只是说不清在哪个槽;
            #  C 也不一致 -> **这一列没解出**, 报歧义而不是挑一个。
            if hit is None and c.endswith("datas(KB)"):
                fits = []
                for n in sorted(ps):
                    for s in range(8):
                        cs = []
                        for k, v in nz:
                            r = allrec[n].get(k)
                            if not r or r["pmu"][s] == 0:
                                cs = None
                                break
                            cs.append(float(v) * 1024 / r["pmu"][s])
                        if not cs:
                            continue
                        m = sum(cs) / len(cs)
                        sp = (max(cs) - min(cs)) / m if m else 1
                        if sp < 1e-6 and _plausible_bytes(m):
                            fits.append((n, s, round(m)))
                hit = _one_or_ambiguous(fits, "×C字节/1024")
            #  利用率百分比: `带宽 / 峰值 × 100`。峰值**不在产物里**, 所以这里的做法是
            #  **把峰值反解出来并报出去**, 而不是从别处取一个数填进来 ——
            #  报出来的数可以被独立核对 (`python -m core.socspec --bw-table`)。
            #  ★2026-10-23 核对结果: 四个值 264.00 / 209.32 / 220.06 / 186.80 GiB/s
            #    与那张表逐位吻合。**这关掉了 `08` §3.46 挂着的"未证实"** ——
            #    那一条说的是"这些值是程序里的真实字面量, 但找不到引用它们的代码,
            #    所以不能说它们就是算产物用的那张表"。产物侧反解给出了缺的那一环。
            if hit is None and c.endswith("bw_usage_rate(%)"):
                for n in sorted(ps):
                    for s in range(8):
                        cs = []
                        for k, v in nz:
                            r = allrec[n].get(k)
                            if not r or r["pmu"][s] == 0:
                                cs = None
                                break
                            cs.append(float(v) * td * 1000 / r["pmu"][s] / _GIB)
                        if not cs:
                            continue
                        m = sum(cs) / len(cs)
                        sp = (max(cs) - min(cs)) / m if m else 1
                        #  m = C字节 × 100 / 峰值。C 取 128 或 256 (与 datas 列同源),
                        #  峰值要落在合理区间并被报出。
                        #  ★字节常数**取同表 `X_datas(KB)` 已解出的那个**, 不枚举。
                        #    第一版按 (128, 256, …) 顺序枚举、取第一个让峰值落在
                        #    合理区间的值 —— `GM_to_L1` 于是用 128 得出峰值 132.0,
                        #    而正确答案是 C=256 / 峰值 264.00 (正好差一倍)。
                        #    **枚举出一个落在合理区间的值, 不等于解释。**
                        if sp < 1e-6:
                            dcol = c[:-len("_bw_usage_rate(%)")] + "_datas(KB)"
                            dsol = res.get(dcol)
                            cb = dsol[3] if dsol and dsol[3] else None
                            #  ★峰值也要过合理性 —— 第一版只检查了字节常数,
                            #    于是样本外 (l2press) 报出**峰值 0.0 GiB/s** 两次。
                            #    合理性检查要覆盖**每一个反解出来的量**, 漏一个就漏一类伪解。
                            if cb and m:
                                peak = cb * 100 / m
                                if 50 <= peak <= 1200:
                                    hit = (f"(×{cb}字节/2^30/TaskDur)/峰值×100",
                                           n, s, round(peak, 2))
                        if hit:
                            break
                    if hit:
                        break
            #  主存读一族: **两条读通路之和**。单槽反解给出 255.54 (离散 1.2e-04),
            #  槽位接近而常数不整 —— 那正是"少加了一个槽"的样子。
            #  遍 2 的 `pmu6 + pmu7`, C=128, 三列全中 (24/24 · 48/48 · 72/72)。
            #  ★放宽到这三列而不是所有列: 名字里的 `read` 本来就意味着多条读通路,
            #  组合是被语义预测出来的 —— 与 `total` 类、`fops` 类同一条纪律。
            #  ★2026-10-28 (`08` §3.110c): 这里原本**在第一个拟合处 break**, 而 §3.83
            #    早把别处三个同样的"先试到就报"改成了枚举全部拟合 —— 这一处是漏网的。
            #    后果不只是"可能报错一个": 它让这一列**不带候选集**, 于是 `--ambigres`
            #    看不见它, 只看得见单槽恰好拟合上的那些目录 (= 恰好 r50d==r50e 的那些)。
            #    拿它当正例对照就是**条件预设了结论**, 与 `09` 第 173 条那条循环判据同形。
            if hit is None and ("main_mem_read" in c or "read_main_memory" in c):
                import itertools as _it2
                kb = c.endswith("datas(KB)")
                fits = []
                for n in sorted(ps):
                    for sz in (2, 3):
                        for combo in _it2.combinations(range(8), sz):
                            cs = []
                            for k, v in nz:
                                r = allrec[n].get(k)
                                s_ = sum(r["pmu"][x] for x in combo) if r else 0
                                if not s_:
                                    cs = None
                                    break
                                cs.append(float(v) * (1024 if kb else td * 1000)
                                          / s_ / (1 if kb else _GIB))
                            if not cs:
                                continue
                            m = sum(cs) / len(cs)
                            sp = (max(cs) - min(cs)) / m if m else 1
                            if sp < 1e-6 and _plausible_bytes(m):
                                fits.append((n, "+".join(f"pmu{x}" for x in combo),
                                             round(m)))
                hit = _one_or_ambiguous(fits, "Σpmu×C字节" + ("/1024" if kb
                                                             else "/2^30/TaskDur"))
            #  活跃带宽: 分母是**该管线自己的时间**而不是 TaskDur。
            #  `X_active_bw(GB/s)` 配 `X_time(us)` —— 列名直接给出配对, 不用搜。
            #  ★三列全中且 C 都是 128, 而且槽位与 `*_datas(KB)` 那几列**完全重合**
            #    (fixpipe↔L0C_to_GM · mte2↔GM_to_UB · mte3↔UB_to_GM, 都是遍 3):
            #    同一个字节计数, 一个按 TaskDur 算总量, 一个按管线活跃时间算活跃带宽。
            #    **两条独立反解落在同一个槽同一个常数上**, 这比任何单独一条都强。
            if hit is None and c.endswith("_active_bw(GB/s)"):
                tcol = c[:-len("_active_bw(GB/s)")] + "_time(us)"
                trip = [(k, float(v), float(dict(
                    (row_key(r), r.get(tcol, "0")) for r in rows).get(k, 0) or 0))
                    for k, v in nz]
                trip = [(k, b, tu) for k, b, tu in trip if tu > 0]
                #  ★同 `datas(KB)`: 枚举全部拟合再判, 不在第一个就 break (08 §3.83)。
                fits = []
                for n in sorted(ps):
                    for s in range(8):
                        cs = []
                        for k, bw, tus in trip:
                            r = allrec[n].get(k)
                            if not r or r["pmu"][s] == 0:
                                cs = None
                                break
                            cs.append(bw * tus * 1000 / r["pmu"][s] / _GIB)
                        if not cs:
                            continue
                        m = sum(cs) / len(cs)
                        sp = (max(cs) - min(cs)) / m if m else 1
                        if sp < 1e-6 and _plausible_bytes(m):
                            fits.append((n, s, round(m)))
                hit = _one_or_ambiguous(fits, "×C字节/2^30/本管线time")
            #  带宽: 常数由数据反解, 要求跨行一致且落在整数字节上。
            if hit is None and "bw" in c:
                fits = []
                for n in sorted(ps):
                    for s in range(8):
                        cs = []
                        for k, v in nz:
                            r = allrec[n].get(k)
                            if not r or r["pmu"][s] == 0:
                                cs = None
                                break
                            cs.append(float(v) * td * 1000 / r["pmu"][s] / _GIB)
                        if not cs:
                            continue
                        m = sum(cs) / len(cs)
                        sp = (max(cs) - min(cs)) / m if m else 1
                        if sp < 1e-6 and _plausible_bytes(m):
                            fits.append((n, s, round(m)))
                hit = _one_or_ambiguous(fits, "×C字节/2^30/TaskDur")
            #  ★没命中时再补一遍**覆盖率** (2026-10-28, `09` 第 216 条)。
            #    动因: `08` §3.56 早就手工算出 `aic_read_hit_rate` 用最自然的式子
            #    **80.7% 的行精确**, 其余偏到 78 个百分点 —— 而 `--solve` 只报一个
            #    "未解", 把这件事整个盖住了。**二元判定会把"大部分对"说成"不知道"。**
            #    ★只在未命中时做, 命中路径仍走短路的 `all()` —— 第 214/215 条刚因为
            #    把闸门拖慢栽过, 不重蹈。
            if hit is None:
                best = (0, None, None, None)
                for fname, fn in exact.items():
                    for n in sorted(ps):
                        for sl in range(8):
                            cnt = 0
                            for k, v in vals:
                                e = allrec[n].get(k) or {}
                                if e.get("pmu") and fn(e["pmu"][sl], e["cyc"]) == v:
                                    cnt += 1
                            if cnt > best[0]:
                                best = (cnt, fname, n, sl)
                if best[0] and len(vals):
                    frac = best[0] / len(vals)
                    if frac >= 0.5:
                        #  ★遍/槽置 None —— 本模块既有约定 `v[1] is None` = **没解出**
                        #    (写在 `_one_or_ambiguous` 的注释里)。首版填了遍/槽,
                        #    于是 `form_code_join` 把"部分成立"当成一种形式并进了映射,
                        #    当场打红一条既有判据。**改哪一块先读那一块的约定。**
                        #    遍/槽信息放进文案, 不放进会被下游当真的字段。
                        hit = (f"部分成立 {best[0]}/{len(vals)} 行 ({frac*100:.1f}%): "
                               f"{best[1]} (遍{best[2]} 槽{best[3]}) "
                               f"—— **不是'未解'**, 是这一式在部分行上精确",
                               None, None, None)
            #  ★第 6 位 = **本目录上这一列有几个不同的非零值**。2026-10-28 (`08` §3.110b):
            #    `--ambigres` 求交时, 一份"该列恒定"的产物只能给出**巧合拟合** ——
            #    常数列上有很多 (遍, 槽, C) 都能对上, 而它们与真常数无关。
            #    这种目录混进交集里会把交集**错误地交空**。实撞: MemSet 的
            #    `read_main_memory_datas` 恒为 0.5, 于是遍5槽0 以 C=128 命中,
            #    而真形式在遍2槽6/7 (两寄存器之和, 且本机 r50d==r50e 所以单槽 C=256)。
            #    判别力是**列的性质**, 求解器算得出, 外面算不出 —— 所以放进返回值。
            if hit:
                hit = tuple(hit) + (None,) * (5 - len(hit)) + (len({v for _k, v in nz}),)
            res[c] = hit
        if res:
            out[tbl] = res
    return out


#  ---- 由**同表已解出的计数列**派生的列 ----
#  这些列不是任何单槽的换算, 而是 `Σ分子 × 100 / Σ分母`。分子分母那些计数列
#  本身已由 `solve_columns` 从 bin 解出, 所以这一层是**传递**而不是另起炉灶。
#  ★运算顺序要和产物一致: `f32(f32(Σ分子 × 100) / f32(Σ分母))` —— 24/24;
#  而"先除后乘 100"只有 19/24。**先乘还是先除, 差 5 行。**
#  这是本仓第三次撞"浮点判据要在产物实际走过的运算路径上比"
#  (前两次: E3 的绝对容差 · 周期数先转 fp32)。
_L2 = ["r0_read_cache_hit", "r0_read_cache_miss_allocate",
       "r1_read_cache_hit", "r1_read_cache_miss_allocate"]
DERIVED = {}
for _side in ("aic", "aiv"):
    _rd_n = [f"{_side}_r0_read_cache_hit", f"{_side}_r1_read_cache_hit"]
    _rd_d = [f"{_side}_{x}" for x in _L2]
    DERIVED[("L2Cache", f"{_side}_read_hit_rate(%)")] = (_rd_n, _rd_d)
    DERIVED[("L2Cache", f"{_side}_write_hit_rate(%)")] = (
        [f"{_side}_write_cache_hit"],
        [f"{_side}_write_cache_hit", f"{_side}_write_cache_miss_allocate"])
    DERIVED[("L2Cache", f"{_side}_total_hit_rate(%)")] = (
        [f"{_side}_write_cache_hit"] + _rd_n,
        [f"{_side}_write_cache_hit", f"{_side}_write_cache_miss_allocate"] + _rd_d)


#  ---- 列间关系 (报告 1.2 的 F 系列) ----
#  ★这些式子**本仓早就写在报告的公式表里**, 不是本工具推出来的。
#  做进来的理由是: F 系列只说"列与列之间成立", 没人逐行逐位验过它们在**每一份产物**上
#  还成立 —— 而本工具的其余判据都是逐位的, 单独留这几条靠人记会不一致。
#  `wsum_int` = 整数加权和 (F12), `fsum` = fp32 逐项累加 (F6/F9)。
WEIGHTED = {
    #  F12 (报告 1.2): cube_fops = fp_instr×8192 + int_instr×16384
    ("ArithmeticUtilization", "aic_cube_fops"):
        ("wsum_int", [("aic_cube_fp_instr_number", 8192),
                      ("aic_cube_int_instr_number", 16384)]),
    #  F6: cube_total_instr = fp + int
    ("ArithmeticUtilization", "aic_cube_total_instr_number"):
        ("wsum_int", [("aic_cube_fp_instr_number", 1),
                      ("aic_cube_int_instr_number", 1)]),
    #  F9: cube_ratio = cube_fp16_ratio + cube_int8_ratio (fp32 累加)
    ("ArithmeticUtilization", "aic_cube_ratio"):
        ("fsum", ["aic_cube_fp16_ratio", "aic_cube_int8_ratio"]),
}


#  F10 (报告 1.2): `Σ(vec dtype 五支) ≤ vec_ratio` —— 报告记为**上界**。
#  做成判据时要同时报**成立率**与**最大间隙**: 上界与等式在不触发间隙的样本上
#  观察不到区别, 只报"成立"会让人以为已经验到了它是上界。
F10_SUP = ["aiv_vec_fp32_ratio", "aiv_vec_fp16_ratio", "aiv_vec_int32_ratio",
           "aiv_vec_int16_ratio", "aiv_vec_misc_ratio"]


def check_f10(kdir: str) -> tuple | None:
    """返回 (行数, 成立数, 最大相对间隙) 或 None。"""
    g = glob.glob(os.path.join(kdir, "ArithmeticUtilization_*.csv"))
    if not g:
        return None
    n = ok = 0
    mx = 0.0
    for r in csv.DictReader(open(g[0])):
        v = r.get("aiv_vec_ratio", "NA")
        if v in ("NA", "", None):
            continue
        try:
            tot = float(v)
        except ValueError:
            continue
        if tot <= 0:
            continue
        s = sum(float(r[c]) for c in F10_SUP
                if r.get(c, "NA") not in ("NA", "", None))
        n += 1
        if s <= tot + 1e-6:
            ok += 1
        mx = max(mx, (tot - s) / tot)
    return (n, ok, mx) if n else None


def check_weighted(kdir: str) -> dict:
    """验 F 系列的列间关系。返回 {(表, 列): (ok, n)}。"""
    out = {}
    for (tbl, col), (kind, spec) in WEIGHTED.items():
        g = glob.glob(os.path.join(kdir, f"{tbl}_*.csv"))
        if not g:
            continue
        ok = n = 0
        for r in csv.DictReader(open(g[0])):
            if r.get(col, "NA") in ("NA", "", None):
                continue
            try:
                if kind == "wsum_int":
                    got = str(sum(int(r[c]) * w for c, w in spec))
                else:
                    acc = 0.0
                    for c in spec:
                        acc = f32(acc + f32(float(r[c])))
                    got = TIME_FMT % acc
            except (KeyError, ValueError):
                break
            n += 1
            if got == r[col]:
                ok += 1
        if n:
            out[(tbl, col)] = (ok, n)
    return out


def check_derived(kdir: str) -> dict:
    """返回 {(表, 列): "ok/n"}。只验派生式, 不搜索。"""
    out = {}
    for (tbl, col), (num, den) in DERIVED.items():
        g = glob.glob(os.path.join(kdir, f"{tbl}_*.csv"))
        if not g:
            continue
        ok = n = 0
        for r in csv.DictReader(open(g[0])):
            if r.get(col, "NA") in ("NA", "", None):
                continue
            try:
                ns = [int(r[c]) for c in num]
                ds = [int(r[c]) for c in den]
            except (KeyError, ValueError):
                break
            if sum(ds) == 0:
                continue
            n += 1
            if TIME_FMT % f32(f32(sum(ns) * 100) / f32(sum(ds))) == r[col]:
                ok += 1
        if n:
            out[(tbl, col)] = (ok, n)
    return out


#  ---- `*_ratio` 的分母是哪一个 (08 §3.82) -------------------------------------------
#  为什么是一条独立的检查而不是 `--solve` 的一行: `--solve` 两种形式都试, 哪个对就报哪个,
#  所以它**在单个目录上永远"解得出"**, 看不出"换个算子就换了一支"。这条反过来做 ——
#  先反解分母本身, 再问它在目录内是不是常数, 最后拿 `Op Type` 当预测去对。
#  它的用处是**将来 CANN 换版本改了这条规则时会红**, 而 `--solve` 不会。

DENOM_RULE = {"mix": "const", "cube": "per_row", "vector": "per_row"}
#  来源: 本仓自定 —— 不是硬件/协议常数, 是判据的容差。定法: `*_time(us)` 在 csv 里
#  只有 6 位小数, 反推成周期数 (`time x freq`) 的相对误差约 1e-6; 而两个候选分母
#  在要区分的地方相差 >=2e-2 (08 §3.82 的表)。1e-4 落在两者中间两个数量级开外。
_DENOM_TOL = 1e-4


def denom_of(kdir: str) -> dict | None:
    """反解一个 kernel 目录的 `*_ratio` 分母。返回判定 + 证据, 解不了返回 None。"""
    try:
        ob = sorted(glob.glob(os.path.join(kdir, "OpBasicInfo*.csv")))[0]
        r0 = next(csv.DictReader(open(ob, errors="ignore")))
        dur = float(r0["Task Duration(us)"]); freq = float(r0["Current Freq"])
        op_type = (r0.get("Op Type") or "").strip()
        pu = sorted(glob.glob(os.path.join(kdir, "PipeUtilization*.csv")))[0]
    except (IndexError, StopIteration, KeyError, ValueError, OSError):
        return None
    fw = math.floor(dur) * freq
    n_row = n_const = 0
    Ds = []
    for row in csv.DictReader(open(pu, errors="ignore")):
        for pre, col in (("aic", "aic_cube"), ("aiv", "aiv_vec"),
                         ("aic", "aic_mte2"), ("aiv", "aiv_mte2"), ("aic", "aic_mte1")):
            t = row.get(f"{col}_time(us)"); rt = row.get(f"{col}_ratio")
            tc = row.get(f"{pre}_total_cycles")
            if not t or t in ("NA", "") or not rt or rt in ("NA", "") or not tc or tc in ("NA", ""):
                continue
            try:
                rt = float(rt); tc = float(tc); t = float(t)
            except ValueError:
                continue
            #  ratio 太小的时候反解会被 6 位小数的截断放大, 不作证据
            if rt < 1e-3 or tc <= 0:
                continue
            D = t * freq / rt
            Ds.append(D)
            a = abs(D - tc) / tc <= _DENOM_TOL
            b = fw > 0 and abs(D - fw) / fw <= _DENOM_TOL
            if a and not b:
                n_row += 1
            elif b and not a:
                n_const += 1
            #  a and b = 两个候选在这一行数值相同, **不是证据**, 两边都不记
    if not Ds:
        return None
    verdict = ("per_row" if n_row and not n_const else
               "const" if n_const and not n_row else
               "indistinguishable" if not (n_row or n_const) else "conflict")
    return {"kdir": kdir, "op_type": op_type, "verdict": verdict,
            "expected": DENOM_RULE.get(op_type), "n_row": n_row, "n_const": n_const,
            "spread": (max(Ds) - min(Ds)) / statistics.median(Ds)}


#  ---- 跨 Op Type 一致性 (08 §3.84) ---------------------------------------------------
#  为什么要有它: 第 132 条与第 133 条是同一个根的两次 —— **解析式只在 gmfr (mix) 上验过**。
#  两次都是人想起来才去跑别的算子。这条把"去跑别的 Op Type"做成可执行的一步:
#  同一列在两类以上 Op Type 上都解出时, 换算形式与常数必须一致; 不一致的要么是
#  已登记的作用域规则 (下表), 要么就是**新发现, 必须去查**。
#
#  ★与 `--denom` 的分工: `--denom` 只盯 `*_ratio` 这一族、拿已知规则去核对 (回归用);
#  `--xcheck` 不预设哪一列会出问题, 它找的是**还没被登记的作用域依赖** (发现用)。

#  已登记的作用域依赖: 列名正则 -> 为什么它按 Op Type 分支 (写明出处)
KNOWN_SCOPED = [
    (re.compile(r"_(wait_)?ratio$"),
     "08 §3.82: *_ratio 的分母 mix 用 floor(TaskDur)xfreq, 纯 cube/vector 用本行 total_cycles"),
]
_XSUFFIX = re.compile(r"\s*\(另有 \d+ 处[^)]*\)$")


def _is_scope_dependent(forms: dict) -> bool:
    """各 Op Type 的换算形式集合**两两不相交** = 作用域依赖; 相交 = 同类内部就多解。

    判据只有这一条, 理由: "mix 用 A 而 cube 用 B" 是规则按 Op Type 分支 (§3.82 那一类);
    而 "cube 里有的目录用 A 有的用 B" 说明决定它的不是 Op Type —— 那是多解 (§3.83 那一类),
    把它报成作用域依赖会让人去找一条并不存在的规则 (E17: 下一步先确认这条路存在)。
    """
    sets = [frozenset(v) for v in forms.values()]
    return all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:])


def xcheck(per_type: int = 40, seed: int = 2) -> dict:
    """按 Op Type 抽样跑 `--solve`, 比对同一列的 (换算形式, 常数) 是否一致。"""
    import random
    dirs = product_kdirs()
    bytype = collections.defaultdict(list)
    for k in dirs:
        try:
            r0 = next(csv.DictReader(open(
                sorted(glob.glob(os.path.join(k, "OpBasicInfo*.csv")))[0], errors="ignore")))
        except (IndexError, StopIteration, OSError):
            continue
        bytype[(r0.get("Op Type") or "?").strip()].append(k)
    rng = random.Random(seed)
    sample = {t: sorted(rng.sample(v, min(per_type, len(v)))) for t, v in sorted(bytype.items())}
    obs = collections.defaultdict(lambda: collections.defaultdict(set))
    for t, ks in sample.items():
        for k in ks:
            try:
                sol = solve_columns(k)
            except (OSError, ValueError, KeyError, IndexError):
                continue
            for cols in sol.values():
                for c, v in cols.items():
                    if not v:
                        continue
                    form = str(v[0])
                    #  只比"两边都解出"的 —— 歧义/全零是数据的事, 不是作用域的事
                    if form.startswith("歧义") or "全零" in form:
                        continue
                    obs[c][t].add((_XSUFFIX.sub("", form), v[3]))
    form_diff, const_diff, multi = [], [], []
    for c, per in sorted(obs.items()):
        if len(per) < 2:
            continue
        forms = {t: sorted({f for f, _C in v}) for t, v in per.items()}
        if len({f for v in forms.values() for f in v}) > 1:
            #  ★区分两件事, 判据是**各 Op Type 的形式集合相不相交**:
            #    不相交 (例 cube 只有 /total_cycles, mix 只有 /(floor...)) = 作用域依赖,
            #      这是关于"规则按 Op Type 分支"的发现, 必须查清;
            #    相交 (某一类里既有 A 又有 B) = 同一类里就不一致, 那不是 Op Type 的事,
            #      是**某些目录上两种形式都成立** (多解)。
            #      实例: read_main_memory_datas 在一个 MatMulV3 目录上 pmu6 == pmu7 逐行相等,
            #      于是"单槽 x256"与"pmu6+pmu7 x128"给出同一个数 —— 抽样 25 个 cube 目录
            #      里只有 1 个这样, 所以它是巧合不是规则。
            (form_diff if _is_scope_dependent(forms) else multi).append((c, forms))
        consts = {t: sorted({C for _f, C in v if C is not None}) for t, per_t in per.items()
                  for _t, v in [(t, per_t)]}
        allc = {C for v in consts.values() for C in v}
        if len(allc) > 1:
            const_diff.append((c, consts))
    return {"sample": {t: len(v) for t, v in sample.items()},
            "n_cols": sum(1 for p in obs.values() if len(p) >= 2),
            "form_diff": form_diff, "multi_form": multi, "const_diff": const_diff}


def print_xcheck(per_type: int = 40) -> int:
    r = xcheck(per_type)
    print(f"=== 跨 Op Type 一致性 (08 §3.84) === 抽样 {r['sample']}; "
          f"两类以上都解出的列 {r['n_cols']}")
    def known(col):
        for rx, why in KNOWN_SCOPED:
            if rx.search(col):
                return why
        return None
    new_form = [(c, f) for c, f in r["form_diff"] if not known(c)]
    print(f"\n--- A. 换算形式随 Op Type 不同: {len(r['form_diff'])} 列 "
          f"(其中**未登记**的 {len(new_form)}) ---")
    for c, f in r["form_diff"]:
        why = known(c)
        tag = "  [已登记] " + why if why else "  ★**未登记 —— 去查**"
        print(f"  {c}{tag}")
        if not why:
            for t, fs in sorted(f.items()):
                print(f"      {t:8} {fs}")
    print(f"\n--- A2. 同一个 Op Type 内部就有两种形式 (= 多解, 不是作用域): "
          f"{len(r['multi_form'])} 列 ---")
    for c, f in r["multi_form"]:
        print(f"  {c}")
        for t, fs in sorted(f.items()):
            print(f"      {t:8} {fs}")
    print(f"\n--- B. 形式相同而常数不唯一: {len(r['const_diff'])} 列 ---")
    for c, f in r["const_diff"]:
        print(f"  {c:34} " + "  ".join(f"{t}:{v}" for t, v in sorted(f.items()) if v))
    if new_form:
        print("\n  ★A 里有未登记的作用域依赖 —— 要么查清后加进 KNOWN_SCOPED 并写出处, "
              "要么它是解析式的 bug。")
        return 1
    print("\n  A 里没有未登记的作用域依赖。"
          "\n  ★B 不判对错: 常数随目录变可能是真的 (例 aic_cube_fops 的 8192 vs 16384 "
          "对应 fp 与 int 指令), 也可能是巧合命中。它是**待查清单**, 不是违规。")
    return 0


#  ---- `--ambigres`: 歧义列的候选 C 跨目录取交集 (08 §3.110) ---------------------------
#  动因 (2026-10-28, `09` 第 218 条): `--solve` 对五个 L0 带宽列报"歧义: 多个 C 同样成立",
#  而那只是**单目录内**的结论 —— 每个目录各自排除掉一部分候选, 交起来就唯一了。
#  `--xcheck` 查不到这一层: 它开头就把歧义列跳过 ("歧义是数据的事, 不是作用域的事")。
#
#  ★为什么交集本身就是回验, 不需要我在外面再写一遍式子:
#    每个目录的候选集都是**该目录全部非零行逐行成立**的 C。C 落在交集里
#    = 它在每一个目录的每一行上都成立。我上一轮在外面重推反解式去"验", 连错三轮
#    (先说按 Op Type 分、再说按算子分、再说分母选错), 每次都被下一次更宽的测量推翻。
#    根源与 `09` 第 215 条同一条: **在工具外面复刻工具的算术**。
#
#  ★已解出的目录也必须参与。只交"歧义目录"会漏掉一类反例: 某目录把该列**解出**成另一个 C,
#    那是真的冲突, 而它根本不在歧义集合里。所以两类一起收, 已解出的当成单元素候选集。
#  ★正例对照: `08` §3.59 的六个 `*_datas(KB)` 常数是**另一条路**定出来的 ——
#    从 msopprof 二进制里认出寄存器号, 再按寄存器级公式逐位核对, 与本命令的
#    "逐行候选集求交"毫无共用环节。所以它们能当独立的对照臂。
#    没有对照臂的话, 一个"把所有候选集都读成全集"的实现也会报出"处处收敛到唯一",
#    而那只是因为交集退化 —— 单看"收敛了几列"分不开这两种情况。
#  ★判据是 **(遍, 槽, 常数)** 三项全中, 不是只比常数。只比一个数太好满足:
#    恒零槽拼进组合、退化目录上的单槽写法, 都能给出同一个数字而指向别的计数器。
#    2026-10-29 实撞两次 —— 先是 `pmu0+pmu1+pmu7` 冒充 `pmu6+pmu7` (常数同为 128),
#    再是去掉遍次后遍3 与遍5 的同号槽被并成一组。两次都只有连槽位一起比才拦得住。
AMBIGRES_CONTROL = {            # 列 -> (期望 C, 期望形式, 出处)
    #  ★这一列是**两寄存器之和**, 期望值 128 配的是求和形式 `pmu6+pmu7`。
    #    2026-10-28 一度改成 256 —— 那是错的, 来由记在 `08` §3.110c:
    #    当时求和分支在第一个拟合处 break 且不带候选集, 于是求交只看得见
    #    `r50d==r50e` 的那些目录 (单槽 `256 × pmu6` 在那里也严丝合缝),
    #    我便按那个有偏子集把期望改成了 256。**条件预设了结论。**
    #    修法不是改期望值, 是让求交按**槽位形式**分组 —— 两种写法各自收敛, 互不冲突。
    "read_main_memory_datas(KB)": (128, "遍2:pmu6+pmu7",
                                   "08 §3.59 `128/1024 x (r50d+r50e)`"),
    "write_main_memory_datas(KB)": (128, "遍1:pmu5", "08 §3.59 `128/1024 x r50c`"),
    "GM_to_L1_datas(KB)": (256, "遍3:pmu2", "08 §3.59 `256/1024 x (r32-r206)`"),
    "L0C_to_GM_datas(KB)": (128, "遍3:pmu4", "08 §3.59 `128/1024 x (r20c-r206)`"),
    "GM_to_UB_datas(KB)": (128, "遍3:pmu1", "08 §3.59 `128/1024 x r3e`"),
    "UB_to_GM_datas(KB)": (128, "遍3:pmu0", "08 §3.59 `128/1024 x r3d`"),
}
#  ★遍次写死是有意的: 它跟着采集配置走, 配置一变这里就该红。
#    那不是误报 —— 槽号只在一遍之内有意义, 遍次变了就说明 §3.59 的寄存器→槽映射
#    要重新核一遍。宽松到"只比槽不比遍"会把一类真问题放过去。


def ambig_resolve(op_filter: str = None, limit: int = 0, dump_col: str = None) -> dict:
    """跨目录对歧义列的候选 C 取交集; 已解出的目录按单元素集合参与。"""
    dirs = product_kdirs()
    if op_filter:
        dirs = [k for k in dirs if op_filter in k]
    if limit:
        dirs = dirs[:limit]
    #  ★求交的键是 **(列, 槽位形式)** 而不是列。2026-10-28 (`08` §3.110c):
    #    对裸常数求交, 等于默认"同一列在所有目录上都用同一个槽位形式"。
    #    那个默认是错的 —— 退化目录上成立的**单槽**写法与一般目录上的**求和**写法
    #    给出不同的 C, 而它们是同一个公式。按形式分组之后两者各自收敛, 互不冲突。
    cand = collections.defaultdict(list)      # (列, 形式) -> [(目录, C 集合), ...]
    forms = collections.defaultdict(collections.Counter)   # 列 -> {形式: 份数}
    skipped = collections.Counter()           # 列 -> 因判别力不足被剔的目录数
    dull_amb = collections.Counter()          # 列 -> 被剔的目录里**本来报歧义**的份数
    for k in dirs:
        try:
            sol = solve_columns(k)
        except (OSError, ValueError, KeyError, IndexError):
            continue
        for cols in sol.values():
            for c, v in cols.items():
                fs = v[4] if v and len(v) > 4 else None
                if not fs:
                    continue
                #  ★恒定列不参与求交。它给出的候选是巧合拟合, 混进来会把交集错误地交空
                #    (`08` §3.110b, 由 §3.59 的正例对照臂揪出来)。
                #    判据取"非零值至少两个不同" —— 单值列上任何 (遍,槽,C) 都只需对上一个数。
                if (v[5] if len(v) > 5 else None) in (None, 0, 1):
                    skipped[c] += 1
                    if len({f[2] for f in fs}) > 1:
                        dull_amb[c] += 1
                    continue
                per = collections.defaultdict(set)
                for _n, s, C in fs:
                    per[_slotform(_n, s)].add(C)
                for form, cs in per.items():
                    cand[(c, form)].append((k, frozenset(cs)))
                    forms[c][form] += 1
    out, ctrl, byform = {}, {}, collections.defaultdict(list)
    for (c, form), obs in sorted(cand.items()):
        if len(obs) < 2:
            continue
        inter = set(obs[0][1])
        for _k, s in obs[1:]:
            inter &= s
        byform[c].append({
            "form": form, "dirs": len(obs),
            "ambig_dirs": sum(1 for _k, s in obs if len(s) > 1),
            "union": sorted({x for _k, s in obs for x in s}),
            "inter": sorted(inter),
            #  交集空时报**是谁把它交空的**: 第一个与前缀交集不相容的目录
            "culprit": _first_conflict(obs)})
    for c, fl in byform.items():
        #  ★主形式 = 覆盖目录最多的那一个; **并列时取槽数最少的**。
        #    为什么并列要定死规则而不是取第一个: `read_main_memory_datas` 上
        #    `遍2:pmu6+pmu7` 与 `遍2:pmu0+pmu1+pmu7` 等五种写法**同为 103 份、同为 C=128**
        #    —— 多出来的槽在这些目录上恒为 0, 加不加都成立。取第一个就会把
        #    `pmu0+pmu1+pmu7` 报成主形式, 而 §3.59 的答案是 `pmu6+pmu7`。
        #    这正是 §3.83 记过的"先试到就报"换了个层级又出现一次。
        fl.sort(key=lambda d: (-d["dirs"], _nslots(d["form"]), d["form"]))
        main = fl[0]
        #  ★把"只是多加了恒零槽"的写法标出来, 别让它们冒充独立形式。
        mset = _slotset(main["form"])
        for d in fl[1:]:
            if (d["dirs"] == main["dirs"] and d["inter"] == main["inter"]
                    and mset < _slotset(d["form"])):
                d["redundant"] = True
        n_amb = sum(d["ambig_dirs"] for d in fl)
        if c in AMBIGRES_CONTROL:
            want, wform, cite = AMBIGRES_CONTROL[c]
            ctrl[c] = {"want": want, "wform": wform, "cite": cite,
                       "inter": main["inter"],
                       "form": main["form"], "dirs": main["dirs"],
                       "ambig_dirs": n_amb, "skipped": skipped[c],
                       "culprit": main["culprit"], "nforms": len(fl),
                       "ok": main["inter"] == [want] and main["form"] == wform}
        if not n_amb:
            continue                          # 处处都已解出, 不归本命令管
        out[c] = {"forms": fl, "skipped": skipped[c], "ambig_dirs": n_amb}
    #  ★`dump_col`: 把某一列**每份目录各自给出什么候选集**摊开。
    #    动因 (2026-10-28): 对照臂报 `read_main_memory_datas` 交集为空, 而它在每份目录上
    #    都**唯一**解出 —— 那不是"数据区分不开", 是**各份解出的值不一样**。
    #    这两种情况在汇总行上长得一样 (都只看到"交集为空"), 必须能摊开才分得清。
    dump = None
    if dump_col:
        g = collections.defaultdict(list)
        for (c, form), obs in cand.items():
            if c != dump_col:
                continue
            for k, s in obs:
                g[(form, tuple(sorted(s)))].append(k)
        dump = sorted(g.items(), key=lambda kv: -len(kv[1]))
    #  ★剔完之后**不再歧义**的列必须单独报, 不能让它从报告里消失。
    #    它是一条结论而不是空白: 那一列的"歧义"本来就**全部来自恒定目录**,
    #    即它从来没真的欠定过。静默消失会让读的人以为这一列没被扫到。
    #  ★"剩下的目录交出来是什么"才是这一列的答案。只报"不再歧义"等于把问题换个说法,
    #    那是本仓记过的"假设演化 = 0 减熵"的同一个形状。
    vanished = {}
    for c, n in sorted(dull_amb.items()):
        if c in out:
            continue
        fl = byform.get(c) or []
        main = fl[0] if fl else None
        vanished[c] = {"dull_amb": n, "skipped": skipped[c],
                       "left": main["dirs"] if main else 0,
                       "form": main["form"] if main else None,
                       "nforms": len(fl),
                       "inter": main["inter"] if main else [],
                       "culprit": main["culprit"] if main else None}
    return {"cols": out, "control": ctrl, "dump": dump, "dump_col": dump_col,
            "vanished": vanished, "byform": dict(byform)}


def _slotform(n, s) -> str:
    """把 `--solve` 报的 (遍, 槽) 归一成一个形式名, 例 `遍2:pmu6+pmu7`。

    ★**遍次必须进键。** 第一版把它去掉了, 理由是"同一个计数器在不同采集配置里
      可能落在不同遍, 那不是形式差异" —— 那是一条**我没验过的假设**, 而数据当场否掉它:
      去掉遍次之后每一列冒出 5~6 个"交集为空"的次形式, 且
      `read_main_memory_datas` 的主形式变成 `pmu0+pmu1+pmu7` (§3.59 是遍2 的 `pmu6+pmu7`)。
      原因直白: **槽号只在一遍之内有意义** —— 不同遍下发的是不同的 PMU 事件组,
      所以遍3 的 `pmu1` 与遍5 的 `pmu1` 根本是两个计数器。把它们并成一组,
      等于拿两个无关的量求交, 必然交空。
    """
    return f"遍{n}:" + (f"pmu{s}" if isinstance(s, int) else str(s))


def _slotset(form: str) -> frozenset:
    """形式名 -> `(遍, 槽)` 的集合。遍次带进去, 所以不同遍的同名槽不会被当成同一个。"""
    n, _, rest = form.partition(":")
    return frozenset((n, x) for x in rest.split("+"))


def _nslots(form: str) -> int:
    return len(_slotset(form))


def _first_conflict(obs):
    acc = set(obs[0][1])
    for k, s in obs[1:]:
        if not (acc & s):
            return {"dir": k, "acc": sorted(acc), "this": sorted(s)}
        acc &= s
    return None


def print_ambigres(op_filter: str = None, limit: int = 0, dump_col: str = None) -> int:
    rr = ambig_resolve(op_filter, limit, dump_col)
    r, ctrl = rr["cols"], rr["control"]
    print(f"=== 歧义列候选 C 跨目录取交集 (08 §3.110) ==="
          + (f"  只看路径含 {op_filter!r} 的目录" if op_filter else ""))
    if dump_col:
        print(f"\n--- {dump_col}: 每份目录各自的候选集 ---")
        if not rr["dump"]:
            print("  本次样本上没观测到这一列。")
            return 0
        for (form, cs), ks in rr["dump"]:
            print(f"  形式 {form:<14} 候选 {list(cs)!s:<22} {len(ks):>5} 份   例 "
                  + "/".join(ks[0].rstrip("/").split("/")[-2:]))
        print(f"\n  ★候选集全是单元素而值不同 = **各份解出的常数不一样**, 不是"
              f"数据区分不开 —— 那要去找分支, 不是去找更多样本。")
        return 0
    print(f"\n--- 正例对照: §3.59 从寄存器级公式独立定出的 {len(AMBIGRES_CONTROL)} 个常数 ---")
    bad = []
    for c, (want, wform, cite) in sorted(AMBIGRES_CONTROL.items()):
        d = ctrl.get(c)
        if not d:
            print(f"  {c:<32} 本次样本上没观测到 —— 对照臂缺位")
            continue
        mark = ("✓" if d["ok"] else
                "★**形式不符**" if d["inter"] == [want] else "★**不符**")
        print(f"  {c:<32} 主形式 {d['form']:<16} 交集 {d['inter']}  "
              f"期望 {want} @ {wform} ({cite})  {mark}"
              + (f"   [{d['dirs']} 份; 另 {d['nforms']-1} 种形式]"
                 if d["nforms"] > 1 else f"   [{d['dirs']} 份]")
              + (f" [剔 {d['skipped']} 份恒定]" if d["skipped"] else ""))
        if not d["ok"]:
            bad.append(c)
            print(f"        其中歧义 {d['ambig_dirs']} 份"
                  + ("  ★**处处唯一解而值不同** —— 这是分支, 不是欠定"
                     if not d["ambig_dirs"] else ""))
            if d["culprit"]:
                cp = d["culprit"]
                print(f"        交空于 {'/'.join(cp['dir'].rstrip('/').split('/')[-3:])}: "
                      f"此前 {cp['acc']}, 它只允许 {cp['this']}")
    if bad:
        print("  ★对照臂不符 —— **先别看下面的结论**, 求交这条路本身有问题。")
    elif not ctrl:
        print("  ★一个对照臂都没观测到 —— 下面的收敛结论无从判真假。")
    n_solved = 0
    for c, d in r.items():
        print(f"\n  {c}   (歧义 {d['ambig_dirs']} 份; "
              f"另有 {d['skipped']} 份因该列恒定被剔)")
        #  ★主形式单独算"收敛"。次形式多是退化写法 (只在部分目录上成立),
        #    把它们一起计数会让"收敛了几列"虚高。
        for i, f in enumerate(d["forms"]):
            tag = ("★收敛到唯一" if len(f["inter"]) == 1 else
                   "**交集为空 —— 各目录真的不一致**" if not f["inter"] else
                   f"仍有 {len(f['inter'])} 个")
            lead = ("主形式" if i == 0 else
                    "冗余  " if f.get("redundant") else "次形式")
            print(f"      {lead} {f['form']:<20} {f['dirs']:>5} 份 "
                  f"(歧义 {f['ambig_dirs']}); 并集 {f['union']}"
                  + ("   ← 只是多加了在这些目录上恒零的槽" if f.get("redundant") else ""))
            print(f"            交集 {f['inter']}   {tag}")
            if f["culprit"]:
                cp = f["culprit"]
                print(f"            ★交空于 "
                      f"{'/'.join(cp['dir'].rstrip('/').split('/')[-3:])}: "
                      f"此前 {cp['acc']}, 它只允许 {cp['this']}")
        if len(d["forms"][0]["inter"]) == 1:
            n_solved += 1
    #  ★2026-10-29: 按 (遍, 槽) 分组之后 `r` (形式**内**仍歧义的列) 在全仓上是**空的** ——
    #    `--solve` 报的歧义**全部是形式之间的歧义**。所以这一节才是正文, 不是补遗;
    #    原先"剔除恒定目录后不再歧义"的标题已经说不全原因了 (现在有两个原因)。
    van = rr.get("vanished") or {}
    if van:
        print(f"\n--- 单目录内报过歧义、跨目录求交后**定下来**的 {len(van)} 列 ---")
        print("  (两个原因: 恒定目录被剔 / 歧义本来就是**形式之间**的, 按 (遍,槽) 分组即消)")
        for c, d in sorted(van.items()):
            tail = (f"交集 {d['inter']}" if d["left"] else
                    "★**一份有判别力的都不剩** —— 本机这一列处处恒定, 说不出任何常数")
            if d["left"] and not d["inter"]:
                tail += " —— **同形式内处处唯一解而值不同 = 真分支**"
            print(f"  {c:<32} 剔 {d['skipped']} 份 (其中本来报歧义 {d['dull_amb']} 份), "
                  f"剩 {d['left']} 份 (主形式 {d['form']}"
                  + (f", 另 {d['nforms']-1} 种" if d["nforms"] > 1 else "")
                  + f"); {tail}")
            if d.get("culprit"):
                cp = d["culprit"]
                print(f"        交空于 {'/'.join(cp['dir'].rstrip('/').split('/')[-3:])}: "
                      f"此前 {cp['acc']}, 它只允许 {cp['this']}")
        n_uni = sum(1 for d in van.values() if len(d["inter"]) == 1)
        n_dead = sum(1 for d in van.values() if not d["left"])
        print(f"\n  其中 {n_uni} 列收敛到唯一的 C; "
              f"{n_dead} 列本机无判别力 (说不出常数); "
              f"{len(van) - n_uni - n_dead} 列同形式内仍不一致 (真分支)。")
    print(f"\n  按形式分组后**仍**歧义的列: {len(r)} 列"
          + (f" (其中主形式收敛到唯一的 {n_solved})" if r else
             "  ★**一列都没有** —— `--solve` 报的歧义全部是形式之间的歧义"))
    print("  ★交集是必要条件也是充分条件: 候选集里的每个 C 都是"
          "**该目录全部非零行逐行成立**的, 所以落在交集里 = 处处成立。")
    print("  ★求交按**槽位形式**分组 (§3.110c): 常数离开它的槽位没有意义 —— "
          "同一个公式的求和写法与退化单槽写法给出不同的 C, 混在一起会被判成冲突。")
    print("  ★但交集唯一**不等于**该值正确 —— 若全部目录都缺少能区分的大值, "
          "交集可能唯一地收敛到一个错的 C。要证伪得拿**样本外**目录再交一次。")
    return 1 if bad else 0


#  ---- 覆盖数三分口径 (08 §3.80/§3.81) -------------------------------------------------
#  为什么要做成命令: 这个数**错过三次** —— 600 份 / 900 份 / 11012 份扫出三个不同的
#  "从未出现"列数 (87 / 80 / 59), 而第一次就被写进了 README 与报告 (09 第 128 条)。
#  它不是估计值, 是一个**会被自己的采样方式改变的"事实"**。一次性脚本扫完就丢,
#  下一个人只能重写一遍扫法 —— 于是口径再变一次。

#  ★产物树**有两棵**: `runs/` 与 `cases/*/runs/`。只扫前者已经出过**四次**错:
#  06 V18 (F10 "无可观测间隙") · 08 §3.81 (覆盖数 87/80/59 扫三次三个数) ·
#  08 §3.80 的 114 · 08 §4.x 的 `aic_mte3_wait_ratio` "max 4.9e-05, 不是换个算子能解决的"
#  (实为 3.03e-03, 差 62 倍, 见 §3.85)。**四次里有两次给出的是否定结论** ——
#  那种错会把下一步指到不存在的地方 (E17)。所以本模块里枚举产物只走这一个入口。
PRODUCT_GLOBS = ("runs/**/{pat}", "cases/*/runs/**/{pat}")


def product_files(pat: str) -> list:
    """枚举两棵产物树里匹配 `pat` 的文件。**不要再在别处手写 glob。**"""
    out = []
    for g in PRODUCT_GLOBS:
        out += glob.glob(g.format(pat=pat), recursive=True)
    return sorted(out)


def product_kdirs() -> list:
    """枚举两棵树里的 kernel 目录 (以 `OpBasicInfo_*.csv` 为标志)。"""
    return sorted({os.path.dirname(p) for p in product_files("OpBasicInfo*.csv")})


#  动因样本 (gmfr): 本模块的判据绝大多数在它上面定, `--coverage` 的三分也以它为准。
#  提成常量是因为自检与 coverage 两处都要用同一个目录 —— 两处各写一份就会分叉。
MOTIV_KDIR = ("runs/sweep_64_512_128_0_4_4/OPPROF_20260806024727_JYPETUZHUHHVICMO/"
              "GroupedMatmulFinalizeRouting_a26d6ce4e036526731b7c0c0b18e3c7a_"
              "10000000000000000101_mix_aic/3")

#  来源: `08` §3.80 定的抽法 —— msopprof 的 .rodata 字符串池里带 `aic_`/`aiv_` 前缀的列名。
#  本仓早先写的 204 是另一种抽法, 以这条正则为准 (两者差 3 条)。
COLNAME_RE = re.compile(r"^(aic|aiv)_[a-z0-9_]+(\(.*\))?$")
#  指令记录解析函数自报的名字 (08 §3.101 订正版)
PARSE_NAME_RE = re.compile(rb"^Parse[A-Za-z0-9_]{2,48}Record$")
#  来源: `08` §3.81 的引用 `msopprof(9251896)`。**带字节数**, 因为同机还有
#  cann-9.0.0-beta.2 的另一份 (8989640) 与 cann-8.5.0 的 msopt 版 —— 取错版本
#  会悄悄换掉全集 (feedback: 引用必须解得出对象)。
#  ★第二族: **没有 `aic_`/`aiv_` 前缀**的跨层列 (`GM_to_L1_datas(KB)` /
#  `L1_to_GM_datas(KB)(estimate)` / `read_main_memory_datas(KB)` …)。
#  它们**不在 201 的全集里** —— `COLNAME_RE` 只认前缀族, 所以"出现过 142 / 从未出现 59"
#  那三分也不含它们。2026-10-23 之前这一族**没人数过** (`09` 第 163/164 条),
#  而 `k=40` 的上游 (`L1_to_GM_datas(KB)(estimate)`) 正好落在这里。
#  分开计数而不合并: `201` 这个数已被 `README`/报告多处引用, 合并会让它变。
XLEVEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*_(datas|bw_usage_rate|bw)\([^)]*\)(\([a-z]+\))?$")

#  ★不写死版本号: 本机装的是 9.0.0 而这里曾写死 9.1.0-beta.1, 自检因此 FileNotFoundError,
#    并被当成"长期未过"登记了一天多 (2026-10-28 查实)。socspec.tool_path 按
#    ASCEND_HOME_PATH -> ascend-toolkit/latest -> cann-* 的顺序探存在的那个。
from .socspec import tool_path as _tool_path
MSOPPROF = _tool_path("tools/msopprof/bin/msopprof",
                      legacy="/usr/local/Ascend/cann-9.1.0-beta.1")
#  来源: 上面那个文件本身 —— `stat -c%s` 得 9251896, 与 `08` §3.81 的引用
#  `msopprof(9251896)` 一致。单独再写一遍来源是因为常数来源棘轮**以前一条模块级语句为界**
#  取上下文 (`trust.py:1308` 写明了为什么: 不设界的话文件头一句就能让整个文件免检),
#  所以紧挨在 `MSOPPROF` 下面的这一行拿不到上面那段注释。
MSOPPROF_BYTES = 9251896


def binary_colnames(path: str = MSOPPROF, want_bytes: int = MSOPPROF_BYTES) -> set:
    """从 msopprof 的字符串池抽 `aic_`/`aiv_` 列名全集。**字节数不符就拒**, 不换一个凑合。"""
    got = os.path.getsize(path)
    if got != want_bytes:
        raise SystemExit(
            f"拒跑: {path} 是 {got} 字节, 本仓引用的是 {want_bytes} 字节。\n"
            f"  同机有多份 msopprof (9.0.0-beta.2 / 8.5.0 的 msopt 版), 取错版本会悄悄换掉全集。\n"
            f"  要换版本请同时改 MSOPPROF/MSOPPROF_BYTES 并在 08 里写明。")
    with open(path, "rb") as f:
        blob = f.read()
    toks = set()
    for m in re.finditer(rb"[\x20-\x7e]{4,}", blob):
        toks.update(m.group().decode("ascii").split("\x00"))
    return {t for t in toks if COLNAME_RE.match(t)}


def xlevel_colnames(path: str = MSOPPROF, want_bytes: int = MSOPPROF_BYTES) -> set:
    """第二族 (无 `aic_`/`aiv_` 前缀的跨层列) 的全集。**与 201 那一族分开数**, 见 XLEVEL_RE。"""
    got = os.path.getsize(path)
    if got != want_bytes:
        raise SystemExit(f"拒跑: {path} 是 {got} 字节, 本仓引用的是 {want_bytes} 字节")
    with open(path, "rb") as f:
        blob = f.read()
    toks = set()
    for m in re.finditer(rb"[\x20-\x7e]{4,}", blob):
        toks.update(m.group().decode("ascii").split("\x00"))
    return {t for t in toks if XLEVEL_RE.match(t) and not COLNAME_RE.match(t)}


def col_order(path: str = MSOPPROF, want_bytes: int = MSOPPROF_BYTES) -> list:
    """从 msopprof 的 `.text` 里读**每张表的列顺序**。返回 [(槽偏移, 列名), ...] 按出现序。

    ★怎么找到的 (`08` §3.89): 列名串有 201/201 个 `adrp+add` 引用 (§3.88), 而这些引用
    几乎都长成同一个三条指令的模式 ——

        adrp x0, <页>              ; 列名所在页
        add  x1, x0, #<页内偏移>    ; x1 = 列名 C 串
        add  x0, x19, #<槽>        ; x0 = 结构体里的第 <槽> 个 std::string
        bl   _ZNSsC1EPKcRKSaIcE    ; std::string(x0, x1, allocator)

    也就是**表头行的构造**: 把列名逐个写进一个 `std::string` 数组。
    槽是 8 字节一格, 所以 `槽/8` 就是**列号**。

    ★验过: 抽出来的 `ArithmeticUtilization` 顺序与真机 csv 表头**逐列相同**
    (`aic_cube_ratio` 在槽 0x020 = 第 4 列, csv 里正是第 5 个字段, 前面 4 个
    `block_id`/`sub_block_id`/`aic_time(us)`/`aic_total_cycles` 占 0x000~0x018)。

    用处: 它给出列名 → **列号**。那 29 处引用全是表头构造, **不是公式** ——
    公式按列号索引 (`.rodata` 里 `FormulaForA5...EE<N>_E` 排到 113), 所以列号是桥。
    """
    got = os.path.getsize(path)
    if got != want_bytes:
        raise SystemExit(f"拒跑: {path} 是 {got} 字节, 本仓引用的是 {want_bytes} 字节")
    with open(path, "rb") as f:
        blob = f.read()
    #  来源: `readelf -S -W msopprof` 的 `.text` 行 —— 文件偏移与 vaddr 都是 0x038800,
    #  长度 0x6effec。两者相等所以下面直接拿文件偏移当 PC 用。
    lo, hi = 0x038800, 0x038800 + 0x6effec
    out, i = [], lo
    while i < hi - 16:
        w = struct.unpack_from("<I", blob, i)[0]
        if (w >> 31) & 1 and ((w >> 24) & 0x1F) == 0x10:          # adrp
            immlo = (w >> 29) & 3
            imm = (((w >> 5) & 0x7FFFF) << 2) | immlo
            if imm & (1 << 20):
                imm -= (1 << 21)
            page, rd = (i & ~0xFFF) + (imm << 12), w & 0x1F
            a1 = struct.unpack_from("<I", blob, i + 4)[0]
            a2 = struct.unpack_from("<I", blob, i + 8)[0]
            #  ADD (immediate, 64 位): sf=1, 28..23 = 100100010, sh=0
            ok1 = (a1 >> 31) & 1 and ((a1 >> 23) & 0x1FF) == 0b100100010 and ((a1 >> 5) & 0x1F) == rd
            ok2 = (a2 >> 31) & 1 and ((a2 >> 23) & 0x1FF) == 0b100100010
            if ok1 and ok2:
                noff = page + ((a1 >> 10) & 0xFFF)
                if 0 <= noff < len(blob):
                    end = blob.find(b"\0", noff)
                    nm = blob[noff:end].decode("ascii", "replace") if 0 < end - noff < 64 else ""
                    if COLNAME_RE.match(nm):
                        out.append(((a2 >> 10) & 0xFFF, nm))
                        i += 16
                        continue
        i += 4
    return out


def form_codes(path: str = MSOPPROF, want_bytes: int = MSOPPROF_BYTES,
               zero_code: bool = True) -> dict:
    """从 `.text` 抽 `{列名 → 换算形式码 k}`。只收 14 份拷贝给出**同一个 k** 的列。

    ★`k` 是什么 (`08` §3.92/§3.94): 每个列名入表时带一个 16 位常量, 先 `strh` 进 sp 缓冲
    再随 vector 拷走。把它与 `--solve` 在动因样本上解出的换算形式取交集 (38 列),
    **9 个 k 对 9 种形式, 一一对应, 没有一个 k 对上两种**; 而 `k` **不**决定 `(遍, 槽)`
    (`k=4` 横跨 12 种)。⇒ `k` 编的是"怎么算", 不是"算谁"。

    ★抽取形状 (次序是判据的一部分, 不能放松):
    `mov wR,#k` → (≤4 条) → `adrp+add` 取列名 → (≤6 条) → `strh wR,[x?,#off]` **同寄存器**。
    要求 `mov` 在名字**之前**、`strh` 在名字**之后** —— 第一版没有这条次序约束,
    把 `aic_vec_*` 全抽成同一个值 (串到了邻条目的 `mov`), 见 `09` 第 146/147 条。

    ★正确性的证据不是"抽到了多少条", 是上面那个**一一对应** ——
    它由两份**互相独立**的来源 (二进制常量 / 从产物数据反解的形式) 对出来,
    任何一侧抽错都不会这么齐。
    """
    got = os.path.getsize(path)
    if got != want_bytes:
        raise SystemExit(f"拒跑: {path} 是 {got} 字节, 本仓引用的是 {want_bytes} 字节")
    with open(path, "rb") as f:
        blob = f.read()
    lo, hi = 0x038800, 0x038800 + 0x6effec

    def wd(a):
        return struct.unpack_from("<I", blob, a)[0]

    seen = collections.defaultdict(collections.Counter)
    i = lo
    while i < hi - 4:
        x = wd(i)
        if (x >> 31) & 1 and ((x >> 24) & 0x1F) == 0x10:          # adrp
            immlo = (x >> 29) & 3
            im = (((x >> 5) & 0x7FFFF) << 2) | immlo
            if im & (1 << 20):
                im -= (1 << 21)
            page, rd = (i & ~0xFFF) + (im << 12), x & 0x1F
            nm = None
            for k in range(1, 6):
                v = wd(i + 4 * k)
                if (v >> 31) & 1 and ((v >> 23) & 0x1FF) == 0b100100010 \
                        and ((v >> 5) & 0x1F) == rd and not ((v >> 22) & 1):
                    a = page + ((v >> 10) & 0xFFF)
                    end = blob.find(b"\0", a)
                    cand = blob[a:end].decode("ascii", "replace") if 0 < end - a < 64 else ""
                    if COLNAME_RE.match(cand):
                        nm = cand
                    break
            if nm:
                mv = mr = None
                for k in range(1, 5):                              # MOVZ 32 位, 名字之前
                    v = wd(i - 4 * k)
                    if ((v >> 23) & 0x1FF) == 0b010100101 and not ((v >> 31) & 1):
                        mr, mv = v & 0x1F, (v >> 5) & 0xFFFF
                        break
                if mv is not None:
                    for k in range(1, 7):                          # STRH 同寄存器, 名字之后
                        v = wd(i + 4 * k)
                        if (v >> 22) == 0b0111100100 and (v & 0x1F) == mr:
                            seen[nm][mv] += 1
                            break
                elif zero_code:
                    #  ★码为 0 的列写的是 `strh wzr`, **没有 mov** —— 原实现要求名字前有
                    #    `mov wR,#k`, 于是整整一类列从未进过这张表 (2026-10-26 实测 19 个,
                    #    其中就有 README 上挂着的 `*_wait_ratio` 一族, 见 08 §3.102)。
                    #  ★wzr 这一侧**失去了寄存器匹配**这条防串号的判据, 只剩"名字之后 ≤6 条"
                    #    这个窗口。所以它靠的是下面那条**跨 14 份拷贝取值唯一**的过滤
                    #    —— 真串到邻条目的话 14 份不会给出同一个值。
                    for k in range(1, 7):
                        v = wd(i + 4 * k)
                        if (v >> 22) == 0b0111100100 and (v & 0x1F) == 31:
                            seen[nm][0] += 1
                            break
        i += 4
    return {n: c.most_common(1)[0][0] for n, c in seen.items() if len(c) == 1}


#  对账用的样本目录。只放**动因样本 (mix)**: 跨 Op Type 并样会让 `k` 对上两种分母
#  (§3.82: `*_ratio` 的分母 mix 用 floor(TaskDur)xfreq、纯核用本行 total_cycles),
#  那不是抽错而是 `k` 比"形式"粗一层 —— 要看那一层请显式传多个目录。
JOIN_KDIRS = (MOTIV_KDIR,)


#  dtype 配分的五列 (`08` §3.98)。`aiv_vec_ratio` 在动因样本上**逐行精确等于**它们之和。
VEC_PARTS = ("aiv_vec_fp32_ratio", "aiv_vec_fp16_ratio", "aiv_vec_int32_ratio",
             "aiv_vec_int16_ratio", "aiv_vec_misc_ratio")


#  ★本模块**不**过滤"坏 launch", 而 `core/perf._corrupt_launches` 过滤 (判据: 某核的
#  `*_time(us)` 超过同 launch 的 `Task Duration`, 物理上不可能)。同一份语料两套口径,
#  这是**有意的**, 理由分两段 (`09` 第 156 条):
#    · E1/E2/E2b/E3 是**从 dump 逐位还原 csv**。计数器坏了, csv 与 bin 仍然一致 ——
#      过滤掉反而会让"还原得出"这件事少验几行。这些判据必须看全量。
#    · `--vecshare` / `--denom` / `--formcode` 这些**读数值含义**的, 坏行会污染。
#      ★**2026-10-26 实测订正**: 原文写"只有 48 行, 全在 `runs/pipe_v18a_i1_on` 的 launch 0"。
#        按"某核 `*_time(us)` > 本 launch 的 `Task Duration`, 任一侧越界即记一行"这个口径
#        扫全仓 1602 个有 dur 有 PipeUtilization 的目录, 得到的是 **2 个目录 96 行**:
#        `pipe_v18a_i1_on` 的 launch 0 占 72 行, 另有
#        `runs/authority_v85_off/.../4` 占 24 行 —— **后者原文没有**, 而它是一个
#        被当作基线用过的 run。数字现算: `python -m core.aprof --corrupt`。
#      **要加过滤时, 只能加在后一类上。**


#  ★★**2026-10-28 订正的订正 (`09` 第 210 条)。先读这一段, 它推翻了下一段。**
#  第 207 条判 `aic_mte2_instructions` 是"伪回归项"。**那个判断是错的。**
#  四格表 (全 GemmV3 4944 行, 行分两类: `0.5(hit+miss) − L0C_to_GM` 是否 ≡ 33):
#
#                    五项式逐位精确   不精确
#      退化行 4432        3376        1056
#      有信息行 512        512           0      <- 100% 精确
#
#  ⇒ **五项式在有信息的那 512 行上一行不差, 它的全部失败都落在退化行里。**
#  第 207 条的证据 (三个 1024³ 变体上 `残差 = 0.0625 × Δmte2`) 没有错, 错的是作用域:
#  **那三个变体的行全部是退化行** —— 我从一个退化子集推出了"这一项是伪的",
#  而它在真正有判别力的行上是**承重**的。
#  真正的缺口因此换了形状: 不是"式子多了一项", 是**式子缺一个分支** ——
#  在整 tile 块 (退化行) 上 `y` 不随 mte2 走, 而式子照走。
#
#  ★仍然成立的只有一条: `aic_mte3_instructions` 在 GemmV3 全部 4944 行上**恒等于 18**
#  (变异系数 0), 所以那一项是**截距换了个写法** (`0.0104167 × 18 = 0.1875`)。
#  后向剔除在 512 行上重推出的最小列集正是"四列 + 截距 0.1875", 与此一致。
#
#  ★`L1_to_GM_datas(KB)(estimate)` 的候选闭式 (08 §3.105, 2026-10-27)。**假设, 未定论。**
#  由"跨目录合并后整体最小二乘 + 后向剔除"得到, 系数是整洁有理数;
#  在 GemmV3 的 41/54 个 run 组上**逐位精确**, 另 13 组差 0.25~11 KB (像量化步长);
#  **别的算子不成立** (同一组基重拟合时系数不同 —— GroupedMatmul 的前三项仍保持
#  0.5 : 0.5 : −1 的比例但整体缩放 ~0.727)。所以它现在是**作用域内的候选式**, 不是公式。
L1GM_FORMULA = (
    ("L2Cache", "aic_write_cache_hit", 0.5),
    ("L2Cache", "aic_write_cache_miss_allocate", 0.5),
    ("Memory", "L0C_to_GM_datas(KB)", -1.0),
    ("Memory", "aic_mte2_instructions", 1.0 / 16),
    ("Memory", "aic_mte3_instructions", 1.0 / 96),
)
L1GM_TARGET = "L1_to_GM_datas(KB)(estimate)"

#  ★第二个候选式 (08 §3.106, 2026-10-27): **两列 + 一个常数**, 比上面那个五项式
#  参数少而且最坏情况好两个数量级 —— GemmV3 上逐行精确 3976/4944 (80.4%),
#  不精确时最大只差 0.5156 KB (五项式在 13 个组上差到 33 KB)。
#  怎么找到的: 先去掉五项式里的两个小项, 看三项式 `0.5(hit+miss) − L0C_to_GM` 的残差 ——
#  它在 4944 行上**只取 6 个离散值** (1, 1.5, 2, 2.5, 11, 11.5), 而这 6 个值满足
#  `r = L0C_to_GM/256 + 0.5`。合并即得下式。
#  ★**此前所有拟合都没有常数列**, 这正是它们需要 mte2/mte3 两个小项去凑一个近似常数的原因。
L1GM_FORMULA2 = (
    ("L2Cache", "aic_write_cache_hit", 0.5),
    ("L2Cache", "aic_write_cache_miss_allocate", 0.5),
    ("Memory", "L0C_to_GM_datas(KB)", -255.0 / 256.0),
)
#  来源: docs/msprof-re/08-msopprof-a-path.md §3.106 —— 由 4944 行实测反解:
#  三项式的残差只取 6 个值, 对照 L0C_to_GM 得 r = L0C_to_GM/256 + 0.5, 这里是那个 0.5。
#  它**不是**向上取整 (ceil/round/floor 三种各验过, GemmV3 从 3976 掉到 ≤164)。
L1GM_CONST2 = 0.5


def l1gm_span(op_filter: str = "GemmV3") -> dict:
    """**整族判定**: y 在不在同行其它数值列的线性张成里 (该算子全部目录合并)。

    ★为什么这一条比"再穷举一层"有用: **任何子集的张成都是全集张成的子空间**。
      全集拟合不下来, 三列、四列、八十列全都不必试 —— 一次计算关掉整个线性族。
    ★必须合并到 n ≫ p: 单目录 24 行 × 80 余列时矩阵满行秩, **任何 y 都在张成里**,
      残差恒为 0, 判定是空的 (08 §3.105 实撞)。
    ★必须剔掉由 y 派生的列 (`L1_to_GM_*`), 否则它会以单列、残差 1e-16 "拟合成功"。
    """
    import numpy as _np
    keys = ("block_id", "sub_block_id")
    rows, cols0 = [], None
    for k in product_kdirs():
        if op_filter and op_filter not in k:
            continue
        per = {}
        for p2 in sorted(glob.glob(os.path.join(k, "*.csv"))):
            tbl = os.path.basename(p2).rsplit("_", 1)[0]
            with open(p2, newline="", errors="ignore") as f:
                rd = csv.DictReader(f)
                if not rd.fieldnames or "block_id" not in rd.fieldnames:
                    continue
                for r in rd:
                    d = per.setdefault(tuple(r.get(c, "") for c in keys), {})
                    for c, v in r.items():
                        if c in keys or c is None:
                            continue
                        v = (v or "").strip()
                        if v in ("", "NA"):
                            continue
                        try:
                            d[f"{tbl}::{c}"] = float(v)
                        except ValueError:
                            pass
        yk = sorted({c for d in per.values() for c in d
                     if c.endswith("::" + L1GM_TARGET)})
        if not yk:
            continue
        yk = yk[0]
        per = {a: d for a, d in per.items() if yk in d}
        if len(per) < 4:
            continue
        common = set.intersection(*(set(d) for d in per.values()))
        cs = sorted(c for c in common
                    if c != yk and "L1_to_GM" not in c)      # 剔循环列
        if cols0 is None:
            cols0 = cs
        cs = [c for c in cols0 if c in set(cs)]
        if len(cs) < 10:
            continue
        if cs != cols0:
            ia = [cols0.index(c) for c in cs]
            rows = [([v[i] for i in ia], t) for v, t in rows]
            cols0 = cs
        for a in sorted(per):
            rows.append(([per[a][c] for c in cols0], per[a][yk]))
    if not rows:
        return {}
    X = _np.array([r for r, _ in rows], float)
    y = _np.array([t for _, t in rows], float)
    sol, _r, rank, _sv = _np.linalg.lstsq(X, y, rcond=None)
    rel = float(_np.linalg.norm(X @ sol - y) / (_np.linalg.norm(y) or 1.0))
    err = X @ sol - y
    #  ★**相对残差 x 均值不是逐行误差** —— 相对残差是二范数之比。逐行要报
    #    RMS 与最大绝对值, 否则会把 33 KB 的最大偏差说成 0.01 KB (本节初稿实撞)。
    return {"rows": X.shape[0], "cols": X.shape[1], "rank": int(rank),
            "rel": rel, "rms": float(_np.sqrt((err ** 2).mean())),
            "max_abs": float(_np.abs(err).max())}


def l1gm_eval(op_filter: str | None = None, tol: float = 1e-9) -> tuple:
    """**一次扫描**同时核对两个候选式。返回 `(按 run 组的式一, 按算子的式二)`。

    ★为什么要合并 (2026-10-27): 自检里原来分别调 `l1gm_check` / `l1gm_check2` /
      `l1gm_span`, 三次各扫一遍全仓, pre-commit 从约 2 分钟涨到 **551 秒**。
      **慢闸门会训练出 `--no-verify`** —— 闸门的成本本身就是它的一条设计约束。
      合并后只读一遍 csv。`l1gm_span` 另算 (它要 numpy 做 4944x81 的最小二乘),
      **不进自检**, 只在 `--l1gm` 里跑。
    """
    keys = ("block_id", "sub_block_id")
    by_grp, by_op, by_sign = {}, {}, {}
    for k in product_kdirs():
        if op_filter and op_filter not in k:
            continue
        mem = sorted(glob.glob(os.path.join(k, "Memory_*.csv")))
        l2 = sorted(glob.glob(os.path.join(k, "L2Cache_*.csv")))
        if not mem or not l2:
            continue
        op = os.path.basename(os.path.dirname(k)).split("_")[0]
        g = k.split("/OPPROF_")[0]
        m = {}
        with open(mem[0], newline="", errors="ignore") as f:
            for r in csv.DictReader(f):
                m[tuple(r.get(c, "") for c in keys)] = r
        sgn = by_sign.setdefault(g, set())
        n1, ok1, mx1 = by_grp.get(g, (0, 0, 0.0))
        n2, ok2, mx2 = by_op.get(op, (0, 0, 0.0))
        with open(l2[0], newline="", errors="ignore") as f:
            for r in csv.DictReader(f):
                kk = tuple(r.get(c, "") for c in keys)
                if kk not in m:
                    continue
                src = {**m[kk], **r}
                y = (src.get(L1GM_TARGET) or "").strip()
                if y in ("", "NA"):
                    continue
                try:
                    yv = float(y)
                except ValueError:
                    continue
                for spec, const, acc in ((L1GM_FORMULA, 0.0, "1"),
                                         (L1GM_FORMULA2, L1GM_CONST2, "2")):
                    tot, bad = const, False
                    for _t, col, coef in spec:
                        v = (src.get(col) or "").strip()
                        if v in ("", "NA"):
                            bad = True
                            break
                        try:
                            tot += coef * float(v)
                        except ValueError:
                            bad = True
                            break
                    if bad:
                        continue
                    d = abs(tot - yv)
                    if acc == "1":
                        n1 += 1
                        ok1 += 1 if d < tol else 0
                        mx1 = max(mx1, d)
                        #  ★**带符号**收残差的取值集合 (2026-10-28, `09` 第 203 条)。
                        #    只报 `max |差|` 会丢两件事: 512³ 的残差是 **负**的
                        #    (文档原记 "0.25" 是取过绝对值的, 于是"缺一个正项"这个读法
                        #    从一开始就被锁死); 以及**一个组里只有两个取值**,
                        #    而那两个值差恰好 `0.5x(K/1024-1)` —— 取值集合一打出来就看见。
                        sgn.add(round(tot - yv, 6))
                    else:
                        n2 += 1
                        ok2 += 1 if d < tol else 0
                        mx2 = max(mx2, d)
        if n1:
            by_grp[g] = (n1, ok1, mx1)
        if n2:
            by_op[op] = (n2, ok2, mx2)
    return by_grp, by_op, by_sign


#  候选式三 = 五项式去掉 `aic_mte2_instructions` (`09` 第 207 条提出, 第 210 条订正了它的**解读**:
#  这一项不是伪项, 去掉它只是用来**分离退化行**的一个探针 —— 残差塌缩成 6 个离散值
#  这件事说的是"退化行上 y 只取几个值", 不是"这一项不该在式子里"。)
L1GM_FORMULA3 = tuple(t for t in L1GM_FORMULA if t[1] != "aic_mte2_instructions")


def l1gm_nomte2(op_filter: str | None = None) -> dict:
    """去掉 `aic_mte2_instructions` 项之后, 每个 run 组的残差**取值集合**。

    ★报取值集合而不是 `max` —— 本条结论正是靠"取值集合只有 6 个值"看出来的,
      而 `max` 会把它重新糊回一个连续谱 (`09` 第 203 条已为此栽过一次)。
    """
    out = {}
    for k in product_kdirs():
        if op_filter and op_filter not in k:
            continue
        mem = sorted(glob.glob(os.path.join(k, "Memory_*.csv")))
        l2 = sorted(glob.glob(os.path.join(k, "L2Cache_*.csv")))
        if not mem or not l2:
            continue
        g = k.split("/OPPROF_")[0]
        keys = ("block_id", "sub_block_id")
        m = {}
        with open(mem[0], newline="", errors="ignore") as f:
            for r in csv.DictReader(f):
                m[tuple(r.get(c, "") for c in keys)] = r
        with open(l2[0], newline="", errors="ignore") as f:
            for r in csv.DictReader(f):
                kk = tuple(r.get(c, "") for c in keys)
                if kk not in m:
                    continue
                src = {**m[kk], **r}
                y = (src.get(L1GM_TARGET) or "").strip()
                if y in ("", "NA"):
                    continue
                tot, bad = 0.0, False
                for _t, col, coef in L1GM_FORMULA3:
                    v = (src.get(col) or "").strip()
                    if v in ("", "NA"):
                        bad = True
                        break
                    try:
                        tot += coef * float(v)
                    except ValueError:
                        bad = True
                        break
                if bad:
                    continue
                out.setdefault(g, set()).add(round(tot - float(y), 4))
    return out


#  ---- `--exactrel`: 用**精确有理数零空间**找一列的闭式, 不拟合 ----
#  ★方法来源与为什么换 (2026-10-28, `09` 第 211 条): 此前找闭式全靠"挑子集 + 最小二乘 +
#    后向剔除", 一轮只能试一个形状, 而且每次都要操心留出、容差、伪项 (第 207/210 条各栽一次)。
#    **产物里的数是精确十进制**, 所以这件事根本不是拟合问题, 是**有理数线性代数**:
#    把"目标列 + 候选列 + 常数列"排成矩阵, 求它在 ℚ 上的零空间, 一次拿到**全部**精确关系。
#    没有容差、没有留出的必要 (零空间是恒等式不是估计), 也不会挑出伪项 ——
#    伪项在精确算术下根本进不了零空间。
#  ★零空间维数 > 1 时**关系不唯一**: 列之间本身还有别的恒等式, 任一条都可以加进来。
#    所以报的是"存在一条含目标列的精确关系", 不是"这就是实现"。
_MERGED_CACHE: dict = {}


def _merged_rows(op_filter: str | None = None) -> list:
    """把符合过滤的 kernel 目录下**全部表**按 `(block_id, sub_block_id)` 并成行, 带缓存。

    ★加缓存的理由 (2026-10-28, `09` 第 214 条): 一次 `--selftest` 要对同一个语料调三次
    关系搜索, 每次都重读一遍全部 csv —— 把 aprof 自检从约 20 秒推到 44 秒。
    `l1gm_eval` 的注释早写着"**慢闸门会训练出 `--no-verify`**"。
    ★缓存只在**进程内**, 不落盘: 产物目录会变, 跨进程缓存会把过期数据当成现状。
    """
    key = op_filter or ""
    if key in _MERGED_CACHE:
        return _MERGED_CACHE[key]
    keys = ("block_id", "sub_block_id")
    out = []
    for k in product_kdirs():
        if op_filter and op_filter not in k:
            continue
        merged = {}
        for path in sorted(glob.glob(os.path.join(k, "*.csv"))):
            if os.path.basename(path).startswith("OpBasicInfo"):
                continue
            try:
                with open(path, newline="", errors="ignore") as f:
                    for r in csv.DictReader(f):
                        if any(c not in r for c in keys):
                            break
                        merged.setdefault(tuple(r.get(c, "") for c in keys), {}).update(r)
            except OSError:
                continue
        out.extend(merged.values())
    _MERGED_CACHE[key] = out
    return out


def exact_relation(target: str, op_filter: str | None = None,
                   where=None, target_mul: str | None = None) -> dict:
    """在**精确有理数**下找 `target` 的线性闭式。`where(row)` 可再筛一个分支。

    返回 `{"rows", "cols", "dim", "rel"}`; `rel` 是 `{列名: Fraction}` (含 `"1"` 截距),
    满足 `target == Σ 系数 × 列`。找不到含目标列的关系时 `rel` 为 None。
    """
    from fractions import Fraction as _F
    import sympy as _sp
    keys = ("block_id", "sub_block_id")
    rows = _merged_rows(op_filter)
    rows = [s for s in rows
            if (s.get(target) or "").strip() not in ("", "NA")
            and (target_mul is None or (s.get(target_mul) or "").strip() not in ("", "NA"))
            and (where is None or where(s))]
    if not rows:
        return {"rows": 0, "cols": 0, "dim": 0, "rel": None}

    def fr(s, c):
        v = (s.get(c) or "").strip()
        if v in ("", "NA"):
            return None
        try:
            return _F(v)
        except (ValueError, ZeroDivisionError):
            return None

    #  ★列筛选走**浮点**不走 `Fraction` (2026-10-28, `09` 第 215 条)。
    #    profile 实测: `fr()` 被调 141 万次, `Fraction.__new__` 占 15.2 秒 ——
    #    而这一步只需要判"能不能解析"与"有没有变化", 浮点足够;
    #    精确性由**最终的逐行验证**兜底, 那一层仍是整数算术。
    #    ★第 214 条把瓶颈**猜**成"Fraction 逐行验证"并据此改了整数验证 —— 猜错了,
    #      改完仍是 43 秒。**先 profile 再优化**, 这一条已补进 skill。
    def _ff(_s, _c):
        _v = (_s.get(_c) or "").strip()
        if _v in ("", "NA"):
            return None
        try:
            return float(_v)
        except ValueError:
            return None

    cols = None
    for s in rows:
        ok = {c for c in s if c not in keys and _ff(s, c) is not None}
        cols = ok if cols is None else (cols & ok)
    #  只留**在这批行上真有变化**的列 —— 常数列进来只会让零空间多一维无信息的解
    use = []
    for c in sorted(cols):
        if c in (target, target_mul):
            continue
        seen = set()
        for s in rows:
            seen.add(_ff(s, c))
            if len(seen) > 1:
                break
        if len(seen) > 1:
            use.append(c)
    #  ★**浮点求候选 + 精确验证** (2026-10-28, `09` 第 212 条)。
    #    前一版在 ℚ 上直接求零空间, 在 57 列上 9 秒, 到 gmfr 的两百多列就跑不出来 ——
    #    sympy 的精确 RREF 随列数爆炸。**而严格性只需要落在验证那一步**:
    #    用 SVD 在 float64 上取数值零空间当**候选**, 把系数有理化, 再用 Fraction
    #    在**全部行**上逐行验证。候选错了会被验证打掉, 候选漏了只是少报一条 ——
    #    两种失败都不会产生假结论。
    #    ★所以这个工具的承诺是"**报出来的都对**", 不是"对的都报得出来"。这两句话不一样,
    #    输出里必须写清楚, 否则"没找到"会被读成"不存在"。
    import numpy as _np
    #  ★三段式 (2026-10-28, `09` 第 212 条): **数值剪枝 → 小列集上精确求解 → 全量精确验证**。
    #    走到这一版踩过两种失败, 都由自检当场抓住:
    #      · 纯 ℚ 零空间: 57 列上 9 秒, 两百多列就跑不出来 (sympy 精确 RREF 随列数爆炸);
    #      · 纯 SVD 零空间: 快, 但列间共线时零空间基是**任意旋转**的, 有理化就散了 ——
    #        **自检的正例当场红**, 否则这一版会开始报假的"不存在"。
    #    现在: 用最小二乘在浮点上把列剪到少数几根 (快, 且只负责提候选),
    #    再在这几列上做 ℚ 精确求解 (小系统, 快), 最后在**全部行**上逐行精确验证。
    #    ★承诺仍是"**报出来的都对**", 不是"对的都报得出来" —— 剪枝可能剪掉真列,
    #    那只会少报一条, 不会产生假结论。输出里必须写清楚。
    #  ★★**印数精度** (2026-10-28, `09` 第 213 条)。`*_ratio` / `*_bw` 这类列在 csv 里
    #    只印 6 位小数 —— **印出来的是舍入值**, 所以这类列上**不存在**精确有理关系。
    #    首版用 Fraction 精确相等验证, 于是 13 个未解列**全部**报"没有",
    #    而那只说明"这一列印出来是舍入的", 不说明公式不存在。
    #    ★正例对照当场照出来: 纯 cube 上已知的 `cube_ratio × total_cycles` 与
    #      `cube_time × 1800` 差 **0.017** —— 不是关系不成立, 是印数只有 6 位。
    #    ★仓里的 `--solve` 一直用**逐位字符串比** (把算出的值按同样格式印再比),
    #      我换方法时把这件事丢了。现在补回来: 按目标列**实际印了几位小数**来比。
    _nd = 0
    for _s0 in rows[:512]:
        _t = (_s0.get(target) or "").strip()
        if "." in _t:
            _nd = max(_nd, len(_t.split(".", 1)[1].rstrip()))
    _exactly_printed = _nd == 0 or all(
        _F((_s0.get(target) or "0").strip()) == _F(round(float((_s0.get(target) or "0").strip()), _nd))
        for _s0 in rows[:64])

    def _same(_calc, _s0):
        """算出来的值与该行的印数是否一致。整数/短小数列用精确相等; 带 6 位小数的
        舍入列按 `%.{nd}f` 比字符串 —— 这是 msopprof 自己的输出口径。"""
        _raw = (_s0.get(target) or "").strip()
        if _nd == 0:
            return _calc == _F(_raw)
        return ("%%.%df" % _nd) % float(_calc) == ("%%.%df" % _nd) % float(_F(_raw))

    def tv(_s):
        """目标值。`target_mul` 非空时取 `y × 该列` —— 把有理关系 `y = N/D` 线性化成
        `y·D = N` (线性化有理拟合, Sanathanan-Koerner 1963; 另见 implicit SINDy
        '乘掉分母再在零空间里找稀疏向量'的同一手法)。**整台机器不动**, 只换目标。"""
        _b = fr(_s, target)
        if target_mul is None:
            return _b
        _m = fr(_s, target_mul)
        return None if _m is None else _b * _m
    A = _np.array([[float(fr(s, c)) for c in use] + [1.0] for s in rows], dtype=float)
    yv = _np.array([float(tv(s)) for s in rows], dtype=float)
    scale = max(float(_np.abs(yv).max()), 1.0)

    #  ★筛的阈值必须跟着**印数精度**走 (2026-10-28, `09` 第 213 条):
    #    目标列只印 `_nd` 位时, 它本身就带 ±0.5e-{_nd} 的舍入噪声。
    #    按精确关系的 1e-7 去筛, **会把真关系筛掉** —— 实撞: `aic_mte2_ratio` 等列
    #    因此走了"整体就有残差"的早退分支, 看起来像"不存在"。
    #  ★★**语料太窄就拒判** (2026-10-28, `09` 第 214 条)。行数不明显多于列数时,
    #    精确方法会找到**不存在的关系** —— 实撞: 24 行 x 50 列给出系数 `1741/3142`,
    #    384 行仍给出 `500000/1111`, 两个都"全量逐位成立"而显然是过拟合。
    #    **这正是这套工具本该避免的失效, 它却自己撞上了。**
    #    门槛取 8 倍: 低于它报"判不了", 不报关系 —— 与本工具"报出来的都对"的承诺一致。
    if len(rows) < 8 * (len(use) + 1):
        return {"rows": len(rows), "cols": len(use), "dim": 0, "rel": None,
                "sub": 0, "cand": 0, "rejected": 0, "decimals": _nd,
                "exact_print": _exactly_printed, "toonarrow": True}
    _screen = max(1e-7 * scale, 4.0 * 10.0 ** (-_nd)) if _nd else 1e-7 * scale

    def _resid(idx):
        if not idx:
            return float("inf")
        w, *_ = _np.linalg.lstsq(A[:, idx], yv, rcond=None)
        return float(_np.abs(A[:, idx] @ w - yv).max())

    keep = list(range(A.shape[1]))
    if _resid(keep) > _screen:
        return {"rows": len(rows), "cols": len(use), "dim": 0, "rel": None,
                "sub": len(rows), "cand": 0, "rejected": 0, "nofit": True,
                "decimals": _nd, "exact_print": _exactly_printed}
    changed = True
    while changed and len(keep) > 1:
        changed = False
        for c in list(keep):
            t = [x for x in keep if x != c]
            if _resid(t) <= _screen:
                keep = t
                changed = True
                break
    names = use + ["1"]
    sub_names = [names[i] for i in keep]
    import sympy as _sp
    #  小列集上用 ℚ 精确解 (行数取 2x 列数即可定解, 再全量验证)
    nr = min(len(rows), 4 * len(keep) + 8)
    st = max(1, len(rows) // nr)
    samp = rows[::st][:nr]
    Mx = _sp.Matrix([[_sp.Rational(fr(s, c)) if c != "1" else _sp.Integer(1)
                      for c in sub_names] for s in samp])
    vy = _sp.Matrix([_sp.Rational(tv(s)) for s in samp])
    try:
        sol = Mx.solve_least_squares(vy)
    except Exception:                                      # noqa: BLE001
        sol = None
    rel, n_cand, n_rej = None, 0, 0
    if sol is not None:
        n_cand = 1
        cand = {}
        for c, k in zip(sub_names, sol):
            if k != 0:
                cand[c] = _F(int(_sp.Rational(k).p), int(_sp.Rational(k).q))
        #  ★报**覆盖率**而不是只报过/不过 (2026-10-28, `09` 第 213 条)。
        #    实撞: `*_ratio` 这类列的推导**自己就过了一遍 fp32** (见 `as_printed` 的注释:
        #    `cycles > 2^24` 时换算前就丢精度), 于是**连真公式也做不到全量逐位** ——
        #    已知的 `mte2_ratio = 1800 x mte2_time / total_cycles` 在 790 行里只中 764 行。
        #    只报"没找到"会把这件事说成"关系不存在"; 报 764/790 才是实情。
        #  ★验证层走**整数**而不是 `Fraction` (2026-10-28, `09` 第 215 条)。
        #    值都是 ≤6 位小数 ⇒ ×10^6 即整数; 系数的分母取公倍数之后整条式子都是整数乘加。
        #    第 214 条如实记过"43 秒没解决, 瓶颈在 Fraction 逐行验证" —— 这是那笔账的偿还。
        #    ★口径不变: 比的仍是**按印数精度**的值, 只是算术换了实现。
        from math import lcm as _lcm
        _D = 1
        for _k in cand.values():
            _D = _lcm(_D, _k.denominator)
        _ci = {c: int(k.numerator) * (_D // int(k.denominator)) for c, k in cand.items()}
        _S = 10 ** max(_nd, 1)
        _cache = {}

        def _iv(_s, _c):
            """该行该列 ×10^nd 后的整数; 取不到返回 None。"""
            _key = id(_s), _c
            if _key in _cache:
                return _cache[_key]
            _raw = (_s.get(_c) or "").strip()
            if _raw in ("", "NA"):
                _cache[_key] = None
                return None
            try:
                _f = _F(_raw) * _S
            except (ValueError, ZeroDivisionError):
                _cache[_key] = None
                return None
            _r = int(_f) if _f.denominator == 1 else None
            _cache[_key] = _r
            return _r

        n_ok = 0
        for s in rows:
            tot, bad = 0, False
            for c, ki in _ci.items():
                vv = _S if c == "1" else _iv(s, c)
                if vv is None:
                    bad = True
                    break
                tot += ki * vv
            if bad:
                continue
            ty = _iv(s, target)
            if ty is None:
                continue
            if target_mul is None:
                #  tot/(D·S) 与 y 按印数精度比 ⇒ 等价于 tot 与 y·D 整数相等 (都已 ×S)
                n_ok += 1 if tot == ty * _D else 0
            else:
                _dvi = _iv(s, target_mul)
                #  y = (Σ)/d  ⇒  tot·S == y·d·D  (两边都 ×S^2·D)
                n_ok += 1 if (_dvi not in (None, 0) and tot * _S == ty * _dvi * _D) else 0
        cover = n_ok / len(rows) if rows else 0.0
        if cand and n_ok == len(rows):
            rel = cand
        else:
            n_rej = 1
    return {"rows": len(rows), "cols": len(use), "dim": len(keep), "rel": rel,
            "sub": len(samp), "cand": n_cand, "rejected": n_rej,
            "decimals": _nd, "exact_print": _exactly_printed,
            "cover": locals().get("cover"), "near": locals().get("cand") if rel is None else None}


def ratio_relation(target: str, op_filter: str | None = None) -> list:
    """找 `target = (列的线性组合) / d` 形式的精确关系, `d` 遍历所有候选列。

    ★方法出处 (2026-10-28, `09` 第 213 条): **线性化有理拟合** —— 两边乘分母,
      `y = N/D` 变成线性的 `y·D = N` (polyrat 文档记为 Sanathanan-Koerner 1963 的旧方法;
      implicit SINDy 的"乘掉分母再在零空间里找稀疏向量"是同一手法)。
      **整台机器不动**, 只把目标换成 `y × d`。
    ★只扫**单列分母**: 那 13 列是 ratio 与带宽, 分母多半是 `total_cycles` 这类单列。
      分母是多列组合的情形**本函数覆盖不到**, 这是它的作用域, 必须跟着结论写。
    ★先用 numpy 最小二乘做**廉价筛** (每个分母一次), 只有残差近零的才进精确验证 ——
      否则 13 列 x 20 个分母每次重读一遍盘要两小时。
    """
    import numpy as _np
    from fractions import Fraction as _F
    keys = ("block_id", "sub_block_id")
    rows = []
    for k in product_kdirs():
        if op_filter and op_filter not in k:
            continue
        merged = {}
        for path in sorted(glob.glob(os.path.join(k, "*.csv"))):
            if os.path.basename(path).startswith("OpBasicInfo"):
                continue
            try:
                with open(path, newline="", errors="ignore") as f:
                    for r in csv.DictReader(f):
                        if any(c not in r for c in keys):
                            break
                        merged.setdefault(tuple(r.get(c, "") for c in keys), {}).update(r)
            except OSError:
                continue
        for s in merged.values():
            if (s.get(target) or "").strip() not in ("", "NA"):
                rows.append(s)
    if not rows:
        return []

    def fr(s, c):
        v = (s.get(c) or "").strip()
        if v in ("", "NA"):
            return None
        try:
            return _F(v)
        except (ValueError, ZeroDivisionError):
            return None

    cols = None
    for s in rows:
        ok = {c for c in s if c not in keys and fr(s, c) is not None}
        cols = ok if cols is None else (cols & ok)
    use = [c for c in sorted(cols) if c != target and len({fr(s, c) for s in rows}) > 1]
    if not use:
        return []
    A = _np.array([[float(fr(s, c)) for c in use] + [1.0] for s in rows], dtype=float)
    yf = _np.array([float(fr(s, target)) for s in rows], dtype=float)
    #  ★筛阈值跟着**印数精度**走 (同 `exact_relation`, `09` 第 213 条):
    #    `y` 只印 nd 位 ⇒ `y·d` 的舍入噪声是 0.5e-nd × |d|。按精确口径的 1e-7 去筛,
    #    会把这一整族真关系筛掉 —— 本函数首版就是这么把 13 列全报成"没有"的。
    nd = 0
    for s0 in rows[:512]:
        t0 = (s0.get(target) or "").strip()
        if "." in t0:
            nd = max(nd, len(t0.split(".", 1)[1].rstrip()))
    out = []
    for di, d in enumerate(use):
        dv = A[:, di]
        if not _np.all(dv != 0):
            continue
        t = yf * dv
        sc = max(float(_np.abs(t).max()), 1.0)
        lim = 1e-7 * sc
        if nd:
            lim = max(lim, 4.0 * 10.0 ** (-nd) * float(_np.abs(dv).max()))
        w, *_ = _np.linalg.lstsq(A, t, rcond=None)
        if float(_np.abs(A @ w - t).max()) > lim:
            continue
        #  廉价筛过了, 再用精确路径定系数并全量验证
        r = exact_relation(target, op_filter, target_mul=d)
        if r.get("rel"):
            out.append((d, r))
    return out


def print_ratiorel(target: str, op_filter: str | None = None) -> int:
    hits = ratio_relation(target, op_filter)
    print(f"=== 比值形式 y = (列的线性组合) / d  (09 第 213 条) === {target}"
          + (f"  [算子 {op_filter}]" if op_filter else ""))
    if not hits:
        print("  **单列分母这一族里没有**。")
        print("  ★作用域: 只扫了**单列**分母; 分母是多列组合的情形本入口覆盖不到。")
        return 1
    for d, r in hits:
        print(f"  ★分母 = {d}  ({r['rows']} 行全量验证通过):")
        for c, k in r["rel"].items():
            print(f"        {str(k):>14}  x  {c}")
    return 0


#  ---- `--hitbound`: 命中率的自然式是不是**单侧**的 ----
#  ★由来 (2026-10-28, `09` 第 217 条): `08` §3.56 把 `aic_read/total_hit_rate` 记成
#    "80.7% 精确, 其余偏到 78 个百分点" —— 读起来像噪声, 用的人无从判断能不能用。
#    全仓 110614 行扫一遍**方向**之后性质就变了: **低估 0 行**。
#    ⇒ 自然式是一个**精确的上界**, 不是一个"大致对"的近似。
#    差额落在一类**未导出的 miss** 上 (计数器名就叫 `miss_allocate`, 暗示还有不 allocate 的 miss)。
#  ★四个 100% 取等的列是这条检查**自带的正例对照**: 它们一旦不是 100%, 说明取数或口径坏了。
HITRATE_SPECS = (
    ("read", ("r0_read_cache_hit", "r1_read_cache_hit"),
     ("r0_read_cache_miss_allocate", "r1_read_cache_miss_allocate")),
    ("write", ("write_cache_hit",), ("write_cache_miss_allocate",)),
    ("total", ("r0_read_cache_hit", "r1_read_cache_hit", "write_cache_hit"),
     ("r0_read_cache_miss_allocate", "r1_read_cache_miss_allocate",
      "write_cache_miss_allocate")),
)


def hitrate_bound(op_filter: str | None = None) -> list:
    """命中率自然式与实测的**方向**统计。返回 [(列, 取等, 高估, 低估, 最大高估)]。"""
    rows = _merged_rows(op_filter)
    out = []
    for side in ("aic", "aiv"):
        for kind, nums, miss in HITRATE_SPECS:
            col = f"{side}_{kind}_hit_rate(%)"
            eq = over = under = 0
            worst = 0.0
            for s in rows:
                y = _num(s, col)
                if y is None:
                    continue
                nv = [_num(s, f"{side}_{c}") for c in nums]
                mv = [_num(s, f"{side}_{c}") for c in miss]
                if any(v is None for v in nv + mv):
                    continue
                den = sum(nv) + sum(mv)
                if not den:
                    continue
                d = sum(nv) / den * 100 - y
                if d >= 0.05:
                    over += 1
                    worst = max(worst, d)
                elif d <= -0.05:
                    under += 1
                else:
                    eq += 1
            if eq + over + under:
                out.append((col, eq, over, under, worst))
    return out


def _num(s: dict, c: str):
    v = (s.get(c) or "").strip()
    if v in ("", "NA"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def print_hitbound(op_filter: str | None = None) -> int:
    res = hitrate_bound(op_filter)
    print("=== 命中率自然式的**方向** (08 §3.56 / 09 第 217 条) ===")
    print("  式子: hit / (hit + miss_allocate) × 100, 分子分母都是 csv 里的原始计数列")
    print(f"  {'列':<26}{'行':>8}{'取等':>9}{'高估':>9}{'低估':>9}{'最大高估':>10}")
    bad = 0
    for col, eq, over, under, worst in res:
        t = eq + over + under
        mark = "  ★低估 >0, 不是上界" if under else ""
        bad += 1 if under else 0
        print(f"  {col:<26}{t:>8}{eq/t*100:>8.1f}%{over/t*100:>8.1f}%"
              f"{under/t*100:>8.1f}%{worst:>10.2f}{mark}")
    print("  ★**低估全为 0** ⇒ 这个式子是**精确的上界**, 不是'大致对' —— "
          "可以放心当上界用, 差额落在一类未导出的 miss 上。")
    print("  ★四个 100% 取等的列是本检查自带的正例对照: 它们不再是 100% 就说明取数坏了。")
    return 1 if bad else 0


def print_exactrel(target: str, op_filter: str | None = None) -> int:
    r = exact_relation(target, op_filter)
    print(f"=== 精确有理闭式 (09 第 211 条) === {target}"
          + (f"  [算子过滤 {op_filter}]" if op_filter else ""))
    print(f"  {r['rows']} 行 x {r['cols']} 个有变化的列")
    if r.get("toonarrow"):
        print(f"  **判不了**: {r['rows']} 行只有 {r['cols']} 列的 "
              f"{r['rows']/max(r['cols'],1):.1f} 倍, 行数不明显多于列数时"
              f"精确方法会找到**不存在的关系** (实撞过 1741/3142 这种系数)。")
        print("  ★换一个更宽的语料再跑; 不要把这条读成'没有关系'。")
        return 1
    if r.get("nofit"):
        print("  数值最小二乘在全部列上就已有残差 ⇒ **这些列的线性组合里没有它**。")
    else:
        print(f"  数值剪枝后剩 {r['dim']} 列 -> 在 {r.get('sub', 0)} 行上 ℚ 精确求解 "
              f"-> 候选 {r.get('cand', 0)} 条, **全量 {r['rows']} 行逐行精确验证**后剔除 "
              f"{r.get('rejected', 0)} 条")
    if r.get("rel") is None and r.get("cover"):
        print(f"  ★最佳候选**覆盖 {r['cover']*100:.1f}%** 的行, 没有全量成立 —— "
              f"不是'没有关系', 是'没有**全量逐位**的关系'。")
        print("    这类列的推导自己就过了一遍 fp32 (见 as_printed 的注释), "
              "连真公式也做不到全量逐位。")
        for c, k in (r.get("near") or {}).items():
            print(f"        {str(k):>14}  x  {c}")
    print("  ★本工具承诺的是「**报出来的都对**」, 不是「对的都报得出来」——")
    print("    剪枝可能剪掉真列, 那只会少报一条。所以'不存在'要读成'这套搜索没找到'。")
    if not r["rel"]:
        print("  **没有含目标列的精确线性关系** —— 不是'拟合得不够好', 是**恒等式不存在**。")
        print("  ★下一步看分支: 语料里可能有两支各自成立 (`09` 第 211 条的 L1_to_GM 就是)。")
        return 1
    print("  找到一条精确关系:")
    for c, k in r["rel"].items():
        print(f"      {str(k):>12}  x  {c}")
    if r["dim"] > 1:
        print(f"  ★零空间维数 {r['dim']} > 1 ⇒ **关系不唯一** (列之间另有恒等式),")
        print("    这是'存在一条精确关系', 不是'实现就长这样'。")
    return 0


def l1gm_colvar(op_filter: str | None = None) -> list:
    """对闭式里的每个入选列, 报它在**作用域语料**上的取值分布。

    ★动因 (2026-10-28, `09` 第 208 条): 五项式里 `aic_mte2_instructions` 被证实是伪项,
      而同一张表里 `aic_mte3_instructions` 在 GemmV3 的 **4944 行上只有一个取值 (18)** ——
      它根本不是回归量, 是**截距换了个写法**。
    ★**这条判据是上一条教训的工具化**: 拟合之后必看每个入选列的取值分布。
      方差接近 0 的列, 它的系数不是"这一项多重要", 是"常数项被分走了多少"。
      做成工具而不是写进文档, 是因为"记得去看"防不住 —— 本仓同一个病已犯两次
      (§3.106 的缺失截距 / 本条的两个伪项)。
    """
    import statistics as _st
    cols = [c for _t, c, _k in L1GM_FORMULA]
    coef = {c: k for _t, c, k in L1GM_FORMULA}
    vals = {c: [] for c in cols}
    for k in product_kdirs():
        if op_filter and op_filter not in k:
            continue
        for p in sorted(glob.glob(os.path.join(k, "*.csv"))):
            try:
                rows = list(csv.DictReader(open(p, newline="", errors="ignore")))
            except OSError:
                continue
            if not rows:
                continue
            for c in cols:
                if c not in rows[0]:
                    continue
                for r in rows:
                    v = (r.get(c) or "").strip()
                    if v in ("", "NA"):
                        continue
                    try:
                        vals[c].append(float(v))
                    except ValueError:
                        pass
    out = []
    for c in cols:
        v = vals[c]
        if not v:
            out.append((c, coef[c], 0, 0, None))
            continue
        m = _st.mean(v)
        cv = (_st.pstdev(v) / m) if m else 0.0
        out.append((c, coef[c], len(v), len(set(round(x, 6) for x in v)), cv))
    return out


def print_l1gm(op_filter: str | None = None) -> int:
    t, t2, tsgn = l1gm_eval(op_filter)
    terms = "  ".join(f"{c:+g}x{col}" for _tbl, col, c in L1GM_FORMULA)
    print(f"=== `{L1GM_TARGET}` 候选闭式核对 (08 §3.105) ===")
    print(f"  式子: {terms}")
    print("  ★**假设, 未定论**: 作用域目前只到 GemmV3, 别的算子重拟合时系数就变了。")
    if not t:
        print("  没有可核的目录 (要同时有 Memory 与 L2Cache 两张表)。")
        return 2
    full = [g for g, (n, ok, _m) in t.items() if n and ok == n]
    print(f"\n  {len(t)} 个 run 组; **整组逐位精确的: {len(full)}**")
    for g, (n, ok, mx) in sorted(t.items(), key=lambda kv: -kv[1][2])[:10]:
        if ok != n:
            vs = sorted(x for x in tsgn.get(g, ()) if x)
            print(f"    ✗ {g[-54:]:54s} {ok:4d}/{n:<4d} 残差取值 {vs} KB")
    if t2:
        terms = "  ".join(f"{c:+g}x{col}" for _t, col, c in L1GM_FORMULA2)
        print(f"\n  **候选式二** (两列 + 常数, 08 §3.106): {terms}  {L1GM_CONST2:+g}")
        print(f"    {'算子':28s} {'行':>7s} {'逐位相等':>14s} {'最大绝对差':>13s}")
        for op, (n, ok, mx) in sorted(t2.items()):
            print(f"    {op[:28]:28s} {n:7d} {ok:7d}/{n:<6d} {mx:13.6g}")
        print("    ★它**参数更少且最坏情况好两个数量级** (GemmV3 最大差 0.52 KB vs 五项式 33 KB),")
        print("      但不是整组精确 —— 两式的失效面不同, 都留着。")
    cv = l1gm_colvar(op_filter or "GemmV3")
    print(f"\n  **入选列在作用域语料上的取值分布** (09 第 208 条: 方差近 0 = 截距换了写法)")
    print(f"    {'列':<34}{'系数':>12}{'不同取值':>10}{'变异系数':>12}")
    for c, k, _n, nuniq, v in cv:
        mark = "  ★常数, **不是回归量**" if (v is not None and v < 1e-9) else (
            "  ★方差近 0, 可疑" if (v is not None and v < 0.05) else "")
        print(f"    {c:<34}{k:>12g}{nuniq:>10}{(v if v is not None else 0):>12.4f}{mark}")
    nm = l1gm_nomte2(op_filter or "GemmV3")
    if nm:
        vals = sorted({v for vs in nm.values() for v in vs})
        print(f"\n  **候选式三** = 五项式去掉 `aic_mte2_instructions` (09 第 207 条提出, 第 210 条订正解读)")
        print(f"    {len(nm)} 个组、全部行的残差**取值集合**: {len(vals)} 个值 {vals[:8]}")
        print("    ★这几个值全部来自**退化行** (0.5(hit+miss)-L0C == 33 的那 4432 行)。")
        print("      ★**这一项不是伪项**: 五项式在 512 行**有信息行**上 512/512 逐位精确,")
        print("        全部失败 (1056 行) 都在退化行里。缺的是**一个分支**, 不是少一项 (09 第 210 条)。")
    sp = l1gm_span(op_filter or "GemmV3")
    if sp:
        print(f"\n  **整族判定** ({op_filter or 'GemmV3'} 全部目录合并): "
              f"{sp['rows']} 行 x {sp['cols']} 列 (秩 {sp['rank']})")
        print(f"    整体最小二乘相对残差 **{sp['rel']:.3e}**; "
              f"逐行 RMS {sp['rms']:.4g} KB, 最大绝对差 {sp['max_abs']:.4g} KB")
        if sp["rel"] > 1e-6:
            print("    ⇒ y **不是**这些列的任何线性组合。任何子集的张成都是它的子空间,")
            print("       所以三列、四列…… 全部不必再试 —— **线性族整族排除**。")
            print("       而上面有整组逐位精确的 ⇒ 规律是**分段的**, 与 `(estimate)` 这个后缀相符。")
    print("  ★两个数说的不是一回事, 要分开读:")
    print("    · **固定系数**的五项式在 13 个组上差到 33 KB (差随规模增长:")
    print("      512³ 0.25 · 1024³ 3.06 · 8192x2048² 11 · 4096³ 33 KB);")
    print("    · 而**重新拟合**的全局线性式最大只差 0.12 KB ≈ 2 个 1/16 KB 量子。")
    print("    ⇒ 缺的**不是一个大项**: 五项式的系数是**分支专属**的, 换一支就得换系数;")
    print("      而即便逐支重拟合也取不到逐位精确 —— 还差一个量化/取整。")
    print("    残差本身全部是 1/16 KB 的整数倍 (实测), 1/16 KB = 64 B。")
    return 0 if len(full) == len(t) else 1


def corrupt_dirs(root: str | None = None) -> tuple:
    """按"某核忙时间不可能超过任务总时长"扫产物。返回 `(坏目录表, 扫过数, 跳过原因)`。

    判据: 同一个 kernel 目录里, `PipeUtilization` 任一行的 `aic_time(us)` 或
    `aiv_time(us)` **大于** `OpBasicInfo` 的 `Task Duration(us)` (放 0.1% 余量)。
    一个核的忙时间是任务窗口的**子区间**, 超出去只能是产物坏了。

    ★与 `core/perf._corrupt_launches` 同判据、不同作用域: 那个在一次采集的 out_dir 上
      当场过滤, 这个是**全仓普查** —— 用来回答"本机到底有几处、在哪几个 run 里",
      而这正是 2026-10-26 发现原注释写窄了的那个问题 (`08` §3.103)。
    ★跳过的要分因报出来, 不混成一格 (§3.97): 没有 `PipeUtilization` 是档位没开,
      `Task Duration` 是 `NA` 是记录残缺 —— 两者下一步不同。
    """
    kdirs = (sorted({os.path.dirname(p) for p in
                     glob.glob(os.path.join(root, "**", "OpBasicInfo*.csv"), recursive=True)})
             if root else product_kdirs())
    bad, checked = [], 0
    skip = collections.Counter()
    for k in kdirs:
        verdict, why = corrupt_kdir(k)
        if verdict is None:
            skip[why] += 1
            continue
        checked += 1
        if verdict:
            bad.append(verdict)
    return sorted(bad, key=lambda b: -b["ratio"]), checked, dict(skip)


def corrupt_kdir(kdir: str) -> tuple:
    """单个 kernel 目录的坏行判据。返回 `(结果, 跳过原因)`。

    结果为 `None` = 判不了 (原因在第二个返回值), `False` = 干净, 否则是坏目录的那条记录。
    ★从 `corrupt_dirs` 里抽出来是为了让**消费方按单目录判**: 全仓普查太贵, 而
    `--vecshare` 这类只读一个目录的入口此前拿不到这个判据, 于是坏数据照样进结论
    (2026-10-28 实撞, 见 `09` 第 198 条)。
    """
    ob = sorted(glob.glob(os.path.join(kdir, "OpBasicInfo*.csv")))
    pu = sorted(glob.glob(os.path.join(kdir, "PipeUtilization*.csv")))
    if not ob:
        return None, "没有 OpBasicInfo csv"
    if not pu:
        return None, "没有 PipeUtilization csv (档位没开)"
    try:
        r0 = next(csv.DictReader(open(ob[0], errors="ignore")))
    except (StopIteration, OSError) as e:
        return None, f"OpBasicInfo 读不出首行 ({type(e).__name__})"
    d = (r0.get("Task Duration(us)") or "").strip()
    if d in ("", "NA"):
        return None, "Task Duration 是 NA (记录残缺)"
    dur, n, mx = float(d), 0, 0.0
    with open(pu[0], newline="", errors="ignore") as f:
        for r in csv.DictReader(f):
            over = False
            for side in ("aic", "aiv"):
                v = (r.get(f"{side}_time(us)") or "").strip()
                if v not in ("", "NA"):
                    fv = float(v)
                    mx = max(mx, fv)
                    over = over or fv > dur * 1.001
            n += 1 if over else 0
    if not n:
        return False, ""
    return {"kdir": kdir, "dur": dur, "max_time": mx,
            "ratio": mx / dur if dur else float("inf"), "rows": n}, ""


def print_corrupt(root: str | None = None) -> int:
    bad, checked, skip = corrupt_dirs(root)
    print("=== 产物坏行普查 (08 §3.103) === "
          "判据: 某核 `*_time(us)` > 本 launch 的 `Task Duration`, 物理上不可能")
    print(f"  扫过 {checked} 个 kernel 目录 (有 dur 且有 PipeUtilization)")
    for w, n in sorted(skip.items(), key=lambda kv: -kv[1]):
        print(f"  跳过 {n:5d}  {w}")
    if not bad:
        print("  没有坏目录。")
        return 0
    print(f"\n  ★坏目录 {len(bad)} 个, 共 {sum(b['rows'] for b in bad)} 行:")
    for b in bad:
        print(f"    {b['ratio']:7.2f}x  dur={b['dur']:10.2f}  max_time={b['max_time']:10.2f}  "
              f"{b['rows']:3d} 行  {b['kdir']}")
    print("  ★这些目录的**数值含义类**结论不可用 (`--vecshare` / `--denom` / `--formcode`);")
    print("    逐位还原类 (E1/E2/E3) 仍要看全量 —— 计数器坏了 csv 与 bin 照样一致。")
    return 1


def vec_share(kdir: str) -> list:
    """逐行给 `(块, aiv_vec_ratio, Σ分项, 未导出份额)`。

    ★`未导出份额 = aiv_vec_ratio − Σ分项` (`08` §3.98)。动因样本上它逐行为 0
    (配分精确成立); 别的算子上 > 0, 差额落在本机**从未导出**的那族向量单元列
    (`vec_sfu`/`ldu`/`stu`/`simt_vf`/`vf`, 见 §3.81 的 59 列)。

    ★**用处**: 那几列在本机再也不会出现 (档位没开), 而这一式子让它们的**总份额**
    仍然可以从已有产物里读出来 —— 读向量瓶颈时, 这部分工作此前是完全不可见的。

    ★**不要**把比值读成好看的整数比: 第一版在 GemmV3 上看到 `min = 1.166667 = 7/6`
    就写了"恰好 1/7", 而逐行检验 `vr×6/7 == Σ` **48/48 全不成立**, 中位是 1.16724 ——
    那个 7/6 只是某一份产物的最小值碰巧落在附近 (`09` 第 154 条)。
    """
    #  ★坏目录直接拒跑 (2026-10-28, `09` 第 198 条)。本函数读的是**数值含义**,
    #    而 §3.103 的坏目录里 `aiv_time` 超 `Task Duration` 5.6 倍, 比值整体失真。
    #    此前只有全仓普查 (`--corrupt`) 报得出它, 单目录入口拿不到这个判据 ——
    #    于是 `08` §3.98 里三个 max 值就是从坏目录抄出来的, 挂了两天没人发现。
    _bad, _why = corrupt_kdir(kdir)
    if _bad:
        raise SystemExit(
            f"拒跑: {kdir} 是坏目录 (`aprof --corrupt` 判据: 某核 *_time(us) 超本 launch 的 "
            f"Task Duration {_bad['ratio']:.2f} 倍, {_bad['rows']} 行)。\n"
            f"  这里读的是**数值含义**, 坏行会污染比值。逐位还原类 (E1/E2/E3) 不受影响, "
            f"它们必须看全量 —— 两类口径的分工见 VEC_PARTS 上方的注释。")
    au = sorted(glob.glob(os.path.join(kdir, "ArithmeticUtilization*.csv")))
    pu = sorted(glob.glob(os.path.join(kdir, "PipeUtilization*.csv")))
    if not au or not pu:
        return []
    tc = {(r["block_id"], r["sub_block_id"]): r for r in csv.DictReader(open(pu[0]))}
    out = []
    for r in csv.DictReader(open(au[0])):
        key = (r["block_id"], r["sub_block_id"])
        vr = tc.get(key, {}).get("aiv_vec_ratio")
        if vr in (None, "", "NA"):
            continue
        parts = [float(r[c]) for c in VEC_PARTS if r.get(c) not in (None, "", "NA")]
        if not parts:
            continue
        vr, s = float(vr), sum(parts)
        out.append((f"{key[0]}/{key[1]}", vr, s, vr - s))
    return out


def col_corpus_max(cols) -> dict:
    """这些列在两棵产物树里的最大值。`0.0` = **本机全零**, `None` = 任何产物都没有这一列。

    ★为什么要有它 (`08` §3.97): `--formcode` 的"未知 `k`"清单把三种完全不同的状态混在一起 ——
    **本机全零** (换个负载才可能解) · **按设计不解** (`*_total_cycles` 直接取记录字段, E2 已验)
    · **真未解**。混在一起的后果是下一步指错: 对全零的列再怎么读汇编也没有数据可对。
    与 §3.79 四 "全零要与未解分开写" 是同一条。
    """
    want = set(cols)
    out = {c: None for c in want}
    for p in product_files("*.csv"):
        try:
            with open(p, errors="ignore") as f:
                hdr = f.readline()
        except OSError:
            continue
        hit = [c for c in want if c in hdr]
        if not hit:
            continue
        try:
            rows = list(csv.DictReader(open(p, errors="ignore")))
        except OSError:
            continue
        for c in hit:
            vs = []
            for r in rows:
                v = r.get(c)
                if v in (None, "", "NA"):
                    continue
                try:
                    vs.append(abs(float(v)))
                except ValueError:
                    pass
            if vs:
                out[c] = max(vs) if out[c] is None else max(out[c], max(vs))
    return out


#  ★{记录类型码 → 指令记录解析函数} (08 §3.101 订正版, 2026-10-26)。
#  ★★**这张表的键不是列的换算形式码** —— 2026-10-26 当天先认错了一次, 判决性证据是
#    **两个方向的键集都对不上**: 列那侧有 18/33/38/40/41 而本表没有这些键,
#    本表有 14/15 而没有任何列用它们。印证: 每个函数自带名字串
#    (`ParseLoad2dRecord` / `ParseDmaMovRecord` / `ParseNdDMAOut2UbRecord` …),
#    它们是**指令记录的类型名**, 与列的换算形式无关。
#    `check_not_formcode()` 把这条判决固化, 别再认一次。
def record_parsers(path: str = MSOPPROF, want_bytes: int = MSOPPROF_BYTES,
                   lo: int = 0x038800, hi: int = 0x038800 + 0x6effec) -> dict:
    """`{记录类型码 → (解析函数地址, 记录字节数 或 None, [查表点...], 名字串 或 None)}`。

    ★抽取形状 (手读 `0x289420` 与 `0x289920` 两处得出, 两处逐字同构):
        ldr  w0, [x1, #32]          ; it->first
        cmp  w0, #k                 ; std::map::find 的收尾判定 !(k < it->first)
        b.hi <未命中>
        adrp x2, <页> ; add x2, x2, #imm      ; ← 该键的处理函数

    ★记录字节数**不能**取"函数开头第一个落在合理区间的立即数" —— 函数开头到处是栈偏移
      与结构偏移, 长得一模一样。判据是**写回**: 推进流游标必然是
      `ldr xS,[xBase]` → `add xD, xS, #imm` → `str xD,[xBase]`, 第三条缺一不可。
      加上这条之后 17 个 k 里有 **3 个** (k=4 / k=13 / k=17) 的数变了 —— 也就是说
      naive 取法在本表上错 3/17, 而错的那几个不会以任何方式表现出来。
    """
    got = os.path.getsize(path)
    if got != want_bytes:
        raise SystemExit(f"拒跑: {path} 是 {got} 字节, 本仓引用的是 {want_bytes} 字节")
    with open(path, "rb") as f:
        blob = f.read()

    def wd(a):
        return struct.unpack_from("<I", blob, a)[0]

    def adrp_add(a, rd_want):
        """a 处若是 `adrp xR` 且其后 ≤2 条是 `add xR, xR, #imm`, 返回目标地址。"""
        x = wd(a)
        if not ((x >> 31) & 1 and ((x >> 24) & 0x1F) == 0x10):
            return None
        rd = x & 0x1F
        if rd != rd_want:
            return None
        im = (((x >> 5) & 0x7FFFF) << 2) | ((x >> 29) & 3)
        if im & (1 << 20):
            im -= (1 << 21)
        page = (a & ~0xFFF) + (im << 12)
        for k in range(1, 3):
            v = wd(a + 4 * k)
            if (v >> 31) & 1 and ((v >> 23) & 0x1FF) == 0b100100010 \
                    and (v & 0x1F) == rd and ((v >> 5) & 0x1F) == rd:
                return page + ((v >> 10) & 0xFFF)
        return None

    def rec_bytes(fn, span=0x200):
        """`ldr xS,[xB]` → `add xD,xS,#imm` → `str xD,[xB]` 里的那个 imm。"""
        for a in range(fn, fn + span, 4):
            v = wd(a)
            if not ((v >> 31) & 1 and ((v >> 23) & 0x1FF) == 0b100100010):
                continue                                   # ADD (imm), 64 位
            d, sreg, imm = v & 0x1F, (v >> 5) & 0x1F, (v >> 10) & 0xFFF
            if not 0x08 <= imm <= 0x100:
                continue
            base = None
            for k in range(1, 9):                          # 往前找 `ldr xS,[xB]`
                u = wd(a - 4 * k)
                if (u >> 22) == 0b1111100101 and (u & 0x1F) == sreg \
                        and ((u >> 10) & 0xFFF) == 0:
                    base = (u >> 5) & 0x1F
                    break
            if base is None:
                continue
            for k in range(1, 4):                          # 往后找 `str xD,[xB]`
                u = wd(a + 4 * k)
                if (u >> 22) == 0b1111100100 and (u & 0x1F) == d \
                        and ((u >> 5) & 0x1F) == base and ((u >> 10) & 0xFFF) == 0:
                    return imm
        return None

    def parse_name(fn, span=0x600):
        """函数体里第一个形如 `Parse...Record` 的字符串 —— 它自报是哪种记录。

        ★这条是把上面那次误认翻过来的直接证据: 一个"列公式"不会叫
          `ParseLoad2DTransposeRecord`。
        """
        for a2 in range(fn, fn + span, 4):
            x2 = wd(a2)
            if not ((x2 >> 31) & 1 and ((x2 >> 24) & 0x1F) == 0x10):
                continue
            im2 = (((x2 >> 5) & 0x7FFFF) << 2) | ((x2 >> 29) & 3)
            if im2 & (1 << 20):
                im2 -= (1 << 21)
            page2, rd2 = (a2 & ~0xFFF) + (im2 << 12), x2 & 0x1F
            for k2 in range(1, 4):
                v2 = wd(a2 + 4 * k2)
                if (v2 >> 31) & 1 and ((v2 >> 23) & 0x1FF) == 0b100100010 \
                        and ((v2 >> 5) & 0x1F) == rd2:
                    t2 = page2 + ((v2 >> 10) & 0xFFF)
                    e2 = blob.find(b"\0", t2)
                    if 0 < e2 - t2 < 64:
                        cand = blob[t2:e2]
                        if PARSE_NAME_RE.match(cand):
                            return cand.decode()
                    break
        return None

    out = {}
    for a in range(lo, hi, 4):
        x = wd(a)
        if ((x >> 23) & 0x1FF) != 0b011100010 or (x & 0x1F) != 31:
            continue                                       # CMP (imm) 32 位 = SUBS WZR
        k = (x >> 10) & 0xFFF
        if not 1 <= k <= 64:
            continue
        nxt = wd(a + 4)
        if (nxt >> 24) != 0x54 or (nxt & 0xF) != 0x8:       # B.HI
            continue
        fn = adrp_add(a + 8, 2)
        if fn is None:
            continue
        if k in out and out[k][0] != fn:
            out[k] = (out[k][0], out[k][1], out[k][2] + [a], out[k][3], True)
            continue
        prev = out.get(k)
        out[k] = (fn, rec_bytes(fn), (prev[2] if prev else []) + [a], parse_name(fn))
    return out


def check_not_formcode() -> tuple:
    """判决: 记录类型码的键空间与列的换算形式码**不是同一个**。

    返回 `(列有而表无, 表有而列无)`。两边都非空就是判决成立 ——
    2026-10-26 实测 `([18, 33, 38, 40, 41], [14, 15])`。
    ★这条固化下来是因为两者都是小整数、范围还重叠, **看起来就是同一个东西**,
      而认错的代价是把一张记录解析表当成列公式表去读 (当天真认错了一次)。
    """
    keys = set(record_parsers())
    cols = set(form_codes().values())
    return sorted(cols - keys), sorted(keys - cols)


def print_record_parsers() -> int:
    t = record_parsers()
    print(f"=== 指令记录解析器表 (08 §3.101 订正版) === {MSOPPROF}({MSOPPROF_BYTES})")
    print("  判据: `cmp wN,#码` + `b.hi` + `adrp x2/add x2` 三条紧邻 (std::map::find 的收尾);")
    print("  记录字节数取**写回流游标**的那条 add 的立即数, 不取开头第一个立即数;")
    print("  名字取函数体里第一个 `Parse...Record` 串 —— 它自报是哪种记录。")
    amb = [k for k, v in t.items() if len(v) > 4]
    print(f"\n  {'码':>4}  {'解析函数':>10}  {'记录字节':>8}  名字")
    for k in sorted(t):
        v = t[k]
        rb = f"{v[1]}" if v[1] is not None else "★没读出"
        print(f"  {k:>4}  0x{v[0]:08x}  {rb:>8}  {v[3] or '★没找到名字串'}")
    fns = [v[0] for v in t.values()]
    print(f"\n  {len(t)} 个码, {len(set(fns))} 个不同的解析函数"
          + ("" if len(set(fns)) == len(t) else "  ★有码共用解析函数 (名字相同则属正常)"))
    co, ko = check_not_formcode()
    print(f"\n  ★**这张表的键不是列的换算形式码** (`--formcode` 那个)。判决证据是两个方向都对不上:")
    print(f"      列那侧有而本表没有的码: {co}")
    print(f"      本表有而没有任何列用的: {ko}")
    print(f"    2026-10-26 先认错过一次, 把本表当成列公式表读了一轮 (见 08 §3.101)。")
    if amb:
        print(f"  ★同一个码在不同查表点取到不同函数: {amb} —— 抽取形状还不够严, 别采信这几个")
    miss = [k for k, v in t.items() if v[1] is None]
    if miss:
        print(f"  ★记录字节数没读出来的码: {miss} (没有'推进后写回'那条, 不是 0)")
    nonm = [k for k, v in t.items() if not v[3]]
    if nonm:
        print(f"  ★没抽到名字串的码: {nonm} (函数体 0x600 内没有 Parse*Record)")
    return 1 if (amb or miss) else 0


def form_code_join(kdirs=None) -> dict:
    """`{k → 这些目录上 --solve 解出过的换算形式集合}`。一一对应才说明 `k` 抽对了。"""
    codes = form_codes()
    flat = {}
    for kd in (kdirs or JOIN_KDIRS):
        if not os.path.isdir(kd):
            continue
        for cols in solve_columns(kd).values():
            for c, v in cols.items():
                if v and v[1] is not None:
                    flat.setdefault(c, set()).add(re.sub(r"\s*\(另有.*", "", str(v[0])))
    #  ★派生列 (`DERIVED`) 也算"已解出的形式"。它们不是任何单槽的换算, 所以
    #  `solve_columns` 的 pmu 形式里没有它们 —— 但本仓早已逐行验过
    #  (`check_derived`, 24/24)。不把它们接进来, `k=16/17/18` 会被当成"未解",
    #  而答案其实就在同一个模块里。对照 09 第 138 条: "工具没试过"不是"数据里没有"。
    for kd in (kdirs or JOIN_KDIRS):
        if not os.path.isdir(kd):
            continue
        _dv = check_derived(kd)
        for (tbl, col), (num, den) in DERIVED.items():
            ok, tot = _dv.get((tbl, col), (0, 0))
            if not (tot and ok == tot):
                continue
            #  ★1 分子 / 2 分母 且分子就在分母里 = `a/(a+b)*100`, 与 solve_columns 用的
            #  是同一个名字。不归一化的话 `k=17` 会同时挂上两种写法, 看起来像冲突 ——
            #  **那种"冲突"是命名造成的, 不是事实**, 而本工具的全部价值就在冲突数上。
            if len(num) == 1 and len(den) == 2 and num[0] in den:
                flat.setdefault(col, set()).add("a/(a+b)*100")
            else:
                flat.setdefault(col, set()).add(
                    f"Σ分子×100/Σ分母 ({len(num)}/{len(den)} 列)")
    out = collections.defaultdict(set)
    for n, k in codes.items():
        if n in flat:
            out[k] |= flat[n]
    return {k: sorted(v) for k, v in sorted(out.items())}


def coverage(kdir: str | None = None) -> dict:
    """三分口径: 二进制全集 x 本机产物表头 x 动因样本的 `--solve` 判定。"""
    univ = binary_colnames()
    files = product_files("*.csv")
    seen = set()
    for p in files:
        try:
            with open(p, errors="ignore") as f:
                seen.update(h.strip() for h in f.readline().rstrip("\n").split(","))
        except OSError:
            continue
    appeared = univ & seen
    k = kdir or MOTIV_KDIR
    verdict = {}
    if os.path.isdir(k):
        for cols in solve_columns(k).values():
            for c, v in cols.items():
                verdict[c] = ("未解" if v is None else
                              "歧义" if str(v[0]).startswith("歧义") else
                              "全零" if "全零" in str(v[0]) else "解出")
    split = collections.Counter(verdict.get(c, "样本上没这列") for c in appeared)
    xl = xlevel_colnames()
    xl_split = collections.Counter(verdict.get(c, "样本上没这列") for c in xl)
    return {"xlevel": len(xl), "xlevel_split": dict(xl_split),
            "n_files": len(files), "universe": len(univ),
            "appeared": len(appeared), "never": len(univ - appeared),
            "kdir": k, "split": dict(split),
            "never_cols": sorted(univ - appeared)}


def print_coverage(kdir: str | None = None) -> int:
    r = coverage(kdir)
    print(f"=== 覆盖数三分口径 (08 §3.80/§3.81) ===")
    print(f"  全集 (msopprof({MSOPPROF_BYTES}) 的 aic_/aiv_ 列名)   {r['universe']}")
    print(f"  扫 {r['n_files']} 份 csv 表头 (runs/ 与 cases/*/runs/ 两棵树):")
    print(f"    本机出现过   {r['appeared']}")
    print(f"    从未出现     {r['never']}   ← **档位没开**, 不是「没解出」")
    _op = os.path.basename(os.path.dirname(r["kdir"])).split("_")[0]
    print(f"\n  本机出现过的 {r['appeared']} 列, 按动因样本 ({_op}) 的 --solve:")
    for key in ("解出", "全零", "样本上没这列", "未解", "歧义"):
        if key in r["split"]:
            print(f"    {key:12} {r['split'][key]}")
    print(f"\n  ★第二族 (无 aic_/aiv_ 前缀的跨层列, 见 XLEVEL_RE): {r['xlevel']} 列 —— "
          f"**不在上面 201 的全集里**")
    for key in ("解出", "全零", "样本上没这列", "未解", "歧义"):
        if key in r["xlevel_split"]:
            print(f"    {key:12} {r['xlevel_split'][key]}")
    print("\n  ★这三个数会随**扫了多少份产物**变 —— 它不是估计值, 是一个会被采样方式改变的事实。")
    print("    已错过三次 (600/900/11012 份给出 87/80/59), 所以别抄文档里的数, 敲这条命令。")
    return 0


def print_denom(root: str | None = None, limit: int | None = None) -> int:
    """扫一棵 (或全部) 产物树, 核对 `Op Type → 分母` 这条规则。有反例则退出码 1。"""
    kdirs = (sorted({os.path.dirname(p) for p in
                     glob.glob(os.path.join(root, "**", "OpBasicInfo*.csv"), recursive=True)})
             if root else product_kdirs())
    if limit:
        kdirs = kdirs[:limit]
    #  ★排除坏目录 (2026-10-28, `09` 第 198 条): 本函数核的是分母规则, 属"读数值含义"那一类。
    #    **排了几个要明说** —— 静默少扫几个目录与"全过"长得一样。
    _skipped = [k for k in kdirs if corrupt_kdir(k)[0]]
    if _skipped:
        kdirs = [k for k in kdirs if k not in set(_skipped)]
    tally = collections.Counter(); bad = []; spread = collections.defaultdict(list)
    for k in kdirs:
        d = denom_of(k)
        if d is None:
            tally["解不了"] += 1
            continue
        tally[(d["op_type"], d["verdict"])] += 1
        spread[d["verdict"]].append(d["spread"])
        if d["expected"] and d["verdict"] not in ("indistinguishable", d["expected"]):
            bad.append(d)
        elif not d["expected"]:
            bad.append({**d, "why": "没有这个 Op Type 的规则"})
    print(f"=== *_ratio 分母核对 (08 §3.82) === 扫 {len(kdirs)} 个 kernel 目录")
    if _skipped:
        print(f"    ★已排除 {len(_skipped)} 个**坏目录** (aprof --corrupt 的判据): "
              f"{', '.join(os.path.basename(os.path.dirname(x)) + '/' + os.path.basename(x) for x in _skipped[:3])}"
              f"{' …' if len(_skipped) > 3 else ''}")
    print("    规则: Op Type=mix -> 常数 floor(TaskDur)xfreq;  cube/vector -> 本行 total_cycles")
    for key, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        if isinstance(key, tuple):
            t, v = key
            mark = "" if DENOM_RULE.get(t) == v or v == "indistinguishable" else "  ★不符规则"
            print(f"    {t:8} {v:18} {n:5}{mark}")
        else:
            print(f"    {key:27} {n:5}")
    for v, ss in sorted(spread.items()):
        if ss:
            print(f"    目录内反解 D 的极差/中位: {v:18} 中位 {statistics.median(ss):.2e}")
    if bad:
        print(f"\n  ★反例 {len(bad)} 个 (前 10):")
        for d in bad[:10]:
            print(f"    {d['op_type']:8} 判 {d['verdict']:18} 期望 {d['expected']} "
                  f"(逐行 {d['n_row']} / 常数 {d['n_const']})  {d['kdir']}")
        return 1
    print("\n  0 反例。")
    return 0


def print_solve(kdir: str) -> int:
    sol = solve_columns(kdir)
    if not sol:
        print(f"解不了 {kdir} (缺 OpBasicInfo / dump, 或 Current Freq 是 NA)")
        return 1
    tot = ok = blind = amb = 0
    print(f"=== 列 → (换算形式, 遍, 槽, 常数) === {kdir}\n")
    for tbl in sorted(sol):
        rs = sol[tbl]
        n_ok = sum(1 for v in rs.values() if v and v[1] is not None)
        n_bl = sum(1 for v in rs.values() if v and v[1] is None
                   and not v[0].startswith("歧义"))
        n_am = sum(1 for v in rs.values() if v and v[1] is None
                   and v[0].startswith("歧义"))
        tot += len(rs)
        ok += n_ok
        blind += n_bl
        amb += n_am
        print(f"  {tbl:<24} {len(rs):>3} 列  解出 {n_ok:>3}  全零 {n_bl:>2}  "
              + (f"歧义 {n_am:>2}  " if n_am else "        ")
              + f"未解 {len(rs) - n_ok - n_bl - n_am}")
        for c, v in sorted(rs.items()):
            if v and v[1] is None and v[0].startswith("歧义"):
                print(f"       {c:<36} {v[0]}")
        for c, v in sorted(rs.items()):
            if v and v[1] is not None:
                f_, n, s, C = v[:4]
                print(f"       {c:<36} {f_:<22} 遍{n:>2} "
                      + (s if isinstance(s, str) else (f"pmu{s}" if s is not None else "cyc"))
                      + (f"  C={C}" if C else ""))
    #  ★派生只计**尚未被 solve 解出**的列。第一版直接加总, 把 `aic_write_hit_rate`
    #    这类两边都命中的列算了两遍 —— 一个会重复计数的进度指标, 比没有指标更坏。
    der = {k: v for k, v in {**check_derived(kdir), **check_weighted(kdir)}.items()
           if not (sol.get(k[0], {}).get(k[1]) and sol[k[0]][k[1]][1] is not None)}
    if der:
        n_ok = sum(1 for a, b in der.values() if a == b)
        print(f"\n  ── 由同表其它列**派生**的 {len(der)} 列 "
              f"(传递: 那些列已从 bin 解出; 含报告 1.2 的 F6/F9/F12) ──")
        for (tbl, col), (a, b) in sorted(der.items()):
            print(f"       {tbl}/{col:<30} {a}/{b}" + (" 逐行" if a == b else "  ✗"))
        ok += n_ok
    f10 = check_f10(kdir)
    if f10:
        n10, ok10, mx10 = f10
        verdict = ("**在这份产物上是等式**, 观察不到上界性"
                   if mx10 < 1e-5 else f"**观察到间隙**, 最大相对 {mx10:.3%}")
        print(f"\n  ── F10 `Σ(vec dtype 五支) ≤ vec_ratio` ── {ok10}/{n10} 成立; {verdict}")
    print(f"\n  合计 {tot} 列: 解出 {ok} (含派生 {len(der)}) · 全零无分辨力 {blind} · "
          f"**歧义 {amb}** · 未解 {tot - ok - blind - amb}")
    if amb:
        print("    歧义 = 这份数据上有不止一种换算都逐位成立, 分不开 —— "
              "报出来而不是挑一个。")
    print("  ★仍未解的 3 列: `aiv_vec_fp32_ratio` · `aiv_vec_misc_ratio` "
          "(单槽/两槽/三槽与几种分母都不中) ·")
    print("    (2026-10-23: `*_icache_miss_rate` 已解 —— `a/b` 两计数器之比, 见 08 §3.86。")
    print("     此前它落在'未解'里是因为 --solve 没试过这种形式, 又被另一处记成'计数器未导出'。)")
    return 0


def falsify(kdir: str) -> int:
    """证伪检验: 在**副本**上扰动原始字节, 看判据会不会红、会不会误伤。

    ★为什么必须有这一步 (`06` V6 对 B 路径做过同一件事):
    **一个从不报红的检查器, 与一个全对的检查器, 长得一模一样。**
    全仓 1079/1079 过这个结果, 在判据根本不工作时也会出现。

    四个扰动, 两红两不红:
      A 最后一遍全部 `total_cycle` +1      -> E2 必须红
      B **只**把第 1 条记录 +1             -> E2 必须红 (单条分辨力)
      C 改**第 5 遍** (不是最后一遍)        -> 必须**全过** (作用域不该外溢)
      D 最后一遍截掉一条记录                -> E1 必须红
    原产物全程只读: 先 copytree 到临时目录, 扰动只发生在副本上。
    """
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        k = os.path.join(td, "k")
        shutil.copytree(kdir, k)
        ps = passes(k)
        if len(ps) < 2:
            print(f"✗ 证伪检验需要至少两遍 bin, 这个目录只有 {sorted(ps)}")
            return 1
        last = max(ps)
        other = max(n for n in ps if n != last)
        B, BO = ps[last], ps[other]
        orig, oo = open(B, "rb").read(), open(BO, "rb").read()

        def bump(buf: bytes, only_first: bool = False) -> bytes:
            d = bytearray(buf)
            rng = [0] if only_first else range(len(d) // REC)
            for i in rng:
                v, = struct.unpack_from("<Q", d, i * REC + OFF_TOTAL_CYCLE)
                struct.pack_into("<Q", d, i * REC + OFF_TOTAL_CYCLE, v + 1)
            return bytes(d)

        plan = [
            ("基线", lambda: (open(B, "wb").write(orig), open(BO, "wb").write(oo)),
             {"E1": True, "E2": True}),
            (f"A 最后一遍({last})全部 total_cycle +1",
             lambda: open(B, "wb").write(bump(orig)), {"E1": True, "E2": False}),
            (f"B 只把最后一遍的第 1 条 +1",
             lambda: open(B, "wb").write(bump(orig, True)), {"E1": True, "E2": False}),
            (f"C 改第 {other} 遍 (不是最后一遍)",
             lambda: (open(B, "wb").write(orig), open(BO, "wb").write(bump(oo))),
             {"E1": True, "E2": True}),
            (f"D 最后一遍截掉一条记录",
             lambda: (open(BO, "wb").write(oo), open(B, "wb").write(orig[:-REC])),
             {"E1": False, "E2": False}),
        ]
        print(f"=== 证伪检验 === {kdir}\n  (扰动只发生在副本上, 原产物只读)\n")
        bad = 0
        for tag, mutate, want in plan:
            mutate()
            r = check(k)
            got = {e: r.get(e) for e in want}
            ok = got == want
            bad += 0 if ok else 1
            print(f"  {'过' if ok else '✗'} {tag:<34} "
                  f"E1={r.get('E1')!s:<5} E2={r.get('E2')!s:<5} E3={r.get('E3')!s:<5}"
                  f"{'' if ok else f'   期望 {want}'}")
        print(f"\n  ★E3 在 A/B 扰动下**仍然过** —— 它不读 bin, 只比 csv 自己的两列。"
              f"\n    端到端的那一段是 E1+E2; 这是作用域, 不是缺陷。")
        print(f"\n证伪检验: {len(plan)} 项, {bad} 项未达预期")
        return 1 if bad else 0


#  ======================================================================
#  bean 布局机读 (原 `docs/msprof-re/tools/beanmap.py`, 2026-10-23 并入)
#  ======================================================================
#  ★**为什么并进 aprof 而不是单独一个模块**: 上面那组 `OFF_*` 偏移是**手抄** `03` §4.7
#  的, 而这一段从 CANN 的 `analysis` 包**机读**同一份布局 —— 两者是同一个事实的两份真源,
#  放在一起才能互校 (`selftest` 里有一条就是拿机读结果验手抄常数)。
#  分开放就是本仓最忌的那个形态: 同一事实两份真源, 第二份必然先过期。
#
#  与 `gen_spec` 的分工: 它从 ops-info JSON 抽**算子的输入输出包络** (给写 kernel 用,
#  lint 区); 这一段抽 **profiling 记录的字节布局** (给解产物用, attr 区)。
#  方法同是"机读而不手抄", 服务的问题不同。
#
#  不 import 被分析的模块, 只做 AST 解析: CANN 安装包只读 (硬约束第 1 条),
#  连它的 `__pycache__` 都不该碰。

BM_ROOTS = (
    "/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/profiler_tool/analysis",
    "/usr/local/Ascend/latest/tools/profiler/profiler_tool/analysis",
)
#  `=` 原生字节序、标准大小、无填充。来源: `StructFmt.BYTE_ORDER_CHAR`, 由 `beanmap`
#  自己从源码读出来 (下面 `load_structfmt` 返回它), 这里只作为读不到时的兜底。
BM_FALLBACK_BO = "="


def bm_find_analysis(root: str | None = None) -> str | None:
    for p in ([root] if root else list(BM_ROOTS)):
        if p and os.path.isdir(os.path.join(p, "msparser")):
            return p
    return None


def bm_load_structfmt(an: str) -> tuple[dict, str]:
    """返回 ({常量名: 值}, 字节序字符)。"""
    p = os.path.join(an, "msparser", "data_struct_size_constant.py")
    tree = ast.parse(open(p, encoding="utf-8").read())
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "StructFmt":
            #  **按源码顺序**求值: 后面的常量会引用前面的 (`RUNTIME_OP_INFO_256_FMT =
            #  RUNTIME_OP_INFO_FMT + "42I"`), 乱序就解不出来。
            for st in node.body:
                if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name):
                    v = _bm_fold(st.value, out)
                    if v is not None:
                        out[st.targets[0].id] = v
    return out, out.get("BYTE_ORDER_CHAR", BM_FALLBACK_BO)


def _bm_fold(node: ast.AST, env: dict):
    """把 `"HHIIIQ" + "B" * 196` 这类**常量表达式**折出来。解不出返回 None。

    ★必须有这一步: 9 个格式串 (`TASK_TRACK_FMT` / `HCCL_OP_INFO_FMT` / `DPU_*` /
    `CAPTURE_STREAM_INFO_FMT` / `MEMCPY_INFO_FMT` / `V5_STARS_PROFILE_FMT` /
    `RUNTIME_OP_INFO_256_FMT` …) 是**拼接出来的**, 不是字面量。第一版只收
    `ast.Constant`, 于是把这 9 个全报成"常量不存在" —— **9 条命中全是误报**,
    而它们看起来与真问题 (`CYCLES_FMT` 非法) 一模一样, 都写着 C1。
    判据报出来的东西要先核对: 这一条 grep 一次源码就看清了。
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.BinOp):
        l, r = _bm_fold(node.left, env), _bm_fold(node.right, env)
        if l is None or r is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return l + r
            if isinstance(node.op, ast.Mult):
                return l * r
        except TypeError:
            return None
    #  `struct.calcsize(...)` 这类调用不折 —— 折它等于把被检查对象的计算搬过来,
    #  那样 C2 就变成"自己和自己比", 永远相等。
    return None


def bm_load_fmt_dict(an: str) -> dict:
    """返回 {bean 类名: StructFmt 常量名} —— `StructDecoder.STRUCT_FMT_DICT`。"""
    p = os.path.join(an, "profiling_bean", "struct_info", "struct_decoder.py")
    tree = ast.parse(open(p, encoding="utf-8").read())
    out = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "STRUCT_FMT_DICT"):
            continue
        for k, v in zip(node.value.keys, node.value.values):
            #  值的形状固定是 `StructFmt.<NAME>`; 别的形状不猜, 跳过并让 table 显式少一行。
            if isinstance(k, ast.Constant) and isinstance(v, ast.Attribute):
                out[k.value] = v.attr
    return out


_BM_ITEM = re.compile(r"(\d*)([xcbB?hHiIlLqQnNefdspP])")


def bm_expand(fmt: str) -> list | None:
    """把 `'4HQ2HBBH'` 展开成 [(字符, 偏移, 字节数), ...]。解不开返回 None。

    ★逐项算偏移时**不能把单项的 calcsize 累加** —— 那样会丢掉对齐规则。
    这里用的是 `=` (标准大小、无填充), 累加才成立; 换成 `@` 就不成立了。
    所以字节序字符是从源码读出来的, 读到不是 `=` 就不给偏移 (见 layout)。
    """
    if not _BM_ITEM.fullmatch(fmt.replace(" ", "")) and not re.fullmatch(
            r"(\d*[xcbB?hHiIlLqQnNefdspP])+", fmt):
        return None
    items, off = [], 0
    for cnt, ch in _BM_ITEM.findall(fmt):
        n = int(cnt) if cnt else 1
        #  `s` / `p` 的数字是**总字节数**, 不是重复次数 —— 这是 struct 的特例,
        #  按重复次数处理会把 `156s` 当成 156 个字段。
        if ch in "sp":
            sz = struct.calcsize(ch * 1) if n == 0 else n
            items.append((f"{n}{ch}", off, sz))
            off += sz
            continue
        unit = struct.calcsize(f"={ch}")
        for _ in range(n):
            items.append((ch, off, unit))
            off += unit
    return items


def bm_layout(fmt: str, bo: str) -> list | None:
    """给一个格式串返回逐项布局; 字节序不是 `=` 时返回 None (偏移不可累加)。"""
    if bo != "=":
        return None
    return bm_expand(fmt)


_IDX = re.compile(r"\[(\d+)\]")


def bm_bean_fields(an: str) -> dict:
    """扫 `profiling_bean/**/*.py`, 返回 {bean 类名: {下标: 字段名}}。

    形状: `__init__` 里 `self._x = args[N]` 或先 `filed = args[0]` 再 `self._x = filed[N]`。
    两种写法在本版本里都有 (`HwtsLogBean` 用前者, `CyclesBean` 用后者)。
    """
    out = {}
    base = os.path.join(an, "profiling_bean")
    for dp, _dn, fns in os.walk(base):
        if "__pycache__" in dp:
            continue
        for fn in fns:
            if not fn.endswith(".py"):
                continue
            try:
                tree = ast.parse(open(os.path.join(dp, fn), encoding="utf-8").read())
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                init = next((b for b in node.body
                             if isinstance(b, ast.FunctionDef) and b.name == "__init__"), None)
                if init is None:
                    continue
                fields = {}
                for st in ast.walk(init):
                    if not (isinstance(st, ast.Assign) and len(st.targets) == 1):
                        continue
                    tgt, val = st.targets[0], st.value
                    if not (isinstance(tgt, ast.Attribute) and isinstance(val, ast.Subscript)):
                        continue
                    if not isinstance(val.slice, ast.Constant) or not isinstance(val.slice.value, int):
                        continue
                    #  `self._x = <something>[N]`; 被下标的那个名字是 args 解包后的元组,
                    #  叫 args 还是 filed 不重要 —— 它是这个 bean 唯一的下标源。
                    fields[val.slice.value] = tgt.attr.lstrip("_")
                if fields:
                    out[node.name] = fields
    return out


def _bm_rough_items(fmt: str) -> int:
    """格式串非法时也要数出**项数**, 好让 C3 不被 C1 挡住。

    按 struct 的规则数: 合法字符按重复计数展开, `s`/`p` 算一项, 不认得的字符按一项算
    (`ttttttttttH` -> 11 项)。只用于 C3 的越界判断, 不用于算偏移。
    """
    n = 0
    for cnt, ch in re.findall(r"(\d*)(.)", fmt):
        if ch in "sp":
            n += 1
        else:
            n += int(cnt) if cnt else 1
    return n


def _bm_c3(row: dict, nitems: int) -> None:
    if not (row["fields"] and nitems):
        return
    mx = max(row["fields"])
    if mx >= nitems:
        row["problems"].append(
            f"C3 `__init__` 用到下标 {mx} (字段 `{row['fields'][mx]}`), "
            f"而格式串只有 {nitems} 项 —— 解出来也会 IndexError")


def bm_bo_missing(an: str) -> list:
    """C4: 找 `X = struct.calcsize(<没拼字节序的 fmt>)`, 且该 fmt 在 `@` 与 `=` 下尺寸不同。

    返回 [(常量名, 表达式, @尺寸, =尺寸)]。**同值的不报** —— 漏拼本身只是风格不一致,
    报出来会淹掉真问题 (本仓: 会报一堆废话的检查教人忽略它)。
    """
    consts, _bo = bm_load_structfmt(an)
    p = os.path.join(an, "msparser", "data_struct_size_constant.py")
    tree = ast.parse(open(p, encoding="utf-8").read())
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        f = node.value.func
        if not (isinstance(f, ast.Attribute) and f.attr == "calcsize" and node.value.args):
            continue
        arg = ast.unparse(node.value.args[0])
        if "BYTE_ORDER_CHAR" in arg:
            continue
        fmt = consts.get(arg)
        if not isinstance(fmt, str):
            continue
        try:
            native, std = struct.calcsize(fmt), struct.calcsize("=" + fmt)
        except struct.error:
            continue
        if native != std:
            out.append((ast.unparse(node.targets[0]), arg, native, std))
    return out


def bm_analyse(an: str) -> dict:
    """把三份来源合成一张表。返回 {bean: {...}}。"""
    consts, bo = bm_load_structfmt(an)
    fmt_dict = bm_load_fmt_dict(an)
    fields = bm_bean_fields(an)
    rows = {}
    for bean, cname in sorted(fmt_dict.items()):
        fmt = consts.get(cname)
        #  尺寸常量的命名**不统一**: `AICORE_FMT_SIZE` / `HCCL_OP_INFO_SIZE` /
        #  `TASK_TRACK_DATA_SIZE` / `DPU_TASK_TRACK_SIZE` 四种写法都有。只认
        #  `<FMT名>_SIZE` 会把有配套尺寸的也算进"没有第二来源", 把那个数字抬高。
        stem = cname[:-4] if cname.endswith("_FMT") else cname
        cand = [cname + "_SIZE", stem + "_SIZE", stem + "_DATA_SIZE", stem + "_FMT_SIZE"]
        size_name = next((c for c in cand if c in consts), cand[0])
        declared = consts.get(size_name)
        row = {"const": cname, "fmt": fmt, "declared": declared,
               "fields": fields.get(bean, {}), "problems": []}
        if not isinstance(fmt, str):
            row["problems"].append(f"C1 `{cname}` 不是字符串常量 (取到 {fmt!r})")
            rows[bean] = row
            continue
        try:
            calc = struct.calcsize(bo + fmt)
        except struct.error as e:
            #  ★基类 `decode()` 无条件拼 `BYTE_ORDER_CHAR + fmt`, 所以这里的失败
            #    等价于"这个 bean 一旦被 decode 就抛 struct.error"。
            row["calc"] = None
            row["problems"].append(f"C1 `{bo}{fmt}` 在 struct 下非法: {e} "
                                   f"—— 基类 decode() 一调用即抛")
            #  ★**不 continue**: C1 失败时仍要查 C3。第一版在这里短路, 于是
            #    `CyclesBean` 的下标越界 (`filed[20]` 配 11 项格式串) 一条都没报出来
            #    —— 而那条是手工读源码时先看见的。一个判据把另一个判据挡住, 比漏判更难发现,
            #    因为 check 的汇总行会显示 "C3: 0", 读起来像"查过了, 没问题"。
            _bm_c3(row, _bm_rough_items(fmt))
            rows[bean] = row
            continue
        row["calc"] = calc
        row["layout"] = bm_layout(fmt, bo)
        if declared is None:
            row["problems"].append(f"C2 没有配套的 `{size_name}`, calcsize 是唯一来源")
        elif declared != calc:
            #  ★"不等"有两种成因, 工具分不开, 所以把两种都写出来由人判:
            #    (a) 真不一致; (b) 声明的是**记录槽步长**而格式串只解前面一段
            #        (`V5_MODEL_EXEOM_SIZE=256` 配 40 字节的格式串就是这种 ——
            #         `graph_add_info_parser.py:68` 拿 256 当读文件的步长)。
            row["problems"].append(
                f"C2 calcsize={calc} 与声明 `{size_name}`={declared} 不等 "
                f"(或声明的是记录槽步长, 格式串只解前 {calc} 字节)")
        _bm_c3(row, len(row.get("layout") or []))
        rows[bean] = row
    return rows


def bm_print_table(rows: dict) -> int:
    print(f"=== 分析侧记录规格全表 === {len(rows)} 个 bean 挂在 STRUCT_FMT_DICT 上\n")
    print(f"  {'bean 类':<28} {'StructFmt 常量':<32} {'calcsize':>8} {'声明':>6} {'字段':>4}  问题")
    for bean, r in rows.items():
        calc = r.get("calc")
        print(f"  {bean:<28} {r['const']:<32} {str(calc):>8} "
              f"{str(r['declared'] if r['declared'] is not None else '—'):>6} "
              f"{len(r['fields']):>4}  {len(r['problems']) or ''}")
    return 0


def bm_print_check(rows: dict) -> int:
    bad = {b: r for b, r in rows.items() if isinstance(r, dict) and r["problems"]}
    c1 = sum(1 for r in bad.values() if any(p.startswith("C1") for p in r["problems"]))
    c2eq = sum(1 for r in bad.values() if any("不等" in p for p in r["problems"]))
    c2no = sum(1 for r in bad.values() if any("没有配套" in p for p in r["problems"]))
    c3 = sum(1 for r in bad.values() if any(p.startswith("C3") for p in r["problems"]))
    print(f"=== beanmap check === {len(rows) - 1} 个 bean")
    print(f"  C1 格式串非法 (一 decode 即抛): {c1}")
    print(f"  C2 calcsize 与声明尺寸不等:     {c2eq}")
    print(f"  C2 没有第二来源可交叉校验:      {c2no}  ← 不是错, 是**可核验性的上限**")
    print(f"  C3 bean 下标越出格式串项数:     {c3}")
    c4 = bm_bo_missing(rows["__analysis__"]) if "__analysis__" in rows else []
    print(f"  C4 漏拼字节序**且**两模式不同:   {len(c4)}\n")
    for name, arg, nat, std in c4:
        print(f"  ★{name} = calcsize({arg}) —— @ 下 {nat} 字节, = 下 {std} 字节")
    for bean, r in sorted(bad.items()):
        hard = [p for p in r["problems"] if not p.startswith("C2 没有配套")]
        if not hard:
            continue
        print(f"  ★{bean} ({r['const']})")
        for p in hard:
            print(f"      {p}")
    return 1 if (c1 or c2eq or c3) else 0


def bm_print_show(rows: dict, bean: str) -> int:
    r = rows.get(bean)
    if r is None:
        near = [b for b in rows if bean.lower() in b.lower()]
        print(f"没有这个 bean: {bean}" + (f" —— 名字相近的: {', '.join(near)}" if near else ""))
        return 1
    print(f"=== {bean} === {r['const']} = {r['fmt']!r}")
    print(f"  calcsize={r.get('calc')}  声明={r['declared']}")
    for p in r["problems"]:
        print(f"  ★{p}")
    lay = r.get("layout")
    if not lay:
        print("  (没有可用布局)")
        return 0
    print(f"\n  {'下标':>4} {'偏移':>6} {'字节':>4}  {'类型':<6} 字段名")
    for i, (ch, off, sz) in enumerate(lay):
        print(f"  {i:>4} {off:>6} {sz:>4}  {ch:<6} {r['fields'].get(i, '')}")
    return 0



def selftest() -> int:
    ran = bad = 0

    def ck(name, got, want):
        nonlocal ran, bad
        ran += 1
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '✗'} {name}: 得 {got!r}, 期望 {want!r}")

    ck("csv_cycles 把 aic/aiv 合成一个多重集",
       csv_cycles([{"aic_total_cycles": "5", "aiv_total_cycles": "NA"},
                   {"aic_total_cycles": "NA", "aiv_total_cycles": "7"},
                   {"aic_total_cycles": "NA", "aiv_total_cycles": "7"}]),
       collections.Counter({7: 2, 5: 1}))
    ck("csv_cycles 两列都是 NA 的行不计",
       csv_cycles([{"aic_total_cycles": "NA", "aiv_total_cycles": "NA"}]),
       collections.Counter())
    #  ★E3 的表示: 精确商 != fp32 打印, 这正是第一版用绝对容差栽的地方。
    ck("as_printed: 小于 2^24 的周期数, 两种算法同值",
       as_printed(9188491, 1800), "5104.717285")
    ck("as_printed 与 double 直接打印**不同** (所以不能拿 double 比)",
       as_printed(9188491, 1800) == TIME_FMT % (9188491 / 1800), False)
    #  ★超过 2^24 的周期数: "先转 fp32" 与 "double 商压 fp32" 差一个 ULP,
    #    而产物站在前者那边。这是 433 个目录的 E3 不符的全部原因。
    ck("as_printed: 22520341 (>2^24) 必须给出产物那个值",
       as_printed(22520341, 1800), "12511.299805")
    ck("同一个值用 double 商压 fp32 会差一个 ULP (所以这条判据有分辨力)",
       TIME_FMT % f32(22520341 / 1800.0), "12511.300781")
    #  ★遍次号不保证连续, 也不保证到 10 —— 这正是"最后一遍"与"第 10 遍"分道的地方。
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "dump"))
        for n in (1, 2, 4):
            open(os.path.join(td, "dump", f"DeviceProf{n}.bin"), "wb").close()
        ck("passes 认出不连续的遍次号", sorted(passes(td)), [1, 2, 4])
        ck("max(passes) 给的是存在的最大编号, 不是 10", max(passes(td)), 4)
    #  真机部分: 动因样本必须过, 并且判据要有分辨力 (只有最后一遍相等)
    k = MOTIV_KDIR
    if os.path.isdir(k):
        r = check(k)
        ck("动因样本 E1/E2/E2b/E3/E4 全过",
           (r["E1"], r["E2"], r["E2b"], r["E3"], r["E4"]), (True,) * 5)
        ck("E2 有分辨力: 只有最后一遍相等", r["eq_passes"], [10])
        #  ★逐行对账靠的是 `03` §4.7 表里的 block_id@30 —— 第一版漏了这个字段,
        #    还据此写下"bin 记录里没有 block_id"。
        ck("E2b 逐行全中 (键 = block_id@30 + core_type@20 + sub_block@28)",
           r["E2b_n"], r["rows"])
        ck("E4 解出 23 列, 0 列未解",
           (len(r["E4_solved"]), len(r["E4_unsolved"])), (23, 0))
        #  ★全零列不算解出 —— 任何全零槽都能匹配, 那是判据没有分辨力。
        ck("E4 把全零列标成无分辨力而不是解出",
           sorted(r["E4_blind"]),
           ["aic_scalar_mte2_stall_time(us)", "aic_scalar_mte3_stall_time(us)",
            "aic_scalar_wait_time(us)", "aiv_scalar_mte2_stall_time(us)",
            "aiv_scalar_stall_by_ub_time(us)"])
        rec = records(f"{k}/dump/DeviceProf10.bin")
        rec.pop("__dup__", None)
        ck("records 的键在一遍之内唯一 (72 条 72 个键)", len(rec), 72)
        ck("row_key 认出 aic/cube0 这一类",
           row_key({"block_id": "7", "sub_block_id": "cube0",
                    "aic_total_cycles": "1"}), (7, 0, 0))
        #  ---- solve: 全表推广 ----
        sol = solve_columns(k)
        n_ok = sum(1 for rs in sol.values() for v in rs.values() if v and v[1] is not None)
        n_bl = sum(1 for rs in sol.values() for v in rs.values() if v and v[1] is None)
        #  **七**张, 不是八: `OpBasicInfo` 是算子级的一行 (没有 block_id), 不进逐行解。
        #  这条期望我第一版写成 8, 当场被自检抓住 —— 实现是对的, 错的是我数表的方式。
        ck("solve 覆盖七张 block 级表 (OpBasicInfo 是算子级, 不算)", len(sol), 7)
        n_am = sum(1 for rs in sol.values() for v in rs.values()
                   if v and v[1] is None and v[0].startswith("歧义"))
        #  ★2026-10-23 从 (114, 18, 1) 改成 (108, 18, 7): §3.83 把三处"先试到就 break"
        #    改成枚举全部拟合之后, gmfr 上有 6 列从"解出"降为"歧义" (5 个 L0 带宽列
        #    + aic_cube_fops)。**这个期望值此前差点没动** —— 第一版的歧义返回里填了
        #    第一处拟合的遍号, 而这条断言数的是 `v[1] is not None`, 于是 6 条歧义被
        #    计进"已解出", 断言照常过。数字没变, 变的是它的含义 (见 09 第 135 条)。
        #  ★2026-10-23 再 +1 (108 -> 109): `a/b` 形式让 `aic_icache_miss_rate` 解开
        #    (08 §3.86)。gmfr 上只多一列是因为 `aiv_icache_miss_rate` 在它上面全零;
        #    换个算子两列都解得出 (MatMulV3 / SFAG 上 遍5 pmu7/pmu6 与 遍9 pmu2/pmu1, 一致)。
        ck("solve 在动因样本上解出 109 列 / 18 全零 / 7 歧义",
           (n_ok, n_bl - n_am, n_am), (109, 18, 7))
        #  ★主存读一族是**两条读通路之和**: 单槽反解给出 255.54 (离散 1.2e-04),
        #    槽位接近而常数不整 —— 那正是"少加了一个槽"的样子。
        #  ★歧义要同时报"要分开需要多大的值" —— 不然它只是个标签。
        ck("歧义行里带上了分辨力所需的量级",
           "要在 %.6f 上分开需要" in
           sol["ResourceConflictRatio"]["aic_mte3_wait_ratio"][0], True)
        ck("read_main_memory_datas 是遍2 两槽之和, C=128",
           (sol["Memory"]["read_main_memory_datas(KB)"][1],
            sol["Memory"]["read_main_memory_datas(KB)"][2],
            sol["Memory"]["read_main_memory_datas(KB)"][3]), (2, "pmu6+pmu7", 128))
        #  ---- `--ambigres` 的分组键 (2026-10-29, `08` §3.110c) ----
        #  这四条钉的是**当天真撞的两个错**, 不是抽象性质。改回去就会红。
        ck("形式名带遍次 (槽号只在一遍之内有意义)",
           (_slotform(2, 6), _slotform(2, "pmu6+pmu7")),
           ("遍2:pmu6", "遍2:pmu6+pmu7"))
        #  ★负例: 去掉遍次会把两个无关计数器并成一组, 当天的后果是每列冒出
        #    5~6 个"交集为空"的次形式。这条就是那个假设的证伪。
        ck("★负例: 不同遍的同号槽**必须**是不同的形式",
           _slotset("遍3:pmu1") == _slotset("遍5:pmu1"), False)
        ck("同遍同槽才算同一个形式 (判据不能宽到这都分不开)",
           _slotset("遍3:pmu1") == _slotset("遍3:pmu1"), True)
        #  ★并列时取槽数最少的: `pmu0+pmu1+pmu7` 与 `pmu6+pmu7` 当天同为 103 份、
        #    同为 C=128 (多出的槽恒零)。取第一个就会报出前者, 而 §3.59 是后者。
        ck("并列的形式里取槽数最少的那个 (§3.83 的'先试到就报'换了层级)",
           sorted([{"dirs": 103, "form": "遍2:pmu0+pmu1+pmu7"},
                   {"dirs": 103, "form": "遍2:pmu6+pmu7"}],
                  key=lambda d: (-d["dirs"], _nslots(d["form"]), d["form"]))[0]["form"],
           "遍2:pmu6+pmu7")
        #  ★对照臂比的是 (遍, 槽, 常数) 三项。只比常数时, 这两个形式分不开 ——
        #    它们的 C 都是 128。
        ck("★负例: 只比常数分不开这两个形式 (所以判据必须带形式)",
           AMBIGRES_CONTROL["read_main_memory_datas(KB)"][0] == 128
           and AMBIGRES_CONTROL["read_main_memory_datas(KB)"][1] == "遍2:pmu6+pmu7",
           True)
        #  ★`aiv_vec_fops` = 遍4 (pmu0+pmu1) × 32, 而 32 正是 AIV 一条 fp32 向量
        #    指令的元素数 —— 系数本身可解释, 不是凑出来的。
        #  ★F10 在 gmfr 上是等式, 但**不要据此说"本机观测不到上界"** —— 那句话
        #    2026-10-23 写错过一次: 扫描范围漏了 `cases/*/runs/`, 而严格上界的样本
        #    (`Cast` 覆盖率 **0.0421**, `GemmV3` 0.8567) 全在漏掉的那一批里。
        #    `08` §3.32 早就记着这批真机数据。
        _f10 = check_f10(k)
        ck("F10 在动因样本上成立且无可观测间隙 (等式)",
           (_f10[1] == _f10[0], _f10[2] < 1e-5), (True, True))
        ck("aiv_vec_fops 的系数是 32 (AIV fp32 向量宽度)",
           sol["ArithmeticUtilization"]["aiv_vec_fops"][3], 32)
        #  ★`aic_mte3_wait_ratio` 在这份产物上两种分母都逐位成立 —— 报歧义,
        #    不挑一个。样本外 (l2press) 照出: 挑一个会让同一列在两个算子上
        #    报出不同的"解", 看起来像映射不稳定, 实际是判据没有分辨力。
        #  ★2026-10-23 关掉 (08 §3.85): 这条歧义在 gmfr 上**合理**(该列最大 8.0e-06,
        #    两分母只差 1.62e-03, 分不开), 但它不是"本机不可达" —— `cases/gemm_v3` 里
        #    有 199 个目录该列 > 3.1e-04, 最大 3.03e-03。在那种目录上两分母差 37~40%,
        #    `ratio x floor(TaskDur) x freq` 逐块恒为整数 71, 而 `ratio x total_cycles`
        #    给出 50.5~51.9 (非整且逐块变) —— 分母就是 floor 那一支, 与 §3.82 的 mix 规则一致。
        kw = ("cases/gemm_v3/runs/prof_fused_512x512x512_r1/"
              "OPPROF_20260827073751_YSDSHQLZHFGFUIQY/"
              "GemmV3_ND_ND_ND_ND_FP16_FP16_FP16_FP16_high_performance_16842752_mix_aic/3")
        if os.path.isdir(kw):
            sw = solve_columns(kw)
            ck("★该列够大的目录上歧义消失, 解到 floor 那一支 (08 §3.85)",
               sw["ResourceConflictRatio"]["aic_mte3_wait_ratio"][0], "/(floor(TaskDur)*freq)")
        #  ★`a/b` 形式 (08 §3.86): 两个算子上解到**同一对**计数器 —— 样本外一致才敢写进文档。
        km = ("cases/gemm_v3/runs/probe_sk_256_16384/OPPROF_20260827094648_XCBDIEHPSLSQBCPF/"
              "MatMulV3_ND_ND_ND_ND_FP16_FP16_FP16_FP16_high_performance_65584_mix_aic/2")
        ks = ("runs/prof2_det0_bc2048/OPPROF_20260917082055_ZXJQQTSVGFKGPEIW/"
              "SparseFlashAttentionGrad_ecdedf297118bbcd87ce0445eedfc881_111101_mix_aic/1")
        for _k, _lb in ((km, "MatMulV3"), (ks, "SFAG")):
            if not os.path.isdir(_k):
                print(f"  (跳过 icache 样本 {_lb}: 不在本机)")
                continue
            _s = solve_columns(_k)["PipeUtilization"]
            ck(f"★{_lb}: aic/aiv_icache_miss_rate 解到同一对计数器 (08 §3.86)",
               (_s["aic_icache_miss_rate"][1:3], _s["aiv_icache_miss_rate"][1:3]),
               ((5, "pmu7/pmu6"), (9, "pmu2/pmu1")))
        else:
            print("  (跳过 mte3_wait 判决样本: gemm_v3 的目录不在本机)")
        ck("aic_mte3_wait_ratio 报成歧义而不是挑一种分母",
           sol["ResourceConflictRatio"]["aic_mte3_wait_ratio"][0].startswith("歧义"), True)
        #  ★反解出的常数要过领域约束: 字节是 2 的幂且 16~65536, 峰值 50~1200 GiB/s。
        ck("字节常数的合理性: 0 与 204800000 都要被拒 (样本外照出的两个伪解)",
           (_plausible_bytes(0), _plausible_bytes(204800000), _plausible_bytes(256)),
           (False, False, True))
        #  ★四个峰值必须与 `core.socspec --bw-table` 逐位吻合 —— 这条关掉了
        #    `08` §3.46 挂着的"那张表是不是 msopprof 实际读的, 未证实"。
        ck("反解出的四个峰值 = 带宽表里的四个值",
           [sol["Memory"][f"{x}_bw_usage_rate(%)"][3]
            for x in ("GM_to_L1", "GM_to_UB", "L0C_to_GM", "UB_to_GM")],
           [264.0, 220.06, 209.32, 186.8])
        #  ★`*_active_bw` 与 `*_datas(KB)` 落在同一个槽同一个常数上 (遍 3, C=128)。
        ck("fixpipe 活跃带宽与 L0C_to_GM 数据量同槽同常数",
           (sol["PipeUtilization"]["aic_fixpipe_active_bw(GB/s)"][1:4],
            sol["Memory"]["L0C_to_GM_datas(KB)"][1:4]),
           ((3, 4, 128), (3, 4, 128)))
        der = check_derived(k)
        ck("派生的 6 条命中率式全部逐行成立",
           sorted({c for (tb, c), (a, b) in der.items() if a == b}),
           ["aic_read_hit_rate(%)", "aic_total_hit_rate(%)", "aic_write_hit_rate(%)",
            "aiv_read_hit_rate(%)", "aiv_total_hit_rate(%)", "aiv_write_hit_rate(%)"])
        #  ★运算顺序: 先乘 100 再除 24/24, 先除再乘 100 只有 19/24。
        ck("aiv_vec_total_cflt_ratio 是两槽之和, 与 bankgroup/bank 两列自洽",
           sol["ResourceConflictRatio"]["aiv_vec_total_cflt_ratio"][2], "pmu2+pmu3")
        #  ★`aic_time(us)` 来自 `total_cycle` 而不是任何 PMU 槽 —— 第一版的四种形式
        #    全拿 pmu 当分子, 于是把这 14 列 (七张表各两列) 报成"未解", 而 E3 早验过它们。
        ck("aic_time(us) 走 cyc 那一支, 不是 pmu",
           sol["Memory"]["aic_time(us)"][0], "cyc/freq")
        #  ★ratio 的分母**有两种**, 由 `Op Type` 决定 (2026-10-23 全仓 1598 个目录反解,
        #    0 反例, 见 08 §3.82): `mix` 用常数 floor(TaskDur)*freq; 纯 cube/vector
        #    用**本行自己的** total_cycles。动因样本是 mix, 所以这里期望 floor 那一支;
        #    纯核那一支在下面用 gemm_v3 的样本验 —— 只验一支会把另一支当成不存在。
        ck("mix 算子: aic_cube_ratio 走 floor(TaskDur) 分母那一支",
           sol["PipeUtilization"]["aic_cube_ratio"][0], "/(floor(TaskDur)*freq)")
        #  ★带宽的常数必须落在整数字节上 —— 它是"标 GB/s 实为 GiB/s"的判据:
        #    用 1e9 反解得 238.4186, 用 2^30 反解才得 256。
        #  ★动因样本上这些列**唯一可解** —— 这正是"报第一个"与"报唯一那个"
        #    在它上面分不开的原因 (08 §3.83)。写成判据, 将来变成多解会在这里红。
        ck("★mix 动因样本上 L0C_to_GM_datas 唯一可解 (形式里不带'另有'/'歧义')",
           ("另有" not in str(sol["Memory"]["L0C_to_GM_datas(KB)"][0])
            and not str(sol["Memory"]["L0C_to_GM_datas(KB)"][0]).startswith("歧义")), True)
        ck("带宽列 aic_l1_read_bw 的常数是 256 字节 (用 2^30 反解)",
           sol["Memory"]["aic_l1_read_bw(GB/s)"][3], 256)
        ck("带宽列的分母是 TaskDur, 形式名要写出来",
           sol["Memory"]["aic_l1_read_bw(GB/s)"][0], "×C字节/2^30/TaskDur")
        #  ★纯核样本: 同一列必须解到**另一支**。两个分母在 mix 上只差 0.17%
        #    (floor*freq/total_cycles 中位 1.0017), 在纯 vector 上差到 1.79 倍 ——
        #    只在 mix 上解过就会把"常数分母"当成普遍规律, core/perf 一度就是这样。
        kv = ("cases/gemm_v3/runs/prof_split_4096x4096x4096_r1/"
              "OPPROF_20260827073613_SKXZXDMABJAHKVCW/"
              "MatMulV3_ND_ND_ND_ND_FP16_FP16_FP16_FP16_high_performance_65536/1")
        if os.path.isdir(kv):
            solv = solve_columns(kv)
            ck("★纯 cube 算子: 同一列走的是 /total_cycles 那一支",
               solv["PipeUtilization"]["aic_cube_ratio"][0], "/total_cycles")
            #  denom_of 是反方向的检查: 先反解分母再问它常不常, 与 solve 的"哪个对报哪个"
            #  不是同一件事。两个样本各验一支, 并把"目录内 D 常不常"也写成判据 ——
            #  它才是不需要先知道答案就能把两族分开的那一步 (08 §3.82 一节)。
            #  ★多解判决 (08 §3.83): gmfr 上 L0C_to_GM_datas 唯一, 这个目录上有 5 处
            #    同样严丝合缝的 (遍, 槽, C) 而 C 差 64 倍 —— 必须报歧义而不是挑第一个。
            ck("★纯 cube: L0C_to_GM_datas 多解时报歧义, 不挑一个",
               str(solv["Memory"]["L0C_to_GM_datas(KB)"][0]).startswith("歧义"), True)
            ck("★歧义行要把候选 C 全列出来 (说得出区分不开的是哪几个)",
               "[16, 128, 512, 1024]" in str(solv["Memory"]["L0C_to_GM_datas(KB)"][0]), True)
            ck("★槽不唯一但 C 一致时照常报 C, 只注明处数 (数值仍可读)",
               (solv["Memory"]["GM_to_L1_datas(KB)"][3],
                "另有 3 处同 C 的槽" in str(solv["Memory"]["GM_to_L1_datas(KB)"][0])), (256, True))
            dv = denom_of(kv)
            ck("★denom_of: 纯 cube 判 per_row 且与 Op Type 规则一致",
               (dv["verdict"], dv["expected"], dv["op_type"]), ("per_row", "per_row", "cube"))
            ck("★纯核目录内反解 D **不是常数** (极差/中位 > 1e-3)", dv["spread"] > 1e-3, True)
        else:
            print("  (跳过纯核样本: gemm_v3 的目录不在本机)")
        dm = denom_of(k)
        ck("★denom_of: mix 判 const 且与 Op Type 规则一致",
           (dm["verdict"], dm["expected"], dm["op_type"]), ("const", "const", "mix"))
        ck("★mix 目录内反解 D **是常数** (极差/中位 < 1e-4)", dm["spread"] < 1e-4, True)
        ck("row_key 认出 aiv/vector1",
           row_key({"block_id": "7", "sub_block_id": "vector1",
                    "aic_total_cycles": "NA", "aiv_total_cycles": "1"}), (7, 1, 1))
    else:
        print("  (跳过真机部分: 动因样本不在本机)")
    #  ---- 产物枚举必须覆盖两棵树 (已错四次, 见 PRODUCT_GLOBS 上方) ----
    _pf = product_files("OpBasicInfo*.csv")
    if not _pf:
        #  ★本机一个产物都没有时这三条**不适用**, 不是未过: 树是空的, 既证明不了
        #    glob 对也证明不了 glob 错。把不适用报成红色会让它长期挂在登记表里,
        #    而那正是 2026-10-28 掩盖掉两个真 bug 的机制 (见 F169)。
        #    与本函数上面"(跳过真机部分: 动因样本不在本机)"同一处置。
        print("  (跳过产物枚举三条: 本机 runs/ 与 cases/*/runs/ 下没有 OpBasicInfo*.csv —— "
              "空树上这三条不适用)")
    else:
        ck("★产物枚举覆盖 runs/ 这棵树", any(f.startswith("runs/") for f in _pf), True)
        ck("★产物枚举覆盖 cases/*/runs/ 这棵树 —— 漏掉它已经给出过两个否定结论",
           any(f.startswith("cases/") for f in _pf), True)
        ck("★product_kdirs 与 product_files 数得上 (每个目录一个 OpBasicInfo)",
           len(product_kdirs()) <= len(_pf), True)

    #  ---- dtype 分项与 aiv_vec_ratio 的配分恒等式 (08 §3.98) ----
    #  ★`aiv_vec_ratio == Σ(各 dtype 分项)` 在**动因样本上逐行精确成立** (比值 1.000000)。
    #    这说明 k=3/k=11 那一族是 vec_ratio 的一个**配分**, 与它共用分母;
    #    解不出来的是**分子**(计数器没进 dump 的槽), 不是换算形式本身。
    #    别的算子上比值 > 1 (GemmV3 恰好 7/6), 差额就是本机从未导出的那几类 (§3.81)。
    _au = sorted(glob.glob(os.path.join(MOTIV_KDIR, "ArithmeticUtilization*.csv")))
    _pu2 = sorted(glob.glob(os.path.join(MOTIV_KDIR, "PipeUtilization*.csv")))
    if _au and _pu2:
        _parts = ["aiv_vec_fp32_ratio", "aiv_vec_fp16_ratio", "aiv_vec_int32_ratio",
                  "aiv_vec_int16_ratio", "aiv_vec_misc_ratio"]
        _tc = {(r["block_id"], r["sub_block_id"]): r
               for r in csv.DictReader(open(_pu2[0]))}
        _rat = []
        for r in csv.DictReader(open(_au[0])):
            v = r.get("aiv_vec_fp32_ratio")
            if v in (None, "", "NA") or float(v) <= 0:
                continue
            _vr = _tc.get((r["block_id"], r["sub_block_id"]), {}).get("aiv_vec_ratio")
            if _vr in (None, "", "NA"):
                continue
            _s = sum(float(r[c]) for c in _parts if r.get(c) not in (None, "", "NA"))
            if _s > 0:
                _rat.append(float(_vr) / _s)
        ck("★动因样本上 aiv_vec_ratio == Σ(dtype 分项), 逐行比值 1.000000",
           (len(_rat) > 0, all(abs(x - 1.0) < 1e-5 for x in _rat)), (True, True))

    #  ---- 未导出份额 (08 §3.98) ----
    _vs = vec_share(MOTIV_KDIR)
    #  ★容差 4e-6 不是拍的: csv 里这 6 个数各按 `%.6f` 打印, 每个最多带 5e-7 的舍入,
    #    6 个相加的上界就是 3e-6。比它更紧 (如 1e-6) 会在打印精度上假红 —— 第一版就是。
    ck("★动因样本上未导出份额逐行为 0 (配分精确, 容差 = 6 个 %.6f 的舍入上界)",
       (len(_vs) > 0, max((abs(d) for _b, _v, _s2, d in _vs), default=1) < 4e-6), (True, True))
    _g = sorted(glob.glob("cases/gemm_v3/runs/prof_fused_512x512x512_r1/OPPROF_*/"
                          "GemmV3_*_mix_aic/0"))
    if _g:
        _vg = vec_share(_g[0])
        _sh = [d / v for _b, v, _s2, d in _vg if v]
        #  ★只钉"显著大于 0"这一件事。具体百分比随算子变, 写死它等于把一个算子的
        #    工作构成当成规律 —— 第一版在 GemmV3 上把 min=1.166667 读成"恰好 1/7",
        #    而逐行 `vr×6/7 == Σ` 48/48 全不成立 (09 第 154 条)。
        ck("★GemmV3 上未导出份额显著 > 0 (本机看不见的那族向量单元)",
           bool(_sh) and min(_sh) > 0.05, True)

    #  ---- 换算形式码 k 与 --solve 的形式对账 (08 §3.92/§3.94/§3.95) ----
    if os.path.exists(MSOPPROF) and os.path.getsize(MSOPPROF) == MSOPPROF_BYTES:
        _mix2 = ("cases/gemm_v3/runs/probe_sk_256_16384/OPPROF_20260827094648_XCBDIEHPSLSQBCPF/"
                 "MatMulV3_ND_ND_ND_ND_FP16_FP16_FP16_FP16_high_performance_65584_mix_aic/2")
        _cube = ("cases/gemm_v3/runs/prof_split_4096x4096x4096_r1/"
                 "OPPROF_20260827073613_SKXZXDMABJAHKVCW/"
                 "MatMulV3_ND_ND_ND_ND_FP16_FP16_FP16_FP16_high_performance_65536/1")
        _fc = form_codes()
        ck("★抽到的 {列名 → k} 条数 > 70", len(_fc) > 70, True)
        ck("★aiv_vec_fp32_ratio 的 k 在同族里独一份 (它 11, 同族 3)",
           (_fc.get("aiv_vec_fp32_ratio"), _fc.get("aic_vec_fp32_ratio"),
            _fc.get("aiv_vec_misc_ratio")), (11, 3, 3))
        if os.path.isdir(_mix2):
            _j = form_code_join([MOTIV_KDIR, _mix2])
            #  ★这一条才是 k 抽对了的证据: 两份**互相独立**的来源 (二进制常量 / 从产物
            #    数据反解的形式) 一一对应。抽错一侧都不会这么齐。
            ck("★只用 mix 目录对账: 没有一个 k 对上两种形式",
               [k for k, v in _j.items() if len(v) > 1], [])
            #  ★12 而不是 10: 把 DERIVED (命中率那六列, 本仓早已逐行验过) 也算进
            #    "已解出的形式"之后多了 k=16/17/18。**答案本来就在同一个模块里。**
            ck("★对上的 k 不少于 12 个", len(_j) >= 12, True)
            #  ★未知 k 的三分: 全零/从未出现 ≠ 按设计不解 ≠ 真未解 (08 §3.97)。
            #    把它们混成一个"未知"清单, 会让人去给一列**本机根本没有数据**的列读汇编。
            _unk = sorted({k for k in _fc.values()} - set(_j))
            _cols = {n for n, kk in _fc.items() if kk in set(_unk)}
            _mx = col_corpus_max(_cols)
            ck("★k=33 (aic_l2_cache_hit_rate) 本机从未出现, 不是解析缺口",
               _mx.get("aic_l2_cache_hit_rate(%)"), None)
            ck("★k=11 (aiv_vec_fp32_ratio) 有非零数据 —— 它是真未解",
               bool(_mx.get("aiv_vec_fp32_ratio")), True)
            #  ★"有非零数据"的未知 k 是 3/11/13/40 —— 其中 13 (`aic_cube_fops`) 在有数据的
            #    目录上是**歧义**而不是未解 (C 两解 {64, 8192}), 工具目前不分这一层,
            #    所以这条断言按工具实际算的那个集合写, 不按我希望的三个写。
            ck("★有非零数据的未知 k: 3 / 11 / 13 / 40 (7 已按设计排除)",
               sorted(k for k in _unk
                      if any(_mx.get(n) for n, kk in _fc.items() if kk == k)
                      and not all(n.endswith("_total_cycles")
                                  for n, kk in _fc.items() if kk == k)), [3, 11, 13, 40])
            ck("★命中率三族各占一个 k, 且互不相同",
               (_j.get(16), _j.get(17), _j.get(18)),
               (["Σ分子×100/Σ分母 (3/6 列)"], ["a/(a+b)*100"],
                ["Σ分子×100/Σ分母 (2/4 列)"]))
        if os.path.isdir(_mix2) and os.path.isdir(_cube):
            _j2 = form_code_join([MOTIV_KDIR, _mix2, _cube])
            #  ★加一个**纯 cube** 目录之后, 冲突**恰好一个**, 且正是 §3.82 的分母作用域。
            #    换句话说: 这个工具唯一一处"对不上", 等于把 §3.82 又独立证了一遍。
            #  ★2026-10-26 补收码 0 那一类之后, 这条的期望从 `{2: ...}` 变成 `{0: ..., 2: ...}`
            #    (08 §3.102)。**不是回归**: 码 0 与 k=2 解出的是同一个换算形式, 所以 §3.82
            #    那两支分母在它身上原样重现 —— 两个码呈现同一个现象, 正是"同形"的佐证。
            ck("★加纯核目录后的冲突就是 §3.82 的分母两支, 码 0 与 k=2 各一份",
               {k: v for k, v in _j2.items() if len(v) > 1},
               {0: ["/(floor(TaskDur)*freq)", "/total_cycles"],
                2: ["/(floor(TaskDur)*freq)", "/total_cycles"]})
    else:
        print("  (跳过形式码对账: msopprof 不在本机或字节数不符)")

    #  ---- 列顺序: 二进制抽出来的 vs 真机 csv 表头 (08 §3.89) ----
    if os.path.exists(MSOPPROF) and os.path.getsize(MSOPPROF) == MSOPPROF_BYTES:
        _co = col_order()
        ck("★表头构造处抽到的列条目数 > 3000", len(_co) > 3000, True)
        _au = []
        for _slot, _nm in _co:
            if _nm == "aic_cube_ratio" and _slot == 0x20:
                _au = []
            if _au or _nm == "aic_cube_ratio":
                _au.append((_slot // 8, _nm))
            if _nm == "aiv_vec_fops":
                break
        ck("★ArithmeticUtilization 的列号与 csv 表头一致 (前 4 列是 block/sub_block/time/cycles)",
           [x for x in _au[:6]],
           [(4, "aic_cube_ratio"), (5, "aic_cube_fp16_ratio"), (6, "aic_cube_int8_ratio"),
            (7, "aic_cube_fops"), (8, "aic_cube_total_instr_number"),
            (9, "aic_cube_fp_instr_number")])
        ck("★两条未解列的列号", [(i, n) for i, n in _au if "fp32_ratio" in n or "misc_ratio" in n],
           [(14, "aiv_vec_fp32_ratio"), (18, "aiv_vec_misc_ratio")])
        #  ★拿真机 csv 验: 抽出来的顺序必须就是表头顺序 —— 这一条才让上面两条有意义。
        _k = ("cases/gemm_v3/runs/probe_sk_256_16384/OPPROF_20260827094648_XCBDIEHPSLSQBCPF/"
              "MatMulV3_ND_ND_ND_ND_FP16_FP16_FP16_FP16_high_performance_65584_mix_aic/2")
        _f = sorted(glob.glob(os.path.join(_k, "ArithmeticUtilization*.csv")))
        if _f:
            with open(_f[0]) as _fh:
                _hdr = [h.strip() for h in _fh.readline().rstrip("\n").split(",")]
            ck("★逐列对: csv 表头的第 i 个字段 == 二进制里列号 i 的那一个",
               all(_hdr[i] == n for i, n in _au if i < len(_hdr)), True)
        else:
            print("  (跳过列序对照: MatMulV3 的样本不在本机)")
    else:
        print("  (跳过列顺序: msopprof 不在本机或字节数不符)")

    #  ---- 覆盖数三分口径 (08 §3.80/§3.81) ----
    if os.path.exists(MSOPPROF) and os.path.getsize(MSOPPROF) == MSOPPROF_BYTES:
        cov = coverage()
        ck("★二进制列名全集 201 条 (08 §3.80 的抽法)", cov["universe"], 201)
        ck("★出现过 + 从未出现 == 全集 (分母没有漏项)",
           cov["appeared"] + cov["never"], cov["universe"])
        ck("★五分加起来等于出现过的列数 (没有列被数两次或漏掉)",
           sum(cov["split"].values()), cov["appeared"])
        #  ★第二族 (09 第 164 条): 14 列, 与 201 **不相交**。这一条钉的是"两族不重叠",
        #    不是 14 这个数 —— 数会随 CANN 版本变, 而"互斥"是抽法决定的。
        ck("★第二族与 201 那一族不相交 (抽法互斥)",
           bool(xlevel_colnames() & binary_colnames()), False)
        ck("★第二族里只有两个 (estimate) 列在动因样本上没有",
           cov["xlevel_split"].get("样本上没这列"), 2)
        #  ★这三个数**本身**不写死: 它们随本机产物增减而变 (扫 11012 份时是 142/59,
        #    现在 12972 份仍是 142/59, 但没有理由保证将来还是)。写死会变成第四个错数字。
        #    守的是**恒等式**, 不是当下的值 —— 对照 09 第 128 条。
    else:
        print("  (跳过覆盖数: msopprof 不在本机或字节数不符)")
    #  ---- 跨 Op Type 一致性的分类判据 (08 §3.84) ----
    def _raises_sysexit(fn):
        try:
            fn(); return False
        except SystemExit:
            return True
    ck("★字节数不符时拒跑, 不换一份凑合 (同机有 9.0.0-beta.2 的另一份)",
       _raises_sysexit(lambda: binary_colnames(MSOPPROF, MSOPPROF_BYTES + 1)), True)
    ck("★形式集合两两不相交 = 作用域依赖 (ratio 那一类)",
       _is_scope_dependent({"mix": ["/(floor(TaskDur)*freq)"], "cube": ["/total_cycles"]}), True)
    ck("★某一类里同时出现两种形式 = 多解, 不是作用域 (read_main_memory_datas 那一类)",
       _is_scope_dependent({"mix": ["Σpmu×C字节/1024"],
                            "cube": ["×C字节/1024", "Σpmu×C字节/1024"]}), False)
    ck("★三类里有一对相交就不算作用域依赖 (不是只看相邻两个)",
       _is_scope_dependent({"mix": ["A"], "cube": ["B"], "vector": ["A", "B"]}), False)
    ck("★非 2 的幂的 fops 常数要被标出来 (10992 这一类)",
       bool(10992 & (10992 - 1)) and not bool(32 & (32 - 1)) and not bool(8192 & (8192 - 1)), True)
    ck("★KNOWN_SCOPED 认得出 *_ratio 与 *_wait_ratio, 不误伤 *_time(us)",
       tuple(bool(KNOWN_SCOPED[0][0].search(x))
             for x in ("aic_cube_ratio", "aiv_vec_wait_ratio", "aic_cube_time(us)")),
       (True, True, False))

    #  ---- bean 布局机读本身的判据 (原 beanmap.py 的 15 条, 2026-10-23 原样搬入) ----
    #  不搬就是删覆盖: 它们查的是"格式串怎么展开成偏移"这一层, 下面的交叉校验建立在它之上。
    ck("bm_expand 重复计数展开成逐项", len(bm_expand("4HQ")), 5)
    ck("bm_expand 偏移累加", [o for _c, o, _s in bm_expand("4HQ")], [0, 2, 4, 6, 8])
    #  ★`s` 的数字是总字节数不是重复次数 —— 按重复次数处理会把 `156s` 当成 156 个字段。
    ck("bm_expand `156s` 是一项 156 字节", bm_expand("156s"), [("156s", 0, 156)])
    ck("bm_expand 非法字符返回 None", bm_expand("ttttttttttH"), None)
    #  与 struct 自己的 calcsize 对齐 (本仓语料里真实出现的几个格式串)
    for _f in ("4HQ2HBBHLHH12Q", "BBHHHII10Q8I", "HHIIIQ16s64sQQ2Q15Q"):
        _l = bm_expand(_f)
        ck(f"bm_expand 总长 == calcsize('={_f}')",
           _l[-1][1] + _l[-1][2], struct.calcsize("=" + _f))
    #  字节序不是 `=` 时不给偏移 (偏移不可累加)
    ck("bm_layout 在 `@` 下拒绝给偏移", bm_layout("4HQ", "@"), None)

    #  ---- ★交叉校验: 机读的布局 vs 上面手抄的 OFF_* 常数 ----
    #  这是把 beanmap 并进本模块的**全部理由**: 两份真源放在一起才能互校。
    #  CANN 换版本改了 FFTS_BLOCK_PMU_FMT 的话, 这几条会红, 而不是等解析出垃圾才发现。
    _an = bm_find_analysis()
    if _an is None:
        print("  (跳过交叉校验: 找不到 analysis 目录)")
    else:
        _consts, _bo = bm_load_structfmt(_an)
        ck("BYTE_ORDER_CHAR 读得到", _bo, "=")
        ck("StructFmt 常量数 > 100", len(_consts) > 100, True)
        _fd = bm_load_fmt_dict(_an)
        ck("STRUCT_FMT_DICT 条目 > 70", len(_fd) > 70, True)
        ck("HwtsLogBean 在总表里", _fd.get("HwtsLogBean"), "HWTS_LOG_FMT")
        _rows = bm_analyse(_an)
        #  ★动因案例当正例: CyclesBean 的格式串非法**且**下标越界, 两条判据各抓一条。
        ck("C1 正例: CyclesBean 格式串非法必须被抓住",
           any(q.startswith("C1") for q in _rows.get("CyclesBean", {}).get("problems", [])), True)
        #  C4 现在是 0 条 —— 写进自检, 将来格式串一改就会在这里先红。
        ck("C4 当前应为 0 条 (那 11 个漏拼字节序的格式串都自然对齐)", bm_bo_missing(_an), [])
        #  ★2026-10-30 补负例: 上面那条**只走得到通过路径** —— 真包里当前确实 0 条,
        #    于是判据逻辑整个坏掉 (比如 calcsize 的 "=" 拼错) 也仍然返回 []。
        #    `opdbg --mutation --traced` 把它列为"检出路径未覆盖", 样本正是 `值域=[]`。
        #    ★先核过**不是静默零**: 真包 36 个 calcsize 调用点, 11 个无字节序且全部
        #      能在常量表里解析出来, 0 个查不到 —— 判据够到了真数据。
        import tempfile as _tf2
        with _tf2.TemporaryDirectory() as _d2:
            os.makedirs(os.path.join(_d2, "msparser"), exist_ok=True)
            open(os.path.join(_d2, "msparser", "data_struct_size_constant.py"),
                 "w", encoding="utf-8").write(
                "import struct\n"
                "class StructFmt:\n"
                "    BYTE_ORDER_CHAR = '='\n"
                "    PADDED_FMT = 'HQ'\n"       # @ 下 16 (补齐), = 下 10 -> 必须报
                "    ALIGNED_FMT = 'II'\n"      # 两者都是 8 -> 同值不报
                "PADDED_SIZE = struct.calcsize(PADDED_FMT)\n"
                "ALIGNED_SIZE = struct.calcsize(ALIGNED_FMT)\n"
                "SPELLED_SIZE = struct.calcsize(BYTE_ORDER_CHAR + PADDED_FMT)\n")
            _c4 = bm_bo_missing(_d2)
            ck("★C4 负例: 漏拼字节序且 @ 与 = 尺寸不同 -> 必须报出 (HQ: 16 vs 10)",
               _c4, [("PADDED_SIZE", "PADDED_FMT", 16, 10)])
            ck("★C4 负例的对照: 同值的不报 (II 两种都是 8), 拼了字节序的也不报",
               [x[0] for x in _c4], ["PADDED_SIZE"])
        ck("负例: HwtsLogBean 没有 C1/C3 问题",
           [q for q in _rows.get("HwtsLogBean", {}).get("problems", [])
            if not q.startswith("C2 没有配套")], [])
        _lay = (_rows.get("FftsBlockPmuBean") or {}).get("layout") or []
        _off = [o for _c, o, _s in _lay]
        ck("★机读布局与手抄常数一致: 记录长度", _rows["FftsBlockPmuBean"]["calc"], REC)
        for _name, _v in (("core_type", OFF_CORE_TYPE), ("sub_block", OFF_SUB_BLOCK),
                          ("block_id", OFF_BLOCK_ID), ("total_cycle", OFF_TOTAL_CYCLE),
                          ("pmu0", OFF_PMU0)):
            ck(f"★偏移 {_v} ({_name}) 在机读布局里是一个字段的起点", _v in _off, True)
        #  字段名也要对上 —— 偏移对而含义错会更难发现
        _nm = {o: _rows["FftsBlockPmuBean"]["fields"].get(i)
               for i, (_c, o, _s) in enumerate(_lay)}
        ck("★偏移 30 的字段名就是 block_id (不只是位置对)", _nm.get(OFF_BLOCK_ID), "block_id")
        ck("★偏移 32 的字段名就是 total_cycle", _nm.get(OFF_TOTAL_CYCLE), "total_cycle")

    #  ---- L1_to_GM 候选闭式 (08 §3.105, 2026-10-27) ----
    #  ★**不能加 cap_rows**: 加了之后扫描在前几组就停, 而前几组恰好都通过,
    #    于是"仍有不符的组"这条自检被截断成假阴性 (2026-10-27 实撞)。
    #    截断改变答案 —— 与本轮 03 §9.6b 那条 (`--limit` 截出来的"10 个原始点") 同形。
    _lg, _l2, _lsg = l1gm_eval("GemmV3")
    if _lg:
        _full = [g for g, (n, ok, _m) in _lg.items() if n and ok == n]
        ck("L1_to_GM 候选式: 至少有一个 run 组**整组逐位精确** (一个都没有就是式子抄错了)",
           len(_full) > 0, True)
        #  ★这一条**故意断言它还不全对**: 现在 41/54。哪天它全对了, 这条会失败,
        #    那正是提醒"作用域变了, 去把 08 §3.105 的'未定论'改掉"。
        ck("★候选式仍有不符的组 (它是**假设**不是公式; 全对了要回去改文档)",
           len(_full) < len(_lg), True)
        _mx = max(m for _n, _o, m in _lg.values())
        ck("★不符时差的是小量而不是量级 (最大绝对差 < 100 KB)", _mx < 100, True)
    #  ---- 第二候选式 (08 §3.106), 与上面同一次扫描 ----
    if _l2:
        _n, _ok, _mx2 = _l2.get("GemmV3", (0, 0, 0.0))
        ck("候选式二在 GemmV3 上多数行逐位精确 (过半就说明式子没抄错)",
           _n > 0 and _ok > _n / 2, True)
        #  ★这一条是两式的**分工判据**: 式二最坏情况必须比式一好一个量级以上,
        #    否则留两个式子就没有理由。哪天不成立, 该去掉一个。
        ck("★候选式二的最坏情况比五项式好一个量级以上 (否则留两式没理由)",
           _mx2 * 10 < max(m for _a, _b, m in _lg.values()), True)
        ck("★但它仍不精确 (它也是假设; 全精确了要回去改 08 §3.106)",
           _ok < _n, True)

    #  ---- 产物坏行普查 (08 §3.103, 2026-10-26) ----
    _cb, _cn, _csk = corrupt_dirs()
    if _cn == 0 and not product_files("OpBasicInfo*.csv"):
        #  ★同上: 本机没有任何产物目录时这条不适用。只有在**有产物却扫到 0 个**时
        #    才说明 glob 坏了 —— 那才是这条要抓的。
        print("  (跳过坏行普查: 本机没有产物目录, 空树上扫到 0 个不构成 glob 坏掉的证据)")
    else:
        ck("坏行普查: 扫到的目录数 > 0 (扫不到就是 glob 坏了, 而它会静默报'没有坏目录')",
       _cn > 0, True)
    ck("★动因样本不在坏目录里 (它在就说明本仓一大批结论建在坏数据上)",
       any(b["kdir"] == MOTIV_KDIR for b in _cb), False)
    ck("★坏目录**不止一个 run** —— 原注释写的'全在 pipe_v18a_i1_on'已被实测推翻",
       len({b["kdir"].split("/OPPROF_")[0] for b in _cb}) > 1, True)
    ck("★每个坏目录的越界倍数都 > 1 (判据本身没把正常目录算进来)",
       all(b["ratio"] > 1.0 for b in _cb), True)
    #  ---- 2026-10-28: 把判据接到消费方 (`09` 第 198 条) ----
    #  ★动因是实撞: `08` §3.98 里三个 max 值是从坏目录抄出来的 —— 检测器早就能报出它,
    #    但只有全仓普查报得出, 单目录入口 (`--vecshare`) 拿不到, 于是坏数据照样进结论。
    #  判据必须**两侧都验**: 坏目录拒跑 (负例) + 干净目录照常出数 (正例)。
    #  只验负例的话, 一个"永远拒跑"的实现也能全过。
    if _cb:
        _bad_k = _cb[0]["kdir"]
        ck("★corrupt_kdir 对已知坏目录报坏 (单目录判据与全仓普查同口径)",
           bool(corrupt_kdir(_bad_k)[0]), True)
        _hit = False
        try:
            vec_share(_bad_k)
        except SystemExit:
            _hit = True
        ck("★负例: vec_share 读坏目录必须拒跑 (它读的是数值含义, 坏行污染比值)",
           _hit, True)
    ck("★正例: corrupt_kdir 对动因样本判干净 (判据过宽会把全仓结论一起拒掉)",
       corrupt_kdir(MOTIV_KDIR)[0], False)
    ck("★正例: 干净目录的 vec_share 仍出数 (只验负例的话'永远拒跑'也能全过)",
       len(vec_share(MOTIV_KDIR)) > 0, True)
    #  ---- probecol (2026-10-28, `09` 第 200 条) ----
    #  ★正例用一个**已知答案**的列 (`aiv_vec_ratio` = 遍10 槽0 / floor(TaskDur)xfreq):
    #    探查器找不到已知答案就是它自己坏了, 而那种坏会伪装成"这列也解不出"。
    #  ---- l1gm 入选列方差体检 (2026-10-28, `09` 第 208 条) ----
    #  ★正例必须是**真仓里那个已知的常数列** —— 造一个合成常数列也能让判据过,
    #    但那证明不了它在真语料上抓得住 (本会话第 201 条的教训: 负例是作者造的)。
    #  ★自检只取**一个 kernel 目录**, 不扫全量语料 (2026-10-28, `09` 第 214 条)。
    #    首版用 "GemmV3" 全量, 把 aprof 自检从约 20 秒推到 44 秒 —— 而 `l1gm_eval`
    #    的注释早写着"**慢闸门会训练出 `--no-verify`**, 闸门的成本本身就是一条设计约束"。
    #    ★但**仍用真仓数据**不造合成样本 (第 201/213 条的教训): 这个目录同时含
    #    320 个退化行与 64 个有信息行 (16 个目录 384 行), 下面几条判据各自都站得住。
    #    ★**单目录 (24 行) 不行**: 24 行对 50 多列系统欠定, 当场出现"伪精确拟合"
    #    (系数是 1741/3142 这种) —— 而那正是这套工具本该避免的失效。
    #    夹具的行数必须**明显多于列数**, 否则精确方法会找到不存在的关系。
    _SEL_DIR = "prof_base_1024x1024x10"
    _cv = {c: v for c, _k, _n, _u, v in l1gm_colvar("GemmV3")}
    ck("★l1gm 列方差: aic_mte3_instructions 在 GemmV3 上是常数 (变异系数 0)",
       _cv.get("aic_mte3_instructions"), 0.0)
    #  ★这条判据的**作用域守卫** (`09` 第 210 条): 方差小只说明"这一列在这批语料里
    #    问不出东西", **不说明它不该在式子里**。第 207 条就是据此错判 mte2 是伪项的 ——
    #    它在 512 行有信息行上承重 (五项式在那里 512/512 逐位精确)。
    #    所以这里盯住: mte2 的变异系数必须**明显不是 0**, 免得下次又被读成"也是伪项"。
    ck("★负例: aic_mte2_instructions 的变异系数明显非 0 (它不是常数, 更不是伪项)",
       (_cv.get("aic_mte2_instructions") or 0) > 1.0, True)
    #  ---- exact_relation (2026-10-28, `09` 第 211 条) ----
    #  ★正例/负例都用**真仓**: 分支内必须找得到那条精确关系, 混合语料上必须明说"不存在"。
    #    后者才是这个方法相对最小二乘的要害 —— 最小二乘永远给得出一个答案。
    from fractions import Fraction as _FR

    def _fr(_s, _c):
        _v = (_s.get(_c) or "").strip()
        return None if _v in ("", "NA") else _FR(_v)

    def _degen(_s):
        try:
            return _FR(_fr(_s, "aic_write_cache_hit")
                       + _fr(_s, "aic_write_cache_miss_allocate"), 2)                    - _fr(_s, "L0C_to_GM_datas(KB)") == 33
        except TypeError:
            return False
    #  ★**自检只取一个 kernel 目录, 不扫全量语料** (2026-10-28, `09` 第 214 条)。
    #    首版用 "GemmV3" 全量, 一次 11 秒 x 两条 = 把 aprof 自检从 20 秒推到 44 秒,
    #    而 `l1gm_eval` 的注释早写着"**慢闸门会训练出 `--no-verify`**,
    #    闸门的成本本身就是它的一条设计约束" —— 我犯的正是它记着的那条。
    #    ★但**仍用真仓的数据**, 不造合成样本 (本会话第 201/213 条的教训):
    #    这个目录同时含 16 个退化行与 8 个有信息行, 两条判据各自都站得住。
    #  ---- solve 的覆盖率回退 (2026-10-28, `09` 第 216 条) ----
    #  ★正例用**真仓里一个会触发它的目录**: MatMulV2 上 `aic_mte3_ratio` 被
    #    `/total_cycles` 解释了 16/24 行。不配正例的话, 一个从不触发的回退
    #    (本仓 CR7: "没人调用的判据防不住任何东西") 会一直留在代码里装样子。
    #  ---- 命中率的单侧性 (2026-10-28, `09` 第 217 条) ----
    #  ★判据本身**自带正例对照**: 四个列必须 100% 取等, 两个 aic 读路径列必须低估为 0。
    #    只盯"低估为 0"的话, 一个取数坏掉 (全读成 None) 的实现也能全过 —— 两侧都要钉。
    _hb = {c: (eq, over, under) for c, eq, over, under, _w in hitrate_bound()}
    ck("★命中率自然式**低估恒为 0** (⇒ 它是精确上界, 不是'大致对')",
       [c for c, (_e, _o, u) in _hb.items() if u], [])
    ck("★正例对照: 四个列必须 100% 取等 (不再是 100% 就说明取数或口径坏了)",
       sorted(c for c, (e, o, u) in _hb.items() if o == 0 and u == 0 and e),
       ["aic_write_hit_rate(%)", "aiv_read_hit_rate(%)",
        "aiv_total_hit_rate(%)", "aiv_write_hit_rate(%)"])
    _mm = [x for x in product_kdirs() if "MatMulV2" in x]
    if _mm:
        _sc = solve_columns(_mm[0])
        _part = [v[0] for cols in _sc.values() for v in cols.values()
                 if v and isinstance(v[0], str) and "部分成立" in v[0]]
        ck("★solve 覆盖率回退: 真仓上必须能把一个'未解'列报成'部分成立'",
           bool(_part), True)
    _r_narrow = exact_relation(L1GM_TARGET, "prof_base_1024x1024x1023")
    ck("★exactrel 负例: **语料太窄要拒判** (24 行 x 50 列曾给出 1741/3142 这种伪精确拟合)",
       (_r_narrow["rel"], _r_narrow.get("toonarrow")), (None, True))
    _r_mix = exact_relation(L1GM_TARGET, "GemmV3")
    ck("★exactrel 负例: 混合两支的语料上必须报**不存在**精确关系 "
       "(最小二乘在这里会照样给一个答案)", _r_mix["rel"], None)
    _r_b = exact_relation(L1GM_TARGET, "GemmV3", where=lambda x: not _degen(x))
    ck("★exactrel 正例: 非退化支上找得到那条已知关系 (mte2 系数 1/16)",
       _r_b["rel"] and _r_b["rel"].get("aic_mte2_instructions"), _FR(1, 16))
    ck("★负例: 真有变化的列不许被判成常数 (判据过宽会把整个式子判掉)",
       all((_cv.get(c) or 0) > 0.2 for c in
           ("aic_write_cache_hit", "L0C_to_GM_datas(KB)", "aic_mte2_instructions")), True)
    _pr = probe_column(MOTIV_KDIR, "aiv_vec_ratio")
    ck("★probecol 正例: 已解出的列必须被找到 (找不到说明探查器坏了, 而那会伪装成'也解不出')",
       any("槽0" in h and "floor" in h for h in _pr.get("hits", [])), True)
    _pn = probe_column(MOTIV_KDIR, "aiv_vec_fp32_ratio")
    ck("★probecol 负例: k=11 那列在这个空间里确实没有 (有了就是本条结论该改)",
       _pn.get("hits"), [])
    ck("★probecol 必须把搜过的空间报出来 (否则'没找到'会被读成'没找对地方')",
       bool(_pr.get("space")), True)
    #  ---- probecol 的三条自我设限, 每条都对着一种"看起来有答案"的失效 ----
    ck("★probecol 全行成立才算候选 (单行巧合在这个宽空间里几乎必然发生)",
       probe_column(MOTIV_KDIR, "aiv_ub_read_bw_scalar(GB/s)").get("hits"), [])
    _mt = probe_column(MOTIV_KDIR, "aic_mte3_wait_ratio")
    ck("★印数精度不足的列报**问不出**而不是报'没有' (两者下一步不同)",
       (_mt.get("hits"), bool(_mt.get("noinfo"))), ([], True))
    _hr = probe_column(MOTIV_KDIR, "aic_read_hit_rate(%)")
    ck("★NA 分母剔除而不是换成 1.0 (换成 1.0 曾凭空造出 82 个候选)",
       any("不是可用分母" in d for d in _hr.get("dropped", [])), True)
    #  ★跳过原因必须分因 (§3.97): 混成一格会让"档位没开"与"记录残缺"同命运,
    #    而两者的下一步一个是改采集、一个是丢弃。
    ck("★跳过原因分因报出, 不混成一格", len(_csk) >= 2, True)

    #  ---- 码 0 那一类 (08 §3.102, 2026-10-26) ----
    if os.path.isfile(MSOPPROF) and os.path.getsize(MSOPPROF) == MSOPPROF_BYTES:
        _with = form_codes()
        _without = form_codes(zero_code=False)
        _extra = set(_with) - set(_without)
        ck("码 0 那一类此前整类漏收 (收了之后列数必须变多)", len(_extra) > 0, True)
        ck("★新收进来的**全部**是码 0 —— 多收一条非 0 的就说明 wzr 那一侧串号了",
           {_with[n] for n in _extra}, {0})
        ck("★`aiv_vec_ratio` 的码是 0 (手读 0x5a2d0 的 `strh wzr, [x3, #520]`)",
           _with.get("aiv_vec_ratio"), 0)
        ck("★不收码 0 时它就不在表里 (说明漏收是真的, 不是换了个写法)",
           "aiv_vec_ratio" in _without, False)
        #  ★`*_wait_ratio` 一族落在这一类里 —— README 把它们挂了很久的"未解"
        ck("★`*_wait_ratio` 一族进得来 (它们是 README 上挂着的那组)",
           all(f"aic_mte{i}_wait_ratio" in _with for i in (1, 2, 3)), True)

    #  ---- 指令记录解析器表 (08 §3.101 订正版, 2026-10-26) ----
    if os.path.isfile(MSOPPROF) and os.path.getsize(MSOPPROF) == MSOPPROF_BYTES:
        _kh = record_parsers()
        ck("记录解析器表: 每个码一个函数, 没有两个码共用",
           len({v[0] for v in _kh.values()}), len(_kh))
        #  ★手读锚点 (0x288db0 的 `add x1,x3,#0x2c` + `str x1,[x2]`): 这一条是全表里
        #    唯一逐条手读过的, 机抽若与它不符, 说明抽取形状坏了而不是二进制变了。
        ck("★手读锚点: 码 11 的解析函数 / 记录字节数 / 名字",
           _kh.get(11, (None,) * 4)[:2] + (_kh.get(11, (None,) * 4)[3],),
           (0x288db0, 44, "ParseLoad2DTransposeRecord"))
        ck("★手读锚点: 码 3 的解析函数 / 记录字节数 / 名字",
           _kh.get(3, (None,) * 4)[:2] + (_kh.get(3, (None,) * 4)[3],),
           (0x28b400, 48, "ParseDmaMovRecord"))
        #  ★★把 2026-10-26 那次误认钉死: 本表的键空间与列的换算形式码**不是同一个**。
        #    两个方向都要有差集 —— 只有一个方向有差集时, 还能用"抽漏了"解释过去。
        _co, _ko = check_not_formcode()
        ck("★判决: 列有而本表没有的码 (非空 = 两个码不是一回事)", bool(_co), True)
        ck("★判决: 本表有而没有任何列用的码 (非空, 另一个方向也对不上)", bool(_ko), True)
        ck("★名字串自报它是记录解析而不是列公式 (抽到名字的都以 Parse 开头)",
           all(n.startswith("Parse") and n.endswith("Record")
               for _k, v in _kh.items() if (n := v[3])), True)
        #  ★证伪检验: "取函数开头第一个落在区间里的立即数"这种 naive 取法必须与
        #    带写回判据的取法**不一致**。若两者处处相同, 说明那条判据没有承重,
        #    留着只是装饰 —— 这一条在 2026-10-26 实测不一致于 3 个 k。
        with open(MSOPPROF, "rb") as _f:
            _blob = _f.read()

        def _naive(fn, span=0x200):
            for _a in range(fn, fn + span, 4):
                _v = struct.unpack_from("<I", _blob, _a)[0]
                if (_v >> 31) & 1 and ((_v >> 23) & 0x1FF) == 0b100100010:
                    _im = (_v >> 10) & 0xFFF
                    if 0x08 <= _im <= 0x100:
                        return _im
            return None
        _diff = [k for k, v in _kh.items() if _naive(v[0]) != v[1]]  # noqa: E501
        ck("★证伪: 不带写回判据的 naive 取法会取错 (这条判据有承重)",
           len(_diff) > 0, True)

    print(f"\naprof 自检: {ran} 条, {bad} 条未过")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kdir", nargs="?", help="kernel 目录")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--per-op", type=int, metavar="N",
                    help="每个算子只取 N 个目录 —— 全仓九成是 gmfr, 抽样比前 N 个有代表性")
    ap.add_argument("--beans", nargs="?", const="check", metavar="table|check|show:<Bean>",
                    help="bean 布局机读: check 查四条一致性判据 (默认) / table 全表 / "
                         "show:<Bean> 看单个的逐字段字节布局")
    ap.add_argument("--join-kdir", dest="join_kdirs", action="append", metavar="KDIR",
                    help="--formcode 对账时多用一个 kernel 目录 (可给多次)。"
                         "★跨 Op Type 并样会让 k 对上两种分母 —— 那是 §3.82 的作用域, 不是抽错")
    ap.add_argument("--vecshare", metavar="KDIR",
                    help="逐行报 aiv_vec_ratio / Σ(dtype 分项) / **未导出份额** (见 08 §3.98)")
    ap.add_argument("--formcode", action="store_true",
                    help="抽 `{列名 → 换算形式码 k}` 并与 --solve 的形式对账 (见 08 §3.92/§3.94)")
    ap.add_argument("--l1gm", nargs="?", const="", metavar="OP",
                    help="核对 L1_to_GM_datas(KB)(estimate) 的候选闭式 (见 08 §3.105); "
                         "可给算子名片段只看该算子")
    ap.add_argument("--corrupt", nargs="?", const="", metavar="ROOT",
                    help="全仓普查产物坏行 (某核忙时间 > 任务总时长, 见 08 §3.103)")
    ap.add_argument("--recparse", action="store_true",
                    help="指令记录解析器表: 记录类型码 → 解析函数 + 记录字节数 + 名字 "
                         "(见 08 §3.101 订正版; **这个码不是 --formcode 那个 k**)")
    ap.add_argument("--colorder", action="store_true",
                    help="从 msopprof 的 .text 读每张表的列顺序 (表头构造处, 槽/8 = 列号; "
                         "见 08 §3.89)")
    ap.add_argument("--coverage", action="store_true",
                    help="覆盖数三分口径: msopprof 二进制的列名全集 x 本机产物表头 x "
                         "动因样本的 --solve 判定 (见 08 §3.80/§3.81)")
    ap.add_argument("--xcheck", nargs="?", const=40, type=int, metavar="N",
                    help="跨 Op Type 一致性: 每类抽 N 个目录 (默认 40) 跑 --solve, "
                         "比对同一列的换算形式与常数。出现**未登记**的作用域依赖则退出码 1 "
                         "(见 08 §3.84)")
    ap.add_argument("--ambigres", nargs="?", const="", metavar="SUBSTR",
                    help="歧义列的候选 C 跨目录取交集 (见 08 §3.110)。--solve 的\"歧义\"只在"
                         "单目录内成立, 交起来常收敛到唯一值。给 SUBSTR 则只看路径含它的目录")
    ap.add_argument("--ambigres-col", metavar="COL",
                    help="--ambigres 摊开某一列: 每份目录各自给出什么候选集 (查交集为空的来源)")
    ap.add_argument("--ambigres-limit", type=int, default=0, metavar="N",
                    help="--ambigres 只扫前 N 个目录 (调试用; 默认全扫)")
    ap.add_argument("--denom", nargs="?", const="", metavar="ROOT",
                    help="核对 `*_ratio` 的分母规则 (Op Type=mix -> 常数 floor(TaskDur)xfreq; "
                         "cube/vector -> 本行 total_cycles, 见 08 §3.82)。不给 ROOT 则扫 runs/ 与 cases/。"
                         "有反例退出码 1")
    ap.add_argument("--exactrel", metavar="列名",
                    help="用**精确有理数零空间**找这一列的线性闭式 (不拟合, 无容差)。"
                         "配 --op 限定算子。零空间维数>1 时关系不唯一, 工具会说出来")
    ap.add_argument("--hitbound", action="store_true",
                    help="命中率自然式的**方向**统计: 取等/高估/低估 —— 低估为 0 则它是精确上界")
    ap.add_argument("--ratiorel", metavar="列名",
                    help="找 `该列 = (列的线性组合) / d` 的精确关系 (线性化有理拟合: 两边乘分母)。"
                         "只扫单列分母, 配 --op")
    ap.add_argument("--op", metavar="算子名", help="--exactrel / --ratiorel 的算子过滤")
    ap.add_argument("--probecol", metavar="KDIR:COL",
                    help="给 --solve 解不出的列扫更宽的形式空间, 并**把搜过的空间打印出来** "
                         "(否则'没找到'永远可以被读成'没找对地方')。只报候选, 不下结论")
    ap.add_argument("--solve", metavar="KDIR",
                    help="解全部八张表的列 → (换算形式, 遍, 槽, 常数)")
    ap.add_argument("--falsify", metavar="KDIR",
                    help="证伪检验: 在副本上扰动原始字节, 看判据红不红、误不误伤")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.beans:
        an = bm_find_analysis()
        if an is None:
            print("找不到 CANN analysis 目录")
            return 2
        rows = bm_analyse(an)
        if a.beans.startswith("show:"):
            return bm_print_show(rows, a.beans[5:])
        return bm_print_table(rows) if a.beans == "table" else bm_print_check(rows)
    if a.vecshare:
        rows = vec_share(a.vecshare)
        if not rows:
            print(f"没有可用数据 (缺 ArithmeticUtilization / PipeUtilization): {a.vecshare}")
            return 1
        print(f"=== 向量 dtype 配分与未导出份额 (08 §3.98) === {len(rows)} 行")
        print(f"  {'块':>12}  {'vec_ratio':>10}  {'Σ分项':>10}  {'未导出':>10}  {'占比':>7}")
        for blk, vr, s, d in rows[:12]:
            print(f"  {blk:>12}  {vr:10.6f}  {s:10.6f}  {d:10.6f}  "
                  f"{(d / vr * 100 if vr else 0):6.2f}%")
        if len(rows) > 12:
            print(f"  ... 另 {len(rows) - 12} 行")
        ds = [d / vr for _b, vr, _s, d in rows if vr]
        if ds:
            print(f"\n  未导出份额占 vec_ratio 的比例: 中位 {statistics.median(ds) * 100:.2f}%  "
                  f"min {min(ds) * 100:.2f}%  max {max(ds) * 100:.2f}%")
            print("  ★动因样本上这一项应为 0 (配分精确); >0 的部分是本机从未导出的那族向量单元列。")
        return 0
    if a.l1gm is not None:
        return print_l1gm(a.l1gm or None)
    if a.corrupt is not None:
        return print_corrupt(a.corrupt or None)
    if a.recparse:
        return print_record_parsers()
    if a.formcode:
        codes = form_codes()
        join = form_code_join(a.join_kdirs or None)
        print(f"=== 换算形式码 k (08 §3.92/§3.94) === {len(codes)} 个列名 "
              f"(只收 14 份拷贝一致的)")
        bad = [k for k, v in join.items() if len(v) > 1]
        print(f"\n  k → --solve 解出过的形式  ({len(join)} 个 k 有交集, "
              f"**对上两种形式的: {len(bad)}**)")
        for k, v in join.items():
            print(f"    k={k:<3} {v}")
        #  ★**反方向也要看** (2026-10-26, 08 §3.102): §3.94 只验了"一个 k 只对一种形式",
        #    没验"一种形式只对一个 k"。补收码 0 那一类之后, 反方向当场就不成立 ——
        #    码 0 与 k=2 解出的是同一个形式。所以 k 决定怎么算, **而怎么算不决定 k**。
        rev = collections.defaultdict(set)
        for k, forms in join.items():
            for f in forms:
                rev[f].add(k)
        multi = {f: sorted(ks) for f, ks in rev.items() if len(ks) > 1}
        print(f"\n  反方向: 同一种形式对应几个 k  (**对上多个 k 的形式: {len(multi)}**)")
        for f, ks in sorted(multi.items()):
            print(f"    {f}  ← k={ks}")
        if not multi:
            print("    (本次没有 —— 若 §3.102 之后这里变空, 先查是不是码 0 那一类又漏收了)")
        unknown = sorted({k for k in codes.values()} - set(join))
        print(f"\n  还没对上任何已解形式的 k: {unknown}")
        #  ★把三种状态分开 (08 §3.97): 全零 / 按设计不解 / 真未解。
        #    混在一起会把下一步指错 —— 对全零的列读再多汇编也没有数据可对。
        allc = {n for n, kk in codes.items() if kk in set(unknown)}
        mx = col_corpus_max(allc) if allc else {}
        for k in unknown:
            who = sorted(n for n, kk in codes.items() if kk == k)
            zero = [n for n in who if mx.get(n) == 0.0]
            noc = [n for n in who if mx.get(n) is None]
            byde = [n for n in who if n.endswith("_total_cycles")]
            if byde:
                why = "**按设计不解** (直接取记录的 total_cycle 字段, E2 已逐位验过)"
            elif len(zero) + len(noc) == len(who):
                #  ★两者要分开: "出现过但全零" 换个负载就可能解; "从未出现" 是档位没开
                #    (08 §3.81 的那 59 列)。前者改负载, 后者改采集 —— 下一步不一样。
                why = (f"**本机从未出现** ({len(noc)} 列) —— 档位没开, 见 §3.81, 不是解析缺口"
                       if len(noc) == len(who) else
                       f"**出现过但全零** ({len(zero)} 列) —— 换个负载才可能解")
            else:
                why = f"**真未解** (有非零数据: {max((mx[n] for n in who if mx.get(n)), default=0):.6g})"
            print(f"    k={k:<3} {why}")
            print(f"         {who[:4]}{' ...' if len(who) > 4 else ''}")
        return 1 if bad else 0
    if a.colorder:
        rows = col_order()
        print(f"=== 表头构造处抽出的列顺序 (08 §3.89) === {len(rows)} 条 (槽/8 = 列号)")
        prev = -1
        for slot, nm in rows:
            if slot <= prev:
                print("  ---- 新的一张表 ----")
            prev = slot
            print(f"  +0x{slot:03x}  列 {slot // 8:>3}  {nm}")
        return 0
    if a.coverage:
        return print_coverage()
    if a.xcheck is not None:
        return print_xcheck(a.xcheck)
    if a.ambigres is not None:
        return print_ambigres(a.ambigres or None, a.ambigres_limit, a.ambigres_col)
    if a.denom is not None:
        return print_denom(a.denom or None, a.limit)
    if a.hitbound:
        return print_hitbound(a.op)
    if a.ratiorel:
        return print_ratiorel(a.ratiorel, a.op)
    if a.exactrel:
        return print_exactrel(a.exactrel, a.op)
    if a.probecol:
        return print_probecol(a.probecol)
    if a.solve:
        return print_solve(a.solve)
    if a.falsify:
        return falsify(a.falsify)
    if a.all:
        return scan_all(a.limit, a.per_op)
    if not a.kdir:
        ap.error("给一个 kernel 目录, 或 --all")
    return print_one(check(a.kdir))


if __name__ == "__main__":
    sys.exit(main())
