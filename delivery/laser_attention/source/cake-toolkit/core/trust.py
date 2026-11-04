"""判据**信任等级**与预测-实测对账台账 —— 让"这条判据现在有多可信"成为机读量。

与已有模块的分工 (新建理由):
  · 与 `core/_refs.py` 的分工: _refs 是**人手写的判据出处卷宗**(静态文本: 判什么/实证/验证日期),
    trust 是**自动累积的信任台账**(预测-实测对账 + 等级计算 + 资格闸门)。trust 读 _refs 当起始证据等级,
    不重复存它的内容。
  · 与 `core/statemap.py` 的分工: statemap 管**方向**(某条优化方向的开/关/复活),
    trust 管**工具**(某条判据可不可信)。关一条方向时, statemap 问 trust "你依据的那条判据有没有资格关"。
  · 与 `core/gatestate.py` 的分工: gatestate 登记"哪个闸门当前未过"(产物侧的红),
    trust 登记"哪条判据当前不可信"(工具侧的红)。

★★为什么要有这个 (2026-09-18 立, 代价 11.94 ms)

  参考实现 (PR 12104) 比本仓最好构建快 11.94 ms。dubin 把五条优化方向逐条告诉了我,
  我**每一条都发现过, 又逐条否决掉了**。否决所依据的是本仓的成本模型 ——
  而那个模型的内存侧下界至今建立在 `dram_GBps = None (uncalibrated)` 上。

  三条从这次代价里提炼的机制, 全部落在本模块:

  1. **预算不是成本。** 静态模型给的是**上界**, 真实成本可能远小于预算。
     用一个只保证"不小于"的量去**筛掉**候选, 方向是反的 —— 上界大不等于收益小,
     上界小才等于收益小。本模块对每条判据强制标注它给的是上界/下界/点估,
     并让"用上界做否决"这件事直接被 `gate` 拒绝。
  2. **等级必须能降。** 搜到的一句说得最准: *"A state machine that only climbs is not
     trust, it is seniority"* —— 一条开始失准的判据必须掉下来, 掉多快是设计的一部分,
     不是事后运维。于是 stale/代理量/对账落空都是降级项, 且降级项**自动生效**。
  3. **"证据不足"不等于"不可信"。** 同一来源: *"absence of evidence and evidence of
     absence are different"*。新建工具是 **T0 未验证**, 不是"低分工具";
     打印时必须写「未验证」而不是「不可信」, 因为二者的处方相反 ——
     前者的处方是**去用它并对账**, 后者的处方是**别用**。

## 等级 T0..T5 怎么算 (照 GRADE: 起始等级由证据类型定, 再逐项升降)

  起始 (来自 _refs 的 status):
      ini_exact 4 | measured 3 | derived 1 | disputed 1 | refuted 0 | 表里没有 0(未验证)
  降级 (每项 -1):
      stale       距上次验证 > STALE_DAYS
      indirect    实证测的是**代理量**不是这条主张本身
      imprecise   只能排序不能定量 (输入参数未标定 / 模型自称只给上界)
      unfalsifiable  没有负例自检 (opdbg --selftest-criteria 抓不住它)
      calib_miss  预测-实测对账落空 (|log2(实测/预测)| > MISS_LOG2)
  升级 (每项 +1):
      oos         样本外检验过
      cross_case  在 >=2 个不同 case 上被端到端实测确认
      calibrated  >=2 次对账且都落在阈值内

  ★**升降不对称 (有意的)**: 降级可以由文本启发触发 (evidence 里写了"只可用于排序"),
    升级**只认台账记录** —— 必须有人真做过一次对账。理由: 文本启发出错时,
    让它把一条好判据压低只是保守, 让它把一条差判据抬高是**静默的假授权**。

## 资格闸门 (本模块唯一的强制项)

  `trust gate --criterion <id> --action refute` ——
  等级 < MIN_LEVEL_TO_REFUTE 或带 imprecise/indirect 的判据, **没有资格单独关闭一条方向**,
  退出码 2。这是 loop 提示词那句"成本模型不够好时, 否决没有资格"的机读版。
"""
from __future__ import annotations
import argparse
import ast, datetime, glob, importlib, json, math, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "core", "_trust_ledger.jsonl")

STALE_DAYS = 120
MISS_LOG2 = 0.5          # |log2(actual/pred)| > 0.5 即偏差 >1.41x 判落空
MIN_LEVEL_TO_REFUTE = 3
DENSITY_FLOOR = 0.34     # 实现类产出 / (实现类 + 关闭类) 低于它就报"产物密度低"
MIN_DENSITY_SAMPLES = 5  # 少于这么多条记录时密度不作数 (空台账会算出"健康"的 1.00)

BASE = {"ini_exact": 4, "measured": 3, "derived": 1, "disputed": 1, "refuted": 0}
DOWN = ["stale", "indirect", "imprecise", "unfalsifiable", "calib_miss",
        "upper_only", "contradicted"]
UP = ["oos", "cross_case", "calibrated"]
MANUAL_UP = ["oos", "cross_case"]     # calibrated 只能由 resolve 对账挣出来, 不许手标
CLEAR_N = 2                            # 一条落空/反驳要被 N 条后续端到端证据才算清掉
MANUAL_UP_CAP = 1                      # 手标升级项合计最多抬 1 级 (免得自写三行拿到否决权)
STALE_STATUSES = ("measured", "derived", "disputed", "ini_exact")

LEVEL_NAME = {
    0: "T0 未验证/已推翻",
    1: "T1 单来源推导",
    2: "T2 有实证但有保留",
    3: "T3 实证可用",
    4: "T4 可回源/多点实证",
    5: "T5 跨 case 标定过",
}

#  ★文本启发只用于**降级**。每条都写清它在找什么形态, 免得后人以为是语义理解。
TEXT_DOWNGRADE = [
    ("imprecise", re.compile(r"只(可)?用于排序|不作承诺|不预测.{0,6}收益|仅供参考|上界|未标定|uncalibrated")),
    ("indirect",  re.compile(r"代理量|proxy|间接")),
]


def _today() -> str:
    return datetime.date.today().isoformat()


# ----------------------------------------------------------------- 台账 I/O
LAST_READ_DAMAGE: list = []      # 上一次 read_ledger 读坏的行号; 打印路径必须**带内**报出来


def read_ledger(path: str | None = None) -> list:
    p = path or LEDGER
    LAST_READ_DAMAGE.clear()
    if not os.path.exists(p):
        return []
    out = []
    for i, ln in enumerate(open(p, encoding="utf-8")):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        try:
            out.append(json.loads(ln))
        except Exception as e:
            #  ★损坏的行不静默跳过 —— 静默跳过会让"台账是空的"与"台账读坏了"长得一样。
            LAST_READ_DAMAGE.append(i + 1)
            print(f"★台账第 {i+1} 行读不了 ({e}), 已跳过 —— 请手工修: {p}", file=sys.stderr)
    return out


def _damage_banner() -> bool:
    """台账有读坏的行时**在标准输出上**说出来。

    ★盲审揪出 (2026-09-18): 原来警告走 stderr 而**错的等级走 stdout** ——
      重定向或翻页时只剩那个错数, 而它看起来完全正常。
      损坏的恰好是一条降级记录时, 等级会静默偏高。
    """
    if not LAST_READ_DAMAGE:
        return False
    print(f"★★ 台账有 {len(LAST_READ_DAMAGE)} 行读不了 (行 {LAST_READ_DAMAGE}) —— "
          f"下面每个等级都**可能偏高**: 读坏的若是降级记录, 它就静默消失了。先修台账再读数。")
    return True


def append_ledger(rec: dict, path: str | None = None) -> None:
    p = path or LEDGER
    rec.setdefault("ts", _today())
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------- 判据来源
def load_refs(mods: list | None = None) -> dict:
    """{'模块.判据id': Ref} —— 从各模块的 REFS 收。两种形态 (dict / list) 都收。"""
    if mods is None:
        try:
            areas = json.load(open(os.path.join(ROOT, "core", "_areas.json"), encoding="utf-8"))
            mods = [m for a in areas["areas"] for m in a["mods"]]
        except Exception:
            mods = []
    out = {}
    for m in mods:
        try:
            mod = importlib.import_module(f"core.{m}")
        except Exception:
            continue
        refs = getattr(mod, "REFS", None)
        if not refs:
            continue
        #  ★列表形态的元素有两种: Ref 对象 (取属性) 与**纯字典** (取下标)。
        #    首版只写 getattr, 而字典没有 `.id` 属性 -> 键恒为空串 -> 同模块的多条
        #    全塌成一个 `"<模块>."` 且只剩最后一条。netgate (4 条) 与 shguard (7 条)
        #    因此在等级表里只各剩 1 条, **静默丢了 9 条** —— 而守这件事的判据写的是
        #    `len(refs) > 100`, 总数 309 照样过, 看不见局部坍塌。
        #    2026-10-30 由 `opdbg --mutation --traced` 顺藤摸到 (它指出那条断言只验条数)。
        def _rkey(r):
            if isinstance(r, dict):
                return r.get("id") or str(r.get("what") or r.get("实证") or "")[:40]
            return getattr(r, "id", "") or str(getattr(r, "what", ""))[:40]
        items = refs.items() if hasattr(refs, "items") else [(_rkey(r), r) for r in refs]
        for k, v in items:
            out[f"{m}.{k}"] = v
    return out


def _ref_text(r) -> str:
    w, e = getattr(r, "what", ""), getattr(r, "evidence", "")
    j = lambda x: " ".join(x) if isinstance(x, (list, tuple)) else str(x or "")
    return j(w) + " " + j(e)


# ----------------------------------------------------------------- 等级计算
def assess(crit: str, refs: dict, ledger: list, today: str | None = None) -> dict:
    """算一条判据的信任等级。返回 {level, base, down, up, flags, calib, why}。"""
    today = today or _today()
    r = refs.get(crit)
    why, down, up = [], [], []

    if r is None:
        base, status = 0, "unlisted"
        why.append("判据出处表里没有这一条 —— **未验证**, 不是不可信。处方是去用它并对账, 不是别用")
    else:
        status = getattr(r, "status", "derived")
        base = BASE.get(status, 1)
        why.append(f"起始 {base} (出处 status={status})")
        age = r.age_days(today) if hasattr(r, "age_days") else -1
        if status in STALE_STATUSES and age > STALE_DAYS:   # 盲审: 原来 ini_exact/disputed 永不过期
            down.append("stale"); why.append(f"stale: 距上次验证 {age} 天 (>{STALE_DAYS})")
        txt = _ref_text(r)
        for flag, pat in TEXT_DOWNGRADE:
            m = pat.search(txt)
            if m and flag not in down:
                down.append(flag)
                why.append(f"{flag}: 出处文本里出现 {m.group(0)!r} [文本启发, 只降不升]")

    #  台账记录 (显式 flag / vote / 对账) —— 升级只认这里
    #  ★本段 2026-09-18 盲审后重写。原版有五处"照接口正常用就能绕过"的洞:
    #    (1) --bound 写进了台账却**没有任何判定路径读它** —— 本模块唯一的强制项没实现;
    #    (2) 升级项不去重, 同一条 flag 敲两遍就多抬一级;
    #    (3) 同一条预算可以 resolve 多次, 一次实测就凑够"两次对账";
    #    (4) 落空永不过期且只能靠手标 calibrated 抬回来 —— 那条手标不受任何校验;
    #    (5) 手写三行 flag 就能给一条 REFS 里根本不存在的判据拿到否决资格。
    #  共同形态: **凡是作者自己一行就能写进去的东西, 都不能单独构成升级依据。**
    manual_up, cases, upper_only = [], set(), False
    resolves, votes, cleared, flag_why = [], [], [], {}
    for rec in ledger:
        c = rec.get("criterion")
        if c != crit and not (c and c.endswith(".*") and crit.startswith(c[:-1])):
            continue
        op = rec.get("op")
        if op == "flag":
            f = rec.get("flag")
            if f in DOWN and f not in down:
                down.append(f); flag_why[f] = f"{f}: 台账 {rec.get('ts')} —— {rec.get('why','')}"
            elif f in MANUAL_UP and f not in manual_up:
                manual_up.append(f); why.append(f"{f}(手标): 台账 {rec.get('ts')} —— {rec.get('why','')}")
        elif op == "unflag":
            #  ★2026-09-18 立: 降级标记的**成因被修好之后**必须能撤销。
            #    原版没有这条路 —— 于是 "dram_GBps 未标定" 这个 imprecise 在标定做完之后
            #    仍然永久压着 hwfloor 四条判据。一个只能降不能升的标记与"永久拉黑"没区别,
            #    而那正是第 0.95 条要防的反面 (只降不升同样不是信任)。
            #    撤销要写 --evidence 指到修它的那次改动, 且**只清它之前的**同名标记 ——
            #    撤销之后又出现的同名标记重新生效。
            f = rec.get("flag")
            if f in down:
                down.remove(f)
                cleared.append(f"{f} 已于 {rec.get('ts')} 撤销: {rec.get('why','')}")
            elif f in manual_up:
                manual_up.remove(f)
        elif op == "predict":
            if rec.get("bound") == "upper":
                upper_only = True
        elif op == "vote":
            if rec.get("retract"):
                #  ★撤票 (2026-09-18 立): 一张**基于不完整证据**投出的票必须能撤。
                #    当天实例: 我看到"桶宽 80 构建通过"就给 ub_budget 投了 contradicted,
                #    而完整证据是 —— 它在真机上算错 33% 的元素, OVER CAP 很可能**是对的**。
                #    台账仍是追加式的 (原票留在文件里), 但**计分时不再计入**;
                #    撤票必须写 --evidence 说明当时漏了什么。
                votes = [v for v in votes if not (v.get("outcome") == rec.get("outcome")
                                                  and v.get("kind") == rec.get("kind"))]
            else:
                votes.append(rec)
        elif op == "resolve":
            resolves.append(rec)

    why += [v for k, v in flag_why.items() if k in down] + cleared

    #  ★(1) 上界模型: 只要这条判据出过 upper 预算, 它给的就是上界 —— 上界只能排序。
    if upper_only and "upper_only" not in down:
        down.append("upper_only")
        why.append("upper_only: 台账里有 --bound upper 的预算 —— 上界只能排序, 不能筛掉候选")

    #  ★(3)(4) 对账: 按**预算 id** 去重 (同一条预算 resolve 多次只算一次),
    #    并按时间序算"还没被清掉的落空": 一条落空要 CLEAR_N 条后续命中才算修好。
    #  ★2026-09-22 codex 评审第二条, 已用夹具复现并修复。
    #    原实现只按**判据 ID** 聚合, 对账序列不分主张。夹具: 一条"端到端节省时间"的预测
    #    落空 + 两条"落 DRAM 字节"的预测命中 -> calib_miss 被清掉, 等级 T2 升 T4,
    #    否决资格恢复。**证据不是可互换的积分**: 另一个量上的成功不能代替对这个量的修正。
    #    真实台账里这个混存已经存在: hwfloor.HF2 的 P1/P2/P4 是 ms, P3 是 GB。
    #    对策: 按**主张**分桶 (predict 的 claim, 缺则用 unit, 再缺则单独成桶),
    #    落空只能被**同一主张**的后续命中清掉; 升级也必须落在单一主张内挣到。
    claim_of = {}
    for rec in ledger:
        if rec.get("op") == "predict" and rec.get("id"):
            claim_of[rec["id"]] = (str(rec.get("claim") or "").strip()
                                   or str(rec.get("unit") or "").strip()
                                   or "(未声明主张)")
    seen, buckets = set(), {}
    for r in resolves:
        rid = r.get("id")
        if rid in seen:                       # 同一条预算重复对账, 只认第一次
            continue
        seen.add(rid)
        key = claim_of.get(rid) or (str(r.get("unit") or "").strip() or "(未声明主张)")
        buckets.setdefault(key, []).append(r.get("verdict"))
    open_by_claim, hits_total, hits_by_claim = {}, 0, {}
    for key, seq in buckets.items():
        open_miss, hits_after, hits_k = 0, 0, 0
        for v in seq:
            if v in ("miss", "violated"):
                open_miss += 1; hits_after = 0
            elif v == "hit":
                hits_k += 1; hits_after += 1
                if open_miss and hits_after >= CLEAR_N:
                    open_miss -= 1; hits_after = 0
            elif v == "loose" and "imprecise" not in down:
                down.append("imprecise")
                why.append(f"imprecise: 主张『{key}』上有一条预算被**大幅超额满足** —— "
                           f"该界太松, 定不了量")
        if open_miss:
            open_by_claim[key] = open_miss
        hits_by_claim[key] = hits_k
        hits_total += hits_k
    if open_by_claim and "calib_miss" not in down:
        down.append("calib_miss")
        detail = "; ".join(f"『{k}』{n} 条" for k, n in sorted(open_by_claim.items()))
        why.append(f"calib_miss: 以下主张上的对账落空尚未被 {CLEAR_N} 次**同主张**后续命中"
                   f"清掉 —— {detail} (共对账 {len(seen)} 条预算, 命中 {hits_total})。"
                   f"★另一个量上的命中不计入: 证据不是可互换的积分")
    #  升级同样要落在单一主张内挣到: 跨主张各命中一次不构成"这条判据已标定"。
    best_claim = max(hits_by_claim.items(), key=lambda kv: kv[1], default=(None, 0))
    if best_claim[1] >= 2 and not open_by_claim:
        up.append("calibrated")
        why.append(f"calibrated: 主张『{best_claim[0]}』上 {best_claim[1]} 次对账命中, "
                   f"且**所有主张**都无未清落空 (只能这样挣到, 不许手标)")
    elif hits_total >= 2 and best_claim[1] < 2 and not open_by_claim:
        why.append(f"未升 calibrated: 共 {hits_total} 次命中但分散在 {len(hits_by_claim)} 个"
                   f"主张上, 单个主张最多 {best_claim[1]} 次 —— 跨主张不累加")

    #  ★投票: 确认要端到端; 反驳门槛低 (保守方向), 但同样可被 CLEAR_N 次后续端到端确认清掉。
    contra = 0
    for v in votes:
        if v.get("outcome") == "confirmed" and v.get("kind") == "end_to_end":
            cases.add(v.get("case") or "?")
            if contra:
                contra = max(0, contra - 1) if len(cases) >= CLEAR_N else contra
        elif v.get("outcome") == "contradicted":
            contra += 1; cases.clear()
    if contra and "contradicted" not in down:
        down.append("contradicted")
        why.append(f"contradicted: {contra} 条反驳票未被 {CLEAR_N} 个不同 case 的端到端确认清掉")
    if len(cases) >= 2:
        up.append("cross_case"); why.append(f"cross_case: 端到端确认于 {sorted(cases)}")

    #  ★(2)(5) 手标升级项合计最多抬 MANUAL_UP_CAP 级, 且**不能让未登记的判据拿到否决资格**。
    manual_gain = min(len(manual_up), MANUAL_UP_CAP)
    if len(manual_up) > manual_gain:
        why.append(f"手标升级项 {manual_up} 合计只抬 {manual_gain} 级 (上限 {MANUAL_UP_CAP}) —— "
                   f"自己一行写进去的东西不能单独构成升级依据")
    up_all = list(up) + manual_up

    lvl = 0 if status == "refuted" else max(0, min(5, base - len(down) + len(up) + manual_gain))
    #  未登记的判据 (REFS 里没有) 无论台账怎么写都不得拿到否决资格 ——
    #  否则"升级只认台账"退化成"认一条自己刚写进去的行"。
    blocked = {"imprecise", "indirect", "upper_only"} & set(down)
    can_refute = (lvl >= MIN_LEVEL_TO_REFUTE and not blocked and status != "unlisted")
    if status == "unlisted" and lvl >= MIN_LEVEL_TO_REFUTE:
        why.append("★判据出处表里没有这一条 —— 无论台账等级多高都不给否决资格, "
                   "先给它写一条 Ref(what/status/evidence/verified)")
    return {"criterion": crit, "level": lvl, "base": base, "status": status,
            "down": down, "up": up_all, "why": why, "can_refute": can_refute}


# ----------------------------------------------------------------- 各子命令
VERDICT_ADOPTED = ["hwfloor", "ub_budget"]   # 已在结论处自报资格的模块; opdbg 报还差谁


def verdict_banner(module: str, *, refusing: bool = False, prefix: str = "") -> list:
    """让一条判据在**给出结论的同一处**报出自己的信任等级。返回要打印的行。

    ★2026-09-18 立, 代价是一个被错杀的候选。
      我用 `ub_budget` 报的「UB 196608/196608, 余量 0, OVER CAP」毙掉了"桶宽 64->80"这个候选,
      写下"这条路直接封死"。而 ub_budget 的 `U5_absolute_peak` 在判据表里是 **refuted**,
      `U1_static_ub_estimate` 是 T0 —— 按本仓自己的资格闸门, 它**没有否决权**。
      闸门是我建的, 我自己没调。

    ★所以修法不能是"下次记得敲 trust gate" —— **那与它要防的遗漏是同一形态**
      (CLAUDE.md: 工具不被触发 = 工具不存在; 约束要在动手那一刻自动触发)。
      唯一能拦住的位置是**结论本身**: 数字和它的资格一起出现, 不需要谁记得去查。

    refusing=True (这次的结论是"不行/超了/不通过") 且有判据 <T3 时, 打一条大字:
      它不是在说"别信这个数", 是在说**这个数没有资格替你关掉一条路, 去实测**。
    """
    try:
        refs, led = load_refs([module]), read_ledger()
        rows = [assess(k, refs, led) for k in sorted(refs)]
    except Exception as e:
        return [f"{prefix}★{module} 的信任等级取不到 ({type(e).__name__}) —— 按最低信任处理。"]
    if not rows:
        return [f"{prefix}★{module} **没有判据出处表** —— 它的结论是**未验证**的断言 "
                f"(不是不可信: 处方是去用它并对账)。补法见 core/_refs.py。"]
    weak = [r for r in rows if not r["can_refute"]]
    out = [f"{prefix}★判据信任: {len(rows)} 条, "
           f"{len(rows)-len(weak)} 条有否决资格 (>=T{MIN_LEVEL_TO_REFUTE}), {len(weak)} 条没有"
           + (f" —— {', '.join(r['criterion'].split('.',1)[1][:26] for r in weak[:4])}" if weak else "")]
    if refusing and weak:
        out += [f"{prefix}★★ 本次结论是**否定**的, 而上面 {len(weak)} 条判据没有否决资格。",
                f"{prefix}   **它们不足以替你关掉这条路** —— 想关就去实测 (建一次/测一次),",
                f"{prefix}   或先给它对账: python -m core.trust predict/resolve。",
                f"{prefix}   查详情: python -m core.trust show -q {module} -v"]
    return out


def cmd_show(a) -> int:
    refs, ledger = load_refs(), read_ledger(a.file)
    keys = sorted(set(refs) | {r["criterion"] for r in ledger if r.get("criterion") and not r["criterion"].endswith(".*")})
    if a.q:
        keys = [k for k in keys if a.q.lower() in k.lower()]
    rows = [assess(k, refs, ledger) for k in keys]
    rows.sort(key=lambda x: (x["level"], x["criterion"]))
    print("=== 判据信任等级 (T0..T5; 低等级 = **鼓励去质疑并实测**, 不是别用) ===")
    for r in rows:
        if a.max_level is not None and r["level"] > a.max_level:
            continue
        tag = "".join(f" -{d}" for d in r["down"]) + "".join(f" +{u}" for u in r["up"])
        gate = "" if r["can_refute"] else "   ★无资格单独关闭方向"
        print(f"  T{r['level']}  {r['criterion']:42s} [{r['status']}]{tag}{gate}")
        if a.verbose:
            for w in r["why"]:
                print(f"         · {w}")
    n_low = sum(1 for r in rows if r["level"] < MIN_LEVEL_TO_REFUTE)
    print(f"\n共 {len(rows)} 条; 其中 {n_low} 条 < T{MIN_LEVEL_TO_REFUTE} —— "
          f"这些条目**不得单独用来关闭一条优化方向**, 要关必须真去实现并端到端实测。")
    print("  等级怎么来的: trust show -q <词> -v ; 记一条预算: trust predict ; 对账: trust resolve")
    return 0


def cmd_predict(a) -> int:
    pid = a.id or f"P{sum(1 for r in read_ledger(a.file) if r.get('op') == 'predict') + 1}"
    if a.bound not in ("upper", "lower", "point"):
        print("--bound 必填: upper(上界) / lower(下界) / point(点估)。"
              "**上界不得用于筛掉候选** —— 上界大不等于收益小。", file=sys.stderr)
        return 2
    append_ledger({"op": "predict", "id": pid, "criterion": a.criterion, "claim": a.claim,
                   "value": a.value, "unit": a.unit, "bound": a.bound, "case": a.case}, a.file)
    print(f"已记预算 {pid}: {a.criterion} → {a.value} {a.unit} ({a.bound})")
    if a.bound == "upper":
        print("★这是**上界**。它能回答'最多能省多少', 不能回答'这个方案不值得做' —— "
              "真实成本可能远小于它。用它做否决会被 trust gate 拒。")
    print(f"  实测出来后: python -m core.trust resolve --id {pid} --actual <数>")
    return 0


def cmd_resolve(a) -> int:
    led = read_ledger(a.file)
    pred = next((r for r in reversed(led) if r.get("op") == "predict" and r.get("id") == a.id), None)
    if pred is None:
        print(f"台账里没有预算 id={a.id}; 现有: "
              f"{[r['id'] for r in led if r.get('op')=='predict']}", file=sys.stderr)
        return 2
    #  ★盲审揪出: 同一条预算可以被 resolve 多次, 于是"一次实测"就能凑够"两次对账命中"。
    dup = [r for r in led if r.get("op") == "resolve" and r.get("id") == a.id]
    if dup and not a.redo:
        print(f"{a.id} 已经对过账 ({dup[-1]['ts']}: {dup[-1]['actual']} → {dup[-1]['verdict']})。"
              f"\n一条预算只算一次对账 —— 要记新的一次实测请 **新开一条 predict**; "
              f"确实是要更正上一条才加 --redo。", file=sys.stderr)
        return 2
    p, act = float(pred["value"]), float(a.actual)
    if p <= 0 or act <= 0:
        print("预测与实测都要 > 0 才能算比值", file=sys.stderr)
        return 2
    ratio = act / p
    off = abs(math.log2(ratio))
    bound = pred.get("bound", "point")
    thr = 2 ** MISS_LOG2
    #  ★盲审揪出: 原来不看 bound, 于是"上界被满足"(实测远小于上界) 也被判 miss;
    #    而且比值 <1 时还打印 "0.30x > 1.41x" 这种假比较。
    #    上界的正确判读: 实测**超过**它 = 界被违反 (硬证伪); 实测远低于它 = 界太松 (定不了量)。
    if bound == "upper":
        verdict = "violated" if ratio > thr else ("loose" if ratio < 1 / thr else "hit")
    elif bound == "lower":
        verdict = "violated" if ratio < 1 / thr else ("loose" if ratio > thr else "hit")
    else:
        verdict = "miss" if off > MISS_LOG2 else "hit"
    append_ledger({"op": "resolve", "id": a.id, "criterion": pred.get("criterion"),
                   "predicted": p, "actual": act, "unit": pred.get("unit"), "bound": bound,
                   "ratio": round(ratio, 3), "verdict": verdict, "note": a.note,
                   "case": pred.get("case")}, a.file)
    print(f"{a.id} {pred.get('criterion')}: 预测 {p} ({bound}) → 实测 {act} {pred.get('unit','')}  "
          f"比值 {ratio:.2f}x  → **{verdict}**")
    if verdict == "violated":
        d = "超过了上界" if bound == "upper" else "低于了下界"
        print(f"★实测{d} ({ratio:.2f}x) —— 这是对这条判据的**硬证伪**, 不是精度问题。")
        print(f"  这条判据已自动降级, 并进入待修 test_case: python -m core.trust cases")
        print(f"  ——**修模型优先于继续优化**: 界错了方向, 据它做的每一次筛选都不作数。")
    elif verdict == "loose":
        d = "远低于上界" if bound == "upper" else "远高于下界"
        print(f"★实测{d} ({ratio:.2f}x) —— 界没被违反, 但它**太松, 定不了量**, 已标 imprecise。")
        print(f"  松的界只能排序。要它能定量, 得把模型里被这个比值指向的那一项参数改到能给出更接近的界。")
    elif verdict == "miss":
        print(f"★偏差超阈值 ({ratio:.2f}x, 阈值 {1/thr:.2f}~{thr:.2f}x)。这条判据已自动降级:")
        print(f"    python -m core.trust cases")
        print(f"  ——**修模型优先于继续优化**: 一个偏 {ratio:.2f} 倍的预算会筛掉本该做的候选。")
    return 0


def cmd_vote(a) -> int:
    if a.retract and not a.evidence:
        print("★撤票必须给 --evidence —— 说明当时漏了什么。不写理由的撤票与改账无异。", file=sys.stderr)
        return 2
    if a.kind != "end_to_end" and a.outcome == "confirmed":
        print("★只有 --kind end_to_end 的确认才计入升级。"
              "'工具被调用了很多次'不是票 —— 那样长出来的是资历不是信任。", file=sys.stderr)
    append_ledger({"op": "vote", "criterion": a.criterion, "outcome": a.outcome,
                   "kind": a.kind, "evidence": a.evidence, "case": a.case,
                   **({"retract": True} if a.retract else {})}, a.file)
    print(f"{'已撤回' if a.retract else '已投票'}: {a.criterion} {a.outcome} ({a.kind})")
    return 0


def cmd_flag(a) -> int:
    if a.flag not in DOWN + UP:
        print(f"--flag 取值: {DOWN + UP}", file=sys.stderr)
        return 2
    if a.clear:
        if not a.evidence:
            print("★撤销一个标记必须给 --evidence —— 指到把它的成因修掉的那次改动 "
                  "(commit / 标定 / 实测)。不给证据的撤销与直接删记录无异。", file=sys.stderr)
            return 2
        append_ledger({"op": "unflag", "criterion": a.criterion, "flag": a.flag,
                       "why": a.evidence, "case": a.case}, a.file)
        print(f"已撤销 {a.criterion} 的 {a.flag} 标记。撤销之后再出现的同名标记会重新生效。")
        return 0
    if a.flag == "calibrated":
        #  ★盲审揪出: 这条手标是整套等级的后门 —— 它不受任何校验, 一行就能把落空盖掉。
        print("★calibrated **不能手标** —— 它只能由 predict/resolve 对账挣出来 "
              f"(≥2 次命中且无未清落空)。先 trust predict, 实测后 trust resolve。", file=sys.stderr)
        return 2
    if a.flag in UP and not a.evidence:
        print("★升级项必须给 --evidence (哪次对账/哪个 case)。升级只认台账, 不认文本。", file=sys.stderr)
        return 2
    if a.flag in MANUAL_UP:
        print(f"★手标升级项合计最多抬 {MANUAL_UP_CAP} 级, 且**不给未登记判据否决资格** —— "
              f"自己一行写进去的东西不能单独构成升级依据。")
    append_ledger({"op": "flag", "criterion": a.criterion, "flag": a.flag,
                   "why": a.why or a.evidence, "case": a.case}, a.file)
    print(f"已标记 {a.criterion}: {a.flag}")
    return 0


def cmd_gate(a) -> int:
    r = assess(a.criterion, load_refs(), read_ledger(a.file))
    print(f"{a.criterion}: T{r['level']} [{r['status']}]"
          + (f"  降级 {r['down']}" if r["down"] else "") + (f"  升级 {r['up']}" if r["up"] else ""))
    for w in r["why"]:
        print(f"    · {w}")
    if a.action != "refute":
        return 0
    if r["can_refute"]:
        print(f"\n✓ 有资格: T{r['level']} >= T{MIN_LEVEL_TO_REFUTE} 且无 imprecise/indirect。"
              f"关闭时仍要走 core.statemap add --kind refuted --proxy false。")
        return 0
    print(f"\n✗ **没有资格用它关闭一条方向** (T{r['level']}"
          + (", 带 " + "/".join(sorted({'imprecise','indirect'} & set(r['down']))) if {'imprecise','indirect'} & set(r['down']) else "")
          + ")。")
    print("  可以做的三件事, 按代价从低到高:")
    print(f"   1. 给这条判据对账: trust predict --bound lower/point → 实测 → trust resolve; "
          f"{CLEAR_N} 条不同预算命中即 +calibrated (记 --bound upper 反而会降级 —— 上界本来就不能否决)")
    print("   2. 真去实现那条方向并端到端测 (statemap add --attempted), 结果不论都是合格产出")
    print("   3. 若只想标'暂不做', 用 --kind open/minimized, **不要写成 refuted**")
    return 2


def cmd_cases(a) -> int:
    """对账落空的条目 = 待修的模型 test_case (dubin 原话: 误差大就作为 test_case)。

    ★盲审揪出 (2026-09-18): 原版用 `flag calibrated` 的 `id` 当"已修"集合, 而 cmd_flag
      写的记录**根本没有 id 字段**, 于是那个集合恒为 {None}, 待修表只增不减;
      同一函数的 `return len(rows) and 0` 又让退出码恒为 0, 没法当闸门用。
      现在"已修"与 assess 用**同一条规则**: 一条落空要 CLEAR_N 次后续命中才算清掉。
    """
    led = read_ledger(a.file)
    dmg = _damage_banner()
    print("=== 待修 test_case: 预算与实测偏差超阈值的条目 ===")
    #  按判据分组, 复用 assess 的时序规则找出"还开着的"那些落空
    by_crit: dict = {}
    for r in led:
        if r.get("op") == "resolve":
            by_crit.setdefault(r.get("criterion"), []).append(r)
    rows = []
    for crit, rs in by_crit.items():
        seen, seq = set(), []
        for r in rs:
            if r.get("id") in seen:
                continue
            seen.add(r.get("id")); seq.append(r)
        open_rows, hits_after = [], 0
        for r in seq:
            if r.get("verdict") in ("miss", "violated", "loose"):
                open_rows.append(r); hits_after = 0
            elif r.get("verdict") == "hit":
                hits_after += 1
                if open_rows and hits_after >= CLEAR_N:
                    open_rows.pop(0); hits_after = 0
        rows += open_rows
    if not rows:
        print("  (空) —— 没有**还开着的**对账落空。注意: 没记 predict 就永远不会有落空, "
              "空表也可能意味着**从来没对过账**。")
        print(f"  台账里共 {sum(len(v) for v in by_crit.values())} 条对账记录。")
        return 2 if dmg else 0
    for r in sorted(rows, key=lambda x: x.get("ts", "")):
        print(f"  [{r.get('ts')}] {r.get('criterion')}  预测 {r['predicted']} "
              f"({r.get('bound','point')}) → 实测 {r['actual']} {r.get('unit','')}  "
              f"{r['ratio']}x  [{r['verdict']}]")
        if r.get("note"):
            print(f"        {r['note']}")
    print(f"\n共 {len(rows)} 条。修法: 找出模型里被这个比值指向的那一项参数/系数, 改完"
          f"**新开一条 predict** 再实测对账; 连续 {CLEAR_N} 次命中才算把这条清掉 "
          f"(calibrated 不能手标)。")
    return 1


def cmd_density(a) -> int:
    """产物密度: 关闭类产出 vs 实现类产出。dubin 2026-09-18: 大量探针否定方案, 缺实现探针。"""
    p = a.states
    if not os.path.exists(p):
        print(f"没有台账 {p}", file=sys.stderr)
        return 2
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    #  ★盲审揪出: 原版 built 里把 refuted+attempted 又加了一遍, 同一条记录在分子分母各算一次
    #    (3 条真做过 + 3 条纯推理关掉 -> 真实密度 0.50 而打印 0.33);
    #    样本闸门又是对被放大后的 tot 判, 于是 3 条记录也能绕过"这个数不作数"。
    #    现在每条记录**恰好归一类**, 闸门看的是记录条数。
    closed = [r for r in recs if r.get("kind") == "refuted" and not r.get("attempted")]
    attempted = [r for r in recs if r.get("kind") == "refuted" and r.get("attempted")]
    built = [r for r in recs if r.get("kind") in ("state", "minimized")] + attempted
    tot = len(closed) + len(built)
    d = len(built) / tot if tot else 0.0
    print(f"=== 产物密度 ({p}) ===")
    print(f"  纯推理关掉 (refuted 且没 attempted) {len(closed):3d}")
    print(f"  真做出来测过 (state/minimized/refuted+attempted) {len(built):3d}  其中真试过才关掉的 {len(attempted)}")
    print(f"  密度 = 做出来 / (做出来 + 纯推理关掉) = {d:.2f}   (地板 {DENSITY_FLOOR})")
    #  ★样本太少时这个数不作数, 且必须**显式说出来**。
    #  2026-09-18 第一次跑真 case 就撞到: 台账里只有 1 条 (真实的关闭全写在散文里,
    #  从没走过 statemap), 于是密度算出 1.00 = "健康" —— 而实际形态恰是密度最低的那一种。
    #  "台账是空的"与"密度确实高"长得一样, 是本模块最危险的一种输出。
    if len(recs) < MIN_DENSITY_SAMPLES:
        print(f"\n★**这个数不作数**: 台账里只有 {len(recs)} 条记录 (<{MIN_DENSITY_SAMPLES})。")
        print("  空台账算出的高密度与真的高密度长得一样。若本 case 实际上关掉过方向,")
        print("  说明那些关闭**从没进过台账** —— 先补记 (core.statemap add --kind refuted), 再看这个数。")
        return 2
    if d < DENSITY_FLOOR:
        print("\n★密度偏低: 大量方向被**推理关掉**而没有一条被真做出来测过。")
        print("  这个形态的危险在于它感觉像收敛 —— 候选空间在缩小, 但缩小它的证据从没被验过。")
        print("  处方: 下一跳强制做一次实现 (哪怕不完美), 用它反过来检验模型的预算是不是夸大了。")
        return 1
    return 0


AUDIT_HEADER = """你在做一次**盲审**。下面给你的是一个诊断工具的源码与它自称的能力。
★你拿不到作者的推理过程、也拿不到它服务的那个 case —— 这是有意的:
  同一份上下文喂出来的复核会复现同一个盲点, 那不是独立意见, 是回声。

请只回答三个问题, 每条都要指到**具体行号**:
  1. 这个工具在什么输入下会给出**错的答案**(不是崩, 是给一个看起来合理的错数)?
  2. 它自称防住的那件事, 有没有一条**绕过路径** —— 即照着它的接口正常使用, 却仍然掉进它说要防的坑?
  3. 它的自检里有没有**期望值本身就写错**的 (自检过了但判据是错的)?

不要提风格、命名、类型标注。不要复述它的设计。找不到就明说找不到。
"""


def cmd_audit(a) -> int:
    """出一份**盲审包**: 只给源码与自称能力, 不给作者的推理与 case 上下文。

    ★2026-09-18 立。上一次找人复核时我把自己的 spec 与 case 文件一并给了复核方,
      它于是沿着我的框架复核, 确认了我的结论 —— 同一个盲点复现了一遍。
      独立性不可能来自同一份上下文, 所以这个包**必须减去**东西, 不是增加东西。
    """
    src = os.path.join(ROOT, "core", f"{a.module}.py")
    if not os.path.exists(src):
        print(f"没有 core/{a.module}.py", file=sys.stderr)
        return 2
    body = open(src, encoding="utf-8").read()
    out = a.out or os.path.join(ROOT, "runs", f"audit_{a.module}_{_today()}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(AUDIT_HEADER)
        f.write(f"\n模块: core/{a.module}.py  ({len(body.splitlines())} 行)\n\n```python\n")
        f.write(body)
        f.write("\n```\n")
    print(f"盲审包已写到 {out} ({len(body.splitlines())} 行源码)")
    print("★派给一个**不带本会话上下文**的 subagent 读这个文件, 不要把 case 目录或本会话的结论一并给它。")
    return 0


def cmd_adoption(a) -> int:
    """哪些"会给出否决结论"的模块**还没有在结论处自报资格**。

    ★这张表存在的理由: verdict_banner 只在调用它的模块里起作用。
      漏掉的模块与从来没有这条机制**长得一模一样** —— 而那正是它要防的形态。
    """
    import re as _re
    areas = json.load(open(os.path.join(ROOT, "core", "_areas.json"), encoding="utf-8"))
    judging = [m for a_ in areas["areas"] if a_["id"] in ("model", "lint", "gate")
               for m in a_["mods"]]
    miss = []
    for m in judging:
        f = os.path.join(ROOT, "core", f"{m}.py")
        if os.path.exists(f):
            src = open(f, encoding="utf-8", errors="replace").read()
        else:
            #  包形态的模块 (如 core/opgen/): 扫包内全部 .py —— 原先 open(目录) 直接
            #  IsADirectoryError, 整个 adoption 子命令起不来 (2026-10-20 审计实撞)
            pkg = os.path.join(ROOT, "core", m)
            if not os.path.isdir(pkg):
                continue
            import glob as _glob
            src = "\n".join(open(p, encoding="utf-8", errors="replace").read()
                            for p in sorted(_glob.glob(os.path.join(pkg, "*.py"))))
        if "verdict_banner" not in src:
            miss.append(m)
    print("=== 会给出否决结论、但**结论处不自报资格**的模块 ===")
    print(f"  已接入: {', '.join(sorted(set(judging) - set(miss))) or '(无)'}")
    print(f"  ★仍缺 ({len(miss)}): {', '.join(sorted(miss))}")
    print("  接法: 在打印结论的那一行之后加")
    print("      from core.trust import verdict_banner")
    print("      for ln in verdict_banner('<模块名>', refusing=<本次结论是否为否定>): print(ln)")
    print("  ★不是'提醒去查 trust gate' —— 那与它要防的遗漏是同一形态 (工具不被触发 = 工具不存在)。")
    return 1 if miss else 0


def cmd_banner(a) -> int:
    r = assess(a.criterion, load_refs(), read_ledger(a.file))
    b = "上界" if a.bound == "upper" else ("下界" if a.bound == "lower" else "点估")
    #  ★盲审揪出: 原版 --bound 只决定前半句措辞, 后半句仍看 can_refute, 于是打出过
    #    "本数是上界, 可用于排序与否决" 这种自相矛盾的横幅。上界**恒不可用于否决**。
    ok = r["can_refute"] and a.bound != "upper"
    print(f"[{a.criterion} T{r['level']} {LEVEL_NAME[r['level']]} · 本数是{b}, "
          + ("可用于排序与否决" if ok else "**只可用于排序, 不可用于否决候选**") + "]")
    return 0 if ok else 1


# ----------------------------------------------------------------- 自检
class _R:
    def __init__(self, what, status, evidence, verified):
        self.what, self.status, self.evidence, self.verified = what, status, evidence, verified
    def age_days(self, today=None):
        return (datetime.date.fromisoformat(today or _today()) - datetime.date.fromisoformat(self.verified)).days


def _selftest() -> int:
    T = "2026-09-18"
    refs = {
        "m.fresh":   _R("一条干净的实测判据", "measured", "runs/x 16 点零反例", "2026-09-10"),
        "m.stale":   _R("久未重验", "measured", "runs/y", "2026-01-01"),
        "m.upper":   _R("给的是理想下界, **只可用于排序, 绝对值不可信**", "measured", "runs/z", "2026-09-10"),
        "m.derived": _R("由结构推出", "derived", "推导见注释", "2026-09-10"),
        "m.dead":    _R("已被实测推翻", "refuted", "f640 一测就翻", "2026-09-10"),
    }
    bad = 0

    n_ck = 0

    def ck(name, got, want):
        nonlocal bad, n_ck
        n_ck += 1
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '★不过'} {name}: 得 {got}, 期望 {want}")

    ck("干净实测 = T3 且有资格否决",
       (lambda r: (r["level"], r["can_refute"]))(assess("m.fresh", refs, [], T)), (3, True))
    ck("过期 -1 → T2 且失去否决资格",
       (lambda r: (r["level"], r["can_refute"]))(assess("m.stale", refs, [], T)), (2, False))
    ck("文本里写了'只可用于排序' → imprecise 降级且无资格",
       (lambda r: (r["level"], "imprecise" in r["down"], r["can_refute"]))(assess("m.upper", refs, [], T)),
       (2, True, False))
    ck("derived 起始 T1", assess("m.derived", refs, [], T)["level"], 1)
    ck("refuted 恒为 T0", assess("m.dead", refs, [], T)["level"], 0)
    ck("表里没有的判据 = T0 未验证(unlisted), 不是 refuted",
       (lambda r: (r["level"], r["status"]))(assess("m.nosuch", refs, [], T)), (0, "unlisted"))

    # --- citecheck 判据负例 (C2/C3) ---
    _t = ("见 `<analysis>/msmodel/task_time/hwts_aiv_model.py:66-67`, 又见 "
          "`tools/msopprof/bin/msopprof(9251896 字节) @0x4fab38`, 另有一处裸的 @5221176。")
    _src, _bin, _bare, _ = _cite_scan(_t)
    ck("三类引用各抽到一条", (len(_src), len(_bin), len(_bare)), (1, 1, 1))
    ck("源码引用抽出的是区间不是单行", _src[0][1:], (66, 67))
    ck("C5: `旧→新` 里的旧值是历史记录, 不算活引用 (否则判据对着自己的修复记录报警)",
       _cite_scan("重定位 `hwts_aiv_model.py:137→66-67`")[0], [])
    ck("C5: 同一行里的**新**值仍然要被检 (别把整行都放过)",
       [c[0] for c in _cite_scan("重定位 `a.py:137→66-67` 另见 `b.py:10`")[0]], ["b.py"])
    ck("C5: 普通引用不受影响", len(_cite_scan("见 `x.py:10-12`")[0]), 1)
    ck("C6: `PREFIX=/path` 被拆成前缀与路径", _cite_split_root("AN=/tmp/x"), ("AN", "/tmp/x"))
    ck("C6: 裸路径不被误拆 (绝对路径里也可能有等号)", _cite_split_root("/tmp/a=b"), (None, "/tmp/a=b"))
    _CITE_PREFIXES[:] = ["AN", "INC"]
    ck("C6: $AN/ 前缀被剥掉", _cite_strip_prefix("$AN/x/y.py", _CITE_PREFIXES), "x/y.py")
    ck("C6: <AN>/ 与裸 AN/ 两种写法同样剥掉",
       (_cite_strip_prefix("<AN>/y.py", _CITE_PREFIXES), _cite_strip_prefix("INC/z.h", _CITE_PREFIXES)),
       ("y.py", "z.h"))
    ck("C6: 不在前缀表里的首段**不许**被剥 (否则会把 core/perf.py 剥成 perf.py 乱配)",
       _cite_strip_prefix("core/perf.py", _CITE_PREFIXES), "core/perf.py")
    ck("C7: preset 探测到的 root 都是真实目录 (探不到的必须跳过而不是留下坏 root)",
       all(os.path.isdir(r.split("=", 1)[-1]) for r in _cite_preset_roots()), True)
    #  ★按路径比, 不认目录名。上一版写的是 r.endswith("op_debug_agent") —— 钉的是本仓
    #   当年的签出目录名; 本仓改名 cake-toolkit 之后这条必挂, 而 _cite_preset_roots
    #   一直是对的。判据要守的是"仓根在不在表里", 跟目录叫什么无关。
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ck("C7: preset 至少给出仓根 (按路径比, 不认目录名)", _REPO in _cite_preset_roots(), True)
    #  ★并且要真的走一遍缺 CANN 那一路。上一版的标签写着"换台机器 CANN 没了也要能扫",
    #   但本机 CANN 齐全, 那个分支从没被进过, 断言恒真 —— 标签承诺的事一次也没测。
    def _no_cann_roots():
        import glob as _g
        _env, _real = os.environ.get("ASCEND_HOME_PATH"), _g.glob
        os.environ["ASCEND_HOME_PATH"] = "/nonexistent-cann-for-selftest"
        _g.glob = lambda q, *a, **k: [] if q.startswith("/usr/local/Ascend/cann-") else _real(q, *a, **k)
        try:
            return _cite_preset_roots()
        finally:
            _g.glob = _real
            os.environ.pop("ASCEND_HOME_PATH", None) if _env is None else os.environ.__setitem__("ASCEND_HOME_PATH", _env)
    _NC = _no_cann_roots()
    ck("C7: CANN 探不到时仓根仍在表里 (这才是这条判据的字面意思)", _REPO in _NC, True)
    ck("C7 负例: CANN 探不到时不许留下 Ascend 下的坏 root",
       [r for r in _NC if "/usr/local/Ascend/" in r], [])
    #  ---- 跨篇指针的标识符按位置归属 (09 第 174 条) ----
    #  实撞的那一行 (07 篇 :186) 列了四个指针只写一个标识符, 整行口径让另外三个
    #  **必然**判"附近找不到" —— 15 条"指不到"里 3 条是这么来的。
    _P1 = "- `01:363` 全部选项 + `01:36` 定位 + `01:1184` `ACP_PIPE_FD` + `02:226`"
    _pp, _oo = _ptr_tokens(_P1)
    ck("指针归属: 一行四个指针都被认出来", len(_pp), 4)
    ck("指针归属: 标识符只归给它左边最近的那个 (第三个)",
       [sorted(_oo[i]) for i in range(4)], [[], [], ["ACP_PIPE_FD"], []])
    ck("指针归属: 没有指针的行不返回归属表", _ptr_tokens("`ACP_PIPE_FD` 单独一行")[1], {})
    ck("指针: `[12:20]` 是 Python 切片不是指针 (07 篇 :246 实撞)",
       _ptr_tokens("`FFTS_PMU_FMT`, pmu=filed[12:20]@48")[0], [])
    ck("指针: 切片守卫不许误伤正常指针 (前面是反引号/空格/括号都算正常)",
       [m.group(0) for _p, m in _ptr_tokens("见 `04:503` 与 (06:10) 两处")[0]],
       ["`04:503", "06:10"])
    ck("指针归属: 落在所有指针之前的标识符归第一个",
       sorted(_ptr_tokens("`NS_TO_US` 见 `02:873` 与 `04:514`")[1][0]), ["NS_TO_US"])
    #  ---- C11 裸模块名解到本仓 (09 第 173 条) ----
    #  本机实测: 751 条引用里 499 条直拼路径命中, 249 条靠后缀唯一命中, 3 条绝对路径;
    #  249 条里落在本仓的裸名只有 5 条 —— 全部在 09 篇, 已改成 core/<file>。
    ck("C11: 裸模块名 + 命中落在本仓 = 要拦 (下一次交付把 core/ 拷进 cases/ 就变歧义)",
       _cite_bare_repo_hit("trust.py", os.path.join(_RT_REPO, "core", "trust.py")), True)
    ck("C11: 同一个文件写成 core/trust.py 就不拦 (这正是处方)",
       _cite_bare_repo_hit("core/trust.py", os.path.join(_RT_REPO, "core", "trust.py")), False)
    ck("C11: 裸名命中**外部只读树**不拦 (CANN 版本钉死, 本机 198 条长期稳定)",
       _cite_bare_repo_hit("utils.py", "/usr/local/Ascend/x/common_func/utils.py"), False)
    ck("C11: 绝对路径引用不走这条 (它自带唯一对象)",
       _cite_bare_repo_hit(os.path.join(_RT_REPO, "core/trust.py"),
                           os.path.join(_RT_REPO, "core", "trust.py")), False)
    #  ---- C12 发现编号 F<n> 解不出 (2026-10-29) ----
    #  ★第一条是**判据抓不抓得住自己的动因**。动因是 `F170` 被抄进
    #    `core/_known_failing.json` 而本仓只有 F1~F13 —— 所以扫描范围必须含那个文件,
    #    只扫 docs 的版本抓不住它, 那就是一条"看起来立住了"的判据。
    _c12_doc = "| F7 定义行\n正文里引用 F7 与 F170。\n"
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        os.makedirs(os.path.join(_td, "docs/msprof-re"))
        os.makedirs(os.path.join(_td, "core"))
        open(os.path.join(_td, "docs/msprof-re/x.md"), "w",
             encoding="utf-8").write(_c12_doc)
        open(os.path.join(_td, "core/_known_failing.json"), "w",
             encoding="utf-8").write('[{"原因": "见 F170 的定位"}]')
        _bad, _def = fid_unresolved(_td)
        ck("★C12 正例: 动因那条 (登记表里的 F170) 必须被抓住",
           sorted(_bad), ["170"])
        ck("★C12 判据覆盖登记表, 不只覆盖 docs (只扫 docs 就抓不住动因)",
           any(p.endswith("_known_failing.json") for p, _i, _t in _bad["170"]), True)
        ck("C12 负例: 有定义位的编号不许报 (判据过宽会把全仓染红)",
           "7" in _bad, False)
    ck("C12 负例: dtype 记号 =F32 / /S32 不是引用",
       bool(_C12_USE.search("VMUL=F32 与 VADD=F32/S32")), False)
    ck("C12 负例: 反引号里的提及不算引用 (处方: 引用写裸编号, 提及写 `F170`)",
       bool(_C12_USE.search(_C12_CODE.sub(" ", "外部编号 `F170` 本仓解不出"))), False)
    ck("C12 正例: 裸写的编号算引用 (否则上一条就把判据架空了)",
       bool(_C12_USE.search("见 F170 的定位")), True)
    _CITE_PREFIXES.clear()
    ck("C2: 合规二进制引用**不**被同时记成裸偏移 (否则写对了也报违规)",
       _cite_scan("`a/b(123 字节) @0x10`")[2], [])
    ck("C2: 裸偏移必须被抓住 (这条判据松掉 = 121 个解不开的引用重演)",
       _cite_scan("落在 @5208112 处")[2], ["5208112"])
    ck("C2: 行号里的冒号数字不该被当成裸偏移", _cite_scan("`x.py:350`")[2], [])
    ck("C2: `@1800MHz` 是频率不是偏移 (判据过宽的真实误报之一)",
       _cite_scan("≈ 4494 us @1800MHz")[2], [])
    ck("C2: `@1024³` 是形状不是偏移 (真实误报之二)", _cite_scan("gemm_v3 @1024³: scalar")[2], [])
    # ★以下三条的动因案例是**工具自己犯的错**, 所以直接把那个案例写成自检正例:
    #   `02` 篇先后声明了两个二进制, 原实现只取第一条, 于是 §9.11 的偏移被拿去验了
    #   §9.9 的那个文件 —— 不报错, 只是判据指错对象, 正是 C2 要防的那一类。
    _d2 = "<!-- citecheck:bin a/x.so(11) -->\n见 @0x10\n<!-- citecheck:bin b/y.so(22) -->\n见 @0x20\n"
    ck("C2: 一篇里的多条文件级声明都要认 (不能只取第一条)",
       [(d[0], d[1]) for d in _cite_bin_decl(_d2)], [("a/x.so", 11), ("b/y.so", 22)])
    ck("C2: 裸偏移归**最近的在前声明**, 不是全归第一条",
       [_cite_bin_decl(_d2)[0 if pos < _cite_bin_decl(_d2)[1][2] else 1][0]
        for pos in _cite_scan(_d2)[3]], ["a/x.so", "b/y.so"])
    # ---- 第五类: 二进制日志出处 (`__FILE__:__LINE__` 字面量) ----
    #   动因: 闭源 CANN 的日志行自带 `prof_inner_api.cpp:96`, 长得和源码引用一样,
    #   被判"解不到文件"。判得对但分类错 —— 处置相反 (坏引用要改掉, 这一类要换对象验)。
    _SO = "/usr/local/Ascend/cann-9.1.0-beta.1/aarch64-linux/lib64/libprofapi.so"
    ck("第五类: 声明被解出 (basename -> 二进制 + 字节数)",
       _cite_logsrc_decl("<!-- citecheck:logsrc prof_inner_api.cpp libprofapi.so(198592) -->"),
       {"prof_inner_api.cpp": ("libprofapi.so", 198592)})
    ck("第五类: 没有声明时不认 (默认仍按源码引用处理, 不静默放行)",
       _cite_logsrc_decl("见 prof_inner_api.cpp:96"), {})
    if os.path.isfile(_SO):
        _rt = [os.path.dirname(_SO)]
        ck("第五类: 文件名串真在该二进制里 = 过",
           _cite_check_logsrc("prof_inner_api.cpp", "libprofapi.so", 198592, _rt, "")[0], "ok")
        ck("★第五类: 文件名串**不在**该二进制里必须不过 (receive_data.cpp 属另一个库) ——"
           "没有这一步, 把任意 .cpp 挂到任意 .so 上都能过闸",
           _cite_check_logsrc("receive_data.cpp", "libprofapi.so", 198592, _rt, "")[0], "no-basename")
        ck("★第五类: 字节数不符必须不过 (与 C2 同一条理由: 防验错对象)",
           _cite_check_logsrc("prof_inner_api.cpp", "libprofapi.so", 1, _rt, "")[0], "size-mismatch")
    else:
        print("  跳过 3 条 (本机没有 libprofapi.so)")

    #  ---- docindex: 动因案例 (`03` 篇加索引之前的形态) 当正例 ----
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _d:
        def _w(name, lines, corr, idx):
            t = ("x\n" * lines) + ("已被订正\n" * corr) + ("★ 前置索引\n" if idx else "")
            open(os.path.join(_d, name), "w", encoding="utf-8").write(t)
        _w("long_corr_noidx.md", 1300, 5, False)   # 动因案例的形态
        _w("long_corr_idx.md", 1300, 5, True)
        _w("short_corr.md", 50, 9, False)
        _w("long_nocorr.md", 1300, 0, False)
        _hit = {os.path.basename(x[0]) for x in docs_missing_index(_d)}
        ck("docindex 正例: 长 + 多处订正 + 无索引 必须报 (`03` 篇加索引前的形态)",
           "long_corr_noidx.md" in _hit, True)
        ck("docindex 负例: 有索引就不报", "long_corr_idx.md" in _hit, False)
        ck("docindex 负例: 短文档不报 (短到能通读, 从中间读不是问题)",
           "short_corr.md" in _hit, False)
        ck("docindex 负例: 没被订正过的长文档不报 (前后不一致才是真风险)",
           "long_nocorr.md" in _hit, False)
    ck("C2: `(data)` 标记 (.rodata 跳转表) 与 `(code)` 同等对待, 不按字符串验",
       _cite_scan("`p/q.so(7) @0x237e4(data)`")[1][0][2], "0x237e4(data)")
    ck("C2: 内联合规形态也要收 0x 十六进制与 (code) 后缀 (原式在此抛 ValueError)",
       [c[2] for c in _cite_scan("`p/q.so(7) @0x1edeac(code)`")[1]], ["0x1edeac(code)"])
    _g = _cite_git_resolve(["20260921", "1000003", "0000000000000000000000000000000000000000"], _RT_REPO)
    ck("C4: 不是 commit 的十六进制串不得记为 git 引用 (随机种子 20260921)", _g.get("20260921"), None)
    ck("C4: 十进制常数 1000003 同理 (它恰好也是合法十六进制)", _g.get("1000003"), None)
    ck("C2: 收窄之后真偏移仍然抓得住 (别把判据收成永远不报)",
       _cite_scan("落在 @5208112 处")[2], ["5208112"])
    _rt = os.path.dirname(os.path.abspath(__file__))
    ck("C3: 越界行号判得出 (trust.py 自己没有第 999999 行)",
       _cite_resolve("trust.py", _rt, "")[1] is not None, True)
    ck("C3: root 下不存在的文件 = 解不到, 不回退搜全盘",
       _cite_resolve("no_such_file_xyz.py", _rt, "")[0], None)
    ck("C3: 占位符被剥掉后按相对路径解 (剥不掉则永远解不到)",
       _cite_resolve("<R>/trust.py", _rt, "<R>")[0] is not None, True)

    _sz = os.path.getsize(os.path.join(_rt, "trust.py"))
    ck("C2: 字节数对不上必须判 size-mismatch (这一步没有, 就会拿 msopprof 去验 msprof 的偏移)",
       _cite_check_bin("trust.py", _sz + 1, "0x10", [_rt], "")[0], "size-mismatch")
    ck("C2: 字节数对得上才往下验偏移 (判定不再是 size-mismatch)",
       _cite_check_bin("trust.py", _sz, "0x10", [_rt], "")[0] != "size-mismatch", True)
    ck("C2: 文件解不到时不得报'偏移不对'(两种毛病不能混成一种)",
       _cite_check_bin("no_such_bin_xyz", 1, "0x0", [_rt], "")[0], "no-file")
    ck("C2: 代码段偏移带 (code) 标记, 不按字符串验 (否则一条合法引用把全篇判红)",
       _cite_check_bin("trust.py", os.path.getsize(os.path.join(_rt, "trust.py")),
                       "0x10(code)", [_rt], "")[0], "code")

    #  ★升降不对称: 文本启发不能升级
    up_txt = {"m.oos": _R("这条做过样本外检验, 跨 case 存活", "derived", "样本外四点全中", "2026-09-10")}
    ck("文本里写了'样本外'也**不升级** (升级只认台账)",
       assess("m.oos", up_txt, [], T)["level"], 1)
    ck("台账里两个不同 case 的端到端确认 → +cross_case",
       assess("m.derived", refs, [
           {"op": "vote", "criterion": "m.derived", "outcome": "confirmed", "kind": "end_to_end", "case": "a"},
           {"op": "vote", "criterion": "m.derived", "outcome": "confirmed", "kind": "end_to_end", "case": "b"}], T)["level"], 2)
    ck("同一个 case 投两票不算跨 case",
       assess("m.derived", refs, [
           {"op": "vote", "criterion": "m.derived", "outcome": "confirmed", "kind": "end_to_end", "case": "a"},
           {"op": "vote", "criterion": "m.derived", "outcome": "confirmed", "kind": "end_to_end", "case": "a"}], T)["level"], 1)
    ck("对账落空 → 降级并失去否决资格",
       (lambda r: (r["level"], r["can_refute"]))(assess("m.fresh", refs, [
           {"op": "resolve", "criterion": "m.fresh", "verdict": "miss"}], T)), (2, False))
    #  ★下面三条的夹具 2026-09-18 盲审后重写: 原夹具的 resolve 记录**没有 id**,
    #  于是"两条 hit 行"被当成两次对账 —— 而真实语义是"两条**不同预算**各对账一次"。
    #  自检过了, 但它编码的判据 (台账里有两条 hit 行) 与要验的性质 (两次独立对账) 不是一回事。
    ck("两条不同预算各命中一次 → +calibrated",
       assess("m.fresh", refs, [
           {"op": "resolve", "id": "A", "criterion": "m.fresh", "verdict": "hit"},
           {"op": "resolve", "id": "B", "criterion": "m.fresh", "verdict": "hit"}], T)["level"], 4)
    ck("同一条预算 resolve 两次只算一次对账 (凑不出 calibrated)",
       assess("m.fresh", refs, [
           {"op": "resolve", "id": "A", "criterion": "m.fresh", "verdict": "hit"},
           {"op": "resolve", "id": "A", "criterion": "m.fresh", "verdict": "hit"}], T)["level"], 3)
    ck("两中一空 → 净降一级 (不是抵消), 且失去否决资格",
       (lambda r: (r["level"], r["can_refute"]))(assess("m.fresh", refs, [
           {"op": "resolve", "id": "A", "criterion": "m.fresh", "verdict": "hit"},
           {"op": "resolve", "id": "B", "criterion": "m.fresh", "verdict": "hit"},
           {"op": "resolve", "id": "C", "criterion": "m.fresh", "verdict": "miss"}], T)), (2, False))
    ck(f"落空之后连续 {CLEAR_N} 次命中 -> 清掉, 等级回来 (落空不许永久封死)",
       (lambda r: (r["level"], r["can_refute"]))(assess("m.fresh", refs, [
           {"op": "resolve", "id": "A", "criterion": "m.fresh", "verdict": "miss"},
           {"op": "resolve", "id": "B", "criterion": "m.fresh", "verdict": "hit"},
           {"op": "resolve", "id": "C", "criterion": "m.fresh", "verdict": "hit"}], T)), (4, True))
    ck("通配 criterion 'm.*' 的 flag 落到具体条目上",
       "imprecise" in assess("m.fresh", refs, [
           {"op": "flag", "criterion": "m.*", "flag": "imprecise", "why": "dram_GBps 未标定"}], T)["down"], True)
    ck("等级不会超过 T5",
       assess("m.ini", {"m.ini": _R("可回源", "ini_exact", "ini 逐位相同", "2026-09-10")}, [
           {"op": "vote", "criterion": "m.ini", "outcome": "confirmed", "kind": "end_to_end", "case": "a"},
           {"op": "vote", "criterion": "m.ini", "outcome": "confirmed", "kind": "end_to_end", "case": "b"},
           {"op": "resolve", "id": "A", "criterion": "m.ini", "verdict": "hit"},
           {"op": "resolve", "id": "B", "criterion": "m.ini", "verdict": "hit"}], T)["level"], 5)
    ck("等级不会低于 T0",
       assess("m.derived", refs, [{"op": "flag", "criterion": "m.derived", "flag": f, "why": "x"}
                                  for f in DOWN], T)["level"], 0)
    #  ============ 2026-09-18 盲审回归用例 (每条对应一个被揪出的洞) ============
    ck("盲审#2: --bound upper 的预算 -> upper_only 降级, **上界不得用于否决**",
       (lambda r: ("upper_only" in r["down"], r["can_refute"]))(assess("m.fresh", refs, [
           {"op": "predict", "id": "A", "criterion": "m.fresh", "value": 6.77, "bound": "upper"}], T)),
       (True, False))
    ck("盲审#2: --bound lower 的预算不触发 upper_only",
       "upper_only" in assess("m.fresh", refs, [
           {"op": "predict", "id": "A", "criterion": "m.fresh", "value": 6.77, "bound": "lower"}], T)["down"],
       False)
    ck("盲审#1c: 同一条 flag 敲两遍不多抬一级",
       assess("m.derived", refs, [
           {"op": "flag", "criterion": "m.derived", "flag": "oos", "why": "x"},
           {"op": "flag", "criterion": "m.derived", "flag": "oos", "why": "x"}], T)["level"], 2)
    ck("盲审#2b: 手标升级项合计最多抬 1 级 (自写三行拿不到否决资格)",
       (lambda r: (r["level"], r["can_refute"]))(assess("m.derived", refs, [
           {"op": "flag", "criterion": "m.derived", "flag": f, "why": "x"} for f in MANUAL_UP], T)),
       (2, False))
    ck("盲审#2b: REFS 里没有的判据, 台账再高也不给否决资格",
       (lambda r: (r["level"], r["can_refute"]))(assess("zz.nosuch", refs, [
           {"op": "flag", "criterion": "zz.nosuch", "flag": "oos", "why": "x"},
           {"op": "vote", "criterion": "zz.nosuch", "outcome": "confirmed", "kind": "end_to_end", "case": "a"},
           {"op": "vote", "criterion": "zz.nosuch", "outcome": "confirmed", "kind": "end_to_end", "case": "b"},
           {"op": "resolve", "id": "A", "criterion": "zz.nosuch", "verdict": "hit"},
           {"op": "resolve", "id": "B", "criterion": "zz.nosuch", "verdict": "hit"}], T)), (3, False))
    ck("盲审#1f: ini_exact 也会过期 (原来只有 measured/derived 会)",
       assess("m.old", {"m.old": _R("可回源", "ini_exact", "ini", "2020-01-01")}, [], T)["level"], 3)
    ck(f"盲审#2c: 反驳票可被 {CLEAR_N} 个不同 case 的端到端确认清掉 (不是永久)",
       "contradicted" in assess("m.fresh", refs, [
           {"op": "vote", "criterion": "m.fresh", "outcome": "contradicted", "kind": "reasoning"},
           {"op": "vote", "criterion": "m.fresh", "outcome": "confirmed", "kind": "end_to_end", "case": "a"},
           {"op": "vote", "criterion": "m.fresh", "outcome": "confirmed", "kind": "end_to_end", "case": "b"}],
           T)["down"], False)
    ck("盲审#1h: bound=upper 且实测远小于上界 -> loose(界太松) 而不是 miss",
       "imprecise" in assess("m.fresh", refs, [
           {"op": "resolve", "id": "A", "criterion": "m.fresh", "verdict": "loose"}], T)["down"], True)

    ck("unflag: 成因修好后可撤销降级标记",
       "imprecise" in assess("m.fresh", refs, [
           {"op": "flag", "criterion": "m.fresh", "flag": "imprecise", "why": "未标定"},
           {"op": "unflag", "criterion": "m.fresh", "flag": "imprecise", "why": "已标定"}], T)["down"],
       False)
    ck("unflag 只清它**之前**的同名标记, 之后又出现的重新生效",
       "imprecise" in assess("m.fresh", refs, [
           {"op": "flag", "criterion": "m.fresh", "flag": "imprecise", "why": "未标定"},
           {"op": "unflag", "criterion": "m.fresh", "flag": "imprecise", "why": "已标定"},
           {"op": "flag", "criterion": "m.fresh", "flag": "imprecise", "why": "又发现另一处未标定"}],
           T)["down"], True)

    #  ★密度: 样本太少必须报"不作数"而不是报健康
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(json.dumps({"kind": "state", "to": "localized"}) + "\n")
        few = f.name
    class _A: pass
    a = _A(); a.states = few
    ck("台账只有 1 条时密度报'不作数'(退出码 2), 不报健康", cmd_density(a), 2)
    with open(few, "w", encoding="utf-8") as f:
        for _ in range(6):
            f.write(json.dumps({"kind": "refuted", "hypothesis": "h"}) + "\n")
    ck("6 条全是关闭类 -> 密度 0.00 报低 (退出码 1)", cmd_density(a), 1)
    #  盲审#1a: 3 条真试过 + 3 条纯推理关掉 = 真实密度 0.50, 原版算 0.33 并报低
    with open(few, "w", encoding="utf-8") as f:
        for _ in range(3):
            f.write(json.dumps({"kind": "refuted", "hypothesis": "h", "attempted": True}) + "\n")
        for _ in range(3):
            f.write(json.dumps({"kind": "refuted", "hypothesis": "h"}) + "\n")
    ck("盲审#1a: refuted+attempted 只归一类, 密度 0.50 判健康 (退出码 0)", cmd_density(a), 0)
    #  盲审#1b: 样本闸门看记录条数, 不看被放大后的 tot
    with open(few, "w", encoding="utf-8") as f:
        for _ in range(3):
            f.write(json.dumps({"kind": "refuted", "hypothesis": "h", "attempted": True}) + "\n")
    ck("盲审#1b: 只有 3 条记录 -> 仍报'这个数不作数'(退出码 2)", cmd_density(a), 2)
    os.unlink(few)

    #  ---- 跨主张抵消 (2026-09-22 codex 评审第二条, 夹具已复现原缺陷) ----
    #  原实现只按判据 ID 聚合对账序列, 于是"端到端节省时间"落空可以被"落 DRAM 字节"
    #  的两次命中清掉, 等级 T2 -> T4 且否决资格恢复。证据不是可互换的积分:
    #  一条反例只能由修正模型、明确收窄适用域, 或证明该反例的实验无效来解决。
    _cr = {"c.x": _R("跨主张夹具", "measured", "e", "2026-09-16")}

    def _pair(i, claim, unit, verdict):
        return [{"op": "predict", "id": i, "criterion": "c.x", "claim": claim,
                 "unit": unit, "value": "1", "case": "t", "ts": "2026-09-16"},
                {"op": "resolve", "id": i, "criterion": "c.x", "unit": unit,
                 "verdict": verdict, "case": "t", "ts": "2026-09-16"}]

    _cross = (_pair("P1", "端到端节省时间", "ms", "miss")
              + _pair("P2", "落 DRAM 字节", "GB", "hit")
              + _pair("P3", "落 DRAM 字节", "GB", "hit"))
    _a = assess("c.x", _cr, _cross, today=T)
    ck("跨主张: 时间落空**不得**被两次字节命中清掉 (calib_miss 仍在)",
       "calib_miss" in _a["down"], True)
    ck("跨主张: 未清落空时不得恢复否决资格", _a.get("can_refute"), False)
    _same = (_pair("P1", "端到端节省时间", "ms", "miss")
             + _pair("P2", "端到端节省时间", "ms", "hit")
             + _pair("P3", "端到端节省时间", "ms", "hit"))
    _b = assess("c.x", _cr, _same, today=T)
    ck("★正例: **同主张**两次命中必须能清掉落空 (不能只会降不会升)",
       "calib_miss" not in _b["down"] and "calibrated" in _b["up"], True)
    _split = _pair("P2", "落 DRAM 字节", "GB", "hit") + _pair("P3", "端到端节省时间", "ms", "hit")
    _c = assess("c.x", _cr, _split, today=T)
    ck("跨主张: 两个主张各命中一次**不累加**成 calibrated",
       "calibrated" in _c["up"], False)

    #  ---- deadcatch: 宽且静默的 except 把 NameError 吞成阴性结果 ----
    _DC_REAL = ("import glob\n"
                "def f(p):\n"
                "    try:\n"
                "        tree = ast.parse(open(p).read())\n"
                "        return tree\n"
                "    except Exception:\n"
                "        return ''\n")
    ck("负例: try 里用了没 import 的 ast, 而 except 宽且静默 -> 必须报出",
       [m for _ln, m in deadcatch_src(_DC_REAL)], [["ast"]])
    ck("★正例: 补上 import 之后不报 (这正是 09 第 87 条的修法)",
       deadcatch_src("import ast\n" + _DC_REAL), [])
    ck("★正例: except 收窄到 SyntaxError 就不是本检查的对象 (宽度才是问题)",
       deadcatch_src(_DC_REAL.replace("except Exception", "except SyntaxError")), [])
    ck("★正例: except 块里有 print 就不算静默 (出了事会说话)",
       deadcatch_src(_DC_REAL.replace("return ''", "print('x')")), [])
    ck("★正例: 名字定义过就不报 (只抓**未定义**的)",
       deadcatch_src("import glob\ndef f(p):\n    try:\n        return glob.glob(p)\n"
                     "    except Exception:\n        return []\n"), [])
    ck("★正例: 语法错的源码返回空表而不是抛 (判不了就不判)",
       deadcatch_src("def ( bad syntax"), [])

    #  ---- silentskip: 不进任何分母的跳过 (09 第 176 条) ----
    #  夹具照着 envcheck E8 的真实形状写: 先解析出一张表, 再逐项查,
    #  查不到的那一项 `continue` 掉, 而末尾照样打一个总结数。
    _SS_BAD = ("def report(items, tbl):\n"
               "    bad = 0\n"
               "    for it in items:\n"
               "        v = tbl.get(it)\n"
               "        if v is None:\n"
               "            continue\n"
               "        if v > 90:\n"
               "            bad += 1\n"
               "    print(f'{bad} 条')\n")
    ck("silentskip 负例: 跳过点在任何计数之前且不留痕 -> 必须报出",
       [(f, c) for _ln, f, c in silentskip_src(_SS_BAD)], [("report", ("bad",))])
    ck("★正例: 跳过时 print 一句就不算静默 (那一条说得出去向)",
       silentskip_src(_SS_BAD.replace("            continue\n",
                                      "            print(it); continue\n")), [])
    ck("★正例: 先 `tot += 1` 再跳过 -> 这一条仍在分母里, 不报",
       silentskip_src(_SS_BAD.replace("        v = tbl.get(it)\n",
                                      "        bad += 0\n        v = tbl.get(it)\n")), [])
    ck("★正例: 函数不打印计数就不在本检查的范围内",
       silentskip_src(_SS_BAD.replace("    print(f'{bad} 条')\n", "    return bad\n")), [])
    ck("★正例: 语法错的源码返回空表而不是抛 (与 deadcatch 同口径)",
       silentskip_src("def ( bad syntax"), [])
    #  ---- ssok 标注 (2026-10-26, 09 第 177 条) ----
    _SS_OK = _SS_BAD.replace("        if v is None:\n",
                             "        if v is None:   # ssok: 表里没有的不在口径内\n")
    ck("ssok: 带理由的标注 -> 判为已判, 理由取得出来",
       [(w, bool(mk)) for _l, _f, _c, w, mk in silentskip_src(_SS_OK, with_marks=True)],
       [("表里没有的不在口径内", True)])
    ck("★ssok 不是消音: 结构仍然是命中, 只是带上了理由",
       [(f, c) for _l, f, c in silentskip_src(_SS_OK)], [("report", ("bad",))])
    ck("★负例: 没有理由的裸 `# ssok:` 不算判过 (理由是这条标注的全部价值)",
       [w for _l, _f, _c, w, _m in silentskip_src(
           _SS_BAD.replace("        if v is None:\n",
                           "        if v is None:   # ssok:\n"), with_marks=True)], [None])
    ck("★负例: 字符串字面量里的 ssok: 不算标注 (只认真注释)",
       [w for _l, _f, _c, w, _m in silentskip_src(
           _SS_BAD.replace("    bad = 0\n", "    bad = 0\n    s = '# ssok: 假的'\n"),
           with_marks=True)], [None])
    ck("★负例: 标注贴在别处 (函数首行) 不生效 —— 否则一句话能消掉整函数的命中",
       [w for _l, _f, _c, w, _m in silentskip_src(
           _SS_BAD.replace("    bad = 0\n", "    bad = 0   # ssok: 贴错地方\n"),
           with_marks=True)], [None])
    #  ★2026-10-26 实撞的那一例: 前一处命中行尾的标注不许顺延到紧接着的下一处。
    _SS_TWO = ("def report(items, tbl):\n"
               "    bad = 0\n"
               "    for it in items:\n"
               "        if it is None:\n"
               "            continue   # ssok: 前一处的理由\n"
               "        try:\n"
               "            v = tbl[it]\n"
               "        except KeyError:\n"
               "            continue\n"
               "        if v > 90:\n"
               "            bad += 1\n"
               "    print(f'{bad} 条')\n")
    ck("★负例: 前一处命中的标注不许顺延到下一处 (opbuild 457/461 实撞)",
       [(l, w) for l, _f, _c, w, _m in silentskip_src(_SS_TWO, with_marks=True)],
       [(5, "前一处的理由"), (9, None)])
    ck("★负例: 占位形式 `ssok: <理由>` 不算标注 (讲解本约定的注释会被误收)",
       _ss_marks("x = 1  # ssok: <理由>\n"), {})
    ck("_ss_marks: tokenize 失败时报 -1 而不是报'没有标注'",
       -1 in _ss_marks("def ( bad syntax  # ssok: x\n"), True)
    #  ★收文件这一步也要有判据: 第一次真跑时就是它漏了 core/opdbg (无后缀的入口脚本),
    #  仓内工具报 26 处而会话内临时脚本报 29 处, 差的 5 处全在那个文件里。
    _ss_files = _ss_pyfiles([os.path.join(_RT_REPO, "core")])
    ck("收文件: 无后缀但带 python shebang 的脚本要收进来 (core/opdbg)",
       any(os.path.basename(f) == "opdbg" for f in _ss_files), True)
    ck("收文件: 子目录里的 .py 也要收 (core/opgen/)",
       any("/opgen/" in f for f in _ss_files), True)
    ck("收文件: 带别的后缀的文件不许混进来 (.json/.md/.so)",
       [f for f in _ss_files
        if os.path.splitext(f)[1] not in ("", ".py")], [])

    #  ---- constcheck: 三类必须分得开, 否则满屏误报 ----
    import tempfile as _tfc
    with _tfc.TemporaryDirectory() as _dc:
        _f = os.path.join(_dc, "m.py")
        open(_f, "w", encoding="utf-8").write(
            "#  来源: msparser/stars/stars_log_parser.py:28 DEFAULT_FMT_SIZE\n"
            "FRAME = 64\n"
            "#  来源: common_func/ms_constant/number_constant.py\n"   # 路径无行号, 仍算来源
            "MAGIC = 23130\n"
            "#  本仓自定: 这个阈值外部没有来源可引\n"
            "STALE_DAYS = 90\n"
            "BARE = 128\n"                                           # 什么都没有
            "N_SEEN = 0\n"                                           # 计数器
            "DERIVED = 8 * 1024\n"                                   # 表达式, 不在范围内
            "\n\ndef bump():\n    global N_SEEN\n    N_SEEN += 1\n")
        _r = constcheck([_f])
        _names = {k: {x[2] for x in v} for k, v in _r.items()}
        ck("负例: 什么来源都没有的常数必须报出来 (BARE)",
           "BARE" in _names["missing"], True)
        ck("★正例: 带 file:line 的算有来源 (FRAME)", "FRAME" in _names["sourced"], True)
        ck("★正例: 只写文件名没写行号**也算**来源 —— 能不能解到行是 citecheck 的事",
           "MAGIC" in _names["sourced"], True)
        ck("★正例: 标了`本仓自定`的不报 (外部没有来源可引的阈值)",
           "STALE_DAYS" in _names["selfset"], True)
        ck("★正例: 被 global 赋值的是**计数器不是常数**, 跳过 (第一版把它报成缺来源)",
           "N_SEEN" in _names["mutable"], True)
        ck("★正例: 表达式赋值不在范围内 (它自带推导, 不该要求再引来源)",
           all("DERIVED" not in v for v in _names.values()), True)
        _f2 = os.path.join(_dc, "n.py")
        open(_f2, "w", encoding="utf-8").write(
            "#  来源: a/b.py:12\n"
            "FIRST = 1\n"
            "SECOND = 2\n")   # 紧跟其后, 但那条来源是 FIRST 的
        _n2 = {k: {x[2] for x in v} for k, v in constcheck([_f2]).items()}
        ck("★正例: 上下文以前一条语句为界 —— 前一个常数的来源不算后一个的",
           ("FIRST" in _n2["sourced"] and "SECOND" in _n2["missing"]), True)

    #  ---- constcheck 棘轮: 历史欠账不挡路, 只拦变多 ----
    _HEAD = "A = 1\nB = 2\n"                       # 两条都缺来源 (基线 2)
    _SAME = "A = 1\nB = 2\n"
    _MORE = "A = 1\nB = 2\nC = 3\n"               # 新增一条缺来源
    _FIXED = "#  来源: a/b.py:1\nA = 1\nB = 2\n"  # 补了一条
    _w = lambda now, head: constcheck_ratchet([("m.py", now, head)])
    ck("★正例: 历史欠账原样留着**不拦** (全量设闸会训练出 --no-verify)",
       _w(_SAME, _HEAD), [])
    ck("负例: 新增一条缺来源必须拦", bool(_w(_MORE, _HEAD)), True)
    ck("★正例: 补了来源使缺口变少, 不拦", _w(_FIXED, _HEAD), [])
    ck("★负例: 新增文件基线按 0 算 —— 新文件里一条都不许没来源",
       bool(_w("X = 9\n", None)), True)
    ck("★正例: 新增文件但常数都带来源, 不拦",
       _w("#  来源: a/b.py:1\nX = 9\n", None), [])
    ck("★正例: 语法错的 HEAD 版本不许把基线算成 0 从而误拦 (旧文件本就有欠账)",
       _w(_SAME, "A = 1\nB = 2\ndef (\n"), [])

    #  ---- num: 标记 (09 第 169 条) ----
    #  ★整块受 aprof 的 msopprof 字节闸约束: doc_numbers -> aprof.coverage() ->
    #   binary_colnames(), 字节数不符时 raise SystemExit。2026-10-29 实撞: 本机
    #   cann-9.1.0-beta.1 已撤, tool_path 只解得到 cann-9.0.0 的那份
    #   (9120712 != 所钉的 9251896), SystemExit 就从 _selftest **中间**穿出去,
    #   后面 7 条断言 + `load_refs` 那条 + 收尾的 `trust 自检: N 条` 全没跑,
    #   外面只看到 rc=1, 于是被当成"某条判据不过"登记了一天多。
    #  ★不重钉那个字节数: 实测 9.0.0 这份的列面**确实不同** —— `^aic_(vec|ub)`
    #   只有 19 条, 而 `08` §3.81 记的"59 列里 20 列"属于这一族 (子集 20 > 全集 19),
    #   `--colorder` 3768 vs §3.81 的 4039。闸正在做它该做的事, 重钉会让
    #   201/142/59 与 4039 这些数失去依据。按本文件上面 libprofapi 那处的写法跳过。
    #  ★这一族还有**第二个**前提, 探针查不到: fixture 里写的 `59` 就是 `cov["never"]`,
    #   而 never 依赖 `runs/` 与 `cases/*/runs/` 两棵产物树。2026-10-29 在本检出上实测:
    #   产物树为空 -> appeared 0 / never 201, 于是即便字节数相符这 7 条也会红 (报 2 条
    #   而不是 1 条)。所以跳过信息只陈述探针查过的那一项, **不声称它是唯一的拦路者**。
    def _docnum_ready() -> bool:
        import importlib as _il
        try:
            _ap = _il.import_module("core.aprof")
            return (os.path.exists(_ap.MSOPPROF)
                    and os.path.getsize(_ap.MSOPPROF) == _ap.MSOPPROF_BYTES)
        except (OSError, ImportError, AttributeError):
            return False

    if _docnum_ready():
      try:
          #  ★正例/负例都造在临时目录上, 不依赖真仓的内容 —— 真仓的三个标记是**内容**,
          #    自检要验的是**机制**: 数字不符时它报不报。
          import tempfile as _tf
          _d = _tf.mkdtemp()
          with open(os.path.join(_d, "a.md"), "w") as _f:
              _f.write("| 从未出现 | **59** | <!-- num:never -->\n"
                       "| 从未出现 | **87** | <!-- num:never -->\n"
                       "| 无标记的一行 | **87** |\n")
          with open(os.path.join(_d, "b.md"), "w") as _f:
              #  自指控制的负例: 讲机制的那行把标记写在反引号里, **不该**被当成真标记
              _f.write("行尾写 `<!-- num:never -->`, 键取 coverage() 的 never。写错的值 87。\n")
          _r = doc_numbers(_d)
          ck("★num 标记: 数字对的不报, 错的报, 没标记的不看, 反引号里的示例不算", len(_r), 1)
          ck("★报出来的是那一行写错的值", (_r[0][3], _r[0][4]) if _r else None, (59, [87]))

          #  ---- 2026-10-27 补三条: 旧写法的两个洞 + 弱判据分流 (09 第 195 条) ----
          #  动因: 给 report.html 的一个数挂标记后做负例检验, **把 126 改成 999 竟然不红** ——
          #  同行的括注里还写着一次 126, 加上日期 `2026-10-27` 又贡献三个候选。
          #  判据能过不等于能抓; 这正是同一个会话里早些时候那条恒真判据的同型。
          _d2 = _tf.mkdtemp()
          with open(os.path.join(_d2, "a.md"), "w") as _f:
              _f.write("单位数的量 **7** <!-- num:split_open -->\n"                      # 1
                       "单位数写错 **8** <!-- num:split_open -->\n"                      # 2
                       "同行还有别的数 **59** 与 87 <!-- num:never -->\n"                # 3
                       "只带日期 2026-10-27 的 **59** <!-- num:never -->\n"              # 4
                       "子串不算: 只在 1591 里出现过 <!-- num:never -->\n")             # 5
          _r2 = doc_numbers(_d2)
          _by = {ln: (k, v, got) for _p, ln, k, v, got, _t in _r2}
          ck("★C-num1 单位数的量必须可判 (旧写法下限 2 位, 个位数永远匹配不到 -> 假红)",
             1 in _by, False)
          ck("★C-num1b 单位数写错时必须红 (否则这一族量挂了标记等于没挂)", 2 in _by, True)
          ck("★C-num2 同行有别的数时报**判据弱**而不是不符 (数是对的, 该改的是写法)",
             "判据弱" in _by.get(3, ("", 0, 0))[0], True)
          ck("★C-num3 只带日期不算弱 (日期是每条订正注记都会带的, 不剥就全是弱)",
             4 in _by, False)
          ck("★C-num4 负例: 现算值作为**子串**出现不算命中 (59 在 1591 里)",
             5 in _by and "判据弱" not in _by[5][0], True)
      except SystemExit as _e:
        #  ★探针复刻的是闸的条件而不是闸本身, 两者会漂移, 所以照样兜住:
        #   宁可报"没跑", 不能让它再把后面的断言吞掉。
        print(f"  跳过 7 条 (doc_numbers 拒跑: {str(_e).splitlines()[0]})")
    else:
        print("  跳过 7 条 (msopprof 字节数与本仓所钉的不符; 这一族另需 runs/ 产物树非空,"
              " 见块首注释 —— 两个前提本检出都不满足)")

    #  ★真仓判据必须能被本模块读到 —— 否则等级表是空的而看不出来
    real = load_refs()
    ck("能从真仓读到 >100 条判据 (读不到就是等级表静默变空)", len(real) > 100, True)
    #  ★2026-10-30 补: 上面只验**条数**, 而等级表的失效形态不止"整表变空" ——
    #    **局部坍塌**同样致命且总数看不出来。这条判据就是这么挖出来的:
    #    `opdbg --mutation --traced` 指出中性值清空字段却保留键、len() 不变所以自检绿,
    #    顺着查下去发现 netgate/shguard 的 11 条登记在表里只剩 2 条 (见 load_refs 的注释)。
    #    判据取"键名不得以点结尾" —— 空判据 id 正是坍塌的签名, 且不误伤
    #    pipeplan 那类没有 what 的常量来源登记 (它们是三元组, 本来就不是判据)。
    ck("★等级表不得有空判据 id (键名以点结尾 = 同模块多条塌成一条)",
       [k for k in real if k.endswith(".")], [])
    ck("★列表形态的 REFS 必须逐条进表 (netgate 4 条 + shguard 7 条)",
       (len([k for k in real if k.startswith("netgate.")]),
        len([k for k in real if k.startswith("shguard.")])), (4, 7))
    print(f"trust 自检: {n_ck} 条, {bad} 条未过")
    bad |= selftest_reservecheck()
    return 1 if bad else 0


#  ★常数来源检查 (constcheck, 2026-10-22 立)。
#
#  从哪来 (03 §9.31/§9.32 实撞): `stars_demux.py` 的默认帧长 128 是从邻居
#  `ffts_profile.data` 那里顺过来的, 而 `stars_soc.data` 实际是 64。回扫波及面时
#  一眼就能分出哪些结论受影响 —— 分界线正是**这个数字当初有没有一手来源**:
#  由 struct 串算出来的 (`FFTS_PMU_FMT` -> 128) 一条没中, 靠"看着像"定的那条中了。
#  那条分界线当时只在人脑子里, 没有任何机读形式。本检查就是把它写下来。
#
#  ★它**不判断常数对不对**, 只判断"说得出从哪来吗"。一个标了来源的错常数照样过,
#  但下次回扫时它至少是可追的 —— 而没有来源的常数连重查的入口都没有。
#
#  三类, 必须分开 (混在一起会制造满屏误报):
#    1. 断言**外部事实** (CANN 枚举值 / 帧长 / 硬件参数) -> 必须有来源;
#    2. **本仓自定**的阈值与开关 (`STALE_DAYS` 这种) -> 外部没有来源可引, 标 `本仓自定`;
#    3. 模块级**可变状态** (用 `global` 累加的计数器) -> 根本不是常数, 跳过。
#       第 3 类分得出来: 看有没有函数对它 `global` 后赋值。实测 `reconcile.N_OK`
#       就是这一类, 第一版扫描把它当常数报了。
_PROV_RE = re.compile(
    r"[\w/.$<>+-]+\.(?:py|cpp|h|hpp|cc|so|json|bin)(?::\d+)?"  # 源码路径, 行号可缺
    #  ★行号可缺是故意的: 一条**只写了文件名**的注释仍然说得出来源, 本检查管的就是
    #    "说不说得出"。能不能解到那一行是 `citecheck` 的职责, 两条判据不要互相顶替
    #    (第一版要求必带 :NN, 于是把 msprof_raw.EXPECT_SIZE 那条真来源报成了缺来源)。
    r"|实测|实撞|标定|来源|出自|§\d+\.\d+"                  # 实证 / 文档节号
    r"|\d{4}-\d{2}-\d{2}|REFS|\bRef\(")                  # 日期 / 判据登记
_SELFSET_RE = re.compile(r"本仓自定|本仓定|自定阈值|策略值")
_PROV_CTX_LINES = 9


def _mutable_globals(tree) -> set:
    """找出所有被某个函数 `global` 之后赋值的模块级名字 —— 它们是状态不是常数。"""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            out.update(node.names)
    return out


def _num_const_targets(node) -> list:
    """这条模块级赋值是不是"纯数值"常数? 是则返回被赋的名字, 否则空表。"""
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return []
    tg = node.targets if isinstance(node, ast.Assign) else [node.target]
    if not all(isinstance(t, ast.Name) for t in tg):
        return []
    v = node.value
    if v is None:
        return []

    def _isnum(e):
        return (isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
                and not isinstance(e.value, bool))

    if _isnum(v):
        return [t.id for t in tg]
    if isinstance(v, (ast.Tuple, ast.List, ast.Set)) and v.elts and all(_isnum(e) for e in v.elts):
        return [t.id for t in tg]
    if isinstance(v, ast.Dict) and v.values and all(_isnum(e) for e in v.values):
        return [t.id for t in tg]
    return []


#  ★静默吞异常检查 (deadcatch, 2026-10-22 立)。
#
#  从哪来 (09 第 87 条实撞): `core/expdesign.py` 加了一段用 `ast` 提 docstring 的代码,
#  而该模块**从来没有 `import ast`**。`except Exception` 把 `NameError` 一并接住,
#  于是那段功能对每个模块都静默返回空串 —— 整个功能是空操作, 调用方看不出异常。
#  我据此做了三版"对照实验"并写了一整条总账, 全部建立在那个被吞掉的 NameError 上。
#
#  **宽 except 把「没写对」伪装成「没命中」**: 语法级的错误表现成语义级的阴性结果,
#  而阴性结果不会让人去查代码, 只会让人去编解释。
#
#  ★本检查**不报所有宽 except** (全仓 121 处, 多数合法 —— `int(x)` 失败就跳过这行,
#  异常本身就是预期结果)。只报一种**可静态证实**的形态:
#  try 块里引用了本模块根本没定义的名字, 而异常被宽且静默地吞掉。
_DC_BROAD = {"Exception", "BaseException"}
#  本仓自定: 块里出现这些调用就算有报告, 不算静默。
_DC_REPORTING = {"print", "warn", "error", "exception", "warning", "info", "debug"}


def _dc_broad(h) -> bool:
    if h.type is None:
        return True
    if isinstance(h.type, ast.Name):
        return h.type.id in _DC_BROAD
    if isinstance(h.type, ast.Tuple):
        return any(isinstance(e, ast.Name) and e.id in _DC_BROAD for e in h.type.elts)
    return False


def _dc_silent(h) -> bool:
    """块里没有 raise、也没有任何报告性调用 —— 出了事一声不吭。"""
    for n in ast.walk(ast.Module(body=h.body, type_ignores=[])):
        if isinstance(n, ast.Raise):
            return False
        if isinstance(n, ast.Call):
            f = n.func
            nm = f.id if isinstance(f, ast.Name) else (
                f.attr if isinstance(f, ast.Attribute) else "")
            if nm in _DC_REPORTING:
                return False
    return True


def _dc_defined(tree) -> set:
    """模块里定义过的名字: import / 赋值 / def / class / 形参 / except as / global。"""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.Global):
            out.update(n.names)
    return out


def deadcatch_src(src: str) -> list:
    """返回 [(行号, 未定义的名字表)] —— 被静默吞掉的潜伏 NameError。"""
    import builtins as _b
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    defined = _dc_defined(tree) | set(dir(_b))
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Try):
            continue
        if not any(_dc_broad(h) and _dc_silent(h) for h in n.handlers):
            continue
        used = {m.id for m in ast.walk(ast.Module(body=n.body, type_ignores=[]))
                if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Load)}
        miss = sorted(used - defined)
        if miss:
            out.append((n.lineno, miss))
    return out


#  ★静默跳过检查 (silentskip, 2026-10-24 立, 09 第 176 条)。
#  与 deadcatch 同族: deadcatch 抓"这段代码从未执行过而调用方拿到阴性结果",
#  silentskip 抓"这一条从未被检查过而它不在任何分母里"。两者的共同后果是
#  **一个看着正常的绿灯**, 而且都不会报错。
#  判法: 在一个**会把计数打出来**的函数里, 找出循环体内的 `continue`, 当它
#    (a) 归属的最近循环确实在计数 (有 AugAssign), 且
#    (b) 位于该循环内**任何计数之前**, 且
#    (c) 跳过时不 append / 不 print —— 不留任何痕迹
#  三条同时成立时报出来。实撞: `core/envcheck.py` 的 E8 按这条被抓住,
#  它在本机**一张卡都没解析出来**却照常打"没有占死的卡" (见 09 第 176 条)。
#  ★误报的两个来源已修掉, 都值得记: (1) 内层循环的 continue 被记到外层名下,
#    对外层而言它当然"在计数之前"; (2) 只看计数器自增, 于是"跳过时 append 进一张
#    会被打印的表"也被当成静默。两条修完候选从 64 降到 29。
def _ss_nearest_loop(fn) -> dict:
    """{continue 节点: 它**最近的**那个循环}。嵌套函数不跨。"""
    out = {}

    def walk(node, cur):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            nxt = ch if isinstance(ch, (ast.For, ast.While, ast.AsyncFor)) else cur
            if isinstance(ch, ast.Continue) and cur is not None:
                out[ch] = cur
            walk(ch, nxt)
    walk(fn, None)
    return out


def _ss_records(node) -> bool:
    """这段语句里有没有"留痕"的动作: 计数自增 / append / print。"""
    for x in ast.walk(node):
        if isinstance(x, ast.AugAssign):
            return True
        if isinstance(x, ast.Call):
            f = x.func
            if isinstance(f, ast.Attribute) and f.attr in ("append", "add", "update", "setdefault"):
                return True
            if isinstance(f, ast.Name) and f.id == "print":
                return True
    return False


#  ★逐条判过之后要把结论留在代码旁边 (2026-10-26, 09 第 177 条)。
#  31 处首扫命中里只有 13 处是真缺陷, 其余 18 处是"过滤掉本来就不在这个口径里的输入"。
#  不留标注的话每轮都要把同样的 18 处重判一遍, 而**重判的成本会让人学会整条忽略**。
#  形式: 在 guard 语句那几行里写一句注释 `# ssok: <理由>`。
#  ★标注本身不能变成又一次静默跳过 —— 所以已判的**照样打印**(只是不置退出码),
#    且**标注与命中对不上时要报失效**: 代码改过之后标注会留在原地, 而一条贴错位置的
#    标注会把后来真正出现的命中吞掉。
_SS_MARK = re.compile(r"#.*\bssok:\s*(\S.*?)\s*$")


def _ss_marks(src: str) -> dict:
    """{行号: 理由} —— 只认**真注释** (tokenize), 字符串字面量里的 `ssok:` 不算。"""
    import io as _io
    import tokenize as _tk
    out = {}
    try:
        for tok in _tk.generate_tokens(_io.StringIO(src).readline):
            if tok.type != _tk.COMMENT:
                continue
            m = _SS_MARK.search(tok.string)
            #  ★占位形式 `<理由>` 不算标注 —— 否则**讲解这条约定的那行注释本身**
            #    会被当成一条标注, 然后报成"标注失效" (本检查在自己身上实撞)。
            if m and not m.group(1).startswith("<"):
                out[tok.start[0]] = m.group(1)
    except Exception:                                        # noqa: BLE001
        #  ★不吞成"没有标注": tokenize 失败时返回空表会让**全部**标注失效,
        #    于是已判的 18 处集体回到未判, 看起来像是代码退化了。调用方据
        #    `_SS_MARK_BROKEN` 区分"没有标注"与"标注读不出来"。
        out[-1] = ""
    return out


def silentskip_src(src: str, with_marks: bool = False) -> list:
    """返回 [(行号, 函数名, 该循环的计数器名)] —— 不进任何分母的跳过。

    `with_marks=True` 时每项追加 (理由 或 None, 标注所在行 或 0)。
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    marks = _ss_marks(src) if with_marks else {}
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                   and x.func.id == "print" for x in ast.walk(fn)):
            continue                      # 不打印计数的函数不在此列
        for cont, loop in _ss_nearest_loop(fn).items():
            ctr = sorted({t.id for x in ast.walk(loop) if isinstance(x, ast.AugAssign)
                          for t in [x.target] if isinstance(t, ast.Name)})
            if not ctr:
                continue
            seen, guard = False, None
            for st in loop.body:
                if any(c is cont for c in ast.walk(st)):
                    guard = st
                    break
                if any(isinstance(x, ast.AugAssign) for x in ast.walk(st)):
                    seen = True
            if seen or guard is None or _ss_records(guard):
                continue
            if not with_marks:
                out.append((cont.lineno, fn.name, ctr[:3]))
                continue
            #  标注只认 guard **自己**的行范围。★原来还认 guard 上面那一行, 当天就
            #  吃了亏: `core/opbuild.py:457` 的标注落进了紧邻其下那条 `try` 的范围里,
            #  把 461 这处真缺陷判成了"已判"。上一行往往是**前一处命中**的结尾。
            lo = getattr(guard, "lineno", cont.lineno)
            hi = getattr(guard, "end_lineno", None) or cont.lineno
            mk = next((l for l in range(lo, hi + 1) if l in marks), 0)
            out.append((cont.lineno, fn.name, ctr[:3], marks.get(mk), mk))
    if not with_marks:
        return sorted(set((a, b, tuple(c)) for a, b, c in out))
    return sorted(set((a, b, tuple(c), d, e) for a, b, c, d, e in out))


def _ss_pyfiles(roots: list) -> list:
    """要扫的 Python 源 —— **包括没有 `.py` 后缀的脚本**。

    ★2026-10-24 首次运行时发现的缺口: 只按 `**/*.py` 收文件会漏掉 `core/opdbg`,
    而它是本仓的一站式入口, 前面两轮恰恰在它里面抓出过问题 (09 第 171 条的"32 条"、
    第 176 条扫描里的 5 处)。会话内的临时脚本手工把它加在列表末尾, 落进 `core/` 时
    丢了这一句 —— **工具化会丢掉临时脚本里那些"手工加的一行"**, 两边的数对不上才看见。
    判法用 shebang, 不用文件名白名单: 名字会变, `#!...python` 不会。
    """
    import glob as _g
    out = []
    for r in roots:
        if os.path.isfile(r):
            out.append(r)
            continue
        if not os.path.isdir(r):
            continue
        for p in sorted(_g.glob(os.path.join(r, "**", "*"), recursive=True)):
            if not os.path.isfile(p):
                continue
            if p.endswith(".py"):
                out.append(p)
                continue
            if os.path.splitext(p)[1]:          # 有别的后缀 -> 不是脚本
                continue
            try:
                with open(p, "rb") as fh:
                    head = fh.readline(200)
            except OSError:
                continue
            if head.startswith(b"#!") and b"python" in head:
                out.append(p)
    return sorted(set(out))


def print_silentskip(roots: list) -> int:
    files = _ss_pyfiles(roots)
    hits, unread, judged, stale, broken = [], [], [], [], []
    for f in files:
        try:
            src = open(f, encoding="utf-8").read()
        except OSError as e:
            #  ★这一条是**本检查对自己的应用**: 原来这里也是光秃秃的 continue,
            #  读不开的文件既不扫也不报, 而抬头那句"N 个文件"照样把它算进去 ——
            #  正是本检查要抓的形状。读不开就报出来, 别让它混在"扫过了"里。
            unread.append((f, type(e).__name__))
            continue
        marks = _ss_marks(src)
        if -1 in marks:
            broken.append(f)
            marks.pop(-1)
        used = set()
        for ln, fn, ctr, why, mk in silentskip_src(src, with_marks=True):
            (judged if why else hits).append((f, ln, fn, ctr, why))
            if mk:
                used.add(mk)
        stale += [(f, l, marks[l]) for l in sorted(set(marks) - used)]
    print(f"=== 静默跳过检查 (core/trust silentskip) === "
          f"{len(files) - len(unread)}/{len(files)} 个文件扫过")
    for f, why in unread:
        print(f"  ★读不开, **没扫**: {f} ({why})")
    print("  判的是: 统计型循环里**不留痕的跳过** —— 这一条既不计数也不记录,")
    print("  于是它从分母里消失, 覆盖率反而变好看。")
    if judged:
        print(f"\n  已判为口径内 (`# ssok:`) **{len(judged)} 处** —— 不置退出码, "
              f"但**照样列出来**: 一条标注贴错位置会把后来的真命中吞掉。")
        for f, ln, fn, _c, why in judged:
            print(f"    · {f}:{ln}  {fn}()  {why[:72]}")
    for f in broken:
        print(f"  ★**标注读不出来** (tokenize 失败), 该文件按未判算: {f}")
    for f, ln, why in stale:
        print(f"  ★**标注失效**: {f}:{ln} 写着 `ssok: {why[:40]}`, "
              f"但这一行附近现在**不是命中** —— 代码改过了, 标注要跟着挪或删")
    if not hits:
        print("\n  没有未判的命中。")
        return 1 if (stale or broken) else 0
    print(f"\n  **未判 {len(hits)} 处**:")
    for f, ln, fn, ctr, _w in hits:
        print(f"    {f}:{ln}  {fn}()  该循环的计数器: {', '.join(ctr)}")
    print("\n  ★**命中不等于错**: 过滤掉本来就不属于这个统计口径的输入是对的")
    print("    (例 `if not f.endswith('.py'): continue`)。要逐条看的是:")
    print("    **被跳掉的那一条, 本来算不算在打印出来的那个数里**。")
    print("    判完在 guard 那几行里写 `# ssok: <理由>`, 下一轮就不必重判。")
    return 1


def print_deadcatch(roots: list) -> int:
    import glob as _g
    files = []
    for r in roots:
        if os.path.isdir(r):
            files += sorted(_g.glob(os.path.join(r, "**", "*.py"), recursive=True))
        elif r.endswith(".py"):
            files.append(r)
    hits, unread = [], []
    for f in files:
        try:
            src = open(f, encoding="utf-8").read()
        except OSError as e:
            unread.append((f, type(e).__name__))     # 同 silentskip: 读不开要报, 不能混进"扫过了"
            continue
        for ln, miss in deadcatch_src(src):
            hits.append((f, ln, miss))
    print(f"=== 静默吞异常检查 (core/trust deadcatch) === "
          f"{len(files) - len(unread)}/{len(files)} 个文件扫过")
    for f, why in unread:
        print(f"  ★读不开, **没扫**: {f} ({why})")
    if not hits:
        print("  没有「try 块引用未定义名字 + 宽且静默 except」的组合。")
        print("  ★这**不等于**没有宽 except: 本仓有一百多处, 多数合法")
        print("    (`int(x)` 失败就跳过, 异常本身就是预期结果)。")
        print("    本检查只报可静态证实的那一种 —— 把「没写对」吞成「没命中」。")
        return 0
    print(f"  ★**{len(hits)} 处**: try 块引用了本模块没定义的名字, 而异常被静默吞掉\n")
    for f, ln, miss in hits:
        print(f"    {f}:{ln}  未定义: {miss}")
    print("\n  后果: 这段代码**从未真正执行过**, 而调用方拿到的是一个看着正常的阴性结果。")
    return 1


#  ---- 预留标记守卫 --------------------------------------------------------
#  沉淀动机 (2026-10-30 用户指出, 实例是本仓自己的 F210):
#    `vc_l1l0_map.h` 的 L1_DIAG_PING/PONG 注释写着「Phase 3 预留: 对角阵与累加器驻留」,
#    那正是 MindIE Laser 把 softmax 重缩放搬到 cube 的那块地址 —— 早先有人读懂并留了位。
#    2026-10-30 我为了给 PV 做双缓冲, **读着那一行就把四个槽拿走了**, 没问 Phase 3 是什么;
#    两小时后又从 Laser 源码把同一条结论重推了一遍。
#  ★这不是"没记下来", 是"记下来了没取出来"。加台账无用 —— 台账已经 6700 行。
#    能拦住的只有: **动用预留时必须留痕**。本闸不禁止动用, 只不许悄悄动用。
RESERVE_MARKERS = ("预留", "保留给", "待实现", "尚未实现", "暂未", "TODO", "FIXME")
_RESERVE_ACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_reserve_ack.json")


def _reserve_acks() -> list:
    try:
        with open(_RESERVE_ACK, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def reserve_ack(fragment: str, reason: str, owner: str) -> int:
    acks = _reserve_acks()
    acks.append({"fragment": fragment, "reason": reason, "owner": owner,
                 "t": datetime.date.today().isoformat()})
    with open(_RESERVE_ACK, "w", encoding="utf-8") as f:
        json.dump(acks, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"已登记动用预留: {fragment!r}")
    return 0


def reservecheck_diff(diff: str, acks: list = None) -> list:
    """从 `git diff -U0` 文本里挑出**被删且未被重新加入**的预留标记行。

    Explicit 预留/保留给 markers remain protected in every file. Weak TODO or
    not-yet-implemented text in prose is not a resource reservation. Matching
    moves are confined to one file, so copying a marker elsewhere cannot hide
    a deletion. This is a scoped heuristic, not a complete resource proof.
    """
    acks = [a.get("fragment", "") for a in (acks if acks is not None else _reserve_acks())]
    removed, added, path, old_path, out = [], {}, "", "", []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            path, old_path = "", ""
        elif line.startswith("--- a/"):
            old_path = line[6:]
        elif line.startswith("+++ b/"):
            path = line[6:]
        elif line == "+++ /dev/null":
            path = old_path
        elif line.startswith("-") and not line.startswith("---"):
            removed.append((path, line[1:]))
        elif line.startswith("+") and not line.startswith("+++"):
            added.setdefault(path, []).append(line[1:])
    for path, text in removed:
        if not any(m in text for m in RESERVE_MARKERS):
            continue
        explicit = "预留" in text or "保留给" in text
        prose = os.path.splitext(path)[1].lower() in (".md", ".rst", ".txt", ".adoc")
        if prose and not explicit:
            continue
        core = " ".join(text.split())
        same_file = " ".join("\n".join(added.get(path, [])).split())
        if core and core in same_file:
            continue
        if any(a and a in text for a in acks):
            continue
        out.append((path, core))
    return out


def reservecheck(staged: bool = True) -> int:
    cmd = ["git", "diff", "--cached", "-U0"] if staged else ["git", "diff", "-U0"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode:
            print(f"reservecheck: git diff rc={result.returncode}: {result.stderr}", file=sys.stderr)
            return 2
        diff = result.stdout
    except Exception as e:
        print(f"reservecheck: 取 diff 失败 {e}", file=sys.stderr)
        return 2
    hits = reservecheck_diff(diff)
    if not hits:
        return 0
    print("★本次改动动用/删除了带**预留标记**的行, 但没有留痕:", file=sys.stderr)
    for path, text in hits:
        print(f"    {path}: {text[:110]}", file=sys.stderr)
    print("  ★先问一句「这处预留是给什么留的」—— 本仓的 F210 正是读着那一行把四个 L1 槽",
          file=sys.stderr)
    print("    拿走了, 而它留的是 Laser 把 softmax 重缩放搬到 cube 的那块地址。", file=sys.stderr)
    print("  确认要动用就留痕: python -m core.trust reserveack \"<片段>\" "
          "--reason \"...\" --owner \"<归属>\"", file=sys.stderr)
    return 1


def selftest_reservecheck() -> int:
    bad = 0
    print("=== reservecheck 自检 ===")
    D = "+++ b/a.h\n"
    cases = [
        ("正例: 删掉带「预留」的行且未重新加入 -> 必须拦",
         D + "-constexpr int X = 4;   // Phase 3 预留: 对角阵\n+constexpr int X = 5;\n", [], 1),
        ("负例: 同一行只是挪了位置 (删了又加) -> 不拦",
         D + "-constexpr int X = 4;   // Phase 3 预留: 对角阵\n"
             "+constexpr int X = 4;   // Phase 3 预留: 对角阵\n", [], 0),
        ("负例: 删的行没有预留标记 -> 不拦",
         D + "-constexpr int Y = 1;\n+constexpr int Y = 2;\n", [], 0),
        ("负例: 已登记的片段 -> 不拦",
         D + "-constexpr int X = 4;   // Phase 3 预留: 对角阵\n+constexpr int X = 5;\n",
         [{"fragment": "Phase 3 预留"}], 0),
        ("正例: TODO 同样算预留标记",
         D + "-// TODO: 等 CANN 9.1 再开\n+// 开了\n", [], 1),
        ("E99普通规划进度文档不当资源预留",
         "+++ b/docs/plan.md\n-P1尚待实现，P6暂停。\n+P1对象已实现，其他阶段尚待实现。\n", [], 0),
        ("文档明确资源预留仍保护",
         "+++ b/docs/plan.md\n-L1槽位预留给对角阵\n+挪作他用\n", [], 1),
        ("跨文件复制不能洗掉原预留删除",
         D + "-// 预留资源X\n+++ b/other.h\n+// 预留资源X\n", [], 1),
        ("删除整个文件仍保护并保留旧路径",
         "--- a/deleted.h\n+++ /dev/null\n-// 预留资源X\n", [], 1),
    ]
    for title, diff, acks, want in cases:
        got = len(reservecheck_diff(diff, acks))
        ok = (got > 0) == (want > 0)
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} {title} (实得 {got} 条)")
    deleted = reservecheck_diff("--- a/deleted.h\n+++ /dev/null\n-// 预留资源X\n", [])
    ok = bool(deleted) and deleted[0][0] == "deleted.h"
    bad += not ok
    print(f"  {'过' if ok else '**未过**'} 删除文件路径来源保留")
    from unittest.mock import patch
    import contextlib, io
    runtime = [(subprocess.CompletedProcess([], 128, "", "git read failed"), 2),
               (subprocess.CompletedProcess([], 0, "", ""), 0), (OSError("unavailable"), 2)]
    for result, want in runtime:
        kwargs = {"side_effect": result} if isinstance(result, Exception) else {"return_value": result}
        with patch.object(subprocess, "run", **kwargs), contextlib.redirect_stderr(io.StringIO()):
            got = reservecheck()
        ok = got == want
        bad += not ok
        print(f"  {'过' if ok else '**未过**'} diff读取结果/异常不假装通过 (rc={got}, want={want})")
    print(f"reservecheck 自检: {len(cases)+1+len(runtime)} 条, {bad} 条未过")
    return 1 if bad else 0


def constcheck(paths: list) -> dict:
    """返回 {"sourced": [...], "selfset": [...], "mutable": [...], "missing": [...]}。

    每项是 (文件, 行号, 名字)。只看**模块级**纯数值常数 —— 函数内的局部数字不在
    范围内 (它们通常由参数推得, 且作用域小); 含表达式的赋值也不算 (那些自带推导)。
    """
    out = {"sourced": [], "selfset": [], "mutable": [], "missing": []}
    for f in paths:
        try:
            src = open(f, encoding="utf-8").read()
        except OSError:
            continue
        for k, v in constcheck_src(f, src).items():
            out[k] += v
    return out


def constcheck_src(name: str, src: str) -> dict:
    """同 `constcheck`, 但吃源码文本 —— 棘轮要拿 HEAD 版本比, 那份不在磁盘上。"""
    out = {"sourced": [], "selfset": [], "mutable": [], "missing": []}
    for _ in (0,):
        f = name
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        lines = src.splitlines()
        mut = _mutable_globals(tree)
        #  ★上下文**以前一条模块级语句为界**, 再往上最多 _PROV_CTX_LINES 行。
        #    第一版只按"往上数 9 行"取, 于是**前一条常数的来源注释会算到后一条头上** ——
        #    自检里 `BARE = 128` 紧跟在一条带来源的常数后面, 被判成"有来源"。
        #    没有边界的话, 文件头写一句 `来源: xxx.py:1` 就能让整个文件免检。
        _prev_end = 0
        for node in tree.body:
            names = _num_const_targets(node)
            lo = max(_prev_end, node.lineno - 1 - _PROV_CTX_LINES)
            _prev_end = node.end_lineno
            for name in names:
                rec = (f, node.lineno, name)
                if name in mut:
                    out["mutable"].append(rec)
                    continue
                ctx = "\n".join(lines[lo:node.end_lineno])
                if _SELFSET_RE.search(ctx):
                    out["selfset"].append(rec)
                elif _PROV_RE.search(ctx):
                    out["sourced"].append(rec)
                else:
                    out["missing"].append(rec)
    return out


#  ★棘轮 (2026-10-22): 全仓还有 45 条缺来源, 直接设闸会让每次提交都失败, 而
#    "每次都失败的闸"等于没有闸 —— 它只会训练出 `--no-verify` 的习惯
#    (pre-commit 自己的注释里就写着这一条的代价)。
#    所以只拦**本次提交让某个文件的缺来源数变多**: 历史欠账不挡路, 新写的常数必须带来源。
#    与 `gatestate --check` 的登记表是同一个思路的两种实现 —— 那边登记具体项,
#    这边比计数。比计数不需要维护登记表, 代价是同一文件里"补一条又新增一条"看不出来。
def constcheck_ratchet(pairs: list) -> list:
    """pairs = [(路径, 现在的源码, HEAD 的源码或 None)]。返回变差的 [(路径, 旧, 新)]。

    HEAD 侧为 None = 新增文件, 基线按 0 算 —— 新文件里的常数一条都不许没来源。
    """
    worse = []
    for path, now_src, head_src in pairs:
        if not _parses(now_src):
            continue            # 本次就是语法错, 由导入检查那一关去拦, 不在这里重复报
        n_now = len(constcheck_src(path, now_src)["missing"])
        if head_src is None:
            n_head = 0          # 新增文件: 基线 0, 一条都不许没来源
        elif not _parses(head_src):
            #  ★HEAD 版本解析不了 -> **比不了**, 跳过这个文件。
            #    算成 0 会把"正在修一个本来就坏掉的文件"误拦 (自检里有这条负例):
            #    旧文件的欠账会全部算成本次新增的。
            continue
        else:
            n_head = len(constcheck_src(path, head_src)["missing"])
        if n_now > n_head:
            worse.append((path, n_head, n_now))
    return worse


def _parses(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def print_constcheck(paths: list, limit: int = 40) -> int:
    r = constcheck(paths)
    n = sum(len(v) for v in r.values())
    print(f"=== 常数来源检查 (core/trust constcheck) === {len(paths)} 个文件, "
          f"{n} 个模块级数值常数")
    print(f"    有来源 {len(r['sourced'])} · 标为本仓自定 {len(r['selfset'])} · "
          f"可变状态(跳过) {len(r['mutable'])} · **缺来源 {len(r['missing'])}**\n")
    if not r["missing"]:
        print("  没有缺来源的常数。")
        return 0
    print("  下面这些既没引到一手来源, 也没标 `本仓自定` —— 两者选一补上:\n")
    cur = None
    for f, ln, name in r["missing"][:limit]:
        if f != cur:
            cur = f
            print(f"  {f}")
        print(f"    :{ln:<5} {name}")
    if len(r["missing"]) > limit:
        print(f"    … 另有 {len(r['missing']) - limit} 个 (--limit 调)")
    print("\n  ★本检查不判断常数对不对, 只判断**说不说得出从哪来**。"
          "标了来源的错常数照样过,")
    print("    但它至少可追; 没来源的常数连重查的入口都没有 (03 §9.31 的帧长就是这么留下的)。")
    return 1


REFS = {
    "TR5_judged_must_stay_visible": _R(
        ["逐条判过的结论要留在**代码旁边**且**继续显示**, 不能把它做成消音开关。",
         "silentskip 的 `# ssok: 理由` 三条约束: 已判的照样逐条打印 (只是不置退出码) /",
         "标注与命中对不上要报失效 / 没有理由的裸标注不算判过。"],
        "measured",
        ["2026-10-26: 首版允许标注写在 guard **上面那一行**, 当天就吃了亏 —— 上一行往往是",
         " 前一处命中的结尾, 于是一处后缀过滤上的标注顺延进紧邻其下那条 try 的范围,",
         " 把一处真缺陷 (读不开的源文件静默丢, 构建前守卫的需求集因此不完整) 判成了'已判'。",
         " 收紧成只认 guard 自己的行范围后, 同一份代码上未判数从 12 回到 13。",
         " 另一次: 讲解本约定的那行注释自己被收成一条标注, 然后报成'标注失效'。"],
        "2026-10-26"),
    "TR1_upper_bound_cannot_refute": _R(
        ["用**上界**去筛掉候选是反的 —— 上界大不等于收益小; 只有下界小才等于收益小。",
         "本仓把它做成硬闸门: predict 必填 --bound, gate 对 imprecise/上界类判据直接拒。"],
        "measured",
        ["2026-09-18 sfag_train: 参考实现 PR12104 比本仓最好构建快 11.94ms, 而五条方向我全发现过又全否决了。",
         "否决所依据的内存侧下界建立在 core/chip_calib.json 的 dram_GBps=None(uncalibrated) 上,",
         "且 hwfloor 算出的合计超额 6.77ms 对实测差值 11.94ms 偏 1.76 倍 —— 预算本身就低估 1.76x。"],
        "2026-09-18"),
    "TR3_citation_must_be_openable": _R(
        ["一条第三方打不开的引用, 让整条结论退回到'作者说的'。两条硬形式:",
         "C2 二进制偏移必须写成 `<路径>(<字节数>) @0x<偏移>` —— 裸 @ 记违规;",
         "C3 源码引用的 root 由调用方钉死, 工具**不替你搜版本**, 搜出多份即报歧义。"],
        "measured",
        ["2026-10-20 msprof-re 实撞两次: (a) 01 篇 121 个裸 `@<偏移>` 事后 0/121 可解,",
         " 120 个落在代码段上, 且当时没记是哪个二进制多大, 换一份同名文件完全对不上;",
         " (b) 04 篇 4 条 `.py:NN` 行号越界, 重定位时 `find | head -1` 落到 cann-8.5.0,",
         " 而本机有 4 棵 analysis 树 3 个版本 (hwts_aiv_model.py 9.1.0 是 76 行, 8.5.0 是 64 行)。"],
        "2026-10-20"),
    "TR2_trust_must_decay": _R(
        ["只升不降的等级是资历不是信任。stale/代理量/对账落空必须自动降级, 且降级立即生效。"],
        "derived",
        ["设计出处: 反复出现的形态是'一条判据建立时对, 后来 CANN/硬件/形状变了还在被引用'。",
         "本仓已有三次实例 (gmlayout A6 / pipeplan mte3_bw / perf 冲突比分母), 见 core/_refs.py 开头。",
         "外部参照: reputation 系统设计里的 time-decay —— 'A state machine that only climbs is not trust, it is seniority'。"],
        "2026-09-18"),
    "TR3_no_evidence_ne_low_trust": _R(
        ["'表里没有这一条'要显示成 **T0 未验证**, 不是'不可信' —— 两者处方相反:",
         "未验证的处方是**去用它并对账**, 不可信的处方是**别用**。"],
        "derived",
        ["外部参照: 'absence of evidence and evidence of absence are different' (reputation 系统设计通则)。",
         "本仓动机: 24 个模块目前完全没有 REFS, 若一律显示低分, 结果是新工具一建好就没人敢用, RSI 停转。"],
        "2026-09-18"),
    "TR4_votes_need_end_to_end": _R(
        ["只有**端到端实测确认**才计入升级投票; '这个工具被调用了很多次'不是票。"],
        "derived",
        ["动机: 调用次数是使用惯性, 与判据对不对无关。本仓 detsync_lint H25/H26 引入后被调用多次,",
         "而它们因为 printer 的 KeyError **一次都没成功打印过** —— 调用次数在那段时间里是满的。"],
        "2026-09-18"),
}


# ---------------------------------------------------------------- citecheck
# 引用可达性 —— "这条引用现在还指得到吗"
#
# 与本模块其余部分的关系: assess/gate 回答"这条判据现在还准不准", citecheck 回答
# 它背后那条引用**还能不能被第三方打开**。一条指不到的引用不是小瑕疵 —— 它让整条
# 结论退回到"作者说的", 而本模块存在的理由就是不接受这个。
#
# 2026-10-20 立, 代价是两次真撞:
#   (1) msprof-re 的 01 篇写了 121 个裸 `@<偏移>`, 事后**一个都解不开** —— 120 个落在
#       代码段上, 而且当时没写清是哪个二进制、多大, 换一份同名文件就对不上。
#       → 判据 C2: 二进制偏移必须写成 `<路径>(<字节数>) @0x<偏移>`, 裸 @ 一律记为违规。
#   (2) 04 篇的 `<analysis>/...py:NN` 有 4 条行号越界。重定位时我先 `find | head -1`
#       找 analysis 目录, 落到了 cann-8.5.0 那一份 —— 本机有 **4 棵 analysis 树、3 个
#       版本**, 同名文件行数不同 (hwts_aiv_model.py: 9.1.0 是 76 行, 8.5.0 是 64 行)。
#       → 判据 C3: root 必须由调用方钉死, 不替人搜; 搜出多份时报歧义而不是挑一份。

_CITE_SRC = re.compile(r"`?(?P<path>(?:<analysis>/|[\w./<>-]*?)[\w./-]+\.(?:py|cpp|h|hpp|cc|ini|json))"
                       r":(?P<a>\d+)(?:-(?P<b>\d+))?`?")
# 内联合规形态 `<路径>(<字节数>) @<偏移>`。偏移两种写法都收: 十进制, 或 `0x` 开头的十六进制。
# ★2026-10-21 实撞: 原式把 `0x` 当前缀吃掉却只保留裸十六进制, 下游按十进制解析 ——
#   于是本来合法的 `libprofimpl.so(2754496) @0x1edeac(code)` 让 citecheck 自己抛
#   ValueError 崩掉。崩溃点不在引用上, 在工具上, 而它崩在"第一次有人把十六进制写进
#   内联形态"的那一刻 —— 此前十六进制只出现在文件级声明的 BARE 分支里, 两条路径
#   对偏移的语法约定一直不一致, 没人踩到就一直没暴露。现在两条路径收同一种写法。
_CITE_BIN_OK = re.compile(r"(?P<path>[\w./-]+)\((?P<size>\d+)\s*(?:字节|bytes?)?\)\s*"
                          r"@(?P<off>0[xX][0-9a-fA-F]+|\d+)(?P<tag>\((?:code|data)\))?")
# `@0x...(code)` = 指向**代码段**而非字符串的偏移。不打这个标记就会被判 not-a-string ——
# 那是对的判定但不是缺陷, 所以给它一个显式出口, 而不是让一条合法引用把全篇判红。
# 尾部的 (?![A-Za-z³²¹⁰]) 不是洁癖: 不加它, `@1800MHz` (频率) 与 `@1024³` (形状)
# 都会被当成文件偏移报成 C2 违规 —— 两条都是 2026-10-20 全目录扫描时的真实误报。
_CITE_BIN_BARE = re.compile(
    r"(?<![\w(])@(?P<off>0[xX][0-9a-fA-F]+|\d{4,})(?P<tag>\((?:code|data)\))?(?![A-Za-z³²¹⁰])")


_CITE_BIN_DECL = re.compile(r"<!--\s*citecheck:bin\s+(?P<path>\S+?)\((?P<size>\d+)\)\s*-->")


def _cite_bin_decl(text: str):
    """文件级二进制声明 `<!-- citecheck:bin <路径>(<字节数>) -->`。

    为什么要有它: 一篇文档里上百个偏移全指同一个二进制时, 每条内联重复一遍路径与字节数
    只会把正文撑坏, 而**省略它就退回 C2 违规**。声明把归属提到文件级, 一样机读、一样逐条验,
    且**字节数仍在**, 所以"验错对象"这一类仍然被挡住。

    ★2026-10-21 改成返回**全部**声明而不只是第一条: 一篇里可以有多段, 各自管一个二进制。
    原先用 `.search` 只取第一条, 后面的段被静默忽略 —— `02` 篇 §9.11 声明了 `libprofapi.so`,
    而它的偏移全被拿去验 §9.9 声明的 `libprofimpl.so`。**这正是 C2 规则本身要防的"验错对象",
    由实现这条规则的工具犯下**: 它不报错, 只是把判据指向了另一个文件。
    归属按**最近的在前声明**算; 偏移出现在所有声明之前时归第一条 (文件头声明仍覆盖全篇)。
    """
    return [(m.group("path"), int(m.group("size")), m.start())
            for m in _CITE_BIN_DECL.finditer(text)]


# 第五类: **闭源二进制打印出来的 `__FILE__:__LINE__` 字面量** (2026-10-22 加)。
#   来路: 反向 msprof 采集链路时, 日志行自带 `prof_inner_api.cpp:96` 这样的出处。
#   它长得和源码引用一模一样, 于是被 `_CITE_SRC` 抓走并判"解不到文件" —— **判得对但分类错**:
#   那不是一条坏引用, 是一条**验证对象不是文件**的引用 (CANN 闭源, 磁盘上没有这个 .cpp)。
#   两者处置相反: 坏引用要改掉, 这一类要**换个对象去验**。
#   声明形态 `<!-- citecheck:logsrc <basename.cpp> <二进制>(<字节数>) -->`, 验两件事:
#     1. 二进制在不在、字节数对不对 (与 C2 同一条理由: 防验错对象);
#     2. **那个文件名字符串确实在这个二进制里** —— 抄错文件名、或把它挂到不产生这条日志的
#        二进制上, 都会被这一步抓住。实测有区分力: `prof_inner_api.cpp` 与
#        `prof_cann_plugin.cpp` 在 libprofapi.so 里各命中 1 次, 而 `receive_data.cpp` **0 次**
#        (它属于另一个库)。
#   **行号本身验不了**, 没有源码可比 —— 这一点在输出里明写, 不假装验过。
_CITE_LOGSRC_DECL = re.compile(
    r"<!--\s*citecheck:logsrc\s+(?P<base>[\w.-]+\.(?:cpp|cc|h|hpp|py))\s+"
    r"(?P<path>\S+?)\((?P<size>\d+)\)\s*-->")


def _cite_logsrc_decl(text: str) -> dict:
    """文件级声明: 哪些 basename 是二进制里的 `__FILE__` 字面量, 各自归哪个二进制。"""
    return {m.group("base"): (m.group("path"), int(m.group("size")))
            for m in _CITE_LOGSRC_DECL.finditer(text)}


def _cite_check_logsrc(base: str, binpath: str, size: int, roots: list, placeholder: str):
    """验一条 `__FILE__:__LINE__` 字面量引用。返回 (判定, 说明)。"""
    full = None
    for r in (roots or []):
        full, _ = _cite_resolve(binpath, r, placeholder)
        if full is not None:
            break
    if full is None:
        return "no-file", "解不到二进制"
    actual = os.path.getsize(full)
    if actual != size:
        return "size-mismatch", f"字节数不符: 文中 {size}, 实际 {actual} (★多半是验错了对象)"
    tbl = _bin_strings(full)
    if not tbl:
        return "no-strings", "strings 跑不出来, 文件名未验"
    if any(base == s or base in s for s in tbl.values()):
        return "ok", f"`{base}` 在 {os.path.basename(full)} 里 (★行号无源码可比, 未验)"
    return "no-basename", f"★`{base}` 这个串**不在** {os.path.basename(full)} 里"


# 第四类: git commit 哈希当出处。与前三类同源 —— 它一样会指不到 (rebase / 换仓 / 手误),
# 而且一样**看不出自己已经失效**。误报形态也相同: 一串 7+ 位十六进制未必是哈希
# (op-debug-agent-90 2026-10-20 实撞两条: opfuzz 的随机种子 20260921 的子串、
#  alignlint 的分段常数 1000003), 所以只认**真能被 git 解析**的, 解不出就不记为引用。
_CITE_GIT = re.compile(r"(?<![\w])(?P<sha>[0-9a-f]{7,40})(?![\w])")


def _cite_git_resolve(shas: list, repo: str) -> dict:
    """批量问一次 git: {sha: 主题行 or None}。一次 cat-file, 不是每条起一个进程。"""
    if not shas:
        return {}
    import subprocess
    out = {}
    try:
        r = subprocess.run(["git", "-C", repo, "cat-file", "--batch-check=%(objectname) %(objecttype)"],
                           input="\n".join(shas), capture_output=True, text=True, timeout=120)
        for sha, line in zip(shas, r.stdout.splitlines()):
            out[sha] = None if "missing" in line else line.split()[0]
    except Exception:
        return {}
    full = [v for v in out.values() if v]
    if full:
        try:
            r = subprocess.run(["git", "-C", repo, "log", "--no-walk", "--format=%h %s"] + full,
                               capture_output=True, text=True, timeout=120)
            subj = {ln.split(None, 1)[0]: ln.split(None, 1)[1] for ln in r.stdout.splitlines() if " " in ln}
        except Exception:
            subj = {}
        for k, v in list(out.items()):
            if v:
                out[k] = subj.get(v[:len(k)], subj.get(v[:7], "(解到对象, 未取到主题)"))
    return out


def _cite_scan(text: str):
    """把一段 markdown 里的引用抽成三类。返回 (源码引用, 合规二进制引用, 裸偏移)。"""
    # 紧跟 →/-> 的是**改之前的值**, 不是活引用 (总账里 `旧→新` 这种写法)。
    # 不加这条, "记录一条坏引用"就会被判成"有一条坏引用" —— 判据对着自己的修复记录报警,
    # 而这种误报会直接训练人忽略它的输出。2026-10-20 实撞 4 条。
    src = [(m.group("path"), int(m.group("a")), int(m.group("b") or m.group("a")))
           for m in _CITE_SRC.finditer(text)
           if not text[m.end():m.end() + 2].lstrip().startswith(("→", "->"))]
    binok = [(m.group("path"), int(m.group("size")), m.group("off") + (m.group("tag") or ""))
             for m in _CITE_BIN_OK.finditer(text)]
    # 只排除**落在合规引用跨度之内**的那个 @ ——
    # 按"附近若干字符"排除会把紧跟在合规引用后面的裸偏移一并放过 (判据过宽)。
    spans = [m.span() for m in _CITE_BIN_OK.finditer(text)]
    hits = [m for m in _CITE_BIN_BARE.finditer(text)
            if not any(lo <= m.start() < hi for lo, hi in spans)]
    bare = [m.group("off") + (m.group("tag") or "") for m in hits]
    return src, binok, bare, [m.start() for m in hits]


#  ---- 长文档的前置索引 ----
#  ★2026-10-22 实撞 (msprof 逆向): `03` 篇 2300 行按时间顺序写成, 我从第 9 节的缺口表进入,
#  直接去量字节 —— 而 §4.7 早已把那个结构逐字段写明, 连样本条数都对得上。
#  **用三节重推了同一篇前半部分已有的东西, 净增量只有四条。**
#  更要紧的是: 两天前 `08` 篇撞过同一个毛病并为此加了前置索引, **而缓解只打在那一篇上**。
#  一条沉淀如果只落在它被发现的那个位置, 它防住的就只有那一个位置 —— 所以这里做成判据,
#  对全目录生效, 而不是再手写一份索引了事。
_DOCIDX_MARK = "前置索引"
_DOCIDX_CORR = re.compile(r"订正|推翻|已被.*?订正|本节已被|★.*?后补")


def docs_missing_index(root: str, min_lines: int = 1200, min_corr: int = 3) -> list:
    """找**长且被订正过多次、却没有前置索引**的文档。返回 [(路径, 行数, 订正句数)]。

    判据的两个阈值都不是精确值, 是"从中间读会出事"的粗筛:
      · 行数 —— 长到不可能通读;
      · 订正句数 —— 文中已有结论被后面推翻过, 说明**前后不一致是这篇的常态**。
    两个都满足而没有索引 = 下一个从中间进入的人 (多半是三天后的自己) 会拿到过期答案。

    **它不检查索引的内容**, 只检查有没有。内容要人写 —— 工具判不了"这张表指对了没有"。
    """
    out = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.md"), recursive=True)):
        try:
            txt = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        n = txt.count("\n") + 1
        corr = len(_DOCIDX_CORR.findall(txt))
        if n >= min_lines and corr >= min_corr and _DOCIDX_MARK not in txt:
            out.append((p, n, corr))
    return out


def print_docs_missing_index(root: str = "docs", min_lines: int = 1200, min_corr: int = 3) -> int:
    hits = docs_missing_index(root, min_lines, min_corr)
    print(f"=== 长文档缺前置索引: {len(hits)} 篇 (阈值 >= {min_lines} 行且 >= {min_corr} 处订正) ===")
    print("  长 + 被订正过多次 + 没索引 = 从中间读会拿到过期答案。索引的**内容**要人写, 本检查只判有没有。\n")
    for p, n, c in hits:
        print(f"  {os.path.basename(p):36s} {n:5d} 行, {c:3d} 处订正")
    if hits:
        print("\n  补法: 文件开头加一段 `★ 前置索引`, 至少两列 —— "
              "\"要查什么 → 看哪一节(不要重推)\" 与 \"本篇哪些结论被后面订正过\"。")
    return 0


def _cite_rng(l1: int, l2: int) -> str:
    return str(l1) if l1 == l2 else f"{l1}-{l2}"


def _cite_split_root(spec: str):
    """`--root` 支持两种写法: `/path` 或 `PREFIX=/path`。

    后者让文档里的缩写前缀 (`$AN/`, `$INC/`, `<analysis>/`) 各自对上自己的树。
    不支持它的时候, 这类引用会被统一记成"解不到文件" —— 那是**工具的局限**,
    很容易被读成"文档里有 133 条坏引用"。两者的处置完全不同, 所以要分开。
    """
    if "=" in spec and not spec.startswith("/"):
        pre, _, path = spec.partition("=")
        return pre.strip(), path
    return None, spec


def _cite_strip_prefix(path: str, prefixes) -> str:
    for pre in prefixes:
        for form in (f"${pre}/", f"<{pre}>/", f"{pre}/"):
            if path.startswith(form):
                return path[len(form):]
    return path


#  跨篇指针 `NN:LINE` 与引用句里的反引号标识符。**定义在模块级** ——
#  2026-10-24 实撞: 它们原来是 `cmd_citecheck` 的局部变量, 把归属逻辑提成模块级函数后
#  立刻 NameError, 而**整个指针检查块被异常吞掉**, citecheck 照常打完别的行、
#  只是那一段凭空消失。自检替我抓住了 (函数直接被调), 但真正的教训是:
#  **一段只在某个分支里跑的检查, 出错时的表现是"没有输出"而不是"报错"。**
_DOCPTR = re.compile(r"`?(\d\d):(\d+)")
_DOCTOK = re.compile(r"`([A-Za-z_][A-Za-z0-9_()%/.\-]{3,})`")


def _ptr_tokens(line: str):
    """一行里的跨篇指针, 以及**归属它的**反引号标识符 (09 第 174 条)。

    返回 `([(起始位置, match), ...], {指针序号: {标识符}})`。
    每个标识符归给它**左边最近的那个指针**; 落在所有指针之前的归第一个。
    """
    #  ★紧挨在 `[` 右边的不是指针, 是 Python 切片 —— `filed[12:20]` 会被这条正则抓成
    #  "指向第 12 篇第 20 行"。本机实撞 1 条 (07 篇 :246), 它恰好指向一篇不存在的 `12`
    #  所以被静默跳过; 若写的是 `[04:503]` 就会混进计数而没有任何迹象。
    ptrs = [(m.start(), m) for m in _DOCPTR.finditer(line)
            if not (m.start() and line[m.start() - 1] == "[")]
    own = {i: set() for i in range(len(ptrs))}
    if not ptrs:
        return ptrs, own
    for tm in _DOCTOK.finditer(line):
        t = tm.group(1)
        if re.match(r"^\d\d$", t):
            continue
        k = 0
        for j, (pos, _m) in enumerate(ptrs):
            if pos <= tm.start():
                k = j
        own[k].add(t)
    return ptrs, own


def _cite_bare_repo_hit(path: str, full: str) -> bool:
    """这条引用是不是"裸模块名 + 命中落在本仓"? (C11, 09 第 173 条)

    判真的那一刻引用**是可解的** —— C11 要拦的不是现在解不开, 是**现在唯一所以能解**。
    本仓每次交付把整个 `core/` 拷进 `cases/*/submit_*/ut/core/`, 同名多一份,
    `_cite_resolve` 的"唯一命中才算解开"就把这条引用从可解变成解不到。
    落在 CANN / analysis 等只读外部树的裸名不在此列 (版本钉死, 不会被交付包加份)。
    """
    if path.startswith("/"):
        return False
    rel = _cite_strip_prefix(path, _CITE_PREFIXES).lstrip("/")
    if "/" in rel:
        return False
    return os.path.realpath(full).startswith(_RT_REPO + os.sep)


def _cite_resolve(path: str, root: str, placeholder: str):
    """把一条引用路径落到磁盘。返回 (绝对路径 or None, 该文件行数 or None)。

    只在 root 下解析。**不搜全盘** —— 搜出来的那一份未必是作者当时读的那一份。
    """
    if path.startswith("/"):                 # 绝对路径自带 root, 不再往 --root 下拼
        return (path, sum(1 for _ in open(path, "rb"))) if os.path.isfile(path) else (None, None)
    rel = path.replace(placeholder, "").lstrip("/") if placeholder and path.startswith(placeholder) else path
    rel = _cite_strip_prefix(rel, _CITE_PREFIXES).lstrip("/")
    cand = os.path.join(root, rel)
    if os.path.isfile(cand):
        return cand, sum(1 for _ in open(cand, "rb"))
    # 作者常只写末几段路径 (例 `_constant.py:350`), 在 root 下按后缀匹配, 唯一命中才算解开。
    # 索引**按 root 建一次**: 逐条引用各走一遍 os.walk 时, 整棵 CANN 树上单篇文档要跑几分钟。
    idx = _ROOT_INDEX.get(root)
    if idx is None:
        idx = {}
        for dp, _dn, fns in os.walk(root):
            for fn in fns:
                idx.setdefault(fn, []).append(os.path.join(dp, fn))
        _ROOT_INDEX[root] = idx
    hits = [h for h in idx.get(os.path.basename(rel), []) if h.endswith("/" + rel) or h.endswith("/" + os.path.basename(rel)) and rel == os.path.basename(rel)]
    if len(hits) != 1:
        return None, None                      # 0 条 = 没有; >1 条 = 歧义, 不挑一份
    return hits[0], sum(1 for _ in open(hits[0], "rb"))


_STRINGS_CACHE: dict = {}
_ROOT_INDEX: dict = {}
_CITE_PREFIXES: list = []
_RT_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bin_strings(path: str) -> dict:
    """`strings -n 3 -t d` 的 {十进制偏移: 串}。一个二进制只跑一次。"""
    if path not in _STRINGS_CACHE:
        import subprocess
        tbl = {}
        try:
            out = subprocess.run(["strings", "-n", "3", "-t", "d", path],
                                 capture_output=True, text=True, errors="replace", timeout=120).stdout
            for line in out.splitlines():
                q = line.strip().split(None, 1)
                if len(q) == 2 and q[0].isdigit():
                    tbl[int(q[0])] = q[1]
        except Exception:
            tbl = {}
        _STRINGS_CACHE[path] = tbl
    return _STRINGS_CACHE[path]


def _cite_check_bin(path: str, size: int, off: str, roots: list, placeholder: str):
    """验一条二进制偏移引用。返回 (判定, 说明)。

    三步, 缺一步都能让一条引用看起来成立而其实指错对象:
      1. 文件在不在;
      2. **字节数对不对** —— 这一步是 C2 的全部意义。2026-10-20 实撞: 01 篇 121 个偏移
         被拿去验 `msopprof`(9251896 字节), 判了 0/121 "全部落在代码段"; 而文中写的是
         `msprof`(8142256 字节), 换对文件后 **121/121 命中**。字节数一比就不会验错对象;
      3. 偏移落不落在一条可打印串的起点上 (`strings -t d` 的口径)。
    """
    full, _ = _cite_resolve(path, roots[0] if roots else "/", placeholder) if roots else (None, None)
    for r in roots[1:]:
        if full:
            break
        full, _ = _cite_resolve(path, r, placeholder)
    if full is None:
        return "no-file", "解不到文件"
    actual = os.path.getsize(full)
    if actual != size:
        return "size-mismatch", f"字节数不符: 文中 {size}, 实际 {actual} (★多半是验错了对象)"
    if off.endswith(("(code)", "(data)")):
        # `(code)` 代码段 / `(data)` 非字符串的只读数据 (跳转表、常量表)。两者判法相同 ——
        # 本工具只验到"文件存在且字节数相符", 偏移本身要人去 objdump 复核。
        # ★2026-10-21 加 `(data)`: `libprofapi.so @0x237e4` 是 .rodata 里的跳转表,
        # 标成 `(code)` 会过闸但**写的是假话**; 为了让一条引用过闸而误标它的性质,
        # 正是判据开始腐烂的方式。
        return "code", f"非字符串偏移 ({off[-5:-1]}), 本工具只验到文件与字节数 (@{off[:-6]})"
    n = int(off, 16) if off.lower().startswith("0x") else int(off)
    # 十进制与 0x 十六进制都收; (code) 后缀已在上面提前返回。
    tbl = _bin_strings(full)
    if not tbl:
        return "no-strings", "strings 跑不出来, 偏移未验"
    if n in tbl:
        return "ok", tbl[n][:70]
    return "not-a-string", f"@{off} 不是任何可打印串的起点"


def _cite_preset_roots() -> list:
    """`--preset msprof-re` 的 root 集 —— **自己探测, 不让人手拼七个 --root**。

    为什么要有它: 上一版这条命令只写在 README 里, 七个 root 手拼一次要一分钟,
    于是没有人会跑它。本仓的 citecheck 自己就是从"只写在文档里的纪律没人执行"来的,
    这条命令不该重蹈。探测不到的 root **跳过而不是报错** —— 换台机器也要能跑。
    """
    import glob as _g
    cann = os.environ.get("ASCEND_HOME_PATH") or ""
    if not cann or not os.path.isdir(cann):
        cands = sorted(_g.glob("/usr/local/Ascend/cann-*"), reverse=True)
        cann = cands[0] if cands else ""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    if cann:
        an = os.path.join(cann, "tools/profiler/profiler_tool/analysis")
        if os.path.isdir(an):
            out += [f"AN={an}", f"analysis={an}"]
        inc = os.path.join(cann, "aarch64-linux/pkg_inc/profiling")
        if os.path.isdir(inc):
            out.append(f"INC={inc}")
        asc = os.path.join(cann, "aarch64-linux/ascendc/include/basic_api/impl")
        if os.path.isdir(asc):
            out.append(asc)
        out.append(cann)
    # ★只在文件系统上找 torch_npu 的位置, **不 import 它**。
    #  2026-10-20 被 `opdbg --selftest-areas` 当场抓住: trust 登记在 nav 区 (定义是不上机),
    #  而 `import torch_npu` 会真的碰设备。一个静态引用检查器本来就不该拉起设备运行时 ——
    #  这里要的只是一个目录路径, import 是最贵也最不该用的那种拿法。
    for sp in _g.glob("/usr/local/python*/lib/python*/site-packages") + list(sys.path):
        if sp and os.path.isdir(os.path.join(sp, "torch_npu")):
            out.append(sp)
            break
    out.append(repo)
    return out


def cmd_silentskip(a) -> int:
    return print_silentskip(a.paths or ["core"])


def cmd_deadcatch(a) -> int:
    return print_deadcatch(a.paths or ["core", "docs/msprof-re/tools"])


def cmd_constcheck(a) -> int:
    if getattr(a, "staged", False):
        return _constcheck_staged(a.limit)
    import glob as _g
    roots = a.paths or ["core", "docs/msprof-re/tools"]
    files = []
    for r in roots:
        if os.path.isdir(r):
            files += sorted(_g.glob(os.path.join(r, "**", "*.py"), recursive=True))
        elif r.endswith(".py"):
            files.append(r)
    return print_constcheck(files, a.limit)


def _constcheck_staged(limit: int) -> int:
    """棘轮模式: 只看本次 `git add` 过的 py 文件, 拦缺来源数**变多**的那些。"""
    import subprocess as _sp

    def _git(*args):
        r = _sp.run(["git", *args], capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None

    names = _git("diff", "--cached", "--name-only")
    if names is None:
        print("  (不在 git 仓里, 跳过棘轮)")
        return 0
    files = [n for n in names.split()
             if n.endswith(".py")
             and (n.startswith("core/") or n.startswith("docs/msprof-re/tools/"))]
    if not files:
        return 0
    pairs = []
    for f in files:
        try:
            now = open(f, encoding="utf-8").read()
        except OSError:
            continue          # 本次删掉的文件, 不参与
        pairs.append((f, now, _git("show", f"HEAD:{f}")))
    worse = constcheck_ratchet(pairs)
    if not worse:
        print(f"✓ 常数来源棘轮: {len(pairs)} 个文件, 缺来源数没有变多")
        return 0
    print(f"✗ 常数来源棘轮: {len(worse)} 个文件的缺来源数**变多了**")
    for path, old, new in worse[:limit]:
        print(f"    {path}: {old} -> {new}")
        for f2, ln, name in constcheck_src(path, open(path, encoding="utf-8").read())["missing"]:
            print(f"        :{ln:<5} {name}")
    print("  新写的数值常数要么引一手来源 (注释里写 `来源: <文件>:<行>`),")
    print("  要么标 `本仓自定` (外部没有来源可引的阈值)。历史欠账不挡路, 只拦变多。")
    return 1


def cmd_citecheck(a) -> int:
    raw_roots, placeholder = list(a.root), a.placeholder
    if a.preset == "msprof-re":
        raw_roots = _cite_preset_roots() + raw_roots
        if not a.docs:
            a.docs = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "docs", "msprof-re")]
    roots = []
    _CITE_PREFIXES.clear()
    for spec in raw_roots:
        pre, r = _cite_split_root(spec)
        if pre:
            _CITE_PREFIXES.append(pre)
        if not os.path.isdir(r):
            print(f"[错] --root 指不到: {r}")
            return 2
        roots.append(r)
    files = []
    for p in a.docs:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".md"))
        else:
            files.append(p)
    tot = ok = oor = nofile = 0
    bad_rows, bare_rows, binok_rows = [], [], []
    barerepo_rows = []        # 裸模块名 + 命中落在本仓 (见下方 C11)
    #  解法三分 (09 第 173 条): 直拼 root/<写下的路径> / 后缀唯一命中 / 绝对路径。
    #  **第二类是脆的那一类** —— 它成立的前提是同名文件只有一份, 而这个前提会变。
    #  打出来是为了让"靠什么解开的"可数, 不只是"解没解开"。
    n_direct = n_fallback = n_abs = 0
    logsrc_rows = []
    for f in files:
        text = open(f, encoding="utf-8", errors="replace").read()
        src, binok, bare, bare_pos = _cite_scan(text)
        decls = _cite_bin_decl(text)
        if decls:                              # 文件级声明: 裸偏移按它归属, 照样逐条验
            for o, pos in zip(bare, bare_pos):
                d = decls[0]                   # 落在所有声明之前 → 归第一条
                for cand in decls:             # 否则归**最近的在前声明**
                    if cand[2] <= pos:
                        d = cand
                binok.append((d[0], d[1], o))
            bare = []
        base = os.path.basename(f)
        logsrc = _cite_logsrc_decl(text)
        for path, l1, l2 in src:
            pb = os.path.basename(path)
            if pb in logsrc:          # 声明过的字面量: 换对象验, 不按源码路径解
                bp, bsz = logsrc[pb]
                v, why = (_cite_check_logsrc(pb, bp, bsz, roots, placeholder)
                          if roots else ("skip", "未给 --root"))
                logsrc_rows.append((base, f"{path}:{_cite_rng(l1, l2)}", bp, v, why))
                continue
            tot += 1
            if not roots:
                continue
            full = n = None
            for r in roots:               # 多棵树: 第一棵解开就停, 全不开才算解不到
                full, n = _cite_resolve(path, r, placeholder)
                if full is not None:
                    break
            if full is None:
                nofile += 1
                bad_rows.append((base, f"{path}:{_cite_rng(l1, l2)}", "解不到文件 (或 root 下多份同名)"))
            elif max(l1, l2) > n:
                oor += 1
                bad_rows.append((base, f"{path}:{_cite_rng(l1, l2)}", f"行号越界: 末行 {max(l1, l2)} > 该文件 {n} 行"))
            else:
                ok += 1
                if path.startswith("/"):
                    n_abs += 1
                elif os.path.isfile(os.path.join(
                        r, _cite_strip_prefix(path, _CITE_PREFIXES).lstrip("/"))):
                    n_direct += 1
                else:
                    n_fallback += 1
                #  ---- C11: 裸模块名命中本仓 = 今天唯一所以能解 (2026-10-24, 09 第 173 条) ----
                #  `_cite_resolve` 的后缀回退要求**唯一命中**, 所以多一份同名就从"能解"
                #  变"解不到"。本仓的 `core/` 会被每次交付整个拷进
                #  `cases/*/submit_*/ut/core/` —— 实撞: `errattr.py` 已有 6 份, 于是
                #  `errattr.py:15` 这条引用在写下的当天就解不开。
                #  **区别在于命中落在哪棵树**: CANN / analysis 是只读外部树, 版本钉死,
                #  裸名在那里长期稳定 (本机 198 条); 落在本仓的裸名则是定时炸弹,
                #  下一次交付就把它引爆。所以只拦后者, 不拦前者。
                if _cite_bare_repo_hit(path, full):
                    barerepo_rows.append((base, f"{path}:{_cite_rng(l1, l2)}",
                                          os.path.relpath(os.path.realpath(full), _RT_REPO)))
        for path, size, off in binok:
            verdict, why = _cite_check_bin(path, size, off, roots, placeholder) if roots else ("skip", "未给 --root")
            binok_rows.append((base, path, size, off, verdict, why))
        for off in bare:
            bare_rows.append((base, off))

    if a.git_repo:
        shas = sorted({m.group("sha") for f in files
                       for m in _CITE_GIT.finditer(open(f, encoding="utf-8", errors="replace").read())})
        res = _cite_git_resolve(shas, a.git_repo)
        hit = {k: v for k, v in res.items() if v}
        print(f"  git 出处引用   {len(shas)} 个候选十六进制串: 真能解出 commit 的 {len(hit)}, "
              f"其余 {len(shas) - len(hit)} 个不是哈希 (种子/常数等, 不计为引用)")
        for k in list(hit)[:a.limit]:
            print(f"      {k}  {hit[k][:66]}")
    print(f"扫 {len(files)} 份文档")
    print(f"  源码行号引用   {tot} 条", end="")
    if roots:
        print(f": 可解 {ok} (直拼 {n_direct} / 后缀唯一命中 {n_fallback} / 绝对路径 {n_abs}), "
              f"行号越界 {oor}, 解不到文件 {nofile}")
    else:
        print("  (未给 --root, 只统计不解析 —— C3: 不替调用方猜版本)")
    bin_ok = sum(1 for r in binok_rows if r[4] == "ok")
    bin_code = sum(1 for r in binok_rows if r[4] == "code")
    bin_bad = [r for r in binok_rows if r[4] not in ("ok", "skip", "code")]
    print(f"  合规二进制引用 {len(binok_rows)} 条  (`<路径>(<字节数>) @偏移`)"
          + (f": 验过 {bin_ok}, 代码段 {bin_code}, 不符 {len(bin_bad)}" if roots else "  (未给 --root, 未验)"))
    for base, path, size, off, verdict, why in bin_bad[:a.limit]:
        print(f"      ★{base:30s} {os.path.basename(path)}({size}) @{off}  {verdict}: {why}")
    ls_bad = [r for r in logsrc_rows if r[3] not in ("ok", "skip")]
    if logsrc_rows:
        print(f"  二进制日志出处 {len(logsrc_rows)} 条  (`__FILE__:__LINE__` 字面量, 闭源无源码)"
              + (f": 验过 {len(logsrc_rows) - len(ls_bad)}, 不符 {len(ls_bad)}" if roots
                 else "  (未给 --root, 未验)"))
        print("      ★只验到「文件名串在该二进制里」+「字节数相符」; **行号无源码可比, 未验**。")
        for base, cite, bp, verdict, why in ls_bad[:a.limit]:
            print(f"      ★{base:30s} {cite:40s} -> {os.path.basename(bp)}  {verdict}: {why}")
    #  ---- 跨文档行号指针 `NN:LINE` (2026-10-23, 08 §3.87) ----
    #  同一套说明书内部互引时用的是"篇号:行号"。**行号指向的是一份还在长的文档** ——
    #  被引篇每插一段, 后面所有指针就偏一次, 而指针本身永远"在范围内", 不会报错。
    #  实撞: `07` 篇 117 条这种指针里, 可检验的 44 条有 **42 条**已经指不到它要指的东西
    #  (`04:832` 本该是 `block_detail`, 实际落在一段 `ge_summary` 的 SQL 里;
    #   `02:1030` 本该是 `MSTX_INJECTION_PATH`, 实际落在 §8.3 讲记录 magic 的段落)。
    #  判法只用引用句里**自己写出的反引号标识符**: 它在目标行附近找不到 = 指不到。
    #  ★报而不拦: 这是启发式 (引用句没写标识符就判不了, 本机 120 条里 76 条如此),
    #  而且当下的失败率会让它拦下每一次提交。**处方是把行号换成节号** (`§3.82` 这种,
    #  插段不会让它漂), 不是逐条修行号。
    _dir = os.path.dirname(files[0]) if files else ""
    _repo = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True).stdout.strip() or "."
    _rev_miss = set()
    _n_unpinned = 0
    _n_nodoc = 0
    _nodoc_ids: dict = __import__("collections").Counter()
    _unpinned_docs = set()

    def _docs_at(rev):
        """取 `NN` → 行表。`rev` 为 None 用工作区, 否则用 `git show <rev>:<path>`。"""
        out = {}
        for _f in sorted(glob.glob(os.path.join(_dir, "*.md"))):
            _b = os.path.basename(_f)
            if not re.match(r"^\d\d-", _b):
                continue
            if rev is None:
                out[_b[:2]] = open(_f, encoding="utf-8", errors="replace").read().splitlines()
            else:
                #  ★`git show <rev>:<path>` 的 path 必须是**仓内相对路径**;
                #    传绝对路径会静默失败 (returncode!=0), 结果是整张表为空,
                #    而下面的循环对"取不到被引篇"是 `continue` —— **指针会一条不剩地消失**,
                #    计数从 121 掉到 4 而不报任何错。第一版就是这么过去的。
                _rel = os.path.relpath(os.path.abspath(_f), _repo)
                r = subprocess.run(["git", "-C", _repo, "show", f"{rev}:{_rel}"],
                                   capture_output=True, text=True)
                if r.returncode == 0:
                    out[_b[:2]] = r.stdout.splitlines()
                else:
                    _rev_miss.add((rev, _rel))
        return out

    #  ★`citecheck:docrev <rev>` —— 一篇文档可以声明"我的行号指的是哪一版"。
    #  `07` 篇实测: 它引入时 (`6079e95`) 29/44 指得到, 到现在只剩 7/44, **而 0 条越界** ——
    #  行号漂移不会报错, 只会悄悄指到别处。钉版本比逐条改 117 个行号便宜得多,
    #  而且钉住之后不会再漂。新写的跨篇引用用节号 (`§3.82`), 插段不影响。
    #  ★必须是**文件抬头的 HTML 注释**形式, 且只在前 60 行里找。
    #  不设这两条的话, **讲这条机制的文档会把自己声明成已钉版本**:
    #  08 §3.87 与 09 第 139 条的正文里写着 `citecheck:docrev 6079e95` 作为示例,
    #  第一版于是把 08/09 当成已钉, 拿 6079e95 去解它们 (那一版这两篇还不存在)。
    #  与 expdesign 的 `_E16_SELFREF` 是同一类自指控制。
    _REV = re.compile(r"^\s*<!--\s*citecheck:docrev\s+([0-9a-f]{7,40})\s*-->\s*$", re.M)
    _cache = {}
    _n_ptr = _n_ok = _n_bad = _n_na = 0
    _bad_ptr = []
    for _f in files:
        _base = os.path.basename(_f)
        _src = open(_f, encoding="utf-8", errors="replace").read()
        _mr = _REV.search("\n".join(_src.splitlines()[:60]))
        _rev = _mr.group(1) if _mr else None
        #  ★只检验**声明了版本**的那些篇。没声明的只数不判, 理由有两条:
        #  (1) 一个行号指针不说清"指的是哪一版"时, 它本来就没有可检验的对象
        #      (与 C3 "不替调用方猜版本" 同一条);
        #  (2) 否则会把**叙述里引用的指针**当成指针本身 —— 08 §3.87 与 09 第 139 条
        #      正文里写着 `04:832` / `03:250` 当反例, 第一版把它们也数了进去,
        #      计数 121 -> 124、指不到 15 -> 18, 而那 3 条根本不是引用。
        if _rev is None:
            _n_unpinned += 1
            _unpinned_docs.add(_base)
            continue
        if _rev not in _cache:
            _cache[_rev] = _docs_at(_rev)
        _byid = _cache[_rev]
        for _i, _l in enumerate(_src.splitlines(), 1):
            #  ---- 标识符按**位置**归到指针, 不按整行 (09 第 174 条) ----
            #  第一版把**整行**的反引号标识符当成每一个指针的检验集。一行只有一个指针时
            #  没差别, 但本篇是综述, 一行里常列三四个指针:
            #    `- **acp**: 已覆盖 (`01:363-379` 全部选项 + `01:36` 定位 + `01:1184`
            #     `ACP_PIPE_FD` + `02:226`)`
            #  这一行里只有 `ACP_PIPE_FD` 一个标识符, 它属于 `01:1184`; 而整行口径把它
            #  同时派给另外三个指针, 于是那三条**必然**判"附近找不到"。
            #  15 条"指不到"里有 3 条是这么来的 —— **误报, 且全部集中在列举句**。
            #  改成: 每个标识符归给它**左边最近的那个指针**; 落在所有指针之前的归第一个。
            #  归不到标识符的指针记"判不了", 与原本就没标识符的那一类同列。
            _ptrs, _own = _ptr_tokens(_l)
            for _idx, (_pos, _m) in enumerate(_ptrs):
                _tgt = _byid.get(_m.group(1))
                if _tgt is None:
                    #  指向一篇取不到的文档 (该版本里还没有, 或篇号根本不存在)。
                    #  ★原来这里是光秃秃的 `continue` —— 这类指针**既不计入分母也不报**,
                    #  于是"117 条全检过"读起来像全覆盖, 而实际有几条根本没进过检查。
                    _n_nodoc += 1
                    _nodoc_ids[_m.group(1)] += 1
                    continue
                _n_ptr += 1
                _ln = int(_m.group(2))
                _toks = _own[_idx]
                if not _toks or _ln > len(_tgt):
                    _n_na += 1
                    continue
                #  ±20 行: 引用句常写"这一段讲 X", 标识符未必正好落在被引的那一行。
                #  窗口敏感性实测 (07 篇 44 条, 当前版): ±3 得 2 · ±20 得 7 · ±80 得 28,
                #  而**整篇都找不到该标识符的只有 5 条** —— 所以它指的篇基本是对的,
                #  偏的是行。取 ±20 是因为再宽就等于只验"在不在这一篇", 判不出漂移。
                _win = " ".join(_tgt[max(0, _ln - 21):_ln + 20])
                if any(t in _win for t in _toks):
                    _n_ok += 1
                else:
                    _n_bad += 1
                    _bad_ptr.append((_base, _i, _m.group(0), sorted(_toks)[:2]))
    if _n_ptr or _n_unpinned:
        print(f"  跨文档行号指针 {_n_ptr} 条 (`NN:LINE`, 仅限已钉版本的篇): 指得到 {_n_ok}, "
              f"**指不到 {_n_bad}**, 判不了 {_n_na} (引用句里没写标识符)")
        if _n_unpinned:
            print(f"      另有 {_n_unpinned} 条在**未钉版本**的篇里 ({', '.join(sorted(_unpinned_docs))}): "
                  f"只数不判 —— 不说清指哪一版就没有可检验的对象 (C3)。")
        print("      ★报而不拦 (判法是启发式: 引用句没写反引号标识符就判不了)。"
              "行号漂移**永远在范围内**, 不会越界报错。")
        print("      处方二选一: 文档头写 `citecheck:docrev <rev>` 钉版本 (`07` 篇已钉), "
              "或新写的引用改用节号 (`§3.82` 这种, 插段不影响)。")
        if _n_nodoc:
            #  ★这一行与上面的"判不了"不同: 判不了的指针**进过检查**, 只是没抓手;
            #  这里的指针**连目标都没取到**。分开报, 否则 117 条读起来像全覆盖。
            print(f"      ★另 {_n_nodoc} 条指向**取不到的篇**, 未进上面任何一档: "
                  + ", ".join(f"{k} 篇 {v} 条" for k, v in sorted(_nodoc_ids.items())))
        if _rev_miss:
            #  ★2026-10-24 改: 原来这里无条件告警"指向它们的指针一条都没被检查"。
            #  实测本机这 4 篇 (08/09/10/11) **被指的次数是 0** —— 告警说的是一个
            #  不存在的缺口, 而它读起来像丢了覆盖。现在按**真被指的条数**分两种说法。
            _miss_ids = {os.path.basename(f)[:2] for _r, f in _rev_miss}
            _hit = sum(v for k, v in _nodoc_ids.items() if k in _miss_ids)
            _what = (f"**其中 {_hit} 条指针指向它们, 这 {_hit} 条没被检查**"
                     if _hit else "**本篇没有指向它们的指针, 覆盖上没有缺口**")
            print(f"      {len(_rev_miss)} 篇在被钉的那一版里取不到 (当时还不存在或改过名), "
                  + _what + ": "
                  + ", ".join(f"{r[:7]}:{os.path.basename(f)}" for r, f in sorted(_rev_miss)))
        for _b in _bad_ptr[:a.limit]:
            print(f"      {_b[0]:30s}:{_b[1]:<5d} {_b[2]:10s} 附近找不到 {_b[3]}")
        if len(_bad_ptr) > a.limit:
            print(f"      ... 另 {len(_bad_ptr) - a.limit} 条")
    try:
        _dn = doc_numbers()
        #  ★"数错了"与"判据抓不住"是两件事, 分开报 (2026-10-27, 09 第 195 条)。
        #  合成一个计数时, 一行"弱"会被读成"有一个数写错了"去改数 —— 而数是对的,
        #  该改的是那一行的写法。反过来也一样: 把弱算进不符里, 0 不符就永远达不到,
        #  于是整条判据会被当成噪声忽略。
        _weak = [r for r in _dn if "判据弱" in r[2]]
        _wrong = [r for r in _dn if "判据弱" not in r[2]]
        print(f"  带 num: 标记的数   与现算值不符的 **{len(_wrong)}** 行 "
              f"(标记法见 doc_numbers 的 docstring)")
        for _f, _i, _k, _v, _got, _t in _wrong[:5]:
            print(f"      ★{os.path.basename(_f)}:{_i}  num:{_k} 现算={_v} 文中={_got}  {_t[:60]}")
        print(f"  ★其中判据**抓不住**的   {len(_weak)} 行 "
              f"(现算值在, 但同行还有别的数 —— 改错了也不会红; 报而不拦)")
        for _f, _i, _k, _v, _got, _t in _weak[:5]:
            print(f"      ·{os.path.basename(_f)}:{_i}  num:{_k.split(' ')[0]} "
                  f"现算={_v} 同行候选={_got}  {_t[:50]}")
        if _weak:
            print("      补法: 让**被标记的数独占该行里唯一的数字** —— "
                  "历史值与日期挪到相邻行/句, 判据取的是整行任一个数等于现算值。")
    except (SystemExit, ImportError, KeyError) as _e:
        print(f"  带 num: 标记的数   跳过 ({_e})")
    print(f"  ★裸偏移引用   {len(bare_rows)} 条  (C2 违规: 没写是哪个文件、多大, 第三方复核不了)")
    #  ★**无条件打这一行**, 哪怕是 0: 只在非 0 时才出现的计数等于没有基线 ——
    #  读的人分不清"这次是 0"和"这版工具没查这件事"(第 140/159 条都是这么漏掉的)。
    print(f"  ★C11 裸名解到本仓 {len(barerepo_rows)} 条  (今天唯一所以能解, 交付一拷即歧义)")
    try:
        _fid_bad, _fid_def = fid_unresolved()
        print(f"  ★C12 发现编号 F<n> 解不出 {len(_fid_bad)} 个  "
              f"(本仓定义位 {len(_fid_def)} 个: F"
              + "/F".join(sorted(_fid_def, key=int)) + ")")
        for _n, _hits in sorted(_fid_bad.items(), key=lambda kv: int(kv[0])):
            print(f"      ★F{_n} 用了 {len(_hits)} 次但本仓没有定义位 —— "
                  f"外部编号就别只写编号, 写清事实或标明它解不出")
            for _p, _i, _t in _hits[:3]:
                print(f"         {os.path.basename(_p)}:{_i}  {_t}")
    except (OSError, re.error) as _e:
        print(f"  ★C12 发现编号 跳过 ({_e})")
    for base, off in bare_rows[:a.limit]:
        print(f"      {base:34s} @{off}")
    if len(bare_rows) > a.limit:
        print(f"      ... 另 {len(bare_rows) - a.limit} 条")
    if barerepo_rows:
        print(f"\n★C11 裸模块名解到本仓 {len(barerepo_rows)} 条 —— **今天唯一所以能解**, "
              f"下一次交付把 core/ 拷进 cases/ 就变歧义:")
        for base, cite, rp in barerepo_rows[:a.limit]:
            print(f"   {base:34s} {cite:40s} -> {rp}  (写成 {rp}:<行> 即可)")
        if len(barerepo_rows) > a.limit:
            print(f"   ... 另 {len(barerepo_rows) - a.limit} 条")
    if bad_rows:
        print(f"\n解不开的 {len(bad_rows)} 条:")
        for base, cite, why in bad_rows[:a.limit]:
            print(f"   {base:34s} {cite:58s} {why}")
        if len(bad_rows) > a.limit:
            print(f"   ... 另 {len(bad_rows) - a.limit} 条")
    return 1 if (bad_rows or bare_rows or bin_bad or ls_bad or barerepo_rows) else 0


#  ---- 文档里"现算得出"的数 (09 第 169 条) ----
#  ★**靠标记, 不靠猜**。两版推断式设计都否掉了:
#    v1 (第 141 条): 关键词共现 —— 92 条未解项出 12 对候选, 基本全噪声;
#    v2 (本条): 只扫表格行 + 关键词在首格 + 取第二格的数 —— 6 条命中**全是假阳性**
#       (`08`/`09` 里是**历史记录**里保留的旧值 87; `README` 两行是 59 的**子项** 20 与 39;
#        另一行的"全集"指的是**产物列全集 175**, 与二进制列名全集不是一个量)。
#  ⇒ 一行是不是"该量的权威行", **推不出来, 只能由写的人声明**。
#  这与 `citecheck:bin` / `citecheck:docrev` 是同一个办法 (第 139 条): 把推断换成声明。
#
#  用法: 在那一行行尾写 `<!-- num:<键> -->`, 键取 `coverage()` 的
#  `universe` / `appeared` / `never`。检查只看带标记的行。
#  ★自指控制: **反引号里的标记不算**。讲这条机制的那段文字本身会写
#  `<!-- num:never -->` 当示例 —— 第 140 条就是这么被 `citecheck:docrev` 咬过一次
#  (讲机制的文档把自己声明成了已钉版本)。这里一开始就加上, 不等它再咬一次。
_DOCNUM = re.compile(r"(?<!`)<!--\s*num:([a-z_]+)\s*-->")


#  ---- C12: 发现编号 `F<n>` 必须在本仓有定义位 (2026-10-29) ----
#  ★动因是我自己犯的: 同事口头提到一条编号 `F170`, 我把它抄进
#    `core/_known_failing.json` 的说明里。本仓的 F 编号空间实际只有 F1~F13,
#    于是那行字指向一个**不存在的对象** —— 而读到它的人无从知道该去哪里查。
#    与 C11 同一族 (引用必须解得出对象), 只是对象换成了本仓自有的编号空间。
#  ★扫描范围**包含 `core/_known_failing.json`**: 只扫 docs 的话这条判据抓不住
#    当初那个错 —— 登记表也是别人要读的共享文件。
#    判据能不能抓住它自己的动因, 是立判据时第一个该问的 (`09` 第 173 条同形)。
C12_FILES = ("docs/msprof-re/*.md", "docs/msprof-re/*.html", "core/_known_failing.json")
_C12_USE = re.compile(r'(?<![A-Za-z0-9_=/])F(\d{1,3})(?![0-9])')
#  定义位: 表格行首 `| F7 …` 或行首加粗 `**F10`。
_C12_DEF = re.compile(r'^\|?\s*\*{0,2}F(\d{1,3})\*{0,2}[′\s]', re.M)
#  代码片段与 dtype 记号不算引用 (`VMUL`=F32 / `/S32`), 所以 `=`、`/` 前缀在 _C12_USE
#  里就排除掉了, 反引号与 <code> 段在这里整段剔除。
_C12_CODE = re.compile(r'<code>.*?</code>|`[^`]*`', re.S)


def fid_unresolved(root: str = ".") -> tuple:
    """返回 (用过但没有定义位的编号 -> [(文件, 行号, 整行)], 有定义位的编号集合)。"""
    import glob as _g
    uses, defs = {}, set()
    for pat in C12_FILES:
        for p in sorted(_g.glob(os.path.join(root, pat))):
            try:
                body = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for m in _C12_DEF.finditer(body):
                defs.add(m.group(1))
            for i, line in enumerate(_C12_CODE.sub(" ", body).splitlines(), 1):
                for m in _C12_USE.finditer(line):
                    uses.setdefault(m.group(1), []).append((p, i, line.strip()[:90]))
    return ({n: v for n, v in uses.items() if n not in defs}, defs)


def doc_numbers(root: str = "docs/msprof-re") -> list:
    """带 `<!-- num:<键> -->` 标记的行里, 数字与**现算值**不符的。

    返回 `[(路径, 行号, 键, 现算, 文中, 原文)]`。**只认标记**, 不猜哪一行是权威行。
    """
    import importlib
    ap = importlib.import_module("core.aprof")
    cov = ap.coverage()
    #  ★可被标记的量 = `coverage()` 的键 + 下面几个。加新量的门槛只有一条:
    #  **算得快** (见上方注释; 一条没人跑的检查等于没有)。
    vals = dict(cov)
    vals["xlevel"] = cov["xlevel"]
    for k, v in cov.get("split", {}).items():
        vals["split_" + {"解出": "solved", "全零": "zero", "歧义": "ambig",
                         "未解": "open", "样本上没这列": "absent"}.get(k, k)] = v
    try:
        vals["symptoms"] = len(importlib.import_module("core.symptom").SYMPTOMS)
    except (ImportError, AttributeError):
        pass
    try:
        vals["formcodes"] = len(ap.form_codes())
    except (OSError, SystemExit):
        pass
    #  ★`openitems` 加进来的理由 (2026-10-27, 09 第 195 条): 这个数**已经过期过两次**
    #  (67 → 105 → 126), 而每次都是人手改的。`09` 第 168 条写明同一份索引"五个地方各自
    #  过期过", 并给了切分 —— 表格行的语境就是"当前状态", 必须跟着现算值走。
    #  门槛是"算得快": `open_items("docs")` 实测 0.08 秒, 比已登记的 `coverage()` (1.71 秒)
    #  快 20 倍, 远在门槛之内。
    try:
        vals["openitems"] = len(
            importlib.import_module("core.expdesign").open_items("docs"))
    except (ImportError, AttributeError, OSError):
        pass
    cov = vals
    out = []
    files = sorted(glob.glob(os.path.join(root, "*.md"))) + \
        sorted(glob.glob(os.path.join(root, "*.html")))
    for f in files:
        try:
            lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            continue
        for i, ln in enumerate(lines, 1):
            m = _DOCNUM.search(ln)
            if not m:
                continue
            key = m.group(1)
            if key not in cov:
                out.append((f, i, key, None, None, ln.strip()[:100]))
                continue
            #  ★判据取"整行里任一个数等于现算值"就算过, 所以**候选集越大它越抓不住**。
            #  2026-10-27 负例检验实撞: 把一行的 `126` 改成 `999` 而 `citecheck` 不红 ——
            #  因为同一行的括注里还写着一次 126, 而且日期 `2026-10-27` 又贡献了
            #  `2026`/`10`/`27` 三个候选。日期是**每一条订正注记都会带**的东西,
            #  于是这条判据在带日期的行上近乎不可证伪。
            #  处置两条: (a) 先剥掉 ISO 日期再取数; (b) 剥完仍有多个候选时**显式报弱**,
            #  因为"过"在那种行上不代表数对。
            body = _DOCNUM.sub("", ln)
            body = re.sub(r"\d{4}-\d{2}-\d{2}", " ", body)
            #  ★直接找**现算值本身**, 不再"先收集 2~5 位候选再看包不包含"。
            #  旧写法有两个洞, 2026-10-27 各撞一次:
            #    · 宽度下限 2 位 ⇒ 个位数的量 (如 `未解 7`) **永远匹配不到**, 报成假红。
            #      这多半就是那张覆盖表里四行一直没挂标记的原因 —— 挂了就红, 于是不挂。
            #    · 只看"包含", 同行数字越多越抓不住 (见下面的判据弱)。
            nums = [int(x) for x in re.findall(r"\b\d{1,6}\b", body)]
            if not re.search(rf"\b{cov[key]}\b", body):
                out.append((f, i, key, cov[key], nums[:3], ln.strip()[:100]))
            elif len(set(nums)) > 1:
                #  阈值是 >1 不是 >2: 现算值本身就占一个候选, 所以
                #  "值 + 任意一个别的数" 就已经让判据抓不住了 (改错了仍有那个值兜着)。
                #  第一版写 >2, 自检当场照出来 —— 两个数的那行它不报。
                #  现算值在里面, 但同行还有别的数 —— 记成"弱", 不记成"不符"。
                out.append((f, i, key + " (判据弱: 同行多数)", cov[key],
                            sorted(set(nums))[:5], ln.strip()[:100]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="判据信任等级与预测-实测对账台账")
    ap.add_argument("--file", help="台账路径 (默认 core/_trust_ledger.jsonl)")
    ap.add_argument("--refs", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("show", help="列所有判据的信任等级")
    s.add_argument("-q", help="关键词过滤"); s.add_argument("-v", "--verbose", action="store_true")
    s.add_argument("--max-level", type=int, help="只看不高于这个等级的")

    s = sub.add_parser("predict", help="记一条预算 (模型算出来的数)")
    s.add_argument("--criterion", required=True); s.add_argument("--claim", required=True)
    s.add_argument("--value", type=float, required=True); s.add_argument("--unit", default="")
    s.add_argument("--bound", required=True, choices=["upper", "lower", "point"],
                   help="upper=上界(**不得用于否决候选**) / lower=下界 / point=点估")
    s.add_argument("--case", default=""); s.add_argument("--id")

    s = sub.add_parser("resolve", help="用实测给某条预算对账")
    s.add_argument("--id", required=True); s.add_argument("--actual", type=float, required=True)
    s.add_argument("--note", default="")
    s.add_argument("--redo", action="store_true", help="更正已对过账的那一条 (默认拒绝重复对账)")

    s = sub.add_parser("vote", help="投一票 (只有端到端确认计入升级)")
    s.add_argument("--criterion", required=True)
    s.add_argument("--outcome", required=True, choices=["confirmed", "contradicted"])
    s.add_argument("--kind", default="end_to_end", choices=["end_to_end", "inspection", "reasoning"])
    s.add_argument("--evidence", default=""); s.add_argument("--case", default="")
    s.add_argument("--retract", action="store_true",
                   help="撤回同类型的票 (当时的证据不完整)。必须给 --evidence 说明漏了什么")

    s = sub.add_parser("flag", help="显式标一个升/降级项")
    s.add_argument("--criterion", required=True); s.add_argument("--flag", required=True)
    s.add_argument("--why", default=""); s.add_argument("--evidence", default=""); s.add_argument("--case", default="")
    s.add_argument("--clear", action="store_true",
                   help="撤销这个标记 (它的成因已被修好)。必须给 --evidence 指到修它的那次改动")

    s = sub.add_parser("gate", help="这条判据有没有资格关闭一条方向")
    s.add_argument("--criterion", required=True)
    s.add_argument("--action", default="refute", choices=["refute", "cite"])

    s = sub.add_parser("reservecheck", help="动用/删除带预留标记的行时必须留痕 (提交闸)")
    s.add_argument("--staged", action="store_true", default=True)
    s = sub.add_parser("reserveack", help="登记一次「确认动用这处预留」")
    s.add_argument("fragment")
    s.add_argument("--reason", required=True)
    s.add_argument("--owner", required=True)
    sub.add_parser("cases", help="对账落空的条目 = 待修的模型 test_case")

    sub.add_parser("adoption", help="哪些判据模块还没在结论处自报资格")

    s = sub.add_parser("density", help="产物密度: 关闭类 vs 实现类产出")
    s.add_argument("states", help="<case>/states.jsonl")

    s = sub.add_parser("audit", help="出一份盲审包 (只给源码, 不给作者的推理与 case 上下文)")
    s.add_argument("--module", required=True); s.add_argument("--out")

    s = sub.add_parser("citecheck", help="引用可达性: 文档里的 file:line / 二进制偏移现在还指得到吗")
    s.add_argument("docs", nargs="*", help="markdown 文件或目录 (给 --preset 时可省)")
    s.add_argument("--preset", choices=["msprof-re"],
                   help="自动探测 root 集并默认扫 docs/msprof-re/ —— 省掉手拼七个 --root")
    s.add_argument("--root", action="append", default=[],
                   help="源码树根, 可给多次。可写 `/path`, 也可写 `PREFIX=/path` 让文档里的"
                        "缩写前缀 ($AN/ / <analysis>/ / INC/) 对上自己的树 "
                        "(C3: 版本必须由调用方钉死, 本工具不替你搜)")
    s.add_argument("--placeholder", default="<analysis>", help="文档里代表 root 的占位符")
    s.add_argument("--limit", type=int, default=20, help="每类最多列几条")
    s.add_argument("--git-repo", help="把文中的 7-40 位十六进制串拿到这个仓里解 commit")

    s = sub.add_parser("deadcatch",
                       help="静默吞异常: try 里引用未定义名字而 except 宽且不作声 "
                            "(把「没写对」吞成「没命中」, 09 第 87 条实撞)")
    s.add_argument("paths", nargs="*", help="py 文件或目录 (默认 core/ 与 docs/msprof-re/tools/)")

    s = sub.add_parser("silentskip",
                       help="静默跳过: 统计型循环里不留痕的 continue "
                            "(这一条不进任何分母, 覆盖率反而变好看; 09 第 176 条实撞 envcheck E8)")
    s.add_argument("paths", nargs="*", help="py 文件或目录 (默认 core/)")

    s = sub.add_parser("constcheck",
                       help="常数来源: 模块级数值常数说不说得出从哪来 (03 §9.31 的帧长就是没来源留下的)")
    s.add_argument("paths", nargs="*",
                   help="py 文件或目录 (默认 core/ 与 docs/msprof-re/tools/)")
    s.add_argument("--limit", type=int, default=40)
    s.add_argument("--staged", action="store_true",
                   help="棘轮模式 (提交闸用): 只拦本次暂存让缺来源数变多的文件")

    s = sub.add_parser("docindex",
                       help="长文档缺前置索引: 长 + 被订正过多次 + 没索引 = 从中间读会拿到过期答案")
    s.add_argument("root", nargs="?", default="docs", help="扫哪个目录 (默认 docs/)")
    s.add_argument("--min-lines", type=int, default=1200)
    s.add_argument("--min-corrections", type=int, default=3)

    s = sub.add_parser("banner", help="出一行可粘贴的横幅 (标明上界/下界与是否可用于否决)")
    s.add_argument("--criterion", required=True)
    s.add_argument("--bound", default="upper", choices=["upper", "lower", "point"])

    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.refs:
        from core import _refs
        return _refs.print_refs("trust", REFS) or 0
    fn = {"show": cmd_show, "predict": cmd_predict, "resolve": cmd_resolve, "vote": cmd_vote,
          "flag": cmd_flag, "gate": cmd_gate, "cases": cmd_cases, "adoption": cmd_adoption, "density": cmd_density,
          "banner": cmd_banner, "audit": cmd_audit, "citecheck": cmd_citecheck, "constcheck": cmd_constcheck, "deadcatch": cmd_deadcatch, "silentskip": cmd_silentskip,
          "reservecheck": lambda a: reservecheck(True),
          "reserveack": lambda a: reserve_ack(a.fragment, a.reason, a.owner),
          "docindex": lambda a: print_docs_missing_index(a.root, a.min_lines, a.min_corrections),
          }.get(a.cmd)
    if fn is None:
        ap.print_help()
        return 0
    return fn(a)


if __name__ == "__main__":
    sys.exit(main())
