"""判据出处表 —— 让每条判据可被证伪, 而不只是被断言。

**为什么需要这个 (2026-08-27 立)**: 本仓 20 个模块里只有 pipeplan 与 detreduce 有自校准入口,
其余 18 个的判据无法被证伪。而 dual-loop 的核心断言是"方法必须押注" ——
一条不能被产物证伪的判据不是方法, 是断言。

三次真实代价:
  1. gmlayout 的 A6 (原子写起始地址要 512B 对齐) 被我当"已测事实"写进判据并给了改法,
     f640 一测就翻 (补齐后带宽 30.22 -> 29.55, 更差)。**据那条错结论白做了一次构建。**
  2. pipeplan 的 mte3_bw() 是 Little 定律的单流形式, 在**交织写**上被证伪
     (界写 256B 砍到 32B 应慢一个数量级, 实测纹丝不动)。
  3. perf 的冲突比折算, 分母错当成向量周期 (实为 task_cyc), 低估 3 倍。

三次都是**判据本身过期或错误**, 而工具没有任何机制告诉使用者"这条判据的实证基础是什么、
还成不成立"。本模块给一个统一的最小 schema, 并让 opdbg 能横向聚合出"哪些判据该重验了"。

## schema

    REFS = {
        "判据 id": Ref(what="它判什么",
                       status="ini_exact|measured|derived|disputed|refuted",
                       evidence="实证点: 哪次构建/哪个 runs 目录/哪条真机日志",
                       verified="YYYY-MM-DD"),
    }

status 的含义 (**refuted 必须留在表里**, 删掉等于让后人重新推一遍):
  ini_exact = 与平台 ini 逐位相同, 可回源硬核对
  measured  = 本仓真机标定, ini 无声明; evidence 必须能指到具体 runs 目录
  derived   = 由上面两类推导, 推导本身要写在 evidence 里
  disputed  = 两个来源不一致且现有数据分不出; **只用于排序, 不作承诺**
  refuted   = 曾经相信、已被实测推翻; 保留原文 + 推翻它的实证点
"""
from __future__ import annotations
import datetime
import os as _os
import subprocess as _sp
import sys as _sys
import re as _re

STATUS_ORDER = ["refuted", "disputed", "measured", "derived", "ini_exact"]
STALE_DAYS = 120          # 超过这个天数的 measured/derived 提示重验 (硬件/CANN 版本会变)


#  ---- 判据的适用域 (2026-10-31 立) ------------------------------------------------
#  来历: 合入 SMLA 分支时一次进来两类"长得像本机判据"的条目 —— DSL 栈 (cannbotdsl) 的,
#  与另一台机器 (Ascend950PR) 的。两者用错的**代价方向不同**:
#    · 栈用错 = 噪音。在 AscendC 的 kernel 上读到 DSL 的判词, 浪费的是注意力。
#    · 芯片用错 = 错结论。950PR 的 cube 峰值 373.5 TFLOPS 拿到 910B2 上算地板会直接偏,
#      而本仓已经因为"跨芯片借用"栽过 (mmplan 的 cv_direct_path 沿用 910B 的 False,
#      把 950PR 上确实存在的直通整条藏掉)。
#  ★所以适用域**写在条目上, 不靠使用处记忆**, 并守两条:
#    1. 带域的条目**永远**打可见标签 —— 过滤是便利, 标签才是保证。
#    2. 隐藏必须留痕: 隐了几条、叫什么、怎么看, 都在表尾报出来。静默隐藏是另一种掩盖。
#  不带 scope 的条目一律 any, 所以既有的几百条一个字都不用改。
_SOC_CACHE = []           # 懒探测一次; [] = 还没探, [None] = 探不到


def local_soc():
    """本机 SoC (小写), 探不到返回 None。**不回落到任何具体型号** —— 与 socspec 同口径。"""
    if not _SOC_CACHE:
        try:
            from core.socspec import detect_soc
            _SOC_CACHE.append(detect_soc()[0].lower())
        except Exception:      # noqa: BLE001 —— 没有 CANN 的机器上 --refs 仍要能看
            _SOC_CACHE.append(None)
    return _SOC_CACHE[0]


def local_stack() -> str:
    """本机默认语言栈。要改走环境变量 OPDBG_STACK, 不在模块里写死某一条线。"""
    return (_os.environ.get("OPDBG_STACK") or "ascendc").lower()


def _norm_scope(scope) -> dict:
    """scope 收 {'stack':…, 'soc':…}, 也收单字符串简写 ('dsl' / 'ascend950pr')。"""
    if not scope:
        return {"stack": "any", "soc": "any"}
    if isinstance(scope, str):
        scope = {"soc": scope} if scope.lower().startswith("ascend") else {"stack": scope}
    return {"stack": (scope.get("stack") or "any").lower(),
            "soc": (scope.get("soc") or "any").lower()}


def _soc_match(want: str, have) -> bool:
    """型号比对按**系列前缀双向**匹配: 条目写 ascend950pr 而本机 ascend950pr_957b, 算中。"""
    if want == "any" or not have:
        return True
    return want.startswith(have) or have.startswith(want)


def scope_tag(ref) -> str:
    """条目的可见标签。无域返回空串 —— 本机本栈的条目不该被标签淹没。"""
    s = getattr(ref, "scope", None) or {"stack": "any", "soc": "any"}
    bits = [s["stack"].upper()] if s["stack"] != "any" else []
    if s["soc"] != "any":
        #  ★型号原样大写, 不做好看的大小写修饰 —— 标签是给人核对用的, 与 chip_calib
        #    的键同形才核得动。
        bits.append(s["soc"].upper())
    return "[" + "\u00b7".join(bits) + "] " if bits else ""


def in_scope(ref, stack=None) -> bool:
    """这条判据适不适用于当前 (栈, 芯片)。

    ★探不到 SoC 时**不按芯片隐藏** —— 隐藏要以"确知不适用"为前提, 探测失败不是那个前提。
      这种时候标签仍然照打, 读者自己判。
    """
    s = getattr(ref, "scope", None) or {"stack": "any", "soc": "any"}
    st = (stack or local_stack()).lower()
    if st == "all":
        #  ★唯一的逃生口, 两个维度一起开。留**一个**而不是每维一个: 逃生口多了就没人
        #    记得哪个开了哪个没开, 而"我以为我看到了全部"比"我知道我没看全"更危险。
        return True
    if s["stack"] != "any" and s["stack"] != st:
        return False
    return _soc_match(s["soc"], local_soc())


class Ref:
    def __init__(self, what, status: str, evidence, verified: str, scope=None):
        assert status in STATUS_ORDER, status
        #  ★what / evidence 可以给**字符串列表**, 按行拼接, 作者不必手写 chr(92)+"n"。
        #  2026-09-17 立: 同一天里三条 REFS 条目都因为手写换行转义写坏了字符串字面量
        #  (写成了 `\"` 即"把引号转义掉"而不是 `\n"` 即"换行后收引号"), 整个模块 SyntaxError。
        #  pycheck 每次都当场抓住, 代价不高, 但同类失败三次就该结构性预防而不是"下次小心"。
        #  给列表时每个元素是一行, 不需要任何转义。
        self.what = "\n".join(what) if isinstance(what, (list, tuple)) else what
        self.evidence = "\n".join(evidence) if isinstance(evidence, (list, tuple)) else evidence
        self.status, self.verified = status, verified
        #  适用域 —— 见上面「判据的适用域」。不给就是 any, 既有条目不必改。
        self.scope = _norm_scope(scope)

    def age_days(self, today: str | None = None) -> int:
        try:
            d0 = datetime.date.fromisoformat(self.verified)
            d1 = datetime.date.fromisoformat(today) if today else datetime.date.today()
            return (d1 - d0).days
        except Exception:
            return -1


def print_refs(mod: str, refs: dict, today: str | None = None,
               stack: str | None = None, show_all: bool = False) -> int:
    """打印一个模块的判据表。返回需要注意的条数 (refuted + disputed + 过期)。

    不适用于当前 (栈, 芯片) 的条目默认不列, 但**条数与名字都在表尾报出来** ——
    见上面「判据的适用域」的第 2 条: 静默隐藏是另一种掩盖。`show_all=True` 全列。
    """
    order = sorted(refs, key=lambda x: (STATUS_ORDER.index(refs[x].status), x))
    out = order if show_all else [k for k in order if in_scope(refs[k], stack)]
    hidden = [k for k in order if k not in set(out)]
    print(f"=== {mod} 判据出处 ({len(out)} 条) ===")
    hot = 0
    for k in out:
        r = refs[k]
        age = r.age_days(today)
        mark = ""
        if r.status == "refuted":
            mark = "  ★已被推翻 —— 保留在表里是为了不让后人重新推一遍"
            hot += 1
        elif r.status == "disputed":
            mark = "  ★两来源不一致, 只用于排序不作承诺"
            hot += 1
        elif r.status in ("measured", "derived") and age > STALE_DAYS:
            mark = f"  ★距上次验证 {age} 天, 建议重验 (硬件/CANN 版本会变)"
            hot += 1
        print(f"  [{r.status:9s}] {scope_tag(r)}{k:22s} {r.what}")
        print(f"      实证: {r.evidence}")
        print(f"      验证于 {r.verified}" + (f" ({age} 天前)" if age >= 0 else "") + mark)
    if hot:
        print(f"\n★本表有 {hot} 条需要注意 (已推翻 / 有争议 / 过期)。"
              f"用它们做决定前先看 evidence 那一行指向的实证点。")
    if hidden:
        print(f"\n☆另有 {len(hidden)} 条不属于当前适用域 (栈 {stack or local_stack()} / "
              f"芯片 {local_soc() or '探测不到'}) 未列出: {', '.join(hidden)}\n"
              f"  要看全部: OPDBG_STACK=all 再跑一遍这条命令。")
    return hot


#  ---- 判据的元审计: 判据本身够不够严谨 -------------------------------------------------
#  print_refs 只查"过期/推翻/争议"这三种**状态**问题, 查不出判据本身武断。
#  2026-08-28 一轮里连撞三种形态, 每种都让人白做了工:
#    (a) 未实测却给指令 —— 只是推导, 输出时却写成"应该改成 X";
#    (b) 给了改法却不给前置条件 —— H10 的"软件流水错开一拍"在标量已饱和时净亏, 实撞 20.12->22.40;
#    (c) 给了某一维的上界, 使用者当成净收益 —— W7 的 wave_eff 提到 1.000 反而慢 73%。
#  下面三条规则就是把这三种形态变成可执行的检查。规则本身也可能过宽或过窄,
#  所以命中一律是**提示**, 由人判断, 不阻断任何流程。
#  ★指示性措辞要分两类, 混在一起会误报 —— 本审计的第一版就栽在这里:
#    ADVICE  = 对使用者的**优化建议** ("改成 X" / "建议用 Y")。这类必须给前置条件或反例,
#              因为一条"永远该这么改"的建议是可疑的 (H10 与 W7 都被自己的产物打翻过)。
#    CONTRACT= 工具**自身的行为契约** ("解析不出必须报 UNRESOLVED 并拒绝判过")。
#              这类要的是实证而不是反例 —— 让它"有条件地不报"恰恰是制造假安全。
_ADVICE = ("应该", "应当", "改成", "该用", "建议", "要把", "得把", "改法")
_CONTRACT = ("必须", "不要", "不得", "禁止")
_IMPERATIVE = _ADVICE + _CONTRACT
#  hedge 词表要认变体。第一版只写了"未实测", 于是把 evidence 里明明白白写着
#  "未**单独**实测, 但同轮 f640 证明方向一致"的那条判成了"未实测却给指令"。
_HEDGE = ("未实测", "未单独实测", "尚未实测", "没有实测", "推导", "推出", "只用于排序",
          "不作承诺", "上界", "不预测", "前置条件", "除非", "反例", "净收益一律实测",
          "适用边界", "方向一致", "本模型算不出")
#  锚点不必是数字。一条 evidence 只要能让人**回去核对**就够了 ——
#  引用的原始报错串 ('missing but already registered worktree') 与数字一样硬。
#  第一版只认数字与路径, 于是把两条 evidence 里明明引了报错原文的判据判成"无锚点"。
_ANCHOR_HINT = ("cases/", "runs/", "us", "%", "MB", "KB", "GB", "cycle", "实测", "真机",
                "Error", "error", "报 '", "报错", "实撞", "复现", "日志", "'", "\"",
                "逐位", "挂死", "崩")


def audit_refs(mod: str, refs: dict) -> list:
    """按三条元规则审判据本身。返回 [(id, 规则, 说明)]; 只提示, 不阻断。"""
    out = []
    #  ★REFS 在仓里有**两种形态**: dict{id: Ref} 与 list[Ref] (netgate / shguard 走后者)。
    #    原来这里只认 dict, 于是 `sorted(list[dict])` 直接抛 TypeError, 整个
    #    `--audit-criteria` 在字母序走到 netgate 时崩掉 —— 它后面的模块一个都没审,
    #    而**崩掉的入口不会打印"我没审"**。core/opdbg 的 capability_index 两种形态都收,
    #    同一份数据的两个读取点口径不一致 (2026-10-26 查 silentskip 时撞出, 09 第 177 条)。
    pairs = (sorted(refs.items()) if hasattr(refs, "items") else
             [(getattr(r, "id", "") or str(getattr(r, "what", ""))[:40], r) for r in refs])
    for k, r in pairs:
        #  ★schema 不统一本身就是一条要报的问题: 值不是 Ref 就没法做任何元审计,
        #    而"审计跳过了它"和"它通过了审计"在输出上必须区分得开。
        if not all(hasattr(r, a) for a in ("what", "status", "evidence")):
            #  ★pipeplan 用的是 (ini_键名, status, evidence) 三元组 —— 那是**参数出处表**
            #    (每条对应一个数值从哪来), 与本模块管的**判据出处表** (每条对应一条判断规则)
            #    是两回事。参数表不给改法, 三条元规则一条都不适用。
            #    本审计的第一版把它们混为一谈, 一口气报了 17 条"未用统一 schema" —— 又一次武断。
            if isinstance(r, tuple) and len(r) == 3:
                out.append((k, "参数出处表(本审计不覆盖)",
                            "这是数值参数的来源记录, 不是判断规则; 三条元规则针对的是"
                            "'判据的措辞有没有超出实证', 对参数表不适用。"))
            else:
                out.append((k, "未用统一 schema",
                            f"该条的值是 {type(r).__name__}, 既不是 Ref 也不是参数三元组, "
                            f"元审计无法进行。"))
            continue
        txt = f"{r.what} {r.evidence}"
        imperative = any(w in r.what for w in _IMPERATIVE)
        hedged = any(w in txt for w in _HEDGE)
        #  规则一: 未实测却给指令
        if r.status in ("derived", "assumed") and imperative and not hedged:
            out.append((k, "未实测却给指令",
                        f"status={r.status} 却在 what 里给出指示性措辞, 且 evidence 未注明这是推导。"
                        f"读的人会当成已证实的做法。改法: 要么补实证升 measured, 要么在 what 里写清"
                        f"'这是推导, 未实测'及适用边界。"))
        #  规则二: 自称实测, 但关键词匹配没认出可核对的锚点。
        #  ★这条规则**误报率高, 只作提示**。它被自己的产物连续收窄了四轮:
        #    第一版只认数字与路径 -> 把引了原始报错串的两条判成无锚点;
        #    放宽到报错串之后 -> 又把"128x256 通过 / 256x256 被 GetTiling 拒绝"这种
        #    通过/拒绝对照、以及主动写明"目录存在性证据, 不等于该算子已经在用"的判据判成无锚点。
        #  继续加词的终点是"什么都算锚点", 规则本身就失效了。所以定位改成
        #  **提请人工确认**而不是断言不合格 —— 一条会误报的检查, 措辞必须配得上它的准确率。
        if r.status == "measured" and not any(a in r.evidence for a in _ANCHOR_HINT):
            out.append((k, "锚点待确认",
                        f"status=measured, 但按关键词没认出 evidence 里的可核对锚点。"
                        f"**这条规则误报率高** —— 请人工看一眼: evidence 能不能让人回去核对? "
                        f"能 (有数字/路径/报错原文/通过与拒绝的对照/明确写出证据类型与局限) 就忽略本条; "
                        f"不能则补锚点或把 status 降回 derived。"))
        #  规则三: 给了**优化建议**却不给前置条件或反例。只对 ADVICE 触发, 不对工具契约触发。
        if any(w in r.what for w in _ADVICE) and not any(
                w in txt for w in ("前置条件", "除非", "反例", "不适用", "上界",
                                   "净收益一律实测", "适用边界", "方向一致")):
            out.append((k, "改法无边界",
                        f"给了改法却没写它在什么条件下不成立。一条永远成立的改法是可疑的 —— "
                        f"本仓已有两条改法被自己的产物打翻 (H10 在标量饱和时净亏, "
                        f"W7 按均衡选 tile 慢 73%)。补一个已知反例或前置条件。"))
    return out


def print_audit(mod: str, refs: dict) -> int:
    """打印一个模块的措辞审计。返回**需要收紧的条数** —— 不覆盖的那类不计入。"""
    hits = audit_refs(mod, refs)
    skipped = [h for h in hits if h[1].startswith("参数出处表")]
    real = [h for h in hits if not h[1].startswith("参数出处表")]
    if skipped and not real:
        print(f"=== {mod}: {len(skipped)} 条是参数出处表, 本审计不覆盖 (三条元规则针对判断规则) ===")
        return 0
    if not real:
        return 0
    print(f"=== {mod}: {len(real)} 条判据需要收紧措辞"
          + (f" (另有 {len(skipped)} 条是参数表, 不覆盖)" if skipped else "") + " ===")
    for k, rule, why in real:
        print(f"  [{rule}] {k}")
        print(f"      {why}")
    return len(real)


#  ---- 判据编号唯一性 (跨模块的公共守卫) ---------------------------------------
#  2026-10-27 同一天撞两次: hangdiag 加新判据时先占了 C12 (已属 branch_guard)、
#  再占了 C13 (已属 dead_workspace); shipcheck 加新判据时占了 S9 (已属产物一致性)。
#  两次都是靠自检打印出同名两行才看出来 —— **肉眼发现不是机制**。
#  第一次只在 hangdiag 里加了守卫, 于是第二次在 shipcheck 上照撞 ——
#  单模块的守卫防不住下一个模块, 故提到公共层, 由 opdbg 的判据自检统一扫。
#  ───────── git 钩子环境的唯一剥除清单 (2026-10-31) ─────────
#
#  git 执行钩子时导出 `GIT_DIR` 等变量, 而 **`GIT_DIR` 的优先级高于 `-C <路径>`**。
#  子进程继承到它, 那些"建隔离夹具仓"的写法就全部失去隔离: `git -C <夹具目录> init`
#  重新初始化的是**调用者那个仓**, 随后的 add/commit/config 全落在它上面。
#  ★**主检出里不一定复现**: 那里 git 导出的是相对路径 `.git`, 按夹具目录解析反而正常;
#    **链接工作树**里它是绝对路径才会打穿。而链接工作树与主检出**共享 .git/config 与 refs**,
#    所以受害的是主检出 —— 这是 2026-10-31 我在主检出里反复复现不出、而仓却一直被改坏的原因。
#  本清单取两个独立发现的**并集**: dispatcher 侧 (core/opdbg 的 selftest_env, 覆盖自检子进程)
#  与模块侧 (goodstate.git_env, 覆盖生产路径的 git 调用) 必须用同一份 ——
#  两份清单会漂, 而漂掉的那一项恰恰是下次打穿的入口。
GIT_HOOK_ENV = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_PREFIX",
    "GIT_QUARANTINE_PATH",            # 来自 dispatcher 侧那份 (提交隔离区)
    "GIT_NAMESPACE", "GIT_CONFIG", "GIT_CONFIG_GLOBAL",   # 来自模块侧那份
    #  ★身份也要剥 (2026-10-31 补): `git commit` 会把 GIT_AUTHOR_* 导给钩子, 而
    #    **它们顶得过夹具自己的 `-c user.name=t`** —— 夹具仓的提交作者于是变成调用者的身份。
    #    后果不是"提交署错名"这么轻: 任何**对夹具作者做断言**的用例都变成不确定的,
    #    它在终端里绿、在钩子里红, 而两边跑的是同一份代码。
    #    实撞: goodstate 的「复原命令取 HEAD 的真作者」连拦四次提交, 单跑五遍却 5/5 过。
    #    本仓的生产路径都不自动 commit (G6 明确不替人提交), 所以剥掉它们没有副作用。
    "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_AUTHOR_DATE",
    "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL", "GIT_COMMITTER_DATE",
)


def clean_git_env() -> dict:
    """当前环境减去 git 钩子导出的那些 —— 起 git 子进程一律用它。"""
    return {k: v for k, v in _os.environ.items() if k not in GIT_HOOK_ENV}


def _mutation_exempt(path: str) -> dict:
    """模块声明的变异豁免: `_MUTATION_EXEMPT = {"函数名": "理由"}`。理由是必填的。"""
    import ast
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except Exception:
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "_MUTATION_EXEMPT":
                    try:
                        v = ast.literal_eval(node.value)
                    except Exception:
                        return {}
                    return {k: s for k, s in v.items() if isinstance(s, str) and s.strip()}
    return {}


def _selftest_argv(root: str, module: str):
    """从 opdbg 的 CRITERIA_EXTRA 表里取该模块自检的真实调用参数。

    ★不另立一套映射: 那张表是唯一的真源, 复制一份就会与它漂移 ——
    而漂移的症状是"基线不绿", 看着像模块坏了。
    """
    import ast
    try:
        src = open(_os.path.join(root, "core", "opdbg"), encoding="utf-8").read()
    except OSError:
        return None
    m = _re.search(r"^CRITERIA_EXTRA\s*=\s*(\[)", src, _re.M)
    if not m:
        return None
    depth, i = 0, m.start(1)
    while i < len(src):
        if src[i] == "[":
            depth += 1
        elif src[i] == "]":
            depth -= 1
            if depth == 0:
                break
        i += 1
    try:
        table = ast.literal_eval(src[m.start(1):i + 1])
    except Exception:
        return None
    for row in table:
        if isinstance(row, (tuple, list)) and row and row[0] == module:
            #  ★表里有的行第二项是 None (如 ablate), 直接 list() 会 TypeError ——
            #    我自己的代码撞的, 不是模块的问题。缺省回落到 --selftest。
            return list(row[1]) if len(row) > 1 and row[1] else ["--selftest"]
    return None


def _criterion_owners(path: str) -> set:
    """源文件里 `check="XX_..."` 字面量各自所属的 def 名 —— 判据函数名的真来源。"""
    import re as _r
    try:
        src = open(path, encoding="utf-8").read()
    except OSError:
        return set()
    spans = [(m.group(1), m.start()) for m in _r.finditer(r"^def (\w+)\(", src, _r.M)]
    def owner(pos):
        name = None
        for fn, st in spans:
            if st <= pos:
                name = fn
            else:
                break
        return name
    out = set()
    #  ★判据标签的写法**不统一**, 认得太少就会静默把整个模块报成"本工具不适用":
    #    check="XX1_y"      关键字实参 (hangdiag / detsync_lint / shipcheck / oplog)
    #    "check": "XX1_y"   字典字面量 (ub_budget / mmplan)
    #    crit = "XX1"       局部变量再拼 (alignlint)
    #    首版只认第一种, 于是 ub_budget / mmplan / alignlint 全被判成不适用。
    pats = (r'check\s*=\s*"[A-Za-z]+\d*_[a-z0-9_]+"',
            r'"check"\s*:\s*"[A-Za-z]+\d*_[a-z0-9_]+"',
            r'\bcrit\s*=\s*"[A-Za-z]+\d+"')
    for pat in pats:
        for m in _r.finditer(pat, src):
            o = owner(m.start())
            if o:
                out.add(o)
    return out


#  ───────── 无标签模块的变异算子 (2026-10-30 立) ─────────
#
#  `_criterion_owners` 只能靠 `check="XX_..."` 这类标签挑判据, 于是**没有标签的模块
#  整个落在视野外** —— 全仓 33 个模块报"不适用"。实测过它们不是标签写错: 逐模块量下来
#  只有 4 个 (abgate/aprof/dslir/socspec) 的判据以返回字典的键存在, **其余 29 个一个
#  标签都没有**, 而它们确有完整的多例自检 (hwfloor 20 条 / abgate 34 条 / tkcheck 28 条)。
#  判据就是 analyze / verdict_lines / o1_single_source 这类普通函数名。
#
#  所以换一条路挑判据: **让自检自己说它调用了谁** (sys.settrace), 零标签依赖。
#  再按观测到的**返回值形状**造"一切正常"的中性值去替换。
#
#  ★硬守卫 (建这个算子时当场撞的): 中性值**必须与观测值不同**, 否则是空变异。
#    首版 neutral() 对 str/int/None 原样返回, 四条判据因此被误报成"摆设" ——
#    量具带着它自己要抓的那个缺陷。现在空变异单独报一类, 不与"摆设"混。
_TRACE_DRIVER = r'''
import importlib, sys, os, json, copy, types
mod, mode, ARGV = sys.argv[1], sys.argv[2], sys.argv[3:]
M = importlib.import_module("core." + mod)

#  ★自检入口**不统一**, 不能硬调 M.selftest(): `trust` 的是 `_selftest()` (下划线,
#    由 main 分发), `perf` 根本没有裸的那个 (只有 selftest_cflt_wall 等一组)。
#    首版硬调, 于是这两个最大的模块一律报"追踪不到被调用的函数" —— 而那是**静默空手**:
#    看起来像跑过了。回落到按 opdbg 的 CRITERIA_EXTRA 表给的 argv 调 main()。
def run_selftest():
    fn = getattr(M, "selftest", None)
    if callable(fn):
        return fn()
    sys.argv = ["core." + mod] + ARGV
    try:
        return M.main()
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0

if mode == "--trace":
    names = {n for n, f in vars(M).items()
             if isinstance(f, types.FunctionType) and f.__module__ == M.__name__
             and n not in ("selftest", "main") and not n.startswith("_")}
    called = set()
    def tracer(frame, event, arg):
        if event == "call" and frame.f_code.co_filename == M.__file__:
            called.add(frame.f_code.co_name)
        return None
    sys.settrace(tracer)
    try:
        run_selftest()
    finally:
        sys.settrace(None)
    print(json.dumps(sorted(called & names)))
    sys.exit(0)

orig = getattr(M, mode, None)
if orig is None:
    sys.exit(97)
seen = []
def capture(*a, **k):
    r = orig(*a, **k)
    seen.append(r)
    return r
setattr(M, mode, capture)
try:
    run_selftest()
except BaseException:
    pass
setattr(M, mode, orig)
if not seen:
    sys.exit(2)                      # 自检没调用到它

def neutral(v):
    """造"一切正常 / 无发现"的同形状值。不保证与原值不同 —— 由下面的守卫兜。"""
    if isinstance(v, bool):  return True          # 判决位: 恒判通过
    if isinstance(v, list):  return []
    if isinstance(v, tuple): return tuple(neutral(x) for x in v)
    if isinstance(v, dict):  return {k: neutral(x) for k, x in v.items()}
    #  ★集合要单列 —— 漏了它就走到最后一行原样返回 = 空变异假象。
    #    2026-10-30 实撞: devguard.self_chain (PID 集合) 与 schedtab.gm_tensors
    #    ({'vg','out'}) 都因此被报成"检出路径未覆盖"。
    if isinstance(v, (set, frozenset)): return type(v)()
    if isinstance(v, str):   return ""
    if v is None:            return None
    if isinstance(v, (int, float)): return 0
    return v

NV = [neutral(x) for x in seen]
if not any(a != b for a, b in zip(seen, NV)):
    #  ★空变异分两种, 必须让读的人一眼看出是哪种 —— 所以把观测到的返回值样本打出来:
    #    (a) 判据恒返回"无发现"值 ([] / 全 True) = 真缺口, 自检只走通过路径;
    #    (b) 函数按设计恒返回常量 (CLI 入口 return 0 / 打印函数) = 无从断言, 不是缺口。
    #    首版只报"空变异", 于是 watchdog.probe (恒 return 0 的入口) 与
    #    hwfloor.check_no_divergence (真缺口) 混在一类里, 读的人得逐个翻源码。
    uniq = {repr(x)[:40] for x in seen}
    print("SAMPLE %d次观测 值域=%s" % (len(seen), " | ".join(sorted(uniq)[:3])))
    sys.exit(4)

calls = {"i": 0}
def mutant(*a, **k):
    i = calls["i"]; calls["i"] += 1
    return copy.deepcopy(NV[i] if i < len(NV) else NV[-1])
setattr(M, mode, mutant)
try:
    rc = run_selftest()
except BaseException:
    sys.exit(1)                      # 变异令自检崩 -> 算翻转, 但证据弱, 另记一类
#  ★退出码写成**显式常量**而不是 `0 if rc else 3`: 首版那样写, 调用方把 0 读成了
#    "摆设" (0 其实是"翻了"), 于是 13 条已断言的判据被报成摆设。
sys.exit(5 if rc else 3)             # 5=自检翻转(已断言)  3=自检仍绿(摆设)
'''


def traced_criteria(module: str, timeout: int = 180) -> dict:
    """无标签模块的变异自检: 判据由"自检调用了谁"挑出, 中性值按返回值形状造。

    ★**本算子的精确语义是「自检调用了它但不依赖它的返回值」, 不是「它是摆设判据」。**
      挑判据靠的是追踪, 于是 `cann_root` (查环境变量) 与 `predict`
      (`(blocks+depth)*t_block` 一行算式) 这类也会被收进来。
      2026-10-30 全仓首扫 24 个模块报出 10 条, 抽查五条里只有两条
      (`watchdog.probe` 静默期归因 · `loopguard.elites` 有成文规则"格间不比较") 是判据。
      ⇒ **读这份输出时先问"它是不是在下判决"**: 判据落进 weak 才叫摆设, 必须补自检;
        辅助函数落进 weak 只说明它的行为没被测, 补不补看它坏了会不会静默传播。

    四类结果, **空变异必须与摆设分开报**:
      asserted — 变异后自检退出码翻转, 返回值确实被断言;
      weak     — 变异真的改了返回值而自检仍绿;
      vacuous  — 中性值与每一次观测都相同, 变异无从发生。
                 这**不是**"测不出", 是一条更强的结论: 自检只观测得到通过值,
                 **判据的检出路径一次没走过** (hwfloor 的 check_no_divergence 即此例:
                 自检写 `if div: bad += len(div)`, 而它真实恒返回 [])。
      crash    — 变异令自检抛异常。算翻转但证据弱 (只证明被调用, 没证明判词被检查)。
    """
    import tempfile
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    argv = _selftest_argv(root, module)
    if argv is None:
        return dict(module=module, status="opdbg 的自检表里没有它")
    base = _sp.run([_sys.executable, "-m", f"core.{module}"] + argv,
                   capture_output=True, cwd=root, timeout=timeout)
    if base.returncode != 0:
        return dict(module=module, status=f"基线不绿 rc={base.returncode}")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(_TRACE_DRIVER)
        drv = fh.name
    try:
        import json as _json
        #  ★argv 必须跟着传下去 —— 驱动回落到 main() 时要靠它 (trust/perf 没有裸
        #    selftest())。首版漏传, 于是 sys.argv 只有模块名, main() 什么也没跑而
        #    追踪到 0 个函数, 看起来像"这个模块没有判据"。
        got = _sp.run([_sys.executable, drv, module, "--trace"] + list(argv),
                      capture_output=True, text=True, cwd=root, timeout=timeout)
        try:
            names = _json.loads(got.stdout.strip().splitlines()[-1])
        except Exception:
            return dict(module=module, status="追踪不到被调用的函数",
                        detail=got.stderr[-200:])
        exempt = _mutation_exempt(_os.path.join(root, "core", f"{module}.py"))
        out = dict(module=module, status="ok", asserted=[], weak=[],
                   vacuous=[], crash=[], uncalled=[],
                   exempt={k: v for k, v in exempt.items() if k in names})
        for n in names:
            if n in exempt:
                continue
            r = _sp.run([_sys.executable, drv, module, n] + list(argv), capture_output=True,
                        text=True, cwd=root, timeout=timeout)
            kind = {5: "asserted", 3: "weak", 4: "vacuous",
                    1: "crash", 2: "uncalled"}.get(r.returncode, "uncalled")
            if kind == "vacuous":
                #  带上观测样本, 让"真缺口"与"恒返回常量的入口"当场可分
                s = [x for x in r.stdout.splitlines() if x.startswith("SAMPLE ")]
                out["vacuous"].append(f"{n}  [{s[-1][7:]}]" if s else n)
            else:
                out[kind].append(n)
        out["total"] = sum(len(out[k]) for k in
                           ("asserted", "weak", "vacuous", "crash"))
        return out
    finally:
        _os.unlink(drv)


def unasserted_criteria(module: str, timeout: int = 180) -> dict:
    """变异自检: 把模块里的每条 check_* 换成恒返回空, 看它的 --selftest 退出码会不会翻。

    回答的问题是"**这条判据的自检是不是真的接上了**" —— 不是能靠读代码判断的事。
    2026-10-28 一个会话里同一形态撞了七次, 每次都是判据在、返回值也对, 而接线把它丢了:
      - ub_budget 的打印路径取全文件和而不是 per_function 最大;
      - abgate 的 G5 放在参数序之前, 一命中就 UnboundLocalError;
      - oplog 解析失配时报"失败不在 device 侧", 把答案盖掉;
      - 全仓 14 个模块的 __main__ 写成裸 main(), --selftest 退出码恒为 0,
        而 opdbg 按退出码判 PASS —— detsync_lint 的"1 条未过"被盖了至少一个月;
      - detsync_lint 的覆盖守卫按 cases 表统计, 用例放在它之后就不算覆盖;
      - 它的三态断言硬编码 level == "错误", 对唯一一条发 "严重" 的判据失明。
    每一次都是**注入一个故意的失败、看退出码变不变**才发现的。本函数把这个动作自动化。

    ★每个变异起一个**子进程**: 同进程串跑会互相污染 (实测批量扫描时 hangdiag 报
      "基线不绿", 单独跑却是 rc=0)。隔离的代价是慢, 故这条不进每次提交的闸,
      它是周期性体检。
    ★驱动写成临时文件再跑, 不用 `python -c`: 嵌套引号会把判据名里的转义弄坏
      (首版就这么静默返回了"0 条判据")。
    """
    import tempfile
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    #  ★自检的调用方式**不统一**, 而且不是"换个开关"那么简单: shipcheck 要
    #    `x --selftest-criteria` (位置参数占位), opfuzz 要 `--op x --selftest`。
    #    猜开关的那版把它们一律报成"基线不绿", 看着像模块坏了而其实是入口不对。
    #    唯一的真源是 opdbg 的 CRITERIA_EXTRA 表, 这里读它, 不另立一套。
    argv = _selftest_argv(root, module)
    if argv is None:
        return dict(module=module, status="opdbg 的自检表里没有它", total=0, weak=[])
    base = _sp.run([_sys.executable, "-m", f"core.{module}"] + argv,
                   capture_output=True, cwd=root, timeout=timeout)
    if base.returncode != 0:
        return dict(module=module, status=f"基线不绿 rc={base.returncode}", total=0, weak=[])
    driver = (
        "import sys\n"
        "import importlib\n"
        "M = importlib.import_module('core.' + sys.argv[1])\n"
        "if len(sys.argv) > 2 and sys.argv[2]:\n"
        "    if not hasattr(M, sys.argv[2]):\n"
        "        sys.exit(97)\n"
        "    setattr(M, sys.argv[2], lambda *a, **k: [])\n"
        "    rc = M.selftest()\n"
        "    sys.exit(rc if isinstance(rc, int) else 0)\n"
        "names = sorted(n for n in sys.argv[3].split(',') if n)\n"
        "sys.stdout.write(chr(10).join(n for n in names if callable(getattr(M, n, None))))\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(driver)
        drv = fh.name
    try:
        #  ★判据函数名**不靠命名约定猜**: 按 `check="XX_..."` 字面量归属到它所在的 def,
        #    与 duplicate_criterion_numbers 同一套。只认 check_* 的那版在 abgate 上
        #    静默返回"0 条判据" —— 它的判据是 null_floor / verdict 这类普通函数名。
        owners = _criterion_owners(_os.path.join(root, "core", f"{module}.py"))
        if not owners:
            return dict(module=module,
                        status="判据标签不是已识别的三种写法, 本工具不适用",
                        total=0, weak=[])
        got = _sp.run([_sys.executable, drv, module, "", ",".join(sorted(owners))],
                      capture_output=True, text=True, cwd=root, timeout=timeout)
        names = [n for n in got.stdout.split() if n]
        if not names:
            return dict(module=module, status="取不到判据名", total=0, weak=[],
                        detail=got.stderr[-200:])
        #  ★正当豁免: 有的判据函数**本来就不该**进自检 —— netgate 的 run 要真实网络,
        #    它的自检文档写着"在自检里跑它等于让判据的对错取决于跑自检时的网络状况,
        #    那不是自检"。这类必须报成豁免而不是缺口, 否则工具会逼人去写一个假的自检。
        #    豁免要在模块里**写明理由**: `_MUTATION_EXEMPT = {"函数名": "理由"}`。
        exempt = _mutation_exempt(_os.path.join(root, "core", f"{module}.py"))
        weak = []
        for n in names:
            if n in exempt:
                continue
            r = _sp.run([_sys.executable, drv, module, n, ""], capture_output=True,
                        cwd=root, timeout=timeout)
            if r.returncode == 0:
                weak.append(n)
        return dict(module=module, status="ok", total=len(names), weak=weak,
                    exempt={k: v for k, v in exempt.items() if k in names})
    finally:
        _os.unlink(drv)

def duplicate_criterion_numbers(path: str) -> dict:
    """同一个编号前缀 (C12 / S9 / A13 ...) 跨**两个不同的检查函数**即算撞号。

    同一条判据出多个结果码 (C14 的 agree/disagree/unresolved) 不算 ——
    按 `check="XX_..."` 字面量所属的 def 归属判, 不按码名的词形猜。
    返回 {编号: [函数名, ...]}, 空字典表示没有撞号。
    """
    import collections
    import re as _re
    try:
        src = open(path, encoding="utf-8").read()
    except OSError:
        return {}
    spans = [(m.group(1), m.start()) for m in _re.finditer(r"^def (\w+)\(", src, _re.M)]

    def owner(pos: int) -> str:
        name = "<module>"
        for fn, st in spans:
            if st <= pos:
                name = fn
            else:
                break
        return name

    by = collections.defaultdict(set)
    for m in _re.finditer(r'check="([A-Z]+\d+)_[a-z0-9_]+"', src):
        by[m.group(1)].add(owner(m.start()))
    return {k: sorted(v) for k, v in by.items() if len(v) > 1}


def selftest_scope() -> int:
    """适用域的六条负例/正例。

    ★第 5 条是最容易写反的那条: 探测不到 SoC 时如果按"不匹配"处理, 一台没装 CANN 的
      机器上整张判据表会**静默空掉**, 而空表看起来跟"本模块没有判据"一模一样。
    """
    import contextlib
    import io as _io
    passed = failed = 0

    def chk(cond, msg):
        nonlocal passed, failed
        print(("  ✓ " if cond else "  ✗ ") + msg)
        passed, failed = passed + bool(cond), failed + (not cond)

    R = {"plain": Ref(what="无域", status="measured", evidence="e", verified="2026-10-01"),
         "dsl": Ref(what="DSL", status="measured", evidence="e", verified="2026-10-01",
                    scope="dsl"),
         "soc": Ref(what="别的芯片", status="measured", evidence="e", verified="2026-10-01",
                    scope="ascend950pr")}

    def render(**kw):
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_refs("selftest", R, today="2026-10-31", **kw)
        return buf.getvalue()

    _SOC_CACHE[:] = ["ascend910b2"]
    out = render(stack="ascendc")
    chk("plain" in out, "无域条目在默认视图里必须在 (既有条目一条不少)")
    chk("[DSL] dsl" not in out, "栈不符的条目默认不列")
    chk("dsl" in out.split("☆")[-1], "被隐藏的条目名字必须出现在表尾 —— 静默隐藏是另一种掩盖")
    chk("soc" in out.split("☆")[-1], "芯片不符的条目同样隐藏并留痕")

    out = render(stack="all")
    chk("[DSL] dsl" in out and "[ASCEND950PR] soc" in out,
        "OPDBG_STACK=all 两个维度一起开, 且带域条目必须带可见标签")

    _SOC_CACHE[:] = [None]
    body = render(stack="ascendc").split("\u2606")[0]
    chk("[ASCEND950PR] soc" in body,
        "探测不到 SoC 时**不按芯片隐藏**, 条目要留在正文里 (否则没装 CANN 的机器整表静默空掉)")
    _SOC_CACHE.clear()

    chk(scope_tag(R["dsl"]) and scope_tag(R["soc"]), "带域条目的标签永远非空")
    chk(scope_tag(R["plain"]) == "", "无域条目不打标签 (别让标签淹没本机本栈的条目)")

    print(f"\n判据适用域自检: {passed} 过 / {failed} 未过")
    return 1 if failed else 0


if __name__ == "__main__":
    #  ★本模块是库, 这个入口只为**让适用域这条机制可被自检**: 它会隐藏条目,
    #    而会隐藏东西的机制必须有负例看着 (本仓 "判据不能只是断言" 的同一条)。
    _sys.exit(selftest_scope() if "--selftest" in _sys.argv else
              (print("用法: python -m core._refs --selftest") or 0))
