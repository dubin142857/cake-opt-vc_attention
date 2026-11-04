"""GM/workspace 容量 vs 写入范围 静态一致性检查 (算子无关, build 前拦静默越界).

沉淀动机 (dual-loop, 同类第 4 次): "缓冲容量不随写入范围缩放" 这一类缺陷反复出现, 失效模式是**静默写坏
相邻内存** —— 不报错、不崩, 只是结果错, 且往往错在与越界点无关的地方 (被踩的是别的缓冲/元数据).
编译期查不出, CPU 复现查不出, 真机也常常只表现为数值异常. 故必须静态拦.

**本工具与具体算子无关**, 只依赖 CANN/AscendC 的通用结构与命名约定:
  host 侧: 形如 `<xxx>WorkspaceSize<xxx> = <expr>;` / `workspaces[i] = <expr>;` 的容量赋值
  kernel 侧: `GlobalTensor<T> buf;` + 对 buf 的写入 (DataCopyPad(buf[EXPR],..) / InitOutput(buf[EXPR],..) /
             buf.SetGlobalBuffer(base + OFFSET_EXPR))
  缩放符号: shape 类 (m,n,k,batch,groupNum,totalInGroup,...) 与 tiling 类 (base[MNK],single[MN],
            parallNum,coreNum,blockDim,...) —— 均为 CANN 惯用名, 可用 --symbols 覆盖/扩充.

两条判据 (均为机械规则, 无需构建/真机):
  W1 容量不随写入范围缩放: 写入范围表达式含缩放符号 S, 而容量表达式不含 S
     -> shape/tiling 一变大即越界. (实例: 容量固定 96MB 常量, 而写入 batch*n; 容量写死 256*128,
        而写入 baseM*baseN)
  W2 容量含裸数字常量: 容量表达式里出现 >=64 的裸字面量, 而 kernel 用 tiling 符号索引同一缓冲
     -> 典型的"tile 尺寸被硬编码", 改 tiling 时必忘同步.

用法:
  python -m core.wscap <kernel.h> --host <tiling.cpp> [--symbols a,b,c] [--json]
  python -m core.wscap --scan <ops_root>        # 批量扫一棵算子树 (每个算子自动配对 host/kernel)
"""
from __future__ import annotations
import argparse, json, os, re, sys, glob

from ._refs import Ref, print_refs
from . import _dialect

REFS = {
    "W1_capacity_not_scaling": Ref(
        what="host 算的 workspace 容量不随某个决定写入量的符号缩放 —— 形状一变就写到界外",
        status="measured",
        evidence="sfag/gmfr 上反复用到的那一类: 容量表达式里少了 singleN / coreNum 这种因子时, "
                 "小 shape 恰好够用而大 shape 静默越界。判据把 host 的容量赋值与 kernel 的写入"
                 "各自依赖哪些符号并排列出来, 对不上就报。",
        verified="2026-09-17"),
    "W3_external_index_unguarded": Ref(
        what="GM 索引掺入从**输入张量** GetValue 读出的值, 而该值在 kernel 内从未与上界比较",
        status="measured",
        evidence="gmfr 实撞: 单个越界的 row_index 砸中共享尺度表, 99.5% 输出错且不报错。\n"
                 "★2026-09-17 跨算子复验时, 本判据在 gmfr 上**报出了它自己的动机实例** ——\n"
                 "  yGm 的索引掺入 tokenRanksGm 的 GetValue 值 (outRow), 无上下界检查。\n"
                 "  判据都是事后写的, 能在真实算子上重新报出来才算数。",
        verified="2026-09-17"),
    "W6_row_stride_misalign": Ref(
        what="GM 行主布局的行跨距不是 32B 的倍数 -> 非首行行首地址未对齐 -> ADDR_MISALIGN (507015)",
        status="measured",
        evidence="原子写的三条约束是**不同**的东西: 地址对齐(崩) / 长度 32B(静默丢数) / "
                 "长度 512B(仅性能)。W6 管第一条, W6b 管 UB 源侧同一约束。",
        verified="2026-09-16"),
    "_误报普查_2026-09-17": Ref(
        what="W1/W2/W3 的误报密度 —— 需要 host 才能激活, 所以用少量算子配齐 host 来验, 不是 60 个",
        status="measured",
        evidence="不给 host 时: 451 个文件扫过、识别到 311 个 GM 缓冲 (**看得见**), 判据零命中 ——\n"
                 "这是作用域限制不是通过, GM 容量来自 host tiling。\n"
                 "配齐 host 后逐个验:\n"
                 "  sfag (12/12 Pass 的良品): 列出 14 处容量符号与 4 个写入点, W1/W2/W3 **零命中** (正例过)\n"
                 "  gmfr: 命中 1 条 W3, 正是它自己的动机实例 (见上条) —— 负例过\n"
                 "★局限如实记: gmfr 的 host 里识别到**容量赋值 0 处**, 所以 W1/W2 在该算子上未激活,\n"
                 "  本次只验到了 W3。",
        verified="2026-09-17"),
}

# CANN 惯用缩放符号 (shape 维 / tiling 参数). 不含任何算子专属名.
SHAPE_SYMS = ["m", "n", "k", "batch", "groupNum", "totalInGroup", "quantGroupNum", "outputBs", "seqLen",
              "numHeads", "hiddenSize", "tokenNum", "expertNum", "topK"]
TILING_SYMS = ["baseM", "baseN", "baseK", "singleM", "singleN", "singleK",
               # TCubeTiling 标准字段名 (matmul 库 tiling 结构体), 与上面的简称并列收录:
               "singleCoreM", "singleCoreN", "singleCoreK", "depthA1", "depthB1",
               "parallNum", "coreNum", "blockDim", "ubCalSize", "vBaseM",
               "stepM", "stepN", "stepKa", "stepKb", "windowSize"]
# tile 形状因子: 两侧都直接取自 matmulTiling, 故 host/kernel **同名**; 也正是改 tiling 时真正变动的量.
# 并行度类符号 (parallNum/coreNum/blockDim/CV_PARALL_NUM) host 与 kernel 惯用名不同, 拿它们比会全是噪声.
TILE_SYMS = {"baseM", "baseN", "baseK", "singleM", "singleN", "singleK",
             "singleCoreM", "singleCoreN", "singleCoreK", "stepM", "stepN", "stepKa", "stepKb"}
CAP_LHS = re.compile(r"\b(\w*[Ww]ork[Ss]pace\w*)\s*(?:\[[^\]]*\])?\s*(?:=|\+=)\s*([^;]+);")
GM_DECL = re.compile(r"GlobalTensor\s*<[^>]+>\s+(\w+)\s*;")
LITERAL = re.compile(r"(?<![\w.])(\d{2,})(?![\w.])")
# CANN kernel 入口签名惯例: (GM_ADDR x, ..., GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling).
# 只有从 workspace 指针派生的 GlobalTensor 其容量才由 host 侧 workspaceSize 决定; 输入/输出张量的
# 容量来自张量本身 shape, 不该用 W1 判 (否则每个算子的 yGm 都误报).
WS_NAME = re.compile(r"(?<![A-Za-z0-9_])(?:usr|user|deter\w*|sync\w*)?[Ww]ork[Ss]pace\w*(?![A-Za-z0-9_])|"
                     r"(?<![A-Za-z0-9_])(?:ws|wsGm|usrWs|userWs)(?![A-Za-z0-9_])")


def workspace_derived(k_src: str, bufs) -> set:
    """哪些 GlobalTensor 的基址派生自 workspace 指针 (含经局部指针中转的两跳)."""
    roots = set()
    for m in re.finditer(r"__gm__[^;=]*?\*\s*(\w+)\s*=\s*([^;]+);", k_src):
        if WS_NAME.search(m.group(2)):
            roots.add(m.group(1))
    for _ in range(2):  # 传递闭包 (ws -> 局部指针 -> 再偏移)
        for m in re.finditer(r"\b(\w+)\s*=\s*([^;]+);", k_src):
            if any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(r)}(?![A-Za-z0-9_])", m.group(2)) for r in roots):
                roots.add(m.group(1))
    out = set()
    for buf in bufs:
        for m in re.finditer(rf"{re.escape(buf)}\s*\.SetGlobalBuffer\s*\(([^;]*?)\)\s*;", k_src, re.S):
            arg = m.group(1)
            if WS_NAME.search(arg) or any(
                    re.search(rf"(?<![A-Za-z0-9_]){re.escape(r)}(?![A-Za-z0-9_])", arg) for r in roots):
                out.add(buf)
    return out


def input_derived(k_src: str, bufs, ws_bufs) -> set:
    """能**证明**基址来自 kernel 入口输入指针的 GlobalTensor.

    只有这类张量的 GetValue 才是"外部不可信数据". 若某张量在本文件里没有 SetGlobalBuffer (常见于
    经参数传入的中间张量, 如 lightning_indexer 的 vec1ParamGm = 前级算出的参数), 则来源无法判定,
    W3 不报 —— 检查器只在能证明外部来源时才失败, 免得把内部产生的偏移也判成越界风险.
    """
    out = set()
    for buf in bufs:
        if buf in ws_bufs:
            continue
        if re.search(rf"{re.escape(buf)}\s*\.SetGlobalBuffer\s*\(", k_src):
            out.add(buf)
    return out


def external_index_vars(k_src: str, idx_expr: str) -> tuple[set, bool]:
    """索引表达式里哪些变量源自输入张量 GetValue; 以及其派生链上是否出现过范围比较.

    派生链: v = ....GetValue(..) 起, 任何 u = f(v) 也算派生 (常见 rowSigned -> outRow 的 cast 中转).
    """
    asgn = {}
    for m in re.finditer(r"\b(\w+)\s*=\s*([^;]+);", k_src):
        asgn.setdefault(m.group(1), []).append(m.group(2))
    # 纯别名边: RHS 去掉 cast/括号后只剩单个标识符 (rowSigned -> outRow 的 static_cast 属此类).
    # 只沿这种边扩散; 一旦掺入算术就停 —— 否则会把循环计数器 i 之类拉进链, 而 "i < n" 到处都有,
    # 会把毫无检查的代码判成已检查 (本工具自测时真撞).
    def alias_up(v):
        out = set()
        for rhs in asgn.get(v, []):
            bare = re.sub(r"static_cast\s*<[^>]*>|[()\s]", "", rhs)
            if re.fullmatch(r"[A-Za-z_]\w*", bare):
                out.add(bare)
        return out

    def src_tensors(v, depth=0):
        """v 的值最终从哪些 GlobalTensor 的 GetValue 来 (空集 = 不是外部取值)."""
        if depth > 4:
            return set()
        out = set()
        for r in asgn.get(v, []):
            out |= set(re.findall(r"(\w+)\s*(?:\.|->)\s*GetValue\s*\(", r))
        for u in alias_up(v):
            out |= src_tensors(u, depth + 1)
        return out

    # 流不敏感的代价: 同名变量在文件里可能有多处赋值 (循环计数器 + 某处的 GetValue). 若并非全部赋值
    # 都来自 GetValue, 则"该索引确实携带外部值"无法确定 -> 记为不确定, 由调用方降为 info.
    # (跨算子扫描实测: nsa_selected_attention_infer 的 n2Idx 有 6 处赋值, 仅 1 处是 GetValue.)
    seeds, srcs, certain = set(), set(), True
    for v in re.findall(r"\b([a-zA-Z_]\w*)\b", idx_expr):
        st = src_tensors(v)
        if st:
            seeds.add(v)
            srcs |= st
            rhss = asgn.get(v, [])
            if len(rhss) > 1 and not all("GetValue" in r for r in rhss):
                certain = False
    if not seeds:
        return set(), False, set(), True
    chain = set(seeds)
    for _ in range(4):  # 沿纯别名边双向闭包
        for v in list(chain):
            chain |= alias_up(v)
        for lhs in asgn:
            if alias_up(lhs) & chain:
                chain.add(lhs)
    guarded = any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(c)}(?![A-Za-z0-9_])\s*(?:<|>=|>|<=)|"
                            rf"(?:<|>=|>|<=)\s*(?:static_cast\s*<[^>]*>\s*\()?\s*"
                            rf"(?<![A-Za-z0-9_]){re.escape(c)}(?![A-Za-z0-9_])", k_src) for c in chain)
    return seeds, guarded, srcs, certain


def strip_comments(src: str) -> str:
    out, i, n = [], 0, len(src)
    while i < n:
        if src[i:i + 2] == "//":
            j = src.find("\n", i); j = n if j < 0 else j
            out.append(" " * (j - i)); i = j
        elif src[i:i + 2] == "/*":
            j = src.find("*/", i); j = n if j < 0 else j + 2
            out.append("".join("\n" if c == "\n" else " " for c in src[i:j])); i = j
        else:
            out.append(src[i]); i += 1
    return "".join(out)


def strip_strings(src: str) -> str:
    """把 "..." 字面量换成等长空白 (保留行号). 必要性: 日志格式串里普遍写着 `baseN=%u`,
    不剥的话"符号赋值"正则会把格式串当成赋值式, 推断按 gcd 合并所有赋值 -> 一条垃圾就把结论压到 1."""
    out, i, n, ins = [], 0, len(src), False
    while i < n:
        c = src[i]
        if ins:
            out.append("\n" if c == "\n" else " ")
            if c == "\\" and i + 1 < n:
                out.append(" "); i += 2; continue
            if c == '"':
                ins = False
            i += 1
        else:
            if c == '"':
                ins = True; out.append(" ")
            else:
                out.append(c)
            i += 1
    return "".join(out)


def syms_in(expr: str, vocab) -> set:
    """表达式里出现的缩放符号 (作为独立标识符, 忽略 tiling->/this-> 前缀与 static_cast)."""
    e = re.sub(r"\b(?:tiling|tilingData|this)\s*(?:->|\.)\s*", "", expr)
    e = re.sub(r"static_cast\s*<[^>]*>", "", e)
    # CANN tiling 结构体一律通过生成的访问器读写 (get_singleCoreM() / set_stepM(..)), 前缀里的下划线
    # 会让"独立标识符"的边界判定失效, 故先剥掉 get_/set_ 前缀再匹配. 属所有算子通用写法, 非个例适配.
    e = re.sub(r"\b(?:get|set)_(\w+)", r"\1", e)
    return {s for s in vocab if re.search(rf"(?<![A-Za-z0-9_]){s}_?(?![A-Za-z0-9_])", e)}


def inline_locals(host_src: str, expr: str, vocab=(), depth: int = 2) -> str:
    """把容量表达式里引用的局部变量替换成它自己的赋值式 (最多 depth 层).

    必要性: 容量常写成 `userWorkspaceSize[0] = userSize + sysWorkspaceSize;`, 真正的缩放符号藏在
    userSize 的定义里. 不展开就会把正确代码判成"容量不随 shape 缩放" (跨算子扫描时真撞: 误报
    moe_compute_expert_tokens).

    ★只在展开能**带来**符号信息时才展开. 本函数不认作用域, 而 tiling 文件里同名局部量 (baseM/baseN)
    在各分支函数各有一份定义, 若把 `baseM` 替换成别处的 `baseM = 16`, 既取错值又把符号本身抹掉
    (实测: 因此把 W4 判成反的). 故 RHS 若不含任何 shape/tiling 符号也不含 get_*() 取值, 就保留原标识符.
    """
    defs = {}
    for m in re.finditer(r"\b(?:size_t|uint\d+_t|int\d+_t|auto)\s+(\w+)\s*=\s*([^;]+);", host_src):
        rhs = m.group(2)
        informative = bool(re.search(r"get_\w+\s*\(", rhs)) or bool(syms_in(rhs, vocab))
        if informative:
            defs.setdefault(m.group(1), rhs)
    for _ in range(depth):
        prev = expr
        for v, rhs in defs.items():
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(v)}(?![A-Za-z0-9_])", expr):
                expr = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(v)}(?![A-Za-z0-9_])", f"({rhs})", expr)
        if expr == prev:
            break
    return expr


SCALES_WITH_SOMETHING = re.compile(r"get_\w+\s*\(|(?:tiling|tilingData)\s*(?:->|\.)\s*\w+|"
                                   r"GetStorageShape|GetDim\s*\(|\bshape\w*\b")


def host_capacities(host_src: str, vocab):
    """host 侧容量赋值 -> [(行号, 变量, 表达式, 符号集, 裸字面量集)]"""
    caps = []
    for m in CAP_LHS.finditer(host_src):
        lhs, raw = m.group(1), " ".join(m.group(2).split())
        # 排除"取 workspace 尺寸数组指针"这类非容量赋值 (如 size_t* userWorkspaceSize =
        # context->GetWorkspaceSizes(1);). 它不含 shape 项, 若当成容量会被判成"常量容量",
        # 进而把随 shape 缩放的真容量误判成高置信缺陷 (跨算子扫描实测: moe_compute_expert_tokens).
        stmt_start = host_src.rfind(";", 0, m.start()) + 1
        if "*" in host_src[stmt_start:m.start()] or "GetWorkspaceSizes" in raw:
            continue
        # W1 看展开后的表达式 (缩放符号常藏在中间变量里); W2 看原始表达式 —— 展开会把 SYS_WORKSPACE
        # 这类命名常量的数值搬进来, 而"命名常量"恰恰不是 W2 要抓的"写在容量点上的裸数字".
        expr = inline_locals(host_src, raw, vocab)
        ln = host_src[:m.start()].count("\n") + 1
        lits = {int(x) for x in LITERAL.findall(raw) if int(x) >= 64}
        # sizeof(...) 里的数字与 1024/MB 类换算不算 tile 硬编码
        lits -= {1024, 4096 if "MB" in expr or "1024" in expr else -1}
        caps.append(dict(line=ln, lhs=lhs, expr=expr, syms=syms_in(expr, vocab), lits=sorted(lits),
                         scaled=bool(SCALES_WITH_SOMETHING.search(expr))))
    return caps


def kernel_gm_writes(k_src: str, vocab):
    """kernel 侧对 GM 缓冲的写入 -> {buf: {"syms":set, "exprs":[(行号, 片段)], "getvalue": bool}}"""
    bufs = {b: dict(syms=set(), exprs=[], getvalue=False, ext_vars=set(), ext_srcs=set(), ext_guarded=False, ext_certain=True, ws=False)
            for b in GM_DECL.findall(k_src)}
    if not bufs:
        return bufs
    names = "|".join(map(re.escape, bufs))
    # 写入点: DataCopyPad(buf[..],..) / DataCopy*(buf[..],..) / InitOutput<..>(buf[..],..) / buf.SetValue(
    pat = re.compile(rf"(?:DataCopyPad|DataCopy\w*|InitOutput\s*<[^>]*>)\s*\(\s*({names})\s*\[([^\]]*)\]")
    for m in pat.finditer(k_src):
        buf, idx = m.group(1), " ".join(m.group(2).split())
        ln = k_src[:m.start()].count("\n") + 1
        bufs[buf]["syms"] |= syms_in(idx, vocab)
        bufs[buf]["exprs"].append((ln, f"{buf}[{idx[:56]}]"))
    # SetGlobalBuffer 的基址偏移也计入 (决定该缓冲起点, 越界同样致命)
    for m in re.finditer(rf"({names})\s*\.SetGlobalBuffer\s*\(([^;]*?)\)\s*;", k_src, re.S):
        buf, arg = m.group(1), " ".join(m.group(2).split())
        bufs[buf]["syms"] |= syms_in(arg, vocab)
    # 索引里是否掺入了"从输入张量取出的值"(GetValue) —— 外部数据当偏移, 需值域检查 (见 W3)
    for m in pat.finditer(k_src):
        buf, idx = m.group(1), m.group(2)
        seeds, guarded, srcs, certain = external_index_vars(k_src, idx)
        if seeds:
            bufs[buf]["ext_certain"] = bufs[buf]["ext_certain"] and certain
            bufs[buf]["getvalue"] = True
            bufs[buf]["ext_vars"] |= seeds
            bufs[buf]["ext_srcs"] |= srcs
            bufs[buf]["ext_guarded"] = bufs[buf]["ext_guarded"] or guarded
    return bufs


SIZEOF_OF = {"int8_t": 1, "uint8_t": 1, "half": 2, "bfloat16_t": 2, "int16_t": 2, "uint16_t": 2,
             "float": 4, "int32_t": 4, "uint32_t": 4, "int64_t": 8, "uint64_t": 8}
GM_DECL_T = re.compile(r"GlobalTensor\s*<\s*([\w:]+)\s*>\s+(\w+)\s*;")
# host 侧"按 32B 对齐算出来"的写法。命中任一即认为跨距是按构造对齐的。
ALIGNED_IDIOM = re.compile(r"CeilAlign|AlignUp|RoundUp"
                           # C++ 整数字面量可带 U/L 后缀 (8ULL), 不吃掉后缀就匹配不上正确写法
                           r"|Ceil\s*\([^;]*?\)\s*\*\s*(?:8|16|32)[uUlL]*\b"
                           r"|/\s*(?:8|16|32)[uUlL]*\s*\)?\s*\*\s*(?:8|16|32)[uUlL]*\b"
                           r"|&\s*~\s*\(?\s*(?:7|15|31)[uUlL]*\b")


def parse_assume(s: str) -> dict:
    """输入契约 -> {符号: 已知模数}. 形如 'n%32==0,curVecBaseN%32==0' (逗号分隔).

    存在意义: 对齐类判据的**可达性**取决于算子的输入契约, 不写清就只能按最坏情形报, 真命中会被
    不可达项埋掉. gmfr 实例: aclnn 入口校验 `n mod 32 == 0`(EZ1001), 于是 n/curVecBaseN 一路
    都是 32 的倍数, 按 n%8 报的那几条全都够不着; 而 `alignBaseN>>3` 只剩 4 的倍数, 那条是真的.
    """
    out = {}
    for part in s.split(","):
        m = re.fullmatch(r"\s*(\w+)\s*%\s*(\d+)\s*==\s*0\s*", part)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def implied_multiple(expr: str, assume: dict) -> int:
    """expr 在契约下**可证**是几的倍数 (证不出返回 1).

    只做能闭式推的三种: 裸符号 / 右移或除以 2 的幂 (模数同步缩小) / 乘常数 (模数放大).
    除法是重点 —— "按 32B 对齐构造出来的量除以 8" 之后只剩 4 的倍数, 对齐保证降级正发生在这.
    """
    e = re.sub(r"\b(?:tiling|tilingData|vecAParams|this|mnConfig)\s*(?:->|\.)\s*", "", expr).strip(" ()")
    m = re.fullmatch(r"(\w+)", e)
    if m:
        return assume.get(m.group(1), 1)
    m = re.fullmatch(r"(\w+)\s*>>\s*(\d+)", e)
    if m:
        base = assume.get(m.group(1), 1)
        return max(1, base >> int(m.group(2)))
    m = re.fullmatch(r"(\w+)\s*/\s*(\d+)", e)
    if m:
        base, d = assume.get(m.group(1), 1), int(m.group(2))
        return max(1, base // d) if d and base % d == 0 else 1
    m = re.fullmatch(r"(\w+)\s*\*\s*(\d+)", e) or re.fullmatch(r"(\d+)\s*\*\s*(\w+)", e)
    if m:
        a, b = m.group(1), m.group(2)
        sym, c = (a, int(b)) if a.isidentifier() else (b, int(a))
        return assume.get(sym, 1) * c
    return 1


def _fold_ternary(e: str) -> str:
    """把 C++ 三目 `c ? a : b` 折成 `(a + b)`.

    为什么是加法: 本推断里 Add 的语义就是取 gcd, 而"两个分支的公共可整除数"正是 gcd —— 折成加法
    即得到可靠下界。为什么要折: 不折则整条赋值解析失败, 而多处赋值按 gcd 合并, 一条解析不了就把
    结论压到"证不出"。条件里普遍带括号 (`(a.x == b.y - 1) ? ..`), 故必须扫描配对而不能用正则.
    """
    for _ in range(4):
        q = e.find("?")
        if q < 0:
            break
        d = 0
        c = -1
        for i in range(q + 1, len(e)):          # 找同层的 ':'
            if e[i] == "(":
                d += 1
            elif e[i] == ")":
                if d == 0:
                    break
                d -= 1
            elif e[i] == ":" and d == 0:
                c = i
                break
        if c < 0:
            break
        d = 0
        st = 0
        for i in range(q - 1, -1, -1):          # 向左找条件的起点 (括号配平, 或撞到分隔符)
            ch = e[i]
            if ch == ")":
                d += 1
            elif ch == "(":
                if d == 0:
                    st = i + 1
                    break
                d -= 1
            elif ch in ",=" and d == 0:
                st = i + 1
                break
        e = e[:st] + "(" + e[q + 1:c] + " + " + e[c + 1:] + ")"
    return e


def _to_py(expr: str) -> str:
    """C++ 尺寸表达式 -> 可交给 ast 的形式. 只做词法层替换, 不改语义."""
    e = re.sub(r"static_cast\s*<[^>]*>\s*", "", expr)
    #  剥所有 `结构体.` / `结构体->` 前缀: 这类尺寸表达式里承载取值的是**字段名**, 结构体名
    #  只是转手容器 (tiling-> / mnConfig. / syncConfig. / vecAParams. 在 CANN 里都出现).
    #  列白名单必然漏 —— syncConfig.baseN 就是这么漏掉的.
    e = re.sub(r"\b\w+\s*(?:->|\.)\s*", "", e)
    e = re.sub(r"\b(?:get|set)_(\w+)", r"\1", e)
    e = re.sub(r"(\d+)[uUlL]+", r"\1", e)            # 8ULL -> 8
    #  C 风格函数式转型 uint64_t(X) / int32_t(X): 不剥就成了未知函数调用 -> 返回 1, 而多处
    #  赋值取 gcd 会被这个 1 毒化, 把已经可证的量压成"证不出".
    e = re.sub(r"\b(?:u?int(?:8|16|32|64)_t|size_t|uint32|uint64)\s*\(", "(", e)
    e = re.sub(r"\b\w+\s*::\s*", "", e)               # Ops::Base::CeilAlign -> CeilAlign
    #  CeilAlign/AlignUp/RoundUp(x,m) 返回 m 的倍数; Ceil(x,m) 是上取整除法, **不**保证 —— 
    #  只有 Ceil(x,m)*m 才保证, 由 Mult 分支合成. 早先把两者折成同一个名字是错的.
    e = re.sub(r"\b(?:CeilAlign|AlignUp|RoundUp)\s*\(", "AlignTo(", e)
    e = _fold_ternary(e)
    return e.strip().rstrip(";")


_BLOCKERS: set = set()


def infer_with_blockers(expr, assume, srcs=()):
    """(可证倍数, 卡住证明的符号集). 卡住 = 该符号的取值不在给到的源码里 (常见于 matmul
    tiling 库产出的 singleCoreN 之类) —— 此时判据既不能判可达也不能判不可达, 必须说出来,
    否则读的人只能自己重推一遍."""
    _BLOCKERS.clear()
    v = infer_multiple(expr, assume, srcs)
    return v, set(_BLOCKERS)


def infer_multiple(expr: str, assume: dict, srcs=(), depth: int = 4, seen=None) -> int:
    """在输入契约下推 expr **可证**是几的倍数 (推不出返回 1).

    契约只给到入口维度 (如 n%32==0), 但判据要用的往往是它派生出来的量 (curVecBaseN / alignBaseN /
    boundGroups). 不传播就只能按最坏情形报, 于是不可达项把真命中埋掉 —— gmfr 上正是这样:
    n%32==0 已经保证 curVecBaseN 是 32 的倍数, 而 alignBaseN>>3 除完只剩 4 的倍数, 只有后者是真的.

    传播规则 (都是可证的下界, 不猜): 加减取 gcd, 乘取积, 右移/整除按可整除性缩小,
    Ceil(x, m)*m 与 (x + m-1)/m*m 认作 m 的倍数. 符号沿 kernel/host 的赋值式递归展开.
    """
    import ast as _ast
    from math import gcd
    seen = seen or set()
    blockers = _BLOCKERS

    def walk(node) -> int:
        if isinstance(node, _ast.Constant) and isinstance(node.value, int):
            #  0 是 gcd 的单位元 ("任意数都整除 0"), 不能当成 1 —— 当成 1 会把结论压到"证不出".
            return node.value
        if isinstance(node, _ast.Name):
            nm = node.id
            if nm in assume:
                return assume[nm]
            if nm in seen:
                #  自引用 = 归纳情形 (`x = cond ? f : x` / `x += b`): 这一支不带来新约束, 返回 0
                #  (gcd 的单位元) 而非 1. 返回 1 会把"保持原值"的分支误当成"证不出"而压掉结论.
                #  注意只有自引用能返回 0 —— 真正未知的符号必须返回 1, 否则不可靠.
                return 0
            if depth <= 0:
                return 1
            best = 0
            for s in srcs:
                if not s:
                    continue
                #  排除比较式 (`baseN == 0` 的右半会被当成赋值而毒化 gcd), 但**收下 += / -=**:
                #  归纳变量 `x = a; x += b` 保证 x 恒是 gcd(a,b) 的倍数, 而 gcd 正是本推断合并多处
                #  赋值的运算, 所以增量式与普通赋值同样处理即为可靠下界. gmfr 的 offsetN 正是这类
                #  (`offsetN = 0; offsetN += baseN`), 不收下就推不出它是 baseN 的倍数.
                ss = strip_strings(s)
                #  for 头里的步进式没有分号结尾, 单靠上面那条正则抓不到 —— 而循环变量的步长正是
                #  它可整除性的来源 (gmfr 的 offsetN += baseN). 故单列一条模式.
                rhss = [m.group(1) for m in
                        re.finditer(rf"(?<![=!<>*/%&|^])\b{re.escape(nm)}_?\s*[+\-]?=(?!=)\s*([^;]+);", ss)]
                rhss += [m.group(1) for m in
                         re.finditer(rf"for\s*\([^;{{}}]*;[^;{{}}]*;\s*{re.escape(nm)}\s*[+\-]=\s*([^)]+)\)", ss)]
                #  丢掉括号不配平的捕获: 以分号为终止的那条正则会越过 for 头的右括号一直吃到下一个
                #  分号 (`offsetN += baseN) { if (...);`), 抓出的是垃圾, 解析失败即被算成"证不出"而
                #  毒化 gcd. 合法右值必然括号配平, 以此过滤最省且不误杀.
                rhss = [r for r in rhss if r.count("(") == r.count(")") and "{" not in r]
                for rhs in rhss:
                    #  剥掉 tiling->/mnConfig. 前缀后, `mnConfig.baseN = tiling->matmulTiling.baseN`
                    #  会变成 `baseN = baseN` 的同义式. 它不携带信息, 却会被算成"证不出"(1)而毒化 gcd,
                    #  故跳过. 这类跨结构体的同名转手在 CANN tiling 里是普遍写法, 不是个例适配.
                    if _to_py(rhs) == nm:
                        continue
                    v = infer_multiple(rhs, assume, srcs, depth - 1, seen | {nm})
                    best = v if best == 0 else gcd(best, v)   # 多处赋值取 gcd (必须对所有赋值都成立)
            if not best and blockers is not None:
                blockers.add(nm)          # 证明卡在这个符号上 (取值不在给到的源码里)
            return best or 1
        if isinstance(node, _ast.BinOp):
            l, r = walk(node.left), walk(node.right)
            op = node.op
            if isinstance(op, (_ast.Add, _ast.Sub)):
                return gcd(l, r) or 1
            if isinstance(op, _ast.Mult):
                return l * r
            if isinstance(op, _ast.RShift):
                k = node.right.value if isinstance(node.right, _ast.Constant) else None
                return max(1, l >> k) if k is not None and l % (1 << k) == 0 else 1
            if isinstance(op, (_ast.Div, _ast.FloorDiv)):
                d = node.right.value if isinstance(node.right, _ast.Constant) else None
                return max(1, l // d) if d and l % d == 0 else 1
            if isinstance(op, _ast.LShift):
                k = node.right.value if isinstance(node.right, _ast.Constant) else None
                return l << k if k is not None else 1
            return 1
        if isinstance(node, _ast.Call):
            fn = getattr(node.func, "id", "")
            if fn == "AlignTo" and len(node.args) == 2:
                return walk(node.args[1])       # 对齐到 m -> 是 m 的倍数
            if fn == "Ceil" and len(node.args) == 2:
                return 1        # 上取整除法本身不保证; Ceil(x,m)*m 由 Mult 分支合成
            return 1
        if isinstance(node, _ast.UnaryOp):
            return walk(node.operand)
        return 1

    try:
        return walk(_ast.parse(_to_py(expr), mode="eval").body)
    except Exception:
        return implied_multiple(expr, assume)      # 解析不了就退回单符号/单除法那套


def row_stride_align(k_src: str, host_src: str | None, assume: dict | None = None):
    """W6: GM 行主布局的**行跨距**不是 32B 的倍数 -> 非首行的行首地址未对齐 -> ADDR_MISALIGN (507015).

    形态: kernel 写 buf[row * STRIDE + col] (STRIDE 是 tiling 字段), 且该写是 DataCopy* / 原子写。
    硬件要求 DataCopy 的 GM 地址 32B 对齐, 而行首地址 = base + row·STRIDE·sizeof(T)。STRIDE·sizeof(T)
    不是 32 的倍数时, row 取某些值就偏移几个字节 —— **首行永远是对的**, 所以小 shape / row=0 的用例
    全过, 换个 shape 才崩, 且报的是 aicore exception 而非越界, 极易误判成同步问题。
    (2026-08-19 gmfr 实撞: 界累加器 stride = ceil(n/8), n=4096 -> 512 恰好对齐所以一直没暴露,
     n=4000 -> 500, 500·4=2000B 不是 32 的倍数, 奇数行即崩。)

    判据: host 对该 STRIDE 的赋值里没有 32B 对齐惯用写法 (CeilAlign / Ceil(..,8)*8 / /8*8 / &~7),
    且它不是显然的 32B 倍数字面量 -> 报。只看 host 单点赋值, 因为跨距该由 host 单点算出。
    """
    hz = []
    types = {name: t.split("::")[-1] for t, name in GM_DECL_T.findall(k_src)}
    if not types:
        return hz
    names = "|".join(map(re.escape, types))
    # buf[ A * STRIDE + ... ] —— STRIDE 取乘号右侧的标识符 (可带 tiling-> 前缀)
    pat = re.compile(rf"DataCopy\w*\s*\(\s*({names})\s*\[\s*[^\]]*?\*\s*"
                     rf"(?:tiling\s*->\s*)?(\w+)\s*\+[^\]]*\]")
    seen = set()
    for m in pat.finditer(k_src):
        buf, stride = m.group(1), m.group(2)
        esz = SIZEOF_OF.get(types.get(buf, ""), 0)
        if not esz or (buf, stride) in seen:
            continue
        seen.add((buf, stride))
        need = 32 // esz if esz <= 32 else 1
        if need <= 1:
            continue
        im, blk = infer_with_blockers(stride, assume or {}, (k_src, host_src))
        if im % need == 0:
            continue          # 输入契约已保证该跨距是 need 的倍数 -> 不可达, 不报
        ln = k_src[:m.start()].count("\n") + 1
        # host 侧对该跨距的赋值 (成员名惯例带尾下划线)
        asg = []
        if host_src:
            asg = re.findall(rf"\b{re.escape(stride)}_?\s*=\s*([^;]+);", host_src)
        if not asg:
            continue                       # host 里找不到单点赋值, 交给 W4 的符号集比对, 这里不猜
        if any(ALIGNED_IDIOM.search(a) for a in asg):
            continue
        # 赋值本身就是 32B 倍数的字面量也放过
        if all(re.fullmatch(r"\s*\d+\s*", a) and int(a) % need == 0 for a in asg):
            continue
        hz.append(dict(check="W6_row_stride_misalign", buf=buf, line=ln, stride=stride,
                       msg=(f"kernel 在 L{ln} 按 {buf}[row*{stride}+..] 行主写 GM (元素 {types[buf]}, "
                            f"{esz}B), 行首地址 = base + row·{stride}·{esz}B。DataCopy 要求 GM 地址 32B 对齐, "
                            f"故 {stride} 必须是 {need} 的倍数, 但 host 的赋值 '{asg[0].strip()[:60]}' 里没有"
                            f"对齐动作。后果: row=0 恒对, 某些 row 偏几个字节 -> ADDR_MISALIGN / 507015 "
                            f"aicore exception, 换 shape 才复现且报错不指向布局。"
                            f"修: host 把 {stride} 按 {need} 向上取整 (Ceil({stride},{need})*{need}), "
                            f"补出来的尾部槽位不对应真实列, 消费侧须能把它们与真实值区分开。"
                            + (f" 证明卡在 {sorted(blk)[:3]}: 这些符号的取值不在给到的源码里"
                               f"(如 singleCoreN 由 matmul tiling 库产出), 故既判不了可达也判不了不可达 —— "
                               f"要么 host 加显式断言把它变成源码事实, 要么 --assume 把它作为已知契约传进来。"
                               if blk else ""))))
    # W6b: **UB 源**侧的行跨距被除法/右移打散。DataCopy 对 UB 源地址同样要求 32B 对齐, 而 UB 里的行宽
    #      通常由一个"按 32B 对齐构造出来"的量 (alignBaseN 之类) 得到 —— 一旦对它做 /8 或 >>3 (例如把
    #      逐元素布局压成逐组布局), 对齐性就没了, 而**首行地址永远是对的**, 于是只有部分行崩。
    #      2026-08-19 gmfr 实撞: 界的 UB 行宽 = alignBaseN/8, 整行宽 128/8=16 个 int32=64B 恒对齐,
    #      末块窄到 32 时 32/8=4 个 int32=16B, 奇数行偏 16B -> ADDR_MISALIGN, 且 GM 侧完全正常。
    ubpat = re.compile(r"DataCopy\w*\s*\([^;]*?,\s*(\w+)\s*\[\s*(\w+)\s*\*\s*([^\],]*?)\s*\]")
    for m in ubpat.finditer(k_src):
        ub, row, stride = m.group(1), m.group(2), m.group(3).strip()
        if not re.search(r"/\s*\d|>>\s*\d", stride):
            continue                      # 跨距里没有除法 -> 沿用来源的对齐性, 不报
        if re.search(r"\bBoundUbStride\b|CeilAlign|AlignUp|RoundUp", stride):
            continue
        if infer_multiple(stride, assume or {}, (k_src, host_src)) % 8 == 0:
            continue          # 契约保证除完之后仍是 8 个元素 (32B/4B) 的倍数 -> 不可达
        ln = k_src[:m.start()].count("\n") + 1
        hz.append(dict(check="W6b_ub_row_stride_misalign", buf=ub, line=ln, stride=stride,
                       msg=(f"L{ln} 以 {ub}[{row}*{stride}] 逐行取 UB 源做 DataCopy —— 行跨距里带除法/右移, "
                            f"把上游按 32B 对齐构造出来的量除小了, 对齐性不再成立。DataCopy 对 UB 源地址"
                            f"同样要求 32B 对齐, 而 {row}=0 恒对, 只有部分行崩 -> 换 shape 才复现的 "
                            f"ADDR_MISALIGN / 507015, 且 GM 侧地址完全正常, 极易查错方向。"
                            f"修: 把行跨距向上补到 32B/sizeof(元素) 的倍数, 并让**产出该布局的一侧**"
                            f"(归约/搬运) 与**取行的一侧**共用同一个跨距函数, 不各写一遍。")))
    return hz


def region_base_exprs(k_src: str):
    """kernel 侧"区域基址"表达式: SetGlobalBuffer(workspace + EXPR) 里的 EXPR (含经中间变量的两跳).

    这类表达式与 host 侧的容量表达式是**同一个布局公式的两份拷贝** (host 用它算总容量, kernel 用它算
    下一区域的起点). 两份不一致 = 后一区域落进前一区域内部, 被覆写; 症状常是数值异常而非报错.
    """
    out = []
    inter = {}
    for m in re.finditer(r"\b(?:uint64_t|uint32_t|size_t)\s+(\w+)\s*=\s*([^;]+);", k_src):
        inter[m.group(1)] = m.group(2)
    for m in re.finditer(r"(\w+)\s*\.SetGlobalBuffer\s*\(([^;]*?)\)\s*;", k_src, re.S):
        buf, arg = m.group(1), " ".join(m.group(2).split())
        if not WS_NAME.search(arg):
            continue
        expr = arg
        for _ in range(2):   # 展开中间变量 (deterBase / stagingBytes 之类)
            for v, rhs in inter.items():
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(v)}(?![A-Za-z0-9_])", expr):
                    expr = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(v)}(?![A-Za-z0-9_])", f"({rhs})", expr)
        ln = k_src[:m.start()].count("\n") + 1
        out.append(dict(buf=buf, line=ln, expr=" ".join(expr.split())))
    return out


def runtime_capacity_check(host_src: str, missing) -> bool:
    """host 侧是否存在"含 missing 维的表达式 vs 容量变量"的运行期比较 (OP_CHECK_IF / if / assert 皆可)."""
    if not host_src:
        return False
    for m in re.finditer(r"[^;{}]*(?:<|>|<=|>=)[^;{}]*", host_src):
        seg = m.group(0)
        if WS_NAME.search(seg) and all(
                re.search(rf"(?<![A-Za-z0-9_]){s}_?(?![A-Za-z0-9_])", seg) for s in missing):
            return True
    return False



MEMBER_DECL = re.compile(r"^\s*(?:mutable\s+)?(?:u?int(?:8|16|32|64)_t|size_t|bool|float|double|"
                         r"ge::\w+|uint\d+|int)\s+(\w+_)\s*;\s*$", re.M)


MEMBER_DECL = re.compile(r"^[ \t]*(?:mutable\s+)?(?:u?int(?:8|16|32|64)_t|size_t|bool|float|double|"
                         r"ge::\w+|int|unsigned)(?:\s*\*)?\s+(\w+_)\s*;[ \t]*$", re.M)


def uninit_members(host_src: str, hdr_src: str):
    """tiling 类成员在**声明处缺初值** -> 存在读到未初始化内存的可能.

    ★判据选择说明: "该成员是否真的在某条路径上先读后写" 需要真正的数据流分析, 正则做不到 —— 本判据
    先后试过按大括号深度、按反路径 return 覆盖两版, 都产生大量误报, 已放弃. 改判可机械核验且修复代价
    为零的形式: 声明处有无初值. 这条规则没有"误报"概念 (声明处初始化是 C++ 惯例, 加 `= 0` 永远安全),
    而它确实能拦住真实缺陷.

    真实缺陷 (gmfr): `int64_t tuningConfig_;` 的全部赋值包在 `if (context_->GetDeterministic() == 0)`
    内且无 else, 确定性场景从不赋值; 真机 tiling 日志打出 avg_m=825319265, 使同一输入选出不同 tiling
    (性能与结果随运行漂移, 不报错不崩). 该缺陷在上游长期潜伏 —— 原本只在一个窄 shape 分支里被读到,
    一旦被提升为通用判据就立刻发作.

    合并成一条输出, 免得刷屏掩盖其它判据.
    """
    names = MEMBER_DECL.findall(hdr_src)
    if not names:
        return []
    return [dict(check="W5_member_no_initializer", buf=f"{len(names)} 个成员",
                 msg=("tiling 类成员声明处缺初值: " + ", ".join(names[:14])
                      + (" ..." if len(names) > 14 else "")
                      + ". 若某成员的赋值全在条件块内, 条件不成立的路径就读到未初始化内存, 使同一输入"
                        "选出不同 tiling (性能/结果随运行漂移, 不报错). 修: 声明处一律给初值, 如 `= 0`."))]


def analyze(kernel_path: str, host_path: str | None, vocab, hdr_path: str | None = None,
            assume: dict | None = None):
    k_src = strip_comments(open(kernel_path, errors="ignore").read())
    bufs = kernel_gm_writes(k_src, vocab)
    ws_bufs = workspace_derived(k_src, bufs)
    in_bufs = input_derived(k_src, bufs, ws_bufs)
    host_src = strip_comments(open(host_path, errors="ignore").read()) if host_path else ""
    caps = host_capacities(host_src, vocab) if host_path else []
    cap_syms = set().union(*[c["syms"] for c in caps]) if caps else set()
    hz = []
    for buf, info in sorted(bufs.items()):
        if not info["exprs"]:
            continue
        info["ws"] = buf in ws_bufs
        # W1: 写入范围含容量不含的缩放符号. 只对 workspace 派生缓冲成立 —— 输入/输出张量的容量由
        # 张量 shape 决定, 不出现在 host 的 workspaceSize 表达式里, 拿它比会把每个算子的 yGm 都误报.
        missing = sorted(info["syms"] - cap_syms) if info["ws"] else []
        if caps and missing:
            # 容量确实不随该维缩放, 但若 host 侧已有"该维 vs 容量"的运行期校验 (平台固定容量时唯一可行的
            # 补偿手段), 则失效模式从"静默写坏"变成"明确报错", 降级为提示而非失败.
            # 置信度分级. 高置信的形态是: host 侧存在一个**纯常量**容量 (不含任何 shape/tiling 项),
            # 而 kernel 却按 shape 往里写 —— 这正是 gmfr 那条 (确定性区固定 96MB vs 写 batch*n).
            # 反之若每个容量表达式都随某个量缩放, 那么"符号名不匹配"更可能是 host/kernel 命名不一致
            # (跨算子扫描实测: moe_compute_expert_tokens 的 numOfExpert 对应 kernel 的 n), 降为 info 由人判.
            # 排除 system/sys 前缀: 平台固定预留本就是常量, 每个算子都有, 不能据此判全仓高置信.
            has_const_cap = any((not c["scaled"]) and (not c["syms"])
                                and not re.match(r"sys", c["lhs"], re.I) for c in caps)
            if runtime_capacity_check(host_src, missing):
                hz.append(dict(check="W1_info_runtime_checked", buf=buf, missing=missing, severity="info",
                               where=info["exprs"][:2],
                               msg=(f"GM '{buf}' 的容量不随 {'/'.join(missing)} 缩放, 但 host 侧已有该维与容量变量的"
                                    f"运行期比较 -> 超限时会明确报错而非静默越界. 可接受 (平台固定容量常见).")))
            else:
                hz.append(dict(check="W1_capacity_not_scaling", buf=buf, missing=missing,
                               where=info["exprs"][:2],
                               severity=None if has_const_cap else "info",
                               msg=(f"GM '{buf}' 的写入范围随 {'/'.join(missing)} 增长, 但 host 侧所有容量表达式"
                                    f"都不含这些符号, 也没有运行期校验 -> 该维一变大即静默越界. "
                                    f"修: 容量按写入范围推导; 平台容量固定改不了时, 至少加 OP_CHECK_IF 报错.")))
        # W3: 外部取值当索引且该值 (及其派生链) 全程没做过范围比较
        ext_input_srcs = sorted(info["ext_srcs"] & in_bufs)
        if info["getvalue"] and not info["ext_guarded"] and ext_input_srcs:
            hz.append(dict(check="W3_external_index_unguarded", buf=buf, where=info["exprs"][:2],
                           vars=sorted(info["ext_vars"]), srcs=ext_input_srcs,
                           severity=None if info["ext_certain"] else "info",
                           msg=(f"GM '{buf}' 的索引掺入了从输入张量 {'/'.join(ext_input_srcs)} GetValue 读出的值 "
                                f"({'/'.join(sorted(info['ext_vars']))}), 该值及其派生变量在 kernel 内从未与任何"
                                f"上界做比较 -> 非法输入会写到该缓冲之外 (可能砸中相邻元数据/别的缓冲, 且不报错). "
                                f"修: 写前做有符号值域检查 (下界 <0 与上界 >=容量 都要拦).")))
    # W4: 同一布局公式在 host(容量) 与 kernel(下一区域基址) 各写一份, 两份符号集不一致 -> 后一区域
    #     落进前一区域内部被覆写. (gmfr 实撞: host 改成 baseM*(baseN*stepN) 而 kernel 的 deterBase 仍按
    #     baseM*baseN 算 -> 确定性累加器被 matmul 输出覆写 -> det_on 全 NaN, det_off 却正常.)
    for rb in region_base_exprs(k_src):
        base_syms = syms_in(rb["expr"], vocab)
        if not base_syms:
            continue
        for c in caps:
            cap_syms = c["syms"]
            # 只比"看起来是同一区域公式"的一对: 双方都含 tiling 类符号且有交集
            if not (cap_syms & TILE_SYMS) or not (base_syms & TILE_SYMS):
                continue
            if not (cap_syms & base_syms & TILE_SYMS):
                continue
            diff = (cap_syms ^ base_syms) & TILE_SYMS
            if diff:
                hz.append(dict(check="W4_layout_formula_mismatch", buf=rb["buf"], line=rb["line"],
                               missing=sorted(diff),
                               msg=(f"kernel 在 L{rb['line']} 用 {sorted(base_syms)} 算 '{rb['buf']}' 的区域基址, "
                                    f"而 host L{c['line']} 的 {c['lhs']} 用 {sorted(cap_syms)} 算同一布局的容量 —— "
                                    f"两份公式差 {sorted(diff)}. 同一布局公式写两遍必然改一处忘一处, 后果是后一区域"
                                    f"落进前一区域内部被覆写 (症状常是数值异常/NaN 而非报错). "
                                    f"修: 两侧用同一组符号推导, 或把该偏移放进 tiling 由 host 单点计算.")))
                break

    # W6: GM 行主布局的行跨距未按 32B 对齐 (非首行行首地址失配 -> ADDR_MISALIGN)
    hz += row_stride_align(k_src, host_src if host_path else None, assume)
    # W5: tiling 类成员声明无初值且赋值都在条件块内
    if host_path:
        hdr = hdr_path or re.sub(r"\.cpp$", ".h", host_path)
        if os.path.exists(hdr):
            hz += uninit_members(strip_comments(open(host_path, errors="ignore").read()),
                                 strip_comments(open(hdr, errors="ignore").read()))
    # W2: 容量含裸字面量, 而 kernel 用 tiling 符号索引
    tiling_used = set().union(*[i["syms"] for i in bufs.values()]) if bufs else set()
    for c in caps:
        if c["lits"] and (tiling_used & set(TILING_SYMS)):
            hz.append(dict(check="W2_hardcoded_capacity", buf=c["lhs"], line=c["line"],
                           lits=c["lits"], expr=c["expr"][:90],
                           msg=(f"容量表达式含裸字面量 {c['lits']} (行 {c['line']}), 而 kernel 用 tiling 符号"
                                f"({'/'.join(sorted(tiling_used & set(TILING_SYMS)))}) 索引缓冲 -> 疑似 tile 尺寸被硬编码, "
                                f"改 tiling 时必忘同步. 修: 容量由同一组 tiling 变量推导.")))
    return hz, bufs, caps


def pair_of(op_dir: str):
    """给定算子目录, 自动配对 (kernel.h, host tiling.cpp). 取体量最大的作主文件."""
    ks = sorted(glob.glob(os.path.join(op_dir, "op_kernel", "*.h")), key=os.path.getsize, reverse=True)
    hs = sorted(glob.glob(os.path.join(op_dir, "op_host", "*tiling*.cpp")), key=os.path.getsize, reverse=True)
    return (ks[0] if ks else None), (hs[0] if hs else None)


def selftest() -> int:
    """判据自检: 每条判据一个**已知违规**的最小输入 + 一个正例防判据过宽。

    2026-10-31 立。本模块此前**完全没有自检** —— `opdbg --selftest-criteria` 把它列在
    「仍缺判据自检 (判据改坏了不会红)」那一行里。判据有标签 (W1..W6) 而无人断言,
    等于每条都只是声明。

    ★夹具必须贴近真实内核的结构, 不能只写一句写入: W1 要求缓冲**由 workspace 指针派生**
      (靠 `SetGlobalBuffer` 的实参判), W3 要求取值来源是**输入派生**张量。
      只写 `DataCopyPad(wsGm[...])` 而不写 SetGlobalBuffer 时四条判据全部静默不报,
      而那看起来就像"判据没问题"。
    ★写这些用例时自己栽的两次, 都留在注释里:
      1. 用 `wsGm.SetValue(...)` 当写入点 —— 识别不到。见下面 W0 那条。
      2. W6 的 `need` 是**元素数**不是字节数: float 的 need = 32/4 = 8, 而我先写了
         `rowStride_ = 40` 当负例 —— 40 能被 8 整除 (40x4=160B 正好对齐), 判据不报是对的,
         红的是我的期望。改用 gmfr 实撞的 500 (500x4=2000B, 不是 32 的倍数)。
    """
    import tempfile
    vocab = SHAPE_SYMS + TILING_SYMS
    bad, cases = 0, []

    def hits(kernel: str, host: str | None = None, hdr: str | None = None):
        with tempfile.TemporaryDirectory() as d:
            kp = os.path.join(d, "k.h")
            open(kp, "w", encoding="utf-8").write(kernel)
            hp = hdp = None
            if host is not None:
                hp = os.path.join(d, "t.cpp")
                open(hp, "w", encoding="utf-8").write(host)
            if hdr is not None:
                hdp = os.path.join(d, "t.h")
                open(hdp, "w", encoding="utf-8").write(hdr)
            hz, bufs, _ = analyze(kp, hp, vocab, hdr_path=hdp)
            return {h["check"] for h in hz}, bufs

    def ck(name, got, want):
        cases.append((name, got == want, got))

    #  公共夹具: workspace 派生的 wsGm (W1/W6 都要它被判成 ws 派生)
    INIT_WS = ("void Init(__gm__ uint8_t* workspace){\n"
               "  wsGm.SetGlobalBuffer((__gm__ float*)workspace);\n}\n")
    K_SCALE = ("GlobalTensor<float> wsGm;\n" + INIT_WS +
               "void P(){ DataCopyPad(wsGm[batch * n], ub, p); }\n")

    #  ---- W1: 写入随某维增长而容量不随它缩放 ----
    h, _ = hits(K_SCALE, "uint64_t workspaceSize = 96 * 1024 * 1024;\n")
    ck("★W1 负例: 写入随 batch 增长而容量是纯常量 -> 必须报",
       "W1_capacity_not_scaling" in h, True)
    h, _ = hits(K_SCALE, "uint64_t workspaceSize = batch * n * sizeof(float);\n")
    ck("W1 正例: 容量随同一组符号缩放 -> 不许报", "W1_capacity_not_scaling" in h, False)

    #  ---- W2: 容量写死字面量而 kernel 按 tiling 符号索引 ----
    K_TILE = ("GlobalTensor<float> wsGm;\n" + INIT_WS +
              "void P(){ DataCopyPad(wsGm[baseM * 4], ub, p); }\n")
    h, _ = hits(K_TILE, "uint64_t workspaceSize = 65536;\n")
    ck("★W2 负例: 容量写死 65536 而 kernel 按 baseM 索引 -> 必须报",
       "W2_hardcoded_capacity" in h, True)
    h, _ = hits(K_TILE, "uint64_t workspaceSize = baseM * 4 * sizeof(float);\n")
    ck("W2 正例: 容量由 baseM 推导 -> 不许报", "W2_hardcoded_capacity" in h, False)

    #  ---- W3: 外部取值当索引且全程没做范围比较 ----
    K_EXT = ("GlobalTensor<int32_t> idxGm;\nGlobalTensor<float> wsGm;\n"
             "void Init(__gm__ uint8_t* idx, __gm__ uint8_t* workspace){\n"
             "  idxGm.SetGlobalBuffer((__gm__ int32_t*)idx);\n"
             "  wsGm.SetGlobalBuffer((__gm__ float*)workspace);\n}\n"
             "void P(){ int32_t r = idxGm.GetValue(0); DataCopyPad(wsGm[r * 64], ub, p); }\n")
    h, _ = hits(K_EXT, "uint64_t workspaceSize = batch * 64;\n")
    ck("★W3 负例: 输入张量取出的值当索引且未做值域检查 -> 必须报",
       "W3_external_index_unguarded" in h, True)
    #  ★正例: 同样的形态, 但来源**不是**输入派生 (本文件里没有 SetGlobalBuffer) ->
    #    来源无法证明就不报, 这是判据自己声明的作用域, 不是漏报。
    K_EXT_UNKNOWN = ("GlobalTensor<int32_t> midGm;\nGlobalTensor<float> wsGm;\n" + INIT_WS +
                     "void P(){ int32_t r = midGm.GetValue(0); DataCopyPad(wsGm[r * 64], ub, p); }\n")
    h, _ = hits(K_EXT_UNKNOWN, "uint64_t workspaceSize = batch * 64;\n")
    ck("W3 正例: 来源不是输入派生 (证不出外部来源) -> 不许报",
       "W3_external_index_unguarded" in h, False)

    #  ---- W5: host 头文件成员无初始化 ----
    K_MIN = "GlobalTensor<float> wsGm;\n" + INIT_WS + "void P(){ DataCopyPad(wsGm[0], ub, p); }\n"
    h, _ = hits(K_MIN, "uint64_t workspaceSize = 1024;\n",
                hdr="class T {\n  uint32_t coreNum_;\n  int64_t tileLen_;\n};\n")
    ck("★W5 负例: 成员声明无初始化 -> 必须报", "W5_member_no_initializer" in h, True)
    h, _ = hits(K_MIN, "uint64_t workspaceSize = 1024;\n",
                hdr="class T {\n  uint32_t coreNum_ = 0;\n  int64_t tileLen_ = 0;\n};\n")
    ck("W5 正例: 成员都带初始化 -> 不许报", "W5_member_no_initializer" in h, False)

    #  ---- W6: 行跨距不是 32B 的倍数 ----
    K_ROW = ("GlobalTensor<float> wsGm;\n" + INIT_WS +
             "void P(){ DataCopyPad(wsGm[r * rowStride + c], ub, p); }\n")
    h, _ = hits(K_ROW, "rowStride_ = 500;\nuint64_t workspaceSize = batch * 500;\n")
    ck("★W6 负例: 跨距 500 (500x4=2000B, 非 32B 倍数) -> 必须报",
       "W6_row_stride_misalign" in h, True)
    h, _ = hits(K_ROW, "rowStride_ = AlignUp(cols, 8);\nuint64_t workspaceSize = batch * rowStride_;\n")
    ck("W6 正例: host 用 AlignUp 构造跨距 -> 不许报", "W6_row_stride_misalign" in h, False)
    h, _ = hits(K_ROW, "rowStride_ = 64;\nuint64_t workspaceSize = batch * 64;\n")
    ck("W6 正例: 跨距字面量本身是 8 元素的倍数 -> 不许报",
       "W6_row_stride_misalign" in h, False)

    #  ---- W0 (不是判据, 是**写入识别的作用域**) ----
    #  `kernel_gm_writes` 的注释里写着写入点包含 `buf.SetValue(`, 而正则只有
    #  DataCopyPad / DataCopy* / InitOutput —— **SetValue 没进去**。
    #  只用 SetValue 写 GM 的缓冲因此对 W1/W2/W3/W4 全部不可见。
    #  ★实测过影响面再决定怎么处理: 真实内核里 419 个文件在 GM 张量上用 SetValue
    #    (单个最多 180 次), 所以形态是常见的; 但把它并入识别后, 60 个真实算子对上的
    #    命中数 12 -> 12, **一条新发现都没有**。
    #    ⇒ 所以这里**只钉住现状**, 不擅自扩大判据的覆盖面 (本模块自己写着"判据过宽
    #      会让使用者学会忽略它"), 并把这条不一致留成一条会红的用例:
    #      哪天识别面真的改了, 这条会立刻提醒改注释或改用例, 不会静默漂。
    K_SETVAL = ("GlobalTensor<float> wsGm;\n" + INIT_WS +
                "void P(){ wsGm.SetValue(batch * n, 1.0f); }\n")
    _h, bufs = hits(K_SETVAL, "uint64_t workspaceSize = 96 * 1024 * 1024;\n")
    ck("★作用域: 只用 SetValue 写 GM 时**识别不到写入** (注释与实现不符, 已实测无新发现)",
       bufs.get("wsGm", {}).get("exprs"), [])

    print("\n=== wscap 判据自检 ===")
    for name, ok, got in cases:
        if not ok:
            bad += 1
        print(f"  {'过' if ok else '**未过**'} {name}" + ("" if ok else f"  (实得 {got})"))
    print(f"wscap 自检: {len(cases)} 条, {bad} 条未过")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", nargs="?", help="kernel .h")
    ap.add_argument("--host", help="host tiling .cpp (提供才能做 W1/W2 跨文件比对)")
    ap.add_argument("--symbols", default="", help="追加缩放符号 (逗号分隔), 用于非常规命名的算子")
    ap.add_argument("--scan", help="批量扫算子树 (目录下每个含 op_kernel/op_host 的算子)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="判据自检: 每条判据一个已知违规输入 + 一个正例防过宽")
    ap.add_argument("--refs", action="store_true",
                    help="打印本模块判据的出处/实证/最后验证日期 (含误报普查记录)")
    ap.add_argument("--op", default="",
                    help="算子名. 给了就自动从 aclnn 库抽入口校验的维度约束当契约 "
                         "(core.gen_spec entry_constraints), 不必手写 --assume")
    ap.add_argument("--cann", default=os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-8.5.0"))
    ap.add_argument("--assume", default="",
                    help="输入契约, 形如 'n%%32==0,curVecBaseN%%32==0'. 对齐类判据的可达性取决于它; "
                         "不给就按最坏情形报, 真命中会被不可达项埋掉 (gmfr: aclnn 校验 n mod 32 == 0)")
    a = ap.parse_args()
    if a.refs:
        print_refs("wscap", REFS)
        return 0
    if a.selftest:
        return selftest()
    vocab = SHAPE_SYMS + TILING_SYMS + [s for s in a.symbols.split(",") if s]
    asm = parse_assume(a.assume) if a.assume else {}
    if a.op:
        from core.gen_spec import entry_constraints
        ec = entry_constraints(a.op, a.cann)
        asm.update({k: v for k, v in ec["assume"].items() if k not in asm})
        if ec["assume"]:
            print("aclnn 入口契约 (自动抽): " +
                  ", ".join(f"{k}%{v}==0" for k, v in sorted(ec["assume"].items())) +
                  "  -- 对齐判据的可达性按它判\n")

    if a.scan:
        ops = sorted({os.path.dirname(d) for d in glob.glob(os.path.join(a.scan, "**", "op_kernel"), recursive=True)})
        rows, tot, unpaired = [], 0, []
        #  ★逐个报进度: 这一趟在 ops-nn 上要几分钟, 不打进度就与"卡死"完全无法区分
        #    (2026-09-17 实撞: 跑满 550 秒超时, 一行输出都没有)。走 stderr, 不污染表格。
        print(f"[scan] {len(ops)} 个候选算子目录", file=sys.stderr, flush=True)
        for _i, op in enumerate(ops, 1):
            if _i % 20 == 0 or _i == len(ops):
                print(f"[scan] {_i}/{len(ops)} ... {os.path.basename(op)}",
                      file=sys.stderr, flush=True)
            kp, hp = pair_of(op)
            if not kp or not hp:
                #  ★配不出 kernel/host 的算子目录要报出来: 抬头打了"N 个候选算子目录",
                #    收尾又打"N 个算子, M 个有命中", 两个 N 是同一个数, 而中间这批
                #    根本没扫 —— 没扫与扫过没问题在表里长得一样
                #    (2026-10-26, silentskip 扫出, 09 第 177 条)。
                unpaired.append(os.path.basename(op))
                continue
            #  批量扫描时逐算子取自己的 aclnn 契约: 目录名 snake_case 即算子名 CamelCase (CANN 惯例),
            #  故不需要人工逐个指定. 不带契约的话对齐类判据全按最坏情形报, 一棵树扫下来真命中会被埋掉.
            opname = "".join(w.capitalize() for w in os.path.basename(op).split("_"))
            asm_op = dict(asm)
            try:
                from core.gen_spec import entry_constraints
                asm_op.update({k: v for k, v in entry_constraints(opname, a.cann)["assume"].items()
                               if k not in asm_op})
            except Exception:
                pass
            try:
                hz, _, _ = analyze(kp, hp, vocab, assume=asm_op)
            except Exception as e:
                rows.append((os.path.basename(op), -1, f"ERROR {type(e).__name__}")); continue
            hard = [h for h in hz if h.get("severity") != "info"]
            tot += len(hard)
            if hz:
                kinds = ",".join(sorted({h["check"].split("_")[0] +
                                         ("" if h.get("severity") != "info" else "(info)") for h in hz}))
                rows.append((os.path.basename(op), len(hard), kinds))
        print(f"=== wscap --scan {a.scan}: {len(ops) - len(unpaired)} 个算子扫过, "
              f"{len(rows)} 个有命中, 硬失败 {tot} 条 (info 级不计) ===")
        if unpaired:
            print(f"  ★另有 {len(unpaired)} 个目录**配不出 kernel/host 对, 没扫**: "
                  f"{', '.join(unpaired[:8])}")
        for nm, cnt, kinds in rows:
            print(f"  {nm:46s} {cnt:3d}  {kinds}")
        if a.json:
            print(json.dumps(rows, ensure_ascii=False))
        return 0

    if not a.kernel:
        ap.error("需给 kernel .h, 或用 --scan <ops_root>")
    hz, bufs, caps = analyze(a.kernel, a.host, vocab, assume=asm)
    print(f"=== wscap: {a.kernel}" + (f"  +  {a.host}" if a.host else "  (未给 --host, 只能查 W3)") + " ===")
    print(f"GM 缓冲 {len([b for b,i in bufs.items() if i['exprs']])} 个有写入 | host 容量赋值 {len(caps)} 处 | 命中 {len(hz)} 条")
    for c in caps:
        print(f"  [容量] L{c['line']:<5d} {c['lhs']:<22s} 缩放符号={sorted(c['syms']) or '(无!)'}"
              + (f"  裸字面量={c['lits']}" if c["lits"] else ""))
    for b, i in sorted(bufs.items()):
        if i["exprs"]:
            print(f"  [写入] {b:<18s} {'workspace派生' if i['ws'] else '张量(不参与W1)'}"
                  f"  缩放符号={sorted(i['syms']) or '(常量)'}"
                  + ("  含GetValue索引" + ("(已有值域检查)" if i["ext_guarded"] else "(无检查!)")
                     if i["getvalue"] else ""))
    for h in hz:
        print(f"\n[{h['check']}] {h['buf']}")
        print(f"  {h['msg']}")
        for ln, frag in h.get("where", []):
            print(f"    kernel L{ln}: {frag}")
    if not hz:
        #  ★0 命中先问方言 (2026-10-21 SMLA): W1/W2/W3 都按 AscendC 的 GlobalTensor +
        #    GetValue 写法判; 源码里没有这些记号时, "无命中" 是没看见。
        if not _dialect.not_effective("wscap", [a.kernel, a.host], "AscendC 的 GlobalTensor / GetValue 写法"):
            print("\n无 W1/W2/W3 命中. 注: 静态启发式, 非完备; 运行期越界仍建议配合 workspace 尾部哨兵值检查.")
    if a.json:
        print(json.dumps(hz, ensure_ascii=False, default=str))
    hard = [h for h in hz if h.get("severity") != "info"]
    if hz and not hard:
        print("\n只有 info 级命中 (已有运行期校验补偿), 放行.")
    sys.exit(1 if hard else 0)


import sys

if __name__ == "__main__":
    #  ★必须把 main() 的返回码交出去。此前写成裸 `main()`, 于是
    #    `python -m core.<mod> --selftest` **无论自检过不过, 退出码恒为 0** ——
    #    而 opdbg 的判据普查正是按子进程退出码判 PASS 的, 于是真实的
    #    "N 条未过" 被报成 PASS。2026-10-28 在 detsync_lint 上实撞并全仓同改。
    _rc = main()
    sys.exit(_rc if isinstance(_rc, int) else 0)
