"""确定性/跨核同步 hazard 静态预检 (build 前, 料敌于先; dual-loop 沉淀自 v7 并发 bug).
不需构建/不需真机, 扫 kernel 源码, 报"会破坏确定性"的常见模式 —— 在烧 6min 构建+真机前拦下.
覆盖(源自 gmfr v7/v8 实撞):
  H1 跨核共享可见性: 某 GlobalTensor 在"单核守卫(GetBlockIdx()!=0 return / ==0)"内被**写**, 又在无守卫处被**读**,
     二者仅靠 SyncAll -> SyncAll 不保证跨核 GM 写可见 -> race. (v7 的 scolGm/invScolGm 正是此模式)
     修法: 每核冗余算写自己分片 / 或写后显式 drain + 消费前 flag, 而非裸 SyncAll.
  H3 UB越界(整维载入分块缓冲): 某UB载入随整维(n/m)增长的字节, 而缓冲容量不随该整维缩放(baseN/ubCalSize/常量) -> 越界踩踏 -> 非确定.
     --shape 给具体值时算出精确翻转点(如 scale bf16 载入1024B容量: n>512 越界). 修: 按 baseN/ubCalSize 分块载.
  H6 跨核标志计数不平衡(提示级): CrossCoreSetFlag 与 CrossCoreWaitFlag 处于不同循环控制下 -> 两侧次数相等
     依赖隐式不变量(如 "Iterate() 次数 == curSingleN/baseN"), 改 tiling 的 single*/base*/step* 即打破,
     症状是真机 aicore timeout 507014 (核挂死不报错). 沉淀自 gmfr: singleM=2*baseM 使标志数 2:等待数 1.
     命中不等于有 bug (隐式不变量本身很常见), 但改 tiling 前必须复核这两个循环次数是否仍相等.
  H32 C220显式Cube/AIV分支的同通道同循环从raw GetBlockIdx起步: 提示复核子核编号到逻辑Cube核号的映射。
     仅直接局部赋值模式，不解析helper/完整预处理/调度，不据无命中证明安全，不把合法lane调度判为错误。
  H2 确定性路径用浮点原子加: det 路径(deterministicFlag==1)出现 SetAtomicAdd<float/half/bfloat> = 时序相关 = 不可复现.
     (int32/int 原子加才顺序无关; 快路径 float 原子加是预期的非确定, 不算)
用法: python -m core.detsync_lint <kernel.h> [--verbose]
"""
from __future__ import annotations
import re, sys, os, argparse

from ._refs import Ref, print_refs
from . import _dialect

def strip_comments(src):
    """去 // 与 /* */ 注释, 注释字符换成空格 (保留换行/字符位置 -> 行号不变). 防注释里的模式假触发 H4/H5 等."""
    out = []; i = 0; n = len(src)
    while i < n:
        if src[i:i + 2] == '//':
            j = src.find('\n', i); j = n if j < 0 else j
            out.append(' ' * (j - i)); i = j
        elif src[i:i + 2] == '/*':
            j = src.find('*/', i); j = n if j < 0 else j + 2
            out.append(''.join('\n' if c == '\n' else ' ' for c in src[i:j])); i = j
        else:
            out.append(src[i]); i += 1
    return ''.join(out)

def split_funcs(src):
    """按 __aicore__ ... ::Name( 切成函数体 (粗粒度, 够用)."""
    marks = [(m.start(), m.group(1)) for m in re.finditer(r'::(\w+)\s*\(', src)]
    funcs = []
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(src)
        funcs.append((name, src[pos:end]))
    return funcs

def is_single_core(body):
    """函数体是否有单核守卫 (只核0/单核执行)."""
    return bool(re.search(r'GetBlockIdx\(\)\s*!=\s*0\s*\)\s*\{?\s*return', body) or
                re.search(r'if\s*\(\s*GetBlockIdx\(\)\s*==\s*0\s*\)', body))

def gm_tensors(src):
    return set(re.findall(r'GlobalTensor<[^>]+>\s+(\w+)\s*;', src))

# ---- H3: UB 缓冲越界 (整维载入分块缓冲 -> OOB -> 踩踏邻接缓冲 -> run-to-run 非确定) ----
# 沉淀自 gmfr v7/v8: ComputeScolVec 把整宽 [n] scale 载入 baseN 宽的 scaleInQueue, n>512 时 UB 越界.
# 判据(机械, 无需构建): 某 UB 载入的传输字节 = f(整维 n/m/k/batch), 而目标缓冲容量表达式**不含该整维**
# (只含 baseN/ubCalSize/常量) -> 容量不随整维缩放 -> 整维一大必越界. 与 detrace 尺寸扫描的翻转信号互为印证.
FULL_DIMS = ["totalM", "batch", "n", "m", "k"]   # 随问题规模增长的整维 (载入这些整宽=smell)
_SIZEOF = {"float": 4, "int32_t": 4, "int32": 4, "uint32_t": 4, "float32": 4,
           "bfloat16_t": 2, "half": 2, "float16": 2, "int16_t": 2,
           "int8_t": 1, "uint8_t": 1, "int64_t": 8, "uint64_t": 8}

def _norm(expr):
    """规范化 C++ 尺寸表达式: 去 tiling->/static_cast/typename P::/matmulTiling. 前缀与空格."""
    e = expr
    e = re.sub(r'static_cast<[^>]+>\s*\(', '(', e)
    e = e.replace("tiling->", "").replace("typename P::", "").replace("matmulTiling.", "")
    return e.strip()

def _full_dims_in(expr):
    """表达式里出现的整维符号 (作为独立标识符)."""
    e = _norm(expr)
    return [d for d in FULL_DIMS if re.search(rf'(?<![A-Za-z_]){d}(?![A-Za-z0-9_])', e)]

def _eval_bytes(expr, vals, scale_b, out_b):
    """按具体 shape 求值字节数 (失败返回 None). sizeof(TYPE)->字节, 符号->vals."""
    e = _norm(expr)
    smap = dict(_SIZEOF); smap["SCALE_TYPE"] = scale_b; smap["DTYPE_OUT"] = out_b
    smap["cT::T"] = 4; smap["cT::T "] = 4
    e = re.sub(r'sizeof\(\s*([\w:]+)\s*\)', lambda m: str(smap.get(m.group(1).strip(), "None")), e)
    for k, v in vals.items():
        e = re.sub(rf'(?<![A-Za-z_]){k}(?![A-Za-z0-9_])', str(v), e)
    if re.search(r'[A-Za-z_]', e) or "None" in e:
        return None   # 仍有未解析符号
    try:
        return int(eval(e, {"__builtins__": {}}))
    except Exception:
        return None

#  ---- H25: DMA **传输粒度** 太碎 --------------------------------------------------
#  沉淀动机 (2026-09-18, sfag 复盘 + dubin 指出"这么多没建好的, 为啥不建"):
#    sfag 的 K 与 K_rope 是两个独立 GM 张量, 按选中 token 分两次 gather:
#      intriParamsRope.blockLen = selectedBlockSize * dimRope * sizeof(T1) = 1*64*2 = **128 B**
#    selectedBlockSize=1 (逐 token 选择), 于是**每个选中 token 一次 128 字节的随机 DMA**,
#    另加一次 1024 B 的。而 UB 那侧早就按连续布局准备好了
#    (`dimDAlign = (dimD + dimRope + ...)` = 576 = nope+rope), 缺的连续性在 **GM 那一端**。
#    把两者按 token 打包成 576 连续元素, 两次 DMA (1024+128) 变成一次 1152 B。
#  ★我当时把 `aiv_mte2 +5854us` 记成"**局部性假设**", 还准备设计一个零构建的缓存实验去验证它
#    —— 而真实原因是源码里写着的一个常量 128。**一次 grep 的事, 我要去做实验。**
#  ★与 H9 的分工: H9 管**原子写**的长度颗粒 (32B 对齐, 违反=静默丢更新);
#    本条管**任意 DataCopy 的传输粒度** (不影响正确性, 只影响带宽), 所以级别是提示不是错误。
#  ★判据只给上界与线索, **不给处方**: 粒度小可能是数据布局决定的, 打包与否要看生产者那一端
#    能不能改 (见 expdesign E3: 要学/改数据布局去读**写它**的那一端)。
_DMA_EFFICIENT_B = 512      # 低于此值按"碎"报; 512B 的依据见 REFS


#  ---- H26: 跨核汇合的**粒度**比它真正要保证的东西粗 -------------------------------
#  沉淀动机 (2026-09-18, sfag 复盘): 确定性路径每个 s1 行做一次**全体 AIV 汇合**
#  (CrossCoreSetFlag/WaitFlag), 1024 次。而逐位一致真正要求的只是
#  **同一个地址上的累加次序固定** —— 全局时序是一个**严格更强**的条件, 它顺带保证了
#  逐地址有序, 代价是让输出地址根本不相交、永远不会碰面的核也互相等待。
#  ★我在这条线上做了两轮 (g8/g9, +3.05%), **两轮都只在挪等待的位置**, 一次也没问过
#    "该不该是全体、该不该每轮一次"。然后在结论行写下"屏障线已榨到头" ——
#    那句话只在**放置**这一层成立, 限定语在从工作笔记进入结论时丢了。
#  ★与 H21 的分工: H21 管**核内**屏障与它保护的 UB 槽的粒度错配; 本条管**跨核**汇合。
#    仓里核内那侧判据齐备, 跨核这侧此前完全空白 —— 而昂贵的那个在跨核。
#  ★判据给线索不给处方: 参与者集合能静态看出来 (全体), **真实冲突集合看不出来**
#    (取决于运行期索引)。所以本条是**待核对项**, 不是错误。
# ======================================================================
#  share —— 每核共享状态 x 占用方 x 重叠窗口 的体检表 (H30/H31)
#
#  来由 (vc_attention 2026-10-28, 代价六次构建): 一个核里两个 MatmulImpl 的竞态,
#  三次"切开某一样共享物"的尝试全部不成立 ——
#    isA2B2Shared 切开 L0A/L0B      -> 仍 100%
#    再关 enUnitFlag (换 fixpipe 路径) -> 仍 100%
#    只下 M->MTE1 跨流水序           -> 53.2%
#    再加 FIX->S                    -> 34.0%
#    PipeBarrier<PIPE_ALL>          -> 0%
#  缺的不是某一条边, 而是**共享物太多, 人列不全**。所以要的不是又一条判据,
#  是一张把"谁占了什么、两个发射点之间实际有哪些序"摊开的表。
# ======================================================================

_SH_MM = re.compile(r"\b(?:matmul::)?(?:Matmul|MatmulImpl)\s*<([^;{}]*?)>\s*([A-Za-z_]\w*)\s*;")
_SH_REGIST = re.compile(r"\bREGIST_MATMUL_OBJ\s*\(([^;]*?)\)\s*;", re.S)
_SH_A2B2 = re.compile(r"\bisA2B2Shared\b\s*=\s*true")
#  ★`MatmulImpl<...>` 是**内部**类型, 写它就是明确绕开 KFC 客户端, 两个实例直接驱动
#    本核的 cube; `Matmul<...>` / `matmul::Matmul<...>` 是**公开**类型, 在混合核下
#    解析成客户端, cube 由服务端按消息执行。这条区分决定 H30 该报错误级还是提示级 ——
#    不区分的话, 在 CANN 发行语料的 57 个多实例内核上会报 38 条, 其中 36 条是误报。
_SH_IMPL_DECL = re.compile(r"\bMatmulImpl\s*<[^;{}]*?>\s*[A-Za-z_]\w*\s*;")
#  ★TSCM<TPosition::GM, 2> 的首个模板实参是**源**位置, TSCM 自己在 L1 ——
#    按首参归类会把它错报成 GM (本工具首次上真机就撞了这个)。故单独捕获容器名。
_SH_TBUF = re.compile(r"\bT(Buf|Que|QueBind|SCM)\s*<\s*(?:TPosition::)?(\w+)[^;>]*>\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*(?:\[|;)")
_SH_FETCH = re.compile(r"\b(\w+)\s*=\s*[^;]*?\bFetchEventID\s*(?:<\s*HardEvent::(\w+)\s*>\s*\(\s*\)|\(\s*HardEvent::(\w+)\s*\))")
_SH_ALLOC = re.compile(r"\bAllocEventID\s*<\s*HardEvent::(\w+)\s*>")
_SH_ISSUE = re.compile(r"\b([A-Za-z_]\w*)\s*\.\s*(?:template\s+)?(IterateAll|Iterate)\s*<?")
_SH_BARRIER = re.compile(r"\bPipeBarrier\s*<\s*(\w+)\s*>")
_SH_SETWAIT = re.compile(r"\b(?:Set|Wait)Flag\s*<\s*HardEvent::(\w+)\s*>")
_SH_CC = re.compile(r"\bCrossCore(?:Set|Wait)Flag\s*(?:<[^>]*>)?\s*\(\s*([A-Za-z_]\w*)")

_POS2HW = {"A1": "L1", "B1": "L1", "TSCM": "L1", "C1": "L1",
           "A2": "L0A", "B2": "L0B", "CO1": "L0C", "CO2": "UB",
           "VECIN": "UB", "VECOUT": "UB", "VECCALC": "UB", "LCM": "UB"}


def _sh_owners(src):
    """返回 [(实例名, 模板实参原文)], 认成员声明与 REGIST_MATMUL_OBJ 两种写法。"""
    out = {}
    for m in _SH_MM.finditer(src):
        out[m.group(2)] = m.group(1)
    for m in _SH_REGIST.finditer(src):
        args = [a.strip() for a in m.group(1).split(",")]
        for a in args[2::2]:
            if re.fullmatch(r"[A-Za-z_]\w*", a or "") and a not in out:
                out[a] = "(REGIST_MATMUL_OBJ)"
    return out


def share_report(src: str) -> dict:
    """把每核共享状态与发射点之间的序整理成结构化结果。纯函数。"""
    clean = strip_comments(src)
    lines = clean.split("\n")
    owners = _sh_owners(clean)
    shared = bool(_SH_A2B2.search(clean))

    # --- 内核自己声明的缓冲, 按硬件归类 ---
    bufs = {}
    for m in _SH_TBUF.finditer(clean):
        hw = "L1" if m.group(1) == "SCM" else _POS2HW.get(m.group(2), m.group(2))
        for nm in re.split(r"\s*,\s*", m.group(3)):
            bufs.setdefault(hw, []).append(nm)

    # --- 事件号: FetchEventID 只查不占, AllocEventID 才占 ---
    fetched, alloced = [], sorted({m.group(1) for m in _SH_ALLOC.finditer(clean)})
    for m in _SH_FETCH.finditer(clean):
        fetched.append((m.group(1), m.group(2) or m.group(3)))
    raw_events = sorted({m.group(1) for m in _SH_SETWAIT.finditer(clean)})
    cc_flags = sorted({m.group(1) for m in _SH_CC.finditer(clean)})

    # --- 发射点序列 ---
    issues = []
    for i, ln in enumerate(lines, 1):
        for m in _SH_ISSUE.finditer(ln):
            if m.group(1) in owners:
                issues.append((i, m.group(1)))

    # --- 相邻且属主不同的发射点之间, 实际存在哪些序 ---
    windows = []
    pairs = list(zip(issues, issues[1:]))
    #  ★发射点在循环体里时, **末个 -> 下一轮首个**也是一个重叠窗口, 而且方向相反。
    #    漏掉它等于只堵一边 —— vc_attention 上那个方向的重叠恰是软件流水的收益所在
    #    (两边都堵实测 +24%), 所以必须报出来由人判, 不能静默省略。
    wrap = None
    if len(issues) >= 2 and issues[-1][1] != issues[0][1]:
        #  ★回边的那段文本必须按**顶层构造边界**截断。首版把"末个发射点之后的全部行
        #    + 首个发射点之前的全部行"都算进去, 于是把另一个类 (AIV 那侧) 的事件
        #    全混了进来 —— 一份要给人读的表这样就没用了。顶层构造以行首非空白起始。
        fwd = []
        for ln in lines[issues[-1][0]:]:
            if ln[:1].strip():
                break
            fwd.append(ln)
        bwd, start = [], 0
        for i in range(issues[0][0] - 2, -1, -1):
            if lines[i][:1].strip():
                start = i + 1
                break
        bwd = lines[start:issues[0][0] - 1]
        body = "\n".join(fwd) + "\n" + "\n".join(bwd)
        wrap = ((issues[-1][0], issues[-1][1]), (issues[0][0], issues[0][1]), body)
        pairs.append((wrap[0], wrap[1]))
    for a, b in pairs:
        if a[1] == b[1]:
            continue
        if wrap and a is wrap[0] and b is wrap[1]:
            seg, back = wrap[2], True
        else:
            seg, back = "\n".join(lines[a[0]:b[0] - 1]), False
        windows.append(dict(
            frm=a, to=b, back_edge=back,
            barriers=sorted({m.group(1) for m in _SH_BARRIER.finditer(seg)}),
            events=sorted({m.group(1) for m in _SH_SETWAIT.finditer(seg)}),
            crosscore=sorted({m.group(1) for m in _SH_CC.finditer(seg)}),
            ends=sorted({m.group(1) for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\.\s*End\s*\(", seg)
                         if m.group(1) in owners}),
        ))
    return dict(owners=owners, a2b2_shared=shared, bufs=bufs,
                fetched=fetched, alloced=alloced, raw_events=raw_events,
                cc_flags=cc_flags, issues=issues, windows=windows)


def check_share(src: str) -> list:
    """H30: 两个 Matmul 实例的发射点之间没有全屏障。H31: 事件号靠 FetchEventID 取。"""
    r = share_report(src)
    hz = []
    if len(r["owners"]) >= 2:
        clean = strip_comments(src)
        #  显式本地路径: 声明用内部类型 MatmulImpl, 且本文件没有 REGIST_MATMUL_OBJ
        explicit_local = bool(_SH_IMPL_DECL.search(clean)) and "REGIST_MATMUL_OBJ" not in clean
        for w in r["windows"]:
            if "PIPE_ALL" in w["barriers"]:
                continue
            res = ["L0A/L0B (每个实例都 InitBuffer 整块" +
                   ("; 已开 isA2B2Shared 只切开了它" if r["a2b2_shared"] else "; 未开 isA2B2Shared") + ")",
                   "fixpipe 的 dtype/layout 配置 (每核寄存器)",
                   "M_MTE1 / MTE1_M 的事件号"]
            have = []
            if w["ends"]:
                have.append("`%s.End()`" % "`, `".join(w["ends"]))
            if w["barriers"]:
                have.append("PipeBarrier<%s>" % ">, PipeBarrier<".join(w["barriers"]))
            if w["events"]:
                have.append("SetFlag/WaitFlag: " + ", ".join(w["events"]))
            if w["crosscore"]:
                have.append("CrossCoreFlag: " + ", ".join(w["crosscore"]))
            hz.append(dict(
                check="H30_matmul_issue_overlap",
                level="错误" if explicit_local else "提示",
                flag="%s@L%d -> %s@L%d%s" % (w["frm"][1], w["frm"][0], w["to"][1], w["to"][0],
                                             " (循环回边)" if w.get("back_edge") else ""),
                msg=("两个 Matmul 实例的发射点之间**没有 PipeBarrier<PIPE_ALL>**。"
                     "这段窗口里实际存在的序: " + ("; ".join(have) if have else "**一条都没有**") + "。"
                     "而这两方共享的每核状态至少有: " + "; ".join(res) + "。"
                     "★`End()` 只做 FIX_M (挡 M 流水), 挡不住 MTE1 与标量; "
                     "单条跨流水序也不够 —— vc_attention 实测阶梯: 无序 100%, "
                     "只下 M->MTE1 53.2%, 再加 FIX->S 34.0%, 全屏障 0% (48 趟/格)。"
                     "症状是**间歇**的数值不确定, 触发率只随时序变化, "
                     "消费侧越快命中率越高 —— 跑多少用例全过都不构成没有它的证据。"
                     + ("★本文件用**内部类型** `MatmulImpl` 声明且没有 REGIST_MATMUL_OBJ, "
                        "即明确绕开 KFC 客户端、两个实例直接驱动本核的 cube。"
                        "CANN 发行语料里 57 个多实例内核, 这样写的只有 1 个 "
                        "(mat_mul_v3 的 deterministic splitk), 且它的流程里带 PipeBarrier<PIPE_ALL> —— "
                        "**这个配置几乎没有先例**。"
                        if explicit_local else
                        "★本文件用**公开类型** `Matmul` 声明 (或已有 REGIST_MATMUL_OBJ), "
                        "混合核下它解析成 KFC 客户端, cube 由服务端按消息执行, 不是本地并发, "
                        "故只给提示级。要确认: 看整个编译单元有没有 REGIST_MATMUL_OBJ —— "
                        "★**要找到父目录**: 按架构分的子目录 (arch32 / arch35) 里只放实现头, "
                        "注册写在上一层的 .cpp 里; 只搜同目录会把它们误判成「没有注册」。"
                        "再看编译模式是不是混合核。若确属本地路径, 按错误级处理。"
                        "★信噪比已标定 (2026-10-28, CANN 发行语料 57 个多实例内核): "
                        "38 个有未加全屏障的窗口, 其中 2 个是显式本地 MatmulImpl (错误级, 同一文件两处路径), "
                        "**36 个全部是 KFC 路径, 逐个在编译单元内找到了注册, 无漏网的错误级**。"))))
    for name, evt in r["fetched"]:
        hz.append(dict(
            check="H31_event_id_not_reserved", level="提示", flag="%s=FetchEventID(%s)" % (name, evt),
            msg=("`FetchEventID` 返回 `sff0(eventOccupy)` 却**不置位** —— 它只查不占, "
                 "所以同一事件类型的**所有调用方拿到的是同一个号** (通常是 0), "
                 "包括算子库内部那些。Set/Wait 紧邻时无害; 一旦这对 Set/Wait 之间还夹着"
                 "别人的同类事件, 就会互相消耗额度。要独占得用 `AllocEventID<HardEvent::%s>()` "
                 "(它会 sbitset1 占号), 用完 `ReleaseEventID`。" % evt)))
    return hz


def print_share(src: str) -> None:
    r = share_report(src)
    print("=== 每核共享状态 x 占用方 ===")
    ow = ", ".join(sorted(r["owners"])) or "(无)"
    if len(r["owners"]) >= 2:
        tag = "已开 isA2B2Shared (只切开 L0 缓冲)" if r["a2b2_shared"] else "★未开 isA2B2Shared"
        print(f"  [L0A/L0B]      {ow}    每个实例都 InitBuffer 整块; {tag}")
        print(f"  [L0C/CO1]      {ow}    按 tiling 大小顺序分配, 区间不重叠")
        print(f"  [fixpipe 配置]  {ow}    dtype/layout/quant 是**每核寄存器**")
        for nm, targs in sorted(r["owners"].items()):
            print(f"       {nm}: <{targs.strip()[:72]}>")
    elif r["owners"]:
        print(f"  [Cube]         {ow}    单一实例, 无实例间共享")
    for hw, names in sorted(r["bufs"].items()):
        print(f"  [{hw:<12}] 内核: {', '.join(sorted(set(names)))}")
    if r["alloced"]:
        print(f"  [事件号-已占]   AllocEventID: {', '.join(r['alloced'])}")
    for nm, evt in r["fetched"]:
        print(f"  [事件号-未占]   {nm} = FetchEventID({evt})   ★只查不占, 与库内同号")
    if r["cc_flags"]:
        print(f"  [跨核 flag]     {', '.join(r['cc_flags'])}")
    print()
    print("=== 发射点与它们之间实际存在的序 ===")
    if not r["issues"]:
        print("  (没找到 Matmul 的发射点)")
    for w in r["windows"]:
        tag = "  (循环回边: 下一轮)" if w.get("back_edge") else ""
        print(f"  L{w['frm'][0]} {w['frm'][1]} -> L{w['to'][0]} {w['to'][1]}{tag}")
        bits = []
        if w["ends"]:      bits.append("End(): " + ", ".join(w["ends"]))
        if w["barriers"]:  bits.append("PipeBarrier: " + ", ".join(w["barriers"]))
        if w["events"]:    bits.append("事件序: " + ", ".join(w["events"]))
        if w["crosscore"]: bits.append("跨核: " + ", ".join(w["crosscore"]))
        print("      " + ("; ".join(bits) if bits else "**一条序都没有**"))
        print("      " + ("✓ 有 PIPE_ALL 全屏障" if "PIPE_ALL" in w["barriers"]
                          else "★无全屏障 —— 上面列的共享状态在这段窗口里都可能被两方同时碰"))


def check_crosscore_barrier_granularity(src):
    hz = []
    for name, body in split_funcs(src):
        if "CrossCoreSetFlag" not in body and "CrossCoreWaitFlag" not in body:
            continue
        atomic = bool(re.search(r"SetAtomicAdd\s*<", body))
        if not atomic:
            continue
        #  汇合在循环体内 = 每轮一次。
        #  ★必须真的做大括号配对: 第一版用"从 for( 向后 4000 字符"近似, 于是**循环外**的
        #    汇合也落进窗口而误报。自检里那条"汇合在循环外不该报"当场抓住 (2026-09-18)。
        in_loop = False
        for lm in re.finditer(r"\bfor\s*\(|\bwhile\s*\(", body):
            ob = body.find("{", lm.end())
            if ob < 0:
                continue
            depth, close = 0, -1
            for i in range(ob, len(body)):
                if body[i] == "{":
                    depth += 1
                elif body[i] == "}":
                    depth -= 1
                    if depth == 0:
                        close = i
                        break
            if close < 0:
                continue
            if re.search(r"CrossCore(Set|Wait)Flag", body[ob:close]):
                in_loop = True
                break
        if not in_loop:
            continue
        ln = src.find(body[:60])
        ln = src[:ln].count(chr(10)) + 1 if ln >= 0 else 0
        hz.append(dict(check="H26_crosscore_barrier_granularity", level="提示",
                       line=ln, buffer=name,
                       err=(f"`{name}` 在循环里做**跨核汇合**且同一函数内有浮点原子加 —— "
                            f"这是「每轮让全体核互等」的形态。**这是待核对项, 不是错误。**\n"
                            f"    要核对的是: 逐位一致真正需要的是**同一个地址上的累加次序固定**, "
                            f"而全体汇合是一个**严格更强**的条件。\n"
                            f"    问三个问题:\n"
                            f"      1) 参与汇合的核里, 有多少对真的会写到**同一个**输出地址?\n"
                            f"      2) 若输出地址由每核独占的索引段决定, 这个汇合**整个不需要** "
                            f"(单 owner 独占的累加不受结合律限制, 见 H2 末句)。\n"
                            f"      3) 若确有冲突, 能不能只在**真冲突的核之间**定序 (链式/令牌), "
                            f"而不是全体停下?\n"
                            f"    ★sfag 2026-09-18 实撞: 每个 s1 行一次全体 AIV 汇合, 共 1024 次。"
                            f"我在这条线上做了两轮改动, **两轮都只在挪等待的位置**, "
                            f"一次也没问过该不该是全体 —— 然后写下了「屏障线已榨到头」。\n"
                            f"    ★真实冲突集合静态看不出来 (取决于运行期索引), 所以判据只能提问。"
                            f"要定量就去看 core.hwfloor 超额表里散射那一行的倍数。")))
    return hz



#  ---- H28: 全核屏障之前有**提前返回** —— 死锁 -------------------------------------
#  SyncAll() / 全核汇合要求**每一个**核都到达。函数里任何一条在屏障之前的 `return`
#  都会让那个核缺席, 其余核无限等待 —— 表现为真机挂死、无错误码、msprof 采不到,
#  与 hangdiag 查的那一类症状一模一样, 但成因在单文件内, 静态就能看出来。
#
#  典型来源: 把「本核没活干」写成提前返回。它在没有屏障时完全正确, 加屏障那一刻起
#  就变成死锁 —— 而加屏障的人通常只看自己新加的那几行。
#  正解: 把「有没有活」变成一个标志, 空闲核照样走到屏障:
#      const bool active = (bi < nTask) && (a0 < axis);
#      if (active) { ...干活... }
#      AscendC::SyncAll();
#      if (active) { ...接着干... }
#
#  出处: cummin 2026-09-20 把两阶段扫描从「两次 kernel launch」改成「单 kernel +
#  SyncAll」时, 原函数开头两处 early return 正是这个形态, 改屏障前先拆掉了它们。
BARRIER_CALLS = ("SyncAll", "CrossCoreWaitFlag", "SyncBlock")


def check_early_return_before_barrier(src: str) -> list:
    import re as _re
    hz = []
    lines = src.splitlines()
    # 切出每个 __aicore__ / __global__ 函数的行区间
    fpat = _re.compile(r"(?:__aicore__|__global__)[^;{}]*?\b([A-Za-z_]\w*)\s*\([^;{}]*?\)\s*\{")
    for m in fpat.finditer(src):
        name = m.group(1)
        i = src.index("{", m.end() - 1)
        depth, j = 0, i
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        seg = src[i:j]
        base_ln = src[:i].count("\n") + 1
        bpos = [seg.index(b) for b in BARRIER_CALLS if b in seg]
        if not bpos:
            continue
        first_barrier = min(bpos)
        rets = [mm.start() for mm in _re.finditer(r"\breturn\s*;", seg[:first_barrier])]
        if not rets:
            continue
        bline = base_ln + seg[:first_barrier].count("\n")
        for r in rets:
            ln = base_ln + seg[:r].count("\n")
            hz.append(dict(check="H28_early_return_before_barrier", level="严重", line=ln,
                           msg=(f"函数 {name} 在第 {bline} 行的全核屏障**之前**有 `return;` "
                                f"(第 {ln} 行) —— 走这条 return 的核不会到达屏障, 其余核无限等待。"
                                f"把「本核没活干」改成标志: `const bool active = ...;` "
                                f"`if (active) {{...}}` 屏障 `if (active) {{...}}`")))
    return hz

#  ---- H27: 核间划分的**余数**整块压在一个核上 -------------------------------------
#  沉淀动机 (2026-09-18, sfag g15 实测 +1.91%):
#    `firstCoreKSize = total / nCores;`  (向下取整)
#    `cur = (idx == nCores-1) ? (total % nCores + firstCoreKSize) : firstCoreKSize;`
#    本 shape: 2048/48 = 42 余 32 -> 47 个核各 42 行, **一个核 74 行**, max/mean = 1.735。
#    改成 `first + (idx < rem ? 1 : 0)` 后 max/mean = 1.008, 实测 34.790 -> 34.125 ms。
#  ★与 mmplan W7 (波量化) 的分工: W7 建模的是 **ceil 均分**的形态
#    (max/mean = ceil(tiles/cores)*cores/tiles), 对本内核算出 1.008 —— 而实际是 1.735。
#    W7 是参数化的 planner, 不看源码; 本条从**源码**认出是哪种划分形态。两者互补, 不重复。
#  ★为什么它值得单列一条判据: g15 是本轮**唯一靠通读源码找到**的量化候选,
#    `whatfor 负载均衡` / `whatfor 余数` 当时零命中 —— 仓里没有任何判据在看这件事。
#  ★级别提示: 余数压在末核不影响正确性, 只影响并行效率; 且是否值得改要看该阶段
#    有没有跨核会合 (有会合 -> 长尾无法跨轮摊平, 值得改; 无会合 -> 可能被下一轮盖住)。
def check_core_partition_remainder(src, vals=None, scale_b=2, out_b=2):
    vals = vals or {}
    hz = []
    #  first = A / B;  ... (idx == B - 1) ? (A % B + first) : first
    pat = re.compile(
        r"(\w+)\s*=\s*([A-Za-z_]\w*)\s*/\s*([A-Za-z_]\w*)\s*;"          # first = A / B;
        r"(?:.|\n){0,400}?"
        r"\?\s*\(\s*\2\s*%\s*\3\s*\+\s*\1\s*\)", re.M)
    for m in pat.finditer(src):
        first, A, B = m.group(1), m.group(2), m.group(3)
        ln = src[:m.start()].count(chr(10)) + 1
        na, nb = vals.get(A), vals.get(B)
        quant = ""
        if isinstance(na, (int, float)) and isinstance(nb, (int, float)) and nb:
            fl = int(na // nb); rem = int(na % nb)
            if rem == 0:
                continue                      # 整除时没有余数问题, 不报
            mx = fl + rem; mean = na / nb
            quant = (f"\n    定量: {A}={int(na)} / {B}={int(nb)} = {fl} 余 {rem} -> "
                     f"{int(nb)-1} 个核各 {fl}, **一个核 {mx}**, max/mean = **{mx/mean:.3f}**"
                     f" (均摊后 {(fl+1)/mean:.3f})")
        hz.append(dict(check="H27_core_partition_remainder", level="提示", line=ln,
                       buffer=first,
                       err=(f"`{first} = {A} / {B}` 向下取整, 而余数 `{A} % {B}` **整块给了最后一个核**。"
                            f"{quant}\n"
                            f"    改法: `{first} + (idx < {A} % {B} ? 1 : 0)`, 起始偏移相应改成 "
                            f"`idx * {first} + Min(idx, {A} % {B})`。\n"
                            f"    ★**起始偏移有几处就要改几处** —— sfag 2026-09-18 共 5 处, "
                            f"最初只改了 3 处, 靠 grep 才抓到 isSmallS2 分支里的两处。\n"
                            f"    ★值不值得改要看这一段有没有**跨核会合**: 有会合则长尾无法跨轮摊平, "
                            f"值得改 (sfag 实测 +1.91%); 没有会合则可能被下一轮盖住。\n"
                            f"    ★与 mmplan W7 的分工: W7 建模 **ceil 均分**的波量化, 对本形态算出 "
                            f"1.008 而实际 1.735 —— 形态不同, 模型不通用。")))
    return hz


def check_dma_granularity(src, vals=None, scale_b=2, out_b=2):
    """扫 DataCopyExtParams 的 blockLen, 报**每次搬得太碎**的那些。

    解不出具体字节时**必须说出来** —— "没查成"与"查过没问题"长得一样。
    """
    vals = vals or {}
    hz = []
    #  两种写法都要认 —— 第一版只认大括号初始化, 而真实代码里绝大多数是**逐字段赋值**,
    #  于是承载结论的那一条 (intriParamsRope.blockLen = selectedBlockSize*dimRope*sizeof(T1))
    #  一条都没报出来。样本外检验当场抓住 (2026-09-18)。
    #  ★blockLen 不等于一次搬运的长度 —— 要连 blockCount 与 stride 一起看。
    #    2026-09-18 实撞 (样本外, sfag 真源码): `outParamRope.blockLen = 128` 被报成"太碎",
    #    而同一处 `blockCount = curGatherSize(64)` 且 **srcStride = dstStride = 0** ——
    #    那是**一次连续的 8 KB 搬运**, 不是 64 次 128 B。判据只看 blockLen 就是假阳性。
    #    真正碎的是读侧: intriParamsRope 的 srcStride 非零 (按 kv token 跨距), 那才是
    #    128 B 的散读。**修假阴性不能换假阳性** —— 这一条差点让我去改一个本来就不碎的地方。
    #  ★**捕获跨距的值再判, 不用前瞻断言**。第一版写成 `=\s*(?!0\s*;)`, 而 `\s*` 会
    #    回溯到零宽, 于是前瞻看到的是空格不是 '0', 断言反而成立 —— 判据看起来还在报,
    #    而"修好了"其实一行没生效。这类失效最隐蔽: 输出没变, 容易以为是判据本来就该报。
    stride0 = set()
    for nm in set(re.findall(r"(\w+)\s*\.\s*blockLen\s*=", src)):
        vals_ = re.findall(rf"{re.escape(nm)}\s*\.\s*(?:src|dst)Stride\s*=\s*([^;]+);", src)
        if vals_ and all(v.strip() == "0" for v in vals_):
            stride0.add(nm)
    cands = []
    for m in re.finditer(r"DataCopyExtParams\s+(\w+)\s*\{\s*([^,]+),\s*([^,{}]+)[,}]", src):
        cands.append((m.group(1), m.group(3).strip(), m.start()))
    for m in re.finditer(r"(\w+)\s*\.\s*blockLen\s*=\s*([^;]+);", src):
        cands.append((m.group(1), m.group(2).strip(), m.start()))
    #  同一个 params 在不同分支会被赋多次 (sfag 的 intriParamsRope 有三处), 按
    #  (名字, 表达式) 去重而不是只按名字 —— 只按名字会漏掉别的分支里更碎的那一次。
    #  局部标量别名先展开 (selectedBlockSizeDrope = selectedBlockSize * dimRope),
    #  否则表达式解不出, 本条只会报"算不出"而不是报出真实字节数。
    alias = {}
    for m in re.finditer(r"\b(?:u?int\d*_t|int|size_t)\s+(\w+)\s*=\s*([^;=]+);", src):
        alias.setdefault(m.group(1), m.group(2).strip())
    seen = {}
    for name, blk_len, pos in cands:
        key = (name, re.sub(r"\s+", "", blk_len))
        if key in seen:
            continue
        seen[key] = True
        ln = src[:pos].count(chr(10)) + 1
        expanded = blk_len
        for _ in range(3):
            for k, v in alias.items():
                expanded = re.sub(r"(?<![A-Za-z_])" + re.escape(k) + r"(?![A-Za-z0-9_])",
                                  "(" + v + ")", expanded)
        #  模板类型的宽度 (sizeof(T1) 这类) 解不出是最常见的卡点 —— 允许用
        #  --shape 传 sizeof_T1=2 显式给出, 而**不猜默认值**: 猜错会把 128B 算成 256B,
        #  正好跨过阈值, 把该报的漏掉。
        for _m in re.finditer(r"sizeof\(\s*([\w:]+)\s*\)", expanded):
            _k = "sizeof_" + _m.group(1).strip()
            if _k in vals:
                expanded = expanded.replace(_m.group(0), str(vals[_k]))
        nb = _eval_bytes(expanded, vals, scale_b, out_b)
        if nb is None:
            hz.append(dict(check="H25_dma_granularity_unresolved", level="提示", line=ln,
                           buffer=name,
                           err=(f"`{name}.blockLen = {blk_len}` **算不出字节数**, 本条没查成 "
                                f"(不是查过没问题)。补 --shape 里缺的量再跑 —— **最常见的缺项是模板类型宽度**, "
                                f"形如 sizeof_T1=2。"
                                f"传输粒度直接决定有效带宽, 而它只能从源码读、测不出来。")))
            continue
        if nb >= _DMA_EFFICIENT_B:
            continue
        if name in stride0:
            #  跨距为 0 -> blockCount 个 block 首尾相接, 实际是一次连续搬运。
            #  这里不报, 但**要说出来**为什么不报 —— 否则"没查"与"查过没问题"又分不开。
            hz.append(dict(check="H25_dma_contiguous_ok", level="提示", line=ln, buffer=name,
                           err=(f"`{name}.blockLen = {blk_len}` = {nb} B/块, 低于 {_DMA_EFFICIENT_B} B, "
                                f"**但 srcStride/dstStride 显式置 0** —— blockCount 个块首尾相接, "
                                f"实际是**一次连续搬运**, 不算碎, 故不报。\n"
                                f"    ★记下这条是为了让'没查'与'查过没问题'分得开。要确认实际长度, "
                                f"看它运行时的 blockCount (sfag: curGatherSize=64 -> 64x128 = 8 KB 连续)。")))
            continue
        #  按粒度估一个**乐观**的带宽占比: 粒度不足一个 512B 事务时, 事务本身的固定开销
        #  按比例摊薄 —— 这只给量级, 不作承诺。
        frac = nb / _DMA_EFFICIENT_B
        hz.append(dict(check="H25_dma_granularity", level="提示", line=ln, buffer=name,
                       err=(f"`{name}.blockLen = {blk_len}` = **{nb} B/次**, 低于 {_DMA_EFFICIENT_B} B。"
                            f"每次搬这么少时, 事务的固定开销摊不开 —— 同样的字节总量会拆成约 "
                            f"{1/frac:.1f} 倍的事务数, 有效带宽大致按这个比例掉。\n"
                            f"    ★先看这个张量在 GM 上是不是可以与相邻张量**按行打包**: "
                            f"两个分开的张量各搬一次, 打包后是一次。\n"
                            f"    ★sfag 2026-09-18 实撞: key 与 key_rope 分两次 gather, rope 那次 128 B; "
                            f"而 UB 侧的行宽早就是 nope+rope 连续的 —— 缺的连续性在 GM 那一端。\n"
                            f"    ★**判据给线索不给处方**: 能不能打包取决于**写它的那一端**"
                            f" (expdesign E3), 静态判不了。本条只负责让你看见 {nb} 这个数。")))
    return hz


def ub_buffers(src):
    """UB 缓冲 -> 容量表达式(字节). 来自 InitBuffer(第3实参=单张容量) 与 tmpBuff.GetWithOffset(count*sizeof)."""
    caps = {}
    for m in re.finditer(r'InitBuffer\(\s*(\w+)\s*,\s*[^,]+,\s*(.+?)\)\s*;', src, re.S):
        caps[m.group(1)] = m.group(2).strip()
    return caps

def alloc_map(body, caps):
    """本函数内 localVar -> 容量表达式. 来自 queue.AllocTensor 与 tmpBuff.GetWithOffset<T>(count,off)."""
    amap = {}
    for m in re.finditer(r'(\w+)\s*=\s*(\w+)\s*\.\s*(?:template\s+)?AllocTensor', body):
        if m.group(2) in caps:
            amap[m.group(1)] = caps[m.group(2)]
    for m in re.finditer(r'(\w+)\s*=\s*\w+\.\s*(?:template\s+)?GetWithOffset<\s*([\w:]+)\s*>\(\s*([^,]+?)\s*,', body):
        amap[m.group(1)] = f"({m.group(3)}) * sizeof({m.group(2)})"
    return amap

def datacopy_loads(body):
    """本函数内 (dstLocal, blockLen表达式, 行号). 关联 DataCopyExtParams.blockLen(第2字段) 与 DataCopyPad(dst,src,param)."""
    # 参数结构体 -> blockLen 表达式
    params = {}
    for m in re.finditer(r'DataCopyExtParams\s+(\w+)\s*\{\s*[^,]+,\s*(.+?),', body):
        params[m.group(1)] = m.group(2).strip()
    loads = []
    for m in re.finditer(r'DataCopyPad\(\s*(\w+)(?:\[[^\]]*\])?\s*,\s*(\w+)(?:\[[^\]]*\])?\s*,\s*(\w+)', body):
        dst, src, pv = m.group(1), m.group(2), m.group(3)
        if pv in params:
            ln = body[:m.start()].count("\n")   # 相对行(用于近似定位)
            loads.append((dst, params[pv], ln, m.group(0)))
    return loads

def check_ub_oob(src, shape=None, scale_b=2, out_b=2):
    """H3: 整维载入分块缓冲. shape 给了则算精确翻转点."""
    caps = ub_buffers(src)
    funcs = split_funcs(src)
    hz = []
    for name, body in funcs:
        amap = alloc_map(body, caps)
        for dst, blockLen, _, snippet in datacopy_loads(body):
            cap = amap.get(dst)
            if not cap:
                continue
            load_dims = _full_dims_in(blockLen)     # 载入随哪些整维缩放
            cap_dims = _full_dims_in(cap)            # 容量随哪些整维缩放
            bad = [d for d in load_dims if d not in cap_dims]   # 载入含的整维, 容量不含 -> 不缩放
            if not bad:
                continue
            h = dict(check="H3_ub_oob", buffer=dst, func=name, load=blockLen.strip(), cap=cap,
                     dims=bad, msg=(f"UB '{dst}' (容量 {cap.strip()}, 不随 {'/'.join(bad)} 缩放) 被载入 "
                                    f"{blockLen.strip()} (随 {'/'.join(bad)} 增长) 于 {name} -> {'/'.join(bad)} 大时 UB 越界"
                                    f"踩踏邻接缓冲 = run-to-run 非确定. 改: 按 baseN/ubCalSize 分块载入."))
            if shape:
                cb = _eval_bytes(cap, shape, scale_b, out_b)
                # 求翻转: 找 load 里的整维符号, 令其他符号取 shape 值, 解 load_bytes>cap_bytes
                if cb and len(bad) == 1:
                    d = bad[0]
                    per = _eval_bytes(blockLen, {**shape, d: 1}, scale_b, out_b)  # 每单位整维的字节
                    if per:
                        flip = cb // per
                        cur = shape.get(d)
                        h["quant"] = (f"容量={cb}B, 每单位{d}载入={per}B -> 翻转点 {d}>{flip} 越界"
                                      + (f"; 当前 {d}={cur} => {'越界!' if cur and cur > flip else 'fits'}" if cur else ""))
            hz.append(h)
    return hz

# ---- H4: aicore 禁 float<->unsigned 标量直转 (bisheng: "cast between floating and unsigned integer variable
#         is not allowed in aicore function"). 沉淀自 gmfr v8.5: static_cast<float>(tiling->k) 编译错.
#         判据: static_cast<float|double>(X) 或 (float)X 或 float(X), 其中 X 含 tiling-> 字段或已知无符号变量,
#         且未先 static_cast<int..> 中转 -> 编译必错. 修: 经 signed int32/int64 中转.
_UINT_DECL = re.compile(r'\b(?:uint32_t|uint64_t|uint16_t|uint8_t|size_t)\s+(\w+)')

def _extract_paren(src, open_pos):
    """从 '(' 位置抽取到匹配 ')' 的内部字符串 (计深度)."""
    depth = 0
    for i in range(open_pos, len(src)):
        if src[i] == '(':
            depth += 1
        elif src[i] == ')':
            depth -= 1
            if depth == 0:
                return src[open_pos + 1:i], i
    return src[open_pos + 1:], len(src)

def check_float_uint_cast(src):
    """H4: 扫 float<->unsigned 直转. 返回 hazard 列表 (含近似行号)."""
    uint_vars = set(_UINT_DECL.findall(src))            # 全文件无符号局部/成员变量名 (粗粒度足够)
    hz = []
    # float<-... : static_cast<float>( / static_cast<double>(
    for m in re.finditer(r'static_cast<\s*(float|double)\s*>\s*\(', src):
        inner, _ = _extract_paren(src, m.end() - 1)
        inner_s = inner.strip()
        # 已用 signed int 中转 -> 安全 (我的修法)
        if re.match(r'static_cast<\s*(?:int8_t|int16_t|int32_t|int64_t|int)\s*>', inner_s):
            continue
        has_tiling = 'tiling->' in inner_s
        has_uint_var = any(re.search(rf'(?<![A-Za-z_]){v}(?![A-Za-z0-9_])', inner_s) for v in uint_vars)
        if has_tiling or has_uint_var:
            ln = src[:m.start()].count("\n") + 1
            why = "tiling-> 无符号字段" if has_tiling else "无符号变量"
            hz.append(dict(check="H4_float_uint_cast", buffer=f"L{ln}", line=ln,
                           msg=(f"L{ln}: static_cast<{m.group(1)}>({inner_s[:48]}...) 直转{why} -> aicore 编译错 "
                                f"'cast between floating and unsigned integer not allowed'. 修: 先 static_cast<int32_t>() 再转 float.")))
    # 反向 (unsigned<-float) 实战未撞, 且 static_cast<uint>(...sizeof(float)...) 是尺寸计算非转换 (误报源), 故暂不覆盖;
    # 若日后撞到 float->unsigned 再精确补 (要求内层是 float 声明变量, 排除 sizeof).
    return hz

# ---- H5: TQue depth 越界双入队 (DeQue 前 EnQue 次数 > 队列 depth) -> 真机 aicore timeout(507014) 卡死.
#         沉淀自 gmfr v8.5: vhi+vlo 都 EnQue 同一 TQue<VECOUT,1> (2 EnQue 无中间 DeQue) -> 死锁.
#         判据: TQue<pos,DEPTH>/TQueBind<...,DEPTH> 取 DEPTH; 函数内按文本序跑 EnQue(+1)/DeQue(-1) 余额, max>DEPTH 即报.
def queue_depths(src):
    """queue 变量名 -> 声明 depth. TQue<pos,DEPTH> qname; / TQueBind<src,dst,DEPTH> qname;"""
    d = {}
    for m in re.finditer(r'\bTQue\s*<[^,>]+,\s*(\d+)\s*>\s+(\w+)\s*;', src):
        d[m.group(2)] = int(m.group(1))
    for m in re.finditer(r'\bTQueBind\s*<[^,]+,[^,]+,\s*(\d+)\s*(?:,[^>]*)?>\s+(\w+)\s*;', src):
        d[m.group(2)] = int(m.group(1))
    return d

def check_queue_depth(src):
    """H5: 每函数每队列文本序 EnQue/DeQue 余额, max outstanding > depth -> 死锁."""
    depths = queue_depths(src)
    if not depths:
        return []
    hz = []
    for name, body in split_funcs(src):
        for q, dep in depths.items():
            # 文本序事件: EnQue(+1)/DeQue(-1)/return(reset). 在 return 处清零 —— 互斥分支(det if{..EnQue;return;} 与
            # 非 det fallthrough EnQue)只有一条路径执行, 不该累计; 真 bug(两 EnQue 无 return/DeQue 间隔)才累计越界.
            evs = []
            for m in re.finditer(rf'\b{q}\s*\.\s*(?:template\s+)?EnQue\s*\(', body):
                evs.append((m.start(), +1))
            for m in re.finditer(rf'\b{q}\s*\.\s*(?:template\s+)?DeQue\b', body):
                evs.append((m.start(), -1))
            for m in re.finditer(r'\breturn\b', body):
                evs.append((m.start(), 0))       # 0 = reset 标记
            if not any(d != 0 for _, d in evs):
                continue
            evs.sort()
            bal = 0; mx = 0; peak_pos = 0
            for pos, delta in evs:
                if delta == 0:
                    bal = 0; continue            # return: 路径终结, 余额清零
                bal += delta
                if bal < 0:
                    bal = 0                      # DeQue 多于 EnQue (跨函数消费), 归零不为负
                if bal > mx:
                    mx = bal; peak_pos = pos
            if mx > dep:
                ln = body[:peak_pos].count("\n") + (src[:src.find(body)].count("\n") if body in src else 0) + 1
                hz.append(dict(check="H5_tque_overflow", buffer=f"{q}@{name}",
                               msg=(f"队列 '{q}' (depth={dep}) 在 '{name}' 出现 {mx} 个 outstanding EnQue (DeQue 前入队 "
                                    f"{mx} 次 > depth {dep}) -> 真机 aicore timeout(507014) 死锁, 非编译错. "
                                    f"修: 次要输出改独立队列或持久 TBuf+V_MTE3/MTE3_V 手动同步; 或每次 DeQue 后再下一个 EnQue.")))
    return hz

# ---- H6: 跨核标志 (CrossCoreSetFlag/CrossCoreWaitFlag) 计数不平衡 -> 真机 aicore timeout(507014).
#      CV 并行算子的通用形态: AIC 侧每产出一块就置一次标志, AIV 侧每消费一块就等一次. 二者计数相等
#      往往依赖**隐式不变量** (如"Iterate() 次数 == curSingleN/baseN"), 这个不变量不写在任何地方,
#      改 tiling 的 single*/base*/step* 任一项就可能打破, 症状是核挂死而非报错.
#      本检查抽出每个 flag 的置位/等待点及其所在循环的控制表达式, 控制量不同即提示需复核.

#  ★跨核标志的实参常常是**三元选相**: `CrossCoreSetFlag<2,PIPE_MTE3>(m == 0 ? X_PING : X_PONG)`。
#  按"左括号后的第一个标识符"去取, 取到的是**条件变量** (m), 不是标志名 —— 于是 PING/PONG
#  那一对的收支从来没被算过, 而表里会凭空多出几个叫 taskMod / taskMod1 的假标志。
#  2026-09-17 sfag 实撞: H7 报 `taskMod 置位 2 次 / 等待 3 次`, 那压根不是标志。
#  取法: 有 `? A : B` 就取两支, 并把 _PING/_PONG(以及 _0/_1) 后缀归一成一个**通道名** ——
#  一对相位是同一条通道的两个槽, 分开计数两边都不平, 合起来才对得上。
_CC_TERNARY = re.compile(r"\?\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)")
_CC_PHASE_SUFFIX = re.compile(r"_(PING|PONG|Ping|Pong|ping|pong|[01])$")


def _cc_channel(name: str) -> str:
    """把一对相位归一成通道名: CUBE_WAIT_VEC_PING -> CUBE_WAIT_VEC_*"""
    n = _CC_PHASE_SUFFIX.sub("", name or "")
    return (n + "_*") if n != name else name


def _cc_flag_channel(src: str, lparen: int) -> str:
    """从 `(` 的位置读出这次调用的标志通道名。"""
    depth, i = 0, lparen
    while i < len(src):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                break
        elif src[i] == ";" and depth <= 0:
            break
        i += 1
    arg = src[lparen + 1:i]
    m = _CC_TERNARY.search(arg)
    if m:
        a, b = _cc_channel(m.group(1)), _cc_channel(m.group(2))
        return a if a == b else "%s|%s" % (a, b)
    m2 = re.match(r"\s*([A-Za-z_]\w*)", arg)
    return m2.group(1) if m2 else "?"


def _mutually_exclusive(src: str, lines: list) -> bool:
    """多处置位是否落在同一 if-else 链的**不同分支** —— 那样运行时只走一条, 不是重复置位。

    ★为什么要这条 (2026-08-27 实撞): GemmV3 的 RunVector 在 `if (mTileHalfActual != 0) {...}`
      与 `else {...}` 里各发一次 V_NOTIFY_C, H7 按"同一循环上下文"聚合就报成"置位 2 次 / 等待 1 次"。
      那是误报 —— 判据过宽会变成狼来了, 而 H7 抓的真 bug (置位与等待次数真不等) 很值钱, 不能让它被淹掉。
    判法: 相邻两处之间的源码必须恰好跨过一个 else, 且花括号收支平衡 (说明在同一嵌套层)。
    """
    if len(lines) < 2:
        return False
    offs = []
    for ln in sorted(lines):
        idx, pos = 1, 0
        for pos, ch in enumerate(src):
            if idx == ln:
                break
            if ch == "\n":
                idx += 1
        offs.append(pos)
    for a, b in zip(offs, offs[1:]):
        seg = src[a:b]
        if "else" not in seg or seg.count("{") != seg.count("}"):
            return False
    return True


#  ---- H10: SetFlag 与 WaitFlag **紧邻** -> 这两条 pipe 被串起来, 双缓冲白写 ----------------
#  本判据抓的不是死锁而是**性能**: `SetFlag<MTE1_M>(ev); WaitFlag<MTE1_M>(ev); Mmad(...)` 这种写法
#  让当前迭代必须等自己的搬运做完才能算, 而下一片的搬运又排在这条计算之后 —— 两条 pipe 排队执行。
#  ★实证 (GemmV3 2026-08-27): RunCube 的 fragment 循环正是这个形状, L0A/L0B 明明按奇偶做了
#    pingpong 却没用上; msprof 量出的 pipe 重叠度 (Σpipe/aic_time) 只有 1.23,
#    同 shape 的 AscendC matmul 库是 2.25, 两者 cube 总工作量相当 (152 vs 159 核·us)。
#  改法是软件流水错开一拍: 先发起下一片的搬运, 再等本片、算本片。
#  ★**命中不等于它是主要矛盾**, 已两次实测证伪:
#    (1) 2026-09-11 FIAS: cube 侧两处 `mm?ResL0C.Set<M_FIX>(); Wait<M_FIX>(); Fixpipe(...)` 命中本判据,
#        且 "cube 443 + fixpipe 156 = 599us ≈ aic_time 565us" 看着像全串。把两半拆开、中间插进
#        另一条 matmul 之后: fixpipe 忙时 156.1->153.3us **几乎没动** (说明它本来就在重叠),
#        aic_scalar 311->331us, aic_time 565->583us, 端到端 614.8->623.8us —— 净亏。
#        事后看这正落在本判据自己写的前置条件里: 标量已占 aic_time 55%。**先查前置条件再动手**。
#    (2) 2026-08-27 实测: GemmV3 照此改完, 重叠度 1.22->1.23 纹丝不动,
#    因为那里只是三条 pipe 之间的一道缝, 而更宽的那道 (MTE2 与 cube 之间, 成因是 L1 只有两组
#    pingpong buffer 且释放 flag 只在最后一片置位) 没动。用本判据定位后, **先用
#    core/perf --signals 看"三条 pipe 的活跃时间之和是不是就等于 aic_time"** ——
#    严格相加说明是全链串行, 那就要逐道缝找, 不能只修工具指出的这一处。
#  判据强度: 紧邻本身只算**提示** (有时确实没有可重叠的工作); 但若同一 flag 出现两个 event id
#  (pingpong 迹象) 还紧邻, 那就是**双缓冲写了没用**, 升为错误级。
#  ★事件 id 常常是**数组下标**形式 (`vToMte3Id[0]`, `mte3ToVId[runInfo.loop % DB]`) —— 早期这里只写了
#    `(\w+)`, 于是所有带下标的 Set/Wait 一条都匹配不上, H10/H7 对这类写法全盲。
#    2026-09-11 FIAS 实撞: fia_block_vec_noquant_mla.h 的两处紧邻 (L513/L672) 就是这个形状。
_EVID = r"(\w+(?:\[[^\]\n]*\])?)"
_SETF = re.compile(r"SetFlag<\s*(?:AscendC::)?HardEvent::(\w+)\s*>\s*\(\s*" + _EVID + r"\s*\)")
_WAITF = re.compile(r"WaitFlag<\s*(?:AscendC::)?HardEvent::(\w+)\s*>\s*\(\s*" + _EVID + r"\s*\)")
#  ★同一个判据的**成员形式**: `buf.Set<HardEvent::M_FIX>(); buf.Wait<HardEvent::M_FIX>();`
#    2026-09-11 FIAS 实撞: core/perf 的 MFU 分解把人路由到 H10, 而 H10 只认自由函数形式,
#    于是它对着正是这个形状的文件报"0 条"。那三处紧邻实测占端到端 39% 的流水互等。
_SETM = re.compile(r"(\w+)\s*\.\s*Set<\s*(?:AscendC::)?HardEvent::(\w+)\s*>\s*\(\s*\)")
_WAITM = re.compile(r"(\w+)\s*\.\s*Wait<\s*(?:AscendC::)?HardEvent::(\w+)\s*>\s*\(\s*\)")
#  这个对象是不是从多缓冲池里取的 (取到了就说明本来有 pingpong 可用)
_FROM_POOL = re.compile(r"(\w+)\s*=\s*(\w*[Bb]uffers\w*)\s*\.\s*Get(?:Pre|Reused|Vec|Cube)?\s*\(\s*\)")
_MULTIBUF_POOL = re.compile(r"BuffersPolicy(?:DB|3buff|4buff)\s*<[^>]*>\s*(\w+)")


def _h10_paired(a: str, b: str) -> bool:
    """两个事件 id 名字是不是**同一资源的两个槽** (xxxPing/xxxPong, xxx0/xxx1)。

    ★2026-09-16 用 60 个真实算子普查时定的: 原来按"这个文件里该事件类型有几个 id"判
      双缓冲, 于是 sfag (12/12 Pass 的良品) 被报 18 条**错误级**。错误级的含义是
      "做了双缓冲却写成串行", 名字不成对就没有这个前提。
    """
    na = re.sub(r"(?i)(ping|pong|\d+)$", "", a)
    nb = re.sub(r"(?i)(ping|pong|\d+)$", "", b)
    return na == nb and a != b and bool(na)


def check_setwait_adjacent(src: str) -> list:
    code_only = strip_comments(src)
    hz, evs = [], {}
    for m in _SETF.finditer(code_only):
        evs.setdefault(m.group(1), set()).add(m.group(2))
    for m in _SETF.finditer(code_only):
        ev_kind, ev_id = m.group(1), m.group(2)
        tail = code_only[m.end():m.end() + 200]
        w = _WAITF.search(tail)
        if not w or w.group(1) != ev_kind or w.group(2) != ev_id:
            continue
        between = tail[:w.start()]
        if between.count(";") > 1:          # 中间还有别的语句 -> 不算紧邻
            continue
        line = code_only[:m.start()].count("\n") + 1
        #  ★pingpong 的证据必须是"另一个 id 就在**这一处附近**", 不能是"整个文件里这个
        #    事件类型有好几个 id" —— 大文件里同一类型事件本来就会在不同函数各用各的 id,
        #    那不是双缓冲。2026-09-16 用 60 个真实算子普查时发现: 按整文件统计时
        #    sfag (12/12 Pass 的良品) 被报 18 条**错误级**, conv3d_backprop_input_v2 被报 9 条。
        #    错误级的含义是"做了双缓冲却写成了串行", 证据不到位就只是噪声。
        _W = 1500
        near = code_only[max(0, m.start() - _W):m.end() + _W]
        #  证据一: 这个 id 本身是**从两个值里选出来**的 —— `ev = (k & 1) ? ID1 : ID0`
        #  或 `ev = TAB[i & 1]`。这是双缓冲最常见的写法, 文本上只有一个 id, 按"有几个 id"
        #  统计根本看不见它。
        sel = bool(re.search(re.escape(ev_id) + r"\s*=\s*[^;]*\?[^;]*:", near)
                   or re.search(re.escape(ev_id) + r"\s*=\s*\w+\s*\[[^\]]*&\s*1\s*\]", near))
        #  证据二: 附近另有一个**名字成对**的 id (xxxPing/xxxPong, xxx0/xxx1)。
        #  名字不成对的两个事件是**不同用途**, 不是同一资源的双槽 —— sfag 的
        #  castEvent / cvtEvent / event 就挨在二十行内, 同为 MTE2_V, 但各做各的事。
        pair = any(_h10_paired(ev_id, other)
                   and re.search(r"<\s*(?:AscendC::)?HardEvent::" + re.escape(ev_kind)
                                 + r"\s*>\s*\(\s*" + re.escape(other), near)
                   for other in evs.get(ev_kind, ()) if other != ev_id)
        pingpong = sel or pair
        #  ★证据三 (2026-09-20 实撞, 代价一次构建 + 88% 算错的候选):
        #    紧邻**不等于冗余**。若 WaitFlag 之后**紧接着**就有一条消费刚搬进来的那块数据的
        #    指令, 那这个等待是**必需的** —— 删掉它, 那条指令就作用在没搬完的数据上。
        #    g20 实例: `DataCopy(dkInUb,...)` / Set+Wait / `Muls(dkInUb, dkInUb, ...)`。
        #    我当时用 grep 过滤过的视图看代码, 过滤器里没有 `Muls`, 于是那条依赖是**不可见的**,
        #    据此判"这对 Set/Wait 冗余" -> 删掉 -> dk/dk_rope 88.4%/90.1% 算错。
        #    ⇒ 本判据只说"这里是串行点", **不说"这里可以删"**。判能不能删要看紧跟着的消费者。
        after = code_only[m.end():m.end() + 300]
        nxt = re.search(r"\b(Muls|Adds|Add|Sub|Mul|Cast|Duplicate|Exp|Div|Max|Min|Select|"
                        r"Compares?|ReduceSum|Brcb|DataCopy\w*)\s*\(\s*(\w+)", after)
        consumer = nxt.group(0)[:40] + "..." if nxt else None
        hz.append(dict(check="H10_setwait_adjacent", buffer=f"{ev_kind}({ev_id})",
                       level="错误" if pingpong else "提示",
                       msg=(f"L{line}: SetFlag<{ev_kind}>({ev_id}) 与 WaitFlag 紧邻 —— "
                            f"这两条 pipe 会排队执行, 当前迭代必须等自己的搬运做完才能算, "
                            f"而下一片的搬运又排在这条计算之后。"
                            + (f"\n      ★**紧跟着就是 `{consumer}`** —— 若它消费的正是刚搬进来的那块, "
                               f"**这个等待是必需的, 删不得** (2026-09-20 实撞: 删掉致 88% 元素算错)。"
                               f"本判据只说'这里是串行点', 不说'这里可以删'。" if consumer else "")
                            + (f"★该 flag 有 {sorted(evs[ev_kind])} 两个 event id, 说明本来做了 pingpong "
                               f"双缓冲, 但这么写等于没用上。" if pingpong else "")
                            + f"改法: 软件流水错开一拍 —— 先发起下一片的搬运, 再等本片、算本片。"
                            f"★前置条件: 该循环的**标量流水尚有余量**。先看 core/perf --signals 的 "
                            f"aic_scalar_mte1_stall_time —— 它若已占 aic_time 三成以上, 说明标量本身就是 "
                            f"mte1 的阻塞源, 此时用'保存上一拍参数'的写法排流水会再加标量, 净亏。"
                            f"(GemmV3 实撞: stall 7.49us 占 37%, 排完一拍后 aic_time 20.12->22.40, "
                            f"重叠度 1.22->1.16 反而更串行。) 标量吃紧时改用循环展开两拍 —— "
                            f"prev 状态成为同拍内的局部量, 标量总量不变而指令序改善。"
                            f"效果用 core/perf --signals 的 'pipe 重叠度' 验 (接近 1 = 串行)。")))

    #  ---- 成员形式: buf.Set<HardEvent::X>(); buf.Wait<HardEvent::X>(); ----
    multibuf = set(_MULTIBUF_POOL.findall(code_only))          # 声明成多缓冲的池名
    from_pool = {m.group(1): m.group(2) for m in _FROM_POOL.finditer(code_only)}
    for m in _SETM.finditer(code_only):
        obj, ev_kind = m.group(1), m.group(2)
        tail = code_only[m.end():m.end() + 200]
        w = _WAITM.search(tail)
        if not w or w.group(1) != obj or w.group(2) != ev_kind:
            continue
        if tail[:w.start()].count(";") > 1:     # 中间还有别的语句 -> 不算紧邻
            continue
        line = code_only[:m.start()].count("\n") + 1
        pool = from_pool.get(obj)
        pingpong = pool is not None and pool in multibuf
        hz.append(dict(check="H10_setwait_adjacent", buffer=f"{obj}.{ev_kind}",
                       level="错误" if pingpong else "提示",
                       msg=(f"L{line}: {obj}.Set<{ev_kind}>() 与 {obj}.Wait<{ev_kind}>() 紧邻 —— "
                            f"这两条 pipe 会排队执行, 当前迭代必须等自己这一步做完才能进下一步。"
                            + (f"★{obj} 取自多缓冲池 {pool}, 说明本来有 pingpong 可用, 这么写等于没用上。"
                               if pingpong else "")
                            + f"改法: 把两半拆开, 中间插进**另一个任务/另一条 matmul** 的同类工作, "
                              f"让两条 pipe 并行; 或用硬件 unitFlag 直通 (MMParam/FixpipeParams 的 unitFlag)。"
                            + f"★★前置条件必须先验, 否则两种改法都白做: 去 msprof 看这条被串起来的 pipe "
                              f"**是不是真的没在重叠** —— 用 `core/perf --signals` 的 pipe 重叠度, "
                              f"以及该 pipe 的忙时相对 aic_time 的占比。"
                            + f"★FIAS 950PR_957b 上这处 (mm1/mm2 的 matmul 紧跟自己的 fixpipe) 被**两条独立手段"
                              f"先后证伪**: (a) 2026-09-11 软件错拍 —— fixpipe 忙时 156.1 -> 153.3us 几乎没动 "
                              f"(说明它本来就在重叠) 而 aic_scalar 311 -> 331us, 端到端 614.8 -> 623.8us 净亏; "
                              f"(b) 2026-09-14 硬件 unitFlag (精度逐位相同) —— 558.64 -> 559.27us, **中性**。"
                              f"两次都说明: `cube 忙时 + fixpipe 忙时 ≈ aic_time` 这个算式看着像全串, "
                              f"但它**不构成证据** —— 必须直接看重叠度。")))

    return hz


#  ---- H11: **同核** SetFlag/WaitFlag 在循环体内收支不平衡 -> 死锁 -----------------------
#  H7 查的是跨核 (CrossCoreSetFlag/WaitFlag), 同核的 SetFlag<HardEvent::X>(id) / WaitFlag 一样会死锁,
#  而且更容易在改动中被无意破坏 —— 2026-08-28 实撞: 为做诊断把 RunVector 里一个 `if (cond) {...}`
#  改成 `if (false && cond)`, 块内的 WaitFlag<MTE3_MTE2>(EVENT_ID0) 随之被跳过, 而块外的
#  SetFlag 还在, 循环内 Set 1 次 Wait 0 次 -> 真机直接挂死, 驱动无任何输出。
#  判法: 只看**循环体内**的次数 (循环外的初始 Set 与收尾 Wait 是另一对, 本就不该相等)。
#  ★本判据抓不住什么 (必须写明, 否则它会给人虚假的安全感): 一侧被**恒假条件**或运行期条件跳过时,
#    源码里两侧都在, 静态计数相等 —— 上面那次实撞 (`if (false && cond)`) 正属此类, H11 报不出来。
#    静态判据只能抓"源码里就写得不平衡"的情形; 条件相关的不平衡要靠真机跑一次 (症状是驱动无输出)。
#  ★事件号还有一种**成员调用**写法: `SetFlag<E>(buf.GetEventID<E>())`。原模式只认
#    `[\w\[\]\+\. ]`, 碰到里面的 `<`/`(` 就整条匹配不上 —— 于是 H11/H12/H15 对**整份**
#    这么写的内核全盲 (FIAS arch35 的 KV 槽旗标无一例外是这个形式)。判据看不见 = 假阴性,
#    比不报更危险。2026-09-11 由 FIAS 行分组改造反向补上。
_EVID_ANY = r"([\w\[\]\+\. ]+?(?:<[^<>]*>\s*\(\s*\))?)"
_SETF_ANY = re.compile(r"SetFlag<\s*(?:AscendC::)?HardEvent::(\w+)\s*>\s*\(\s*" + _EVID_ANY + r"\s*\)\s*;")
_WAITF_ANY = re.compile(r"WaitFlag<\s*(?:AscendC::)?HardEvent::(\w+)\s*>\s*\(\s*" + _EVID_ANY + r"\s*\)\s*;")


# ---- H21: **逐轮同步一个每 N 轮才切换的资源** —— 屏障粒度比它保护的资源细一档,
#      于是循环体被逐轮串行化, 访存请求无法在途重叠。
#      形态特征: 一个循环体内同时出现同一事件的 WAIT_FLAG 与 SET_FLAG (一进一出, 账面平),
#      而这对旗标保护的缓冲**只在某个计数器达到阈值时才翻转** (循环里有
#      `if (cnt == CAP) { ... pingpong = 1 - pingpong; cnt = 0; }` 这样的槽切换)。
#      2026-09-15 sparse_flash_attention_grad 实撞: GatherKVNonOptimized 每 2 个 token
#      就等一次 MTE3->MTE2, 而槽每 64 个 token 才换一次 —— 同槽内的 32 轮写的是 UB 上
#      互不重叠的偏移, 本来不需要任何轮间同步。后果是 **AIV 每核 GM->UB 只跑到
#      17.6 GB/s, 是 mte2_GBps_per_core=132 的 13%**: 不是带宽不够, 是被旗标卡住。
#      与 H10 的分工: H10 看 Set 与 Wait 在**文本上**是否紧邻 (同一轮内立即等),
#      H21 看的是**跨迭代**的那一对 —— 轮末 Set、下一轮开头 Wait, H10 看不见。
#      ★清单式, 不是判定: 有些循环的资源确实每轮都换 (真正的单缓冲), 那样逐轮同步是对的。
#      要核对的是"这对旗标保护的东西, 是不是每轮都被覆盖"。
def check_barrier_finer_than_resource(src: str) -> list:
    code = strip_comments(src)
    _EV = re.compile(r"(?:AscendC::)?(?:SET_FLAG|WAIT_FLAG)\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)")
    _SLOT = re.compile(r"(\w+)\s*=\s*1\s*-\s*\1\b")           # pingpong = 1 - pingpong
    hz = []
    for m in re.finditer(r"\b(for|while)\s*\(", code):
        i, d = m.end() - 1, 0
        while i < len(code):
            if code[i] == "(":
                d += 1
            elif code[i] == ")":
                d -= 1
                if d == 0:
                    break
            i += 1
        j = i + 1
        while j < len(code) and code[j] in " \t\r\n":
            j += 1
        if j >= len(code) or code[j] != "{":
            continue
        k, d = j, 0
        while k < len(code):
            if code[k] == "{":
                d += 1
            elif code[k] == "}":
                d -= 1
                if d == 0:
                    break
            k += 1
        body = code[j + 1:k]
        if not _SLOT.search(body):
            continue          # 循环里没有槽切换 -> 谈不上"每 N 轮才换"
        #  槽切换必须在一个条件里 (否则就是每轮都换, 逐轮同步是对的)
        slot_m = _SLOT.search(body)
        head = body[:slot_m.start()]
        if head.count("if") == 0:
            continue
        #  ★只报 WAIT 落在循环体**顶层**的那一对: 那才是"每轮无条件等一次"。
        #  WAIT 若已经包在 if 里 (如 `if (cnt == 0) WAIT(...)`), 它就已经是槽级的, 不该再报;
        #  写出分支内部的 WAIT (等本槽读完再写出) 同理是必要的。
        #  不加这一条, 判据会把**已经改对的代码**也报出来 —— 2026-09-15 收紧前它在修好的
        #  源码上仍命中 8 处, 提示级的噪声一多就没人看了。
        def _top_level(pos):
            d = 0
            for ch in body[:pos]:
                if ch == "{":
                    d += 1
                elif ch == "}":
                    d -= 1
            return d == 0
        waits, setz = set(), set()
        for em in _EV.finditer(body):
            tok = body[max(0, em.start() - 10):em.start() + 9]
            if "SET_FLAG" in tok:
                setz.add(em.groups())
            elif _top_level(em.start()):
                waits.add(em.groups())
        both = waits & setz
        for ev in sorted(both):
            hz.append(dict(check="H21_barrier_finer_than_resource", level="提示",
                           buffer=f"{ev[0]}_{ev[1]}({ev[2]})",
                           line=code[:j].count("\n") + 1,
                           msg=(f"循环体内对事件 `{ev[2]}` ({ev[0]}->{ev[1]}) 既 WAIT 又 SET, "
                                f"而同一循环里的缓冲槽是**有条件**翻转的 "
                                f"(`{slot_m.group(0)}` 在 if 之内) —— 屏障粒度比它保护的资源细一档。"
                                f"**这是待核对清单**: 请确认同一个槽内的连续几轮写的是不是 UB 上"
                                f"互不重叠的偏移; 若是, 这对旗标只需在**槽开始**时等一次、"
                                f"**写出后**放一次, 逐轮同步纯属把循环串行化。"
                                f"★2026-09-15 实撞: gather 每 2 个 token 等一次而槽每 64 个才换, "
                                f"AIV 每核 GM->UB 只跑到能力的 13% (17.6 / 132 GB/s)。"
                                f"与 H10 的分工: H10 看同一轮内文本紧邻的 Set/Wait, "
                                f"这条看的是轮末 Set、下一轮开头 Wait 的**跨迭代**那一对。")))
    return hz


def check_intracore_flag_balance(src: str) -> list:
    """同核 Set/Wait 在**同一条件块内**的收支平衡。

    ★按条件块分组而不是按整个循环体 —— 2026-08-28 自测实撞: 按循环体加总时,
      本来正确的 gemm_v3 kernel 被报 4 条 (它的跨 tile 预取块 `if (k == last && 还有下一个 tile)`
      里 Set/Wait 是成对的, 块外另有一对, 加总后自然不等)。判据过宽就是狼来了。
    """
    code_only = strip_comments(src)
    lines = code_only.split("\n")
    #  为每行算出它所处的"条件/循环块"路径 (用各层块的起始行号串成 key)
    ctx, stack, depth = [], [], 0
    for i, ln in enumerate(lines):
        opens, closes = ln.count("{"), ln.count("}")
        starts_block = bool(re.search(r"\b(if|else|for|while)\b", ln)) and opens > 0
        ctx.append(tuple(stack))
        if starts_block:
            stack.append(i)
        depth += opens - closes
        while stack and depth <= len(stack) - 1:
            stack.pop()
    from collections import Counter
    sets, waits = Counter(), Counter()
    in_loop = _loop_mask(lines)
    for i, ln in enumerate(lines):
        if not in_loop[i]:
            continue
        for m in _SETF_ANY.finditer(ln):
            sets[(m.group(1), m.group(2).strip(), ctx[i])] += 1
        for m in _WAITF_ANY.finditer(ln):
            waits[(m.group(1), m.group(2).strip(), ctx[i])] += 1
    hz = []
    for key in set(sets) | set(waits):
        ev, eid, _ = key
        ns, nw = sets[key], waits[key]
        if ns == nw or ns == 0 or nw == 0:
            continue
        hz.append(dict(check="H11_intracore_flag_balance", buffer=f"{ev}({eid})", level="错误",
                       msg=(f"**同一条件块内** SetFlag<{ev}>({eid}) 出现 {ns} 次而 WaitFlag 出现 {nw} 次 —— "
                            f"次数不等即死锁 (真机挂死, 驱动无任何输出, 也不报错)。"
                            f"注意: 条件块外的另一对、以及循环外的初始化/收尾, 都不计入本组。")))
    return hz


#  ---- H15: 同一事件的 Set 与 Wait 挂在**不同的守卫条件**下 -----------------------------------
#  H11 只数同一条件块内的次数, 两侧分别落在不同的 `if` 里时它两组各看到"只有一侧", 直接跳过 ——
#  正是它自己写明的盲区 ("条件相关的不平衡要靠真机跑一次")。而这种写法在做"多任务共用一份资源"
#  的改造时必然出现: FIAS 2026-09-11 的行分组把一块 KV 喂给组内 G 行, 于是
#  `WaitFlag<MTE2_MTE1>` 收进 `if (isFirstOfBlock)`、`SetFlag<MTE1_MTE2>` 收进 `if (isLastOfBlock)`。
#  这么写**可以是对的**, 但对不对取决于两个条件在运行期各触发多少次 —— 源码上已看不出来。
#  故本判据不判错, 只强制作者把"为什么次数相等"说清楚: 两侧守卫不同就报出来。
#  ★抓不住什么: 守卫写在**调用方**时 (本例的预取就是 —— SetFlag 在 PreloadKvTile 内部无守卫,
#    守卫在调用它的 ExecuteTask 里) 本判据看不见, 因为它只在单文件内按词法作用域看。
_IF_HEAD = re.compile(r"\b(?:else\s+)?if\s*\((.+)\)\s*\{?\s*$")


def check_flag_guard_asymmetry(src: str) -> list:
    code_only = strip_comments(src)
    lines = code_only.split("\n")
    #  ★作用域栈必须按**花括号深度**弹出, 不能按栈长度。早期版本写的是
    #    `while stack and depth <= len(stack)-1: pop()` —— depth 数的是文件里所有层
    #    (namespace/class/函数都算), 与"我压了几个 if"根本不是一回事, 于是 `if` 块结束后
    #    没弹干净, 后面的语句被安上了上一个 if 的条件。
    #    ★这种错法比不报更坏: 它会把一条**正确**的代码报成"两侧守卫不同", 把人引去查一个
    #    不存在的挂死原因。FIAS 2026-09-14 实撞: 真机确实挂死, 而本判据指的那处是对的。
    guards, stack, depth = [], [], 0
    for ln in lines:
        guards.append(tuple(g for g in (c for _, c in stack) if g))
        opens, closes = ln.count("{"), ln.count("}")
        after_close = depth - closes
        while stack and stack[-1][0] >= after_close:
            stack.pop()
        if opens > 0 and re.search(r"\b(if|else|for|while)\b", ln):
            m = _IF_HEAD.search(ln.strip())
            cond = m.group(1).strip() if m else ("else" if re.search(r"\belse\b", ln) else "")
            stack.append((after_close, cond))
        depth = after_close + opens
    setg, waitg = {}, {}
    for i, ln in enumerate(lines):
        g = guards[i]
        for mm in _SETF_ANY.finditer(ln):
            setg.setdefault((mm.group(1), mm.group(2).strip()), set()).add(g)
        for mm in _WAITF_ANY.finditer(ln):
            waitg.setdefault((mm.group(1), mm.group(2).strip()), set()).add(g)
    hz = []
    for key in set(setg) & set(waitg):
        ev, eid = key
        sg, wg = setg[key], waitg[key]
        if sg == wg:
            continue
        if not any(x for x in sg) and not any(x for x in wg):
            continue
        fmt = lambda gs: " | ".join(" && ".join(g) if g else "(无守卫)" for g in sorted(gs))
        hz.append(dict(check="H15_flag_guard_asymmetry", buffer=f"{ev}({eid})", level="提示",
                       msg=(f"SetFlag<{ev}>({eid}) 的守卫是 [{fmt(sg)}], 而 WaitFlag 的守卫是 [{fmt(wg)}] "
                            f"—— 两侧条件不同, 收支是否相等只在运行期才成立, 源码上看不出来。"
                            f"次数一旦不等就是真机挂死 (驱动无任何输出, 也不报错)。"
                            f"要么让两侧守卫相同, 要么在注释里写明这两个条件在一轮资源生命周期里"
                            f"各触发恰好一次的理由。")))
    return hz


#  ---- H16: Fixpipe 的双目标模式与类型转换互斥 ------------------------------------------------
#  CANN 自己的实现里有明文断言 (dav_3510/kernel_operator_fixpipe_impl.h:230):
#      "Dual destination mode can be enabled only when normal DMA or NZ2ND is enabled
#       with any other Fixpipe functions bypassed."
#  即 dualDstCtl != 0 时, quantPre 必须是 NoQuant, reluEn 必须是 false, 且 format 不能是
#  COLUMN_MAJOR。但那是 **ASCENDC_ASSERT (debug 级)**, release 版不触发 —— 于是唯一的症状是
#  上机 507015 aicore exception + "The configuration of FIXP is illegal. subErrType: 0x4",
#  编译零告警。2026-09-14 FIAS 实撞: 把 mm2 结果改成 fp16 落盘 (靠 quantPre 转换) 而 dualDstCtl
#  仍是 1 (一次写两个 AIV 的 UB), 三条用例全崩。改法: 拆成两次单目标 fixpipe, 用 subBlockId
#  选目标 AIV, 第二次的源在 L0C 上偏移 halfM*16 个元素。
_FIXP_DECL = re.compile(r"FixpipeParams\w*<\s*(?:CO2Layout::)?(\w+)\s*>\s+(\w+)\s*;")


def check_fixpipe_dualdst_quant(src: str) -> list:
    code_only = strip_comments(src)
    hz = []
    for m in _FIXP_DECL.finditer(code_only):
        fmt, name = m.group(1), m.group(2)
        #  只看这个变量到下一次同名声明 / 函数结尾之间的赋值
        seg = code_only[m.end(): m.end() + 4000]
        dual = re.search(re.escape(name) + r"\s*\.\s*dualDstCtl\s*=\s*([^;]+);", seg)
        quant = re.search(re.escape(name) + r"\s*\.\s*quantPre\s*=\s*([^;]+);", seg)
        relu = re.search(re.escape(name) + r"\s*\.\s*reluEn\s*=\s*(true)\s*;", seg)
        if dual is None:
            continue
        dv = dual.group(1).strip()
        if dv in ("0", "0U", "0b00", "false"):
            continue
        bad = []
        if quant is not None and "NoQuant" not in quant.group(1):
            bad.append(f"quantPre = {quant.group(1).strip()}")
        if relu is not None:
            bad.append("reluEn = true")
        if fmt == "COLUMN_MAJOR":
            bad.append("format = COLUMN_MAJOR")
        if not bad:
            continue
        hz.append(dict(check="H16_fixpipe_dualdst_vs_quant", buffer=name, level="错误",
                       msg=(f"{name}.dualDstCtl = {dv} (双目标模式) 与 {', '.join(bad)} **同时**设置 —— "
                            f"CANN 的 dav_3510 实现里有明文断言: 双目标模式只允许在 normal DMA / NZ2ND "
                            f"且其余 Fixpipe 功能全旁路时使用。该断言是 debug 级, release 版不触发, "
                            f"所以编译零告警而上机必崩 (507015 aicore exception + "
                            f"'The configuration of FIXP is illegal')。"
                            f"改法: 拆成两次**单目标** fixpipe (dualDstCtl=0), 用 subBlockId 选目标 AIV, "
                            f"第二次的源在 L0C 上偏移 halfM*16 个元素。")))
    return hz



#  ---- H18: 跨核缓冲的槽数 < 软件流水的级数 -> 必然死锁 ---------------------------------------
#  三级软件流水 (本轮同时处理任务 L / L-1 / L-2) 配一个按块号轮转的 S 槽环时, 有一条纯算术判据:
#
#      设一个槽连续服务 G 个任务 (行分组时 G = 组内行数, 不分组时 G = 1), 则同一槽的复用间隔是
#      G*S 个任务; 而本轮最早发出的操作 (任务 L 的预取) 要等的槽, 由任务 L-G*S 在它的第
#      stages-1 级 (即第 L-G*S+stages-1 轮) 释放。要求该轮严格早于本轮:
#          (L - G*S) + (stages - 1) < L   <=>   G*S >= stages
#
#  G=1 时就退化成 S >= stages。S < stages 时**不报错也不告警**, 症状是真机彻底挂死: 驱动无任何
#  输出, 不超时, 不报 aicore exception —— 因为那一轮的 Wait 在等一个本轮稍后才会发出的 Set。
#  2026-09-14 FIAS 实撞: 为了给行分组腾 L1, KV 槽从 3 降到 2 而流水仍是 3 级。矩形分核下 G=2
#  时 2*2>=3 成立, 但非矩形路径的 G 恒为 1, 1*2>=3 不成立 —— 所有非矩形形状挂死。
#  修法二选一: 要么把 S 提回 >= stages; 要么在源码里写明 G 的下界以及"每个组都满 G 行"的理由
#  (末组不足 G 行就会掉回 G=1)。
_PIPE_STAGE = re.compile(r"\[\s*\(?\s*loop\s*-\s*(\d+)\s*U?L?\s*\)?\s*%")
_PIPE_STAGE0 = re.compile(r"\[\s*loop\s*%")
#  Buffer<...> name[SIZE]  /  Buffer<...> *name[SIZE]
_SLOT_ARRAY = re.compile(r"Buffer\s*<[^;{}]*?>\s*\*?\s*(\w+)\s*\[\s*([\w]+)\s*\]")
_CONST_INT = re.compile(r"constexpr\s+\w+\s+(\w+)\s*=\s*(\d+)\s*U?\s*;")


def _int_env(src: str) -> dict:
    """把 `constexpr T NAME = <整数>;` 收成名字->值, 供槽数/取模数解析。"""
    return {m.group(1): int(m.group(2)) for m in _CONST_INT.finditer(src)}


def check_slot_ring_vs_pipeline_depth(src: str) -> list:
    code_only = strip_comments(src)
    stages = 0
    for m in _PIPE_STAGE.finditer(code_only):
        stages = max(stages, int(m.group(1)) + 1)
    if stages == 0:
        if _PIPE_STAGE0.search(code_only):
            stages = 1
    if stages < 2:
        return []          # 不是软件流水, 本判据不适用
    env = _int_env(code_only)
    hz = []
    seen = set()

    def _report(name, size, how):
        if (name, size) in seen:
            return
        seen.add((name, size))
        hz.append(dict(check="H18_slot_ring_vs_pipeline_depth", buffer=name, level="提示",
                       msg=(f"跨核槽环 {name} 只有 {size} 个槽, 而软件流水有 {stages} 级 "
                            f"(本轮同时处理任务 L .. L-{stages - 1}){how}。判据: 一个槽连续服务 G 个任务时, "
                            f"复用间隔 G*{size} 必须 >= {stages}, 否则任务 L 的预取要等的槽, 要到本轮**稍后**"
                            f"才被释放 —— 那一轮的 Wait 永远等不到。G=1 时即 {size} >= {stages} 不成立。"
                            f"症状是真机彻底挂死: 驱动无任何输出, 不超时, 也不报 aicore exception。"
                            f"要么把槽数提回 {stages}, 要么在源码里写明 G 的下界, 并说明为什么**每个**组"
                            f"(包括末组和非矩形退化路径) 都满 G 行。")))

    for m in _SLOT_ARRAY.finditer(code_only):
        name, size_tok = m.group(1), m.group(2)
        size = int(size_tok) if size_tok.isdigit() else env.get(size_tok)
        if size is None or size >= stages:
            continue
        _report(name, size, "")

    #  槽数是运行期量时 (`n = <cond> ? A : B;`), 只认**确实被用来索引槽环**的那一个:
    #  槽环 arr[idx] 的下标 idx 是某个字段, 而该字段由 `... = <something> % n;` 得出。
    #  不做这层追踪就会把任何取模数 (序列长度、头数 …) 都当成槽数报出来。
    arrays = {m.group(1) for m in _SLOT_ARRAY.finditer(code_only)}
    idx_fields = set()
    for arr in arrays:
        for m in re.finditer(re.escape(arr) + r"\s*\[\s*([\w\.\->]+)\s*\]", code_only):
            tok = m.group(1)
            if tok.isdigit():
                continue
            idx_fields.add(tok.split(".")[-1].split("->")[-1])
    #  ★槽号的**来源**必须单调递增, 否则 G*S 的复用间隔只在"不回绕"的那段里成立。
    #    s2 块下标这类量在行组/分段的边界上会被重新赋值 (回到本段起点), 槽号跟着跳回去,
    #    于是新一段的头几个任务会撞上紧邻的前几个任务刚用过的槽 —— 复用间隔局部塌成 1~2,
    #    症状与槽数不足完全一样 (真机静默挂死 / 507014 aicore timeout), 但 S >= stages 仍成立,
    #    上面那条判据看不出来。判据: 取模的左操作数除了"初始化成字面量"与自增之外还被赋过值,
    #    就说明它会回绕。2026-09-14 FIAS 实撞: kvSlot = s2Cur % 2, 而 s2Cur 有四处重新赋值,
    #    尾核末段宽度为奇数时必超时。
    for fld in idx_fields:
        for m in re.finditer(r"[\w\.\->]*\b" + re.escape(fld) + r"\s*=\s*([\w\.\->]+)\s*%\s*(\w+)\s*;", code_only):
            src_var = m.group(1).split(".")[-1].split(">")[-1]
            resets = [a for a in re.findall(re.escape(src_var) + r"\s*=\s*([^;=][^;]*);", code_only)
                      if not re.fullmatch(r"\s*\d+\s*U?L?L?\s*", a)]
            if resets:
                hz.append(dict(check="H18_slot_index_not_monotonic", buffer=src_var, level="提示",
                               msg=(f"槽号取自 {src_var} % ... , 而 {src_var} 在 {len(resets)} 处被**重新赋值**"
                                    f"(例如 `{src_var} = {resets[0].strip()[:40]}`), 不是只增不减。"
                                    f"槽环的复用间隔 G*S 只在下标单调时才成立: 下标一回绕, 新一段的头几个"
                                    f"任务就会撞上紧邻的前几个任务刚用过的槽, 间隔局部塌成 1~2, 而这时"
                                    f"S >= 流水级数仍然成立, H18 的槽数判据看不出来。"
                                    f"症状是真机静默挂死或 507014 aicore timeout, 且**与形状有关**"
                                    f"(回绕点的奇偶决定撞不撞), 小 shape 的门禁往往漏过。"
                                    f"改法: 槽号改由一个**逐块递增、从不回绕**的计数取模。")))
        for m in re.finditer(r"[\w\.\->]*\b" + re.escape(fld) + r"\s*=\s*[^;]*?%\s*(\w+)\s*;", code_only):
            var = m.group(1)
            if var.isdigit():
                vals = [int(var)]
            elif var in env:
                vals = [env[var]]
            else:
                #  成员变量: 收它所有的整数赋值 (含三元的两支), 取最小值
                vals = [int(v) for v in re.findall(
                    re.escape(var) + r"\s*=\s*[^;]*?(\d+)\s*U?\s*[;:]", code_only)]
                vals += [env[t] for t in re.findall(
                    re.escape(var) + r"\s*=\s*[^;]*?([A-Z_][A-Z0-9_]+)", code_only) if t in env]
            if not vals:
                continue
            lo = min(vals)
            if lo >= stages:
                continue
            _report(var, lo, f" (槽数是运行期量, 取到的最小值是 {lo})")
    return hz


#  ---- H17: 从**多缓冲池**取出的 buffer 从不等自己的反依赖事件 -> 写覆盖上一代仍在被读的数据 ----
#  多缓冲池的 Get() 是轮转的, 意味着这块存储**一定会被复用**。复用前必须等"上一代的消费者用完"
#  (L0A/L0B 上就是 `Wait<HardEvent::M_MTE1>`, L0C 上是 `Wait<HardEvent::FIX_M>`)。
#  ★2026-09-14 FIAS 实撞: 为省标量指令, 把 L0A 的 `Wait/Set<M_MTE1>` 去掉, 理由是"同一代的
#    A/B 由同一条 Mmad 消费, 等 B 的旗标就够"。真机 **507015 aicore exception**。
#    前提错在两个池的轮转步调并不一致 —— mm1 的切 K 每片段 A/B 各取一次 (+5/+5), 而 mm2 的切 N
#    是 A 取 1 次、B 取 4 次 (A 在 n 循环外只载入一次)。同一对池被两条 matmul 以不同节奏共用,
#    B 的旗标根本不代表 A 已被释放。
#  => 判据不去证明"步调是否一致"(那要跨函数分析), 而是直接要求: 取自多缓冲池的 buffer
#     必须有自己的 Wait。想共用旗标就得先证明两个池自始至终同步轮转, 那是注释里要写的事。
_BUF_USE = re.compile(r"\b(\w+)\s*\.\s*(?:template\s+)?(?:GetTensor|Get|Set|Wait|SetCrossCore|WaitCrossCore)\b")


def check_pool_buf_missing_wait(src: str) -> list:
    code_only = strip_comments(src)
    multibuf = set(_MULTIBUF_POOL.findall(code_only))
    if not multibuf:
        return []
    hz = []
    #  ★不能复用 _FROM_POOL: 它要求池名里带 "Buffers", 那是命名习惯不是语言规则,
    #    池换个名字判据就全盲。这里直接按**已声明为多缓冲的池名**来匹配赋值。
    pool_alt = "|".join(re.escape(x) for x in sorted(multibuf))
    get_re = re.compile(r"(\w+)\s*=\s*(" + pool_alt + r")\s*\.\s*Get(?:Pre|Reused|Vec|Cube)?\s*\(\s*\)")
    seen = set()
    for m in get_re.finditer(code_only):
        name, pool = m.group(1), m.group(2)
        if (name, pool) in seen:
            continue
        seen.add((name, pool))
        n_wait = len(re.findall(re.escape(name) + r"\s*\.\s*(?:template\s+)?Wait\s*<", code_only))
        n_set = len(re.findall(re.escape(name) + r"\s*\.\s*(?:template\s+)?Set\s*<", code_only))
        n_use = len(_BUF_USE.findall(code_only.replace(m.group(0), "")))
        if n_wait == 0 and n_use > 0:
            hz.append(dict(check="H17_pool_buf_no_wait", buffer=f"{name}<-{pool}", level="错误",
                           msg=(f"`{name}` 取自**多缓冲池** {pool} (轮转 = 这块存储一定会被复用), "
                                f"却**从不 Wait 自己的反依赖事件**。复用前不等上一代的消费者用完, "
                                f"就会在它还在被读时覆盖 —— 真机症状是 `507015 aicore exception` 或静默错数。"
                                f"★若你是**故意**用另一个 buffer 的旗标代管, 先证明两个池自始至终**同步轮转**: "
                                f"同一对池常被多条 matmul 以不同节奏共用 (FIAS 实撞: 切 K 里 A/B 各取一次, "
                                f"切 N 里 A 取 1 次而 B 取 4 次), 步调一旦不同, 代管的旗标就不成立。")))
        elif n_wait > 0 and n_set == 0:
            hz.append(dict(check="H17_pool_buf_no_wait", buffer=f"{name}<-{pool}", level="提示",
                           msg=(f"`{name}` 取自多缓冲池 {pool}, 有 Wait 而**没有配对的 Set** —— "
                                f"下一代取到同一块时会永远等不到, 真机挂死。")))
    return hz


def _loop_mask(lines: list) -> list:
    """标出每行是否在 for/while 体内。"""
    mask, stack, d = [False] * len(lines), [], 0
    for i, ln in enumerate(lines):
        if re.search(r"\b(for|while)\s*\(", ln):
            stack.append(d)
        mask[i] = bool(stack)
        d += ln.count("{") - ln.count("}")
        while stack and d <= stack[-1]:
            stack.pop()
    return mask



#  ---- H12: 循环外的初始置位缺失 -> 循环内第一次等待永远等不到 ------------------------------
#  H11 查的是循环**内**的收支; 但流水常见的写法是"循环外先置一次位(表示 buffer 空着), 循环内
#  先等后置"。整体重写某个函数时最容易连那一次初始置位一起删掉 —— 症状是真机挂死、驱动零输出、
#  没有任何报错, 与 H7/H11 抓的死锁形态完全一样却查不出来。
#  ★2026-08-28 实撞: 重写 GemmV3 的 RunVector 时把开头的 SetFlag<MTE3_MTE2>(EVENT_ID0) 一并删了,
#    循环内 Set/Wait 各一次仍然平衡, H11 报不出来, 花了一轮构建 + 一次真机挂死才定位。
#  判法: 循环内存在 WaitFlag<E>(id) 且**该等待位于同循环的 SetFlag<E>(id) 之前**时,
#    循环外必须有一次 SetFlag<E>(id) 打底。
#  ---- 变量 -> 可能的 event id 集合 -------------------------------------------------------
#  循环外的初始置位常写成字面量 `SetFlag<M_MTE1>(EVENT_ID0)`, 而循环内的等待用变量
#  `WaitFlag<M_MTE1>(eventFrag)` —— 按字符串比对会认不出这是同一个 buffer, 于是在**完全正确的
#  代码**上报"缺初始置位"。GemmV3 的 base kernel 上实撞三条误报。
#  这里把形如
#      event_t x = cond ? EVENT_ID0 : EVENT_ID1;
#      event_t x = static_cast<event_t>(EVENT_ID0 + (i & 0b11));
#  的赋值解析出来: 前者能确定取值集合 {ID0, ID1}; 后者带运行时算式, **静态确定不了**,
#  这种情况必须降级为提示而不是断言缺失 —— 判据宁可漏报也不能在正确代码上喊死锁。
_EVID = re.compile(r"\bEVENT_ID(\d)\b")
_EVDECL = re.compile(r"\bevent_t\s+(\w+)\s*=\s*([^;]+);")


def _event_var_values(code_only: str) -> tuple:
    """返回 (确定取值的变量表, 取值无法静态确定的变量名集合)."""
    exact, opaque = {}, set()
    for m in _EVDECL.finditer(code_only):
        var, rhs = m.group(1), m.group(2)
        ids = {f"EVENT_ID{d}" for d in _EVID.findall(rhs)}
        if not ids:
            opaque.add(var)
            continue
        #  只有算术**紧邻 EVENT_ID 本身**才说明 id 由运行时算出、字面量只是基址。
        #  三元表达式 `cond ? EVENT_ID0 : EVENT_ID1` 的条件里可以有任意算术,
        #  取值集合仍然是 {ID0, ID1} —— 早先按"右侧出现算术"一刀切, 把它误判成不可确定。
        if re.search(r"EVENT_ID\d\s*(?:[+\-*]|<<|>>)|(?:[+\-*]|<<|>>)\s*EVENT_ID\d", rhs):
            opaque.add(var)
        else:
            exact[var] = ids
    return exact, opaque


def _expand_ids(tok: str, exact: dict) -> set:
    """把一个 id 记号展开成它可能代表的字面量集合; 展不开就原样返回."""
    tok = tok.strip()
    return exact.get(tok, {tok})


def check_missing_prologue_setflag(src: str) -> list:
    code_only = strip_comments(src)
    lines = code_only.split("\n")
    in_loop = _loop_mask(lines)
    exact, opaque = _event_var_values(code_only)

    #  循环外的置位集合, 变量一律展开成它可能代表的字面量
    outer_sets = set()
    for i, ln in enumerate(lines):
        if in_loop[i]:
            continue
        for m in _SETF_ANY.finditer(ln):
            for eid in _expand_ids(m.group(2), exact):
                outer_sets.add((m.group(1), eid))

    #  循环内: 记录每个 (event, id记号) 第一次出现的是 Wait 还是 Set
    first_kind, first_line, first_tok = {}, {}, {}
    for i, ln in enumerate(lines):
        if not in_loop[i]:
            continue
        # VC split-pack E73: Set;Wait on one line must retain column order.
        # Scanning all waits first turns a valid local handoff into a deadlock.
        events = [(m.start(), "Wait", m) for m in _WAITF_ANY.finditer(ln)]
        events += [(m.start(), "Set", m) for m in _SETF_ANY.finditer(ln)]
        for _, kind, m in sorted(events, key=lambda item: item[0]):
            k = (m.group(1), m.group(2).strip())
            if k not in first_kind:
                first_kind[k], first_line[k], first_tok[k] = kind, i + 1, m.group(2).strip()

    # A dynamic cell and a different cell of the same event array may alias.
    # Without index/call-graph analysis this is unknown, not a proven deadlock.
    array_sets=[]
    for m in _SETF_ANY.finditer(code_only):
        cell=re.fullmatch(r"(\w+)\s*\[\s*([^]\n]+)\s*\]",m.group(2).strip())
        if cell:array_sets.append((m.group(1),cell.group(1),cell.group(2).strip()))
    hz = []
    for k, kind in first_kind.items():
        if kind != "Wait":
            continue
        ev, tok = k
        want = _expand_ids(tok, exact)
        have = {eid for eid in want if (ev, eid) in outer_sets}
        if have == want:                      # 每个可能取值都有循环外置位 -> 合法
            continue
        cell=re.fullmatch(r"(\w+)\s*\[\s*([^]\n]+)\s*\]",tok)
        if cell:
            array,index=cell.group(1),cell.group(2).strip()
            uncertain=[other for event,name,other in array_sets if event==ev and name==array
                       and other!=index and (not index.isdigit() or not other.isdigit())]
            if uncertain:
                hz.append(dict(check="H12_prologue_unverifiable",buffer=f"{ev}({tok})",level="提示",
                    msg=(f"L{first_line[k]}: {tok} 首次为Wait，同数组还有不同cell的Set {sorted(set(uncertain))}。"
                         "动态索引/跨函数置位尚未求解，无法证明初始许可齐备或缺失；需沿实际槽位执行序列"
                         "核对每个Set/Wait和阶段末净收支，保留原始记录与负例回放。无错误级命中不代表安全。")))
                continue
        if tok in opaque:
            #  id 由运行时算出 (如 EVENT_ID0 + (i & 0b11)), 静态定不了取值集合。
            #  此时既不能断言缺失, 也不能放过 —— 报成提示, 把要人工核对的点说清楚。
            hz.append(dict(check="H12_prologue_unverifiable", buffer=f"{ev}({tok})", level="提示",
                           msg=(f"L{first_line[k]}: 循环内先 WaitFlag<{ev}>({tok}) 后 SetFlag, 而 {tok} 的 "
                                f"event id 由运行时算出 (右侧含算术), **静态确定不了它会取哪几个值**, "
                                f"所以判据无法证明循环外的初始置位是否配齐。请人工核对: 循环外的 "
                                f"SetFlag<{ev}> 条数必须等于该变量实际轮转的路数 —— 四路轮转就要四个, "
                                f"少一个就是第一次等待永远等不到, 真机挂死且零输出、无报错。")))
            continue
        missing = sorted(want - have)
        detail = (f"(该变量可能取 {sorted(want)}, 其中 {missing} 在循环外没有置位)"
                  if tok in exact else "")
        hz.append(dict(check="H12_missing_prologue_setflag", buffer=f"{ev}({tok})", level="错误",
                       msg=(f"L{first_line[k]}: 循环内先 WaitFlag<{ev}>({tok}) 后 SetFlag, 但**循环外没有**"
                            f"对应的初始 SetFlag{detail} —— 第一次等待永远等不到, 真机挂死且驱动零输出、"
                            f"无报错。流水写法的惯例是循环外先置一次位表示\"这块 buffer 空着\"; "
                            f"整体重写某个函数时最容易把它连带删掉。")))
    return hz


#  ---- H19: UB 张量被 MTE2 写入后由 MTE3 读出, 中间一条同步链都没有 -------------------------
#  `DataCopy(ubT, xxxGm[...])` 走 MTE2 把 GM 数据填进 UB; 随后 `DataCopy(yyyGm[...], ubT)` 走 MTE3
#  把同一块 UB 写回 GM。两条**不同流水**之间没有硬件互锁, 必须有旗标链:
#      SetFlag/WaitFlag<MTE2_MTE3>        或      WaitFlag<MTE2_V> ... SetFlag/WaitFlag<V_MTE3>
#  (后者是"中间夹了向量运算"时的常见写法, 靠 V 传递次序)。缺了这条链, MTE3 可能在 MTE2 填完之前
#  就开始读, 且**同一块 UB 往往还被别的阶段复用**, 于是污染会传到看起来毫不相干的输出上。
#  ★2026-09-15 sparse_flash_attention_grad 实撞, 找了五个单变量构建才定位:
#    `ScatterRowsToGm` 里那对 V_MTE3 被 `if constexpr (WRITE_DV)` 包着, KV merge 下不发 ——
#    出厂靠**调用方**在 `Add(dk, dv)` 之后补一次。把 dK/dV 合并成一次矩阵乘之后那个 Add 没了,
#    链就断了。症状是**随机**整 token NaN (每趟受害集合不同), 而且因为该 UB 还被 dS 输出复用,
#    连不经过 scatter 的 dq 都会 NaN, 极易误判成"合并算错了"。
#  ★判据分两级: 消费者是本函数内的直接 DataCopy 写 GM -> 错误; 消费者是把该 UB 传进
#    `*ToGm` / `*CopyOut` / `Scatter*` 这类调用 -> 只给提示 (同步可能在被调函数里, 静态看不见,
#    但这正是上面那个坑的形状, 值得去看一眼)。
#  ★**抓不住的 (必须写明, 否则会被当成'扫过就没问题')**: 旗标链**文本上存在但被条件包住**时,
#    本判据按文本判定会认为链在, 于是放过。实测: 上面那次实撞的 `ScatterAddDeter` 就是这一形态
#    (`if (!cubeMerged) { ... SetFlag<V_MTE3> ... }`), H19 没抓到; 抓到的是同一份文件里
#    `ScatterAddHead64` 那处**整段删掉**的。要覆盖条件形态需要控制流分析, 成本远高于本判据。
#  ★误报面 (2026-09-15 实测): 扫 ops-transformer 的 60 个 arch35 kernel, 只在 1 个文件上命中 3 条,
#    其余 59 个零命中 —— 判据够窄, 不是狼来了。
_H19_IN = re.compile(r"\bDataCopy(?:Pad)?\s*\(\s*(\w+)\s*(?:\[[^\]]*\])?\s*,\s*(\w*[Gg]m)\b")
_H19_OUT = re.compile(r"\bDataCopy(?:Pad)?\s*\(\s*(\w*[Gg]m)\s*(?:\[[^\]]*\])?\s*,\s*(\w+)\b")
_H19_CALLOUT = re.compile(r"\b(\w*(?:ToGm|CopyOut|Scatter\w*|StoreToGm))\s*<[^;{}]*>\s*\(([^;]*)\)|"
                          r"\b(\w*(?:ToGm|CopyOut|Scatter\w*|StoreToGm))\s*\(([^;]*)\)")
#  ★同步链有两种写法, 判据必须都认: 模板 `SetFlag<HardEvent::MTE2_MTE3>(id)` 与
#  宏 `SET_FLAG(MTE2, MTE3, id)`。2026-09-16 实撞: 只认模板那种, 于是在一个大量用宏的
#  算子上报出 **18 条全是误报** —— 那些位置的 SET_FLAG(MTE2, MTE3, ...) 就写在写出点前面。
#  代价不是这 18 条本身, 而是**真信号被淹没**: 我连着几轮把 H19 当"既有告警"跳过,
#  直觉上知道它不对却没去验 —— 判据过宽就是狼来了, 哪天它报真问题也会被同样跳过。
_H19_CHAIN_DIRECT = re.compile(
    r"(?:Set|Wait)Flag<\s*(?:AscendC::)?HardEvent::MTE2_MTE3\s*>"
    r"|(?:SET|WAIT)_FLAG\s*\(\s*MTE2\s*,\s*MTE3\s*,")
_H19_CHAIN_V1 = re.compile(
    r"WaitFlag<\s*(?:AscendC::)?HardEvent::MTE2_V\s*>"
    r"|WAIT_FLAG\s*\(\s*MTE2\s*,\s*V\s*,")
_H19_CHAIN_V2 = re.compile(
    r"(?:Set|Wait)Flag<\s*(?:AscendC::)?HardEvent::V_MTE3\s*>"
    r"|(?:SET|WAIT)_FLAG\s*\(\s*V\s*,\s*MTE3\s*,")


def _h19_has_chain(seg: str) -> bool:
    if _H19_CHAIN_DIRECT.search(seg):
        return True
    m1 = _H19_CHAIN_V1.search(seg)
    if m1 and _H19_CHAIN_V2.search(seg, m1.end()):
        return True
    return "PipeBarrier<PIPE_ALL>" in seg


def check_ub_mte2_to_mte3_chain(src: str) -> list:
    """H19: 同一块 UB 上 MTE2 生产 -> MTE3 消费, 中间没有 MTE2->MTE3 或 MTE2->V->MTE3 的旗标链。"""
    hz = []
    code = strip_comments(src)
    #  split_funcs 按 `::Name(` 切, 只认成员函数定义; 自由函数 / 片段切不出来 —— 那时整份当一段。
    funcs = split_funcs(code) or [("(整份源码)", code)]
    for fname, body in funcs:
        for mi in _H19_IN.finditer(body):
            ub = mi.group(1)
            if ub.endswith("Gm") or ub.endswith("gm"):
                continue                      # GM->GM 不在本判据范围
            rest = body[mi.end():]
            #  同一 UB 的下一个消费点: 直接写 GM (错误) 还是传进 *ToGm/Scatter* 调用 (提示)
            direct, callish = None, None
            for mo in _H19_OUT.finditer(rest):
                if mo.group(2) == ub:
                    direct = mo.start(); break
            for mc in _H19_CALLOUT.finditer(rest):
                args = mc.group(2) or mc.group(4) or ""
                if re.search(r"\b" + re.escape(ub) + r"\b", args):
                    callish = mc.start(); break
            pick = min([x for x in (direct, callish) if x is not None], default=None)
            if pick is None:
                continue
            if _h19_has_chain(rest[:pick]):
                continue
            line = body[:mi.start()].count("\n") + 1
            is_direct = (pick == direct)
            hz.append(dict(
                check="H19_ub_mte2_to_mte3_no_chain",
                buffer=f"{fname}:{ub}", level=("错误" if is_direct else "提示"),
                msg=(f"函数 {fname} 内 L{line} 起: UB 张量 `{ub}` 由 MTE2 (DataCopy 自 *Gm) 填入, "
                     + ("随后被 MTE3 (DataCopy 写 *Gm) 读出" if is_direct
                        else "随后被传进 *ToGm/Scatter* 一类调用 (可能在被调函数里写 GM)")
                     + ", 中间没有 MTE2->MTE3、也没有 MTE2->V->MTE3 的旗标链。"
                     + ("两条流水之间没有硬件互锁, MTE3 可能读到没填完的 UB。"
                        if is_direct else
                        "同步可能在被调函数里, 但**被调函数里的同步常被 if constexpr 关掉** "
                        "(出厂靠调用方补), 删掉调用方那段时链就断了 —— 去确认一眼。")
                     + " 症状是随机、每趟受害集合不同的脏数据; 若该 UB 还被别的阶段复用, "
                       "污染会传到看起来不相干的输出上。")))
    return hz


#  ---- H13: SetFlag<A_B> 之前, A 流水上没有任何操作 -> 该旗标永远发不出去 -------------------
#  `SetFlag<HardEvent::A_B>(id)` 的语义是"等 A 流水上已下发的操作做完, 再置位给 B"。
#  若 A 流水上一条操作都没有, 这个置位就永远不会真正发出, 等它的一侧死等 —— 真机挂死、零输出、无报错。
#  跨核的 `CrossCoreSetFlag<_, PIPE_A>` 同理。
#  ★2026-08-28 一轮里撞了三次, 每次都以为是别的原因:
#    (a) cube 侧 early-return 分支用 CrossCoreSetFlag<.., PIPE_FIX> 发通知, 而那条路径没有 fixpipe;
#    (b) 二分调试删掉全部向量计算后, SetFlag<V_MTE3> 等不到 V 流水;
#    (c) 再删掉 CopyOut 后, SetFlag<MTE3_MTE2> 等不到 MTE3 流水。
#  后两次是**二分调试自己制造的**, 一度让人以为原版有问题 —— 所以这条判据不只防写错,
#  也防"删代码做二分"时踩空。
_PIPE_OPS = {
    "MTE1": ("LoadData", "LoadCbufTo", "l1_to_l0"),
    "MTE2": ("DataCopy", "CopyGmTo", "gm_to_", "MOV_OUT_TO_L1"),
    "MTE3": ("DataCopy", "CopyUbufTo", "ub_to_gm", "CopyCcToGm"),
    "V":    ("Cast", "Muls", "Axpy", "Add", "Sub", "Mul", "Div", "Max", "Min", "Exp", "Duplicate", "Brcb"),
    "M":    ("Mmad", "Iterate"),
    "FIX":  ("Fixpipe", "CopyCcToGm", "GetTensorC", "IterateAll"),
}
_SETF_EV = re.compile(r"SetFlag<\s*(?:AscendC::)?HardEvent::(\w+?)_(\w+?)\s*>")
_CCSETF_PIPE = re.compile(r"CrossCoreSetFlag<[^,]*,\s*PIPE_(\w+)\s*>")


def check_setflag_pipe_has_work(src: str) -> list:
    """**只查一种明确模式**: 早退分支内发跨核通知, 而该分支上没有对应流水的操作。

    ★为什么只查这一种 (2026-08-28 自测): 先写成"函数级 —— SetFlag<A_B> 所在函数里要有 A 流水操作",
      在正确的 gemm_v3 kernel 上误报 11 条。成因是 split_funcs 按 `::Name(` 切, 会把
      `AscendC::Mmad(` 这类调用误当函数边界, 切出的块太碎。要准确判需要控制流分析, 成本过高。
      而判据过宽就是狼来了 —— 所以收窄到误报为零的那个模式:
          if (<条件>) { ... CrossCoreSetFlag<_, PIPE_X>(...); return; }
      这正是实撞的第一种情况: 没分到活的核走早退分支, 仍要给对侧发通知, 而那条路径上一个
      X 流水的操作都没有, 通知永远发不出去, 对侧死等。
    抓不住的 (必须写明): 循环体内被条件跳过的流水操作、跨函数的情形、以及二分调试删代码时
      把某条流水删空 —— 后者只能靠"删代码时同时删掉配对的 flag"这条纪律。
    """
    code_only = strip_comments(src)
    hz = []
    #  找 `{ ... CrossCoreSetFlag<_,PIPE_X>(..); ... return; }` 的最小块
    for m in _CCSETF_PIPE.finditer(code_only):
        pipe = m.group(1)
        ops = _PIPE_OPS.get(pipe)
        if not ops:
            continue
        #  向前找最近的 `{`, 向后找最近的 `}`; 块内必须有 return 才算早退分支
        lb = code_only.rfind("{", 0, m.start())
        rb = code_only.find("}", m.end())
        if lb < 0 or rb < 0:
            continue
        block = code_only[lb:rb]
        if "return" not in block:
            continue
        if any(o in block for o in ops):
            continue
        line = code_only[:m.start()].count("\n") + 1
        hz.append(dict(check="H13_setflag_pipe_idle", buffer=f"PIPE_{pipe}", level="错误",
                       msg=(f"L{line}: 早退分支内 `CrossCoreSetFlag<_, PIPE_{pipe}>` —— 该分支上找不到任何 "
                            f"{pipe} 操作 ({'/'.join(ops[:3])}...)。这个置位要等 {pipe} 流水上已下发的操作做完"
                            f"才真正发出, 而这条路径一个都没有, 于是永远不发, 对侧死等: 真机挂死、零输出、无报错。"
                            f"改法: 让这条路径也走一次该流水的操作, 或从根上不产生空闲分支 "
                            f"(例如把 blockDim 设成正好有活干的核数)。")))
    return hz

def _c220_role_sources(src):
    """Explicit C220 preprocessor roles only, preserving source offsets.

    This is not a C++ preprocessor: unknown conditions retain the enclosing
    role. Runtime ASCEND_IS_AIV branches and helper call graphs are out of scope.
    """
    role_text = {"cube": [], "vec": []}
    stack = []
    for line in strip_comments(src).splitlines(keepends=True):
        directive = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", line)
        if directive:
            kind, condition = directive.groups()
            roles = {r for r, macro in (("cube", "__DAV_C220_CUBE__"), ("vec", "__DAV_C220_VEC__"))
                     if re.search(rf"\b{macro}\b", condition)}
            role = next(iter(roles)) if len(roles) == 1 else None
            if kind == "ifndef" or (kind in ("if", "elif") and re.search(r"!\s*defined", condition)):
                role = {"cube": "vec", "vec": "cube"}.get(role)
            if kind in ("if", "ifdef", "ifndef"):
                stack.append(role)
            elif stack and kind == "elif":
                stack[-1] = role
            elif stack and kind == "else":
                stack[-1] = {"cube": "vec", "vec": "cube"}.get(stack[-1])
            elif stack and kind == "endif":
                stack.pop()
        active = next((x for x in reversed(stack) if x), None)
        blank = "".join("\n" if x == "\n" else " " for x in line)
        for role in role_text:
            role_text[role].append(line if not directive and active == role else blank)
    return {role: "".join(lines) for role, lines in role_text.items()}


def check_mix_vector_core_index(src):
    """H32 advisory: identical paired loops starting from two raw block indices.

    C220 AIV GetBlockIdx includes the subblock. Only direct assignments in the
    loop's immediate lexical scope and explicitly guarded C220 roles are read.
    An intentional lane schedule may be valid: report a review, never an error.
    """
    sites = {"cube": {}, "vec": {}}
    for role, code in _c220_role_sources(src).items():
        for loop in re.finditer(r"\bfor\s*\(", code):
            control, end = _extract_paren(code, loop.end() - 1)
            origin = re.search(r"=\s*(\w+)\s*$", control.split(";")[0])
            if not origin:
                continue
            alias = origin.group(1)
            # Limit alias resolution to this loop's immediate lexical scope.
            braces = []
            for i, ch in enumerate(code[:loop.start()]):
                if ch == "{": braces.append(i)
                elif ch == "}" and braces: braces.pop()
            begin = braces[-1] + 1 if braces else 0
            definitions = list(re.finditer(rf"\b{re.escape(alias)}\s*=\s*([^;]+);", code[begin:loop.start()]))
            if not definitions or not re.fullmatch(r"(?:AscendC::)?GetBlockIdx\s*\(\s*\)", definitions[-1].group(1).strip()):
                continue
            tail = code[begin + definitions[-1].end():loop.start()]
            if re.search(rf"\b{re.escape(alias)}\s*(?:[-+*/%&|^]=|\+\+|--)", tail):
                continue  # A later mutation needs dataflow analysis, outside this direct-alias check.
            body_start = end + 1
            while body_start < len(code) and code[body_start].isspace(): body_start += 1
            if body_start >= len(code) or code[body_start] != "{":
                continue
            depth, body_end = 1, body_start + 1
            while body_end < len(code) and depth:
                if code[body_end] == "{": depth += 1
                elif code[body_end] == "}": depth -= 1
                body_end += 1
            body = code[body_start:body_end]
            for flag in re.finditer(r"CrossCore(Set|Wait)Flag\s*(?:<[^>]*>)?\s*\(", body):
                channel = _cc_flag_channel(body, flag.end() - 1)
                key = (re.sub(r"\s+", "", control), channel, flag.group(1))
                sites[role].setdefault(key, code[:body_start+flag.start()].count("\n") + 1)
    hazards = []
    for (control, channel, kind), line in sorted(sites["vec"].items()):
        counterpart = (control, channel, "Wait" if kind == "Set" else "Set")
        if counterpart not in sites["cube"]:
            continue
        hazards.append(dict(check="H32_mix_vector_core_index", level="提示", line=line, buffer=channel,
            msg=(f"C220两侧通道 '{channel}' 的相同循环均直接从 GetBlockIdx() 别名起步。"
                 "MIX子核比>1时AIV编号包含subblock，和Cube核号不同；若任务按Cube配对，"
                 "需以 GetBlockIdx()/GetSubBlockNum() 等逻辑核号映射复核循环和共享地址。"
                 "合法的独立lane调度也可能使用raw编号，本条仅提示复核，不证明死锁或替代设备验证。")))
    return hazards


def check_crosscore_flags(src):
    def enclosing_loops(pos):
        """从 pos 往前扫, 用花括号配平找出包含该位置的循环头 (最内两层)."""
        depth, i, loops = 0, pos, []
        while i > 0 and len(loops) < 2:
            c = src[i]
            if c == "}":
                depth += 1
            elif c == "{":
                if depth == 0:
                    head = src[max(0, i - 200):i]
                    m = re.search(r"(for|while)\s*\(([^{]*)$", head, re.S)
                    if m:
                        loops.append(" ".join(m.group(2).split())[:90])
                else:
                    depth -= 1
            i -= 1
        return loops

    sites = {}
    for m in re.finditer(r"CrossCore(Set|Wait)Flag\s*(?:<[^>]*>)?\s*\(", src):
        kind = m.group(1)
        flag = _cc_flag_channel(src, m.end() - 1)
        sites.setdefault(flag, {"Set": [], "Wait": []})[kind].append(
            (src[:m.start()].count("\n") + 1, enclosing_loops(m.start())))
    hz = []
    for flag, d in sorted(sites.items()):
        if not d["Set"] or not d["Wait"]:
            continue
        set_ctl = {l for _, ls in d["Set"] for l in ls}
        wait_ctl = {l for _, ls in d["Wait"] for l in ls}
        if set_ctl != wait_ctl:
            hz.append(dict(check="H6_crosscore_flag_imbalance", buffer=flag,
                           msg=(f"标志 '{flag}' 的置位与等待处于不同的循环控制下 —— 置位: "
                                f"{sorted(set_ctl) or ['(无循环)']} (行 {[l for l, _ in d['Set']]}); "
                                f"等待: {sorted(wait_ctl) or ['(无循环)']} (行 {[l for l, _ in d['Wait']]}). "
                                f"两侧次数相等依赖隐式不变量, 改 tiling 的 single*/base*/step* 会打破它, "
                                f"症状是真机 aicore timeout 507014 (核挂死, 不报错). "
                                f"改 tiling 前必须复核这两个循环的次数是否仍相等.")))
    # ---- H7: **同一控制上下文**下置位与等待的次数不相等 -> 不是"依赖不变量", 是直接死锁.
    #      H6 只比循环上下文, 比不了次数; 2026-08-17 撞的那个 bug 两侧都在"无循环"里, H6 看不见:
    #        AIV 侧 CrossCoreSetFlag<2,...>(SYNC_RETRY_V2C) 写了 1 次
    #        AIC 侧 CrossCoreWaitFlag(SYNC_RETRY_V2C)       连写了 2 次
    #      当时的错误推理是"一个 AIC 配两个 AIV, 所以要等两次"。实际 CrossCoreSetFlag<2,...> 是
    #      **组语义**: 一对 AIV 的两次置位合成 AIC 可消费的**一个**。依据是既有协议
    #      SYNC_AIV_TO_AIC 就是"每个 AIV 各发一次、AIC 每块等一次"。代价: 一次构建 + 一次挂死
    #      (真机 aicore timeout 507014) + 一轮定位。本判据把它压到 build 前一条命令。
    for flag, d in sorted(sites.items()):
        if not d["Set"] or not d["Wait"]:
            continue
        from collections import Counter
        sc = Counter(tuple(ls) for _, ls in d["Set"])
        wc = Counter(tuple(ls) for _, ls in d["Wait"])
        for ctx in set(sc) & set(wc):
            if sc[ctx] == wc[ctx]:
                continue
            side = "Wait" if wc[ctx] > sc[ctx] else "Set"
            lines = sorted(l for l, ls in d[side] if tuple(ls) == ctx)
            if _mutually_exclusive(src, lines):
                continue          # 同一 if-else 链的不同分支, 运行时只走一条
            adjacent = len(lines) > 1 and (max(lines) - min(lines)) <= 4
            hz.append(dict(check="H7_crosscore_flag_count", buffer=flag, level=("错误" if adjacent else "提示"),
                           msg=(f"标志 '{flag}' 在**同一控制上下文** {list(ctx) or ['(无循环)']} 下, "
                                f"置位 {sc[ctx]} 次 / 等待 {wc[ctx]} 次 —— 次数不相等即死锁 "
                                f"(真机 aicore timeout 507014, 核挂死不报错)。"
                                f"多出的一侧在行 {lines}{' (相邻, 极可能是手写重复)' if adjacent else ''}。"
                                f"★CrossCoreSetFlag<2,...> 是**组语义**: 一对 AIV 的两次置位合成 AIC "
                                f"可消费的**一个**, 不要按'一个 AIC 配两个 AIV'去乘 2。"
                                f"对照既有协议 SYNC_AIV_TO_AIC: 每个 AIV 各发一次、AIC 每块等一次。")))
    return hz


# ---- H20: **孤儿补偿循环** —— 真实站点已被删除, 只剩空转凑计数的循环还在发旗标.
#      形态特征: 一个 for/while 循环, 循环体除 CrossCoreSetFlag/CrossCoreWaitFlag 之外
#      没有任何语句 —— 它不承载数据依赖, 唯一作用是让本核的收发次数与别的核对齐
#      (典型写法 `if (本核任务数 < 最大任务数) for (...) { Set(F); Wait(F); }`).
#      为什么 H7 抓不住: 补偿循环自己发自己等, 同一控制上下文下次数天然相等, 账面是平的.
#      2026-09-15 sparse_flash_attention_grad 实撞: 确定性 scatter 改成每核独立后删掉了
#      SCATTER_VECTOR_SYNC_FLAG 的全部真实站点, 两处补偿循环没跟着删 -> 真机 507014
#      aicore timeout. 代价: 一次构建 + 一次挂死 + 一轮定位.
#      规则: 删一个跨核旗标必须**成对**删 —— 真实站点与补偿站点一起去.
def check_orphan_compensation_loop(src: str) -> list:
    code = strip_comments(src)
    _CALL = re.compile(r"(?:AscendC::)?CrossCore(Set|Wait)Flag\s*(?:<[^>]*>)?\s*\([^;]*;")
    comp_sites = {}          # flag -> [行号]
    comp_spans = []          # (起, 止) 纯补偿循环在 code 里的字节区间
    for m in re.finditer(r"\b(for|while)\s*\(", code):
        # 括号配平找循环头的右括号
        i, d = m.end() - 1, 0
        while i < len(code):
            if code[i] == "(":
                d += 1
            elif code[i] == ")":
                d -= 1
                if d == 0:
                    break
            i += 1
        j = i + 1
        while j < len(code) and code[j] in " \t\r\n":
            j += 1
        if j >= len(code) or code[j] != "{":
            continue          # 无花括号的单语句循环体, 不是补偿循环的写法
        k, d = j, 0
        while k < len(code):
            if code[k] == "{":
                d += 1
            elif code[k] == "}":
                d -= 1
                if d == 0:
                    break
            k += 1
        body = code[j + 1:k]
        calls = list(_CALL.finditer(body))
        if not calls:
            continue
        # 循环体去掉这些调用后若只剩空白, 即"空转"
        rest = _CALL.sub("", body).strip()
        if rest:
            continue
        comp_spans.append((j, k))
        for c in calls:
            comp_sites.setdefault(
                _cc_flag_channel(body, body.index("(", c.start())), []).append(
                code[:j + 1 + c.start()].count("\n") + 1)

    hz = []
    for flag, lines in sorted(comp_sites.items()):
        real = []
        for c in _CALL.finditer(code):
            if _cc_flag_channel(code, code.index("(", c.start())) != flag:
                continue
            if any(a <= c.start() <= b for a, b in comp_spans):
                continue
            real.append(code[:c.start()].count("\n") + 1)
        if real:
            continue          # 还有真实站点, 补偿循环是配套的, 正常
        hz.append(dict(check="H20_orphan_compensation_loop", buffer=flag, level="错误",
                       line=min(lines),
                       msg=(f"标志 '{flag}' 只剩**空转补偿循环**在收发 (行 {sorted(set(lines))}), "
                            f"全文没有任何一处承载真实数据依赖的站点。补偿循环的次数是按"
                            f"'别的核要发多少次'算的, 而别的核已经不发了 —— 真机 aicore "
                            f"timeout 507014 (核挂死, 不报错)。"
                            f"★删一个跨核旗标必须**成对**删: 真实站点与补偿站点一起去。"
                            f"若该旗标的真实站点在**另一个文件**, 本判据会误报 —— "
                            f"先 grep 全仓确认, 再决定删不删。")))
    return hz


# ---- H8: 清零/初始化的范围**跨了区段** -> 与别的核正在写的区段撞车 (真机: 输出近似全 0 + det_bit=False)
#      形态特征: InitOutput/Duplicate 的元素数由**两个区段偏移相减**算出 (如
#      (boundOffsetBytes - fpAccOffsetBytes)/4 + ...), 说明这一趟清零横跨了多个逻辑区段。
#      危险在于: 清零函数与随后写这些区段的函数之间**往往没有 SyncAll** —— 核 A 已在写自己
#      那段 scol, 核 B 还在清零把它抹掉, 抹多少随时序变。
#      2026-08-17 gmfr 实撞: 为了顺带清界累加器, 把 ZeroFpAcc 的范围从 batch*n 扩成
#      "(boundOffset-fpAccOffset)/4 + batch*G", 于是连 scol/invScol 一起清了; 而
#      ZeroFpAcc() 与紧随的 ComputeScolVec() 之间没有屏障 -> 46 例 UT 掉到 32/46,
#      两次运行 100% 元素不同、中位相对差 ~1.0 (即一次近似全 0)。
#      修法: 按区段分多次清 (gmfr 改成 ZeroRange(acc) + ZeroRange(bound), 跳过中间的 scol/invScol)。
# ---- H22: **全局原子模式的作用域** —— SetAtomicAdd<T>() 之后, 该作用域内被写的
#      每一个 GlobalTensor 的元素类型都必须是 T。原子模式是**全局状态**, 不是某次
#      DataCopy 的参数: 为一个输出改了它, 同作用域内别的输出会被一起改掉。
#      2026-09-15 sparse_flash_attention_grad 实撞: ScatterAddDeter 里把 dk 的累加改成
#      `SetAtomicAdd<int32_t>()` 走定点, 而同一作用域内还有 `DataCopy(dvOutGm, dvInUb)`,
#      dvOutGm 是 GlobalTensor<float> —— dv 的 fp32 数据被按 int32 原子加,
#      KV_MERGE=0 且 det=1 时 dv **99.95% 元素不同**。KV merge (目标场景) 下 dv 根本不写,
#      所以从 v3a 一直到验证矩阵首跑才暴露。
#      ★这是同一类失败的第三次 (另两次: 写方被模板条件关掉而读方没跟上 = hangdiag C5;
#      因子前移只撤了一条通路上的补偿 = hangdiag C6)。三者的共同形态是
#      "改动落在**共享位置**, 只考虑了自己走的那条分支"。
def _cx_branch(code: str, pos: int) -> dict:
    """求 pos 处所在的 `if constexpr` 分支路径: {条件名: 取该支时为 True/False}。

    ★为什么需要它: 同一个条件的两个**独立** if constexpr 块 (一个管 SetAtomicAdd,
    一个管 DataCopy) 在结构上不是同一条 if-else 链, `_mutually_exclusive` 认不出来,
    于是把"分别在两个块的对应分支里"这种**正确**写法报成错。2026-09-15 修 sfag 的
    dv 缺陷时当场撞到: 改对了反而被判据拦下。
    """
    path, i, depth = {}, pos, 0
    while i > 0:
        c = code[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                head = code[max(0, i - 260):i]
                m = re.search(r"if\s+constexpr\s*\(\s*(!?)\s*([A-Za-z_]\w*)\s*\)\s*$", head)
                if m:
                    path.setdefault(m.group(2), m.group(1) != "!")
                elif re.search(r"\belse\s*$", head):
                    #  else 支: 往前找与之配对的 if constexpr
                    j, d2 = i - 1, 0
                    while j > 0:
                        if code[j] == "}":
                            d2 += 1
                        elif code[j] == "{":
                            d2 -= 1
                            if d2 == 0:
                                h2 = code[max(0, j - 260):j]
                                m2 = re.search(r"if\s+constexpr\s*\(\s*(!?)\s*([A-Za-z_]\w*)\s*\)\s*$", h2)
                                if m2:
                                    path.setdefault(m2.group(2), m2.group(1) == "!")
                                break
                        j -= 1
            else:
                depth -= 1
        i -= 1
    return path


def _cx_incompatible(code: str, a: int, b: int) -> bool:
    """两个位置是否处在同一 constexpr 条件的**相反**分支 (即不可能同时存在)。"""
    pa, pb = _cx_branch(code, a), _cx_branch(code, b)
    return any(k in pb and pa[k] != pb[k] for k in pa)


def check_atomic_mode_scope(src: str) -> list:
    #  ★按**函数**分段处理, 不在全文件里收集类型: 同一个名字在不同函数里可以是不同类型。
    #  2026-09-15 样本外检验实撞: `dkOutGm` 在 ScatterAddUnDeter 里是 GlobalTensor<float>、
    #  在 ScatterAddDeter 里是 GlobalTensor<int32_t>, 全文件收集会把两者混起来,
    #  于是给 det=0 那条**正确**的路径报一条不存在的错。判据的自检当时是全过的 ——
    #  这类混淆只有拿真实源码跑才露出来。
    _code = strip_comments(src)
    _parts = [b for _n, b in split_funcs(_code)]
    if not _parts:
        #  split_funcs 按 `::Name(` 切; 切不出来 (自由函数、片段) 就按整段算,
        #  否则判据会在这类输入上静默失效 —— 那比误报更坏。
        _parts = [_code]
    hz = []
    for _body in _parts:
        hz += _atomic_mode_scope_one(_body)
    return hz


def _atomic_mode_scope_one(code: str) -> list:
    gt = {}
    for m in re.finditer(r"GlobalTensor\s*<\s*([A-Za-z_]\w*)\s*>\s+(\w+)", code):
        gt[m.group(2)] = m.group(1)
    _SET = re.compile(r"SetAtomicAdd\s*<\s*([A-Za-z_]\w*)\s*>\s*\(")
    _NONE = re.compile(r"SetAtomic(?:None|Add\s*<)")
    _COPY = re.compile(r"\bDataCopy(?:Pad)?\s*\(\s*(\w+)\s*[\[(,]")
    hz = []
    for m in _SET.finditer(code):
        mode = m.group(1)
        #  作用域: 到下一个 SetAtomicNone / 下一个 SetAtomicAdd 为止
        nxt = len(code)
        for nm in _NONE.finditer(code, m.end()):
            nxt = nm.start()
            break
        seg = code[m.end():nxt]
        bad = {}
        for cm in _COPY.finditer(seg):
            dst = cm.group(1)
            t = gt.get(dst)
            if t is None or t == mode:
                continue
            #  ReinterpretCast 到 mode 的写是刻意的, 不算
            tail = seg[cm.start():cm.start() + 200]
            if f"ReinterpretCast<{mode}>" in tail.replace(" ", ""):
                continue
            _cl = code[:m.end() + cm.start()].count("\n") + 1
            _sl = code[:m.start()].count("\n") + 1
            #  ★互斥分支不算: `if constexpr (C) { SetAtomicAdd<int32_t>(); ... 写 int32 }
            #  else { SetAtomicAdd<float>(); ... 写 fp32 }` 是**正确**的写法, 两支不会同时存在。
            #  2026-09-15 修 dv 缺陷时就是这么改的, 判据当场把改对的代码报成错 ——
            #  复用 H7 那条已有的互斥判断 (_mutually_exclusive), 不另造一套。
            if _mutually_exclusive(code, [_sl, _cl]) or \
                    _cx_incompatible(code, m.start(), m.end() + cm.start()):
                continue
            bad.setdefault(f"{dst}<{t}>", _cl)
        for nm, ln in sorted(bad.items()):
            hz.append(dict(check="H22_atomic_mode_scope", level="错误", line=ln,
                           buffer=f"SetAtomicAdd<{mode}> -> {nm}",
                           msg=(f"`SetAtomicAdd<{mode}>()` 的作用域内写了 `{nm}`, 元素类型与原子模式"
                                f"不一致。**原子模式是全局状态, 不是某次 DataCopy 的参数** —— "
                                f"为一个输出改了它, 同作用域内别的输出会被一起改掉, "
                                f"结果是那个输出的数据按错误的类型做原子加: 编译通过、不报错、只算错。"
                                f"★2026-09-15 实撞: dk 改走 int32 定点后, 同作用域内 dv 的 fp32 写"
                                f"被按 int32 原子加, KV_MERGE=0 且 det=1 时 dv 99.95% 元素不同; "
                                f"而目标场景 (KV merge) 下 dv 根本不写, 所以拖了三个版本才暴露。"
                                f"改法: 要么让两个输出走同一种模式 (如 dv 也定点), "
                                f"要么在切换处夹 SetAtomicNone 分隔两段作用域。")))
    return hz


# ---- H23: **手工分配的 L1 常驻块被覆写却没有 WAR 保护** —— 结果随时序变化, 精度可能仍过。
#      形态: 用 GetBuffer<BufferType::ASCEND_CB>(offset) 拿到一块 L1, 往它 CopyGmToL1,
#      而该写入点之前没有 WaitFlag<MTE1_MTE2> 之类"等上一次读完"的保护。
#      为什么 H17 抓不到: H17 只认**从池取**的 buffer (BuffersPolicy*/AllocBuffer),
#      认不出手工 GetBuffer 拿到的常驻块。
#      2026-09-15 sparse_flash_attention_grad 实撞: 把 selected kv 搬进 L1 常驻块供
#      mm1/mm2 共用, 省掉 mm2 的一次 GM 读。**四项精度全过而逐位确定性 False** ——
#      Mmad 发出后不等完成, cube2(task N) 还在读常驻块时 cube1(task N+1) 已开始覆写,
#      竞争窗口小所以偏差落在容差内, 只有逐位比对才看得出来。
#      ★这类缺陷最隐蔽的地方就在这里: **精度过不能证明没有数据竞争**。
def check_l1_resident_no_war(src: str) -> list:
    code = strip_comments(src)
    #  手工分配的 L1 块名
    resident = set()
    for m in re.finditer(r"(\w+)\s*=\s*\w+\.GetBuffer\s*<\s*BufferType::ASCEND_CB[^>]*>", code):
        resident.add(m.group(1))
    #  ★分配与使用常常**不在同一个文件**: GetBuffer 在 cube_op.h, CopyGmToL1 在 cubeN_sfag.h。
    #  2026-09-15 实撞: 判据的自检用例是自包含的所以两条都过, 拿真实源码一跑却零命中。
    #  所以在本文件找不到 GetBuffer 时, 退而认 `l1_` 前缀的目标名 —— 保护检查本身是
    #  文件内的, 有 WaitFlag 的那些 (l1_key_tensor 等) 照样不报。
    hz = []
    for m in re.finditer(r"CopyGmToL1\s*\(\s*(\w+)", code):
        dst = m.group(1)
        if dst not in resident and not dst.startswith("l1_"):
            continue
        #  往前看 600 字符内有没有"等读完"的保护
        #  窗口取 1400 字符: WaitFlag 与 CopyGmToL1 之间常隔十几行 (含 if constexpr 分支
        #  与偏移计算)。2026-09-15 用 600 字符时把 cube1 的 rope 块误报成无保护。
        head = code[max(0, m.start() - 1400):m.start()]
        if re.search(r"WaitFlag\s*<\s*(?:AscendC::)?HardEvent::MTE1_MTE2", head):
            continue
        ln = code[:m.start()].count("\n") + 1
        hz.append(dict(check="H23_l1_resident_no_war", level="错误", line=ln, buffer=dst,
                       msg=(f"往手工分配的 L1 块 `{dst}` 写入 (行 {ln}), 而写入点之前没有 "
                            f"`WaitFlag<HardEvent::MTE1_MTE2>` 之类**等上一次 Mmad 读完**的保护。"
                            f"Mmad 发出后不等完成, 下一轮的覆写会与上一轮的读撞车, "
                            f"结果随运行时序变化。"
                            f"★**精度过不能证明没有这个问题** —— 2026-09-15 实撞时四项精度全过而"
                            f"逐位确定性 False, 竞争窗口小, 偏差落在容差内。"
                            f"H17 抓不到这一形态, 因为它只认从池取的 buffer, 认不出 GetBuffer。"
                            f"改法: 给该块配一个专用事件做 WAR 保护 (注意 EVENT_ID 可能已用满), "
                            f"或让它 ping-pong, 或放弃常驻改回逐次搬运。")))
    return hz


def check_zero_span(src):
    hz = []
    for m in re.finditer(r"(InitOutput|Duplicate)\s*<[^>]*>\s*\([^;]*;", src):
        stmt = m.group(0)
        line = src[:m.start()].count("\n") + 1
        offs = set(re.findall(r"(\w*[Oo]ffset\w*)", stmt))
        # 语句本身或它赋值来源里有"偏移相减"的痕迹
        near = src[max(0, m.start() - 600):m.start()]
        # 允许 tiling-> / obj. 前缀, 否则 "boundOffsetBytes - tiling->fpAccOffsetBytes" 匹配不上
        _OFF = r"(?:\w+\s*(?:->|\.)\s*)?(\w*[Oo]ffset\w*)"
        span_expr = re.search(_OFF + r"\s*-\s*" + _OFF, stmt + near)
        if len(offs) >= 2 or span_expr:
            pair = f"{span_expr.group(1)} - {span_expr.group(2)}" if span_expr else ", ".join(sorted(offs))
            hz.append(dict(check="H8_zero_span_crosses_regions", buffer=stmt.split("(")[1][:24],
                           msg=(f"第 {line} 行的清零范围由**区段偏移相减**得出 ({pair}) —— 这一趟清零"
                                f"横跨了多个逻辑区段。若清零函数与随后写这些区段的函数之间没有 SyncAll, "
                                f"就会出现'核 A 在写、核 B 还在清'的写-写竞态: 抹掉多少随时序变, "
                                f"症状是输出近似全 0 且 det_bit=False (gmfr 2026-08-17 实撞, "
                                f"46 例 UT 掉到 32/46)。修法: 按区段分多次清, 跳过中间不该动的段。")))
    return hz


SIZEOF_H9 = {"int8_t": 1, "uint8_t": 1, "half": 2, "bfloat16_t": 2, "int16_t": 2, "uint16_t": 2,
             "float": 4, "int32_t": 4, "uint32_t": 4, "int64_t": 8, "uint64_t": 8}
_ALIGNED_H9 = re.compile(r"CeilAlign|AlignUp|RoundUp|UbStride"
                         r"|Ceil\s*\([^)]*\)\s*\*\s*(?:8|16|32)[uUlL]*"
                         r"|/\s*(?:8|16|32)[uUlL]*\s*\)?\s*\*\s*(?:8|16|32)[uUlL]*"
                         r"|&\s*~\s*\(?\s*(?:7|15|31)[uUlL]*")


def check_atomic_write_granule(src, assume=None, host_src=None):
    """H9: 原子写的**长度**不是 32B 的整数倍 -> 同一目标被多核并发原子加时丢更新.

    机理: 不足一个 32B 颗粒的原子加要对该颗粒做读-改-写. 两个核同时往同一颗粒的不同部分加, 后写的
    那次用的是它读到的旧值, 前一次的增量被抹掉. 结果既数值错又**随时序变** -> det_bit=False.
    与"地址对齐"是两回事: 地址不对齐是硬件直接报 ADDR_MISALIGN(崩), 长度不足颗粒是**静默丢数**.
    也与 MTE3 的 512B 规则无关 —— 512B 是带宽饱和点(性能), 32B 是原子性颗粒(正确性).

    2026-08-19 gmfr 实测四点: 界的写长度 16B/48B -> det_bit=False; 32B/64B -> True.

    判据: DataCopyExtParams 的 blockLen 形如 `EXPR * sizeof(T)`, 且该 params 用在原子区间内的
    DataCopyPad 上, 而 EXPR 没有对齐动作 -> 报. EXPR 必须是 32/sizeof(T) 的倍数.
    """
    hz = []
    # params 名 -> (blockLen 里的 EXPR, 元素类型, 行号)
    pmap = {}
    for m in re.finditer(r"DataCopyExtParams\s+(\w+)\s*\{\s*1\s*,\s*([^,]+),", src):
        nm, blen = m.group(1), m.group(2)
        t = re.search(r"sizeof\s*\(\s*([\w:]+)\s*\)", blen)
        if not t:
            continue
        expr = blen[:t.start()].replace("static_cast<uint32_t>", "").strip(" (*")
        if not expr:
            #  blockLen 恰为 sizeof(T) = **单个元素**: 这是"全部核往同一个固定地址做原子聚合"
            #  (如 SetAtomicMax 汇总一个标量), 与"各核写一个区间、区间只部分覆盖 32B 颗粒"不是
            #  同一情形. gmfr 实测该写法在 50/50 UT 与多轮重试下 det_bit 恒 True, 故不报.
            continue
        # 一跳内联: 长度常写成一个局部变量, 对齐动作在它的定义里 (如 bndStrideW = BoundUbStride(..)).
        # 不内联就把已经对齐好的写法误报成风险 —— 与 wscap.inline_locals 同一理由.
        if re.fullmatch(r"\w+", expr):
            d = re.search(rf"\b(?:const\s+)?u?int\d*_t\s+{re.escape(expr)}\s*=\s*([^;]+);", src)
            if d:
                expr = d.group(1).strip()
        pmap[nm] = (expr, t.group(1).split("::")[-1], src[:m.start()].count("\n") + 1)
    if not pmap:
        return hz
    # 原子区间: SetAtomicXxx() ... SetAtomicNone()
    for a in re.finditer(r"SetAtomic(?:Add|Max|Min)\s*<[^>]*>\s*\(\s*\)", src):
        end = src.find("SetAtomicNone", a.end())
        region = src[a.end():end if end > 0 else len(src)]
        for w in re.finditer(r"DataCopyPad\s*\(\s*(\w+)\s*\[[^\]]*\]\s*,\s*[^;]*?,\s*(\w+)\s*\)", region):
            gm, pnm = w.group(1), w.group(2)
            if pnm not in pmap:
                continue
            expr, ty, _ = pmap[pnm]
            esz = SIZEOF_H9.get(ty, 0)
            if not esz:
                continue
            need = 32 // esz if esz < 32 else 1
            if need <= 1 or _ALIGNED_H9.search(expr):
                continue
            # 输入契约可证是 need 的倍数 -> 不可达, 不报 (契约由 core.gen_spec 从 aclnn 库自动抽)
            from core.wscap import infer_with_blockers
            im, blk = infer_with_blockers(expr, assume or {}, (src, host_src))
            if im % need == 0:
                continue
            if re.fullmatch(r"\s*\d+\s*", expr) and int(expr) % need == 0:
                continue
            ln = src[:a.end() + w.start()].count("\n") + 1
            key = (gm, pnm)
            if key in [(h["buffer"], h.get("params")) for h in hz]:
                continue
            hz.append(dict(check="H9_atomic_write_granule", buffer=gm, params=pnm, line=ln,
                           msg=(f"第 {ln} 行在原子区间内以 {pnm} 写 '{gm}', 长度 = ({expr})·{esz}B, "
                                f"而 {expr} 没有对齐动作 -> 长度可能不是 32B 的整数倍。不足一个 32B 颗粒的"
                                f"原子加要读-改-写该颗粒, 多核并发累加同一输出时**丢更新**: 数值错且随时序变, "
                                f"表现为 det_bit=False。注意这跟地址对齐不同 —— 地址不对齐会直接 "
                                f"ADDR_MISALIGN 崩掉, 长度不足颗粒是静默丢数。"
                                f"修: 把 {expr} 向上补到 {need} 的倍数, 补出的槽位写 0 (原子加 0 无害), "
                                f"并让消费侧能把'补出来的 0'与真实值区分开。"
                                + (f" 证明卡在 {sorted(blk)[:3]}: 取值不在源码里(如 singleCoreN 由 matmul "
                                   f"tiling 库产出), 既判不了可达也判不了不可达 —— host 加显式断言, "
                                   f"或 --assume 把它作为已知契约传进来。" if blk else ""))))
    return hz


#  ---- H14: 跨核等待挡的流水 != 消费该数据的流水 -> 读到旧值 --------------------------
#  实撞 (sparse_flash_attention, S21/S22, 2026-09-10): cube 侧
#      v0ResGm.WaitCrossCore();      // buffer.h: GM+BACKWARD 在 AIC 侧发的是
#                                    //   CrossCoreWaitFlag<..., PIPE_MTE2>  —— 只挡 MTE2
#      GatherKvToL1(...)             // 里面用**标量读** d.GetValue() 取跨核描述符
#  标量流水不受那条等待约束, 可以越过它先读, 于是读到上一轮的旧表。
#  代价形态很隐蔽: 表上带身份标签时表现为"标签常不符"而回退到慢路径 (性能损失, 精度仍对),
#  没有标签时就是**静默错数**。此前两轮排查 (换槽位查找 / 加自旋) 都归错了因, 花了两次构建。
#  判据: WaitCrossCore() 之后, 在同一函数体内 (含它直接调用的、本文件里定义的函数) 出现
#        GlobalTensor 的标量读 GetValue(, 而中间没有把标量排到 MTE2 之后的事件对。
_SCALAR_AFTER_WAIT_BARRIER = re.compile(r"HardEvent::(MTE2_S|FIX_S|M_S|MTE3_S)")

def _split_funcs_ln(src: str):
    """把源码粗切成函数体, 同时给出起始行号。既认 `Class<..>::Name(` 也认裸 `Name(` 定义
    —— split_funcs 只认前者, 自检里的裸函数会被整段跳过 (写这条判据时实撞)。"""
    marks = [(m.start(), m.group(1))
             for m in re.finditer(r'(?:::|\b__aicore__[^\n(]*?\b)(\w+)\s*\([^;{]*\)\s*\{', src)]
    out = []
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(src)
        out.append((name, src[pos:end], src[:pos].count("\n") + 1))
    return out


def check_crosscore_wait_pipe(src: str) -> list:
    hz = []
    funcs = _split_funcs_ln(src)
    bodies = {name: body for name, body, _ln in funcs}
    for name, body, base_ln in funcs:
        for m in re.finditer(r"(\w+)\s*\.\s*WaitCrossCore\s*(<[^>]*>)?\s*\(\s*\)", body):
            tail = body[m.end():]
            if _SCALAR_AFTER_WAIT_BARRIER.search(tail.split("GetValue(")[0] if "GetValue(" in tail else tail):
                continue
            #  本函数体内的标量读
            hit = None
            mg = re.search(r"(\w+)\s*(\[[^\]]*\])?\s*\.\s*GetValue\s*\(", tail)
            if mg:
                hit = f"{mg.group(1)}.GetValue()"
            else:
                #  一层展开: 等待之后调用的、本文件里定义的函数里有没有标量读
                for mc in re.finditer(r"(?:this\s*->\s*)?(\w+)\s*\(", tail):
                    callee = mc.group(1)
                    cb = bodies.get(callee)
                    if cb and "GetValue(" in cb and not _SCALAR_AFTER_WAIT_BARRIER.search(cb):
                        hit = f"{callee}() 内的 GetValue()"
                        break
            if hit is None:
                continue
            ln = base_ln + body[:m.start()].count("\n")
            hz.append(dict(check="H14_crosscore_wait_pipe_mismatch", buffer=m.group(1), level="错误", line=ln,
                           msg=(f"L{ln}: `{m.group(1)}.WaitCrossCore()` 之后用**标量读** ({hit}) 取跨核数据, "
                                f"而 WaitCrossCore 对 GM/UB 缓冲发的是 `CrossCoreWaitFlag<..., PIPE_MTE2/PIPE_FIX/PIPE_V>` "
                                f"—— 只挡住那一条流水, 标量流水可以越过它先读到旧值。"
                                f"症状: 带身份标签时表现为标签频繁不符而落到慢路径 (精度仍对, 但白跑), "
                                f"不带标签时是静默错数。"
                                f"改法: 等待之后补一对 `SetFlag/WaitFlag<HardEvent::MTE2_S>`, "
                                f"把标量读排到那条等待之后; 或者改用 MTE2 把数据搬进 UB 再读 "
                                f"(向量侧读 UB 走 V 流水, 与 WaitCrossCore 的 PIPE_V 天然有序)。"
                                f"★代价: 屏障会让消费核真的等生产核, 实测 +5.8us/97.1us; "
                                f"要消掉它得让生产提前一拍 (生产 T+1 的数据时消费 T)。")))
    return hz


def flag_ledger(src):
    """快速 debug: 撞到 507014 时先看这张账表, 一眼定位是哪个旗标不平衡。"""
    sites = {}
    for m in re.finditer(r"CrossCore(Set|Wait)Flag\s*(?:<[^>]*>)?\s*\(", src):
        kind = m.group(1)
        flag = _cc_flag_channel(src, m.end() - 1)
        sites.setdefault(flag, {"Set": [], "Wait": []})[kind].append(src[:m.start()].count("\n") + 1)
    print("=== 跨核旗标账表 (507014 先看这个) ===")
    print(f"{'旗标':<24}{'置位次数':>9}{'等待次数':>9}  置位行 / 等待行")
    for flag, d in sorted(sites.items()):
        print(f"{flag:<24}{len(d['Set']):>9}{len(d['Wait']):>9}  {d['Set']} / {d['Wait']}")
    print("\n判读: 置位/等待的**站点数**不等不一定错 (可能分处互斥分支);")
    print("      但同一控制上下文下次数不等**必然**死锁 —— 那是 H7 判据。")
    print("      CrossCoreSetFlag<2,...> 是组语义: 一对 AIV 的两次置位 = AIC 可消费的一个。")
    return 0



# ---- H24: **首尾条件在循环边界参数化后不再互补** —— 挂死 (含同步原语时) 或数值错。
#      形态: 一个 `for (v = B; v < E; v += s)` 的循环体里 (或它调用的函数里, 同文件)
#      有两个守卫 `v == 0` 与 `v + s >= N`。只要 B 是 0 且 E 是 N, 这两个条件就互补 ——
#      每轮循环"首"一次"尾"一次, 配对的 Wait/Set 刚好各一次。
#      **一旦把循环起止参数化** (分段、分核、断点续跑), B 不再是 0 而 E 不再是 N,
#      于是首段的核只撞得上"首"、末段的核只撞得上"尾"、中间的核两个都撞不上。
#      2026-09-15 sparse_flash_attention_grad v8a 实撞:
#        for (blkCntOffset = KvSegBegin(n); blkCntOffset < KvSegEnd(n); blkCntOffset += sco)
#        if (blkCntOffset == 0)          { WaitFlag<MTE1_MTE2>(MM_L1_QUERY_EVENTS); ... }
#        if (blkCntOffset + sco >= n)    { SetFlag <MTE1_MTE2>(MM_L1_QUERY_EVENTS);     }
#      kvSeg_==0 的核只 Wait 不 Set -> 下一批的 Wait 无人配对 -> 507014 aicore timeout,
#      挂的正是 blk:0 与 blk:8 (两个 kvSeg_==0 的核)。
#      ★为什么 `--flags` 查不到: 那张账表只统计 CrossCoreSetFlag/WaitFlag, 而这是**核内**事件。
#      ★为什么"站点数配平"也查不到: 站点数一个没变, 变的是**每个核跑到它的次数**。
def _norm_expr(e: str) -> str:
    return re.sub(r"\s+", "", e or "")


def _take_expr(code: str, i: int) -> str:
    """从 i 开始取一个**括号配平**的表达式, 在深度 0 的 `)` `;` `,` `&&` `||` 处停。

    ★不要用正则的 `[\w().]+?` 之类去截: 2026-09-15 H24 首版就是这么写的,
    自检的自包含用例全过, 拿真实源码一跑三条误报 —— 懒惰量词把
    `KvSegBegin(actualSelectedBlockCount)` 截成缺右括号的前缀, 贪婪量词又把外层的
    `)` 吞进来, 两头都对不上循环边界的字面。表达式要按括号深度扫, 不能按字符类扫。
    """
    depth, j, n = 0, i, len(code)
    while j < n:
        c = code[j]
        if c in "([":
            depth += 1
        elif c in ")]":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0:
            if c in ";,":
                break
            if code.startswith("&&", j) or code.startswith("||", j):
                break
        j += 1
    return code[i:j].strip()


def check_loop_boundary_guards(src: str) -> list:
    code = strip_comments(src)

    #  1) 收集每个循环变量的起点/终点/步长 (同一变量可能出现在多个循环里)
    begins, ends, steps = {}, {}, {}
    loop_pat = re.compile(
        r"for\s*\(\s*(?:[\w:]+[\w:<>\s*&]*?\s+)?(\w+)\s*=\s*([^;]+?)\s*;"
        r"\s*\1\s*<\s*([^;]+?)\s*;\s*\1\s*\+=\s*([^)]+?)\s*\)")
    for m in loop_pat.finditer(code):
        v, b, e, st = m.groups()
        begins.setdefault(v, set()).add(_norm_expr(b))
        ends.setdefault(v, set()).add(_norm_expr(e))
        steps.setdefault(v, set()).add(_norm_expr(st))
    if not begins:
        return []

    SYNC = re.compile(r"(?:Set|Wait)Flag\s*<|CrossCore(?:Set|Wait)Flag")

    def _level_and_body(pos: int) -> tuple:
        """守卫命中点之后的一段代码里有没有同步原语 —— 有则挂死风险 (错误级)。"""
        tail = code[pos:pos + 500]
        return ("错误", True) if SYNC.search(tail) else ("提示", False)

    hz = []
    #  2) 段首形守卫: `v == <A>` (含 `xxx = (v == <A>)` 这种赋给 bool 的)
    for v, bset in begins.items():
        for m in re.finditer(rf"\b{re.escape(v)}\s*==\s*", code):
            raw_a = _take_expr(code, m.end())
            if not raw_a:
                continue
            a = _norm_expr(raw_a)
            ln = code[:m.start()].count("\n") + 1
            lvl, has_sync = _level_and_body(m.start())
            if a not in bset:
                hz.append(dict(check="H24_loop_boundary_guard", level=lvl, line=ln, buffer=v,
                               msg=(f"行 {ln} 用 `{v} == {raw_a}` 判\"第一轮\", "
                                    f"而 `{v}` 的循环**起点**是 {sorted(bset)} —— 这个守卫一次都不会成立"
                                    f"(或只对其中一部分核成立)。"
                                    + (f"该守卫段内有 Set/WaitFlag: 它与配对的\"最后一轮\"守卫一旦不再互补, "
                                       f"就是**只 Wait 不 Set** 或反之, 症状是 aicore timeout (507014)。"
                                       if has_sync else
                                       f"该守卫段内没有同步原语, 所以是数值风险不是挂死风险: "
                                       f"每轮的前置计算 (行和/预加载/初始化) 会被整段跳过。")
                                    + f"改法: 把 0 换成该循环真正的起点表达式。"
                                      f"★2026-09-15 sfag v8a 实撞: 循环起止改成 [KvSegBegin, KvSegEnd) 后, "
                                      f"`blkCntOffset == 0` 只有 kvSeg_==0 的核成立, 而配对的 SetFlag 只有"
                                      f"末段核成立, 挂在 blk:0 / blk:8。`--flags` 查不到 —— 它只统计跨核旗标。")))
            elif len(bset) > 1:
                hz.append(dict(check="H24_loop_boundary_guard", level="提示", line=ln, buffer=v,
                               msg=(f"行 {ln} 的 `{v} == {raw_a}` 只对 `{v}` 的**其中一种**"
                                    f"循环起点成立 (该文件里有 {sorted(bset)} 共 {len(bset)} 种)。"
                                    f"确认走另一种起点的那条路径是不是也要这段前置。")))

    #  3) 段尾形守卫: `v + <s> >= <B>`
    for v, eset in ends.items():
        sset = steps.get(v, set())
        for m in re.finditer(rf"\b{re.escape(v)}\s*\+\s*(\w+)\s*>=\s*", code):
            raw_b = _take_expr(code, m.end())
            if not raw_b:
                continue
            st, b = _norm_expr(m.group(1)), _norm_expr(raw_b)
            if sset and st not in sset:
                continue          #  不是拿循环步长在比, 与本判据无关
            ln = code[:m.start()].count("\n") + 1
            lvl, has_sync = _level_and_body(m.start())
            if b not in eset:
                hz.append(dict(check="H24_loop_boundary_guard", level=lvl, line=ln, buffer=v,
                               msg=(f"行 {ln} 用 `{v} + {m.group(1)} >= {raw_b}` 判\"最后一轮\", "
                                    f"而 `{v}` 的循环**终点**是 {sorted(eset)} —— 两者不是同一个量。"
                                    + (f"该守卫段内有 Set/WaitFlag: 与配对的\"第一轮\"守卫不互补 = "
                                       f"**只 Set 不 Wait** 或反之, 症状是 aicore timeout (507014)。"
                                       if has_sync else
                                       f"该守卫段内没有同步原语, 是数值风险: 尾块大小/收尾归约/下一轮预加载"
                                       f"会落在错误的那一轮上, 或一轮都不落。")
                                    + f"改法: 把它换成该循环真正的终点表达式。"
                                      f"注意这里常有**两个语义混在一个标志里**: \"整行的最后一块\"(管尾块不满) "
                                      f"与 \"本核这一段的最后一块\"(管收尾动作)。未分段时两者相等, 分段后必须拆开 —— "
                                      f"2026-09-15 sfag v8a 就是把 isLastBasicBlock 拆成它与 isSegLastBlock 才修好。")))
            elif len(eset) > 1:
                hz.append(dict(check="H24_loop_boundary_guard", level="提示", line=ln, buffer=v,
                               msg=(f"行 {ln} 的 `{v} + {m.group(1)} >= {raw_b}` 只对 `{v}` 的**其中一种**"
                                    f"循环终点成立 (该文件里有 {sorted(eset)} 共 {len(eset)} 种)。")))
    return hz


# ---- H2: **det 路径上的浮点原子加** —— 跨核累加顺序不固定, 逐位确定性无保证。
#      2026-09-15 sparse_flash_attention_grad 实撞, 而**判据当时零命中**:
#      它识别 det 路径只认三种写法 (`deterministicFlag == 1` / 函数名含 FRDeterministic
#      或 FpAcc), 全是 gmfr 那个算子的命名。本 case 的 `ScatterAddDeter` 一个都不匹配,
#      于是 v6b 把 KV_MERGE=0 那支从 int32 定点"回退"成 fp32 原子加时**没有任何告警**,
#      逐位确定性一路丢进交付包, 直到交付前补验才发现 (base/v3c 实测 True, v6b False)。
#      ★两级判定:
#        - 同一函数里既有 int 原子加又有浮点原子加 -> **错误级**。作者自己用 int32 是为了
#          确定性 (整数加法结合律), 那么走浮点的那一支就是**已知**会丢确定性的分支。
#        - 只有浮点原子加 -> 提示级。确定性只能靠外部把累加串行化 (每批一次的全局汇合),
#          而那个机制常常在别处被删掉, 在本函数里一点看不出来 —— 这正是 v6b 的经过:
#          v3c 删了汇合并用定点顶上, v6b 回退了原子类型却没把汇合一并恢复。
def check_float_atomic_in_det(src: str) -> list:
    code = strip_comments(src)
    FLOAT_AT = re.compile(r"SetAtomicAdd<\s*(float|half|bfloat16_t|DTYPE_OUT)\s*>")
    INT_AT = re.compile(r"SetAtomicAdd<\s*(?:u?int(?:8|16|32|64)_t|int)\s*>")
    hz = []
    for name, body in split_funcs(code):
        #  ★det 路径的识别: 名字带 Deter/Deterministic 的算它, 但要排掉 UnDeter /
        #  NonDeterministic 那一类反义词 —— 它们恰恰是 det=0 的路径, 报了就是误报。
        low = name.lower()
        #  ★识别 det 路径不能只看函数名: 2026-09-16 实撞 —— 为了让 det=1 复用块级 scatter,
        #  一个名为 `ScatterAddUnDeter` 的函数被 det 路径调用了, 名字从此不再可靠。
        #  加一条更硬的信号: 函数体里用 `if constexpr (IS_DETERMINISTIC)` 分叉的, 就是 det 相关。
        is_det = (re.search(r"deterministicflag\s*==\s*1", body.lower()) is not None
                  or "frdeterministic" in low or "fpacc" in low
                  or re.search(r"if\s+constexpr\s*\(\s*!?\s*IS_DETERMINISTIC\s*\)", body) is not None
                  or ("deter" in low and "undeter" not in low and "nondeter" not in low))
        if not is_det:
            continue
        fl = list(FLOAT_AT.finditer(body))
        if not fl:
            continue
        im = INT_AT.search(body)
        has_int = im is not None
        for m in fl:
            #  ★"互斥"不等于"安全" —— 要看互斥的是**哪个条件**。
            #    按 IS_DETERMINISTIC 分叉: false 那支本来就是 det=0 路径, 走浮点天经地义 -> 放行;
            #    按别的条件分叉 (如 MERGE_DKDV): 两个实例化**都在 det=1 下**, 走浮点那个
            #      实例化就是丢确定性的那一个 -> 照报。
            #  2026-09-16 差点在这里放宽过头: 先写成"只要互斥就不报", 自检里那条来自
            #  v6b 真实缺陷的负例 (int32/float 按 MERGE_DKDV 分叉) 当场变成不命中 ——
            #  判据的放宽必须拿已知的真实负例回归, 不能只看新案例过不过。
            if _cx_branch(body, m.start()).get("IS_DETERMINISTIC") is False:
                continue
            ln = code[:code.find(body) + m.start()].count("\n") + 1 if body in code else 0
            if has_int:
                hz.append(dict(check="H2_float_atomic_in_det", level="错误", line=ln, buffer=name,
                               dtype=m.group(1),
                               msg=(f"det 路径函数 `{name}` 里**同时**有 int 原子加与 "
                                    f"`SetAtomicAdd<{m.group(1)}>`。用 int 正是为了让累加顺序无关"
                                    f"(整数加法结合律), 所以走浮点的那一支就是**已知**会丢逐位确定性的分支。"
                                    f"★2026-09-15 实撞: v6b 为修另一个缺陷 (dv 被 int32 原子模式波及, 见 H22) "
                                    f"把 KV_MERGE=0 那支回退成 fp32 原子加, 而 v3c 删掉的全局汇合没有一并恢复 —— "
                                    f"精度全过、det=0 全过, 只有逐位确定性悄悄从 True 变成 False。"
                                    f"改法: 让两个输出走**同一种**模式 (都定点), 或把被删掉的串行化机制恢复回来。"
                                    f"两条都做不到就不能声称 det=1。")))
            else:
                hz.append(dict(check="H2_float_atomic_in_det", level="提示", line=ln, buffer=name,
                               dtype=m.group(1),
                               msg=(f"det 路径函数 `{name}` 用 `SetAtomicAdd<{m.group(1)}>` —— 浮点加法不满足结合律, "
                                    f"跨核并发累加的顺序不固定, 逐位确定性只能靠**外部**把累加串行化"
                                    f"(典型是每批一次的全局汇合)。确认那个机制还在。"
                                    f"★它常常在别处被删掉, 而本函数里一点看不出来: 2026-09-15 实撞的正是这一幕 —— "
                                    f"删汇合的那一版用 int32 定点顶上了, 后来有人把原子类型回退成 fp32, "
                                    f"汇合却没跟着回来。单 owner 独占的累加不受此限。")))
    return hz


def analyze(src, assume=None, host_src=None):
    gms = gm_tensors(src)
    funcs = split_funcs(src)
    # 记录每个 GM buffer: 在哪些函数被写(dst)/读(src), 该函数是否单核
    written_single, read_multi = {}, {}
    for name, body in funcs:
        sc = is_single_core(body)
        for g in gms:
            # 写: DataCopyPad(g ...) 或 DataCopyPad(g[...]) 作为第一实参(dst=GM)
            if re.search(rf'DataCopyPad\(\s*{g}\b', body) or re.search(rf'DataCopy\w*\(\s*{g}\b', body) \
               or re.search(rf'InitOutput<[^>]*>\(\s*{g}\b', body):
                if sc: written_single.setdefault(g, []).append(name)
            # 读: DataCopyPad(dst, g[...] / g,) 作为第二实参(src=GM)
            if re.search(rf'DataCopyPad\(\s*\w[^,]*,\s*{g}\b', body) or re.search(rf'DataCopy\w*\(\s*\w[^,]*,\s*{g}\b', body):
                if not sc: read_multi.setdefault(g, []).append(name)
    hazards = []
    for g in written_single:
        if g in read_multi:
            hazards.append(dict(check="H1_cross_core_visibility", buffer=g,
                                written_single_core_in=written_single[g], read_multi_core_in=read_multi[g],
                                msg=f"GM '{g}' 单核写(在 {written_single[g]})+ 多核读(在 {read_multi[g]}), 仅 SyncAll 分隔 "
                                    f"-> 跨核 GM 写可见性无保证 = race. 改: 每核冗余算自己分片 / 显式 drain+flag."))
    hazards += check_float_atomic_in_det(src)   # H2: det 路径浮点原子加
    hazards += check_float_uint_cast(src)   # H4: aicore float<->unsigned 直转 (编译必错)
    hazards += check_queue_depth(src)       # H5: TQue depth 越界双入队 (真机死锁)
    hazards += check_zero_span(src)          # H8: 清零范围跨区段 (写-写竞态)
    hazards += check_atomic_write_granule(src, assume, host_src)  # H9: 原子写长度不足 32B 颗粒 (并发丢更新)
    hazards += check_crosscore_flags(src)   # H6: 跨核标志置位/等待计数不平衡 (真机死锁)
    hazards += check_mix_vector_core_index(src)  # H32: 相同循环可能使用不同的Cube/AIV编号
    hazards += check_orphan_compensation_loop(src)  # H20: 真实站点已删, 只剩空转补偿循环
    hazards += check_barrier_finer_than_resource(src)  # H21: 屏障粒度比它保护的资源细一档
    hazards += check_crosscore_barrier_granularity(src)  # H26: 跨核汇合比它要保证的东西粗
    hazards += check_core_partition_remainder(src)  # H27: 核间划分的余数整块压在末核
    hazards += check_early_return_before_barrier(src)  # H28: 全核屏障前的 early return = 死锁
    hazards += check_atomic_mode_scope(src)  # H22: 全局原子模式波及同作用域的别的输出
    hazards += check_l1_resident_no_war(src)  # H23: L1 常驻块被覆写却没有 WAR 保护
    hazards += check_loop_boundary_guards(src)  # H24: 首尾条件在循环边界参数化后不再互补
    hazards += check_setwait_adjacent(src)  # H10: Set/Wait 紧邻 -> 两条 pipe 串行 (性能, 非死锁)
    hazards += check_share(src)  # H30/H31: 多 Matmul 实例的重叠窗口 / 事件号未占号
    hazards += check_intracore_flag_balance(src)  # H11: 同核 Set/Wait 循环内收支不平衡 (死锁)
    hazards += check_missing_prologue_setflag(src)  # H12: 循环外缺初始置位 (死锁, H11 的盲区)
    hazards += check_flag_guard_asymmetry(src)  # H15: Set/Wait 守卫条件不同 (收支只在运行期成立, H11 的盲区)
    hazards += check_pool_buf_missing_wait(src)  # H17: 多缓冲池取出的 buffer 从不等反依赖事件 (写覆盖)
    hazards += check_slot_ring_vs_pipeline_depth(src)  # H18: 槽数 < 流水级数 (真机静默挂死)
    hazards += check_fixpipe_dualdst_quant(src)  # H16: fixpipe 双目标模式与类型转换互斥 (上机才崩)
    hazards += check_setflag_pipe_has_work(src)  # H13: SetFlag 依赖的流水上没有操作 (死锁)
    hazards += check_crosscore_wait_pipe(src)  # H14: 跨核等待挡的流水 != 消费流水 (读旧值)
    hazards += check_ub_mte2_to_mte3_chain(src)  # H19: UB 上 MTE2 生产->MTE3 消费缺旗标链 (随机脏数据)
    return hazards

def _parse_shape(s):
    """--shape n=4096,baseN=256,ubCalSize=4096,e=8,m=512,batch=512,coreNum=24 -> dict."""
    d = {}
    for kv in s.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1); d[k.strip()] = int(v)
    return d


def _selftest_checks() -> list:
    """本模块里所有"吃一段源码、吐一串 hazard"的判据函数。

    签名契约: 第一个参数收源码, 其余参数都有默认值。新判据只要符合这个形状,
    自检自动带上它 —— 不需要在任何白名单里登记。
    """
    import inspect
    out = []
    for nm, fn in sorted(globals().items()):
        if not nm.startswith("check_") or not callable(fn):
            continue
        try:
            ps = list(inspect.signature(fn).parameters.values())
        except (TypeError, ValueError):
            continue
        if not ps or ps[0].default is not inspect.Parameter.empty:
            continue
        if any(q.default is inspect.Parameter.empty for q in ps[1:]):
            continue          #  还要别的必填参数 (如 host_src), 自检这里喂不出来
        if ps[0].name == "srcs":
            continue          #  ★吃 {文件: 源码} 字典的跨文件判据 (H16) 不符合"第一个参数
                              #    收一段源码"的契约, 通用循环喂字符串给它会当场 AttributeError。
                              #    它有自己的用例 (_selftest_h16), 不走这条路。
        out.append(fn)
    return out


def selftest() -> int:
    import tempfile
    mix_index = """
#ifdef __DAV_C220_CUBE__
void C() {
    const auto core = GetBlockIdx();
    for (uint64_t task = core; task < tasks; task += cores) {
        CrossCoreSetFlag<2,PIPE_FIX>(READY);
    }
}
#endif
#ifdef __DAV_C220_VEC__
void V() {
    const auto lane = GetBlockIdx();
    const auto core = GetBlockIdx();
    for (uint64_t task = core; task < tasks; task += cores) {
        CrossCoreWaitFlag(READY);
        DataCopy(out[lane], in[lane], 16);
    }
}
#endif
"""
    fixed_index = mix_index.replace("const auto lane = GetBlockIdx();\n    const auto core = GetBlockIdx();",
                                    "const auto lane = GetBlockIdx();\n    const auto core = GetBlockIdx()/GetSubBlockNum();")
    cases = [
        ("H12 数组索引未知: helper置位s/caller等待slot只提示", None, """
void Seed(){for(int s=0;s<2;++s){SetFlag<HardEvent::M_MTE1>(ids[s]);}}
void Run(){Seed();for(int k=0;k<n;++k){WaitFlag<HardEvent::M_MTE1>(ids[slot]);SetFlag<HardEvent::M_MTE1>(ids[slot]);}}
"""),
        ("H12 负例: 固定不同cell不该放过缺项", True, """
void Run(){SetFlag<HardEvent::M_MTE1>(ids[1]);for(int k=0;k<n;++k){WaitFlag<HardEvent::M_MTE1>(ids[0]);SetFlag<HardEvent::M_MTE1>(ids[0]);}}
"""),
        ("H12 负例: 没有其他cell置位仍是错误", True, """
void Run(){for(int k=0;k<n;++k){WaitFlag<HardEvent::M_MTE1>(ids[slot]);SetFlag<HardEvent::M_MTE1>(ids[slot]);}}
"""),
        ("H12 负例: 不同数组的置位不能当未知别名", True, """
void Seed(){for(int s=0;s<2;++s){SetFlag<HardEvent::M_MTE1>(other[s]);}}
void Run(){for(int k=0;k<n;++k){WaitFlag<HardEvent::M_MTE1>(ids[slot]);SetFlag<HardEvent::M_MTE1>(ids[slot]);}}
"""),
        ("H32 负例: C220同通道同循环但两侧都用raw编号，仅提示", None, mix_index),
        ("H32 正例: GetBlockIdx/GetSubBlockNum归一且raw lane仅作寻址", False, fixed_index),
        ("H32 正例: 先读lane再归一core不该误报", False,
         fixed_index.replace("GetBlockIdx()/GetSubBlockNum()", "lane/GetSubBlockNum()")),
        ("H32 正例: raw core后续复合赋值归一不猜未归一", False,
         mix_index.replace("const auto lane = GetBlockIdx();\n    const auto core = GetBlockIdx();",
                           "const auto lane = GetBlockIdx();\n    auto core = GetBlockIdx(); core /= GetSubBlockNum();")),
        ("H32 正例: 只有AIV角色不能假设Cube任务配对", False,
         "#ifdef __DAV_C220_VEC__" + mix_index.split("#ifdef __DAV_C220_VEC__",1)[1]),
        ("H32 正例: 不同flag通道不猜配对", False,
         mix_index.replace("CrossCoreWaitFlag(READY)", "CrossCoreWaitFlag(OTHER)")),
        ("H32 正例: 不同循环控制不据字面core猜同任务", False,
         mix_index.replace("task += cores) {\n        CrossCoreWaitFlag", "task += vecCores) {\n        CrossCoreWaitFlag")),
        ("H32 正例: 注释中的风险代码不命中", False, "/*"+mix_index+"*/"),
        ("H32 负例: defined角色和嵌套未知条件仍只报提示", None,
         mix_index.replace("#ifdef __DAV_C220_VEC__", "#if defined(__DAV_C220_VEC__)\n#ifdef DEBUG")+"\n#endif\n"),
        ("H32 正例: 宏名子串不能当C220角色", False,
         mix_index.replace("__DAV_C220_", "NOT___DAV_C220_")),
        ("H32 正例: OR两角色条件无法分工，不猜角色", False,
         mix_index.replace("#ifdef __DAV_C220_VEC__", "#if defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)")),
        ("H32 正例: C310不套用C220接口推论", False,
         mix_index.replace("__DAV_C220_", "__DAV_C310_")),
        ("H19 正例: 同步链用**宏**写 (SET_FLAG(MTE2,MTE3,..)) 也要认 "
         "(2026-09-16 实撞: 只认模板那种, 在一个大量用宏的算子上报出 18 条全误报)", False, '''
__aicore__ inline void VecOp::GatherKVOptimized(const RunInfo &runInfo) {
    DataCopyPad(gatherTensor[0], keyGm[off], intriParams, padParams);
    SET_FLAG(MTE2, MTE3, mte3WaitMte2EventId);
    WAIT_FLAG(MTE2, MTE3, mte3WaitMte2EventId);
    DataCopyPad(selectedKWorkspaceGm[wsAddr], gatherTensor, outParamK);
}
'''),
        ("H19 负例: 宏形式也没有时仍须报 (别把判据改瘫)", True, '''
__aicore__ inline void VecOp::GatherKVOptimized(const RunInfo &runInfo) {
    DataCopyPad(gatherTensor[0], keyGm[off], intriParams, padParams);
    DataCopyPad(selectedKWorkspaceGm[wsAddr], gatherTensor, outParamK);
}
'''),
        ("H2 负例(错误级): det 函数里 int 与 float 原子加并存 —— 走 float 那支是已知丢确定性的分支 "
         "(2026-09-15 sfag v6b 实撞, 而判据当时零命中: det 路径只认 gmfr 的命名)", True, '''
__aicore__ inline void VecOp::ScatterAddDeter(const RunInfo &runInfo) {
    if constexpr (MERGE_DKDV) {
        SetAtomicAdd<int32_t>();
    } else {
        SetAtomicAdd<float>();
    }
    DataCopy(dkOutGm, dkInUb, n);
    SetAtomicNone();
}
'''),
        ("H2 负例(提示级): det 函数里只有 float 原子加 —— 确定性只能靠外部串行化, 而那机制可能已被删", "提示", '''
__aicore__ inline void VecOp::ScatterAddDeter(const RunInfo &runInfo) {
    SetAtomicAdd<float>();
    DataCopy(dkOutGm, dkInUb, n);
    SetAtomicNone();
}
'''),
        ("H2 正例: det 函数里只有 int 原子加, 顺序无关", False, '''
__aicore__ inline void VecOp::ScatterAddDeter(const RunInfo &runInfo) {
    SetAtomicAdd<int32_t>();
    DataCopy(dkOutGmInt, dkInUb.ReinterpretCast<int32_t>(), n);
    SetAtomicNone();
}
'''),
        ("H2 正例: UnDeter 是 det=0 的路径, 它的 float 原子加不该报", False, '''
__aicore__ inline void VecOp::ScatterAddUnDeter(const RunInfo &runInfo) {
    SetAtomicAdd<float>();
    DataCopy(dkOutGm, dkInUb, n);
    SetAtomicNone();
}
'''),
        ("H2 正例: NonDeterministic 同理, 反义词不许被前缀匹配吃掉", False, '''
__aicore__ inline void Op::ProcessNonDeterministic(GM_ADDR dk) {
    SetAtomicAdd<float>();
    DataCopy(dkOutGm, dkInUb, n);
    SetAtomicNone();
}
'''),
        ("H24 负例: 循环边界参数化后首尾守卫不再互补, 且守卫内有同步原语 "
         "(2026-09-15 sfag v8a 实撞 507014, 挂在 blk:0/blk:8)", True, """__aicore__ void f() {
    for (blk = KvSegBegin(total); blk < KvSegEnd(total); blk += step) {
        Compute();
    }
}
__aicore__ void Compute() {
    if (blk == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(L1_Q_EVENT);
        Preload();
    }
    if (blk + step >= total) {
        SetFlag<HardEvent::MTE1_MTE2>(L1_Q_EVENT);
    }
}"""),
        ("H24 正例: 循环边界参数化后, 首尾守卫一并锚到本段边界", False, """__aicore__ void f() {
    for (blk = KvSegBegin(total); blk < KvSegEnd(total); blk += step) {
        Compute();
    }
}
__aicore__ void Compute() {
    if (blk == KvSegBegin(total)) {
        WaitFlag<HardEvent::MTE1_MTE2>(L1_Q_EVENT);
        Preload();
    }
    if (blk + step >= KvSegEnd(total)) {
        SetFlag<HardEvent::MTE1_MTE2>(L1_Q_EVENT);
    }
}"""),
        ("H24 正例: 未分段 —— 起点就是字面 0, 终点就是那个量, 两个守卫本来就互补", False,
         """__aicore__ void f() {
    for (blk = 0; blk < total; blk += step) {
        Compute();
    }
}
__aicore__ void Compute() {
    if (blk == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(L1_Q_EVENT);
    }
    if (blk + step >= total) {
        SetFlag<HardEvent::MTE1_MTE2>(L1_Q_EVENT);
    }
}"""),
        ("H24 负例 (数值侧, 提示级): 守卫内没有同步原语, 前置计算被整段跳过", "提示",
         """__aicore__ void f() {
    for (blk = KvSegBegin(total); blk < KvSegEnd(total); blk += step) {
        bool calcRowsumFront = (blk == 0);
        CalRowsum(calcRowsumFront);
    }
}"""),
        ("H24 覆盖守卫 (括号配平): 边界表达式带嵌套括号时不许误报 —— "
         "首版用懒惰/贪婪量词截表达式, 自检全过而真实源码上三条误报", False,
         """__aicore__ void f() {
    for (blk = KvSegBegin(Min(cnt, maxBlk)); blk < KvSegEnd(Min(cnt, maxBlk)); blk += step) {
        if (blk == KvSegBegin(Min(cnt, maxBlk))) {
            WaitFlag<HardEvent::MTE1_MTE2>(L1_Q_EVENT);
        }
        if (blk + step >= KvSegEnd(Min(cnt, maxBlk))) {
            SetFlag<HardEvent::MTE1_MTE2>(L1_Q_EVENT);
        }
    }
}"""),
        ("H27 负例: 余数整块给最后一个核 (floor + 全部余数)", None,
         """__aicore__ void VecOp<T>::part() {
    int64_t first = total / nCores;
    int64_t cur = (idx == nCores - 1) ? (total % nCores + first) : first;
}"""),
        ("H27 正例: 已经均摊 (floor + 1) 的写法不该报", False,
         """__aicore__ void VecOp<T>::part2() {
    int64_t first = total / nCores;
    int64_t rem = total % nCores;
    int64_t cur = first + ((idx < rem) ? 1 : 0);
}"""),
        ("H27 正例: 只有整除没有余数分支的不该报", False,
         """__aicore__ void VecOp<T>::part3() {
    int64_t first = total / nCores;
    int64_t cur = first;
}"""),
        ("H26 负例: 循环里全体跨核汇合 + 浮点原子加 (每轮让全体核互等)", None,
         """__aicore__ void VecOp<T>::ScatterAddDeter() {
    for (int64_t i = 0; i < n; i++) {
        CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_FLAG);
        SetAtomicAdd<float>();
        DataCopyPad(gm, ub, p);
        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_FLAG);
    }
}"""),
        ("H26 正例: 有汇合但没有原子加 (单 owner 写, 不受结合律限制) 不该报", False,
         """__aicore__ void VecOp<T>::noAtomic() {
    for (int64_t i = 0; i < n; i++) {
        CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_FLAG);
        DataCopyPad(gm, ub, p);
        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_FLAG);
    }
}"""),
        ("H26 正例: 有原子加但汇合在循环外 (只汇合一次) 不该报", False,
         """__aicore__ void VecOp<T>::syncOnce() {
    for (int64_t i = 0; i < n; i++) {
        SetAtomicAdd<float>();
        DataCopyPad(gm, ub, p);
    }
    CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_FLAG);
}"""),
        ("H25 负例: DataCopyExtParams 每次只搬 128B, 远低于有效粒度", None,
         """__aicore__ void f() {
    DataCopyExtParams p;
    p.blockLen = 128;
    DataCopyPad(dst, src, p);
}"""),
        ("H25 正例: blockLen 小但两端跨距都是 0 (blockCount 个块首尾相接=连续搬运), 不该报错", None,
         """__aicore__ void C::g() {
    DataCopyExtParams q;
    q.blockCount = 64;
    q.blockLen = 128;
    q.srcStride = 0;
    q.dstStride = 0;
    DataCopyPad(dst, src, q);
}"""),
        ("H25 正例: 每次搬 1024B, 不该报 (判据过宽会天天响)", False,
         """__aicore__ void f() {
    DataCopyExtParams p;
    p.blockLen = 1024;
    DataCopyPad(dst, src, p);
}"""),
        ("H25 负例: 大括号初始化形态也要认 (第一版只认这种, 反而漏了逐字段赋值)", None,
         """__aicore__ void f() {
    DataCopyExtParams q{1, 64, 0, 0, 0};
    DataCopyPad(dst, src, q);
}"""),
        ("H7 负例: 同一循环里置位 2 次 / 等待 1 次 (真死锁)", True, """__aicore__ void f() {
    for (int i = 0; i < n; ++i) {
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_A);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_A);
        AscendC::CrossCoreWaitFlag(FLAG_A);
    }
}"""),
        ("H7 正例: 两次置位在 if/else 两个分支 (运行时只走一条)", False, """__aicore__ void f() {
    for (int i = 0; i < n; ++i) {
        if (cond) {
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_A);
        } else {
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_A);
        }
        AscendC::CrossCoreWaitFlag(FLAG_A);
    }
}"""),
    ]
    cases += [
        ("H23 负例: L1 常驻块被覆写却没有 WAR 保护 (2026-09-15 sfag 实撞, 精度过但逐位不确定)", True, """__aicore__ void f() {
    l1_kv_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);
    for (int32_t nIdx = 0; nIdx < n; nIdx += step) {
        CopyGmToL1(l1_kv_tensor, selectedKWorkspaceGm[off], singleN, dimDqk, dimDqk);
        MmadInnerWithSync<T1>(l0c, q, l1_kv_tensor, a, b, mmParam, f1, f2, f3, true, outGm[0]);
    }
}"""),
        ("H23 正例: 覆写前等上一次 Mmad 读完", False, """__aicore__ void f() {
    l1_kv_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);
    for (int32_t nIdx = 0; nIdx < n; nIdx += step) {
        WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MM_L1_KV_EVENT);
        CopyGmToL1(l1_kv_tensor, selectedKWorkspaceGm[off], singleN, dimDqk, dimDqk);
        MmadInnerWithSync<T1>(l0c, q, l1_kv_tensor, a, b, mmParam, f1, f2, f3, true, outGm[0]);
        SetFlag<AscendC::HardEvent::MTE1_MTE2>(MM_L1_KV_EVENT);
    }
}"""),
        ("H22 负例: 原子模式改成 int32 后, 同作用域内 fp32 输出被一起改掉 (2026-09-15 sfag 实撞)", True, """__aicore__ void f() {
    GlobalTensor<int32_t> dkOutGm = dkWorkspaceGmInt[off];
    GlobalTensor<float> dvOutGm = dvWorkspaceGm[off2];
    SetAtomicAdd<int32_t>();
    DataCopy(dkOutGm[baseRow * dimDAlign], dkInUb.ReinterpretCast<int32_t>(), n);
    DataCopy(dvOutGm[baseRow * dimD2Align], dvInUb, m);
    SetAtomicNone();
}"""),
        ("H22 正例: 两个输出与原子模式同类型", False, """__aicore__ void f() {
    GlobalTensor<float> dkOutGm = dkWorkspaceGm[off];
    GlobalTensor<float> dvOutGm = dvWorkspaceGm[off2];
    SetAtomicAdd<float>();
    DataCopy(dkOutGm[baseRow * dimDAlign], dkInUb, n);
    DataCopy(dvOutGm[baseRow * dimD2Align], dvInUb, m);
    SetAtomicNone();
}"""),
        ("H22 正例: 中间夹 SetAtomicNone 分隔成两段作用域", False, """__aicore__ void f() {
    GlobalTensor<int32_t> dkOutGm = dkWorkspaceGmInt[off];
    GlobalTensor<float> dvOutGm = dvWorkspaceGm[off2];
    SetAtomicAdd<int32_t>();
    DataCopy(dkOutGm[baseRow * dimDAlign], dkInUb.ReinterpretCast<int32_t>(), n);
    SetAtomicNone();
    SetAtomicAdd<float>();
    DataCopy(dvOutGm[baseRow * dimD2Align], dvInUb, m);
    SetAtomicNone();
}"""),
        ("H21 提示: 逐轮等一个每 N 轮才换的槽 (2026-09-15 sfag 实撞, AIV 只跑到能力的 13%)", None, """__aicore__ void f() {
    for (uint32_t i = 0; i < n; i += 2) {
        event_t ev = pingpong ? evPong : evPing;
        WAIT_FLAG(MTE3, MTE2, ev);
        DataCopyPad(gatherTensor[cur * dim], keyGm[idx], params, pad);
        cur += 2;
        if (cur == CAP) {
            SET_FLAG(MTE2, MTE3, ev2);
            WAIT_FLAG(MTE2, MTE3, ev2);
            DataCopyPad(wsGm[outOff], gatherTensor, outParam);
            SET_FLAG(MTE3, MTE2, ev);
            pingpong = 1 - pingpong;
            cur = 0;
        } else {
            SET_FLAG(MTE3, MTE2, ev);
        }
    }
}"""),
        ("H21 正例: WAIT 已经条件化成槽级 (改对之后的形态), 不该再报", False, """__aicore__ void f() {
    for (uint32_t i = 0; i < n; i += 2) {
        event_t ev = pingpong ? evPong : evPing;
        if (cur == 0) {
            WAIT_FLAG(MTE3, MTE2, ev);
        }
        DataCopyPad(gatherTensor[cur * dim], keyGm[idx], params, pad);
        cur += 2;
        if (cur == CAP) {
            DataCopyPad(wsGm[outOff], gatherTensor, outParam);
            SET_FLAG(MTE3, MTE2, ev);
            pingpong = 1 - pingpong;
            cur = 0;
        }
    }
}"""),
        ("H21 正例: 槽每轮无条件翻转 (真双缓冲), 逐轮同步是对的", False, """__aicore__ void f() {
    for (int64_t loop = 0; loop < maxLoops; loop++) {
        event_t ev = pingPongIdx == 0 ? evA : evB;
        WAIT_FLAG(MTE3, MTE2, ev);
        DataCopy(dkInUb, srcGm[loop * rowSize], rowSize);
        DataCopy(dstGm[loop * rowSize], dkInUb, rowSize);
        SET_FLAG(MTE3, MTE2, ev);
        pingPongIdx = 1 - pingPongIdx;
    }
}"""),
        ("H20 负例: 真实站点已删, 只剩 if 里的空转补偿循环 (2026-09-15 sfag 实撞)", True, """__aicore__ void f() {
    if (processBS1ByCore < formerCoreProcessNNum && cubeBlockIdx < usedCoreNum) {
        CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
        for (int32_t i = 0; i < n; ++i) {
            CrossCoreSetFlag<0, PIPE_MTE3>(SCATTER_VECTOR_SYNC_FLAG);
            CrossCoreWaitFlag<0, PIPE_MTE3>(SCATTER_VECTOR_SYNC_FLAG);
        }
    }
    CrossCoreSetFlag<2, PIPE_FIX>(SCATTER_SYNC_FLAG);
}"""),
        ("H20 正例: 补偿循环之外还有承载依赖的真实站点 (配套关系, 正常)", False, """__aicore__ void f() {
    for (int32_t t = 0; t < task; ++t) {
        ScatterAdd(dst, src);
        CrossCoreSetFlag<0, PIPE_MTE3>(VEC_FLAG);
        CrossCoreWaitFlag<0, PIPE_MTE3>(VEC_FLAG);
    }
    if (task < maxTask) {
        for (int32_t i = task; i < maxTask; ++i) {
            CrossCoreSetFlag<0, PIPE_MTE3>(VEC_FLAG);
            CrossCoreWaitFlag<0, PIPE_MTE3>(VEC_FLAG);
        }
    }
}"""),
        ("H20 正例: 跨核汇合 Set 紧跟 Wait 但不在循环里 (barrier 惯用法)", False, """__aicore__ void f() {
    CubeCompute(cubeOp);
    CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
    CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
    CrossCoreSetFlag<2, PIPE_FIX>(SCATTER_SYNC_FLAG);
}"""),
        ("H14 负例: 跨核等待后直接标量读跨核数据 (等待只挡 MTE2, 标量越过)", True, """__aicore__ void f() {
    v0ResGm.WaitCrossCore();
    int32_t nRun = d.GetValue(1);
    Gather(nRun);
}"""),
        ("H14 正例: 等待后补了 MTE2->S 事件对再标量读", False, """__aicore__ void f() {
    v0ResGm.WaitCrossCore();
    SetFlag<HardEvent::MTE2_S>(evt);
    WaitFlag<HardEvent::MTE2_S>(evt);
    int32_t nRun = d.GetValue(1);
}"""),
        ("H10 负例: SetFlag 与 WaitFlag 紧邻且该 flag 有 pingpong 两个 id", True, """__aicore__ void f() {
    for (int k = 0; k < n; ++k) {
        event_t ev = (k & 1) ? EVENT_ID1 : EVENT_ID0;
        LoadCbufToCa(a, b);
        SetFlag<HardEvent::MTE1_M>(ev);
        WaitFlag<HardEvent::MTE1_M>(ev);
        Mmad(c, a, b);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    }
}"""),
        ("H11 负例: 循环内同一 flag 置位 2 次而等待 1 次", True, """__aicore__ void f() {
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (int i = 0; i < n; ++i) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        CopyIn(a);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        Compute(a);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}"""),
        ("H11 正例: 循环内 Set 与 Wait 各 1 次 (循环外另有一对)", False, """__aicore__ void f() {
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (int i = 0; i < n; ++i) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        CopyIn(a);
        Compute(a);
        CopyOut(a);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}"""),
        ("H19 负例: UB 被 MTE2 填入后直接被 MTE3 写回 GM, 中间无旗标链", True, """__aicore__ void f() {
    DataCopy(dkInTensor, mm4ResGm[off], 4608);
    DataCopyPad(dkOutGm[dst], dkInTensor, outParam);
}"""),
        ("H19 正例: 中间有 MTE2->MTE3 旗标链", False, """__aicore__ void f() {
    DataCopy(dkInTensor, mm4ResGm[off], 4608);
    SetFlag<HardEvent::MTE2_MTE3>(e);
    WaitFlag<HardEvent::MTE2_MTE3>(e);
    DataCopyPad(dkOutGm[dst], dkInTensor, outParam);
}"""),
        ("H19 正例: 中间夹向量运算, 走 MTE2->V->MTE3 链", False, """__aicore__ void f() {
    DataCopy(dkInTensor, mm4ResGm[off], 4608);
    SetFlag<HardEvent::MTE2_V>(e1);
    WaitFlag<HardEvent::MTE2_V>(e1);
    Muls(dkInTensor, dkInTensor, s, 4608);
    SetFlag<HardEvent::V_MTE3>(e2);
    WaitFlag<HardEvent::V_MTE3>(e2);
    DataCopyPad(dkOutGm[dst], dkInTensor, outParam);
}"""),
        ("H19 提示: UB 传进 Scatter*ToGm 一类调用而无链 (同步可能在被调函数里)", "提示", """__aicore__ void f() {
    DataCopy(dkInTensor, mm4ResGm[off], 4608);
    SetFlag<HardEvent::MTE2_V>(e1);
    WaitFlag<HardEvent::MTE2_V>(e1);
    ScatterRowsToGm<false>(dkOutGm, dvOutGm, dkInTensor, dvInTensor, topkUb, false, gmOffset, n, ci, rows, cur, e2);
}"""),
        ("H12 负例: 循环内先 Wait 后 Set, 而循环外没有初始置位", True, """__aicore__ void f() {
    for (int i = 0; i < n; ++i) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        CopyIn(a);
        Compute(a);
        CopyOut(a);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}"""),
        ("H12 正例: 同行Set后Wait保持字符次序 (VC split-pack E73)", False, """__aicore__ void f() {
    for (int i=0; i<n; ++i) {
        SetFlag<HardEvent::V_MTE2>(vm2); WaitFlag<HardEvent::V_MTE2>(vm2);
        SetFlag<HardEvent::MTE2_V>(m2v); WaitFlag<HardEvent::MTE2_V>(m2v);
    }
}"""),
        ("H12 负例: 同行Wait后Set仍缺prologue", True, """__aicore__ void f() {
    for (int i=0; i<n; ++i) {
        WaitFlag<HardEvent::V_MTE2>(vm2); SetFlag<HardEvent::V_MTE2>(vm2);
    }
}"""),
        ("H12 负例: 同行不同event分别判断次序", True, """__aicore__ void f() {
    for (int i=0; i<n; ++i) {
        SetFlag<HardEvent::V_MTE2>(vm2); WaitFlag<HardEvent::MTE2_V>(m2v); WaitFlag<HardEvent::V_MTE2>(vm2); SetFlag<HardEvent::MTE2_V>(m2v);
    }
}"""),
        ("H12 正例: 循环外有初始置位", False, """__aicore__ void f() {
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (int i = 0; i < n; ++i) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        CopyIn(a);
        Compute(a);
        CopyOut(a);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}"""),
        ("H13 负例: 早退分支内发 PIPE_FIX 通知而该分支无 fixpipe", True, """__aicore__ void f() {
    if (coreOff >= m) {
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(C_NOTIFY_V);
        return;
    }
    Mmad(c, a, b);
    IterateAll(gm);
    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(C_NOTIFY_V);
}"""),
        ("H13 正例: 早退分支内的通知用的流水在该分支上有操作", False, """__aicore__ void f() {
    if (coreOff >= m) {
        DataCopy(gm, ub, params);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(C_NOTIFY_V);
        return;
    }
    Mmad(c, a, b);
}"""),
        ("H10 负例(成员形式): buf.Set<M_FIX>() 紧跟 buf.Wait<M_FIX>(), 且 buf 取自多缓冲池",
         True, """BuffersPolicyDB<BufferType::L0C> mmL0CBuffers;
__aicore__ void f() {
    Buffer<BufferType::L0C> l0c = mmL0CBuffers.Get();
    Mmad(l0c, a, b);
    l0c.Set<HardEvent::M_FIX>();
    l0c.Wait<HardEvent::M_FIX>();
    Fixpipe(ub, l0c);
}
"""),
        ("H10 正例(成员形式): 中间插了另一条 matmul, 已错开", False,
         """BuffersPolicyDB<BufferType::L0C> mmL0CBuffers;
__aicore__ void f() {
    Buffer<BufferType::L0C> l0c = mmL0CBuffers.Get();
    Mmad(l0c, a, b);
    l0c.Set<HardEvent::M_FIX>();
    Mmad(l0c2, c, d);
    l0c2.Set<HardEvent::M_FIX>();
    l0c.Wait<HardEvent::M_FIX>();
    Fixpipe(ub, l0c);
}
"""),
        ("H10 负例: 名字成对 (xxxPing/xxxPong) 且紧邻 -> 双缓冲写成了串行, 要判错误级", True, """
__aicore__ void f() {
    DataCopy(a, gm, n);
    SetFlag<HardEvent::MTE2_V>(gatherPing);
    WaitFlag<HardEvent::MTE2_V>(gatherPing);
    Cast(b, a, RoundMode::CAST_NONE, n);
    DataCopy(c, gm2, n);
    SetFlag<HardEvent::MTE2_V>(gatherPong);
    WaitFlag<HardEvent::MTE2_V>(gatherPong);
}
"""),
        ("H10 正例: Set 与 Wait 之间隔着别的语句 (已错开)", False, """__aicore__ void f() {
    for (int k = 0; k < n; ++k) {
        LoadCbufToCa(a, b);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        LoadCbufToCa(a2, b2);
        Mmad(c, a2, b2);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    }
}"""),
        ("H12 正例: 循环外用 EVENT_ID 字面量置位, 循环内用变量等待 (GemmV3 base 的真实写法)", False, """__aicore__ void f() {
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
    for (int k = 0; k < n; ++k) {
        event_t ev = (1ULL - (k & 0b1)) ? EVENT_ID0 : EVENT_ID1;
        WaitFlag<HardEvent::M_MTE1>(ev);
        LoadData(a, b);
        SetFlag<HardEvent::M_MTE1>(ev);
    }
}"""),
        #  ★H16: CANN 的断言是 debug 级, release 版不触发 -> 编译零告警而上机必崩
        ("H16 负例: fixpipe 双目标模式 + quantPre 类型转换同时开", True, """__aicore__ void f() {
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = 512;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.quantPre = QuantMode_t::QF322F16_PRE;
    Fixpipe<half, float, CFG>(dst, l0C, fixpipeParams);
}"""),
        ("H16 正例: 单目标 + quantPre, 合法, 不该报", False, """__aicore__ void f() {
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.dualDstCtl = 0;
    fixpipeParams.quantPre = QuantMode_t::QF322F16_PRE;
    Fixpipe<half, float, CFG>(dst, l0C, fixpipeParams);
}"""),
        ("H16 正例: 双目标 + NoQuant (出厂 mm2 的写法), 不该报", False, """__aicore__ void f() {
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.quantPre = QuantMode_t::NoQuant;
    Fixpipe<float, float, CFG>(dst, l0C, fixpipeParams);
}"""),
        ("H12 负例: 变量可能取两个 id 而循环外只置了一个 (漏一路, 真死锁)", True, """__aicore__ void f() {
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    for (int i = 0; i < n; ++i) {
        event_t ev = (i & 1) ? EVENT_ID0 : EVENT_ID1;
        WaitFlag<HardEvent::M_MTE1>(ev);
        LoadData(a, b);
        SetFlag<HardEvent::M_MTE1>(ev);
    }
}"""),
        ("H12 提示: id 由运行时算出, 静态定不了取值集合 -> 只许提示不许断言", "提示", """__aicore__ void f() {
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
    for (int k = 0; k < n; ++k) {
        event_t ev = static_cast<event_t>(EVENT_ID0 + (k & 0b11));
        WaitFlag<HardEvent::M_MTE1>(ev);
        LoadData(a, b);
        SetFlag<HardEvent::M_MTE1>(ev);
    }
}"""),
        #  ★H15: 两侧守卫不同 -> 收支只在运行期成立 (H11 的盲区: 它按条件块分组, 各组只看到一侧)
        ("H15 负例: 同一事件的 Set 与 Wait 挂在不同守卫下", "提示", """__aicore__ void f() {
    for (int k = 0; k < n; ++k) {
        if (info.isFirstOfBlock) {
            WaitFlag<HardEvent::MTE2_MTE1>(buf.GetEventID<HardEvent::MTE2_MTE1>());
        }
        Mmad(c, a, b);
        if (info.isLastOfBlock) {
            SetFlag<HardEvent::MTE2_MTE1>(buf.GetEventID<HardEvent::MTE2_MTE1>());
        }
    }
}"""),
        #  ★作用域栈按花括号深度弹出的回归: 前面有一个**已经结束**的 if 块时, 后面的语句
        #    不能被安上那个 if 的条件 (早期按栈长度弹出就会, 从而把正确代码报成不对称)。
        ("H15 正例: 前面有已结束的 if 块, 后面的语句不该被安上它的条件", False, """__aicore__ void f() {
    for (int k = 0; k < n; ++k) {
        if (info.isFirstS2Loop) {
            CopyQuery(a);
        }
        WaitFlag<HardEvent::MTE2_MTE1>(buf.GetEventID<HardEvent::MTE2_MTE1>());
        Mmad(c, a, b);
        SetFlag<HardEvent::MTE2_MTE1>(buf.GetEventID<HardEvent::MTE2_MTE1>());
    }
}"""),
        ("H15 正例: 两侧守卫相同, 不该报", False, """__aicore__ void f() {
    for (int k = 0; k < n; ++k) {
        if (info.isFirstOfBlock) {
            WaitFlag<HardEvent::MTE2_MTE1>(buf.GetEventID<HardEvent::MTE2_MTE1>());
            Mmad(c, a, b);
            SetFlag<HardEvent::MTE2_MTE1>(buf.GetEventID<HardEvent::MTE2_MTE1>());
        }
    }
}"""),
        #  ★H16: 多缓冲池取出的 buffer 从不等自己的反依赖事件 (FIAS 2026-09-14 实撞 507015)
        ("H17 负例: 从 DB 池取出的 L0A buffer 被写入却从不 Wait", True, """BuffersPolicyDB<BufferType::L0A> aPool;
BuffersPolicyDB<BufferType::L0B> bPool;
__aicore__ void f() {
    for (int k = 0; k < n; ++k) {
        auto l0aBuffer = aPool.Get();
        auto l0bBuffer = bPool.Get();
        l0bBuffer.template Wait<HardEvent::M_MTE1>();
        LoadDataToL0A(l0aBuffer.template GetTensor<half>(), a, p, 0, 128, 64);
        LoadDataToL0B(l0bBuffer.template GetTensor<half>(), b, p, 0, 128, 128, 1);
        Mmad(c, l0aBuffer.template GetTensor<half>(), l0bBuffer.template GetTensor<half>(), mp);
        l0bBuffer.template Set<HardEvent::M_MTE1>();
    }
}"""),
        ("H17 正例: 两个池各自 Wait/Set, 不该报", False, """BuffersPolicyDB<BufferType::L0A> aPool;
BuffersPolicyDB<BufferType::L0B> bPool;
__aicore__ void f() {
    for (int k = 0; k < n; ++k) {
        auto l0aBuffer = aPool.Get();
        auto l0bBuffer = bPool.Get();
        l0aBuffer.template Wait<HardEvent::M_MTE1>();
        l0bBuffer.template Wait<HardEvent::M_MTE1>();
        LoadDataToL0A(l0aBuffer.template GetTensor<half>(), a, p, 0, 128, 64);
        LoadDataToL0B(l0bBuffer.template GetTensor<half>(), b, p, 0, 128, 128, 1);
        Mmad(c, l0aBuffer.template GetTensor<half>(), l0bBuffer.template GetTensor<half>(), mp);
        l0aBuffer.template Set<HardEvent::M_MTE1>();
        l0bBuffer.template Set<HardEvent::M_MTE1>();
    }
}"""),
        #  ★H18: 槽数 < 软件流水级数 -> 真机静默挂死 (FIAS 2026-09-14 实撞, 驱动无任何输出)
        ("H18 负例: 三级流水配 2 个跨核槽", "提示", """struct K {
    static constexpr uint32_t KV_SLOT_N = 2;
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> *kvSlots_[KV_SLOT_N];
    void Run() {
        for (uint64_t loop = 0; loop < n; ++loop) {
            auto &r0 = taskRunInfo[loop % PRELOAD_N];
            Preload(r0);
            if (loop >= 1UL) { Mm1(taskRunInfo[(loop - 1UL) % PRELOAD_N]); }
            if (loop >= 2UL) { Mm2(taskRunInfo[(loop - 2UL) % PRELOAD_N]); }
        }
    }
};"""),
        ("H18 正例: 三级流水配 3 个跨核槽, 不该报", False, """struct K {
    static constexpr uint32_t KV_SLOT_N = 3;
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> *kvSlots_[KV_SLOT_N];
    void Run() {
        for (uint64_t loop = 0; loop < n; ++loop) {
            auto &r0 = taskRunInfo[loop % PRELOAD_N];
            Preload(r0);
            if (loop >= 1UL) { Mm1(taskRunInfo[(loop - 1UL) % PRELOAD_N]); }
            if (loop >= 2UL) { Mm2(taskRunInfo[(loop - 2UL) % PRELOAD_N]); }
        }
    }
};"""),
        #  ★运行期槽数: 只有**确实被用来索引槽环**的那个变量才算槽数; 序列长度之类的取模数不算
        ("H18 正例: 取模数是序列长度而非槽数, 不该报", False, """struct K {
    static constexpr uint32_t KV_SLOT_N = 3;
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> *kvSlots_[KV_SLOT_N];
    void Run() {
        uint32_t actSeqLensQ_ = 1;
        for (uint64_t loop = 0; loop < n; ++loop) {
            auto &r0 = taskRunInfo[loop % PRELOAD_N];
            info.s1Idx = info.gS1Idx % actSeqLensQ_;
            if (loop >= 2UL) { Mm2(taskRunInfo[(loop - 2UL) % PRELOAD_N]); }
        }
    }
};"""),
        #  ★H18 第二症状: 槽数够, 但槽号的来源会**回绕** -> 复用间隔在边界上局部塌陷
        ("H18 负例: 槽号取自会被重新赋值的块下标 (边界回绕)", "提示", """struct K {
    static constexpr uint32_t KV_SLOT_N = 3;
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> *kvSlots_[KV_SLOT_N];
    void Calc(uint32_t s2Cur) { info.kvSlot = s2Cur % KV_SLOT_N; }
    void Mm1(RunInfoX &r) { Bmm1(*kvSlots_[r.kvSlot], r); }
    void Advance(uint32_t &s2Cur) {
        s2Cur++;
        if (s2Cur >= chunkEnd_) { s2Cur = chunkStart_; }
    }
    void Run() {
        for (uint64_t loop = 0; loop < n; ++loop) {
            auto &r0 = taskRunInfo[loop % PRELOAD_N];
            if (loop >= 2UL) { Mm2(taskRunInfo[(loop - 2UL) % PRELOAD_N]); }
        }
    }
};"""),
        ("H18 正例: 槽号取自只增不减的块计数, 不该报", False, """struct K {
    static constexpr uint32_t KV_SLOT_N = 3;
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> *kvSlots_[KV_SLOT_N];
    void Calc(uint32_t s2Cur) {
        if (s2Cur != prevSlotS2_) { blockSeq_++; }
        prevSlotS2_ = s2Cur;
        info.kvSlot = blockSeq_ % KV_SLOT_N;
    }
    void Mm1(RunInfoX &r) { Bmm1(*kvSlots_[r.kvSlot], r); }
    void Run() {
        for (uint64_t loop = 0; loop < n; ++loop) {
            auto &r0 = taskRunInfo[loop % PRELOAD_N];
            if (loop >= 2UL) { Mm2(taskRunInfo[(loop - 2UL) % PRELOAD_N]); }
        }
    }
};"""),
        #  ★事件号写成 `buf.GetEventID<E>()` 时, 原模式整条匹配不上 -> H11 对这类写法全盲
        ("H11 负例(成员调用形式的事件号): 循环内 Set 2 次 Wait 1 次", True, """__aicore__ void f() {
    for (int k = 0; k < n; ++k) {
        WaitFlag<HardEvent::MTE1_MTE2>(buf.GetEventID<HardEvent::MTE1_MTE2>());
        SetFlag<HardEvent::MTE1_MTE2>(buf.GetEventID<HardEvent::MTE1_MTE2>());
        SetFlag<HardEvent::MTE1_MTE2>(buf.GetEventID<HardEvent::MTE1_MTE2>());
    }
}"""),
        #  ---- 2026-10-28 补: 这六条此前**一条自检用例都没有**, 覆盖守卫一直在报。
        #    判据在而没有用例 = 它上线后是否还抓得住无人知晓。负例进本表 (守卫按它统计),
        #    配对的正例在下面的独立块里直接调对应函数 —— 正例若进本表, 要求 27 条判据
        #    全不命中, 太脆, 一条无关判据的改动就会把它弄红。
        #    ★各条的最小输入形态是**实测出来的**, 不是读码猜的: split_funcs 要 `::Name(`
        #    的类外定义形式, check_atomic_write_granule 的长度里必须带 sizeof(T) 才能定元素宽度,
        #    DataCopyPad 的目的还得带下标。按直觉写的第一版六条全部不命中。
        ("H9 负例: 原子区间内写长度 tailLen*sizeof(float) 没有对齐动作", "提示", """__aicore__ inline void K::Store() {
    SetAtomicAdd<float>();
    DataCopyExtParams p { 1, tailLen * sizeof(float), 0, 0, 0 };
    DataCopyPad(dstGm[offset], src, p);
    SetAtomicNone();
}"""),
        ("H28 负例: 全核屏障之前有 return", True, """__aicore__ void f() {
    if (blockIdx >= used) { return; }
    SyncAll();
}"""),
        ("H4 负例: static_cast<float> 直转无符号变量", "提示", """__aicore__ void f() {
    uint32_t n = t.count;
    float x = static_cast<float>(n - 1);
}"""),
        ("H5 负例: 深度 1 的 TQue 连 EnQue 两次才 DeQue", "提示", """TQue<TPosition::VECIN, 1> q;
__aicore__ inline void K::Process() {
    LocalTensor<half> a = q.AllocTensor<half>();
    q.EnQue(a);
    LocalTensor<half> b = q.AllocTensor<half>();
    q.EnQue(b);
    LocalTensor<half> c = q.DeQue<half>();
    q.FreeTensor(c);
}"""),
        ("H3 负例: 整维 n 载入按 baseN 分配的缓冲", "提示", """__aicore__ inline void K::Init() {
    pipe.InitBuffer(tileBuf, 2, baseN * sizeof(float));
}
__aicore__ inline void K::Process() {
    LocalTensor<float> t = tileBuf.AllocTensor<float>();
    DataCopyExtParams p { 1, n * sizeof(float), 0, 0, 0 };
    DataCopyPad(t, srcGm, p, padParams);
}"""),
        ("H8 负例: 清零范围由区段偏移相减得出", "提示", """__aicore__ void f() {
    uint32_t endOffset = t.hi;
    uint32_t beginOffset = t.lo;
    uint32_t span = endOffset - beginOffset;
    InitOutput<float>(dstGm[beginOffset], span);
}"""),
        #  ★H30: 一个核里两个 Matmul 实例, 发射点之间没有全屏障。
        #    覆盖守卫按**这张表**统计判据有没有被测到 —— 判别力的细分用例在下面
        #    那个独立块里, 但不进这张表就会被报成"没有任何自检用例"。
        ("H30 负例: 两个 Matmul 实例的发射点之间无全屏障", True, """class K {
    MatmulImpl<A, B, C, D, CFG> mmq;
    MatmulImpl<E, F, G, H, CFG> mmv;
    __aicore__ void P() {
        for (int i = 0; i < n; ++i) {
            mmq.IterateAll(s); mmq.End();
            CrossCoreWaitFlag(P_READY);
            mmv.IterateAll(o); mmv.End();
        }
    }
};"""),
    ]
    print(f"=== detsync_lint 判据负例自检 ({len(cases)} 条) ===")
    bad = 0
    _covered = set()
    for name, should_hit, code in cases:
        with tempfile.NamedTemporaryFile("w", suffix=".h", delete=False) as f:
            f.write(code); path = f.name
        code = open(path).read()
        #  ★判据调用串**自动收集**, 不写死。写死的那版每加一条判据要同步改两处
        #  (这里 + 判据前缀白名单), 漏改的症状是自检**静默地不测它** —— 用例挂在那里
        #  永远"不命中", 看起来像判据失效。2026-09-15 加 H24 时又撞了一次:
        #  前缀白名单上次已改成自动取, 而这串调用还是手写的。
        hz = []
        for _fn in _selftest_checks():
            _got = _fn(code)
            if _got:
                _covered.add(_fn.__name__)
            hz += _got
        #  判据前缀直接从用例名的首个 token 取 (`H18 负例: ...` -> `H18_`)。
        #  原来这里是一张写死的白名单 ("H10","H11",...), 每加一条判据就要同步改一次;
        #  漏改的症状是**自检静默地不测它** —— 用例挂在那里永远"不命中", 看起来像判据失效。
        _tok = name.split()[0]
        want = (_tok + "_") if re.fullmatch(r"H\d+", _tok) else "H7"
        matched = [h for h in hz if h.get("check", "").startswith(want)]
        #  ★"严重" 也是错误及以上。此前这里硬编码 `== "错误"`, 于是
        #    check_early_return_before_barrier (全模块唯一一条发 "严重" 的判据)
        #    永远满足不了 True 断言, 被当成提示级 —— 判据在、级别也对,
        #    而**自检的接线把它丢了**, 与本会话反复撞的那一形态同类。
        errs = [h for h in matched if h.get("level") in ("错误", "严重")]
        #  三态断言。加"提示"这一档是因为: 把一条错误级判据悄悄降成提示, 就能骗过
        #  "命中/不命中"的二态自检 —— 那等于判据放水而自检还是绿的。
        if should_hit is True:
            ok, want_txt = bool(errs), "错误级命中"
        elif should_hit is False:
            ok, want_txt = not matched, "不命中"
        else:
            ok, want_txt = (bool(matched) and not errs), "提示级命中(不许错误级)"
        got_txt = ("错误级" if errs else "提示级") if matched else "不命中"
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} {name}   (期望{want_txt}, 实际{got_txt})")
        os.unlink(path)
    #  ★覆盖守卫 (2026-09-15 从 hangdiag 移植): 加了判据却没加自检用例时, 用例会静默地
    #  不被任何断言覆盖 —— 看起来判据在, 实际它上线后还抓不抓得住无人知晓。
    #  这里比对模块里所有 check_* 函数与自检实际调用的集合。
    #  覆盖 = 该判据在**某条自检用例上真的产出过命中**, 运行时统计, 不写白名单。
    #  白名单版有个说不出口的毛病: 加判据时顺手把名字填进去, 守卫就绿了, 而用例
    #  一条没写。2026-09-15 加 H24 时这张表已是本文件里第三处写死的判据清单
    #  (另两处: 判据调用串、判据前缀), 一并改掉。
    _all = {f.__name__ for f in _selftest_checks()}
    #  ★位置有两个不变量, 缺一不可 (2026-10-28 连摆三次才对):
    #    1) 必须在 selftest() 中途那次 `bad = 0` **之后** —— 放在它之前,
    #       自检会照常打印"未过"而 bad 被抹掉, 模块仍报 PASS;
    #    2) 必须在覆盖守卫算 `_missing` 的那行**之前** —— 守卫靠记录
    #       自检期间哪些 check_* 被调用过来判覆盖, 放在它之后,
    #       check_share 会被报成"没有任何自检用例", 尽管用例就在下面。
    #    两个不变量都用"注入一个故意写错的期望"验过, 不靠读代码判断。
    _sh_bad = ("class K {\n  MatmulImpl<A,B,C,D,CFG> mmq;\n  MatmulImpl<E,F,G,H,CFG> mmv;\n"
               "  void P(){ for(int i=0;i<n;++i){ mmq.IterateAll(s); mmq.End();\n"
               "    CrossCoreWaitFlag(P_READY);\n    mmv.IterateAll(o); mmv.End(); } }\n};\n")
    _sh_ok = _sh_bad.replace("CrossCoreWaitFlag(P_READY);",
                             "CrossCoreWaitFlag(P_READY); PipeBarrier<PIPE_ALL>();")
    _sh_one = ("class K {\n  MatmulImpl<A,B,C,D,CFG> mm;\n"
               "  void P(){ for(int i=0;i<n;++i){ mm.IterateAll(s); mm.End(); } }\n};\n")
    _sh_evt = "void f(){ auto e = pipe.FetchEventID(HardEvent::MTE3_S); }\n"
    _sh_alloc = "void f(){ auto e = pipe.AllocEventID<HardEvent::MTE3_S>(); }\n"
    for _nm, _src, _want30, _want31 in (
            ("H30 负例: 两个实例的发射点之间无全屏障 -> 错误级", _sh_bad, True, False),
            ("H30 正例: 有 PipeBarrier<PIPE_ALL> 的那个方向不报", _sh_ok, True, False),
            ("H30 正例: 只有一个实例, 不该报", _sh_one, False, False),
            ("H31 负例: FetchEventID 取号要提示 (只查不占)", _sh_evt, False, True),
            ("H31 正例: AllocEventID 占号不报", _sh_alloc, False, False)):
        _r = check_share(_src)
        _h30 = any(h["check"].startswith("H30") for h in _r)
        _h31 = any(h["check"].startswith("H31") for h in _r)
        _ok = (_h30 == _want30 and _h31 == _want31)
        bad += 0 if _ok else 1
        print(f"  {'过' if _ok else '**未过**'} {_nm}   (H30{'命中' if _h30 else '不命中'}, "
              f"H31{'命中' if _h31 else '不命中'})")
    #  ---- 2026-10-28 补: 上面六条负例的**配对正例**, 防判据过宽。
    #    正例不进 cases 表 —— 那张表要求 27 条判据全不命中, 一条无关判据的改动
    #    就会把它弄红; 这里直接调对应函数, 只对该判据断言。
    for _nm, _fn, _src in (
            ("H9 正例: 长度有 AlignUp 对齐动作 -> 不报",
             check_atomic_write_granule, """__aicore__ inline void K::Store() {
    const uint32_t tailLen = AlignUp(rawLen, 8);
    SetAtomicAdd<float>();
    DataCopyExtParams p { 1, tailLen * sizeof(float), 0, 0, 0 };
    DataCopyPad(dstGm[offset], src, p);
    SetAtomicNone();
}"""),
            ("H28 正例: return 在屏障之后 -> 不报",
             check_early_return_before_barrier, """__aicore__ void f() {
    SyncAll();
    if (blockIdx >= used) { return; }
}"""),
            ("H4 正例: 转的是有符号变量 -> 不报",
             check_float_uint_cast, """__aicore__ void f() {
    int32_t n = t.count;
    float x = static_cast<float>(n - 1);
}"""),
            ("H5 正例: 深度 2 的 TQue 配两次 EnQue -> 不报",
             check_queue_depth, """TQue<TPosition::VECIN, 2> q;
__aicore__ inline void K::Process() {
    LocalTensor<half> a = q.AllocTensor<half>();
    q.EnQue(a);
    LocalTensor<half> b = q.AllocTensor<half>();
    q.EnQue(b);
    LocalTensor<half> c = q.DeQue<half>();
    q.FreeTensor(c);
}"""),
            ("H3 正例: 载入长度与分配的分块宽度同源 -> 不报",
             check_ub_oob, """__aicore__ inline void K::Init() {
    pipe.InitBuffer(tileBuf, 2, baseN * sizeof(float));
}
__aicore__ inline void K::Process() {
    LocalTensor<float> t = tileBuf.AllocTensor<float>();
    DataCopyExtParams p { 1, baseN * sizeof(float), 0, 0, 0 };
    DataCopyPad(t, srcGm, p, padParams);
}"""),
            ("H8 正例: 清零范围是常量, 不由偏移相减得出 -> 不报",
             check_zero_span, """__aicore__ void f() {
    InitOutput<float>(dstGm, 1024);
}""")):
        _got = _fn(_src)
        _ok = not _got
        bad += 0 if _ok else 1
        print(f"  {'过' if _ok else '**未过**'} {_nm}   (实得 {[h.get('check') for h in _got] or '不命中'})")

    #  ★级别判别 (2026-10-28 用 CANN 发行语料的 57 个多实例内核标定):
    #    不区分路径时错误级 38 条, 其中 36 条是误报 —— 它们走公开类型/KFC 客户端,
    #    cube 由服务端按消息执行, 不是本地并发。区分后错误级剩 2 条 (同一文件两处路径)。
    _sh_pub = _sh_bad.replace("MatmulImpl", "Matmul")
    _sh_reg = _sh_bad.replace("mmq.IterateAll",
                              "REGIST_MATMUL_OBJ(&pipe, ws, mmq, &t1, mmv, &t2); mmq.IterateAll")
    for _nm, _src, _want in (
            ("H30 级别: 显式 MatmulImpl 且无 REGIST -> 错误级", _sh_bad, "错误"),
            ("H30 级别: 公开 Matmul 类型 -> 只给提示 (KFC 客户端, 不是本地并发)", _sh_pub, "提示"),
            ("H30 级别: 有 REGIST_MATMUL_OBJ -> 只给提示", _sh_reg, "提示")):
        _lv = {h["level"] for h in check_share(_src) if h["check"].startswith("H30")}
        _ok = _lv == {_want}
        bad += 0 if _ok else 1
        print(f"  {'过' if _ok else '**未过**'} {_nm}   (实得 {_lv or '无命中'})")

    #  ★正例 _sh_ok 仍应命中一条 H30 —— 那是**循环回边**方向 (mmv -> 下一轮 mmq),
    #    它确实没有全屏障。漏报这个方向等于只堵一边, 故必须报出来由人判。
    _wins = share_report(_sh_ok)["windows"]
    _ok = len(_wins) == 2 and _wins[1].get("back_edge") and "PIPE_ALL" not in _wins[1]["barriers"]
    bad += 0 if _ok else 1
    print(f"  {'过' if _ok else '**未过**'} H30 负例: 循环回边方向必须单独成窗并报出")
    #  ★TSCM<TPosition::GM,2> 的首参是**源**位置, 容器自己在 L1 —— 不得按首参归成 GM
    _ok = "L1" in share_report("TSCM<TPosition::GM, 2> vScm;\n")["bufs"]
    bad += 0 if _ok else 1
    print(f"  {'过' if _ok else '**未过**'} share: TSCM 按容器归 L1, 不按首个模板实参归 GM")

    """判据负例自检: 已知违规必须被抓住, 已知合法不许误报 (判据过宽会变成狼来了)。"""


    _missing = sorted(_all - _covered)
    if _missing:
        bad += 1
        print(f"  **未过** 覆盖守卫: 这些判据没有任何自检用例能让它命中: {_missing}")
        print(f"           (共 {len(_all)} 条判据, {len(_all) - len(_missing)} 条有用例)")

    #  ★2026-10-29 补 (`opdbg --mutation` 指出 `analyze` 是摆设): 上面所有用例跑的是
    #    `_selftest_checks()` 里的**单个判据函数**, 从不经过 `analyze` —— 而 `analyze`
    #    才是生产路径上把这些判据**组装**起来的那个。于是"判据单测全绿、而 analyze 漏了
    #    其中一条"这件事, 整套自检看不见。下面两条把它堵上。
    import inspect as _insp
    #  A) 行为: analyze 必须真的把 hazard 吐出来 (判据被废掉时这条立刻红)
    _probe_src = """
__aicore__ inline void VecOp::G(const RunInfo &r) {
    DataCopyPad(gatherTensor[0], keyGm[off], intriParams, padParams);
    DataCopyPad(selectedKWorkspaceGm[wsAddr], gatherTensor, outParamK);
}
"""
    _a_hz = analyze(_probe_src)
    _a_ok = any(h.get("check", "").startswith("H19") for h in _a_hz)
    bad += 0 if _a_ok else 1
    print(f"  [{'OK ' if _a_ok else 'FAIL'}] analyze 组装后仍能吐出 H19 "
          f"(实得 {sorted({h.get('check','') for h in _a_hz})})")

    #  B) 结构: 凡被自检单独测到的判据, 都必须出现在 `analyze` 的生产路径里。
    #     这条抓的是"判据写了、单测也绿, 而 analyze 忘了调它" —— 单测永远发现不了。
    #     ★两条**按设计**不在 analyze 里: main 另行调用它们 (要 shape/字节数参数,
    #       analyze 的签名里没有)。它们列在这里是**声明**, 不是豁免清单的省事写法。
    _outside_analyze = {"check_ub_oob", "check_dma_granularity"}
    try:
        _asrc = _insp.getsource(analyze)
    except (OSError, TypeError):
        _asrc = ""
    _called = set(re.findall(r"\b(check_\w+)\s*\(", _asrc))
    _tested = {fn.__name__ for fn in _selftest_checks()}
    _missing = sorted(_tested - _called - _outside_analyze)
    bad += 0 if not _missing else 1
    print(f"  [{'OK ' if not _missing else 'FAIL'}] analyze 覆盖全部被测判据"
          + (f" —— **漏了 {_missing}**: 它们单测绿而生产路径不调, 等于没上线" if _missing
             else f" ({len(_tested - _outside_analyze)} 条)"))

    print(f"detsync_lint 自检: {len(cases)} 条, {bad} 条未过")
    bad += _selftest_h16()
    bad += _selftest_h10_level()
    bad += _selftest_print_path()
    return 1 if bad else 0


def _selftest_print_path() -> int:
    """★把每条判据的命中记录**真的渲染一遍** —— 自检此前只走返回值, 不走打印路径。

    2026-09-18 实撞 (第二次盲测审计揪出): H25/H26 用 `err=` 建记录, 而打印处硬取
    `h['msg']` 与 `h['func']` -> **判据一命中就 KeyError**。后果:
      单文件模式打印到第 4 条就崩, 后面 41 条全丢; 目录模式在第 6 个文件崩, 后 6 个没扫到;
      `opdbg sweep` 把崩溃的退出码 1 当成"有命中", 报"0 个异常退出"。
      **这两条判据从加进来那天起一次都没成功打印过**, 而 --selftest-criteria 走返回值,
      结构上抓不到这一类。
    判法: 取每条判据在自检用例上产出的记录, 用与 main 相同的字段访问方式渲染一次。
    """
    bad, n = 0, 0
    cases = {
        "check_dma_granularity": "__aicore__ void C::f() { DataCopyExtParams p; p.blockLen = 128; }",
        "check_core_partition_remainder":
            ("__aicore__ void C::p() { int64_t first = total / nCores; "
             "int64_t cur = (idx == nCores - 1) ? (total % nCores + first) : first; }"),
        "check_crosscore_barrier_granularity":
            ("__aicore__ void C::g() { for (int i=0;i<n;i++) { "
             "CrossCoreWaitFlag<0,PIPE_MTE3>(F); SetAtomicAdd<float>(); "
             "CrossCoreSetFlag<0,PIPE_MTE3>(F); } }"),
    }
    cases["check_mix_vector_core_index"] = """
#ifdef __DAV_C220_CUBE__
void C(){ auto core=GetBlockIdx(); for(int task=core;task<n;task+=cores){CrossCoreSetFlag<2,PIPE_FIX>(F);} }
#endif
#ifdef __DAV_C220_VEC__
void V(){ auto core=GetBlockIdx(); for(int task=core;task<n;task+=cores){CrossCoreWaitFlag(F);} }
#endif
"""
    for fname, src in cases.items():
        fn_ = globals().get(fname)
        if not fn_:
            #  ★查不到判据函数要**算未过**, 不能跳过: 改名或删掉一条判据之后,
            #    这里静默少测一条, 而收尾照样打"n 条命中记录全部可渲染" ——
            #    一条自检用"我没测"冒充"我测过了"
            #    (2026-10-26, silentskip 扫出, 09 第 177 条)。
            print(f"  **未过** 打印路径: 自检点名的判据 `{fname}` 在本模块里不存在 "
                  f"(改名了? 删了?) —— 这一条**没测**, 不是测过没问题")
            bad += 1
            continue
        for h in fn_(src):
            n += 1
            try:
                #  与 main 的两处打印完全相同的字段访问方式
                _ = f"[{h['check']}] {h.get('buffer','')}"
                _ = f"{(h.get('msg') or h.get('err') or '(无消息)')}"
                _ = f" (在 {h['func']})" if h.get("func") else ""
            except Exception as e:
                print(f"  **未过** 打印路径: {fname} 的记录渲染时 {type(e).__name__}: {e}")
                bad += 1
    print(f"  {'过' if not bad else '**未过**'} 打印路径自检: {n} 条命中记录全部可渲染"
          f" (判据一命中就崩这一类, 返回值自检抓不到)")
    return bad


def _selftest_h10_level() -> int:
    """H10 的**级别**自检: 紧邻一律报, 但"错误级"只留给真的双缓冲。

    通用自检框架只判命中与否, 判不了级别 —— 而这次收窄改的正是级别:
    2026-09-16 用 60 个真实算子普查, 按"这个文件里该事件类型有几个 id"判双缓冲时,
    sfag (12/12 Pass 的良品) 被报 18 条错误级, conv3d_backprop_input_v2 被报 9 条。
    """
    bad = 0
    print("=== H10 级别自检 (错误级只留给真双缓冲) ===")
    #  正例: 名字不成对 = 不同用途 (sfag 的 castEvent / cvtEvent 就是这样), 只该提示
    a = ("__aicore__ void f() {\n    DataCopy(a, gm, n);\n"
         "    SetFlag<HardEvent::MTE2_V>(castEvent);\n    WaitFlag<HardEvent::MTE2_V>(castEvent);\n"
         "    Cast(b, a, RoundMode::CAST_NONE, n);\n    DataCopy(c, gm2, n);\n"
         "    SetFlag<HardEvent::MTE2_V>(cvtEvent);\n    WaitFlag<HardEvent::MTE2_V>(cvtEvent);\n}\n")
    lv = {h["level"] for h in check_setwait_adjacent(a)}
    ok = lv == {"提示"}
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} 正例: 名字不成对 -> 只提示不报错误级 (实得 {lv})")
    #  负例一: 名字成对 xxxPing / xxxPong
    b = a.replace("castEvent", "gatherPing").replace("cvtEvent", "gatherPong")
    ok = "错误" in {h["level"] for h in check_setwait_adjacent(b)}
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} 负例: 名字成对 (Ping/Pong) -> 错误级")
    #  负例二: 从两个值里选出来的 id —— 文本上只有一个名字, 按"有几个 id"看不见
    c = ("__aicore__ void f() {\n  for (int k = 0; k < n; ++k) {\n"
         "    event_t ev = pingPongFlag ? EVENT_ID1 : EVENT_ID0;\n    DataCopy(a, gm, n);\n"
         "    SetFlag<HardEvent::MTE2_V>(ev);\n    WaitFlag<HardEvent::MTE2_V>(ev);\n  }\n}\n")
    ok = "错误" in {h["level"] for h in check_setwait_adjacent(c)}
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} 负例: `ev = flag ? ID1 : ID0` 选出来的 -> 错误级")
    print(f"H10 级别自检: 3 条, {bad} 条未过")
    return bad


def _selftest_h16() -> int:
    """H16 自检: 跨文件的事件总数失衡要抓住, 而正常的跨文件配对不许误报。"""
    bad = 0
    print("=== H16 自检 (跨文件事件 Set/Wait 总数) ===")
    #  负例: sfag v22a 的真实形态 —— 置位在 A 文件, 等待在 B 文件, 且少一次
    a = ("__aicore__ inline void K::Init() {\n"
         "    SET_FLAG(MTE3, MTE2, gatherEv);\n"
         "    SET_FLAG(MTE3, MTE2, gatherEvPong);\n}\n")
    b = ("__aicore__ inline void V::Gather() {\n"
         "    WAIT_FLAG(MTE3, MTE2, gatherEv);\n}\n")
    hz = check_crossfunction_flag_balance({"a.h": a, "b.h": b})
    got = {h["buffer"]: h["level"] for h in hz}
    ok = got.get("MTE3_MTE2(gatherEvPong)") == "错误"
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} 负例: 跨文件少一次等待且不在循环 -> 错误级 (实得 {got})")
    #  正例: 跨文件但总数对得上 —— 分配时预置、释放时收回是正常写法, 不许报
    #  (第一版报"分散在多个函数", 真仓 14 条全是这个, 纯噪声)
    b2 = ("__aicore__ inline void V::Gather() {\n"
          "    WAIT_FLAG(MTE3, MTE2, gatherEv);\n"
          "    WAIT_FLAG(MTE3, MTE2, gatherEvPong);\n}\n")
    hz = check_crossfunction_flag_balance({"a.h": a, "b.h": b2})
    ok = not hz
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} 正例: 跨文件总数相等 -> 不报 (实得 {[h['buffer'] for h in hz]})")
    #  提示级: 一侧在循环里 —— 词法次数不等于运行期次数, 判不了对错
    c = ("void K::Loop() {\n    for (int i = 0; i < n; ++i) {\n"
         "        SET_FLAG(V, MTE2, evLoop);\n    }\n"
         "    WAIT_FLAG(V, MTE2, evLoop);\n    WAIT_FLAG(V, MTE2, evLoop);\n}\n")
    hz = check_crossfunction_flag_balance({"c.h": c})
    ok = len(hz) == 1 and hz[0]["level"] == "提示"
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} 提示级: 一侧在循环里 -> 不判错误 "
          f"(实得 {[(h['buffer'], h['level']) for h in hz]})")
    #  提示级: id 是**形参** —— 另一侧在调用方且换了名字, 本判据按名字匹配看不到
    d = ("__aicore__ inline void K::Fix(const uint32_t eventId)\n{\n"
         "    SetFlag<HardEvent::FIX_M>(eventId);\n}\n")
    hz = check_crossfunction_flag_balance({"d.h": d})
    ok = len(hz) == 1 and hz[0]["level"] == "提示" and "形参" in hz[0]["msg"]
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} 提示级: id 是形参 -> 降级并说明看不到调用方 "
          f"(实得 {[(h['buffer'], h['level']) for h in hz]})")
    #  两种写法都要认 —— 真仓同一个算子里宏与模板混用
    e = ("void A::f() {\n    SetFlag<HardEvent::MTE3_MTE2>(mix);\n}\n")
    f = ("void B::g() {\n    WAIT_FLAG(MTE3, MTE2, mix);\n    WAIT_FLAG(MTE3, MTE2, mix);\n}\n")
    hz = check_crossfunction_flag_balance({"e.h": e, "f.h": f})
    ok = len(hz) == 1 and "mix" in hz[0]["buffer"]
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} 负例: 模板写法与宏写法要能对上账 "
          f"(实得 {[h['buffer'] for h in hz]})")
    #  正例: pingpong 的位掩码写法 —— `x` 与 `x & 1` 是同一个事件, 不许拆成两个
    #  (2026-09-16 跨算子验证实撞: ops-nn 的 conv3d_backprop_filter_v2 / _input_v2
    #   各被误报两条, 原因就是按原文匹配把掩码版本当成了另一个 id)
    g = ("void C::a() {\n    SetFlag<HardEvent::M_MTE1>(l0aPingPongFlag_);\n}\n")
    h = ("void C::b() {\n    WaitFlag<HardEvent::M_MTE1>(l0aPingPongFlag_ & 1);\n}\n")
    hz = check_crossfunction_flag_balance({"g.h": g, "h.h": h})
    ok = not hz
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} 正例: `x` 与 `x & 1` 是同一事件, 不许拆开 "
          f"(实得 {[x['buffer'] for x in hz]})")
    #  ★三元选相的实参: 取"左括号后第一个标识符"会取到**条件变量**, 不是标志名。
    #  2026-09-17 sfag 实撞: 账表里凭空多出 taskMod / taskMod1 两个假标志,
    #  而真正的 CUBE_WAIT_VEC_PING/PONG 那一对的收支从来没被算过。
    _t = """
      void f() {
        CrossCoreSetFlag<2, PIPE_MTE3>(m == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
        CrossCoreWaitFlag<2, PIPE_MTE2>(m == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
        CrossCoreSetFlag<2, PIPE_FIX>(SCATTER_SYNC_FLAG);
      }
    """
    got = {_cc_flag_channel(_t, _t.index("(", _t.index("CrossCoreSetFlag"))),
           _cc_flag_channel(_t, _t.index("(", _t.index("CrossCoreWaitFlag"))),
           _cc_flag_channel(_t, _t.index("(", _t.index("PIPE_FIX>") ))}
    ok = got == {"CUBE_WAIT_VEC_*", "SCATTER_SYNC_FLAG"}
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} 三元选相取通道名, 不取条件变量 (实得 {sorted(got)})")
    #  正例: 普通实参不受影响
    _t2 = "CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);"
    ok = _cc_flag_channel(_t2, _t2.index("(", 20)) == "SCATTER_SYNC_FLAG"
    bad += 0 if ok else 1
    print(f"  {'过' if ok else '**未过**'} 普通实参照旧取标志名")
    # E83: helper boundaries remove lexical loop context; array cells may alias.
    array_cases = [
        ("void seed(){\n for(int s=0;s<2;++s){\n SetFlag<HardEvent::MTE1_MTE2>(a[s]);\n }\n}\n"
         "void phase(){\n WaitFlag<HardEvent::MTE1_MTE2>(a[0]);\n}\n", "提示", "动态cell正向"),
        ("void phase(){\n SetFlag<HardEvent::MTE2_MTE1>(a[0]);\n}\n"
         "void use(int slot){\n WaitFlag<HardEvent::MTE2_MTE1>(a[slot]);\n}\n", "提示", "动态cell反向"),
        ("SetFlag<HardEvent::MTE2_V>(a[1]);\nWaitFlag<HardEvent::MTE2_V>(a[0]);\n", "错误", "固定不同cell"),
        ("SetFlag<HardEvent::MTE2_V>(a[0x1u]);\nWaitFlag<HardEvent::MTE2_V>(a[0u]);\n", "错误", "hex固定cell"),
        ("SetFlag<HardEvent::MTE2_V>(a[i]);\nWaitFlag<HardEvent::MTE2_V>(b[0]);\n", "错误", "不同数组"),
        ("WaitFlag<HardEvent::MTE2_V>(a[0]);\n", "错误", "完全缺另一侧"),
        ("SetFlag<HardEvent::MTE2_V>(a[i]);\nWaitFlag<HardEvent::V_MTE2>(a[0]);\n", "错误", "不同事件"),
        ("SetFlag<HardEvent::MTE2_V>(a[i]);\nSetFlag<HardEvent::MTE2_V>(a[i]);\n"
         "WaitFlag<HardEvent::MTE2_V>(a[i]);\n", "错误", "同cell次数不等"),
    ]
    for text, level, label in array_cases:
        rows = check_crossfunction_flag_balance({"array.h": text})
        ok = bool(rows) and all(row["level"] == level for row in rows)
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} {label} -> {level}")
    print(f"H16 自检: 16 条, {bad} 条未过")
    return bad


#  ---- H16: 同一 pipe 内事件的**跨函数/跨文件**净计数 -----------------------------------------
#  H11 只数**同一条件块内**的次数, H15 只看单文件内的词法守卫 —— 两条都在自己的注释里写明了
#  同一个盲区: "配对写在调用方时看不见"。sfag 2026-09-16 的 v22a 正是这个形态:
#  det 段在 ..._bs1_basic.h 里连发两次 SET_FLAG(MTE3, MTE2, gather...), 对应的 WAIT_FLAG
#  在 ..._vec_op.h 的另一个函数里。单看任一文件都是"只有一侧", 两条已有判据各自跳过。
#
#  ★两次收窄, 都由真仓打回:
#    1. 第一版按函数聚合 -> 真仓 15 条, 其中唯一那条"错误"是**函数边界识别错**造成的:
#       l0c_event 的 Wait(243 行) 与 Set(275 行) 本在同一函数, 被切到了两个"函数"里。
#       C++ 的函数头形态太多 (模板/尾置返回/宏修饰), 精确切分不可靠 —— 干脆不依赖它,
#       改按 文件:行 记位置。判据的结论本来就只需要**总数**。
#    2. 第一版还报"全局次数相等但分散在多个函数"。真仓 14 条全是 AllocEventID()/FreeEventID()
#       这种分配时预置、释放时收回的对称写法 —— 正常且常见。次数相等就不该占版面。
#
#  ★本判据数的是**源码里出现的次数**, 不是运行期次数 —— 循环里的一次 Set 可以配多次 Wait。
#    两侧不在词法循环时保留既有错误级；形参或同数组动态cell未求解时只能提示。
#    E83 证实 helper 内的 [0] 与循环里的 [s]/[slot] 不能按字面键证明缺项。
_H16_MACRO = re.compile(r"\b(SET_FLAG|WAIT_FLAG)\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*([^)]+?)\s*\)")
_H16_TMPL = re.compile(r"\b(SetFlag|WaitFlag)\s*<\s*(?:AscendC::)?HardEvent::(\w+)\s*>\s*\(\s*([^)]*?)\s*\)")


def _h16_norm_id(eid: str) -> str:
    """事件 id 表达式归一化。

    ★2026-09-16 跨算子验证 (ops-nn 60 个算子) 发现: conv3d_backprop_filter_v2 里
      `SetFlag<M_MTE1>(l0aPingPongFlag_)` 与 `WaitFlag<M_MTE1>(l0aPingPongFlag_ & 1)`
      指的是**同一个** pingpong 事件, 而按原文匹配会拆成两个 id, 于是两边各自"只有一侧",
      报出两条提示。掩码 `& 1` / `& 0x1` 是取 pingpong 位的惯用写法, 不改变指向哪个事件。
    """
    t = eid.strip()
    while t.startswith("(") and t.endswith(")"):
        t = t[1:-1].strip()
    t = re.sub(r"\s*&\s*(?:0[xX])?[0-9a-fA-F]+\s*$", "", t)   # 尾部的位掩码
    return re.sub(r"\s+", "", t)


def _h16_flags(line: str):
    """统一产出 (kind, event, id)。两种写法真仓都在用, 只认一种会漏掉半个算子。"""
    for m in _H16_MACRO.finditer(line):
        yield ("set" if m.group(1) == "SET_FLAG" else "wait"), f"{m.group(2)}_{m.group(3)}", _h16_norm_id(m.group(4))
    for m in _H16_TMPL.finditer(line):
        yield ("set" if m.group(1) == "SetFlag" else "wait"), m.group(2), _h16_norm_id(m.group(3))


def _h29_dynamic_array_peer(event, eid, opposite):
    """Evidence of an unresolved opposite array cell, never proof of balance.

    Only one-sided literal-name gaps qualify. Scalar/fixed-cell mismatches and
    a known same-cell 2:1 mismatch retain their existing error level.
    """
    cell = re.fullmatch(r"(\w+)\[([^]\n]+)\]", eid)
    if not cell:
        return []
    name, index = cell.groups()
    literal = lambda x: bool(re.fullmatch(r"[-+]?(?:0[xX][0-9a-fA-F]+|[0-9]+)[uUlL]*", x))
    peers = []
    for (other_event, other_id), count in opposite.items():
        other = re.fullmatch(r"(\w+)\[([^]\n]+)\]", other_id)
        if count and other and other_event == event and other.group(1) == name:
            other_index = other.group(2)
            if other_index != index and (not literal(index) or not literal(other_index)):
                peers.append(other_index)
    return sorted(set(peers))


def check_crossfunction_flag_balance(srcs) -> list:
    """H16: 把多个文件里同一事件的 Set/Wait 汇总, 报出**总数对不上**的那些。

    srcs: {文件名: 源码}。必须一次喂全部相关文件 —— 只喂一个文件时"另一半在别处"
    与"另一半根本不存在"长得一模一样, 而这两者的处置完全相反。
    """
    from collections import Counter
    sets, waits, where, looped = Counter(), Counter(), {}, {}
    #  ★第三次收窄: 事件 id 是**变量名**, 同一个物理事件在不同作用域叫不同名字。
    #    真仓实例: 一个函数末尾 SetFlag<FIX_M>(eventId) —— eventId 是它的**形参**,
    #    调用方拿 l0c_event 传进来并在那边 Wait。按 id 文本匹配必然把这一对拆开,
    #    于是报成"置位 1 等待 0"。形参是可判的, 判出来就降级并说清本判据看不到调用方。
    param_ids = set()
    _PARAM = re.compile(r"[(,]\s*(?:const\s+)?[\w:<>]+\s*[*&]?\s*([A-Za-z_]\w*)\s*[,)]")
    for src in srcs.values():
        for line in strip_comments(src).split("\n"):
            #  ★不要求本行含 `(` —— 长签名会换行, 真仓那个 eventId 就写在续行上
            #    (`..., const uint32_t eventId)`), 按"本行含 (" 筛会整条漏掉。
            #    正则要求"类型 名字"两个词, 所以函数**调用**的实参不会被误吃。
            if not line.rstrip().endswith(";") and (")" in line or "," in line):
                param_ids.update(_PARAM.findall(line))
    for path, src in srcs.items():
        lines = strip_comments(src).split("\n")
        loops = _loop_mask(lines)
        for i, line in enumerate(lines):
            for kind, ev, eid in _h16_flags(line):
                key = (ev, eid)
                (sets if kind == "set" else waits)[key] += 1
                where.setdefault(key, []).append(
                    f"{os.path.basename(path)}:{i + 1} {kind}")
                looped[key] = looped.get(key, False) or bool(loops[i])
    hz = []
    for key in sorted(set(sets) | set(waits)):
        ev, eid = key
        ns, nw = sets[key], waits[key]
        if ns == nw:
            continue          # 总数对得上就不报 —— 分配/释放式的跨函数配对是正常写法
        if eid in param_ids and (ns == 0 or nw == 0):
            lvl, head = "提示", (f"事件 {ev}({eid}) 全局只有{'置位' if ns else '等待'}一侧"
                                 f"({ns} set / {nw} wait), 而 `{eid}` 是**形参** —— "
                                 f"另一侧在调用方且换了变量名, 本判据按名字匹配看不到它")
        elif not looped.get(key) and (ns == 0 or nw == 0) and _h29_dynamic_array_peer(
                ev, eid, waits if ns else sets):
            peers = _h29_dynamic_array_peer(ev, eid, waits if ns else sets)
            lvl, head = "提示", (f"事件 {ev}({eid}) 字面全局置位 {ns} 次、等待 {nw} 次，"
                                 f"同事件同数组的另一侧使用不同cell {peers}；"
                                 "动态索引/辅助函数调用尚未求解，索引可能别名，"
                                 "不能将字面差数当作运行期配额失衡（unverifiable）")
        elif looped.get(key):
            lvl, head = "提示", (f"事件 {ev}({eid}) 全局置位 {ns} 次、等待 {nw} 次, "
                                 f"且至少一侧在循环里 —— 词法次数不等于运行期次数, 判不了对错")
        else:
            lvl, head = "错误", (f"事件 {ev}({eid}) 全局置位 {ns} 次、等待 {nw} 次, "
                                 f"且**两侧都不在循环里** —— 词法次数即运行期次数, 次数不等即挂死")
        spots = where.get(key, [])
        hz.append(dict(check="H29_crossfunction_flag_balance", buffer=f"{ev}({eid})", level=lvl,
                       msg=(head + "。出现在: " + "; ".join(spots[:8]) +
                            ("" if len(spots) <= 8 else f" (共 {len(spots)} 处)") +
                            "。★核对法: 沿调用链走一遍, 每个置位在被等待之前只发一次。"
                            "H11 (同条件块内) 与 H15 (单文件守卫) 对这种形态都是盲区。"
                            "★只喂了部分文件时, 缺的那一半会让本条误报 —— 要喂全整个 op_kernel 目录。")))
    return hz


def _git_blob(path: str, ref: str):
    """取 path 在 ref 上的内容; 不在 git 里或该 ref 上没这个文件就返回 None。"""
    import subprocess
    d = os.path.dirname(os.path.abspath(path)) or "."
    try:
        r = subprocess.run(["git", "-C", d, "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return None
        root = r.stdout.strip()
        rel = os.path.relpath(os.path.abspath(path), root)
        r2 = subprocess.run(["git", "-C", root, "show", f"{ref}:{rel}"],
                            capture_output=True, text=True, timeout=20)
        return r2.stdout if r2.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _finding_keys(hz, f):
    """一条命中的身份 = (文件, 判据, 对象)。**不含行号** —— 行号会被无关的增删行推移,
    按行号比对会把没动过的老命中全报成'新增'。"""
    return {(os.path.basename(f), h["check"], str(h.get("buffer", ""))) for h in hz}


def _scan_dir(a, ap) -> int:
    """目录模式: 递归扫 .h/.hpp/.cpp, 只打印有命中的文件, 末尾给一张汇总表。"""
    import glob as _g
    files = sorted(f for pat in ("**/*.h", "**/*.hpp", "**/*.cpp")
                   for f in _g.glob(os.path.join(a.kernel, pat), recursive=True))
    if not files:
        print(f"{a.kernel} 下没有 .h/.hpp/.cpp")
        return 2
    hsrc = strip_comments(open(a.host).read()) if a.host else None
    from core.wscap import parse_assume
    asm = parse_assume(a.assume) if a.assume else {}
    if a.op:
        from core.gen_spec import entry_constraints
        asm.update({k: v for k, v in entry_constraints(a.op, a.cann)["assume"].items()
                    if k not in asm})
    shape = _parse_shape(a.shape) if a.shape else None
    base_ref = getattr(a, "base", "") or ""
    base_seen, no_base_files = 0, 0
    if base_ref:
        print(f"=== detsync_lint (目录模式, 只报相对 `{base_ref}` **新增**的命中): "
              f"{a.kernel} —— {len(files)} 个文件 ===\n")
    else:
        print(f"=== detsync_lint (目录模式): {a.kernel} —— {len(files)} 个文件 ===\n")
    total, per_check, hit_files = 0, {}, 0
    jhits = []          # --json 机读输出 (core._jsonout 契约): 与打印同源, 不另算一遍
    unread = []
    for f in files:
        try:
            src = strip_comments(open(f, errors="ignore").read())
        except OSError as e:
            #  ★读不开要报: 抬头已经打了"{len(files)} 个文件", 收尾又打
            #    "hit_files/len(files) 个文件有命中" —— 静默丢弃会让分母虚高,
            #    一个读不开的文件与一个查过没问题的文件长得一模一样
            #    (2026-10-26, silentskip 扫出, 09 第 177 条)。
            unread.append((f, type(e).__name__))
            continue
        hz = (analyze(src, asm, hsrc) + check_ub_oob(src, shape, a.scale_bytes, a.out_bytes)
              + check_dma_granularity(src, shape or {}, a.scale_bytes, a.out_bytes))
        if base_ref:
            bsrc = _git_blob(f, base_ref)
            if bsrc is not None:
                bhz = check_dma_granularity(strip_comments(bsrc), shape or {}, a.scale_bytes, a.out_bytes) \
            + analyze(strip_comments(bsrc), asm, hsrc) + check_ub_oob(
                    strip_comments(bsrc), shape, a.scale_bytes, a.out_bytes)
                old_keys = _finding_keys(bhz, f)
                kept = [h for h in hz
                        if (os.path.basename(f), h["check"], str(h.get("buffer", ""))) not in old_keys]
                base_seen += len(hz) - len(kept)
                hz = kept
            else:
                no_base_files += 1
        if not hz:
            continue
        hit_files += 1
        print(f"--- {os.path.relpath(f, a.kernel)}  ({len(hz)} 条)")
        for h in hz:
            per_check[h["check"]] = per_check.get(h["check"], 0) + 1
            total += 1
            jhits.append({**h, "file": os.path.relpath(f, a.kernel)})
            print(f"    [{h.get('level', '错误')}] {h['check']}  {h.get('buffer', '')}")
            print(f"      {(h.get('msg') or h.get('err') or '(无消息)')}")
        print()
    #  ★H16 只能在目录模式下跑 —— 它比的是**跨文件**的总数, 单文件喂进去时
    #    "另一半在别处"与"另一半根本不存在"长得一模一样, 而两者的处置完全相反。
    srcs = {}
    for f in files:
        try:
            srcs[f] = open(f, errors="ignore").read()
        except OSError:
            continue
    h16 = check_crossfunction_flag_balance(srcs)
    if base_ref and h16:
        #  ★H16 是**跨文件汇总**, 逐文件差分对它无效 —— 第一版就因此把基线上本来就有的
        #  一条报成了"新增" (g7 那次: FIX_M(eventId) 在 g4 上同样存在)。
        #  修假阴性不能换来假阳性, 所以这里把基线的整棵树也喂一遍, 整体对照。
        bsrcs = {}
        for f in files:
            b = _git_blob(f, base_ref)
            if b is not None:
                bsrcs[f] = b
        if bsrcs:
            bkeys = {(h["check"], str(h.get("buffer", "")))
                     for h in check_crossfunction_flag_balance(bsrcs)}
            h16 = [h for h in h16 if (h["check"], str(h.get("buffer", ""))) not in bkeys]
    if h16:
        print(f"--- 跨文件汇总 (H16: 事件 Set/Wait 总数)  ({len(h16)} 条)")
        for h in h16:
            per_check[h["check"]] = per_check.get(h["check"], 0) + 1
            total += 1
            jhits.append(dict(h))
            print(f"    [{h['level']}] {h['check']}  {h['buffer']}")
            print(f"      {(h.get('msg') or h.get('err') or '(无消息)')}")
        print()
    print("=" * 72)
    if base_ref and no_base_files:
        #  ★反向失效守卫: 基线取不到时**不能**静默退回"报全部" —— 那会把既有命中冒充成新增,
        #  而使用者正是为了"只看新增"才加的 --base。第一版就是这样静默退回的。
        if no_base_files >= len(files):
            print(f"★取不到基线 `{base_ref}` 上的**任何**文件 (ref 写错, 或这些文件不在 git 里) ——")
            print(f"  下面是**全部命中**, 不是新增。别把它当差分读。")
        else:
            print(f"★有 {no_base_files}/{len(files)} 个文件在基线 `{base_ref}` 上取不到 "
                  f"(新增的文件, 或 ref 不含它们) —— 这些文件按**全部命中**列出, 不是新增。")
    if unread:
        print(f"★有 {len(unread)}/{len(files)} 个文件**读不开, 根本没查** —— "
              f"它们不是'查过没问题': " + ", ".join(f"{p} ({w})" for p, w in unread[:6]))
    print(f"汇总: {hit_files}/{len(files) - len(unread)} 个文件有命中, 共 {total} 条"
          + (f" (另有 {len(unread)} 个没查成)" if unread else "")
          + (f" (相对 `{base_ref}` 的**新增**; 基线已有 {base_seen} 条未列)" if base_ref else ""))
    for k, v in sorted(per_check.items(), key=lambda x: -x[1]):
        print(f"  {v:4d}  {k}")
    if not total:
        print("  无命中。注: 静态启发式, 非完备。")
    print("★模板组合型 kernel 必须用目录模式 —— 顶层 kernel 头往往一条都没有, "
          "hazard 在被组合的 epilogue/gemm 头里。")
    if getattr(a, "json", ""):
        from core._jsonout import dump
        dump(a.json, "detsync_lint", jhits, hard_fail=bool(total),
             extra={"base": base_ref, "files_scanned": len(files),
                    "no_base_files": no_base_files, "base_seen": base_seen})
        print(f"(机读输出已写 {a.json})")
    return 1 if total else 0



#  ★判据出处表 (2026-09-17 补建)。此前本模块 20+ 条判据**一条出处都没有** ——
#  无法回答"这条的实证是什么、上次验证哪天、有没有已知反例", 而 opdbg 的 REFS_MODULES
#  里也没有它。按 _refs 的立表理由: 一条不能被产物证伪的判据不是方法, 是断言。
#  本表先收**有真实证据**的条目; 没进表的判据就是"目前只是断言", 引用前自己验。
REFS = {
    "H32_mix_vector_core_index": Ref(
        what="C220两侧同控制循环从raw GetBlockIdx起步时，复核逻辑Cube核号与AIV子核映射",
        status="derived",
        evidence=[
            "VC-Attention E75/F296-F299: 原错GlobalVector未归一核号，首个MIX样本未返回；",
            "  修后在重新过三守卫的另一张卡上数值通过，非同卡单因素消融，不宣称唯一因果。",
            "本机CANN9.2 dav_c220/kernel_operator_sys_var_impl.h: AIV GetBlockIdxImpl=",
            "  get_block_idx()*GetTaskRationImpl()+get_subblockid(); GetSubBlockNumImpl为子核比。",
            "只认显式C220角色、同通道、同循环控制和直接局部赋值，提示级，不是完整C++证明。",
            "不套用C310显式双flag规则；未解析helper/运行时角色时不以无命中证明安全。",
        ], verified="2026-11-02"),
    "H27_core_partition_remainder": Ref(
        what="核间划分的**余数整块压在最后一个核**上 —— floor 分配 + 余数全给末核",
        status="measured",
        evidence=[
            "sfag 2026-09-18 实测: `firstCoreKSize = s2RealSize / totalVec` 向下取整,",
            "  `cur = (idx == totalVec-1) ? (s2RealSize % totalVec + first) : first`。",
            "  本 shape 2048/48 = 42 余 32 -> 47 个核各 42 行, **一个核 74 行**, max/mean = 1.734。",
            "  改成 `first + (idx < rem ? 1 : 0)` 后 1.008, 端到端 34.790 -> **34.125 ms (+1.91%)**,",
            "  逐位两档一致、精度过。",
            "★与 mmplan W7 (波量化) 的分工: W7 建模 **ceil 均分** 的形态",
            "  (max/mean = ceil(tiles/cores)*cores/tiles), 对本内核算出 1.008 —— 而实际 1.734。",
            "  W7 是参数化 planner 不看源码; 本条从**源码**认出是哪一种划分形态。互补不重复。",
            "★为什么值得单列: g15 是那一轮**唯一靠通读源码找到**的量化候选 ——",
            "  `whatfor 负载均衡` / `whatfor 余数` 当时零命中, 仓里没有判据在看这件事。",
            "★样本外检验用**仓库历史**做的, 比造用例强: 在 g10-idxub (未修) 上命中 L1694,",
            "  在 g15-balance (已修) 上静默。误报密度: 扫 ops-transformer/attention 的",
            "  **1485 个头文件, 全树只命中这 1 条**。",
            "★提示级不是错误级: 余数压末核不影响正确性, 只影响并行效率; 且是否值得改要看",
            "  该阶段有没有跨核会合 (有会合 -> 长尾无法跨轮摊平, 值得改; 无会合 -> 可能被下一轮盖住)。",
            "★配套的坑: 起始偏移**有几处就要改几处**。sfag 共 5 处, 最初只改了 3 处,",
            "  靠 `grep 'vecBlockIdx \* firstCoreKSize'` 才抓到 isSmallS2 分支里的两处。",
        ],
        verified="2026-09-18"),
    "H25_dma_granularity": Ref(
        what="DataCopy 的**传输粒度**太碎 (< 512B/次) —— 同样的字节总量被拆成更多事务, 有效带宽按比例掉",
        status="measured",
        evidence=[
            "sfag 2026-09-18: key 与 key_rope 是两个独立 GM 张量, 按选中 token 分两次 gather。",
            "  intriParamsRope.blockLen = selectedBlockSize * dimRope * sizeof(T1) = 1*64*2 = **128 B**",
            "  selectedBlockSize=1 (逐 token 选择) -> 每个选中 token 一次 128B 随机 DMA + 一次 1024B。",
            "  而 UB 那侧早就按连续布局准备好了 (dimDAlign = dimD + dimRope = 576),",
            "  **缺的连续性在 GM 那一端** —— 打包成 576 连续元素即可一次 1152 B。",
            "★我当时把 aiv_mte2 +5854us 记成'局部性假设', 还准备设计一个零构建缓存实验去验它;",
            "  真实原因是源码里写着的常量 128。**一次 grep 的事, 我要去做实验。**",
            "★第一版只认大括号初始化 `DataCopyExtParams p{...}`, 漏了真实代码里占绝大多数的",
            "  **逐字段赋值** `p.blockLen = ...`, 于是承载结论的三处一条都没报。样本外检验当场抓住。",
            "★2026-09-18 样本外再修一次 (同一份真源码): 原来把 `outParamRope.blockLen=128` 也报成碎,",
            "  而同一处 `blockCount = curGatherSize(64)` 且 **srcStride = dstStride = 0** ——",
            "  blockCount 个块首尾相接, 实际是**一次连续的 8 KB 搬运**, 不碎。**判据只看 blockLen 是假阳性。**",
            "  真正碎的只有读侧 (intriParamsRope 的 srcStride 按 kv token 跨距, 非零)。",
            "  => 现在跨距两端都为 0 时改报 `H25_dma_contiguous_ok` (**说出来为什么不报**,",
            "     否则'没查'与'查过没问题'又分不开)。**修假阴性不能换假阳性。**",
            "★修它时还踩了一个隐蔽的正则 bug: 写成 `=\\s*(?!0\\s*;)` 做前瞻, 而 `\\s*` 会回溯到零宽,",
            "  前瞻看到的是空格不是 '0', 断言反而成立 —— **判据输出没变, 容易以为本来就该报**。",
            "  改成**捕获跨距的值再判**。这类'修了但没生效'的失效最隐蔽。",
            "★作用范围 (排序时要先算这个): rope 读总量 268 MB = 落 DRAM 28.52 GB 的 **0.94%**;",
            "  它让 gather 的事务数翻倍而只贡献 11% 的字节。所以打包这条**字节收益不到 1%**,",
            "  价值全在事务数上 —— 而事务成本本仓明确要求实测 (feedback-instruction-cost-must-be-measured)。",
            "★真源码验证: 读侧两处 (L304/L1087) 报碎, 写侧 (L314) 报连续, **不误报** 512B 的 K 那条。",
            "★级别是提示不是错误: 粒度不影响正确性, 只影响带宽 (与 H9 的原子写颗粒不同,",
            "  那条违反会静默丢更新)。★判据给线索不给处方: 能不能打包取决于**写它的那一端**。",
        ],
        verified="2026-09-18"),
    "H26_crosscore_barrier_granularity": Ref(
        what="跨核汇合的**粒度**比它真正要保证的东西粗 —— 全局时序严格强于'同一地址上累加有序'",
        status="derived",
        evidence=[
            "sfag 2026-09-18: 确定性路径每个 s1 行做一次**全体 AIV 汇合**, 共 1024 次。",
            "而逐位一致真正要求的只是**同一个地址上的累加次序固定**; 全局时序顺带保证了它,",
            "代价是让输出地址根本不相交、永远不会碰面的核也互相等待。",
            "★我在这条线上做了两轮 (g8 +1.37% / g9 +1.68%), **两轮都只在挪等待的位置**,",
            "  一次也没问过'该不该是全体、该不该每轮一次'。然后在结论行写下「屏障线已榨到头」——",
            "  那句话只在**放置**这一层成立, 限定语从工作笔记进入结论时丢了。",
            "★与 H21 的分工: H21 管**核内**屏障与 UB 槽的粒度错配; 本条管**跨核**汇合。",
            "  仓里核内那侧判据齐备, 跨核这侧此前完全空白 —— 而昂贵的那个在跨核。",
            "★status=derived 而非 measured: 参与者集合静态可见 (全体), **真实冲突集合看不出来**",
            "  (取决于运行期索引), 所以判据只能提问不能下判断, 级别是提示。",
            "★误报控制: 只在'循环体内有跨核汇合 **且** 同函数有浮点原子加'时报。",
            "  真源码 1900 行只命中 1 条 (ScatterAddDeter)。第一版用'从 for( 向后 4000 字符'",
            "  近似循环体, 循环外的汇合也落进窗口而误报 —— 改成真正的大括号配对, 自检抓住的。",
        ],
        verified="2026-09-18"),
    "H10_setwait_adjacent": Ref(
        what="识别'同一轮内 SetFlag 与 WaitFlag 文本紧邻'这一形态 —— 它说明双缓冲没被用上。"
             "★只识别形态, **不预测收益**。",
        status="disputed",
        evidence="正面: 2026-09-15 sfag gather 实测 AIV 每核 GM->UB 只到能力的 13% "
                 "(17.6 / 132 GB/s), 形态与本条一致。"
                 "★反面 (2026-09-17 sfag_train det=1, 两次独立证伪): 照本条改法做了两次 —— "
                 "g1 把 GatherKVNonOptimized 的旗标从逐迭代改为逐批 (每批 32 次空转往返减到 1 次), "
                 "g2 把 ScatterAddDeter 的两次 MTE2->V 往返合成一次。两次**逐位一致均通过**, "
                 "H10 计数 25->23, 而性能 37.807 -> 37.561 (+0.65%) 与 37.758 (+0.13%), "
                 "趟间离散 1.0-1.4% —— **全部落在噪声内, 零收益**。"
                 "结论: 本条指出的是真实的代码形态, 但在该算子上**同步放置不是约束**。"
                 "动手前先用 core/perf --workrate 判是量差还是率差; 率差时同步只是候选之一, "
                 "还要与粒度/对齐/停顿分开。runs/cand_g1, runs/cand_g2。",
        verified="2026-09-17"),
    "H21_barrier_finer_than_resource": Ref(
        what="循环体内对同一事件既 WAIT 又 SET, 而缓冲槽是**有条件**翻转的 —— 屏障粒度比它保护的资源细。"
             "★只识别形态, **不预测收益**。",
        status="disputed",
        evidence="正面: 2026-09-15 sfag 实测 'gather 每 2 个 token 等一次而槽每 64 个才换', "
                 "AIV 每核 GM->UB 13% 能力。"
                 "★反面: 与 H10 同一组实验 (2026-09-17 sfag_train g1) —— 照本条改法把等待挪到槽开始, "
                 "H21 计数 3->2, 逐位通过, 性能零收益。同上: 形态真实, 收益不可由形态推出。",
        verified="2026-09-17"),
    "H29_crossfunction_flag_balance": Ref(
        what="事件 Set/Wait 的跨文件词法收支不等；循环、形参或同数组动态cell未解时只提示，不能冒充运行期配额证明。",
        status="measured",
        evidence="2026-09-17 sfag_train: 目录模式在 op_kernel 上报 MTE3_MTE2(EVENT_ID0/ID1) "
                 "全局 2 set / 5 wait。★该条对**文件子集**会误报, 必须喂整个 op_kernel 目录 —— "
                 "单文件模式下'另一半在别处'与'另一半不存在'长得一样。\n"
                 "★**已知误报形态 (2026-09-17 实撞)**: 同一事件在**互斥的两条主路径**里各有一个 SET \n"
                 "  (`ProcessDeterministic` / `ProcessNonDeterministic`, 由 `if constexpr` 选择), \n"
                 "  本条按词法计数看到 '2 set / 1 wait' 就报错误级 —— 而运行期只会执行其中一条。\n"
                 "  **判据看不见编译期互斥。** 遇到这形态先看那两处 SET 是不是在互斥分支里, 是则忽略。\n"
                 "  (同一天新加的 hangdiag C11 正是专查这种互斥分支的资源不对称, 两者可互相印证。)",
        verified="2026-11-02"),
    "H2_float_atomic_in_det": Ref(
        what="确定性路径里用 float 原子加 —— 加法不满足结合律, 跨核累加顺序不固定即不可复现。",
        status="disputed",
        evidence="★2026-09-17 sfag_train 实测**反例**: baseline det=1 的 ScatterAddDeter 确实在用 "
                 "SetAtomicAdd<float>(vec_op.h L1650), 本条命中; 而 det=1 连跑 3 次 "
                 "dq/dk/dq_rope/dk_rope **逐位一致** (runs/baseline_det1_bsnd.json, "
                 "bitwise_stable=true), 两种 kv_merge 口径都是。"
                 "原因: 该路径的确定性来自**分核划分** (每个输出行由唯一一个核写), 原子加并未真的竞争。"
                 "结论: 命中本条**不等于**不确定 —— 还要看分核是否让同一地址只被一个核触碰。"
                 "本条应作为'去核对分核划分'的提示, 不是判定。",
        verified="2026-09-17"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", nargs="?"); ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--shape", default="", help="H3 定量翻转点: n=..,baseN=..,e=..,batch=.. (逗号分隔)")
    ap.add_argument("--scale-bytes", type=int, default=2, help="SCALE_TYPE 字节 (bf16=2, fp32=4)")
    ap.add_argument("--out-bytes", type=int, default=2, help="DTYPE_OUT 字节")
    ap.add_argument("--flags", action="store_true", help="只打印跨核旗标账表 (507014 快速定位)")
    ap.add_argument("--share", action="store_true",
                    help="只打印**每核共享状态 x 占用方 x 重叠窗口**的体检表。"
                         "一个核里放两个以上 Matmul 实例时先敲它 —— 共享物太多, 人列不全")
    ap.add_argument("--op", default="",
                    help="算子名. 给了就自动从 aclnn 库抽入口校验的维度约束当契约 —— H9 这类对齐判据的"
                         "**可达性**取决于它, 不给就按最坏情形报, 真命中会被不可达项埋掉")
    ap.add_argument("--cann", default=os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-8.5.0"))
    ap.add_argument("--assume", default="", help="手写契约, 形如 'n%%32==0'. 与 --op 可叠加")
    ap.add_argument("--host", default="", help="host tiling .cpp. 契约要沿 baseN/singleN 传播才能判 H9 的可达性, "
                                              "而这些量的取值在 host, 不给就只能按最坏情形报")
    ap.add_argument("--base", default="",
                    help="git ref。给了就**只报相对该基线新增**的命中 (目录模式)。"
                         "这支算子基线上本就有 20 条错误级, 新增的那 1 条会被淹掉 —— "
                         "判据的信噪比决定它会不会被当成背景噪声忽略掉。")
    ap.add_argument("--json", default="", metavar="PATH",
                    help="机读输出落盘 (core._jsonout 契约 v1); prose 输出不变, 两不相扰。"
                         "上层 (candrun) 按它读, 不再刮取'汇总:'行的措辞")
    ap.add_argument("--selftest", action="store_true", help="判据负例自检")
    ap.add_argument("--refs", action="store_true",
                    help="打印本模块判据的出处/实证/最后验证日期 (含 disputed 与 refuted)。"
                         "★没进表的判据就是目前只有断言没有实证, 引用前自己验")
    a = ap.parse_args()
    if getattr(a, "refs", False):
        return 0 if print_refs("detsync_lint", REFS) is None else 0
    if getattr(a, "selftest", False):
        return selftest()
    if getattr(a, "share", False):
        if not a.kernel or os.path.isdir(a.kernel):
            ap.error("--share 需要单个 kernel 源文件")
        print_share(open(a.kernel, encoding="utf-8", errors="ignore").read())
        return 0
    #  ★kernel 给目录就递归扫全部头/源文件, 逐个跑一遍再汇总。
    #    起因 (block_sparse_attention @ 950, 2026-08-28): 这支 kernel 是跨 60+ 个头文件的
    #    模板组合 —— 顶层 kernel 头里一条 hazard 都没有, 而 H10 的三处命中全在被组合的
    #    epilogue 头 (block_epilogue_mask2idx_arch35.hpp) 里。只喂顶层文件时工具报"0 hazard",
    #    看着像判据漏了, 其实是**入口太窄**。实测该文件的三处 H10 正是端到端 27.5% 成本的所在。
    if a.kernel and os.path.isdir(a.kernel):
        return _scan_dir(a, ap)
    src = strip_comments(open(a.kernel).read())   # 去注释防假触发 (行号保留)
    if a.flags:
        return flag_ledger(src)
    from core.wscap import parse_assume
    asm = parse_assume(a.assume) if a.assume else {}
    if a.op:
        from core.gen_spec import entry_constraints
        ec = entry_constraints(a.op, a.cann)
        asm.update({k: v for k, v in ec["assume"].items() if k not in asm})
        if ec["assume"]:
            print("aclnn 入口契约 (自动抽): "
                  + ", ".join(f"{k}%{v}==0" for k, v in sorted(ec["assume"].items()))
                  + "  -- 对齐判据的可达性按它判\n")
    hsrc = strip_comments(open(a.host).read()) if a.host else None
    hz = analyze(src, asm, hsrc)
    shape = _parse_shape(a.shape) if a.shape else None
    hz3 = (check_ub_oob(src, shape, a.scale_bytes, a.out_bytes)
           + check_dma_granularity(src, shape or {}, a.scale_bytes, a.out_bytes))
    print(f"=== detsync_lint: {a.kernel} ===")
    nh4 = sum(1 for h in hz if h['check'] == 'H4_float_uint_cast')
    nh5 = sum(1 for h in hz if h['check'] == 'H5_tque_overflow')
    print(f"GM buffers: {len(gm_tensors(src))} | H1/H2 hazard: {len(hz)-nh4-nh5} | "
          f"H3(UB越界): {len(hz3)} | H4(float↔uint 编译错): {nh4} | H5(TQue 双入队死锁): {nh5} | "
          f"H7(旗标次数不等=死锁): {sum(1 for h in hz if h['check']=='H7_crosscore_flag_count')} | "
          f"H8(清零跨区段=写写竞态): {sum(1 for h in hz if h['check']=='H8_zero_span_crosses_regions')}")
    for h in hz:
        print(f"\n[{h['check']}] {h.get('buffer','')}")
        print(f"  {(h.get('msg') or h.get('err') or '(无消息)')}")
    for h in hz3:
        #  ★所有字段都用 .get —— 判据记录的字段集不是统一的, 打印处硬取会让**新判据一命中就崩**。
        #    2026-09-18 实撞: H25/H26 没有 func 字段, 打印处硬取 h['func'] -> KeyError,
        #    于是它们从加进来那天起一次都没成功打印过。自检只走返回值不走打印路径, 抓不到。
        _fn = f" (在 {h['func']})" if h.get("func") else ""
        print(f"\n[{h['check']}] {h.get('buffer','')}{_fn}")
        print(f"  {(h.get('msg') or h.get('err') or '(无消息)')}")
        if "quant" in h:
            print(f"  定量: {h['quant']}")
    if not hz and not hz3:
        #  ★0 命中先问方言 (2026-10-21 SMLA): 全部判据都建立在 AscendC 的写法上,
        #    对 DSL 源码报 "无 hazard" 是**没看见**, 不是没问题。
        if not _dialect.not_effective("detsync_lint", [a.kernel, a.host], "AscendC 的旗标/TQue 写法"):
            print("无确定性/跨核/UB越界 hazard (H1/H2/H3). 注: 静态启发式, 非完备; 真机确定性仍以 core/detrace divergence 为准.")
    # H1 跨核可见性 / H3 UB越界 / H4 编译必错 / H5 真机死锁 = 硬失败 (build/真机前必拦); H2 需人工确认单owner, 不硬失败.
    hard = any(h['check'] in ('H1_cross_core_visibility', 'H4_float_uint_cast', 'H5_tque_overflow') for h in hz) or bool(hz3)
    if a.json:
        from core._jsonout import dump
        dump(a.json, "detsync_lint", [dict(h) for h in hz + hz3], hard_fail=hard)
        print(f"(机读输出已写 {a.json})")
    sys.exit(1 if hard else 0)

import sys

if __name__ == "__main__":
    #  ★必须把 main() 的返回码交出去。此前写成裸 `main()`, 于是
    #    `python -m core.<mod> --selftest` **无论自检过不过, 退出码恒为 0** ——
    #    而 opdbg 的判据普查正是按子进程退出码判 PASS 的, 于是真实的
    #    "N 条未过" 被报成 PASS。2026-10-28 在 detsync_lint 上实撞并全仓同改。
    _rc = main()
    sys.exit(_rc if isinstance(_rc, int) else 0)


