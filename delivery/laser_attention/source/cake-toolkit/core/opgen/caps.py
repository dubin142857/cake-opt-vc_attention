#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opgen.caps —— **芯片机制能力**的可执行探测表。

与 core/socspec 的分工 (不重复)
------------------------------
socspec 回答"这颗芯片有多大、多快": 容量 (L1/L0A/L0B/L0C/UB)、核数、主频、MAC/cycle。
本模块回答"这颗芯片**让你怎么写**": 矩阵单元能不能把结果直接交给向量核、有没有寄存器级
SIMT、一个矩阵核配几个向量核、跨核旗标怎么发、哪些组合是被硬件禁止的。

前者决定**算得下算不下**, 后者决定**能采用哪一类调度**。算子设计里真正贵的决策全在后者:
同一个 attention, 有 C-V 直通就能做成一个混合核 (中间量不落 GM), 没有就只能拆成三次核
发射经 GM 交接 —— 这两条路的 GM 流量在 SFA case 3 上相差 4.9 倍 (10.7 GB vs 2.2 GB)。
A5(950) 有直通、regbase SIMT; A2/A3 没有。**所以能力必须探测, 不能写死**。

纪律 (与 core/_refs 同源)
-------------------------
每条能力都带 probe(), 且 probe 返回 (值, 证据字符串)。三类取值:
  True/False  探到了明确证据
  None        **探不到**。不等于 False —— 静默回落成 False 会让设计器给出一个
              "看起来在正常工作"的保守方案, 而本仓明令禁止静默回落 (mmplan 2026-09-11
              实撞: cv_direct_path 借用 910B 的 False, 在 950PR 上把直通路径整个藏掉了)。
证据一律指向**可复核的东西**: 平台 ini 的字段、CANN 头文件的行、编译器的宏定义、
仓内跑通过的参考内核。不写"我记得"。
"""
from __future__ import annotations

import os
import re

from .._refs import Ref, print_refs

ASCEND = os.environ.get("ASCEND_HOME_PATH") or "/usr/local/Ascend/cann"

REFS = {
    "CAP_probe_not_assert": Ref(
        what="能力表的每一条都由 probe() 从 ini/头文件/编译器宏取证, 探不到返回 None 而非 False",
        status="derived",
        evidence=[
            "反面教材 mmplan 2026-09-11: cv_direct_path 沿用 910B 的 False, 于是在 950PR 上",
            "把本机确实存在的 cube->UB 直通整条路藏掉 (msprof L0C_to_GM_datas=0 是直通的实证)。",
            "本模块因此不允许跨芯片借用, 也不允许把'探不到'折叠成'没有'。",
        ],
        verified="2026-09-22"),
    "CAP_dualdst_excludes_quant": Ref(
        what="Fixpipe 双目标模式 (dualDstCtl != 0) 与 quantPre/relu/COLUMN_MAJOR 互斥",
        status="measured",
        evidence=[
            "CANN 实现 dav_3510/kernel_operator_fixpipe_impl.h:230 的明文断言;",
            "FIAS 2026-09-14 实撞: mm2 结果靠 quantPre 转 fp16 且 dualDstCtl=1, 三条用例全崩",
            "507015 + 'The configuration of FIXP is illegal'。断言是 debug 级, release 不触发。",
            "判据已固化在 core/detsync_lint 的 H16; 本表只是把它提升成**设计期**约束:",
            "要 C-V 直通就意味着送进 UB 的必须是 L0C 的原始 fp32, 类型转换只能由向量侧做。",
        ],
        verified="2026-09-14"),
}


def _read(path: str) -> str:
    try:
        with open(path, errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _inc_root() -> str | None:
    return _first_existing(os.path.join(ASCEND, "asc", "include"),
                           os.path.join(ASCEND, "x86_64-linux", "asc", "include"),
                           os.path.join(ASCEND, "aarch64-linux", "asc", "include"))


def _impl_root() -> str | None:
    return _first_existing(os.path.join(ASCEND, "asc", "impl"),
                           os.path.join(ASCEND, "x86_64-linux", "asc", "impl"),
                           os.path.join(ASCEND, "aarch64-linux", "asc", "impl"))



def _arch_guard_stack(txt: str, needle: str):
    """返回 needle 所在位置上生效的 **__NPU_ARCH__ 条件栈**。

    为什么要这个: CANN 的头文件把不同架构的字段塞在同一个结构体/同一个文件里, 靠
    #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || ...) 分档。只 grep 字段名
    等于问"这个安装包里有没有这个字段", 而不是"这颗芯片有没有这个能力" ——
    2026-09-22 本模块的首版三条 probe 全踩这个坑, 在 910B/310P 上把 950 专属的
    dualDstCtl 报成"有", 由负例控制抓出。
    """
    pos = txt.find(needle)
    if pos < 0:
        return None
    return _arch_guard_stack_at(txt, pos)


def _arch_guard_stack_at(txt: str, pos: int):
    """同上, 但直接给**位置**。needle 版靠 `txt.find` 定位, 只有当 needle 在全文唯一时才对。

    ★2026-09-30 实撞: opgen.api 用「声明的前 40 个字符」当 needle 查 DataCopyPad 的守卫,
      而 `#if ... #else ...` 两侧的声明前 40 字符**逐字相同**, 于是 11 处声明里 10 处被
      `find` 定位到了别的声明身上 (359/370/388/394/404 全指到第 70 行, 411/416 指到第 127 行)。
      后果是 910B 上把一个 `#if (__NPU_ARCH__==3510)||(==5102)` 里的 950 专属重载
      判成"无守卫, 对所有架构可见"。凡是 needle 不唯一的调用方一律走本函数。
    """
    stack = []
    for m in re.finditer(r"^[ \t]*#\s*(if|ifdef|ifndef|elif|else|endif)\b([^\n]*)", txt, re.M):
        if m.start() > pos:
            break
        kind, rest = m.group(1), m.group(2)
        #  条件带反斜杠续行时把后续行接上 —— 否则 `(== A || == B || \` 这种会漏掉
        #  续行里的 arch 号, 把"本 arch 在名单里"读成"不在"。头文件里实测 30+ 处。
        if rest.rstrip().endswith("\\"):
            tail = txt[m.end():]
            for nxt in tail.splitlines():
                rest = rest.rstrip().rstrip("\\") + " " + nxt.strip()
                if not nxt.rstrip().endswith("\\"):
                    break
        if kind in ("if", "ifdef", "ifndef"):
            stack.append(rest)
        elif kind == "elif":
            if stack:
                stack[-1] = rest
        elif kind == "else":
            if stack:
                stack[-1] = "!(" + stack[-1] + ")"
        elif kind == "endif":
            if stack:
                stack.pop()
    conds = [c for c in stack if "__NPU_ARCH__" in c]
    return conds


def _arch_ok(txt: str, needle: str, arch: str):
    """(可用?, 说明). 字段不存在 -> False; 无架构守卫 -> True; 有守卫则看 arch 在不在里面。"""
    conds = _arch_guard_stack(txt, needle)
    if conds is None:
        return False, f"源码里没有 {needle}"
    if not conds:
        return True, f"{needle} 无 __NPU_ARCH__ 守卫 (对所有架构可见)"
    for c in conds:
        nums = re.findall(r"__NPU_ARCH__\s*==\s*(\d+)", c)
        neg = c.strip().startswith("!(")
        inside = str(arch) in nums
        if nums and (inside == neg):
            return False, f"{needle} 的守卫 {c.strip()} 不含 arch {arch}"
    return True, f"{needle} 的守卫 {' && '.join(x.strip() for x in conds)} 含 arch {arch}"


class Cap:
    """一条机制能力。

    design_yes / design_no 不是修辞: 它们是设计器 (opgen.plan) 真正分支的地方 ——
    有这条能力时允许枚举哪一类调度, 没有时必须改用什么。写不出这两句, 说明这条
    "能力"对设计没有影响, 不该进表。
    """
    __slots__ = ("id", "what", "design_yes", "design_no", "probe")

    def __init__(self, cid, what, design_yes, design_no, probe):
        self.id, self.what = cid, what
        self.design_yes, self.design_no = design_yes, design_no
        self.probe = probe


# ------------------------------------------------------------------ 各条 probe

def _p_arch(soc):
    from .. import socspec
    sp = socspec.spec(soc)
    return sp.get("arch"), f"平台 ini [version] NpuArch = {sp.get('arch')} ({sp['ini']})"


def _p_cv_macro(soc):
    """两侧代码的分界宏。ini 的 CCEC_AIC_version=dav-c310-cube -> __DAV_C310_CUBE__。

    不写死成 __DAV_C310_CUBE__: 换一颗芯片这个串就变, 而写死会让生成的核在别的芯片上
    把两侧代码**同时**编进去 (两侧都满足 #ifndef), 症状是编译过、上板行为荒谬。
    """
    from .. import socspec
    sp = socspec.spec(soc)
    ini = socspec.parse_ini(sp["ini"])
    raw = None
    for sect in ini.values():
        if isinstance(sect, dict) and sect.get("CCEC_AIC_version"):
            raw = sect["CCEC_AIC_version"]
            break
    if not raw:
        return None, "ini 里没有 CCEC_AIC_version, 判不出两侧分界宏"
    macro = "__" + re.sub(r"[^A-Za-z0-9]", "_", raw).upper() + "__"
    return macro, f"ini CCEC_AIC_version = {raw} -> 矩阵侧宏 {macro}"


def _p_cv_direct_ub(soc):
    """矩阵单元能否把 L0C 直接写进对侧向量核的 UB。

    证据取**字段本身的架构守卫**, 不取任何一处用法: FixpipeConfig 的 isToUB 字段被
    `__NPU_ARCH__ == <arch>` 守着才说明这颗芯片有这条通路。
    """
    arch, _ = _p_arch(soc)
    inc = _inc_root()
    if not inc or not arch:
        return None, "找不到 asc/include 或 arch, 无法取证"
    h = os.path.join(inc, "basic_api", "kernel_struct_fixpipe.h")
    txt = _read(h)
    if not txt:
        return None, f"读不到 {h}"
    #  ★只取到第一个成员函数为止: 结构体体内还有构造函数, 一路贪到 "};" 会把守卫
    #    与无关代码搅在一起 (首版如此, 结果这条能力被误报成"探不到")。
    ok, why = _arch_ok(txt, "isToUB", arch)
    if not ok:
        return False, f"L0C->UB 直通不可用: {why}"
    return True, (f"kernel_struct_fixpipe.h: {why}; "
                  f"用法 Fixpipe<T,U,{{CO2Layout::ROW_MAJOR, true}}>(ubTensor, l0cTensor, params)")


def _p_cv_dual_dst(soc):
    """一次 Fixpipe 把 M 对半分给配对的两个向量核 (dualDstCtl=1)。"""
    inc = _inc_root()
    if not inc:
        return None, "找不到 asc/include"
    arch, _ = _p_arch(soc)
    txt = _read(os.path.join(inc, "basic_api", "kernel_struct_fixpipe.h"))
    ok, why = _arch_ok(txt, "dualDstCtl", arch)
    if not ok:
        return False, f"双目标回写不可用: {why}"
    mode = ""
    for root, _, names in os.walk(inc):
        for n in names:
            if "McgShfMode" in _read(os.path.join(root, n)):
                mode = f"; 具名枚举 McgShfMode (DUAL_DST_SPLIT_M / DUAL_DST_SPLIT_N) 在 {n}"
                break
        if mode:
            break
    return True, (f"kernel_struct_fixpipe.h: {why}; =1 按 M 对半分, "
                  "M/2*N 写进每个 AIV 的 UB (M 必须是 2 的倍数), =2 按 N 切 1:2" + mode)


def _p_dualdst_quant_exclusive(soc):
    """双目标模式与 quantPre/relu/COLUMN_MAJOR 是否互斥 (设计期就得知道)。"""
    #  ★只在**本 arch 的**实现目录里找。首版用 os.walk(impl) 全树搜, 命中的是
    #    dav_m510/ —— 另一颗芯片的实现。本模块自己禁止跨芯片借用, 却差点在第一条
    #    证据上犯这个错; 全树搜索在多 arch 并存的 CANN 树里天然会拿到别人的文件。
    impl = _impl_root()
    arch, _ = _p_arch(soc)
    hit = None
    #  ★目录名有两种形态: 950 是 dav_3510 (按 NpuArch), 910B 是 dav_c220 (按 CCEC 版本串)。
    #    两个都从 ini 取, 不猜 —— 首版猜了个 dav_c201, 于是在 910B 上报"无法取证"。
    cand = []
    names = [f"dav_{arch}"] if arch else []
    macro, _ = _p_cv_macro(soc)
    if macro:
        base = macro.strip("_").lower().replace("_cube", "").replace("_vec", "")
        names.append(base)
    if impl:
        for d in names:
            p = os.path.join(impl, "basic_api", d)
            if os.path.isdir(p):
                cand.append(p)
    for p in cand:
        for root, _, names in os.walk(p):
            for n in names:
                if "Dual destination mode can be enabled only when" in _read(os.path.join(root, n)):
                    hit = os.path.join(root, n)
                    break
            if hit:
                break
        if hit:
            break
    if not cand:
        return None, f"CANN impl 树里没有本 arch ({arch}) 的实现目录, 无法就地取证"
    if hit:
        return True, (f"CANN 实现里的明文断言 ({hit}): 双目标模式只允许在 normal DMA / NZ2ND "
                      f"且其余 Fixpipe 功能全旁路时使用 —— 即 quantPre 必须 NoQuant。"
                      f"该断言是 debug 级, release 不触发, 症状是上板 507015。")
    return None, ("在 CANN impl 树里没搜到那条断言原文 (可能装的是裁剪包)。"
                  "**不等于没有这条约束** —— FIAS 2026-09-14 已实撞, 见 core/detsync_lint H16。")


def _p_regbase_vf(soc):
    """寄存器级 SIMT 向量函数 (__simd_vf__ + RegTensor)。"""
    from .. import socspec
    arch, _ = _p_arch(soc)
    #  ★reg_compute/ 头在安装包里对所有芯片都在, 只探它等于问"装了没", 不是"这颗芯片有没有"。
    #    按 arch 分档的证据在 adv_api/detail/**/regbase/<arch>, socspec.paradigms 已经在探。
    par = socspec.paradigms(str(arch)) if arch else {}
    if par.get("regbase") is False:
        return False, (f"adv_api/detail/**/regbase/{arch} 一处未命中 —— "
                       f"该架构没有寄存器级实现档 ({par.get('evidence', '')})")
    inc = _inc_root()
    d = os.path.join(inc, "basic_api", "reg_compute") if inc else None
    has_hdr = bool(d and os.path.isdir(d))
    cc = ""
    for root, _, names in os.walk(os.path.join(ASCEND, "tools")):
        if "__clang_cce_defines.h" in names:
            t = _read(os.path.join(root, "__clang_cce_defines.h"))
            if "cce_simd_vf" in t:
                cc = os.path.join(root, "__clang_cce_defines.h")
            break
    if not has_hdr and not cc:
        return False, "既无 basic_api/reg_compute/ 也无编译器的 cce_simd_vf 属性"
    if not (has_hdr and cc):
        return None, f"只探到一半 (头 {has_hdr} / 编译器属性 {bool(cc)}) —— 判不出可用"
    ops = sorted({m for m in re.findall(r"__simd_callee__ inline \w+ (\w+)\(",
                                        "".join(_read(os.path.join(d, n)) for n in os.listdir(d)))})
    #  ★命名空间随 CANN 版本变, 必须探而不是记: 本机 9.1.0-beta.1 是 AscendC::Reg,
    #    而仓内参考例 examples/attention/flash_attn/L0_minimal 的 VF 代码用 AscendC::MicroAPI
    #    —— 照抄参考例编不过。2026-09-22 实撞。
    ns = []
    for cand, tag in (("namespace Reg {", "AscendC::Reg"),
                      ("namespace MicroAPI {", "AscendC::MicroAPI")):
        if any(cand in _read(os.path.join(d, n)) for n in os.listdir(d)):
            ns.append(tag)
    return True, (f"regbase 按 arch 分档命中 {par.get('regbase_dirs')} 处; "
                  f"命名空间 {' + '.join(ns) if ns else '未识别'}; "
                  f"reg_compute/ {len(os.listdir(d))} 个头, 编译器定义 cce_simd_vf; "
                  f"寄存器级算子 {len(ops)} 个, 含 {', '.join(ops[:10])}")


def _p_mix_ratio(soc):
    """一个矩阵核配几个向量核 —— 决定跨核旗标要发几个。"""
    from .. import socspec
    sp = socspec.spec(soc)
    cc, vc = sp.get("cube_cores"), sp.get("vector_cores")
    if not cc or not vc:
        return None, "ini 里取不到 cube/vector 核数"
    if vc % cc:
        return None, f"向量核 {vc} 不是矩阵核 {cc} 的整数倍, 配对关系需人工确认"
    r = vc // cc
    inc = _inc_root()
    kt = ""
    if inc:
        for root, _, names in os.walk(inc):
            for n in names:
                if f"KERNEL_TYPE_MIX_AIC_1_{r}" in _read(os.path.join(root, n)):
                    kt = f"; 核类型常量 KERNEL_TYPE_MIX_AIC_1_{r} 在 {n}"
                    break
            if kt:
                break
    return r, (f"ini: 矩阵核 {cc} / 向量核 {vc} = 1:{r}{kt}。"
               f"含义: 矩阵侧每个方向要发 {r} 个 CrossCore flag (f, f+16, ...), "
               f"向量侧各等一个; 只发一个则其余 AIV 永久等待。")


def _p_crosscore_flag(soc):
    inc = _inc_root()
    if not inc:
        return None, "找不到 asc/include"
    t = _read(os.path.join(inc, "basic_api", "kernel_operator_block_sync_intf.h"))
    if "CrossCoreSetFlag" not in t:
        return False, "无 CrossCoreSetFlag/CrossCoreWaitFlag"
    sig = re.search(r"template <([^>]*)>\s*\n__aicore__ inline void CrossCoreSetFlag", t)
    return True, (f"CrossCoreSetFlag<modeId, pipe>(flagId) / CrossCoreWaitFlag<modeId, pipe>(flagId)"
                  f"{'; 模板参数 ' + sig.group(1).strip() if sig else ''}。"
                  f"仓内跑通过的用法是 modeId=4 (矩阵侧 PIPE_FIX 发, 向量侧 PIPE_V 等)。")


def _p_simt_api(soc):
    inc = _inc_root()
    d = os.path.join(inc, "simt_api") if inc else None
    if not d or not os.path.isdir(d):
        return False, "无 asc/include/simt_api/"
    arch, _ = _p_arch(soc)
    h = _read(os.path.join(d, "asc_simt.h"))
    guarded = bool(arch) and f"__NPU_ARCH__ == {arch}" in h.replace("==", " == ")
    return True, (f"simt_api/ 存在{'' if guarded else ' (但 asc_simt.h 未按本 arch 分档, 需确认)'}")


CAPS = [
    Cap("npu_arch", "架构号, 所有按 __NPU_ARCH__ 分档的头文件都据它取舍",
        "可以引用该档下的类型 (如 FixpipeParamsArch<arch>)",
        "—", _p_arch),
    Cap("cv_split_macro", "矩阵侧/向量侧代码的分界宏",
        "混合核用 #ifdef <宏> 分两侧, 编译器各编各的",
        "只能用运行期 if ASCEND_IS_AIC (两侧代码都进二进制)", _p_cv_macro),
    Cap("cv_direct_ub", "矩阵单元 Fixpipe 把 L0C 直接写进对侧向量核的 UB",
        "中间量 (分数矩阵 / 矩阵乘结果) 不必落 GM, C-V 可融合成一个核",
        "中间量必须经 GM 交接, 算子拆成多次核发射; 流量按中间量大小翻倍", _p_cv_direct_ub),
    Cap("cv_dual_dst_split_m", "一次 Fixpipe 按 M 对半分给配对的两个向量核",
        "矩阵侧一条指令喂饱两个向量核, 向量侧各拿 M/2 整行 (softmax 这类按行归约需要整行)",
        "要发两次单目标 Fixpipe, 用 subBlockId 选目标, 第二次源在 L0C 偏移 halfM*16",
        _p_cv_dual_dst),
    Cap("dualdst_excludes_quant", "双目标模式与 quantPre/relu/COLUMN_MAJOR 互斥",
        "★这是**禁止项**: 探到 True 意味着走直通时送进 UB 的只能是 L0C 原始 fp32, "
        "类型转换必须由向量侧做",
        "(探不到不等于没有, 见 core/detsync_lint H16)", _p_dualdst_quant_exclusive),
    Cap("regbase_simd_vf", "寄存器级 SIMT 向量函数 (__simd_vf__ + RegTensor)",
        "小分块上的多步向量运算可以整段留在寄存器里, 分块宽度不再决定指令效率 —— "
        "这是分块 softmax 能不被指令固定开销吃掉的前提",
        "每步都要经 UB 往返, 分块宽度必须做大 (>=2048 元素) 才能摊薄固定开销, "
        "而分块做大又与 UB 容量冲突", _p_regbase_vf),
    Cap("mix_aic_to_aiv_ratio", "一个矩阵核配几个向量核",
        "决定混合核的分工粒度与跨核旗标个数",
        "—", _p_mix_ratio),
    Cap("crosscore_flag", "跨核旗标 (CrossCoreSetFlag / CrossCoreWaitFlag)",
        "矩阵侧与向量侧可以在**核内**流水交接, 不必靠多次核发射的 stream 顺序",
        "只能靠多次核发射 + GM 交接", _p_crosscore_flag),
    Cap("simt_api", "SIMT 编程接口 (asc/include/simt_api)",
        "可以按线程模型写向量代码",
        "只能用向量指令模型", _p_simt_api),
]


#  ---- 负例控制 (门禁 G1) ----
def _nc_probe_not_assert():
    """CAP_probe_not_assert 的负例: 探不到时**不得**折叠成 False。

    在 910B2 上 dualdst_excludes_quant 探不到 (本 arch 的实现目录里没有那条断言原文),
    它必须返回 None; 若哪天被改成 False, 本条立刻红。
    """
    v, _ = _p_dualdst_quant_exclusive("ascend910b2")
    return v is None


def _nc_dualdst_excludes_quant():
    """CAP_dualdst_excludes_quant 的负例: 同时设 dualDstCtl 与 quantPre 的源码必须被判据抓住。

    判据本体在 core/detsync_lint 的 H16, 这里喂它一段**已知违规**的最小源码。
    """
    from .. import detsync_lint
    src = ("FixpipeParamsC310<CO2Layout::ROW_MAJOR> fp;\n"
           "fp.dualDstCtl = 1;\n"
           "fp.quantPre = QuantMode_t::F322BF16;\n")
    hits = detsync_lint.check_fixpipe_dualdst_quant(src)
    return any(h.get("check") == "H16_fixpipe_dualdst_vs_quant" for h in hits)


NEGATIVE_CONTROLS = {
    "CAP_probe_not_assert": _nc_probe_not_assert,
    "CAP_dualdst_excludes_quant": _nc_dualdst_excludes_quant,
}


def probe_all(soc=None):
    out = []
    for c in CAPS:
        try:
            val, ev = c.probe(soc)
        except Exception as e:                                  # noqa: BLE001
            val, ev = None, f"probe 抛异常: {type(e).__name__}: {e}"
        out.append((c, val, ev))
    return out


def show(soc=None, verbose=True):
    from .. import socspec
    if soc is None:
        soc = socspec.detect_soc()[0]
    print(f"=== 芯片机制能力 @ {soc}")
    rows = probe_all(soc)
    unknown = 0
    for c, val, ev in rows:
        mark = {True: "有", False: "无", None: "探不到"}.get(val, str(val))
        if val is None:
            unknown += 1
        print(f"\n[{mark:>5}] {c.id:<24} {c.what}")
        print(f"        证据: {ev}")
        if verbose:
            if val is True or (val is not None and val is not False):
                print(f"        ⇒ 可用: {c.design_yes}")
            else:
                print(f"        ⇒ 改用: {c.design_no}")
    print(f"\n{len(rows)} 条能力, {unknown} 条探不到。")
    print("★探不到 ≠ 没有。设计器遇到 None 会**拒绝**给出依赖该能力的方案, 而不是悄悄回落 —— "
          "用别的芯片的结论算出一个'通过', 比报错危险得多。")
    return 0


def as_dict(soc=None):
    return {c.id: {"value": v, "evidence": e} for c, v, e in probe_all(soc)}
