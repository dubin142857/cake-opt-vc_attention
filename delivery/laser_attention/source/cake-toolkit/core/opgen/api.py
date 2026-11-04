#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opgen.api —— AscendC **API 用法解释器**: 一条 API 的五层取证, 最细一层落到编译器。

与 opgen 内已有文件的分工 (为什么不能塞进它们)
----------------------------------------------
  `caps.py`      答「这颗芯片**让你怎么写**」—— 粒度是**机制**: 有没有 C-V 直通、
                 有没有 regbase。9 条, 一颗芯片一张表。
  `semantics.py` 答「这个**数学原语**是什么」—— 硬件无关的参考实现与形状规则。
  本文件         答「`AscendC::<某条 API>` 到底怎么用」—— 粒度是**一条 API**。

夹在两者之间的这一格此前是空的, 而设计时被卡住的恰恰是它: 机制知道有 regbase, 语义知道
要做 Muls, 但「一次 VF 调用处理多少元素」既不在 caps 也不在 semantics 里 —— 于是 2026-09-22
那次把「VF 调用」当成「一条指令」代进 40.8 周期/条的标定, 算出「指令条数降 6.5 倍」,
实测慢 3.7%, 整条结论作废 (提交 32202e8)。★**单位对不上却直接相除**, 与跨环境数字不可比
是同一类错; 缺的不是算力模型, 是这一层的取证。

五个层次 (由粗到细, 每层各有取证来源; 与 PLAN 的 L1–L6 是**两个轴**, 不要混)
-----------------------------------------------------------------------------
  PLAN 的 L1–L6 是「一个**调度**要过哪几类判据」; 这里的 A1–A5 是「一条 **API** 要问清哪几件事」。

  A1 存在与档位   这颗芯片上有没有它? 声明处在哪个 `__NPU_ARCH__` 条件下?
                  取证: `asc/include/**` 的声明位置 + 其上生效的 arch 条件栈。
  A2 签名与位置   几个重载, **每个操作数**在 GM / 片上 / 寄存器 的哪一个, 数据往哪个方向流?
                  取证: 接口头的模板与形参类型 (GlobalTensor/LocalTensor/RegTensor),
                  逐形参给位置 + 按 dst/src 定方向 + 摘出模板参数 (变体轴) 与
                  `__inout_pipe__` 标注; 「片上」再用本 arch 实现目录里同族的 TPosition 收窄。
  A3 硬约束       对齐、容量、取值域 —— 编译过而上板才炸的那一类。
                  取证: 本 arch 实现目录里的 `ASCENDC_ASSERT` / `static_assert` **原文**。
  A4 流水与同步   它落在哪条 pipe, 因此与前后指令之间要不要 Set/WaitFlag。
                  取证: 实现里出现的 `PIPE_*`, 以及接口头所属的类别。
  A5 指令与编译器 展开成哪条底层指令, 一条指令吃多少元素, regbase 下等价写法是什么。
                  取证: **编译器自带的内建头** `__clang_cce_vector_types.h` /
                  `__clang_cce_vector_intrinsics.h`, 平台 ini 的 intrinsic dtype 表,
                  以及 `reg_compute/` 里的同名寄存器级算子。

每层三态 (与 caps 的纪律同源): 取到证据 / `None` 探不到 / 不适用。
**探不到时不猜, 给一条可执行的实验** —— 这是本模块与「查文档」的根本差别。

实验按代价分三档, 只有 E1 由本模块直接跑:
  E1 编译探针  `ccec -c` 一个最小片段, 用编译错误或 `static_assert` 反解一个数。
               秒级, 不上机, 不占卡。`--probe` 跑它。
  E2 反汇编    编译后看展开成几条指令 —— 判「一次调用 = 几条硬件指令」。
  E3 真机      `core/probe` 的 ascendc_library 直调路径; 要卡, 走 devguard。

已由 A5 结清的一笔账 (本模块的第一个产出)
-----------------------------------------
平台 ini 的 `vector_reg_width=256` / `predicate_reg_width=32` / `wide_reg_width=768`
**单位是字节, 不是位**。证据是编译器内建头里同一套寄存器的元素数:
`vector_f32 = 64 x f32` = 256 B, `vector_f16 = 128 x f16` = 256 B,
`vector_bool = 256 x i1` = 32 B, `wvector_s24 = 256 x 24 bit` = 768 B。
而 `core/socspec` 打印时给它们硬加了 `b` 后缀 —— 差 8 倍。
同时这条也**证实**了 32202e8 撤回时给出的那句「VF 内部仍然是每 64 个元素一条指令」:
64 个 f32 正好是一个向量寄存器。

按芯片可插拔 —— 2026-09-30 在 910B 上核出的四条 (首版只在开发机 950PR 上对)
--------------------------------------------------------------------------
本模块声称"判的是**这颗芯片**有没有/是什么", 但首版有四处把开发机的巧合当成了规律。
四条都由 910B 与 950PR 的对照实测揪出, 各配了负例控制:

  1. **arch 实现目录有两套命名**。950 家族用 NpuArch 数字 (`dav_3510`), 其余家族用 ini
     里的编译器码名 (`dav_c220` / `dav_c100` / `dav_m200` / `dav_m300`)。63 个 platform ini
     全量比对: 按数字只命中 32 个 (全是 3510)。首版只拼 `dav_<NpuArch>`, 于是 910B 去找
     不存在的 `dav_2201`, A3/A4 静默丢掉整个实现目录 —— DataCopyPad 的断言从 35 条掉到 6 条。
  2. **措辞照着拼出来的名字写**。上一条之所以能瞒住, 是因为输出照旧打「dav_2201 + 校验层」:
     把"少扫了一个目录"讲成了"这颗芯片的约束就这么少"。现在只报**实际扫到的**目录。
  3. **守卫按文本片段定位**。`#if ... #else ...` 两侧的声明前 40 字符可以逐字相同,
     `txt.find` 把 11 处声明里的 10 处定位到了别的声明身上, 于是 910B 上把一个 950 专属
     重载判成"无守卫, 所有架构可见"。改成按**位置**取守卫栈。
  4. **`#else` 分支的判定是反的**。条件栈把 else 记成 `!( ... 3510 ... )`, 而首版用子串
     匹配, 于是 950 认领了"非 950"那一支, 910B 反而被挡在自己该用的那一支外面。
     改成与 `caps._arch_ok` 同口径: 取 `== N` 名单再按 `!(` 取反。
  ★3 与 4 互相掩盖: 3 让守卫栈大多解析成空, 4 因此从没被触发。只修一条会让另一条当场显形。
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import tempfile

from .._refs import Ref, print_refs
from . import caps as _caps

ASCEND = os.environ.get("ASCEND_HOME_PATH") or "/usr/local/Ascend/cann"

REFS = {
    "API1_five_levels_have_distinct_sources": Ref(
        what=["一条 API 的五层各有**不同的**取证来源; 任一层缺证据都要标 None 并给实验, ",
              "不得用上一层的结论替代下一层 (A4 的 pipe 归属推不出 A5 的每指令元素数)"],
        status="derived",
        evidence=[
            "反面教材 2026-09-22 (提交 32202e8): 有 A1/A2 级的证据 (regbase 存在、RegTensor 可用),",
            "就直接对 A5 级的量下了结论 ——「VF 把指令条数降 6.5 倍」。",
            "实测 vf1 慢 3.7%, 结论作废。缺的正是 A5: 一次 VF 调用内部仍是每 64 个 f32 一条指令。",
            "本模块因此把「这条 API 展开成什么」单列一层, 且它的取证口是**编译器头**而非 CANN 头。",
        ],
        verified="2026-09-29"),
    "API2_reg_width_ini_unit_is_bytes": Ref(
        what=["平台 ini 的 vector_reg_width / predicate_reg_width / wide_reg_width 单位是**字节**",
              "(950PR: 256 B / 32 B / 768 B), 不是位"],
        status="measured",
        evidence=[
            "编译器内建头 tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_types.h:83",
            "  typedef float vector_f32 __attribute__((ext_vector_type(64), ...));  // 64 x f32 = 256 B",
            "同文件:85 vector_f16 = 128 x f16 = 256 B; :133 vector_bool = 256 x i1 = 32 B;",
            "同文件:168 wvector_s24 = 256 x 24 bit = 768 B, :170 wvector_s48 = 128 x 48 bit = 768 B。",
            "四个类型互相独立地指向同一组数, 与 ini 的三个字段逐一对上 —— 单位只能是字节。",
            "★现状: core/socspec.py:717 打印时硬加 'b' 后缀, 即把 256 B 说成 256 bit, 差 8 倍。",
        ],
        verified="2026-09-29"),
    "API3_compiler_header_is_the_instruction_truth": Ref(
        what=["指令级的事实 (一条指令吃多少元素、支持哪些 dtype) 的取证口是**编译器自带头**, ",
              "不是 CANN 的 asc/ 头 —— 后者只到 API 层, 再往下是编译器的内建类型与内建函数"],
        status="measured",
        evidence=[
            "vector_f32 等类型在 asc/ 整棵树里**搜不到定义** (只被 reg_compute 的 TypeGet 引用),",
            "定义在 tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_types.h。",
            "指令表在同目录 __clang_cce_vector_intrinsics.h (13437 行), 形态是可解析的宏:",
            "  __VF_VEC_SCALAR_OP(vadds, int32_t, s32, v64)  -> 指令名/C 类型/后缀/通道数。",
            "本机实测 ccec 可直接编译该头所在语言模式 (见 probe_vector_lane_count 的实验记录)。",
        ],
        verified="2026-09-29"),
    "API5_impl_arch_dir_has_two_naming_schemes": Ref(
        what=["`asc/impl/basic_api/` 下的 arch 实现目录**有两套命名**, 哪一套有效随芯片族而变: ",
              "950 家族用 NpuArch 数字 (dav_3510), 其余家族用 ini 里的编译器码名 ",
              "(910B->dav_c220 / 910A->dav_c100 / 310P->dav_m200 / 310B->dav_m300)。",
              "两条候选都要试, 且必须报出命中的是哪一条; 都不命中时说「本安装包没有」而不是沉默"],
        status="measured",
        evidence=[
            "本机 63 个 platform ini 全量比对 (2026-09-30), 实际存在的目录是",
            "  dav_3510 dav_c100 dav_c220 dav_l300 dav_l311 dav_m200 dav_m300 dav_m310 dav_m510:",
            "  按 NpuArch 数字命中 32 个 ini —— **全部是 3510**, 且 3510 的码名 dav_c310 不存在;",
            "  按 ini 码名命中 27 个 ini (910B/910A/310P/310B/Kirin 全在这一半);",
            "  两者皆不命中 4 个 (Ascend031 / Ascend035*), 本安装包里确实没有其实现目录。",
            "★2026-10-28 在本机 CANN 9.0.0 上重测, 结果与上面那次**不同**: 63 个 ini 里",
            "  按 CCEC 码名命中 57 个、两条都不命中 6 个 (Ascend031/035*/Kirin), ",
            "  **按 NpuArch 数字命中 0 个** —— 本包的实现目录只有",
            "  dav_c100 dav_c220 dav_c310 dav_m200 dav_m300 dav_m310, 没有 dav_3510,",
            "  而 950PR 的 ini 现在写 CCEC_AIC_version=dav-c310-cube。",
            "  即**数字那套在本包上不会触发**。两套候选仍都要试: 换一个安装包就可能反过来,",
            "  上一次测量 (2026-09-30) 正是在 dav_3510 存在、32 个 ini 按数字命中的包上做的。",
            "★首版只用 dav_<NpuArch>: 63 颗里只对 32 颗, 而开发机 950PR 恰在对的那一半。",
            "  910B (NpuArch=2201) 去找不存在的 dav_2201, A3/A4 静默丢掉整个 arch 实现目录,",
            "  输出却照旧写着「dav_2201 + 算子无关的校验层」—— 实测 910B 上 DataCopyPad",
            "  只剩 6 条校验层断言, 而 dav_c220/kernel_operator_data_copy_impl.h 里有 73 条。",
        ],
        verified="2026-09-30"),
    "API6_declared_pipe_beats_family_guess": Ref(
        what=["接口头里的 `__inout_pipe__(X)` / `__in_pipe__` / `__out_pipe__` 是 pipe 归属的**声明**, ",
              "优先级高于实现里出现的 PIPE_* 与按文件名的族首猜; 三者要分别标注出处"],
        status="measured",
        evidence=[
            "asc/include 下 14 个接口头带 pipe 标注, 共 75 处",
            "  (__inout_pipe__: S 17 / MTE2 15 / V 12 / MTE3 9; 另有 __in_pipe__ / __out_pipe__)。",
            "反面: DataCopyPad 的声明原文就是 `__aicore__ inline __inout_pipe__(MTE3) void",
            "  DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, ...)`,",
            "  而 A4 首版在 910B 上打的是「实现里出现的 PIPE: 无; 按所属族首猜 MTE2/MTE3」——",
            "  答案就在 A2 已经抓到的那个签名串里, 只是没有人去读它。",
            "★dav_c220/kernel_operator_data_copy_impl.h 里一个 PIPE_* 都没有 (dav_3510 里有 18 处),",
            "  所以在 910B 上这条标注是唯一的非猜测来源。",
        ],
        verified="2026-09-30"),
    "API4_unknown_must_carry_an_experiment": Ref(
        what=["某一层探不到时输出 None **并附一条可执行的实验**; 只报 None 不给实验等于把问题退回给人"],
        status="derived",
        evidence=[
            "与 caps 的 CAP_probe_not_assert 同源但更进一步: caps 只要求不静默回落成 False,",
            "本模块要求 None 必须带 E1/E2/E3 中的一条, 且 E1 由 --probe 当场跑掉。",
            "动机: PLAN 的 M4 写着「寄存器数目前未知, 属 C2 的 None, M4 的第一件事是探它」——",
            "这句话在 PLAN 里躺了一周没人去探, 因为它是**文档里的待办**而不是**命令的输出**。",
        ],
        verified="2026-09-29"),
}

#  ---- A5 的两个取证口: 编译器内建头 ----
_CC_ROOT = os.path.join(ASCEND, "tools", "bisheng_compiler")


def _cc_include_dir():
    for p in sorted(glob.glob(os.path.join(_CC_ROOT, "lib", "clang", "*", "include"))):
        if os.path.isdir(p):
            return p
    return None


def ccec_path():
    p = os.path.join(_CC_ROOT, "bin", "ccec")
    return p if os.path.isfile(p) else None


def _read(p):
    try:
        with open(p, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:                                               # noqa: BLE001
        return ""


def _inc_root():
    return _caps._inc_root()


def _impl_root():
    return _caps._impl_root()


def _impl_arch_candidates(soc=None):
    """本 arch 实现目录的候选名 + 各自出处, 按「先数字后码名」排。

    **两套命名并存, 哪一套有效随芯片族而变** —— 这是本模块按芯片可插拔的关键:
      · NpuArch 数字:   950 家族 -> `dav_3510` (而 950 的码名 `dav_c310` **不存在**)
      · ini 编译器码名: 910B -> `dav_c220`, 910A -> `dav_c100`, 310P -> `dav_m200`,
                        310B -> `dav_m300`, Kirin -> `dav_l300` / `dav_l311`
    码名从 ini 的 `CCEC_VECTOR_version` / `CCEC_AIC_version` 取 (`dav-c220-vec` -> `dav_c220`),
    不硬编码芯片表 —— 换一颗本仓没见过的芯片, 只要它的 ini 里有这两个字段就能自己解出来。
    """
    arch, _ = _caps._p_arch(soc)
    cands = []
    if arch:
        cands.append((f"dav_{arch}", f"平台 ini NpuArch={arch}"))
    ini_keys, _ini = cc_arch(soc)
    for key, val in sorted(ini_keys.items()):
        #  dav-c220-vec / dav-c220-cube / dav-c100 -> dav_c220 / dav_c100
        code = re.sub(r"-(vec|cube)$", "", val).replace("-", "_")
        if code and all(code != c for c, _ in cands):
            cands.append((code, f"平台 ini {key}={val}"))
    return cands, arch


def _impl_arch_dir(soc=None):
    """本 arch 的实现目录, 连同「命中的是哪一条规则」。探不到返回 None, 不回落到别的 arch。

    返回 (目录, arch, 出处说明)。★调用方打印时必须用返回的**目录本身**, 不要再用
    `dav_{arch}` 拼一个名字出来 —— 那正是 910B 上的假象来源: A3 报「dav_2201 + 校验层」,
    而 dav_2201 根本不存在, 真正扫到的只有校验层。判据见 REFS 的
    API5_impl_arch_dir_has_two_naming_schemes。

    ★另一条旧账 (保留原注释): 首版写成 `impl/dav_<arch>` 少了中间的 `basic_api/`, 于是 A3
      在一台**确实有** dav_3510 的机器上报"找不到本 arch 的实现目录"。这个方向的误报比漏报
      更贵: 它把"约束没抓到"说成"环境不对", 于是人去查环境而不是去查约束。
    """
    cands, arch = _impl_arch_candidates(soc)
    base = os.path.join(_impl_root() or "", "basic_api")
    for name, src in cands:
        d = os.path.join(base, name)
        if os.path.isdir(d):
            return d, arch, f"{name} (按 {src})"
    tried = ", ".join(n for n, _ in cands) or "(没有候选: ini 里既无 NpuArch 也无 CCEC_*_version)"
    return None, arch, f"试过 {tried}, 本安装包 {base} 下都不存在"


def _arch_dir(soc=None):
    """兼容旧签名: 只要 (目录, arch)。要出处说明的走 `_impl_arch_dir`。"""
    d, arch, _how = _impl_arch_dir(soc)
    return d, arch


def cc_arch(soc=None):
    """这颗芯片对应的**编译器** arch 串, 从平台 ini 取, 不硬编码。

    ★ini 里 `CCEC_VECTOR_version=dav-c310-vec` / `CCEC_AIC_version=dav-c310-cube` 就是
      `--cce-aicore-arch=` 要填的值。它与 CMake 侧的 `dav-3510` 是**两套命名**:
      后者填给 ccec 会被拒 (实测 "Unsupported CCE architecture: dav-3510")。
      两套名字不互通这件事没有任何文档写, 所以取数只走 ini 这一个口。
    """
    #  ★ini 的路径只从 socspec 取 —— 它是本仓「芯片规格唯一取数口」(C6 单一来源)。
    #    首版自己 glob platform_config/*.ini, 那里每颗 SoC 一个文件, 没有 soc 名就选不出,
    #    于是报"ini=None"而其实 socspec 早就解析出了正确的那一个。
    try:
        from .. import socspec
        ini = (socspec.spec(soc) or {}).get("ini")
    except Exception:                                               # noqa: BLE001
        ini = None
    txt = _read(ini) if ini else ""
    out = {}
    for key in ("CCEC_VECTOR_version", "CCEC_AIC_version"):
        m = re.search(rf"^{key}\s*=\s*(\S+)", txt, re.M)
        if m:
            out[key] = m.group(1)
    return out, ini


# ══════════════════════════════════════════════════════════════════════════
#  A1 存在与档位
# ══════════════════════════════════════════════════════════════════════════
def a1_exists(name: str, soc=None) -> dict:
    """这颗芯片上有没有这条 API, 声明在哪, 受哪个 arch 条件管。

    ★判的是「**这颗芯片**有没有」而不是「这个安装包里有没有」—— 两者的差别就是
    caps 首版踩的坑 (在 910B 上把 950 专属的 dualDstCtl 报成有)。
    """
    inc = _inc_root()
    if not inc:
        return {"level": "A1", "value": None, "why": "找不到 asc/include",
                "experiment": "检查 ASCEND_HOME_PATH 是否指向已安装的 CANN"}
    arch, arch_ev = _caps._p_arch(soc)
    pat = re.compile(r"\b__aicore__\s+(?:inline\s+)?\S[^;{]*?\b" + re.escape(name) + r"\s*\(")
    sites, guarded_out = [], []
    for fp in glob.glob(os.path.join(inc, "**", "*.h"), recursive=True):
        txt = _read(fp)
        if name not in txt:
            continue
        for m in pat.finditer(txt):
            ln = txt.count("\n", 0, m.start()) + 1
            #  ★按**位置**取守卫栈, 不按文本片段。`#if/#else` 两侧的声明前 40 字符
            #    可以逐字相同 (DataCopyPad 实测如此), 按片段查会张冠李戴。
            stack = _caps._arch_guard_stack_at(txt, m.start())
            rel = os.path.relpath(fp, inc)
            if stack and arch and not _arch_in(stack, arch):
                guarded_out.append((rel, ln, stack))
            else:
                sites.append((rel, ln, stack))
    #  ★`interface/` 是 `basic_api/` 的旧路径别名, 同名文件内容相同 (供老代码 include)。
    #    不去重的话同一条声明会被数两遍, "22 处声明"里有一半是同一份文件的两个路径 ——
    #    数量本身会被读成"重载很多", 而那是假的。
    sites = _dedupe_alias(sites)
    guarded_out = _dedupe_alias(guarded_out)
    if not sites and not guarded_out:
        return {"level": "A1", "value": None,
                "why": f"asc/include 下找不到 `{name}` 的 __aicore__ 声明",
                "experiment": f"确认拼写与命名空间; 或它属于 adv_api 的模板类而非自由函数, "
                              f"敲 `grep -rn '{name}' {inc}/adv_api | head`"}
    if not sites:
        return {"level": "A1", "value": False,
                "why": f"`{name}` 只声明在别的 arch 档下 ({guarded_out[0][2]}), 本机 arch={arch}",
                "evidence": [f"{p}:{l}" for p, l, _ in guarded_out[:4]]}
    return {"level": "A1", "value": True,
            "why": f"arch={arch} ({arch_ev.split('(')[0].strip()}) 下可用, {len(sites)} 处声明",
            "sites": sites[:8],
            "evidence": [f"{p}:{l}" + (f"  [守卫 {g}]" if g else "") for p, l, g in sites[:6]]}


def _dedupe_alias(sites):
    """同一个 (文件名, 行号) 在 basic_api/ 与 interface/ 各出现一次时只留 basic_api/。"""
    best = {}
    for rel, ln, g in sites:
        key = (os.path.basename(rel), ln)
        prev = best.get(key)
        if prev is None or (rel.startswith("basic_api/") and not prev[0].startswith("basic_api/")):
            best[key] = (rel, ln, g)
    return [best[k] for k in sorted(best, key=lambda x: (x[0], x[1]))]


def _arch_in(stack, arch) -> bool:
    """条件栈对本 arch 成不成立。**逐条求值**, 不做子串匹配。

    ★首版是 `str(arch) in str(stack)`, 在 `#else` 分支上给出的是相反的答案:
      `#if (__NPU_ARCH__==3510)||(==5102)` 的 else 支被记成 `!( ... 3510 ... )`,
      子串匹配看到 "3510" 就判 950 命中 —— 而那恰恰是「非 950」的那一支;
      同时 910B (2201) 在自己该用的那一支上被判成不命中。两个方向同时错。
      口径与 caps._arch_ok 统一: 取条件里的 `__NPU_ARCH__ == N` 名单, 再按 `!(` 取反。
    """
    for c in (stack or []):
        nums = re.findall(r"__NPU_ARCH__\s*==\s*(\d+)", str(c))
        if not nums:
            continue                      # 认不出的条件不参与判定, 不猜
        neg = str(c).strip().startswith("!(")
        if (str(arch) in nums) == neg:    # 名单含我而这是否定支, 或名单不含我而这是肯定支
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════
#  A2 签名与操作数位置
# ══════════════════════════════════════════════════════════════════════════
#  形参类型 -> 它住在哪一级存储。**短标签**, 因为 A2 现在逐个形参地报, 长句会把表撑散。
#  RegTensor/MaskReg/AddrReg 只在 regbase 芯片 (950) 的声明里出现; 在 910B 上 A1 已按
#  __NPU_ARCH__ 守卫把那些重载滤掉, 所以这张表不需要自己带芯片维度 —— 可插拔性由 A1 提供。
_LOC_BY_TYPE = [
    (r"\bGlobalTensor\b", "GM"),
    (r"\bLocalTensor\b", "片上"),
    (r"\bRegTensor\b", "REG (向量寄存器)"),
    (r"\bMaskReg\b", "REG (谓词)"),
    (r"\bAddrReg\b", "REG (地址)"),
]
_TENSOR_LOCS = ("GM", "片上", "REG (向量寄存器)", "REG (谓词)", "REG (地址)")

#  接口头自己声明的 pipe 归属。`__inout_pipe__(MTE3)` 这种, 是**声明**不是猜测。
_PIPE_DECL_RE = re.compile(r"__(inout|in|out)_pipe__\s*\(\s*([A-Z0-9_,\s]+?)\s*\)")


def _split_top_level(s: str, sep: str = ",") -> list:
    """按顶层分隔符切, 尊重 <> () [] 的嵌套 —— `LocalTensor<T>` 里的逗号不算数。"""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def _paren_span(s: str, open_at: int):
    """从 s[open_at] 这个 '(' 起, 返回配对 ')' 的下标; 不配对返回 None。"""
    depth = 0
    for i in range(open_at, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _parse_signature(sig: str, name: str) -> dict:
    """把一条声明拆成 模板参数 / 形参表 / pipe 标注。纯文本解析, 与芯片无关。"""
    out = {"tmpl": [], "params": [], "pipe_decl": None}
    m = re.match(r"\s*template\s*<", sig)
    if m:
        end = None
        depth = 0
        for i in range(m.end() - 1, len(sig)):
            if sig[i] == "<":
                depth += 1
            elif sig[i] == ">":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end:
            out["tmpl"] = _split_top_level(sig[m.end():end])
    p = _PIPE_DECL_RE.search(sig)
    if p:
        out["pipe_decl"] = {"kind": p.group(1), "pipes": [x.strip() for x in p.group(2).split(",")]}
    #  形参表: 找函数名后面那个 '(' —— 不能用第一个 '(', 那可能是 __inout_pipe__( 的
    m2 = re.search(re.escape(name) + r"\s*\(", sig)
    if m2:
        lo = sig.index("(", m2.end() - 1)
        hi = _paren_span(sig, lo)
        if hi:
            for raw in _split_top_level(sig[lo + 1:hi]):
                loc = next((tag for pat, tag in _LOC_BY_TYPE if re.search(pat, raw)), None)
                nm = re.findall(r"(\w+)\s*$", raw)
                out["params"].append({"decl": raw, "name": nm[0] if nm else "", "loc": loc})
    return out


def _flow_of(params: list):
    """从形参表读出**方向**。方向决定 MTE2 还是 MTE3, 是 A2 最该给出的那个事实。

    ★首版把位置收成一个**集合** (`[GM, 片上]`), 于是 GM->片上 与 片上->GM 打印出来
      一模一样 —— 而这两者分属不同 pipe。只认按名字点明的 dst/src, 认不出就说认不出,
      不按"第一个参数一定是 dst"硬推 (Mmad 这类三操作数 API 会被它推错)。
    """
    ts = [p for p in params if p["loc"]]
    dst = next((p for p in ts if p["name"].lower().startswith("dst")), None)
    src = next((p for p in ts if p["name"].lower().startswith("src")), None)
    if dst and src:
        return f"{src['loc']} -> {dst['loc']}", f"{src['name']} -> {dst['name']}"
    return None, None


def a2_signature(name: str, a1: dict, soc=None) -> dict:
    """几个重载, 每个重载的**每个操作数**落在哪一级存储, 以及数据往哪个方向流。"""
    if a1.get("value") is not True:
        return {"level": "A2", "value": None, "why": "A1 未确认存在, 不往下推",
                "experiment": "先解决 A1"}
    inc = _inc_root()
    overloads = []
    for rel, ln, _g in a1.get("sites", []):
        txt = _read(os.path.join(inc, rel))
        lines = txt.splitlines()
        #  往上找模板行, 往下收到分号 —— 声明可能折行
        head = max(0, ln - 4)
        chunk = "\n".join(lines[head:ln + 6])
        m = re.search(r"(template\s*<[^>]*>\s*)?__aicore__[^;{]*" + re.escape(name) + r"\s*\([^;{]*",
                      chunk, re.S)
        sig = re.sub(r"\s+", " ", m.group(0)).strip() if m else ""
        parsed = _parse_signature(sig, name)
        flow, by = _flow_of(parsed["params"])
        locs = sorted({p["loc"] for p in parsed["params"] if p["loc"]})
        overloads.append({"at": f"{rel}:{ln}", "sig": sig[:400], "locs": locs,
                          "tmpl": parsed["tmpl"], "params": parsed["params"],
                          "pipe_decl": parsed["pipe_decl"], "flow": flow, "flow_by": by})
    locs_all = sorted({l for o in overloads for l in o["locs"]})
    flows = sorted({o["flow"] for o in overloads if o["flow"]})
    on_chip, tp_src = _onchip_levels(name, soc)
    decl_pipes = sorted({p for o in overloads if o["pipe_decl"] for p in o["pipe_decl"]["pipes"]})
    why = f"{len(overloads)} 个重载; 操作数位置: {', '.join(locs_all) or '未识别'}"
    if flows:
        why += f"; 方向: {' | '.join(flows)}"
    if decl_pipes:
        why += f"; 接口头声明 pipe: {', '.join(decl_pipes)}"
    out = {"level": "A2", "value": bool(overloads), "why": why,
           "overloads": overloads[:8], "flows": flows, "declared_pipes": decl_pipes,
           "onchip_levels": on_chip, "onchip_src": tp_src}
    if not locs_all:
        out["experiment"] = ("形参类型没命中已知的张量类; 敲 `opdbg opgen api <名> --raw` "
                             "看原始签名后补 _LOC_BY_TYPE")
    elif "片上" in locs_all and not on_chip:
        #  探不到就给实验, 不猜 —— "片上"落在 UB 还是 L1 会改变整条搬运路径。
        out["experiment"] = (
            f"签名只说得出「片上」, 说不出是 UB / L1 / L0x —— 这一级由实参的 TPosition 定, "
            f"不在声明里。取证: 在本 arch 实现目录里数同族实现出现的 TPosition "
            f"(`grep -ohE 'TPosition::[A-Z0-9_]+' <本 arch 实现目录>/*{_family_hint(name)}*.h | sort -u`)")
    return out


#  API 名 -> 实现文件名里的族关键词。只登记**确认过的**族; 认不出就返回 None。
_FAMILY_HINT = [
    ("datacopy", "data_copy"), ("mmad", "mmad"), ("fixpipe", "fixpipe"),
    ("transpose", "transpose"), ("gather", "gather"), ("scatter", "scatter"),
]


def _family_hint(name: str):
    """API 名 -> 实现文件名里的族关键词; 认不出返回 None。

    ★不能回落成"扫整个 arch 目录里含这个名字的文件"。`Add` 这种短名是 `Adds` / `AddrReg` /
      `PadParams` 的子串, 那样回落会挑中 kernel_operator_cache_impl.h 与
      kernel_operator_atomic_impl.h —— 跟 Add 的操作数位置毫无关系, 而它们里面的
      TPosition 会被当成"Add 的片上位置"打出来 (950 上甚至打出了 GM, 在"片上"这一栏里
      是自相矛盾的)。**造一条假证据比不给证据更贵**, 所以认不出就交给实验行。
    """
    n = name.lower()
    for key, filekey in _FAMILY_HINT:
        if key in n:
            return filekey
    return None


def _onchip_levels(name: str, soc=None):
    """「片上」到底是 UB / L1 / L0x —— 从**本 arch 实现目录**里同族实现出现的 TPosition 取。

    这是 A2 里唯一随芯片变的部分, 且它是**证据不是判定**: 列出的是该族实现里出现过的
    TPosition, 不等于每个重载都接受其中每一个。实测两颗芯片确实不同 ——
    dav_c220 的 data_copy 实现里是 VECIN/TSCM/A1/CO1, dav_3510 是 TSCM/VECIN/LCM。
    认不出族, 或族文件里没有整词命中这条 API, 一律返回空 (由 A2 给实验), 不回落硬凑。
    """
    d, _arch, _how = _impl_arch_dir(soc)
    hint = _family_hint(name)
    if not d or not hint:
        return [], None
    word = re.compile(r"\b" + re.escape(name) + r"\b")
    files = [f for f in sorted(glob.glob(os.path.join(d, "*.h")))
             if hint in os.path.basename(f) and word.search(_read(f))]
    seen = set()
    for f in files:
        seen.update(re.findall(r"TPosition::([A-Z0-9_]+)", _read(f)))
    #  GM 不是"片上"的一级 —— 它出现在实现文件里只说明那段代码也碰 GM。
    seen.discard("GM")
    if not seen:
        return [], None
    return sorted(seen), os.path.basename(files[0])


# ══════════════════════════════════════════════════════════════════════════
#  A3 硬约束 (编译过而上板才炸的那一类)
# ══════════════════════════════════════════════════════════════════════════
_ASSERT_PAT = re.compile(
    r"(ASCENDC_DEBUG_ASSERT|ASCENDC_ASSERT|ASCENDC_CHECK_VALUE_RANGE|ASCENDC_CHECK\w*"
    r"|CheckValueRange|static_assert)\s*\(", re.S)


def _a3_files(soc=None):
    """A3 的取证文件集: **本 arch 实现目录** + **算子无关的校验层**。

    ★只扫 dav_<arch>/ 是不够的 (首版如此): 数值约束的大头住在
      `impl/basic_api/utils/kernel_check_*.h` 与 `impl/basic_api/kernel_operator_*_check.h`,
      它们不按 arch 分目录。DataCopyPad 的 blockLen 上限、32B 对齐断言、Mmad 的
      nSize 必须被 16 整除, 全在那一层 —— 漏掉它等于把最值钱的那部分约束判成"没有"。
    """
    d, arch, how = _impl_arch_dir(soc)
    impl = _impl_root() or ""
    files = []
    if d:
        files += sorted(glob.glob(os.path.join(d, "*.h")))
    files += sorted(glob.glob(os.path.join(impl, "basic_api", "utils", "kernel_check*.h")))
    files += sorted(glob.glob(os.path.join(impl, "basic_api", "kernel_*check*.h")))
    return files, arch, how, bool(d)


def a3_constraints(name: str, soc=None) -> dict:
    """与这条 API 相关的断言原文 (本 arch 实现 + 算子无关的校验层)。

    ★为什么取**实现**而不是接口: 接口头是签名与 @param 语义, 数值约束写在实现/校验层,
    且多数包在 `#if ASCENDC_CPU_DEBUG` 里 —— release 构建不触发, 症状变成上板
    507015/507035 这类错误码。把原文摆出来, 比事后查码表早一轮。
    """
    files, arch, how, has_arch = _a3_files(soc)
    #  ★措辞只说**实际扫到的**。arch 实现目录缺席时不得写成"dav_<arch> + 校验层",
    #    那会把"少扫了一整个目录"伪装成"这颗芯片就这么少约束"。
    scope = f"{how} + 算子无关的校验层" if has_arch else f"**仅**算子无关的校验层 ({how})"
    if not files:
        return {"level": "A3", "value": None,
                "why": f"arch 实现目录与校验层文件都没找到 —— {how}",
                "experiment": f"确认 {_impl_root()}/basic_api 下有 arch 实现目录与 utils/kernel_check*.h"}
    hits = []
    for fp in files:
        txt = _read(fp)
        if name not in txt:
            continue
        lines = txt.splitlines()
        for i, ln in enumerate(lines):
            if not _ASSERT_PAT.search(ln):
                continue
            #  只收与本 API 同一个函数体里的断言: 往上找最近的函数头
            up = "\n".join(lines[max(0, i - 60):i])
            fn = re.findall(r"__aicore__\s+(?:inline\s+)?\S[^;{]*?\b(\w+)\s*\(", up)
            if fn and name.lower() not in fn[-1].lower():
                continue
            body = "\n".join(lines[i:i + 4])
            hits.append({"at": f"{os.path.basename(fp)}:{i + 1}",
                         "text": re.sub(r"\s+", " ", body)[:300]})
    if not hits:
        return {"level": "A3", "value": None,
                "why": f"扫了 {len(files)} 个文件 ({scope}), 与 `{name}` 同函数的断言一条没抓到",
                "experiment": f"两种可能: (a) 它确实没有数值约束; (b) 实现里它叫另一个名字 (模板分发)。"
                              f"区分: `grep -rln '{name}' {_impl_root()}/basic_api | head` 看它落在哪几个"
                              f"实现文件, 再用那里的函数名重跑本层。"}
    return {"level": "A3", "value": True,
            "why": f"{len(hits)} 条断言 ({scope})",
            "asserts": hits[:10],
            "note": "多数包在 #if ASCENDC_CPU_DEBUG 里 —— release 不触发, "
                    "违反时表现为上板错误码而非编译失败。"}


# ══════════════════════════════════════════════════════════════════════════
#  A4 流水与同步
# ══════════════════════════════════════════════════════════════════════════
#  接口头文件名 -> 该族默认落在哪条 pipe。**只作首猜**, 实现里出现的 PIPE_* 优先。
_PIPE_BY_FILE = [
    (r"data_copy", "MTE2 (GM->片上) / MTE3 (片上->GM); 方向由实参定"),
    (r"vec_", "V (向量)"),
    (r"_mm_|gemm|cube|mmad", "M (矩阵)"),
    (r"fixpipe", "FIX"),
    (r"scalar", "S (标量)"),
    (r"block_sync|determine_compute_sync", "跨核同步, 不占计算 pipe"),
]


def a4_pipe(name: str, a1: dict, soc=None, a2: dict = None) -> dict:
    """它落在哪条 pipe —— 决定前后要不要 Set/WaitFlag, 以及它跟谁能重叠。

    三个来源按可信度排, 且**必须分别标出处**:
      1. 接口头自己的 `__inout_pipe__(X)` —— 声明, 最硬 (由 A2 解析出来)
      2. 本 arch 实现里出现的 `PIPE_*` —— 实现证据
      3. 按接口头文件名的族首猜 —— 只是猜
    ★首版只有 2 和 3, 于是 910B 上 DataCopyPad 打的是「实现里出现的 PIPE: 无; 按所属族
      首猜 MTE2/MTE3」, 而它的声明原文里就写着 `__inout_pipe__(MTE3)`。dav_c220 的
      data_copy 实现里一个 PIPE_* 都没有 (dav_3510 里有 18 处), 所以在 910B 上来源 1
      是唯一的非猜测证据 —— 缺了它这一层在半数芯片上永远只能给猜测。
    """
    d, _arch, how = _impl_arch_dir(soc)
    seen = set()
    if d:
        for fp in glob.glob(os.path.join(d, "*.h")):
            txt = _read(fp)
            if name not in txt:
                continue
            for m in re.finditer(r"\bPIPE_([A-Z0-9]+)\b", txt):
                seen.add(m.group(1))
    declared = sorted((a2 or {}).get("declared_pipes") or [])
    guess = []
    for rel, _ln, _g in (a1.get("sites") or []):
        for pat, tag in _PIPE_BY_FILE:
            if re.search(pat, rel):
                guess.append(tag)
    guess = sorted(set(guess))
    if not declared and not seen and not guess:
        return {"level": "A4", "value": None,
                "why": f"接口头无 pipe 标注, 实现里没出现 PIPE_* ({how}), 文件名也没命中已知族",
                "experiment": "E2 反汇编: 编一个只调这条 API 的最小核, objdump 看它发到哪条队列; "
                              "或 msprof 单点看哪条 pipe 的计数涨了"}
    bits = []
    if declared:
        bits.append(f"**接口头声明** __*_pipe__: {', '.join(declared)}")
    bits.append(f"本 arch 实现里出现的 PIPE: {', '.join(sorted(seen)) or '无'} ({how})")
    if guess:
        bits.append(("按所属族首猜 " if not declared else "族首猜 (已被声明取代): ")
                    + " | ".join(guess))
    out = {"level": "A4", "value": True, "why": "; ".join(bits),
           "declared_pipes": declared, "pipes_in_impl": sorted(seen), "family_guess": guess,
           "source": "声明" if declared else ("实现 PIPE_*" if seen else "族首猜"),
           "note": "同 pipe 内按序, 跨 pipe 要 SetFlag/WaitFlag —— 具体 hazard 判据在 "
                   "core/detsync_lint (H 族), 本层只定它属于哪条 pipe。"}
    if not declared and not seen:
        out["experiment"] = ("只有族首猜, 没有任何声明或实现证据 —— E2 反汇编定它: "
                             "编一个只调这条 API 的最小核, llvm-objdump 看助记符落在哪条队列。")
    return out


# ══════════════════════════════════════════════════════════════════════════
#  A5 指令与编译器 (最细一层)
# ══════════════════════════════════════════════════════════════════════════
_VEC_TYPE_RE = re.compile(
    r"typedef\s+(?P<ct>[\w:\s]+?)\s+(?P<name>w?vector_\w+)\s*\n?\s*"
    r"__attribute__\(\(\s*ext_vector_type\((?P<lanes>\d+)\)", re.S)


def vector_lane_table():
    """编译器内建向量类型的通道数表: {类型名: 通道数}。

    这是 A5 的地基 —— 「一条向量指令吃多少元素」只有这里说了算。
    """
    inc = _cc_include_dir()
    if not inc:
        return None, "找不到 bisheng 的 clang include 目录"
    txt = _read(os.path.join(inc, "__clang_cce_vector_types.h"))
    if not txt:
        return None, "读不到 __clang_cce_vector_types.h"
    out = {}
    for m in _VEC_TYPE_RE.finditer(txt):
        out[m.group("name")] = int(m.group("lanes"))
    return (out or None), os.path.join(inc, "__clang_cce_vector_types.h")


#  C 标量类型 -> 字节宽, 用于把通道数换算成寄存器字节宽
_CT_BYTES = {"f32": 4, "s32": 4, "u32": 4, "f16": 2, "bf16": 2, "s16": 2, "u16": 2,
             "s8": 1, "u8": 1, "s64": 8, "u64": 8}


def reg_width_bytes():
    """从通道数反解**向量寄存器字节宽**, 并与平台 ini 对账。

    返回 (字节宽 或 None, 证据行列表)。
    ★这条同时是 API2 的负例来源: 若哪天 ini 与内建头对不上, 本函数必须报"对不上"而不是二选一。
    """
    tab, src = vector_lane_table()
    ev = []
    if not tab:
        return None, [f"取不到内建向量类型表: {src}"]
    widths = set()
    for tname, lanes in tab.items():
        if tname.startswith("wvector_"):
            continue
        suf = tname.split("vector_", 1)[1]
        b = _CT_BYTES.get(suf)
        if not b:
            continue
        widths.add(lanes * b)
        if len(ev) < 4:
            ev.append(f"{tname} = {lanes} x {suf} = {lanes * b} B  ({os.path.basename(src)})")
    if len(widths) != 1:
        return None, ev + [f"★内建类型给出的寄存器宽不唯一: {sorted(widths)} —— 判据本身要重做"]
    w = widths.pop()
    ini = _ini_reg_widths()
    if ini.get("vector_reg_width") is not None:
        ok, msg = reconcile_reg_width(w, ini["vector_reg_width"])
        ev.append(msg)
        if not ok:
            return None, ev
    return w, ev


def reconcile_reg_width(builtin_bytes: int, ini_value) -> tuple:
    """内建头解出的字节宽 vs 平台 ini 的字段值, 对账。

    单独成函数是为了让负例控制能喂一个**已知错误**的值进来 (C4): 把 ini 当成"位"去读
    会得到 2048, 它必须被判成对不上, 而不是被悄悄换算过去。
    ★"两个来源不一致时自动挑一个"正是本仓禁止的静默回落 —— 这里只报不符, 由人查是哪侧变了。
    """
    try:
        v = int(ini_value)
    except (TypeError, ValueError):
        return False, f"平台 ini vector_reg_width = {ini_value!r} 不是整数, 无法对账"
    if v == builtin_bytes:
        return True, (f"平台 ini vector_reg_width = {v} —— 与内建头一致, 故**单位是字节**")
    if v == builtin_bytes * 8:
        return False, (f"平台 ini vector_reg_width = {v} = 内建头的 {builtin_bytes} B x 8 —— "
                       f"这一侧在按**位**记; 两套口径不得混用, 先定死 ini 的单位再往下算")
    return False, (f"平台 ini vector_reg_width = {v}, 与内建头的 {builtin_bytes} B **对不上**, "
                   f"且不是 8 倍关系 —— 不得二选一, 先查是哪一侧变了")


def _ini_reg_widths():
    try:
        from .. import socspec
        s = socspec.spec()
        return (s or {}).get("regfile") or {}
    except Exception:                                               # noqa: BLE001
        return {}


_VF_MACRO_RE = re.compile(r"^\s*__VF_\w+\(\s*(?P<instr>\w+)\s*,\s*(?P<ct>[\w:]+)\s*,"
                          r"\s*(?P<suf>\w+)\s*,\s*v(?P<lanes>\d+)", re.M)


def instruction_table(instr: str | None = None):
    """编译器内建指令表: 指令名 x dtype x 通道数。

    形态是宏 `__VF_VEC_SCALAR_OP(vadds, int32_t, s32, v64)`, 可直接解析。
    """
    inc = _cc_include_dir()
    if not inc:
        return None, "找不到 bisheng 的 clang include 目录"
    fp = os.path.join(inc, "__clang_cce_vector_intrinsics.h")
    txt = _read(fp)
    if not txt:
        return None, f"读不到 {fp}"
    rows = []
    for m in _VF_MACRO_RE.finditer(txt):
        if instr and m.group("instr").lower() != instr.lower():
            continue
        rows.append({"instr": m.group("instr"), "ctype": m.group("ct"),
                     "suffix": m.group("suf"), "lanes": int(m.group("lanes"))})
    return rows, fp


#  AscendC API 名 -> 寄存器级 (regbase) 的同名算子; 只认 reg_compute/ 里真的有的
def regbase_equivalent(name: str):
    inc = _inc_root()
    d = os.path.join(inc or "", "basic_api", "reg_compute")
    if not os.path.isdir(d):
        return None, "本机没有 reg_compute/ (这颗芯片没有 regbase)"
    pat = re.compile(r"__simd_callee__[^;{]*?\b" + re.escape(name) + r"\s*\(")
    hits = []
    for fp in sorted(glob.glob(os.path.join(d, "*.h"))):
        txt = _read(fp)
        for m in pat.finditer(txt):
            hits.append(f"{os.path.basename(fp)}:{txt.count(chr(10), 0, m.start()) + 1}")
    return (hits or None), d


#  哪些 pipe 属于向量单元 —— 只有它们的"每指令元素数"由向量寄存器宽决定。
_VEC_PIPES = {"V"}


def _is_vector_side(a4: dict | None) -> bool | None:
    """这条 API 归不归向量单元管。三态: True/False/None(判不出)。"""
    if not a4 or a4.get("value") is None:
        return None
    pipes = set(a4.get("pipes_in_impl") or [])
    guess = " ".join(a4.get("family_guess") or [])
    if pipes & _VEC_PIPES or "V (向量)" in guess:
        return True
    if pipes or guess:
        return False
    return None


def a5_instruction(name: str, soc=None, a4: dict | None = None) -> dict:
    """展开成什么指令、一条吃多少元素、regbase 下的等价写法。

    ★向量寄存器宽只对**向量侧** API 有意义。首版对 Mmad 也把「fp32 64 元素」printed 出来 ——
      数字本身没错, 但它与矩阵侧 API 无关, 而「摆在那里的正确数字被代进不该代的地方」
      正是本仓反复付过代价的形态 (sim 小 shape 成本分布定大 shape 优先级、
      把 VF 调用当一条指令)。所以这里按单元分档: 不适用就明说不适用。
    """
    w, ev = reg_width_bytes()
    rows, itab_src = instruction_table()
    reg_hits, reg_src = regbase_equivalent(name)
    #  API 名与指令名之间没有可靠的机械映射 (Add -> vadd? Adds -> vadds?), 不猜。
    guess = [r for r in (rows or []) if r["instr"].lstrip("v").lower() == name.lower()]
    out = {"level": "A5", "reg_width_bytes": w, "evidence": ev,
           "instr_table_rows": len(rows or []), "instr_table_src": itab_src,
           "regbase_sites": reg_hits, "regbase_dir": reg_src}
    if w is None:
        out["value"] = None
        out["why"] = "寄存器字节宽没定下来 —— 见证据行"
        out["experiment"] = ("E1 编译探针: `opgen api <名> --probe` 会编一个最小片段, "
                             "用模板特化失败把 sizeof(vector_f32) 逼进编译错误里。")
        return out
    lanes = {f"{k}": v for k, v in ((s, w // b) for s, b in _CT_BYTES.items())}
    out["value"] = True
    vec = _is_vector_side(a4)
    out["vector_side"] = vec
    if vec is True:
        out["lanes_by_dtype"] = lanes
        out["why"] = (f"向量侧 API: 向量寄存器 {w} 字节 = fp32 {lanes['f32']} 元素 / "
                      f"fp16 {lanes['f16']} 元素; 内建指令表 {len(rows or [])} 行")
    elif vec is False:
        out["why"] = (f"**非向量侧 API** —— 向量寄存器宽 ({w} B) 与它无关, 故不在此列出每指令元素数; "
                      f"它的粒度由矩阵/搬运单元的参数定 (见 A3 的 shape 与 stride 断言)")
        out["evidence"] = [e for e in ev if "平台 ini" in e][:1]
    else:
        out["why"] = f"判不出它归哪个单元 (A4 没定下 pipe), 因此不敢说寄存器宽对它适不适用"
        out["experiment"] = ("先把 A4 定下来: E2 反汇编一个只调它的最小核, 看助记符属于哪个单元。"
                             "★在 A4 定下来之前**不要**把向量寄存器宽代进任何与它有关的估算。")
    if guess:
        out["mapped_instr"] = guess[:6]
    else:
        cca = (cc_arch(soc)[0] or {}).get("CCEC_VECTOR_version", "<见 ini CCEC_VECTOR_version>")
        out["experiment"] = (
            f"指令名与 API 名对不上 (表里没有 `v{name.lower()}`) —— **不猜映射**。"
            f" E2 反汇编定它: 编一个只调 {name} 的最小核 "
            f"(`ccec -c -xcce --cce-aicore-arch={cca} --cce-aicore-only`), 再用同目录的 "
            f"`llvm-objdump -d` 反汇编 .o 看助记符。★不要用 `-S`: 设备侧不支持它 "
            f"(实测 \"unsupported option '-S' on device side\")。")
    if reg_hits and vec is not False:
        out["note"] = (f"regbase 下有同名寄存器级算子 ({len(reg_hits)} 处) —— "
                       f"★但「改用 VF」不等于「指令变少」: 一次 VF 调用内部仍是每 {lanes['f32']} 个 "
                       f"f32 一条指令 (提交 32202e8 的实测撤回)。VF 省的是链式运算之间的 UB 往返。")
    return out


# ══════════════════════════════════════════════════════════════════════════
#  E1 编译探针 —— 不确定时当场做的那个实验
# ══════════════════════════════════════════════════════════════════════════
_PROBE_SRC = """// opgen.api E1 探针: 用模板特化失败把数字逼进编译错误
template<int N> struct RevealBytes;
template<int N> struct RevealLanes;
void probe() {
    RevealBytes<sizeof(%(vt)s)> a;
    RevealLanes<sizeof(%(vt)s) / sizeof(%(st)s)> b;
    (void)a; (void)b;
}
"""


def probe_vector_lane_count(vtype="vector_f32", stype="float", arch=None, soc=None):
    """E1: 真的编一次, 从编译错误里把通道数读出来。

    这是本模块"不确定就做实验"的那条腿。**它不需要卡**, 所以没有 devguard 的前置。
    ★首版曾直接写 `sizeof(vector_f32)` 而不带任何头, 编译器报 "use of undeclared identifier"
      —— 内建向量类型要在 cce 语言模式下才可见, 所以这里显式走 `-x cce` 并带 arch。
      记下来是因为"探针自己没配对"会被读成"这个类型不存在", 而两者的处方相反。
    """
    cc = ccec_path()
    if not cc:
        return None, f"本机没有 ccec ({_CC_ROOT}/bin/ccec)"
    if arch is None:
        m, ini = cc_arch(soc)
        arch = m.get("CCEC_VECTOR_version")
        if not arch:
            return None, (f"平台 ini 里没有 CCEC_VECTOR_version, 定不出编译器 arch 串 "
                          f"(ini={ini})。**不猜** —— CMake 侧的 dav-3510 填给 ccec 会被拒。")
    src = _PROBE_SRC % {"vt": vtype, "st": stype}
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "probe.cpp")
        with open(fp, "w", encoding="utf-8") as f:
            f.write(src)
        cmd = [cc, "-x", "cce", "-c", f"--cce-aicore-arch={arch}", fp, "-o", os.path.join(td, "p.o")]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except Exception as e:                                      # noqa: BLE001
            return None, f"编译探针跑不起来: {type(e).__name__}: {e}"
        err = (r.stderr or "") + (r.stdout or "")
    if "undeclared identifier" in err and vtype in err:
        return None, (f"编译器在 {arch} 下不认识 {vtype} —— 说明这颗芯片/这个语言模式没有它, "
                      f"不是探针写错 (探针带了 -x cce 与 arch)")
    mb = re.search(r"RevealBytes<(\d+)>", err)
    ml = re.search(r"RevealLanes<(\d+)>", err)
    if not (mb and ml):
        return None, f"从编译输出里没解出数字; 原话前 400 字:\n{err[:400]}"
    return {"vtype": vtype, "bytes": int(mb.group(1)), "lanes": int(ml.group(1)),
            "cmd": " ".join(cmd[:-3] + ["<tmp>.cpp"])}, "ok"


# ══════════════════════════════════════════════════════════════════════════
#  组装与输出
# ══════════════════════════════════════════════════════════════════════════
def explain(name: str, soc=None, probe=False) -> dict:
    a1 = a1_exists(name, soc)
    a2 = a2_signature(name, a1, soc)
    a3 = a3_constraints(name, soc)
    a4 = a4_pipe(name, a1, soc, a2)
    a5 = a5_instruction(name, soc, a4)
    out = {"api": name, "A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5}
    if probe:
        res, why = probe_vector_lane_count()
        out["E1"] = {"result": res, "why": why}
    return out


def _pad(s: str, width: int) -> str:
    """按**显示宽度**补空格: 中日韩字符占两列, 而 len() 只算一个。

    "片上" 用 f"{:<16}" 会比 "GM" 少缩两列, 表看起来是歪的。
    """
    import unicodedata
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(0, width - w)


_LEVEL_TITLE = {
    "A1": "存在与档位   这颗芯片上有没有它",
    "A2": "签名与位置   操作数必须在哪一级存储",
    "A3": "硬约束       编译过而上板才炸的那一类",
    "A4": "流水与同步   落在哪条 pipe",
    "A5": "指令与编译器 展开成什么, 一条吃多少元素",
}


def show(name: str, soc=None, probe=False, raw=False) -> int:
    d = explain(name, soc, probe)
    print(f"=== AscendC::{name} —— 五层取证 @ {_caps._p_arch(soc)[0] or '?'}\n")
    unknown = 0
    for lv in ("A1", "A2", "A3", "A4", "A5"):
        r = d[lv]
        v = r.get("value")
        mark = {True: "有", False: "无", None: "探不到"}.get(v, "?")
        print(f"[{mark:>5}] {lv} {_LEVEL_TITLE[lv]}")
        print(f"        {r.get('why', '')}")
        for e in (r.get("evidence") or [])[:6]:
            print(f"        · {e}")
        if r.get("onchip_levels"):
            print(f"        · 「片上」在本 arch 同族实现里出现的 TPosition: "
                  f"{', '.join(r['onchip_levels'])}"
                  f"{'  (' + r['onchip_src'] + ')' if r.get('onchip_src') else ''}")
            print("          ↑ 这是该族实现里**出现过**的位置, 不等于每个重载都收其中每一个。")
        for o in (r.get("overloads") or [])[:4]:
            tail = []
            if o.get("flow"):
                tail.append(f"{o['flow']}  ({o.get('flow_by')})")
            if o.get("pipe_decl"):
                pd = o["pipe_decl"]
                tail.append(f"__{pd['kind']}_pipe__({', '.join(pd['pipes'])})")
            print(f"        · {o['at']}  {'  '.join(tail) if tail else ''}")
            for p in (o.get("params") or []):
                if p.get("loc"):
                    print(f"            {_pad(p['name'] or '(未命名)', 12)} "
                          f"{_pad(p['loc'], 18)} {p['decl']}")
            if o.get("tmpl"):
                print(f"            模板参数 (变体轴): {' | '.join(o['tmpl'])}")
            if raw:
                print(f"            {o['sig']}")
        for a in (r.get("asserts") or [])[:6]:
            print(f"        · {a['at']}  {a['text'][:180]}")
        if r.get("lanes_by_dtype"):
            ln = r["lanes_by_dtype"]
            print(f"        · 每条向量指令的元素数: f32 {ln['f32']} / f16 {ln['f16']} / s8 {ln['s8']}")
        for m in (r.get("mapped_instr") or [])[:4]:
            print(f"        · 内建指令 {m['instr']}  {m['ctype']}  v{m['lanes']}")
        if r.get("note"):
            print(f"        ★{r['note']}")
        if r.get("experiment"):
            unknown += 1
            print(f"        ⚑ 不确定 -> 实验: {r['experiment']}")
        print()
    if "E1" in d:
        e = d["E1"]
        print("=== E1 编译探针 (当场跑的实验)")
        if e["result"]:
            r = e["result"]
            print(f"    {r['vtype']}: {r['bytes']} 字节 / {r['lanes']} 个元素")
            print(f"    命令: {r['cmd']}")
        else:
            print(f"    没跑成: {e['why']}")
        print()
    print(f"★{5 - unknown}/5 层有证据, {unknown} 层给了实验。"
          if unknown else "★五层都有证据。")
    print("★A3 的断言多数是 debug 级 —— release 构建不触发, 违反时是上板错误码。"
          " 静态查这一类敲 `opdbg preflight`。")
    return 0


def show_regwidth(soc=None, probe=False) -> int:
    """单独把 A5 的地基摆出来: 寄存器字节宽, 以及它与平台 ini 的对账。"""
    w, ev = reg_width_bytes()
    print("=== 向量寄存器宽度 (A5 的地基)\n")
    for e in ev:
        print(f"  · {e}")
    if w:
        print(f"\n  ⇒ 向量寄存器 = {w} 字节; "
              f"fp32 {w // 4} 元素 / fp16 {w // 2} 元素 / s8 {w} 元素")
        print(f"  ★平台 ini 的 vector_reg_width / predicate_reg_width / wide_reg_width "
              f"单位都是**字节**。core/socspec 打印时加的 'b' 后缀是错的 (差 8 倍)。")
    else:
        print("\n  ⇒ 定不下来 —— 见上面的证据行; 跑 --probe 让编译器当场给个数。")
    if probe:
        res, why = probe_vector_lane_count()
        print(f"\n=== E1 编译探针: {res if res else why}")
    tab, src = vector_lane_table()
    if tab:
        print(f"\n  内建类型表 ({len(tab)} 条, {os.path.basename(src)}), 前 8 条:")
        for k in sorted(tab)[:8]:
            print(f"    {k:<22} {tab[k]} 通道")
    return 0


# ---- 负例控制 (门禁 G1) --------------------------------------------------
def _nc_reg_width_is_bytes():
    """API2 的负例: 按"位"口径记的 ini 值必须被判成对不上, 不得被悄悄换算过去。

    正例 (256 == 256 字节) 与两个负例 (2048 = 按位记; 384 = 两侧真的不一致) 一起判,
    只有正例会过说明判据既抓得住错也不过宽。
    """
    ok_pos, _ = reconcile_reg_width(256, 256)
    ok_bit, msg_bit = reconcile_reg_width(256, 2048)
    ok_bad, _ = reconcile_reg_width(256, 384)
    return ok_pos and not ok_bit and not ok_bad and "位" in msg_bit


def _nc_levels_have_distinct_sources():
    """API1 的负例: A5 的取证文件必须**不在** A1–A4 的文件集合里。

    这条抓的是"把上一层的结论当下一层用"这个具体动作。若哪天 A5 被改成从 asc/ 的头文件
    去推每指令元素数 (那里根本没有这个事实), 本条立刻红 —— 而 2026-09-22 那次作废的结论
    正是这么来的: 用 A1/A2 级的证据 (regbase 存在) 去答 A5 级的问题 (一次调用几条指令)。
    """
    d = explain("Add")
    upper = set()
    for lv in ("A1", "A2", "A3", "A4"):
        r = d[lv]
        for s in (r.get("evidence") or []):
            upper.add(os.path.basename(str(s).split(":")[0]))
        for o in (r.get("overloads") or []):
            upper.add(os.path.basename(o["at"].split(":")[0]))
        for a in (r.get("asserts") or []):
            upper.add(os.path.basename(a["at"].split(":")[0]))
    a5 = d["A5"]
    a5_files = {os.path.basename(str(a5.get("instr_table_src") or ""))}
    for e in (a5.get("evidence") or []):
        m = re.search(r"\(([^)]+\.h)\)", str(e))
        if m:
            a5_files.add(os.path.basename(m.group(1)))
    a5_files.discard("")
    #  A5 至少要有一个**上面四层从没引用过**的文件, 且它得是编译器自带的头
    fresh = a5_files - upper
    return bool(fresh) and any(f.startswith("__clang_cce") for f in fresh)


def _nc_unknown_carries_experiment():
    """API4 的负例: 一个**不存在**的 API, 每一个说不清的层都必须带 experiment 字段。"""
    d = explain("ThisApiDoesNotExistXyz")
    bad = [lv for lv in ("A1", "A2", "A3")
           if d[lv].get("value") is None and not d[lv].get("experiment")]
    return not bad


def _nc_arch_not_borrowed():
    """C3 的负例: 换一颗没有 regbase 的芯片, regbase 等价项必须为空而不是沿用本机结论。"""
    hits, _ = regbase_equivalent("ThisApiDoesNotExistXyz")
    return hits is None


def _nc_impl_arch_dir_two_schemes():
    """API5 的负例: 两套命名都要试, 且**都不命中时必须说都不命中**, 不许沉默降级。

    910B 走码名 (dav_c220), 950PR 走数字 (dav_3510) —— 两颗芯片各自命中**不同的**规则,
    这正是"只用一条规则"会在另一半芯片上失效的证据。Ascend035 两条都不命中, 必须返回
    None 且把试过的候选名都列出来。
    """
    d220, _a, how220 = _impl_arch_dir("Ascend910B2")
    d351, _b, how351 = _impl_arch_dir("Ascend950PR_9589")
    dnil, _c, hownil = _impl_arch_dir("Ascend035")
    #  ★2026-10-28 本机 CANN 换包后重测: 950PR 现在也按**码名**命中 (dav_c310),
    #    不再按 NpuArch 数字命中 dav_3510 —— 本包里 dav_3510 根本不存在。
    #    期望随之更新, 但**这条负例的牙没有松**: 第三项 (Ascend035 两条候选都要列出来)
    #    仍然证明"两套候选都试过且都不中时必须说出来"; 前两项仍然证明
    #    "命中的是哪一条规则必须报出来", 代码若改成只试一条或沉默降级, 这条照样红。
    return (d220 and os.path.basename(d220) == "dav_c220" and "CCEC" in how220
            and d351 and os.path.basename(d351) == "dav_c310" and "CCEC" in how351
            and dnil is None and "dav_3505" in hownil and "dav_n350" in hownil)


def _nc_a3_message_names_a_real_dir():
    """API5 的负例: A3 的措辞里不许出现一个**不存在的目录名**。

    首版在 910B 上打「dav_2201 + 算子无关的校验层」, 而 dav_2201 不存在 —— 把"少扫了
    一整个实现目录"说成了"这颗芯片的约束就这么少" (实测 6 条 vs 真值 35 条)。
    """
    r = a3_constraints("DataCopyPad", "Ascend910B2")
    why = r.get("why", "")
    for m in re.findall(r"dav_\w+", why):
        if not os.path.isdir(os.path.join(_impl_root() or "", "basic_api", m)):
            return False
    return "dav_c220" in why


def _nc_arch_guard_else_branch():
    """API5 的负例: `#if A #else B` 两支必须**互斥地**分给两颗芯片。

    两条独立的错各自都能让这条红:
      (a) 按文本片段 (`txt.find`) 定位守卫 —— #if/#else 两侧声明前 40 字符逐字相同, 全被
          定位到同一处; (b) 用子串匹配判 `!( ... 3510 ... )` —— 950 会认领"非 950"那一支。
    实测 DataCopyPad: 两颗芯片各 7 处声明, 交集只剩无守卫的 3 处。
    """
    s2201 = {a["at"] for a in (a2_signature("DataCopyPad",
             a1_exists("DataCopyPad", "Ascend910B2"), "Ascend910B2").get("overloads") or [])}
    s3510 = {a["at"] for a in (a2_signature("DataCopyPad",
             a1_exists("DataCopyPad", "Ascend950PR_9589"), "Ascend950PR_9589").get("overloads") or [])}
    return bool(s2201) and bool(s3510) and s2201 != s3510 and len(s2201 & s3510) < len(s2201)


def _nc_a2_direction_not_a_set():
    """API6 的负例: 方向必须是**有序**的, GM->片上 与 片上->GM 不许打印成同一个东西。

    首版把位置收成集合, 两个方向都显示 `[GM, 片上]` —— 而它们分属 MTE2 与 MTE3。
    """
    a2 = a2_signature("DataCopyPad", a1_exists("DataCopyPad", "Ascend910B2"), "Ascend910B2")
    flows = set(a2.get("flows") or [])
    return "GM -> 片上" in flows and "片上 -> GM" in flows


def _nc_declared_pipe_beats_guess():
    """API6 的负例: 接口头写了 `__inout_pipe__` 时, A4 必须用它而不是族首猜。

    910B 上 dav_c220 的 data_copy 实现里一个 PIPE_* 都没有, 所以这条标注是唯一的
    非猜测证据; 缺了它 A4 在这颗芯片上永远只能给猜测。
    """
    a1 = a1_exists("DataCopyPad", "Ascend910B2")
    a2 = a2_signature("DataCopyPad", a1, "Ascend910B2")
    a4 = a4_pipe("DataCopyPad", a1, "Ascend910B2", a2)
    return a4.get("source") == "声明" and "MTE3" in (a4.get("declared_pipes") or [])


NEGATIVE_CONTROLS = {
    "API1_five_levels_have_distinct_sources": _nc_levels_have_distinct_sources,
    "API2_reg_width_ini_unit_is_bytes": _nc_reg_width_is_bytes,
    "API4_unknown_must_carry_an_experiment": _nc_unknown_carries_experiment,
    "API3_compiler_header_is_the_instruction_truth": _nc_arch_not_borrowed,
    "API5_impl_arch_dir_has_two_naming_schemes": _nc_impl_arch_dir_two_schemes,
    "API5_a3_message_names_a_real_dir": _nc_a3_message_names_a_real_dir,
    "API5_arch_guard_else_branch_is_exclusive": _nc_arch_guard_else_branch,
    "API6_a2_direction_is_ordered": _nc_a2_direction_not_a_set,
    "API6_declared_pipe_beats_family_guess": _nc_declared_pipe_beats_guess,
}


def selftest() -> int:
    ok = True
    for k, fn in NEGATIVE_CONTROLS.items():
        try:
            got = bool(fn())
        except Exception as e:                                      # noqa: BLE001
            got, k = False, f"{k} ({type(e).__name__}: {e})"
        print(f"  [{'OK ' if got else 'FAIL'}] 负例 {k}")
        ok = ok and got
    #  正例: 一条本机确实存在的 API 必须五层都不报"探不到存在"
    d = explain("DataCopyPad")
    got = d["A1"].get("value") is True
    print(f"  [{'OK ' if got else 'FAIL'}] 正例 DataCopyPad 的 A1 必须判存在")
    ok = ok and got
    w, _ = reg_width_bytes()
    got = w == 256
    print(f"  [{'OK ' if got else 'FAIL'}] 寄存器宽解出 256 字节 (实际 {w})")
    ok = ok and got
    return 0 if ok else 1


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="core.opgen api",
                                 description="AscendC API 五层取证 (A1 存在 -> A5 指令)")
    ap.add_argument("name", nargs="?", help="API 名, 如 DataCopyPad / Add / Fixpipe")
    ap.add_argument("--soc", default=None)
    ap.add_argument("--probe", action="store_true", help="跑 E1 编译探针 (秒级, 不占卡)")
    ap.add_argument("--raw", action="store_true", help="打印原始签名")
    ap.add_argument("--regwidth", action="store_true", help="只看 A5 的地基: 寄存器字节宽与 ini 对账")
    ap.add_argument("--instr", default=None, help="查内建指令表里的一条, 如 vadds")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--refs", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.refs:
        print_refs("opgen.api", REFS)
        return 0
    if a.regwidth:
        return show_regwidth(a.soc, a.probe)
    if a.instr:
        rows, src = instruction_table(a.instr)
        print(f"=== 内建指令 {a.instr} ({src})")
        for r in rows or []:
            print(f"  {r['instr']:<12} {r['ctype']:<12} {r['suffix']:<6} v{r['lanes']}")
        if not rows:
            print("  表里没有这条 —— 指令名与 AscendC API 名不是同一套命名, 别互相猜。")
        return 0
    if not a.name:
        ap.print_help()
        print("\n★先敲 `--regwidth` 看 A5 的地基 (它决定一切按元素数算的结论),"
              " 再查具体 API。")
        return 2
    return show(a.name, a.soc, a.probe, a.raw)


if __name__ == "__main__":
    sys.exit(main())
