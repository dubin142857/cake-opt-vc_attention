#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mixaddr —— 混合核 (MIX_AIC_1_2) 两侧共享缓冲的**地址表一致性**。

为什么建 (2026-09-22, sparse_flash_attention 的 C-V 融合反向沉淀)
-----------------------------------------------------------------
arch35 的矩阵单元可以把 L0C 经 `Fixpipe<..., {ROW_MAJOR, isToUB=true}>` 直接写进
**对侧向量核的 UB**, 向量侧再原地消费。这是 C-V 融合省掉中间量 GM 往返的机制
(sparse_flash_attention 上这段往返是 case 3 的 4.29 GB, 占该 case 耗时的大头)。

它的交接方式与 TPipe 分配的缓冲**根本不同**: 两侧各自用显式地址声明同一块内存

    LocalTensor<float> ubScore = LocalTensor<float>(TPosition::VECIN, 0, N);      // 矩阵侧
    LocalTensor<float> ubScore = LocalTensor<float>(TPosition::VECIN, 0, N);      // 向量侧

没有任何机制保证这两行一致 —— 它们在不同的 `#ifdef` 分支里, 编译器各编各的,
两边写不同的偏移照样编译通过, 运行时表现为**向量侧读到别的缓冲的内容**: 数值错但
不报错、不挂死, 且错得结构化 (像是算法写错), 极易被误判成 softmax/掩码的逻辑问题。

判据
----
M1  两侧同名共享张量的 (TPosition, 地址表达式, 元素数表达式) 必须逐字相同
M2  矩阵侧声明的共享张量必须是向量侧声明序列的**前缀**(同名同序同位置) ——
    向量侧可以在共享缓冲之后再追加自己的私有缓冲, 反之不行: 矩阵侧多出来或顺序
    不同, 说明两侧对同一段 UB 的划分已经分叉
M3  用了 `Fixpipe` 目的地为 UB 位置 (VECIN/VECOUT/VECCALC) 却没有成对的
    `CrossCoreSetFlag` / `CrossCoreWaitFlag`, 或两侧 flag 号对不上
    ★1 个 AIC 配 2 个 AIV: 矩阵侧每个方向要发**两个** flag (f 与 f+16),
      向量侧各等一个。矩阵侧只发一个 = 另一个 AIV 永久挂起。

不判什么
--------
地址表达式按**字面**比较, 不做符号求值。`ubAddr += A * sizeof(T)` 这类累加序列
两侧只要写法一致就通过, 写法不同 (哪怕数值相同) 会报出来让人自己看 —— 这是刻意的:
让两侧保持逐字相同本身就是该守的纪律, "数值上碰巧相等"不是。
"""
from __future__ import annotations
import argparse
import os
import re
import sys

from ._refs import Ref, print_refs
from . import _dialect

REFS = {
    "M1_mix_shared_addr_agree": Ref(
        what="混合核两侧用显式地址声明的同名共享张量, (TPosition, 地址, 元素数) 必须逐字相同",
        status="derived",
        evidence=[
            "仓内参考 ops-transformer-master/examples/attention/flash_attn/L0_minimal:",
            "fa_block_cube.h:216-222 与 fa_block_vec.h:200-203 两侧逐字相同 (正例, 判据通过);",
            "注入 64 B 偏移后 M1 命中 (负例, 判据可被证伪)。",
            "同构生产用法 mc2/3rd/quant_batch_matmul_v3/.../qbmm_perblock_api_utils.h。",
            "2026-10-20 补: 上述手工双向校验已自动化为 --selftest (正例 1 + 注入 3);",
            "同日跨算子普查 5 个无显式地址共享的 op_kernel (sfag/sfag_train/ffn/",
            "moe_token_unpermute/gmfr v8.4 交付) 全部优雅报不适用, 零误报。",
            "★status 是 derived 而非 measured: 判据本身双向自校验过, 但'两侧地址不一致会",
            "  导致向量侧读到别的缓冲'这条因果还没在本仓真机上实撞。撞到后升 measured。",
        ],
        verified="2026-10-20"),
    "M3_mix_flag_pairs_two": Ref(
        what="1 个 AIC 配 2 个 AIV, 矩阵侧每个方向要发两个 CrossCore flag (f 与 f+16)",
        status="derived",
        evidence=[
            "fa_block_cube.h 的 ComputeMm1/ComputeMm2 各发 (f) 与 (f+16);",
            "qbmm_perblock_api_utils.h:35-46 的 WaitForVector/NotifyVector 同形。",
            "向量侧每核只等/发一个, 硬件按 subBlockIdx 把它映射到 f 或 f+16。",
            "★尚未在本仓真机上跑通混合核 —— 跑通后应升 measured。",
        ],
        verified="2026-09-22"),
}

#  显式地址构造的两种写法: 变量声明式与赋值式
_DECL = re.compile(
    r"LocalTensor\s*<\s*(?P<ty>[^>]+?)\s*>\s*"
    r"(?:(?P<name1>\w+)\s*\(|(?P<name2>\w+)\s*=\s*LocalTensor\s*<[^>]+>\s*\()"
    r"\s*(?:AscendC::)?TPosition::(?P<pos>\w+)\s*,\s*(?P<addr>[^,]+?)\s*,\s*(?P<cnt>[^)]+?)\s*\)")

_UB_POS = ("VECIN", "VECOUT", "VECCALC", "LCM")
_SHARED_POS = _UB_POS + ("A1", "B1", "TSCM")

#  两侧的分界。仓里见过的两套写法都认。
_CUBE_ON = (re.compile(r"#\s*ifdef\s+__DAV_C310_CUBE__"),
            re.compile(r"if\s+ASCEND_IS_AIC\b"))
_VEC_ON = (re.compile(r"#\s*ifndef\s+__DAV_C310_CUBE__"),
           re.compile(r"if\s+ASCEND_IS_AIV\b"))


class Decl:
    __slots__ = ("name", "pos", "addr", "cnt", "ty", "file", "line")

    def __init__(self, name, pos, addr, cnt, ty, file, line):
        self.name, self.pos, self.addr, self.cnt = name, pos, addr, cnt
        self.ty, self.file, self.line = ty, file, line

    def key(self):
        return (self.pos, _norm(self.addr), _norm(self.cnt), _norm(self.ty))

    def where(self):
        return f"{self.file}:{self.line}"

    def __str__(self):
        return f"{self.name} ({self.pos}, addr={self.addr.strip()}, n={self.cnt.strip()})"


def _norm(s):
    return re.sub(r"\s+", "", s or "")


def _side_of(text, upto):
    """upto 之前最后一个还生效的分界标记决定这行属于哪一侧。

    只做最朴素的判断: 数 #if/#endif 的深度太脆 (仓里 cube/vec 分界恰好都是整文件级,
    参考例 fa_block_cube.h / fa_block_vec.h 即此)。所以先看整文件的门, 再看函数内的
    `if ASCEND_IS_AIC` 块。判不出就归 'both', 由调用方按文件名兜底。
    """
    head = text[:upto]
    last, side = -1, None
    for pats, tag in ((_CUBE_ON, "cube"), (_VEC_ON, "vec")):
        for p in pats:
            for m in p.finditer(head):
                if m.start() > last:
                    last, side = m.start(), tag
    return side


def scan(paths):
    cube, vec, other = [], [], []
    fixpipe_ub, setflag, waitflag = [], [], []
    for path in paths:
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        base = os.path.basename(path)
        hint = "cube" if "cube" in base else ("vec" if ("vec" in base or "vector" in base) else None)
        for m in _DECL.finditer(text):
            pos = m.group("pos")
            if pos not in _SHARED_POS:
                continue
            name = m.group("name1") or m.group("name2")
            line = text.count("\n", 0, m.start()) + 1
            d = Decl(name, pos, m.group("addr"), m.group("cnt"), m.group("ty"), path, line)
            side = _side_of(text, m.start()) or hint
            (cube if side == "cube" else vec if side == "vec" else other).append(d)
        for m in re.finditer(r"Fixpipe\s*<[^;]*?>\s*\(\s*(\w+)", text):
            fixpipe_ub.append((path, text.count("\n", 0, m.start()) + 1, m.group(1)))
        for m in re.finditer(r"CrossCoreSetFlag\s*<([^>]*)>\s*\(([^)]*)\)", text):
            setflag.append((path, text.count("\n", 0, m.start()) + 1,
                            _side_of(text, m.start()) or hint, _norm(m.group(2))))
        for m in re.finditer(r"CrossCoreWaitFlag\s*<([^>]*)>\s*\(([^)]*)\)", text):
            waitflag.append((path, text.count("\n", 0, m.start()) + 1,
                             _side_of(text, m.start()) or hint, _norm(m.group(2))))
    return cube, vec, other, fixpipe_ub, setflag, waitflag


def report(cube, vec, other, fixpipe_ub, setflag, waitflag, files_g=None):
    bad = 0
    if not cube and not vec:
        #  ★"不适用" 的理由要问方言 (2026-10-21 SMLA): 在 DSL 源码上它不是"非混合核",
        #    而是"这些记号一个都没有" —— 两种情形对读者的含义完全不同。
        why = ("源码里没有任何 AscendC 地址构造记号 (方言不符, 判据未生效)"
               if _dialect.detect(files_g) != "ascendc" else
               "非混合核, 或缓冲全由 TPipe 分配")
        print(f"未发现显式地址构造的共享张量 —— 本判据不适用 ({why})。")
        return 0

    cmap, vmap = {}, {}
    for d in cube:
        cmap.setdefault(d.name, d)
    for d in vec:
        vmap.setdefault(d.name, d)

    # M1
    m1 = []
    for name, cd in cmap.items():
        vd = vmap.get(name)
        if vd is not None and cd.key() != vd.key():
            m1.append((cd, vd))
    if m1:
        bad += 1
        print(f"[M1] 两侧同名共享张量的声明不一致 {len(m1)} 处 —— 向量侧会读到别的缓冲:")
        for cd, vd in m1:
            print(f"  {cd.name}")
            print(f"    矩阵侧 {cd.where()}  {cd}")
            print(f"    向量侧 {vd.where()}  {vd}")

    # M2  —— 按 TPosition 分别比: L1 与 UB 是两个地址空间, 合成一条序列比会误报。
    #      矩阵侧私有的 L1 缓冲 (l1Q/l1K/l1V 这类) 只在矩阵侧出现是正常的;
    #      **UB 位置**上只在矩阵侧出现才是错 —— 矩阵侧写进去而向量侧没人按这个地址读。
    for pos in sorted({d.pos for d in cube} | {d.pos for d in vec}):
        cn = [d.name for d in cube if d.pos == pos]
        vn = [d.name for d in vec if d.pos == pos]
        both = set(cn) & set(vn)
        cs2 = [n for n in cn if n in both]
        vs2 = [n for n in vn if n in both]
        if cs2 != vs2:
            bad += 1
            print(f"[M2] {pos} 上两侧共享张量的相对顺序不同 —— 对同一段内存的划分已分叉:")
            print(f"    矩阵侧 {cs2}")
            print(f"    向量侧 {vs2}")
        if pos in _UB_POS:
            only_c = [n for n in cn if n not in both]
            if only_c:
                bad += 1
                print(f"[M2] {pos} 上这些张量只在矩阵侧声明: {only_c}")
                print("     矩阵侧 Fixpipe 写进对侧 UB, 向量侧却没有按这个地址声明的张量,"
                      " 写进去的数据没人读。")

    # M3
    if fixpipe_ub:
        cs = [f for f in setflag if f[2] == "cube"]
        cw = [f for f in waitflag if f[2] == "cube"]
        vs = [f for f in setflag if f[2] == "vec"]
        vw = [f for f in waitflag if f[2] == "vec"]
        if not (cs and vw):
            bad += 1
            print("[M3] 有 Fixpipe 但缺跨核同步: "
                  f"矩阵侧 Set {len(cs)} 条 / 向量侧 Wait {len(vw)} 条 —— 向量侧会读到未写完的 UB。")
        else:
            #  1 AIC 配 2 AIV: 矩阵侧每个方向要发两个 flag (f 与 f+16)
            cs_ids = [x[3] for x in cs]
            lone = [i for i in cs_ids if ("+16" not in i and "+FLAG" not in i.upper())
                    and not any((i + "+16") == j or j.startswith(i + "+") for j in cs_ids)]
            if lone:
                bad += 1
                print(f"[M3] 矩阵侧这些 CrossCoreSetFlag 没有配对的 +16 版本: {sorted(set(lone))}")
                print("     1 个 AIC 配 2 个 AIV, 每个方向要发两个 flag; 只发一个 = 另一个 AIV 永久等待。")
            cw_ids = [x[3] for x in cw]
            lone_w = [i for i in cw_ids if ("+16" not in i and "+FLAG" not in i.upper())
                      and not any(j.startswith(i + "+") for j in cw_ids)]
            if lone_w:
                bad += 1
                print(f"[M3] 矩阵侧这些 CrossCoreWaitFlag 没有配对的 +16 版本: {sorted(set(lone_w))}")
            if not vs:
                bad += 1
                print("[M3] 向量侧没有 CrossCoreSetFlag —— 矩阵侧拿不到"
                      "「缓冲已被消费、可以覆写」的额度, 第二轮就会挂住。")

    if other:
        print(f"\n归不到哪一侧的共享张量声明 {len(other)} 处 (判据判不出, 是待查清单不是判定):")
        for d in other[:10]:
            print(f"  {d.where()}  {d}")

    if not bad:
        print(f"矩阵侧 {len(cube)} 个 / 向量侧 {len(vec)} 个共享张量, "
              f"M1 M2 M3 全过。")
    return 1 if bad else 0


def selftest() -> int:
    """判据负例自检: 把 REFS 里 2026-09-22 那次**手工**双向校验自动化成夹具 (2026-10-20)。
    夹具 = 最小 cube/vec 两侧源对; 正例必须 M1/M2/M3 全过, 三个注入变体必须各自命中。"""
    import contextlib
    import io
    import tempfile
    bad = 0

    def _say(ok, msg):
        print(f"  {'过' if ok else '★未过'} {msg}")
        return 0 if ok else 1

    #  全双工同步才是正例: 矩阵侧发 f/f+16 (数据就绪), 并等 g/g+16 (覆写额度);
    #  向量侧等 f、发 g。少任何一个方向都是 M3 真命中, 不是夹具可省略的装饰。
    CUBE_OK = ("#ifdef __DAV_C310_CUBE__\n"
               "LocalTensor<float> ubScore(TPosition::VECIN, 0, 4096);\n"
               "Fixpipe<Cfg>(ubScore, l0c, p);\n"
               "CrossCoreSetFlag<2>(f);\nCrossCoreSetFlag<2>(f + 16);\n"
               "CrossCoreWaitFlag<2>(g);\nCrossCoreWaitFlag<2>(g + 16);\n#endif\n")
    VEC_OK = ("#ifndef __DAV_C310_CUBE__\n"
              "LocalTensor<float> ubScore(TPosition::VECIN, 0, 4096);\n"
              "CrossCoreWaitFlag<2>(f);\nCrossCoreSetFlag<2>(g);\n#endif\n")

    def run_pair(cube_src, vec_src):
        with tempfile.TemporaryDirectory() as td:
            c = os.path.join(td, "fx_cube.h")
            v = os.path.join(td, "fx_vec.h")
            open(c, "w").write(cube_src)
            open(v, "w").write(vec_src)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = report(*scan([c, v]))
            return rc, buf.getvalue()

    print("=== mixaddr 判据负例自检 (正例 1 + 注入 3) ===")
    rc, out = run_pair(CUBE_OK, VEC_OK)
    bad += _say(rc == 0 and "全过" in out, "正例: 两侧逐字一致 -> M1/M2/M3 全过")
    rc, out = run_pair(CUBE_OK, VEC_OK.replace("TPosition::VECIN, 0,", "TPosition::VECIN, 64,"))
    bad += _say(rc == 1 and "[M1]" in out, "注入 64B 地址偏移 -> M1 必抓 (REFS 记载的那个负例)")
    rc, out = run_pair(CUBE_OK.replace("Fixpipe", "LocalTensor<float> ubPriv(TPosition::VECIN, 8192, 128);\nFixpipe"),
                       VEC_OK)
    bad += _say(rc == 1 and "[M2]" in out and "ubPriv" in out,
                "UB 张量只在矩阵侧声明 -> M2 必抓 (写进去的数据没人读)")
    rc, out = run_pair(CUBE_OK, VEC_OK.replace("CrossCoreWaitFlag<2>(f);\n", ""))
    bad += _say(rc == 1 and "[M3]" in out, "有 Fixpipe 而向量侧无 Wait -> M3 必抓")
    print(f"mixaddr 自检: 4 条, {bad} 条未过")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(
        prog="core.mixaddr",
        description="混合核两侧共享 UB/L1 缓冲的地址表一致性 (M1 声明一致 / M2 序列前缀 / M3 跨核 flag 成对)")
    ap.add_argument("paths", nargs="*", help="op_kernel 目录或若干源文件")
    ap.add_argument("--refs", action="store_true", help="列出本模块判据的实证出处与最后验证日期")
    ap.add_argument("--selftest", action="store_true", help="判据负例自检 (正例 1 + 注入 3, 纯夹具)")
    a = ap.parse_args()
    if a.refs:
        print_refs("mixaddr", REFS)
        return 0
    if a.selftest:
        return selftest()
    files = []
    for p in a.paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in names
                          if n.endswith((".h", ".hpp", ".cpp", ".cc"))]
        else:
            files.append(p)
    if not files:
        #  ★目录里只有 .py (DSL 内核) 时也要说清是哪一种"没有"。
        if _dialect.detect([p]) != "ascendc":
            _dialect.not_effective("mixaddr", [p], "AscendC 的 .h/.cpp 源文件")
        else:
            print("没有可扫的源文件。")
        return 0
    return report(*scan(sorted(files)), files_g=sorted(files))


if __name__ == "__main__":
    sys.exit(main())
