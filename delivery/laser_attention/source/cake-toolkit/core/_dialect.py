"""源码方言识别 + 「判据未生效」的统一上报。

★为什么有这个模块 (2026-10-21 SMLA 实撞):
  一整套 build 前静态判据 (ub_budget / alignlint / detsync_lint / wscap / gmtraffic /
  mixaddr) 都是按 **AscendC** 的写法建的 —— 它们找的是 `pipe->InitBuffer(...)`、
  `__aicore__` 函数体、`GlobalTensor` 声明。而 cases/SMLA 用的是 CANNBot-DSL:
  `@jit` + `mem_copy` / `tile_slice`, 一个 AscendC 记号都没有。

  在那份源码上, 这几条判据**全部打印"未命中 / 无问题 / OK" 并 rc=0**。
  于是 `opdbg preflight` 给出 "6 条判据, 0 条未过" —— 而实际上它一条也没看见。
  这是本仓反复栽的那个坑的最高级形态: **"找不到" 被读成 "没有问题"**。
  (同族已两次: perf 找不到 PipeUtilization 就报错说没有数据; hwfloor 的假下界。)

  修法不是给 SMLA 加一套解析器 (那是针对特定算子), 而是让**每一条判据在看不见源码时
  自己举手**: 方言不符 -> 打印 `★未生效`, 并让 dispatcher 把它和 "通过" 分开计数。
  "未生效" 既不吞掉退出码 (不制造永远红的闸), 也不再冒充绿。
"""
from __future__ import annotations
import os

MARK = "★未生效"          # 上行聚合 (opdbg preflight) 认这个前缀, 改它要同步改 opdbg

#  AscendC 的记号: 命中任意一个就认为这条源码在判据的方言里。
_ASCENDC = ("__aicore__", "__global__", "AscendC::", "TPipe", "TQue", "TBuf",
            "GlobalTensor", "kernel_operator.h", "LOCAL_TENSOR")
#  CANNBot-DSL / Python 内核的记号。
_DSL = ("@jit", "mem_copy(", "tile_slice(", "const_expr", "range_constexpr",
        "Channel(", "vec_sync_block_arrive", "async_copy(")

_SRC_EXT = (".h", ".hpp", ".cpp", ".cc", ".cxx", ".py")
_MAX_FILES = 400


def _iter_src(paths, limit: int = _MAX_FILES):
    n = 0
    for p in paths:
        if not p:
            continue
        p = str(p)
        if os.path.isfile(p):
            yield p
            n += 1
            if n >= limit:
                return
            continue
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for f in sorted(files):
                if f.endswith(_SRC_EXT):
                    yield os.path.join(root, f)
                    n += 1
                    if n >= limit:
                        return


def detect(paths) -> str:
    """'ascendc' | 'dsl' | 'unknown'。看的是**记号**, 不是扩展名。"""
    asc = dsl = False
    for f in _iter_src(paths):
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                txt = fh.read()
        except OSError:
            continue
        if not asc and any(m in txt for m in _ASCENDC):
            asc = True
        if not dsl and any(m in txt for m in _DSL):
            dsl = True
        if asc:
            return "ascendc"        # AscendC 优先: 混合仓里 Python 只是脚手架
    if dsl:
        return "dsl"
    return "unknown"


def count(paths, needle: str) -> int:
    """源码里 needle 出现的次数。用来区分**两种 0 命中**: 真没有 vs 解析器没认出写法。"""
    n = 0
    for f in _iter_src(paths):
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                n += fh.read().count(needle)
        except OSError:
            continue
    return n


def not_effective(tool: str, paths, what: str) -> bool:
    """方言不等于本判据的前提时, 打印 `★未生效` 并返回 True。否则返回 False。

    调用点放在**判据已经数出 0 个对象之后** —— 有命中就说明解析成功, 不必报警。
    """
    dialect = detect(paths)
    if dialect == "ascendc":
        return False
    srcs = [p for p in paths if p]
    where = ", ".join(os.path.basename(str(s).rstrip("/")) for s in srcs[:3]) or "(未给源码)"
    hint = ("CANNBot-DSL 内核 (Python, @jit/mem_copy)" if dialect == "dsl"
            else "没认出来的方言")
    print(f"\n{MARK} [{tool}] 本次一条判据也没有执行 —— 源码里找不到 {what}。")
    print(f"         输入: {where}   识别为: {hint}")
    print(f"         本模块只认 AscendC 写法 (__aicore__ / TPipe / InitBuffer / GlobalTensor)。")
    print(f"         ★「无命中」在这里的意思是**没看见**, 不是**没问题**。")
    return True


def missing_source(tool: str, path: str) -> bool:
    """源码路径本身不存在/为空时也要举手 —— 空目录走完 walk 同样产出 0 命中。"""
    if path and os.path.exists(path):
        return False
    print(f"\n{MARK} [{tool}] 源码路径不存在: {path!r}")
    return True


def selftest() -> int:
    bad = 0
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "k.h")
        open(a, "w").write("__aicore__ void K() { AscendC::TPipe p; }\n")
        b = os.path.join(td, "k.py")
        open(b, "w").write("@jit\ndef k():\n    mem_copy(a, b)\n")
        c = os.path.join(td, "empty.py")
        open(c, "w").write("x = 1\n")
        d2 = os.path.join(td, "dsl_only"); os.makedirs(d2)
        open(os.path.join(d2, "k.py"), "w").write("@jit\ndef k():\n    mem_copy(x, y)\n")
        cases = [(a, "ascendc"), (b, "dsl"), (c, "unknown"), (td, "ascendc"), (d2, "dsl")]
        for p, want in cases:
            got = detect([p])
            ok = got == want
            print(f"{'ok ' if ok else 'FAIL'} detect({os.path.basename(p)}) = {got} (want {want})")
            bad += 0 if ok else 1
        #  未生效: AscendC 不该报, dsl 必须报
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r_asc = not_effective("t", [a], "InitBuffer")
            r_dsl = not_effective("t", [b], "InitBuffer")
        ok = (not r_asc) and r_dsl and MARK in buf.getvalue()
        ok2 = count([b], "mem_copy(") == 1
        print(f"{'ok ' if ok2 else 'FAIL'} count: {count([b], 'mem_copy(')} (want 1)")
        ok = ok and ok2
        print(f"{'ok ' if ok else 'FAIL'} not_effective: ascendc={r_asc} dsl={r_dsl}")
        bad += 0 if ok else 1
    return bad


if __name__ == "__main__":
    import sys
    sys.exit(1 if selftest() else 0)
