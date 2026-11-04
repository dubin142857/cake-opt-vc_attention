"""算子 spec 自动生成 — 从 CANN ops-info JSON 抽取机读的合法参数包络, 写成 agent 友好的 op.yaml 骨架.

依据 (方法论 infra 先行): 算子的合法 dtype/format/attr 组合是 CANN 自己声明的机读事实
(aic-<soc>-ops-info-<class>.json). 手抄易错, 自动抽取是唯一正确来源.

用法:
    python -m core.gen_spec GroupedMatmulFinalizeRouting \
        --soc ascend910b --cann /usr/local/Ascend/cann-9.1.0-beta.1 \
        --out cases/grouped_matmul_finalize_routing/op.autogen.yaml

生成的是 op.yaml 的 `spec:` 段骨架 (合法包络 + attr 默认值 + IO dtype/format 矩阵).
问题描述 (problem:) 与参考实现由人/agent 补, 不自动生成.
"""
from __future__ import annotations

import argparse
import glob
import re
import json
import os
from typing import Optional

try:
    import yaml
except ImportError:  # pyyaml 缺失时退化为 json 输出
    yaml = None


def _find_ops_info(cann: str, soc: str, vendor: str | None = None) -> list:
    """ops-info JSON 的位置. 默认是 CANN 的 built-in 树.

    ★为什么要能指向 vendor 包 (2026-09-10, SparseFlashAttention 实撞): 本机装的 CANN 包比
      源码仓旧, 而且没装 opp_kernel —— 新算子 (SparseFlashAttention / FlashAttn / FIA) 在
      built-in 树里**根本没有条目**, gen_spec 只报"未找到", 于是"spec 机读优先"这条硬约束
      就没法执行, 只能手抄。而自编的 customize vendor 包里带着**自己的** ops-info
      (op_impl/ai_core/tbe/config/<soc>/aic-<soc>-ops-info.json), 那正是要抽的东西。
      注意 vendor 包里的文件名没有 -<class> 后缀, 故两种模式都 glob。
    """
    if vendor:
        pats = [os.path.join(vendor, "op_impl/ai_core/tbe/config", soc,
                             f"aic-{soc}-ops-info*.json")]
    else:
        pats = [os.path.join(cann, "opp/built-in/op_impl/ai_core/tbe/config", soc,
                             f"aic-{soc}-ops-info-*.json")]
    out = []
    for p in pats:
        out += glob.glob(p)
    return sorted(out)


def _locate_op(files: list, op: str) -> tuple:
    for f in files:
        d = json.load(open(f))
        for k in d:
            if k.lower() == op.lower():
                return f, d[k]
    return None, None


def extract_spec(op: str, soc: str, cann: str, vendor: str | None = None) -> dict:
    files = _find_ops_info(cann, soc, vendor)
    src, node = _locate_op(files, op)
    if node is None:
        where = f"vendor 包 {vendor}" if vendor else f"CANN built-in 树 {cann}"
        raise SystemExit(
            f"未在 {where} 的 {soc} ops-info 中找到算子 {op}; 已扫 {len(files)} 个 json。"
            f"\n★本机装的 CANN 可能比源码仓旧 —— 新算子在 built-in 树里没有条目。"
            f"先用 core.opbuild 编出 customize vendor 包, 再 --vendor <该包目录> 重抽。")

    inputs, outputs, attrs = [], [], {}
    for key in sorted(node):
        v = node[key]
        if key.startswith("input") and isinstance(v, dict):
            inputs.append({
                "name": v.get("name", key),
                "param_type": v.get("paramType"),
                "dtype": v.get("dtype", "").split(","),
                "format": v.get("format", "").split(","),
            })
        elif key.startswith("output") and isinstance(v, dict):
            outputs.append({
                "name": v.get("name", key),
                "param_type": v.get("paramType"),
                "dtype": v.get("dtype", "").split(","),
                "format": v.get("format", "").split(","),
            })
        elif key.startswith("attr_") and isinstance(v, dict):
            attrs[key[len("attr_"):]] = {
                "type": v.get("type"),
                "default": v.get("defaultValue"),
                "param_type": v.get("paramType"),
            }

    return {
        "op": op,
        "soc": soc,
        "cann": os.path.basename(cann.rstrip("/")),
        "ops_info_source": src,
        "attr_order": node.get("attr", {}).get("list"),
        "inputs": inputs,
        "outputs": outputs,
        "attrs": attrs,
    }


#  ---- aclnn 入口校验的机读抽取 ----
#  为什么必须抽: ops-info JSON 只声明 dtype/format/attr, **不含维度整除类约束**. 那些约束写死在
#  aclnn 参数校验里 (libopapi*.so), 违反时报 EZ1001 而不是走到 kernel. 于是同一条约束决定了
#  "某个对齐缺陷到底可不可达" —— 不抽出来, 静态 lint 只能按最坏情形报, 真命中会被不可达项埋掉.
#  gmfr 实例: aclnn 要求 n mod 32 == 0, 于是按 n%8 报的三条全都够不着; 而界的 alignBaseN>>3
#  除完只剩 4 的倍数, 那条是真的. 判"哪条真"完全取决于这个契约.
#  抽法: 校验消息以明文串存在 .so 里, 且同一算子的全部校验串在字符串表里连续成块, 故按算子名
#  在窗口内取串即可. 属 CANN 通用布局, 与具体算子无关.
_CONSTRAINT_PATTERNS = [
    (re.compile(r"(\w+)\s+mod\s+(\d+)\s+equal\s+0"), "mod"),
    (re.compile(r"(\w+)\s+bigger\s+or\s+equal\s+(\d+)"), "min"),
    (re.compile(r"(\w+)\s+dim\s+num\s+should\s+be\s+equal\s+to\s+(\d+)"), "dims"),
]
_CONSTRAINT_HINT = re.compile(r"mod\s+\d+\s+equal|bigger or equal|do not support|only support|should be equal to")


def _opapi_libs(cann: str) -> list:
    root = os.path.dirname(cann.rstrip("/")) if os.path.basename(cann.rstrip("/")).startswith("cann-") else cann
    here = glob.glob(os.path.join(cann, "*", "lib64", "libopapi*.so"))
    if here:
        return sorted(set(here))     # 优先只看指定版本, 否则每个装过的 CANN 都要 strings 一遍
    return sorted(set(glob.glob(os.path.join(root, "*", "*", "lib64", "libopapi*.so"))))


#  ★`strings` 一个 so 要 0.4 秒, 而 libopapi 系列有 12 个 —— 每调用一次 entry_constraints
#  就重跑一轮约 4 秒。批量扫一棵算子树 (ops-nn 有 361 个算子) 时就是 24 分钟, 且全程无输出,
#  看起来与卡死无法区分 (2026-09-17 实撞: wscap --scan 跑满 550 秒超时, 一行都没打)。
#  按 so 路径缓存行表: 一轮 strings 之后每个算子只在内存里查, 总内存约 25 MB。
_STRINGS_CACHE: dict = {}


def _strings_lines(so: str):
    """so 的 strings 输出按行缓存; 取不到返回 None (调用方跳过该 so)。"""
    if so not in _STRINGS_CACHE:
        import subprocess
        try:
            raw = subprocess.run(["strings", so], capture_output=True, text=True,
                                 errors="ignore", timeout=300).stdout
            _STRINGS_CACHE[so] = raw.splitlines()
        except Exception:
            _STRINGS_CACHE[so] = None
    return _STRINGS_CACHE[so]


def entry_constraints(op: str, cann: str, window: int = 24) -> dict:
    """从 aclnn 库里抽该算子入口校验的维度约束.

    返回 {"messages": [原文], "assume": {符号: 模数}, "min": {符号: 下界}, "libs": [...]}.
    assume 可直接喂给 core.wscap --assume / core.detsync_lint --assume 决定对齐判据的可达性.
    """
    msgs, libs = [], []
    for so in _opapi_libs(cann):
        lines = _strings_lines(so)
        if lines is None:
            continue
        idx = [i for i, s in enumerate(lines) if op in s]
        if not idx:
            continue
        libs.append(so)
        for i in idx:
            for j in range(max(0, i - window), min(len(lines), i + window)):
                s = lines[j]
                if _CONSTRAINT_HINT.search(s) and s not in msgs:
                    msgs.append(s)
    assume, mins = {}, {}
    for s in msgs:
        for rx, kind in _CONSTRAINT_PATTERNS:
            for m in rx.finditer(s):
                sym, val = m.group(1), int(m.group(2))
                if kind == "mod":
                    # 多条约束取**最松**的 (不同模式各有约束, 静态判可达性必须按最松的那条,
                    # 否则会把"某模式下可达"的真缺陷当成不可达而漏报)
                    assume[sym] = val if sym not in assume else __import__("math").gcd(assume[sym], val)
                elif kind == "min":
                    mins[sym] = min(mins.get(sym, val), val)
    return {"messages": msgs, "assume": assume, "min": mins, "libs": libs}


def selftest() -> int:
    """判据负例自检 (纯夹具, 不碰真 CANN): 抽取正确性 + 失败必须显式 + 约束取最松。"""
    import shutil
    import tempfile
    bad = 0

    def _say(ok, msg):
        print(f"  {'过' if ok else '★未过'} {msg}")
        return 0 if ok else 1

    print("=== gen_spec 判据负例自检 (纯夹具) ===")
    with tempfile.TemporaryDirectory() as td:
        soc = "fakesoc"
        node = {"FakeOp": {
            "input0": {"name": "x", "paramType": "required",
                       "dtype": "float16,int8", "format": "ND,FRACTAL_NZ"},
            "output0": {"name": "y", "paramType": "required",
                        "dtype": "float16", "format": "ND"},
            "attr_transpose": {"type": "bool", "defaultValue": "false",
                               "paramType": "optional"},
            "attr": {"list": "transpose"},
        }}
        bdir = os.path.join(td, "opp/built-in/op_impl/ai_core/tbe/config", soc)
        os.makedirs(bdir)
        json.dump(node, open(os.path.join(bdir, f"aic-{soc}-ops-info-fake.json"), "w"))
        #  vendor 包: 文件名**没有** -<class> 后缀 (2026-09-10 SFA 实撞的那个差别)
        vdir = os.path.join(td, "vend/op_impl/ai_core/tbe/config", soc)
        os.makedirs(vdir)
        json.dump({"VendorOnlyOp": {"input0": {"name": "a", "paramType": "required",
                                               "dtype": "int32", "format": "ND"}}},
                  open(os.path.join(vdir, f"aic-{soc}-ops-info.json"), "w"))

        s = extract_spec("FakeOp", soc, td)
        bad += _say(s["inputs"][0]["dtype"] == ["float16", "int8"]
                    and s["attrs"]["transpose"]["default"] == "false"
                    and s["attr_order"] == "transpose",
                    "抽取正确性: dtype 列表 / attr 默认值 / attr_order")
        bad += _say(extract_spec("fakeop", soc, td)["op"] == "fakeop",
                    "算子名大小写不敏感查找")
        try:
            extract_spec("NoSuchOp", soc, td)
            bad += _say(False, "未找到的算子必须显式报错 (实得: 静默返回)")
        except SystemExit:
            bad += _say(True, "未找到的算子显式报错, 不许静默空 spec")
        bad += _say(extract_spec("VendorOnlyOp", soc, td,
                                 vendor=os.path.join(td, "vend"))["inputs"][0]["dtype"] == ["int32"],
                    "vendor 包文件名无 -<class> 后缀也能找到 (2026-09-10 SFA 形态)")

        if shutil.which("strings"):
            lib = os.path.join(td, "fakever", "lib64")
            os.makedirs(lib)
            open(os.path.join(lib, "libopapi_fake.so"), "w").write(
                "FakeOpEntryCheck\nn mod 32 equal 0\nn mod 8 equal 0\n"
                "k bigger or equal 2\nunrelated line\n")
            r = entry_constraints("FakeOp", td)
            bad += _say(r["assume"].get("n") == 8 and r["min"].get("k") == 2 and r["libs"],
                        f"入口约束抽取且 mod 取最松 (gcd(32,8)=8), 实得 assume={r['assume']} min={r['min']}")
            r2 = entry_constraints("NothingHere", td)
            bad += _say(not r2["libs"] and not r2["assume"],
                        "查不到的算子不许产出约束 (查不到与无约束要分得开)")
        else:
            print("  (跳过 入口约束 2 条: 本机无 strings 命令)")
    print(f"gen_spec 自检: {bad} 条未过")
    return 1 if bad else 0


def main() -> None:
    import sys
    if "--selftest" in sys.argv[1:]:
        sys.exit(selftest())
    ap = argparse.ArgumentParser()
    ap.add_argument("op")
    ap.add_argument("--soc", default=None, help="CANN 目录系列名; 不给按本机探测 (socspec.config_dir_name)")
    ap.add_argument("--cann", default=os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.1.0-beta.1"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--vendor", default=None,
                    help="从**自编 vendor 包**里抽 (装好的 vendors/<name> 目录). "
                         "本机 CANN 比源码仓旧时, 新算子只在 vendor 包的 ops-info 里")
    ap.add_argument("--entry-constraints", action="store_true",
                    help="抽 aclnn 入口校验的维度约束 (ops-info JSON 里没有, 只在 libopapi*.so 的校验消息里). "
                         "--assume-only 则只输出可直接喂给 lint 的契约串")
    ap.add_argument("--assume-only", action="store_true")
    a = ap.parse_args()
    if not getattr(a, "soc", None):      # 目录系列名按本机推, 不写死某颗芯片
        from core import socspec
        a.soc = socspec.config_dir_name()


    if a.entry_constraints or a.assume_only:
        r = entry_constraints(a.op, a.cann)
        if a.assume_only:
            print(",".join(f"{k}%{v}==0" for k, v in sorted(r["assume"].items())))
            return
        if not r["libs"]:
            print(f"未在 {a.cann} 的 libopapi*.so 里找到算子名 {a.op}"); return
        print(f"来源: {', '.join(os.path.basename(x) for x in r['libs'])}")
        print(f"\n机读契约 (喂 lint):  --assume '" + ",".join(f"{k}%{v}==0" for k, v in sorted(r["assume"].items())) + "'")
        if r["min"]:
            print("下界: " + ", ".join(f"{k}>={v}" for k, v in sorted(r["min"].items())))
        print(f"\n原文 {len(r['messages'])} 条:")
        for m in r["messages"]:
            print("  " + m[:150])
        return

    spec = {"spec": extract_spec(a.op, a.soc, a.cann, a.vendor)}
    if yaml is not None:
        text = yaml.safe_dump(spec, allow_unicode=True, sort_keys=False)
    else:
        text = json.dumps(spec, indent=2, ensure_ascii=False)

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(text)
        print(f"wrote {a.out}")
    else:
        print(text)


if __name__ == "__main__":
    #  ★必须把 main() 的返回码交出去。此前写成裸 `main()`, 于是
    #    `python -m core.<mod> --selftest` **无论自检过不过, 退出码恒为 0** ——
    #    而 opdbg 的判据普查正是按子进程退出码判 PASS 的, 于是真实的
    #    "N 条未过" 被报成 PASS。2026-10-28 在 detsync_lint 上实撞并全仓同改。
    _rc = main()
    sys.exit(_rc if isinstance(_rc, int) else 0)
