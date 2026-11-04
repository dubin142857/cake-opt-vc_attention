"""socspec — 芯片规格**唯一取数口** (算子无关): 按真机 SoC 精确型号读平台 ini, 并审计各模块的硬编码常数.

为什么需要它 (2026-08-28, 从 block_sparse_attention 上 950 反向沉淀):

本仓的判据常数分散写在各模块里, 且全部标定于 910B/2201:
  ub_budget.UB_CAP = {"ascend910b": 192KB}      # 取不到时 .get(soc, 192*1024) **静默**回落
  mmplan.CAP        = {"ascend910b": {...}}      # L0C 128KB / 24 核 / 1.8GHz
  pipeplan.MACHINE  = {"ascend910b2": {...}}
  sortplan._SOC_INI = {"ascend950pr": "Ascend950PR_9589.ini"}   # 型号写死

★2026-10-20 核: 上面这四行是**写于 2026-08-28 的历史记述, 不是当前清单** —— `sortplan._SOC_INI`
此后已被删除, 现在不存在。当前的活清单有两张, 都机读, 引用时取它们而不是取这段注释:
  · `AUDIT` (本文件 ~600 行): 模块硬编码常数 -> ini 字段的逐条映射, `--audit` 自动覆盖;
  · `--audit` 末尾的"代码里残留的芯片名字面量"扫描: 现报 mmplan / pipeplan / socspec / ub_budget。
一张只写在注释里的清单不会告诉你它已经过期 —— 这一条本身就是它的实例。

搬到 950 上这三类全部失效, 且失效方式不同:
  1. **静默给错数**: ub_budget 在 950 机器上按 192KB 判 UB, 真值 248KB —— 会把可行方案判成溢出.
  2. **KeyError**: mmplan 直接 CAP[soc] 崩, 至少是响的.
  3. **型号选错**: 本机 aclrtGetSocName() = Ascend950PR_957b, 而 sortplan 写死 9589.
     两者差 11 项, 且差的正是编排判据要用的: cube 核 32->28, vector 核 64->56,
     L2 128MB->112MB, ddr_rate 31->27. 按 9589 算出的核利用率与关键路径整体偏乐观.

第 3 条是本模块的直接起因: **芯片型号必须从运行时取, 不能从 npu-smi 的 Chip Name 猜** ——
`npu-smi info` 只给 "Ascend950PR", 后缀 (9589 / 957b / 9599 ...) 是不同 SKU, 核数与 L2 都不同;
`npu-smi info -t board` 的 "NPU Name" 才给 957b, 而 `aclrtGetSocName()` 直接给全名.

定位: 与 _refs 配套 —— _refs 说"这条判据的实证是什么", socspec 说"这台机器的硬参数是什么",
两者合起来才能判"这条判据在**这台**机器上还成不成立".

用法:
    python -m core.socspec                      # 本机: 型号 + 规格 + 范式 + 派生峰值
    python -m core.socspec --compare ascend910b2 ascend950pr_957b   # 换芯片时先看这张差异表
    python -m core.socspec --audit              # 审计各模块硬编码常数 vs 本机 ini
    python -m core.socspec --refs               # 本模块判据出处

其它模块取硬参数一律走 `socspec.spec()`, 不要新开硬编码表.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import re
import subprocess
import sys

from core._refs import Ref, print_refs

ASCEND = os.environ.get("ASCEND_HOME_PATH") or "/usr/local/Ascend/ascend-toolkit/latest"

REFS = {
    "soc_name": Ref(
        what="本机 SoC 精确型号 (决定选哪个平台 ini)",
        status="measured",
        evidence="aclrtGetSocName() @ libascendcl.so 实测返回 Ascend950PR_957b; "
                 "npu-smi info -t board -i 2 的 NPU Name 字段同为 957b. "
                 "★npu-smi info 的 Chip Name 只到 Ascend950PR, 按它猜后缀会选错 SKU",
        verified="2026-08-28"),
    "capacity/cores/freq": Ref(
        what="L1/L0A/L0B/L0C/UB/L2 容量, cube 与 vector 核数, cube 频率",
        status="ini_exact",
        evidence="逐字段取自平台 ini 的 [AICoreSpec]/[SoCInfo], 本模块不改写不折算",
        verified="2026-08-28"),
    "mac_per_cycle": Ref(
        what="每 cube 核每周期 MAC 数 (按 dtype)",
        status="derived",
        evidence="= ini [DtypeMKN] 的 m*n*k. 回归核对: 910B2 DT_INT8=16,32,16 -> 8192, "
                 "24 核 x 8192 x 2 x 1.8GHz = 707.8 TOPS, 与 pipeplan REFS 记的 int8 峰值一致",
        verified="2026-08-28"),
    "dram_bw_model": Ref(
        what="DRAM 带宽 = cube 核数 x ddr_read_rate x 频率",
        status="derived",
        evidence="ini 的 ddr_read_rate 是**每核每周期字节数**的编译器成本模型参数, 不是实测可达带宽. "
                 "910B2 上按此式得 1.38TB/s 而 mmplan 记的实测饱和值是 1.07TB/s (-23%). "
                 "★只可用于排序与上界, 收益一律实测",
        verified="2026-08-28"),
    "paradigm": Ref(
        what="regbase / SIMT 两种编程范式在本机是否可用",
        status="measured",
        evidence="regbase: CANN 树内 asc/impl/adv_api/detail/**/regbase/<arch> 目录按 NpuArch 分档, "
                 "3510 档存在; SIMT: asc/include/simt_api/ 存在且 asc_simt.h 按 __NPU_ARCH__ 分档. "
                 "两者都是**目录存在性**证据, 不等于该算子已经在用",
        verified="2026-08-28"),
}


# ---------------------------------------------------------------- 型号探测

def _platform_dir() -> str:
    for base in (ASCEND, os.path.realpath(ASCEND)):
        for sub in ("x86_64-linux", "aarch64-linux", "."):
            p = os.path.join(base, sub, "data", "platform_config")
            if os.path.isdir(p):
                return p
    raise SystemExit(f"找不到 platform_config 目录 (从 {ASCEND} 起找); 先 source set_env.sh")


def detect_soc() -> tuple[str, str]:
    """返回 (soc_name, 来源). **不给默认值** —— 猜错型号比报错更贵 (见模块 docstring 第 3 条)."""
    try:
        cl = ctypes.CDLL("libascendcl.so")
        cl.aclrtGetSocName.restype = ctypes.c_char_p
        name = cl.aclrtGetSocName()
        if name:
            return name.decode(), "aclrtGetSocName()"
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run(["npu-smi", "info", "-t", "board", "-i", "0"],
                             capture_output=True, text=True, timeout=15).stdout
        chip = re.search(r"Chip Name\s*:\s*(\S+)", out)
        sku = re.search(r"NPU Name\s*:\s*(\S+)", out)
        if chip and sku:
            return f"{chip.group(1)}_{sku.group(1)}", "npu-smi -t board (Chip Name + NPU Name)"
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit("探测不到 SoC 型号 (aclrtGetSocName 与 npu-smi -t board 都没给出); "
                     "用 --soc <型号> 显式指定, 别让下游按默认值算")


def ini_path(soc: str) -> str:
    """型号 -> 平台 ini。大小写不敏感; 前缀模糊命中多个 SKU 时**显式警告**再取最短的。

    ★这里不能悄悄取一个 —— 同系列不同 SKU 的核数/L2/ddr_rate 都不同 (950PR 的 9589 与 957b
      差 11 项), 取错等于后面所有编排判据整体偏乎。
    """
    d = _platform_dir()
    files = [f for f in os.listdir(d) if f.endswith(".ini")]
    low = soc.lower()
    exact = [f for f in files if os.path.splitext(f)[0].lower() == low]
    if exact:
        return os.path.join(d, exact[0])
    hits = sorted((f for f in files if low in os.path.splitext(f)[0].lower()), key=lambda f: (len(f), f))
    if not hits:
        avail = ", ".join(sorted(os.path.splitext(f)[0] for f in files)[:8])
        raise SystemExit(f"{d} 里找不到 soc={soc!r} 的平台 ini. 前几个可用型号: {avail} ...")
    if len(hits) > 1:
        print(f"[warn] soc={soc!r} 前缀命中 {len(hits)} 个 SKU ({', '.join(h[:-4] for h in hits)}), "
              f"取 {hits[0][:-4]}; **不同 SKU 核数/L2/ddr_rate 不同**, 判据要准就给全名 "
              f"(本机全名: 跑 python -m core.socspec 看第一行)", file=sys.stderr)
    return os.path.join(d, hits[0])


def parse_ini(path: str) -> dict:
    sec, out = None, {}
    with open(path, errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if ln.startswith("[") and ln.endswith("]"):
                sec = ln[1:-1]
                continue
            if "=" in ln and sec:
                k, v = ln.split("=", 1)
                out.setdefault(sec, {}).setdefault(k.strip(), v.strip())
    return out


# ---------------------------------------------------------------- 规格

def _i(d: dict, sec: str, key: str) -> int | None:
    v = d.get(sec, {}).get(key)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


#  NpuArch 数字 -> 编译器码名。两套命名哪一套有效随安装包而变, 所以两条都要试。
#  取值来自各 platform ini 的 CCEC_AIC_version (如 dav-c310-cube -> c310),
#  2026-10-28 在本机 CANN 9.0.0 的 63 个 ini 上核过。
_CCEC_CODE_BY_ARCH = {
    "3510": "c310", "2201": "c220", "1001": "c100",
    "2002": "m200", "3003": "m300", "3113": "m310",
}


def tool_path(relpath: str, legacy: str = "") -> str:
    """CANN 包里某个工具的绝对路径。**探而不是写死版本号。**

    2026-10-28 实撞: core/aprof.py 与 core/detrace.py 各自写死了
    `/usr/local/Ascend/cann-9.1.0-beta.1/...`, 而本机装的是 9.0.0 ——
    aprof 的自检因此 FileNotFoundError, 并被当成"长期未过"登记了一天多。
    写死版本号与 CLAUDE.md 第 2 条 (机读优先, 不手抄) 是同一类违反。

    顺序: ASCEND_HOME_PATH -> ascend-toolkit/latest -> /usr/local/Ascend/cann-* (新到旧)。
    ★返回**存在**的那个; 全都不存在时返回按首选根拼出的路径 (让调用方自己报错),
      而不是返回一个空串让错误以别的形态冒出来。
    """
    roots = []
    env = os.environ.get("ASCEND_HOME_PATH")
    if env:
        roots.append(env)
    roots.append("/usr/local/Ascend/ascend-toolkit/latest")
    roots.extend(sorted(glob.glob("/usr/local/Ascend/cann-*"), reverse=True))
    if legacy:
        roots.append(legacy)
    seen = set()
    first = ""
    for r in roots:
        r = os.path.realpath(r)
        if r in seen:
            continue
        seen.add(r)
        cand = os.path.join(r, relpath)
        if not first:
            first = cand
        if os.path.exists(cand):
            return cand
    return first


def paradigms(arch: str) -> dict:
    """本机可用的编程范式. 证据是 CANN 树里的**按 arch 分档目录**, 不是推测."""
    root = None
    for sub in ("x86_64-linux", "aarch64-linux", "."):
        p = os.path.join(ASCEND, sub, "asc")
        if os.path.isdir(p):
            root = p
            break
    if root is None:
        return {"regbase": None, "simt": None, "evidence": "找不到 asc/ 目录, 无法取证"}
    #  ★arch 分档目录有**两套命名**, 哪一套有效随安装包而变 (见 opgen.api 的
    #    API5_impl_arch_dir_has_two_naming_schemes): 一套是 NpuArch 数字 (3510),
    #    一套是编译器码名 (c310 / c220 / m200 ...)。只找数字那套, 在本机 CANN 9.0.0 上
    #    regbase/3510 命中 0 处而 regbase/c310 有 17 处 —— 于是 950 家族的
    #    regbase_simd_vf 被探成 False, 而它实际存在。2026-10-28 实撞并改。
    #    ★两条都要试, 且证据里要写**命中的是哪一条**, 不能只报个布尔值。
    cands, hit_name = [str(arch)], None
    code = _CCEC_CODE_BY_ARCH.get(str(arch))
    if code:
        cands.append(code)
    regbase = []
    for nm in cands:
        regbase = glob.glob(os.path.join(root, "impl", "adv_api", "detail", "**", "regbase", nm),
                            recursive=True)
        if regbase:
            hit_name = nm
            break
    simt = os.path.isdir(os.path.join(root, "include", "simt_api"))
    simt_arch = False
    hdr = os.path.join(root, "include", "simt_api", "asc_simt.h")
    if os.path.exists(hdr):
        with open(hdr, errors="ignore") as f:
            simt_arch = f"__NPU_ARCH__ == {arch}" in f.read().replace("==", " == ")
    return {
        "regbase": bool(regbase),
        "regbase_dirs": len(regbase),
        "simt": simt,
        "simt_arch_guarded": simt_arch,
        "regbase_dir_name": hit_name,
        "evidence": (f"regbase: adv_api/detail/**/regbase/{hit_name} 命中 {len(regbase)} 处"
                     if hit_name else
                     f"regbase: 试过 {', '.join(cands)}, 都未命中")
                    + f"; simt_api/ {'在' if simt else '缺'}",
    }


def default_soc() -> str:
    """本机型号。**所有模块的 --soc 默认值都应该走它**, 不许在代码里写某颗芯片的名字。

    探测不到就抛 —— 回落到某颗具体芯片等于用别人的硬件参数判本机 (本仓明令禁止的静默回落)。
    """
    soc, src = detect_soc()
    if not soc:
        raise RuntimeError(
            "探测不到本机 SoC (aclrtGetSocName)。不回落到任何具体型号 —— "
            "用别的芯片的硬件参数判本机会制造假安全。请显式传 --soc。")
    return soc


#  CANN 的 ops-info / kernel config 目录用的是**系列名**而不是精确 SKU
#  (Ascend950PR_957b -> ascend950, Ascend910B2 -> ascend910b), 各模块原来各写死一份。
_SERIES_RE = re.compile(r"^(ascend[0-9]{3}[a-z]*)", re.I)


def config_dir_name(soc: str | None = None) -> str:
    """把精确 SKU 映射成 CANN 目录用的系列名。写死 'ascend950'/'ascend910b' 的地方都该用它。"""
    full = (soc or default_soc()).lower()
    m = _SERIES_RE.match(full)
    if not m:
        raise ValueError(f"认不出系列名: {soc!r}")
    name = m.group(1)
    #  950PR / 950B... 的目录名只到 ascend950; 910B1/B2/B3 的目录名是 ascend910b
    if name.startswith("ascend950"):
        return "ascend950"
    return name


def spec(soc: str | None = None) -> dict:
    """本机 (或指定型号) 的归一化规格. 下游取硬参数只走这里."""
    src = "--soc 指定"
    if soc is None:
        soc, src = detect_soc()
    path = ini_path(soc)
    d = parse_ini(path)
    arch = d.get("version", {}).get("NpuArch", "?")
    mkn = d.get("DtypeMKN", {})

    def mac(dtype: str) -> int | None:
        v = mkn.get(dtype) or mkn.get("Default")
        try:
            m, n, k = (int(x) for x in v.split(","))
            return m * n * k
        except Exception:  # noqa: BLE001
            return None

    freq_ghz = (_i(d, "AICoreSpec", "cube_freq") or 0) / 1000.0
    cube_cores = _i(d, "SoCInfo", "cube_core_cnt")
    ddr_rate = _i(d, "AICoreMemoryRates", "ddr_read_rate")
    l2_rate = _i(d, "AICoreMemoryRates", "l2_read_rate")
    s = {
        "soc": d.get("version", {}).get("SoC_version", soc),
        "soc_source": src,
        "ini": path,
        "arch": arch,
        "aic_version": d.get("version", {}).get("AIC_version"),
        "cube_cores": cube_cores,
        "vector_cores": _i(d, "SoCInfo", "vector_core_cnt"),
        "freq_ghz": freq_ghz,
        "cap": {
            "L1": _i(d, "AICoreSpec", "l1_size"),
            "L0A": _i(d, "AICoreSpec", "l0_a_size"),
            "L0B": _i(d, "AICoreSpec", "l0_b_size"),
            "L0C": _i(d, "AICoreSpec", "l0_c_size"),
            "UB": _i(d, "AICoreSpec", "ub_size"),
            "BT": _i(d, "AICoreSpec", "bt_size"),
            "L2": _i(d, "SoCInfo", "l2_size"),
            "HBM": _i(d, "SoCInfo", "memory_size"),
        },
        "mac_per_cycle": {"fp16": mac("Default"), "int8": mac("DT_INT8"),
                          "fp32": mac("DT_FLOAT"), "int4": mac("DT_INT4")},
        "rate_B_per_cycle": {k: _i(d, "AICoreMemoryRates", k)
                             for k in sorted(d.get("AICoreMemoryRates", {}))},
        "paradigm": paradigms(arch),
        "regfile": {k: _i(d, "AICoreSpec", k)
                    for k in ("vector_reg_width", "wide_reg_width", "predicate_reg_width")
                    if _i(d, "AICoreSpec", k)},
        #  向量核每周期的计算字节数 ([VectorCoreSpec] vec_calc_size)。没有它就算不出
        #  向量侧的算力天花板, 于是"这个算子慢是不是因为算力没用上"这个问题只能靠猜。
        #  2026-09-21 SFA 实撞: 缺这个数时我先后给出了两个相反的结论, 都没有算术支撑。
        "vec_calc_B_per_cycle": _i(d, "VectorCoreSpec", "vec_calc_size")
                                or _i(d, "AICoreSpec", "vec_calc_size"),
    }
    if cube_cores and freq_ghz:
        m16 = s["mac_per_cycle"]["fp16"] or 0
        s["derived"] = {
            "peak_fp16_TFLOPS": cube_cores * m16 * 2 * freq_ghz / 1000.0,
            # ★模型值不是实测可达值, 见 REFS["dram_bw_model"]
            "dram_model_TBps": (cube_cores * ddr_rate * freq_ghz / 1000.0) if ddr_rate else None,
            "l2_model_TBps": (cube_cores * l2_rate * freq_ghz / 1000.0) if l2_rate else None,
            #  ini 的各级速率是**每核每周期字节数**, 这里换成每核 GB/s, 免得每个模块各乘一遍频率。
            #  ★它们是编译器成本模型参数, 当上限用可以, 当可达带宽用不行 (见 REFS.dram_bw_model)。
            "rate_GBps_per_core": {k: (v * freq_ghz) for k, v in s["rate_B_per_cycle"].items() if v},
        }
        #  ★两个执行单元的算力天花板必须并排给出。只报 cube 峰值会让人以为算力充裕,
        #    而算术若跑在向量单元上, 真正的上限是下面这个小两个数量级的数。
        vcalc = s.get("vec_calc_B_per_cycle")
        vcores = s.get("vector_cores")
        if vcalc and vcores:
            for w, nm in ((4, "fp32"), (2, "fp16")):
                s["derived"][f"peak_vector_{nm}_TMACs"] = (
                    vcores * (vcalc / w) * freq_ghz / 1000.0)
            if m16:
                s["derived"]["peak_cube_fp16_TMACs"] = cube_cores * m16 * freq_ghz / 1000.0
                s["derived"]["cube_over_vector_fp16"] = (
                    s["derived"]["peak_cube_fp16_TMACs"]
                    / s["derived"]["peak_vector_fp16_TMACs"])
    return s


# ---------------------------------------------------------------- 实测标定 (ini 里没有的)

#  ★分两类, 不要混:
#    - **ini 事实** (容量/核数/频率/各级速率/MAC 每周期): 上面 spec() 全从本机 ini 取, 换芯片自动跟着变。
#    - **实测标定** (可达带宽、有效率、有没有 cube->vec 直通): ini 里没有, 只能按芯片实测, 记在这里。
#  两类的失效方式不同: ini 事实错 = 读错文件; 实测标定错 = **拿了别的芯片的数**。
#  后者是本仓反复强调的静默假阴性来源, 所以这里**按 soc 分开存**, 取不到就返回 None,
#  由调用方显式拒判 —— 绝不回落到另一颗芯片。
#
#  每项写清 value / status / evidence / verified:
#    status: measured(饱和实测) | observed(某次负载的观测值, 是下界不是上限) | inferred(由证据推断)
CALIB_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chip_calib.json")


#  msopprof 的"理论带宽"表 —— `*_bw_usage_rate(%)` 的分母就是它 (docs/msprof-re/08 §3.44)。
#  **按模式搜而不是写死偏移**: 偏移随 CANN 版本变, 写死等于下个版本静默读到垃圾。
_MSOPPROF_BIN = "/usr/local/Ascend/{ver}/tools/msopprof/bin/msopprof"
#  前 6 项的含义: 与 Memory.csv 里 `*_bw_usage_rate(%)` 的**列序**一一对应 (实测 5/6 精确吻合)。
BW_TABLE_COLS = ["GM_to_L1", "L1_to_GM", "L0C_to_L1", "L0C_to_GM", "GM_to_UB", "UB_to_GM"]


def msopprof_bw_table(path: str | None = None) -> dict:
    """从 msopprof 二进制里抽"理论带宽"表。返回分组后的 {key: GiB/s}。

    **表的真实结构 (08 §3.46 实测)**: 连续的 `{uint32 key, float value}`, 每条 8 字节。
    key 是 `Failed to get max bandwidth of type : %d` 里的那个 type id。
    本机命中 76 条, 按"key 序列重复即新组"切成 8 组 (6 组 9 条 + 2 组 11 条),
    各组 key 序列完全相同 —— 一组对应一个 SoC (或 SoC 族)。

    ★**首版把它读成"步长 8 的 float, 中间夹零"**: 位置找对了, 理由是错的 ——
    那些"零"是 key 字段。把结构体读成单一字段的数组, 会得到一个自洽但错误的布局,
    而且它不会自己露馅 (08 §3.46)。现在按 {u32,f32} 解, 并用 key 序列切组。

    证据分级 (别混):
      1. 这些值是**程序里的真实字面量** —— 确证 (float32 逐位, 且与本仓五个独立
         反解出来的分母吻合)。
      2. **这张表就是 msopprof 实际读的那张** —— 仍**未证实**: 找不到引用它的代码
         (运行时建 std::map, 见 08 §3.46), 只有模式证据。
      3. key 11 = L0C_to_L1 是**按组内消去法推出的**, 本仓无非零样本可验。
    """
    if path is None:
        cands = sorted(glob.glob("/usr/local/Ascend/*/tools/msopprof/bin/msopprof"))
        if not cands:
            return {"error": "找不到 msopprof 二进制", "groups": []}
        path = cands[-1]
    try:
        d = open(path, "rb").read()
    except OSError as e:
        return {"error": f"读不了 {path}: {e}", "groups": []}
    import struct as _st

    best = None
    for off in range(0, len(d) - 8 * 6, 4):
        ent, o = [], off
        while o + 8 <= len(d):
            k, = _st.unpack_from("<I", d, o)
            v, = _st.unpack_from("<f", d, o + 4)
            if not (0 < k < 200 and 1.0 < v < 1e4):
                break
            ent.append((k, round(v, 5))); o += 8
        if len(ent) >= 18 and (best is None or len(ent) > len(best[1])):
            best = (off, ent)
    if best is None:
        return {"error": "没找到符合 {u32 key, f32 value} 结构的表", "path": path,
                "groups": []}
    off, ent = best
    groups, cur, seen = [], [], set()
    for k, v in ent:
        if k in seen:
            groups.append(cur); cur, seen = [], set()
        cur.append((k, v)); seen.add(k)
    if cur:
        groups.append(cur)
    return {"path": path, "offset": off, "n": len(ent), "groups": groups,
            "cols": {BW_TABLE_COLS[i]: v for i, (k, v) in enumerate(groups[0][:6])}
            if groups else {}}


def print_bw_table() -> int:
    r = msopprof_bw_table()
    if r.get("error"):
        print(f"✗ {r['error']}")
        return 1
    g = r["groups"]
    print(f"=== msopprof 理论带宽表 @ {r['path']} ===")
    print(f"    起点 0x{r['offset']:06x}, 共 {r['n']} 条 {{u32 key, f32 value}}, "
          f"切成 {len(g)} 组 (各组条数 {[len(x) for x in g]})")
    seqs = {}
    for x in g:
        seqs.setdefault(tuple(k for k, _ in x), 0)
        seqs[tuple(k for k, _ in x)] += 1
    print(f"    key 序列共 {len(seqs)} 种: " +
          "; ".join(f"{list(kk)} ×{n} 组" for kk, n in
                    sorted(seqs.items(), key=lambda kv: -kv[1])) + "\n")
    chan = {2: "GM_to_L1", 6: "L1_to_GM", 11: "L0C_to_L1", 12: "L0C_to_GM",
            3: "GM_to_UB", 15: "UB_to_GM"}
    print(f"{'key':>5}{'GiB/s':>10}{'GB/s':>10}   通路 / 证据")
    for k, v in g[0]:
        c = chan.get(k)
        note = ("★**推测** —— 组内消去法, 本仓无非零样本可验" if k == 11
                else ("与本仓实测分母逐位吻合" if c else "通路未知"))
        print(f"{k:>5}{v:>10.2f}{v*2**30/1e9:>10.2f}   {c or '?':<12} {note}")
    print("\n  ★**这张表是不是 msopprof 实际读的那张, 仍未证实** (08 §3.46):")
    print("    查表在运行时建成 std::map (soc 名 -> 索引, 再 (索引,type) -> 带宽),")
    print("    找不到引用这段 .rodata 的代码。已确证的是**这些值是程序里的真实字面量**。")
    return 0


def selftest_bw_table() -> int:
    """定位逻辑的自检: 用合成二进制, 不依赖真 msopprof。"""
    import struct as _st
    import tempfile
    bad = 0
    print("=== socspec msopprof 带宽表定位 自检 (4 条) ===")
    with tempfile.TemporaryDirectory() as t:
        keys = [2, 6, 11, 12, 3, 15, 25, 26, 44]
        vals = [264.0, 199.43, 216.88, 209.32, 220.06, 186.8, 437.5, 210.5, 340.1]
        buf = bytearray(b"\x11" * 64)
        for _grp in range(2):                     # 两组, 验分组逻辑
            for k, v in zip(keys, vals):
                buf += _st.pack("<I", k) + _st.pack("<f", v)
        buf += b"\x00" * 64
        p = os.path.join(t, "fake")
        open(p, "wb").write(bytes(buf))
        r = msopprof_bw_table(p)
        ok = (not r.get("error") and r.get("n") == 18)
        bad += 0 if ok else 1
        print(f"  {'过' if ok else '**未过**'} 负例: 合成表必须被定位到且条数对  "
              f"(n={r.get('n')})")
        ok2 = ([len(x) for x in r.get("groups", [])] == [9, 9])
        bad += 0 if ok2 else 1
        print(f"  {'过' if ok2 else '**未过**'} 负例: 按 key 重复切组 -> 必须切成 9+9  "
              f"({[len(x) for x in r.get('groups', [])]})")
        ok3 = (round(r.get("cols", {}).get("L0C_to_L1") or 0, 2) == 216.88)
        bad += 0 if ok3 else 1
        print(f"  {'过' if ok3 else '**未过**'} 负例: 第 3 条 (key 11) 映射到 L0C_to_L1  "
              f"({r.get('cols', {}).get('L0C_to_L1')})")
        p2 = os.path.join(t, "empty")
        open(p2, "wb").write(b"\x00" * 8192)
        r2 = msopprof_bw_table(p2)
        ok4 = bool(r2.get("error"))
        bad += 0 if ok4 else 1
        print(f"  {'过' if ok4 else '**未过**'} 正例: 没有表的文件必须报错, 不许硬凑  "
              f"({r2.get('error') or r2.get('cols')})")
    print(f"socspec 带宽表自检: 4 条, {bad} 条未过")
    return 1 if bad else 0


def _load_calib() -> dict:
    """按芯片的实测标定从 **JSON 数据文件** 读, 不写在代码里 —— 加一颗芯片只改数据。"""
    try:
        with open(CALIB_JSON, encoding="utf-8") as f:
            return json.load(f).get("chips", {})
    except Exception as e:                      # 数据文件坏了要说话, 不能当"这颗芯片没标定"
        print(f"  ★读不到 {CALIB_JSON}: {e} —— 所有实测标定项将按『无标定』处理", file=sys.stderr)
        return {}


CALIB = _load_calib()


def calib(soc: str | None = None) -> tuple:
    """返回 (本芯片的实测标定 dict, 出处)。**不回落到别的芯片**。

    取不到的项由调用方显式拒判 —— 用另一颗芯片的实测值算出一个"通过", 比报错危险得多。
    """
    if soc is None:
        soc, _ = detect_soc()
    key = (soc or "").lower()
    if key in CALIB:
        return CALIB[key], f"CALIB/{key}"
    return {}, f"无标定/{key}"


def calib_value(name: str, soc: str | None = None):
    """取一项实测标定的值; 没有该芯片的标定或该项标成 uncalibrated 时返回 None。"""
    c, _ = calib(soc)
    e = c.get(name)
    return e["value"] if e else None


def calib_check(name: str, changed, soc: str | None = None) -> tuple:
    """取一项标定, **但先核对本次改动有没有动到它标定时固定的量**。

    返回 ``(value, blockers)``。blockers 非空时 value 恒为 None —— 调用方必须拒判,
    不许"取不到就用 0 顶上"或退回 calib_value。

    ``changed`` 是本次改动改变了哪些量, 一串自由词 (如 ["分核", "流量"]);
    与标定项 ``held_fixed[*].keys`` 做子串双向匹配。

    为什么拒判而不是提示: 标定值自带测量条件, 本仓已**三次**误用 ——
      1. dram_GBps 按负载类型错用;
      2. dram_GBps 按工作集错用 (1690 GB/s 实为 L2 全命中下的观测, 据此算出"已达 96.4%",
         把方向定成"只能减流量", 实际空间是 36%);
      3. l2_working_set_curve 按调度结构错用 (2026-09-15 sfag v8a: 二维分核把瞬时工作集
         144MB 压到 7.1MB, 照曲线 18MB 档预测约 13ms, **实测 44.59ms** —— 不是偏差, 是
         方向相反; 因为它压工作集的同时把行切换次数乘了 8, 而那正是曲线 held_fixed 的量)。
    三次都是"看了值、没看条件", 提示型判据拦不住, 只有取不到值才拦得住。

    未声明 held_fixed 的标定项同样拒判: 没记测量条件的值不许拿来做预测。
    """
    c, _ = calib(soc)
    e = c.get(name)
    if not e:
        return None, [f"无 {soc or '本芯片'} 的 {name} 标定"]
    if e.get("status") == "uncalibrated":
        return None, [f"{name} 已降级为 uncalibrated: {e.get('evidence', '')[:120]}"]
    held = e.get("held_fixed")
    if not held:
        return None, [f"{name} 未声明 held_fixed —— 补齐它标定时固定了什么, 再用它做预测"]

    terms = [t for t in (changed or []) if str(t).strip()]
    blockers = []
    for h in held:
        keys = h.get("keys", []) if isinstance(h, dict) else []
        note = h.get("note", "") if isinstance(h, dict) else str(h)
        for t in terms:
            t = str(t).strip().lower()
            if any(t in str(k).lower() or str(k).lower() in t for k in keys):
                blockers.append(f"本次改动动了『{t}』, 而 {name} 标定时把它固定为: {note}")
                break
    return (None, blockers) if blockers else (e["value"], [])


def selftest_calib_check() -> int:
    """负例自检: 三次真实误用都必须被拦, 而正当引用必须放行。"""
    soc = "ascend910b2"
    cases = [
        #  (标定项, 本次改动改了什么, 期望拦?, 说明)
        ("l2_working_set_curve", ["分核"],        True,
         "v8a 二维分核 —— 实撞: 预测 13ms 实测 44.59ms"),
        ("l2_working_set_curve", ["行切换"],      True,  "同一件事换个说法也要拦"),
        ("l2_working_set_curve", ["核数"],        True,  "减核扫描同样动了调度结构"),
        ("l2_working_set_curve", ["流量"],        True,  "曲线标定时逻辑流量恒定"),
        ("l2_working_set_curve", ["L0C 复用"],    False, "只动 L0C 不动 held_fixed, 放行"),
        #  ★2026-09-22 更新期望 (原为 True/"已降级 uncalibrated, 一律拒判"):
        #    910b2 的 dram_GBps 已于 2026-09-18 重新饱和标定 (status=measured, 四组
        #    held_fixed: 负载/访问形态/粒度/核数), 其中没有"L0C 复用", 所以放行是**对的**。
        #    这条期望自那次标定起就过期了, 而它一直没被发现 —— 因为 core/gatestate 是从
        #    中文输出里正则提取"★未过", 而 socspec 自检打的是"11 条, 1 条不符", 一次都
        #    匹配不到 (codex 评审第一条, 已修)。**它是那条缺陷的第一个真实受害者。**
        ("dram_GBps",            ["L0C 复用"],    False,
         "910b2 的 dram_GBps 已重新实测且 held_fixed 里没有 L0C 复用 -> 放行"),
        ("crosscore_barrier_drain_us", ["分核"],  True,  "汇合开销随参与核数变"),
        ("crosscore_barrier_drain_us", ["量化"],  False, "量化模式与汇合无关, 放行"),
        ("fixpipe_fixed_ns",     ["量化"],        True,  "Fixpipe 固定开销标定时固定了量化模式"),
        ("cv_direct_path",       ["分核"],        True,  "未声明 held_fixed -> 拒判"),
        ("不存在的标定项",         [],             True,  "无此项 -> 拒判"),
    ]
    #  ★"已降级 uncalibrated -> 拒判"这条分支现在**没有活的见证** (本机 chip_calib 里
    #    一个 uncalibrated 条目都没有)。把上面那条期望改成放行的同时, 必须另外给这条
    #    分支补一个注入式负例, 否则改期望就等于静默删掉一条覆盖。
    inj_bad = 0
    _real_calib = calib
    try:
        globals()["calib"] = lambda soc=None: (
            {"降级项": {"status": "uncalibrated", "value": 999,
                        "evidence": "合成条目: 只用于验证降级分支仍然拒判",
                        "held_fixed": [{"keys": ["无关"], "note": ""}]}}, "注入")
        v, blk = calib_check("降级项", [], soc)
        if not (v is None and blk and "uncalibrated" in blk[0]):
            inj_bad = 1
            print(f"  ✗ 注入negative: status=uncalibrated 竟未拒判 -> ({v}, {blk})")
        else:
            print("  ✓ 注入negative: status=uncalibrated -> 拒判 (期望拦)")
    finally:
        globals()["calib"] = _real_calib

    bad = inj_bad
    for name, changed, want_block, why in cases:
        v, bl = calib_check(name, changed, soc)
        got_block = bool(bl)
        ok = got_block == want_block
        if not ok:
            bad += 1
        if got_block and v is not None:
            print(f"  ✗ {name}: 拦了却还返回了值 —— 契约破了")
            bad += 1
        mark = "✓" if ok else "✗"
        print(f"  {mark} calib_check({name!r}, {changed}) -> "
              f"{'拦' if got_block else '放行'} (期望{'拦' if want_block else '放行'})  # {why}")
        if got_block and not ok:
            for b in bl:
                print(f"      {b}")
    print(f"calib_check 自检: {len(cases)} 条, {bad} 条不符")
    return 1 if bad else 0


# ---------------------------------------------------------------- 审计


#  各模块的硬编码常数 -> 本机 ini 字段. 新增模块在此加一行, --audit 自动覆盖.
#  scale: 模块值 = ini 值 * scale (freq 以 GHz 记而 ini 以 MHz 记, 故 1e-3)
AUDIT = [
    ("ub_budget", "UB_CAP",  ("{soc}",),              "AICoreSpec.ub_size",      1.0),
    ("mmplan",    "CAP",     ("{soc}", "L0A"),        "AICoreSpec.l0_a_size",    1.0),
    ("mmplan",    "CAP",     ("{soc}", "L0B"),        "AICoreSpec.l0_b_size",    1.0),
    ("mmplan",    "CAP",     ("{soc}", "L0C"),        "AICoreSpec.l0_c_size",    1.0),
    ("mmplan",    "CAP",     ("{soc}", "UB"),         "AICoreSpec.ub_size",      1.0),
    ("mmplan",    "CAP",     ("{soc}", "L2"),         "SoCInfo.l2_size",         1.0),
    ("mmplan",    "CAP",     ("{soc}", "cores"),      "SoCInfo.cube_core_cnt",   1.0),
    ("mmplan",    "CAP",     ("{soc}", "ghz"),        "AICoreSpec.cube_freq",    1e-3),
    ("mmplan",    "CAP",     ("{soc}", "mac_per_cycle_int8"), "@mac_int8",       1.0),
    ("pipeplan",  "MACHINE", ("{soc}", "cap", "L0A"), "AICoreSpec.l0_a_size",    1.0),
    ("pipeplan",  "MACHINE", ("{soc}", "cap", "L0B"), "AICoreSpec.l0_b_size",    1.0),
    ("pipeplan",  "MACHINE", ("{soc}", "cap", "L0C"), "AICoreSpec.l0_c_size",    1.0),
    ("pipeplan",  "MACHINE", ("{soc}", "cap", "UB"),  "AICoreSpec.ub_size",      1.0),
    ("pipeplan",  "MACHINE", ("{soc}", "freq_ghz"),   "AICoreSpec.cube_freq",    1e-3),
    ("pipeplan",  "MACHINE", ("{soc}", "rate", "MAC_PER_CY"), "@mac_int8",       1.0),
]


def _dig(obj, path):
    for p in path:
        if not isinstance(obj, dict) or p not in obj:
            return None, p
        obj = obj[p]
    return obj, None


def _probe_ub(soc):
    from core import ub_budget
    v, src = ub_budget.ub_cap_for(soc)
    return f"UB={v}B  ({src})"


def _probe_mm(soc):
    from core import mmplan
    c, src = mmplan.cap_for(soc)
    un = [k for k in ("cv_direct_path", "dram_GBps", "mte2_GBps_per_core") if c.get(k) is None]
    return (f"L1={c['L1']} L0C={c['L0C']} UB={c['UB']} cores={c['cores']} ghz={c['ghz']}  ({src})"
            + (f"  ★实测项缺 {un} -> 相关判据会拒判" if un else ""))


def _probe_pipe(soc):
    from core import pipeplan
    m = pipeplan.machine_for(soc)
    has_time = all(m["rate"].get(k) for k in ("MTE1_A", "MTE2_HIT", "CUBE_EFF"))
    return (f"L0C={m['cap'].get('L0C')} UB={m['cap'].get('UB')} ghz={m['freq_ghz']}"
            + ("  时间类判据可用" if has_time else "  ★无本机实测速率 -> 只有容量类判据可用"))


_CHIPLIT = re.compile(r"ascend[0-9]{3}[a-z0-9_-]*", re.I)


def scan_chip_literals(root: str | None = None) -> list:
    """扫 core/*.py 里**会被当成参数用**的芯片名字面量。

    只报"活的"位置 —— 默认参数、赋值右值、字典键、下标; 注释/docstring/help=/evidence= 里的
    说明性提及不报 (它们是证据出处, 本来就该写清是哪颗芯片测的)。

    为什么要这条: 硬件参数一旦按芯片名写死, 换机器时不会报错, 只会**悄悄按别的芯片判**。
    2026-09-11 实撞: mmplan.cap_for 里一行 `base = dict(CAP["ascend910b2"])`,
    让 950PR 上 cv_direct_path 被判成 False (本机其实有 cube->UB 直通), dram_GBps 用了 910B 的 1070。
    """
    import ast
    root = root or os.path.dirname(os.path.abspath(__file__))
    hits = []
    for fn_ in sorted(glob.glob(os.path.join(root, "*.py"))):
        try:
            src = open(fn_, encoding="utf-8").read()
            tree = ast.parse(src)
        except Exception:
            continue
        #  docstring 节点先记下来, 后面跳过
        docs = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                b = getattr(node, "body", None)
                if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                        and isinstance(b[0].value.value, str):
                    docs.add(id(b[0].value))
        for node in ast.walk(tree):
            #  默认参数
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in list(node.args.defaults) + [x for x in node.args.kw_defaults if x]:
                    if isinstance(d, ast.Constant) and isinstance(d.value, str) and _CHIPLIT.fullmatch(d.value.strip()):
                        hits.append((os.path.basename(fn_), d.lineno, "默认参数", d.value))
            #  赋值右值 / 字典键 / 下标
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str) and _CHIPLIT.fullmatch(node.value.value.strip()):
                hits.append((os.path.basename(fn_), node.lineno, "赋值", node.value.value))
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                    and isinstance(node.slice.value, str) and _CHIPLIT.fullmatch(node.slice.value.strip()):
                hits.append((os.path.basename(fn_), node.lineno, "下标取数", node.slice.value))
            if isinstance(node, ast.Dict):
                for k in node.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str) \
                            and _CHIPLIT.fullmatch(k.value.strip()) and id(k) not in docs:
                        hits.append((os.path.basename(fn_), k.lineno, "字典键", k.value))
    return hits


def audit_chip_literals() -> int:
    """把 scan_chip_literals 的结果按『数据表 / 活代码』分开打印。"""
    hits = scan_chip_literals()
    #  按芯片分档的**数据表**本来就要用型号当键 (CALIB/CAP/MACHINE 的回落表), 这类只提示
    DATA_TABLES = {"socspec.py", "chip_calib.json"}
    live = [h for h in hits if h[0] not in DATA_TABLES and h[2] != "字典键"]
    tables = [h for h in hits if h not in live]
    print("\n=== 代码里残留的芯片名字面量 ===")
    if not live:
        print("  无『活代码』残留 (默认参数/赋值/下标里都没有写死型号)")
    for f, ln, kind, v in live:
        print(f"  ★[{kind}] {f}:{ln}  \"{v}\"  —— 换机器时不会报错, 只会悄悄按这颗芯片判")
    if tables:
        print(f"  ※另有 {len(tables)} 处在**按芯片分档的数据表**里 (字典键/回落表), 属正常: "
              f"{sorted({f for f, _, _, _ in tables})}")
    return len(live)


def audit_calib(soc: str | None = None) -> int:
    """审计**实测标定**: ini 里没有、只能按芯片实测的那几项, 本机有没有。

    与硬编码常数审计分开列, 因为两者的补法不同: 前者是让模块回源读 ini,
    后者必须真的上机测一次 —— 没标定时**不许借用别的芯片**。
    """
    s = spec(soc)
    c, src = calib(s["soc"])
    known = sorted({k for v in CALIB.values() for k in v})
    print(f"\n=== 实测标定审计 (本机 {s['soc']}, 出处 {src}) ===")
    bad = 0
    for k in known:
        e = c.get(k)
        if e is None:
            print(f"  ★[本机缺] {k:26s} 别的芯片有 "
                  f"{[n for n, v in CALIB.items() if k in v]} —— **不得借用**, 用到它的判据应拒判")
            bad += 1
        elif e.get("status") == "uncalibrated" or e.get("value") is None:
            print(f"  ★[未标定] {k:26s} {e.get('evidence','')[:72]}")
            bad += 1
        else:
            print(f"    [{e['status']:8s}] {k:26s} = {e['value']}   ({e['verified']})")
    if bad:
        print(f"\n★{bad} 项本机没有可用标定。ini 事实换芯片自动跟着走, 实测标定不会 —— "
              f"用到这些项的判据必须显式拒判, 而不是回落到另一颗芯片。")
    return bad


def audit(soc: str | None = None) -> int:
    s = spec(soc)
    d = parse_ini(s["ini"])
    print(f"=== 硬编码常数审计 (本机 {s['soc']}, arch {s['arch']}) ===")
    print(f"平台 ini: {s['ini']}\n")
    #  模块表的 key 用的是简称 (ascend910b / ascend910b2), 本机全名是 Ascend950PR_957b;
    #  审计时把全名与它的各级前缀都当候选, 一个都不命中就是"本机无标定项".
    full = s["soc"].lower()
    cands = [full, full.split("_")[0], full.split("_")[0][:9], "ascend950", "ascend950pr"]
    bad = miss = 0
    for mod, attr, path, ini_key, scale in AUDIT:
        try:
            m = __import__(f"core.{mod}", fromlist=["x"])
            table = getattr(m, attr)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] core.{mod}.{attr} 取不到 ({type(e).__name__})")
            continue
        hit = next((c for c in cands if c in table), None)
        label = f"{mod}.{attr}[{'/'.join(path).replace('{soc}', hit or '本机')}]"
        if hit is None:
            print(f"  ★[无标定] {label:52s} 表里只有 {sorted(table)} —— 本机型号不在其中")
            miss += 1
            continue
        got, missing_at = _dig(table, [hit] + [p for p in path[1:]])
        if missing_at:
            print(f"  ★[无标定] {label:52s} 缺字段 {missing_at!r}")
            miss += 1
            continue
        if ini_key == "@mac_int8":
            want = s["mac_per_cycle"]["int8"]
            src = "ini [DtypeMKN] DT_INT8 的 m*n*k"
        else:
            sec, key = ini_key.split(".")
            want = _i(d, sec, key)
            src = f"ini {ini_key}"
        if want is None:
            print(f"  [skip] {label:52s} ini 无 {ini_key}")
            continue
        want *= scale
        ok = abs(float(got) - float(want)) < max(1e-9, abs(want) * 1e-9)
        flag = "  ✓" if ok else f"  ★不一致"
        print(f"  {'' if ok else '★'}{label:52s} 模块={got}  {src}={want:g}{flag}")
        if not ok:
            bad += 1
    print()
    if miss:
        print(f"※{miss} 项在模块的**内置表**里没有本机档位。内置表只是 socspec 不可用时的回落, "
              f"不代表模块真会用它 —— 下面『模块实际解析』那一节才是结论。")
    print()
    print("=== 模块在本机实际解析到什么 (这一节才是结论) ===")
    for name, fn in (("ub_budget.ub_cap_for()", _probe_ub),
                     ("mmplan.cap_for()", _probe_mm),
                     ("pipeplan.machine_for()", _probe_pipe)):
        try:
            print(f"  {name:26s} {fn(soc)}")
        except Exception as e:
            print(f"  ★{name:26s} 取不到: {type(e).__name__}: {e}")
            bad += 1
    if bad:
        print(f"★{bad} 项与 ini 不一致: 若是**实测偏离**(如 L1 被 bt_size 切走 1KB) 属正常, "
              f"但必须在该模块 REFS 里写明; 没写明的按手抄漂移处理。")
    audit_calib(soc)
    audit_chip_literals()
    if not miss and not bad:
        print("全部对上。")
    return 1 if (miss or bad) else 0


def compare(a: str, b: str) -> int:
    sa, sb = spec(a), spec(b)
    da, db = parse_ini(sa["ini"]), parse_ini(sb["ini"])
    print(f"=== {sa['soc']} (arch {sa['arch']})  vs  {sb['soc']} (arch {sb['arch']}) ===")
    print(f"  {os.path.basename(sa['ini'])}   {os.path.basename(sb['ini'])}\n")
    n = 0
    for sec in ("version", "SoCInfo", "AICoreSpec", "AICoreMemoryRates"):
        keys = sorted(set(da.get(sec, {})) | set(db.get(sec, {})))
        rows = [(k, da.get(sec, {}).get(k, "-"), db.get(sec, {}).get(k, "-"))
                for k in keys if da.get(sec, {}).get(k) != db.get(sec, {}).get(k)]
        if not rows:
            continue   # ssok: 无差异的 section 对'共 n 项不同'贡献 0
        print(f"[{sec}]")
        for k, va, vb in rows:
            print(f"  {k:26s} {va:>18s} -> {vb:>18s}")
        n += len(rows)
    print(f"\n共 {n} 项不同。★换芯片时这张表里的每一项都可能让某条判据失效, 逐条核对再复用。")
    #  原写 audit_calib(soc), 而 compare(a, b) 里没有 soc —— 必 NameError。
    #  core/pycheck 建好当天扫出来的; 审计第一个 SoC 的标定。
    audit_calib(a)
    return 0


def show(soc: str | None = None) -> int:
    s = spec(soc)
    print(f"=== {s['soc']}  (arch {s['arch']}, {s['aic_version']}) ===")
    print(f"  型号来源: {s['soc_source']}")
    print(f"  平台 ini: {s['ini']}\n")
    print(f"  核数     : cube {s['cube_cores']}  vector {s['vector_cores']}   频率 {s['freq_ghz']} GHz")
    cap = s["cap"]
    print(f"  片上容量 : L1 {cap['L1']/1024:.0f}KB  L0A {cap['L0A']/1024:.0f}KB  "
          f"L0B {cap['L0B']/1024:.0f}KB  L0C {cap['L0C']/1024:.0f}KB  UB {cap['UB']/1024:.0f}KB  "
          f"BT {cap['BT']/1024:.0f}KB")
    print(f"  片外     : L2 {cap['L2']/2**20:.0f}MB   HBM {cap['HBM']/2**30:.0f}GB")
    print(f"  MAC/周期 : " + "  ".join(f"{k} {v}" for k, v in s["mac_per_cycle"].items() if v))
    if s.get("regfile"):
        #  ★单位是**字节**不是位 (2026-09-29 由 opgen.api 的 A5 层定死, 此前这里印的是 "b")。
        #    证据是编译器内建头里同一组寄存器的元素数: vector_f32 = 64 x f32 = 256 B,
        #    vector_f16 = 128 x f16 = 256 B, vector_bool = 256 x i1 = 32 B,
        #    wvector_s24 = 256 x 24 bit = 768 B —— 四个类型各自独立地指向 ini 的三个字段。
        #    (tools/bisheng_compiler/lib/clang/*/include/__clang_cce_vector_types.h:83/85/133/168)
        #    差 8 倍的单位错在按元素数估算时会直接翻成 8 倍的收益幻觉, 所以这里连同**每 dtype
        #    的元素数**一起印 —— 那才是写 kernel 时真正要代进去的数。
        vw = s["regfile"].get("vector_reg_width")
        extra = (f"   (fp32 {vw // 4} 元素 / fp16 {vw // 2} 元素 每条向量指令)"
                 if vw else "")
        print(f"  寄存器   : " + "  ".join(f"{k} {v}B" for k, v in s["regfile"].items()) + extra)
    d = s.get("derived", {})
    if d:
        print(f"\n  派生峰值 (derived, 只作上界与排序, 不作承诺):")
        print(f"    fp16 cube 峰值   {d['peak_fp16_TFLOPS']:.1f} TFLOPS "
              f"= {s['cube_cores']} 核 x {s['mac_per_cycle']['fp16']} MAC/cy x 2 x {s['freq_ghz']}GHz")
        if d.get("dram_model_TBps"):
            print(f"    DRAM 模型带宽    {d['dram_model_TBps']:.2f} TB/s "
                  f"= 核数 x ddr_read_rate({s['rate_B_per_cycle'].get('ddr_read_rate')}B/cy) x 频率")
            print(f"      ★这是编译器成本模型参数, 不是实测可达带宽 (910B2 上该式高估实测 23%)")
        if d.get("l2_model_TBps"):
            print(f"    L2 模型带宽      {d['l2_model_TBps']:.2f} TB/s")
    p = s["paradigm"]
    print(f"\n  编程范式 : regbase {'可用' if p.get('regbase') else '不可用'}"
          f"  SIMT {'可用' if p.get('simt') else '不可用'}"
          f"{' (按 __NPU_ARCH__ 分档)' if p.get('simt_arch_guarded') else ''}")
    print(f"    取证: {p['evidence']}")
    print(f"    ★目录存在只说明**能用**, 不说明当前算子在用; 用不用要看该算子源码.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--soc", default=None, help="型号 (默认按本机 aclrtGetSocName 探测)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="两个型号的规格差异表")
    ap.add_argument("--audit", action="store_true", help="审计各模块硬编码常数 vs 本机 ini")
    ap.add_argument("--refs", action="store_true", help="本模块判据出处")
    ap.add_argument("--selftest", action="store_true",
                    help="判据负例自检 (标定引用的 held_fixed 核对)")
    ap.add_argument("--json", action="store_true", help="按 JSON 输出规格 (给下游取数)")
    ap.add_argument("--bw-table", action="store_true",
                    help="从 msopprof 二进制抽 *_bw_usage_rate(%%) 的分母表 (理论带宽, GiB/s)")
    a = ap.parse_args()
    if a.selftest:
        return selftest_calib_check() | selftest_bw_table()
    if a.bw_table:
        return print_bw_table()
    if a.refs:
        print_refs("socspec", REFS)
        return 0
    if a.compare:
        return compare(*a.compare)
    if a.audit:
        return audit(a.soc)
    if a.json:
        import json
        print(json.dumps(spec(a.soc), ensure_ascii=False, indent=1))
        return 0
    return show(a.soc)


if __name__ == "__main__":
    sys.exit(main())
