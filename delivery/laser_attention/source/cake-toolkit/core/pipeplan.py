"""pipeplan — 手写 cube/vec 流水的**编排前端证伪器** (build 前秒级, 先服务 gmfr).

为什么需要它 (2026-08-12, 一天之内四次撞墙, 每次代价一个 4.5 分钟构建 + 一轮真机对测):

| 撞的 | 症状 | 现有工具 | 本模块的判据 |
|---|---|---|---|
| UB 只剩 2048B 却想加 64KB 缓冲 | 真机 UB 越界 | `ub_budget` 覆盖(静态峰值) | B1 按**时间轴**算, 覆盖跨迭代存活 |
| flag 数 != 等待数 | aicore timeout 507014 | `detsync_lint` H6 提示级 | S1 硬失败 + 前缀余额 |
| `parallNum == nBatch` | **每条流水都变快而端到端 +19%** | 无 | **S2 流水深度 >= 2x 批量** |
| 优化了一个空等 63% 的核 | 改动生效但端到端 0 | 无 | **T1 各 pipe 空闲占比** |

**authority 边界 (与本项目 sim 的纪律一致)**:
  - **结构判据 (B1/B2/S1/S2) 是权威的** —— 它们是容量与计数守恒, 不含性能假设, 违反即真机出事.
  - **时间估算 (T1) 只作排序参考, 不作收益承诺** —— 速率取自本会话真机标定 (见 MACHINE 的出处注释),
    但真机 MTE2 交付率**没有可静态估的模型** (本项目已试过 4 个全失效). T1 的用途是回答
    "哪条 pipe 会成为墙 / 哪条在空闲", 不是回答"快多少".

用法:
    python -m core.pipeplan --demo gmfr_nbatch1      # 复现已知配置, 看判据
    python -m core.pipeplan --demo gmfr_nbatch4_pn4  # 应报 S2
    python -m core.pipeplan --calibrate              # 用三个已实测点自检 (同构性检查)
    python -m core.pipeplan --refs                  # 硬参数出处表 + 回源核对平台 ini
在 Python 里直接搭方案见 build_astationary() 的写法.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 机器模型. 容量是硬规格; 速率全部来自 2026-08-12 本会话的真机标定, 出处逐条注明.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 硬参数与**出处**. 规矩 (CLAUDE.md "spec 机读优先, 不手抄"): 每个数配一条 ref,
# 且凡是平台 ini 里有声明的, 由 verify_refs() **自动回源核对**, 不靠人肉抄写保持同步.
#     python -m core.pipeplan --refs      打印出处表并回源核对
# ---------------------------------------------------------------------------
import glob as _glob
import os as _os
import re as _re

def _ini_path(soc: str | None = None) -> str:
    """定位平台配置. 优先 ASCEND_HOME_PATH, 回退到常见安装位置的 glob.

    ★默认参数不写某颗芯片的名字 —— 不给就按本机探测 (socspec.default_soc)。
    """
    if not soc:
        from core import socspec
        soc = socspec.default_soc()
    name = soc if soc.lower().endswith(".ini") else soc + ".ini"
    name = name[0].upper() + name[1:]
    home = _os.environ.get("ASCEND_HOME_PATH", "")
    cands = ([_os.path.join(home, "aarch64-linux/data/platform_config", name)] if home else []) + \
        _glob.glob("/usr/local/Ascend/*/aarch64-linux/data/platform_config/" + name)
    for c in cands:
        if _os.path.isfile(c):
            return c
    return ""


def _ini_read(soc: str | None = None) -> dict:
    p = _ini_path(soc)
    if not p:
        return {}
    out = {}
    for line in open(p, errors="ignore"):
        m = _re.match(r"\s*([a-z0-9_]+)\s*=\s*(-?\d+)\s*$", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


# {参数: (ini 键 或 None, 状态, 出处说明)}
#   状态 ini_exact = 必须与 ini 逐位相同, verify_refs 会硬核对
#        runtime    = 以 tiling 运行时实际比较的值为准 (与 ini 不同, 见说明)
#        measured   = 本会话真机标定, ini 无声明
#        disputed   = 两个来源不一致且现有数据分不出, 只用于排序不作承诺
REFS = {
    "cap.L0A": ("l0_a_size", "ini_exact", "平台配置 Ascend910B2.ini"),
    "cap.L0B": ("l0_b_size", "ini_exact", "平台配置 Ascend910B2.ini"),
    "cap.L0C": ("l0_c_size", "ini_exact", "平台配置 Ascend910B2.ini (确认 128KB, 非标称 256KB)"),
    "cap.UB":  ("ub_size",   "ini_exact", "平台配置 Ascend910B2.ini"),
    "cap.L1":  (None, "runtime",
                "★ini 写 l1_size=524288(512KB), 但 tiling 运行时从 GetCoreMemSize(L1) 拿到的更小: "
                "真机日志(2026-08-12 复现) 'GMMFR(A8W8) L1 tiling: ... -> L1 384KB/511KB'. "
                "缺的 1KB 正好等于 ini 的 bt_size=1024 (bias table 从 L1 切走), 523264=511KB 与日志逐位吻合 "
                "—— 该归因是假设, 但结论不依赖它 (depthB1=8 的 needL1 恰为未扣除的 524288, 怎样都过不了闸). "
                "★勿据数据手册的 512KB 更正本表: 判据必须对齐 tiling 实际比较的那个值"),
    "rate.MTE1_A": ("l1_to_l0_a_rate", "disputed",
                    "ini 声明 512, dubin 2026-08-12 给出 256, 差 2 倍未解决. "
                    "比值 2:1 两来源一致 -> 解析最优 baseM=sqrt(rA/rB*C)=256/baseN=128 不受影响 "
                    "(且与真机 6 点扫描赢家重合); 受影响的只有 MTE1_EFF 口径 (69.0% vs 34.5%). "
                    "要分需一个只变搬运字节、锁死指令数的单变量真机点"),
    "rate.MTE1_B": ("l1_to_l0_b_rate", "disputed", "同 MTE1_A, ini 声明 256 而本表取 128"),
    "rate.MTE1_EFF": (None, "measured",
        "6 点真机扫描 runs/sweep_*: 纯搬运占实测 aic_mte1 的 64.5~74.4%. "
        "★必须实测的原因: 差额是**每条 MTE1 指令的固定开销**, 平台 ini 只声明搬运速率(l1_to_l0_*_rate), "
        "不声明指令发射与建立的周期数, 无法从任何已声明参数推出"),
    "rate.MTE2_HIT": (None, "measured",
        "探针 F 臂 runs/pbF_l2res_atomic: L2 常驻 + 真原子写压力. "
        "★必须实测的原因: MTE2 的交付率**取决于字节从哪来**(L2 命中 vs DRAM 缺失)与复用距离, "
        "不是一个固定速率. 本项目已试过 4 个静态成本模型(重读次数/GM→L1 字节/指令数/DRAM+L2 最小二乘)"
        "全部与实测矛盾, 详见 DEBUG_FINDINGS 的'四个失效的成本模型'"),
    "rate.MTE2_MISS": (None, "measured",
        "探针 D/E 臂 runs/pbD_seq_atomic runs/pbE_seq_atomic_full: 22.4~28.1 取中值. "
        "★必须实测的原因: 同 MTE2_HIT; 且此值随全卡 DRAM 竞争(AIV 原子写占多少份额)变化, "
        "ini 的 ddr_rate=32 是无竞争的峰值声明, 用它会高估"),
    "rate.MAC_PER_CY": (None, "measured",
        "24 核 x 8192 = 707.8 TOPS int8 峰值反推. "
        "★半可推: ini 的 cube_m/n/k_size=16 是 fp16 的分形(16x16x16=4096), int8 的 k 维翻倍 "
        "(16x16x32=8192) 这一条 ini 未声明, 需按 dtype 另行确认"),
    "rate.CUBE_EFF": (None, "measured",
        "runs/fx_off: aic_cube 1973us 对算力下限 1820.5us. "
        "★必须实测的原因: 差额来自尾块(真实 m-tile 平均 174/256 行)与发射空泡, 与形状分布有关, "
        "ini 不含任何'效率'类声明"),
    "rate.FIXPIPE": (None, "measured",
        "runs/fx_off: aic_fixpipe_active_bw 107.7 GB/s @1.8GHz. "
        "★必须实测的原因: ini 无 L0C→GM 的速率条目 (l0_c_to_ub_rate=256 是到 UB 不是到 GM)"),
    "rate.MTE3_BW_CEIL": (None, "measured",
        "6 点粒度扫描 runs/vW0_512_scalar runs/gran_1024 runs/vW1_2k_scalar runs/gran_4096 "
        "runs/gran_8192 runs/vW2_16k_scalar: 4K/8K/16K 三点饱和值均值 21.90 GB/s (极差 0.19). "
        "★必须实测的原因: 这是**原子加**(读-改-写)的天花板, 而 ini 声明的 ub_to_l2_rate=64 B/cy "
        "= 115 GB/s 是**普通写**的速率 —— 两者差 5.3 倍, 拿 ini 的数推会错 5 倍"),
    "rate.MTE3_CL": (None, "measured",
        "同一组 6 点扫描的两个未饱和点 (512B->18.93e6, 1KB->17.76e6) 取均值. "
        "★必须实测的原因: 它是**在途事务数 C ÷ 每事务延迟 L**, C 与 L 在平台配置里**都没有声明**, "
        "且二者只以比值形式影响结果, 无法从任何已声明参数分解或推出"),
    "rate.VMTE2": (None, "measured",
        "runs/fx_off: aiv_mte2 35783KB/1246.55us = 28.7 GB/s. "
        "★必须实测的原因: 含 staging 的 L2 命中率(实测 97.56%)与指令粒度, 非纯硬件速率"),
    "freq_ghz": ("cube_freq", "ini_exact", "ini cube_freq=1800 (MHz), 本表以 GHz 记故除 1000"),
}

#  ★这张表**只放实测标定**, 容量/频率/MAC 每周期一律从本机 ini 取 (machine_for 里做)。
#    2026-09-11 实撞: 原来整张表按 soc 硬编码且只有 910B2 一档, 在 950PR 上 machine_for(soc) 直接
#    KeyError —— 更糟的是若有人补一行回落, 就会拿 910B 的速率去判 950PR。
#    ★下面的 MTE1_A/MTE1_B 是**实测有效速率**, 与 ini 的 l1_to_l0_a/b_rate 不是一回事:
#      910B2 的 ini 写的是 512/256, 本表是 256/128, 关系未建立。所以它们**不可移植到别的芯片**,
#      换芯片必须重新实测。
MACHINE = {
    "ascend910b2": {
        "cap": {
            "L1": 511 * 1024,
            "L0A": 64 * 1024,
            "L0B": 64 * 1024,
            "L0C": 128 * 1024,
            "UB": 192 * 1024,
        },
        "rate": {
            "MTE1_A": 256.0, "MTE1_B": 128.0, "MTE1_EFF": 0.695,
            "MTE2_HIT": 118.5, "MTE2_MISS": 25.0,
            "MAC_PER_CY": 8192.0, "CUBE_EFF": 0.92,
            "FIXPIPE": 59.8,
            # 原子写不是固定速率, 见 mte3_bw(): BW(B) = min(BW_CEIL, CL*B)  (Little 定律)
            "MTE3_BW_CEIL": 21.90e9, "MTE3_CL": 18.34e6,
            "VMTE2": 15.9,
        },
        "freq_ghz": 1.8,
    }
}


_MACHINE_CACHE = {}

#  ★当前芯片上下文: 嵌套的成本函数拿不到 soc 参数 (它们只接形状), 又不能让它们各自
#    写死一颗芯片。用一个显式设定的模块级变量当"当前芯片", 默认 None = 本机探测。
#    与"静默回落"的区别在于: 它必须被**调用方显式设定**, 且 --calibrate 这类按芯片的
#    检查会在入口处设好, 不会在无人察觉时生效。
_ACTIVE_SOC = None


def set_active_soc(soc: str | None) -> None:
    """设定后续 machine_for(None) 解析到哪颗芯片。传 None 恢复成按本机探测。"""
    global _ACTIVE_SOC
    _ACTIVE_SOC = soc


def rate_of(m: dict, key: str) -> float:
    """取一项速率; 本芯片没有该项实测标定时给出可读的拒判信息, 而不是 KeyError。"""
    v = m.get("rate", {}).get(key)
    if v is None:
        raise SystemExit(
            f"{m.get('soc')} 没有 {key} 的实测速率标定 —— 时间类判据不作数。\n"
            f"  ini 只给容量与成本模型上限, 各 pipe 的**有效速率**必须上机实测。\n"
            f"  补法: 实测后在 core/pipeplan.MACHINE 里加本芯片一档 (别抄别的芯片的);\n"
            f"  或显式 set_active_soc('<已标定型号>') 说明你在按那颗芯片算。")
    return v


def machine_for(soc: str | None = None) -> dict:
    """本机(或指定型号)的 pipeplan 机器参数: **容量/频率/MAC 从 ini 取, 速率从实测表取**。

    没有该芯片的实测速率时不回落 —— 返回的 rate 里缺项, 用到缺项的判据自己拒判。
    这样"容量类"判据换芯片就能用, 而"时间类"判据在没标定时明说不作数。
    """
    soc = soc or _ACTIVE_SOC
    key = (soc or "").lower() or "__auto__"
    if key in _MACHINE_CACHE:
        return _MACHINE_CACHE[key]
    cap, freq, name, rate = None, None, soc, {}
    try:
        from core import socspec
        sp = socspec.spec(soc)
        name = sp["soc"]
        cap = {k: sp["cap"][k] for k in ("L1", "L0A", "L0B", "L0C", "UB") if sp["cap"].get(k)}
        freq = sp["freq_ghz"]
        mac = sp["mac_per_cycle"]
        rate = {"MAC_PER_CY": float(mac.get("int8") or 0) or None}
        #  ini 的 l1_to_l0_* / l0_c_to_ub 是**成本模型上限**, 与本表的实测有效速率不同名不同义,
        #  这里单独带出来供判据当上限用, 不覆盖实测项。
        rb = sp["rate_B_per_cycle"]
        rate["MTE1_A_CEIL"] = rb.get("l1_to_l0_a_rate")
        rate["MTE1_B_CEIL"] = rb.get("l1_to_l0_b_rate")
        rate["FIXPIPE_CEIL"] = rb.get("l0_c_to_ub_rate")
    except Exception as e:
        print(f"  ★socspec 取不到规格 ({e}); 容量类判据将不可用", file=__import__("sys").stderr)
    builtin = MACHINE.get((name or "").lower()) or MACHINE.get((soc or "").lower())
    if builtin:
        cap = cap or dict(builtin["cap"])
        freq = freq or builtin["freq_ghz"]
        merged = dict(builtin["rate"]); merged.update({k: v for k, v in rate.items() if v})
        rate = merged
    elif not cap:
        raise KeyError(f"{soc or '本机'}: 既取不到 ini 也没有实测标定, pipeplan 无法工作")
    else:
        print(f"  ★{name}: 没有本芯片的**实测速率**标定, 只有 ini 的容量与上限 —— "
              f"容量类判据(B1/B2/S1/S2)可用, 时间类(T1/各 pipe 忙时)一律不作数。\n"
              f"    补法: 实测后在 core/pipeplan.MACHINE 里加本芯片一档 (别抄 910B 的)。",
              file=__import__("sys").stderr)
    m = {"cap": cap, "rate": rate, "freq_ghz": freq, "soc": name}
    _MACHINE_CACHE[key] = m
    return m


def verify_refs(soc: str | None = None) -> int:
    """回源核对: ini 里有声明的参数必须对上, 对不上即 FAIL (防手抄漂移)."""
    ini = _ini_read(soc)
    M = machine_for(soc)
    print(f"=== pipeplan 硬参数出处 (soc={soc}) ===")
    print(f"平台配置: {_ini_path(soc) or '★未找到, ini 项无法回源核对'}\n")
    bad = 0
    for key, (ini_key, status, note) in REFS.items():
        grp, name = (key.split(".") + [""])[:2]
        val = M[grp][name] if name else M[grp]
        mark = ""
        if ini_key and ini:
            got = ini.get(ini_key)
            if got is None:
                mark, bad = f"★ini 无 {ini_key}", bad + 1
            elif status == "ini_exact" and float(got) != float(val) and \
                    not (key == "freq_ghz" and abs(got / 1000.0 - val) < 1e-9):
                mark, bad = f"★不一致: ini {ini_key}={got}", bad + 1
            elif status == "disputed":
                mark = f"(ini {ini_key}={got}, 本表取 {val})"
            else:
                mark = f"= ini {ini_key} ✓"
        print(f"  {key:<16} {val:<12} [{status}] {mark}")
        print(f"      {note}")
    print("\n" + ("回源核对通过" if not bad else f"★{bad} 项对不上, 先修表再用"))
    return 0 if not bad else 1


PIPES = ("MTE2", "MTE1", "MAC", "FIXPIPE", "VMTE2", "V", "MTE3", "S")

# c0 与分形常数 (CANN 9.1 头文件: matmul_config_utils.h B8_C0SIZE=32/B32_C0SIZE=8,
# kernel_utils_constants.h BLOCK_CUBE=16)
C0 = {1: 32, 2: 16, 4: 8}          # 按元素字节数取 c0
BLOCK_CUBE = 16


def nz_bytes(rows: int, cols: int, esize: int = 1) -> int:
    """L1 上 NZ 布局的**真实**字节数 (含分形对齐 pad).

    ★2026-08-12 验: GM→L1 对 ND 源走 CopyND2NZ 随路转换 (data_copy_wrapper_nd.h),
      所以 L1 里的东西是 NZ 不是 ND —— 自己往 L1 载 A 时必须按这个算容量, 按 rows*cols 算会低估.
      dstNzC0Stride = Ceil(rows, BLOCK_CUBE) * BLOCK_CUBE (库里同名参数的取法).
      对 gmfr 的 A 面板 (256 x 1536 int8) 两维本就对齐, 结果正好 393216B, 无膨胀;
      尾块 (如 58 行) 会 pad 到 64 行.
    """
    c0 = C0[esize]
    return (cols + c0 - 1) // c0 * ((rows + BLOCK_CUBE - 1) // BLOCK_CUBE * BLOCK_CUBE) * c0 * esize


# ★原子写长度必须是 512B 的倍数 (2026-08-14 实测 12 点, 见 mte3_align_penalty)
MTE3_ALIGN_BYTES = 512

def mte3_align_penalty(gran_bytes: int) -> float:
    """原子写长度不是 512B 倍数时的**额外**时间系数 (1.0 = 无罚).

    ★这条不在 Little 定律模型里, 是独立的一层. 2026-08-14 实测 12 点 (固定原子写总字节,
      只变每条的字节数, 每点两轮, 轮内散布 <1%), 以 2048B 为基准:

        512 的倍数 (无罚):  1024 -> 0.946   1536 -> 1.029   2048 -> 1.000
                            2560 -> 0.903   3072 -> 1.023   4096 -> 0.982
        非 512 的倍数:      2080 -> 1.532   2112 -> 1.439   2176 -> 1.264
                            2304 -> 1.116   2688 -> 1.212   2944 -> 1.193

      罚的大小随"离下一个 512 边界还差多少"单调变化: 余 32B 罚 53%, 余 256B 罚 12%。
      形态接近"按 ceil(B/512)*512 计费"但比它更狠 (2080 的 ceil 模型给 1.231, 实测 1.532)。

    ★两条被实测证伪的判据, 不要再试 (都是先写下预测再测的):
        "32B 对齐即可"  -> 2080 = 65x32 对齐, 却是罚得最狠的一点
        "128B 对齐即可" -> 2176 = 17x128 与 2688 = 21x128 都对齐, 仍罚 26% / 21%

    用法: 设计任何"给每行的写出附加几个值"的布局时先过这里 —— 附加量必须凑够 512B 的倍数,
    否则省下的字节会被对齐罚吃掉还倒亏。gmfr 的界累加器就是这么从 +1.56% 变成 +25% 的。
    """
    r = gran_bytes % MTE3_ALIGN_BYTES
    if r == 0:
        return 1.0
    # 实测拟合: 余数越小罚越重; 端点由 2080(余32,1.53) 与 2304(余256,1.12) 定, 中间线性内插
    frac = r / MTE3_ALIGN_BYTES
    return 1.0 + 0.55 * (1.0 - frac) ** 1.5


def mte3_bw(gran_bytes: int, soc: str | None = None) -> float:
    """AIV 原子写的**有效带宽**(B/cycle/核), 随每条写出的字节数变化.

    ★这条曲线不是拟合出来的经验式, 是 Little 定律的直接形式:
          有效带宽 = min( BW_CEIL ,  (C/L) x B )
      在途事务数 C 与每事务延迟 L 只以比值出现; 粒度小于饱和点时带宽正比于粒度,
      越过饱和点后被天花板封住.

          饱和粒度 B* = BW_CEIL / (C/L) = 21.90e9 / 18.34e6 = 1194 B

    6 点真机验证 (固定 AIC 负载与原子写总字节, 只变粒度), **最大偏差 3.2%**:
        512B 3614us / 1KB 1926 / 2KB 1651 / 4KB 1604 / 8KB 1603 / 16KB 1590
      其中 1KB / 4KB / 8KB 三点是**先有模型后测的**, 用来把两个竞争模型分开 ——
      "固定开销 + 理想带宽"那个模型在这三点上分别错 +19% / -18% / -28%, 已被否.

    ★两个常数为什么必须实测: 见 REFS 里 MTE3_BW_CEIL 与 MTE3_CL 的说明
      (原子加是读-改-写, ini 的 ub_to_l2_rate 是普通写, 差 5.3 倍; C 与 L 平台配置里都没有).

    ★★适用边界 —— 本式是**单流**形式, 对**交织写**不成立 (gmfr 2026-08-25 实测证伪):
      它假设这条 MTE3 上只有一股写在争在途事务。当一股小写与一股大写交织在同一条流水上时,
      小写躲在大写的影子里, **它自己的粒度不进入方程**。
      实证: gmfr 的界写 (256B/行) 与数据写 (2048B/行) 逐行交织。按本式, 界写只能跑
      4.70 GB/s/核, 砍到 32B 会掉到 0.59 GB/s/核 —— 应慢一个数量级; 而消融实测
      (measurement_builds/g64w: 只改写长度 256B->32B) 两个 shape 都落在散布内, 时间纹丝不动。
      同一轮还测出界写的成本 ~100% 是指令与延迟、~0% 是字节。
      **推论: 用本式给"把某股写变小"估收益时, 必须先确认那股写是不是独占这条流水; 不是就别估,
      直接做一个只改写长度的消融构建去测。**
    """
    R = machine_for(soc)["rate"]
    bw_bytes_per_s = min(R["MTE3_BW_CEIL"], R["MTE3_CL"] * gran_bytes)
    return bw_bytes_per_s / (machine_for(soc)["freq_ghz"] * 1e9)


def mte3_saturation_bytes(soc: str | None = None) -> float:
    """原子写的饱和粒度 B* —— 写出粒度取到它以上, MTE3 就不再是粒度的函数."""
    R = machine_for(soc)["rate"]
    return R["MTE3_BW_CEIL"] / R["MTE3_CL"]


def advise_write_gran(admissible: list, soc: str | None = None) -> dict:
    """给定**可选**粒度集合(受 tile 形状/流水深度等约束), 给出该取哪个与理由.

    可选集合是离散的, 由三条约束共同决定 (以 gmfr 为例):
      - 写出必须是 L2 写访问粒度 512B 的整数倍 (实测: AIC fixpipe 67.1MB/136572 次 = 491B/次)
      - UB tile 元素数固定为 ubCalSize, 形状只能是 (ubCalSize/W) x W
      - nBatch 受 parallNum >= 2*nBatch 与 CV_PARALL_NUM_MAX 限制
    """
    star = mte3_saturation_bytes(soc)
    ceil_bw = machine_for(soc)["rate"]["MTE3_BW_CEIL"]
    rows = []
    for g in sorted(admissible):
        bw = min(ceil_bw, machine_for(soc)["rate"]["MTE3_CL"] * g)
        rows.append((g, bw / 1e9, bw / ceil_bw))
    ok = [r for r in rows if r[2] >= 0.95]
    return {"saturation_bytes": star, "table": rows,
            "pick": ok[0][0] if ok else rows[-1][0],
            "why": (f"饱和粒度 B*={star:.0f}B; 可选集合里第一个达到 95% 天花板的是 "
                    f"{ok[0][0]}B ({ok[0][2]:.0%})" if ok else
                    f"饱和粒度 B*={star:.0f}B; 可选集合里没有达到 95% 的, 取最大的 {rows[-1][0]}B")}


def advise_tile(ea: int = 1, eb: int = 1, ec: int = 4, live_acc: int = 1,
                soc: str | None = None, topn: int = 5) -> dict:
    """由 **MTE1 速率比 + L0C 容量 + 累加器活跃份数** 解出最优 (baseM, baseN).

    每条 Mmad 搬 baseM·baseK·eA 进 L0A、baseK·baseN·eB 进 L0B, 做 baseM·baseN·baseK 次乘加:

        每 MAC 的 MTE1 周期 = eA/(rA·baseN) + eB/(rB·baseM)
        约束                baseM·baseN·eC·live_acc <= L0C
        => M* = sqrt( C · (eB·rA)/(eA·rB) ),   C = L0C / (eC·live_acc)

    ★baseK 不进这个式子 —— 它在分子分母同时出现被约掉, 故 baseK 只改指令条数与缓冲深度,
      不改每 MAC 的搬运量. (这条独立解释了'baseK 越大指令越少所以越快'为何与实测反向.)

    ★**只依赖 rA/rB 这个比值, 不依赖绝对速率** —— 缩放 rA,rB 只缩放 f 不移动极小点.
      这一点很要紧: 两个绝对值在 REFS 里是 disputed (ini 512/256 对 dubin 给的 256/128),
      但**比值 2:1 两个来源一致**, 所以本判据不受那个争议影响.

    ★dtype 依赖显式给出, 不写死 (dubin 2026-08-12 提醒):
        ea/eb = A/B 的元素字节 (int8=1, fp16/bf16=2, int4=0.5 用 1/2 表示需自行折算)
        ec    = 累加器元素字节 (int32/fp32=4)
        live_acc = stepM·stepN·dbL0C, 即**同时活在 L0C 的累加器份数**
      举例: int8+int32 单缓冲 -> (256,128); 若 dbL0C=2 则 C 减半 -> 最优移到 (181,90) 附近;
      若 B 换成 int4 (eb=0.5) -> 最优移向方形.

    返回按代价排序的前 topn 个**格点**解 (baseM/baseN 取 BLOCK_CUBE=16 的整数倍), 不是连续解 ——
    连续最优往往不在格点上, 且 N = floor(C/M) 向下取整**会丢 L0C 面积**, 直接四舍五入连续解
    可能落到一个面积利用率很低的点上. 例: live_acc=2 时连续解 181, 但 (176,80) 只用到 86% 的
    L0C 而 (144,112) 用到 98%, 后者反而更优. 故按格点枚举, 表里同时给面积利用率.
    """
    R = machine_for(soc)["rate"]
    ra, rb = R["MTE1_A"], R["MTE1_B"]
    cap = machine_for(soc)["cap"]["L0C"] / (ec * live_acc)          # baseM*baseN 的上界
    cont = (cap * (eb * ra) / (ea * rb)) ** 0.5                  # 连续解 M*
    cands = []
    m = BLOCK_CUBE
    while m <= cap:
        n = int(cap // m) // BLOCK_CUBE * BLOCK_CUBE
        if n >= BLOCK_CUBE:
            f = ea / (ra * n) + eb / (rb * m)
            cands.append((f, m, n, m * n * ec / 1024.0, m * n / cap))
        m += BLOCK_CUBE
    cands.sort()
    best = cands[0]
    return {"continuous_M": cont, "cap_MN": cap, "best": (best[1], best[2]),
            "table": [(mm, nn, f * 1e6, kb, f / best[0], util) for f, mm, nn, kb, util in cands[:topn]],
            "why": (f"连续解 M*=sqrt(C·eB·rA/(eA·rB))={cont:.1f}; 格点最优 "
                    f"({best[1]},{best[2]}), L0C 占 {best[3]:.0f}KB/"
                    f"{machine_for(soc)['cap']['L0C']/1024:.0f}KB x {live_acc} 份")}


# ---------------------------------------------------------------------------
# 需要**外部填充**的量. 工具不猜: 用到而未填时在输出里打 ★TODO 并说明怎么填、不填会错在哪.
#   kind = "告知"  算子算法/形状决定, 模型推不出, 使用者必须给
#          "实测"  不可静态建模, 必须上真机量 (第四层)
#     python -m core.pipeplan --needs    列出全部待填项
# ---------------------------------------------------------------------------
NEEDS = {
    "rows_per_expert": ("告知", "每个专家的行数(分组分布)",
        "从 group_list 拿 —— repro.build_inputs 里是 bincount(exp_ids), 或用甲方真实路由统计. "
        "★不填的后果: 退化为均匀假设, **低估** mTiles 从而低估 B 侧流量. "
        "gmfr 实测: 均匀 400 个 m-tile 对真实 588 个, 差 47%"),
    "mte2_delivery": ("实测", "MTE2 的交付率 (B/cycle)",
        "python -m core.perf --app '<跑真 shape>' --kernel-name <op> --out runs/x, "
        "再读 GM_to_L1_datas / aic_mte2_time. ★不可静态估: 本项目试过 4 个模型 "
        "(重读次数/GM→L1 字节/指令数/DRAM+L2 最小二乘) 全部与实测矛盾"),
    "mte3_const_for_dtype": ("实测", "非 fp32 原子加的 BW_CEIL 与 C/L",
        "现有常数只对 **fp32 SetAtomicAdd** 成立; 2 字节原子加(SetAtomicAdd<half>)走另一条硬件通路. "
        "重测法: core/probe 固定 AIC 负载与原子写总字节, 扫 atomBytes 取 6 点, 用两个未饱和点解 C/L、"
        "三个饱和点取 BW_CEIL 均值"),
    "flag_counter_depth": ("实测", "跨核旗标的计数器深度",
        "CrossCoreSetFlag 在 CANN 头文件里**只有声明, 无深度说明**. 实证只知 parallNum=4/8 可用. "
        "测法: 小 shape 上逐档抬 parallNum 直到挂死或算错. ★不测就抬 parallNum = 赌一个未知上限"),
    "shared_operand": ("告知", "哪个操作数在哪个轴上共享",
        "由算子算法决定, 模型推不出. grouped matmul: B 按专家分片, 共享范围**只在组内**; "
        "普通 matmul: B 在整个 m 轴共享. 它决定 stepM/stepN 哪个方向有意义"),
    "output_scatter": ("告知", "输出是散射还是密集写",
        "散射(如 finalize routing 按 row_index 落行)决定 MTE3 的粒度与累加器工作集; "
        "密集写则 MTE3 粒度由 tile 宽度直接给出"),
}


def print_needs() -> int:
    print("=== pipeplan 需要外部填充的量 (工具不猜) ===\n")
    for k, (kind, what, how) in NEEDS.items():
        print(f"  [{kind}] {k}\n      是什么: {what}\n      怎么填: {how}\n")
    return 0


@dataclass
class OpShape:
    """第二层的输入 —— 算子形状. ★rows_per_expert 不填会低估 B 侧流量, 见 NEEDS."""
    m: int
    n: int
    k: int
    e: int = 1
    rows_per_expert: list = field(default_factory=list)   # ★需告知; 空 = 退化均匀假设 + 告警
    ea: float = 1.0
    eb: float = 1.0
    ec: float = 4.0
    cores: int = 24

    def mtiles(self, baseM: int) -> int:
        """Σ_e ceil(rows_e / baseM). 填充浪费就藏在这个向上取整里."""
        rows = self.rows_per_expert or [self.m // max(self.e, 1)] * max(self.e, 1)
        return sum(-(-int(r) // baseM) for r in rows if r > 0)


def advise_shape(shape: OpShape, live_acc: int = 1, soc: str | None = None, topn: int = 6) -> dict:
    """第二层: 在**第一层的可行域内**枚举 (baseM,baseN), 按 GM→L1 字节排序.

        GM→L1 = (n/baseN)·m·k·eA   +   mTiles(baseM)·k·n·eB
                 └ A 每个 n 块重读一次 ┘   └ B 每个 m-tile 重读一次 ┘

    ★这一层只算**字节**, 不算时间 —— MTE2 的交付率不可静态估 (见 NEEDS 的 mte2_delivery).
    第一层的 MTE1 代价同表给出 —— 字节并列时由它破平 (issue#1 上 (128,256) 与 (256,128) 字节完全并列,
    第一层判 (256,128) 省 25% 的 MTE1 周期, 与真机 6 点扫描的赢家一致).

    ★字节只区分**粗选**, 不区分细选: 6 点 baseM/baseN 扫描里字节最少的 (256,128) 也确实最快,
      字节排序给对了; 但 23 点 step/depth 扫描里赢家与基线**字节几乎相同 (350 对 348 MB/核) 而
      时间差 9%** —— 那一层字节完全没有分辨力, 必须交真机扫描.
    """
    cap = machine_for(soc)["cap"]["L0C"] / (shape.ec * live_acc)
    todo = []
    if not shape.rows_per_expert:
        todo.append("★TODO [告知] rows_per_expert 未填 -> 已按均匀 "
                    f"{shape.m // max(shape.e,1)} 行/组 假设; gmfr 实测该假设把 m-tile 数从 588 低估到 400 "
                    "(差 47%), 会**低估 B 侧流量**. 见 NEEDS")
    rows = []
    bM = BLOCK_CUBE
    while bM <= cap:
        bN = int(cap // bM) // BLOCK_CUBE * BLOCK_CUBE
        if bN >= BLOCK_CUBE and shape.n % bN == 0:
            a = (shape.n / bN) * shape.m * shape.k * shape.ea
            b = shape.mtiles(bM) * shape.k * shape.n * shape.eb
            rows.append((a + b, bM, bN, a, b, shape.mtiles(bM)))
        bM += BLOCK_CUBE
    rows.sort()
    # 第一层的代价 (每 MAC 的 MTE1 周期) 一并算出 —— 字节并列时由它破平, 见下面 tie 的判定
    R = machine_for(soc)["rate"]
    out = []
    for tot, bm, bn, a, bb, mt in rows:
        mte1 = shape.ea / (R["MTE1_A"] * bn) + shape.eb / (R["MTE1_B"] * bm)
        out.append((tot, bm, bn, a, bb, mt, mte1))
    best_bytes = out[0][0]
    tied = [r for r in out if r[0] <= best_bytes * 1.001]        # 字节 0.1% 内视为并列
    tie_note = None
    if len(tied) > 1:
        tied.sort(key=lambda r: r[6])
        tie_note = (f"字节并列 {len(tied)} 个 (差 <0.1%): "
                    + ", ".join(f"({r[1]},{r[2]})" for r in tied)
                    + f" -> 由**第一层**的 MTE1 代价破平, 取 ({tied[0][1]},{tied[0][2]}), "
                      f"比次优省 {(tied[1][6]/tied[0][6]-1)*100:.0f}% 的 MTE1 周期")
    best = (tied[0][1], tied[0][2]) if tied else None
    out.sort(key=lambda r: (r[0], r[6]))
    return {"cap_MN": cap, "todo": todo, "tie": tie_note,
            "table": [(bm, bn, tot / shape.cores / 2 ** 20, a / shape.cores / 2 ** 20,
                       bb / shape.cores / 2 ** 20, mt, tot / best_bytes, mte1 * 1e5)
                      for tot, bm, bn, a, bb, mt, mte1 in out[:topn]],
            "best": best}


@dataclass
class Op:
    """流水里的一步. bytes/macs 二选一; 只填 cycles 也行 (直接给周期)."""
    pipe: str
    name: str = ""
    bytes: float = 0.0
    macs: float = 0.0
    cycles: float = 0.0
    rate_key: str = ""              # 覆盖默认速率 (如 MTE2_HIT / MTE2_MISS / MTE1_A)
    alloc: tuple = ()               # (space, buf_name, bytes)
    free: str = ""                  # buf_name
    reads: tuple = ()               # 依赖的 buf_name
    writes: tuple = ()
    set_flag: str = ""              # 跨核旗标 id
    wait_flag: str = ""


@dataclass
class Program:
    """一个**稳态循环体**的描述. 结构判据按循环体 + 跨迭代存活算."""
    name: str
    ops: list = field(default_factory=list)          # **内层**循环体 (每 inner 次)
    outer_ops: list = field(default_factory=list)    # **外层**每轮一次 (跨核旗标通常在这一层)
    iters: int = 1                  # 外层轮数 (整个 kernel)
    inner: int = 1                  # 每个外层轮里内层跑几次
    persistent: dict = field(default_factory=dict)   # 循环外常驻: {buf: (space, bytes)}
    buf_count: dict = field(default_factory=dict)    # {buf: 缓冲个数}, 默认 1
    # 跨核流水深度: 生产者最多领先多少个"块", 消费者一次消费多少个
    producer_slots: int = 0
    consumer_batch: int = 0
    role: str = ""                  # "AIC" / "AIV"; 跨核旗标必须成对检查, 单侧判不了
    soc: str | None = None

    def add(self, *ops):
        self.ops.extend(ops)
        return self

    def add_outer(self, *ops):
        self.outer_ops.extend(ops)
        return self

    def all_ops(self):
        """展开一个外层轮: 外层 op 各一次 + 内层 op 各 inner 次."""
        return list(self.outer_ops) + list(self.ops) * self.inner


def _op_cycles(op: Op, R: dict) -> float:
    if op.cycles:
        return op.cycles
    if op.macs:
        return op.macs / (R["MAC_PER_CY"] * R["CUBE_EFF"])
    if not op.bytes:
        return 0.0
    key = op.rate_key or {
        "MTE2": "MTE2_MISS", "MTE1": "MTE1_A", "FIXPIPE": "FIXPIPE",
        "VMTE2": "VMTE2",
    }.get(op.pipe, "")
    if op.pipe == "MTE3":
        # 原子写的速率是粒度的函数, 不是常数. rate_key 此处复用为"每条写出的字节数"
        gran = float(op.rate_key) if op.rate_key else 512.0
        return op.bytes / mte3_bw(int(gran))
    if not key:
        return 0.0
    r = R[key]
    if op.pipe == "MTE1":
        r *= R["MTE1_EFF"]
    return op.bytes / r


def check(prog: Program, peer: "Program|None" = None) -> dict:
    M = machine_for(prog.soc)
    C, R = M["cap"], M["rate"]
    fails, warns, notes = [], [], []

    # ---- B1 容量: 按时间轴模拟 alloc/free, 常驻量始终计入; 跨迭代存活用两轮展开捕捉 ----
    live = {}                       # buf -> (space, bytes)
    for b, (sp, by) in prog.persistent.items():
        live[b] = (sp, by)
    peak = {sp: 0.0 for sp in C}
    peak_at = {sp: "" for sp in C}

    def _acct(tag):
        cur = {sp: 0.0 for sp in C}
        for b, (sp, by) in live.items():
            n = prog.buf_count.get(b, 1)
            cur[sp] = cur.get(sp, 0.0) + by * n
        for sp, v in cur.items():
            if v > peak[sp]:
                peak[sp], peak_at[sp] = v, tag

    _acct("循环外常驻")
    for rnd in range(2):            # 两轮: 第二轮起点带着上一轮未释放的量
        for op in prog.all_ops():
            if op.alloc:
                sp, nm, by = op.alloc
                live[nm] = (sp, by)
                _acct(f"{op.name or op.pipe}(第{rnd+1}轮 alloc {nm})")
            if op.free and op.free in live:
                del live[op.free]
    for sp, v in peak.items():
        if v > C[sp]:
            fails.append(f"B1_容量: {sp} 峰值 {v/1024:.1f}KB > {C[sp]/1024:.0f}KB (发生在 {peak_at[sp]})")
        elif v > C[sp] - 1024:
            warns.append(f"B1_余量: {sp} 峰值 {v/1024:.1f}KB, 距上限不足 1KB (本项目判过的安全线)")

    # ---- B2 双缓冲: 被跨 pipe 消费的 buffer 若只有 1 份, 下一轮写必须等本轮读完 ----
    producer, consumer = {}, {}
    for op in prog.all_ops():
        for b in op.writes:
            producer.setdefault(b, op.pipe)
        for b in op.reads:
            consumer.setdefault(b, op.pipe)
    for b, pp in producer.items():
        cp = consumer.get(b)
        if cp and cp != pp and prog.buf_count.get(b, 1) < 2:
            warns.append(f"B2_单缓冲: '{b}' 由 {pp} 产、{cp} 消费而只有 1 份 -> 两者跨迭代串行 "
                         f"(本项目实测同类: depthA1 单缓冲慢 17.7%, 正确性不受影响)")

    # ---- S1 旗标守恒 ----
    # ★跨核旗标的 Set 在**对侧核**的程序里, 单侧程序天然失衡 —— 必须成对检查.
    #   本判据第一版就是漏了这点, 自检时三个已实测点全报 FAIL 才发现: 模型不对就先修模型.
    #   真机对应的坑: singleM=2*baseM 那次 AIC 每块发 2 个 flag 而 AIV 每步只等 1 个 -> 507014.
    def _cnt(pr):
        st, wt = {}, {}
        for o in pr.all_ops():
            if o.set_flag:
                st[o.set_flag] = st.get(o.set_flag, 0) + 1
            if o.wait_flag:
                wt[o.wait_flag] = wt.get(o.wait_flag, 0) + 1
        return st, wt
    sets, waits = _cnt(prog)
    if peer is None:
        if sets or waits:
            notes.append("S1: 未给对侧核程序, 旗标守恒**未判** (跨核计数只能成对判). "
                         "传 peer= 后才是硬判据")
    else:
        psets, pwaits = _cnt(peer)
        for f in set(sets) | set(waits) | set(psets) | set(pwaits):
            emit = sets.get(f, 0) * prog.iters + psets.get(f, 0) * peer.iters
            recv = waits.get(f, 0) * prog.iters + pwaits.get(f, 0) * peer.iters
            if emit != recv:
                fails.append(f"S1_旗标不守恒: '{f}' 全程 Set {emit} 次 / Wait {recv} 次 -> 计数漂移. "
                             f"真机表现为 aicore timeout 507014 (本项目撞过: singleM=2*baseM 时"
                             f"AIC 每块发 2 个而 AIV 每步只等 1 个)")

    # ---- S2 流水深度: 生产者可领先的块数必须 >= 消费者的批量, 否则两边轮流干活 ----
    if prog.consumer_batch > 1:
        slots, batch = prog.producer_slots, prog.consumer_batch
        if slots < 2 * batch:
            fails.append(f"S2_流水深度: 生产者槽数 {slots} < 2x 批量 {batch} -> 生产者发满即停等、"
                         f"消费者攒齐才动, 两边轮流干活. **真机实测 (nBatch=4/parallNum=4): 每条流水的"
                         f"忙时都下降 16~63% 而端到端 +19%, 重叠比 1.82x->1.40x**. 需槽数 >= {2*batch}")

    # ---- T1 时间与空闲 (参考级, 非承诺) ----
    busy = {p: 0.0 for p in PIPES}
    for op in prog.all_ops():
        busy[op.pipe] = busy.get(op.pipe, 0.0) + _op_cycles(op, R)
    total = sum(busy.values())
    wall = max(busy.values()) if total else 0.0     # 本核理想全流水化下界
    us = lambda cy: cy * prog.iters / (M["freq_ghz"] * 1e3)
    per = {p: us(v) for p, v in busy.items() if v}
    return {
        "name": prog.name, "fails": fails, "warns": warns, "notes": notes,
        "peak": {sp: v for sp, v in peak.items() if v},
        "pipe_us": per, "sum_us": us(total), "wall_lb_us": us(wall),
        "bind": max(per, key=per.get) if per else None,
        "idle": {p: 1 - v / us(wall) for p, v in per.items()} if wall else {},
    }


def fmt(r: dict) -> str:
    L = [f"=== pipeplan: {r['name']} ==="]
    L.append("容量峰值: " + "  ".join(f"{k} {v/1024:.1f}KB" for k, v in sorted(r["peak"].items())))
    if r["pipe_us"]:
        L.append("")
        L.append(f"{'pipe':<10}{'忙(us)':>10}{'占理想墙':>10}")
        for p, v in sorted(r["pipe_us"].items(), key=lambda x: -x[1]):
            L.append(f"{p:<10}{v:10.0f}{v/r['wall_lb_us']:9.0%}")
        L.append(f"{'流水和':<10}{r['sum_us']:10.0f}")
        from .hwfloor import FLOOR_KIND
        L.append(f"{'理想墙':<10}{r['wall_lb_us']:10.0f}   <- 全流水化下界, 由 {r['bind']} 定; "
                 f"**只作排序参考, 不作收益承诺**")
        #  ★种类标注 (2026-09-18): critpath 类, 随分核/编排变, 不是硬件能力下界。
        L.append(f"{'':<10}★这是「{FLOOR_KIND['critpath'][0]}」—— {FLOOR_KIND['critpath'][1]}")
        idle = [(p, x) for p, x in r["idle"].items() if x > 0.5]
        if idle:
            L.append("")
            for p, x in sorted(idle, key=lambda x: -x[1]):
                L.append(f"★ {p} 在理想墙下空闲 {x:.0%} —— 优化它的端到端收益 ~0 "
                         f"(2026-08-12 实测同类: AIV 空等 63~67% 时, 省 416us 换来端到端 +1.2%)")
    for f in r["fails"]:
        L.append(f"[FAIL] {f}")
    for w in r["warns"]:
        L.append(f"[WARN] {w}")
    L.append("PASS (结构判据)" if not r["fails"] else "FAILED (结构判据)")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# gmfr 现状的三个已实测配置 —— 用作本模块的同构性检查 (探针的规矩: 先证明像被对标的对象).
# 数量取自 runs/nbp_1 runs/nbp_4 runs/nb8p_4 的真机读数, 不是估的.
# ---------------------------------------------------------------------------
def _gmfr_aiv(nbatch: int, parall: int) -> Program:
    """AIV 侧 issue#1: 外层 = 一个 quad(nbatch 个块), 内层 = 行块.
    旗标在**外层**: 攒齐 nbatch 个 SYNC_AIC_TO_AIV 才动, 末尾回同样多的 SYNC_AIV_TO_AIC.
    数量取自真机 (runs/nbp_1 runs/nb8p_4): 每核 784 块, 每 AIV 分半个 m-tile 即 128 行."""
    baseN, rows, tile = 128, 128, 4096
    wide = baseN * nbatch
    rows_per_iter = tile // wide
    p = Program(name=f"gmfr AIV nBatch={nbatch} parallNum={parall}", role="AIV",
                iters=784 // nbatch, inner=rows // rows_per_iter,
                producer_slots=parall, consumer_batch=nbatch,
                buf_count={"vecIn": 2, "vecOut": 2},
                persistent={"tmpBuff": ("UB", 98304), "scaleQ": ("UB", wide * 4 * 2),
                            "perTokQ": ("UB", 4096), "biasQ": ("UB", 512)})
    for _ in range(nbatch):
        p.add_outer(Op("S", "等 AIC 旗标", wait_flag="AIC2AIV"))
    p.add(Op("VMTE2", "读 staging", bytes=rows_per_iter * wide * 4,
             alloc=("UB", "vecIn", tile * 4), writes=("vecIn",)))
    p.add(Op("V", "反量化+combine", cycles=rows_per_iter * wide / 64,
             reads=("vecIn",), free="vecIn", writes=("vecOut",),
             alloc=("UB", "vecOut", tile * 4)))
    p.add(Op("S", "row_index 标量读", cycles=rows_per_iter * 40))
    p.add(Op("MTE3", "散射原子加", bytes=rows_per_iter * wide * 4,
             rate_key=str(wide * 4),          # 每行写出的字节数 -> mte3_bw()
             reads=("vecOut",), free="vecOut"))
    for _ in range(nbatch):
        p.add_outer(Op("S", "回 AIC 旗标", set_flag="AIV2AIC"))
    return p


def _gmfr_aic(nbatch: int, parall: int) -> Program:
    """AIC 侧 issue#1: 外层 = 一个 quad, 内层 = quad 里的 nbatch 个块.
    每块发一个 SYNC_AIC_TO_AIV; 槽满时等一个 SYNC_AIV_TO_AIC."""
    baseM, baseN, k, bK = 256, 128, 1536, 128
    p = Program(name=f"gmfr AIC nBatch={nbatch} parallNum={parall}", role="AIC",
                iters=784 // nbatch, inner=nbatch,
                producer_slots=parall, consumer_batch=nbatch,
                buf_count={"A_L1": 2, "B_L1": 2, "A_L0": 2, "B_L0": 2},
                persistent={"L0C": ("L0C", baseM * baseN * 4)})
    p.add(Op("MTE2", "载 A 面板", bytes=baseM * k, rate_key="MTE2_HIT",
             alloc=("L1", "A_L1", baseM * bK * 4), writes=("A_L1",)))
    p.add(Op("MTE2", "载 B 面板", bytes=k * baseN, rate_key="MTE2_MISS",
             alloc=("L1", "B_L1", bK * baseN * 4), writes=("B_L1",)))
    p.add(Op("MTE1", "A->L0A", bytes=baseM * k, rate_key="MTE1_A",
             alloc=("L0A", "A_L0", baseM * bK), reads=("A_L1",), writes=("A_L0",)))
    p.add(Op("MTE1", "B->L0B", bytes=k * baseN, rate_key="MTE1_B",
             alloc=("L0B", "B_L0", bK * baseN), reads=("B_L1",), writes=("B_L0",)))
    p.add(Op("MAC", "Mmad", macs=baseM * baseN * k, reads=("A_L0", "B_L0")))
    p.add(Op("FIXPIPE", "排空 L0C", bytes=baseM * baseN * 4, set_flag="AIC2AIV"))
    p.add(Op("S", "等回流", wait_flag="AIV2AIC"))
    return p


def build_astationary(baseM=256, baseN=128, bK=128, stepKb=2, depthB1=4, nbatch=4, parall=8,
                      k=1536, n=4096, mtiles_per_core=25) -> Program:
    """**A 面板驻留 L1 的手写 cube 流水** (待实现方案, 2026-08-12 起草).

    与库当前方案的唯一结构差别: A 面板在 L1 里**跨 n 块存活** ——
      外层 = 一个 m-tile (载 A 一次), 内层 = n 方向的 n/baseN 个块 (B 面板流过).
    库做不到这一点: 它的 A 驻留绑在 stepN 上, 而 stepN>1 要求 stepN 个累加器同时活在 L0C,
    128KB 只装得下一个 (实测 stepN=4 时 GM→L1 反涨 91%).

    它**不改变** L1→L0 的字节 (L0C 仍只装一个累加器, A 仍要为每个 n 块重喂进 L0A) ——
    这条是 2026-08-12 用 core/mmplan 的 MTE1 解析式确认过的, 所以 tile 仍取 256/128.

    ★2026-08-12 读 CANN 头文件验出的三条硬约束 (全部已反映在本函数的参数与容量里):
      1. **不必手写 LoadData/Mmad**: 声明 `MatmulType<TPosition::A1, CubeFormat::NZ, int8_t>`
         即走 `CopyCubeInType::FROM_L1` —— 库的 LoadData 直接返回调用者的 L1 地址,
         AllocTensor/ClearLoadData/Destroy 全是空函数 (copy_cube_in_from_l1.h).
      2. **A 在 L1 是 NZ 不是 ND**, 容量按 nz_bytes() 算 (GM→L1 走 CopyND2NZ 随路转换).
      3. **stepKa 无要求** (2026-08-12 二次核对订正): A 在 L1 时走
         `PhyPosIsL1OrUB(A_TYPE::pos)` 分支, k 偏移是 `kInnerIdx * baseK` **线性寻址不回绕**;
         按 stepKa 回绕的那支只在 A 从 GM 来时走. 面板几何来自 `SetSingleShape`:
         axisL1Len = singleCoreM, kAxisL1Len = singleCoreK (InitSplitAParams).
         故 NZ 的 dstNzC0Stride 按**运行时 curSingleM** 算, 不是 baseM.
      4. **depthA1 对 A 失效**: FROM_L1 下 GetCubeInBufferType() 返回 NONE, 主模板全是空体,
         库**不为 A 分配任何 L1**. 这解开了历史死结: 以前 stepKa=12/depthA1=12 被否是因为库要
         384KB 的 A 缓冲, 现在那 384KB 是调用者自己的面板, 不再重复计账.
    """
    nt = n // baseN
    ksteps = k // bK
    p = Program(name=f"A驻留 手写 {baseM}x{baseN} bK{bK} stepKb{stepKb} nBatch{nbatch}",
                role="AIC", iters=mtiles_per_core, inner=nt,
                producer_slots=parall, consumer_batch=nbatch,
                buf_count={"B_L1": 1, "A_L0": 2, "B_L0": 2},   # B_L1 的份数已在 depthB1 里
                persistent={"A_L1": ("L1", nz_bytes(baseM, k, 1)),   # ★NZ 真实字节, 非 rows*cols
                            "L0C": ("L0C", baseM * baseN * 4)})
    # 外层: A 面板载一次 (原方案是每个 n 块载一次, 即 nt 倍)
    p.add_outer(Op("MTE2", "载 A 面板(整 k, 只此一次)", bytes=baseM * k, rate_key="MTE2_MISS",
                   writes=("A_L1",)))
    # 内层: 每个 n 块
    # B 仍由库管: L1 占用 = depthB1 * baseK * baseN (depthB1 是缓冲份数, 已含双缓冲)
    p.add(Op("MTE2", "载 B 面板", bytes=k * baseN, rate_key="MTE2_MISS",
             alloc=("L1", "B_L1", depthB1 * bK * baseN), writes=("B_L1",)))
    for _ in range(ksteps):
        p.add(Op("MTE1", "A->L0A", bytes=baseM * bK, rate_key="MTE1_A",
                 alloc=("L0A", "A_L0", baseM * bK), reads=("A_L1",), writes=("A_L0",)))
        p.add(Op("MTE1", "B->L0B", bytes=bK * baseN, rate_key="MTE1_B",
                 alloc=("L0B", "B_L0", bK * baseN), reads=("B_L1",), writes=("B_L0",)))
        p.add(Op("MAC", "Mmad", macs=baseM * baseN * bK, reads=("A_L0", "B_L0")))
    p.add(Op("FIXPIPE", "排空 L0C", bytes=baseM * baseN * 4, set_flag="AIC2AIV"))
    p.add(Op("S", "等回流", wait_flag="AIV2AIC"))
    return p


def _astationary_aiv(nbatch=4, parall=8, baseN=128, n=4096, mtiles_per_core=25) -> Program:
    """配套 AIV: 结构与现方案相同 (攒批 nbatch), 只是外层轮数按 n 块数算."""
    rows, tile = 128, 4096
    wide = baseN * nbatch
    rpi = tile // wide
    p = Program(name=f"A驻留 AIV nBatch={nbatch}", role="AIV",
                iters=mtiles_per_core * (n // baseN) // nbatch, inner=rows // rpi,
                producer_slots=parall, consumer_batch=nbatch,
                buf_count={"vecIn": 2, "vecOut": 2},
                persistent={"tmpBuff": ("UB", 98304), "scaleQ": ("UB", wide * 4 * 2),
                            "perTokQ": ("UB", 4096), "biasQ": ("UB", 512)})
    for _ in range(nbatch):
        p.add_outer(Op("S", "等 AIC 旗标", wait_flag="AIC2AIV"))
    p.add(Op("VMTE2", "读 staging", bytes=rpi * wide * 4, alloc=("UB", "vecIn", tile * 4),
             writes=("vecIn",)))
    p.add(Op("V", "反量化+combine", cycles=rpi * wide / 64, reads=("vecIn",), free="vecIn",
             writes=("vecOut",), alloc=("UB", "vecOut", tile * 4)))
    p.add(Op("S", "row_index", cycles=rpi * 40))
    p.add(Op("MTE3", "散射原子加", bytes=rpi * wide * 4,
             rate_key=str(wide * 4),          # 每行写出的字节数 -> mte3_bw()
             reads=("vecOut",), free="vecOut"))
    for _ in range(nbatch):
        p.add_outer(Op("S", "回 AIC 旗标", set_flag="AIV2AIC"))
    return p


DEMOS = {
    "gmfr_nbatch1": lambda: (_gmfr_aiv(1, 4), _gmfr_aic(1, 4)),
    "gmfr_nbatch4_pn4": lambda: (_gmfr_aiv(4, 4), _gmfr_aic(4, 4)),
    "gmfr_nbatch4_pn8": lambda: (_gmfr_aiv(4, 8), _gmfr_aic(4, 8)),
    "astationary": lambda: (_astationary_aiv(), build_astationary()),
}


def calibrate(soc: str | None = None) -> int:
    """同构性检查: 三个已实测点, 工具的**结构判据**必须与真机结论同号.

    ★跑的是**这三个点当初测出来的那颗芯片**, 不是本机 —— 换机器后同构性仍应成立。
      型号取自内置标定表的唯一键, 不在代码里写死名字。
    """
    soc = soc or next(iter(MACHINE))
    set_active_soc(soc)          # 嵌套的成本函数没有 soc 形参, 靠上下文
    try:
        expect = {
            "gmfr_nbatch1":     ("PASS", "基线, 真机 aiv_time 4260us"),
            "gmfr_nbatch4_pn4": ("FAIL", "真机 +19%: 每条流水忙时都降而端到端涨, 重叠比 1.82->1.40"),
            "gmfr_nbatch4_pn8": ("PASS", "真机 -10.5%: 重叠比 1.99x"),
        }
        ok = True
        print("=== pipeplan 同构性检查 (三个已实测点) ===\n")
        for k, (want, why) in expect.items():
            aiv, aic = DEMOS[k]()
            r = check(aiv, peer=aic)
            got = "FAIL" if r["fails"] else "PASS"
            mark = "对上" if got == want else "★对不上"
            if got != want:
                ok = False
            print(f"{k:<22} 工具 {got:<5} 期望 {want:<5} {mark}   ({why})")
            for f in r["fails"]:
                print(f"      -> {f.split(':')[0]}")
        print("\n" + ("同构性检查通过, 结构判据可用于设计新方案" if ok else
                      "★对不上: 先修工具, 不要用它设计 (探针的规矩)"))
    finally:
        set_active_soc(None)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", choices=sorted(DEMOS))
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--refs", action="store_true", help="打印硬参数出处表并回源核对平台 ini")
    ap.add_argument("--needs", action="store_true", help="列出需要外部填充(告知/实测)的量")
    ap.add_argument("--advise", action="store_true",
                    help="按 dtype/累加器份数 解出最优 tile 与写出粒度")
    ap.add_argument("--ea", type=float, default=1, help="A 元素字节 (int8=1, fp16/bf16=2)")
    ap.add_argument("--eb", type=float, default=1, help="B 元素字节")
    ap.add_argument("--ec", type=float, default=4, help="累加器元素字节 (int32/fp32=4)")
    ap.add_argument("--ew", type=float, default=4, help="**写出**元素字节 (det 时是 int32=4; 非 combine 输出 bf16=2)")
    ap.add_argument("--live-acc", type=int, default=1, help="同时活在 L0C 的累加器份数 = stepM*stepN*dbL0C")
    ap.add_argument("--n", type=int, default=4096, help="算子的 n, 用于算 nBatch 的整除约束")
    ap.add_argument("--parall-max", type=int, default=8, help="parallNum 上限 (非硬件参数, 见 REFS)")
    a = ap.parse_args()
    if a.needs:
        return print_needs()
    if a.advise:
        t = advise_tile(a.ea, a.eb, a.ec, a.live_acc)
        print("=== 最优 tile (由 MTE1 速率比 + L0C 容量 + 累加器份数 解出) ===")
        print(f"  {t['why']}")
        print(f"\n  {'baseM':>6}{'baseN':>7}{'每MAC周期':>12}{'L0C/份':>9}{'相对最优':>9}{'L0C面积':>9}")
        for m, n, f, kb, rel, util in t["table"]:
            print(f"  {m:>6}{n:>7}{f:>11.3f}u{kb:>8.0f}K{rel:>9.3f}{util:>8.0%}")
        bM, bN = t["best"]
        adm = [d for d in range(1, a.parall_max // 2 + 1) if a.n % (bN * d) == 0]
        gran = [int(bN * d * a.ew) for d in adm]
        g = advise_write_gran(gran)
        print(f"\n=== 写出粒度 (baseN={bN}, 写出元素 {a.ew:g}B, n={a.n}, parallNum<={a.parall_max}) ===")
        print(f"  可选 nBatch {adm} -> 每行写出 {gran} B")
        for gb, bw, frac in g["table"]:
            print(f"  {gb:>6}B  {bw:5.2f} GB/s  天花板的 {frac:4.0%}   nBatch={gb//int(bN*a.ew)}")
        print(f"  -> {g['why']}")
        return 0
    if a.refs:
        return verify_refs()
    if a.calibrate:
        return calibrate()
    if a.demo:
        aiv, aic = DEMOS[a.demo]()
        print(fmt(check(aic, peer=aiv)))
        print()
        print(fmt(check(aiv, peer=aic)))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
