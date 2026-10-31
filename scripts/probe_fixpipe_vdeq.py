#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fixpipe 逐通道反量化 (VDEQF16) 的**形参**数值探针 —— 不是跑算子。

★为什么要它 (F183 定的次序): 两处本机文档对那条 uint64 的编码**互相矛盾**:
    tbe/tik/debug/intrinsic_cube.py:121
        "bits[0:31] is deq value of fp32, bits[32:35] right move size, bits[36] mcd control"
    asc/impl/basic_api/dav_c220/kernel_operator_fixpipe_v2_impl.h (标量 DEQF16 那一支)
        "deq factor of uint64 bits describe: bits[31:13] is deq value of fp32"
  文档是一条证据, 不是结论 (E23)。冲突只能由数值坐实。

★本探针同时回答三件事, 每件都配一个"会让它无声走错"的替代解释:
  C1 编码: 按"fp32 位模式零扩展成 uint64"构造 -> 结果必须与 NumPy 模型**逐位相同**。
  C2 施加方向: scale **逐列各不相同** (第 0 列与第 1 列差一个量级)。
     若硬件其实按标量或按行施加, C1 会当场崩掉 —— 而若所有列用同一个 scale,
     这两种解释给出同一答案, 样例就钉不住契约 (E6)。
  C3 舍入: 与 NumPy 的 float32 -> float16 (就近偶数) 逐位比, **不设容差**。

★对照组: 同一次 QK 还按现行的 NoQuant 出一份 int32。它必须与 NumPy 的 int8 矩阵乘
  逐位相同 —— 它既是 C1/C3 的参考, 也是"设 FPC 没有破坏别的东西"的旁证。

★取值刻意用小整数且 scale 取 2 的幂: int32 结果在 fp16 内**可精确表示**,
  于是"不逐位相同"只可能是形参或编码错, 不会是舍入 —— 这是本探针能下判决的依据。
  另跑一组非 2 的幂的 scale 专门验 C3 的舍入。
"""
import argparse, json, os, sys
import numpy as np


def _encode(scale: np.ndarray) -> np.ndarray:
    """按待验假设构造逐通道项: fp32 的位模式**零扩展**成 uint64。"""
    return scale.astype(np.float32).view(np.uint32).astype(np.uint64)


def _trunc_mantissa(scale: np.ndarray, keep_low_zero_bits: int) -> np.ndarray:
    """把 fp32 的低 n 位尾数清零, 用来检验'硬件只用 bits[31:n]'这一类假设。"""
    #  ★先在 Python 的任意精度整数里移位再**截到 32 位**, 不要让 numpy 去转一个越界值:
    #    `np.uint32(0xFFFFFFFF << 13)` 会告警 (将来直接报错), 而结果本来就该是截断后的。
    m = np.uint32((0xFFFFFFFF << keep_low_zero_bits) & 0xFFFFFFFF)
    return (scale.astype(np.float32).view(np.uint32) & m).view(np.float32)


def _round_mantissa(scale: np.ndarray, keep_low_zero_bits: int) -> np.ndarray:
    """同上, 但对被丢掉的那几位做**就近偶数**进位, 而不是直接截断。"""
    u = scale.astype(np.float32).view(np.uint32).astype(np.uint64)
    half = np.uint64(1) << np.uint64(keep_low_zero_bits - 1)
    mask = np.uint64((1 << keep_low_zero_bits) - 1)
    low = u & mask
    kept = u >> np.uint64(keep_low_zero_bits)
    up = (low > half) | ((low == half) & ((kept & np.uint64(1)) == np.uint64(1)))
    kept = kept + up.astype(np.uint64)
    return (kept << np.uint64(keep_low_zero_bits)).astype(np.uint32).view(np.float32)


def _rtz_f16(x32: np.ndarray) -> np.ndarray:
    """fp32 -> fp16 的**截断**(向零)转换, 作为 NumPy 就近偶数之外的对照模型。"""
    near = x32.astype(np.float16)
    over = np.abs(near.astype(np.float32)) > np.abs(x32)
    out = near.copy()
    step = np.nextafter(near[over].astype(np.float32),
                        np.zeros(1, dtype=np.float32)).astype(np.float16)
    out[over] = step
    return out


def _candidate_models(ref_i32: np.ndarray, scale: np.ndarray) -> dict:
    """候选模型族。探针的职责不是只说'不符', 是说**哪一个**成立 (E23)。"""
    f = ref_i32.astype(np.float32)
    out = {
        "f32_scale__rte": (f * scale[None, :]).astype(np.float16),
        "f32_scale__rtz": _rtz_f16(f * scale[None, :]),
    }
    for n in (13, 12, 14):
        out[f"trunc{n}_scale__rte"] = (f * _trunc_mantissa(scale, n)[None, :]).astype(np.float16)
        out[f"round{n}_scale__rte"] = (f * _round_mantissa(scale, n)[None, :]).astype(np.float16)
    #  另一支解释: 先把 int32 转 fp16 再乘。大值会整数不精确, 故它若成立会很显眼。
    out["i32_to_f16_then_mul"] = (f.astype(np.float16).astype(np.float32)
                                  * scale[None, :]).astype(np.float16)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=3, help="物理卡号; 0 卡是 CI runner, 禁用")
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--output", default="results/fixpipe_vdeq_probe.json")
    a = ap.parse_args()
    if a.device == 0:
        print("拒绝: 0 卡是管理员部署 CI runner 用的", file=sys.stderr)
        return 2

    import torch, torch_npu  # noqa: F401
    import vc_attention_ascend
    vc_attention_ascend.load_native()
    torch.npu.set_device(a.device)
    dev = f"npu:{a.device}"

    M, N, D = 64, 128, 128
    rng = np.random.default_rng(a.seed)
    #  小整数: |S| <= 128*7*7 = 6272, fp16 整数精确区是 |x| <= 2048,
    #  故再乘一个 2 的幂的小 scale 把它压回精确区 —— 见下面 scale 的取法。
    q = rng.integers(-7, 8, size=(M, D)).astype(np.int8)
    k = rng.integers(-7, 8, size=(N, D)).astype(np.int8)
    ref_i32 = q.astype(np.int32) @ k.astype(np.int32).T          # [M, N]

    cases = []
    #  第一组 (判 C1/C2): scale 是 2 的幂且**逐列不同**, 结果在 fp16 内精确。
    #  第 0 列 2^-10, 第 1 列 2^-6 —— 差一个量级, 标量解释无处藏身。
    pw = np.array([2.0 ** -(10 - (j % 5) * 1) for j in range(N)], dtype=np.float32)
    pw[0], pw[1] = np.float32(2.0 ** -10), np.float32(2.0 ** -6)
    cases.append(("pow2_perchannel", pw, True))
    #  第二组 (判 C3): 非 2 的幂, 必然要舍入; 与 NumPy 的 fp32->fp16 逐位比。
    rnd = (rng.random(N).astype(np.float32) * np.float32(0.01) + np.float32(1e-4))
    cases.append(("nonpow2_perchannel", rnd, False))

    tq = torch.from_numpy(q).to(dev)
    tk = torch.from_numpy(k).to(dev)
    report, ok = {}, True
    for name, scale, exact_expected in cases:
        deq_u64 = _encode(scale)
        tdeq = torch.from_numpy(deq_u64.view(np.int64)).to(dev)   # 只搬位, 不解释
        #  标量路径的两个值: 上半区用 scale[0], 下半区用 scale[1] ——
        #  两者在第一组里差一个量级, 所以"两半区搞反"或"只生效一个"都会当场显形。
        sc_a, sc_b = np.float32(scale[0]), np.float32(scale[1])
        tdeqs = torch.from_numpy(_encode(np.array([sc_a, sc_b],
                                                  dtype=np.float32)).view(np.int64)).to(dev)
        s_i32_t, s_f16_t, s_sc1_t, s_sc2_t = torch.ops.vc_ascend.fixpipe_vdeq_probe(
            tq, tk, tdeq, tdeqs)
        s_i32 = s_i32_t.cpu().numpy()
        s_f16 = s_f16_t.cpu().numpy()
        s_sc1 = s_sc1_t.cpu().numpy()
        s_sc2 = s_sc2_t.cpu().numpy()

        #  对照组: NoQuant 那条出口
        ctrl_ok = bool(np.array_equal(s_i32, ref_i32))
        #  模型: int32 -> fp32 乘逐通道 scale -> 舍成 fp16
        model = (ref_i32.astype(np.float32) * scale[None, :]).astype(np.float16)
        same = bool(np.array_equal(s_f16, model))
        #  C2 的专项: 若硬件其实只认一个标量 (取第 0 项), 模型会变成下面这个。
        #  它必须**不**成立 —— 否则这组样例分不开"逐通道"与"标量"。
        scalar_model = (ref_i32.astype(np.float32) * scale[0]).astype(np.float16)
        distinguishes = not bool(np.array_equal(model, scalar_model))
        scalar_matches = bool(np.array_equal(s_f16, scalar_model))

        bad = int((s_f16 != model).sum())
        #  ★主假设不成立时, 报**哪一个**候选模型成立, 而不是只报"不符"。
        fits = {k: int((s_f16 != v).sum()) for k, v in _candidate_models(ref_i32, scale).items()}
        exact_models = sorted(k for k, v in fits.items() if v == 0)
        #  ---- C4 标量 DEQF16 的编码 (全 64 行一个标量) ----
        #  ★它与向量路径**是两条**: 走 set_quant_pre 这个另一个 SPR, 不碰 FPC。
        #    那句有歧义的 "bits[31:13]" 恰恰长在标量那一支, 所以编码不能从向量那条推。
        sc1_fits = {k: int((s_sc1 != v).sum())
                    for k, v in _candidate_models(ref_i32,
                                                  np.full(N, sc_a, np.float32)).items()}
        sc1_exact = sorted(k for k, v in sc1_fits.items() if v == 0)
        #  ---- C5 M 方向半区拆分 (上半 sc_a, 下半 sc_b, 各发一次 fixpipe) ----
        #  ★待验的是两件事: L0C 的 M 偏移 (每分形内 m_off*16 个元素) 与
        #    "srcStride 仍传 MBLK 而不是 m_size"。任一错都会让某个半区整片不对。
        half = M // 2
        m_top = (ref_i32[:half].astype(np.float32)
                 * _trunc_mantissa(np.full(N, sc_a, np.float32), 13)[None, :]).astype(np.float16)
        m_bot = (ref_i32[half:].astype(np.float32)
                 * _trunc_mantissa(np.full(N, sc_b, np.float32), 13)[None, :]).astype(np.float16)
        top_ok = bool(np.array_equal(s_sc2[:half], m_top))
        bot_ok = bool(np.array_equal(s_sc2[half:], m_bot))
        #  负例: 若两半区其实用了同一个标量, 下半会等于"用 sc_a 算"的结果。
        bot_if_same = (ref_i32[half:].astype(np.float32)
                       * _trunc_mantissa(np.full(N, sc_a, np.float32), 13)[None, :]).astype(np.float16)
        halves_differ = not bool(np.array_equal(m_bot, bot_if_same))
        bot_used_wrong_scalar = bool(np.array_equal(s_sc2[half:], bot_if_same))
        report[name] = {
            "candidate_model_mismatch_counts": fits,
            "models_bit_exact": exact_models,
            "scalar_models_bit_exact": sc1_exact,
            "scalar_mismatch_counts": sc1_fits,
            "msplit_top_ok": top_ok,
            "msplit_bottom_ok": bot_ok,
            "msplit_case_can_distinguish": halves_differ,
            "msplit_bottom_used_the_other_scalar": bot_used_wrong_scalar,
            "control_int32_bitexact": ctrl_ok,
            "vdeq_fp16_bitexact": same,
            "mismatch_count": bad,
            "sample_got": [float(x) for x in s_f16[0, :4]],
            "sample_model": [float(x) for x in model[0, :4]],
            "case_can_distinguish_perchannel_from_scalar": distinguishes,
            "matches_scalar_model_instead": scalar_matches,
            "exact_expected": exact_expected,
        }
        ok = (ok and ctrl_ok and distinguishes and (not scalar_matches) and bool(exact_models)
              and bool(sc1_exact) and top_ok and bot_ok and halves_differ
              and (not bot_used_wrong_scalar))

    #  ★C1 只能由 **2 的幂** 那组来判: 那一组的乘积在 fp16 内精确, 舍入是恒等,
    #    于是"逐位相同"只检验编码。拿非 2 的幂那组去判 C1 会把编码与舍入混为一谈 ——
    #    第一版就是这么写的, 当场把一个**已经成立**的编码判成了不成立。
    report["verdict"] = {
        "C1_encoding_is_fp32_bits_zero_extended": report["pow2_perchannel"]["vdeq_fp16_bitexact"],
        "C2_scale_is_per_channel_along_N": all(
            report[n]["case_can_distinguish_perchannel_from_scalar"]
            and not report[n]["matches_scalar_model_instead"] for n, _, _ in cases),
        #  ★这一条**故意**保留为 False: 它问的是"像不像 NumPy 的直接 fp32->fp16",
        #    答案是不像 —— 真正的语义由下面那条 `C3_models_bit_exact` 给出。
        #    把它改名成"通过"会掩盖掉这个区别, 所以留着, 并在这里写明。
        "C3_is_plain_numpy_f32_to_f16": report["nonpow2_perchannel"]["vdeq_fp16_bitexact"],
        "C3_models_bit_exact": report["nonpow2_perchannel"]["models_bit_exact"],
        "C4_scalar_deqf16_models_bit_exact":
            report["nonpow2_perchannel"]["scalar_models_bit_exact"],
        "C5_m_split_two_scalars_works": all(
            report[n]["msplit_top_ok"] and report[n]["msplit_bottom_ok"]
            and report[n]["msplit_case_can_distinguish"] for n, _, _ in cases),
        "all_pass": ok,
    }
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    for n, _, _ in cases:
        r = report[n]
        print(f"[{n}] 对照 int32 逐位 {r['control_int32_bitexact']} · "
              f"VDEQ fp16 逐位 {r['vdeq_fp16_bitexact']} (不符 {r['mismatch_count']} 个) · "
              f"本例能分开逐通道/标量 {r['case_can_distinguish_perchannel_from_scalar']}")
        print(f"    逐位相同的候选模型: {r['models_bit_exact'] or '无'}")
        print(f"    标量 DEQF16 逐位相同的模型: {r['scalar_models_bit_exact'] or '无'}")
        print(f"    M 半区拆分: 上半 {r['msplit_top_ok']} · 下半 {r['msplit_bottom_ok']} · "
              f"本例能分开两半区 {r['msplit_case_can_distinguish']}"
              + ("  ★下半用错了另一个标量" if r["msplit_bottom_used_the_other_scalar"] else ""))
        if not r["vdeq_fp16_bitexact"]:
            print(f"    实得 {r['sample_got']}")
            print(f"    主假设 {r['sample_model']}")
            top = sorted(r["candidate_model_mismatch_counts"].items(), key=lambda kv: kv[1])[:4]
            print(f"    各候选不符个数(少到多): {top}")
    print(json.dumps(report["verdict"], ensure_ascii=False))
    judged = {k: v for k, v in report["verdict"].items()
              if k not in ("all_pass", "C3_is_plain_numpy_f32_to_f16")}
    n_ok = sum(1 for v in judged.values() if v is True or (isinstance(v, list) and v))
    n_all = len(judged)
    print("结论:", f"{n_all} 条判据全过" if ok
          else f"**未过** ({n_ok}/{n_all}) —— 形参/编码假设被证伪, 见上面的实得与模型")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
