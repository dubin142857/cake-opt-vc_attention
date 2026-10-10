"""Isolated TileLang candidates for the Hopper INT8-QK/FP8-PV experiment.

The original tilelang_expcast backend and all C++ kernels remain the benchmark
controls. These candidates cache quantization scales, compare INT8 MMA vs
WGMMA, and test copy pipelines, PV overlap, packed FP8 decoding, and grouped
reductions. ExpCast FP32 evaluation and online FP32 update order are retained.
"""
from functools import lru_cache

import tilelang_expcast as baseline

NO_WS_VARIANTS = ("cached_no_ws", "cached_no_ws_overlap", "cached_no_ws_int_mass",
                  "cached_no_ws_vec_decode", "cached_no_ws_reduce4", "vec_overlap")
VARIANTS = ("cached", "mma_qk", *NO_WS_VARIANTS)
VEC_DECODE_VARIANTS = ("cached_no_ws_vec_decode", "vec_overlap")
PV_OVERLAP_VARIANTS = ("cached_no_ws_overlap", "vec_overlap")


def compiler_pass_configs(variant):
    import tilelang
    if variant not in VARIANTS:
        raise ValueError(f"Unknown TileLang optimization: {variant}")
    config = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    if variant in NO_WS_VARIANTS:
        # Keep the 128-thread compute CTA without a producer warpgroup.
        # TileLang replaces K/V TMA with cp.async; both GEMMs still use
        # WGMMA on Hopper.
        config[tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED] = True
    return config


def make_primfunc(batch, heads, queries, keys, dim, output_dtype="bfloat16",
                  stages=1, variant="cached"):
    import tilelang.language as T

    if dim not in (64, 128) or stages not in (1, 2):
        raise ValueError("Supported: D64/128, one or two pipeline stages")
    if min(batch, heads, queries, keys) < 1 or output_dtype not in ("float16", "bfloat16"):
        raise ValueError("Expected nonempty FP16/BF16 MHA")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown TileLang optimization: {variant}")
    if variant in NO_WS_VARIANTS and stages != 1:
        raise ValueError("no_ws variants support one stage; two stages spill registers on tail shapes")
    bm, bn = 64, 128
    blocks = (keys + bn - 1) // bn
    qs_len = (queries + bm - 1) // bm * 32
    prelude = (baseline.ROOT / "experiments/expcast/paper_math.cuh").read_text(encoding="utf-8")
    if variant == "cached_no_ws_int_mass":
        # ExpCast emits codes 0..120. Their decoded E4M3 values are integral
        # multiples of 2^-9. Summing 128 codes stays <=2^24 units, so converting
        # the integer sum back to FP32 exactly matches every FP32 sum order.
        prelude += """
__device__ __forceinline__ int vc_opt_prob_units(uint8_t code) {
  const int c = static_cast<int>(code);
  return c < 8 ? c : (((c & 7) + 8) << ((c >> 3) - 1));
}
"""
    if variant in PV_OVERLAP_VARIANTS:
        # wait_wgmma has no shared-buffer read regions. Keep both operands
        # live through the wait so the software pipeline cannot reuse them.
        prelude += """
__device__ __forceinline__ void vc_opt_keep_shared_live(const void* v, const void* p) {
  asm volatile("" : : "l"(v), "l"(p) : "memory");
}
"""
    if variant == "mma_qk":
        # Keep the generic shared-memory K reads live through MMA and an
        # explicit consumer rendezvous. A CTA-wide barrier would deadlock the
        # producer/consumer pipeline. Named barrier 15 is reserved for this
        # helper; the pinned lowering uses named barrier 3 for its consumers.
        prelude += """
__device__ __forceinline__ void vc_mma_finish_k(const void* k) {
  asm volatile("bar.sync 15, 128;" : : "l"(k) : "memory");
}
"""

    @T.prim_func
    def kernel(Q: T.Tensor((batch, heads, queries, dim), T.int8),
               K: T.Tensor((batch, heads, keys, dim), T.int8),
               V: T.Tensor((batch, heads, dim, blocks * bn), T.float8_e4m3fn),
               QS: T.Tensor((batch, heads, qs_len), T.float32),
               KS: T.Tensor((batch, heads, blocks * 4), T.float32),
               VS: T.Tensor((batch, heads, dim), T.float32),
               O: T.Tensor((batch, heads, queries, dim), output_dtype),
               scale: T.float32):
        T.func_attr({"global_symbol": "vc_tilelang_opt"})
        with T.Kernel(T.ceildiv(queries, bm), heads, batch,
                      threads=128, prelude=prelude) as (bx, h, b):
            q = T.alloc_shared((bm, dim), T.int8)
            k = T.alloc_shared((bn, dim), T.int8)
            if variant == "mma_qk":
                # A fragment B operand forces MMA lowering for this GEMM only.
                # Globally disabling WGMMA would also change the FP8 PV path.
                k_register = T.alloc_fragment((bn, dim), T.int8)
            v = T.alloc_shared((dim, bn), T.float8_e4m3fn)
            # Broadcast fragments follow the score layout chosen by the GEMM.
            # Query scales are loop-invariant; make K scale reuse explicit.
            # NVRTC already merges many scalar reads in the original backend.
            query_scale = T.alloc_fragment((bm,), T.float32)
            key_scale = T.alloc_fragment((bn,), T.float32)
            dots = T.alloc_fragment((bm, bn), T.int32)
            logits = T.alloc_fragment((bm, bn), T.float32)
            if variant == "cached_no_ws_reduce4":
                # g%4 keeps the four-lane ownership of the QK fragment. Reduce
                # eight values locally, then four partials and a four-lane
                # AllReduce, with both query rows kept separate.
                grouped_values = T.alloc_fragment((bm, 16, 8), T.float32)
                grouped_partial = T.alloc_fragment((bm, 16), T.float32)
            prob = T.alloc_fragment((bm, bn), T.float8_e4m3fn)
            p_shared = T.alloc_shared((bm, bn), T.float8_e4m3fn)
            pv = T.alloc_fragment((bm, dim), T.float32)
            accum = T.alloc_fragment((bm, dim), T.float32)
            maximum = T.alloc_fragment((bm,), T.float32)
            previous = T.alloc_fragment((bm,), T.float32)
            alpha = T.alloc_fragment((bm,), T.float32)
            mass = T.alloc_fragment((bm,), T.float32)
            if variant == "cached_no_ws_int_mass":
                mass_units = T.alloc_fragment((bm,), T.int32)
            denom = T.alloc_fragment((bm,), T.float32)
            T.copy(Q[b, h, bx * bm:(bx + 1) * bm, :], q)
            for i in T.Parallel(bm):
                row = bx * bm + i
                query_scale[i] = QS[b, h, (row // 16) * 8 + row % 8]
            T.fill(maximum, -5e6)
            T.clear(accum)
            T.clear(denom)
            for tile in T.Pipelined(blocks, num_stages=stages):
                T.copy(K[b, h, tile * bn:(tile + 1) * bn, :], k)
                T.copy(V[b, h, :, tile * bn:(tile + 1) * bn], v)
                for j in T.Parallel(bn):
                    key_scale[j] = KS[b, h, tile * 4 + (j % 8) // 2]
                if variant == "mma_qk":
                    T.copy(k, k_register)
                    T.gemm(q, k_register, dots, transpose_B=True, clear_accum=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    T.evaluate(T.call_extern("handle", "vc_mma_finish_k", T.access_ptr(k, "r")))
                else:
                    T.gemm(q, k, dots, transpose_B=True, clear_accum=True,
                           policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(bm, bn):
                    dequant = T.call_extern("float32", "vc_paper_mul", query_scale[i], key_scale[j])
                    if tile == blocks - 1:
                        score = T.call_extern("float32", "vc_paper_mul", T.cast(dots[i, j], T.float32), dequant)
                        logits[i, j] = T.call_extern("float32", "vc_paper_mul", score, scale)
                    else:
                        factor = T.call_extern("float32", "vc_paper_mul", scale, dequant)
                        logits[i, j] = T.call_extern("float32", "vc_paper_mul", T.cast(dots[i, j], T.float32), factor)
                    if tile * bn + j >= keys:
                        logits[i, j] = -T.infinity(T.float32)
                T.copy(maximum, previous)
                if variant == "cached_no_ws_reduce4":
                    for i, g, r in T.Parallel(bm, 16, 8):
                        grouped_values[i, g, r] = logits[i, (g // 4) * 32 + (r // 2) * 8 + (g % 4) * 2 + r % 2]
                    T.reduce_max(grouped_values, grouped_partial, dim=2)
                    T.reduce_max(grouped_partial, maximum, dim=1, clear=False)
                else:
                    T.reduce_max(logits, maximum, dim=1, clear=False)
                for i in T.Parallel(bm):
                    alpha[i] = T.call_extern("float32", "vc_paper_alpha", previous[i], maximum[i])
                for i, j in T.Parallel(bm, bn):
                    code = T.call_extern("uint8", "vc_paper_code", logits[i, j], maximum[i])
                    prob[i, j] = T.reinterpret(T.float8_e4m3fn, code)
                    if variant == "cached_no_ws_int_mass":
                        dots[i, j] = T.call_extern("int32", "vc_opt_prob_units", code)
                    elif variant not in VEC_DECODE_VARIANTS:
                        logits[i, j] = T.cast(prob[i, j], T.float32)
                if variant in VEC_DECODE_VARIANTS:
                    # Bulk copy uses packed FP8x2 conversion into FP32 without
                    # changing probability bytes or the mass reduction order.
                    T.copy(prob, logits, coalesced_width=2)
                if variant in PV_OVERLAP_VARIANTS:
                    T.copy(prob, p_shared)
                    T.wgmma_gemm(p_shared, v, pv, transpose_B=True, clear_accum=True,
                                 policy=T.GemmWarpPolicy.FullRow)
                    T.reduce_sum(logits, mass, dim=1)
                    for i in T.Parallel(bm):
                        denom[i] = T.call_extern("float32", "vc_paper_add",
                            T.call_extern("float32", "vc_paper_mul", denom[i], alpha[i]), mass[i])
                    for i, d in T.Parallel(bm, dim):
                        accum[i, d] = T.call_extern("float32", "vc_paper_mul", accum[i, d], alpha[i])
                    T.wait_wgmma(0)
                    # The explicit register fence covers only the per-thread
                    # fragment, rather than all bm*dim logical elements.
                    T.warpgroup_fence_operand(pv, num_regs=bm * dim // 128)
                    T.evaluate(T.call_extern("handle", "vc_opt_keep_shared_live",
                                            T.access_ptr(v, "r"), T.access_ptr(p_shared, "r")))
                    for i, d in T.Parallel(bm, dim):
                        accum[i, d] = T.call_extern("float32", "vc_paper_add", accum[i, d], pv[i, d])
                else:
                    if variant == "cached_no_ws_int_mass":
                        T.reduce_sum(dots, mass_units, dim=1)
                        for i in T.Parallel(bm):
                            mass[i] = T.cast(mass_units[i], T.float32) * T.float32(0.001953125)
                    elif variant == "cached_no_ws_reduce4":
                        for i, g, r in T.Parallel(bm, 16, 8):
                            grouped_values[i, g, r] = logits[i, (g // 4) * 32 + (r // 2) * 8 + (g % 4) * 2 + r % 2]
                        T.reduce_sum(grouped_values, grouped_partial, dim=2)
                        T.reduce_sum(grouped_partial, mass, dim=1)
                    else:
                        T.reduce_sum(logits, mass, dim=1)
                    T.copy(prob, p_shared)
                    T.gemm(p_shared, v, pv, transpose_B=True, clear_accum=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    for i in T.Parallel(bm):
                        denom[i] = T.call_extern("float32", "vc_paper_add",
                            T.call_extern("float32", "vc_paper_mul", denom[i], alpha[i]), mass[i])
                    for i, d in T.Parallel(bm, dim):
                        accum[i, d] = T.call_extern("float32", "vc_paper_add",
                            T.call_extern("float32", "vc_paper_mul", accum[i, d], alpha[i]), pv[i, d])
            for i, d in T.Parallel(bm, dim):
                O[b, h, bx * bm + i, d] = T.call_extern("float32", "vc_paper_mul",
                    accum[i, d] / denom[i], VS[b, h, d])
    return kernel


@lru_cache(maxsize=64)
def compile_kernel(batch, heads, queries, keys, dim, dtype, stages=1, variant="cached"):
    import tilelang
    if tilelang.__version__.split("+")[0] != "0.1.15":
        raise RuntimeError("This experiment pins TileLang 0.1.15")
    return tilelang.compile(
        make_primfunc(batch, heads, queries, keys, dim, dtype, stages, variant),
        out_idx=[6], target={"kind": "cuda", "arch": "sm_90a"},
        pass_configs=compiler_pass_configs(variant))


def forward_prepared(state, stages=1, variant="cached"):
    import torch
    from profile_int8_h200 import require_int8_state
    require_int8_state(state)
    if state.get("v_layout") != baseline.V_LAYOUT:
        raise ValueError("TileLang requires natural V from tilelang_expcast.prepare")
    q, k = state["q"], state["k"]
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError("Hopper required; no CPU/FP16-matmul fallback")
    if not state.get("expcast", False):
        raise ValueError("This TileLang optimization requires ExpCast enabled")
    if state["block_values"] or state["means"].numel() or state["permutation"] is not None:
        raise ValueError("TileLang experiment requires V smoothing/grouping OFF")
    if any(not state[n].is_contiguous() or state[n].device != q.device
           for n in ("q", "k", "v", "qs", "ks", "vs")):
        raise ValueError("Expected contiguous, same-device prepared tensors")
    b, h, m, d = q.shape
    dtype = "float16" if state["output_type"] == 0 else "bfloat16"
    with torch.cuda.device(q.device):
        fn = compile_kernel(b, h, m, k.shape[2], d, dtype, stages, variant)
        return fn(*(state[n] for n in ("q", "k", "v", "qs", "ks", "vs")), state["scale"])
