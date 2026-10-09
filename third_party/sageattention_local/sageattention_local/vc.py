"""Sage INT8 QK + optional VC V grouping/smoothing and ExpCast.

Dense noncausal MHA, D=64/128, eager inference on Hopper's default stream.
The prepared native core uses the current stream; upstream V preprocessing
requires the default stream. Callers of the private split API manage dependencies.
Each call recomputes grouping; this entry has no denoising-step layout cache.
"""
import math
import torch


# vc15 fuses final four-byte assembly into vc14's paired conversion.
# The vc13 shared-copy schedules remain explicit experiments after regression.
DEFAULT_EXPCAST_KERNEL = "vc15"
UNIFIED_MODES = {"vc31_unified": 38, "vc32_full_pv": 39, "vc32_packed_mean": 40}
DESCRIPTOR_CANDIDATES = {"vc30_desc": 37}
STREAM_CONVERT_CANDIDATES = {"vc28_stream_i2f": 35, "vc29_interleave_rows": 36}
REGISTER_CANDIDATES = {"vc18_r128": 128, "vc18_r144": 144, "vc18_r192": 192,
                       "vc18_pv64_r128": 128, "vc18_pv64_r144": 144, "vc18_pv64_r168": 168}
PIPELINE_CANDIDATES = {
    "vc19_k": {"k_buffers": 2, "v_buffers": 1, "early_tma_prefetch": 1},
    "vc19_v": {"k_buffers": 1, "v_buffers": 2, "early_tma_prefetch": 1},
    "vc19_kv": {"k_buffers": 2, "v_buffers": 2, "early_tma_prefetch": 1},
    "vc19_kv_late": {"k_buffers": 2, "v_buffers": 2, "early_tma_prefetch": 0},
}
MASS_CANDIDATES = {"vc25_mass_serial": 7, "vc25_mass_overlap": 8}
DUAL_QUERY_CANDIDATES = {"vc26_shared": 1, "vc26_stagger": 2}
ENCODING_CANDIDATES = {"vc24_bitpack": 6}
STATE_CANDIDATES = {"vc23_maxstate": 4, "vc23_cached_bias": 5}
WORK_CANDIDATES = {"vc22_intmax": 1, "vc22_skip": 2, "vc22_both": 3}
HISTORY_CANDIDATES = {"vc21_history_r128": 128, "vc21_history_r168": 168}
COMPUTE_CANDIDATES = {"vc20_pv_qk": 1, "vc20_serial": 2, "vc20_qk_pv": 3}
SAGE_LAYOUT_MODES = {"vc16": 1, "vc16_early_v": 2, "vc16_pack_first": 3,
                     "vc16_pack_early": 4, "vc16_serial": 5,
                     "vc17_g1": 6, "vc17_g4": 7, "vc17_c4": 8, "vc17_direct": 9,
                     "vc18_r128": 10, "vc18_r144": 11, "vc18_r192": 12,
                     "vc18_pv64_r128": 13, "vc18_pv64_r144": 14, "vc18_pv64_r168": 15,
                     **{mode: i+16 for i,mode in enumerate(PIPELINE_CANDIDATES)},
                     **{mode: schedule+19 for mode,schedule in COMPUTE_CANDIDATES.items()},
                     **{mode: i+23 for i,mode in enumerate(HISTORY_CANDIDATES)},
                     **{mode: i+24 for mode,i in WORK_CANDIDATES.items()},
                     **{mode: i+24 for mode,i in STATE_CANDIDATES.items()},
                     **{mode: i+24 for mode,i in ENCODING_CANDIDATES.items()},
                     **{mode: i+24 for mode,i in MASS_CANDIDATES.items()},
                     **{mode: i+32 for mode,i in DUAL_QUERY_CANDIDATES.items()},
                     **STREAM_CONVERT_CANDIDATES, **DESCRIPTOR_CANDIDATES, **UNIFIED_MODES}


def _extension():
    from . import _vc_sm90
    if (getattr(_vc_sm90, "sage_vc_api_version", None) != 1 or
            getattr(_vc_sm90, "sage_vc_kernel_version", None) != 30 or
            getattr(_vc_sm90, "sage_vc_preprocess_version", None) != 1):
        raise RuntimeError("Rebuild sageattention-local with bash scripts/build_sage_local.sh")
    return _vc_sm90


def sage_vc_build_info():
    from .local_info import local_build_info
    return {**local_build_info(), "vc_extension": _extension().__file__,
            "kernel_math_modified": True,
            "algorithm": "Sage per-thread INT8 QK + VC block V residual/mean + ExpCast",
            "qk_hadamard": False, "v_block": 128, "mean_dtype": "float16",
            "stream_convert_candidate": {"mode": "vc28_stream_i2f", "baseline": "vc22_both",
                                         "scope": "INT32-to-FP32 conversion inside encoding groups; unchanged probability bytes and FP32 sums; reported H200 full API 1.0156x over vc22"},
            "interleaved_encode_candidate": {"mode": "vc29_interleave_rows", "baseline": "vc28_stream_i2f",
                                             "scope": "interleave two independent row conversion/encoding chains; same P bytes and FP32 sums; reported H200 full/core API about 1.2% slower than vc28"},
            "full_pv_candidates": {"modes": ["vc32_full_pv", "vc32_packed_mean"], "baseline": "vc31_unified",
                                   "scope": "channel V, full N128 PV; optional batched packed-mean loads (overlap not assumed); identical quantization and two-level FP32 accumulation; H200 speed unverified"},
            "unified_candidate": {"mode": "vc31_unified", "baseline": "vc28_stream_i2f",
                                  "scope": "vc28 probability scheduling plus channel V block means; measured vc31 baseline, D128 smoothed PV uses N64 fragments"},
            "descriptor_candidate": {"mode": "vc30_desc", "baseline": "vc28_stream_i2f",
                                     "scope": "reuse aligned Q/K/V WGMMA descriptor anchors; reported H200 full/core slower than vc28; retain vc28 baseline"},
            "clustering": "8 clusters, 2 Lloyd iterations, stable order; fresh every call",
            "grouping_modes": ["fused", "legacy"], "default_grouping_mode": "fused",
            "grouping_implementation": "fused assignment/statistics for <=8 effective clusters; legacy otherwise",
            "grouping_rounding": "FP32 reduction order changes; labels near boundaries can differ",
            "sage_vc_preprocess_version": _extension().sage_vc_preprocess_version,
            "paper_equivalent": False, "sage_vc_api_version": 1,
            "default_kernel_mode": "optimized", "default_expcast_kernel": DEFAULT_EXPCAST_KERNEL, "kernel_modes": ["optimized", "vc15", "vc14", "vc13_overlap", "vc13_serial", "vc12", "vc11", "vc10", "vc9", "vc7", "vc6", "vc4", *SAGE_LAYOUT_MODES],
            "sage_layout_candidates": {"modes": SAGE_LAYOUT_MODES, "scope": "v_smooth=False except unified vc31/vc32 modes; pack-first modes require expcast=False; vc17 modes require expcast=True; default remains vc15"},
            "probability_candidates": "vc17: group 1/4 fragments, four FP32 chains, or exact direct FP32 decode; vc16 body and unchanged encoding/history",
            "mass_candidates": {"modes": MASS_CANDIDATES, "baseline": "vc22_both", "scope": "completed P kept in registers; previous denominator decoded before/during next QK in source; compiler may merge schedules; no scratch or V smoothing; reported H200 full API did not improve"},
            "dual_query_candidates": {"modes": DUAL_QUERY_CANDIDATES, "baseline": "vc22_both (ExpCast) / vc16 (control)", "scope": "two Q64 warpgroups per CTA share K128/V128; optional QK handoff; original Q64 scales, P encoding and FP32 accumulation; no V smoothing; H200 correctness and speed unverified"},
            "encoding_candidates": {"modes": ENCODING_CANDIDATES, "baseline": "vc22_both", "scope": "exact FP32 bit-bias RNE encoding; unchanged decoded-P FP32 sum; no V smoothing; measured slower than vc22_both on reported H200 workload"},
            "max_state_candidates": {"modes": STATE_CANDIDATES, "baseline": "vc22_both", "register_cap": 168,
                                     "scope": "reuse warp max-update decision; optionally cache affine row bias; v_smooth=False; no full-API improvement over vc22_both in reported H200 test"},
            "work_candidates": {"modes": WORK_CANDIDATES, "baseline": "vc16", "register_cap": 168,
                                "scope": "ExpCast integer maximum / skip identity history; N128 PV, unchanged P and FP32 recurrence; vc22_both measured 1.037x over Sage full API on reported H200 workload"},
            "history_candidates": {"modes": HISTORY_CANDIDATES, "baseline": "vc16", "head_dim": 128,
                                   "history_shared_bytes": 14336,
                                   "scope": "28 of 64 FP32 history values per thread in shared memory, two N64 PV fragments; no V smoothing; tested slower than vc16 on the reported H200 workload"},
            "compute_candidates": {"modes": COMPUTE_CANDIDATES, "baseline": "vc16", "register_cap": 224,
                                   "scope": "PV(i) and QK(i+1) joint group versus serial, single K/V buffers, unchanged vc16 math; no V smoothing; tested slower than vc16 on the reported H200 workload"},
            "pipeline_candidates": {"modes": PIPELINE_CANDIDATES, "baseline": "vc16",
                                    "scope": "K/V ping-pong TMA buffers; vc16 arithmetic, cap 168, N128 PV at D128; no V smoothing; H200 speed unverified"},
            "register_candidates": {"limits": REGISTER_CANDIDATES, "baseline": "vc16 (168)",
                                    "scope": "control and ExpCast, v_smooth=False; vc16 register caps and optional two N64 PV fragments for D128; two-level FP32 accumulation; unmeasured H200 speed"},
            "expcast_probability": "paired saturating u8 RNE/packing + decoded-P FP32 reduction",
            "expcast_interleaving": "vc15: encode two K fragments, issue packed decode, then consume FP32 values",
            "probability_scratch_bytes_per_block": 0,
            "expcast_conversion": "ExpCast-only paired RNE with fused four-byte assembly; exact vc14 P bytes",
            "experimental_probability_scratch_bytes_per_block": 8192,
            "experimental_kernel_modes": ["vc13_serial", "vc13_overlap", *SAGE_LAYOUT_MODES],
            "expcast_numerical_contract": "expcast-only folds coefficients in FP32 before FMA/RNE; byte rounding may differ near boundaries; denominator uses decoded P; extreme ranges use vc11",
            "expcast_pipeline": "vc15: tile sums before PV; history scaling overlaps PV; shared-copy schedules are opt-in experiments",
            "compact_values": "D128/channel: two N64 PV fragments, paired mean restoration, 168-register target",
            "v_scale_mode": "channel", "v_scale_modes": ["channel", "block"],
            "value_restore_rounding": "channel: mean FMA + FP32 buffer, epilogue scale; vc4: per-tile scale",
            "sage_vc_kernel_version": _extension().sage_vc_kernel_version}


def kernel_resources(output_type=1):
    """Report the paired-conversion default and retain historical resources."""
    resources = _extension().kernel_resources(output_type)
    resources["expcast_vc13_overlap"] = resources["expcast_optimized"]
    resources["expcast_optimized"] = resources["expcast_vc15"]
    return resources


def validate_inputs(q, k, v, tensor_layout, is_causal, return_lse, expcast, v_smooth, clusters, iterations):
    if is_causal or return_lse:
        raise ValueError("Sage VC supports noncausal forward without LSE only")
    if tensor_layout not in ("HND", "NHD"):
        raise ValueError("tensor_layout must be HND or NHD")
    if not isinstance(expcast, bool) or not isinstance(v_smooth, bool):
        raise ValueError("expcast and v_smooth must be bool")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (clusters, iterations)) or not (1 <= clusters <= 256 and 1 <= iterations <= 100):
        raise ValueError("Invalid clustering parameters")
    if any(x.ndim != 4 or x.numel() == 0 for x in (q, k, v)):
        raise ValueError("Expected nonempty four-dimensional Q/K/V")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(x.dtype != q.dtype or x.device != q.device for x in (k, v)):
        raise ValueError("Q/K/V require matching FP16/BF16 dtype and device")
    head_axis = 1 if tensor_layout == "HND" else 2
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[head_axis] != k.shape[head_axis] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q/K/V batch/heads/dim must match; GQA is unsupported")
    if q.shape[-1] not in (64, 128):
        raise ValueError("Sage VC requires D=64/128")


@torch.no_grad()
def prepare(q, k, v, *, expcast=True, v_smooth=True, clusters=8, iterations=2, sm_scale=None, v_scale_mode="channel", grouping_mode="fused"):
    """Private default-stream HND preparation, also used by the diagnostic oracle."""
    if v_scale_mode not in ("channel", "block"):
        raise ValueError("v_scale_mode must be 'channel' or 'block'")
    if grouping_mode not in ("fused", "legacy"):
        raise ValueError("grouping_mode must be 'fused' or 'legacy'")
    from .triton.quant_per_thread import per_thread_int8
    from .quant import per_channel_fp8
    ext = _extension()
    permutation = None
    if v_smooth:
        group = ext.group_values if grouping_mode == "fused" else ext.group_values_legacy
        permutation = group(v, clusters, iterations)
        k = k.gather(2, permutation.unsqueeze(-1).expand_as(k))
        v = v.gather(2, permutation.unsqueeze(-1).expand_as(v))
    # Keep Sage's K centering and INT8 quantizer, including its input-dtype subtraction.
    km = k.mean(dim=2, keepdim=True)
    q8, qs, k8, ks = per_thread_int8(q, k, km, BLKQ=64, WARPQ=16, BLKK=128, WARPK=128, tensor_layout="HND")
    if v_smooth:
        v8, vs, means = ext.prepare_values(v.contiguous(), channel_scale=v_scale_mode == "channel")
    else:
        pad = (-v.shape[2]) % 128
        if pad:
            v = torch.cat((v, torch.zeros(v.shape[0], v.shape[1], pad, v.shape[3], device=v.device, dtype=v.dtype)), dim=2)
        v8, vs, _ = per_channel_fp8(v, tensor_layout="HND", smooth_v=False)
        means = torch.empty(0, device=v.device, dtype=torch.float16)
    return {"q": q8, "k": k8, "v": v8, "qs": qs, "ks": ks, "vs": vs, "means": means,
            "expcast": expcast, "block_values": v_smooth, "output_type": 0 if q.dtype == torch.float16 else 1,
            "scale": q.shape[-1]**-0.5 if sm_scale is None else sm_scale, "permutation": permutation}


def forward_prepared(state, *, kernel_mode="optimized"):
    sage_schedule = SAGE_LAYOUT_MODES.get(kernel_mode, 0)
    if kernel_mode in UNIFIED_MODES and state["vs"].ndim!=3:
        raise ValueError("vc31/vc32 require per-channel V scale")
    if kernel_mode in HISTORY_CANDIDATES and state["q"].shape[-1]!=128:
        raise ValueError("Shared-history candidates require D128")
    if sage_schedule:
        if state["block_values"] and kernel_mode not in UNIFIED_MODES:
            raise ValueError("Sage layout candidates require v_smooth=False")
        if 6<=sage_schedule<=9 and not state["expcast"]:
            raise ValueError("Probability candidates require expcast=True")
        if state["expcast"] and sage_schedule in (3, 4):
            raise ValueError("Pack-first candidates require expcast=False")
        kernel_mode = "vc15"
    if kernel_mode not in ("optimized", "vc15", "vc14", "vc13_overlap", "vc13_serial", "vc12", "vc11", "vc10", "vc9", "vc7", "vc6", "vc4"):
        raise ValueError("Unknown kernel_mode; use a documented vc4..vc21 mode or 'optimized'")
    return _extension().forward(*(state[name] for name in ("q", "k", "v", "qs", "ks", "vs", "means",
                                                          "expcast", "block_values", "output_type", "scale")),
                                optimized=kernel_mode != "vc4", packed_probability=kernel_mode in ("optimized", "vc15", "vc14", "vc13_overlap", "vc13_serial", "vc12", "vc11", "vc10", "vc9", "vc7"),
                                compact_values=kernel_mode in ("optimized", "vc15", "vc14", "vc13_overlap", "vc13_serial", "vc12", "vc11", "vc10", "vc9"), pipeline_probability=kernel_mode in ("optimized", "vc15", "vc14", "vc13_overlap", "vc13_serial", "vc12", "vc11", "vc10"),
                                interleaved_probability=kernel_mode in ("optimized", "vc15", "vc14", "vc13_overlap", "vc13_serial", "vc12", "vc11"),
                                affine_probability=kernel_mode in ("optimized", "vc15", "vc14", "vc13_overlap", "vc13_serial", "vc12"),
                                probability_schedule=1 if kernel_mode == "vc13_overlap" else 2 if kernel_mode == "vc13_serial" else 0,
                                paired_probability=kernel_mode in ("optimized", "vc15", "vc14"),
                                fused_probability=kernel_mode in ("optimized", "vc15"), sage_schedule=sage_schedule)


@torch.no_grad()
def sageattn_vc(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False,
               *, expcast=True, v_smooth=True, clusters=8, iterations=2, kernel_mode="optimized", v_scale_mode="channel", grouping_mode="fused"):
    if kernel_mode not in ("optimized", "vc15", "vc14", "vc13_overlap", "vc13_serial", "vc12", "vc11", "vc10", "vc9", "vc7", "vc6", "vc4", *SAGE_LAYOUT_MODES):
        raise ValueError("Unknown kernel_mode; use a documented vc4..vc21 mode or 'optimized'")
    if kernel_mode in UNIFIED_MODES and v_scale_mode!="channel":
        raise ValueError("vc31/vc32 require per-channel V scale")
    if kernel_mode in HISTORY_CANDIDATES and q.shape[-1]!=128:
        raise ValueError("Shared-history candidates require D128")
    if kernel_mode in SAGE_LAYOUT_MODES:
        if v_smooth and kernel_mode not in UNIFIED_MODES:
            raise ValueError("Sage layout candidates require v_smooth=False")
        if 6<=SAGE_LAYOUT_MODES[kernel_mode]<=9 and not expcast:
            raise ValueError("Probability candidates require expcast=True")
        if expcast and SAGE_LAYOUT_MODES[kernel_mode] in (3, 4):
            raise ValueError("Pack-first candidates require expcast=False")
    if v_scale_mode not in ("channel", "block"):
        raise ValueError("v_scale_mode must be 'channel' or 'block'")
    if grouping_mode not in ("fused", "legacy"):
        raise ValueError("grouping_mode must be 'fused' or 'legacy'")
    validate_inputs(q, k, v, tensor_layout, is_causal, return_lse, expcast, v_smooth, clusters, iterations)
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError("Sage VC requires Hopper; no CPU/reference fallback")
    if sm_scale is not None and (not math.isfinite(sm_scale) or not 0 < sm_scale <= 3.4028234663852886e38):
        raise ValueError("sm_scale must be positive and finite FP32")
    with torch.cuda.device(q.device):
        # Upstream global-V preprocessing launches on the default stream.
        if torch.cuda.current_stream(q.device) != torch.cuda.default_stream(q.device):
            raise ValueError("Sage VC currently requires the default CUDA stream")
        if tensor_layout == "NHD":
            q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        q, k, v = (x.contiguous() for x in (q, k, v))
        state = prepare(q, k, v, expcast=expcast, v_smooth=v_smooth, clusters=clusters, iterations=iterations,
                        sm_scale=sm_scale, v_scale_mode=v_scale_mode, grouping_mode=grouping_mode)
        output = forward_prepared(state,kernel_mode=kernel_mode)
        return output.transpose(1, 2).contiguous() if tensor_layout == "NHD" else output
