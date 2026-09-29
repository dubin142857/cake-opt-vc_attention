"""Sage INT8 QK + optional VC V grouping/smoothing and ExpCast.

Dense noncausal MHA, D=64/128, eager inference on Hopper's default stream.
Each call recomputes grouping; this entry has no denoising-step layout cache.
"""
import math
import torch


# H200 same-payload tests rejected the vc13 shared-copy schedules as defaults.
DEFAULT_EXPCAST_KERNEL = "vc12"


def _extension():
    from . import _vc_sm90
    if (getattr(_vc_sm90, "sage_vc_api_version", None) != 1 or
            getattr(_vc_sm90, "sage_vc_kernel_version", None) != 10 or
            getattr(_vc_sm90, "sage_vc_preprocess_version", None) != 1):
        raise RuntimeError("Rebuild sageattention-local with bash scripts/build_sage_local.sh")
    return _vc_sm90


def sage_vc_build_info():
    from .local_info import local_build_info
    return {**local_build_info(), "vc_extension": _extension().__file__,
            "kernel_math_modified": True,
            "algorithm": "Sage per-thread INT8 QK + VC block V residual/mean + ExpCast",
            "qk_hadamard": False, "v_block": 128, "mean_dtype": "float16",
            "clustering": "8 clusters, 2 Lloyd iterations, stable order; fresh every call",
            "grouping_modes": ["fused", "legacy"], "default_grouping_mode": "fused",
            "grouping_implementation": "fused assignment/statistics for <=8 effective clusters; legacy otherwise",
            "grouping_rounding": "FP32 reduction order changes; labels near boundaries can differ",
            "sage_vc_preprocess_version": _extension().sage_vc_preprocess_version,
            "paper_equivalent": False, "sage_vc_api_version": 1,
            "default_kernel_mode": "optimized", "default_expcast_kernel": DEFAULT_EXPCAST_KERNEL, "kernel_modes": ["optimized", "vc13_overlap", "vc13_serial", "vc12", "vc11", "vc10", "vc9", "vc7", "vc6", "vc4"],
            "expcast_probability": "saturating u8 RNE + byte permutation + FP32 reduction",
            "expcast_interleaving": "vc12: encode two K fragments, issue packed decode, then consume FP32 values",
            "probability_scratch_bytes_per_block": 0,
            "experimental_probability_scratch_bytes_per_block": 8192,
            "experimental_kernel_modes": ["vc13_serial", "vc13_overlap"],
            "expcast_numerical_contract": "expcast-only folds coefficients in FP32 before FMA/RNE; byte rounding may differ near boundaries; denominator uses decoded P; extreme ranges use vc11",
            "expcast_pipeline": "vc12: tile sums before PV; history scaling overlaps PV; shared-copy schedules are opt-in experiments",
            "compact_values": "D128/channel: two N64 PV fragments, paired mean restoration, 168-register target",
            "v_scale_mode": "channel", "v_scale_modes": ["channel", "block"],
            "value_restore_rounding": "channel: mean FMA + FP32 buffer, epilogue scale; vc4: per-tile scale",
            "sage_vc_kernel_version": _extension().sage_vc_kernel_version}


def kernel_resources(output_type=1):
    """Relabel native vc13 resources to match this package's default dispatch."""
    resources = _extension().kernel_resources(output_type)
    resources["expcast_vc13_overlap"] = resources["expcast_optimized"]
    resources["expcast_optimized"] = resources["expcast_vc12"]
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
    """Private HND preparation, also returned to the independent diagnostic oracle."""
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
    if kernel_mode not in ("optimized", "vc13_overlap", "vc13_serial", "vc12", "vc11", "vc10", "vc9", "vc7", "vc6", "vc4"):
        raise ValueError("kernel_mode must be 'optimized', 'vc13_overlap', 'vc13_serial', 'vc12', 'vc11', 'vc10', 'vc9', 'vc7', 'vc6' or 'vc4'")
    return _extension().forward(*(state[name] for name in ("q", "k", "v", "qs", "ks", "vs", "means",
                                                          "expcast", "block_values", "output_type", "scale")),
                                optimized=kernel_mode != "vc4", packed_probability=kernel_mode in ("optimized", "vc13_overlap", "vc13_serial", "vc12", "vc11", "vc10", "vc9", "vc7"),
                                compact_values=kernel_mode in ("optimized", "vc13_overlap", "vc13_serial", "vc12", "vc11", "vc10", "vc9"), pipeline_probability=kernel_mode in ("optimized", "vc13_overlap", "vc13_serial", "vc12", "vc11", "vc10"),
                                interleaved_probability=kernel_mode in ("optimized", "vc13_overlap", "vc13_serial", "vc12", "vc11"),
                                affine_probability=kernel_mode in ("optimized", "vc13_overlap", "vc13_serial", "vc12"),
                                probability_schedule=1 if kernel_mode == "vc13_overlap" else 2 if kernel_mode == "vc13_serial" else 0)


@torch.no_grad()
def sageattn_vc(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False,
               *, expcast=True, v_smooth=True, clusters=8, iterations=2, kernel_mode="optimized", v_scale_mode="channel", grouping_mode="fused"):
    if kernel_mode not in ("optimized", "vc13_overlap", "vc13_serial", "vc12", "vc11", "vc10", "vc9", "vc7", "vc6", "vc4"):
        raise ValueError("kernel_mode must be 'optimized', 'vc13_overlap', 'vc13_serial', 'vc12', 'vc11', 'vc10', 'vc9', 'vc7', 'vc6' or 'vc4'")
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
