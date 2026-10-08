"""Experimental vc27: block-scaled FP8 QK + FP8 PV, optional ExpCast.

Hopper, noncausal MHA, FP16/BF16 input/output, D64/D128. K is centered in
FP32. No QK rotation or V smoothing yet. Full API includes fresh quantization;
the split prepare/forward API is for fixed-payload diagnostics only.
"""
import math
import torch
from .vc import _extension, validate_inputs


@torch.no_grad()
def prepare(q, k, v, *, expcast=True, sm_scale=None):
    """Private HND/default-stream preparation; finite contiguous inputs required."""
    if torch.cuda.current_stream(q.device) != torch.cuda.default_stream(q.device):
        raise ValueError("FP8 full preprocessing currently requires the default CUDA stream")
    from .quant import per_channel_fp8
    q8, qs, k8, ks, km = _extension().prepare_qk_fp8(q, k)
    pad = (-v.shape[2]) % 128
    if pad:
        v = torch.cat((v, v.new_zeros(v.shape[0], v.shape[1], pad, v.shape[3])), dim=2)
    v8, vs, _ = per_channel_fp8(v, tensor_layout="HND", smooth_v=False)
    return {"q": q8, "k": k8, "v": v8, "qs": qs, "ks": ks, "vs": vs,
            "means": torch.empty(0, device=q.device, dtype=torch.float16),
            "k_mean": km, "qk_format": "fp8_block", "expcast": expcast,
            "block_values": False, "output_type": 0 if q.dtype == torch.float16 else 1,
            "scale": q.shape[-1] ** -0.5 if sm_scale is None else sm_scale, "permutation": None}


@torch.no_grad()
def forward_prepared(state):
    """Native FP8 core on the current stream; caller manages stream dependencies."""
    if state.get("qk_format") != "fp8_block" or state.get("block_values", False):
        raise ValueError("Expected FP8 block QK without V smoothing")
    return _extension().forward_fp8(*(state[key] for key in
        ("q", "k", "v", "qs", "ks", "vs", "expcast", "output_type", "scale")))


@torch.no_grad()
def sageattn_vc_fp8(q, k, v, tensor_layout="HND", is_causal=False,
                    sm_scale=None, return_lse=False, *, expcast=True):
    """Fresh scaled FP8 quantization and one fused attention launch, vc27 opt-in."""
    validate_inputs(q, k, v, tensor_layout, is_causal, return_lse, expcast, False, 8, 2)
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError("FP8 QK attention requires Hopper; no CPU/reference fallback")
    if sm_scale is not None and (not math.isfinite(sm_scale) or not 0 < sm_scale <= 3.4028234663852886e38
                                or torch.tensor(sm_scale, dtype=torch.float32).item() == 0):
        raise ValueError("sm_scale must be positive and finite FP32")
    with torch.cuda.device(q.device):
        if torch.cuda.current_stream(q.device) != torch.cuda.default_stream(q.device):
            raise ValueError("FP8 full API currently requires the default CUDA stream")
        if tensor_layout == "NHD":
            q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        q, k, v = (x.contiguous() for x in (q, k, v))
        out = forward_prepared(prepare(q, k, v, expcast=expcast, sm_scale=sm_scale))
        return out.transpose(1, 2).contiguous() if tensor_layout == "NHD" else out
