"""INT8 QK + ExpCast + decoded-E4M3/FP16 PV, with bounded native workspace."""

from pathlib import Path
import math
import torch
import torch.nn.functional as F
from .reference import Q_BLOCK, K_BLOCK, K_TILE, e4m3_table, quantize_e4m3_decoded

_loaded = False


def load_native():
    global _loaded
    if not _loaded:
        import torch_npu  # noqa: F401; register PrivateUse1 before loading
        library = Path(__file__).with_name("libvc_ascend_ops.so")
        if not library.is_file():
            raise RuntimeError("Ascend native library missing. Run bash scripts/run_ascend910b.sh on the NPU machine.")
        torch.ops.load_library(str(library))
        _loaded = True
    return torch.ops.vc_ascend


def _check_inputs(q, k, v, layout, require_npu):
    if layout not in ("NHD", "HND"):
        raise ValueError("layout must be NHD [B,N,H,D] or HND [B,H,N,D]")
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("Q/K/V must be rank four")
    if any(x.device != q.device or x.dtype != q.dtype for x in (k, v)):
        raise ValueError("Q/K/V must have the same device and dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("FP16/BF16 inputs only")
    if require_npu and q.device.type != "npu":
        raise ValueError("This API requires NPU tensors; there is no CPU/CUDA fallback")
    h_axis, n_axis = (2, 1) if layout == "NHD" else (1, 2)
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[h_axis] != k.shape[h_axis] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Only MHA with matching B/H/D and equal K/V shapes is supported")
    if q.shape[-1] not in (64, 128) or min(*q.shape, *k.shape) <= 0:
        raise ValueError("Nonempty inputs and D=64/128 required")
    if q.shape[n_axis] >= 2**31 or k.shape[n_axis] >= 2**31:
        raise ValueError("Sequence length exceeds native indexing contract")
    if any(x.requires_grad for x in (q, k, v)):
        raise ValueError("Inference-only operator: backward is not implemented")


def _quantize_blocks(x, block, pad_to=None):
    """按 block 做分块量化; pad_to 给定时填充到它的倍数 (须为 block 的整数倍)。

    填充粒度与量化粒度分开: 内核的 matmul tile 可以宽于量化块, 此时张量要按 tile
    宽度填充, 而 scale 仍按量化块产出 —— 多出来的那些块覆盖的全是填充出的零 token,
    由内核的尾块掩码置 -inf, 不进入结果。
    """
    b, h, n, d = x.shape
    pad_to = block if pad_to is None else pad_to
    padded = F.pad(x.float(), (0, 0, 0, (-n) % pad_to))
    blocks = padded.reshape(b, h, -1, block, d)
    amax = blocks.abs().amax(dim=(-2, -1), keepdim=True)
    scale = torch.where(amax > 0, (amax / 127.0).clamp_min(torch.finfo(torch.float32).tiny), torch.ones_like(amax))
    quant = (blocks / scale).round().clamp(-127, 127).to(torch.int8).reshape(b, h, -1, d)
    # Eight identical FP32 lanes provide aligned, in-bounds DMA metadata loads.
    scales = scale.reshape(b, h, -1, 1).expand(-1, -1, -1, 8).contiguous()
    return quant.contiguous(), scales


@torch.no_grad()
def prepare(q, k, v, *, layout="NHD", v_quantize=True, check_finite=True, _reference_only=False):
    """Fresh preprocessing. _reference_only is solely for explicit reference tests."""
    _check_inputs(q, k, v, layout, not _reference_only)
    if check_finite and not all(bool(torch.isfinite(x).all().item()) for x in (q, k, v)):
        raise ValueError("Finite Q/K/V required")
    if check_finite:
        # Centering may double |K|. Reject inputs whose FP32 score bound overflows.
        qmax, kmax = float(q.abs().amax().item()), float(k.abs().amax().item())
        if 2 * qmax * kmax * math.sqrt(q.shape[-1]) > torch.finfo(torch.float32).max / 4:
            raise ValueError("Q/K range exceeds the FP32 score contract")
    if layout == "NHD":
        q, k, v = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    else:
        q, k, v = (x.contiguous() for x in (q, k, v))
    # Center K in FP32. This preserves exact dense softmax before quantization.
    kf = k.float()
    kf = kf - kf.mean(dim=2, keepdim=True)
    if check_finite and not bool(torch.isfinite(kf).all().item()):
        raise ValueError("Centered K exceeds the FP32 preprocessing contract")
    qi, qs = _quantize_blocks(q, Q_BLOCK)
    ki, ks = _quantize_blocks(kf, K_BLOCK, pad_to=K_TILE)
    vf = v.float()
    vmax = vf.abs().amax(dim=2, keepdim=True)
    # Keep PV operands finite in FP16 even for large BF16 values.
    divisor = 448.0 if v_quantize else 1.0
    vs = torch.where(vmax > 0, (vmax / divisor).clamp_min(torch.finfo(torch.float32).tiny), torch.ones_like(vmax))
    normalized_v = vf / vs
    if v_quantize:
        normalized_v = quantize_e4m3_decoded(normalized_v)
    vh = F.pad(normalized_v.half(), (0, 0, 0, (-v.shape[2]) % K_TILE)).contiguous()
    payload = {"q": qi, "k": ki, "v": vh, "q_scale": qs, "k_scale": ks,
            "v_scale": vs.squeeze(2).contiguous(), "lut": e4m3_table(q.device),
            "queries": q.shape[2], "keys": k.shape[2], "scale": 1 / math.sqrt(q.shape[-1]),
            "dtype": q.dtype, "layout": layout, "v_quantize": bool(v_quantize)}
    if q.device.type == "npu":
        import torch_npu
        # Logical contiguous is insufficient if PTA retained a private format.
        for name in ("q", "k", "v", "q_scale", "k_scale", "v_scale", "lut"):
            payload[name] = torch_npu.npu_format_cast(payload[name], 2)  # ACL_FORMAT_ND
    return payload


def _native_args(payload, expcast):
    return (payload["q"], payload["k"], payload["v"], payload["q_scale"],
            payload["k_scale"], payload["v_scale"], payload["lut"],
            payload["queries"], payload["keys"], payload["scale"], bool(expcast))


def _format_output(out, payload):
    out = out[:, :, :payload["queries"]].to(payload["dtype"])
    return out.transpose(1, 2).contiguous() if payload["layout"] == "NHD" else out.contiguous()


@torch.no_grad()
def run_prepared(payload, *, expcast=True, tile_m=64):
    """Native launch, allocation and final cast/layout; excludes quantization."""
    if payload["q"].device.type != "npu":
        raise ValueError("run_prepared requires NPU tensors")
    #  ★只剩 64: 手写 cube 的几何是 M=64 x N=128, 且 ROWS=M/2=32 恰好是一个
    #    Q_BLOCK —— 向量侧的 scale 逻辑依赖这一条。Q 量化块仍是 32, 与 tile_m 无关。
    if tile_m != 64:
        raise ValueError("tile_m must be 64 (手写 cube 的几何); Q 量化仍是 Q32")
    op = load_native()
    args = _native_args(payload, expcast)
    #  非 tiled 的 forward 入口按 vc_tiling.h 的 M 走, 现在也是 64, 两者等价; 统一走 tiled。
    out = op.forward_tiled(*args, int(tile_m))
    return _format_output(out, payload)


@torch.no_grad()
def profile_prepared(payload, *, expcast=True, tile_m=64):
    """Instrumented output + [core,role,sample,field] system ticks, not production timings."""
    if payload["q"].device.type != "npu":
        raise ValueError("profile_prepared requires NPU tensors")
    if tile_m != 64:
        raise ValueError("tile_m must be 64 (手写 cube 的几何)")
    out, records = load_native().profile_tiled(*_native_args(payload, expcast), int(tile_m))
    return _format_output(out, payload), records


@torch.no_grad()
def attention(q, k, v, *, layout="NHD", expcast=True, v_quantize=True,
              causal=False, v_smooth=False, check_finite=True, tile_m=64):
    if causal or v_smooth:
        raise NotImplementedError("This 910B port is dense noncausal, without V clustering/smoothing")
    return run_prepared(prepare(q, k, v, layout=layout, v_quantize=v_quantize,
                               check_finite=check_finite), expcast=expcast, tile_m=tile_m)
