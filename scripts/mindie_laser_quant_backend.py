"""Explicit INT8-QK Laser adapter; decoded E4M3 V still uses FP16 native PV.

This module never dispatches to the older vc_ascend kernel or a CPU fallback.
The pure preparation helper is separately callable for numerical tests.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ascend.vc_attention_ascend.reference import quantize_e4m3_decoded

QK_BLOCK = 128
SEQUENCE_PAD = 256
CONTRACT = "laser_qk_int8_block128_v_e4m3_decoded_fp16_no_reordering_v1"


@dataclass
class QuantLaserPayload:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    dtype: torch.dtype
    shape: tuple
    keys: int
    scale: float
    v_quantize: bool


def _require_npu(q):
    if q.device.type != "npu":
        raise ValueError("Quantized Laser requires NPU tensors; there is no automatic fallback")


def _validate_inputs(q, k, v):
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("Expected rank-four BNSD Q/K/V")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Quantized Laser supports FP16/BF16 inputs")
    if any(x.dtype != q.dtype or x.device != q.device for x in (k, v)):
        raise ValueError("Q/K/V dtype and device must match")
    if k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Dense MHA requires matching B/H/D and K/V shapes")
    if min(*q.shape, *k.shape) < 1 or q.shape[-1] not in (64, 128):
        raise ValueError("Nonempty D64/128 tensors required")
    if max(q.shape[2], k.shape[2]) >= 2**31:
        raise ValueError("Sequence length exceeds the native indexing contract")
    if any(x.requires_grad for x in (q, k, v)):
        raise ValueError("Quantized Laser is inference-only")


def quantize_blocks(x):
    """Symmetric per-128-token/per-head INT8, including padded scale entries."""
    b, h, n, d = x.shape
    padded = F.pad(x.float(), (0, 128 - d, 0, (-n) % SEQUENCE_PAD))
    blocks = padded.reshape(b, h, -1, QK_BLOCK, 128)
    maximum = blocks.abs().amax(dim=(-2, -1), keepdim=True)
    scale = torch.where(maximum > 0,
                        (maximum / 127.0).clamp_min(torch.finfo(torch.float32).tiny),
                        torch.ones_like(maximum))
    quant = (blocks / scale).round().clamp(-127, 127).to(torch.int8)
    return quant.reshape(padded.shape).contiguous(), scale.reshape(b, h, -1).contiguous()


@torch.no_grad()
def prepare_quant_tensors(q, k, v, *, v_quantize=True, check_finite=True, quantizer=None):
    """Fresh tensor preprocessing, also exposed explicitly to CPU reference tests.

    check_finite=False requires finite, representable input from the caller. The
    benchmark validates its deterministic inputs before timing that fast path.
    """
    _validate_inputs(q, k, v)
    if check_finite and not all(bool(torch.isfinite(x).all().item()) for x in (q, k, v)):
        raise ValueError("Finite Q/K/V required")
    quantizer = quantize_blocks if quantizer is None else quantizer
    qi, qs = quantizer(q)
    ki, ks = quantizer(k)
    if v_quantize:
        vf = v.float()
        maximum = vf.abs().amax(dim=2, keepdim=True)
        vs = torch.where(maximum > 0,
                         (maximum / 448.0).clamp_min(torch.finfo(torch.float32).tiny),
                         torch.ones_like(maximum))
        decoded = quantize_e4m3_decoded(vf / vs).half()
        vs = vs.squeeze(2)
    else:
        # The QK-only ablation preserves baseline V conversion exactly.
        decoded = v.half()
        vs = torch.ones((*v.shape[:2], v.shape[-1]), dtype=torch.float32, device=v.device)
    if check_finite and not bool(torch.isfinite(decoded).all().item()):
        raise ValueError("V cannot be represented by the requested FP16 PV path")
    vh = F.pad(decoded, (0, 128 - v.shape[-1], 0, (-v.shape[2]) % SEQUENCE_PAD)).contiguous()
    vs = F.pad(vs, (0, 128 - v.shape[-1]), value=1).contiguous()
    tensors = [qi, ki, vh, qs, ks, vs]
    if q.device.type == "npu":
        import torch_npu
        tensors = [torch_npu.npu_format_cast(tensor, 2) for tensor in tensors]
    return QuantLaserPayload(*tensors, q.dtype, tuple(q.shape), k.shape[2],
                             1 / math.sqrt(q.shape[-1]), bool(v_quantize))


class MindIEQuantLaser:
    def __init__(self, native, *, v_quantize=True, expcast=False, check_finite=True, metadata=None,
                 quantizer=None):
        self.native = native
        self.quantizer = quantize_blocks if quantizer is None else quantizer
        self.v_quantize = bool(v_quantize)
        self.expcast = bool(expcast)
        self.check_finite = bool(check_finite)
        self.metadata = dict(metadata or {}, contract=CONTRACT, qk_arithmetic="INT8 Cube -> INT32",
                             pv_arithmetic="FP16 Cube", native_fp8=False,
                             v_reordering=False, k_smoothing=False, expcast=self.expcast)

    def prepare(self, q, k, v):
        _require_npu(q)
        return prepare_quant_tensors(q, k, v, v_quantize=self.v_quantize,
                                     check_finite=self.check_finite, quantizer=self.quantizer)

    def quantize_qk(self, q, k):
        """Separate component timing: includes fresh Q/K outputs and scale allocation."""
        qi, qs = self.quantizer(q)
        ki, ks = self.quantizer(k)
        return qi, qs, ki, ks

    def run_native(self, payload):
        """Raw native API, including allocation/dispatch; not a kernel-only timer."""
        _require_npu(payload.q)
        out = self.native(payload.q, payload.k, payload.v, payload.q_scale,
                          payload.k_scale, payload.scale, payload.keys, self.expcast)
        if (not isinstance(out, torch.Tensor) or out.shape != payload.q.shape
                or out.dtype != torch.float32 or out.device != payload.q.device):
            raise RuntimeError("Quantized Laser must return one padded FP32 BNSD tensor")
        return out

    def postprocess(self, out, payload):
        # Native PV used normalized decoded V. Dequantization is part of BOTH
        # prepared and full timing; output follows the baseline FP16 round-trip.
        out = out[:, :, :payload.shape[2], :payload.shape[3]]
        if payload.v_quantize:
            out = out * payload.v_scale[:, :, None, :payload.shape[3]]
        return out.half().to(payload.dtype).contiguous()

    def run_prepared(self, payload):
        return self.postprocess(self.run_native(payload), payload)

    def __call__(self, q, k, v):
        return self.run_prepared(self.prepare(q, k, v))

    def describe_payload(self, payload):
        return {"contract": CONTRACT, "input_shape_bnsd": list(payload.shape),
                "input_dtype": str(payload.dtype), "valid_keys": payload.keys,
                "native_shapes": {name: list(getattr(payload, name).shape)
                                  for name in ("q", "k", "v", "q_scale", "k_scale", "v_scale")},
                "native_dtypes": {name: str(getattr(payload, name).dtype)
                                  for name in ("q", "k", "v", "q_scale", "k_scale", "v_scale")},
                "scale": payload.scale, "qk_block_tokens": QK_BLOCK,
                "v_quantize": payload.v_quantize, "expcast": self.expcast,
                "v_contract": "per-channel E4M3 numerics decoded before FP16 PV" if payload.v_quantize else "plain FP16 V",
                "native_fp8": False, "v_reordering": False, "k_smoothing": False,
                "check_finite_in_preprocessing": self.check_finite,
                "qk_preprocessing": self.metadata.get("qk_preprocessing", "torch"),
                "output_v_scale_multiply": payload.v_quantize,
                "prepared_scope": "fixed quantized Q/K and decoded V + native allocation/dispatch + optional V output scale + FP16 round-trip/cast/trim/contiguous"}


def load_quant_laser(bridge_path, *, preprocessing="torch", preprocess_build=None,
                     torch_namespace="vc_laser_quant", **options):
    if preprocessing not in ("torch", "fused"):
        raise ValueError("preprocessing must be torch or fused")
    if preprocessing == "fused" and preprocess_build is None:
        raise ValueError("Fused preprocessing requires its build manifest")
    import torch_npu  # noqa: F401
    torch.ops.load_library(str(bridge_path))
    native = getattr(torch.ops, torch_namespace).la.default
    schema = native._schema
    types = [str(argument.type) for argument in schema.arguments]
    if (types != ["Tensor"] * 5 + ["float", "int", "bool"]
            or [str(x.type) for x in schema.returns] != ["Tensor"]):
        raise RuntimeError(f"Unrecognized {torch_namespace}::la schema: {schema}")
    metadata = {"dispatch": f"torch.ops.{torch_namespace}.la.default", "bridge_path": str(bridge_path),
                "native_schema": str(schema), "qk_preprocessing": "torch"}
    quantizer = None
    if preprocessing == "fused":
        from mindie_laser_quant_fused import load_fused_quantizer
        quantizer, preprocessing_metadata = load_fused_quantizer(preprocess_build)
        metadata.update(preprocessing_metadata)
    return MindIEQuantLaser(native, metadata=metadata, quantizer=quantizer, **options)
