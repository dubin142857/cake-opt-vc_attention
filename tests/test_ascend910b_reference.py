"""Numerical contract checks runnable without CANN; not native NPU validation."""
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ascend"))
from vc_attention_ascend.api import prepare, attention, run_prepared
from vc_attention_ascend.reference import (
    e4m3_table, quantize_e4m3_decoded, payload_reference, expcast_decode,
)


def inputs(nq=33, nk=129, d=64, dtype=torch.float16):
    g = torch.Generator().manual_seed(913)
    return [torch.randn((2, 3, n, d), generator=g).to(dtype) for n in (nq, nk, nk)]


def prepared(q, k, v, **kwargs):
    return prepare(q, k, v, layout="HND", _reference_only=True, **kwargs)


def test_e4m3_decode_all_expcast_codes():
    actual = e4m3_table()[:121]
    expected = torch.arange(121, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.half().float(), actual, rtol=0, atol=0)


def test_e4m3_encode_rounding_boundaries_and_saturation():
    positive = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    mid = (positive[1:] + positive[:-1]) / 2
    eps = torch.full_like(mid, float("inf"))
    x = torch.cat((positive, mid, torch.nextafter(mid, eps), torch.nextafter(mid, -eps), torch.tensor([450., 1e6])))
    x = torch.cat((x, -x))
    expected = x.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(quantize_e4m3_decoded(x), expected, rtol=0, atol=0)


def test_expcast_endpoints():
    actual = expcast_decode(torch.tensor([0., -100., -float("inf")]))
    torch.testing.assert_close(actual, torch.tensor([256., 0., 0.]), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("nq,nk", [(1, 1), (17, 65), (33, 129), (65, 257)])
def test_uniform_attention_all_keys_and_tail(dtype, nq, nk):
    q, k, v = inputs(nq, nk, dtype=dtype)
    q.zero_()
    p = prepared(q, k, v)
    decoded_v = p["v"][:, :, :nk].float() * p["v_scale"].unsqueeze(2)
    expected = decoded_v.mean(2, keepdim=True).expand(-1, -1, nq, -1)
    for cast in (False, True):
        torch.testing.assert_close(payload_reference(p, expcast=cast), expected, rtol=2e-5, atol=1e-6)
    assert torch.count_nonzero(p["q"][:, :, nq:]) == 0
    assert torch.count_nonzero(p["k"][:, :, nk:]) == 0
    assert torch.count_nonzero(p["v"][:, :, nk:]) == 0


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("value", [0., 2., -1.75])
def test_constant_value_invariant(d, value):
    q, k, v = inputs(33, 385, d)
    v.fill_(value)
    p = prepared(q, k, v)
    actual = payload_reference(p, expcast=True)
    torch.testing.assert_close(actual, torch.full_like(actual, value), rtol=2e-5, atol=2e-6)


def test_control_against_independent_dense_decoded_inputs_single_tile():
    q, k, v = inputs(17, 65)
    p = prepared(q, k, v)
    # Independent INT64 dot avoids premature Q/K dequantization changing the
    # rounding of FP16 P at a boundary (the native Cube returns exact INT32).
    dot = p["q"][:, :, :17].long() @ p["k"][:, :, :65].long().transpose(-1, -2)
    gain = (p["q_scale"][:, :, 0, 0] * p["k_scale"][:, :, 0, 0]) * p["scale"]
    s = dot.float() * gain[..., None, None]
    prob = torch.exp(s - s.amax(-1, keepdim=True))
    result = prob.half().float() @ p["v"][:, :, :65].float() / prob.sum(-1, keepdim=True)
    result *= p["v_scale"].unsqueeze(2)
    torch.testing.assert_close(payload_reference(p, expcast=False), result, rtol=1e-5, atol=2e-6)


def test_large_and_tiny_bfloat16_v_no_fp16_overflow():
    q, k, v = inputs(dtype=torch.bfloat16)
    v *= 1e10
    v[..., 1] = 1e-35
    p = prepared(q, k, v)
    assert torch.isfinite(p["v"]).all()
    assert p["v"].abs().max() <= 448
    assert torch.isfinite(payload_reference(p)).all()


def test_no_silent_fallback_and_unsupported_features():
    q, k, v = inputs()
    with pytest.raises(ValueError, match="NPU"):
        attention(q, k, v, layout="HND")
    with pytest.raises(ValueError, match="NPU"):
        run_prepared(prepared(q, k, v))
    with pytest.raises(NotImplementedError):
        attention(q, k, v, causal=True)
    with pytest.raises(NotImplementedError):
        attention(q, k, v, v_smooth=True)


def test_bad_shapes_gradients_and_nonfinite_rejected():
    q, k, v = inputs()
    with pytest.raises(ValueError, match="matching"):
        prepared(q[:, :1], k, v)
    q.requires_grad_(True)
    with pytest.raises(ValueError, match="Inference"):
        prepared(q, k, v)
    q = q.detach()
    q[..., 0] = float("nan")
    with pytest.raises(ValueError, match="Finite"):
        prepared(q, k, v)
    q, k, v = inputs(dtype=torch.bfloat16)
    q.zero_()
    k.fill_(torch.finfo(torch.bfloat16).max)
    with pytest.raises(ValueError, match="FP32 preprocessing"):
        prepared(q, k, v)


def test_selected_queries_preserve_quantization_groups():
    p = prepared(*inputs(65, 257))
    ids = torch.tensor([0, 31, 32, 63, 64])
    torch.testing.assert_close(payload_reference(p, query_indices=ids), payload_reference(p).index_select(2, ids),
                               rtol=1e-6, atol=1e-7)


def test_layout_consistency():
    q, k, v = inputs()
    p = prepared(q, k, v)
    n = prepare(*(x.transpose(1, 2) for x in (q, k, v)), layout="NHD", _reference_only=True)
    for key in ("q", "k", "v", "q_scale", "k_scale", "v_scale"):
        torch.testing.assert_close(p[key], n[key], rtol=0, atol=0)


def test_v_fp16_mode_is_explicit_and_finite():
    p = prepared(*inputs(), v_quantize=False)
    assert p["v_quantize"] is False
    assert p["v"].abs().max() <= 1
    assert torch.isfinite(payload_reference(p)).all()
