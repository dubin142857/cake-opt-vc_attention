"""CPU tensor/ABI tests; no claims about native Cube or NPU execution."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mindie_laser_quant_backend as backend
import mindie_laser_quant_reference as reference


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_block_scales_padding_and_v_decode_are_explicit(dtype):
    q, k, v = reference.structured_case(dtype, 64, 257, 385)
    payload = backend.prepare_quant_tensors(q, k, v)
    assert payload.q.shape == (1, 2, 512, 128)
    assert payload.k.shape == payload.v.shape == (1, 2, 512, 128)
    assert payload.q_scale.shape == payload.k_scale.shape == (1, 2, 4)
    assert payload.q.dtype == payload.k.dtype == torch.int8
    assert payload.v.dtype == torch.float16
    assert payload.q_scale.dtype == payload.k_scale.dtype == torch.float32
    assert payload.scale == pytest.approx(64**-.5)
    for tensor, count in ((payload.q, 257), (payload.k, 385), (payload.v, 385)):
        assert tensor.is_contiguous()
        assert torch.count_nonzero(tensor[:, :, count:]) == 0
        assert torch.count_nonzero(tensor[..., 64:]) == 0
    assert not torch.equal(payload.q_scale[..., 0], payload.q_scale[..., 1])
    assert not torch.equal(payload.k_scale[..., 0], payload.k_scale[..., 1])
    # Compare E4M3 numbers to PyTorch's independent native format conversion.
    if hasattr(torch, "float8_e4m3fn"):
        normalized = v.float() / payload.v_scale[:, :, None, :64]
        expected = normalized.to(torch.float8_e4m3fn).half()
        torch.testing.assert_close(payload.v[:, :, :385, :64], expected, rtol=0, atol=0)
    plain = backend.prepare_quant_tensors(q, k, v, v_quantize=False)
    torch.testing.assert_close(plain.v[:, :, :385, :64], v.half(), rtol=0, atol=0)
    assert torch.equal(plain.v_scale, torch.ones_like(plain.v_scale))


def test_zeros_have_finite_unit_scales_and_zero_decoded_output():
    inputs = [torch.zeros(1, 2, n, 128, dtype=torch.bfloat16) for n in (17, 129, 129)]
    payload = backend.prepare_quant_tensors(*inputs)
    for scale in (payload.q_scale, payload.k_scale, payload.v_scale):
        assert torch.equal(scale, torch.ones_like(scale))
    for expcast in (False, True):
        output = reference.decoded_reference(payload, expcast=expcast, structured=True)
        assert output.shape == inputs[0].shape
        assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("v_quantize", [False, True])
def test_native_abi_output_scale_and_fresh_full_preprocessing(monkeypatch, v_quantize):
    monkeypatch.setattr(backend, "_require_npu", lambda tensor: None)
    calls = []

    def native(q, k, v, qs, ks, scale, keys, expcast):
        calls.append((q, k, v, qs, ks, scale, keys, expcast))
        # Q=0 makes attention uniform. This double tests data/scale plumbing;
        # it deliberately does not implement the quantized Laser algorithm.
        return v[:, :, :keys].float().mean(2, keepdim=True).expand(q.shape).contiguous()

    adapter = backend.MindIEQuantLaser(native, v_quantize=v_quantize, expcast=True)
    q = torch.zeros(1, 2, 17, 64, dtype=torch.bfloat16)
    k = torch.zeros(1, 2, 129, 64, dtype=torch.bfloat16)
    v = torch.full_like(k, 3)
    v[..., 0], v[..., 1], v[..., 2] = 0, 2, .03125
    payload = adapter.prepare(q, k, v)
    output = adapter.run_prepared(payload)
    torch.testing.assert_close(output, v[:, :, :17], rtol=0, atol=0)
    assert output.is_contiguous() and output.dtype == q.dtype
    assert calls[-1][-3:] == (64**-.5, 129, True)
    torch.testing.assert_close(adapter(q, k, v * 2), output * 2, rtol=0, atol=0)
    assert not adapter.describe_payload(payload)["native_fp8"]
    adapter.native = lambda *args: torch.zeros_like(q)
    with pytest.raises(RuntimeError, match="padded FP32"):
        adapter.run_prepared(payload)


def test_public_adapter_has_no_cpu_fallback_and_rejects_invalid_input():
    x = torch.zeros(1, 2, 17, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="NPU"):
        backend.MindIEQuantLaser(None).prepare(x, x, x)
    bad = x.clone()
    bad[..., 0] = float("nan")
    with pytest.raises(ValueError, match="Finite"):
        backend.prepare_quant_tensors(x, bad, x)
    with pytest.raises(ValueError, match="matching"):
        backend.prepare_quant_tensors(x, x, x[:, :1])
    with pytest.raises(ValueError, match="inference-only"):
        backend.prepare_quant_tensors(x.requires_grad_(), x, x)


@pytest.mark.parametrize("dim", [64, 128])
def test_structured_oracle_detects_scale_loss_and_expcast_change(dim):
    inputs = reference.structured_case(torch.bfloat16, dim, 257, 385)
    payload = backend.prepare_quant_tensors(*inputs)
    expected = reference.decoded_reference(payload, expcast=True, structured=True)
    ordinary = reference.decoded_reference(payload, expcast=False, structured=True)
    difference = (expected[..., 2:].float() - ordinary[..., 2:].float()).square().mean().sqrt()
    rms = expected[..., 2:].float().square().mean().sqrt()
    assert float(difference / rms) > .01
    wrong = replace(payload, q_scale=payload.q_scale.clone())
    wrong.q_scale[..., 1] = wrong.q_scale[..., 0]
    changed = reference.decoded_reference(wrong, expcast=True, structured=True)
    assert not torch.equal(changed[:, :, 128:256], expected[:, :, 128:256])
    wrong = replace(payload, v_scale=torch.ones_like(payload.v_scale))
    changed = reference.decoded_reference(wrong, expcast=True, structured=True)
    assert not torch.allclose(changed.float(), expected.float(), rtol=.1, atol=.1)


def test_reference_rejects_uncontrolled_online_maxima():
    q = torch.ones(1, 1, 3, 64, dtype=torch.float16)
    payload = backend.prepare_quant_tensors(q, q, q)
    with pytest.raises(ValueError, match="zero maximum"):
        reference.decoded_reference(payload, structured=True)


def test_e4m3_rne_boundaries_subnormals_signed_values_and_saturation():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("Independent torch E4M3 conversion requires float8_e4m3fn")
    finite = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    middle = (finite[:-1] + finite[1:]) / 2
    values = torch.cat((finite, middle,
                        torch.nextafter(middle, torch.full_like(middle, float("inf"))),
                        torch.nextafter(middle, torch.full_like(middle, -float("inf"))),
                        torch.tensor([0., 448., 464., 500., 1e-30])))
    values = torch.cat((values, -values))
    expected = values.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    actual = backend.quantize_e4m3_decoded(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
