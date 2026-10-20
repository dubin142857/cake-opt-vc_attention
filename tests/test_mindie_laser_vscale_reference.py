"""CPU numerical contracts and validator fault injection; no NPU speed claims."""
from dataclasses import replace
from pathlib import Path
import struct
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mindie_laser_quant_backend as backend
import mindie_laser_quant_reference as old_reference
import mindie_laser_vscale_reference as reference


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(4, previous))
    yield
    torch.set_num_threads(previous)


def _truncated_float(value):
    """Independent scalar encoding, rather than reusing oracle bit views."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0] & 0xffffe000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def test_scalar_gain_order_and_half_transport_are_explicit():
    dot = torch.tensor([23457, -79147, 2097152, 0], dtype=torch.int32)
    qs, ks = torch.tensor([.0171]), torch.tensor([.0273])
    output, detail = reference.scale_scores(dot, qs, ks, 128 ** -.5, details=True)
    product = torch.tensor(float(qs) * float(ks), dtype=torch.float32).item()
    gain = _truncated_float(product)
    coefficient = torch.tensor(gain * 64, dtype=torch.float32) * torch.tensor(128 ** -.5).half().float()
    expected = (dot.float().div(64).half().float() * coefficient.half().float()).half()
    assert torch.equal(output, expected)
    assert detail["gain"].item() == gain
    assert detail["coefficient"].item() == coefficient.item()
    assert bool(detail["fast"].all())
    assert detail["transport"][2].item() == 32768


def test_full_signed_int8_bound_needs_one_over_64():
    q = torch.full((1, 128), -128, dtype=torch.int8)
    dots = q.int() @ q.int().T
    assert dots.item() == 2097152
    assert torch.isinf((dots.float() / 32).half()).all()
    output, detail = reference.scale_scores(
        dots, torch.tensor(.001), torch.tensor(.001), 128 ** -.5, details=True)
    assert detail["transport"].item() == 32768
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("coefficient", [0., 2 ** -14, 65504.])
def test_half_fast_domain_includes_zero_and_exact_boundaries(coefficient):
    # attention scale 1/64 cancels transport's factor64; gain values are
    # chosen with already-zero low13 mantissa bits.
    _, actual, fast = reference.gain_coefficient(
        torch.tensor(coefficient), torch.tensor(1.), 1 / 64)
    assert actual.item() == coefficient
    assert fast.item()


@pytest.mark.parametrize("gain", [2 ** -16, 2 ** -26, 65536.])
def test_unsafe_scalar_uses_fp32_without_zero_times_half_infinity(gain):
    dots = torch.tensor([0, 1, -2048384], dtype=torch.int32)
    if gain > 65504:
        dots[-1] = -1
    output, detail = reference.scale_scores(
        dots, torch.tensor(gain), torch.tensor(1.), 1 / 64, details=True)
    assert not bool(detail["fast"].any())
    expected = (detail["transport"].float() * detail["coefficient"]).half()
    assert torch.equal(output, expected)
    assert torch.isfinite(output).all()
    if gain < 2 ** -25:
        assert detail["coefficient"].half().item() == 0
        assert output[-1].item() != 0
    if gain > 65504:
        assert torch.isinf(detail["coefficient"].half()).all()
        assert output[0].item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["zero", "tiny_gain", "large_gain", "signed_int8_bound"])
def test_range_payload_oracle_and_constant_channels(dtype, kind):
    payload, structure = reference.range_payload(dtype, kind)
    result = reference.decoded_reference(payload, structured=structure)
    assert result.shape == payload.shape and result.dtype == dtype
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[..., 0], torch.zeros_like(result[..., 0]), rtol=0, atol=0)
    torch.testing.assert_close(result[..., 1], torch.full_like(result[..., 1], 2), rtol=0, atol=0)
    # Scalar-unsafe fixtures must actually select the fallback path.
    _, _, fast = reference.gain_coefficient(
        payload.q_scale[..., :1], payload.k_scale[..., :1], payload.scale)
    if kind in ("tiny_gain", "large_gain", "signed_int8_bound"):
        assert not bool(fast.any())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [64, 128])
def test_random_rounding_change_is_measured_separately_from_oracle_agreement(dtype, dim):
    generator = torch.Generator().manual_seed(918)
    inputs = [torch.randn((1, 2, n, dim), generator=generator).to(dtype)
              for n in (129, 513, 513)]
    payload = backend.prepare_quant_tensors(*inputs, v_quantize=False)
    old = old_reference.decoded_reference(payload)
    new = reference.decoded_reference(payload)
    error = (new.float() - old.float()).square().sum().sqrt() / old.float().square().sum().sqrt()
    assert 0 < error < reference.ROUNDING_DRIFT_LIMIT
    # Query subsets preserve sample order and do not truncate K. Changing the
    # CPU GEMM M dimension can alter FP32 PV summation at a final cast tie.
    ids = torch.tensor([128, 1, 65, 0])
    sampled = reference.decoded_reference(payload, ids, chunk=2)
    expected = new.index_select(2, ids)
    upward = torch.nextafter(expected, torch.full_like(expected, float("inf")))
    downward = torch.nextafter(expected, torch.full_like(expected, -float("inf")))
    spacing = torch.maximum(upward.float() - expected.float(), expected.float() - downward.float())
    # Public output passes through FP16 first, including its subnormal step.
    spacing = spacing.clamp_min(2 ** -24)
    assert bool(((sampled.float() - expected.float()).abs() <= spacing).all())


def test_structured_oracle_detects_wrong_scale_block_mapping():
    case = dict(batch=2, heads=3, queries=257, keys=385, dim=128)
    payload = backend.prepare_quant_tensors(
        *reference._mapping_inputs(case, torch.float16), v_quantize=False)
    indices = reference._selected_queries(case["queries"])
    correct = reference.decoded_reference(payload, indices, structured=True)
    wrong = replace(payload, q_scale=payload.q_scale.flip(-1))
    changed = reference.decoded_reference(wrong, indices, structured=True)
    error = (correct[..., 2:].float() - changed[..., 2:].float()).square().sum().sqrt()
    assert error > .01
    assert payload.q_scale.shape[:2] == (2, 3)
    assert all(index in indices.tolist() for index in (127, 128, 129, 255, 256))


def test_oracle_rejects_other_features_and_uncontrolled_maxima():
    positive = torch.ones(1, 1, 3, 64, dtype=torch.float16)
    payload = backend.prepare_quant_tensors(positive, positive, positive, v_quantize=False)
    with pytest.raises(ValueError, match="zero maximum"):
        reference.decoded_reference(payload, structured=True)
    with pytest.raises(ValueError, match="original exp"):
        reference.decoded_reference(payload, expcast=True)
    with pytest.raises(ValueError, match="unquantized"):
        reference.decoded_reference(replace(payload, v_quantize=True))
    with pytest.raises(ValueError, match="Query indices"):
        reference.decoded_reference(payload, torch.tensor([3]))
    with pytest.raises(ValueError, match="finite"):
        reference.scale_scores(torch.tensor(1), torch.tensor(float("inf")), torch.tensor(1), .125)


class _CPUAdapter:
    """Test double for validator orchestration, not a native implementation."""
    expcast = False
    v_quantize = False

    def __init__(self, *, candidate, stale_scales=False):
        self.candidate, self.stale_scales = candidate, stale_scales
        self.scales = None
        self.native_calls = 0

    def prepare(self, *inputs):
        self.scales = None
        return backend.prepare_quant_tensors(*inputs, v_quantize=False)

    def run_native(self, payload):
        self.native_calls += 1
        # Native outputs include all padded Q rows/features and are FP32.
        p = replace(payload, shape=tuple(payload.q.shape), dtype=torch.float32)
        if self.stale_scales:
            if self.scales is None:
                self.scales = (p.q_scale.clone(), p.k_scale.clone())
            p = replace(p, q_scale=self.scales[0], k_scale=self.scales[1])
        oracle = reference.decoded_reference if self.candidate else old_reference.decoded_reference
        return oracle(p).float()

    def postprocess(self, output, payload):
        return output[:, :, :payload.shape[2], :payload.shape[3]].half().to(payload.dtype).contiguous()

    def __call__(self, *inputs):
        payload = self.prepare(*inputs)
        return self.postprocess(self.run_native(payload), payload)


def test_remote_validator_passes_cpu_oracles_and_records_every_gate(monkeypatch):
    case = dict(name="test_mapping", batch=2, heads=3, queries=17, keys=129, dim=64)
    monkeypatch.setattr(reference, "MAPPING_CASES", (case,))
    adapters = {"baseline": _CPUAdapter(candidate=False), "reuse": _CPUAdapter(candidate=True)}
    report, saves = {}, []
    result = reference.validate_vector_scale(adapters, "cpu", report, lambda: saves.append(True))
    assert all(row["status"] == "passed" for row in result)
    mappings = [row for row in result if row["name"] == case["name"]]
    assert len(mappings) == 2
    for row in mappings:
        assert [x["update"] for x in row["rounds"]] == [
            "initial", "unchanged", "q_scale_in_place", "k_scale_in_place", "unchanged"]
        assert all(x["pointers_unchanged"] for x in row["rounds"])
        assert row["full_prepared"]["passed"]
        assert row["rounds"][2]["decoded_reference"]["passed"]
    assert len(result) == 10
    assert saves


def test_remote_validator_rejects_stale_scale_pointer_cache(monkeypatch):
    monkeypatch.setattr(reference, "MAPPING_CASES", (
        dict(name="scale_refresh", batch=1, heads=2, queries=17, keys=129, dim=64),))
    adapters = {"baseline": _CPUAdapter(candidate=False),
                "reuse": _CPUAdapter(candidate=True, stale_scales=True)}
    report = {}
    with pytest.raises(AssertionError, match="drift|decoded"):
        reference.validate_vector_scale(adapters, "cpu", report, lambda: None)
    assert report["vector_scale_validation"][-1]["status"] == "failed"
    assert "error" in report["vector_scale_validation"][-1]
