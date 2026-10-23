"""Independent binary16 encoding/tail oracle; no CANN installation required."""
import math
import struct

import pytest
import torch

from scripts import mindie_laser_fp16_expcast_math as ref


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def independent_half(code):
    exponent, fraction = divmod(code, 1024)
    return fraction * 2.0**-24 if exponent == 0 else (1 + fraction / 1024) * 2.0**(exponent - 15)


def round_half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def test_constants_are_binary16_adaptation_not_e4m3_codes():
    assert ref.GAIN == fp32(1024 * math.log2(math.e))
    assert ref.BIAS == fp32((15 + 12 - 0.35 / 8) * 1024)
    assert ref.CODE_MAX == (15 + 12) * 1024
    assert not ref.CONTRACT["paper_exact"] and not ref.CONTRACT["fp8"]


def test_every_finite_nonpositive_fp16_input_and_negative_infinity_matches_independent_rounds():
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    values = values[~torch.isnan(values) & (values <= 0)]
    expected_codes, expected_output = [], []
    for value in values.tolist():
        affine = fp32(fp32(value * ref.GAIN) + ref.BIAS)
        code = round(max(0.0, min(float(ref.CODE_MAX), affine)))
        expected_codes.append(code)
        expected_output.append(round_half(independent_half(code) * 2.0**-12))
    assert torch.equal(ref.encode(values), torch.tensor(expected_codes, dtype=torch.int16))
    assert torch.equal(ref.probability(values).view(torch.int16),
                       torch.tensor(expected_output, dtype=torch.float16).view(torch.int16))


def test_all_encoded_values_decode_with_binary16_subnormal_rounding():
    codes = torch.arange(ref.CODE_MAX + 1, dtype=torch.int32)
    output = ref.decode(codes)
    expected = torch.tensor([round_half(independent_half(c) * 2.0**-12)
                             for c in codes.tolist()], dtype=torch.float16)
    assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
    assert torch.isfinite(output).all() and torch.all(output[1:] >= output[:-1])
    assert output[0] == 0 and output[-1] == 1
    # Exactly half the smallest subnormal rounds to even zero; the next bit
    # code must survive. A naive direct bias-15 map cannot represent this tail.
    assert output[2048] == 0 and output[2049] == 2.0**-24


def test_probability_error_envelope_over_entire_fp16_input_domain():
    values = torch.arange(65536, dtype=torch.int32).short().view(torch.float16)
    values = values[torch.isfinite(values) & (values <= 0)]
    expected = values.double().exp()
    actual = ref.probability(values).double()
    # Numerical specification check, not a model-quality or attention gate.
    # Add half a subnormal ULP where relative error alone becomes misleading.
    assert torch.all((actual - expected).abs() <= .031 * expected + 2.0**-25)
    assert torch.all((actual >= 0) & (actual <= 1))


def test_masks_and_tail_probabilities_retain_true_fp16_range():
    scores = torch.tensor([0, -10, -12, -16, -17, -18, -60000, -float("inf")], dtype=torch.float16)
    output = ref.probability(scores)
    assert output[0] == .97802734375
    assert torch.all(output[1:5] > 0) and torch.all(output[1:5] < 2.0**-14)
    assert torch.equal(output[5:], torch.zeros(3, dtype=torch.float16))
    assert ref.encode(scores)[1] > 120  # full binary16 encoding, not shifted FP8


def test_approximate_map_requires_online_reference_and_same_probability_denominator():
    p = ref.probability(torch.tensor([0, -1, -2], dtype=torch.float16)).double()
    assert abs((p[2] / p[1]).item() - math.exp(-1)) > 1e-3
    # P(0) != 1 is intentional. Normalizing by the actual stored P preserves
    # uniform attention; mixing an exact-exp denominator would fail this.
    uniform = ref.probability(torch.zeros(32, dtype=torch.float16)).double()
    assert torch.equal(uniform / uniform.sum(), torch.full((32,), 1 / 32, dtype=torch.float64))


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [.125]])
def test_out_of_contract_scores_are_rejected(values):
    with pytest.raises(ValueError):
        ref.probability(torch.tensor(values, dtype=torch.float16))


def test_wrong_input_type_and_codes_fail_closed():
    with pytest.raises(ValueError, match="FP16"):
        ref.probability(torch.zeros(2, dtype=torch.float32))
    for codes in (torch.tensor([-1]), torch.tensor([27649]), torch.tensor([1.0])):
        with pytest.raises(ValueError):
            ref.decode(codes)
