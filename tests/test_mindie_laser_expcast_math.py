"""Probability math and exact source transformation, without a CANN installation."""
from pathlib import Path
import struct

import pytest
import torch

from scripts import mindie_laser_expcast_patch as patch
from scripts import mindie_laser_expcast_reference as ref


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def independent_decode(code):
    exponent, mantissa = divmod(code, 8)
    value = mantissa * 2.0 ** -9 if exponent == 0 else (1 + mantissa / 8) * 2.0 ** (exponent - 7)
    return value / 256


def test_all_121_codes_match_independent_e4m3_decode():
    codes = torch.arange(121, dtype=torch.int16)
    expected = torch.tensor([independent_decode(int(c)) for c in codes], dtype=torch.float16)
    actual = ref.decode_codes(codes)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert actual[0] == 0 and actual[-1] == 1
    assert actual[1] == 2.0 ** -17  # E4M3 and FP16 subnormal case.
    assert torch.all(actual[1:] > actual[:-1])


def test_every_finite_nonpositive_fp16_bitpattern_matches_two_round_oracle():
    # Enumerate the entire 16-bit space, then select the required finite domain.
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    values = values[torch.isfinite(values) & (values >= -60000) & (values <= 0)]
    expected_codes = []
    for value in values.tolist():
        affine = fp32(fp32(value * ref.GAIN) + ref.BIAS)
        expected_codes.append(round(max(0.0, min(120.0, affine))))
    expected = torch.tensor(expected_codes, dtype=torch.int16)
    assert torch.equal(ref.expcast_codes(values), expected)
    probabilities = ref.expcast_probability(values)
    decoded = torch.tensor([independent_decode(c) for c in expected_codes], dtype=torch.float16)
    assert torch.equal(probabilities.view(torch.int16), decoded.view(torch.int16))
    assert torch.isfinite(probabilities).all()
    assert torch.all((probabilities >= 0) & (probabilities <= 1))


def test_masked_tail_zero_and_valid_row_has_nonzero_denominator():
    shifted = torch.tensor([[0, -0.25, -1, -10, -60000, -float("inf")]], dtype=torch.float16)
    probabilities = ref.expcast_probability(shifted)
    assert torch.equal(probabilities[0, -2:], torch.zeros(2, dtype=torch.float16))
    assert probabilities[0, 0] == 1
    assert probabilities.float().sum() >= 1
    # No 256x scale enters the existing half-precision section reduction.
    assert ref.expcast_probability(torch.zeros(16384, dtype=torch.float16)).sum() == 16384


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [0.125]])
def test_reference_rejects_out_of_contract_scores(values):
    with pytest.raises(ValueError):
        ref.expcast_probability(torch.tensor(values, dtype=torch.float16))


@pytest.mark.parametrize("codes", [[-1], [121]])
def test_decode_rejects_out_of_contract_codes(codes):
    with pytest.raises(ValueError):
        ref.decode_codes(torch.tensor(codes))


def source_fixture(newline=b"\n"):
    # Unrelated history exp deliberately coexists with the probability site.
    text = (b"// upstream copyright\n" + patch.INCLUDE_ANCHOR +
            b"vexp(subMaxValueResultVectorUbAddr, subMaxValueResultVectorUbAddr, 1, 1, 1, 8, 8);\n" +
            patch.BULK_VEXP + b"\nset_flag(PIPE_V, PIPE_MTE3, event_id);\nprocess_calc_sum(...);\n")
    return text.replace(b"\n", newline)


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_transform_is_exactly_reversible_and_preserves_history(newline):
    original = source_fixture(newline)
    changed = patch.patch_vector_forward(original)
    assert patch.unpatch_vector_forward(changed) == original
    assert changed.count(b"vexp(") == 1
    assert b"vexp(subMaxValueResultVectorUbAddr" in changed
    assert changed.endswith(b"process_calc_sum(...);" + newline)
    assert changed.count(b"vc_laser_expcast_probability(") == 1
    with pytest.raises(ValueError, match="already"):
        patch.patch_vector_forward(changed)


def test_changed_or_duplicated_source_anchor_fails_closed():
    original = source_fixture()
    for invalid in (original.replace(b"param.section_block_num * rowPerTime *", b"changed *"),
                    original + patch.BULK_VEXP,
                    original.replace(patch.INCLUDE_ANCHOR, b"")):
        with pytest.raises(ValueError, match="anchors changed"):
            patch.patch_vector_forward(invalid)


def test_pinned_source_roundtrip_if_checkout_is_available():
    source = Path(__file__).resolve().parents[1] / "build/mindie-sd-reference/csrc/ops/laser_attention/op_kernel/VectorForward.h"
    if not source.is_file():
        pytest.skip("Pinned upstream checkout is not required for the CPU unit suite")
    original = source.read_bytes()
    changed = patch.patch_vector_forward(original)
    assert original.count(b"vexp(") == 2
    assert changed.count(b"vexp(") == 1
    assert patch.unpatch_vector_forward(changed) == original


def test_scratch_and_repeat_bounds_for_every_pinned_section_size():
    # Upstream count is section_block_num * rowPerTime(4) * block_size(128).
    # Each score ping buffer contains 16384 half elements; scratch is 32768 B.
    for count in range(128, 16385, 128):
        covered = 0
        for offset in range(0, count, 8192):
            chunk = min(count - offset, 8192)
            assert chunk * 4 <= 32768
            assert 1 <= chunk // 64 <= 255
            assert 1 <= chunk // 128 <= 255
            assert chunk % 128 == 0
            covered += chunk
        assert covered == count


def test_contract_names_agree():
    assert patch.CONTRACT == ref.CONTRACT
