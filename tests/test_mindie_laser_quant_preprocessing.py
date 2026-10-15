"""CPU regressions for the narrow FP32 INT8 rounding-boundary allowance."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mindie_laser_quant_backend as backend
from mindie_laser_quant_preprocessing import check_preprocessing
from mindie_laser_quant_reference import structured_case, validate_small


def _inputs(dtype=torch.float16):
    q = torch.zeros((1, 1, 17, 64), dtype=dtype)
    k = torch.zeros((1, 1, 129, 64), dtype=dtype)
    q[0, 0, 0, :5] = torch.tensor([3, .25, -.25, .5, -.5], dtype=dtype)
    k[0, 0, 0, :5] = q[0, 0, 0, :5]
    k[0, 0, 128, 0] = 7
    v = torch.ones_like(k)
    v[..., 0], v[..., 1] = 0, .25
    return q, k, v


def _with_codes(x, scales):
    """Regenerate codes using the actual supplied scale, including padding."""
    padded = F.pad(x.float(), (0, 128 - x.shape[-1], 0, (-x.shape[-2]) % 256))
    blocks = padded.reshape(*x.shape[:2], -1, 128, 128)
    return (blocks / scales[..., None, None]).round().clamp(-127, 127).to(torch.int8).reshape(padded.shape)


def _assert_result(inputs, payload, passed, expected=None):
    result = check_preprocessing(inputs, payload, expected)
    assert result["passed"] is passed, result
    assert isinstance(result["errors"], list)
    assert bool(result["errors"]) is not passed, result
    # Diagnostics are persisted in the benchmark report, including failures.
    json.dumps(result, allow_nan=False)
    return result


@pytest.fixture
def small_case():
    inputs = _inputs()
    return inputs, backend.prepare_quant_tensors(*inputs)


@pytest.fixture(scope="module")
def reported_case():
    inputs = structured_case(torch.float16, 64, 17, 4097)
    return inputs, backend.prepare_quant_tensors(*inputs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("v_quantize", [False, True])
def test_exact_cpu_payload_passes_with_and_without_cached_expected(dtype, v_quantize):
    inputs = _inputs(dtype)
    expected = backend.prepare_quant_tensors(*inputs, v_quantize=v_quantize)
    _assert_result(inputs, expected, True)
    _assert_result(inputs, expected, True, expected)


def test_reported_six_k_differences_are_one_ulp_from_the_same_half_integer(reported_case):
    inputs, expected = reported_case
    scales = expected.k_scale.repeat_interleave(128, dim=-1)[..., :4097, None]
    ratio = inputs[1].float() / scales
    midpoint = ratio.floor() + .5
    near = (ratio != midpoint) & (
        (torch.nextafter(ratio, torch.full_like(ratio, -float("inf"))) == midpoint)
        | (torch.nextafter(ratio, torch.full_like(ratio, float("inf"))) == midpoint)
    )
    indices = {tuple(index) for index in near.nonzero().tolist()}
    assert indices == {(0, head, token, 32) for head in (0, 1) for token in (639, 2047, 3455)}
    assert torch.equal(ratio[near], torch.full_like(ratio[near], -63.499996185302734))
    actual = replace(expected, k=expected.k.clone())
    for index in indices:
        assert expected.k[index].item() == -63
        actual.k[index] = -64
    _assert_result(inputs, actual, True, expected)


def test_near_half_integer_does_not_allow_the_wrong_neighbor(reported_case):
    inputs, expected = reported_case
    actual = replace(expected, k=expected.k.clone())
    actual.k[0, 0, 639, 32] = -62
    _assert_result(inputs, actual, False, expected)


@pytest.mark.parametrize("delta", [-2, 2])
def test_near_half_integer_never_allows_a_two_code_difference(reported_case, delta):
    inputs, expected = reported_case
    actual = replace(expected, k=expected.k.clone())
    actual.k[0, 0, 639, 32] = int(expected.k[0, 0, 639, 32]) + delta
    _assert_result(inputs, actual, False, expected)


@pytest.mark.parametrize("field", ["q", "k"])
@pytest.mark.parametrize("delta", [-1, 1])
def test_nonboundary_one_code_difference_fails(small_case, field, delta):
    inputs, expected = small_case
    index = (0, 0, 0, 1)
    scale = getattr(expected, field + "_scale")[0, 0, 0]
    ratio = inputs[0 if field == "q" else 1][index].float() / scale
    assert abs(float(ratio - (ratio.floor() + .5))) > .01
    changed = getattr(expected, field).clone()
    changed[index] = int(changed[index]) + delta
    _assert_result(inputs, replace(expected, **{field: changed}), False, expected)


@pytest.mark.parametrize("field", ["q", "k", "v"])
@pytest.mark.parametrize("padding", ["sequence", "feature"])
def test_nonzero_padding_fails(small_case, field, padding):
    inputs, expected = small_case
    changed = getattr(expected, field).clone()
    index = (0, 0, 17 if field == "q" else 129, 0) if padding == "sequence" else (0, 0, 0, 64)
    assert changed[index].item() == 0
    changed[index] = 1
    _assert_result(inputs, replace(expected, **{field: changed}), False, expected)


@pytest.mark.parametrize("field", ["q", "k"])
def test_input_zeros_must_remain_exact_zero(small_case, field):
    inputs, expected = small_case
    changed = getattr(expected, field).clone()
    changed[0, 0, 1, 1] = 1
    _assert_result(inputs, replace(expected, **{field: changed}), False, expected)


@pytest.mark.parametrize("field", ["q", "k"])
def test_bad_scale_fails_even_when_codes_match_that_scale(small_case, field):
    inputs, expected = small_case
    scales = getattr(expected, field + "_scale").clone()
    scales[0, 0, 0] *= 2
    codes = _with_codes(inputs[0 if field == "q" else 1], scales)
    actual = replace(expected, **{field: codes, field + "_scale": scales})
    _assert_result(inputs, actual, False, expected)


def test_small_scale_error_cannot_hide_under_an_absolute_tolerance():
    q = torch.full((1, 1, 17, 64), 1e-30, dtype=torch.bfloat16)
    inputs = (q, q.clone(), torch.zeros_like(q))
    expected = backend.prepare_quant_tensors(*inputs)
    scales = expected.q_scale.clone()
    scales[..., 0] *= 2
    assert 0 < float(abs(scales[0, 0, 0] - expected.q_scale[0, 0, 0])) < 1e-7
    actual = replace(expected, q_scale=scales, q=_with_codes(q, scales))
    _assert_result(inputs, actual, False, expected)


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_scales_must_be_finite_and_positive(small_case, value):
    inputs, expected = small_case
    scales = expected.q_scale.clone()
    scales[0, 0, 0] = value
    _assert_result(inputs, replace(expected, q_scale=scales), False, expected)


def test_two_ulp_scale_drift_fails(small_case):
    inputs, expected = small_case
    scales = expected.q_scale.clone()
    upper = torch.tensor(float("inf"))
    scales[0, 0, 0] = torch.nextafter(torch.nextafter(scales[0, 0, 0], upper), upper)
    actual = replace(expected, q_scale=scales, q=_with_codes(inputs[0], scales))
    _assert_result(inputs, actual, False, expected)


@pytest.mark.parametrize("field,index", [("q_scale", (0, 0, 1)),
                                         ("v_scale", (0, 0, 0)),
                                         ("v_scale", (0, 0, 64))])
def test_zero_blocks_zero_channels_and_padding_require_exact_unit_scale(small_case, field, index):
    inputs, expected = small_case
    scales = getattr(expected, field).clone()
    assert scales[index].item() == 1
    scales[index] = torch.nextafter(scales[index], torch.tensor(float("inf")))
    _assert_result(inputs, replace(expected, **{field: scales}), False, expected)


@pytest.mark.parametrize("field", ["q", "k"])
@pytest.mark.parametrize("direction", [-float("inf"), float("inf")])
def test_one_ulp_actual_scale_with_regenerated_codes_is_valid(small_case, field, direction):
    inputs, expected = small_case
    scales = getattr(expected, field + "_scale").clone()
    scales[0, 0, 0] = torch.nextafter(scales[0, 0, 0], torch.tensor(direction))
    actual = replace(expected, **{field + "_scale": scales,
                                field: _with_codes(inputs[0 if field == "q" else 1], scales)})
    _assert_result(inputs, actual, True, expected)


@pytest.mark.parametrize("value,wrong", [(.5, 1), (1.5, 1), (2.5, 3),
                                         (-.5, -1), (-1.5, -1), (-2.5, -3)])
def test_power_of_two_scale_preserves_exact_ties_to_even(value, wrong):
    q = torch.zeros((1, 1, 17, 64), dtype=torch.float16)
    q[0, 0, 0, 0], q[0, 0, 0, 1] = 127, value
    inputs = (q, q.clone(), torch.zeros_like(q))
    expected = backend.prepare_quant_tensors(*inputs)
    assert expected.q_scale[0, 0, 0].item() == 1
    assert expected.q[0, 0, 0, 1].item() == round(value)
    actual = replace(expected, q=expected.q.clone())
    actual.q[0, 0, 0, 1] = wrong
    _assert_result(inputs, actual, False, expected)


@pytest.mark.parametrize("v_quantize", [False, True])
def test_v_corruption_is_not_covered_by_the_int8_exception(v_quantize):
    inputs = _inputs()
    expected = backend.prepare_quant_tensors(*inputs, v_quantize=v_quantize)
    actual = replace(expected, v=expected.v.clone())
    actual.v[0, 0, 0, 1] += 1
    _assert_result(inputs, actual, False, expected)


def test_plain_v_scale_remains_exact_unit():
    inputs = _inputs()
    expected = backend.prepare_quant_tensors(*inputs, v_quantize=False)
    scales = expected.v_scale.clone()
    scales[0, 0, 1] = torch.nextafter(scales[0, 0, 1], torch.tensor(float("inf")))
    _assert_result(inputs, replace(expected, v_scale=scales), False, expected)


@pytest.mark.parametrize("field", ["q", "k", "v", "q_scale", "k_scale", "v_scale"])
def test_payload_tensor_dtype_mismatch_is_a_reported_failure(small_case, field):
    inputs, expected = small_case
    changed = getattr(expected, field).to(torch.float64)
    _assert_result(inputs, replace(expected, **{field: changed}), False, expected)


@pytest.mark.parametrize("field", ["q", "k", "v", "q_scale", "k_scale", "v_scale"])
def test_payload_tensor_shape_mismatch_is_a_reported_failure(small_case, field):
    inputs, expected = small_case
    changed = getattr(expected, field)[..., :-1]
    _assert_result(inputs, replace(expected, **{field: changed}), False, expected)


def test_validate_small_saves_failed_preprocessing_before_any_native_call():
    class CorruptPreprocessing:
        v_quantize = True
        expcast = False

        def prepare(self, *inputs):
            payload = backend.prepare_quant_tensors(*inputs)
            # Preserve the separate head-count-one exact-scale RNE probe.
            if inputs[0].shape[1] == 2:
                payload = replace(payload, k=payload.k.clone())
                payload.k[0, 0, 0, 0] = 1  # This original input is exactly zero.
            return payload

        def describe_payload(self, payload):
            return {}

        def run_prepared(self, payload):
            pytest.fail("Native execution must not follow failed preprocessing validation")

        def __call__(self, *inputs):
            pytest.fail("Full API must not follow failed preprocessing validation")

    report, saved = {}, []
    with pytest.raises(AssertionError, match="preprocessing failed"):
        validate_small(CorruptPreprocessing(), "cpu", report,
                       lambda: saved.append(copy.deepcopy(report)), torch)
    assert saved[-1] == report
    assert report["preprocessing_rounding_probes"][0]["passed"] is True
    assert len(report["small_accuracy"]) == 1
    failure = saved[-1]["small_accuracy"][0]
    assert failure["status"] == "failed"
    assert failure["preprocessing_cpu_comparison"]["passed"] is False
    assert failure["preprocessing_cpu_comparison"]["errors"]
    assert "full_vs_prepared" not in failure
    json.dumps(saved[-1], allow_nan=False)
