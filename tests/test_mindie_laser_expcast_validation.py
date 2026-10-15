"""Structured CPU oracles must distinguish changed probabilities and padding."""
import pytest
import torch

from scripts import mindie_laser_expcast_validation as validation


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [64, 128])
def test_structured_oracle_separates_softmax_and_handles_all_tails(dtype, dim):
    for nq, nk in validation.SHAPES:
        qkv = validation.structured_case(dtype, dim, nq, nk, torch)
        out = validation.structured_reference(*qkv, torch)
        dense_probability = validation.structured_reference(*qkv, torch, expcast=False)
        metric = validation.error_metrics(dense_probability[..., 2:], out[..., 2:], torch)
        assert out.shape == (1, 2, nq, dim) and out.dtype == dtype
        assert torch.equal(out[..., 0], torch.zeros_like(out[..., 0]))
        assert torch.equal(out[..., 1], torch.full_like(out[..., 1], 2))
        if nk == 1:
            assert metric["relative_rmse"] == 0
        else:
            assert metric["relative_rmse"] > .02


def test_zero_padding_is_observably_different_from_masked_tail():
    q, k, v = validation.structured_case(torch.float16, 128, 17, 129, torch)
    wanted = validation.structured_reference(q, k, v, torch)
    k_bad = torch.cat((k, torch.zeros((1, 2, 127, 128), dtype=k.dtype)), dim=2)
    v_bad = torch.cat((v, torch.zeros_like(k_bad[:, :, 129:])), dim=2)
    unmasked = validation.structured_reference(q, k_bad, v_bad, torch)
    assert validation.error_metrics(unmasked, wanted, torch)["relative_rmse"] > .5
    assert not torch.allclose(unmasked[..., 1], wanted[..., 1])


class CPUAdapter:
    def __init__(self, expcast=True, corrupt_constant=False):
        self.expcast = expcast
        self.corrupt_constant = corrupt_constant

    def prepare(self, *inputs):
        return inputs

    def run_prepared(self, payload):
        result = validation.structured_reference(*payload, torch, expcast=self.expcast)
        if self.corrupt_constant:
            result[..., 1] = 1
        return result

    def __call__(self, *inputs):
        return self.run_prepared(inputs)

    def describe_payload(self, payload):
        return {"scope": "CPU test double, never evidence of NPU execution"}


def test_validation_callback_rejects_unchanged_softmax_and_saves_evidence():
    report, snapshots = {}, []
    with pytest.raises(AssertionError, match="decoded_reference_all_rows_nonconstant_channels"):
        validation.validate_small(CPUAdapter(expcast=False), "cpu", report,
                                  lambda: snapshots.append(True), torch)
    assert snapshots
    row = report["small_accuracy"][0]
    assert row["full_vs_prepared"]["passed"]
    assert not row["decoded_reference_all_rows_nonconstant_channels"]["passed"]
    assert row["ordinary_softmax_vs_decoded_reference"]["relative_rmse"] > .02


def test_validation_callback_rejects_wrong_constant_despite_probability_match():
    report = {}
    with pytest.raises(AssertionError):
        validation.validate_small(CPUAdapter(corrupt_constant=True), "cpu", report, lambda: None, torch)
    row = report["small_accuracy"][0]
    assert row["decoded_reference_all_rows_nonconstant_channels"]["passed"]
    assert row["constant_value_padding_check"] == "failed"


def test_validation_callback_records_complete_cpu_control_run():
    report = {}
    rows = validation.validate_small(CPUAdapter(), "cpu", report, lambda: None, torch)
    assert len(rows) == 24
    assert all(row["status"] == "passed" for row in rows)
    assert all(row["decoded_reference_all_rows_nonconstant_channels"]["passed"] for row in rows)
    assert all("passed" not in row["dense_fp32_all_rows_all_keys"] for row in rows)


def test_oracle_rejects_unmodelled_qk_and_nonzero_block_max():
    q, k, v = validation.structured_case(torch.float16, 64, 17, 129, torch)
    q[..., 1] = 1
    with pytest.raises(ValueError, match="single-feature"):
        validation.structured_reference(q, k, v, torch)
    q[..., 1] = 0
    k[..., 0] -= 1
    with pytest.raises(ValueError, match="zero maximum"):
        validation.structured_reference(q, k, v, torch)


def test_nonfinite_output_is_an_implementation_failure():
    result = validation.error_metrics(torch.tensor([float("nan")]), torch.ones(1), torch, .01)
    assert not result["finite"] and not result["passed"]
