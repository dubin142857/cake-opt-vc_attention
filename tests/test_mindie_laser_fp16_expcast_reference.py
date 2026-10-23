"""CPU oracle coverage only; no claim of CCE compilation or NPU correctness."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import mindie_laser_fp16_expcast_reference as reference
import benchmark_mindie_laser_quant as quant


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(min(old,4))
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("dtype", [torch.float16,torch.bfloat16])
@pytest.mark.parametrize("tail", [False,True])
def test_signal_reference_constants_and_own_algorithm(dtype, tail):
    payload = reference.signal_payload(dtype, tail=tail)
    actual = reference.decoded_reference(payload)
    assert actual.shape == payload.shape and actual.dtype == dtype
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[...,0],torch.zeros_like(actual[...,0]))
    assert torch.equal(actual[...,1],torch.full_like(actual[...,1],2))
    # Independent one-row numerator/denominator computed directly from fixture K
    # codes. This checks the QK transport/gain contract as well as P semantics.
    x = payload.k[0,0,:payload.keys,0].half()
    p = reference.probability(x).double()
    expected = (p @ payload.v[0,0,:payload.keys].double()/p.sum()).half().to(dtype)
    torch.testing.assert_close(actual[0,0,0],expected,rtol=0,atol=0)


@pytest.mark.parametrize("dtype", [torch.float16,torch.bfloat16])
def test_old_exponential_cannot_pass_new_algorithm_gate(dtype):
    payload = reference.signal_payload(dtype)
    expected = reference.decoded_reference(payload)
    unchanged = reference.decoded_reference(payload, probability_fn=lambda x:x.float().exp().half())
    error = quant.common.error_metrics(unchanged[...,2:],expected[...,2:])
    assert error["relative_rmse"] > .01
    with pytest.raises(AssertionError,match="implementation RMSE"):
        reference.checked(error,"old exp")


@pytest.mark.parametrize("dtype", [torch.float16,torch.bfloat16])
def test_amplified_tail_rejects_probability_flush_to_zero(dtype):
    payload = reference.signal_payload(dtype,tail=True)
    expected = reference.decoded_reference(payload)
    flushed = reference.decoded_reference(payload, probability_fn=lambda x:
        torch.where(reference.probability(x)<2**-14,0,reference.probability(x)))
    error = quant.common.error_metrics(flushed[...,2:],expected[...,2:])
    assert error["relative_rmse"] == 1
    assert (expected[...,2:6]>0).all()
    with pytest.raises(AssertionError):
        reference.checked(error,"FTZ")


def test_changing_max_refuses_misleading_global_oracle():
    payload = reference.signal_payload(torch.float16)
    payload.k[:,:,128,0] = 3
    with pytest.raises(ValueError,match="shared maximum"):
        reference.decoded_reference(payload)


def test_shared_nonzero_max_is_valid():
    payload = reference.signal_payload(torch.float16)
    original = reference.decoded_reference(payload)
    payload.k[:,:,:payload.keys,0] += 3
    assert torch.equal(reference.decoded_reference(payload),original)


@pytest.mark.parametrize("case", reference.CASES, ids=lambda case:case["name"])
def test_mapping_fixtures_obey_shared_max_and_cover_keys(case):
    inputs = reference._mapping_inputs(case,torch.float16)
    payload = reference.prepare_quant_tensors(*inputs,v_quantize=False)
    selected = reference._selected_queries(case["queries"])
    out = reference.decoded_reference(payload,selected)
    assert out.shape == (case["batch"],case["heads"],selected.numel(),case["dim"])
    assert torch.isfinite(out).all()


def test_fixture_inventory_covers_dtype_dim_padding_and_history():
    assert {case["dim"] for case in reference.CASES} == {64,128}
    assert {case["dim"] for case in reference.CASES if case["keys"]>4096} == {64,128}
    assert any(case["queries"]>256 for case in reference.CASES)
    assert any(case["keys"]==1 for case in reference.CASES)


def test_invalid_payload_and_selected_rows_are_rejected():
    payload = reference.signal_payload(torch.float16)
    for ids in (torch.tensor([]),torch.tensor([-1]),torch.tensor([17]),torch.tensor([[0]])):
        with pytest.raises(ValueError,match="query indices"):
            reference.decoded_reference(payload,ids)
    with pytest.raises(ValueError,match="plain FP16 V"):
        reference.decoded_reference(replace(payload,v_quantize=True))
    with pytest.raises(ValueError,match="positive chunk"):
        reference.decoded_reference(payload,chunk=0)


@pytest.mark.parametrize("field", ["relative_rmse","max_abs"])
@pytest.mark.parametrize("bad", [float("nan"),float("inf")])
def test_nonfinite_diagnostics_never_pass(field,bad):
    error = dict(relative_rmse=0,max_abs=0)
    error[field] = bad
    with pytest.raises(AssertionError):
        reference.checked(error,"finite diagnostic",float("inf"))
