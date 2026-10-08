"""Reproduce the 1/8 NaN channels and verify the zero-safe V quantization pass."""
import importlib.util
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("zero_v_validation",ROOT/"scripts/validate_fp8_qk.py")
validation=importlib.util.module_from_spec(spec);spec.loader.exec_module(validation)


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
@pytest.mark.parametrize("dim",[64,128])
def test_zero_v_channels_reproduce_failure_and_preserve_other_bytes(dtype,dim):
    probe=pytest.importorskip("_vc_fp8_values_probe")
    # Same zero-channel ratio and padding as the user's failing K257 case.
    v=torch.arange(dim,device="cuda",dtype=dtype).remainder(8).view(1,1,dim,1).expand(1,2,dim,384).contiguous()/8
    v[...,257:]=0
    old,old_scale=probe.legacy(v)
    fixed,scale=probe.safe(v)
    assert torch.isnan(old.float()).float().mean().item()==.125
    assert torch.isfinite(fixed.float()).all()
    torch.testing.assert_close(scale,old_scale,rtol=0,atol=0)
    zero=(old_scale==0)
    assert torch.count_nonzero(fixed.view(torch.uint8)[zero])==0
    assert torch.equal(fixed.view(torch.uint8)[~zero],old.view(torch.uint8)[~zero])


def test_native_value_channels_tails_and_streams():
    ext=pytest.importorskip("_vc_sm90")
    if not hasattr(ext,"prepare_global_values"): pytest.skip("vc27.1 native build required")
    assert validation.validate_global_values(ext)["cases"]==60


def test_payload_failure_identifies_preprocessing_stage():
    state={key:torch.zeros(2,3) for key in ("q","k","v","qs","ks","vs")}
    validation.require_finite_payload(state,"known case")
    state["v"][0,0]=float("nan")
    with pytest.raises(AssertionError,match="payload v; attention not launched"):
        validation.require_finite_payload(state,"known case")
