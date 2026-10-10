"""Exact regrouping and fallback state; asynchronous PV requires remote H200."""
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from validate_fused_quad_math import check_case,inputs


@pytest.fixture(scope="module")
def extension():
    if not torch.cuda.is_available():pytest.skip("CUDA helper test")
    return pytest.importorskip("_vc_mass_tail_math")


CASES=[(p,s) for p in ("random","stable","fallback","tiny","mixed") for s in (1,7,296)]
CASES += [(p,1) for p in ("codes","rounding_ties","all_zero","all_max")]


@pytest.mark.parametrize("pattern,steps",CASES)
@pytest.mark.parametrize("side",[False,True])
def test_mass_tail_probability_is_bitwise_exact(extension,pattern,steps,side):
    row=check_case(extension,pattern,steps,torch.device("cuda:0"),side)
    assert row["P_and_state_bitwise_equal"] and row["fp64_mass_oracle_exact"]


def test_fallback_and_regular_warps_leave_no_stale_carried_pairs(extension):
    x,scale,initial=inputs("random",7,torch.device("cuda:0"))
    scale=scale.view(7,2,4,32)
    scale[:,:,:,0]=1e-38  # One unsafe lane makes its whole warp take fallback.
    scale[::2,:,1::2,:]=.1
    scale[1::2,:,::2,:]=1e-6
    a=extension.check_probability(x,scale.reshape(7,2,128),initial,False)
    b=extension.check_probability(x,scale.reshape(7,2,128),initial,True)
    for old,new in zip(a,b):
        torch.testing.assert_close(new.view(torch.uint8),old.view(torch.uint8),rtol=0,atol=0)
