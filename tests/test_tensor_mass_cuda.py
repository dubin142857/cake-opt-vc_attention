"""Actual FP16 MMA/FP32 reduction gates; native FP8 decode needs SM89+."""
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from validate_fused_quad_math import check_case,inputs
from validate_tensor_mass import validate as validate_mass


@pytest.fixture(scope="module")
def extension():
    if not torch.cuda.is_available():pytest.skip("CUDA required")
    return pytest.importorskip("_vc_tensor_mass_math")


def test_byte_exhaustion_subnormals_and_lane_isolation(extension):
    report=validate_mass(extension,torch.device("cuda:0"))
    assert len(report["cases"])==193 and report["status"]=="passed"


CASES=[(p,s) for p in ("random","stable","fallback","tiny","mixed") for s in (1,7,296)]
CASES += [(p,1) for p in ("codes","rounding_ties","all_zero","all_max")]


@pytest.mark.parametrize("pattern,steps",CASES)
@pytest.mark.parametrize("side",[False,True])
def test_P_state_and_mass_exact(extension,pattern,steps,side):
    row=check_case(extension,pattern,steps,torch.device("cuda:0"),side)
    assert row["P_and_state_bitwise_equal"] and row["fp64_mass_oracle_exact"]


def test_independent_warps_may_take_different_fallback_routes(extension):
    x,scale,initial=inputs("random",7,torch.device("cuda:0"))
    scale=scale.view(7,2,4,32)
    scale[:,:,::2,0]=1e-38 # a single lane forces its entire warp to fall back
    scale[::2,:,1::2,:]=.1
    a=extension.check_probability(x,scale.reshape(7,2,128),initial,False)
    b=extension.check_probability(x,scale.reshape(7,2,128),initial,True)
    for old,new in zip(a,b):
        torch.testing.assert_close(new.view(torch.uint8),old.view(torch.uint8),rtol=0,atol=0)
