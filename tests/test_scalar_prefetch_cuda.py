"""Real GPU checks of carried scalars and unchanged P/max/alpha/mass."""
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from validate_fused_quad_math import check_case,inputs


@pytest.fixture(scope="module")
def extension():
    if not torch.cuda.is_available():pytest.skip("CUDA helper test")
    return pytest.importorskip("_vc_scalar_prefetch_math")


CASES=[(p,s) for p in ("random","stable","fallback","tiny","mixed") for s in (1,7,296)]
CASES += [(p,1) for p in ("codes","rounding_ties","all_zero","all_max")]


@pytest.mark.parametrize("pattern,steps",CASES)
@pytest.mark.parametrize("side",[False,True])
def test_carried_gain_and_range_predicate_are_exact(extension,pattern,steps,side):
    row=check_case(extension,pattern,steps,torch.device("cuda:0"),side)
    assert row["P_and_state_bitwise_equal"] and row["fp64_mass_oracle_exact"]


def test_mixed_lane_and_warp_transitions_do_not_reuse_stale_gain(extension):
    x,scale,initial=inputs("random",7,torch.device("cuda:0"))
    scale=scale.view(7,2,4,32)
    scale[:,:,:,0]=1e-38
    scale[::2,:,1::2,:]=.1
    scale[1::2,:,::2,:]=1e-6
    a=extension.check_probability(x,scale.reshape(7,2,128),initial,False)
    b=extension.check_probability(x,scale.reshape(7,2,128),initial,True)
    for old,new in zip(a,b):
        torch.testing.assert_close(new.view(torch.uint8),old.view(torch.uint8),rtol=0,atol=0)


def test_normal_threshold_and_adjacent_fp32_scales(extension):
    x,scale,initial=inputs("random",7,torch.device("cuda:0"))
    # Exercise either side of the positive-normal guard plus lane-varying gain.
    threshold=torch.tensor(torch.finfo(torch.float32).tiny,device="cuda")
    below=torch.nextafter(threshold,torch.zeros_like(threshold))
    above=torch.nextafter(threshold,torch.full_like(threshold,float("inf")))
    values=torch.stack((below,threshold,above,threshold*2))
    scale.copy_(values[torch.arange(128,device="cuda")%4])
    scale[1::2]*=1e30
    a=extension.check_probability(x,scale,initial,False)
    b=extension.check_probability(x,scale,initial,True)
    for old,new in zip(a,b):
        torch.testing.assert_close(new.view(torch.uint8),old.view(torch.uint8),rtol=0,atol=0)
