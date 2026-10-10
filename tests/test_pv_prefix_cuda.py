"""Execute the split probability helper; full WGMMA tested by H200 launcher."""
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from validate_fused_quad_math import check_case


@pytest.fixture(scope="module")
def extension():
    if not torch.cuda.is_available():pytest.skip("CUDA helper test")
    return pytest.importorskip("_vc_pv_prefix_math")


CASES=[(p,s) for p in ("random","stable","fallback","tiny","mixed") for s in (1,7,296)]
CASES += [(p,1) for p in ("codes","rounding_ties","all_zero","all_max")]


@pytest.mark.parametrize("pattern,steps",CASES)
@pytest.mark.parametrize("side",[False,True])
def test_prefix_probability_states_are_bitwise_exact(extension,pattern,steps,side):
    row=check_case(extension,pattern,steps,torch.device("cuda:0"),side)
    assert row["P_and_state_bitwise_equal"] and row["fp64_mass_oracle_exact"]
