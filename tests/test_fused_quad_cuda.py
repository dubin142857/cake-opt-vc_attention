"""Exact probability gate on actual GPU helpers; no attention speed claim."""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_fused_quad_math import check_case
probe = pytest.importorskip("_vc_fused_quad_math")


@pytest.mark.parametrize("side", [False, True])
@pytest.mark.parametrize("pattern,steps", [(p, s) for p in
    ("random", "stable", "fallback", "tiny", "mixed") for s in (1, 7, 296)] +
    [(p, 1) for p in ("codes", "rounding_ties", "all_zero", "all_max")])
def test_fused_quad_probability_exact(pattern, steps, side):
    result = check_case(probe, pattern, steps, torch.device("cuda:0"), side)
    assert result["P_and_state_bitwise_equal"] and result["fp64_mass_oracle_exact"]
