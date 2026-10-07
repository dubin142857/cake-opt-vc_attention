"""Optional any-CUDA test of real shared-history code; not a WGMMA test."""
import pytest
import torch

probe=pytest.importorskip("_sage_vc_history_probe")


@pytest.mark.parametrize("steps,blocks",[(1,1),(2,3),(5,7),(296,4)])
@pytest.mark.parametrize("retain_last",[False,True])
def test_shared_history_matches_independent_fp32_recurrence(steps,blocks,retain_last):
    torch.manual_seed(2121+steps)
    # Every row/channel differs: catch swapped query rows, slots and CTAs.
    products=torch.randn(steps,blocks,64,64,device="cuda")
    products*=torch.linspace(.001,100,64,device="cuda")
    alphas=torch.rand(steps,blocks,64,device="cuda")
    alphas[...,::3]=1
    alphas[...,1::7]=0
    expected=torch.zeros_like(products[0])
    for product,alpha in zip(products,alphas):
        expected=expected*alpha.unsqueeze(-1)+product
    # Same production shared helper, arbitrary rows and zero/one rescaling.
    torch.testing.assert_close(probe.run(products,alphas,retain_last),expected,rtol=0,atol=0)
