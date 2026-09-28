"""Optional native-helper test. Build tests/cuda/sage_vc_tile_probe.cu as
_sage_vc_tile_probe with torch's CUDAExtension, then put it on PYTHONPATH.
The installed production attention does not contain this probe.
"""
import importlib.util
from pathlib import Path
import pytest
import torch

probe = pytest.importorskip("_sage_vc_tile_probe")
spec = importlib.util.spec_from_file_location("vc_oracle", Path(__file__).resolve().parents[1] /
    "third_party/sageattention_local/sageattention_local/vc_reference.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("mixed_scales", [False, True])
@pytest.mark.parametrize("channel_scale", [False, True])
@pytest.mark.parametrize("packed_probability", [False, True])
def test_real_expcast_register_packing_online_rescale_and_mean_restore(seed,optimized,mixed_scales,channel_scale,packed_probability):
    torch.manual_seed(seed)
    device = "cuda"
    scores = torch.randn(64,128,device=device) * 5
    scores[:,117:] = -float("inf")
    out, product = (torch.randn_like(scores) for _ in range(2))
    m = torch.randn(64,device=device) * 2 + 1
    d = torch.rand(64,device=device) * 200
    scale = torch.rand(128,device=device) + 0.01
    mean = (torch.randn(128,device=device) * 8).half()
    codes, actual, maximum, denom = probe.run(scores,out,product,m,d,scale,mean,optimized,mixed_scales,channel_scale,packed_probability)
    if packed_probability:
        previous = probe.run(scores,out,product,m,d,scale,mean,optimized,mixed_scales,channel_scale,False)
        for new,old in zip((codes,actual,maximum,denom),previous):
            torch.testing.assert_close(new,old,rtol=0,atol=0)
    factors = .137 * (1 + (torch.arange(128,device=device,dtype=torch.float32) % 8) // 2) if mixed_scales else .25
    expected_max = torch.maximum(m,(scores * factors).amax(-1))
    alpha = (m-expected_max).exp()
    expected_codes = oracle.expcast_codes(scores * factors - expected_max[:,None])
    mass = expected_codes.view(torch.float8_e4m3fn).float().sum(-1)
    if optimized:
        # Independent FP64 emulation of each FP32 FMA's single rounding.
        if channel_scale:
            centered = (product.double() + mass.double()[:,None] * mean.double()).float()
            expected = (out * alpha[:,None] + centered) * scale
        else:
            centered = (product.double() + mass.double()[:,None] * mean.double()).float()
            expected = ((out * alpha[:,None]).double() + centered.double() * scale.double()).float()
        old = probe.run(scores,out,product,m,d,scale,mean,False,mixed_scales)
        for actual_item,old_item in zip((codes,maximum,denom),(old[0],old[2],old[3])):
            torch.testing.assert_close(actual_item,old_item,rtol=0,atol=0)
    else:
        expected = out * alpha[:,None] + (product + mass[:,None] * mean.float()) * scale
    torch.testing.assert_close(codes,expected_codes,rtol=0,atol=0)
    torch.testing.assert_close(maximum,expected_max,rtol=0,atol=0)
    torch.testing.assert_close(denom,d * alpha + mass,rtol=1e-6,atol=1e-5)
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-5)


@pytest.mark.parametrize("packed_probability", [False, True])
def test_hot_expcast_boundaries_and_all_nonnegative_fp8_pairs(packed_probability):
    # Exercise each rounding threshold and both adjacent FP32 inputs, masked
    # -inf, very negative scores, negative zero, and all 121^2 decoded pairs.
    device = "cuda"
    boundaries = ((torch.arange(120,device=device,dtype=torch.float64) + .5 -
                   119.6500015258789) / 11.541560173034668).float()
    shifted = torch.cat((torch.linspace(-16,0,65537,device=device), boundaries,
        torch.nextafter(boundaries, torch.full_like(boundaries,-float("inf"))),
        torch.nextafter(boundaries, torch.full_like(boundaries,float("inf"))),
        torch.tensor([-float("inf"), -1e30, -0., 0.],device=device)))
    ids = torch.arange(shifted.numel(),device=device).remainder(121 * 121)
    lo, hi = ids.remainder(121), ids // 121
    encoded, decoded = probe.run_helpers(shifted, (lo | (hi << 8)).int(), packed_probability)
    torch.testing.assert_close(encoded,oracle.expcast_codes(shifted),rtol=0,atol=0)
    expected = torch.stack((lo,hi),-1).byte().view(torch.float8_e4m3fn).float()
    torch.testing.assert_close(decoded,expected,rtol=0,atol=0)


def test_streamed_probability_matches_vc6_across_long_online_history():
    torch.manual_seed(920)
    device = "cuda"
    shape = (64,128)
    mean = torch.randn(128,device=device).half()
    scale = torch.ones(128,device=device)
    out = torch.randn(shape,device=device)
    maximum = torch.linspace(-20,20,64,device=device)
    denominator = torch.logspace(-6,8,64,device=device)
    for tile in range(64):
        scores = torch.randn(shape,device=device)*(1+tile%11)
        # Extreme tail, ordinary tail, uniform rows, and old/new online maxima.
        scores[:16, 1 if tile%2 else 117:] = -float("inf")
        scores[16:32] = 0
        product = torch.randn_like(scores)
        old = probe.run(scores,out,product,maximum,denominator,scale,mean,True,True,True,False)
        new = probe.run(scores,out,product,maximum,denominator,scale,mean,True,True,True,True)
        for a,b in zip(new,old): torch.testing.assert_close(a,b,rtol=0,atol=0)
        _,out,maximum,denominator = old
