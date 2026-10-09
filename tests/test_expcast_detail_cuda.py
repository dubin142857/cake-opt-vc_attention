"""Production vs generated detailed helper, exact FP8 bytes and FP32 state."""
import pytest
import torch

probe=pytest.importorskip("_vc_expcast_detail_math")


@pytest.mark.parametrize("steps",[1,7,296])
@pytest.mark.parametrize("pattern",["random","stable","fallback","tiny"])
@pytest.mark.parametrize("side_stream",[False,True])
def test_detail_preserves_probability_and_reports_routes(steps,pattern,side_stream):
    torch.manual_seed(3300+steps)
    scores=torch.randint(-(2**21),2**21+1,(steps,2,128,8,8),device="cuda",dtype=torch.int32)
    scales=(torch.arange(128,device="cuda").remainder(4)+1).float().expand(steps,2,128).contiguous()*1e-6
    initial=torch.full((2,128,2),-5e6,device="cuda")
    if pattern=="stable":scores[:]=scores[:1].clone()
    if pattern=="fallback":scales*=1000
    if pattern=="tiny":scales.fill_(1e-38)
    baseline=probe.run(scores,scales,initial,False)
    stream=torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):actual=probe.run(scores,scales,initial,True)
    stream.synchronize()
    for a,b in zip(actual[:2],baseline[:2]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    decoded=actual[0].view(torch.uint8).view(torch.float8_e4m3fn).float().view(steps,2,128,4,2,2,4)
    torch.testing.assert_close(actual[1][...,4:6],decoded.sum((3,4,6)),rtol=0,atol=0)
    clocks=actual[2].cpu()
    fallback=pattern in ("fallback","tiny")
    assert (clocks[...,27]==(2 if fallback else 1)).all()
    ids=[10,11,12,26] if fallback else list(range(10,27))
    timestamps=clocks[...,ids]
    assert (timestamps>0).all() and (timestamps[...,1:]>timestamps[...,:-1]).all()
    if fallback:assert (clocks[...,13:26]==0).all()
