"""GPU helper checks; full Hopper attention is validated by the remote runner."""
import pytest
import torch

probe=pytest.importorskip("_vc_split_mass_math")


def check(x,scale,initial,side_stream=False):
    expected=probe.run(x,scale,initial,False)
    stream=torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):actual=probe.run(x,scale,initial,True)
    stream.synchronize()
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    steps,blocks=x.shape[:2]
    codes=actual[0].view(torch.uint8).reshape(steps,blocks,128,4,2,2,4)
    decoded=codes.contiguous().view(torch.float8_e4m3fn).float().double().sum((3,4,6)).float()
    torch.testing.assert_close(actual[1][...,4:6],decoded,rtol=0,atol=0)
    return codes


@pytest.mark.parametrize("steps",[1,7,296])
@pytest.mark.parametrize("pattern",["random","stable","fallback","tiny","transition"])
@pytest.mark.parametrize("side_stream",[False,True])
def test_fp32_split_matches_vc28_bytes_max_alpha_mass(steps,pattern,side_stream):
    torch.manual_seed(steps+3400)
    x=torch.randint(-2**21,2**21+1,(steps,2,128,8,8),device="cuda",dtype=torch.int32)
    scale=(torch.arange(128,device="cuda").remainder(4)+1).float().expand(steps,2,128).contiguous()*1e-6
    initial=torch.full((2,128,2),-5e6,device="cuda")
    if pattern=="stable":x[:]=x[:1].clone()
    if pattern=="fallback":scale*=1000
    if pattern=="tiny":scale.fill_(1e-38)
    if pattern=="transition":
        x.fill_(-100000)
        scale.copy_(torch.linspace(.01,.00001,steps,device="cuda").view(-1,1,1))
    check(x,scale,initial,side_stream)


def test_all_positive_e4m3_pairs_including_zero_subnormal_and_max():
    pairs=121*121;blocks=(pairs+127)//128
    ids=torch.arange(blocks*128,device="cuda").remainder(pairs)
    lo,hi=ids//121,ids%121
    codes=torch.stack([lo,hi,lo,hi,hi,lo,hi,lo],-1)
    scale=torch.full((1,blocks,128),1e-5,device="cuda")
    gain=scale.flatten().double()*11.541560173034668
    scores=((codes.double()-119.6500015258789)/gain[:,None]).round().clamp_max(0).int()
    x=scores[:,None,:].expand(-1,8,-1).contiguous().reshape(1,blocks,128,8,8)
    initial=torch.zeros(blocks,128,2,device="cuda")
    actual=check(x,scale,initial)
    # Each row has sixteen of each member of the pair.
    first=actual.reshape(blocks*128,4,2,2,4)[:,0,0,0,:]
    torch.testing.assert_close(first[:,0].long(),lo,rtol=0,atol=0)
    torch.testing.assert_close(first[:,1].long(),hi,rtol=0,atol=0)
