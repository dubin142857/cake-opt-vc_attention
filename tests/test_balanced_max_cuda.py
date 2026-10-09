"""Real CUDA tree and probability helpers; independent PyTorch integer oracle."""
import pytest
import torch

probe=pytest.importorskip("_vc_balanced_max_math")


@pytest.mark.parametrize("pattern",["random","each_position","ties","negative"])
def test_integer_max_tree_all_positions_and_extreme_int32(pattern):
    torch.manual_seed(3500)
    x=torch.randint(-2**31,2**31-1,(8192,8,8),device="cuda",dtype=torch.int32)
    if pattern=="each_position":
        x.fill_(-2**31)
        i=torch.arange(x.shape[0],device="cuda")
        pos=i%32;k=pos//4
        for r in range(2):x[i,k,r*2+(pos%4//2)*4+pos%2]=2**31-1
    if pattern=="ties":x.fill_(2**31-1)
    if pattern=="negative":x.fill_(-2**31)
    out=probe.max_tree(x)
    expected=torch.stack([x[:,:,[r*2,r*2+1,r*2+4,r*2+5]].amax((1,2)) for r in range(2)],-1)
    torch.testing.assert_close(out[...,0],expected,rtol=0,atol=0)
    torch.testing.assert_close(out[...,1],expected,rtol=0,atol=0)


@pytest.mark.parametrize("steps",[1,7,296])
@pytest.mark.parametrize("pattern",["random","stable","fallback","tiny","transition"])
@pytest.mark.parametrize("side_stream",[False,True])
def test_full_probability_helper_bitwise_parity(steps,pattern,side_stream):
    torch.manual_seed(3500+steps)
    x=torch.randint(-2**21,2**21+1,(steps,2,128,8,8),device="cuda",dtype=torch.int32)
    scale=(torch.arange(128,device="cuda").remainder(4)+1).float().expand(steps,2,128).contiguous()*1e-6
    initial=torch.full((2,128,2),-5e6,device="cuda")
    if pattern=="stable":x[:]=x[:1].clone()
    if pattern=="fallback":scale*=1000
    if pattern=="tiny":scale.fill_(1e-38)
    if pattern=="transition":
        x.fill_(-100000)
        scale.copy_(torch.linspace(.01,.00001,steps,device="cuda").view(-1,1,1))
    expected=probe.run(x,scale,initial,False)
    stream=torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):actual=probe.run(x,scale,initial,True)
    stream.synchronize()
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    decoded=actual[0].view(torch.uint8).view(torch.float8_e4m3fn).float().reshape(steps,2,128,4,2,2,4)
    torch.testing.assert_close(actual[1][...,4:6],decoded.double().sum((3,4,6)).float(),rtol=0,atol=0)
