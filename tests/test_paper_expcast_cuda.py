"""Opt-in native helper tests: paper bytes, exact local sums, scheduling parity."""
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from benchmark_paper_expcast import encoding_inputs
probe=pytest.importorskip("_vc_paper_math")


def test_paper_encoding_boundaries():
    x=encoding_inputs("cuda")
    oracle=(x.double()*11.541560173034668+119.6500015258789).float().round().clamp(0,120).byte()
    torch.testing.assert_close(probe.encode(x),oracle,rtol=0,atol=0)


@pytest.mark.parametrize("steps",[1,7,296])
@pytest.mark.parametrize("pattern",["random","stable","large","tiny"])
@pytest.mark.parametrize("side_stream",[False,True])
def test_probability_matches_paper_and_other_schedule(steps,pattern,side_stream):
    torch.manual_seed(steps)
    x=torch.randint(-2**21,2**21+1,(steps,2,128,8,8),device="cuda",dtype=torch.int32)
    scale=(torch.arange(128,device="cuda").remainder(4)+1).float().expand(steps,2,128).contiguous()*1e-6
    if pattern=="stable":x[:]=x[:1].clone()
    if pattern=="large":scale*=1000
    if pattern=="tiny":scale.fill_(1e-38)
    initial=torch.full((2,128,2),-5e6,device="cuda")
    p,stats=probe.run(x,scale,initial,False)
    stream=torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):op,os=probe.run(x,scale,initial,True)
    stream.synchronize()
    torch.testing.assert_close(op,p,rtol=0,atol=0);torch.testing.assert_close(os,stats,rtol=0,atol=0)
    decoded=p.view(torch.uint8).view(torch.float8_e4m3fn).float().view(steps,2,128,4,2,2,4)
    torch.testing.assert_close(stats[...,4:6],decoded.sum((3,4,6)),rtol=0,atol=0)
    # Independent FP32 score scaling/subtraction, widened FMA, RNE oracle.
    # CUDA fast math flushes denormal scale; use its documented FTZ behavior.
    effective=torch.where(scale.abs()<torch.finfo(torch.float32).tiny,torch.zeros_like(scale),scale)
    scores=x.float()*effective[...,None,None]
    maximum=scores.reshape(steps,2,128,8,2,2,2).amax((3,4,6))
    maximum=maximum.reshape(steps,2,32,4,2).amax(3,keepdim=True).expand(-1,-1,-1,4,-1).reshape(steps,2,128,2)
    maximum=maximum.cummax(0).values.clamp_min(-5e6)
    torch.testing.assert_close(stats[...,:2],maximum,rtol=0,atol=0)
    shifted=scores.reshape(steps,2,128,8,2,2,2)-maximum[...,None,None,:,None]
    codes=(shifted.double()*11.541560173034668+119.6500015258789).float().round().clamp(0,120).byte()
    # Kernel groups 2 K fragments, then row, then the row's 4 values.
    codes=codes.permute(0,1,2,3,5,4,6).reshape(steps,2,128,4,2,2,4)
    torch.testing.assert_close(p.view(torch.uint8).reshape_as(codes),codes,rtol=0,atol=0)
