"""Real vc22 helpers, including exact codes, DPX max and history recurrence.

Non-Hopper CUDA implements the DPX intrinsic in software; this checks values,
not Hopper instruction throughput or WGMMA integration.
"""
import pytest
import torch

probe=pytest.importorskip("_sage_vc_work_probe")


@pytest.mark.parametrize("steps,blocks",[(1,1),(2,3),(7,2),(296,1)])
@pytest.mark.parametrize("pattern",["random","stable","rising","fallback"])
def test_exact_probability_and_independent_history(steps,blocks,pattern):
    torch.manual_seed(2222+steps)
    scores=torch.randint(-(2**21),2**21+1,(steps,blocks,128,8,8),device="cuda",dtype=torch.int32)
    scales=(torch.arange(128,device="cuda").remainder(4)+1).float().expand(steps,blocks,128).contiguous()*1e-6
    initial=torch.full((blocks,128,2),-5e6,device="cuda")
    if pattern=="stable":
        scores[1:]=scores[:1].clone()
    elif pattern=="rising":
        scores=torch.full_like(scores,-100000)
        scores+=torch.arange(steps,device="cuda",dtype=torch.int32).view(-1,1,1,1,1)*5000
    elif pattern=="fallback":
        scales*=1000
        initial[...,0]=129
        initial[...,1]=-129
    products=torch.randn(scores.shape,device="cuda")*torch.linspace(.01,10,8,device="cuda")
    reference=probe.run(scores,scales,initial,products,0)
    for mode in (1,2,3):
        actual=probe.run(scores,scales,initial,products,mode)
        for a,b in zip(actual,reference):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    packed,stats,output=reference
    # Independent row layout, lane-specific scaling, then four-lane max.
    row_ids=([0,1,4,5],[2,3,6,7])
    local=torch.stack([scores[...,ids].amax((-1,-2)).float()*scales for ids in row_ids],-1)
    local=local.view(steps,blocks,32,4,2).amax(3,keepdim=True).expand(-1,-1,-1,4,-1).reshape(steps,blocks,128,2)
    maxima=torch.maximum(local.cummax(0).values,initial.unsqueeze(0))
    torch.testing.assert_close(stats[...,:2],maxima,rtol=0,atol=0)
    # Decode bytes independently. Layout is [K-pair, K-in-pair, row, four bytes].
    decoded=packed.view(torch.uint8).view(torch.float8_e4m3fn).float().view(steps,blocks,128,4,2,2,4)
    mass=decoded.sum((3,4,6))
    torch.testing.assert_close(stats[...,4:6],mass,rtol=0,atol=0)
    denom=torch.ones_like(initial)
    history=torch.zeros_like(products[0])
    for step in range(steps):
        alpha=stats[step,...,2:4]
        denom=denom*alpha+mass[step]
        expanded=alpha[...,[0,0,1,1,0,0,1,1]].unsqueeze(-2)
        history=history*expanded+products[step]
        torch.testing.assert_close(stats[step,...,6:8],denom,rtol=0,atol=0)
    torch.testing.assert_close(output,history,rtol=0,atol=0)
    if pattern=="stable" and steps>1:
        assert (stats[1:,...,2:4]==1).all().item()


def test_sparse_row_updates_and_nondefault_stream():
    # Only one lane group changes its maximum: other warps can skip scaling.
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        scores=torch.zeros(3,2,128,8,8,device="cuda",dtype=torch.int32)
        scores[1,0,4,3,0]=2**21
        scores[2,1,127,7,7]=-(2**21)
        scales=torch.full((3,2,128),1e-5,device="cuda")
        initial=torch.zeros(2,128,2,device="cuda")
        products=torch.randn(scores.shape,device="cuda")
        baseline=probe.run(scores,scales,initial,products,0)
        actual=probe.run(scores,scales,initial,products,3)
    stream.synchronize()
    for a,b in zip(actual,baseline): torch.testing.assert_close(a,b,rtol=0,atol=0)
