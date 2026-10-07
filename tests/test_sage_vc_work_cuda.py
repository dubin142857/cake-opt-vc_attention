"""Real vc22-vc25 helpers, including exact codes, max state and history recurrence.

Non-Hopper CUDA implements the DPX intrinsic in software; this checks values,
not Hopper instruction throughput or WGMMA integration.
"""
import pytest
import torch

probe=pytest.importorskip("_sage_vc_work_probe")


@pytest.mark.parametrize("steps",[1,2,7,296])
@pytest.mark.parametrize("fallback",[False,True])
def test_deferred_mass_zero_prologue_and_masked_tail(steps,fallback):
    torch.manual_seed(2525+steps)
    scores=torch.randint(-10000,10001,(steps,1,128,8,8),device="cuda",dtype=torch.int32)
    scales=torch.full((steps,1,128),.1 if fallback else .0001,device="cuda")
    initial=torch.full((1,128,2),-5e6,device="cuda")
    products=torch.randn(scores.shape,device="cuda")
    baseline=probe.run(scores,scales,initial,products,0,True)
    for mode in (7,8):
        actual=probe.run(scores,scales,initial,products,mode,True)
        for a,b in zip(actual,baseline):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    # Independent check: masked coordinates encode zero, including single tile.
    codes=baseline[0][-1].view(torch.uint8).view(128,4,2,2,4)
    codes=codes.permute(0,1,2,4,3).reshape(128,32,2)
    masked=torch.arange(32,device="cuda")[None,:]>=1+torch.arange(128,device="cuda")[:,None]%32
    assert (codes[masked]==0).all().item()


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
    for mode in (1,2,3,4,5,6,7,8):
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
        actuals=[probe.run(scores,scales,initial,products,mode) for mode in (3,4,5,6,7,8)]
    stream.synchronize()
    for actual in actuals:
        for a,b in zip(actual,baseline): torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize("pattern",["stable_varying_gain","negative_to_safe","safe_to_fallback","tiny_update"])
def test_maxstate_refresh_and_affine_range_transitions(pattern):
    steps=9
    torch.manual_seed(2323)
    scores=torch.randint(-10000,10001,(steps,2,128,8,8),device="cuda",dtype=torch.int32)
    scales=torch.linspace(.0001,.0003,steps,device="cuda").view(-1,1,1).expand(steps,2,128).contiguous()
    initial=torch.full((2,128,2),10.,device="cuda")
    if pattern=="negative_to_safe":
        scores.fill_(-10000)
        scales.copy_(torch.linspace(.02,.001,steps,device="cuda").view(-1,1,1))
        initial.fill_(-1000.)
    elif pattern=="safe_to_fallback":
        scores.fill_(10000)
        scales.copy_(torch.linspace(.001,.02,steps,device="cuda").view(-1,1,1))
        initial.zero_()
    elif pattern=="tiny_update":
        scores.fill_(10000)
        scales.copy_(torch.linspace(1e-13,2e-13,steps,device="cuda").view(-1,1,1))
        initial.zero_()
    products=torch.randn(scores.shape,device="cuda")
    baseline=probe.run(scores,scales,initial,products,3)
    for mode in (4,5,6,7,8):
        actual=probe.run(scores,scales,initial,products,mode)
        for a,b in zip(actual,baseline): torch.testing.assert_close(a,b,rtol=0,atol=0)
    stats=baseline[1]
    if pattern in ("stable_varying_gain","tiny_update"):
        assert (stats[...,2:4]==1).all().item()
    if pattern=="stable_varying_gain":
        assert not torch.equal(baseline[0][0],baseline[0][-1])
    if pattern=="negative_to_safe":
        assert (stats[0,...,:2]<-128).all().item() and (stats[-1,...,:2].abs()<128).all().item()
    if pattern=="safe_to_fallback":
        assert (stats[0,...,:2].abs()<128).all().item() and (stats[-1,...,:2]>128).all().item()


def test_bit_encoder_half_integer_boundaries_and_random_affine():
    # Adjacent FP32 numbers around every byte boundary, including exact ties.
    centers=torch.arange(120,device="cuda",dtype=torch.float32)+.5
    bits=centers.view(torch.int32)[:,None]+torch.arange(-8,9,device="cuda",dtype=torch.int32)
    values=bits.reshape(-1).view(torch.float32)
    values=torch.cat((values,torch.tensor([-float("inf"),-1e30,-1.,-0.,0.,120.49],device="cuda")))
    inputs=torch.stack((values,torch.ones_like(values),torch.zeros_like(values)),1)
    codes,u=probe.encode(inputs)
    expected=torch.round(u.clamp(0,255)).int()
    torch.testing.assert_close(codes[:,0],expected,rtol=0,atol=0)
    torch.testing.assert_close(codes[:,1],expected,rtol=0,atol=0)
    # Nontrivial affine cancellation and different scale exponents, with
    # positive result below 120.5; rounded FMA is checked against CUDA oracle.
    torch.manual_seed(2424)
    for batch in range(8):
        score=torch.randint(-(2**21),2**21+1,(262144,),device="cuda").float()
        gain=torch.exp2(torch.empty_like(score).uniform_(-24,-12))
        target=torch.empty_like(score).uniform_(-1000,119.7)
        bias=target-score*gain
        codes,u=probe.encode(torch.stack((score,gain,bias),1))
        assert (u<120.5).all().item()
        torch.testing.assert_close(codes[:,1],codes[:,0],rtol=0,atol=0)
        torch.testing.assert_close(codes[:,1],torch.round(u.clamp(0,255)).int(),rtol=0,atol=0)


@pytest.mark.parametrize("scale",[0.,2.**-126,2.**-120,2.**-119,2.**-118,1e-6,1e-3])
def test_bit_encoding_scale_guard_and_extreme_int_scores(scale):
    torch.manual_seed(2425)
    scores=torch.randint(-(2**21),2**21+1,(7,1,128,8,8),device="cuda",dtype=torch.int32)
    scores[0,...,0,0]=2**21
    scores[0,...,1,1]=-(2**21)
    scales=torch.full((7,1,128),scale,device="cuda")
    initial=torch.zeros(1,128,2,device="cuda")
    products=torch.randn(scores.shape,device="cuda")
    expected=probe.run(scores,scales,initial,products,3)
    actual=probe.run(scores,scales,initial,products,6)
    for a,b in zip(actual,expected): torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize("pattern",["zero","maximum","each_code","mixed"])
def test_register_mass_exact_e4m3_sum(pattern):
    torch.manual_seed(2525)
    codes=torch.randint(0,121,(121,128,64),device="cuda",dtype=torch.uint8)
    if pattern=="zero": codes.zero_()
    elif pattern=="maximum": codes.fill_(120)
    elif pattern=="each_code": codes.copy_(torch.arange(121,device="cuda",dtype=torch.uint8)[:,None,None])
    actual=probe.sum_p(codes.view(torch.int32))
    expected=codes.view(torch.float8_e4m3fn).float().view(121,128,4,2,2,4).sum((2,3,5))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
