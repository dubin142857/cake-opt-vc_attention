"""CPU contracts and known answers; real Hopper validation is a separate script."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "third_party/sageattention_local/sageattention_local"


def load(name):
    spec = importlib.util.spec_from_file_location("test_sage_" + name, PACKAGE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle, api = load("vc_reference"), load("vc")


def test_expcast_bytes_and_packing_known_values():
    codes = oracle.expcast_codes(torch.tensor([-float("inf"), -20., 0., 1.]))
    assert codes.tolist() == [0, 0, 120, 120]
    assert codes.view(torch.float8_e4m3fn).float().tolist() == [0., 0., 256., 256.]
    perm = oracle.value_permutation(256, "cpu")
    assert perm[:16].tolist() == [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15]
    assert perm.sort().values.tolist() == list(range(256))


def payload(nq, nk, block_values, expcast):
    dim, tiles = 64, (nk+127)//128
    return {"q": torch.zeros(1,1,nq,dim,dtype=torch.int8),
            "k": torch.zeros(1,1,nk,dim,dtype=torch.int8),
            "v": torch.zeros(1,1,dim,tiles*128,dtype=torch.float8_e4m3fn),
            "qs": torch.ones(1,1,((nq+63)//64)*32), "ks": torch.ones(1,1,tiles*4),
            "vs": torch.ones(1,1,tiles,dim) if block_values else torch.ones(1,1,dim),
            "means": torch.zeros(1,1,tiles,dim,dtype=torch.float16) if block_values else torch.empty(0,dtype=torch.float16),
            "expcast": expcast, "block_values": block_values, "output_type": 1, "scale": dim**-0.5}


@pytest.mark.parametrize("expcast", [False, True])
def test_block_mean_restoration_and_tail_weight_known_answer(expcast):
    state = payload(65,257,True,expcast)
    state["means"][0,0,:, :] = torch.tensor([1., -2., 6.])[:,None]
    state["vs"][0,0,:, :] = torch.tensor([2., 3., 4.])[:,None]
    out = oracle.reference(state, chunk=7)
    expected = (128*2 + 128*(-6) + 24) / 257
    torch.testing.assert_close(out,torch.full_like(out,expected),rtol=2e-6,atol=1e-6)


@pytest.mark.parametrize("block_values", [False, True])
@pytest.mark.parametrize("expcast", [False, True])
def test_single_key_preserves_nonzero_value(block_values, expcast):
    state = payload(17,1,block_values,expcast)
    data = state["v"].float()
    data[...,0] = torch.arange(64).remainder(9)-4
    state["v"] = data.to(torch.float8_e4m3fn)
    state["vs"].fill_(0.125)
    state["means"].fill_(2.)
    out = oracle.reference(state)
    # Ordinary Sage denominator is unrounded P, numerator is FP8(P).
    weight = 1. if expcast else 448. / (2**float(torch.tensor(8.807)))
    expected = ((data[...,0] * weight + (2 if block_values else 0)) * 0.125).unsqueeze(2).expand_as(out)
    torch.testing.assert_close(out,expected,rtol=1e-6,atol=1e-6)


def test_query_sampling_keeps_original_scale_indices():
    state = payload(129,129,False,True)
    torch.manual_seed(778)
    state["q"].random_(-3,4)
    state["k"].random_(-3,4)
    state["qs"].uniform_(0.1,0.3)
    state["ks"].uniform_(0.2,0.5)
    state["v"] = torch.randn_like(state["v"].float()).to(torch.float8_e4m3fn)
    ids = [0,7,8,15,16,63,64,128]
    all_rows = oracle.reference(state,chunk=32)
    sampled = oracle.reference(state,query_indices=ids,chunk=3)
    torch.testing.assert_close(sampled,all_rows[:,:,ids],rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize("expcast",[False,True])
def test_channel_scale_epilogue_matches_expanded_per_tile_scales(expcast):
    torch.manual_seed(918)
    state = payload(65,257,True,expcast)
    state["q"].random_(-4,5)
    state["k"].random_(-4,5)
    state["qs"].fill_(.1)
    state["ks"].fill_(.15)
    state["v"] = torch.randn_like(state["v"].float()).to(torch.float8_e4m3fn)
    state["means"].normal_()
    scales = torch.linspace(.01,1,64).view(1,1,64)
    channel = {**state,"vs":scales}
    old = {**state,"vs":scales.unsqueeze(2).expand(1,1,3,64).contiguous()}
    torch.testing.assert_close(oracle.reference(channel),oracle.reference(old),rtol=5e-5,atol=1e-6)


@pytest.mark.parametrize("expcast",[False,True])
def test_shared_scale_nonzero_mean_known_answer(expcast):
    state = payload(65,257,True,expcast)
    state["vs"] = torch.full((1,1,64),.25)
    state["means"][0,0] = torch.tensor([8.,-24.,96.])[:,None]
    out = oracle.reference(state)
    expected = (128*2-128*6+24)/257
    torch.testing.assert_close(out,torch.full_like(out,expected),rtol=2e-6,atol=1e-6)


@pytest.mark.parametrize("mode,flags",[("optimized",(True,True,True,True,True)),("vc10",(True,True,True,True,False)),("vc9",(True,True,True,False,False)),("vc7",(True,True,False,False,False)),("vc6",(True,False,False,False,False)),("vc4",(False,False,False,False,False))])
def test_kernel_mode_dispatch_preserves_previous_schedules(monkeypatch,mode,flags):
    from types import SimpleNamespace
    calls = []
    def forward(*args,**kwargs):
        calls.append((args,kwargs))
        return "output"
    monkeypatch.setattr(api,"_extension",lambda:SimpleNamespace(forward=forward))
    keys = ("q","k","v","qs","ks","vs","means","expcast","block_values","output_type","scale")
    state = {key:object() for key in keys}
    assert api.forward_prepared(state,kernel_mode=mode) == "output"
    assert calls == [(tuple(state[key] for key in keys),dict(optimized=flags[0],packed_probability=flags[1],compact_values=flags[2],pipeline_probability=flags[3],interleaved_probability=flags[4]))]


@pytest.mark.parametrize("kwargs,match", [
    ({"is_causal": True}, "noncausal"), ({"return_lse": True}, "LSE"),
    ({"tensor_layout": "XYZ"}, "tensor_layout"), ({"expcast": 1}, "bool"),
    ({"clusters": 0}, "clustering"), ({"iterations": True}, "clustering"),
    ({"kernel_mode": "unknown"}, "kernel_mode"),
    ({"v_scale_mode": "unknown"}, "v_scale_mode"),
])
def test_unsupported_modes_rejected_before_native_dispatch(kwargs,match):
    q = torch.zeros(1,2,5,64,dtype=torch.float16)
    with pytest.raises(ValueError,match=match):
        api.sageattn_vc(q,q,q,**kwargs)


def test_gqa_cpu_and_mixed_dtypes_are_not_silent_fallbacks():
    q = torch.zeros(1,2,5,64,dtype=torch.float16)
    with pytest.raises(ValueError,match="GQA"):
        api.sageattn_vc(q,q[:,:1],q[:,:1])
    with pytest.raises(ValueError,match="matching"):
        api.sageattn_vc(q,q.float(),q)
    with pytest.raises(ValueError,match="Hopper"):
        api.sageattn_vc(q,q,q)
