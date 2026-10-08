"""Independent FP8 block-scale oracle, dispatch, and native preparation tests."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/"third_party/sageattention_local/sageattention_local"


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle=load("fp8_test_oracle",PACKAGE/"vc_reference.py")
validator=load("fp8_test_validator",ROOT/"scripts/validate_fp8_qk.py")


@pytest.mark.parametrize("cast",[False,True])
def test_fp8_oracle_preserves_block_scales_when_sampling(cast):
    torch.manual_seed(2711)
    q=torch.randn(1,2,129,64)
    k=torch.randn(1,2,257,64)
    q[:,:,64:128]*=4
    k[:,:,128:256]*=.125
    q8,qs=validator.quant_reference(q,64)
    k8,ks=validator.quant_reference(k,128)
    v=torch.zeros(1,2,64,384)
    v[...,:257]=torch.randn(1,2,64,257)
    perm=oracle.value_permutation(384,"cpu")
    state=dict(q=q8,k=k8,v=v.index_select(-1,perm.argsort()).to(torch.float8_e4m3fn),
               qs=qs,ks=ks,vs=torch.ones(1,2,64),qk_format="fp8_block",expcast=cast,block_values=False,scale=.125)
    all_rows=oracle.reference(state,chunk=32)
    ids=[0,7,63,64,65,127,128]
    sampled=oracle.reference(state,query_indices=ids,chunk=3)
    torch.testing.assert_close(sampled,all_rows[:,:,ids],rtol=2e-6,atol=2e-6)


@pytest.mark.parametrize("cast",[False,True])
@pytest.mark.parametrize("nk",[1,129,257])
def test_zero_logits_tail_and_v_scale_known_answer(cast,nk):
    q=torch.zeros(1,1,65,64).to(torch.float8_e4m3fn)
    k=torch.zeros(1,1,nk,64).to(torch.float8_e4m3fn)
    n=((nk+127)//128)*128
    v=torch.zeros(1,1,64,n)
    v[...,:nk]=2
    state=dict(q=q,k=k,v=v.index_select(-1,oracle.value_permutation(n,"cpu").argsort()).to(torch.float8_e4m3fn),
               qs=torch.ones(1,1,2),ks=torch.ones(1,1,(nk+127)//128),vs=torch.ones(1,1,64)*3,
               qk_format="fp8_block",expcast=cast,block_values=False,scale=.125)
    out=oracle.reference(state)
    weight=1. if cast else 448/(2**float(torch.tensor(8.807)))
    torch.testing.assert_close(out,torch.full_like(out,6*weight),rtol=2e-6,atol=2e-6)


def test_fp8_wrapper_dispatch_and_unsupported_cases(monkeypatch):
    package=ModuleType("fp8_stub_package"); package.__path__=[str(PACKAGE)]
    monkeypatch.setitem(sys.modules,"fp8_stub_package",package)
    vc=load("fp8_stub_package.vc",PACKAGE/"vc.py")
    monkeypatch.setitem(sys.modules,"fp8_stub_package.vc",vc)
    api=load("fp8_stub_package.vc_fp8",PACKAGE/"vc_fp8.py")
    calls=[]
    ext=SimpleNamespace(forward_fp8=lambda *args:calls.append(args) or "fp8 output")
    monkeypatch.setattr(api,"_extension",lambda:ext)
    state={k:i for i,k in enumerate(("q","k","v","qs","ks","vs","expcast","output_type","scale"))}
    state["qk_format"]="fp8_block"
    assert api.forward_prepared(state)=="fp8 output"
    assert calls==[tuple(range(9))]
    with pytest.raises(ValueError,match="FP8 block"): api.forward_prepared({**state,"qk_format":"int8"})
    with pytest.raises(ValueError,match="FP8 block"): api.forward_prepared({**state,"block_values":True})
    q=torch.zeros(1,1,65,64,dtype=torch.bfloat16)
    with pytest.raises(ValueError,match="Hopper"): api.sageattn_vc_fp8(q,q,q)
    with pytest.raises(ValueError,match="noncausal"): api.sageattn_vc_fp8(q,q,q,is_causal=True)
    with pytest.raises(ValueError,match="GQA"): api.sageattn_vc_fp8(q,q.expand(1,2,65,64),q)


def test_implementation_gate_rejects_bad_and_nonfinite_results():
    ref=torch.ones(3,4)
    validator.check_implementation(ref,ref)
    with pytest.raises(AssertionError): validator.check_implementation(ref*1.03,ref)
    with pytest.raises(AssertionError): validator.check_implementation(ref*float("nan"),ref)


def test_native_qk_preparation_on_cuda():
    ext=pytest.importorskip("_vc_sm90")
    if not hasattr(ext,"prepare_qk_fp8") or not torch.cuda.is_available(): pytest.skip("vc27 CUDA build required")
    result=validator.validate_quantization(ext)
    assert result["cases"]==36


def test_native_fp8_checks_payload_before_dispatch():
    ext=pytest.importorskip("_vc_sm90")
    if not hasattr(ext,"forward_fp8") or not torch.cuda.is_available(): pytest.skip("vc27 CUDA build required")
    q=torch.zeros(1,1,65,64,device="cuda",dtype=torch.int8)
    k=torch.zeros(1,1,129,64,device="cuda",dtype=torch.int8)
    v=torch.zeros(1,1,64,256,device="cuda").to(torch.float8_e4m3fn)
    qs=torch.ones(1,1,2,device="cuda");ks=qs.clone();vs=torch.ones(1,1,64,device="cuda")
    with pytest.raises(RuntimeError): ext.forward_fp8(q,k,v,qs,ks,vs,True,1,.125)
    with pytest.raises(RuntimeError): ext.prepare_qk_fp8(q,k)
    with pytest.raises(RuntimeError): ext.fp8_kernel_resources(3,128)
