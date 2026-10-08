"""Validation stream contracts; Hopper math is not emulated by these tests."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation = load("vc_validation_stream_test", ROOT / "scripts/validate_sage_vc.py")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Stream execution requires CUDA")
def test_prepared_core_waits_for_inputs_and_returns_ready_output():
    caller, worker = torch.cuda.Stream(), torch.cuda.Stream()
    seen=[]
    def forward(state, *, kernel_mode):
        assert torch.cuda.current_stream()==worker
        seen.append(kernel_mode)
        torch.cuda._sleep(1000000)
        return state["q"] + 3
    with torch.cuda.stream(caller):
        q=torch.zeros(4096,device="cuda")
        torch.cuda._sleep(1000000)
        q.fill_(2)
        out=validation.dual_query_forward_on_stream(
            SimpleNamespace(forward_prepared=forward),{"q":q},"vc26_stagger",worker)
        assert torch.cuda.current_stream()==caller
        consumed=out*2
        del out
    caller.synchronize()
    torch.testing.assert_close(consumed,torch.full_like(consumed,10),rtol=0,atol=0)
    assert seen==["vc26_stagger"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Stream execution requires CUDA")
def test_dual_validator_keeps_real_public_wrapper_on_default_stream(monkeypatch):
    # Use the real public wrapper and CUDA streams on SM86. Only quantization
    # and attention math are stand-ins; there is no Hopper kernel execution.
    package=ROOT / "third_party/sageattention_local/sageattention_local"
    api=load("vc_stream_api_test",package / "vc.py")
    oracle=load("vc_stream_oracle_test",package / "vc_reference.py")
    module=ModuleType("sageattention_local")
    module.vc_reference=oracle
    monkeypatch.setitem(sys.modules,"sageattention_local",module)
    monkeypatch.setitem(sys.modules,"sageattention_local.vc_reference",oracle)
    monkeypatch.setattr(torch.cuda,"get_device_capability",lambda *_:(9,0))
    monkeypatch.setattr(validation,"DUAL_QUERY_SHAPES",[(193,513)])
    default=torch.cuda.default_stream()
    preparations, public_calls, native_calls=[],[],[]

    def prepare(q,k,v,**kwargs):
        assert torch.cuda.current_stream(q.device)==default
        preparations.append(q.shape)
        n=k.shape[2]
        decoded=torch.nn.functional.pad(v.float().transpose(2,3),(0,(-n)%128))
        perm=oracle.value_permutation(decoded.shape[-1],q.device)
        return dict(q=q,k=k,v=decoded.index_select(-1,perm.argsort()).to(torch.float8_e4m3fn),
                    vs=torch.ones_like(v[:,:,0,:],dtype=torch.float32),expcast=kwargs["expcast"])

    def forward(state,*,kernel_mode):
        stream=torch.cuda.current_stream(state["q"].device)
        native_calls.append((kernel_mode,stream==default))
        decoded=state["v"].float().index_select(-1,oracle.value_permutation(state["v"].shape[-1],state["q"].device))
        weight=1. if state["expcast"] else 448./(2**float(torch.tensor(8.807)))
        return ((decoded[...,:state["k"].shape[2]].mean(-1)*state["vs"]*weight)
                .unsqueeze(2).expand_as(state["q"]).to(state["q"].dtype).contiguous())

    real_public=api.sageattn_vc
    def public(*args,**kwargs):
        assert torch.cuda.current_stream(args[0].device)==default
        public_calls.append(kwargs["tensor_layout"])
        return real_public(*args,**kwargs)

    monkeypatch.setattr(api,"prepare",prepare)
    monkeypatch.setattr(api,"forward_prepared",forward)
    monkeypatch.setattr(api,"_extension",lambda:SimpleNamespace(sage_vc_kernel_version=23))
    monkeypatch.setattr(api,"sageattn_vc",public)
    report=validation.validate_dual_query(api,"cuda:0")
    assert report["status"]=="passed" and len(report["cases"])==16
    assert report["known_answer_cases"]==16
    assert public_calls.count("HND")==public_calls.count("NHD")==16
    assert len(preparations)==40
    # Baseline on default; candidates on side stream AND default public API.
    assert all(is_default for mode,is_default in native_calls if mode in ("vc16","vc22_both"))
    assert sum(not is_default for _,is_default in native_calls)==64
    assert report["streams"]["preprocessing"]==report["streams"]["public_api"]=="default"
    assert report["streams"]["candidate_core"]=="nondefault"
    # Keep the production guard: callers cannot safely move preprocessing.
    with torch.cuda.stream(torch.cuda.Stream()):
        q=torch.zeros(1,1,65,64,device="cuda",dtype=torch.float16)
        with pytest.raises(ValueError,match="default CUDA stream"):
            real_public(q,q,q,expcast=True,v_smooth=False,kernel_mode="vc26_stagger")
