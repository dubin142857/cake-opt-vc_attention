"""Experiment routing and async register-lifetime invariants."""
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import mass_tail_native as native
import benchmark_cuda_expcast as bench


def test_two_distinct_paths_with_scale_prefetch_baseline():
    a=bench.parse_args(["--experiment","mass_tail"])
    assert a.output.name=="h200_cuda_expcast_mass_tail.json"
    assert a.trace_dir.name=="cuda_expcast_mass_tail_traces"
    assert bench.variant_modes(native,"vc28_mass_tail",native.BASELINE)=={
        "vc28_scale_prefetch":False,"vc28_mass_tail":True}
    assert native.forward is native.base.forward
    assert native.EXACT_OUTPUT


def test_full_api_reprepares_but_core_reuses_same_payload():
    payload=object();prepared=[]
    def prepare(*args,**kwargs):
        assert kwargs=={"expcast":True,"v_smooth":False}
        state=object();prepared.append(state);return state
    backend=SimpleNamespace(forward=lambda ext,state,flag:(state,flag))
    core,full=bench.make_calls(None,SimpleNamespace(prepare=prepare),(),payload,backend,
                              "vc28_mass_tail",native.BASELINE)
    assert core[native.BASELINE]()==(payload,False)
    assert core["vc28_mass_tail"]()==(payload,True)
    assert full[native.BASELINE]()==(prepared[0],False)
    assert full["vc28_mass_tail"]()==(prepared[1],True)
    assert prepared[0] is not prepared[1]


def test_no_extra_mma_fence_commit_wait_or_tail_change():
    original=native.base.candidate_kernel()
    new=native.candidate_kernel()
    marker="  {\n    p ^= 1;"
    assert new.rsplit(marker,1)[1]==original.rsplit(marker,1)[1]
    assert original in native.generate_source()
    for op in ("warpgroup_arrive()","warpgroup_commit_batch()","warpgroup_wait<0>()",
               "wgmma::wgmma_f8f8f32<","load_async_4D("):
        assert new.count(op)==original.count(op)
    start=new.index("    // Only decoded half pairs")
    end=new.index("    // Read the NEXT tile",start)
    assert new.rfind("warpgroup_commit_batch()",0,start)>new.index("// A separate")
    assert "RS_f8" not in new[start:end] and "RO_temp" not in new[start:end].split("\n",1)[1]


def test_prepare_and_finish_do_not_read_pending_mma_inputs():
    helper=native.helper_source()
    finish=helper.split("void vc_probability_finish_mass_tail(",1)[1].split("template<bool BitEncode",1)[0]
    assert "const VCLateMass& late, float (&mass)[2]" in finish
    assert "packed" not in finish and "scores" not in finish and "wgmma" not in finish
    assert "__hadd" not in helper
    assert "__half2 last[2][2] = {};" in helper
    assert "vc_probability_prepare_integer<false,false,true,false>" in native.check_source()


def test_mixed_warp_pipeline_uses_modified_payload_for_both_arms(monkeypatch):
    import torch
    monkeypatch.setattr(torch,"float16",torch.float32)
    monkeypatch.setattr(torch,"bfloat16",torch.float32)
    calls=[]
    def prepare(q,k,v,**kw):return dict(q=q,qs=torch.ones(2,2,4,32))
    def output(state):
        qs=state["qs"].view(2,2,4,4,8)
        assert torch.all(qs[:,:,:,1::2,:]==128) and torch.all(qs[:,:,:,::2,:]==1)
        return torch.ones_like(state["q"])
    def forward(ext,state,flag):calls.append((state,flag));return output(state)
    monkeypatch.setattr(native,"forward",forward)
    rows=native.validate_pipeline(None,SimpleNamespace(prepare=prepare),
        SimpleNamespace(reference=lambda s,chunk:output(s)),torch.device("cpu"),lambda rows:None)
    assert len(rows)==8 and all(row["status"]=="passed" for row in rows)
    for i in range(0,len(calls),2):
        assert calls[i][0] is calls[i+1][0]
        assert [calls[i][1],calls[i+1][1]]==[False,True]
