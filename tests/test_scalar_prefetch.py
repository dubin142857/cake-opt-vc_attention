"""Check native scheduling boundaries, remote gates and same-input comparisons."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import scalar_prefetch_native as native
import benchmark_cuda_expcast as bench


def test_opt_in_route_uses_measured_baseline():
    a=bench.parse_args(["--experiment","scalar_prefetch"])
    assert a.output.name=="h200_cuda_expcast_scalar_prefetch.json"
    assert bench.EXPERIMENTS[a.experiment][0] is native
    assert bench.variant_modes(native,"vc28_scalar_prefetch",native.BASELINE)=={
        "vc28_scale_prefetch":False,"vc28_scalar_prefetch":True}
    assert native.EXACT_OUTPUT


def test_baseline_tail_and_mma_counts_preserved():
    old=native.base.candidate_kernel();new=native.candidate_kernel()
    assert old in native.generate_source()
    # Tail consumes an already-rounded dequant scale; all remaining arithmetic
    # and output layout remain byte-for-byte the baseline source region.
    assert new[new.index("    // convert RS to float"):]==old[old.index("    // convert RS to float"):]
    for op in ("warpgroup_arrive()","warpgroup_commit_batch()","warpgroup_wait<0>()",
               "wgmma::wgmma_f8f8f32<","load_async_4D("):
        assert new.count(op)==old.count(op)
    assert "float dequant_scale = prepared.dequant_scale;" in new


def test_lookahead_is_between_pv_commit_and_wait_and_does_not_touch_operands():
    s=native.candidate_kernel()
    begin=s.index("    // Prepare only NEXT-tile")
    end=s.index("    // Scale history",begin)
    prefix=s[:begin]
    assert prefix.rfind("warpgroup_commit_batch()")>prefix.rfind("// wait for V")
    assert s.index("warpgroup_wait<0>()",begin)>end
    assert "K_scale[k_scale_idx + iter * k_scale_advance_offset]" in s[begin:end]
    assert "RS_f8" not in s[begin:end] and "RO_temp" not in s[begin:end].split("\n",1)[1]


def test_only_score_independent_guards_move():
    old=(native.VC/"work_reduction.cuh").read_text(encoding="utf-8")
    new=native.helper_source()
    begin=old.index("    #pragma unroll\n    for (int r=0;r<2;++r)")
    end=old.index("    if constexpr (StreamConvert)",begin)
    assert old[begin:end] in new
    assert "bool safe=prepared.scale_safe &&" in new
    assert "fabsf(maximum[q][0])<=128.f && fabsf(maximum[q][1])<=128.f" in new
    assert "if (__all_sync(0xffffffff,safe))" in new
    assert "s.dequant_scale = q_scale * k_scale;" in new
    assert "s.score_scale = attention_scale * s.dequant_scale;" in new


def resources(dim):
    return dict(registers_per_thread=168,local_bytes=0,
                dynamic_shared_bytes=320*dim,static_shared_bytes=128)


def test_resource_gate_checks_all_output_types_and_dimensions():
    calls=[]
    def get(dim,dtype):
        calls.append((dim,dtype))
        return {"expcast_plain":resources(dim),"expcast_probe":resources(dim)}
    result=native.check_resources(SimpleNamespace(resources=get))
    assert result["status"]=="passed" and calls==[(64,0),(64,1),(128,0),(128,1)]


@pytest.mark.parametrize("field,value",[("registers_per_thread",169),("local_bytes",4),
    ("dynamic_shared_bytes",57344),("static_shared_bytes",256)])
def test_resource_gate_rejects_remote_compiler_regressions(field,value):
    def get(dim,dtype):
        new=resources(dim);new[field]=value
        return {"expcast_plain":resources(dim),"expcast_probe":new}
    with pytest.raises(RuntimeError,match="resource gate failed"):
        native.check_resources(SimpleNamespace(resources=get))


def test_pipeline_transition_validation_shares_modified_payload(monkeypatch):
    import torch
    monkeypatch.setattr(torch,"float16",torch.float32)
    monkeypatch.setattr(torch,"bfloat16",torch.float32)
    calls=[]
    def prepare(q,k,v,**kw):
        return dict(q=q,qs=torch.ones(2,2,4,32),ks=torch.ones(2,2,(k.shape[2]+127)//128,4))
    def output(state):
        qs=state["qs"].view(2,2,4,4,8)
        assert torch.all(qs[:,:,:,1::2,:]==128) and torch.all(qs[:,:,:,::2,:]==1)
        assert torch.all(state["ks"][:,:,::2,:]==1e-36)
        assert torch.all(state["ks"][:,:,1::2,:]==1)
        return torch.ones_like(state["q"])
    def forward(ext,state,flag):calls.append((state,flag));return output(state)
    monkeypatch.setattr(native,"forward",forward)
    rows=native.validate_pipeline(None,SimpleNamespace(prepare=prepare),
        SimpleNamespace(reference=lambda state,chunk:output(state)),torch.device("cpu"),lambda rows:None)
    assert len(rows)==16 and all(r["status"]=="passed" for r in rows)
    for i in range(0,len(calls),2):
        assert calls[i][0] is calls[i+1][0]
        assert [calls[i][1],calls[i+1][1]]==[False,True]
