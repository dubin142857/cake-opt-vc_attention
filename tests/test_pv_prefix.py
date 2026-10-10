"""Pipeline isolation, serial/overlap routing and source-level lifetime guards."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import pv_prefix_native as native
import benchmark_cuda_expcast as bench


def test_baseline_is_measured_scale_prefetch_and_all_modes_are_distinct():
    assert native.BASELINE=="vc28_scale_prefetch"
    a=bench.parse_args(["--experiment","pv_prefix"])
    assert a.output.name=="h200_cuda_expcast_pv_prefix.json"
    assert a.trace_dir.name=="cuda_expcast_pv_prefix_traces"
    calls=[]
    def prepare(*args,**kw):
        assert kw==dict(expcast=True,v_smooth=False)
        obj=object();calls.append(obj);return obj
    backend=SimpleNamespace(EXTRA_VARIANTS=native.EXTRA_VARIANTS,
        forward=lambda ext,s,flag:(s,2 if flag else 0),forward_mode=lambda ext,s,mode:(s,mode))
    state=object()
    core,full=bench.make_calls(None,SimpleNamespace(prepare=prepare),(),state,backend,
                              "vc28_prefix_overlap",native.BASELINE)
    for name,mode in ((native.BASELINE,0),("vc28_prefix_overlap",2),("vc28_prefix_serial",1)):
        assert core[name]()==(state,mode)
        assert full[name]()==(calls[-1],mode)
    assert len(set(calls))==3


def test_flag_dispatch_does_not_treat_integer_serial_as_boolean_overlap(monkeypatch):
    monkeypatch.setattr(native.base,"forward",lambda ext,state,mode:mode)
    assert native.forward(None,None,False)==0
    assert native.forward(None,None,True)==2
    assert native.forward_mode(None,None,1)==1


def test_baseline_tail_and_scale_prefetch_are_preserved():
    source=native.generate_source()
    original=native.base.candidate_kernel()
    assert original in source
    marker="  {\n    p ^= 1;"
    assert native.candidate_kernel().rsplit(marker,1)[1]==original.rsplit(marker,1)[1]
    assert source.count("current_k_scale = K_scale[k_scale_idx + iter * k_scale_advance_offset]")==2
    assert "clock64" not in source and "torch::zeros" not in source
    assert 'm.def("check_probability",&check_probability)' in source


def test_only_unissued_operand_is_read_and_fourth_mma_is_fenced():
    s=native.candidate_kernel()
    start=s.index("    if constexpr (PrefixOverlap) {")
    end=s.index("    // Read the NEXT tile",start)
    overlap=s[start:end]
    assert "vc_probability_finish_prefix(late,RS_f8[0][3],tile_mass[0]);" in overlap
    assert overlap.index("finish_prefix")<overlap.index("warpgroup_arrive()")<overlap.index("wgmma_f8f8f32")
    assert "warpgroup_wait" not in overlap
    assert "&sV[96]" in overlap
    assert "v_it < (PrefixOverlap ? 3 : num_tiles_pv_inner)" in s
    helper=native.helper_source().split("void vc_probability_finish_prefix(",1)[1]
    assert "const uint32_t (&last_packed)[4]" in helper
    assert "RO_temp" not in helper and "wgmma" not in helper


def test_helper_oracle_reuses_vc28_not_balanced_or_fused_quad():
    s=native.check_source()
    assert "vc_probability_prepare_integer<false,false,true,false>" in s
    assert "vc_probability_prepare_prefix" in s and "vc_probability_finish_prefix" in s
    assert "prepare_balanced" not in s and "prepare_fused" not in s


def test_source_contains_serial_and_overlap_specializations():
    s=native.generate_source()
    assert "int Mode>" in s
    assert "true,0>(q,k,v" in s and "true,1>(q,k,v" in s and "true,2>(q,k,v" in s
    assert '"prefix_serial",resources_for<D,Out,true,1>()' in s
    assert "mode>=0 && mode<=2" in s


def test_mixed_warp_validation_uses_all_three_modes_and_modified_state(monkeypatch):
    import torch
    monkeypatch.setattr(torch,"float16",torch.float32)
    monkeypatch.setattr(torch,"bfloat16",torch.float32)
    calls=[]
    def prepare(q,k,v,**kw):
        return dict(q=q,k=k,v=v,qs=torch.ones(2,2,4,32))
    def output(state):
        qs=state["qs"].view(2,2,4,4,8)
        assert torch.all(qs[:,:,:,1::2,:]==128) and torch.all(qs[:,:,:,::2,:]==1)
        return torch.ones_like(state["q"])
    def forward(ext,state,mode):
        calls.append((state,mode));return output(state)
    monkeypatch.setattr(native,"forward_mode",forward)
    rows=native.validate_pipeline(None,SimpleNamespace(prepare=prepare),
        SimpleNamespace(reference=lambda s,chunk:output(s)),torch.device("cpu"),lambda rows:None)
    assert len(rows)==8 and all(row["status"]=="passed" for row in rows)
    for i in range(0,len(calls),3):
        group=calls[i:i+3]
        assert [x[1] for x in group]==[0,1,2]
        assert group[0][0] is group[1][0] is group[2][0]
