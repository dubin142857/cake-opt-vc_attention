"""Diagnosis must reject missing GPU activity and must keep V grouping disabled."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import attention_phase_probe as phase
import profile_expcast_h200 as focus


def test_route_uses_same_tensors_and_no_v_smoothing():
    base = dict(q=torch.zeros(1,1,1,64,dtype=torch.int8), k=torch.zeros(1,1,1,64,dtype=torch.int8),
                v=torch.zeros(1,1,64,128).to(torch.float8_e4m3fn), qs=torch.ones(1), ks=torch.ones(1),
                vs=torch.ones(1), means=torch.empty(0), permutation=None, block_values=False,
                output_type=1, scale=.125, expcast=True)
    calls = []
    vc = SimpleNamespace(forward_prepared=lambda s, **kw: calls.append((s,kw)),
                         sageattn_vc=lambda *args, **kw: calls.append(kw))
    # make_calls builds Sage closures but does not invoke them here.
    states, core, full = focus.make_calls(vc, None, None, (), base)
    assert set(states)=={"control", "expcast"}
    for key in ("q", "k", "v", "qs", "ks", "vs"):
        assert states["control"][key] is states["expcast"][key]
    for name, cast in (("control",False),("expcast",True)):
        full[name]()
        assert calls[-1]["v_smooth"] is False and calls[-1]["expcast"] is cast
        core[name]()
        assert calls[-1][1]["kernel_mode"]==focus.MODE
    with pytest.raises(ValueError,match="forbids"):
        focus.make_calls(vc,None,None,(),{**base,"block_values":True})


@pytest.mark.parametrize("args", [["--phase-tile-stride","0"], ["--phase-cta-stride","-1"],
                                  ["--kernel-seconds","nan"], ["--check-queries","-1"],
                                  ["--v-smooth"], ["--dtype","float32"]])
def test_bad_args_fail_before_loading_gpu(args):
    with pytest.raises(SystemExit):
        focus.parse_args(args)


def test_default_shape_and_no_smooth_option():
    a = focus.parse_args([])
    assert (a.batch,a.queries,a.tokens,a.heads,a.dim)==(1,37741,37741,28,128)
    assert not hasattr(a,"v_smooth")
    assert focus.parse_args(["--probability-detail"]).kernel_phases


def test_net_savings_preserves_trial_pairing_and_negative_results():
    r=focus.expcast_savings({"control":[20.,30.,40.],"expcast":[19.,33.,38.]})
    assert r["paired_trial_saved_ms"]==[1.,-3.,2.]
    assert r["median_saved_ms"]==1.
    assert r["min_saved_ms"]==-3.
    with pytest.raises(ValueError):focus.expcast_savings({"control":[1.],"expcast":[]})


def span(name,start,end,category="user_annotation"):
    return {"ph":"X","cat":category,"name":name,"ts":start,"dur":end-start}


def test_cupti_sums_gpu_work_without_host_or_idle_time():
    events = [span("expcast_stage/qk_int8",0,100), span("Q_quant",10,20,"kernel"),
              span("K_quant",30,50,"kernel"), span("launch",2,7,"cuda_runtime"),
              span("expcast_stage/attention",100,300),span("core",120,270,"kernel")]
    r=focus.summarize_stage_trace(events,["qk_int8","attention"])
    assert r["qk_int8"]["kernel_ms"]==.030
    assert r["qk_int8"]["launches"]==2
    assert r["attention"]["kernel_ms"]==.150


@pytest.mark.parametrize("kind", ["missing","outside","overlap","nonfinite","no_gpu"])
def test_cupti_refuses_partial_or_ambiguous_stage_attribution(kind):
    events=[span("expcast_stage/attention",0,100),span("core",10,80,"kernel")]
    expected=["attention"]
    if kind=="missing": expected.append("qk_int8")
    if kind=="outside": events.append(span("other",150,180,"kernel"))
    if kind=="overlap":
        events.append(span("expcast_stage/qk_int8",0,100));expected.append("qk_int8")
    if kind=="nonfinite": events[1]["dur"]=float("nan")
    if kind=="no_gpu": events.pop()
    with pytest.raises(RuntimeError):
        focus.summarize_stage_trace(events,expected)


@pytest.mark.parametrize("keys,stride", [(1,8),(128,8),(129,8),(1024,8),(1025,8),(37741,8),(37741,1)])
def test_clock_samples_cover_final_tile_without_double_count(keys,stride):
    tiles=sorted(set(range(0,(keys+127)//128,stride)) | {((keys+127)//128)-1})
    clocks=(torch.arange(10,dtype=torch.int64)*10+100).repeat(3,len(tiles),1)
    out=phase.summarize_clocks(clocks,keys,stride)
    assert out["sampled_tile_indices"]==tiles
    assert out["final_masked_tile"]["samples"]==3
    if len(tiles)>1:
        assert out["main_tiles"]["samples"]==3*(len(tiles)-1)
        rows=out["main_tiles"]["phases"]
        assert sum(r["sampled_elapsed_share_percent"] for r in rows.values())==pytest.approx(100)
        assert all(r["median_cycles"]==10 for r in rows.values())
    else:
        assert out["main_tiles"] is None


@pytest.mark.parametrize("kind", ["missing","backward","slots"])
def test_clock_parser_refuses_missing_and_invalid_samples(kind):
    clocks=(torch.arange(10,dtype=torch.int64)+100).repeat(2,2,1)
    if kind=="missing":clocks[1,1,8]=0
    if kind=="backward":clocks[0,0,5]=101
    if kind=="slots":clocks=clocks[:,:1]
    with pytest.raises(RuntimeError):phase.summarize_clocks(clocks,129,8)


def test_generation_does_not_modify_production_and_has_exact_boundary_order():
    path=phase.VC/"sage_layout.cuh"
    original=path.read_bytes()
    text=phase.instrument_loop(original.decode())
    # Both normal and masked blocks must have one ordered copy of each marker.
    body=text.split("probe.start(")[1:]
    assert len(body)==2
    for b in body:
        found=[int(v.split(")")[0]) for v in b.split("probe.mark(")[1:]]
        assert found==list(range(1,10))
    assert "probe.start(num_iterations - 1)" in text
    assert text.count("warpgroup_wait<0>()")==original.decode().count("warpgroup_wait<0>()")
    assert text.count("__syncthreads()")==original.decode().count("__syncthreads()")
    assert path.read_bytes()==original
    generated=phase.generate_source()
    assert "#define PHASE_CTA_STRIDE 64" in generated
    assert "Copyright (c) 2024 by SageAttention team" in generated
    with pytest.raises(RuntimeError,match="source changed"):
        phase.instrument_loop(original.decode().replace("    // wait for V", "    // moved wait"))


def test_sampling_stride_is_not_shell_or_code_text():
    with pytest.raises(ValueError):phase.generate_source(0,8)


def test_sm86_fails_explicitly_and_preserves_diagnostic(tmp_path,monkeypatch):
    a=focus.parse_args(["--output",str(tmp_path/"report.json"),"--trace-dir",str(tmp_path/"traces")])
    monkeypatch.setattr(focus,"parse_args",lambda:a)
    monkeypatch.setattr(torch.cuda,"set_device",lambda _:None)
    monkeypatch.setattr(torch.cuda,"get_device_capability",lambda _:(8,6))
    with pytest.raises(RuntimeError,match="Hopper required"):focus.main()
    assert '"status": "failed"' in a.output.read_text()


def detail_records():
    clocks=torch.zeros(2,2,phase.DETAIL_WIDTH,dtype=torch.int64)
    clocks[...,:10]=torch.tensor([100,200,300,1000,3000,4000,5000,6000,7000,8000])
    clocks[:,0,10:27]=torch.arange(1100,2800,100)
    clocks[:,0,27]=1
    clocks[:,1,27]=4
    return clocks


def test_detail_groups_are_summed_before_statistics():
    clocks=detail_records()
    out=phase.summarize_probability_clocks(clocks,129,8,True)["probability_detail"]
    assert out["regular_samples"]==2 and out["masked_tail_samples"]==2
    assert out["helper_median_cycles"]==1600
    rows=out["phases"]
    assert {k:r["median_cycles"] for k,r in rows.items()}==dict(zip(phase.DETAIL_PHASES,[100,100,100,400,400,500]))
    assert sum(r["sampled_helper_share_percent"] for r in rows.values())==pytest.approx(100)


def test_fallback_is_counted_but_not_misreported_as_regular_work():
    clocks=detail_records()
    clocks[1,0,13:26]=0
    clocks[1,0,27]=2
    out=phase.summarize_probability_clocks(clocks,129,8,True)["probability_detail"]
    assert out["regular_samples"]==1 and out["fallback_samples"]==1
    clocks[0,0]=clocks[1,0].clone()
    out=phase.summarize_probability_clocks(clocks,129,8,True)["probability_detail"]
    assert out["regular_samples"]==0 and out["phases"]=={}


@pytest.mark.parametrize("bad",["missing_route","unknown_route","tail_work","missing_clock","wrong_order","outside_parent","fallback_work"])
def test_detail_missing_or_stale_records_are_rejected(bad):
    clocks=detail_records()
    if bad=="missing_route":clocks[0,0,27]=0
    if bad=="unknown_route":clocks[0,0,27]=7
    if bad=="tail_work":clocks[0,1,12]=100
    if bad=="missing_clock":clocks[0,0,15]=0
    if bad=="wrong_order":clocks[0,0,12]=1000
    if bad=="outside_parent":clocks[0,0,26]=3500
    if bad=="fallback_work":clocks[0,0,27]=2
    with pytest.raises(RuntimeError):phase.summarize_probability_clocks(clocks,129,8,True)


def test_control_and_single_masked_tile_have_no_detail_work():
    clocks=detail_records();clocks[...,10:27]=0;clocks[...,27]=3
    assert phase.summarize_probability_clocks(clocks,129,8,False)["probability_detail"]["status"]=="not_applicable_control"
    clocks=detail_records()[:,1:]
    result=phase.summarize_probability_clocks(clocks,1,8,True)["probability_detail"]
    assert result["regular_samples"]==0 and result["masked_tail_samples"]==2


def test_detail_codegen_preserves_four_interleaved_groups_and_tail_path():
    generated=phase.generate_source(probability_detail=True)
    assert "#define PHASE_RECORD_WIDTH 28" in generated
    body=phase.instrument_probability_helpers()
    assert body.index("vc_int_affine_quad(") < body.index("probe.mark(14+")
    assert body.index("probe.mark(14+") < body.index("__nv_cvt_fp8x2_to_halfraw2") < body.index("probe.mark(15+")
    assert body.index("probe.mark(15+") < body.index("float2 values=") < body.index("probe.mark(16+")
    assert "for (int base=0;base<NK;base+=2)" in body
    assert "probe.value(27,2)" in body
    assert "if constexpr (PhaseTrace && WorkReduction==9)" in generated
    assert generated.count("probe.value(27, ExpCast ? 4 : 3)")==1
