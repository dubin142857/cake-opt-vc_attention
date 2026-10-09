"""Numerical contract and benchmark safeguards, no TileLang import required."""
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import benchmark_paper_expcast as bench
import paper_expcast_native as native
import tilelang_expcast as tl


def test_default_route_and_shape():
    a=bench.parse_args([])
    assert (a.batch,a.queries,a.tokens,a.heads,a.dim)==(1,37741,37741,28,128)
    assert not a.skip_flash4 and not a.skip_cupti
    assert not hasattr(a,"v_smooth")


@pytest.mark.parametrize("args",[["--dim","96"],["--queries","0"],["--check-queries","-1"],
                                 ["--kernel-seconds","nan"],["--kernel-trials","0"]])
def test_reject_invalid_benchmark(args):
    with pytest.raises(SystemExit):bench.parse_args(args)


def test_generated_native_has_both_direct_formula_paths_no_clocks():
    s=native.generate_source()
    assert s.count("vc_paper_prepare<PaperInterleaved>(RS,m,")==1
    assert s.count("vc_paper_prepare<PaperInterleaved>(RS_f32,m,")==1
    assert "clock64" not in s and "torch::zeros" not in s
    assert "q.scalar_type()==torch::kInt8" in s
    assert "PaperInterleaved=false" in s


def test_encoding_domain_and_known_paper_example():
    x=bench.encoding_inputs("cpu")
    y=(x.double()*11.541560173034668+119.6500015258789).float().round().clamp(0,120).byte()
    assert x.numel()==65536+360+3 and y.min()==0 and y.max()==120
    shifted=torch.tensor(-1.60/1.4426950408889634)
    code=(shifted.double()*11.541560173034668+119.6500015258789).float().round().byte()
    assert code.item()==107
    assert code.view(torch.float8_e4m3fn).float().item()==88


def test_state_guard_rejects_v_smoothing_before_dispatch():
    from types import SimpleNamespace
    s=dict(q=torch.zeros(1,1,1,64,dtype=torch.int8),k=torch.zeros(1,1,1,64,dtype=torch.int8),
           v=torch.zeros(1,1,64,128).to(torch.float8_e4m3fn),qs=torch.ones(1),ks=torch.ones(1),
           vs=torch.ones(1),means=torch.ones(1),block_values=True,permutation=None)
    with pytest.raises(ValueError,match="smoothing OFF"):native.forward(SimpleNamespace(),s)


def test_no_tilelang_import_for_cli_help():
    # This module's optional dependency is imported only when compiling/running.
    assert "tilelang" not in tl.__dict__


def test_installer_pins_torch_and_uses_shared_environment():
    script=(native.ROOT/"scripts/run_expcast_tilelang_h200.sh").read_text()
    assert "--system-site-packages" in script and "torch==" in script
    assert "tilelang==0.1.15" in script and "--constraint" in script


@pytest.mark.parametrize("keys",[128,129,257])
def test_uniform_tile_regression_has_distinct_cross_boundary_values(keys):
    q,k,v=bench.uniform_tile_inputs(torch.float32,64,keys,"cpu")
    assert not q.any() and not k.any() and v.shape==(1,2,keys,64)
    tokens=torch.arange(keys)
    expected=(1+2*(tokens//128)+(tokens%16)/16).mean()
    torch.testing.assert_close(v[0,0,:,0].mean(),expected)
    if keys>128:
        assert v[0,0,128,0]!=v[0,0,0,0]
        assert abs(v[0,0,128:,0].mean()-expected)>.5


def test_prepared_layouts_cannot_be_silently_interchanged():
    from types import SimpleNamespace
    s=dict(q=torch.zeros(1,1,1,64,dtype=torch.int8),k=torch.zeros(1,1,1,64,dtype=torch.int8),
           v=torch.zeros(1,1,64,128).to(torch.float8_e4m3fn),qs=torch.ones(1),ks=torch.ones(1),
           vs=torch.ones(1),means=torch.empty(0),block_values=False,permutation=None)
    with pytest.raises(ValueError,match="natural V"):tl.forward_prepared(s)
    with pytest.raises(ValueError,match="Sage-permuted V"):
        native.forward(SimpleNamespace(),{**s,"v_layout":tl.V_LAYOUT})


def test_full_recomputes_backend_preprocessing_core_reuses_payload(monkeypatch):
    from types import SimpleNamespace
    calls=[]
    sage,natural=object(),object()
    def prep(label,result):
        def inner(*args,**kwargs):calls.append(label);return result
        return inner
    monkeypatch.setattr(bench,"implementations",lambda ext:{"cuda_paper":lambda s:s,"tilelang_paper_s1":lambda s:s})
    monkeypatch.setattr(tl,"prepare",prep("natural",natural))
    core,full=bench.make_paper_calls(None,SimpleNamespace(prepare=prep("sage",sage)),(1,2,3),sage,natural)
    assert core["cuda_paper"]() is sage and core["tilelang_paper_s1"]() is natural and calls==[]
    for _ in range(2):
        assert full["cuda_paper"]() is sage
        assert full["tilelang_paper_s1"]() is natural
    assert calls==["sage","natural","sage","natural"]


def test_logical_payload_check_rejects_changed_value_byte():
    from types import SimpleNamespace
    order=torch.tensor([0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15])
    ref=SimpleNamespace(value_permutation=lambda n,device:order.to(device))
    natural_bytes=torch.arange(64,80,dtype=torch.uint8).reshape(1,1,1,16)
    packed=torch.empty_like(natural_bytes);packed[...,order]=natural_bytes
    base={k:torch.ones(1) for k in ("q","k","qs","ks","vs")}
    sage={**base,"v":packed.view(torch.float8_e4m3fn)}
    natural={**base,"v":natural_bytes.clone().view(torch.float8_e4m3fn)}
    assert bench.check_logical_payloads(sage,natural,ref)["logical_v_bytes_exact"]
    natural["v"].view(torch.uint8)[...,3]+=1
    with pytest.raises(AssertionError,match="Natural V bytes"):
        bench.check_logical_payloads(sage,natural,ref)
