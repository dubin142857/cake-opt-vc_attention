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
