"""Benchmark routing and generation guards for the CUDA-only experiment."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import benchmark_cuda_expcast as bench
import split_mass_native as native


def test_defaults_keep_h200_main_workload_and_external_baselines():
    a=bench.parse_args([])
    assert (a.batch,a.queries,a.tokens,a.heads,a.dim)==(1,37741,37741,28,128)
    assert not a.skip_flash4 and not a.skip_cupti
    assert not hasattr(a,"v_smooth")


@pytest.mark.parametrize("args",[["--dim","96"],["--queries","0"],["--check-queries","-1"],
                                 ["--kernel-seconds","nan"],["--repeats","0"]])
def test_invalid_benchmark_inputs(args):
    with pytest.raises(SystemExit):bench.parse_args(args)


def test_core_reuses_payload_full_prepares_fresh_and_modes_are_distinct(monkeypatch):
    calls=[]
    def prepare(*qkv,**options):
        calls.append(options);return object()
    vc=SimpleNamespace(prepare=prepare)
    monkeypatch.setattr(native,"forward",lambda ext,state,split:(state,split))
    state=object();core,full=bench.make_calls(None,vc,("q","k","v"),state)
    for name,split in bench.VARIANTS.items():
        assert core[name]()==(state,split)
        first=full[name]();second=full[name]()
        assert first[0] is not second[0] and first[1] is split and second[1] is split
    assert calls==[dict(expcast=True,v_smooth=False)]*4


def test_generated_candidate_changes_only_main_integer_probability_dispatch():
    source=native.generate_source()
    assert "bool SplitMass=false" in source and "if constexpr (SplitMass)" in source
    assert source.count("vc_probability_prepare_split_mass<false,false,true,false>(RS,")==1
    assert "clock64" not in source and "torch::zeros" not in source
    # Tail probability and history are retained from the original kernel.
    original=(native.VC/"sage_layout.cuh").read_text(encoding="utf-8")
    anchor="(num_iterations - 1) * k_scale_advance_offset"
    end=original.index("\n}",original.index(anchor))+2
    tail=original[original.index(anchor):end]
    assert tail in source


@pytest.mark.parametrize("option,value",[("expcast",False),("block_values",True),
                                        ("permutation",object()),("v_layout","channel_major_natural")])
def test_prepared_guard_rejects_incompatible_experiments(option,value):
    s=dict(q=torch.zeros(1,1,1,64,dtype=torch.int8),k=torch.zeros(1,1,1,64,dtype=torch.int8),
           v=torch.zeros(1,1,64,128).to(torch.float8_e4m3fn),qs=torch.ones(1),ks=torch.ones(1),
           vs=torch.ones(1),means=torch.empty(0),block_values=False,permutation=None,expcast=True)
    s[option]=value
    with pytest.raises(ValueError,match="Requires ExpCast"):native.forward(SimpleNamespace(),s)
