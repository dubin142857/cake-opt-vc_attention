"""Experimental routing and exact source-boundary guards, no GPU required."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import benchmark_cuda_expcast as bench
import balanced_max_native as native


def test_default_experiment_and_history_have_distinct_output_paths():
    current=bench.parse_args([])
    old=bench.parse_args(["--experiment","split_mass"])
    assert current.experiment=="balanced_max"
    assert current.output.name=="h200_cuda_expcast_balanced_max.json"
    assert old.output.name=="h200_cuda_expcast_split_mass.json"
    assert current.trace_dir!=old.trace_dir
    explicit=bench.parse_args(["--output","custom.json","--trace-dir","custom"])
    assert explicit.output==Path("custom.json") and explicit.trace_dir==Path("custom")


@pytest.mark.parametrize("experiment",["split_mass","balanced_max"])
def test_candidate_calls_match_selected_backend_and_never_reuse_full_preprocessing(experiment):
    _,name,_=bench.EXPERIMENTS[experiment]
    prepared=[]
    def prepare(*qkv,**kwargs):
        assert kwargs==dict(expcast=True,v_smooth=False)
        state=object();prepared.append(state);return state
    backend=SimpleNamespace(forward=lambda ext,state,flag:(state,flag))
    state=object();core,full=bench.make_calls(None,SimpleNamespace(prepare=prepare),(1,2,3),state,backend,name)
    assert core["vc28_jit"]()==(state,False)
    assert core[name]()==(state,True)
    assert full[name]()[1] is True and full["vc28_jit"]()[1] is False
    assert len(prepared)==2 and prepared[0] is not prepared[1]


def test_helper_preserves_encoding_mass_guard_and_fallback_verbatim():
    original=(native.VC/"work_reduction.cuh").read_text(encoding="utf-8")
    start=original.index("template<bool BitEncode=false")
    end=original.index("template<bool SkipIdentity",start)
    expected=original[start:end].replace("vc_probability_prepare_integer(","vc_probability_prepare_balanced_max(")
    expected=expected.replace("vc_integer_row_max(scores[q],r)","vc_integer_row_max_balanced(scores[q],r)")
    assert native.helper_source().endswith(expected)


def test_kernel_tail_unchanged_and_candidate_is_isolated_from_split_mass():
    source=native.generate_source()
    assert "BalancedMax=false" in source
    assert "if constexpr (BalancedMax)" in source
    assert source.count("vc_probability_prepare_balanced_max<false,false,true,false>(RS,")==1
    assert "vc_probability_split_mass" not in source
    assert "clock64" not in source and "torch::zeros" not in source
    original=(native.VC/"sage_layout.cuh").read_text(encoding="utf-8")
    start=original.index("(num_iterations - 1) * k_scale_advance_offset")
    end=original.index("\n}",start)+2
    assert original[start:end] in source
