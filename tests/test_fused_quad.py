"""The candidate must compare against balanced max and preserve unrelated paths."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_cuda_expcast as bench
import fused_quad_native as native
from validate_fused_quad_math import inputs


def test_benchmark_names_balanced_baseline_and_reprepares_full_inputs():
    calls = []
    def prepare(*args, **kw):
        assert kw == dict(expcast=True, v_smooth=False)
        s = object(); calls.append(s); return s
    backend = SimpleNamespace(forward=lambda ext, s, flag: (s, flag))
    state = object()
    core, full = bench.make_calls(None, SimpleNamespace(prepare=prepare), (), state, backend,
                                  "vc28_fused_quad", "vc28_balanced_max")
    assert "vc28_jit" not in core
    assert core["vc28_balanced_max"]() == (state, False)
    assert core["vc28_fused_quad"]() == (state, True)
    assert full["vc28_balanced_max"]() == (calls[0], False)
    assert full["vc28_fused_quad"]() == (calls[1], True)
    assert calls[0] is not calls[1]


def test_new_experiment_is_explicit_and_has_separate_results():
    a = bench.parse_args(["--experiment", "fused_quad"])
    assert a.output.name == "h200_cuda_expcast_fused_quad.json"
    assert a.trace_dir.name == "cuda_expcast_fused_quad_traces"
    assert bench.parse_args([]).experiment == "balanced_max"
    assert bench.EXPERIMENTS["fused_quad"][0] is native


def test_only_guarded_probability_helper_changes():
    baseline = native.balanced.helper_source()
    start = baseline.index("template<bool BitEncode=false")
    expected = baseline[start:].replace("vc_probability_prepare_balanced_max(",
                                        "vc_probability_prepare_fused_quad(")
    expected = expected.replace("vc_probability_int_affine_tile<InterleaveRows>(",
                                "vc_probability_fused_quad_tile(")
    assert native.helper_source().startswith(baseline)
    assert native.helper_source().endswith(expected)
    source = native.generate_source()
    assert "if constexpr (FusedQuad) vc_probability_prepare_fused_quad" in source
    assert "else vc_probability_prepare_balanced_max<(WorkReduction==6)" in source
    assert "else vc_probability_prepare_integer<" not in source
    original = (native.VC / "sage_layout.cuh").read_text(encoding="utf-8")
    start = original.index("(num_iterations - 1) * k_scale_advance_offset")
    end = original.index("\n}", start)+2
    assert original[start:end] in source
    assert "torch::zeros" not in source and "clock64" not in source
    assert 'm.def("check_probability",&check_probability)' in source


@pytest.mark.parametrize("pattern", ["random", "stable", "fallback", "tiny", "mixed", "codes", "rounding_ties", "all_zero", "all_max"])
def test_oracle_patterns_are_repeatable_without_advancing_global_rng(pattern):
    import torch
    state = torch.random.get_rng_state().clone()
    a, b = inputs(pattern, 7, "cpu"), inputs(pattern, 7, "cpu")
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert torch.equal(torch.random.get_rng_state(), state)
