"""CPU checks for benchmark payload adaptation, selection, and isolation."""
import builtins
import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def benchmark(monkeypatch):
    """Imports/utility calls must work without GPU, TileLang, or VC native."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    real_import = builtins.__import__

    def isolated_import(name, *args, **kwargs):
        if name == "tilelang" or name.startswith(("tilelang.", "tvm", "sageattention_local", "paper_expcast_native")):
            raise AssertionError(f"CPU benchmark utility unexpectedly loaded {name}")
        return real_import(name, *args, **kwargs)

    def no_cuda(*args, **kwargs):
        raise AssertionError("CPU benchmark utility unexpectedly called CUDA")

    monkeypatch.setattr(builtins, "__import__", isolated_import)
    for name in ("set_device", "get_device_capability", "get_device_name", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, no_cuda)
    return load_file("tilelang_opt_benchmark_cpu", ROOT / "scripts" / "benchmark_tilelang_opt.py")


@pytest.fixture
def reference():
    # Load the independent oracle file directly to avoid Sage package imports.
    return load_file("vc_reference_cpu", ROOT / "third_party" / "sageattention_local" /
                     "sageattention_local" / "vc_reference.py")


@pytest.mark.parametrize("tokens", [128, 256, 512])
def test_inverse_v_permutation_preserves_every_fp8_bit_pattern(benchmark, reference, tokens):
    # Byte checks include signed zero, subnormals, and NaN encodings. Numerical
    # equality is insufficient for this layout-only adaptation.
    raw = (torch.arange(2 * 3 * 7 * tokens) % 256).to(torch.uint8).reshape(2, 3, 7, tokens)
    natural = {"v": raw.view(torch.float8_e4m3fn), "q": object(), "vs": object(),
               "v_layout": "channel_major_natural", "expcast": True}
    original = raw.clone()
    state = benchmark.oracle_state(natural, reference)
    permutation = reference.value_permutation(tokens, torch.device("cpu"))
    decoded = state["v"].view(torch.uint8).index_select(-1, permutation)
    assert torch.equal(decoded, original)
    assert torch.equal(natural["v"].view(torch.uint8), original)
    assert state["v"].is_contiguous()
    assert state["v_layout"] == "sage_permuted"
    assert state["q"] is natural["q"] and state["vs"] is natural["vs"]
    assert state["expcast"]
    # Guard against replacing argsort with the forward permutation. Sage's
    # permutation is not its own inverse.
    assert not torch.equal(permutation.index_select(0, permutation), torch.arange(tokens))


def test_inverse_v_permutation_has_expected_first_storage_group(benchmark, reference):
    raw = torch.arange(128, dtype=torch.uint8).reshape(1, 1, 1, 128)
    state = benchmark.oracle_state({"v": raw.view(torch.float8_e4m3fn)}, reference)
    assert state["v"].view(torch.uint8).flatten()[:16].tolist() == [
        0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]


def test_cli_defaults_keep_original_controls_and_only_valid_candidates(benchmark):
    args = benchmark.parse_args([])
    assert (args.batch, args.heads, args.queries, args.tokens, args.dim, args.dtype) == (
        1, 28, 37741, 37741, 128, "bfloat16")
    assert args.variants == ["cached_s1", "cached_s2", "mma_qk_s1", "mma_qk_s2"]
    assert args.output == Path("results/h200_tilelang_opt.json")
    configs = benchmark.configurations(args)
    assert list(configs) == ["baseline_s1", "baseline_s2", *args.variants]
    assert configs["baseline_s1"] == (benchmark.baseline_backend, 1, "baseline")
    assert configs["baseline_s2"] == (benchmark.baseline_backend, 2, "baseline")
    assert all(configs[name][0] is benchmark.optimized_backend for name in args.variants)


def test_cli_candidate_subset_is_deduplicated_and_overlap_is_explicit(benchmark):
    args = benchmark.parse_args(["--variants", "mma_qk_s2", "cached_s1", "mma_qk_s2",
                                 "--include-overlap", "--device", "cuda:3",
                                 "--validate-only", "--skip-sage", "--skip-flash4"])
    assert args.variants == ["mma_qk_s2", "cached_s1"]
    assert args.device == "cuda:3"
    assert args.validate_only and args.skip_sage and args.skip_flash4
    configs = benchmark.configurations(args)
    assert list(configs) == ["baseline_s1", "baseline_s2", "mma_qk_s2", "cached_s1", "overlap_s1", "overlap_s2"]
    assert configs["mma_qk_s2"] == (benchmark.optimized_backend, 2, "mma_qk")
    assert configs["cached_s1"] == (benchmark.optimized_backend, 1, "cached")
    assert configs["overlap_s2"] == (benchmark.baseline_backend, 2, "overlap")


@pytest.mark.parametrize("arguments", [
    ["--variants", "reduce8_s1"], ["--queries", "0"], ["--dim", "32"],
    ["--check-queries", "-1"], ["--kernel-seconds", "nan"],
])
def test_cli_rejects_removed_candidate_and_invalid_dimensions(benchmark, arguments):
    with pytest.raises(SystemExit) as error:
        benchmark.parse_args(arguments)
    assert error.value.code == 2


def test_dispatch_preserves_backend_stage_and_same_payload_without_late_binding(benchmark, monkeypatch):
    invoked = []

    def backend(label):
        def call(state, stage, variant):
            invoked.append((label, state, stage, variant))
            return label, stage, variant
        return call

    monkeypatch.setattr(benchmark.baseline_backend, "forward_prepared", backend("original"))
    monkeypatch.setattr(benchmark.optimized_backend, "forward_prepared", backend("candidate"))
    configs = benchmark.configurations(benchmark.parse_args(["--include-overlap"]))
    calls = benchmark.implementations(configs)
    payload = object()
    for name, call in calls.items():
        module, stage, variant = configs[name]
        expected = "original" if module is benchmark.baseline_backend else "candidate"
        assert call(payload) == (expected, stage, variant)
    assert all(item[1] is payload for item in invoked)
    assert len(invoked) == len(configs)
