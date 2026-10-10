"""CPU checks for benchmark payload adaptation, selection, and isolation."""
import builtins
import importlib.util
import json
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
    assert args.variants == ["cached_s1", "cached_s2", "cached_no_ws_s1"]
    assert args.validation_case == "all"
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
    ["--variants", "reduce8_s1"], ["--variants", "cached_no_ws_s2"],
    ["--variants", "cached_no_ws_overlap_s2"], ["--variants", "cached_no_ws_int_mass_s2"],
    ["--queries", "0"], ["--dim", "32"],
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
    configs = benchmark.configurations(benchmark.parse_args([
        "--include-overlap", "--variants", "cached_s1", "cached_s2", "mma_qk_s1", "mma_qk_s2", "cached_no_ws_s1",
        "cached_no_ws_overlap_s1", "cached_no_ws_int_mass_s1"]))
    calls = benchmark.implementations(configs)
    payload = object()
    for name, call in calls.items():
        module, stage, variant = configs[name]
        expected = "original" if module is benchmark.baseline_backend else "candidate"
        assert call(payload) == (expected, stage, variant)
    assert all(item[1] is payload for item in invoked)
    assert len(invoked) == len(configs)


def test_no_warp_specialization_maps_to_stage_one_and_can_be_selected_alone(benchmark):
    assert "cached_no_ws_s1" in benchmark.parse_args([]).variants
    args = benchmark.parse_args(["--variants", "cached_no_ws_s1"])
    configs = benchmark.configurations(args)
    assert list(configs) == ["baseline_s1", "baseline_s2", "cached_no_ws_s1"]
    assert configs["cached_no_ws_s1"] == (benchmark.optimized_backend, 1, "cached_no_ws")


@pytest.mark.parametrize("name, variant", [
    ("cached_no_ws_overlap_s1", "cached_no_ws_overlap"),
    ("cached_no_ws_int_mass_s1", "cached_no_ws_int_mass"),
])
def test_unmeasured_no_warp_specialization_candidates_are_explicit_and_stage_one(benchmark, name, variant):
    assert name not in benchmark.parse_args([]).variants
    configs = benchmark.configurations(benchmark.parse_args(["--variants", name]))
    assert list(configs) == ["baseline_s1", "baseline_s2", name]
    assert configs[name] == (benchmark.optimized_backend, 1, variant)


def test_targeted_validation_selects_both_tail_patterns(benchmark):
    args = benchmark.parse_args(["--validate-only", "--validation-case", "tail65_129"])
    assert args.validate_only
    assert benchmark.validation_cases(args.validation_case) == [
        ("random", 65, 129), ("uniform_tiles", 65, 129)]
    assert len(benchmark.validation_cases("all")) == 6


def test_targeted_tail_inputs_match_full_suite_rng_for_every_dtype_and_dim(benchmark):
    full_generator = torch.Generator(device="cpu").manual_seed(3333)
    tail_generator = torch.Generator(device="cpu").manual_seed(3333)
    checked = []
    # Use the actual full-loop order and retain generator state across all four
    # combinations; D128 must include draws consumed by the earlier D64 cases.
    for dtype in (torch.float16, torch.bfloat16):
        for dim in (64, 128):
            full = dict(benchmark.yield_selected_inputs("all", dtype, dim, "cpu", full_generator))
            tail = dict(benchmark.yield_selected_inputs("tail65_129", dtype, dim, "cpu", tail_generator))
            assert set(tail) == {("random", 65, 129), ("uniform_tiles", 65, 129)}
            for case, qkv in tail.items():
                for actual, expected in zip(qkv, full[case]):
                    assert actual.dtype == expected.dtype == dtype
                    assert torch.equal(actual, expected), (dtype, dim, case)
                checked.append((dtype, dim, case))
            assert torch.equal(tail_generator.get_state(), full_generator.get_state())
    assert len(checked) == 8


def test_stability_synchronizes_before_reading_and_records_success(benchmark):
    expected = torch.ones(1, 2, 65, 128)
    actual = torch.zeros_like(expected)
    events, records = [], []

    def call():
        events.append("launch")
        return actual

    def synchronize():
        events.append("sync")
        actual.copy_(expected)

    def publish(row):
        events.append("publish")
        records.append(row)

    benchmark.check_exact_run(call, synchronize, expected, stream="default", run_index=2, on_result=publish)
    assert events == ["launch", "sync", "publish"]
    assert records[0] == {"run_index": 2, "stream": "default", "status": "passed",
                          "all_row_metrics": {"relative_rmse": 0., "max_abs": 0.,
                                              "checked_queries": 65, "exact_match": True}}


def test_stability_locates_tail_head_and_publishes_failure_before_raise(benchmark):
    expected = torch.ones(1, 2, 65, 128)
    actual = expected.clone()
    actual[0, 1, 64, 84] += .000549
    records = []
    with pytest.raises(AssertionError, match="Exact stability failed on non_default run 1"):
        benchmark.check_exact_run(lambda: actual, lambda: None, expected,
                                  stream="non_default", run_index=1, on_result=records.append)
    assert len(records) == 1
    row = records[0]
    assert row["status"] == "failed" and row["mismatched_elements"] == 1
    assert row["affected_rows"] == {"count": 1, "first32": [[0, 1, 64]],
                                    "by_head": [{"batch": 0, "head": 1, "count": 1,
                                                 "first32_queries": [64]}]}
    assert row["first_bad"]["coordinate"] == [0, 1, 64, 84]
    assert row["first_bad"]["actual"] == actual[0, 1, 64, 84].item()
    assert row["first_bad"]["expected"] == 1.
    assert row["actual_all_finite"] and row["expected_all_finite"]
    assert row["first_bad"]["actual_finite"] and row["first_bad"]["expected_finite"]
    assert row["all_row_metrics"]["relative_rmse"] < .001
    assert not row["all_row_metrics"]["exact_match"]


def test_nonfinite_stability_failure_remains_valid_json(benchmark):
    expected = torch.ones(1, 2, 65, 128)
    actual = expected.clone()
    actual[0, 1, 64, 84] = float("nan")
    actual[0, 1, 64, 85] = float("inf")
    records = []
    with pytest.raises(AssertionError, match="Exact stability failed"):
        benchmark.check_exact_run(lambda: actual, lambda: None, expected,
                                  stream="default", run_index=1, on_result=records.append)
    row = records[0]
    assert row["mismatched_elements"] == 2
    assert not row["actual_all_finite"] and row["expected_all_finite"]
    assert row["first_bad"]["actual"] == "nan"
    assert not row["first_bad"]["actual_finite"]
    assert row["all_row_metrics"]["relative_rmse"] is None
    json.dumps(row, allow_nan=False)
