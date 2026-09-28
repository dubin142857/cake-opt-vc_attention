"""CPU regression checks for benchmark adapters, reference and measurement math.

These do not replace running the actual third-party kernels on Hopper.
"""
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


spec = importlib.util.spec_from_file_location(
    "vc_benchmark_cuda", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_cuda.py"
)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize("name", ["flash2", "flash3", "flash4", "sage", "sage_local"])
def test_adapter_preserves_values_layout_scale_and_output(monkeypatch, name):
    torch.manual_seed(12)
    qkv = tuple(torch.randn(2, 3, n, 8) for n in (5, 7, 7))
    shd = tuple(x.transpose(1, 2).contiguous() for x in qkv)
    calls = []

    def installed_function(q, k, v, **kwargs):
        calls.append(kwargs)
        assert all(x.is_contiguous() for x in (q, k, v))
        if name in ("sage", "sage_local"):
            assert kwargs == dict(tensor_layout="HND", is_causal=False, sm_scale=8**-0.5, return_lse=False)
        else:
            assert kwargs["causal"] is False
            assert kwargs["softmax_scale"] == 8**-0.5
            if name == "flash2":
                assert kwargs == dict(softmax_scale=8**-0.5, causal=False, dropout_p=0.0)
            else:
                assert set(kwargs) == {"softmax_scale", "causal"}
            assert q.shape == (2, 5, 3, 8)
            q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        for actual, expected in zip((q, k, v), qkv):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        result = F.scaled_dot_product_attention(q, k, v)
        if name not in ("sage", "sage_local"):
            result = result.transpose(1, 2).contiguous()
        return (result, "unused lse") if name in ("flash3", "flash4") else result

    distribution, module_name, function_name, _ = benchmark.EXTERNAL[name]
    module = SimpleNamespace(__file__="installed/interface.py", **{function_name: installed_function})
    module.local_build_info = lambda: {"upstream_commit": "pinned", "extensions": {"qattn_sm90": "local/native.so"}}
    monkeypatch.setattr(benchmark.importlib, "import_module", lambda path: module if path == module_name else None)
    monkeypatch.setattr(benchmark.metadata, "version", lambda package: "test-version" if package == distribution else None)
    backend = benchmark.external_backend(name, qkv, shd)
    actual = benchmark.output_hnd(backend.call(), backend.layout, qkv[0].shape, qkv[0].dtype)
    expected = torch.softmax(qkv[0] @ qkv[1].transpose(-2, -1) / 8**0.5, -1) @ qkv[2]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert len(calls) == 1
    assert backend.info["version"] == "test-version"
    assert backend.info["interface"] == f"{module_name}.{function_name}"
    if name == "sage_local":
        assert backend.info["source_build"]["extensions"]["qattn_sm90"] == "local/native.so"


def test_local_sage_is_optional_and_never_substitutes_installed_sage(monkeypatch):
    assert "sage_local" not in benchmark.parse_args([]).backends
    args = benchmark.parse_args(["--backends", "sage", "sage_local", "--baseline", "sage"])
    assert args.backends == ["sage", "sage_local"]
    seen = []

    def missing(name):
        seen.append(name)
        raise ModuleNotFoundError("not built", name="sageattention_local")

    monkeypatch.setattr(benchmark.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="sage_local: cannot load"):
        benchmark.load_external("sage_local")
    assert seen == ["sageattention_local"]


def test_sage_comparison_uses_installed_output_as_reference_and_correct_speed_ratio():
    samples = {"sage": torch.tensor([2., 2.]), "sage_local": torch.tensor([3., 3.])}
    times = {"sage": {"median_ms": 24.}, "sage_local": {"median_ms": 30.}}
    comparison = benchmark.compare_sage_outputs(samples, times)
    assert comparison["local_speedup_vs_installed"] == pytest.approx(0.8)
    assert comparison["relative_rmse_local_vs_installed"] == pytest.approx(0.5)
    assert comparison["max_abs_local_vs_installed"] == pytest.approx(1.)
    assert not comparison["exact_sample_match"]
    assert comparison["acceptance_threshold"] is None


def test_fa3_legacy_interface_only_when_modern_module_is_absent(monkeypatch):
    seen = []
    legacy = SimpleNamespace(flash_attn_func=lambda: None, __file__="legacy.py")

    def import_module(name):
        seen.append(name)
        if name == "flash_attn_3.flash_attn_interface":
            raise ModuleNotFoundError("No module named flash_attn_3", name="flash_attn_3")
        assert name == "flash_attn_interface"
        return legacy

    monkeypatch.setattr(benchmark.importlib, "import_module", import_module)
    monkeypatch.setattr(benchmark, "distribution_version", lambda _: "source-install")
    function, layout, info = benchmark.load_external("flash3")
    assert function is legacy.flash_attn_func
    assert layout == "BSHD"
    assert info["interface"] == "flash_attn_interface.flash_attn_func"
    assert seen == ["flash_attn_3.flash_attn_interface", "flash_attn_interface"]


@pytest.mark.parametrize("name,missing", [("flash3", "flash_attn_3._C"), ("flash4", "flash_attn.cute")])
def test_broken_installation_is_not_silently_replaced(monkeypatch, name, missing):
    seen = []

    def import_module(module):
        seen.append(module)
        raise ModuleNotFoundError(f"No module named {missing}", name=missing)

    monkeypatch.setattr(benchmark.importlib, "import_module", import_module)
    with pytest.raises(RuntimeError, match=f"{name}: cannot load") as error:
        benchmark.load_external(name)
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    assert len(seen) == 1


@pytest.mark.parametrize("count", [0, 1, 3, 100])
def test_sampled_reference_uses_all_keys_and_handles_chunk_tail(count):
    torch.manual_seed(1)
    q, k, v = (torch.randn(2, 3, n, 8) for n in (7, 11, 11))
    indices = benchmark.query_indices(7, count)
    assert indices == sorted(set(indices))
    assert len(indices) == (7 if count == 0 else min(count, 7))
    assert indices[0] == 0
    if len(indices) > 1:
        assert indices[-1] == 6
    dense = F.scaled_dot_product_attention(q, k, v)
    actual = benchmark.fp32_reference(q, k, v, indices, chunk=2)
    torch.testing.assert_close(actual, dense[:, :, indices, :], atol=1e-6, rtol=1e-5)
    assert actual.device.type == "cpu"


def test_timer_waits_for_gpu_and_converts_to_per_call_milliseconds(monkeypatch):
    events = []
    clock = iter([10.0, 10.06])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda device: events.append(("sync", device)))

    def call():
        events.append("call")
        return "output"

    output, ms = benchmark.measure_trial(call, repeats=3, device="cuda:1")
    assert output == "output"
    assert ms == pytest.approx(20)
    assert events == [("sync", "cuda:1"), "call", "call", "call", ("sync", "cuda:1")]


def test_speedup_direction_and_error_units():
    rows = {"flash3": {"median_ms": 1.0}, "flash4": {"median_ms": 0.5}, "vc": {"median_ms": 2.0}}
    vc_ratios = benchmark.add_speedups(rows, "flash3")
    assert rows["flash4"]["speedup_vs_baseline"] == 2.0
    assert rows["vc"]["speedup_vs_baseline"] == 0.5
    assert vc_ratios == {"flash3": 0.5, "flash4": 0.25}
    metrics = benchmark.error_metrics(torch.tensor([2.0, 2.0]), torch.ones(2))
    assert metrics == {"relative_rmse_vs_fp32": 1.0, "max_abs_error_vs_fp32": 1.0}


def test_vc_ablation_dispatch_freezes_each_config(monkeypatch):
    import vc_attention

    configs = []
    def attention(q, k, v, cfg):
        configs.append(cfg)
        return q, {}

    monkeypatch.setattr(vc_attention, "attention", attention)
    qkv = (torch.zeros(1, 2, 3, 8),) * 3
    backends = benchmark.make_backends(["vc"], qkv, vc_ablation=True)
    for backend in backends:
        backend.call()
    assert [x.name for x in backends] == ["vc", "vc_control", "vc_expcast", "vc_v_smooth"]
    assert [(x.v_smooth, x.expcast) for x in configs] == [(True, True), (False, False), (False, True), (True, False)]
    assert all(x.backend == "cuda_fp8" for x in configs)


def test_nonfinite_output_is_rejected():
    output = torch.full((1, 2, 3, 8), float("nan"))
    with pytest.raises(ValueError, match="NaN/Inf"):
        benchmark.output_hnd(output, "BHND", output.shape, output.dtype)


def test_schedule_and_cache_benchmark_preserves_separate_contracts(monkeypatch):
    import vc_attention
    calls = []

    def attention(q, k, v, cfg, **kwargs):
        calls.append((cfg, kwargs))
        return q, {}

    monkeypatch.setattr(vc_attention, "attention", attention)
    qkv = (torch.zeros(1, 2, 3, 8),) * 3
    rows = benchmark.make_backends(["vc"], qkv, False, vc_schedule_compare=True, vc_reuse_layout=True)
    assert [r.name for r in rows] == ["vc", "vc_serial", "vc_reuse"]
    assert len(calls) == 1 and calls[0][1]["step"] == 0
    cache = calls[0][1]["cache"]
    calls.clear()
    for row in rows:
        row.call()
    assert [cfg.cuda_schedule for cfg, _ in calls] == ["batched", "serial", "batched"]
    assert calls[0][1] == calls[1][1] == {}
    assert calls[2][1]["cache"] is cache and calls[2][1]["step"] == 1
    assert [r.info["layout_cache_reused"] for r in rows] == [False, False, True]


def test_explicit_subset_uses_selected_baseline():
    args = benchmark.parse_args(["--backends", "sage", "vc"])
    assert args.baseline == "sage"
    assert args.backends == ["sage", "vc"]
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--backends", "sage", "--baseline", "flash3"])


@pytest.mark.parametrize("python_schedule,native_version,accepted", [
    (None, None, False), ("batched", None, False), (None, 1, False), ("batched", 1, True),
])
def test_runtime_guard_rejects_old_pip_install_before_large_benchmark(monkeypatch, python_schedule, native_version, accepted):
    import vc_attention
    monkeypatch.setattr(vc_attention, "Config", lambda: SimpleNamespace(cuda_schedule=python_schedule))
    ext = SimpleNamespace(scheduling_version=native_version, numerical_contract_version=2, __file__="test/native.so")
    monkeypatch.setattr(benchmark.importlib, "import_module", lambda _: ext)
    monkeypatch.setattr(benchmark, "distribution_version", lambda *a, **kw: "test")
    if accepted:
        info = benchmark.require_batched_runtime()
        assert info["cuda_schedule"] == "batched" and info["extension_file"] == "test/native.so"
    else:
        with pytest.raises(RuntimeError, match="git pull does not update"):
            benchmark.require_batched_runtime()


def test_runtime_version_uses_loaded_package_not_stale_checkout_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark.metadata, "version", lambda _: "0.1.0")
    metadata_dir = tmp_path / "vc_attention-0.3.0.dist-info"
    metadata_dir.mkdir()
    (metadata_dir / "METADATA").write_text("Metadata-Version: 2.1\nName: vc-attention\nVersion: 0.3.0\n", encoding="utf-8")
    package = tmp_path / "vc_attention" / "__init__.py"
    assert benchmark.distribution_version("vc-attention", package_file=package) == "0.3.0"


def test_fused_benchmark_requires_native_entry_and_dispatches_explicitly(monkeypatch):
    import vc_attention
    ext = SimpleNamespace(scheduling_version=1, numerical_contract_version=2)
    monkeypatch.setattr(benchmark.importlib, "import_module", lambda _: ext)
    with pytest.raises(RuntimeError, match="0.7.0"):
        benchmark.require_batched_runtime("fused")
    ext.fused_api_version = 1
    with pytest.raises(RuntimeError, match="0.7.0"):
        benchmark.require_batched_runtime("fused")
    ext.fused_pipeline_version = 1
    with pytest.raises(RuntimeError, match="0.7.0"):
        benchmark.require_batched_runtime("fused")
    ext.fused_d128_version = 1
    with pytest.raises(RuntimeError, match="0.7.0"):
        benchmark.require_batched_runtime("fused")
    ext.fused_interleaved_version = 1
    assert benchmark.require_batched_runtime("fused")["selected_schedule"] == "fused"
    assert benchmark.parse_args([]).vc_schedule == "fused"
    configs = []
    monkeypatch.setattr(vc_attention, "attention", lambda q, k, v, cfg: configs.append(cfg))
    qkv = (torch.zeros(1, 2, 3, 8),) * 3
    rows = benchmark.make_backends(["vc"], qkv, True, vc_schedule="fused")
    for row in rows:
        row.call()
    assert len(configs) == 4 and all(cfg.cuda_schedule == "fused" for cfg in configs)


def test_fused_ab_comparison_preserves_preprocessing_and_selects_both_kernels(monkeypatch):
    import vc_attention
    configs = []
    monkeypatch.setattr(vc_attention, "attention", lambda q, k, v, cfg: configs.append(cfg))
    qkv = (torch.zeros(1, 2, 3, 8),) * 3
    rows = benchmark.make_backends(["vc"], qkv, False, vc_schedule="fused", vc_fused_compare=True)
    assert [r.name for r in rows] == ["vc", "vc_fused_baseline"]
    for row in rows:
        row.call()
    assert configs[0].cuda_fused_pipeline and not configs[1].cuda_fused_pipeline
    assert replace(configs[1], cuda_fused_pipeline=True) == configs[0]
    assert all(not r.info["layout_cache_reused"] for r in rows)
    assert benchmark.parse_args(["--backends", "vc", "--vc-fused-compare"]).vc_fused_compare
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--backends", "vc", "--vc-schedule", "batched", "--vc-fused-compare"])
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--backends", "sage", "--vc-fused-compare"])


def test_specialization_ab_preserves_preprocessing_and_can_include_both_controls(monkeypatch):
    import vc_attention
    configs = []
    monkeypatch.setattr(vc_attention, "attention", lambda q, k, v, cfg: configs.append(cfg))
    qkv = (torch.zeros(1, 2, 3, 128),) * 3
    rows = benchmark.make_backends(["vc"], qkv, False, vc_schedule="fused",
                                    vc_specialize_compare=True, vc_fused_compare=True)
    assert [r.name for r in rows] == ["vc", "vc_generic", "vc_fused_baseline"]
    assert [r.info["cuda_fused_variant"] for r in rows] == ["interleaved", "pipeline", "baseline"]
    for row in rows:
        row.call()
    assert replace(configs[1], cuda_fused_specialize=True) == configs[0]
    assert replace(configs[2], cuda_fused_pipeline=True) == configs[0]
    assert all(not r.info["layout_cache_reused"] for r in rows)
    assert benchmark.parse_args(["--backends", "vc", "--vc-specialize-compare"]).vc_specialize_compare
    for invalid in (["--dim", "64"], ["--vc-schedule", "batched"], ["--backends", "sage"]):
        with pytest.raises(SystemExit):
            benchmark.parse_args(["--vc-specialize-compare", *invalid])


def test_interleaved_ab_preserves_all_quantization_settings_and_does_not_duplicate_softmax_arms(monkeypatch):
    import vc_attention
    configs = []
    monkeypatch.setattr(vc_attention, "attention", lambda q, k, v, cfg: configs.append(cfg))
    qkv = (torch.zeros(1, 2, 65, 128),) * 3
    rows = benchmark.make_backends(["vc"], qkv, True, vc_schedule="fused", vc_interleave_compare=True)
    assert [r.name for r in rows] == ["vc", "vc_d128", "vc_control", "vc_expcast", "vc_expcast_d128", "vc_v_smooth"]
    for row in rows:
        row.call()
    assert replace(configs[1], cuda_fused_interleave=True) == configs[0]
    assert replace(configs[4], cuda_fused_interleave=True) == configs[3]
    assert [r.info["cuda_fused_variant"] for r in rows] == ["interleaved", "d128", "d128", "interleaved", "d128", "d128"]
    assert all(not r.info["layout_cache_reused"] for r in rows)
    assert benchmark.parse_args(["--backends", "vc", "--vc-interleave-compare"]).vc_interleave_compare
    for invalid in (["--dim", "64"], ["--vc-schedule", "batched"], ["--backends", "sage"]):
        with pytest.raises(SystemExit):
            benchmark.parse_args(["--vc-interleave-compare", *invalid])


@pytest.mark.parametrize("smooth,cast", [(False, False), (False, True), (True, False), (True, True)])
def test_reference_check_preserves_full_q_and_config_and_exposes_native_error(smooth, cast):
    from vc_attention import Config, attention

    generator = torch.Generator().manual_seed(41)
    q, k, v = (torch.randn(1, 2, n, 8, generator=generator) for n in (33, 67, 67))
    # This unsampled row controls the Q scale for the first block. A reference
    # that slices Q before attention will silently use different quantization.
    q[:, :, 1, :] *= 64
    indices = [0, 32]
    cfg = Config(backend="cuda_fp8", q_block=32, kv_block=32, k_quant_block=64,
                 clusters=3, iterations=3, mean_dtype="bfloat16", v_smooth=smooth, expcast=cast)
    reference_cfg = replace(cfg, backend="reference")
    reference, _ = attention(q, k, v, reference_cfg)
    sample = reference[:, :, indices, :].float()
    # Plant an implementation error independently of the quantization error.
    native_sample = sample + 0.125
    golden = benchmark.fp32_reference(q, k, v, indices, chunk=1)
    comparison = benchmark.compare_vc_reference((q, k, v), asdict(cfg), native_sample, golden, indices)
    assert comparison["reference_config"] == asdict(reference_cfg)
    assert comparison["reference_stats"]["backend"] == "reference"
    assert comparison["reference_stats"]["expcast"] is cast
    assert comparison["reference_stats"]["smoothing_active"] is smooth
    assert comparison["native_vs_reference"]["max_abs_error_vs_reference"] == pytest.approx(0.125, abs=1e-6)
    expected = (native_sample.double() - sample.double()).norm() / sample.double().norm()
    assert comparison["native_vs_reference"]["relative_rmse_vs_reference"] == pytest.approx(expected.item())
    expected_reference_error = (sample.double() - golden.double()).norm() / golden.double().norm()
    assert comparison["reference_vs_fp32"]["relative_rmse_vs_fp32"] == pytest.approx(expected_reference_error.item())
    # Same-config agreement is possible even with a nonzero FP32 accuracy error.
    identical = benchmark.compare_vc_reference((q, k, v), asdict(cfg), sample, golden, indices)
    assert identical["native_vs_reference"]["relative_rmse_vs_reference"] == 0
    assert identical["reference_vs_fp32"]["relative_rmse_vs_fp32"] > 0


def test_reference_check_requires_a_native_vc_backend():
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--backends", "sage", "--vc-reference-check"])
    args = benchmark.parse_args(["--backends", "vc", "--vc-ablation", "--vc-reference-check", "--check-queries", "0"])
    assert args.vc_reference_check and args.vc_ablation and args.check_queries == 0


@pytest.mark.parametrize("count,scope", [(3, "all"), (2, "sampled")])
def test_report_identifies_actual_accuracy_scope_and_percent_units(capsys, count, scope):
    result = dict(version="test", median_ms=1, min_ms=1, max_ms=1, speedup_vs_baseline=1,
                  relative_rmse_vs_fp32=0.05, max_abs_error_vs_fp32=0.02)
    report = {
        "arguments": {"baseline": "vc", "queries": 3},
        "accuracy": {"query_indices": list(range(count))},
        "results": {"vc": result}, "vc_speedup_vs": {},
        "vc_reference_check": {"results": {"vc": {
            "native_vs_fp32": {"relative_rmse_vs_fp32": 0.05},
            "reference_vs_fp32": {"relative_rmse_vs_fp32": 0.049},
            "native_vs_reference": {"relative_rmse_vs_reference": 0.001, "max_abs_error_vs_reference": 0.0002},
        }}},
    }
    benchmark.print_report(report)
    text = capsys.readouterr().out
    assert f"{scope} {count}/3 query rows" in text
    assert "PERCENT" in text and "5.000000" in text and "4.900000" in text and "0.100000" in text
    assert "no pass/fail threshold" in text
