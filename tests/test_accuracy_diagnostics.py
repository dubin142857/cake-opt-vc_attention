"""Numerical and orchestration checks for the standalone stage diagnostic."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from vc_attention.core import Config, attention


spec = importlib.util.spec_from_file_location(
    "vc_accuracy_diagnostic", Path(__file__).resolve().parents[1] / "scripts" / "diagnose_cuda_accuracy.py",
)
diagnostic = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diagnostic
spec.loader.exec_module(diagnostic)


def inputs(dtype=torch.float32):
    gen = torch.Generator().manual_seed(42)
    q, k = [torch.randn(1, 2, n, 16, generator=gen) for n in (9, 19)]
    v = torch.randn(1, 2, 19, 7, generator=gen) + 1.2345
    return tuple(x.to(dtype) for x in (q, k, v))


@pytest.mark.parametrize("smooth", [False, True])
def test_unquantized_streaming_with_preprocessing_and_tail_matches_dense(smooth):
    q, k, v = inputs()
    cfg = Config(q_block=7, kv_block=8, clusters=3, v_smooth=smooth)
    prepared = diagnostic.prepare((q, k, v), cfg, 0.07)
    actual = diagnostic.stage_output(prepared, quantize_qk=False, probability="fp32", quantize_v=False)
    expected = (q @ k.transpose(-2, -1) * 0.07).softmax(-1) @ v
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("stage", list(diagnostic.STAGES))
def test_each_partial_quantization_stage_against_independent_single_tile_oracle(stage):
    q, k, v = inputs()
    cfg = Config(q_block=32, kv_block=32, k_smooth=False, qk_hadamard=False)
    prepared = diagnostic.prepare((q, k, v), cfg, 0.17)
    qk, probability, quantize_v = diagnostic.STAGES[stage]
    actual = diagnostic.stage_output(prepared, quantize_qk=qk, probability=probability, quantize_v=quantize_v)
    if qk:
        # One quantization block, no preprocessing: independently encode both operands.
        qscale = q.abs().amax((-2, -1), keepdim=True) / 448
        kscale = k.abs().amax((-2, -1), keepdim=True) / 448
        q = (q / qscale).to(torch.float8_e4m3fn).float() * qscale
        k = (k / kscale).to(torch.float8_e4m3fn).float() * kscale
    scores = (q @ k.transpose(-2, -1)) * 0.17
    shifted = scores - scores.amax(-1, keepdim=True)
    if probability == "expcast":
        codes = (shifted.double() * 11.541560173034668 + 119.6500015258789).float().round().clamp(0, 120).byte()
        p = codes.view(torch.float8_e4m3fn).float() / 256
        numerator_p = p
    else:
        p = shifted.exp()
        numerator_p = (p * 256).to(torch.float8_e4m3fn).float() / 256 if probability == "e4m3" else p
    if quantize_v:
        scale = v.abs().amax(-2, keepdim=True) / 448
        v = (v / scale).to(torch.float8_e4m3fn).float() * scale
    expected = numerator_p @ v / p.sum(-1, keepdim=True)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_all_stage_groups_match_public_reference_with_same_layout_and_record_cast(dtype):
    qkv = inputs(dtype)
    cfg = Config(q_block=7, kv_block=8, clusters=3)
    result = diagnostic.diagnose(qkv, 0.11, cfg, [0, 4, 8], 2, native=False, progress=lambda _: None)
    for group in result.values():
        assert set(group["stages"]) == set(diagnostic.STAGES)
        assert not group["native_checks"]
        for check in group["reference_checks"].values():
            assert check["max_abs"] == 0
        baseline = group["stages"]["none"]
        assert baseline["fp32_output_vs_dense"]["relative_rmse"] < 1e-6
        if dtype != torch.float32:
            assert baseline["cast_vs_fp32_output"]["relative_rmse"] > 0
    assert result["v_smooth_off"]["permutation_sha256"] is None
    assert result["v_smooth_on"]["permutation_sha256"] is not None
    json.dumps(result, allow_nan=False)


def test_sampling_never_recomputes_q_quantization_scales():
    q, k, v = inputs()
    q[:, :, 1, :] *= 100  # Unsampled outlier must still determine the block scale.
    cfg = Config(q_block=32, kv_block=32, clusters=1)
    prepared = diagnostic.prepare((q, k, v), cfg, 0.25)
    full = diagnostic.stage_output(prepared, quantize_qk=True, probability="e4m3", quantize_v=True)
    sampled = diagnostic.diagnose((q, k, v), 0.25, cfg, [0, 8], 1, native=False, progress=lambda _: None)
    golden = diagnostic.dense_reference((q, k, v), [0, 8], 1, 0.25)
    expected = diagnostic.metrics(full[:, :, [0, 8], :], golden)
    assert sampled["v_smooth_off"]["stages"]["qk+p+v"]["fp32_output_vs_dense"] == expected


def test_native_comparison_freezes_layout_and_reports_known_perturbation(monkeypatch):
    seen = []

    def fake_native(q, k, v, cfg, *, cache, scale):
        if cfg.backend != "cuda_fp8":
            return attention(q, k, v, cfg, cache=cache, scale=scale)
        seen.append((cfg.v_smooth, cfg.expcast))
        assert cache.config == cfg
        reference_cfg = replace(cfg, backend="reference")
        output, stats = attention(q, k, v, reference_cfg,
                                  cache=diagnostic.FixedLayout(reference_cfg, cache.prepared), scale=scale)
        return output + 0.125, stats

    monkeypatch.setattr(diagnostic, "attention", fake_native)
    # Supported native geometry; native arithmetic is explicitly stubbed here.
    result = diagnostic.diagnose(inputs(), 0.25, Config(clusters=2), [0, 8], 1,
                                 native=True, progress=lambda _: None)
    assert seen == [(False, False), (False, True), (True, False), (True, True)]
    for group in result.values():
        for row in group["native_checks"].values():
            assert row["cuda_vs_reference"]["max_abs"] == pytest.approx(0.125, abs=2e-7)
            assert row["cuda_vs_reference"]["relative_rmse"] > 0


def test_capture_preserves_tensors_full_context_and_model_scale(tmp_path):
    qkv = inputs(torch.bfloat16)
    path = tmp_path / "post_rope.pt"
    torch.save(dict(zip(("q", "k", "v"), qkv), metadata={"scale": 0.123}), path)
    args = diagnostic.parse_args(["--capture", str(path), "--reference-only", "--device", "cpu"])
    actual, scale, source = diagnostic.load_inputs(args, torch.device("cpu"))
    assert scale == 0.123
    for x, expected in zip(actual, qkv):
        torch.testing.assert_close(x, expected, rtol=0, atol=0)
    assert source["dtype_preserved"]
    assert len(source["inputs"]["q"]["sha256"]) == 64
    args.scale = 0.17
    assert diagnostic.load_inputs(args, torch.device("cpu"))[1] == 0.17


def test_cli_defaults_full_queries_and_requires_explicit_native_opt_out():
    args = diagnostic.parse_args([])
    assert args.check_queries == 0 and not args.reference_only
    assert (args.seed, args.queries, args.tokens, args.heads, args.dim, args.dtype) == (1234, 1024, 1024, 2, 128, "bfloat16")
    with pytest.raises(ValueError, match="scale"):
        diagnostic.prepare(inputs(), Config(), float("nan"))
    with pytest.raises(SystemExit):
        diagnostic.parse_args(["--check-queries", "-1"])


def test_report_labels_percent_and_native_scope(capsys):
    rows = diagnostic.diagnose(inputs(), 0.25, Config(clusters=1), [0, 8], 2, native=False, progress=lambda _: None)
    rows["v_smooth_off"]["stages"]["qk"]["fp32_output_vs_dense"]["relative_rmse"] = 0.05
    diagnostic.print_report({"results": rows})
    text = capsys.readouterr().out
    assert "PERCENT" in text and "5.000000" in text
    assert "FP32-out" in text and "cast-out" in text
    assert "not kernel performance" in text and "hold grouping fixed" in text


@pytest.mark.hopper
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
                    reason="requires a Hopper GPU and the compiled extension")
def test_real_native_diagnostic_uses_fixed_layout_and_restores_constant_values():
    q = torch.zeros(1, 1, 3, 16, device="cuda")
    k = torch.zeros(1, 1, 5, 16, device="cuda")
    v = torch.full_like(k, 3)
    result = diagnostic.diagnose((q, k, v), 0.25, Config(clusters=1), [0, 1, 2], 2,
                                 native=True, progress=lambda _: None)
    for group in result.values():
        for row in group["native_checks"].values():
            assert row["stats"]["native_fp8_matmul"]
            assert not row["stats"]["layout_refreshed"]
            assert row["cuda_vs_dense"]["max_abs"] < 2e-5
