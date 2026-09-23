"""Fixed-payload diagnostics: actual CUDA helpers; GEMM needs Hopper."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from vc_attention.core import Config, _divide_fp32, attention
from vc_attention.cuda_backend import _extension


script_dir = str(Path(__file__).resolve().parents[1] / "scripts")
sys.path.insert(0, script_dir)
try:
    import diagnose_cuda_payloads as payloads
finally:
    sys.path.remove(script_dir)


cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("vc_attention._cuda_ext") is None,
    reason="requires CUDA and the rebuilt extension",
)
hopper = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
                            reason="requires Hopper FP8 Tensor Cores")


def test_error_aggregation_is_element_weighted_and_tracks_first_tile():
    d = payloads.Differences()
    d.add("probe", torch.tensor([2.]), torch.tensor([1.]), {"tile": 0})
    d.add("probe", torch.ones(3), torch.ones(3), {"tile": 1})
    result = d.report()["probe"]
    assert result["relative_rmse"] == 0.5  # sqrt(sum(error^2)/sum(reference^2))
    assert result["different_fraction"] == 0.25
    assert result["first_difference"] == {"tile": 0}


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda)])
def test_scale_division_matches_independent_rounded_quotient(device):
    x = torch.linspace(0.1, 16, 2048, dtype=torch.float32).to(device)
    expected = (x.cpu().double() / 448).float()
    torch.testing.assert_close(_divide_fp32(x, 448).cpu(), expected, atol=0, rtol=0)


@cuda
@pytest.mark.parametrize("block", [32, 128, 256])
def test_native_qk_bytes_and_scales_match_reference_including_padding(block):
    torch.manual_seed(21)
    x = torch.randn(2, 2, 259, 37, device="cuda").to(torch.bfloat16).float()
    actual = _extension().debug_quantize(x, block)
    expected = payloads.reference_qk(x, block)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0, rtol=0)


@cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("smooth", [False, True])
def test_value_encoding_and_scaled_mean_precision(dtype, smooth):
    # A power-of-two number of BF16 values has an exactly summable FP32 mean.
    torch.manual_seed(8)
    x = (torch.randn(1, 2, 256, 21, device="cuda") * 0.25 + 2).bfloat16().float()
    cfg = Config(mean_dtype=dtype, kv_block=128, v_smooth=smooth)
    actual = _extension().debug_values(x, cfg.kv_block, smooth, {"float32": 0, "float16": 1, "bfloat16": 2}[dtype])
    expected = payloads.reference_values(x, cfg, smooth)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0, rtol=0)


@cuda
@pytest.mark.parametrize("cast", [False, True])
def test_native_softmax_tail_previous_state_and_accumulation(cast):
    ext = _extension()
    device = "cuda"
    torch.manual_seed(4)
    scores = torch.randn(32, 64, device=device)
    qs, ks = torch.tensor(0.75, device=device), torch.tensor(0.5, device=device)
    maximum = torch.linspace(-3, 2, 32, device=device)
    denom = torch.ones(32, device=device)
    args = scores, qs, ks, maximum, denom, 0.25, 19, 35, cast
    actual = ext.debug_softmax(*args)
    expected = payloads.python_softmax(*args)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    for a, e in zip(actual[1:], expected[1:]):
        torch.testing.assert_close(a, e, atol=1e-5, rtol=2e-6)
    p, alpha, mass, _, new_denom = actual
    previous = torch.randn(32, 32, device=device)
    pv = torch.randn_like(previous)
    vs = torch.rand(32, device=device)
    mean = torch.randn(32, device=device).half()
    acc, out = ext.debug_accumulate(previous, pv, mass, alpha, vs, mean, new_denom, 19)
    expected_acc = payloads.python_accumulate(previous, pv, mass, alpha, vs, mean)
    torch.testing.assert_close(acc, expected_acc, atol=0, rtol=0)
    torch.testing.assert_close(out[:19], expected_acc[:19] / new_denom[:19, None], atol=2e-6, rtol=2e-6)
    assert torch.count_nonzero(out[19:]).item() == 0


@cuda
def test_diagnostic_shape_guards_prevent_invalid_kernel_inputs():
    ext = _extension()
    with pytest.raises(RuntimeError):
        ext.debug_quantize(torch.ones(4, 4, device="cuda"), 32)
    with pytest.raises(RuntimeError):
        ext.debug_values(torch.ones(1, 1, 3, 16, device="cuda"), 7, True, 1)
    with pytest.raises(RuntimeError):
        ext.debug_softmax(torch.zeros(32, 32, device="cuda"), torch.ones(1, device="cuda"),
                          torch.ones(1, device="cuda"), torch.zeros(31, device="cuda"),
                          torch.zeros(31, device="cuda"), 1., 32, 32, False)


@cuda
def test_replay_orchestration_with_real_kernels_and_explicit_fp32_gemm_stub():
    ext = _extension()

    class ExplicitFP32Gemm:
        # Only a test double for Ampere; CLI never substitutes for native GEMM.
        def debug_gemm(self, a, b):
            return payloads.decode(a) @ payloads.decode(b).T

        def __getattr__(self, name):
            return getattr(ext, name)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(123)
    q, k = [torch.randn(1, 2, n, 37, device="cuda").bfloat16() for n in (35, 145)]
    v = torch.randn(1, 2, 145, 21, device="cuda").bfloat16()
    cfg = Config(q_block=32, kv_block=32, clusters=1, v_smooth=True)
    prepared = payloads.stages.prepare((q, k, v), cfg, 37**-0.5)
    ref, native, _ = payloads.prepare_packs((q, k, v), cfg, prepared, ext)
    for cast in (False, True):
        py, simulated, stats = payloads.replay_packed(native, (q, k, v), cfg, 37**-0.5, cast, ExplicitFP32Gemm())
        torch.testing.assert_close(py, simulated, atol=2e-5, rtol=2e-5)
        assert stats["QK_same_bytes"]["max_abs"] == 0
        assert stats["PV_same_bytes"]["max_abs"] == 0
    # Post-GEMM scales differ in rounding from the original pre-dequantized
    # reference, but on these well-conditioned inputs outputs should be close.
    result, _, _ = payloads.replay_packed(ref, (q, k, v), cfg, 37**-0.5, False)
    expected, _ = attention(q, k, v, cfg, cache=payloads.stages.FixedLayout(cfg, prepared))
    assert payloads.metrics(result.to(q.dtype).float(), expected.float())["relative_rmse"] < 0.01


@hopper
@pytest.mark.hopper
def test_real_fp8_gemm_known_signed_payloads_and_full_replay():
    ext = _extension()
    a = torch.tensor([1., -2., 0.5, 4.], device="cuda").repeat(32, 8).to(torch.float8_e4m3fn).view(torch.uint8)
    b = torch.tensor([-1., 2., 0., 0.25], device="cuda").repeat(64, 8).to(torch.float8_e4m3fn).view(torch.uint8)
    torch.testing.assert_close(ext.debug_gemm(a, b), payloads.decode(a) @ payloads.decode(b).T, atol=0, rtol=0)
    q = torch.zeros(1, 1, 19, 16, device="cuda")
    k = torch.zeros(1, 1, 35, 16, device="cuda")
    v = torch.full_like(k, 3)
    cfg = Config(v_smooth=True, clusters=1)
    prepared = payloads.stages.prepare((q, k, v), cfg, 0.25)
    _, native, _ = payloads.prepare_packs((q, k, v), cfg, prepared, ext)
    _, actual, _ = payloads.replay_packed(native, (q, k, v), cfg, 0.25, True, ext)
    expected, _ = attention(q, k, v, replace(cfg, backend="cuda_fp8", expcast=True))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_granularity_sweep_keeps_attention_tiles_fixed():
    torch.manual_seed(9)
    qkv = tuple(torch.randn(1, 1, n, 16) for n in (19, 145, 145))
    cfg = Config(clusters=1)
    prepared = payloads.stages.prepare(qkv, cfg, 0.25)
    ids = [0, 18]
    golden = payloads.stages.dense_reference(qkv, ids, 2, 0.25)
    rows = payloads.qk_granularity_sweep(prepared, qkv, ids, golden)
    assert len(rows) == 4
    assert all(r["attention_q_tile"] == 128 and r["attention_kv_tile"] == 128 for r in rows.values())
    output = payloads.stages.stage_output(prepared, quantize_qk=True, probability="fp32", quantize_v=False)
    expected = payloads.metrics(payloads.stages.select_output(output, ids), golden)
    assert rows["q128_k256"]["qk_only"]["fp32_output"] == expected
