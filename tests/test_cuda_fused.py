"""Fused register/shared-memory path; native FP8 tests require Hopper.

The explicit emulator exercises the SAME kernel with scalar warp-MMA emulation
on development GPUs. It is never selected by the public attention API and is
not evidence of native FP8 performance or hardware accumulation agreement.
"""
from dataclasses import replace
import importlib.util
import os

import pytest
import torch

from vc_attention import Config, attention
from vc_attention.cuda_backend import _extension


has_cuda = torch.cuda.is_available() and importlib.util.find_spec("vc_attention._cuda_ext") is not None
has_hopper = has_cuda and torch.cuda.get_device_capability()[0] == 9
if os.environ.get("VC_REQUIRE_CUDA") == "1" and not has_hopper:
    raise RuntimeError("VC_REQUIRE_CUDA=1: fused FP8 verification requires Hopper")


@pytest.fixture(params=["emulated", "native"])
def launch(request):
    if not has_cuda or (request.param == "native" and not has_hopper):
        pytest.skip("native fused FP8 requires Hopper; emulator requires CUDA")
    ext = _extension()
    assert ext.fused_api_version == 1 and ext.fused_pipeline_version == 1, "Rebuild the 0.5.0 extension"

    def call(q, k, v, cfg, *, baseline=False):
        if request.param == "native":
            fn = ext.forward_fused_baseline if baseline else ext.forward_fused
        else:
            fn = ext.debug_fused_baseline_emulated if baseline else ext.debug_fused_emulated
        return fn(q, k, v, cfg.q_block, cfg.k_quant_block, cfg.kv_block,
                  cfg.v_smooth, cfg.expcast,
                  {"float32": 0, "float16": 1, "bfloat16": 2}[cfg.mean_dtype],
                  q.shape[-1] ** -0.5, cfg.k_smooth, cfg.qk_hadamard, False)
    return call


def test_fused_configuration_is_explicit_and_rejects_unsupported_kv_block():
    assert Config(backend="cuda_fp8", cuda_schedule="fused").cuda_schedule == "fused"
    assert Config().cuda_fused_pipeline is True
    with pytest.raises(ValueError, match="cuda_fused_pipeline"):
        Config(cuda_fused_pipeline="false")
    with pytest.raises(ValueError, match="kv_block=128"):
        Config(backend="cuda_fp8", cuda_schedule="fused", kv_block=64)


@pytest.mark.parametrize("shape", [
    (1, 1, 1, 1, 128, 256, "float16"),
    (33, 128, 32, 32, 32, 128, "float32"),
    (65, 129, 33, 65, 96, 384, "bfloat16"),
    (97, 256, 65, 37, 96, 256, "float16"),
    (129, 641, 128, 128, 256, 768, "float16"),
    (3, 769, 16, 1, 32, 384, "float32"),
])
@pytest.mark.parametrize("smooth,cast", [(False, False), (False, True), (True, False), (True, True)])
def test_pipeline_matches_retained_core_across_prefetch_tails_and_scale_boundaries(launch, shape, smooth, cast):
    nq, nk, d, dv, qb, kq, mean_dtype = shape
    generator = torch.Generator().manual_seed(503)
    q = torch.randn(1, 1, nq, d, generator=generator).cuda()
    k = torch.randn(1, 1, nk, d, generator=generator).cuda()
    v = torch.randn(1, 1, nk, dv, generator=generator).cuda()
    for start in range(0, nk, 128):
        k[:, :, start:start + 128] *= 1 + start / 256
        v[:, :, start:start + 128] += 2 * ((start // 128) % 3 - 1)
    cfg = Config(backend="cuda_fp8", cuda_schedule="fused", clusters=1,
                 v_smooth=smooth, expcast=cast, q_block=qb, k_quant_block=kq, mean_dtype=mean_dtype)
    expected = launch(q, k, v, cfg, baseline=True)
    actual = launch(q, k, v, cfg)
    # Same preprocessing, MMA order and scalar arithmetic. Scheduling and
    # async copies must not alter a single result, including padded channels.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 1, 1, 1, 16, 13), (2, 2, 65, 259, 37, 21),
                                    (1, 2, 129, 257, 128, 128), (1, 1, 33, 131, 64, 64)])
@pytest.mark.parametrize("smooth,cast", [(False, False), (False, True), (True, False), (True, True)])
def test_fused_signed_data_tail_heads_scales_and_mean_vs_independent_reference(launch, shape, smooth, cast):
    torch.backends.cuda.matmul.allow_tf32 = False
    b, h, nq, nk, d, dv = shape
    generator = torch.Generator().manual_seed(113)
    q = torch.randn(b, h, nq, d, generator=generator)
    k = torch.randn(b, h, nk, d, generator=generator)
    v = torch.randn(b, h, nk, dv, generator=generator)
    k[:, :, :128] *= 0.2  # Two KV tiles share one K quantization scale.
    v[:, :, :128] += 2.0  # Tile-dependent means must be restored and rescaled.
    cfg = Config(backend="cuda_fp8", cuda_schedule="fused", clusters=1,
                 v_smooth=smooth, expcast=cast)
    expected, _ = attention(q, k, v, replace(cfg, backend="reference"))
    actual = launch(q.cuda(), k.cuda(), v.cuda(), cfg).cpu()
    error = actual - expected
    relative = error.norm() / expected.norm().clamp_min(1e-6)
    # Independent Q/K preprocessing and MMA reduction can change FP8 rounding.
    # These are implementation-agreement limits, NOT model quality thresholds.
    assert relative < 0.015
    assert error.abs().max() < 0.06
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("cast", [False, True])
@pytest.mark.parametrize("mean_dtype", ["float32", "float16", "bfloat16"])
def test_fused_known_answer_maximum_change_and_tail(launch, cast, mean_dtype):
    q = torch.ones(2, 2, 65, 16, device="cuda")
    k = torch.cat([torch.full((2, 2, 128, 16), -20., device="cuda"),
                   torch.full((2, 2, 3, 16), 20., device="cuda")], 2)
    offset = torch.arange(4, device="cuda").reshape(2, 2, 1, 1)
    v = torch.cat([torch.full((2, 2, 128, 21), 3., device="cuda"),
                   torch.full((2, 2, 3, 21), 9., device="cuda")], 2) + offset
    cfg = Config(backend="cuda_fp8", cuda_schedule="fused", clusters=1,
                 v_smooth=True, expcast=cast, mean_dtype=mean_dtype)
    actual = launch(q, k, v, cfg)
    torch.testing.assert_close(actual, (9 + offset).expand(2, 2, 65, 21).float(), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_noncontiguous_input_current_stream_and_updated_values(launch, dtype):
    q = torch.zeros(1, 2, 35, 32, device="cuda", dtype=dtype)
    k = torch.zeros(1, 2, 129, 32, device="cuda", dtype=dtype)
    v = torch.full((1, 2, 129, 21), 2., device="cuda", dtype=dtype)
    q, k, v = (torch.stack([x, x], -1)[..., 0] for x in (q, k, v))
    cfg = Config(backend="cuda_fp8", cuda_schedule="fused", v_smooth=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        first = launch(q, k, v, cfg)
        second = launch(q, k, v + 5, cfg)
    stream.synchronize()
    assert first.dtype == dtype
    torch.testing.assert_close(first, torch.full_like(first, 2), rtol=0, atol=0)
    torch.testing.assert_close(second, torch.full_like(second, 7), rtol=0, atol=0)


@pytest.mark.skipif(not has_hopper, reason="public fused API requires Hopper")
@pytest.mark.hopper
@pytest.mark.parametrize("cast", [False, True])
def test_public_fused_vs_batched_with_fixed_grouping(cast):
    torch.manual_seed(907)
    q, k, v = (torch.randn(1, 2, n, 128, device="cuda", dtype=torch.bfloat16) for n in (137, 389, 389))
    cfg = Config(backend="cuda_fp8", cuda_schedule="fused", v_smooth=True, expcast=cast, clusters=1)
    actual, stats = attention(q, k, v, cfg)
    control, _ = attention(q, k, v, replace(cfg, cuda_schedule="batched"))
    error = actual.float() - control.float()
    assert error.norm() / control.float().norm() < 0.02
    assert error.abs().max() < 0.05
    assert stats["native_fused_attention"] and stats["native_fp8_matmul"]
    assert not stats["preprocessing_fused"]
    assert stats["cuda_fused_pipeline"]
    baseline, baseline_stats = attention(q, k, v, replace(cfg, cuda_fused_pipeline=False))
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    assert not baseline_stats["cuda_fused_pipeline"]
