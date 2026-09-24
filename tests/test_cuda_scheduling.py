"""Batched CUDA scheduling, with real helper kernels on Ampere and FP8 on Hopper."""
from dataclasses import replace
import importlib.util

import pytest
import torch

from vc_attention import Config, attention
from vc_attention.cuda_backend import _extension


cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("vc_attention._cuda_ext") is None,
    reason="requires CUDA and rebuilt extension",
)
hopper = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
                            reason="requires Hopper FP8 GEMM")


def test_schedule_validation():
    assert Config().cuda_schedule == "batched"
    with pytest.raises(ValueError, match="cuda_schedule"):
        Config(cuda_schedule="unknown")


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "expanded"])
def test_native_finite_check_preserves_validation_and_strides(dtype, layout):
    ext = _extension()
    tensors = [torch.ones(2, 3, n, 16, device="cuda", dtype=dtype) for n in (17, 35, 35)]
    def view(x):
        if layout == "strided":
            return torch.stack((x, x), -1)[..., 0]
        if layout == "expanded":
            return x[:1].expand_as(x)
        return x
    assert ext.all_finite(*(view(x) for x in tensors))
    for index, bad in enumerate((float("nan"), float("inf"), -float("inf"))):
        current = [x.clone() for x in tensors]
        current[index][0, 1, -1, 3] = bad
        assert not ext.all_finite(*(view(x) for x in current))
        with pytest.raises(ValueError, match="finite"):
            attention(*(view(x) for x in current), Config(backend="cuda_fp8"))


@cuda
@pytest.mark.parametrize("cast", [False, True])
@pytest.mark.parametrize("key_tile", [0, 1, 2])
@pytest.mark.parametrize("qb,rows,start", [(32, 96, 0), (96, 192, 960)])
def test_batched_softmax_and_accumulation_match_independent_serial_tiles(cast, key_tile, qb, rows, start):
    ext = _extension()
    torch.manual_seed(71)
    heads, bk, vp, nk = 3, 32, 32, 75
    nq = start + rows - 11
    opts = dict(device="cuda", dtype=torch.float32)
    scores = torch.randn(heads, rows, bk, **opts)
    pv = torch.randn(heads, rows, vp, **opts)
    qs = torch.rand(heads, (nq + qb - 1) // qb, **opts) + 0.1
    ks = torch.rand(heads, 2, **opts) + 0.1
    vs = torch.rand(heads, 3, vp, **opts) + 0.1
    mean = torch.randn(heads, 3, vp, **opts).half()
    maximum = torch.randn(heads, rows, **opts)
    denom = torch.rand(heads, rows, **opts) + 1
    accum = torch.randn(heads, rows, vp, **opts)
    if key_tile == 0:
        # The first tile must not read uninitialized state, even NaN garbage.
        maximum.fill_(float("nan")); denom.fill_(float("nan")); accum.fill_(float("nan"))
    tensors = [scores, pv, qs, ks, vs, mean, maximum, denom, accum]
    actual = ext.debug_batched_step(tensors, qb, 64, nq, nk, start, key_tile, 0.125, cast)
    for h in range(heads):
        for row in range(0, rows, qb):
            valid = min(qb, nq - start - row)
            sl = slice(row, row + qb)
            old_max = torch.full((qb,), -float("inf"), **opts) if key_tile == 0 else maximum[h, sl]
            old_denom = torch.zeros(qb, **opts) if key_tile == 0 else denom[h, sl]
            old_accum = torch.zeros(qb, vp, **opts) if key_tile == 0 else accum[h, sl]
            expected = ext.debug_softmax(scores[h, sl], qs[h, (start + row) // qb], ks[h, key_tile // 2],
                                         old_max, old_denom, 0.125, valid, min(bk, nk - key_tile * bk), cast)
            for got, want in zip(actual[:5], expected):
                torch.testing.assert_close(got[h, sl], want, atol=0, rtol=0)
            acc, out = ext.debug_accumulate(old_accum, pv[h, sl], expected[2], expected[1],
                                            vs[h, key_tile], mean[h, key_tile], expected[4], valid)
            if key_tile == 2:
                torch.testing.assert_close(actual[6][h, start + row:start + row + valid], out[:valid], atol=0, rtol=0)
            else:
                torch.testing.assert_close(actual[5][h, sl], acc, atol=0, rtol=0)
    if start:
        assert torch.count_nonzero(actual[6][:, :start]) == 0


@hopper
@pytest.mark.hopper
@pytest.mark.parametrize("cast", [False, True])
@pytest.mark.parametrize("smooth", [False, True])
def test_batched_fp8_matches_serial_signed_inputs_multiple_heads_and_query_chunks(cast, smooth):
    torch.manual_seed(71)
    q = torch.randn(2, 2, 1057, 37, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 2, 259, 37, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 2, 259, 21, device="cuda", dtype=torch.bfloat16)
    # One cluster makes the common permutation identity; retain mean recovery.
    cfg = Config(backend="cuda_fp8", expcast=cast, v_smooth=smooth, clusters=1)
    old, _ = attention(q, k, v, replace(cfg, cuda_schedule="serial"))
    actual, stats = attention(q, k, v, cfg)
    err = actual.float() - old.float()
    relative = err.square().sum().sqrt() / old.float().square().sum().sqrt()
    assert relative < 0.015  # GEMM geometry may change FP8 accumulation/boundary rounding.
    assert err.abs().max() < 0.04
    assert stats["cuda_schedule"] == "batched"
    # Compare the SAME re-quantized input across streams. In unsmoothed FP8,
    # V + 8 changes the per-channel scale and rounding; output(V + 8) need not
    # equal output(V) + 8. Ordinary E4M3 P also uses an unquantized denominator.
    expected_shifted, _ = attention(q, k, v + 8, cfg)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        shifted, _ = attention(q, k, v + 8, cfg)
    stream.synchronize()
    torch.testing.assert_close(shifted, expected_shifted, atol=0, rtol=0)


@hopper
@pytest.mark.hopper
@pytest.mark.parametrize("cast", [False, True])
@pytest.mark.parametrize("smooth", [False, True])
def test_batched_known_answer_with_distinct_heads_and_tail_chunk(cast, smooth):
    q = torch.zeros(2, 3, 1057, 16, device="cuda")
    k = torch.zeros(2, 3, 35, 16, device="cuda")
    values = torch.arange(1, 7, device="cuda").reshape(2, 3, 1, 1).float()
    v = values.expand_as(k).contiguous()
    cfg = Config(backend="cuda_fp8", v_smooth=smooth, clusters=1, expcast=cast)
    result, _ = attention(q, k, v, cfg)
    torch.testing.assert_close(result, values.expand_as(q), atol=1e-5, rtol=1e-5)
    # Uniform probabilities and constant V give a known answer even in FP8.
    # Check descriptor reuse with NEW values produced on the caller's stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        shifted, _ = attention(q, k, v + 8, cfg)
    stream.synchronize()
    torch.testing.assert_close(shifted, (values + 8).expand_as(q), atol=1e-5, rtol=1e-5)
