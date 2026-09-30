"""Executable candidate checks; build scripts/benchmark_expcast.cu first.

The integer candidate is diagnostic only and is compared to PyTorch's decoded
FP8 values accumulated in FP64, not a copy of the candidate's bit arithmetic.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ext=pytest.importorskip("_vc_expcast_bench")
pytestmark=pytest.mark.skipif(getattr(ext,"benchmark_api_version",0)!=6,reason="Build the v6 component benchmark")


def load_benchmark():
    path=Path(__file__).resolve().parents[1]/"scripts/benchmark_expcast.py"
    spec=importlib.util.spec_from_file_location("component_benchmark",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_all_fp8_pairs_subnormals_and_encoding_rounding_boundaries():
    report=load_benchmark().validate_components(ext)
    assert report["status"]=="passed"
    assert report["decode_rows"]==121*121+4096+121


@pytest.mark.parametrize("iterations",[1,3,128])
@pytest.mark.parametrize("blocks",[1,3])
def test_benchmark_checksums_preserve_every_candidate_input(iterations,blocks):
    for group,modes in ((0,(0,1)),(1,(0,1,2))):
        outputs=[]
        for mode in modes:
            out=torch.empty(blocks,128,device="cuda")
            ext.component_run(out,group,mode,iterations)
            outputs.append(out)
        torch.cuda.synchronize()
        assert outputs[0].view(torch.int32).count_nonzero().item()>0
        for output in outputs[1:]:
            assert torch.equal(output.view(torch.int32),outputs[0].view(torch.int32))


@pytest.mark.parametrize("rows",[1,127,129,257])
def test_integer_decode_exact_maximum_minimum_and_tail(rows):
    codes=torch.zeros(rows,32,device="cuda",dtype=torch.uint8)
    codes[::2]=120
    codes[1::4]=1
    if rows>2:
        codes[-1,:]=torch.tensor([0,1,7,8,15,16,119,120],device="cuda",dtype=torch.uint8).repeat(4)
    expected=codes.view(torch.float8_e4m3fn).float().double().sum(-1).float()
    actual=ext.decode_check(codes)
    torch.testing.assert_close(actual,expected[:,None].expand_as(actual),rtol=0,atol=0)


def test_component_rejects_invalid_domains_and_alignment():
    # Some Windows PyTorch builds omit the C++ exception text.
    with pytest.raises(RuntimeError):
        ext.decode_check(torch.full((1,32),121,device="cuda",dtype=torch.uint8))
    misaligned=torch.zeros(33,device="cuda",dtype=torch.uint8)[1:].view(1,32)
    with pytest.raises(RuntimeError):
        ext.decode_check(misaligned)
    for invalid in (float("nan"),.01,float("inf")):
        with pytest.raises(RuntimeError):
            ext.encode_check(torch.tensor([0.,-1.,-2.,invalid],device="cuda"),0)
    with pytest.raises(RuntimeError):
        ext.encode_check(torch.zeros(3,device="cuda"),0)
    with pytest.raises(RuntimeError):
        ext.component_run(torch.empty(1,127,device="cuda"),0,0,1)
    with pytest.raises(RuntimeError):
        ext.component_run(torch.empty(1,128,device="cuda"),2,0,1)


def test_stage_report_retains_trials_resources_and_nonadditive_scope():
    args=SimpleNamespace(blocks=2,iterations=3,trials=3)
    output=torch.empty(args.blocks,128,device="cuda")
    report=load_benchmark().benchmark_components(ext,args,output)
    assert len(report["trial_order"])==3
    assert "not additive" in report["timing_caveat"]
    for group in report["groups"].values():
        baseline=group["variants"][group["baseline"]]["median_ms"]
        for row in group["variants"].values():
            assert len(row["trials_ms"])==3 and row["median_ms"]>0
            assert row["baseline_over_variant"]==pytest.approx(baseline/row["median_ms"])
            assert row["registers_per_thread"]>0 and row["local_bytes_per_thread"]>=0


def test_probe_respects_nondefault_cuda_stream():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        codes=torch.full((129,32),120,device="cuda",dtype=torch.uint8)
        decoded=ext.decode_check(codes)
        encoded=ext.encode_check(torch.zeros(260,device="cuda"),0)
    stream.synchronize()
    assert torch.equal(decoded,torch.full_like(decoded,8192.))
    assert torch.equal(encoded,torch.full_like(encoded,120))


def test_affine_encoding_oracle_and_reported_rounding_difference():
    report=load_benchmark().validate_affine(ext)
    assert report["independent_fp64_fma_oracle_exact"]
    assert report["values"]==56*4096
    assert report["maximum_code_difference"]<=1
    assert 0<=report["vc11_byte_mismatch_percent"]<=100


def test_affine_probe_rejects_mismatched_coefficients():
    with pytest.raises(RuntimeError):
        ext.affine_check(torch.zeros(6,device="cuda"),torch.ones(2,device="cuda"),torch.zeros(3,device="cuda"))


def test_paired_conversion_all_rounding_boundaries_and_finite_bit_patterns():
    # Identity affine map exposes the production packed RNE helper directly.
    # Includes ties, either neighbouring float, every output-byte pair, signed
    # zeros, subnormals, infinities and signed integer overflow/underflow.
    ties=torch.arange(-1,257,device="cuda",dtype=torch.float32)+.5
    values=torch.cat((ties,torch.nextafter(ties,torch.full_like(ties,-float("inf"))),
        torch.nextafter(ties,torch.full_like(ties,float("inf"))),
        torch.tensor([-float("inf"),float("inf"),-0.,0.,1e-40,-1e-40,-1e30,1e30],device="cuda")))
    boundary_pairs=torch.cartesian_prod(values,values).contiguous()
    gen=torch.Generator(device="cuda").manual_seed(2414)
    bits=torch.randint(-(2**31),2**31-1,(262144,),device="cuda",dtype=torch.int32,generator=gen)
    random_values=bits.view(torch.float32)
    random_values=random_values[torch.isfinite(random_values)]
    random_pairs=random_values[:random_values.numel()//2*2].reshape(-1,2)
    scores=torch.cat((boundary_pairs,random_pairs))
    gain=torch.ones(scores.shape[0],device="cuda");bias=torch.zeros_like(gain)
    old=ext.affine_check(scores,gain,bias)
    actual=ext.affine_check(scores,gain,bias,True)
    expected=scores.round().clamp(0,255).byte()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(actual,old,rtol=0,atol=0)
    # Four-byte path must preserve both byte pairs, including unlike high/low
    # bytes. The independent oracle catches swapped/copied-upper-pair errors.
    count=scores.numel()//4*4
    quads=scores.flatten()[:count].reshape(-1,4).contiguous()
    quad=ext.quad_check(quads,torch.ones(quads.shape[0],device="cuda"),torch.zeros(quads.shape[0],device="cuda"))
    torch.testing.assert_close(quad,expected.flatten()[:count].reshape_as(quad),rtol=0,atol=0)


@pytest.mark.parametrize("iterations",[1,3,128])
def test_paired_probability_checksum_is_exact_vc12(iterations):
    old=torch.empty(3,128,device="cuda");new=torch.empty_like(old)
    ext.run(old,4,iterations)
    for mode in (5,6):
        ext.run(new,mode,iterations)
        torch.testing.assert_close(new,old,rtol=0,atol=0)


def test_quad_probe_rejects_incomplete_groups_and_mismatched_coefficients():
    for size,gains,biases in ((6,2,2),(8,1,2),(8,2,1),(0,0,0)):
        with pytest.raises(RuntimeError):
            ext.quad_check(torch.zeros(size,device="cuda"),torch.ones(gains,device="cuda"),torch.zeros(biases,device="cuda"))


def test_quad_probe_uses_current_stream_and_keeps_four_distinct_bytes():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        scores=torch.arange(256,device="cuda",dtype=torch.float32).reshape(-1,4)
        actual=ext.quad_check(scores,torch.ones(64,device="cuda"),torch.zeros(64,device="cuda"))
    stream.synchronize()
    torch.testing.assert_close(actual,scores.byte(),rtol=0,atol=0)
