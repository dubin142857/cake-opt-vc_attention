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
pytestmark=pytest.mark.skipif(getattr(ext,"benchmark_api_version",0)!=3,reason="Build the v3 component benchmark")


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
