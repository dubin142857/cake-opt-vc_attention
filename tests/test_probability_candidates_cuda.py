"""Real vc17 helpers; direct decode uses an independent PyTorch FP8 oracle."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

ext=pytest.importorskip("_vc_probability_bench")
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("probability_bench",ROOT/"scripts/benchmark_probability.py")
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)


def test_all_codes_histories_fallback_masks_and_independent_tile_sum():
    report=bench.validate_candidates(ext)
    assert report["candidate_history_cases"]==16


@pytest.mark.parametrize("mode",range(1,5))
def test_encoding_boundaries_preserve_p_and_online_state(mode):
    # Identity scale; max remains exactly zero. Four lanes share a logical row.
    ties=((torch.arange(120,device="cuda",dtype=torch.float64)+.5-119.6500015258789)/11.541560173034668).float()
    values=torch.cat((ties,torch.nextafter(ties,torch.full_like(ties,-float("inf"))),
        torch.nextafter(ties,torch.full_like(ties,float("inf"))),torch.tensor([0.,-0.,-float("inf"),-1e30],device="cuda")))
    scores=values.repeat((128*64+values.numel()-1)//values.numel())[:128*64].view(1,128,8,8).contiguous()
    history=torch.zeros(128,2,device="cuda");scales=torch.ones(1,device="cuda")
    old,old_stats=ext.check(scores,scales,history,0)
    actual,actual_stats=ext.check(scores,scales,history,mode)
    torch.testing.assert_close(actual,old,rtol=0,atol=0)
    torch.testing.assert_close(actual_stats,old_stats,rtol=0,atol=0)
    # Independent FP64 emulation of FP32 fused arithmetic and RNE.
    encoded=(scores.double()*11.541560173034668+119.6500015258789).float().round().clamp(0,120).byte()
    encoded=encoded.view(1,128,4,2,8)[...,[0,1,4,5,2,3,6,7]].reshape(1,128,64)
    torch.testing.assert_close(actual.view(torch.uint8),encoded,rtol=0,atol=0)


@pytest.mark.parametrize("iterations",[1,3,128])
def test_micro_checksums_all_candidates_exact(iterations):
    for group,count in ((0,5),(1,3)):
        outputs=[]
        for mode in range(count):
            output=torch.empty(3,128,device="cuda");ext.run(output,group,mode,iterations);outputs.append(output)
        for output in outputs[1:]: torch.testing.assert_close(output,outputs[0],rtol=0,atol=0)


def test_direct_decode_all_byte_pairs_and_nondefault_stream():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ids=torch.arange(121*121,device="cuda")
        codes=torch.stack((ids%121,ids//121),-1).byte()
        actual=ext.decode(codes)
        expected=codes.view(torch.float8_e4m3fn).float()
    stream.synchronize()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_report_keeps_resources_and_nonadditive_scope():
    report=bench.benchmark(ext,SimpleNamespace(blocks=2,iterations=3,trials=3))
    assert "not additive" in report["scope"] and len(report["trial_order"])==3
    for group in report["groups"].values():
        for row in group["variants"].values():
            assert len(row["trials_ms"])==3 and row["median_ms"]>0
            assert row["registers_per_thread"]>0


def test_invalid_probe_domains_and_warp_shape_rejected():
    with pytest.raises(RuntimeError): ext.decode(torch.tensor([121],device="cuda",dtype=torch.uint8))
    for rows in (31,33):
        with pytest.raises(RuntimeError): ext.check(torch.zeros(1,rows,8,8,device="cuda"),torch.ones(1,device="cuda"),torch.zeros(rows,2,device="cuda"),0)
    for mode in (-1,5):
        with pytest.raises(RuntimeError): ext.run(torch.empty(1,128,device="cuda"),0,mode,1)
