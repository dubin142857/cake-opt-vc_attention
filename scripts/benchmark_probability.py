"""vc17 probability candidates on identical inputs; NOT attention latency."""
import argparse
import importlib
import json
import os
from pathlib import Path
import random
import statistics

import torch

MODES=("vc16", "vc17_g1", "vc17_g4", "vc17_c4", "vc17_direct")
GROUPS={"probability":MODES, "decode_sum":("half2_two_chains","half2_four_chains","direct_two_chains"),
        "encode":("vc15_fused_quad",)}


def load_extension():
    try:
        ext=importlib.import_module("_vc_probability_bench")
        if getattr(ext,"probability_api_version",0)==1: return ext
    except ImportError:
        pass
    from torch.utils.cpp_extension import load
    major,minor=torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"]=f"{major}.{minor}"
    os.environ.setdefault("MAX_JOBS","2")
    return load(name="_vc_probability_bench_v1",sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=["-O3","--use_fast_math","--expt-relaxed-constexpr"],verbose=True)


@torch.no_grad()
def validate_candidates(ext,device="cuda"):
    codes=torch.arange(121,device=device,dtype=torch.uint8)
    torch.testing.assert_close(ext.decode(codes),codes.view(torch.float8_e4m3fn).float(),rtol=0,atol=0)
    generator=torch.Generator(device=device).manual_seed(1717)
    cases=0;values=0
    for steps,rows,extreme in ((1,32,False),(7,160,False),(300,128,False),(9,128,True)):
        scores=torch.randn(steps,rows,8,8,device=device,generator=generator)*20
        scales=torch.linspace(.001,.25,steps,device=device)
        history=torch.full((rows,2),-5e6,device=device)
        # All-masked local threads alongside valid lanes, plus an all-masked
        # later tile after a valid prefix; no tile with zero global mass first.
        scores[:,::4,6:]=-float("inf")
        if steps>1: scores[-1]=-float("inf")
        if extreme:
            scores[:-1]*=100
            scales[:3]=torch.tensor([1e-39,.125,10.],device=device)
            history[::4]=300.
        baseline,stats=ext.check(scores,scales,history,0)
        decoded=baseline.contiguous().view(torch.uint8).view(torch.float8_e4m3fn).float()
        decoded=decoded.view(steps,rows,4,2,2,4)
        expected=decoded.double().sum(dim=(2,3,5)).float()
        torch.testing.assert_close(stats[...,:2],expected,rtol=0,atol=0)
        for mode in range(1,len(MODES)):
            actual,actual_stats=ext.check(scores,scales,history,mode)
            torch.testing.assert_close(actual,baseline,rtol=0,atol=0)
            torch.testing.assert_close(actual_stats,stats,rtol=0,atol=0)
            cases+=1
        values+=scores.numel()
    return {"status":"passed","direct_decode_codes":121,"candidate_history_cases":cases,
            "score_values":values,"p_bytes_tile_mass_max_alpha":"exact match to vc16",
            "tile_mass_oracle":"FP64 sum of independently decoded P bytes",
            "scope":"Actual CUDA helpers, not full Hopper attention or model quality"}


@torch.no_grad()
def benchmark(ext,args,device="cuda"):
    validation=validate_candidates(ext,device)
    output=torch.empty(args.blocks,128,device=device)
    jobs=[(group,mode) for group,names in enumerate(GROUPS.values()) for mode in range(len(names))]
    times={job:[] for job in jobs};resources={};references={}
    for group,mode in jobs:
        for _ in range(3): ext.run(output,group,mode,args.iterations)
        actual=output.clone()
        # The encoding checksum is an opaque uint32 bit pattern, not a float.
        if group!=2 and not torch.isfinite(actual).all().item(): raise RuntimeError("Nonfinite probability checksum")
        if mode==0: references[group]=actual
        else: torch.testing.assert_close(actual,references[group],rtol=0,atol=0)
        regs,local=ext.resources(group,mode)
        resources[group,mode]={"registers_per_thread":regs,"local_bytes_per_thread":local}
    torch.cuda.synchronize(device)
    events=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    rng=random.Random(1717);orders=[]
    for _ in range(args.trials):
        order=list(jobs);rng.shuffle(order);orders.append(order)
        for group,mode in order:
            events[0].record();ext.run(output,group,mode,args.iterations)
            events[1].record();events[1].synchronize()
            times[group,mode].append(events[0].elapsed_time(events[1]))
    groups={}
    print("\nProbability candidates: microbenchmark launch ms (NOT additive, NOT attention speedup)")
    print(f"{'group':<14} {'mode':<23} {'median ms':>11} {'base/new':>10} {'regs':>6} {'local B':>9}")
    for group,(name,modes) in enumerate(GROUPS.items()):
        rows={};baseline=statistics.median(times[group,0])
        for mode,label in enumerate(modes):
            median=statistics.median(times[group,mode])
            row={"median_ms":median,"trials_ms":times[group,mode],"baseline_over_variant":baseline/median,**resources[group,mode]}
            rows[label]=row
            print(f"{name:<14} {label:<23} {median:>11.6f} {baseline/median:>9.3f}x {row['registers_per_thread']:>6} {row['local_bytes_per_thread']:>9}")
        groups[name]={"baseline":modes[0],"variants":rows}
    return {"validation":validation,"groups":groups,"trial_order":orders,
            "gpu":torch.cuda.get_device_name(device),"torch":torch.__version__,"cuda":torch.version.cuda,
            "arguments":vars(args),"scope":"Independent register-input microbenchmarks with common checksum work. Probability group uses actual production helpers including maximum/history. Decode group is standalone and has different scheduling. No QK/PV, no preprocessing; group times are not additive or attention percentages."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks",type=int,default=256)
    parser.add_argument("--iterations",type=int,default=128)
    parser.add_argument("--trials",type=int,default=9)
    parser.add_argument("--output",default="results/probability_candidates.json")
    args=parser.parse_args()
    if not 1<=args.blocks<=65535 or not 1<=args.iterations<=100000 or args.trials<1:
        parser.error("blocks=1..65535, iterations=1..100000, trials>=1 required")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    report=benchmark(load_extension(),args)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Report: {path}")
    print("Full-kernel check: profile_sage_vc.py --expcast-only --compare-probability-candidates")


if __name__=="__main__": main()
