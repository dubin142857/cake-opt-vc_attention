"""Isolated CUDA probability-stage benchmark; does NOT measure PV overlap.

JIT builds the repository's real helpers using nvcc. No full attention, QK/PV,
preprocessing or accuracy claim. Use profile_sage_vc.py for the real speedup.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import random
import statistics

import torch


def load_extension():
    try:
        extension = importlib.import_module("_vc_expcast_bench")
        if getattr(extension,"benchmark_api_version",0) == 5:
            return extension
    except ImportError:
        pass
    from torch.utils.cpp_extension import load
    major,minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    os.environ.setdefault("MAX_JOBS","2")
    # New JIT name avoids reusing the old imported DLL in the same process.
    extension = load(name="_vc_expcast_bench_v5",sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=["-O3","--use_fast_math","--expt-relaxed-constexpr"],verbose=True)
    if extension.benchmark_api_version != 5:
        raise RuntimeError("Stale ExpCast benchmark extension")
    return extension


@torch.no_grad()
def validate_components(extension,device="cuda"):
    # FP64 emulation of FP32 FMA, including all integer-rounding boundaries.
    boundaries=((torch.arange(120,device=device,dtype=torch.float64)+.5-119.6500015258789)/11.541560173034668).float()
    x=torch.cat((torch.linspace(-16,0,65536,device=device),boundaries,
        torch.nextafter(boundaries,torch.full_like(boundaries,-float("inf"))),
        torch.nextafter(boundaries,torch.full_like(boundaries,float("inf"))),
        torch.tensor([-float("inf"),-1e30,-0.,0.],device=device)))
    reference=(x.double()*11.541560173034668+119.6500015258789).float().round().clamp(0,120).byte()
    for mode in (0,1):
        torch.testing.assert_close(extension.encode_check(x,mode),reference,rtol=0,atol=0)
    ordinary=torch.tensor([-float("inf"),0.,-0.6931471805599453,-1.3862943611198906,
                           -2.0794415416798357,-5.,-100.,-0.],device=device)
    expected=(ordinary.double().exp()*256).float().to(torch.float8_e4m3fn).view(torch.uint8)
    torch.testing.assert_close(extension.encode_check(ordinary,2),expected,rtol=0,atol=0)
    ids=torch.arange(121*121,device=device)
    pairs=torch.stack((ids%121,ids//121),-1).byte().repeat(1,16)
    generator=torch.Generator(device=device).manual_seed(1910)
    random_codes=torch.randint(0,121,(4096,32),device=device,dtype=torch.uint8,generator=generator)
    uniform=torch.arange(121,device=device,dtype=torch.uint8).view(-1,1).expand(-1,32).contiguous()
    codes=torch.cat((pairs,random_codes,uniform))
    expected=codes.view(torch.float8_e4m3fn).float().double().sum(-1).float()
    actual=extension.decode_check(codes)
    torch.testing.assert_close(actual,expected[:,None].expand_as(actual),rtol=0,atol=0)
    torch.cuda.synchronize(device)
    return {"status":"passed","encoding_values_per_mode":x.numel(),"fp32_fma_rne_boundaries":120,
            "ordinary_exp2_known_values":ordinary.numel(),"decode_rows":codes.shape[0],
            "decode_values_per_row":32,"integer_sum":"bitwise equal to FP64-decoded FP8 tile sums",
            "scope":"diagnostic candidates only; no full attention execution"}


COMPONENT_NAMES = {
    "encode":("expcast_u8","expcast_u32","exp2_fp8_scale256"),
    "decode_sum":("fp32_serial","fp32_two_chains","integer_exact"),
}
COMPONENT_BASELINES = {"encode":0,"decode_sum":1}


@torch.no_grad()
def validate_affine(extension,device="cuda"):
    scales=torch.tensor([2**-20,1e-4,.01,.137,.25,1.,4.,4096.],device=device)
    maxima=torch.tensor([-127.,-16.,-1.,0.,3.,16.,127.],device=device)
    scale,maximum=torch.meshgrid(scales,maxima,indexing="ij")
    scale,maximum=scale.reshape(-1,1),maximum.reshape(-1,1)
    generator=torch.Generator(device=device).manual_seed(2212)
    shifts=-torch.rand(56,4096,device=device,generator=generator)*14-.0001
    scores=((shifts.double()+maximum.double())/scale.double()).float()
    gain=(scale*11.541560173034668).expand(-1,2048).contiguous()
    bias=(-maximum.double()*11.541560173034668+119.6500015258789).float().expand_as(gain).contiguous()
    actual=extension.affine_check(scores,gain,bias)
    paired=extension.affine_check(scores,gain,bias,True)
    torch.testing.assert_close(paired,actual,rtol=0,atol=0)
    expected=(scores.double()*gain.repeat_interleave(2,-1).double()+bias.repeat_interleave(2,-1).double()).float().round().clamp(0,255).byte()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    shifted=scores*scale-maximum
    legacy=(shifted.double()*11.541560173034668+119.6500015258789).float().round().clamp(0,120).byte()
    difference=(actual.int()-legacy.int()).abs()
    if actual.max().item()>120 or difference.max().item()>1:
        raise AssertionError("Folded encoder exceeded bounded code domain/difference")
    decoded=actual.view(torch.float8_e4m3fn).float()
    previous=legacy.view(torch.float8_e4m3fn).float()
    return {"status":"passed","values":scores.numel(),"independent_fp64_fma_oracle_exact":True,
            "vc11_byte_mismatch_percent":100*(actual!=legacy).float().mean().item(),
            "maximum_code_difference":difference.max().item(),
            "decoded_probability_rel_rmse_percent":100*((decoded-previous).square().mean()/previous.square().mean()).sqrt().item(),
            "scope":"moderate-range encoding diagnostics only; FP8 boundary rounding may change; not attention/model quality"}


@torch.no_grad()
def benchmark_components(extension,args,output):
    validation=validate_components(extension,output.device)
    active={name:list(range(3)) for name in COMPONENT_NAMES}
    if torch.cuda.get_device_capability(output.device)<(8,9):
        active["encode"]=[0,1]
    jobs=[(group,index) for group,(name,indices) in enumerate(active.items()) for index in indices]
    times={job:[] for job in jobs}; resources={}
    for group,index in jobs:
        key=list(COMPONENT_NAMES)[group],COMPONENT_NAMES[list(COMPONENT_NAMES)[group]][index]
        regs,local_bytes=extension.component_resources(group,index)
        resources[key]={"registers_per_thread":regs,"local_bytes_per_thread":local_bytes}
        for _ in range(3):extension.component_run(output,group,index,args.iterations)
    torch.cuda.synchronize(output.device)
    # All decode candidates and both ExpCast encoders consume identical inputs.
    # Encoding checksums are raw uint32 bit patterns, not finite float values.
    for group in (0,1):
        outputs=[]
        for index in ((0,1) if group==0 else (0,1,2)):
            extension.component_run(output,group,index,args.iterations)
            outputs.append(output.clone())
        for actual in outputs[1:]:
            if not torch.equal(actual.view(torch.int32),outputs[0].view(torch.int32)):
                raise AssertionError("Candidate benchmark checksums differ")
    torch.cuda.synchronize(output.device)
    events=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    rng=random.Random(1910); orders=[]
    for _ in range(args.trials):
        order=list(jobs);rng.shuffle(order);orders.append(order)
        for group,index in order:
            events[0].record()
            extension.component_run(output,group,index,args.iterations)
            events[1].record();events[1].synchronize()
            times[group,index].append(events[0].elapsed_time(events[1]))
    groups={}
    print("\nIsolated components (ms per whole microbenchmark launch; NOT additive):")
    print(f"{'component':<14} {'variant':<22} {'median ms':>12} {'baseline/new':>14} {'regs':>6} {'local B':>9}")
    for group,(name,names) in enumerate(COMPONENT_NAMES.items()):
        baseline_index=COMPONENT_BASELINES[name]
        baseline=statistics.median(times[group,baseline_index]);rows={}
        for index in active[name]:
            values=times[group,index];median=statistics.median(values);resource=resources[name,names[index]]
            rows[names[index]]={"median_ms":median,"trials_ms":values,"baseline_over_variant":baseline/median,**resource}
            print(f"{name:<14} {names[index]:<22} {median:>12.6f} {baseline/median:>13.3f}x {resource['registers_per_thread']:>6} {resource['local_bytes_per_thread']:>9}")
        groups[name]={"baseline":names[baseline_index],"variants":rows}
    return {"validation":validation,"groups":groups,"trial_order":orders,
            "skipped_modes":[] if 2 in active["encode"] else ["exp2_fp8_scale256: native FP8 conversion needs SM89+"],
            "scope":"synthetic register inputs, 64 values/thread/iteration, identical common input/checksum work within each group; no rowmax, history rescale, QK/PV, preprocessing or attention speedup",
            "baselines":"current vc10 u8 encoding and two-chain standalone tile sum; serial sum is a diagnostic comparison, not the vc9 historical denominator update",
            "ordinary_encoding":"exp2_fp8_scale256 uses the same 256 scale as ExpCast, not Sage's 2^8.807 scale; outputs need not match ExpCast",
            "timing_caveat":"component times are not additive or percentages of the full helper/kernel; independent compilation changes scheduling/register use",
            "integer_candidate":"exact tile sum for 32 codes in [0,120]; experimental, not used by production attention"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks",type=int,default=256)
    parser.add_argument("--iterations",type=int,default=128)
    parser.add_argument("--trials",type=int,default=9)
    parser.add_argument("--stages",action="store_true",help="Also validate and time encoding and decode+sum candidates")
    parser.add_argument("--output",default="results/expcast_micro.json")
    args = parser.parse_args()
    if not 1 <= args.blocks <= 65535 or not 1 <= args.iterations <= 100000 or args.trials < 1:
        parser.error("blocks=1..65535, iterations=1..100000, trials>=1 required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    extension=load_extension()
    output = torch.empty(args.blocks,128,device="cuda",dtype=torch.float32)
    affine_validation=validate_affine(extension,output.device)
    modes = ("sage_softmax","vc9_expcast","vc10_expcast","vc11_expcast","vc12_expcast","vc14_expcast")
    # Sage's inline FP32->E4M3 pack emits a trap below SM89. The ExpCast
    # helpers support SM86 through CUDA's intrinsic decode implementation.
    active = [0,1,2,3,4,5] if torch.cuda.get_device_capability() >= (8,9) else [1,2,3,4,5]
    resources = {}
    for index in active:
        regs,local_bytes = extension.resources(index)
        resources[modes[index]] = {"registers_per_thread":regs,"local_bytes_per_thread":local_bytes}
    for _ in range(3):
        for index in active: extension.run(output,index,args.iterations)
    torch.cuda.synchronize()
    extension.run(output,2,args.iterations)
    vc10_checksum=output.clone()
    extension.run(output,3,args.iterations)
    torch.testing.assert_close(output,vc10_checksum,rtol=0,atol=0)
    extension.run(output,4,args.iterations)
    vc12_checksum=output.clone()
    extension.run(output,5,args.iterations)
    torch.testing.assert_close(output,vc12_checksum,rtol=0,atol=0)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    rng,orders,times = random.Random(710),[],{modes[index]:[] for index in active}
    for trial in range(args.trials):
        order = list(active)
        rng.shuffle(order)
        orders.append([modes[index] for index in order])
        for index in order:
            events[0].record()
            extension.run(output,index,args.iterations)
            events[1].record()
            events[1].synchronize()
            times[modes[index]].append(events[0].elapsed_time(events[1]))
            if not torch.isfinite(output).all().item():
                raise RuntimeError(f"Nonfinite checksum: {modes[index]}")
    report = {"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"torch_cuda":torch.version.cuda,
        "arguments":vars(args),"trial_order":orders,"register_target":168,"compiled_resources":resources,
        "skipped_modes":[] if 0 in active else ["sage_softmax requires SM89+"],
        "scope":"probability helpers with common register-input/checksum work; no QK/PV or preprocessing; cannot measure PV overlap",
        "modes":{name:{"median_ms":statistics.median(values),"trials_ms":values} for name,values in times.items()}}
    report["vc9_over_vc10"] = report["modes"]["vc9_expcast"]["median_ms"]/report["modes"]["vc10_expcast"]["median_ms"]
    report["vc10_over_vc11"] = report["modes"]["vc10_expcast"]["median_ms"]/report["modes"]["vc11_expcast"]["median_ms"]
    report["vc10_vc11_checksum_exact_match"] = True
    report["vc11_over_vc12"] = report["modes"]["vc11_expcast"]["median_ms"]/report["modes"]["vc12_expcast"]["median_ms"]
    report["vc12_over_vc14"] = report["modes"]["vc12_expcast"]["median_ms"]/report["modes"]["vc14_expcast"]["median_ms"]
    report["vc12_vc14_checksum_exact_match"] = True
    report["affine_validation"]=affine_validation
    print("\nComplete probability helpers (no QK/PV or overlap; NOT attention latency):")
    print(f"{'variant':<20} {'median ms':>12} {'vc12/new':>12}")
    baseline=report["modes"]["vc12_expcast"]["median_ms"]
    for name,row in report["modes"].items():
        print(f"{name:<20} {row['median_ms']:>12.6f} {baseline/row['median_ms']:>11.3f}x")
    if args.stages:
        report["components"]=benchmark_components(extension,args,output)
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    print("This is NOT an attention speedup. Validate with profile_sage_vc.py --expcast-only --compare-kernels --kernel-baseline vc12.")


if __name__ == "__main__":
    main()
