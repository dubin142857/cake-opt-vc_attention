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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks",type=int,default=256)
    parser.add_argument("--iterations",type=int,default=128)
    parser.add_argument("--trials",type=int,default=9)
    parser.add_argument("--output",default="results/expcast_micro.json")
    args = parser.parse_args()
    if not 1 <= args.blocks <= 65535 or not 1 <= args.iterations <= 100000 or args.trials < 1:
        parser.error("blocks=1..65535, iterations=1..100000, trials>=1 required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    try:
        extension = importlib.import_module("_vc_expcast_bench")
    except ImportError:
        from torch.utils.cpp_extension import load
        major,minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        os.environ.setdefault("MAX_JOBS","2")
        extension = load(name="_vc_expcast_bench",sources=[str(Path(__file__).with_suffix(".cu"))],
            extra_cuda_cflags=["-O3","--use_fast_math","--expt-relaxed-constexpr"],verbose=True)
    output = torch.empty(args.blocks,128,device="cuda",dtype=torch.float32)
    modes = ("sage_softmax","vc9_expcast","vc10_expcast")
    # Sage's inline FP32->E4M3 pack emits a trap below SM89. The ExpCast
    # helpers support SM86 through CUDA's intrinsic decode implementation.
    active = [0,1,2] if torch.cuda.get_device_capability() >= (8,9) else [1,2]
    resources = {}
    for index in active:
        regs,local_bytes = extension.resources(index)
        resources[modes[index]] = {"registers_per_thread":regs,"local_bytes_per_thread":local_bytes}
    for _ in range(3):
        for index in active: extension.run(output,index,args.iterations)
    torch.cuda.synchronize()
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
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    print("This is NOT an attention speedup. Validate with profile_sage_vc.py --compare-kernels --kernel-baseline vc9.")


if __name__ == "__main__":
    main()
