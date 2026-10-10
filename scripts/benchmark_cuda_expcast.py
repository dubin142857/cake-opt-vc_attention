"""H200 vc28 CUDA candidates; INT8 QK, FP8 PV, V smooth OFF."""
import argparse
import json
from pathlib import Path
import random
import math
import statistics
from types import SimpleNamespace
import torch

from benchmark_cuda import measure_trial,query_indices,fp32_reference,output_hnd
from benchmark_paper_expcast import uniform_tile_inputs
from profile_int8_h200 import load_inputs,metrics,make_flash4
from profile_sage_vc import load_sage_core_ops,sage_core_calls,stats
from kernel_timing import profile_kernels,run_window
import split_mass_native as native
import balanced_max_native
import fused_quad_native
import scale_prefetch_native
import pv_prefix_native

VARIANTS={"vc28_jit":False,"vc28_split_mass":True}
EXPERIMENTS={
    "pv_prefix":(pv_prefix_native,"vc28_prefix_overlap",
        "Scale-prefetch in every JIT arm. Encode all P, defer only last K32 decode/FP32 sum until first three PV operations are pending; serial-prefix control isolates overlap. Same P bytes, mass addition order and two-level PV accumulation; no shared copy."),
    "scale_prefetch":(scale_prefetch_native,"vc28_scale_prefetch",
        "Original vc28 kernel template as baseline. Load the NEXT K scale during current PV and carry one FP32 scalar. Same QK/PV, probability bytes, FP32 mass/history, per-thread scales and tail masking; no additional shared buffers."),
    "fused_quad":(fused_quad_native,"vc28_fused_quad",
        "Balanced max in BOTH arms. Fuse four-byte encoding, FP8 decode and FP32 pair sums; seed chains with the first pair instead of adding zero. Same P rounding, sum order, fallback, tail and PV/history."),
    "balanced_max":(balanced_max_native,"vc28_balanced_max",
        "Only full-tile INT32 local row-max tree: six -> four max dependency levels; same 15 max3 + one max2 per row. Encoding, mass, warp max, fallback, tail and PV/history unchanged."),
    "split_mass":(native,"vc28_split_mass",
        "Only full-tile FP32 decoded-P sum dependencies: two pair chains -> four component chains. Same FP8 bytes, max/alpha, fallback, tail and PV/history accumulation."),
}


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment",choices=tuple(EXPERIMENTS),default="balanced_max")
    for name,default in (("batch",1),("heads",28),("queries",37741),("tokens",37741),
                         ("dim",128),("trials",7),("repeats",10),("seed",0),
                         ("check-queries",128),("kernel-trials",3)):
        p.add_argument("--"+name,type=int,default=default)
    p.add_argument("--dtype",choices=("float16","bfloat16"),default="bfloat16")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--qkv-file",type=Path)
    p.add_argument("--qkv-layout",choices=("NHD","HND"),default="NHD")
    p.add_argument("--validate-only",action="store_true")
    p.add_argument("--skip-flash4",action="store_true")
    p.add_argument("--skip-cupti",action="store_true")
    p.add_argument("--kernel-warmup-seconds",type=float,default=2.)
    p.add_argument("--kernel-seconds",type=float,default=1.)
    p.add_argument("--trace-dir",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args(argv)
    if any(getattr(a,n)<1 for n in ("batch","heads","queries","tokens","trials","repeats","kernel_trials")) or a.dim not in (64,128):
        p.error("Positive dimensions/counts and D64/128 required")
    if a.check_queries<0 or not all(0<x<float("inf") for x in (a.kernel_seconds,a.kernel_warmup_seconds)):
        p.error("Invalid reference count or timing window")
    if a.trace_dir is None:a.trace_dir=Path(f"results/cuda_expcast_{a.experiment}_traces")
    if a.output is None:a.output=Path(f"results/h200_cuda_expcast_{a.experiment}.json")
    return a


def variant_modes(backend,candidate,baseline):
    return {baseline:False,candidate:True,**getattr(backend,"EXTRA_VARIANTS",{})}


def run_variant(backend,ext,state,mode):
    return backend.forward(ext,state,mode) if type(mode) is bool else backend.forward_mode(ext,state,mode)


def make_calls(ext,vc,qkv,state,backend=native,candidate="vc28_split_mass",baseline="vc28_jit"):
    variants=variant_modes(backend,candidate,baseline)
    core={name:lambda split=split:run_variant(backend,ext,state,split) for name,split in variants.items()}
    full={name:lambda split=split:run_variant(backend,ext,vc.prepare(*qkv,expcast=True,v_smooth=False),split)
          for name,split in variants.items()}
    core["vc28_installed"]=lambda:vc.forward_prepared(state,kernel_mode="vc28_stream_i2f")
    full["vc28_installed"]=lambda:vc.sageattn_vc(*qkv,tensor_layout="HND",expcast=True,v_smooth=False,kernel_mode="vc28_stream_i2f")
    return core,full


def paired_timings(baseline,candidate):
    if not baseline or len(baseline)!=len(candidate):
        raise ValueError("Matched nonempty timing trials required")
    if not all(math.isfinite(x) and x>0 for x in [*baseline,*candidate]):
        raise ValueError("Finite positive timing samples required")
    saved=[old-new for old,new in zip(baseline,candidate)]
    return {"paired_saved_ms":saved,"median_saved_ms":statistics.median(saved),
            "min_saved_ms":min(saved),"max_saved_ms":max(saved),
            "faster_trials":sum(x>0 for x in saved),"trials":len(saved),
            "note":"Matched trial indices with randomized within-trial order; descriptive, not a significance test."}


@torch.no_grad()
def validate(ext,vc,reference,device,progress,backend=native,candidate="vc28_split_mass",baseline="vc28_jit"):
    rows=[]
    gen=torch.Generator(device=device).manual_seed(3400)
    for dtype in (torch.float16,torch.bfloat16):
        for d in (64,128):
            cases=[("random",m,n) for m,n in ((1,1),(65,128),(65,129),(193,513))]
            cases += [("uniform",65,n) for n in (128,129,257)]
            for pattern,m,n in cases:
                if pattern=="uniform":qkv=uniform_tile_inputs(dtype,d,n,device)
                else:qkv=tuple(torch.randn(1,2,length,d,device=device,dtype=dtype,generator=gen) for length in (m,n,n))
                state=vc.prepare(*qkv,expcast=True,v_smooth=False)
                old=vc.forward_prepared(state,kernel_mode="vc28_stream_i2f")
                oracle=reference.reference(state,chunk=32).to(dtype)
                row=dict(dtype=str(dtype),dim=d,queries=m,keys=n,pattern=pattern,variants={})
                rows.append(row);progress(rows)
                outputs={}
                for name,split in variant_modes(backend,candidate,baseline).items():
                    print("Validate",name,str(dtype),d,m,n,pattern,flush=True)
                    row["variants"][name]={"status":"running"};progress(rows)
                    out=run_variant(backend,ext,state,split)
                    result={"vs_installed":metrics(out,old,limit=.001),
                            "vs_decoded_reference":metrics(out,oracle,limit=.01)}
                    if split:
                        result["vs_jit_baseline"]=metrics(out,outputs[baseline],limit=.001)
                        result["exact_vs_jit_baseline"]=torch.equal(out,outputs[baseline])
                        if getattr(backend,"EXACT_OUTPUT",False):
                            from validate_scale_prefetch import assert_exact
                            result["bitwise_gate"]=assert_exact(out,outputs[baseline])
                    if pattern=="uniform":
                        perm=reference.value_permutation(state["v"].shape[-1],device)
                        values=state["v"].float().index_select(-1,perm)[...,:n]
                        mean=(values.mean(-1)*state["vs"]).unsqueeze(2).expand_as(out).to(dtype)
                        torch.testing.assert_close(oracle,mean,rtol=.001,atol=.001)
                    stream=torch.cuda.Stream(device=device);stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):side=run_variant(backend,ext,state,split)
                    stream.synchronize();torch.testing.assert_close(side,out,rtol=0,atol=0)
                    outputs[name]=out;result["status"]="passed"
                    row["variants"][name]=result;progress(rows)
    return rows


@torch.no_grad()
def main():
    a=parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);a.trace_dir.mkdir(parents=True,exist_ok=True)
    backend,candidate,change=EXPERIMENTS[a.experiment]
    baseline_name=getattr(backend,"BASELINE","vc28_balanced_max" if a.experiment=="fused_quad" else "vc28_jit")
    report={"status":"starting","contract":"vc28 folded ExpCast; INT8 QK; FP8 PV; V smooth OFF",
            "experiment":a.experiment,"change":change,"baseline":baseline_name,
            "gates":"1% decoded reference; 0.1% ALL-row same-algorithm regression. Implementation, not model quality."}
    def save():a.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    def update(k,v):report[k]=v;save()
    try:
        device=torch.device(a.device);torch.cuda.set_device(device)
        if torch.cuda.get_device_capability(device)!=(9,0):raise RuntimeError("Hopper required")
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(a.seed)
        from sageattention_local import vc,vc_reference,sageattn
        report["environment"]={"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(device),
                               "native_version":vc._extension().sage_vc_kernel_version}
        ext,report["build"]=backend.load_native();save()
        if a.experiment in ("scale_prefetch","pv_prefix"):
            report["build_audit"]=backend.build_audit(ext,vc,report["build"],a.dim,0 if a.dtype=="float16" else 1);save()
            from validate_scale_prefetch import validate as validate_scale
            report["scale_validation"]=validate_scale(ext,backend,vc,vc_reference,device,lambda r:update("scale_validation",r))
        if a.experiment in ("fused_quad","pv_prefix"):
            from validate_fused_quad_math import validate as validate_math
            report["probability_validation"]=validate_math(ext,device,lambda r:update("probability_validation",r),label=a.experiment)
        if a.experiment=="pv_prefix":
            report["pipeline_validation"]=backend.validate_pipeline(ext,vc,vc_reference,device,lambda r:update("pipeline_validation",r))
        report["validation"]=validate(ext,vc,vc_reference,device,lambda r:update("validation",r),backend,candidate,baseline_name)
        if a.validate_only:
            report["status"]="validation_passed";save();print("Validation passed:",a.output);return
        qkv,report["inputs"]=load_inputs(a,device)
        report["args"]={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        state=vc.prepare(*qkv,expcast=True,v_smooth=False)
        core,full=make_calls(ext,vc,qkv,state,backend,candidate,baseline_name)
        sage_native,_=load_sage_core_ops()
        core["sage_cpp"]=sage_core_calls(SimpleNamespace(vc=vc),state,sage_native,sage_native)["sage_cpp"]
        full["sage_cpp"]=lambda:sageattn(*qkv,tensor_layout="HND",is_causal=False)
        if not a.skip_flash4:
            fa=make_flash4(qkv);full["flash4"]=fa.call;report["flash4"]=fa.info
        ids=query_indices(a.queries,a.check_queries);index=torch.tensor(ids,device=device)
        golden=fp32_reference(*qkv,ids,32)
        oracle=vc_reference.reference(state,query_indices=ids,chunk=32).to(qkv[0].dtype)
        baseline=core[baseline_name]();installed=core["vc28_installed"]()
        report["accuracy"]={};report["checked_queries"]=len(ids)
        for name,call in full.items():
            print("Check main shape",name,flush=True)
            out=output_hnd(call(),"BSHD" if name=="flash4" else "BHND",qkv[0].shape,qkv[0].dtype)
            row={"dense_fp32":metrics(out.index_select(2,index).float().cpu(),golden)}
            if name in core:torch.testing.assert_close(out,core[name](),rtol=0,atol=0)
            if name.startswith("vc28"):
                row.update(vs_jit_baseline=metrics(out,baseline,limit=.001),exact_vs_jit_baseline=torch.equal(out,baseline),
                           vs_installed=metrics(out,installed,limit=.001),
                           decoded_reference=metrics(out.index_select(2,index),oracle,limit=.01))
                if name in variant_modes(backend,candidate,baseline_name) and getattr(backend,"EXACT_OUTPUT",False):
                    from validate_scale_prefetch import assert_exact
                    row["bitwise_gate"]=assert_exact(out,baseline)
            report["accuracy"][name]=row;save()
        raw=ext.resources(a.dim,0 if a.dtype=="float16" else 1)
        report["compiled_resources"]={baseline_name:raw["expcast_plain"],candidate:raw["expcast_probe"]}
        if a.experiment=="pv_prefix":report["compiled_resources"]["vc28_prefix_serial"]=raw["prefix_serial"]
        samples={"full":{n:[] for n in full},"core":{n:[] for n in core}}
        for call in full.values():run_window(call,a.kernel_warmup_seconds,device)
        rng=random.Random(a.seed+3400);report["trial_order"]=[]
        for trial in range(a.trials):
            order=[(scope,n) for scope,group in samples.items() for n in group];rng.shuffle(order)
            report["trial_order"].append(order)
            for scope,name in order:
                _,ms=measure_trial((full if scope=="full" else core)[name],a.repeats,device)
                samples[scope][name].append(ms)
            update("api",{scope:{n:stats(v) for n,v in group.items()} for scope,group in samples.items()})
            print(f"Trial {trial+1}/{a.trials} complete",flush=True)
        print("Uninstrumented synchronized API ms; full includes fresh preprocessing:")
        print(f"{'variant':<24} {'full API':>11} {'fixed core':>11} {'Sage/API':>10} {'FA4/API':>10} {'FP32 RMSE %':>13}")
        for name in full:
            ms=report["api"]["full"][name]["median_ms"]
            c=f"{report['api']['core'][name]['median_ms']:.4f}" if name in core else "--"
            sage=report["api"]["full"]["sage_cpp"]["median_ms"]/ms
            fa=f"{report['api']['full']['flash4']['median_ms']/ms:.3f}x" if "flash4" in full else "--"
            print(f"{name:<24} {ms:>11.4f} {c:>11} {sage:>9.3f}x {fa:>10} {100*report['accuracy'][name]['dense_fp32']['relative_rmse']:>13.6f}")
        print("Ratios use FULL API only; fixed core includes allocation/dispatch and excludes preprocessing.")
        comparison={scope+"_old_over_new":report["api"][scope][baseline_name]["median_ms"]/report["api"][scope][candidate]["median_ms"] for scope in ("full","core")}
        comparison.update(report["accuracy"][candidate]["vs_jit_baseline"])
        comparison["exact_match"]=report["accuracy"][candidate]["exact_vs_jit_baseline"]
        update("comparison",comparison)
        paired={scope:{"jit_vs_candidate":paired_timings(group[baseline_name],group[candidate]),
                       "installed_vs_jit":paired_timings(group["vc28_installed"],group[baseline_name]),
                       "installed_vs_candidate":paired_timings(group["vc28_installed"],group[candidate])}
                for scope,group in samples.items()}
        for scope,group in samples.items():
            for name in getattr(backend,"EXTRA_VARIANTS",{}):
                paired[scope][name+"_vs_candidate"]=paired_timings(group[name],group[candidate])
        update("paired_comparisons",paired)
        print(f"Same-payload {baseline_name} / {candidate} (>1 faster); ALL output rows:",json.dumps(comparison))
        for scope,group in paired.items():
            for name,row in group.items():
                print(f"Paired {scope} {name}: saved {row['median_saved_ms']:+.4f} ms; "
                      f"faster {row['faster_trials']}/{row['trials']} trials (descriptive only)")
        print("Compiled resources:",json.dumps(report["compiled_resources"]))
        if not a.skip_cupti:
            report["kernel_windows"]=profile_kernels({**core,**({"flash4":full["flash4"]} if "flash4" in full else {})},a,device,on_result=lambda r:update("kernel_windows",r))
        report["status"]="passed";save();print("Saved:",a.output)
    except Exception as e:
        report.update(status="failed",error=f"{type(e).__name__}: {e}");save();raise


if __name__=="__main__":main()
