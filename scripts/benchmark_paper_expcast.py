"""Paper ExpCast: identical logical payloads, natural V for TileLang on H200."""
import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace
import torch

from benchmark_cuda import measure_trial,query_indices,fp32_reference,output_hnd
from profile_int8_h200 import load_inputs,metrics,make_flash4
from profile_sage_vc import load_sage_core_ops,sage_core_calls,stats
from kernel_timing import profile_kernels,run_window
import paper_expcast_native as native
import tilelang_expcast as tl_backend

TILELANG_VARIANTS={
    "tilelang_paper_s1":(1,"baseline"),
    "tilelang_paper_s2":(2,"baseline"),
    "tilelang_overlap_s1":(1,"overlap"),
    "tilelang_overlap_s2":(2,"overlap"),
}


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in (("batch",1),("heads",28),("queries",37741),("tokens",37741),
                         ("dim",128),("trials",5),("repeats",10),("seed",0),("check-queries",128),
                         ("kernel-trials",3)):
        p.add_argument("--"+name,type=int,default=default)
    p.add_argument("--dtype",choices=("float16","bfloat16"),default="bfloat16")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--qkv-file",type=Path)
    p.add_argument("--qkv-layout",choices=("NHD","HND"),default="NHD")
    p.add_argument("--skip-flash4",action="store_true")
    p.add_argument("--skip-cupti",action="store_true")
    p.add_argument("--validate-only",action="store_true")
    p.add_argument("--kernel-warmup-seconds",type=float,default=2.)
    p.add_argument("--kernel-seconds",type=float,default=1.)
    p.add_argument("--trace-dir",type=Path,default=Path("results/paper_expcast_overlap_traces"))
    p.add_argument("--output",type=Path,default=Path("results/h200_paper_expcast_tilelang_overlap.json"))
    a=p.parse_args(argv)
    if any(getattr(a,n)<1 for n in ("batch","heads","queries","tokens","trials","repeats","kernel_trials")) or a.dim not in (64,128):
        p.error("Positive dimensions/counts and D64/128 required")
    if a.check_queries<0 or not all(0<x<float("inf") for x in (a.kernel_seconds,a.kernel_warmup_seconds)):
        p.error("Invalid reference count or timing window")
    return a


def implementations(ext):
    return {
        "cuda_paper":lambda s:native.forward(ext,s,False),
        "cuda_paper_interleaved":lambda s:native.forward(ext,s,True),
        **{name:lambda s,stages=stages,schedule=schedule:tl_backend.forward_prepared(s,stages,schedule)
           for name,(stages,schedule) in TILELANG_VARIANTS.items()},
    }


def check_logical_payloads(sage_state,natural_state,reference):
    """Untimed check only: equal quantization, different V storage order."""
    for key in ("q","k","qs","ks","vs"):
        torch.testing.assert_close(sage_state[key],natural_state[key],rtol=0,atol=0,
                                   msg=f"Independent preprocessing differs: {key}")
    v=sage_state["v"]
    expected=v.view(torch.uint8).index_select(-1,reference.value_permutation(v.shape[-1],v.device))
    torch.testing.assert_close(natural_state["v"].view(torch.uint8),expected,rtol=0,atol=0,
                               msg="Natural V bytes differ from Sage after accounting for storage order")
    return {"qk_and_scales_exact":True,"logical_v_bytes_exact":True,
            "tilelang_v_layout":tl_backend.V_LAYOUT,"check_outside_timing":True}


def make_paper_calls(ext,vc,qkv,sage_state,natural_state):
    """Core has no layout conversion; full calls their own fresh preprocessing."""
    core,full={},{}
    for name,fn in implementations(ext).items():
        if name.startswith("tilelang"):
            core[name]=lambda fn=fn:fn(natural_state)
            full[name]=lambda fn=fn:fn(tl_backend.prepare(*qkv))
        else:
            core[name]=lambda fn=fn:fn(sage_state)
            full[name]=lambda fn=fn:fn(vc.prepare(*qkv,expcast=True,v_smooth=False))
    return core,full


def encoding_inputs(device):
    boundaries=((torch.arange(120,device=device,dtype=torch.float64)+.5-119.6500015258789)/11.541560173034668).float()
    return torch.cat((torch.linspace(-32,0,65536,device=device),boundaries,
        torch.nextafter(boundaries,torch.full_like(boundaries,-float("inf"))),
        torch.nextafter(boundaries,torch.full_like(boundaries,float("inf"))),
        torch.tensor([-float("inf"),-0.,0.],device=device)))


def uniform_tile_inputs(dtype,dim,keys,device):
    """Zero logits and distinct V tiles expose premature shared-buffer reuse."""
    q=torch.zeros(1,2,65,dim,device=device,dtype=dtype)
    k=torch.zeros(1,2,keys,dim,device=device,dtype=dtype)
    token=torch.arange(keys,device=device).view(1,1,keys,1)
    channel=torch.arange(dim,device=device).view(1,1,1,dim)
    head=torch.arange(2,device=device).view(1,2,1,1)
    v=(1+2*(token//128)+(token%16)/16+(channel%8)/8+head/4).to(dtype)
    return q,k,v


def save_tilelang_source(state,stages,directory,schedule="baseline"):
    b,h,m,d=state["q"].shape
    dtype="float16" if state["output_type"]==0 else "bfloat16"
    fn=tl_backend.compile_kernel(b,h,m,state["k"].shape[2],d,dtype,stages,schedule)
    path=directory/f"{dtype}_d{d}_m{m}_n{state['k'].shape[2]}_{schedule}_s{stages}.cu"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(fn.get_kernel_source(),encoding="utf-8")
    return str(path)


@torch.no_grad()
def validate(ext,vc,reference,device,save_progress,source_dir):
    import tilelang
    x=encoding_inputs(device)
    expected=(x.double()*11.541560173034668+119.6500015258789).float().round().clamp(0,120).byte()
    encoder=tilelang.compile(tl_backend.make_encoder(x.numel()),out_idx=[1],
        target={"kind":"cuda","arch":"sm_90a"},
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True})
    torch.testing.assert_close(encoder(x),expected,rtol=0,atol=0)
    result={"encoding_values":x.numel(),"encoding_exact":True,"cases":[]}
    save_progress(result)
    calls=implementations(ext)
    generator=torch.Generator(device=device).manual_seed(3333)
    for dtype in (torch.float16,torch.bfloat16):
        for d in (64,128):
            cases=[("random",m,n) for m,n in ((1,1),(65,129),(193,256))]
            cases += [("uniform_tiles",65,n) for n in (128,129,257)]
            for pattern,m,n in cases:
                if pattern=="uniform_tiles":
                    q,k,v=uniform_tile_inputs(dtype,d,n,device)
                else:
                    q=torch.randn(1,2,m,d,device=device,dtype=dtype,generator=generator)
                    k=torch.randn(1,2,n,d,device=device,dtype=dtype,generator=generator)
                    v=torch.randn(k.shape,device=device,dtype=dtype,generator=generator)
                    if n==1:q.zero_();k.zero_()
                state=vc.prepare(q,k,v,expcast=True,v_smooth=False)
                natural=tl_backend.prepare(q,k,v)
                oracle=reference.reference(state,chunk=32).to(dtype)
                row={"pattern":pattern,"dtype":str(dtype),"dim":d,"queries":m,"keys":n,"variants":{}}
                row["payload_check"]=check_logical_payloads(state,natural,reference)
                result["cases"].append(row)
                # Publish the active case before dispatch so a failure is actionable.
                save_progress(result)
                if pattern=="uniform_tiles":
                    decoded=state["v"].float().index_select(-1,reference.value_permutation(state["v"].shape[-1],device))
                    mean=(decoded[...,:n].mean(-1)*state["vs"]).unsqueeze(2).expand_as(oracle).to(dtype)
                    torch.testing.assert_close(oracle,mean,rtol=.001,atol=.001)
                baseline=None
                tilelang_baselines={}
                for name,call in calls.items():
                    print("Validate",name,pattern,str(dtype),d,m,n,flush=True)
                    result["active_variant"]=name
                    row["variants"][name]={"status":"running"}
                    payload=natural if name.startswith("tilelang") else state
                    if name.startswith("tilelang"):
                        stages,schedule=TILELANG_VARIANTS[name]
                        row["variants"][name]["generated_source"]=save_tilelang_source(payload,stages,source_dir,schedule)
                    save_progress(result)
                    out=call(payload)
                    error=row["variants"][name]
                    error["decoded_reference"]=metrics(out,oracle)
                    save_progress(result)
                    if error["decoded_reference"]["relative_rmse"]>.01:
                        raise AssertionError(f"{name}, {pattern}, D{d}, M{m}, N{n}: decoded reference RMSE exceeds 1%: {error}")
                    if baseline is None:baseline=out
                    else:error["cuda_paper_all_rows"]=metrics(out,baseline,limit=.001 if name.startswith("cuda") else .01)
                    if name.startswith("cuda"):
                        error["exact_vs_cuda_paper"]=torch.equal(out,baseline)
                    if name.startswith("tilelang"):
                        if schedule=="baseline":tilelang_baselines[stages]=out
                        else:
                            error["tilelang_baseline_all_rows"]=metrics(out,tilelang_baselines[stages],limit=.001)
                            error["exact_vs_tilelang_baseline"]=torch.equal(out,tilelang_baselines[stages])
                    stream=torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):side=call(payload)
                    stream.synchronize()
                    torch.testing.assert_close(side,out,rtol=0,atol=0)
                    error["status"]="passed"
                    save_progress(result)
    result.pop("active_variant",None)
    return result


@torch.no_grad()
def main():
    a=parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.trace_dir.mkdir(parents=True,exist_ok=True)
    report={"status":"starting","contract":"INT8 QK; Eq7 direct FP32 mul/sub/FMA/RNE; FP8 PV; decoded-P denominator; V smooth OFF",
            "tilelang_v_layout":tl_backend.V_LAYOUT,
            "tilelang_schedules":TILELANG_VARIANTS,
            "schedule_contract":"Overlap decoded-P sum and separate FP32 history multiply with asynchronous fresh PV; wait before FP32 add and before releasing V/P. Same formula and payloads.",
            "tilelang_preprocessing":"Direct natural-order channel quantization; no Sage permutation or inverse permutation. Padding/transpose/quantization included in full API."}
    def save():a.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    def update(key,value):report[key]=value;save()
    try:
        device=torch.device(a.device);torch.cuda.set_device(device)
        if torch.cuda.get_device_capability(device)!=(9,0):raise RuntimeError("Hopper required")
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(a.seed)
        import tilelang
        from sageattention_local import vc,vc_reference,sageattn
        report["environment"]={"gpu":torch.cuda.get_device_name(device),"torch":torch.__version__,
            "cuda":torch.version.cuda,"tilelang":tilelang.__version__,"native_version":vc._extension().sage_vc_kernel_version}
        ext,report["native_build"]=native.load_native();save()
        report["validation"]=validate(ext,vc,vc_reference,device,lambda r:update("validation",r),a.trace_dir/"validation_sources")
        if a.validate_only:
            report["status"]="validation_passed";save();print("Validation passed:",a.output);return
        qkv,report["inputs"]=load_inputs(a,device)
        report["args"]={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        state=vc.prepare(*qkv,expcast=True,v_smooth=False)
        natural=tl_backend.prepare(*qkv)
        report["payload_check"]=check_logical_payloads(state,natural,vc_reference)
        calls=implementations(ext)
        core,full=make_paper_calls(ext,vc,qkv,state,natural)
        core["vc28_folded"]=lambda:vc.forward_prepared(state,kernel_mode="vc28_stream_i2f")
        full["vc28_folded"]=lambda:vc.sageattn_vc(*qkv,tensor_layout="HND",expcast=True,v_smooth=False,kernel_mode="vc28_stream_i2f")
        sage_native,_=load_sage_core_ops()
        core["sage_cpp"]=sage_core_calls(SimpleNamespace(vc=vc),state,sage_native,sage_native)["sage_cpp"]
        full["sage_cpp"]=lambda:sageattn(*qkv,tensor_layout="HND",is_causal=False)
        if not a.skip_flash4:
            fa4=make_flash4(qkv);report["flash4"]=fa4.info
            full["flash4"]=fa4.call
        ids=query_indices(a.queries,a.check_queries)
        index=torch.tensor(ids,device=device)
        golden=fp32_reference(*qkv,ids,32)
        oracle=vc_reference.reference(state,query_indices=ids,chunk=32).to(qkv[0].dtype)
        report["accuracy_scope"]={"queries":len(ids),"all_keys":a.tokens,
            "gates":"1% decoded-reference and TileLang/native; 0.1% CUDA or TileLang same-algorithm schedule regression. Implementation only, not model quality."}
        report["accuracy"]={}
        baseline=core["cuda_paper"]()
        for name,call in full.items():
            print("Check main shape",name,flush=True)
            out=output_hnd(call(),"BSHD" if name=="flash4" else "BHND",qkv[0].shape,qkv[0].dtype)
            row={"dense_fp32":metrics(out.index_select(2,index).float().cpu(),golden)}
            if name in core:torch.testing.assert_close(out,core[name](),rtol=0,atol=0)
            if name in calls:
                row["paper_reference"]=metrics(out.index_select(2,index),oracle,limit=.01)
                row["cuda_paper_all_rows"]=metrics(out,baseline,limit=.001 if name.startswith("cuda") else .01)
                row["exact_vs_cuda_paper"]=torch.equal(out,baseline)
                if name in TILELANG_VARIANTS and TILELANG_VARIANTS[name][1]=="overlap":
                    stages,_=TILELANG_VARIANTS[name]
                    old=core[f"tilelang_paper_s{stages}"]()
                    row["tilelang_baseline_all_rows"]=metrics(out,old,limit=.001)
                    row["exact_vs_tilelang_baseline"]=torch.equal(out,old)
            report["accuracy"][name]=row;save()
        resources=ext.resources(a.dim,0 if a.dtype=="float16" else 1)
        report["native_resources"]={"cuda_paper":resources["expcast_plain"],
                                    "cuda_paper_interleaved":resources["expcast_probe"]}
        report["tilelang_sources"]={}
        for name,(stages,schedule) in TILELANG_VARIANTS.items():
            fn=tl_backend.compile_kernel(a.batch,a.heads,a.queries,a.tokens,a.dim,a.dtype,stages,schedule)
            path=a.trace_dir/f"{name}.cu"
            path.write_text(fn.get_kernel_source(),encoding="utf-8")
            report["tilelang_sources"][name]=str(path)
        for call in full.values():run_window(call,a.kernel_warmup_seconds,device)
        samples={"full":{n:[] for n in full},"core":{n:[] for n in core}}
        rng=random.Random(a.seed+3330)
        report["trial_order"]=[]
        for trial in range(a.trials):
            order=[(scope,n) for scope,group in samples.items() for n in group];rng.shuffle(order)
            report["trial_order"].append(order)
            for scope,name in order:
                _,ms=measure_trial((full if scope=="full" else core)[name],a.repeats,device)
                samples[scope][name].append(ms)
            update("api",{scope:{n:stats(v) for n,v in group.items()} for scope,group in samples.items()})
            print(f"Trial {trial+1}/{a.trials} complete",flush=True)
        print("Uninstrumented synchronized API ms; full includes each backend's fresh preprocessing:")
        print(f"{'variant':<26} {'full API':>11} {'fixed core':>11} {'Sage/API':>10} {'FA4/API':>10} {'FP32 RMSE %':>13}")
        for name in full:
            ms=report["api"]["full"][name]["median_ms"]
            c=f"{report['api']['core'][name]['median_ms']:.4f}" if name in core else "--"
            sage=report["api"]["full"]["sage_cpp"]["median_ms"]/ms
            fa=f"{report['api']['full']['flash4']['median_ms']/ms:.3f}x" if "flash4" in full else "--"
            print(f"{name:<26} {ms:>11.4f} {c:>11} {sage:>9.3f}x {fa:>10} {100*report['accuracy'][name]['dense_fp32']['relative_rmse']:>13.6f}")
        print("Sage/API and FA4/API use FULL API times; >1 means faster. Fixed core excludes preprocessing, includes allocation/dispatch.")
        print("TileLang uses natural-order FP8 V with no per-tile reorder. Padding/transpose/quantization are included in FULL API. Logical Q/K/V bytes and scales checked exactly.")
        print("vc28_folded uses folded affine rounding; paper variants use direct Eq7 evaluation. Sampled FP32 errors are not model-quality gates.")
        print("Paper implementation agreement, ALL main-shape output rows:")
        for name in calls:
            row=report["accuracy"][name]
            print(name,"vs CUDA paper RMSE %",100*row["cuda_paper_all_rows"]["relative_rmse"],
                  "max abs",row["cuda_paper_all_rows"]["max_abs"],"exact",row["exact_vs_cuda_paper"])
        report["schedule_comparison"]={}
        print("TileLang baseline / overlap; identical natural-V payloads and direct ExpCast formula:")
        for stages in (1,2):
            old,new=f"tilelang_paper_s{stages}",f"tilelang_overlap_s{stages}"
            row={scope+"_old_over_new":report["api"][scope][old]["median_ms"]/report["api"][scope][new]["median_ms"]
                 for scope in ("full","core")}
            row.update(report["accuracy"][new]["tilelang_baseline_all_rows"])
            row["exact_match"]=report["accuracy"][new]["exact_vs_tilelang_baseline"]
            report["schedule_comparison"][str(stages)]=row
            print(f"  s{stages}: full {row['full_old_over_new']:.3f}x, core {row['core_old_over_new']:.3f}x, "
                  f"ALL-row RMSE {100*row['relative_rmse']:.6f}%, max abs {row['max_abs']:.8f}, exact={row['exact_match']}")
        if not a.skip_cupti:
            capture={**core,**({"flash4":full["flash4"]} if "flash4" in full else {})}
            report["kernel_windows"]=profile_kernels(capture,a,device,
                on_result=lambda r:update("kernel_windows",r),
                kernel_markers={name:("vc_tilelang_paper",) for name in core if name.startswith("tilelang")})
        report["status"]="passed";save()
        print("Saved:",a.output)
    except Exception as e:
        report.update(status="failed",error=f"{type(e).__name__}: {e}");save()
        raise


if __name__=="__main__":main()
