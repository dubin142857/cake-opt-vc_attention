"""Same-payload paper ExpCast: native CUDA schedules and TileLang on H200."""
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
    p.add_argument("--trace-dir",type=Path,default=Path("results/paper_expcast_traces"))
    p.add_argument("--output",type=Path,default=Path("results/h200_paper_expcast_tilelang.json"))
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
        "tilelang_paper_s1":lambda s:tl_backend.forward_prepared(s,1),
        "tilelang_paper_s2":lambda s:tl_backend.forward_prepared(s,2),
    }


def encoding_inputs(device):
    boundaries=((torch.arange(120,device=device,dtype=torch.float64)+.5-119.6500015258789)/11.541560173034668).float()
    return torch.cat((torch.linspace(-32,0,65536,device=device),boundaries,
        torch.nextafter(boundaries,torch.full_like(boundaries,-float("inf"))),
        torch.nextafter(boundaries,torch.full_like(boundaries,float("inf"))),
        torch.tensor([-float("inf"),-0.,0.],device=device)))


@torch.no_grad()
def validate(ext,vc,reference,device,save_progress):
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
            for m,n in ((1,1),(65,129),(193,256)):
                q=torch.randn(1,2,m,d,device=device,dtype=dtype,generator=generator)
                k=torch.randn(1,2,n,d,device=device,dtype=dtype,generator=generator)
                v=torch.randn(k.shape,device=device,dtype=dtype,generator=generator)
                # Exact uniform-attention case; remaining shapes exercise maxima/tails.
                if n==1:q.zero_();k.zero_()
                state=vc.prepare(q,k,v,expcast=True,v_smooth=False)
                oracle=reference.reference(state,chunk=32).to(dtype)
                row={"dtype":str(dtype),"dim":d,"queries":m,"keys":n,"variants":{}}
                baseline=None
                for name,call in calls.items():
                    print("Validate",name,str(dtype),d,m,n,flush=True)
                    out=call(state)
                    error={"decoded_reference":metrics(out,oracle,limit=.01)}
                    if baseline is None:baseline=out
                    else:error["cuda_paper_all_rows"]=metrics(out,baseline,limit=.001 if name.startswith("cuda") else .01)
                    if name.startswith("cuda"):
                        error["exact_vs_cuda_paper"]=torch.equal(out,baseline)
                    stream=torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):side=call(state)
                    stream.synchronize()
                    torch.testing.assert_close(side,out,rtol=0,atol=0)
                    row["variants"][name]=error
                result["cases"].append(row)
                save_progress(result)
    return result


@torch.no_grad()
def main():
    a=parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.trace_dir.mkdir(parents=True,exist_ok=True)
    report={"status":"starting","contract":"INT8 QK; Eq7 direct FP32 mul/sub/FMA/RNE; FP8 PV; decoded-P denominator; V smooth OFF"}
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
        report["validation"]=validate(ext,vc,vc_reference,device,lambda r:update("validation",r))
        if a.validate_only:
            report["status"]="validation_passed";save();print("Validation passed:",a.output);return
        qkv,report["inputs"]=load_inputs(a,device)
        report["args"]={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        state=vc.prepare(*qkv,expcast=True,v_smooth=False)
        calls=implementations(ext)
        core={name:lambda fn=fn:fn(state) for name,fn in calls.items()}
        full={name:lambda fn=fn:fn(vc.prepare(*qkv,expcast=True,v_smooth=False)) for name,fn in calls.items()}
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
            "gates":"1% decoded-reference and TileLang/native; 0.1% CUDA schedule regression. Implementation only, not model quality."}
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
            report["accuracy"][name]=row;save()
        resources=ext.resources(a.dim,0 if a.dtype=="float16" else 1)
        report["native_resources"]={"cuda_paper":resources["expcast_plain"],
                                    "cuda_paper_interleaved":resources["expcast_probe"]}
        report["tilelang_sources"]={}
        for stages in (1,2):
            fn=tl_backend.compile_kernel(a.batch,a.heads,a.queries,a.tokens,a.dim,a.dtype,stages)
            path=a.trace_dir/f"tilelang_paper_s{stages}.cu"
            path.write_text(fn.get_kernel_source(),encoding="utf-8")
            report["tilelang_sources"][str(stages)]=str(path)
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
        print("Uninstrumented synchronized API ms; full includes fresh shared preprocessing:")
        print(f"{'variant':<26} {'full API':>11} {'fixed core':>11} {'Sage/API':>10} {'FA4/API':>10} {'FP32 RMSE %':>13}")
        for name in full:
            ms=report["api"]["full"][name]["median_ms"]
            c=f"{report['api']['core'][name]['median_ms']:.4f}" if name in core else "--"
            sage=report["api"]["full"]["sage_cpp"]["median_ms"]/ms
            fa=f"{report['api']['full']['flash4']['median_ms']/ms:.3f}x" if "flash4" in full else "--"
            print(f"{name:<26} {ms:>11.4f} {c:>11} {sage:>9.3f}x {fa:>10} {100*report['accuracy'][name]['dense_fp32']['relative_rmse']:>13.6f}")
        print("Sage/API and FA4/API use FULL API times; >1 means faster. Fixed core excludes preprocessing, includes allocation/dispatch.")
        print("vc28_folded uses folded affine rounding; paper variants use direct Eq7 evaluation. Sampled FP32 errors are not model-quality gates.")
        print("Paper implementation agreement, ALL main-shape output rows:")
        for name in calls:
            row=report["accuracy"][name]
            print(name,"vs CUDA paper RMSE %",100*row["cuda_paper_all_rows"]["relative_rmse"],
                  "max abs",row["cuda_paper_all_rows"]["max_abs"],"exact",row["exact_vs_cuda_paper"])
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
