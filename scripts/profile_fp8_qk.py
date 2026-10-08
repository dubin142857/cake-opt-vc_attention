"""Compare vc27 FP8 QK with INT8 vc16/vc22 and original Sage on identical inputs.

Fixed-core pairs share payloads within each dtype. INT8/FP8 change both QK
format and scale granularity; this is an algorithm comparison, not a dtype-only
instruction microbenchmark. All variants keep the same unsmoothed FP8 V.
"""
import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace
import torch
from benchmark_cuda import measure_trial, query_indices, fp32_reference
from profile_sage_vc import stats, load_sage_core_ops, sage_core_calls, trace_layout_samples
from validate_fp8_qk import metrics, check_implementation


def make_calls(vc, fp8, sage_api, sage_native, qkv, int_state, fp8_state):
    core, full, states = {}, {}, {}
    for cast in (False,True):
        int_name="int8_expcast_vc22" if cast else "int8_control_vc16"
        mode="vc22_both" if cast else "vc16"
        state={**int_state,"expcast":cast}
        states[int_name]=state
        core[int_name]=lambda s=state,m=mode:vc.forward_prepared(s,kernel_mode=m)
        full[int_name]=lambda c=cast,m=mode:vc.sageattn_vc(*qkv,expcast=c,v_smooth=False,kernel_mode=m)
        name="fp8_expcast" if cast else "fp8_control"
        state={**fp8_state,"expcast":cast}
        states[name]=state
        core[name]=lambda s=state:fp8.forward_prepared(s)
        full[name]=lambda c=cast:fp8.sageattn_vc_fp8(*qkv,expcast=c)
    core["sage_cpp"]=sage_core_calls(SimpleNamespace(vc=vc),int_state,sage_native,sage_native)["sage_cpp"]
    full["sage_cpp"]=lambda:sage_api(*qkv,tensor_layout="HND",is_causal=False)
    return core,full,states


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name,value in (("batch",1),("queries",37741),("tokens",37741),("heads",28),
                       ("warmup",3),("repeats",5),("trials",5),("check-queries",128),
                       ("kernel-samples",20),("kernel-trials",3),("seed",0)):
        p.add_argument("--"+name,type=int,default=value)
    p.add_argument("--dim",type=int,choices=(64,128),default=128)
    p.add_argument("--dtype",choices=("float16","bfloat16"),default="bfloat16")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--trace-dir",type=Path)
    p.add_argument("--output",type=Path,default=Path("results/h200_vc27_fp8.json"))
    args=p.parse_args(argv)
    for name in ("batch","queries","tokens","heads","warmup","repeats","trials","kernel_samples","kernel_trials"):
        if getattr(args,name)<1: p.error(name+" must be positive")
    if args.check_queries<0: p.error("check-queries must be nonnegative (0 means all)")
    return args


@torch.no_grad()
def main():
    args=parse_args()
    from sageattention_local import vc, vc_fp8, vc_reference, sageattn
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device)!=(9,0): raise RuntimeError("Hopper required")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(args.seed)
    dtype=getattr(torch,args.dtype)
    q=torch.randn(args.batch,args.heads,args.queries,args.dim,device=device,dtype=dtype)
    k=torch.randn(args.batch,args.heads,args.tokens,args.dim,device=device,dtype=dtype)
    v=torch.randn_like(k)
    qkv=(q,k,v)
    int_state=vc.prepare(*qkv,v_smooth=False)
    fp8_state=vc_fp8.prepare(*qkv)
    for name in ("v","vs"):
        assert torch.equal(int_state[name].view(torch.uint8),fp8_state[name].view(torch.uint8)), "V payload differs"
    native,_=load_sage_core_ops()
    core,full,states=make_calls(vc,vc_fp8,sageattn,native,qkv,int_state,fp8_state)
    ids=query_indices(args.queries,args.check_queries)
    index=torch.tensor(ids,device=device)
    print(f"Input NHD shape: {(args.batch,args.queries,args.heads,args.dim)}; all {args.tokens} keys",flush=True)
    print(f"Accuracy: {len(ids)}/{args.queries} query rows; FP8 implementation check precedes timing",flush=True)
    golden=fp32_reference(*qkv,ids,32)
    accuracy={}
    for name,call in core.items():
        out=call()
        # Every row must be finite and match its freshly preprocessed public API.
        torch.testing.assert_close(full[name](),out,rtol=0,atol=0)
        if not torch.isfinite(out).all(): raise AssertionError(name+" produced NaN/Inf")
        sample=out.index_select(2,index).float().cpu()
        entry={"vs_dense_fp32":metrics(sample,golden)}
        if name.startswith("fp8_"):
            oracle=vc_reference.reference(states[name],query_indices=ids,chunk=32).to(dtype).float().cpu()
            entry["vs_decoded_fp8_reference"]=check_implementation(sample,oracle)
        accuracy[name]=entry
        print("Checked",name,json.dumps(entry),flush=True)
    for name,call in [*core.items(),*full.items()]:
        for _ in range(args.warmup): call()
    samples={kind:{name:[] for name in core} for kind in ("core","full_api")}
    rng=random.Random(args.seed+2700)
    orders=[]
    for trial in range(args.trials):
        order=[(kind,name) for kind in samples for name in core]
        rng.shuffle(order);orders.append(order)
        for kind,name in order:
            _,elapsed=measure_trial((core if kind=="core" else full)[name],args.repeats,device)
            samples[kind][name].append(elapsed)
        print(f"Trial {trial+1}/{args.trials} complete",flush=True)
    rows={name:{kind:stats(samples[kind][name]) for kind in samples} for name in core}
    print("Uninstrumented synchronized API wall time; milliseconds:")
    print(f"{'variant':<25} {'full API':>12} {'fixed core':>12} {'FP32 RMSE %':>14} {'FP8 REF %':>12}")
    for name,row in rows.items():
        ref=accuracy[name].get("vs_decoded_fp8_reference")
        ref_text=f"{100*ref['relative_rmse']:.6f}" if ref else "--"
        print(f"{name:<25} {row['full_api']['median_ms']:>12.4f} {row['core']['median_ms']:>12.4f} "
              f"{100*accuracy[name]['vs_dense_fp32']['relative_rmse']:>14.6f} {ref_text:>12}")
    report={"environment":{"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),
                           "native_kernel_version":vc._extension().sage_vc_kernel_version},
            "args":{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
            "quantization":{"int8":"Sage per-thread; input-dtype K centering",
                            "fp8":"E4M3 block Q64/K128; FP32 K centering; scales=max(amax/448,1e-12)",
                            "v":"same unsmoothed Sage per-channel FP8 payload on these random inputs; FP8 entry additionally handles zero channels", "hadamard":False},
            "rows":rows,"accuracy":accuracy,"trial_order":orders,
            "compiled_resources":vc._extension().fp8_kernel_resources(0 if dtype==torch.float16 else 1,args.dim),
            "scope":"Random inputs; fresh preprocessing in full API. Fixed core includes allocation/dispatch. No model-quality or dtype-only speedup claim."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Preserve uninstrumented results even if CUPTI is unavailable.
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if args.trace_dir:
        report["cupti"]=trace_layout_samples(core,args,device)
        args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("Saved:",args.output,flush=True)


if __name__ == "__main__": main()
