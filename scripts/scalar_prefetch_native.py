"""Prepare the next tile's ExpCast scalars while current PV is pending."""
import hashlib
from attention_phase_probe import ROOT, VC, replace_exact
import scale_prefetch_native as base

EXACT_OUTPUT = True
BASELINE = "vc28_scale_prefetch"


def helper_source():
    source=(VC/"work_reduction.cuh").read_text(encoding="utf-8")
    start=source.index("template<bool BitEncode=false")
    end=source.index("template<bool SkipIdentity",start)
    source=source[start:end].replace("vc_probability_prepare_integer(","vc_probability_prepare_scalars(")
    source=replace_exact(source,"float (&mass)[NQ][2]) {",
        "float (&mass)[NQ][2], const VCExpCastScalars& prepared) {\n"
        "  static_assert(NQ==1 && !BitEncode && !EncodeOnly && StreamConvert && !InterleaveRows);")
    # Only the StreamConvert branch is touched; all score-dependent max/range
    # checks and the warp-wide route decision stay at their original position.
    source=replace_exact(source,
        "      float gain=__fmul_rn(scale,11.541560173034668f);\n"
        "      bool safe=scale>=1.1754943508222875e-38f && isfinite(gain) &&",
        "      float gain=prepared.gain;\n"
        "      bool safe=prepared.scale_safe &&")
    return (ROOT/"experiments/expcast/scalar_prefetch.cuh").read_text(encoding="utf-8")+"\n"+source


def candidate_kernel():
    source=base.candidate_kernel()
    source=replace_exact(source,"sage_vc_attention_kernel_scale_prefetch(",
                         "sage_vc_attention_kernel_scalar_prefetch(")
    source=replace_exact(source,
        "  float current_k_scale = K_scale[k_scale_idx];\n  float original_sm_scale = sm_scale;",
        "  float original_sm_scale = sm_scale;\n"
        "  VCExpCastScalars prepared = vc_expcast_scalars(q_scale,K_scale[k_scale_idx],original_sm_scale);")
    source=replace_exact(source,
        "    float dequant_scale = q_scale * current_k_scale;\n    sm_scale = original_sm_scale * dequant_scale;",
        "    sm_scale = prepared.score_scale;")
    source=replace_exact(source,
        "vc_probability_prepare_integer<(WorkReduction==6),carry_mass,(WorkReduction>=9),(WorkReduction==10)>(RS,m,sm_scale,RS_f8,probability_alpha,tile_mass);",
        "vc_probability_prepare_scalars<false,false,true,false>(RS,m,sm_scale,RS_f8,probability_alpha,tile_mass,prepared);")
    source=replace_exact(source,
        "    // Read the NEXT tile's scale while this PV group is pending.\n"
        "    current_k_scale = K_scale[k_scale_idx + iter * k_scale_advance_offset];",
        "    // Prepare only NEXT-tile scalars; never touch pending P/RO_temp.\n"
        "    prepared = vc_expcast_scalars(q_scale,\n"
        "        K_scale[k_scale_idx + iter * k_scale_advance_offset],original_sm_scale);")
    source=replace_exact(source,
        "    float dequant_scale = q_scale * current_k_scale;\n    sm_scale = original_sm_scale;",
        "    float dequant_scale = prepared.dequant_scale;\n    sm_scale = original_sm_scale;")
    return source


def check_source():
    source=(ROOT/"experiments/expcast/fused_quad_check.cuh").read_text(encoding="utf-8")
    source=source.replace("fused_quad_check_kernel","scalar_prefetch_check_kernel")
    # Match production's cross-tile carry and include alternating fast/fallback
    # scales. Unit test scores are already INT32; scale is the full score scale.
    source=replace_exact(source,"  for(int step=0;step<steps;++step) {",
        "  sage_vc::VCExpCastScalars prepared;\n"
        "  if constexpr (Fused) prepared=sage_vc::vc_expcast_scalars(1.f,scale[row],1.f);\n"
        "  for(int step=0;step<steps;++step) {")
    source=replace_exact(source,
        "if constexpr (Fused) sage_vc::vc_probability_prepare_fused_quad<false,false,true,false>(x,m,scale[step*threads+row],p,alpha,mass);",
        "if constexpr (Fused) sage_vc::vc_probability_prepare_scalars<false,false,true,false>(x,m,scale[step*threads+row],p,alpha,mass,prepared);")
    source=source.replace("vc_probability_prepare_balanced_max<","vc_probability_prepare_integer<")
    return replace_exact(source,"    #pragma unroll\n    for(int k=0;k<4;++k)",
        "    if constexpr (Fused) {\n"
        "      if (step+1<steps) prepared=sage_vc::vc_expcast_scalars(1.f,scale[(step+1)*threads+row],1.f);\n"
        "    }\n    #pragma unroll\n    for(int k=0;k<4;++k)")


def generate_source():
    source=base.generate_source()
    source=replace_exact(source,base.candidate_kernel(),base.candidate_kernel()+helper_source()+candidate_kernel())
    source=replace_exact(source,
        "if constexpr (Trace) return sage_vc::sage_vc_attention_kernel_scale_prefetch<",
        "if constexpr (Trace) return sage_vc::sage_vc_attention_kernel_scalar_prefetch<")
    source=replace_exact(source,
        "else return sage_vc::sage_vc_attention_kernel_sage_layout<",
        "else return sage_vc::sage_vc_attention_kernel_scale_prefetch<")
    source=source.replace("Scale prefetch experiment requires ExpCast","Scalar prefetch experiment requires ExpCast")
    return replace_exact(source,"PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {",
        check_source()+'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n  m.def("check_probability",&check_probability);')


def load_native():
    return base.base.load_source(generate_source(),"_vc_scalar_prefetch_")


def build_audit(ext,vc,build,dim,output_type):
    result=base.build_audit(ext,vc,build,dim,output_type)
    result["baseline_source"]="Unmodified scale_prefetch_native candidate"
    result["baseline_source_sha256"]=hashlib.sha256(base.candidate_kernel().encode()).hexdigest()
    return result


forward=base.forward


def check_resources(ext):
    """Reject resource regressions before remote correctness/timing work."""
    rows=[]
    for dim in (64,128):
        for output_type in (0,1):
            raw=ext.resources(dim,output_type)
            old,new=raw["expcast_plain"],raw["expcast_probe"]
            if (new["registers_per_thread"]>168 or new["local_bytes"]!=0 or
                    new["dynamic_shared_bytes"]!=320*dim or
                    new["static_shared_bytes"]>old["static_shared_bytes"]):
                raise RuntimeError(f"Scalar-prefetch resource gate failed: D{dim}, output {output_type}: {new}")
            rows.append({"dim":dim,"output_type":output_type,"baseline":old,"candidate":new})
    return {"status":"passed","rows":rows,
            "scope":"Compiled resources; not achieved occupancy or a performance guarantee"}


def validate_pipeline(ext,vc,reference,device,progress):
    """Alternating scalar predicates and independent per-warp fallback state."""
    import torch
    from validate_scale_prefetch import assert_exact
    from profile_int8_h200 import metrics
    rows=[]
    gen=torch.Generator(device=device).manual_seed(4200)
    for dtype in (torch.float16,torch.bfloat16):
        for dim in (64,128):
            for n in (128,129,257,513):
                qkv=tuple(torch.randn(2,2,length,dim,device=device,dtype=dtype,generator=gen)
                          for length in (193,n,n))
                state=vc.prepare(*qkv,expcast=True,v_smooth=False)
                qs=state["qs"].clone().view(2,2,-1,4,8)
                qs[:,:,:,1::2,:]*=128
                ks=state["ks"].clone().view(2,2,-1,4)
                ks[:,:,::2,:]*=1e-36
                state={**state,"qs":qs.reshape_as(state["qs"]),"ks":ks.reshape_as(state["ks"])}
                row={"dtype":str(dtype),"dim":dim,"keys":n,
                     "pattern":"alternating tiny/normal K scales; alternate warps with 128x Q scale",
                     "status":"running"}
                rows.append(row);progress(rows)
                print("Validate scalar-prefetch transitions",str(dtype),dim,n,flush=True)
                old=forward(ext,state,False);out=forward(ext,state,True)
                row["vs_baseline"]=assert_exact(out,old)
                row["decoded_reference"]=metrics(out,reference.reference(state,chunk=32).to(dtype),limit=.01)
                row["status"]="passed";progress(rows)
    return rows
