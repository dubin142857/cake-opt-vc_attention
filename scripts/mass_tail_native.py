"""Carry four decoded pairs across a single PV batch; scale-prefetch baseline."""
import hashlib
from attention_phase_probe import ROOT, VC, replace_exact
import scale_prefetch_native as base

EXACT_OUTPUT = True
BASELINE = "vc28_scale_prefetch"


def helper_source():
    source=(VC/"work_reduction.cuh").read_text(encoding="utf-8")
    start=source.index("template<bool BitEncode=false")
    end=source.index("template<bool SkipIdentity",start)
    source=source[start:end].replace("vc_probability_prepare_integer(","vc_probability_prepare_mass_tail(")
    source=replace_exact(source,"float (&mass)[NQ][2]) {",
        "float (&mass)[NQ][2], VCLateMass& late) {\n"
        "  static_assert(NQ==1 && !BitEncode && !EncodeOnly && StreamConvert && !InterleaveRows);")
    source=replace_exact(source,
        "vc_probability_int_affine_tile<InterleaveRows>(scores[q],maximum[q],gain,packed[q],mass[q]);",
        "vc_probability_mass_tail_tile(scores[q],maximum[q],gain,packed[q],mass[q],late);")
    return (ROOT/"experiments/expcast/mass_tail.cuh").read_text(encoding="utf-8")+"\n"+source


def check_source():
    source=(ROOT/"experiments/expcast/fused_quad_check.cuh").read_text(encoding="utf-8")
    source=source.replace("fused_quad_check_kernel","mass_tail_check_kernel")
    source=replace_exact(source,
        "if constexpr (Fused) sage_vc::vc_probability_prepare_fused_quad<false,false,true,false>(x,m,scale[step*threads+row],p,alpha,mass);",
        "if constexpr (Fused) {\n"
        "      sage_vc::VCLateMass late;\n"
        "      sage_vc::vc_probability_prepare_mass_tail<false,false,true,false>(x,m,scale[step*threads+row],p,alpha,mass,late);\n"
        "      sage_vc::vc_probability_finish_mass_tail(late,mass[0]);\n    }")
    return source.replace("vc_probability_prepare_balanced_max<","vc_probability_prepare_integer<")


def candidate_kernel():
    source=base.candidate_kernel()
    source=replace_exact(source,"sage_vc_attention_kernel_scale_prefetch(","sage_vc_attention_kernel_mass_tail(")
    marker="  {\n    p ^= 1;"
    main,tail=source.rsplit(marker,1)
    main=replace_exact(main,"    bool probability_rescale=false;",
        "    VCLateMass late;\n    bool probability_rescale=false;")
    main=replace_exact(main,
        "vc_probability_prepare_integer<(WorkReduction==6),carry_mass,(WorkReduction>=9),(WorkReduction==10)>(RS,m,sm_scale,RS_f8,probability_alpha,tile_mass);",
        "vc_probability_prepare_mass_tail<false,false,true,false>(RS,m,sm_scale,RS_f8,probability_alpha,tile_mass,late);")
    main=replace_exact(main,"    // Read the NEXT tile's scale",
        "    // Only decoded half pairs and FP32 scalar mass, never pending P/RO_temp.\n"
        "    vc_probability_finish_mass_tail(late,tile_mass[0]);\n"
        "    // Read the NEXT tile's scale")
    return main+marker+tail


def generate_source():
    source=base.generate_source()
    source=replace_exact(source,base.candidate_kernel(),base.candidate_kernel()+helper_source()+candidate_kernel())
    source=replace_exact(source,
        "if constexpr (Trace) return sage_vc::sage_vc_attention_kernel_scale_prefetch<",
        "if constexpr (Trace) return sage_vc::sage_vc_attention_kernel_mass_tail<")
    source=replace_exact(source,
        "else return sage_vc::sage_vc_attention_kernel_sage_layout<",
        "else return sage_vc::sage_vc_attention_kernel_scale_prefetch<")
    source=source.replace("Scale prefetch experiment requires ExpCast","Mass tail experiment requires ExpCast")
    return replace_exact(source,"PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {",
        check_source()+'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n  m.def("check_probability",&check_probability);')


def load_native():
    return base.base.load_source(generate_source(),"_vc_mass_tail_")


def build_audit(ext,vc,build,dim,output_type):
    result=base.build_audit(ext,vc,build,dim,output_type)
    result["baseline_source"]="Unmodified scale_prefetch_native candidate"
    result["baseline_source_sha256"]=hashlib.sha256(base.candidate_kernel().encode()).hexdigest()
    return result


forward=base.forward


def validate_pipeline(ext,vc,reference,device,progress):
    import torch
    from validate_scale_prefetch import assert_exact
    from profile_int8_h200 import metrics
    rows=[]
    gen=torch.Generator(device=device).manual_seed(3900)
    for dtype in (torch.float16,torch.bfloat16):
        for n in (128,129,257,513):
            qkv=tuple(torch.randn(2,2,length,128,device=device,dtype=dtype,generator=gen)
                      for length in (193,n,n))
            state=vc.prepare(*qkv,expcast=True,v_smooth=False)
            qs=state["qs"].clone().view(2,2,-1,4,8)
            qs[:,:,:,1::2,:]*=128
            state={**state,"qs":qs.reshape_as(state["qs"])}
            row={"dtype":str(dtype),"keys":n,"pattern":"alternate warps with 128x Q scale",
                 "status":"running"}
            rows.append(row);progress(rows)
            print("Validate mass-tail mixed-warp paths",str(dtype),n,flush=True)
            old=forward(ext,state,False)
            out=forward(ext,state,True)
            row["vs_baseline"]=assert_exact(out,old)
            row["decoded_reference"]=metrics(out,reference.reference(state,chunk=32).to(dtype),limit=.01)
            row["status"]="passed";progress(rows)
    return rows
