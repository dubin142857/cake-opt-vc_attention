"""Scale-prefetch baseline / serial prefix / prefix PV overlap, no package edits."""
import hashlib
from attention_phase_probe import ROOT, VC, replace_exact
import scale_prefetch_native as base

EXACT_OUTPUT = True
BASELINE = "vc28_scale_prefetch"
EXTRA_VARIANTS = {"vc28_prefix_serial": 1}


def helper_source():
    return (ROOT / "experiments/expcast/pv_prefix.cuh").read_text(encoding="utf-8")


def check_source():
    source = (ROOT / "experiments/expcast/fused_quad_check.cuh").read_text(encoding="utf-8")
    source = source.replace("fused_quad_check_kernel", "pv_prefix_check_kernel")
    source = replace_exact(source,
        "if constexpr (Fused) sage_vc::vc_probability_prepare_fused_quad<false,false,true,false>(x,m,scale[step*threads+row],p,alpha,mass);",
        "if constexpr (Fused) {\n"
        "      sage_vc::VCLateProbability late;\n"
        "      sage_vc::vc_probability_prepare_prefix(x,m,scale[step*threads+row],p,alpha,mass,late);\n"
        "      sage_vc::vc_probability_finish_prefix(late,p[0][3],mass[0]);\n    }")
    return source.replace("vc_probability_prepare_balanced_max<", "vc_probability_prepare_integer<")


def candidate_kernel():
    source = base.candidate_kernel()
    source = replace_exact(source, "bool PreloadMeans=false>", "bool PreloadMeans=false, bool PrefixOverlap=false>")
    source = replace_exact(source, "sage_vc_attention_kernel_scale_prefetch(", "sage_vc_attention_kernel_pv_prefix(")
    tail_marker = "  {\n    p ^= 1;"
    source, tail = source.rsplit(tail_marker, 1)
    source = replace_exact(source, "    bool probability_rescale=false;",
                           "    VCLateProbability late;\n    bool probability_rescale=false;")
    source = replace_exact(source,
        "vc_probability_prepare_integer<(WorkReduction==6),carry_mass,(WorkReduction>=9),(WorkReduction==10)>(RS,m,sm_scale,RS_f8,probability_alpha,tile_mass);",
        "vc_probability_prepare_prefix(RS,m,sm_scale,RS_f8,probability_alpha,tile_mass,late);")
    # Only MAIN tiles. Masked-tail probability and PV are copied untouched.
    source = replace_exact(source, "    // wait for V\n    wait(&barrier_V, p);\n\n    // A separate",
        "    if constexpr (!PrefixOverlap)\n"
        "      vc_probability_finish_prefix(late,RS_f8[0][3],tile_mass[0]);\n"
        "    // wait for V\n    wait(&barrier_V, p);\n\n    // A separate")
    start = source.index("    // A separate")
    end = source.index("    // Read the NEXT tile's scale", start)
    region = source[start:end]
    region = replace_exact(region, "v_it < num_tiles_pv_inner;", "v_it < (PrefixOverlap ? 3 : num_tiles_pv_inner);")
    region += """    if constexpr (PrefixOverlap) {
      // First three P operands and RO_temp belong to pending WGMMA. The
      // helper touches ONLY the unissued fourth P operand and scalar state.
      vc_probability_finish_prefix(late,RS_f8[0][3],tile_mass[0]);
      // Every warp executes this fence, even if its probability took fallback.
      wgmma::warpgroup_arrive();
      wgmma::wgmma_f8f8f32<head_dim,1,CTA_K>(RO_temp[0],RS_f8[0][3],&sV[96]);
      wgmma::warpgroup_commit_batch();
    }
"""
    return source[:start] + region + source[end:] + tail_marker + tail


def generate_source():
    source = base.generate_source()
    source = replace_exact(source, base.candidate_kernel(), base.candidate_kernel()+helper_source()+candidate_kernel())
    start = source.index("template<int D, typename Out, bool Cast, bool Trace>")
    end = source.index("\n}\n", start)+3
    args = """<64,128,128,D,
    QuantGranularity::kPerThread,QuantGranularity::kPerThread,Out,
    MaskMode::kNone,false,true,true,false,false,true,0,168,false,9"""
    source = source[:start] + """template<int D, typename Out, bool Cast, int Mode>
auto phase_kernel() {
  static_assert(Cast && Mode>=0 && Mode<=2);
  if constexpr (Mode==0) return sage_vc::sage_vc_attention_kernel_scale_prefetch"""+args+""">;
  else return sage_vc::sage_vc_attention_kernel_pv_prefix"""+args+""",false,false,false,(Mode==2)>;
}
""" + source[end:]
    source = source.replace("bool Cast, bool Trace>", "bool Cast, int Mode>")
    source = source.replace("bool Cast,bool Trace>", "bool Cast,int Mode>")
    source = source.replace("Cast,Trace", "Cast,Mode")
    source = source.replace("bool trace)", "int64_t mode)")
    source = replace_exact(source,
        "  if (trace) launch_phase<D,Out,true,true>(q,k,v,qs,ks,vs,out,clocks,scale);\n"
        "  else launch_phase<D,Out,true,false>(q,k,v,qs,ks,vs,out,clocks,scale);",
        "  if (mode==2) launch_phase<D,Out,true,2>(q,k,v,qs,ks,vs,out,clocks,scale);\n"
        "  else if (mode==1) launch_phase<D,Out,true,1>(q,k,v,qs,ks,vs,out,clocks,scale);\n"
        "  else launch_phase<D,Out,true,0>(q,k,v,qs,ks,vs,out,clocks,scale);")
    source = source.replace("scale,cast,trace", "scale,cast,mode")
    source = replace_exact(source, '  TORCH_CHECK(cast,"Scale prefetch experiment requires ExpCast");',
                           '  TORCH_CHECK(cast && mode>=0 && mode<=2,"PV prefix needs ExpCast and mode 0..2");')
    source = replace_exact(source,
        '{"expcast_probe",resources_for<D,Out,true,true>()}',
        '{"expcast_probe",resources_for<D,Out,true,2>()},\n          {"prefix_serial",resources_for<D,Out,true,1>()}')
    return replace_exact(source, "PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {",
        check_source()+'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n  m.def("check_probability",&check_probability);')


def load_native():
    return base.base.load_source(generate_source(), "_vc_pv_prefix_")


def build_audit(ext,vc,build,dim,output_type):
    result=base.build_audit(ext,vc,build,dim,output_type)
    result["baseline_source"]="Unmodified scale_prefetch_native candidate, including original vc28 probability helpers"
    result["baseline_source_sha256"]=hashlib.sha256(base.candidate_kernel().encode()).hexdigest()
    return result


def forward_mode(ext, state, mode):
    # Use the established checks for INT8 QK, physical Sage V, no smoothing.
    return base.forward(ext, state, mode)


def forward(ext, state, optimized=False):
    return forward_mode(ext, state, 2 if optimized else 0)


def validate_pipeline(ext,vc,reference,device,progress):
    import torch
    from validate_scale_prefetch import assert_exact
    from profile_int8_h200 import metrics
    rows=[]
    gen=torch.Generator(device=device).manual_seed(3800)
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
            print("Validate prefix mixed-warp paths",str(dtype),n,flush=True)
            old=forward_mode(ext,state,0)
            serial=forward_mode(ext,state,1)
            out=forward_mode(ext,state,2)
            row["serial_vs_baseline"]=assert_exact(serial,old)
            row["overlap_vs_baseline"]=assert_exact(out,old)
            row["decoded_reference"]=metrics(out,reference.reference(state,chunk=32).to(dtype),limit=.01)
            row["status"]="passed";progress(rows)
    return rows
