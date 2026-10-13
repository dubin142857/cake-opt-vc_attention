"""Reduce exactly decoded FP8 P with FP16 MMA/FP32 accumulation."""
import hashlib
from attention_phase_probe import ROOT, VC, replace_exact
import scale_prefetch_native as base

EXACT_OUTPUT = True
BASELINE = "vc28_scale_prefetch"


def helper_source():
    source=(VC/"work_reduction.cuh").read_text(encoding="utf-8")
    start=source.index("template<bool BitEncode=false")
    end=source.index("template<bool SkipIdentity",start)
    source=source[start:end].replace("vc_probability_prepare_integer(","vc_probability_prepare_tensor_mass(")
    source=replace_exact(source,"vc_probability_int_affine_tile<InterleaveRows>(",
                         "vc_probability_tensor_mass_tile(")
    return (ROOT/"experiments/expcast/tensor_mass.cuh").read_text(encoding="utf-8")+"\n"+source


def candidate_kernel():
    source=base.candidate_kernel().replace("sage_vc_attention_kernel_scale_prefetch(",
                                         "sage_vc_attention_kernel_tensor_mass(")
    return replace_exact(source,"vc_probability_prepare_integer<(","vc_probability_prepare_tensor_mass<(")


def check_source():
    source=(ROOT/"experiments/expcast/fused_quad_check.cuh").read_text(encoding="utf-8")
    source=source.replace("fused_quad_check_kernel","tensor_mass_check_kernel")
    source=source.replace("vc_probability_prepare_fused_quad<","vc_probability_prepare_tensor_mass<")
    source=source.replace("vc_probability_prepare_balanced_max<","vc_probability_prepare_integer<")
    return source+"\n"+(ROOT/"experiments/expcast/tensor_mass_check.cuh").read_text(encoding="utf-8")


def generate_source():
    source=base.generate_source()
    source=replace_exact(source,base.candidate_kernel(),base.candidate_kernel()+helper_source()+candidate_kernel())
    source=replace_exact(source,"if constexpr (Trace) return sage_vc::sage_vc_attention_kernel_scale_prefetch<",
                         "if constexpr (Trace) return sage_vc::sage_vc_attention_kernel_tensor_mass<")
    source=replace_exact(source,"else return sage_vc::sage_vc_attention_kernel_sage_layout<",
                         "else return sage_vc::sage_vc_attention_kernel_scale_prefetch<")
    return replace_exact(source,"PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {",check_source()+
        '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n  m.def("check_probability",&check_probability);\n'
        '  m.def("check_mass",&check_mass);')


def load_native():
    return base.base.load_source(generate_source(),"_vc_tensor_mass_")


def build_audit(ext,vc,build,dim,output_type):
    result=base.build_audit(ext,vc,build,dim,output_type)
    result["baseline_source"]="Unmodified scale_prefetch_native candidate"
    result["baseline_source_sha256"]=hashlib.sha256(base.candidate_kernel().encode()).hexdigest()
    return result


forward=base.forward


def check_resources(ext):
    from scalar_prefetch_native import check_resources as check
    return check(ext,label="Tensor-mass")


def validate_pipeline(ext,vc,reference,device,progress):
    # Same payload/warp-transition cases; forward ABI is identical.
    from scalar_prefetch_native import validate_pipeline as check
    return check(ext,vc,reference,device,progress,label="tensor-mass")
