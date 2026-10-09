"""Same-build balanced-max / fused encode-decode-pair-sum experiment."""
from attention_phase_probe import ROOT, VC, replace_exact
import balanced_max_native as balanced


def helper_source():
    original = balanced.helper_source()
    start = original.index("template<bool BitEncode=false")
    helper = original[start:].replace("vc_probability_prepare_balanced_max(",
                                      "vc_probability_prepare_fused_quad(")
    helper = replace_exact(helper, "vc_probability_int_affine_tile<InterleaveRows>(",
                           "vc_probability_fused_quad_tile(")
    return original + (ROOT / "experiments/expcast/fused_quad.cuh").read_text(encoding="utf-8") + helper


def generate_source():
    source = balanced.generate_source()
    source = replace_exact(source, balanced.helper_source(), helper_source())
    source = source.replace("BalancedMax", "FusedQuad").replace("Balanced max experiment", "Fused quad experiment")
    source = replace_exact(source, "vc_probability_prepare_balanced_max<false,false,true,false>(RS,",
                           "vc_probability_prepare_fused_quad<false,false,true,false>(RS,")
    source = replace_exact(source, "vc_probability_prepare_integer<(WorkReduction==6)",
                           "vc_probability_prepare_balanced_max<(WorkReduction==6)")
    source = replace_exact(source, "PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {",
        (ROOT / "experiments/expcast/fused_quad_check.cuh").read_text(encoding="utf-8") +
        '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n  m.def("check_probability",&check_probability);')
    return source


def load_native():
    return balanced.base.load_source(generate_source(), "_vc_fused_quad_")


forward = balanced.forward
