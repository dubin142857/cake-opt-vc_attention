"""JIT vc28 baseline / balanced integer max tree. Same encoding and FP32 mass."""
from attention_phase_probe import ROOT,VC,replace_exact
import split_mass_native as base


def helper_source():
    source=(VC/"work_reduction.cuh").read_text(encoding="utf-8")
    start=source.index("template<bool BitEncode=false")
    end=source.index("template<bool SkipIdentity",start)
    source=source[start:end].replace("vc_probability_prepare_integer(","vc_probability_prepare_balanced_max(")
    source=replace_exact(source,"vc_integer_row_max(scores[q],r)","vc_integer_row_max_balanced(scores[q],r)")
    return (ROOT/"experiments/expcast/balanced_max.cuh").read_text(encoding="utf-8")+"\n"+source


def generate_source():
    source=base.generate_source()
    source=replace_exact(source,base.helper_source(),helper_source())
    source=source.replace("SplitMass","BalancedMax").replace("Split mass experiment","Balanced max experiment")
    return replace_exact(source,"vc_probability_prepare_split_mass<false,false,true,false>",
                         "vc_probability_prepare_balanced_max<false,false,true,false>")


def load_native():
    return base.load_source(generate_source(),"_vc_balanced_max_")


forward=base.forward
