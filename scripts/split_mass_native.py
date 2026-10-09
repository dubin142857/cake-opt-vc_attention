"""JIT vc28 baseline / split FP32 mass chains; no installed-package changes."""
import hashlib
import os
from pathlib import Path
from attention_phase_probe import ROOT, VC, replace_exact

EXPERIMENT=ROOT/"experiments/expcast"


def helper_source():
    source=(VC/"work_reduction.cuh").read_text(encoding="utf-8")
    start=source.index("template<bool BitEncode=false")
    end=source.index("template<bool SkipIdentity",start)
    source=source[start:end].replace("vc_probability_prepare_integer(","vc_probability_prepare_split_mass(")
    source=replace_exact(source,"vc_probability_int_affine_tile<InterleaveRows>(",
                         "vc_probability_split_mass_tile(")
    return (EXPERIMENT/"split_mass.cuh").read_text(encoding="utf-8")+"\n"+source


def generate_source():
    source=(VC/"vc_sm90.cu").read_text(encoding="utf-8")
    prefix=source[:source.index("template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t NUM_THREADS")]
    kernel=(VC/"sage_layout.cuh").read_text(encoding="utf-8")
    kernel=replace_exact(kernel,"bool PreloadMeans=false>","bool PreloadMeans=false, bool SplitMass=false>")
    old="vc_probability_prepare_integer<(WorkReduction==6),carry_mass,(WorkReduction>=9),(WorkReduction==10)>(RS,m,sm_scale,RS_f8,probability_alpha,tile_mass);"
    kernel=replace_exact(kernel,old,
        "{ if constexpr (SplitMass) vc_probability_prepare_split_mass<false,false,true,false>(RS,m,sm_scale,RS_f8,probability_alpha,tile_mass); else "+old+" }")
    # Both modes use the same native validation, allocation and TMA setup.
    wrapper=(ROOT/"scripts/attention_phase_probe.cu").read_text(encoding="utf-8")
    wrapper=replace_exact(wrapper,"Trace ? clocks.data_ptr() : nullptr","nullptr")
    start=wrapper.index("  auto count=");end=wrapper.index("  if (D==64",start)
    wrapper=wrapper[:start]+"  torch::Tensor clocks;\n"+wrapper[end:]
    wrapper=wrapper.replace("std::vector<torch::Tensor> forward(","torch::Tensor forward(")
    wrapper=wrapper.replace("  return {out,clocks};","  return out;")
    wrapper=replace_exact(wrapper,'  auto B=q.size(0)',
        '  TORCH_CHECK(cast,"Split mass experiment requires ExpCast");\n  auto B=q.size(0)')
    return prefix+'#include "fp8_qk_mma.cuh"\n'+helper_source()+kernel.replace("#pragma once","")+"\n}\n"+wrapper


def load_native():
    from torch.utils.cpp_extension import load,_get_build_directory
    source=generate_source();digest=hashlib.sha256(source.encode())
    for path in sorted(VC.parent.rglob("*.cuh")):digest.update(path.read_bytes())
    name="_vc_split_mass_"+digest.hexdigest()[:16]
    path=Path(_get_build_directory(name,verbose=False))/"kernel.cu"
    path.write_text(source,encoding="utf-8")
    os.environ.setdefault("MAX_JOBS","2")
    ext=load(name=name,sources=[str(path)],extra_include_paths=[str(VC)],
        extra_cuda_cflags=["-O3","--use_fast_math","--expt-relaxed-constexpr",
                           "-gencode=arch=compute_90a,code=sm_90a","-Xptxas=-v"],
        extra_ldflags=["cuda.lib"] if os.name=="nt" else ["-lcuda"],verbose=True)
    return ext,{"source_hash":digest.hexdigest(),"generated_source":str(path)}


def forward(ext,state,split=False):
    from profile_int8_h200 import require_int8_state
    require_int8_state(state)
    if (not state["expcast"] or state["block_values"] or state["means"].numel()
            or state["permutation"] is not None or state.get("v_layout","sage_permuted")!="sage_permuted"):
        raise ValueError("Requires ExpCast, Sage V layout and V smoothing OFF")
    return ext.forward(*(state[k] for k in ("q","k","v","qs","ks","vs","output_type","scale")),True,split)
