"""Optional helper-only GPU validation; supports software FP8 decode on SM86."""
from pathlib import Path
import sys
import hashlib
import torch
from torch.utils.cpp_extension import load, _get_build_directory

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from scalar_prefetch_native import helper_source,check_source,VC
from validate_fused_quad_math import validate

source=Path(__file__).with_name("scalar_prefetch_probe.cu")
digest=hashlib.sha256(source.read_bytes()+helper_source().encode()+check_source().encode())
for path in sorted(VC.parent.rglob("*.cuh")):digest.update(path.read_bytes())
name="_vc_scalar_prefetch_math_"+digest.hexdigest()[:16]
directory=Path(_get_build_directory(name,verbose=False))
(directory/"scalar_prefetch_check.cuh").write_text(check_source(),encoding="utf-8")
(directory/"scalar_prefetch_helper.cuh").write_text(helper_source(),encoding="utf-8")
ext=load(name=name,sources=[str(source)],extra_include_paths=[str(directory)],
         extra_cuda_cflags=["-O3","--use_fast_math","--expt-relaxed-constexpr"],verbose=True)
validate(ext,torch.device("cuda:0"),label="Scalar prefetch")
