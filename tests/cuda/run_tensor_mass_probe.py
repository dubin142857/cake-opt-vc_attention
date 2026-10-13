"""Optional denominator/helper GPU gates; no full attention or timing claims."""
from pathlib import Path
import sys
import hashlib
import torch
from torch.utils.cpp_extension import load, _get_build_directory

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from tensor_mass_native import helper_source,check_source,VC
from validate_fused_quad_math import validate
from validate_tensor_mass import validate as validate_mass

source=Path(__file__).with_name("tensor_mass_probe.cu")
digest=hashlib.sha256(source.read_bytes()+helper_source().encode()+check_source().encode())
for path in sorted(VC.parent.rglob("*.cuh")):digest.update(path.read_bytes())
name="_vc_tensor_mass_math_"+digest.hexdigest()[:16]
directory=Path(_get_build_directory(name,verbose=False))
(directory/"tensor_mass_check.cuh").write_text(check_source(),encoding="utf-8")
(directory/"tensor_mass_helper.cuh").write_text(helper_source(),encoding="utf-8")
ext=load(name=name,sources=[str(source)],extra_include_paths=[str(directory)],
         extra_cuda_cflags=["-O3","--use_fast_math","--expt-relaxed-constexpr"],verbose=True)
validate_mass(ext,torch.device("cuda:0"))
validate(ext,torch.device("cuda:0"),label="Tensor mass")
