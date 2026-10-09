"""python3 tests/cuda/run_fused_quad_probe.py; optional helper-only GPU checks."""
from pathlib import Path
import sys
import hashlib
import torch
from torch.utils.cpp_extension import load, _get_build_directory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from fused_quad_native import helper_source, VC
from validate_fused_quad_math import validate

source = Path(__file__).with_name("fused_quad_probe.cu")
helper = helper_source()
digest = hashlib.sha256(helper.encode()+source.read_bytes())
digest.update((ROOT / "experiments/expcast/fused_quad_check.cuh").read_bytes())
for path in sorted(VC.parent.rglob("*.cuh")):
    digest.update(path.read_bytes())
name = "_vc_fused_quad_math_"+digest.hexdigest()[:16]
directory = Path(_get_build_directory(name, verbose=False))
(directory / "fused_quad_generated.cuh").write_text(helper, encoding="utf-8")
ext = load(name=name, sources=[str(source)], extra_include_paths=[str(directory)],
           extra_cuda_cflags=["-O3", "--use_fast_math", "--expt-relaxed-constexpr"], verbose=True)
validate(ext, torch.device("cuda:0"))
