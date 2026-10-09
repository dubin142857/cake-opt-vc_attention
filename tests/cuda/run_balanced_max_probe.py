"""python3 tests/cuda/run_balanced_max_probe.py; optional GPU helper checks."""
from pathlib import Path
import sys
import hashlib
from torch.utils.cpp_extension import load,_get_build_directory
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from balanced_max_native import helper_source,VC

source=Path(__file__).with_name("balanced_max_probe.cu")
helper=helper_source()
digest=hashlib.sha256(helper.encode()+source.read_bytes())
for path in sorted(VC.parent.rglob("*.cuh")):digest.update(path.read_bytes())
name="_vc_balanced_max_math_"+digest.hexdigest()[:16]
directory=Path(_get_build_directory(name,verbose=False))
(directory/"balanced_max_generated.cuh").write_text(helper,encoding="utf-8")
ext=load(name=name,sources=[str(source)],extra_include_paths=[str(directory)],
         extra_cuda_cflags=["-O3","--use_fast_math","--expt-relaxed-constexpr"],verbose=True)
sys.modules["_vc_balanced_max_math"]=ext
raise SystemExit(pytest.main(["-q",str(ROOT/"tests/test_balanced_max_cuda.py")]))
