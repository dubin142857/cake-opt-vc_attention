"""python3 tests/cuda/run_paper_expcast_probe.py; optional pytest/nvcc helper tests."""
from pathlib import Path
import sys
import torch
from torch.utils.cpp_extension import load
import pytest

ROOT=Path(__file__).resolve().parents[2]
ext=load(name="_vc_paper_math",sources=[str(Path(__file__).with_name("paper_expcast_probe.cu"))],
         extra_cuda_cflags=["-O3","--use_fast_math","--expt-relaxed-constexpr"],verbose=True)
sys.modules["_vc_paper_math"]=ext
raise SystemExit(pytest.main(["-q",str(ROOT/"tests/test_paper_expcast_cuda.py")]))
