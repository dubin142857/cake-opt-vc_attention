"""Opt-in CUDA helper parity test, independent of the installed attention package.

Run: python3 tests/cuda/run_expcast_detail_probe.py
Requires pytest, nvcc and PyTorch's C++ extension build tools; no downloads.
"""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from attention_phase_probe import VC, RECORDER, instrument_probability_helpers


def generate_test_source(row_max="vc28"):
    prefix = """#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "../math.cuh"
#include "../qattn/attn_utils.cuh"
#define PHASE_RECORD_WIDTH 28
#define PHASE_CTA_STRIDE 1
#define PHASE_TILE_STRIDE 1
namespace sage_vc {
#include "vc_math.cuh"
#include "probability_variants.cuh"
#include "work_reduction.cuh"
"""
    wrapper = Path(__file__).with_name("expcast_detail_probe.cu").read_text(encoding="utf-8")
    balanced = ((ROOT / "experiments/expcast/balanced_max.cuh").read_text(encoding="utf-8")
                if row_max == "balanced_max" else "")
    return (prefix + RECORDER + balanced + instrument_probability_helpers(row_max) + "\n}\n" +
            wrapper.replace('#include "probability_detail_test_generated.cuh"', ""))


def main():
    from torch.utils.cpp_extension import load, _get_build_directory
    import pytest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-max", choices=("vc28", "balanced_max"), default="vc28")
    args = parser.parse_args()
    source = generate_test_source(args.row_max)
    digest = hashlib.sha256(source.encode())
    for path in sorted(VC.parent.rglob("*.cuh")):
        digest.update(path.read_bytes())
    name = "_vc_expcast_detail_math_" + digest.hexdigest()[:16]
    path = Path(_get_build_directory(name, verbose=False)) / "probe.cu"
    path.write_text(source, encoding="utf-8")
    ext = load(name=name, sources=[str(path)], extra_include_paths=[str(VC)],
               extra_cuda_cflags=["-O3", "--use_fast_math", "--expt-relaxed-constexpr"], verbose=True)
    sys.modules["_vc_expcast_detail_math"] = ext
    return pytest.main(["-q", str(ROOT / "tests/test_expcast_detail_cuda.py")])


if __name__ == "__main__":
    raise SystemExit(main())
