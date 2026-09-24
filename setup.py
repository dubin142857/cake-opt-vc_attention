"""Build against the torch already installed in the target environment."""
import os
import re
import subprocess
import sys

from setuptools import setup
extensions = []
commands = {}
if os.environ.get("VC_ATTENTION_BUILD_CUDA", "1") != "0":
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    if CUDA_HOME is None or torch.version.cuda is None:
        raise RuntimeError(
            "CUDA Toolkit and a CUDA-enabled PyTorch are required. Install with "
            "pip install . --no-build-isolation. For reference-only use, set "
            "VC_ATTENTION_BUILD_CUDA=0 explicitly."
        )
    compiler = os.path.join(CUDA_HOME, "bin", "nvcc.exe" if sys.platform == "win32" else "nvcc")
    compiler_info = subprocess.check_output([compiler, "--version"], text=True)
    compiler_version = re.search(r"release (\d+)\.(\d+)", compiler_info)
    if not compiler_version or tuple(map(int, compiler_version.groups())) < (12, 4):
        raise RuntimeError("vc-attention 0.7.0 fused WGMMA build requires nvcc >= 12.4. Set CUDA_HOME to a compatible toolkit.")
    # The build host need not have a GPU. H800/H100 are the intended targets.
    # WGMMA requires the architecture-specific sm_90a image. Keep any extra
    # development targets (e.g. sm_86 for diagnostic tests), but always include it.
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "9.0a").replace(" ", ";").split(";")
    arch_list = ["9.0a" if a in ("9.0", "9.0+PTX", "Hopper") else a for a in arch_list if a]
    if not any(a.startswith("9.0a") for a in arch_list):
        arch_list.append("9.0a")
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(dict.fromkeys(arch_list))
    extensions = [CUDAExtension(
        "vc_attention._cuda_ext",
        sources=["csrc/bindings.cpp", "csrc/kernels.cu", "csrc/fused.cu"],
        depends=["csrc/vc_cuda.h", "csrc/wgmma.cuh", "csrc/fused_pipeline.cuh", "csrc/fused_d128.cuh",
                 "csrc/wgmma_async.cuh", "csrc/fused_interleaved.cuh"],
        libraries=["cublasLt"],
        extra_compile_args={
            "cxx": ["/O2", "/std:c++17"] if sys.platform == "win32" else ["-O3", "-std=c++17"],
            # ExpCast explicitly uses __fmaf_rn; keep other arithmetic rounding
            # stable for reference comparisons, independent of auto-contraction.
            "nvcc": ["-O3", "-std=c++17", "--fmad=false", "-lineinfo"],
        },
    )]
    commands["build_ext"] = BuildExtension

setup(
    name="vc-attention",
    version="0.7.0",
    description="VC-Attention reference and CUDA C++ FP8 backend for Hopper",
    packages=["vc_attention"],
    package_dir={"": "src"},
    ext_modules=extensions,
    cmdclass=commands,
    python_requires=">=3.9",
    install_requires=["torch>=2.3"],
    extras_require={"test": ["pytest", "numpy"]},
    include_package_data=False,
)
