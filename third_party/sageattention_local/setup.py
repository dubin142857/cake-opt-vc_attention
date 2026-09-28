"""Independent Hopper build of the pinned SageAttention sources; see UPSTREAM.json.

The upstream setup.py is retained under upstream/. Kernel math is unchanged.
Only _qattn_sm90 and _fused are built; no installed sageattention code is linked.
"""
import os
from pathlib import Path
import re
import subprocess
import sys

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
extensions, commands = [], {}
if "sdist" not in sys.argv:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    if CUDA_HOME is None or torch.version.cuda is None:
        raise RuntimeError("CUDA-enabled PyTorch and nvcc >=12.3 are required; build on the H200 host.")
    if not hasattr(torch.library, "custom_op"):
        raise RuntimeError("This pinned SageAttention wrapper requires torch.library.custom_op (PyTorch >=2.4).")
    nvcc = str(Path(CUDA_HOME) / "bin" / ("nvcc.exe" if sys.platform == "win32" else "nvcc"))
    version = re.search(r"release (\d+)\.(\d+)", subprocess.check_output([nvcc, "--version"], text=True))
    if not version or tuple(map(int, version.groups())) < (12, 3):
        raise RuntimeError("Hopper SageAttention requires nvcc >=12.3")
    # Explicit target makes cross-compilation independent of the build host GPU.
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    threads = int(os.environ.get("SAGE_LOCAL_NVCC_THREADS", "2"))
    if threads < 1:
        raise ValueError("SAGE_LOCAL_NVCC_THREADS must be positive")
    cxx = (["/O2", "/std:c++17", "/DENABLE_BF16"] if sys.platform == "win32" else
           ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"])
    nvcc_flags = ["-O3", "-std=c++17", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                  "--use_fast_math", f"--threads={threads}", "-Xptxas=-v", "-diag-suppress=174",
                  "-gencode=arch=compute_90a,code=sm_90a"]
    headers = [str(p.relative_to(ROOT)) for p in (ROOT / "csrc").rglob("*") if p.suffix in (".h", ".cuh")]
    for name, sources in (
        ("_qattn_sm90", ["csrc/qattn/pybind_sm90.cpp", "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu"]),
        ("_fused", ["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"]),
    ):
        extensions.append(CUDAExtension(
            "sageattention_local." + name, sources=sources, depends=headers,
            libraries=["cuda"] if name == "_qattn_sm90" else [],
            extra_compile_args={"cxx": cxx, "nvcc": nvcc_flags},
        ))
    commands = {"build_ext": BuildExtension}

setup(
    name="sageattention-local", version="2.2.0+vc2",
    description="Independent Hopper source build of SageAttention v2.2.0 for comparison",
    author="SageAttention team (upstream); local build integration in vc-attention",
    url="https://github.com/thu-ml/SageAttention",
    license="Apache-2.0", license_files=["LICENSE", "NOTICE"],
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["sageattention_local", "sageattention_local.*"]),
    package_data={"sageattention_local": ["UPSTREAM.json"]},
    python_requires=">=3.9", install_requires=["torch>=2.4", "triton>=3.0"],
    ext_modules=extensions, cmdclass=commands,
)
