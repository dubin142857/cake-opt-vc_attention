#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 scripts/verify_sage_local_source.py
python3 -c 'import torch, triton; print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "triton:", triton.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name()); assert torch.cuda.get_device_capability() == (9, 0), "Hopper GPU required"; assert hasattr(torch.library, "custom_op"), "PyTorch >=2.4 required"'
nvcc --version
export TORCH_CUDA_ARCH_LIST=9.0a
export MAX_JOBS="${MAX_JOBS:-4}"
export SAGE_LOCAL_NVCC_THREADS="${SAGE_LOCAL_NVCC_THREADS:-2}"
# Independent package and extension names; no uninstall/upgrade of sageattention.
python3 -m pip install ./third_party/sageattention_local --no-build-isolation --no-deps
python3 scripts/validate_sage_local.py --output results/sage_local_validation.json
