#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Existing Torch/Sage/FA4/nvcc. Resource and exact-output gates precede timing.
exec "${PYTHON:-python3}" scripts/benchmark_cuda_expcast.py --experiment scalar_prefetch --trials 9 --kernel-trials 5 "$@"
