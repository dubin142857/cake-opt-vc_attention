#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Existing Torch/Sage/FA4/nvcc only. Validation precedes timing.
exec "${PYTHON:-python3}" scripts/benchmark_cuda_expcast.py --experiment mass_tail --trials 9 --kernel-trials 5 "$@"
