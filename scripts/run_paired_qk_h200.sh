#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Existing Torch/Sage/FA4/nvcc only; no additional packages or ncu needed.
# INT8 QK, ExpCast only, V smoothing OFF. Exact checks precede timing.
exec "${PYTHON:-python3}" scripts/benchmark_cuda_expcast.py --experiment paired_qk --trials 9 --kernel-trials 5 "$@"
