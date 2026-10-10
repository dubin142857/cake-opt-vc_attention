#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Existing Torch/Sage/FA4/nvcc only. Validation runs before timed APIs.
exec "${PYTHON:-python3}" scripts/benchmark_cuda_expcast.py --experiment pv_prefix --trials 9 --kernel-trials 5 "$@"
