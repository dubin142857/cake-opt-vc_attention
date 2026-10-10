#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Uses existing Torch, Sage, FA4 and nvcc. No package rebuild or download.
exec "${PYTHON:-python3}" scripts/benchmark_cuda_expcast.py --experiment scale_prefetch --trials 9 --kernel-trials 5 "$@"
