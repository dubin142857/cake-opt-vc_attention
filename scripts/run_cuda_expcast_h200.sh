#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Existing Torch, sageattention-local, FA4 and nvcc only; no installation.
exec "${PYTHON:-python3}" scripts/benchmark_cuda_expcast.py "$@"
