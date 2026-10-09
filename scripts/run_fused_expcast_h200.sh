#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Uses installed dependencies and nvcc; no downloads or package rebuild.
exec "${PYTHON:-python3}" scripts/benchmark_cuda_expcast.py --experiment fused_quad "$@"
