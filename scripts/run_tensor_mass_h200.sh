#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python3}" scripts/benchmark_cuda_expcast.py --experiment tensor_mass --trials 9 --kernel-trials 5 "$@"
