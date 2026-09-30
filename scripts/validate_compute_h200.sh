#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Use installed build dependencies; no downloads or disassembler required.
# Build script validates the carried QK result, tile phases and tails first.
PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 bash scripts/build_sage_local.sh

python3 scripts/profile_sage_vc.py --compare-compute-candidates \
  --batch 1 --queries 37741 --tokens 37741 --heads 28 --dim 128 \
  --dtype bfloat16 --warmup 3 --repeats 10 --trials 9 \
  --kernel-samples 20 --kernel-trials 3 \
  --trace-dir results/h200_vc20_traces \
  --output results/h200_vc20_compute.json
