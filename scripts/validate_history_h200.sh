#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Use installed dependencies; no downloads or disassembler required.
# This also runs D128 history/tail and independent known-answer checks.
PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 bash scripts/build_sage_local.sh

python3 scripts/profile_sage_vc.py --compare-history-candidates \
  --batch 1 --queries 37741 --tokens 37741 --heads 28 --dim 128 \
  --dtype bfloat16 --warmup 3 --repeats 10 --trials 9 \
  --kernel-samples 20 --kernel-trials 3 \
  --trace-dir results/h200_vc21_traces \
  --output results/h200_vc21_history.json
