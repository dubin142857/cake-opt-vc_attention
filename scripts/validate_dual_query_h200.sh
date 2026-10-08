#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Installed dependencies only. Build runs correctness checks before timing,
# including vc26 Q tails, Q64 scale indexing, shared reuse and current stream.
# ncu/nvdisasm are not needed.
PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 bash scripts/build_sage_local.sh
python3 scripts/profile_sage_vc.py --compare-dual-query-candidates \
  --batch 1 --queries 37741 --tokens 37741 --heads 28 --dim 128 \
  --dtype bfloat16 --warmup 3 --repeats 10 --trials 9 \
  --kernel-samples 20 --kernel-trials 3 \
  --trace-dir results/h200_vc26_traces \
  --output results/h200_vc26_dual_query.json
