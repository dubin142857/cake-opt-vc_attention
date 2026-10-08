#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# --skip-build reuses an already installed vc26 binary. Correctness checks
# still run and must pass before timing. ncu/nvdisasm are not needed.
if [[ $# == 0 ]]; then
  PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 bash scripts/build_sage_local.sh
elif [[ $# == 1 && "$1" == "--skip-build" ]]; then
  python3 scripts/validate_sage_vc.py --output results/sage_vc_validation.json
else
  echo "Usage: bash scripts/validate_dual_query_h200.sh [--skip-build]" >&2
  exit 2
fi
python3 scripts/profile_sage_vc.py --compare-dual-query-candidates \
  --batch 1 --queries 37741 --tokens 37741 --heads 28 --dim 128 \
  --dtype bfloat16 --warmup 3 --repeats 10 --trials 9 \
  --kernel-samples 20 --kernel-trials 3 \
  --trace-dir results/h200_vc26_traces \
  --output results/h200_vc26_dual_query.json
