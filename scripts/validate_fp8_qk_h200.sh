#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# == 0 ]]; then
  PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 bash scripts/build_sage_local.sh
elif [[ $# == 1 && "$1" == "--skip-build" ]]; then
  python3 scripts/validate_sage_vc.py --output results/sage_vc_validation.json
else
  echo "Usage: bash scripts/validate_fp8_qk_h200.sh [--skip-build]" >&2
  exit 2
fi
python3 scripts/validate_fp8_qk.py --output results/h200_vc27_validation.json
python3 scripts/profile_fp8_qk.py --output results/h200_vc27_fp8.json --trace-dir results/h200_vc27_traces
