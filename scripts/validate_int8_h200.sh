#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# == 0 ]]; then
  PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 bash scripts/build_sage_local.sh
elif [[ $# == 1 && "$1" == "--skip-build" ]]; then
  python3 scripts/validate_sage_local.py --output results/sage_local_validation.json
  python3 scripts/validate_sage_vc.py --output results/sage_vc_validation.json
else
  echo "Usage: bash scripts/validate_int8_h200.sh [--skip-build]" >&2
  exit 2
fi
# No FP8 QK entry is invoked. CUPTI is optional in the Python profiler.
python3 scripts/profile_int8_h200.py --output results/h200_int8_main.json
