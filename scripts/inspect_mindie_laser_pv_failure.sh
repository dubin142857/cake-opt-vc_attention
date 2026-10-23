#!/usr/bin/env bash
# Inspect existing failure logs and compiled artifacts; never launch the NPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p results
echo "Offline PV failure inspection: existing logs/binaries only; no build, device launch, or download"
"$PYTHON_BIN" -u scripts/inspect_mindie_laser_pv_failure.py "$@" \
  2>&1 | tee results/mindie_laser_pv_failure.log
