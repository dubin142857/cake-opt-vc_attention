#!/usr/bin/env bash
# Offline only: summarize an existing capture; never launch or rebuild an op.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/summarize_mindie_laser_msopprof.sh (no arguments needed)" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "Offline summary: existing files only; no NPU, profiler, build, or download"
"$PYTHON_BIN" scripts/summarize_mindie_laser_msopprof.py \
  --report results/mindie_laser_msopprof.json
