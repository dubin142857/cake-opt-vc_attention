#!/usr/bin/env bash
# Reuse the successfully compiled extension; no rebuild, no MindIE reinstall.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/run_ascend910b_accuracy.sh (no arguments needed)" >&2
  exit 2
fi
mkdir -p results
"${PYTHON_BIN:-python3}" scripts/diagnose_ascend910b_accuracy.py --device "${NPU_DEVICE:-0}" \
  --output results/ascend910b_accuracy_diagnosis.json 2>&1 | tee results/ascend910b_accuracy_diagnosis.log
