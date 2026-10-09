#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Existing Torch, sageattention-local, FA4 and nvcc only; no installation.
# Separate report/trace defaults; original vc28 focus script stays available.
exec "${PYTHON:-python3}" scripts/profile_expcast_h200.py \
  --expcast-backend balanced_max --probability-detail --trials 7 --phase-trials 5 "$@"
