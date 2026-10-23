#!/usr/bin/env bash
# Restore the remotely validated V-only PV prefetch path using existing artifacts.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/run_mindie_laser_stable.sh (no arguments needed)" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPU_DEVICE="${NPU_DEVICE:-0}"
if [[ ! "$NPU_DEVICE" =~ ^[0-9]+$ ]]; then
  echo "NPU_DEVICE must be a nonnegative index" >&2
  exit 2
fi
for manifest in results/mindie_laser_quant_vscale_build.json \
    results/mindie_laser_quant_vscale_pv_prefetch_build.json \
    results/mindie_laser_quant_preprocess_build.json \
    results/mindie_laser_build.json results/mindie_laser_bridge.json; do
  if [[ ! -f "$manifest" ]]; then
    echo "Missing $manifest; restore the previous successful build artifacts before running." >&2
    echo "This entry only reuses existing builds; it will not rebuild or download sources." >&2
    exit 1
  fi
done
LASER_CANN_ROOT="$("$PYTHON_BIN" -c 'import json; print(json.load(open("results/mindie_laser_quant_vscale_pv_prefetch_build.json"))["cann_root"])')"
if [[ ! -f "$LASER_CANN_ROOT/bin/setenv.bash" ]]; then
  echo "Recorded CANN environment is missing: $LASER_CANN_ROOT/bin/setenv.bash" >&2
  exit 1
fi
set +u
source "$LASER_CANN_ROOT/bin/setenv.bash"
set -u
export ASCEND_HOME_PATH="$LASER_CANN_ROOT"
echo "Restore verified vector_pv_prefetch; reuse existing binaries, no build or download"
echo "Compare original_laser / vector_scale / vector_pv_prefetch; no internal probes or profiler"
echo "BF16 BNSD=[1,28,37741,128]; validate outputs, then 9 interleaved API timing trials"
# The existing runner verifies source/binary provenance before launching its worker.
"$PYTHON_BIN" -u scripts/profile_mindie_laser_pv_prefetch.py \
  --candidate pv_prefetch --shape 1 28 37741 128 --warmup 5 --repeats 10 --trials 9 \
  --device "$NPU_DEVICE" --output results/mindie_laser_stable.json \
  2>&1 | tee results/mindie_laser_stable.log
