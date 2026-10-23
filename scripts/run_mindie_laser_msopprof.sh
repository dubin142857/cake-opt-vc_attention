#!/usr/bin/env bash
# Read existing builds and collect hardware counters; no compilation/download.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/run_mindie_laser_msopprof.sh (no arguments needed)" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPU_DEVICE="${NPU_DEVICE:-0}"
LASER_MSOPPROF_EXPCAST="${LASER_MSOPPROF_EXPCAST:-0}"
if [[ ! "$NPU_DEVICE" =~ ^[0-9]+$ ]]; then
  echo "NPU_DEVICE must be a physical nonnegative device index" >&2
  exit 2
fi
if [[ "$LASER_MSOPPROF_EXPCAST" != 0 && "$LASER_MSOPPROF_EXPCAST" != 1 ]]; then
  echo "LASER_MSOPPROF_EXPCAST must be 0 or 1" >&2
  exit 2
fi
LASER_CANN_ROOT="$("$PYTHON_BIN" -c 'import json; print(json.load(open("results/mindie_laser_quant_vscale_pv_prefetch_build.json"))["cann_root"])')"
if [[ ! -f "$LASER_CANN_ROOT/bin/setenv.bash" ]]; then
  echo "Recorded CANN environment is missing: $LASER_CANN_ROOT/bin/setenv.bash" >&2
  exit 1
fi
export ASCEND_HOME_PATH="$LASER_CANN_ROOT"
set +u
source "$LASER_CANN_ROOT/bin/setenv.bash"
set -u
export ASCEND_HOME_PATH="$LASER_CANN_ROOT"
echo "msOpProf: original FP16 Laser and verified INT8 vector_pv_prefetch; BF16 BNSD=[1,28,37741,128]"
echo "One native call per provider; kernel-name filtered, profiler warmup=5, Default counter groups"
variants=(original_laser vector_pv_prefetch)
if [[ "$LASER_MSOPPROF_EXPCAST" == 1 ]]; then
  variants+=(fp16_expcast_control fp16_expcast)
fi
"$PYTHON_BIN" -u scripts/collect_mindie_laser_msopprof.py \
  --device "$NPU_DEVICE" --shape 1 28 37741 128 \
  --variants "${variants[@]}" \
  --output results/mindie_laser_msopprof.json
