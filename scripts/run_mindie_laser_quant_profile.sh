#!/usr/bin/env bash
# Reuse the last successful INT8 Laser / fused-preprocessor build. No compilation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/run_mindie_laser_quant_profile.sh (no arguments needed)" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPU_DEVICE="${NPU_DEVICE:-0}"
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  for cann_env in /usr/local/Ascend/ascend-toolkit/set_env.sh \
      /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
      /usr/local/Ascend/cann/set_env.sh "${HOME}/Ascend/ascend-toolkit/set_env.sh"; do
    if [[ -f "$cann_env" ]]; then
      set +u
      source "$cann_env"
      set -u
      break
    fi
  done
fi
: "${ASCEND_HOME_PATH:?Source the CANN Toolkit set_env.sh first}"
if [[ -f "$ASCEND_HOME_PATH/bin/setenv.bash" ]]; then
  set +u
  source "$ASCEND_HOME_PATH/bin/setenv.bash"
  set -u
fi
for manifest in results/mindie_laser_quant_build.json results/mindie_laser_quant_preprocess_build.json; do
  if [[ ! -f "$manifest" ]]; then
    echo "Missing $manifest; first complete bash scripts/run_mindie_laser_quant.sh" >&2
    exit 1
  fi
done
mkdir -p results
echo "Reuse existing INT8 Laser binaries; no rebuild or package installation"
echo "BF16 BNSD=[1,28,37741,128]; original exp, FP16 PV; no V smoothing/reordering"
echo "One process, paired order, 9 trials; optional NPU profiler runs AFTER API timing"
"$PYTHON_BIN" -u scripts/profile_mindie_laser_quant.py \
  --quant-build results/mindie_laser_quant_build.json \
  --preprocess-build results/mindie_laser_quant_preprocess_build.json \
  --shape 1 28 37741 128 --warmup 5 --repeats 10 --trials 9 --trace \
  --device "$NPU_DEVICE" --output results/mindie_laser_quant_profile.json \
  2>&1 | tee results/mindie_laser_quant_profile.log
