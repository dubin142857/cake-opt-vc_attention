#!/usr/bin/env bash
# Fixed M32/M64 VC / MindIE-SD Laser / CANN diagnostic on DAV-2201 (A2/A3).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/run_ascend910b_profile.sh (no arguments needed)" >&2
  exit 2
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  for cann_env in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
    /usr/local/Ascend/cann/set_env.sh \
    "${HOME}/Ascend/ascend-toolkit/set_env.sh"; do
    if [[ -f "$cann_env" ]]; then
      echo "Loading CANN environment: $cann_env"
      set +u
      source "$cann_env"
      set -u
      break
    fi
  done
fi
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export NPU_DEVICE="${NPU_DEVICE:-0}"
echo "Ascend M32/M64 profile: NHD=[1,37741,28,128], BF16, NPU=${NPU_DEVICE}"
echo "Pipeline: environment -> build -> accuracy gates -> API timings -> instrumented phase ticks"
echo "MindIE-SD must already be installed; probe ticks are NOT production kernel percentages."
exec bash scripts/run_ascend910b.sh --with-laser --long-shape --profile
