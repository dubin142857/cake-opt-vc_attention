#!/usr/bin/env bash
# Fixed long-sequence VC / MindIE-SD Laser / CANN comparison on Ascend 910B.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/run_ascend910b_laser.sh (no arguments needed)" >&2
  exit 2
fi

# Keep an already selected environment; otherwise source a standard CANN install.
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  for cann_env in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
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
echo "910B comparison: NHD=[1,37741,28,128], BF16, NPU=${NPU_DEVICE}"
echo "Pipeline: environment check -> build/install VC -> validate -> benchmark"
echo "MindIE-SD (mindiesd) must already be installed in this Python environment."

exec bash scripts/run_ascend910b.sh --with-laser --long-shape -- \
  --backends vc_expcast vc_control cann_fusion_attention mindie_laser \
  --dtype bfloat16 --warmup 2 --repeats 3 --trials 5 --check-queries 32 \
  --output results/ascend910b_vs_laser.json
