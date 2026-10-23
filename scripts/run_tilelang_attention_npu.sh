#!/usr/bin/env bash
# One-command ordinary attention benchmark. Its environment/cache is independent.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/run_tilelang_attention_npu.sh (no arguments needed)" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPU_DEVICE="${NPU_DEVICE:-0}"
if [[ ! "$NPU_DEVICE" =~ ^[0-9]+$ ]]; then
  echo "NPU_DEVICE must be a nonnegative integer" >&2
  exit 2
fi
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
# Setup installs only the official pinned Ascend backend into its own venv.
# TILELANG_NPU_PYTHON can select an existing compatible backend; setup validates it.
source scripts/setup_tilelang_attention_npu.sh
: "${TILELANG_NPU_PYTHON:?TileLang NPU setup did not export TILELANG_NPU_PYTHON}"
cache_root="$(pwd)/.cache/tilelang-attention-npu/runtime"
export TILELANG_ATTENTION_CACHE_ROOT="$cache_root"
# The benchmark adds the real SoC, pinned backend commit and CANN manifest hash
# before loading TileLang, preventing reuse across incompatible compiler setups.
export TILELANG_CACHE_DIR="$cache_root/tilelang"
export ASCEND_CACHE_PATH="$cache_root/ascend"
export XDG_CACHE_HOME="$cache_root/xdg"
export ACL_OP_COMPILER_CACHE_DIR="$cache_root/compiler"
export TL_RUN_MODE=npu
mkdir -p "$TILELANG_CACHE_DIR" "$ASCEND_CACHE_PATH" "$XDG_CACHE_HOME" "$ACL_OP_COMPILER_CACHE_DIR" results
echo "Ordinary noncausal attention: BF16 BNSD=[1,28,37741,128], NPU=$NPU_DEVICE"
echo "Validate all small outputs and sampled long Q against independent FP32 CPU oracle before timing"
echo "Reuse verified calls for calibration/warmup; plan at most 10s/provider, except one non-preemptible NPU call"
"$TILELANG_NPU_PYTHON" -u scripts/benchmark_tilelang_attention_npu.py \
  --device "$NPU_DEVICE" --shape 1 28 37741 128 --dtype bfloat16 \
  --warmup 1 --repeats 5 --trials 3 --target-trial-ms 200 --timing-budget-seconds 10 \
  --output results/tilelang_attention_npu.json \
  2>&1 | tee results/tilelang_attention_npu.log
