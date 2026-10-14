#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPU_DEVICE="${NPU_DEVICE:-0}"
skip_build=0
long_shape=0
laser_args=()
benchmark_args=()
while (($#)); do
  case "$1" in
    --skip-build) skip_build=1; shift ;;
    --long-shape) long_shape=1; shift ;;
    --with-laser) laser_args=(--require-laser); shift ;;
    --) shift; benchmark_args=("$@"); break ;;
    *) echo "Usage: bash scripts/run_ascend910b.sh [--skip-build] [--long-shape] [--with-laser] [-- benchmark options]" >&2; exit 2 ;;
  esac
done
mkdir -p results
"$PYTHON_BIN" scripts/check_ascend910b.py --device "$NPU_DEVICE" "${laser_args[@]}" --output results/ascend910b_environment.json
if [[ -z "${SOC_VERSION:-}" ]]; then
  SOC_VERSION="$("$PYTHON_BIN" -c 'import json; print(json.load(open("results/ascend910b_environment.json"))["compiler_soc"])')"
fi
export SOC_VERSION
if (( !skip_build )); then
  "$PYTHON_BIN" -m pip install ./ascend --no-build-isolation --no-deps -v 2>&1 | tee results/ascend910b_build.log
fi
"$PYTHON_BIN" scripts/validate_ascend910b.py --device "$NPU_DEVICE" 2>&1 | tee results/ascend910b_validation.log
shape_args=(--queries 1024 --keys 1024 --heads 2 --dim 128)
if ((${#laser_args[@]})); then
  shape_args=(--queries 4096 --keys 4096 --heads 2 --dim 128)
fi
if ((long_shape)); then
  shape_args=(--queries 37741 --keys 37741 --heads 28 --dim 128)
fi
"$PYTHON_BIN" scripts/benchmark_ascend910b.py --device "$NPU_DEVICE" "${shape_args[@]}" "${laser_args[@]}" "${benchmark_args[@]}" 2>&1 | tee results/ascend910b_benchmark.log
