#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Compile only in a private checkout. No NPU context or package installation.
: "${SOC_VERSION:?Set the exact preflight compiler SoC, e.g. Ascend910_9382}"
if [[ ! "$SOC_VERSION" =~ ^Ascend910_93[0-9][0-9]$ ]]; then
  echo "Expected an exact A3/910C Ascend910_93xx SoC, got $SOC_VERSION" >&2
  exit 2
fi
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_JOBS="${MAX_JOBS:-2}"
build_dir="ascend/build/h26-${SOC_VERSION}"
output_dir="$PWD/results/ascend910c-h26-build"
mkdir -p "$output_dir"
cmake -S ascend -B "$build_dir" \
  "-DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}" \
  "-DSOC_VERSION=${SOC_VERSION}" \
  "-DPython3_EXECUTABLE=$PYTHON_BIN" \
  "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$output_dir" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --parallel "$MAX_JOBS"
sha256sum "$output_dir/libvc_ascend_ops.so"
