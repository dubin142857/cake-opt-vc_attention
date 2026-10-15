#!/usr/bin/env bash
# Build just the isolated MindIE Laser baseline, then compare with installed Laser.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/run_mindie_laser_local.sh (no arguments needed)" >&2
  exit 2
fi
ROOT="$(pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPU_DEVICE="${NPU_DEVICE:-0}"
LASER_BUILD_JOBS="${LASER_BUILD_JOBS:-4}"
if [[ ! "$LASER_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "LASER_BUILD_JOBS must be a positive integer" >&2
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
for tool in git cmake g++ make; do
  command -v "$tool" >/dev/null || { echo "Missing build tool: $tool" >&2; exit 1; }
done
mkdir -p results build
echo "MindIE Laser only: unchanged algorithm, independent VcLaserAttention registration"
echo "BF16 NHD=[1,37741,28,128]; installed Laser vs local source Laser"
"$PYTHON_BIN" scripts/check_ascend910b.py --device "$NPU_DEVICE" --require-laser \
  --output results/mindie_laser_environment.json
BUILD_SOC="$("$PYTHON_BIN" - <<'PY'
import json
d=json.load(open('results/mindie_laser_environment.json'))
soc=d['runtime_soc']
if soc.startswith('Ascend910_93'):
    print('ascend910_93')
elif soc.startswith('Ascend910B'):
    print('ascend910b')
else:
    raise SystemExit('Laser source baseline requires A2/A3: '+soc)
PY
)"
SOURCE_DIR="$ROOT/build/mindie_laser_source"
SOURCE_MANIFEST="$ROOT/build/mindie_laser_source_manifest.json"
"$PYTHON_BIN" -u scripts/prepare_mindie_laser_source.py \
  --source-dir "$SOURCE_DIR" --manifest "$SOURCE_MANIFEST" \
  2>&1 | tee results/mindie_laser_prepare.log
"$PYTHON_BIN" -u scripts/build_mindie_laser_local.py \
  --source-dir "$SOURCE_DIR" --cann-root "$ASCEND_HOME_PATH" --soc "$BUILD_SOC" \
  --jobs "$LASER_BUILD_JOBS" --install-dir "$ROOT/build/mindie_laser_install" \
  --output results/mindie_laser_build.json \
  2>&1 | tee -a results/mindie_laser_build.log
"$PYTHON_BIN" -u scripts/build_mindie_laser_bridge.py \
  --source-dir "$SOURCE_DIR" --cann-root "$ASCEND_HOME_PATH" \
  --jobs "$LASER_BUILD_JOBS" --build-dir "$ROOT/build/mindie_laser_bridge" \
  --output results/mindie_laser_bridge.json \
  2>&1 | tee -a results/mindie_laser_bridge.log
VENDOR_ROOT="$("$PYTHON_BIN" - <<'PY'
import json
print(json.load(open('results/mindie_laser_build.json'))['vendor_root'])
PY
)"
"$PYTHON_BIN" -u scripts/benchmark_mindie_laser_local.py \
  --vendor-root "$VENDOR_ROOT" --bridge-path "$ROOT/build/mindie_laser_bridge/libvc_laser_bridge.so" \
  --source-manifest "$SOURCE_MANIFEST" --build-manifest "$ROOT/results/mindie_laser_build.json" \
  --device "$NPU_DEVICE" --output results/mindie_laser_local_compare.json \
  2>&1 | tee results/mindie_laser_compare.log
