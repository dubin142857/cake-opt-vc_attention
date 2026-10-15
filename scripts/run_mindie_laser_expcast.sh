#!/usr/bin/env bash
# Compare isolated original Laser and its experimental probability transform.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/run_mindie_laser_expcast.sh (no arguments needed)" >&2
  exit 2
fi
ROOT="$(pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPU_DEVICE="${NPU_DEVICE:-0}"
LASER_BUILD_JOBS="${LASER_BUILD_JOBS:-4}"
LASER_PHASE_PROBE="${LASER_PHASE_PROBE:-0}"
if [[ ! "$LASER_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "LASER_BUILD_JOBS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$LASER_PHASE_PROBE" =~ ^[01]$ ]]; then
  echo "LASER_PHASE_PROBE must be 0 or 1" >&2
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
echo "MindIE Laser: original baseline vs experimental bulk probability ExpCast"
echo "BF16 BNSD=[1,28,37741,128]; independent sources, installations and runtime processes"
"$PYTHON_BIN" scripts/check_ascend910b.py --device "$NPU_DEVICE" --require-laser \
  --output results/mindie_laser_expcast_environment.json
BUILD_SOC="$("$PYTHON_BIN" - <<'PY'
import json
with open('results/mindie_laser_expcast_environment.json', encoding='utf-8') as stream:
    soc = json.load(stream)['runtime_soc']
if soc.startswith('Ascend910_93'):
    print('ascend910_93')
elif soc.startswith('Ascend910B'):
    print('ascend910b')
else:
    raise SystemExit('Laser ExpCast comparison requires A2/A3: ' + soc)
PY
)"
BASELINE_SOURCE="$ROOT/build/mindie_laser_source"

build_variant() {
  local variant="$1"
  local stem="$2"
  local source_dir="$ROOT/build/${stem}_source"
  local source_manifest="$ROOT/build/${stem}_source_manifest.json"
  local prepare_args=(--variant "$variant" --source-dir "$source_dir" --manifest "$source_manifest")
  if [[ "$variant" != "baseline" ]]; then
    # Git objects from the pinned baseline are sufficient; never fetch again
    # or copy its generated custom operator/build cache into another variant.
    prepare_args+=(--reference-dir "$BASELINE_SOURCE" --offline)
  fi
  "$PYTHON_BIN" -u scripts/prepare_mindie_laser_source.py "${prepare_args[@]}" \
    2>&1 | tee "results/${stem}_prepare.log"
  "$PYTHON_BIN" -u scripts/build_mindie_laser_local.py \
    --source-dir "$source_dir" --source-manifest "$source_manifest" \
    --cann-root "$ASCEND_HOME_PATH" --soc "$BUILD_SOC" --jobs "$LASER_BUILD_JOBS" \
    --install-dir "$ROOT/build/${stem}_install" --output "results/${stem}_build.json" \
    2>&1 | tee -a "results/${stem}_build.log"
}

build_variant baseline mindie_laser
build_variant expcast mindie_laser_expcast
compare_args=(
  --baseline-build results/mindie_laser_build.json
  --expcast-build results/mindie_laser_expcast_build.json
  --bridge-manifest results/mindie_laser_bridge.json
  --device "$NPU_DEVICE"
  --output results/mindie_laser_expcast_compare.json
)
if [[ "$LASER_PHASE_PROBE" == "1" ]]; then
  echo "Build optional instrumented phase probes in two additional directories"
  build_variant baseline_probe mindie_laser_baseline_probe
  build_variant expcast_probe mindie_laser_expcast_probe
  compare_args+=(
    --baseline-probe-build results/mindie_laser_baseline_probe_build.json
    --expcast-probe-build results/mindie_laser_expcast_probe_build.json
  )
fi
"$PYTHON_BIN" -u scripts/build_mindie_laser_bridge.py \
  --source-dir "$BASELINE_SOURCE" --cann-root "$ASCEND_HOME_PATH" \
  --jobs "$LASER_BUILD_JOBS" --build-dir "$ROOT/build/mindie_laser_bridge" \
  --output results/mindie_laser_bridge.json \
  2>&1 | tee -a results/mindie_laser_bridge.log
"$PYTHON_BIN" -u scripts/benchmark_mindie_laser_expcast.py "${compare_args[@]}" \
  2>&1 | tee results/mindie_laser_expcast_compare.log
