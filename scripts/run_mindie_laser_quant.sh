#!/usr/bin/env bash
# Original Laser vs native INT8 QK with multi-op or fused Q/K preprocessing.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/run_mindie_laser_quant.sh (no arguments needed)" >&2
  exit 2
fi
ROOT="$(pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPU_DEVICE="${NPU_DEVICE:-0}"
LASER_BUILD_JOBS="${LASER_BUILD_JOBS:-4}"
LASER_QUANT_EXPERIMENTS="${LASER_QUANT_EXPERIMENTS-0}"
if [[ ! "$LASER_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "LASER_BUILD_JOBS must be a positive integer" >&2
  exit 2
fi
if [[ "$LASER_QUANT_EXPERIMENTS" != 0 && "$LASER_QUANT_EXPERIMENTS" != 1 ]]; then
  echo "LASER_QUANT_EXPERIMENTS must be 0 or 1" >&2
  exit 2
fi
VARIANTS=(installed baseline qk_int8 qk_int8_fused)
if [[ "$LASER_QUANT_EXPERIMENTS" == 1 ]]; then
  VARIANTS+=(qkv_int8_e4m3 qkv_int8_e4m3_expcast)
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
echo "Laser quantization: ${VARIANTS[*]}"
echo "Mainline: native INT8 QK + original Laser softmax + FP16 V/PV"
echo "Compare multi-op Q/K preprocessing with the fused Ascend C candidate"
if [[ "$LASER_QUANT_EXPERIMENTS" == 1 ]]; then
  echo "Experiments enabled: E4M3 V decoded to FP16 PV / ExpCast"
fi
echo "BF16 BNSD=[1,28,37741,128]; fresh preprocessing, full API and fixed prepared timings"
echo "A2/A3 FP16 PV; no native FP8, no V reordering, no K smoothing"
"$PYTHON_BIN" scripts/check_ascend910b.py --device "$NPU_DEVICE" --require-laser \
  --output results/mindie_laser_quant_environment.json
RUNTIME_SOC="$("$PYTHON_BIN" - <<'PY'
import json
with open('results/mindie_laser_quant_environment.json', encoding='utf-8') as stream:
    print(json.load(stream)['runtime_soc'])
PY
)"
BUILD_SOC="$("$PYTHON_BIN" - <<'PY'
import json
with open('results/mindie_laser_quant_environment.json', encoding='utf-8') as stream:
    soc = json.load(stream)['runtime_soc']
if soc.startswith('Ascend910_93'):
    print('ascend910_93')
elif soc.startswith('Ascend910B'):
    print('ascend910b')
else:
    raise SystemExit('Quantized Laser comparison requires A2/A3: ' + soc)
PY
)"

# Reuse an already verified original baseline; avoid recompiling it on every
# quantized kernel iteration. The benchmark repeats provenance checks later.
if "$PYTHON_BIN" - "$ASCEND_HOME_PATH" "$BUILD_SOC" <<'PY'
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, 'scripts')
import benchmark_mindie_laser_local as baseline
try:
    build_path = Path('results/mindie_laser_build.json').resolve()
    bridge_path = Path('results/mindie_laser_bridge.json').resolve()
    build = json.loads(build_path.read_text(encoding='utf-8'))
    bridge = json.loads(bridge_path.read_text(encoding='utf-8'))
    args = SimpleNamespace(build_manifest=str(build_path), source_manifest=build['source_manifest']['path'],
                           vendor_root=build['vendor_root'], bridge_manifest=str(bridge_path),
                           bridge_path=bridge['bridge_path'])
    baseline.provenance(args)
    if Path(build['definitions']['CUSTOM_ASCEND_CANN_PACKAGE_PATH']).resolve() != Path(sys.argv[1]).resolve():
        raise ValueError('Existing baseline targets another CANN installation')
    if build['soc'] != sys.argv[2]:
        raise ValueError('Existing baseline targets another SoC')
except (OSError, ValueError, KeyError, TypeError) as error:
    print('Original baseline needs build: ' + str(error), flush=True)
    raise SystemExit(1)
print('Reusing verified original Laser baseline and bridge', flush=True)
PY
then
  :
else
  SOURCE_DIR="$ROOT/build/mindie_laser_source"
  SOURCE_MANIFEST="$ROOT/build/mindie_laser_source_manifest.json"
  "$PYTHON_BIN" -u scripts/prepare_mindie_laser_source.py \
    --source-dir "$SOURCE_DIR" --manifest "$SOURCE_MANIFEST" \
    2>&1 | tee results/mindie_laser_prepare.log
  "$PYTHON_BIN" -u scripts/build_mindie_laser_local.py \
    --source-dir "$SOURCE_DIR" --source-manifest "$SOURCE_MANIFEST" \
    --cann-root "$ASCEND_HOME_PATH" --soc "$BUILD_SOC" --jobs "$LASER_BUILD_JOBS" \
    --install-dir "$ROOT/build/mindie_laser_install" --output results/mindie_laser_build.json \
    2>&1 | tee -a results/mindie_laser_build.log
  "$PYTHON_BIN" -u scripts/build_mindie_laser_bridge.py \
    --source-dir "$SOURCE_DIR" --cann-root "$ASCEND_HOME_PATH" --jobs "$LASER_BUILD_JOBS" \
    --build-dir "$ROOT/build/mindie_laser_bridge" --output results/mindie_laser_bridge.json \
    2>&1 | tee -a results/mindie_laser_bridge.log
fi
"$PYTHON_BIN" -u scripts/build_mindie_laser_quant.py \
  --cann-root "$ASCEND_HOME_PATH" --soc "$BUILD_SOC" --jobs "$LASER_BUILD_JOBS" \
  --output results/mindie_laser_quant_build.json \
  2>&1 | tee results/mindie_laser_quant_build.log
"$PYTHON_BIN" -u scripts/build_mindie_laser_quant_preprocess.py \
  --cann-root "$ASCEND_HOME_PATH" --soc "$RUNTIME_SOC" --jobs "$LASER_BUILD_JOBS" \
  --output results/mindie_laser_quant_preprocess_build.json \
  2>&1 | tee results/mindie_laser_quant_preprocess_build.log
"$PYTHON_BIN" -u scripts/benchmark_mindie_laser_quant.py \
  --baseline-build results/mindie_laser_build.json \
  --quant-build results/mindie_laser_quant_build.json \
  --preprocess-build results/mindie_laser_quant_preprocess_build.json \
  --bridge-manifest results/mindie_laser_bridge.json \
  --device "$NPU_DEVICE" --output results/mindie_laser_quant_compare.json \
  --variants "${VARIANTS[@]}" \
  2>&1 | tee results/mindie_laser_quant_compare.log
