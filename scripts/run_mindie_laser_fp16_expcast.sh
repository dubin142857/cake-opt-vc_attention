#!/usr/bin/env bash
# Build only the isolated FP16 ExpCast provider from the validated local checkout.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/run_mindie_laser_fp16_expcast.sh (no arguments needed)" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPU_DEVICE="${NPU_DEVICE:-0}"
LASER_BUILD_JOBS="${LASER_BUILD_JOBS:-4}"
if [[ ! "$NPU_DEVICE" =~ ^[0-9]+$ || ! "$LASER_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPU_DEVICE must be nonnegative; LASER_BUILD_JOBS must be positive" >&2
  exit 2
fi
for manifest in results/mindie_laser_quant_vscale_pv_prefetch_build.json \
    results/mindie_laser_quant_preprocess_build.json \
    results/mindie_laser_build.json results/mindie_laser_bridge.json; do
  if [[ ! -f "$manifest" ]]; then
    echo "Missing $manifest; this experiment needs the previous successful Laser builds." >&2
    exit 1
  fi
done
# Host-only checks fail before compiling if the stable artifacts were changed.
"$PYTHON_BIN" - <<'PY'
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path('scripts').resolve()))
import benchmark_mindie_laser_quant as quant
from mindie_laser_quant_fused import validate_preprocess_manifest
from prepare_mindie_laser_quant_vscale_pv_prefetch_source import IDENTITY, validate_prepared_source
args = SimpleNamespace(quant_build='results/mindie_laser_quant_vscale_pv_prefetch_build.json',
    baseline_build='results/mindie_laser_build.json', bridge_manifest='results/mindie_laser_bridge.json',
    preprocess_build='results/mindie_laser_quant_preprocess_build.json',
    _run_dir='results/preflight', device=0, warmup=5, repeats=10, trials=9, cpu_threads=4)
quant.quant_provenance(quant.build_arguments(args, 'qk_int8_fused'),
    identity=IDENTITY, source_validator=validate_prepared_source)
quant.baseline.provenance(quant.build_arguments(args, 'baseline'))
validate_preprocess_manifest(args.preprocess_build)
print('Verified existing V-only PV prefetch, original FP16 Laser and fused quantizer artifacts', flush=True)
PY
LASER_CANN_ROOT="$("$PYTHON_BIN" -c 'import json; print(json.load(open("results/mindie_laser_quant_vscale_pv_prefetch_build.json"))["cann_root"])')"
BUILD_SOC="$("$PYTHON_BIN" -c 'import json; print(json.load(open("results/mindie_laser_quant_vscale_pv_prefetch_build.json"))["soc"])')"
SOURCE_REFERENCE="$("$PYTHON_BIN" -c 'import json; print(json.load(open("results/mindie_laser_quant_vscale_pv_prefetch_build.json"))["source_dir"])')"
if [[ ! -f "$LASER_CANN_ROOT/bin/setenv.bash" ]]; then
  echo "Recorded CANN environment is missing: $LASER_CANN_ROOT/bin/setenv.bash" >&2
  exit 1
fi
export ASCEND_HOME_PATH="$LASER_CANN_ROOT"
set +u
source "$LASER_CANN_ROOT/bin/setenv.bash"
set -u
export ASCEND_HOME_PATH="$LASER_CANN_ROOT"
echo "Build only the independent FP16 ExpCast candidate; reuse existing stable/FP16/quantizer binaries"
echo "Local source cache: $SOURCE_REFERENCE; no upstream download"
"$PYTHON_BIN" -u scripts/build_mindie_laser_quant_vscale_pv_expcast.py \
  --reference-dir "$SOURCE_REFERENCE" --cann-root "$LASER_CANN_ROOT" \
  --soc "$BUILD_SOC" --jobs "$LASER_BUILD_JOBS" \
  --output results/mindie_laser_quant_vscale_pv_expcast_build.json \
  2>&1 | tee results/mindie_laser_quant_vscale_pv_expcast_build.log
echo "FP16 ExpCast: INT8 QK, FP16 probability/PV, no V reordering or smoothing"
echo "Validate first, then compare original Laser / stable / new control / new ExpCast"
echo "BF16 BNSD=[1,28,37741,128]; 9 paired trials, no internal probes or profiler"
"$PYTHON_BIN" -u scripts/profile_mindie_laser_fp16_expcast.py \
  --shape 1 28 37741 128 --warmup 5 --repeats 10 --trials 9 \
  --device "$NPU_DEVICE" --output results/mindie_laser_fp16_expcast.json \
  2>&1 | tee results/mindie_laser_fp16_expcast.log
