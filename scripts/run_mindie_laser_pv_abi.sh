#!/usr/bin/env bash
# Isolate the extended probe ABI with the unchanged production Cube/Vector body.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/run_mindie_laser_pv_abi.sh (no arguments needed)" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPU_DEVICE="${NPU_DEVICE:-0}"
LASER_BUILD_JOBS="${LASER_BUILD_JOBS:-4}"
if [[ ! "$LASER_BUILD_JOBS" =~ ^[1-9][0-9]*$ || ! "$NPU_DEVICE" =~ ^[0-9]+$ ]]; then
  echo "LASER_BUILD_JOBS must be positive and NPU_DEVICE must be a nonnegative index" >&2
  exit 2
fi
mkdir -p results
# Host-only provenance gate before building. Does not import torch or open NPU.
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path('scripts').resolve()))
import benchmark_mindie_laser_quant as quant
from mindie_laser_quant_fused import validate_preprocess_manifest
from prepare_mindie_laser_quant_vscale_pv_prefetch_source import IDENTITY, validate_prepared_source
args = SimpleNamespace(quant_build='results/mindie_laser_quant_vscale_pv_prefetch_build.json',
    preprocess_build='results/mindie_laser_quant_preprocess_build.json',
    _run_dir='results/preflight', device=0, warmup=0, repeats=1, trials=1, cpu_threads=4)
production = quant.build_arguments(args, 'qk_int8_fused')
quant.quant_provenance(production, identity=IDENTITY, source_validator=validate_prepared_source)
validate_preprocess_manifest(args.preprocess_build)
print('Verified existing V-only production and fused-preprocessing artifacts (host checks only)', flush=True)
PY
LASER_CANN_ROOT="$("$PYTHON_BIN" -c 'import json; print(json.load(open("results/mindie_laser_quant_vscale_pv_prefetch_build.json"))["cann_root"])')"
BUILD_SOC="$("$PYTHON_BIN" -c 'import json; print(json.load(open("results/mindie_laser_quant_vscale_pv_prefetch_build.json"))["soc"])')"
SOURCE_REFERENCE="$("$PYTHON_BIN" -c 'import json; print(json.load(open("results/mindie_laser_quant_vscale_pv_prefetch_build.json"))["source_dir"])')"
if [[ ! -f "$LASER_CANN_ROOT/bin/setenv.bash" ]]; then
  echo "Recorded CANN environment is missing: $LASER_CANN_ROOT/bin/setenv.bash" >&2
  exit 1
fi
set +u
source "$LASER_CANN_ROOT/bin/setenv.bash"
set -u
export ASCEND_HOME_PATH="$LASER_CANN_ROOT"
echo "Build only the independent ABI-only provider; production and failed v2 artifacts are preserved"
echo "Original V-only Cube/Vector source; zero trace output; no timing hooks; source cache: $SOURCE_REFERENCE"
"$PYTHON_BIN" -u scripts/build_mindie_laser_cube_probe.py --kind pv_abi \
  --reference-dir "$SOURCE_REFERENCE" --cann-root "$LASER_CANN_ROOT" --soc "$BUILD_SOC" \
  --jobs "$LASER_BUILD_JOBS" --output results/mindie_laser_cube_pv_abi_probe_build.json \
  2>&1 | tee results/mindie_laser_cube_pv_abi_probe_build.log
echo "Small-input ABI isolation only; no long-shape timing and no launch of the failed instrumented v2"
"$PYTHON_BIN" -u scripts/validate_mindie_laser_pv_abi.py \
  --device "$NPU_DEVICE" --output results/mindie_laser_pv_abi.json \
  2>&1 | tee results/mindie_laser_pv_abi.log
