#!/usr/bin/env bash
# Compare compile-time disabled PV hooks against the working production body.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/run_mindie_laser_pv_const.sh (no arguments needed)" >&2
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
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path('scripts').resolve()))
import validate_mindie_laser_pv_abi as validation
passed = json.loads(Path('results/mindie_laser_pv_abi.json').read_text(encoding='utf-8'))
if passed.get('status') != 'passed':
    raise SystemExit('Complete the ABI-only small validation first; constant-plain does not replace that check')
args = validation.resolve_defaults(validation.parser().parse_args([]))
args._run_dir = 'results/preflight'
validation.verify_builds(args, validation.resolve_providers(args))
print('Verified production, ABI-only and fused quantizer artifacts; prior ABI-only validation passed', flush=True)
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
echo "Build independent constant-plain: original v2 header/hooks, SetupCubeProbe(trace, 0)"
echo "Only the new provider is built from local Git objects; existing artifacts are reused"
"$PYTHON_BIN" -u scripts/build_mindie_laser_cube_probe.py --kind pv_const \
  --reference-dir "$SOURCE_REFERENCE" --cann-root "$LASER_CANN_ROOT" --soc "$BUILD_SOC" \
  --jobs "$LASER_BUILD_JOBS" --output results/mindie_laser_cube_pv_const_probe_build.json \
  2>&1 | tee results/mindie_laser_cube_pv_const_probe_build.log
echo "Offline symbol-size evidence: constant-plain (probe) vs ABI-only (baseline); no disassembly or device launch"
"$PYTHON_BIN" -u scripts/inspect_mindie_laser_pv_failure.py --symbols-only \
  --probe-build results/mindie_laser_cube_pv_const_probe_build.json \
  --baseline-build results/mindie_laser_cube_pv_abi_probe_build.json \
  --output results/mindie_laser_pv_const_symbols.json \
  2>&1 | tee results/mindie_laser_pv_const_symbols.log
echo "Six small correctness cases only; exact padded output and zero trace; constant mode 0"
"$PYTHON_BIN" -u scripts/validate_mindie_laser_pv_abi.py --kind pv_const \
  --device "$NPU_DEVICE" --output results/mindie_laser_pv_const.json \
  2>&1 | tee results/mindie_laser_pv_const.log
