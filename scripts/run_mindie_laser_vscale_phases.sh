#!/usr/bin/env bash
# Two isolated Cube diagnostics, using local pinned source and existing baselines.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/run_mindie_laser_vscale_phases.sh (no arguments needed)" >&2
  exit 2
fi
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
mkdir -p results
for manifest in results/mindie_laser_quant_vscale_build.json results/mindie_laser_quant_preprocess_build.json \
    results/mindie_laser_build.json results/mindie_laser_bridge.json; do
  if [[ ! -f "$manifest" ]]; then
    echo "Missing $manifest; existing current Vector-scale, FP16 and fused quantizer builds are required" >&2
    exit 1
  fi
done
"$PYTHON_BIN" - "$ASCEND_HOME_PATH" "$NPU_DEVICE" <<'PY'
import ctypes, json, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path('scripts').resolve()))
import benchmark_mindie_laser_quant as quant
from mindie_laser_quant_fused import validate_preprocess_manifest
from profile_mindie_laser_quant import verify_runtime_target
from mindie_laser_fp16_trace_worker import verify_runtime_target as verify_fp16_target
from prepare_mindie_laser_quant_vscale_source import IDENTITY, validate_prepared_source
args = SimpleNamespace(quant_build='results/mindie_laser_quant_vscale_build.json',
    baseline_build='results/mindie_laser_build.json', bridge_manifest='results/mindie_laser_bridge.json',
    preprocess_build='results/mindie_laser_quant_preprocess_build.json',
    _run_dir='results/preflight', device=int(sys.argv[2]), warmup=5, repeats=10, trials=9, cpu_threads=4)
old = quant.build_arguments(args, 'qk_int8_fused')
fp16 = quant.build_arguments(args, 'baseline')
quant.quant_provenance(old, identity=IDENTITY, source_validator=validate_prepared_source)
quant.baseline.provenance(fp16)
validate_preprocess_manifest(args.preprocess_build)
import torch, torch_npu
torch.npu.set_device(args.device)
acl = ctypes.CDLL('libascendcl.so')
acl.aclrtGetSocName.restype = ctypes.c_char_p
soc = acl.aclrtGetSocName()
if not soc:
    raise RuntimeError('aclrtGetSocName returned null')
soc = soc.decode()
verify_runtime_target(json.loads(Path(args.quant_build).read_text()),
    json.loads(Path(args.preprocess_build).read_text()), soc, sys.argv[1])
verify_fp16_target(json.loads(Path(args.baseline_build).read_text()), soc, sys.argv[1])
print('Verified existing current Vector-scale, original FP16 and fused quantizer for', soc, flush=True)
PY
BUILD_SOC="$("$PYTHON_BIN" -c 'import json; print(json.load(open("results/mindie_laser_quant_vscale_build.json"))["soc"])')"
SOURCE_REFERENCE="$("$PYTHON_BIN" -c 'import json; print(json.load(open("results/mindie_laser_quant_vscale_build.json"))["source_dir"])')"
echo "Validate/reuse diagnostic builds; build current Vector-scale probe and rebuild FP16 probe only if needed, from local source: $SOURCE_REFERENCE"
echo "No upstream download; existing production operators are reused"
for kind in vscale fp16; do
  "$PYTHON_BIN" -u scripts/build_mindie_laser_cube_probe.py \
    --kind "$kind" --reference-dir "$SOURCE_REFERENCE" \
    --cann-root "$ASCEND_HOME_PATH" --soc "$BUILD_SOC" --jobs "$LASER_BUILD_JOBS" \
    --output "results/mindie_laser_cube_${kind}_probe_build.json" \
    2>&1 | tee "results/mindie_laser_cube_${kind}_probe_build.log"
done
echo "BF16 BNSD=[1,28,37741,128]; original exp and FP16 PV, current Vector-scale INT8 and original FP16 Cube diagnostics"
echo "Exact ALL-output gates, paired native calibration, then sparse QK/output and PV/history samples"
"$PYTHON_BIN" -u scripts/profile_mindie_laser_vscale_phases.py \
  --shape 1 28 37741 128 --warmup 5 --repeats 10 --trials 9 --samples 5 \
  --device "$NPU_DEVICE" --output results/mindie_laser_vscale_phases.json \
  2>&1 | tee results/mindie_laser_vscale_phases.log
