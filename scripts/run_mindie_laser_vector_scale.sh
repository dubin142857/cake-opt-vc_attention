#!/usr/bin/env bash
# Build only the independent Vector-scale candidate; preserve all baseline caches.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/run_mindie_laser_vector_scale.sh (no arguments needed)" >&2
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
for manifest in results/mindie_laser_quant_qresident_build.json results/mindie_laser_quant_preprocess_build.json \
    results/mindie_laser_build.json results/mindie_laser_bridge.json; do
  if [[ ! -f "$manifest" ]]; then
    echo "Missing $manifest; first complete bash scripts/run_mindie_laser_q_resident.sh" >&2
    exit 1
  fi
done
# Fail before compilation if any baseline/library/source cache is stale.
"$PYTHON_BIN" - "$ASCEND_HOME_PATH" "$NPU_DEVICE" <<'PY'
import ctypes, json, os, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path('scripts').resolve()))
import benchmark_mindie_laser_quant as quant
from mindie_laser_quant_fused import validate_preprocess_manifest
from profile_mindie_laser_quant import verify_runtime_target
from prepare_mindie_laser_quant_qresident_source import IDENTITY, validate_prepared_source
args = SimpleNamespace(quant_build='results/mindie_laser_quant_qresident_build.json',
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
preprocess = json.loads(Path(args.preprocess_build).read_text())
for manifest in (args.quant_build, args.baseline_build):
    verify_runtime_target(json.loads(Path(manifest).read_text()), preprocess, soc, sys.argv[1])
print('Verified existing Q-resident INT8, FP16 and fused-preprocessing builds for', soc, flush=True)
PY
BUILD_SOC="$("$PYTHON_BIN" -c 'import json; print(json.load(open("results/mindie_laser_quant_qresident_build.json"))["soc"])')"
SOURCE_REFERENCE="$("$PYTHON_BIN" -c 'import json; print(json.load(open("results/mindie_laser_quant_qresident_build.json"))["source_dir"])')"
echo "Only build the Vector-scale candidate; existing Q-resident/FP16/quantizer are reused"
echo "Reuse verified source checkout: $SOURCE_REFERENCE (no upstream download)"
"$PYTHON_BIN" -u scripts/build_mindie_laser_quant_vscale.py \
  --reference-dir "$SOURCE_REFERENCE" \
  --cann-root "$ASCEND_HOME_PATH" --soc "$BUILD_SOC" --jobs "$LASER_BUILD_JOBS" \
  --output results/mindie_laser_quant_vscale_build.json \
  2>&1 | tee results/mindie_laser_quant_vscale_build.log
echo "BF16 BNSD=[1,28,37741,128]; INT8 QK fixed Cube transport + Vector scaling experiment; original exp and FP16 PV"
echo "FP16 rounding changes: validate candidate decoded reference and all-output drift before timing"
"$PYTHON_BIN" -u scripts/profile_mindie_laser_scale_reuse.py --experiment vector-scale \
  --shape 1 28 37741 128 --warmup 5 --repeats 10 --trials 9 --trace \
  --device "$NPU_DEVICE" --output results/mindie_laser_vector_scale.json \
  2>&1 | tee results/mindie_laser_vector_scale.log
