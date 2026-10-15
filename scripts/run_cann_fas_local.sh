#!/usr/bin/env bash
# Rebuild official CANN FlashAttentionScore before changing its mathematics.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (($#)); then
  echo "Usage: bash scripts/run_cann_fas_local.sh (no arguments needed)" >&2
  exit 2
fi
ROOT="$(pwd -P)"
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
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPU_DEVICE="${NPU_DEVICE:-0}"
CANN_FAS_JOBS="${CANN_FAS_JOBS:-8}"
if [[ ! "$CANN_FAS_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CANN_FAS_JOBS must be a positive integer" >&2
  exit 2
fi
for command_name in git cmake g++ make; do
  command -v "$command_name" >/dev/null || { echo "Missing build tool: $command_name" >&2; exit 1; }
done
mkdir -p results build
echo "Official CANN v9.0.1 FlashAttentionScore source reproduction"
echo "BF16 NHD=[1,37741,28,128]; original softmax, no VC modifications"
echo "First build fetches pinned upstream sources and third-party dependencies if not cached."
"$PYTHON_BIN" scripts/check_ascend910b.py --device "$NPU_DEVICE" --output results/cann_fas_environment.json
BUILD_SOC="$("$PYTHON_BIN" - <<'PY'
import json
d=json.load(open('results/cann_fas_environment.json'))
soc=d['runtime_soc']
if soc.startswith('Ascend910_93'):
    print('ascend910_93')
elif soc.startswith('Ascend910B'):
    print('ascend910b')
else:
    raise SystemExit('Official source reproduction supports only A2/A3 DAV-2201: '+soc)
PY
)"
# The upstream build takes a chip FAMILY, unlike our direct-invoke extension.
SOURCE_DIR="$ROOT/build/cann_fas_source"
SOURCE_MANIFEST="$ROOT/build/cann_fas_source_manifest.json"
INSTALL_DIR="$ROOT/build/cann_fas_install"
VENDOR_ROOT="$INSTALL_DIR/vendors/vc_fas_transformer"
"$PYTHON_BIN" scripts/prepare_cann_fas_source.py --source-dir "$SOURCE_DIR" --manifest "$SOURCE_MANIFEST" \
  2>&1 | tee results/cann_fas_prepare.log
echo "Compile official single-op custom package for $BUILD_SOC"
(
  source "$ROOT/scripts/cann_fas_build_compat.sh"
  cd "$SOURCE_DIR"
  bash build.sh --pkg --soc="$BUILD_SOC" --ops=flash_attention_score \
    --vendor_name=vc_fas --cann_3rd_lib_path="$SOURCE_DIR/third_party" \
    --build-type=Release --disable_asan -j"$CANN_FAS_JOBS" -O3
) 2>&1 | tee results/cann_fas_build.log

shopt -s nullglob
packages=("$SOURCE_DIR"/build_out/cann-ops-transformer-vc_fas_linux*.run)
if ((${#packages[@]} != 1)); then
  echo "Expected exactly one vc_fas package; found ${#packages[@]}. See results/cann_fas_build.log" >&2
  exit 1
fi
mkdir -p "$INSTALL_DIR"
# Only install inside this workspace. Do not source its environment in the parent:
# the benchmark deliberately selects providers in separate clean processes.
bash "${packages[0]}" --quiet --install-path="$INSTALL_DIR" 2>&1 | tee results/cann_fas_install.log
if [[ ! -f "$VENDOR_ROOT/op_api/lib/libcust_opapi.so" ]]; then
  echo "Custom op API was not installed at the expected vendor path: $VENDOR_ROOT" >&2
  exit 1
fi
"$PYTHON_BIN" - "$SOURCE_MANIFEST" "${packages[0]}" "$VENDOR_ROOT" "$BUILD_SOC" <<'PY'
from pathlib import Path
import hashlib, json, sys
manifest, package, vendor = (Path(x) for x in sys.argv[1:4])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
d=json.loads(manifest.read_text())
bins=sorted(vendor.glob('op_impl/ai_core/tbe/kernel/**/*.o'))
if not bins:
    raise SystemExit('No compiled device .o files in custom package; refusing a source-only/JIT baseline')
d['build']={'soc_family':sys.argv[4], 'package':str(package), 'package_sha256':sha(package),
            'vendor_root':str(vendor), 'device_binary_count':len(bins),
            'device_binaries_sha256':{str(p.relative_to(vendor)):sha(p) for p in bins},
            'environment':json.loads(Path('results/cann_fas_environment.json').read_text()),
            'scope':'Host API/tiling source markers; unchanged device source. Binary presence alone does not prove runtime kernel selection.'}
manifest.write_text(json.dumps(d,indent=2)+'\n')
Path('results/cann_fas_build_manifest.json').write_text(json.dumps(d,indent=2)+'\n')
print('Packaged device binaries:',len(bins))
PY
"$PYTHON_BIN" scripts/benchmark_cann_fas_local.py --vendor-root "$VENDOR_ROOT" \
  --source-manifest "$ROOT/results/cann_fas_build_manifest.json" --device "$NPU_DEVICE" \
  --output results/cann_fas_local_compare.json 2>&1 | tee results/cann_fas_compare.log
