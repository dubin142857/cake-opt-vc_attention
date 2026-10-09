#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Dedicated venv reuses the installed Torch/Sage/FA4 and existing CUDA toolkit.
# Use TILELANG_PYTHON=/path/to/python3 to use a prepared environment instead.
base_python="${PYTHON:-python3}"
skip_install=0
if [[ "${1:-}" == "--skip-install" ]]; then skip_install=1; shift; fi
if [[ -n "${TILELANG_PYTHON:-}" ]]; then
  tl_python="$TILELANG_PYTHON"
else
  if [[ ! -x .venv-tilelang-h200/bin/python3 ]]; then
    "$base_python" -m venv --system-site-packages .venv-tilelang-h200
  fi
  tl_python="$PWD/.venv-tilelang-h200/bin/python3"
fi
if [[ "$skip_install" == 0 ]]; then
  constraints="$(mktemp)"
  trap 'rm -f -- "$constraints"' EXIT
  "$tl_python" - "$constraints" <<'PY'
import sys,torch
from pathlib import Path
Path(sys.argv[1]).write_text('torch=='+torch.__version__+'\n')
print('Keeping existing Torch:',torch.__version__)
PY
  "$tl_python" -m pip install --constraint "$constraints" 'tilelang==0.1.15'
  rm -f -- "$constraints"
  trap - EXIT
fi
"$tl_python" - <<'PY'
import torch,tilelang
from sageattention_local import vc
assert tilelang.__version__.split('+')[0]=='0.1.15',tilelang.__version__
assert torch.cuda.is_available(),'CUDA required'
print('Torch:',torch.__version__,'TileLang:',tilelang.__version__,'VC native:',vc._extension().sage_vc_kernel_version)
PY
exec "$tl_python" scripts/benchmark_paper_expcast.py "$@"
