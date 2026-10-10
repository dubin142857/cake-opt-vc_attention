#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Reuse the dedicated environment; do not install packages or rebuild C++.
tl_python="${TILELANG_PYTHON:-$PWD/.venv-tilelang-h200/bin/python3}"
if [[ ! -x "$tl_python" ]]; then
  printf '%s\n' "Existing TileLang environment missing: $tl_python" >&2
  printf '%s\n' 'Set TILELANG_PYTHON to a prepared TileLang 0.1.15 Python.' >&2
  exit 1
fi
"$tl_python" - <<'PY'
import torch, tilelang
assert tilelang.__version__.split('+')[0] == '0.1.15', tilelang.__version__
assert torch.cuda.is_available(), 'CUDA required'
print('Torch:', torch.__version__, 'TileLang:', tilelang.__version__,
      'CUDA:', torch.version.cuda)
PY
exec "$tl_python" scripts/benchmark_tilelang_opt.py "$@"
