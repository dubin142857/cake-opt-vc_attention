#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${FP16_EXPCAST_PYTHON:-}" ]]; then
  exec "$FP16_EXPCAST_PYTHON" scripts/check_fp16_expcast.py "$@"
fi

for expcast_python in \
  "$PWD/.venv-tilelang-opt/Scripts/python.exe" \
  "$PWD/.venv-tilelang-h200/bin/python3" \
  python3 python; do
  if "$expcast_python" -c 'import numpy' >/dev/null 2>&1; then
    exec "$expcast_python" scripts/check_fp16_expcast.py "$@"
  fi
done

printf '%s\n' 'NumPy is required; set FP16_EXPCAST_PYTHON to a Python with NumPy installed.' >&2
exit 1
