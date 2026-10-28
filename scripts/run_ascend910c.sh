#!/usr/bin/env bash
set -euo pipefail

# H26 is one DAV-2201 kernel for A2/A3. Core count is queried at runtime;
# this entrypoint pins the selected device to the A3/910C SoC family.
if [[ ! "${NPU_DEVICE:-}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "Set NPU_DEVICE to an explicitly coordinated, idle physical davinci device ID." >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec bash scripts/run_ascend910b.sh --require-a3 "$@"
