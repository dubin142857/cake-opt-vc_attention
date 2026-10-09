#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Uses the installed vc32 package. Only the isolated phase probe is JIT compiled.
# No installation/download, V clustering or V smoothing.
python3 scripts/profile_expcast_h200.py --kernel-phases "$@"
