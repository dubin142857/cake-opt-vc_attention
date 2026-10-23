#!/usr/bin/env bash
# Retired after the instrumented PV kernel failed small-input validation.
set -euo pipefail
echo "PV constant-mode diagnostic retired: its first small-input synchronization timed out." >&2
echo "Use the verified production path: bash scripts/run_mindie_laser_stable.sh" >&2
echo "Existing diagnostic sources and failure reports are retained; nothing was launched." >&2
exit 2
