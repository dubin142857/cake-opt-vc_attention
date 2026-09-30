#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Local package build also runs the full Hopper correctness checks.
bash scripts/build_sage_local.sh
python3 scripts/benchmark_probability.py --trials 9 --output results/h200_vc17_probability.json
python3 scripts/profile_sage_vc.py --expcast-only --compare-probability-candidates --warmup 3 --repeats 10 --trials 9 --kernel-samples 20 --kernel-trials 3 --trace-dir results/h200_vc17_traces --output results/h200_vc17_attention.json
python3 scripts/inspect_probability_codegen.py --output results/h200_vc17_codegen.json
