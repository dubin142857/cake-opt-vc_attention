#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Self-attention Q/K/V: BSHD [1,37741,28,128], BHND [1,28,37741,128].
# All backends process the full sequence. Only correctness metrics sample Q.
# Extra arguments can override timing/backend/output defaults.
python3 scripts/benchmark_cuda.py \
  --batch 1 --queries 37741 --tokens 37741 --heads 28 --dim 128 \
  --dtype bfloat16 --warmup 2 --repeats 5 --trials 3 \
  --check-queries 128 --reference-chunk 32 \
  --output results/h200_shape_1_37741_28_128.json "$@"
