#!/usr/bin/env bash
# Source this file to export the verified NPU Python. Caller positional
# arguments deliberately do not become bootstrap arguments.
_tilelang_attention_npu_setup() {
  local setup_root setup_python setup_report selected_python
  setup_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
  setup_python="${PYTHON_BIN:-python3}"
  setup_report="$setup_root/results/tilelang_attention_npu_setup.json"
  unset TL_ROOT TVM_IMPORT_PYTHON_PATH TVM_LIBRARY_PATH TVM_PREBUILD_PATH TVM_SOURCE_DIR
  unset TL_TEMPLATE_PATH TL_CUTLASS_PATH TL_COMPOSABLE_KERNEL_PATH SKIP_LOADING_TILELANG_SO PYPI_BUILD
  export ACL_OP_INIT_MODE=1
  export PYTHONNOUSERSITE=1
  export TL_RUN_MODE=npu
  "$setup_python" "$setup_root/scripts/setup_tilelang_attention_npu.py" || return 1
  selected_python="$("$setup_python" -c 'import json,sys; r=json.load(open(sys.argv[1], encoding="utf-8")); assert r["status"] == "ready"; print(r["python_executable"])' "$setup_report")" || return 1
  export TILELANG_NPU_PYTHON="$selected_python"
}
if _tilelang_attention_npu_setup; then
  unset -f _tilelang_attention_npu_setup
else
  unset -f _tilelang_attention_npu_setup
  return 1 2>/dev/null || exit 1
fi
