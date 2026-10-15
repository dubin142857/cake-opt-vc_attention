"""Exercise CMake -> Bash preparation arguments, without CANN or an NPU.

The regression was an unset CHECK_COMPATIBLE in execute_process: CMake omits
the empty unquoted value, so Bash consumes the next option as its value and
silently stops parsing at the following SoC. These tests run both actual tools.
"""
from pathlib import Path
import os
import shutil
import subprocess

import pytest

from scripts import build_mindie_laser_local as driver


ROOT = Path(__file__).resolve().parents[1]
PIN = "71098c20ff90838d1fd1ca0a40515c751a3869b0"
PREPARE_PATH = "csrc/ops/cmake/scripts/prepare.sh"
CONFIG_PATH = "csrc/ops/cmake/config.cmake"

# Offline fallback excerpts from the pinned files above. When the inspected
# checkout exists, git show reads these sections directly from its pinned blob.
# Keep the original argument consumption and unknown-option behavior verbatim.
# Upstream excerpts: Copyright (c) Huawei Technologies Co., Ltd. 2026-2026.
# All rights reserved. MindIE is licensed under Mulan PSL v2:
# http://license.coscl.org.cn/MulanPSL2
# Provided AS IS, without warranties including non-infringement,
# merchantability or fitness for a particular purpose. See that license.
PARSER_EXCERPT = r'''while [[ $# -gt 0 ]]; do
    case $1 in
    -s)
        PATH_TO_SOURCE="$2"
        shift 2
        ;;
    -b)
        PATH_TO_BUILD="$2"
        shift 2
        ;;
    -p)
        ASCEND_CANN_PACKAGE_PATH="$2"
        shift 2
        ;;
    --autogen-dir)
        ASCEND_AUTOGEN_DIR="$2"
        shift 2
        ;;
    --build-open-project)
        BUILD_OPEN_PROJECT="$2"
        shift 2
        ;;
    --binary-out-dir)
        ASCEND_BINARY_OUT_DIR="$2"
        shift 2
        ;;
    --impl-out-dir)
        ASCEND_IMPL_OUT_DIR="$2"
        shift 2
        ;;
    --op-build-tool)
        OP_BUILD_TOOL="$2"
        shift 2
        ;;
    --ascend-cmake-dir)
        ASCEND_CMAKE_DIR="$2"
        shift 2
        ;;
    --tiling-key)
        TILING_KEY="$2"
        shift 2
        ;;
    --ops-compile-options)
        OPS_COMPILE_OPTIONS="$2"
        shift 2
        ;;
    --check-compatible)
        CHECK_COMPATIBLE="$2"
        shift 2
        ;;
    --ascend-compute_unit)
        ASCEND_COMPUTE_UNIT="$2"
        shift 2
        ;;
    --ascend-op-name)
        ASCEND_OP_NAME="$2"
        shift 2
        ;;
    --op_debug_config)
        OP_DEBUG_CONFIG="$2"
        shift 2
        ;;
    *)
        break
        ;;
    esac
done'''

CALL_EXCERPT = r'''execute_process(COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/prepare.sh
                -s ${CMAKE_CURRENT_SOURCE_DIR}
                -b ${CMAKE_CURRENT_BINARY_DIR}/prepare_build
                -p ${ASCEND_CANN_PACKAGE_PATH}
                --autogen-dir ${ASCEND_AUTOGEN_DIR}
                --build-open-project ${BUILD_OPEN_PROJECT}
                --binary-out-dir ${ASCEND_BINARY_OUT_DIR}
                --impl-out-dir ${ASCEND_IMPL_OUT_DIR}
                --op-build-tool ${OP_BUILD_TOOL}
                --ascend-cmake-dir ${ASCEND_CMAKE_DIR}
                --tiling-key ${EP_TILING_KEY}
                --ops-compile-options ${EP_OPS_COMPILE_OPTIONS}
                --check-compatible ${CHECK_COMPATIBLE}
                --ascend-compute_unit ${EP_ASCEND_COMPUTE_UNIT}
                --op_debug_config ${OP_DEBUG_CONFIG}
                --ascend-op-name "${ASCEND_OP_NAME}"
                RESULT_VARIABLE result
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE PREPARE_BUILD_OUTPUT_VARIABLE)'''


def pinned_sections():
    for reference in (ROOT / "build/mindie-sd-reference", ROOT / "build/mindie_laser_source"):
        if not (reference / ".git").exists():
            continue
        blobs = []
        for path in (PREPARE_PATH, CONFIG_PATH):
            result = subprocess.run(["git", "-C", str(reference), "show", f"{PIN}:{path}"],
                                    capture_output=True, check=False)
            if result.returncode:
                break
            blobs.append(result.stdout.decode("utf-8"))
        if len(blobs) == 2:
            prepare, config = blobs
            parser = prepare[prepare.index("while [[ $# -gt 0 ]]"):prepare.index("\nfunction clean()")].strip()
            start = config.index("execute_process(COMMAND bash")
            end_marker = "OUTPUT_VARIABLE PREPARE_BUILD_OUTPUT_VARIABLE)"
            call = config[start:config.index(end_marker, start) + len(end_marker)]
            return parser, call
    return PARSER_EXCERPT, CALL_EXCERPT


@pytest.fixture
def invoke_prepare(tmp_path):
    cmake = shutil.which("cmake")
    bash = str(Path("C:/Program Files/Git/bin/bash.exe")) if os.name == "nt" else shutil.which("bash")
    if not cmake or not bash or not Path(bash).is_file():
        pytest.skip("Regression integration test requires CMake and Bash (Git Bash on Windows)")
    parser, call = pinned_sections()
    shell = tmp_path / "prepare-arguments.sh"
    # Do not call upstream main/clean/build: execute the actual parser only.
    shell.write_text(parser + '\nprintf "CHECK=%s\\nSOC=%s\\nOP=%s\\n" '
                     '"$CHECK_COMPATIBLE" "$ASCEND_COMPUTE_UNIT" "$ASCEND_OP_NAME"\n',
                     encoding="utf-8", newline="\n")
    call = call.replace("COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/prepare.sh",
                        'COMMAND "${TEST_BASH}" "${TEST_PREPARE_SCRIPT}"', 1)
    cmake_script = tmp_path / "call-prepare.cmake"
    cmake_script.write_text('''
set(ASCEND_CANN_PACKAGE_PATH "stub-cann")
set(ASCEND_AUTOGEN_DIR "stub-autogen")
set(ASCEND_BINARY_OUT_DIR "stub-binary")
set(ASCEND_IMPL_OUT_DIR "stub-impl")
set(OP_BUILD_TOOL "stub-op-build")
set(ASCEND_CMAKE_DIR "stub-cmake")
set(EP_TILING_KEY "FALSE")
set(EP_OPS_COMPILE_OPTIONS "FALSE")
set(EP_ASCEND_COMPUTE_UNIT "${ASCEND_COMPUTE_UNIT}")
set(OP_DEBUG_CONFIG "false")
''' + call + '''
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Prepare parser process failed: ${result}")
endif()
file(WRITE "${CAPTURE_PATH}" "${PREPARE_BUILD_OUTPUT_VARIABLE}")
''', encoding="utf-8", newline="\n")
    capture = tmp_path / "arguments.txt"

    def invoke(definitions):
        command = [cmake, *[f"-D{key}={value}" for key, value in definitions.items()],
                   f"-DTEST_BASH={Path(bash).as_posix()}",
                   f"-DTEST_PREPARE_SCRIPT={shell.as_posix()}",
                   f"-DCAPTURE_PATH={capture.as_posix()}", "-P", str(cmake_script)]
        # No ambient values may mask the missing option in the baseline case.
        env = {key: value for key, value in os.environ.items()
               if key not in ("CHECK_COMPATIBLE", "ASCEND_COMPUTE_UNIT", "ASCEND_OP_NAME", "BASH_ENV", "ENV")}
        completed = subprocess.run(command, env=env, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=30)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return dict(line.split("=", 1) for line in capture.read_text(encoding="utf-8").splitlines())

    return invoke


def test_empty_cmake_compatibility_value_silently_discards_soc_and_operator(invoke_prepare, tmp_path):
    definitions = driver.definitions(tmp_path / "source", tmp_path / "cann", "ascend910_93", tmp_path / "install")
    definitions.pop("CHECK_COMPATIBLE", None)
    result = invoke_prepare(definitions)
    assert result == {"CHECK": "--ascend-compute_unit", "SOC": "", "OP": ""}


def test_driver_preserves_requested_soc_and_operator_through_real_cmake_bash(invoke_prepare, tmp_path):
    definitions = driver.definitions(tmp_path / "source", tmp_path / "cann", "ascend910_93", tmp_path / "install")
    result = invoke_prepare(definitions)
    assert result == {"CHECK": "TRUE", "SOC": "ascend910_93", "OP": "vc_laser_attention"}
