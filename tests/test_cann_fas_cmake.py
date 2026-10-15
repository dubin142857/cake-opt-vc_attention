"""Exercise dependency pruning with real CMake; no CANN or C++ compiler needed."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


MODULE = Path(__file__).resolve().parents[1] / "scripts/cann_fas_acl_only.cmake"


def run(*args):
    return subprocess.run(list(map(str, args)), text=True, encoding="utf-8",
                          errors="replace", capture_output=True)


@pytest.fixture
def cmake_tools():
    cmake = shutil.which("cmake")
    # NINJA provides an optional explicit path for Windows developer shells.
    ninja = os.environ.get("NINJA") or shutil.which("ninja")
    if not cmake or not ninja:
        pytest.skip("CMake >= 3.19 and Ninja are needed for the build-graph tests")
    return cmake, ninja


def configure(tmp_path, tools, *, extra="", flags=()):
    source = tmp_path / "source"
    build = tmp_path / "build"
    source.mkdir()
    # Custom targets provide observable stand-ins for the upstream object and
    # runtime libraries, without requiring a CANN/C++ toolchain on CI.
    (source / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.19)
project(cann_ops-transformer LANGUAGES NONE)
set(BUILD_OPEN_PROJECT ON)
set(ASCEND_OP_NAME flash_attention_score)
add_custom_target(abseil_build_transformer
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/abseil.built)
add_custom_target(ascend_protobuf_build_transformer
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/protobuf.built)
add_dependencies(ascend_protobuf_build_transformer abseil_build_transformer)
add_custom_target(op_transformer_onnx_plugin_obj ALL
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/onnx.built)
add_dependencies(op_transformer_onnx_plugin_obj ascend_protobuf_build_transformer)
foreach(name IN ITEMS cust_opapi cust_proto cust_opmaster ops_transformer_kernel)
  add_custom_target(${name} ALL
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/${name}.built)
endforeach()
file(WRITE ${CMAKE_BINARY_DIR}/runtime.info "core package\\n")
install(FILES ${CMAKE_BINARY_DIR}/runtime.info DESTINATION .)
""" + extra + "\nreturn()\n",
        encoding="utf-8",
    )
    cmake, ninja = tools
    result = run(cmake, "-S", source, "-B", build, "-G", "Ninja",
                 f"-DCMAKE_MAKE_PROGRAM={Path(ninja).as_posix()}",
                 f"-DCMAKE_PROJECT_INCLUDE={MODULE.as_posix()}", *flags)
    return build, result


def build_ok(cmake, build, *args):
    result = run(cmake, "--build", build, *args)
    assert result.returncode == 0, result.stdout + result.stderr


def test_default_build_skips_onnx_but_builds_and_installs_core(tmp_path, cmake_tools):
    build, result = configure(tmp_path, cmake_tools)
    assert result.returncode == 0, result.stdout + result.stderr
    build_ok(cmake_tools[0], build)
    for name in ("cust_opapi", "cust_proto", "cust_opmaster", "ops_transformer_kernel"):
        assert (build / f"{name}.built").exists()
    for name in ("onnx", "protobuf", "abseil"):
        assert not (build / f"{name}.built").exists()
    result = run(cmake_tools[0], "--install", build, "--prefix", tmp_path / "install")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "install/runtime.info").exists()
    # Explicitly requested ONNX still works; its dependency graph is intact.
    build_ok(cmake_tools[0], build, "--target", "op_transformer_onnx_plugin_obj")
    assert all((build / f"{name}.built").exists() for name in ("onnx", "protobuf", "abseil"))


def test_needed_core_dependency_is_preserved(tmp_path, cmake_tools):
    build, result = configure(tmp_path, cmake_tools, extra=
        "add_dependencies(cust_opapi ascend_protobuf_build_transformer)")
    assert result.returncode == 0, result.stdout + result.stderr
    build_ok(cmake_tools[0], build)
    assert (build / "protobuf.built").exists()
    assert (build / "abseil.built").exists()
    assert not (build / "onnx.built").exists()


@pytest.mark.parametrize("extra", [
    "set(ENABLE_BUILT_IN ON)",
    "set(ENABLE_TEST ON)",
    "set(ENABLE_STATIC ON)",
    "set(BUILD_OPS_RTY_KERNEL ON)",
    "set(ASCEND_OP_NAME ALL)",
    "add_custom_target(op_transformer_onnx_plugin)",
])
def test_refuses_unreviewed_build_modes(tmp_path, cmake_tools, extra):
    _, result = configure(tmp_path, cmake_tools, extra=extra)
    assert result.returncode != 0
    assert "CANN FAS" in result.stdout + result.stderr
