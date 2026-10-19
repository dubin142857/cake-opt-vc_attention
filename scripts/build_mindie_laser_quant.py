#!/usr/bin/env python3
"""Build the isolated native INT8-QK Laser operator and its torch_npu bridge.

No torch import occurs in this driver. Runtime metadata is discovered in a
short-lived subprocess; the benchmark loads the resulting library separately.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = {
    "op_name": "vc_laser_quant_attention",
    "vendor": "vc_laser_quant",
    "op_type": "VcLaserQuantAttention",
    "native_api": "aclnnVcLaserQuantAttention",
}
MARKER_TOKEN = "vc_mindie_laser_quant_v1"
MARKER_ENVIRONMENT = "VC_MINDIE_LASER_QUANT_MARKER_FILE"
BRIDGE_SOURCE = ROOT / "ascend/laser_quant/bridge"
BRIDGE_NAME = "libvc_laser_quant_bridge.so"
CMAKE_GENERATOR = "Unix Makefiles"
PATH_CACHE_KEYS = {
    "CMAKE_HOME_DIRECTORY", "CANN_ROOT", "TORCH_ROOT", "TORCH_NPU_ROOT",
    "VC_LASER_ADAPTER_DIR", "Python3_EXECUTABLE",
}
TOOLCHAIN_KEYS = {
    "CMAKE_TOOLCHAIN_FILE", "CMAKE_C_COMPILER_LAUNCHER", "CMAKE_CXX_COMPILER_LAUNCHER",
    "CMAKE_USER_MAKE_RULES_OVERRIDE", "CMAKE_USER_MAKE_RULES_OVERRIDE_CXX",
    "CMAKE_USER_MAKE_RULES_OVERRIDE_C", "CMAKE_PROJECT_TOP_LEVEL_INCLUDES",
}
ADAPTER_FILES = (
    "csrc/plugin/pytorch_npu_helper.h",
    "csrc/plugin/find_op_path.cpp",
    "csrc/plugin/find_op_path.h",
    "LICENSE.md",
)


def _module(name):
    return importlib.import_module(("scripts." if __package__ else "") + name)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_bytes(source, *arguments):
    return subprocess.run(["git", "-C", str(source), *arguments], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def copy_pinned_adapter(source, destination):
    """Copy Git object bytes, never generated or locally modified adapter files."""
    pinned = _module("prepare_mindie_laser_source").SOURCE_LOCK.commit
    actual = git_bytes(source, "rev-parse", "HEAD").decode("ascii").strip()
    if actual != pinned:
        raise ValueError(f"Quantized Laser bridge requires pinned source {pinned}, got {actual}")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    for relative in ADAPTER_FILES:
        data = git_bytes(source, "show", f"{pinned}:{relative}")
        target = destination / Path(relative).name
        if not target.exists() or target.read_bytes() != data:
            target.write_bytes(data)
        records[relative] = {
            "path": str(target.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
        }
    return {"upstream_commit": pinned, "files": records}


def discover_runtime():
    code = (
        "import json,pathlib,sys,torch,torch_npu; "
        "print(json.dumps({'torch_root':str(pathlib.Path(torch.__file__).resolve().parent),"
        "'torch_npu_root':str(pathlib.Path(torch_npu.__file__).resolve().parent),"
        "'torch_abi':int(torch._C._GLIBCXX_USE_CXX11_ABI),"
        "'python_executable':sys.executable,'torch_version':str(torch.__version__),"
        "'torch_npu_version':str(torch_npu.__version__)}))"
    )
    result = subprocess.run([sys.executable, "-c", code], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Startup diagnostics may precede the final JSON line.
    lines = result.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError("torch/torch_npu runtime discovery returned no metadata")
    metadata = json.loads(lines[-1])
    if metadata.get("torch_abi") not in (0, 1):
        raise ValueError("Unsupported PyTorch C++ ABI value")
    for key in ("torch_root", "torch_npu_root", "python_executable"):
        metadata[key] = str(Path(metadata[key]).resolve())
    return metadata


def check_cmake_cache(cache, expected):
    """Refuse a build directory configured for another source/runtime/toolkit."""
    cache = Path(cache)
    if not cache.is_file():
        return
    values = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        key, value = line.split("=", 1)
        values[key.split(":", 1)[0]] = value
    reject_toolchain_overrides(values, f"bridge CMake cache {cache}")
    for key, wanted in expected.items():
        if key not in values:
            raise ValueError(f"Existing bridge CMake cache lacks {key}; use a fresh bridge directory: {cache}")
        cached = values[key]
        equal = cached == str(wanted)
        if key in PATH_CACHE_KEYS:
            equal = Path(cached).resolve() == Path(wanted).resolve()
        if not equal:
            raise ValueError(f"Bridge CMake cache mismatch for {key}: {cached!r} != {str(wanted)!r}; "
                             f"use a separate bridge directory: {cache}")


def reject_toolchain_overrides(values, context):
    """Do not silently reuse a configured launcher/toolchain or CMake hook."""
    for key, value in values.items():
        project_hook = key.startswith("CMAKE_PROJECT_") and key.endswith(("_INCLUDE", "_INCLUDE_BEFORE"))
        if value and (key in TOOLCHAIN_KEYS or project_hook):
            raise ValueError(f"Unsupported toolchain override {key}={value!r} in {context}; "
                             "use the isolated bridge's declared build configuration")


def build_bridge(source, cann, bridge_dir, jobs, output, *, runner=None, runtime=None,
                 identity=None, torch_namespace="vc_laser_quant", bridge_source=None,
                 bridge_name=None):
    identity = dict(IDENTITY if identity is None else identity)
    bridge_source = BRIDGE_SOURCE if bridge_source is None else Path(bridge_source).resolve()
    bridge_name = BRIDGE_NAME if bridge_name is None else bridge_name
    source, cann, bridge_dir, output = (Path(item).resolve() for item in (source, cann, bridge_dir, output))
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if runner is None:
        runner = _module("build_cann_fas_local").run_visible
    report = {
        "status": "running", "native_api": identity["native_api"], "op_type": identity["op_type"],
        "bridge_dir": str(bridge_dir), "upstream_source_dir": str(source),
        "cann_root": str(cann), "torch_namespace": torch_namespace,
        "torch_schema": "la(Tensor q, Tensor k, Tensor v, Tensor q_scale, Tensor k_scale, float scale, int keys, bool expcast) -> Tensor",
        "steps": [],
    }
    write_json(output, report)
    try:
        environment = os.environ.copy()
        reject_toolchain_overrides(environment, "bridge build environment")
        if environment.get("CMAKE_GENERATOR", CMAKE_GENERATOR) != CMAKE_GENERATOR:
            raise ValueError(f"Bridge requires CMAKE_GENERATOR={CMAKE_GENERATOR!r}")
        metadata = discover_runtime() if runtime is None else dict(runtime)
        adapter_dir = bridge_dir / "adapter"
        expected = {
            "CMAKE_HOME_DIRECTORY": str(bridge_source.resolve()),
            "CMAKE_GENERATOR": CMAKE_GENERATOR,
            "CMAKE_BUILD_TYPE": "Release",
            "CANN_ROOT": str(cann), "TORCH_ROOT": metadata["torch_root"],
            "TORCH_NPU_ROOT": metadata["torch_npu_root"], "TORCH_ABI": str(metadata["torch_abi"]),
            "VC_LASER_ADAPTER_DIR": str(adapter_dir),
            "Python3_EXECUTABLE": metadata["python_executable"],
        }
        check_cmake_cache(bridge_dir / "CMakeCache.txt", expected)
        report["runtime"] = metadata
        report["adapter"] = copy_pinned_adapter(source, adapter_dir)
        report["bridge_sources"] = [{"path": str(path), "sha256": sha256_file(path)}
                                    for path in (bridge_source / "register.cpp", bridge_source / "CMakeLists.txt")]
        definitions = {key: value for key, value in expected.items()
                       if key not in ("CMAKE_HOME_DIRECTORY", "CMAKE_GENERATOR")}
        configure = ["cmake", "-S", str(bridge_source), "-B", str(bridge_dir), "-G", CMAKE_GENERATOR]
        configure += [f"-D{key}={value}" for key, value in definitions.items()]
        commands = [
            ("quant bridge configure", configure),
            ("quant bridge compile", ["cmake", "--build", str(bridge_dir), "--parallel", str(jobs)]),
        ]
        for phase, command in commands:
            elapsed = runner(command, ROOT, environment, phase)
            report["steps"].append({"phase": phase, "command": command, "elapsed_seconds": elapsed})
            write_json(output, report)
        binary = bridge_dir / bridge_name
        if not binary.is_file() or binary.stat().st_size == 0:
            raise RuntimeError(f"Quantized Laser bridge build did not produce {binary}")
        report.update(status="passed", bridge_path=str(binary), bridge_sha256=sha256_file(binary))
        write_json(output, report)
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_json(output, report)
        raise


def build(source, reference, source_manifest, cann, soc, jobs, install_dir, bridge_dir, output,
          *, preparer=None, source_validator=None, operator_builder=None, bridge_builder=None,
          runner=None, identity=None, marker_token=None, marker_environment=None):
    """Compose preparation, existing operator builder and isolated bridge build."""
    identity = dict(IDENTITY if identity is None else identity)
    marker_token = MARKER_TOKEN if marker_token is None else marker_token
    marker_environment = MARKER_ENVIRONMENT if marker_environment is None else marker_environment
    source, reference, source_manifest, cann, install_dir, bridge_dir, output = (
        Path(item).resolve() for item in
        (source, reference, source_manifest, cann, install_dir, bridge_dir, output))
    if jobs < 1:
        raise ValueError("jobs must be positive")
    operator_output = output.with_name(output.stem + "_operator.json")
    bridge_output = output.with_name(output.stem + "_bridge.json")
    report = {"status": "running", **identity, "cann_root": str(cann), "soc": soc, "jobs": jobs,
              "source_dir": str(source), "install_dir": str(install_dir),
              "source_manifest": {"path": str(source_manifest)},
              "operator_manifest": str(operator_output), "bridge_manifest": str(bridge_output),
              "marker_token": marker_token, "marker_environment": marker_environment}
    write_json(output, report)
    try:
        if identity != IDENTITY and (preparer is None or source_validator is None or bridge_builder is None):
            raise ValueError("An independent operator requires explicit preparation, source validation and bridge builders")
        if preparer is None or source_validator is None:
            preparation = _module("prepare_mindie_laser_quant_source")
            preparer = preparation.prepare if preparer is None else preparer
            source_validator = preparation.validate_prepared_source if source_validator is None else source_validator
        if operator_builder is None:
            operator_builder = _module("build_mindie_laser_local").build
        if bridge_builder is None:
            bridge_builder = build_bridge
        preparer(source, reference, source_manifest)
        source_validator(source, source_manifest)
        report["source_manifest"]["sha256"] = sha256_file(source_manifest)
        write_json(output, report)
        kwargs = {"source_manifest": source_manifest, "identity": identity,
                  "source_validator": source_validator}
        if runner is not None:
            kwargs["runner"] = runner
        native = operator_builder(source, cann, soc, jobs, install_dir, operator_output, **kwargs)
        if native.get("native_api") != identity["native_api"] or native.get("op_type") != identity["op_type"]:
            raise ValueError("Native build manifest does not identify the quantized Laser operator")
        report.update(native)
        # Keep the combined report running until bridge verification succeeds.
        report.update(status="running", **identity,
                      source_manifest={"path": str(source_manifest), "sha256": sha256_file(source_manifest)},
                      operator_manifest=str(operator_output), bridge_manifest=str(bridge_output),
                      marker_token=marker_token, marker_environment=marker_environment)
        write_json(output, report)
        bridge_kwargs = {} if runner is None else {"runner": runner}
        bridge = bridge_builder(source, cann, bridge_dir, jobs, bridge_output, **bridge_kwargs)
        if native.get("status") != "passed" or bridge.get("status") != "passed":
            raise RuntimeError("Quantized Laser operator and bridge must both pass before publishing the combined manifest")
        if bridge.get("native_api") != identity["native_api"] or bridge.get("op_type") != identity["op_type"]:
            raise ValueError("Quantized Laser bridge identity does not match the operator")
        binary = Path(bridge["bridge_path"]).resolve()
        actual_hash = sha256_file(binary)
        if actual_hash != bridge.get("bridge_sha256"):
            raise ValueError("Quantized Laser bridge hash differs from its build manifest")
        report.update(status="passed", bridge_path=str(binary), bridge_sha256=actual_hash,
                      bridge_manifest_sha256=sha256_file(bridge_output))
        write_json(output, report)
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_json(output, report)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cann-root", type=Path, required=True)
    result.add_argument("--soc", required=True)
    result.add_argument("--jobs", type=int, default=4)
    result.add_argument("--source-dir", type=Path, default=ROOT / "build/mindie_laser_quant_source")
    result.add_argument("--reference-dir", type=Path, default=ROOT / "build/mindie-sd-reference")
    result.add_argument("--source-manifest", type=Path, default=ROOT / "build/mindie_laser_quant_source_manifest.json")
    result.add_argument("--install-dir", type=Path, default=ROOT / "build/mindie_laser_quant_install")
    result.add_argument("--bridge-dir", type=Path, default=ROOT / "build/mindie_laser_quant_bridge")
    result.add_argument("--output", type=Path, default=ROOT / "results/mindie_laser_quant_build.json")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        report = build(arguments.source_dir, arguments.reference_dir, arguments.source_manifest,
                       arguments.cann_root, arguments.soc, arguments.jobs, arguments.install_dir,
                       arguments.bridge_dir, arguments.output)
    except Exception:
        traceback.print_exc()
        print(f"Failure evidence: {arguments.output.resolve()}", file=sys.stderr, flush=True)
        return 1
    print(f"Quantized Laser vendor: {report['vendor_root']}", flush=True)
    print(f"Quantized Laser bridge: {report['bridge_path']}", flush=True)
    print(f"Manifest: {arguments.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
