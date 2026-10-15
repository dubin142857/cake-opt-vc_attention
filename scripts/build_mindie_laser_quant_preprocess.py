#!/usr/bin/env python3
"""Incremental direct-launch build of one fused A2/A3 Q/K quantization kernel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

try:
    from .build_cann_fas_local import read_cache, run_visible
except ImportError:
    from build_cann_fas_local import read_cache, run_visible

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ascend/laser_quant_preprocess"
SOURCE_NAMES = ("CMakeLists.txt", "quantize_contract.h", "quantize_kernel.cpp", "quantize_torch.cpp")
LIBRARY_NAME = "libvc_laser_quant_preprocess.so"
CONTRACT_VERSION = 1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def runtime_details():
    if sys.platform != "linux":
        raise ValueError("Build on the remote Linux machine with CANN, torch and torch_npu installed")
    import torch
    import torch_npu
    return {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "python_executable": sys.executable, "python": sys.version,
            "torch_root": str(Path(torch.__file__).resolve().parent),
            "torch_npu_root": str(Path(torch_npu.__file__).resolve().parent),
            "torch_abi": int(torch._C._GLIBCXX_USE_CXX11_ABI)}


def check_cache(path, definitions):
    path = Path(path)
    if not path.exists():
        return
    cache = read_cache(path)
    for key, value in definitions.items():
        if key not in cache or cache[key] != str(value):
            raise ValueError(f"Existing cache mismatch for {key}; preserve it and use another --build-dir")
    for key, value in cache.items():
        if value and ("TOOLCHAIN_FILE" in key or "COMPILER_LAUNCHER" in key or
                      (key.startswith("CMAKE_PROJECT") and "INCLUDE" in key)):
            raise ValueError(f"Unsupported CMake toolchain override {key}; build directory preserved")


def build(cann_root, soc, build_dir, jobs, output, *, runtime=None, runner=None):
    report = {"status": "running", "contract_version": CONTRACT_VERSION,
              "torch_api": "torch.ops.vc_laser_quant_preprocess.quantize",
              "scope": "Pure AIV fused per-128-token quantization; no attention or OPP package rebuild"}
    output = Path(output).resolve()
    write_json(output, report)
    try:
        if not re.fullmatch(r"Ascend910(?:B(?:[1-4]|2C)|_93[0-9]{2})", soc):
            raise ValueError("--soc requires the exact A2/A3 runtime SoC, e.g. Ascend910_9382")
        if jobs < 1:
            raise ValueError("--jobs must be positive")
        for key in ("CMAKE_TOOLCHAIN_FILE", "CMAKE_PROJECT_TOP_LEVEL_INCLUDES",
                    "CMAKE_CXX_COMPILER_LAUNCHER"):
            if os.environ.get(key):
                raise ValueError(f"Unsupported inherited CMake override {key}")
        runtime = runtime_details() if runtime is None else runtime
        runner = run_visible if runner is None else runner
        cann = Path(cann_root).resolve()
        build_dir = Path(build_dir).resolve()
        definitions = {
            "ASCEND_CANN_PACKAGE_PATH": str(cann), "SOC_VERSION": soc,
            "Python3_EXECUTABLE": runtime["python_executable"],
            "CMAKE_HOME_DIRECTORY": str(SOURCE), "CMAKE_GENERATOR": "Unix Makefiles",
            "CMAKE_BUILD_TYPE": "Release", "VC_TORCH_ROOT": runtime["torch_root"],
            "VC_TORCH_NPU_ROOT": runtime["torch_npu_root"],
            "VC_TORCH_VERSION": runtime["torch"], "VC_TORCH_NPU_VERSION": runtime["torch_npu"],
            "VC_TORCH_ABI": str(runtime["torch_abi"]),
        }
        check_cache(build_dir / "CMakeCache.txt", definitions)
        source_files = [{"path": str(SOURCE / name), "sha256": sha(SOURCE / name)} for name in SOURCE_NAMES]
        report.update(soc=soc, cann_root=str(cann), build_dir=str(build_dir),
                      runtime=runtime, definitions=definitions, source_files=source_files)
        write_json(output, report)
        build_dir.mkdir(parents=True, exist_ok=True)
        configure_definitions = {k: v for k, v in definitions.items()
                                 if k not in ("CMAKE_HOME_DIRECTORY", "CMAKE_GENERATOR")}
        configure = ["cmake", "-S", str(SOURCE), "-B", str(build_dir), "-G", "Unix Makefiles",
                     *[f"-D{k}={v}" for k, v in configure_definitions.items()]]
        compile_ = ["cmake", "--build", str(build_dir), "--parallel", str(jobs)]
        report["commands"] = [configure, compile_]
        write_json(output, report)
        runner(configure, ROOT, os.environ.copy(), "quant_preprocess_configure")
        runner(compile_, ROOT, os.environ.copy(), "quant_preprocess_build")
        library = build_dir / LIBRARY_NAME
        if not library.is_file():
            raise RuntimeError("Build did not produce fused preprocessing library")
        if any(sha(record["path"]) != record["sha256"] for record in source_files):
            raise RuntimeError("Native sources changed during compilation; rerun the incremental build")
        report.update(status="passed", library_path=str(library), library_sha256=sha(library))
        write_json(output, report)
        return report
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        write_json(output, report)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cann-root", "--toolkit", dest="cann_root", type=Path, required=True)
    result.add_argument("--soc", required=True)
    result.add_argument("--build-dir", type=Path, default=ROOT / "build/mindie_laser_quant_preprocess")
    result.add_argument("--jobs", type=int, default=4)
    result.add_argument("--output", type=Path, default=ROOT / "results/mindie_laser_quant_preprocess_build.json")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        report = build(args.cann_root, args.soc, args.build_dir, args.jobs, args.output)
        print("Fused Q/K preprocessing:", report["library_path"], flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
