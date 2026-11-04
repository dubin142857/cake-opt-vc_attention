#!/usr/bin/env python3
"""Build the pinned official FAS package without upstream build.sh's clean step."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "scripts/cann_fas_acl_only.cmake"


def read_cache(path):
    result = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith(("#", "//")) or "=" not in line:
                continue
            key, value = line.split("=", 1)
            result[key.split(":", 1)[0]] = value
    return result


def definitions(source, cann, soc):
    # Equivalent to the prior build.sh --pkg/--ops/--vendor/Release/-O3 flags.
    return {
        "BUILD_OPEN_PROJECT": "ON", "VENDOR_NAME": "vc_fas",
        "ASCEND_OP_NAME": "flash_attention_score", "ASCEND_COMPUTE_UNIT": soc,
        "ENABLE_BUILT_IN": "OFF", "ENABLE_OPS_HOST": "ON", "ENABLE_OPS_KERNEL": "ON",
        "ENABLE_BUILD_PKG": "ON", "ENABLE_TILING_SINK": "OFF", "ENABLE_UT_EXEC": "TRUE",
        "ENABLE_TEST": "OFF", "ENABLE_STATIC": "OFF", "ENABLE_EXPERIMENTAL": "OFF",
        "ENABLE_ASAN": "OFF", "CMAKE_BUILD_TYPE": "Release", "CMAKE_BUILD_MODE": "-O3",
        "CHECK_COMPATIBLE": "TRUE", "CANN_3RD_LIB_PATH": str(source / "third_party"),
        "CUSTOM_ASCEND_CANN_PACKAGE_PATH": str(cann), "CMAKE_PROJECT_INCLUDE": str(HOOK),
    }


def validate_cache(cache, source, wanted):
    """Never silently reuse another op/SoC/vendor/toolkit build directory."""
    paths = {"CMAKE_HOME_DIRECTORY": str(source),
             "CUSTOM_ASCEND_CANN_PACKAGE_PATH": wanted["CUSTOM_ASCEND_CANN_PACKAGE_PATH"],
             "CANN_3RD_LIB_PATH": wanted["CANN_3RD_LIB_PATH"]}
    for key, value in paths.items():
        if key in cache and Path(cache[key]).resolve() != Path(value).resolve():
            raise ValueError(f"Existing {key}={cache[key]} differs from {value}; build directory preserved")
    for key in ("VENDOR_NAME", "ASCEND_OP_NAME", "ASCEND_COMPUTE_UNIT", "CMAKE_BUILD_TYPE", "CMAKE_BUILD_MODE"):
        if key in cache and cache[key].rstrip(";") != wanted[key]:
            raise ValueError(f"Existing {key}={cache[key]} differs from {wanted[key]}; build directory preserved")
    for key in ("ENABLE_BUILT_IN", "ENABLE_TEST", "ENABLE_STATIC", "ENABLE_EXPERIMENTAL", "ENABLE_ASAN"):
        if key in cache and cache[key].upper() not in ("OFF", "FALSE", "0", "NO", ""):
            raise ValueError(f"Existing {key}={cache[key]} is incompatible; build directory preserved")
    for key in ("BUILD_OPEN_PROJECT", "ENABLE_OPS_HOST", "ENABLE_OPS_KERNEL", "ENABLE_BUILD_PKG"):
        if key in cache and cache[key].upper() not in ("ON", "TRUE", "1", "YES"):
            raise ValueError(f"Existing {key}={cache[key]} is incompatible; build directory preserved")
    previous_hook = cache.get("CMAKE_PROJECT_INCLUDE")
    if previous_hook and Path(previous_hook).resolve() != HOOK.resolve():
        raise ValueError("Existing CMAKE_PROJECT_INCLUDE is not ours; build directory preserved")
    if cache.get("CMAKE_GENERATOR", "Unix Makefiles") != "Unix Makefiles":
        raise ValueError("Only the upstream Unix Makefiles generator is supported")


def build_environment(base, source, cann):
    env = dict(base)
    arch = platform.machine()
    opp = env.get("ASCEND_OPP_PATH", str(cann / "opp"))
    env.update(BASE_PATH=str(source), BUILD_PATH=str(source / "build"),
               ASCEND_HOME_PATH=str(cann), ASCEND_OPP_PATH=opp,
               INCLUDE_PATH=str(cann / "include"), ACLNN_INCLUDE_PATH=str(cann / "include/aclnn"),
               COMPILER_INCLUDE_PATH=str(cann / "include"), GRAPH_INCLUDE_PATH=str(cann / "include/graph"),
               GE_INCLUDE_PATH=str(cann / "include/ge"), INC_INCLUDE_PATH=str(Path(opp) / "built-in/op_proto/inc"),
               LINUX_INCLUDE_PATH=str(cann / f"{arch}-linux/include"),
               EAGER_LIBRARY_OPP_PATH=str(Path(opp) / "lib64"), EAGER_LIBRARY_PATH=str(cann / "lib64"),
               GRAPH_LIBRARY_STUB_PATH=str(cann / "lib64/stub"), GRAPH_LIBRARY_PATH=str(cann / "lib64"),
               EAGER_INCLUDE_OPP_ACLNNOP_PATH=str(cann / f"{arch}-linux/include/aclnnop"))
    compiler = shutil.which("bisheng", path=env.get("PATH"))
    if not compiler:
        raise ValueError("bisheng not found after CANN bin/setenv.bash; source the complete Toolkit")
    env["BISHENG_REAL_PATH"] = compiler
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (str(source / "build"), env.get("LD_LIBRARY_PATH"))))
    return env


def run_visible(command, cwd, env, phase):
    """Keep real compiler output and a heartbeat; interrupt our own process group."""
    print(f"[{phase}] " + " ".join(command), flush=True)
    started, stop = time.monotonic(), threading.Event()
    process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=(os.name == "posix"))

    def heartbeat():
        while not stop.wait(30):
            print(f"[{phase}] process {process.pid} alive, elapsed {time.monotonic()-started:.0f}s; "
                  "this is a heartbeat, not a progress estimate", flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except KeyboardInterrupt:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
        raise
    finally:
        stop.set()
        thread.join(timeout=1)
    return time.monotonic() - started


def build(source, cann, soc, jobs, output, runner=run_visible):
    source, cann, output = Path(source).resolve(), Path(cann).resolve(), Path(output).resolve()
    build_dir = source / "build"
    cache = read_cache(build_dir / "CMakeCache.txt")
    wanted = definitions(source, cann, soc)
    report = {"status": "running", "source_dir": str(source), "build_dir": str(build_dir),
              "resume_existing_cache": bool(cache), "deletes_build_directory": False,
              "soc": soc, "jobs": jobs, "definitions": wanted, "steps": [],
              "scope": "Original FAS device/API/tiling sources; ONNX-only build targets excluded from ALL"}

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name(output.name + ".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temp.replace(output)

    save()
    try:
        if jobs < 1 or soc not in ("ascend910b", "ascend910_93"):
            raise ValueError("Positive jobs and an A2/A3 SoC family are required")
        validate_cache(cache, source, wanted)
        if not HOOK.is_file() or not (source / "CMakeLists.txt").is_file():
            raise ValueError("CMake source or ACLNN-only hook is missing")
        env = build_environment(os.environ, source, cann)
        report["hook_sha256"] = hashlib.sha256(HOOK.read_bytes()).hexdigest()
        # CMake tracks changes to an included hook itself after the first configure.
        generated = ("Makefile", "CMakeFiles/Makefile.cmake", "cmake_install.cmake", "CPackConfig.cmake")
        configure = (not cache or cache.get("CMAKE_PROJECT_INCLUDE") != str(HOOK)
                     or not all((build_dir / name).is_file() for name in generated)
                     or any(key not in cache for key in wanted))
        commands = []
        if configure:
            commands.append(("configure", ["cmake", "-S", str(source), "-B", str(build_dir),
                                            "-G", "Unix Makefiles", *[f"-D{k}={v}" for k, v in wanted.items()]]))
        commands.append(("build_package", ["cmake", "--build", str(build_dir), "--target", "package", "--parallel", str(jobs)]))
        print("Resume existing objects/dependencies; no clean step." if cache else "Configure first build; future runs reuse this directory.", flush=True)
        for phase, command in commands:
            step = {"phase": phase, "command": command, "status": "running"}
            report["steps"].append(step)
            save()
            step["elapsed_seconds"] = runner(command, source, env, phase)
            step["status"] = "passed"
            save()
        report["status"] = "passed"
        save()
        return report
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        if report["steps"] and report["steps"][-1]["status"] == "running":
            report["steps"][-1]["status"] = report["status"]
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--cann-root", required=True)
    parser.add_argument("--soc", required=True, choices=("ascend910b", "ascend910_93"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output", default="results/cann_fas_build_steps.json")
    args = parser.parse_args()
    try:
        build(args.source_dir, args.cann_root, args.soc, args.jobs, args.output)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"CANN build failed; existing build files retained: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
