#!/usr/bin/env python3
"""Build only pinned MindIE Laser host/device code; install inside the workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

try:
    from scripts.build_cann_fas_local import read_cache, run_visible
    from scripts.prepare_mindie_laser_source import (
        NATIVE_API, OP_TYPE, SOURCE_LOCK, default_manifest, validate_prepared_source,
    )
except ModuleNotFoundError:
    from build_cann_fas_local import read_cache, run_visible
    from prepare_mindie_laser_source import (
        NATIVE_API, OP_TYPE, SOURCE_LOCK, default_manifest, validate_prepared_source,
    )


ROOT = Path(__file__).resolve().parents[1]
GENERATED_FILES = ("Makefile", "CMakeFiles/Makefile.cmake", "cmake_install.cmake", "CPackConfig.cmake")


def definitions(source: Path, cann: Path, soc: str, install: Path) -> dict:
    return {
        "BUILD_OPEN_PROJECT": "ON", "ASCEND_OP_NAME": "vc_laser_attention",
        "VENDOR_NAME": "vc_laser", "ASCEND_COMPUTE_UNIT": soc,
        "CUSTOM_ASCEND_CANN_PACKAGE_PATH": str(cann),
        "ENABLE_OPS_HOST": "ON", "ENABLE_OPS_KERNEL": "ON", "PREPARE_BUILD": "OFF",
        "CMAKE_BUILD_TYPE": "Release", "CMAKE_INSTALL_PREFIX": str(install),
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON", "ENABLE_CCACHE": "OFF",
        "Python3_EXECUTABLE": sys.executable,
        # Upstream config.cmake passes this unquoted to prepare.sh. An empty
        # value consumes --ascend-compute_unit and loses both SoC and op name.
        "CHECK_COMPATIBLE": "TRUE",
    }


def kernel_build_graph(build_dir: Path, soc: str) -> dict:
    """Check configure-time kernel discovery, not just a successful host build.

    Upstream globs these scripts during configure. Scripts generated later by
    make cannot repair an already empty ops_kernel target without reconfigure.
    """
    op = "vc_laser_attention"
    prefix = f"{OP_TYPE}-{op}-"
    gen = build_dir / "binary" / soc / "gen"
    scripts = sorted(path for path in gen.glob(prefix + "*.sh")
                     if path.is_file() and path.stat().st_size
                     and path.stem.removeprefix(prefix).isdigit())
    target_file = build_dir / "CMakeFiles/TargetDirectories.txt"
    targets = set()
    if target_file.is_file():
        for line in target_file.read_text(encoding="utf-8").splitlines():
            targets.add(line.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].removesuffix(".dir"))
    required = {"ops_kernel", op, f"{op}_{soc}"}
    required.update(f"{op}_{soc}_{path.stem.removeprefix(prefix)}" for path in scripts)
    missing = sorted(required - targets)
    return {"complete": bool(scripts) and not missing,
            "compile_scripts": [str(path.resolve()) for path in scripts],
            "target_file": str(target_file.resolve()), "missing_targets": missing}


def validate_cache(cache: dict, source: Path, wanted: dict) -> None:
    path_keys = {"CMAKE_HOME_DIRECTORY": str(source / "csrc/ops"),
                 "CUSTOM_ASCEND_CANN_PACKAGE_PATH": wanted["CUSTOM_ASCEND_CANN_PACKAGE_PATH"],
                 "CMAKE_INSTALL_PREFIX": wanted["CMAKE_INSTALL_PREFIX"]}
    for key, value in path_keys.items():
        if key in cache and Path(cache[key]).resolve() != Path(value).resolve():
            raise ValueError(f"Existing {key} differs from requested {value}; build directory preserved")
    for key, value in wanted.items():
        if key not in path_keys and key in cache and cache[key] != value:
            raise ValueError(f"Existing {key}={cache[key]} differs from {value}; build directory preserved")
    if cache.get("CMAKE_GENERATOR", "Unix Makefiles") != "Unix Makefiles":
        raise ValueError("Existing CMAKE_GENERATOR is incompatible; build directory preserved")
    for key in ("CMAKE_PROJECT_INCLUDE", "CMAKE_PROJECT_INCLUDE_BEFORE", "CMAKE_TOOLCHAIN_FILE"):
        if cache.get(key):
            raise ValueError(f"Unrecognized {key}; build directory preserved")


def build_environment(base: dict, cann: Path) -> dict:
    if os.name != "posix":
        raise ValueError("Building requires Linux and the installed Ascend CANN Toolkit")
    setup = cann / "bin/setenv.bash"
    if not setup.is_file() or not (cann / "tools/opbuild/op_build").is_file():
        raise ValueError("CANN Toolkit bin/setenv.bash or tools/opbuild/op_build is missing")
    loaded = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", 'source "$1" >/dev/null && env -0',
         "mindie-laser-env", str(setup)], env=dict(base), capture_output=True, check=True,
    )
    env = dict(base)
    for entry in loaded.stdout.split(b"\0"):
        if b"=" in entry:
            key, value = entry.split(b"=", 1)
            env[os.fsdecode(key)] = os.fsdecode(value)
    env["ASCEND_HOME_PATH"] = str(cann)
    compiler = shutil.which("bisheng", path=env.get("PATH"))
    if not compiler:
        raise ValueError("bisheng not found after loading the requested CANN Toolkit")
    env["BISHENG_REAL_PATH"] = compiler
    return env


def fingerprint(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty build artifact: {path}")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def collect_artifacts(build_dir: Path, install: Path, soc: str) -> dict:
    packages = sorted(build_dir.glob("CANN-custom_ops-*.run"))
    if len(packages) != 1:
        raise ValueError(f"Expected one generated CANN custom operator package; found {len(packages)}")
    vendor = install / "packages/vendors/vc_laser"
    api = vendor / "op_api/lib/libcust_opapi.so"
    tiling = sorted((vendor / "op_impl/ai_core/tbe/op_tiling/lib").rglob("libcust_opmaster_rt2.0.so"))
    if len(tiling) != 1:
        raise ValueError("Expected one installed Laser host tiling library")
    device_dir = vendor / "op_impl/ai_core/tbe/kernel" / soc / "vc_laser_attention"
    device = sorted(path for path in device_dir.rglob("*")
                    if path.is_file() and path.suffix in (".o", ".bin"))
    if not device:
        raise ValueError("No installed Laser device binary was produced")
    return {"package": fingerprint(packages[0]), "vendor_root": str(vendor.resolve()),
            "host_libraries": [fingerprint(api), *[fingerprint(path) for path in tiling]],
            "device_binaries": [fingerprint(path) for path in device]}


def build(source, cann, soc, jobs, install_dir, output, *,
          source_manifest=None, runner=run_visible):
    source, cann = Path(source).resolve(), Path(cann).resolve()
    install, output = Path(install_dir).resolve(), Path(output).resolve()
    manifest_path = Path(source_manifest or default_manifest(source)).resolve()
    build_dir = source / "build/build"
    cache = read_cache(build_dir / "CMakeCache.txt")
    wanted = definitions(source, cann, soc, install)
    report = {
        "schema_version": 1, "status": "running", "source_dir": str(source),
        "build_dir": str(build_dir), "install_dir": str(install), "soc": soc, "jobs": jobs,
        "native_api": NATIVE_API, "op_type": OP_TYPE,
        "definitions": wanted, "steps": [], "resume_existing_cache": bool(cache),
        "deletes_build_directory": False,
        "scope": "Only vc_laser_attention host/device custom operator; no MindIE wheel, plugin or TIK build",
        "installation_mode": "CMake staging inside workspace; no package installer or system installation",
    }

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)

    save()
    try:
        if isinstance(jobs, bool) or jobs < 1 or soc not in ("ascend910b", "ascend910_93"):
            raise ValueError("Positive jobs and A2/A3 SoC are required")
        if not install.is_relative_to(ROOT.resolve()) or install == ROOT.resolve():
            raise ValueError("Install directory must be inside the workspace")
        if install == source or install.is_relative_to(source) or source.is_relative_to(install):
            raise ValueError("Install and source directories must be separate inside the workspace")
        if install == cann or install.is_relative_to(cann) or cann.is_relative_to(install):
            raise ValueError("Install directory must not overlap the CANN Toolkit")
        validate_cache(cache, source, wanted)
        manifest = validate_prepared_source(source, manifest_path)
        report["source_manifest"] = fingerprint(manifest_path)
        report["upstream"] = manifest["upstream"]
        env = build_environment(os.environ, cann)
        env.update(BUILD_JOB_NUM=str(jobs), CMAKE_BUILD_PARALLEL_LEVEL=str(jobs))
        # Upstream nested prepare and generated kernel scripts also resolve
        # python3 through PATH; keep them in the driver interpreter environment.
        env["PATH"] = os.pathsep.join((str(Path(sys.executable).parent), env.get("PATH", "")))
        graph = kernel_build_graph(build_dir, soc)
        report["device_graph_before"] = graph
        configure = (not cache or any(key not in cache for key in wanted)
                     or not all((build_dir / name).is_file() for name in GENERATED_FILES)
                     or not graph["complete"])
        report["configure_required"] = configure
        if cache and not graph["complete"]:
            print("Incomplete Laser device build graph; reconfiguring in place, preserving host objects.", flush=True)
        commands = []
        if configure:
            commands.append(("configure", ["cmake", "-S", str(source / "csrc/ops"), "-B", str(build_dir),
                                            "-G", "Unix Makefiles", *[f"-D{k}={v}" for k, v in wanted.items()]]))
        commands.append(("build_package", ["cmake", "--build", str(build_dir), "--target", "package",
                                           "--parallel", str(jobs)]))
        commands.append(("install_workspace", ["cmake", "--install", str(build_dir), "--prefix", str(install)]))
        for phase, command in commands:
            step = {"phase": phase, "command": command, "status": "running"}
            report["steps"].append(step)
            save()
            step["elapsed_seconds"] = runner(command, source, env, phase)
            if phase == "configure":
                graph = kernel_build_graph(build_dir, soc)
                report["device_graph_after_configure"] = graph
                save()
                if not graph["complete"]:
                    raise ValueError(
                        "Laser device build graph is empty/incomplete after configure: "
                        f"{len(graph['compile_scripts'])} compile scripts; missing targets {graph['missing_targets']}. "
                        "Host-only packaging is stopped. Check prepare.sh SoC/operator arguments and configure log.")
            step["status"] = "passed"
            save()
        report.update(collect_artifacts(build_dir, install, soc))
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "build/mindie_laser_source")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--cann-root", type=Path, required=True)
    parser.add_argument("--soc", choices=("ascend910b", "ascend910_93"), required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--install-dir", type=Path, default=ROOT / "build/mindie_laser_install")
    parser.add_argument("--output", type=Path, default=ROOT / "results/mindie_laser_build.json")
    args = parser.parse_args()
    try:
        build(args.source_dir, args.cann_root, args.soc, args.jobs, args.install_dir, args.output,
              source_manifest=args.source_manifest)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"MindIE Laser build failed; build files preserved: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
