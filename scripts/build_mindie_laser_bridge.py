#!/usr/bin/env python3
"""Build only the MindIE Laser PTA bridge; no wheel or other MindIE plugins."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from build_cann_fas_local import read_cache, run_visible

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "71098c20ff90838d1fd1ca0a40515c751a3869b0"
ADAPTER_FILES = ("la.cpp", "la.h", "pytorch_npu_helper.h", "find_op_path.cpp", "find_op_path.h")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_adapter(source, destination):
    """Read exact committed bytes; only change the ACLNN operator name."""
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_COMMIT:
        raise ValueError("MindIE source commit differs from the pinned Laser baseline")
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in ADAPTER_FILES:
        original = subprocess.check_output(
            ["git", "-C", str(source), "show", f"{UPSTREAM_COMMIT}:csrc/plugin/{name}"])
        content = original
        if name == "la.cpp":
            if original.count(b'"aclnnLaserAttention"') != 1:
                raise ValueError("Unexpected upstream Laser native API declaration")
            content = original.replace(b'"aclnnLaserAttention"', b'"aclnnVcLaserAttention"')
        target = destination / name
        if not target.exists() or target.read_bytes() != content:
            target.write_bytes(content)
        records[name] = {"upstream_sha256": hashlib.sha256(original).hexdigest(),
                         "compiled_sha256": sha(target)}
    license_bytes = subprocess.check_output(["git", "-C", str(source), "show", f"{UPSTREAM_COMMIT}:LICENSE.md"])
    license_path = destination / "LICENSE.md"
    if not license_path.exists() or license_path.read_bytes() != license_bytes:
        license_path.write_bytes(license_bytes)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--cann-root", required=True)
    parser.add_argument("--build-dir", default="build/mindie_laser_bridge")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", default="results/mindie_laser_bridge.json")
    args = parser.parse_args(argv)
    report = {"status": "running", "native_api": "aclnnVcLaserAttention",
              "torch_api": "torch.ops.vc_laser.la", "upstream_commit": UPSTREAM_COMMIT,
              "scope": "Upstream la.cpp/helper; ACLNN symbol renamed; original queue/stream behavior retained"}
    path = Path(args.output).resolve()
    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    save()
    try:
        if sys.platform != "linux" or args.jobs < 1:
            raise ValueError("Build requires Linux and a positive --jobs value")
        import torch
        import torch_npu
        source, cann, build_dir = map(lambda x: Path(x).resolve(),
                                     (args.source_dir, args.cann_root, args.build_dir))
        build_dir.mkdir(parents=True, exist_ok=True)
        report["upstream_source_dir"] = str(source)
        report["adapter_files"] = prepare_adapter(source, build_dir / "adapter")
        definitions = {"CMAKE_BUILD_TYPE": "Release", "Python3_EXECUTABLE": sys.executable,
                       "VC_LASER_ADAPTER_DIR": str(build_dir / "adapter"),
                       "TORCH_ROOT": str(Path(torch.__file__).resolve().parent),
                       "TORCH_NPU_ROOT": str(Path(torch_npu.__file__).resolve().parent),
                       "CANN_ROOT": str(cann), "TORCH_ABI": str(int(torch._C._GLIBCXX_USE_CXX11_ABI))}
        cache = read_cache(build_dir / "CMakeCache.txt")
        # A different Python/Torch/CANN toolchain needs a separate build directory.
        for key, value in definitions.items():
            if key in cache and cache[key] != value:
                raise ValueError(f"Existing bridge cache {key} differs; preserve it and use another --build-dir")
        report["definitions"] = definitions
        save()
        run_visible(["cmake", "-S", str(ROOT / "ascend/laser_bridge"), "-B", str(build_dir),
                     "-G", "Unix Makefiles", *[f"-D{k}={v}" for k,v in definitions.items()]],
                    ROOT, os.environ.copy(), "bridge_configure")
        run_visible(["cmake", "--build", str(build_dir), "--parallel", str(args.jobs)],
                    ROOT, os.environ.copy(), "bridge_build")
        bridge = build_dir / "libvc_laser_bridge.so"
        if not bridge.is_file():
            raise RuntimeError("Bridge library missing after successful build")
        report.update(status="passed", bridge_path=str(bridge), bridge_sha256=sha(bridge))
        save()
        print("Laser bridge:", bridge, flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        save()
        print(report["error"], file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    sys.exit(main())
