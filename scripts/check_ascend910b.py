"""Read-only target-machine preflight. Writes evidence even on failure."""
import argparse
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def inspect(device=0, require_laser=False):
    report = {"python": sys.version, "platform": platform.platform(), "device_index": device,
              "target_label": "910B_23", "native_fp8_matmul": False, "require_laser": require_laser, "errors": []}
    try:
        report["mindiesd_installed_version"] = importlib.metadata.version("mindiesd")
    except importlib.metadata.PackageNotFoundError:
        report["mindiesd_installed_version"] = None
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/nonexistent"))
    report["cann_root"] = str(cann)
    report["cmake"] = shutil.which("cmake")
    helpers = [cann / p / "ascendc.cmake" for p in (
        "tools/tikcpp/ascendc_kernel_cmake", "compiler/tikcpp/ascendc_kernel_cmake",
        "ascendc_devkit/tikcpp/samples/cmake")]
    report["ascendc_cmake"] = next((str(p) for p in helpers if p.is_file()), None)
    for name, present in (("ASCEND_HOME_PATH", cann.is_dir()), ("cmake", report["cmake"]),
                          ("Ascend C CMake helper", report["ascendc_cmake"])):
        if not present:
            report["errors"].append(f"Missing {name}; source the CANN Toolkit set_env.sh")
    versions = list(cann.glob("*install.info")) + list(cann.glob("*/ascend_toolkit_install.info"))
    report["cann_version_files"] = {str(p): p.read_text(errors="replace") for p in versions if p.is_file()}
    if shutil.which("npu-smi"):
        try:
            proc = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30)
            report["npu_smi"] = proc.stdout + proc.stderr  # diagnostic only, NOT SoC detection
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["npu_smi"] = str(exc)
    try:
        import torch
        import torch_npu
        report.update(torch=torch.__version__, torch_npu=torch_npu.__version__)
        if require_laser:
            from mindie_laser_backend import load_mindie_laser
            adapter = load_mindie_laser()
            report["mindie_laser"] = {**adapter.metadata, "status": "registered; native execution still requires benchmark validation"}
        torch.npu.set_device(device)
        report["device_name"] = torch.npu.get_device_name(device)
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtGetSocName.restype = ctypes.c_char_p
        name = acl.aclrtGetSocName()
        if not name:
            raise RuntimeError("aclrtGetSocName returned null")
        soc = name.decode()
        report["runtime_soc"] = soc
        report["soc_source"] = "aclrtGetSocName after torch.npu.set_device"
        compiler_soc = os.environ.get("SOC_VERSION", soc)
        report["compiler_soc"] = compiler_soc
        if not soc.startswith("Ascend910B"):
            report["errors"].append(f"Expected Ascend910B runtime, found {soc}")
        if not compiler_soc.startswith("Ascend910B"):
            report["errors"].append("SOC_VERSION must be an Ascend910B compiler configuration")
        configs = []
        # Record exact config evidence where CANN exposes platform ini files.
        for root in (cann / "data/platform_config", cann / "aarch64-linux/data/platform_config",
                     cann / "x86_64-linux/data/platform_config"):
            if root.is_dir():
                for path in root.glob("*.ini"):
                    body = path.read_text(errors="replace")
                    if any(line.replace(" ", "").strip() == "SoC_version=" + compiler_soc for line in body.splitlines()):
                        configs.append(str(path))
        report["matching_soc_ini"] = configs
        x = torch.arange(16, dtype=torch.float32, device=f"npu:{device}")
        if not torch.equal((x + 1).cpu(), torch.arange(1, 17, dtype=torch.float32)):
            report["errors"].append("Basic NPU execution check failed")
        report["fusion_attention_available"] = hasattr(torch_npu, "npu_fusion_attention")
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["status"] = "passed" if not report["errors"] else "failed"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--require-laser", action="store_true")
    parser.add_argument("--output", default="results/ascend910b_environment.json")
    args = parser.parse_args()
    report = inspect(args.device, require_laser=args.require_laser)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
