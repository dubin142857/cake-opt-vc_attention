"""Read-only target-machine preflight. Writes evidence even on failure."""
import argparse
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


def resolve_soc(runtime_soc, inherited_soc, cann):
    """Select this device's exact compiler target; never substitute another SKU.

    A2 910B and A3 910_93xx share DAV-2201. This is an architecture mapping,
    not evidence that our native kernel has passed validation on either target.
    """
    runtime_soc = runtime_soc.strip()
    if re.fullmatch(r"Ascend910B(?:[1-4]|2C)", runtime_soc, re.IGNORECASE):
        compiler_soc = "Ascend910B" + runtime_soc[len("Ascend910B"):].upper()
        family = "Atlas A2 / Ascend910B"
    elif re.fullmatch(r"Ascend910_93[0-9]{2}", runtime_soc, re.IGNORECASE):
        compiler_soc = "Ascend910_" + runtime_soc.split("_", 1)[1]
        family = "Atlas A3 / Ascend910_93xx"
    else:
        raise ValueError(f"Unsupported runtime SoC {runtime_soc!r}; expected exact A2 910B or A3 910_93xx SKU")
    result = {"compiler_soc": compiler_soc, "compiler_soc_source": "selected device runtime",
              "inherited_soc_version": inherited_soc, "target_label": family,
              "expected_npu_arch": "2201", "matching_soc_ini": [],
              "soc_ini_evidence": [], "warnings": [], "errors": []}
    if inherited_soc and inherited_soc.strip().casefold() != compiler_soc.casefold():
        result["warnings"].append(
            f"Ignoring inherited SOC_VERSION={inherited_soc}; selected device requires {compiler_soc}")
    # CANN releases use root, aarch64-linux/arm64-linux, or compiler layouts.
    # Compare the SoC_version FIELD, not an ini filename or a prefix match.
    roots = {cann / "data/platform_config", cann / "compiler/data/platform_config"}
    roots.update(cann.glob("*/data/platform_config"))
    paths = {p.resolve() for root in roots for p in root.glob("*.ini")}
    for path in sorted(paths):
        fields = {}
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", ";", "[")) or "=" not in line:
                continue
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
        if fields.get("SoC_version", "").casefold() != compiler_soc.casefold():
            continue
        result["matching_soc_ini"].append(str(path))
        result["soc_ini_evidence"].append({"path": str(path), **{
            k: fields[k] for k in ("SoC_version", "Short_SoC_version", "NpuArch", "CCEC_AIV_version") if k in fields}})
        arch = fields.get("NpuArch")
        if arch and arch not in ("2201", "DAV_2201"):
            result["errors"].append(f"CANN {path} reports NpuArch={arch}; this kernel requires DAV_2201")
    if paths and not result["matching_soc_ini"]:
        result["errors"].append(f"CANN platform_config has no exact SoC_version={compiler_soc}; use a Toolkit supporting this SKU")
    elif not paths:
        result["warnings"].append("No platform ini files exposed by this Toolkit; the CANN compiler must validate the exact runtime SoC")
    return result


def inspect(device=0, require_laser=False, require_a3=False):
    report = {"python": sys.version, "platform": platform.platform(), "device_index": device,
              "target_label": "auto: DAV-2201 A2/A3", "native_fp8_matmul": False,
              "require_laser": require_laser, "require_a3": require_a3,
              "errors": [], "warnings": []}
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
        selection = resolve_soc(soc, os.environ.get("SOC_VERSION"), cann)
        report["errors"].extend(selection.pop("errors"))
        report["warnings"].extend(selection.pop("warnings"))
        report.update(selection)
        if require_a3 and not compiler_soc_is_a3(report["compiler_soc"]):
            report["errors"].append(
                f"Selected device is {report['compiler_soc']}, not an A3/910C Ascend910_93xx SKU")
        x = torch.arange(16, dtype=torch.float32, device=f"npu:{device}")
        if not torch.equal((x + 1).cpu(), torch.arange(1, 17, dtype=torch.float32)):
            report["errors"].append("Basic NPU execution check failed")
        report["fusion_attention_available"] = hasattr(torch_npu, "npu_fusion_attention")
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["status"] = "passed" if not report["errors"] else "failed"
    return report


def compiler_soc_is_a3(soc):
    return bool(re.fullmatch(r"Ascend910_93[0-9]{2}", soc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--require-laser", action="store_true")
    parser.add_argument("--require-a3", action="store_true",
                        help="Reject A2/910B devices for a 910C/A3 run")
    parser.add_argument("--output", default="results/ascend910b_environment.json")
    args = parser.parse_args()
    report = inspect(args.device, require_laser=args.require_laser,
                     require_a3=args.require_a3)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
