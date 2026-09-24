"""Verify that the installed fused core contains native Hopper FP8 QGMMA."""
import argparse
import importlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def inspect_sass(sass):
    variants = {"baseline": {}, "pipeline": {}}
    for section in sass.split("Function : ")[1:]:
        name = section.splitlines()[0]
        match = re.search(r"fused_attention(_pipeline)?ILb([01])ELb0", name)
        if not match:
            continue
        fp8 = re.findall(r"QGMMA\.[^;\n]*\.E4M3\.E4M3[^;\n]*", section)
        if not fp8:
            continue  # Development sm_86 image contains a trap, never a fallback.
        if re.search(r"\bHMMA\.", section):
            raise RuntimeError("Unexpected FP16 HMMA in native fused core")
        copies = len(re.findall(r"\bLDGSTS\.", section))
        variant = "pipeline" if match[1] else "baseline"
        if variant == "pipeline" and not copies:
            raise RuntimeError("Pipelined native core must contain async global-to-shared LDGSTS copies")
        variants[variant]["expcast" if match[2] == "1" else "e4m3"] = {
            "function": name, "fp8_wgmma_instructions": len(fp8), "instruction": fp8[0].strip(),
            "async_copy_instructions": copies,
            "local_load_instructions": len(re.findall(r"\bLDL\b", section)),
            "local_store_instructions": len(re.findall(r"\bSTL\b", section))}
    if any(set(modes) != {"e4m3", "expcast"} for modes in variants.values()):
        raise RuntimeError("Baseline and pipeline modes must contain E4M3 QGMMA; rebuild 0.5.0 with sm_90a")
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuobjdump", default=shutil.which("cuobjdump"))
    parser.add_argument("--binary", help="defaults to the actually imported extension")
    args = parser.parse_args()
    if not args.cuobjdump:
        parser.error("cuobjdump from the CUDA Toolkit must be on PATH, or pass --cuobjdump")
    if args.binary:
        binary = Path(args.binary).resolve()
    else:
        import torch  # Load the extension's PyTorch DLL dependencies first.
        binary = Path(importlib.import_module("vc_attention._cuda_ext").__file__)
    result = subprocess.run([args.cuobjdump, "--dump-sass", str(binary)],
                            capture_output=True, text=True, check=True)
    print(json.dumps({"binary": str(binary), "native_fp8_binary_check": "passed",
                      "kernels": inspect_sass(result.stdout)}, indent=2))


if __name__ == "__main__":
    main()
