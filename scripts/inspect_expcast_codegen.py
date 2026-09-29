"""Inspect vc12/vc14 machine instructions using cuobjdump; no ncu or GPU timing.

Default: inspect the installed extension. --binary can inspect a cross-compiled
object on a machine without Hopper. Counts include all main/tail/fallback paths.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def inspect(text):
    functions={}
    for part in text.split("Function : ")[1:]:
        name=part.splitlines()[0].strip()
        instructions=re.findall(r"/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*",part)
        functions[name]=instructions
    rows=[]
    for name,code in functions.items():
        if "expcast_paired" not in name:
            continue
        suffix=name.split("expcast_paired",1)[1]
        old=[values for key,values in functions.items() if "expcast_affine" in key
             and key.split("expcast_affine",1)[1]==suffix]
        if len(old)!=1:
            raise RuntimeError(f"Cannot identify one vc12 baseline for {name}")
        def counts(instructions):
            ops=Counter(line.split()[0].split(".")[0] for line in instructions if line)
            return {op:ops[op] for op in ("F2IP","PRMT","F2FP","HADD2","FADD","FFMA")}
        before,after=counts(old[0]),counts(code)
        rows.append({"kernel":name,"vc12":before,"vc14":after,
            "paired_conversion_reduced":before["F2IP"]>after["F2IP"]>0,
            "packing_instructions_reduced":before["PRMT"]>after["PRMT"]})
    if len(rows)!=4:
        raise RuntimeError(f"Expected vc14 FP16/BF16, D64/D128 kernels, found {len(rows)}; rebuild package")
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path)
    parser.add_argument("--cuobjdump",help="Explicit cuobjdump executable")
    parser.add_argument("--output",type=Path,default=Path("results/expcast_codegen.json"))
    args=parser.parse_args()
    binary=args.binary
    if binary is None:
        from sageattention_local import vc
        binary=Path(vc._extension().__file__)
    binary=binary.resolve(strict=True)
    tool=args.cuobjdump or shutil.which("cuobjdump")
    if not tool:
        for variable in ("CUDA_HOME","CUDA_PATH"):
            candidate=Path(os.environ.get(variable,"/usr/local/cuda"))/"bin"/("cuobjdump.exe" if os.name=="nt" else "cuobjdump")
            if candidate.is_file():
                tool=str(candidate)
                break
    if not tool:
        raise SystemExit("cuobjdump not found; pass --cuobjdump /path/to/cuda/bin/cuobjdump")
    result=subprocess.run([tool,"--dump-sass",str(binary)],capture_output=True,text=True,errors="replace")
    if result.returncode:
        raise SystemExit(f"cuobjdump failed: {result.stderr}")
    rows=inspect(result.stdout)
    report={"binary":str(binary),"cuobjdump":str(tool),"variants":rows,
        "scope":"Static instruction counts, including main/tail and affine/fallback branches; NOT dynamic counts or a performance measurement"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    for row in rows:
        print(row["kernel"])
        print("  F2IP: {0} -> {1}; PRMT: {2} -> {3}".format(
            row["vc12"]["F2IP"],row["vc14"]["F2IP"],row["vc12"]["PRMT"],row["vc14"]["PRMT"]))
    if not all(row["paired_conversion_reduced"] and row["packing_instructions_reduced"] for row in rows):
        print("This compiler did not reproduce both instruction reductions. Use full H200 measurements to decide.")
    print(report["scope"])
    print(f"Report: {args.output}")


if __name__=="__main__":
    main()
