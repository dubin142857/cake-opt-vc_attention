"""Inspect vc16/vc17 full attention SASS; static counts are not GPU timing."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

NAMES=("vc16","vc17_g1","vc17_g4","vc17_c4","vc17_direct")


def inspect(sass):
    rows=[]
    for part in sass.split("Function : ")[1:]:
        name=part.splitlines()[0].strip()
        if "sage_vc_attention_kernel_sage_layout" not in name: continue
        # ExpCast, late next-V, history overlap; same body for all candidates.
        mode=re.search(r"L8MaskMode0ELb0ELb1ELb1ELb0ELb0ELb1ELi([0-4])EEEv",name)
        dim=re.search(r"ILj64ELj128ELj128ELj(64|128)E",name)
        if not mode or not dim: continue
        code=re.findall(r"/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*",part)
        counts=Counter()
        for line in code:
            tokens=line.split()
            if tokens and tokens[0].startswith("@"): tokens=tokens[1:]
            if tokens: counts[tokens[0].split(".")[0]]+=1
        rows.append({"mode":NAMES[int(mode[1])],"dim":int(dim[1]),
                     "dtype":"bfloat16" if "bfloat16" in name else "float16","kernel":name,
                     "total_static_instructions":sum(counts.values()),"opcodes":dict(sorted(counts.items()))})
    expected={(mode,dim,dtype) for mode in NAMES for dim in (64,128) for dtype in ("float16","bfloat16")}
    keys={(r["mode"],r["dim"],r["dtype"]) for r in rows}
    if keys!=expected or len(rows)!=20: raise RuntimeError(f"Expected 20 vc16/vc17 kernels, found {len(rows)}; rebuild vc17")
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path)
    parser.add_argument("--cuobjdump")
    parser.add_argument("--output",type=Path,default=Path("results/probability_codegen.json"))
    args=parser.parse_args()
    if args.binary is None:
        from sageattention_local import vc
        binary=Path(vc._extension().__file__)
    else: binary=args.binary
    binary=binary.resolve(strict=True)
    tool=args.cuobjdump or shutil.which("cuobjdump")
    if not tool:
        for env in ("CUDA_HOME","CUDA_PATH"):
            candidate=Path(os.environ.get(env,"/usr/local/cuda"))/"bin"/("cuobjdump.exe" if os.name=="nt" else "cuobjdump")
            if candidate.is_file(): tool=str(candidate);break
    if not tool: raise SystemExit("cuobjdump not found; pass --cuobjdump /path/to/cuda/bin/cuobjdump")
    raw=subprocess.run([tool,"--dump-sass",str(binary)],capture_output=True,text=True,errors="replace",check=True)
    rows=inspect(raw.stdout)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    sass_path=args.output.with_suffix(".sass.txt");sass_path.write_text(raw.stdout,encoding="utf-8")
    report={"binary":str(binary),"sass_path":str(sass_path),"variants":rows,
        "scope":"Static counts include main/tail and fallback branches; not dynamic instruction counts or bottleneck percentages. Raw SASS retained for dependency inspection."}
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("D128 BF16 static instructions (NOT speedup):")
    for row in rows:
        if row["dim"]==128 and row["dtype"]=="bfloat16":
            print(row["mode"],{op:row["opcodes"].get(op,0) for op in ("F2IP","F2FP","HADD2","FADD","IADD3","LOP3","SHF","SEL")})
    print(f"Report: {args.output}; raw SASS: {sass_path}")


if __name__=="__main__": main()
