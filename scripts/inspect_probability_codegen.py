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


def dump_sass(tool,binary,output):
    """Preserve subprocess evidence even when disassembly fails or is partial."""
    output.parent.mkdir(parents=True,exist_ok=True)
    sass_path=output.with_suffix(".sass.txt")
    stderr_path=output.with_suffix(".cuobjdump.stderr.txt")
    diagnostics_path=output.with_suffix(".diagnostics.json")
    command=[str(tool),"--dump-sass",str(binary)]
    diagnostics={"command":command,"binary":str(binary),"cuobjdump":str(tool),
        "nvdisasm_on_path":shutil.which("nvdisasm"),"sass_path":str(sass_path),
        "stderr_path":str(stderr_path),"diagnostics_path":str(diagnostics_path)}
    try:
        version=subprocess.run([str(tool),"--version"],capture_output=True,text=True,errors="replace",timeout=10)
        diagnostics["tool_version"]={"returncode":version.returncode,"stdout":version.stdout,"stderr":version.stderr}
    except (OSError,subprocess.TimeoutExpired) as exc:
        diagnostics["tool_version_error"]=str(exc)
    try:
        # Stream to files instead of collecting the complete shared library's
        # disassembly in subprocess.PIPE; keep stderr independently readable.
        with sass_path.open("wb") as stdout,stderr_path.open("wb") as stderr:
            raw=subprocess.run(command,stdout=stdout,stderr=stderr,check=False)
        diagnostics["returncode"]=raw.returncode
    except OSError as exc:
        diagnostics.update(returncode=None,launch_error=str(exc))
    diagnostics_path.write_text(json.dumps(diagnostics,indent=2)+"\n",encoding="utf-8")
    if diagnostics["returncode"]!=0:
        error=stderr_path.read_text(encoding="utf-8",errors="replace").strip()
        if not error: error=diagnostics.get("launch_error") or sass_path.read_text(encoding="utf-8",errors="replace")[-4000:] or "cuobjdump produced no error text"
        failure={**diagnostics,"status":"failed","stage":"cuobjdump","error":error}
        # A failed retry must not leave an old successful JSON looking current.
        output.write_text(json.dumps(failure,indent=2)+"\n",encoding="utf-8")
        raise SystemExit(f"cuobjdump failed (exit {diagnostics['returncode']}):\n{error[:4000]}\n"
            f"Diagnostics: {diagnostics_path}\nFull stderr: {stderr_path}\nRaw/partial stdout: {sass_path}\n"
            "Previously saved attention timings are unaffected. Do not rebuild or rerun timings just to retry this inspection.")
    return sass_path.read_text(encoding="utf-8",errors="replace"),diagnostics


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path)
    parser.add_argument("--cuobjdump")
    parser.add_argument("--output",type=Path,default=Path("results/probability_codegen.json"))
    args=parser.parse_args(argv)
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
    sass,diagnostics=dump_sass(tool,binary,args.output)
    try:
        rows=inspect(sass)
    except RuntimeError as exc:
        report={**diagnostics,"status":"failed","stage":"parse","error":str(exc)}
        args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        raise SystemExit(f"{exc}\nRaw SASS: {diagnostics['sass_path']}\nDiagnostics: {diagnostics['diagnostics_path']}") from exc
    report={**diagnostics,"status":"passed","variants":rows,
        "scope":"Static counts include main/tail and fallback branches; not dynamic instruction counts or bottleneck percentages. Raw SASS retained for dependency inspection."}
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("D128 BF16 static instructions (NOT speedup):")
    for row in rows:
        if row["dim"]==128 and row["dtype"]=="bfloat16":
            print(row["mode"],{op:row["opcodes"].get(op,0) for op in ("F2IP","F2FP","HADD2","FADD","IADD3","LOP3","SHF","SEL")})
    print(f"Report: {args.output}; raw SASS: {diagnostics['sass_path']}")


if __name__=="__main__": main()
