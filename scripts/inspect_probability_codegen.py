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
        # vc18 adds explicit register-cap/PV template arguments. Inspect the
        # original 168-register, N128 probability variants only.
        # Later templates add WorkReduction=0 and FP8QK=false. Require those
        # defaults so vc22 experiments and the new FP8 cores aren't mislabeled.
        mode=re.search(r"L8MaskMode0ELb0ELb1ELb1ELb0ELb0ELb1ELi([0-4])E(?:Li168ELb0E(?:Li0E(?:Lb0E)?)?)?EEv",name)
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


def disassembler_environment(tool,nvdisasm=None):
    """Make Toolkit helpers visible only to the child, preserving explicit overrides."""
    env=os.environ.copy()
    executable="nvdisasm.exe" if os.name=="nt" else "nvdisasm"
    if nvdisasm:
        selected=Path(nvdisasm).expanduser().resolve()
        if not selected.is_file() or not os.access(selected,os.X_OK):
            raise SystemExit(f"--nvdisasm must name an executable file: {selected}")
        source="--nvdisasm"
        env.pop("NVDISASM_PATH",None)
        candidates=[selected]
    elif env.get("NVDISASM_PATH"):
        # Do not reinterpret or silently replace the user's native-tool override.
        return env,{"source":"NVDISASM_PATH","value":env["NVDISASM_PATH"]}
    else:
        resolved_tool=Path(shutil.which(str(tool)) or tool).resolve()
        candidates=[resolved_tool.parent/executable]
        for key in ("CUDA_HOME","CUDA_PATH"):
            if env.get(key): candidates.append(Path(env[key])/"bin"/executable)
        on_path=shutil.which(executable)
        if on_path: candidates.append(Path(on_path))
        if os.name!="nt": candidates.append(Path("/usr/local/cuda/bin")/executable)
        selected=next((p.resolve() for p in candidates if p.is_file() and os.access(p,os.X_OK)),None)
        source="auto" if selected else "not found"
    if selected:
        env["PATH"]=str(selected.parent)+(os.pathsep+env["PATH"] if env.get("PATH") else "")
    return env,{"source":source,"executable":str(selected) if selected else None,
                "searched_paths":[str(p) for p in candidates]}


def dump_sass(tool,binary,output,nvdisasm=None):
    """Preserve subprocess evidence even when disassembly fails or is partial."""
    output.parent.mkdir(parents=True,exist_ok=True)
    sass_path=output.with_suffix(".sass.txt")
    stderr_path=output.with_suffix(".cuobjdump.stderr.txt")
    diagnostics_path=output.with_suffix(".diagnostics.json")
    # Resolve before changing the child's PATH to avoid selecting another cuobjdump.
    tool=shutil.which(str(tool)) or str(tool)
    env,disassembler=disassembler_environment(tool,nvdisasm)
    command=[str(tool),"--dump-sass",str(binary)]
    diagnostics={"command":command,"binary":str(binary),"cuobjdump":str(tool),
        "nvdisasm_on_path":shutil.which("nvdisasm"),"nvdisasm":disassembler,"sass_path":str(sass_path),
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
            raw=subprocess.run(command,stdout=stdout,stderr=stderr,check=False,env=env)
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
        hint=""
        if "nvdisasm" in error.lower() and "could not find executable" in error.lower():
            hint=("nvdisasm is unavailable to cuobjdump. Install the nvdisasm component matching your CUDA Toolkit "
                  "if absent, or pass --nvdisasm /path/to/cuda/bin/nvdisasm.\n")
        raise SystemExit(f"cuobjdump failed (exit {diagnostics['returncode']}):\n{error[:4000]}\n"
            f"{hint}"
            f"Diagnostics: {diagnostics_path}\nFull stderr: {stderr_path}\nRaw/partial stdout: {sass_path}\n"
            "Previously saved attention timings are unaffected. Do not rebuild or rerun timings just to retry this inspection.")
    return sass_path.read_text(encoding="utf-8",errors="replace"),diagnostics


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path)
    parser.add_argument("--cuobjdump")
    parser.add_argument("--nvdisasm",type=Path,help="Explicit nvdisasm executable; otherwise search the selected Toolkit and PATH")
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
    sass,diagnostics=dump_sass(tool,binary,args.output,args.nvdisasm)
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
