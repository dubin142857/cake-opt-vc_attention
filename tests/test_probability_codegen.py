"""Disassembly failures retain evidence without invalidating prior timing files."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("inspect_probability_codegen",ROOT/"scripts/inspect_probability_codegen.py")
codegen=importlib.util.module_from_spec(spec);spec.loader.exec_module(codegen)


@pytest.mark.parametrize("suffix",["", "Li168ELb0E", "Li168ELb0ELi0E", "Li168ELb0ELi0ELb0E"])
def test_template_extensions_retain_probability_candidate_inspection(suffix):
    parts=[]
    for dim in (64,128):
        for dtype in ("6__half","13__nv_bfloat16"):
            for mode in range(5):
                name=(f"sage_vc_attention_kernel_sage_layoutILj64ELj128ELj128ELj{dim}E{dtype}"
                      f"L8MaskMode0ELb0ELb1ELb1ELb0ELb0ELb1ELi{mode}E{suffix}EEv")
                parts.append("Function : "+name+"\n /*0000*/ FADD R1, R2, R3 ; /* fixture */\n")
    if suffix.endswith("Lb0E"):
        # FP8QK=true must not become a second vc16 result.
        parts.append(parts[0].replace("Li0ELb0EEE", "Li0ELb1EEE") if "Li0ELb0E" in suffix else "")
    assert len(codegen.inspect("\n".join(parts)))==20


def test_nonzero_exit_saves_full_stderr_partial_sass_and_overwrites_stale_success(monkeypatch,tmp_path):
    commands=[]
    def run(command,**kwargs):
        commands.append(command)
        if command[-1]=="--version": return SimpleNamespace(returncode=0,stdout="CUDA 13.0",stderr="")
        kwargs["stdout"].write(b"partial disassembly")
        kwargs["stderr"].write(b"specific disassembler failure\n")
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(codegen.subprocess,"run",run)
    output=tmp_path/"report.json";output.write_text('{"status":"passed"}')
    timings=tmp_path/"attention.json";timings.write_text("prior timings")
    with pytest.raises(SystemExit,match="specific disassembler failure"):
        codegen.dump_sass("tool with spaces",tmp_path/"input with spaces.so",output)
    report=json.loads(output.read_text())
    assert report["status"]=="failed" and report["stage"]=="cuobjdump" and report["returncode"]==1
    assert Path(report["sass_path"]).read_text()=="partial disassembly"
    assert Path(report["stderr_path"]).read_text()=="specific disassembler failure\n"
    assert json.loads(Path(report["diagnostics_path"]).read_text())["tool_version"]["stdout"]=="CUDA 13.0"
    assert commands[-1]==["tool with spaces","--dump-sass",str(tmp_path/"input with spaces.so")]
    assert timings.read_text()=="prior timings"


def test_launch_error_retained(monkeypatch,tmp_path):
    def run(*args,**kwargs): raise FileNotFoundError("missing executable")
    monkeypatch.setattr(codegen.subprocess,"run",run)
    output=tmp_path/"subdir"/"report.json"
    with pytest.raises(SystemExit,match="missing executable"):
        codegen.dump_sass("missing",tmp_path/"kernel.so",output)
    report=json.loads(output.read_text())
    assert report["returncode"] is None and "missing executable" in report["launch_error"]
    assert Path(report["stderr_path"]).is_file()


def test_parse_failure_retains_successful_tool_output(monkeypatch,tmp_path):
    binary=tmp_path/"kernel.so";binary.write_bytes(b"fixture")
    output=tmp_path/"report.json"
    def run(command,**kwargs):
        if command[-1]=="--version": return SimpleNamespace(returncode=0,stdout="CUDA version",stderr="")
        kwargs["stdout"].write(b"unrecognized symbols")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(codegen.subprocess,"run",run)
    with pytest.raises(SystemExit,match="Expected 20"):
        codegen.main(["--binary",str(binary),"--cuobjdump","fake","--output",str(output)])
    report=json.loads(output.read_text())
    assert report["status"]=="failed" and report["stage"]=="parse" and report["returncode"]==0
    assert Path(report["sass_path"]).read_text()=="unrecognized symbols"


def make_executable(directory,name):
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/(name+('.exe' if codegen.os.name=='nt' else ''))
    path.write_bytes(b'fixture')
    path.chmod(0o755)
    return path


def test_sibling_nvdisasm_reaches_child_without_changing_parent(monkeypatch,tmp_path):
    monkeypatch.delenv("NVDISASM_PATH",raising=False)
    monkeypatch.setenv("PATH","")
    tool=make_executable(tmp_path/"toolkit with spaces"/"bin","cuobjdump")
    disasm=make_executable(tool.parent,"nvdisasm")
    def run(command,**kwargs):
        if command[-1]=="--version": return SimpleNamespace(returncode=0,stdout="CUDA 13.0",stderr="")
        assert kwargs["env"]["PATH"]==str(disasm.parent)
        assert command[0]==str(tool)
        assert codegen.os.environ["PATH"]==""
        kwargs["stdout"].write(b"sass fixture")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(codegen.subprocess,"run",run)
    sass,diagnostics=codegen.dump_sass(tool,tmp_path/"kernel.so",tmp_path/"result.json")
    assert sass=="sass fixture"
    assert diagnostics["nvdisasm"]["executable"]==str(disasm)


@pytest.mark.parametrize("variable",["CUDA_HOME","CUDA_PATH"])
def test_nvdisasm_from_cuda_environment(monkeypatch,tmp_path,variable):
    for key in ("NVDISASM_PATH","CUDA_HOME","CUDA_PATH"):
        monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv("PATH","")
    root=tmp_path/"cuda"
    disasm=make_executable(root/"bin","nvdisasm")
    monkeypatch.setenv(variable,str(root))
    env,info=codegen.disassembler_environment(tmp_path/"other"/"cuobjdump")
    assert info["executable"]==str(disasm) and env["PATH"]==str(disasm.parent)


def test_explicit_nvdisasm_overrides_child_only(monkeypatch,tmp_path):
    monkeypatch.setenv("NVDISASM_PATH","user override")
    monkeypatch.setenv("PATH","original path")
    env,info=codegen.disassembler_environment("cuobjdump")
    assert env["NVDISASM_PATH"]=="user override" and info["source"]=="NVDISASM_PATH"
    disasm=make_executable(tmp_path/"chosen","nvdisasm")
    env,info=codegen.disassembler_environment("cuobjdump",disasm)
    assert "NVDISASM_PATH" not in env and info["source"]=="--nvdisasm"
    assert env["PATH"]==str(disasm.parent)+codegen.os.pathsep+"original path"
    assert codegen.os.environ["NVDISASM_PATH"]=="user override"


def test_missing_nvdisasm_keeps_actionable_failure(monkeypatch,tmp_path):
    def run(command,**kwargs):
        if command[-1]=="--version": return SimpleNamespace(returncode=0,stdout="CUDA 13.0",stderr="")
        kwargs["stderr"].write(b"cuobjdump fatal: Could not find executable file 'nvdisasm'")
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(codegen.subprocess,"run",run)
    with pytest.raises(SystemExit,match="pass --nvdisasm"):
        codegen.dump_sass("cuobjdump",tmp_path/"kernel.so",tmp_path/"result.json")
