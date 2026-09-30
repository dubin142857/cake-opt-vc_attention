"""Disassembly failures retain evidence without invalidating prior timing files."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("inspect_probability_codegen",ROOT/"scripts/inspect_probability_codegen.py")
codegen=importlib.util.module_from_spec(spec);spec.loader.exec_module(codegen)


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
