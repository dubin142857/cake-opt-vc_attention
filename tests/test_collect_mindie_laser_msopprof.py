"""Host-only command selection and failure boundaries; no profiler/NPU needed."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import collect_mindie_laser_msopprof as collect


def test_help_discovery_and_prefix_before_application(tmp_path):
    help_text = " ".join(sorted(collect.REQUIRED | {"--replay-mode","--kill"}))
    def runner(command, **kwargs):
        assert command == ["/local path/msopprof", "--help"]
        return SimpleNamespace(returncode=0,stdout=help_text,stderr="")
    tool = collect.discover("/cann",tmp_path,explicit="/local path/msopprof",runner=runner)
    app = ["/python path/python3","worker.py","--variant","vector_pv_prefetch"]
    command = collect.command_for(tool,{"kernel_prefix":"VcLaserQuantVscalePvPrefetchAttention"},app,tmp_path/"raw")
    assert command[-len(app):] == app
    assert "--aic-metrics=Default" in command
    assert "--launch-count=1" in command and "--warm-up=5" in command
    assert "--replay-mode=kernel" in command and "--kill=off" in command
    assert not any("Timeline" in value or "Source" in value for value in command)


def test_missing_capability_fails_without_device_command(tmp_path):
    calls=[]
    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0,stdout="--output",stderr="")
    with pytest.raises(RuntimeError,match="no NPU call"):
        collect.discover("/cann",tmp_path,explicit="/tool",runner=runner)
    assert calls == [["/tool","--help"]]
    assert json.loads((tmp_path/"discovery.json").read_text(encoding="utf-8"))["selected"] is None


def test_optional_flags_are_only_used_when_advertised():
    command=collect.command_for(dict(command=["msprof","op"],flags=list(collect.REQUIRED)),
        dict(kernel_prefix="VcLaserAttention"),["python3","worker.py"],Path("out"))
    assert not any(x.startswith("--kill") or x.startswith("--replay-mode") for x in command)
    for prefix in ("", "VcLaser*", "VcLaser|Other", "VcLaser;echo"):
        with pytest.raises(ValueError,match="literal"):
            collect.command_for(dict(command=["tool"],flags=[]),dict(kernel_prefix=prefix),[],Path("out"))


@pytest.mark.parametrize("record", [dict(returncode=1),dict(returncode=0,timed_out=True),dict(returncode=0,interrupted=True)])
def test_failed_process_never_accepts_stale_passed_worker(tmp_path,record):
    path=tmp_path/"worker.json"
    path.write_text('{"status":"passed"}')
    with pytest.raises(RuntimeError,match="no subsequent"):
        collect.require_worker(dict(record,log="error.log"),path)


def test_successful_profiler_exit_still_requires_worker(tmp_path):
    path=tmp_path/"worker.json"
    record=dict(returncode=0,log="tool.log")
    with pytest.raises(RuntimeError,match="no worker"):
        collect.require_worker(record,path)
    path.write_text('{"status":"failed"}')
    with pytest.raises(RuntimeError,match="Worker failed"):
        collect.require_worker(record,path)


def test_replay_must_preserve_same_input_and_sample_rows():
    smoke = dict(input_sha256=["q","k","v"],output_sample_sha256="out",accuracy_scope=dict(query_ids=[0,31]))
    assert collect.compare_replay(smoke,smoke)["sampled_output_exact"]
    for key,value in (("input_sha256",["other"]),("output_sample_sha256","wrong"),
                      ("accuracy_scope",dict(query_ids=[1,30]))):
        with pytest.raises(RuntimeError,match="differs|different"):
            collect.compare_replay(smoke,dict(smoke,**{key:value}))


def test_expcast_requires_current_passed_validation(tmp_path,monkeypatch):
    monkeypatch.setattr(collect,"ROOT",tmp_path)
    (tmp_path/"results").mkdir()
    evidence={key:key for key in ("build_manifest_sha256","source_manifest_sha256","bridge_sha256","device_binary_sha256")}
    evidence["preprocessor"] = dict(build_manifest_sha256="prep-manifest", library_sha256="prep-library")
    prep = dict(build_manifest_sha256="prep-manifest", library_sha256="prep-library")
    # Match profile_mindie_laser_fp16_expcast.verify_builds: the quantizer is
    # shared, so its provenance is a sibling of the attention providers.
    original = {key:value for key,value in evidence.items() if key != "preprocessor"}
    payload=dict(status="passed",workers=dict(paired=dict(status="passed",
        provenance=dict(fp16_expcast=original, preprocessor=prep))))
    (tmp_path/"results/mindie_laser_fp16_expcast.json").write_text(json.dumps(payload))
    info=dict(variant="fp16_expcast",provenance=dict(evidence))
    collect.require_expcast_validation(info)
    info["provenance"]["bridge_sha256"]="changed"
    with pytest.raises(RuntimeError,match="artifacts differ"):
        collect.require_expcast_validation(info)
    info["provenance"]["bridge_sha256"] = evidence["bridge_sha256"]
    info["provenance"]["preprocessor"] = dict(build_manifest_sha256="prep-manifest", library_sha256="rebuilt")
    with pytest.raises(RuntimeError,match="preprocessor differs"):
        collect.require_expcast_validation(info)


def test_timeout_preserves_evidence_even_when_termination_fails(tmp_path, monkeypatch):
    class Process:
        pid = 123
        returncode = None
        def wait(self, **kwargs):
            raise subprocess.TimeoutExpired("worker", kwargs["timeout"])
        def terminate(self):
            raise ProcessLookupError("process group exited")
        kill = terminate
    def killpg(*args):
        raise ProcessLookupError("process group exited")
    monkeypatch.setattr(collect.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(collect.os, "killpg", killpg, raising=False)
    times = iter([0, 2, 3])
    monkeypatch.setattr(collect.time, "monotonic", lambda: next(times))
    record = collect.run_process(["worker"], {}, tmp_path/"log.txt", timeout=1)
    assert record["timed_out"] is True
    assert record["device_tasks_stopped"] is None
    assert record["termination"]["actions"] == ["terminate", "kill"]
    assert len(record["termination"]["errors"]) == 4
    assert record["returncode"] is None
    with pytest.raises(RuntimeError, match="no subsequent"):
        collect.require_worker(record, tmp_path/"worker.json")


def fixture_dependencies(monkeypatch,tmp_path):
    import mindie_laser_msopprof_worker as worker
    import summarize_mindie_laser_msopprof as summary
    monkeypatch.setenv("ASCEND_HOME_PATH",str(tmp_path/"cann"))
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES",raising=False)
    monkeypatch.setattr(worker,"provider_info",lambda name,**kw:dict(variant=name,kernel_prefix="Kernel"+name,
        cann_root=str(tmp_path/"cann"),output=str(kw["output"]),provenance={}))
    monkeypatch.setattr(worker,"worker_environment",lambda env,info:dict(env))
    monkeypatch.setattr(collect,"discover",lambda *a,**kw:dict(command=["tool"],flags=list(collect.REQUIRED)))
    def summarize(run_root):
        evidence = dict(status="partial", manifest_sha256=hashlib.sha256((run_root/"manifest.json").read_bytes()).hexdigest())
        (run_root/"summary.json").write_text(json.dumps(evidence))
        (run_root/"summary.txt").write_text("Partial: no verified target CSV\n")
        return evidence
    monkeypatch.setattr(summary,"write_summary",summarize)


def args_for(tmp_path):
    return SimpleNamespace(output=str(tmp_path/"report.json"),shape=[1,28,37741,128],device=0,
        variants=list(collect.DEFAULTS),tool=None,check_only=False,timeout=10)


def test_failed_smoke_stops_before_profiler_and_next_variant(tmp_path,monkeypatch):
    fixture_dependencies(monkeypatch,tmp_path)
    calls=[]
    def failed(command,env,log_path,**kwargs):
        calls.append(command)
        return dict(command=command,returncode=1,log=str(log_path))
    monkeypatch.setattr(collect,"run_process",failed)
    assert collect.collect(args_for(tmp_path))==1
    assert len(calls)==1 and calls[0][0]==sys.executable
    report=json.loads((tmp_path/"report.json").read_text(encoding="utf-8"))
    assert report["further_device_work_aborted"]
    assert (tmp_path/"report_summary.txt").read_text(encoding="utf-8").startswith("Collection failed")


def test_no_matching_csv_cannot_be_reported_as_success(tmp_path,monkeypatch):
    fixture_dependencies(monkeypatch,tmp_path)
    calls=[]
    def passed(command,env,log_path,**kwargs):
        calls.append(command)
        path=Path(command[command.index("--output")+1])
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(status="passed", input_sha256=["q","k","v"],
            output_sample_sha256="valid", accuracy_scope=dict(query_ids=[0,31]))))
        return dict(command=command,returncode=0,log=str(log_path))
    monkeypatch.setattr(collect,"run_process",passed)
    assert collect.collect(args_for(tmp_path))==2
    assert len(calls)==4
    report = json.loads((tmp_path/"report.json").read_text(encoding="utf-8"))
    assert report["status"]=="partial"
    run_root = Path(report["run_root"])
    summary = json.loads((run_root/"summary.json").read_text(encoding="utf-8"))
    assert summary["manifest_sha256"] == hashlib.sha256((run_root/"manifest.json").read_bytes()).hexdigest()
    assert json.loads((run_root/"manifest.json").read_text(encoding="utf-8"))["status"] == "collection_commands_passed"


def test_check_only_never_starts_application(tmp_path,monkeypatch):
    fixture_dependencies(monkeypatch,tmp_path)
    monkeypatch.setattr(collect,"run_process",lambda *a,**kw:pytest.fail("No application allowed"))
    args=args_for(tmp_path)
    args.check_only=True
    assert collect.collect(args)==0
    assert json.loads((tmp_path/"report.json").read_text(encoding="utf-8"))["status"]=="host_preflight_passed"
