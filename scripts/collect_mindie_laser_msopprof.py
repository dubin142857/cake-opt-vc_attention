#!/usr/bin/env python3
"""Collect msOpProf counters from existing, separately isolated Laser kernels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = ("original_laser", "vector_pv_prefetch")
VARIANTS = (*DEFAULTS, "fp16_expcast_control", "fp16_expcast")
WORKER = ROOT / "scripts/mindie_laser_msopprof_worker.py"
REQUIRED = {"--kernel-name", "--launch-count", "--warm-up", "--aic-metrics", "--output"}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def discover(cann, directory, *, explicit=None, runner=subprocess.run):
    """Read local help only; never install or invoke a profiling application."""
    directory.mkdir(parents=True, exist_ok=True)
    options = []
    if explicit:
        options.append([explicit])
    else:
        for candidate in (shutil.which("msopprof"), str(Path(cann)/"tools/msopprof/bin/msopprof")):
            if candidate and Path(candidate).is_file() and [candidate] not in options:
                options.append([candidate])
        for candidate in (shutil.which("msprof"), str(Path(cann)/"tools/profiler/bin/msprof")):
            if candidate and Path(candidate).is_file() and [candidate,"op"] not in options:
                options.append([candidate,"op"])
    diagnostics = []
    for index, command in enumerate(options):
        record = dict(command=command+["--help"])
        try:
            result = runner(record["command"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=20)
            help_text = result.stdout+"\n"+result.stderr
            path = directory/f"help_{index}.txt"
            path.write_text(help_text, encoding="utf-8")
            flags = set(re.findall(r"--[a-z][a-z0-9-]*", help_text))
            record.update(returncode=result.returncode, help_file=str(path), missing=sorted(REQUIRED-flags))
            diagnostics.append(record)
            if result.returncode == 0 and not record["missing"]:
                selected = dict(command=command, flags=sorted(flags), help_file=str(path))
                write_json(directory/"discovery.json", dict(attempts=diagnostics, selected=selected))
                return selected
        except (OSError, subprocess.TimeoutExpired) as error:
            record["error"] = str(error)
            diagnostics.append(record)
    write_json(directory/"discovery.json", dict(attempts=diagnostics, selected=None))
    raise RuntimeError("No usable msopprof / msprof op found; see tool/discovery.json. "
                       "Use MSOPPROF_BIN for an existing local msopprof executable; no NPU call was made.")


def command_for(tool, info, worker_command, output):
    prefix = info["kernel_prefix"]
    if not re.fullmatch(r"[A-Za-z0-9_]+", prefix):
        raise ValueError("Kernel prefix must be a literal CCE symbol prefix")
    command = [*tool["command"], "--kernel-name="+prefix, "--launch-count=1",
               "--warm-up=5", "--aic-metrics=Default", "--output="+str(output)]
    # One application call; the profiler replays it to collect counter groups.
    # launch-count is a selection limit, not a benchmark repetition count.
    if "--replay-mode" in tool["flags"]:
        command.append("--replay-mode=kernel")
    if "--kill" in tool["flags"]:
        command.append("--kill=off")
    return command+worker_command


def run_process(command, env, log_path, *, timeout):
    """Bound the host process group; a failure never triggers another device job."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for key in ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME"):
        if env.get(key):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
    record = dict(command=command, log=str(log_path), timed_out=False)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != "nt")
        start = time.monotonic()
        try:
            while True:
                remaining = timeout-(time.monotonic()-start)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    record["returncode"] = process.wait(timeout=min(30, remaining))
                    break
                except subprocess.TimeoutExpired:
                    print(f"  running {int(time.monotonic()-start)}s; log: {log_path}", flush=True)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            record.update(timed_out=isinstance(error, subprocess.TimeoutExpired),
                          interrupted=isinstance(error, KeyboardInterrupt), device_tasks_stopped=None)
            termination = record["termination"] = dict(actions=[], errors=[], device_tasks_stopped=None)
            # Exit races and an unresponsive host process must not discard the
            # timeout evidence or accidentally permit the next device job.
            for action, sig in (("terminate", signal.SIGTERM), ("kill", getattr(signal, "SIGKILL", 9))):
                termination["actions"].append(action)
                try:
                    if os.name == "nt":
                        getattr(process, action)()
                    else:
                        os.killpg(process.pid, sig)
                except OSError as stop_error:
                    termination["errors"].append(f"{action}: {type(stop_error).__name__}: {stop_error}")
                try:
                    process.wait(timeout=5)
                    break
                except (subprocess.TimeoutExpired, OSError) as wait_error:
                    termination["errors"].append(f"{action} wait: {type(wait_error).__name__}: {wait_error}")
            record["returncode"] = process.returncode
            termination["host_returncode"] = process.returncode
        record["host_elapsed_seconds"] = time.monotonic()-start
    return record


def require_worker(record, report_path):
    if record.get("timed_out") or record.get("interrupted") or record.get("returncode") != 0:
        raise RuntimeError("Application/profiler failed; no subsequent NPU job launched. See "+record["log"])
    if not report_path.is_file():
        raise RuntimeError("Application produced no worker report; refusing to declare successful collection")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise RuntimeError("Worker failed; see "+str(report_path))
    return report


def compare_replay(smoke, profiled):
    for key in ("input_sha256", "output_sample_sha256"):
        if not smoke.get(key) or smoke[key] != profiled.get(key):
            raise RuntimeError("Profiled replay differs from the unprofiled single call: "+key)
    first = smoke.get("accuracy_scope", {}).get("query_ids")
    if not first or first != profiled.get("accuracy_scope", {}).get("query_ids"):
        raise RuntimeError("Profiled and unprofiled output samples use different query rows")
    return dict(inputs_identical=True, sampled_output_exact=True, query_ids=first,
                scope="Postprocessed valid BF16 query samples, all heads/features; not ALL output equivalence")


def require_expcast_validation(info):
    """Experimental providers require a successful validation of these artifacts."""
    if info["variant"] not in ("fp16_expcast_control", "fp16_expcast"):
        return
    path = ROOT/"results/mindie_laser_fp16_expcast.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    worker = report.get("workers", {}).get("paired", {})
    if report.get("status") != "passed" or worker.get("status") != "passed":
        raise RuntimeError("Validate FP16 ExpCast with run_mindie_laser_fp16_expcast.sh first")
    provenance = worker.get("provenance", {})
    old = provenance.get("fp16_expcast", {})
    now = info["provenance"]
    for key in ("build_manifest_sha256", "source_manifest_sha256", "bridge_sha256", "device_binary_sha256"):
        if key not in now or old.get(key) != now[key]:
            raise RuntimeError("FP16 ExpCast artifacts differ from the validated report: "+key)
    for key in ("build_manifest_sha256", "library_sha256"):
        value = now.get("preprocessor", {}).get(key)
        # The four-arm benchmark stores its shared preprocessor once beside
        # the providers; this single-provider worker nests it in its evidence.
        if not value or provenance.get("preprocessor", {}).get(key) != value:
            raise RuntimeError("FP16 ExpCast preprocessor differs from the validated report: "+key)


def collect(args):
    from mindie_laser_msopprof_worker import provider_info, worker_environment
    from summarize_mindie_laser_msopprof import write_summary
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="mindie_laser_msopprof_", dir=output.parent))
    text_path = output.with_name(output.stem+"_summary.txt")
    text_path.write_text(f"Collection pending; current run: {run_root}\n", encoding="utf-8")
    report = dict(schema="laser-msopprof-v1", status="running", run_root=str(run_root),
                  shape_bnsd=args.shape, dtype="bfloat16", device=args.device,
                  collections=[], smoke=[], requested_metrics="Default", warm_up=5, launch_count=1,
                  scope="Counter diagnostics; isolated processes; not paired API speedup or phase percentages",
                  replay_note="Kernel replay may clear L2 cache; do not compare its time directly with unprofiled API timings")
    manifest_frozen = False
    def save():
        if not manifest_frozen:
            write_json(run_root/"manifest.json", report)
        write_json(output, report)
    save()
    try:
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip():
            raise RuntimeError("Unset ASCEND_RT_VISIBLE_DEVICES before msOpProf; select the physical card with NPU_DEVICE")
        cann = os.environ.get("ASCEND_HOME_PATH")
        if not cann:
            raise RuntimeError("ASCEND_HOME_PATH missing; use run_mindie_laser_msopprof.sh")
        report["tool"] = discover(cann, run_root/"tool", explicit=args.tool)
        save()
        # All source/artifact checks happen before the first NPU launch.
        for name in args.variants:
            path = run_root/name/"profile_worker.json"
            info = provider_info(name, device=args.device, shape=args.shape, output=path)
            if Path(info["cann_root"]).resolve() != Path(cann).resolve():
                raise RuntimeError("Provider CANN root differs from the selected toolkit")
            require_expcast_validation(info)
            report["collections"].append(dict(variant=name, output_dir=str(run_root/name/"profile"),
                kernel_prefix=info["kernel_prefix"], worker_report=str(path), provider=info, status="pending"))
        save()
        if args.check_only:
            report["status"] = "host_preflight_passed"
            save()
            text_path.write_text(f"Host checks passed; no NPU launched. Report: {output}\n", encoding="utf-8")
            print(f"Host checks passed; no NPU launched. Report: {output}", flush=True)
            return 0
        for row in report["collections"]:
            name, directory = row["variant"], run_root/row["variant"]
            def worker_command(path):
                return [sys.executable, "-u", str(WORKER), "--variant", name,
                        "--device", str(args.device), "--shape", *map(str,args.shape), "--output", str(path)]
            smoke_path = directory/"smoke_worker.json"
            smoke_info = provider_info(name, device=args.device, shape=args.shape, output=smoke_path)
            print(f"Unprofiled single-call check: {name}", flush=True)
            evidence = run_process(worker_command(smoke_path), worker_environment(os.environ, smoke_info),
                                   directory/"smoke.log", timeout=args.timeout)
            report["smoke"].append(dict(variant=name, **evidence, worker_report=str(smoke_path)))
            save()
            smoke_report = require_worker(evidence, smoke_path)
            profile_dir = Path(row["output_dir"])
            profile_dir.mkdir(parents=True, exist_ok=True)
            row["command"] = command_for(report["tool"], row["provider"], worker_command(Path(row["worker_report"])), profile_dir)
            row["status"] = "collecting"
            save()
            print(f"msOpProf Default counters: {name}; kernel={row['kernel_prefix']}", flush=True)
            evidence = run_process(row["command"], worker_environment(os.environ, row["provider"]),
                                   directory/"msopprof.log", timeout=args.timeout)
            row.update(evidence)
            save()
            profiled_report = require_worker(evidence, Path(row["worker_report"]))
            row["replay_output_check"] = compare_replay(smoke_report, profiled_report)
            row["status"] = "passed"
            save()
        report["status"] = "collection_commands_passed"
        save()
        # Freeze the collection evidence hashed by the summary. Only the outer
        # status report is updated afterwards, including on a parsing failure.
        manifest_frozen = True
        summary = write_summary(run_root)
        report["summary_json"] = str(run_root/"summary.json")
        report["summary_text"] = str(run_root/"summary.txt")
        report["summary_details_text"] = summary.get("summary_details_text")
        report["summary_status"] = summary["status"]
        report["status"] = "passed" if summary["status"] == "verified" else "partial"
        save()
        shutil.copyfile(run_root/"summary.txt", text_path)
        print(text_path.read_text(encoding="utf-8"), flush=True)
        print(f"Report: {output}\nPaste summary: {text_path}\nRaw files preserved: {run_root}", flush=True)
        return 0 if report["status"] == "passed" else 2
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", further_device_work_aborted=True)
        for row in report["collections"]:
            if row["status"] == "collecting":
                row["status"] = "failed"
        save()
        text_path.write_text(f"Collection failed: {error}\nCurrent logs: {run_root}\nReport: {output}\n",
                             encoding="utf-8")
        print(f"Stopped: {error}\nReport: {output}\nExisting logs preserved: {run_root}", file=sys.stderr, flush=True)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(DEFAULTS))
    parser.add_argument("--shape", nargs=4, type=int, default=[1,28,37741,128])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600, help="Host timeout per invocation; no automatic retry")
    parser.add_argument("--output", default="results/mindie_laser_msopprof.json")
    parser.add_argument("--tool", default=os.environ.get("MSOPPROF_BIN"), help="Existing local msopprof executable")
    parser.add_argument("--check-only", action="store_true", help="Verify tools and artifacts without NPU work")
    args = parser.parse_args(argv)
    if min(args.shape) < 1 or args.shape[-1] not in (64,128) or args.device < 0 or args.timeout < 1:
        parser.error("Require positive shape with D64/128, device>=0, timeout>0")
    if len(set(args.variants)) != len(args.variants):
        parser.error("Variants must be distinct")
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
