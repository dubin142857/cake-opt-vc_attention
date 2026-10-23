#!/usr/bin/env python3
"""Small isolated Laser ABI/constant-mode checks; no device timing or old probe launch."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import traceback
from types import SimpleNamespace

PRODUCTION, ABI = "vector_pv_prefetch", "pv_abi"
KINDS = (ABI, "pv_const")
TRACE_SHAPE = (4, 3, 3, 64)
SCOPE = (
    "ABI-only isolation: original V-only Cube/Vector, extra host-zeroed trace output, mode 0 only. "
    "Small deterministic cases; shared payload; production sync then ABI-only sync; ALL padded outputs exact. "
    "No old instrumented provider, no long-shape timing, no diagnosis or fixed-device-kernel claim. "
    "Worker timeout bounds a host diagnostic process only; termination does not prove device tasks stopped."
)
MANIFESTS = {
    "production_build": "results/mindie_laser_quant_vscale_pv_prefetch_build.json",
    "abi_build": "results/mindie_laser_cube_pv_abi_probe_build.json",
    "preprocess_build": "results/mindie_laser_quant_preprocess_build.json",
}


def kind_config(kind):
    if kind not in KINDS:
        raise ValueError(f"Unsupported isolation kind: {kind}")
    if kind == ABI:
        return dict(label="ABI-only", scope=SCOPE, failure_origin="abi_only_validation")
    return dict(label="constant-mode", failure_origin="constant_mode_validation", scope=(
        "Constant-mode isolation: device initializer gets literal mode 0; host ABI retains mode 0 only. "
        "No timing collection or old runtime-mode probe launch. Small deterministic cases; shared payload; "
        "production sync then candidate sync; ALL padded outputs exact and host-zeroed trace remains zero. "
        "No long-shape timing, performance, root-cause or fixed-device-kernel claim. "
        "Worker timeout bounds a host diagnostic process only; termination does not prove device tasks stopped."
    ))


def resolve_defaults(args):
    """Resolve kind-dependent paths without importing torch or accessing a device."""
    kind = getattr(args, "kind", ABI)
    kind_config(kind)
    args.kind = kind
    if getattr(args, "abi_build", None) is None:
        args.abi_build = f"results/mindie_laser_cube_{kind}_probe_build.json"
    if getattr(args, "output", None) is None:
        args.output = f"results/mindie_laser_{kind}.json"
    return args


def resolve_providers(args):
    resolve_defaults(args)
    import benchmark_mindie_laser_quant as quant
    import prepare_mindie_laser_quant_vscale_pv_prefetch_source as source
    import prepare_mindie_laser_cube_probe_source as abi_source
    providers = {}
    for name, build in ((PRODUCTION, args.production_build), (args.kind, args.abi_build)):
        options = SimpleNamespace(**vars(args), quant_build=build, warmup=0, repeats=1, trials=1)
        provider = quant.build_arguments(options, "qk_int8_fused")
        provider.marker_env = source.MARKER_ENV if name == PRODUCTION else abi_source.marker_environment(args.kind)
        provider.marker_token = source.MARKER_TOKEN if name == PRODUCTION else abi_source.marker_token(args.kind)
        providers[name] = provider
    if len({str(Path(p.vendor_root).resolve()) for p in providers.values()}) != 2:
        raise ValueError("Production and isolation candidate need two distinct vendor roots")
    return providers


def verify_builds(args, providers):
    resolve_defaults(args)
    import benchmark_mindie_laser_quant as quant
    import prepare_mindie_laser_quant_vscale_pv_prefetch_source as source
    import prepare_mindie_laser_cube_probe_source as abi_source
    from mindie_laser_quant_fused import validate_preprocess_manifest
    return {
        PRODUCTION: quant.quant_provenance(providers[PRODUCTION], identity=source.IDENTITY,
                                           source_validator=source.validate_prepared_source),
        args.kind: quant.quant_provenance(providers[args.kind], identity=abi_source.identity(args.kind),
            source_validator=lambda path, manifest: abi_source.validate_prepared_source(path, manifest, kind=args.kind)),
        "preprocessor": validate_preprocess_manifest(args.preprocess_build),
    }


def check_schema(native, *, abi):
    arguments = ["Tensor"]*5+["float", "int", "bool"]+(["int"] if abi else [])
    returns = ["Tensor", "Tensor"] if abi else ["Tensor"]
    schema = native._schema
    if ([str(item.type) for item in schema.arguments] != arguments or
            [str(item.type) for item in schema.returns] != returns):
        raise RuntimeError(f"Unexpected {'ABI-only' if abi else 'production'} schema: {schema}")


def fixture_cases():
    from mindie_laser_vscale_reference import MAPPING_CASES
    return (dict(next(case for case in MAPPING_CASES if case["name"] == "tiny")),
            dict(name="qk_tails", batch=1, heads=2, queries=17, keys=129, dim=64),
            dict(name="three_key_blocks", batch=1, heads=2, queries=65, keys=257, dim=128))


def abi_call(native, payload):
    return native(payload.q, payload.k, payload.v, payload.q_scale, payload.k_scale,
                  payload.scale, payload.keys, False, 0)


def fresh_markers(path, offset, *, pid, token):
    """Require first-call evidence and validate every later exported marker."""
    import benchmark_mindie_laser_quant as quant
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        fresh = stream.read().decode("utf-8")
        end = stream.tell()
    # Identical native metadata may reuse an executor after the first tiling.
    # Absence on a later call is explicitly not new provider evidence.
    records = quant.parse_quant_markers(fresh, pid, token) if fresh.strip() or offset == 0 else []
    return end, dict(records=records, new_records=len(records),
                    reused_first_call_evidence=not records,
                    scope="Host tiling markers; no fresh marker means no per-call identity proof")


def check_result(result, payload, expected, *, kind=ABI):
    import torch
    from profile_mindie_laser_quant import compare_all_rows
    from profile_mindie_laser_vector_scale_bulk import require_exact
    label = kind_config(kind)["label"]
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError(f"{label} must return (attention, trace)")
    out, trace = result
    if (not isinstance(out, torch.Tensor) or out.shape != payload.q.shape or
            out.dtype != torch.float32 or out.device != payload.q.device):
        raise RuntimeError(f"{label} must return padded FP32 attention on the payload device")
    if (not isinstance(trace, torch.Tensor) or tuple(trace.shape) != TRACE_SHAPE or
            trace.dtype != torch.int64 or trace.device != payload.q.device):
        raise RuntimeError(f"{label} trace must be int64 [4,3,3,64] on the payload device")
    host_trace = trace.cpu()
    if bool(host_trace.any()):
        raise AssertionError(f"{label} trace must be entirely zero")
    error = compare_all_rows(out, expected)
    require_exact(error)
    return dict(all_padded_output=error, trace_shape=list(TRACE_SHAPE), trace_dtype=str(trace.dtype), trace_all_zero=True)


def validate_cases(production, native, device, report, save, synchronize, *, after_call=None, kind=ABI):
    import torch
    from mindie_laser_vscale_reference import _mapping_inputs
    from profile_mindie_laser_pv_phases import ProbeValidationGuard
    config = kind_config(kind)
    guard = ProbeValidationGuard(report, save)
    report["cases"] = []
    try:
        # Both dtypes of the original failing tiny fixture precede every tail.
        for case in fixture_cases():
            for dtype in (torch.float16, torch.bfloat16):
                row = dict(case, dtype=str(dtype), status="preparing", probe_mode=0, expcast=False)
                report["cases"].append(row)
                spec = SimpleNamespace(shape=(case["batch"], case["heads"], case["queries"], case["dim"]),
                                       keys=case["keys"], dtype=dtype)
                guard.mark(spec, "prepare", "about_to_prepare", isolated=True)
                inputs = tuple(x.to(device) for x in _mapping_inputs(case, dtype))
                payload = production.prepare(*inputs)
                # Preparation completes before the first native attribution.
                guard.mark(payload, "prepare", "awaiting_sync", isolated=True)
                synchronize()
                guard.mark(payload, PRODUCTION, "about_to_launch", isolated=True)
                expected = production.run_native(payload)
                guard.mark(payload, PRODUCTION, "awaiting_sync", isolated=True)
                synchronize()
                if after_call:
                    guard.mark(payload, PRODUCTION, "host_marker_check", isolated=True)
                    after_call(PRODUCTION, row)
                guard.mark(payload, kind, "about_to_launch", isolated=True)
                result = abi_call(native, payload)
                guard.mark(payload, kind, "awaiting_sync", isolated=True)
                synchronize()
                if after_call:
                    guard.mark(payload, kind, "host_marker_check", isolated=True)
                    after_call(kind, row)
                guard.mark(payload, kind, "trace_and_all_output_cpu", isolated=True)
                row.update(check_result(result, payload, expected, kind=kind), status="passed", probe_mode=0, expcast=False)
                guard.mark(payload, kind, "completed", isolated=True)
                print(f"{config['label']} passed: {dtype} B{case['batch']} H{case['heads']} "
                      f"Q{case['queries']} K{case['keys']} D{case['dim']}; ALL padded outputs exact; trace zero", flush=True)
                del inputs, payload, expected, result
    except Exception as error:
        guard.fail(error, origin=config["failure_origin"])
        if report["cases"]:
            report["cases"][-1].update(status="failed", error=f"{type(error).__name__}: {error}")
        guard.save_failure()
        raise


def worker(args):
    resolve_defaults(args)
    config = kind_config(args.kind)
    import benchmark_mindie_laser_quant as quant
    directory = Path(args._run_dir)/"paired"/"local"
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", stage="provenance", pid=os.getpid(), kind=args.kind, scope=config["scope"], cases=[])
    save = lambda: quant.common.write_json(directory/"report.json", report)
    save()
    try:
        providers = resolve_providers(args)
        report["provenance"] = evidence = verify_builds(args, providers)
        report["marker_files"] = {name: os.environ[p.marker_env] for name, p in providers.items()}
        save()
        import torch
        import torch_npu
        from mindie_laser_quant_backend import load_quant_laser
        from profile_mindie_laser_quant import verify_runtime_target
        from mindie_laser_scale_reuse_checks import inspect_providers
        torch.set_num_threads(min(args.cpu_threads, torch.get_num_threads()))
        torch.npu.set_device(args.device)
        torch.npu.set_option({"ACL_OP_COMPILER_CACHE_MODE": "disable",
                              "ACL_OP_COMPILER_CACHE_DIR": os.environ["ACL_OP_COMPILER_CACHE_DIR"]})
        production = load_quant_laser(providers[PRODUCTION].bridge_path, preprocessing="fused",
            preprocess_build=args.preprocess_build, torch_namespace="vc_laser_quant_vscale_pv_prefetch",
            v_quantize=False, expcast=False, check_finite=False)
        torch.ops.load_library(providers[args.kind].bridge_path)
        native = getattr(torch.ops, f"vc_laser_cube_{args.kind}_probe").la.default
        check_schema(production.native, abi=False)
        check_schema(native, abi=True)
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtGetSocName.restype = ctypes.c_char_p
        soc = acl.aclrtGetSocName()
        if not soc: raise RuntimeError("aclrtGetSocName returned null")
        soc, cann = soc.decode(), os.environ["ASCEND_HOME_PATH"]
        prep = json.loads(Path(args.preprocess_build).read_text(encoding="utf-8"))
        report["runtime_targets"] = {name: verify_runtime_target(evidence[name]["build_manifest"], prep, soc, cann)
                                     for name in providers}
        report["environment"] = dict(torch=torch.__version__, torch_npu=torch_npu.__version__, runtime_soc=soc,
            device_index=args.device, production_schema=str(production.native._schema), abi_schema=str(native._schema))
        offsets = {name: 0 for name in providers}
        def marker_check(name, row):
            path = Path(report["marker_files"][name])
            end, markers = fresh_markers(path, offsets[name], pid=os.getpid(), token=providers[name].marker_token)
            offsets[name] = end
            row.setdefault("host_markers", {})[name] = markers
        report["stage"] = "small_validation"
        save()
        with torch.no_grad():
            validate_cases(production, native, f"npu:{args.device}", report, save,
                           torch.npu.synchronize, after_call=marker_check, kind=args.kind)
        inspect_providers(providers, report, directory, cann_root=cann, soc=evidence[PRODUCTION]["build_manifest"]["soc"])
        quant.inspect_fused_preprocessor(providers[PRODUCTION], report)
        report.update(status="passed", stage="complete", conclusion=f"Small {config['label']} checks passed; no conclusion about old probe failure cause")
        save()
        return 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                      further_device_access_aborted=True)
        # No trace reads, synchronization, provider inspection or new work here.
        try: save()
        except Exception as save_error:
            print(f"Host-only failure report save failed: {save_error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


def terminate_worker(process):
    """Bound host-process cleanup; never claim an accelerator task was cancelled."""
    actions = []
    for label, sig in (("terminate", signal.SIGTERM), ("kill", getattr(signal, "SIGKILL", signal.SIGTERM))):
        try:
            if os.name == "posix": os.killpg(process.pid, sig)
            else: getattr(process, label)()
            actions.append(label)
        except ProcessLookupError:
            actions.append("already_exited")
        except OSError as error:
            actions.append(f"{label}: {type(error).__name__}: {error}")
        try:
            process.wait(timeout=5)
            break
        except subprocess.TimeoutExpired:
            actions.append(label+"_wait_timeout")
        except OSError as error:
            actions.append(f"{label}_wait: {type(error).__name__}: {error}")
    return dict(actions=actions, host_returncode=process.poll(), device_tasks_stopped=None)


def report_summary(report):
    """Extract recorded host evidence only; do not infer a device failure from a deadline."""
    if not isinstance(report, dict):
        raise ValueError("Validation report must be a JSON object")
    workers = report.get("workers")
    child = workers.get("paired", {}) if isinstance(workers, dict) else report
    if not isinstance(child, dict):
        child = {}
    failure = child.get("probe_validation_failure")
    failure = failure if isinstance(failure, dict) else {}
    active = child.get("active_probe_validation")
    active = active if isinstance(active, dict) else {}
    context = failure or active
    summary = dict(status=report.get("status"), worker_status=child.get("status"),
        timed_out=child.get("timed_out"), returncode=child.get("returncode"),
        active={key: context.get(key) for key in ("mode", "stage", "shape_bnsd", "keys", "dtype")},
        first_recorded_validation_error=failure.get("error"),
        worker_error=child.get("error"), parent_error=report.get("error") if child is not report else None,
        host_deadline_seconds=child.get("worker_timeout_seconds"),
        host_timeout_error=child.get("host_timeout_error"), host_termination=child.get("termination"),
        worker_log=child.get("worker_log"), worker_report=child.get("report_path"))
    if child.get("timed_out"):
        summary["deadline_scope"] = (
            "Host worker deadline exceeded; this is not evidence of a device timeout. "
            "Host termination does not prove queued device tasks stopped."
        )
    # A guard can also record a host validation error. Preserve its wording,
    # rather than relabel every exception as a device fault.
    return summary


def print_report_summary(report, *, stream=None):
    stream = sys.stderr if stream is None else stream
    print("Host-only validation report summary (recorded evidence):", file=stream, flush=True)
    print(json.dumps(report_summary(report), ensure_ascii=False, indent=2), file=stream, flush=True)


def summarize_report(path):
    """Read an existing JSON report without provider checks, builds, imports or NPU work."""
    with Path(path).open(encoding="utf-8") as stream:
        report = json.load(stream)
    print_report_summary(report, stream=sys.stdout)
    return 0


def launch(args, providers):
    resolve_defaults(args)
    config = kind_config(args.kind)
    from mindie_laser_scale_reuse_checks import paired_environment
    directory = Path(args._run_dir)/"paired"/"local"
    directory.mkdir(parents=True, exist_ok=True)
    env = paired_environment(os.environ, providers, directory.parent)
    for key in ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker", "--_run-dir", args._run_dir]
    for key in (*MANIFESTS, "device", "cpu_threads", "kind"):
        command += ["--"+key.replace("_", "-"), str(getattr(args, key))]
    logfile, path = directory/"worker.log", directory/"report.json"
    timeout, termination = False, None
    print(f"Starting {config['label']} worker (host timeout {args.worker_timeout:g}s); log: {logfile}", flush=True)
    with logfile.open("wb") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=os.name == "posix")
        try:
            process.wait(timeout=args.worker_timeout)
        except subprocess.TimeoutExpired:
            timeout = True
            termination = terminate_worker(process)
    try:
        result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status": "missing_report"}
    except (OSError, ValueError) as error:
        result = {"status": "invalid_report", "report_error": f"{type(error).__name__}: {error}"}
    result.update(returncode=process.poll(), command=command, worker_log=str(logfile),
                  report_path=str(path), timed_out=timeout, worker_timeout_seconds=args.worker_timeout)
    if timeout:
        deadline_error = "Host worker exceeded diagnostic timeout; no subsequent NPU task will be launched"
        result.update(status="timed_out", prior_worker_status=result["status"], termination=termination,
                      host_timeout_error=deadline_error)
        result.setdefault("error", deadline_error)
    try: print(logfile.read_text(encoding="utf-8", errors="replace"), end="", flush=True)
    except OSError as error: result["log_read_error"] = str(error)
    return result


def parent(args):
    resolve_defaults(args)
    config = kind_config(args.kind)
    import benchmark_mindie_laser_quant as quant
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args._run_dir = tempfile.mkdtemp(prefix=f"mindie_laser_{args.kind}_", dir=output.parent)
    report = dict(status="running", kind=args.kind, scope=config["scope"], run_dir=args._run_dir, workers={})
    save = lambda: quant.common.write_json(output, report)
    save()
    try:
        providers = resolve_providers(args)
        report["provenance"] = verify_builds(args, providers)
        save()
        report["workers"]["paired"] = result = launch(args, providers)
        save()
        if result.get("timed_out") or result.get("returncode") != 0 or result.get("status") != "passed":
            raise RuntimeError(f"{config['label']} validation failed; retain report/worker log and stop device work")
        report["status"] = "passed"
        save()
        print(f"Small {config['label']} validation passed; no performance or old-probe fix claim. Report: {output}", flush=True)
        return 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", no_additional_device_work=True)
        save()
        print(f"{config['label']} validation stopped: {error}\nReport: {output}", file=sys.stderr, flush=True)
        print_report_summary(report)
        return 1


class IsolationParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        return resolve_defaults(super().parse_args(args, namespace))


def parser():
    result = IsolationParser(description=__doc__)
    result.add_argument("--summarize-report", metavar="PATH",
        help="Read existing parent/worker JSON only; no build, provider checks, torch imports or device work")
    result.add_argument("--kind", choices=KINDS, default=ABI)
    for name, default in MANIFESTS.items():
        result.add_argument("--"+name.replace("_", "-"), default=None if name == "abi_build" else default)
    result.add_argument("--output", default=None)
    result.add_argument("--device", type=int, default=0)
    result.add_argument("--cpu-threads", type=int, default=4)
    result.add_argument("--worker-timeout", type=float, default=180)
    result.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--_run-dir", help=argparse.SUPPRESS)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.summarize_report:
        if args._worker:
            cli.error("--summarize-report cannot launch a worker")
        try:
            return summarize_report(args.summarize_report)
        except (OSError, ValueError) as error:
            cli.error(f"Cannot summarize existing report: {type(error).__name__}: {error}")
    if args.device < 0 or args.cpu_threads <= 0 or not 0 < args.worker_timeout < float("inf"):
        cli.error("Need nonnegative device, positive CPU threads and finite positive worker timeout")
    return worker(args) if args._worker else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
