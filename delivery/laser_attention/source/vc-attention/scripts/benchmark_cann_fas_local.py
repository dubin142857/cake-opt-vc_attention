"""Compare installed CANN FAS with the official v9.0.1 source build.

Run on the Linux Ascend host. Each provider runs in a separate Python process;
the parent never imports torch_npu. Timings are synchronized full Python API
wall time, grouped by process, rather than interleaved kernel measurements.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

from prepare_cann_fas_source import MARKER_TOKEN, SOURCE_LOCK


UPSTREAM_COMMIT = SOURCE_LOCK.commit
SHAPE = (1, 28, 37741, 128)
SOURCE_SCOPE = (
    "Official ops-transformer tag v9.0.1 source reproduction; its version.cmake "
    "reports 9.0.0. This does not assert byte identity with installed CANN 9.0.1."
)
PROOF_SCOPE = (
    "The API and successful tiling host markers, loaded library paths/hashes, "
    "and source manifest establish host dispatch evidence. They do not alone "
    "prove the identity of the executed device binary. Archive the build/kernel "
    "manifest and device profiling evidence for that stronger claim."
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def validate_manifest(manifest):
    upstream = manifest.get("upstream", {})
    if upstream.get("commit") != UPSTREAM_COMMIT or upstream.get("tag") != SOURCE_LOCK.tag:
        raise ValueError("Source manifest must pin the official v9.0.1 commit " + UPSTREAM_COMMIT)
    url = upstream.get("url", "").lower().removesuffix(".git").rstrip("/")
    if url != SOURCE_LOCK.url.lower().removesuffix(".git").rstrip("/"):
        raise ValueError("Source manifest upstream must be the pinned official cann/ops-transformer repository")
    if manifest.get("marker_token") != MARKER_TOKEN:
        raise ValueError("Source manifest marker_token is missing or mismatched")
    if manifest.get("kernel_source_unchanged") is not True:
        raise ValueError("Source manifest must attest kernel_source_unchanged=true")
    for name in ("source_dir", "third_party_dir"):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise ValueError(f"Source manifest requires {name}")
    return manifest


def source_evidence(path):
    path = Path(path).resolve()
    manifest = validate_manifest(json.loads(path.read_text(encoding="utf-8")))
    source = Path(manifest["source_dir"])
    if not source.is_absolute():
        source = path.parent / source
    scope = source / "attention" / "flash_attention_score"
    hashes = {}
    if scope.is_dir():
        for directory in ("op_api", "op_host", "op_kernel"):
            for item in sorted((scope / directory).rglob("*")):
                if item.is_file():
                    hashes[str(item.relative_to(source))] = sha256_file(item)
    return {"manifest_path": str(path), "manifest_sha256": sha256_file(path),
            "manifest": manifest, "source_file_sha256": hashes,
            "source_files_available": bool(hashes), "version_scope": SOURCE_SCOPE,
            "kernel_unchanged_evidence": "Build manifest attestation; runtime source hashes are archival, not an independent upstream diff."}


def _custom_library_path(path, removed_roots):
    normalized = path.replace("\\", "/").lower()
    return (any(within(path, root) for root in removed_roots)
            or "/vendors/" in normalized
            or "/op_api/lib" in normalized
            or "/op_tiling/" in normalized and "built-in" not in normalized)


def worker_environment(base, mode, vendor_root, run_dir):
    """Called again after optional set_env.sh, before importing torch/torch_npu."""
    if mode not in ("system", "custom"):
        raise ValueError("Unknown worker mode")
    env = dict(base)
    removed_roots = [p for p in env.pop("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if p]
    removed_roots.append(str(vendor_root))
    # These can inject unrelated code/providers even without CUSTOM_OPP_PATH.
    for name in ("VC_CANN_FAS_MARKER_FILE", "LD_PRELOAD", "LD_AUDIT", "ACLNN_EXTENSION_PATH", "ACLNN_EXTENSION_SWITCH"):
        env.pop(name, None)
    for name in ("LD_LIBRARY_PATH", "PYTHONPATH"):
        paths = [p for p in env.get(name, "").split(os.pathsep)
                 if p and not _custom_library_path(p, removed_roots)]
        if paths:
            env[name] = os.pathsep.join(paths)
        else:
            env.pop(name, None)
    cache = Path(run_dir) / mode / "cache"
    env.update(ACL_OP_COMPILER_CACHE_MODE="disable", ACL_OP_COMPILER_CACHE_DIR=str(cache / "compiler"),
               ASCEND_CACHE_PATH=str(cache / "ascend"), XDG_CACHE_HOME=str(cache / "xdg"))
    if mode == "custom":
        env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor_root)
        opapi = str(Path(vendor_root) / "op_api" / "lib")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (opapi, env.get("LD_LIBRARY_PATH"))))
        env["VC_CANN_FAS_MARKER_FILE"] = str(Path(run_dir) / mode / "markers.jsonl")
    return env


def mapped_libraries(text):
    paths = set()
    for line in text.splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and parts[5].startswith("/") and ".so" in parts[5]:
            path = parts[5].removesuffix(" (deleted)")
            paths.add(str(Path(path).resolve()))
    return sorted(paths)


def verify_libraries(paths, mode, vendor_root):
    """Fail closed on registered vendor packages, including config.ini loading."""
    vendor_root = Path(vendor_root).resolve()
    custom = []
    relevant = []
    for path in paths:
        normal = path.replace("\\", "/").lower()
        if ("opapi" in normal or "optiling" in normal or "op_tiling" in normal
                or "libaclnn_ops" in normal):
            relevant.append(path)
        if ("/vendors/" in normal and ("op_api" in normal or "op_tiling" in normal)
                or "libcust_opapi" in normal
                or "/op_tiling/" in normal and "built-in" not in normal
                or "liboptiling" in normal and "cust" in Path(path).name.lower()):
            custom.append(path)
    unexpected = [p for p in custom if mode == "system" or not within(p, vendor_root)]
    if unexpected:
        raise RuntimeError("Custom vendor contamination in loaded libraries: " + ", ".join(unexpected))
    if mode == "custom":
        api = [p for p in paths if within(p, vendor_root)
               and Path(p).name == "libcust_opapi.so" and "/op_api/lib/" in p.replace("\\", "/")]
        tiling = [p for p in paths if within(p, vendor_root)
                  and "/op_tiling/" in p.replace("\\", "/") and Path(p).name.startswith("lib")]
        if not api or not tiling:
            raise RuntimeError("Custom proof requires vendor libcust_opapi.so and op_tiling/lib*.so in /proc/self/maps")
    return relevant


def parse_markers(text, pid):
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (record.get("token") != MARKER_TOKEN or record.get("phase") not in ("api", "tiling")
                or type(record.get("pid")) is not int or record["pid"] != pid
                or type(record.get("tiling_key")) is not int or record["tiling_key"] < 0):
            raise ValueError("Malformed, foreign-process or wrong-token marker: " + repr(record))
        records.append(record)
    if {item["phase"] for item in records} != {"api", "tiling"}:
        raise RuntimeError("First long-shape call must produce both fresh API and successful tiling markers")
    return records


def error_metrics(actual, expected):
    import torch
    if actual.shape != expected.shape:
        raise ValueError("Output/reference shape mismatch")
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("Output/reference contains non-finite values")
    difference = actual - expected
    rmse = difference.square().mean().sqrt().item()
    reference_rms = expected.square().mean().sqrt().item()
    return {"relative_rmse": rmse / max(reference_rms, 1e-12), "rmse": rmse,
            "reference_rms": reference_rms, "max_abs": difference.abs().max().item()}


def check_error(actual, expected, limit):
    result = error_metrics(actual, expected)
    result["limit"] = limit
    result["passed"] = result["relative_rmse"] <= limit
    return result


def dense_reference(q, k, v, query_ids, chunk=4):
    """FP32 CPU softmax over ALL keys; only query rows are sampled."""
    import torch
    result = []
    # Per head bounds temporary FP32 K/V memory for the long shape.
    for head in range(q.shape[1]):
        kh, vh = k[:, head].float().transpose(-1, -2), v[:, head].float()
        rows = []
        for ids in query_ids.split(chunk):
            qh = q[:, head].index_select(1, ids).float()
            probability = torch.softmax((qh @ kh) / math.sqrt(q.shape[-1]), dim=-1)
            rows.append(probability @ vh)
        result.append(torch.cat(rows, dim=1))
    return torch.stack(result, dim=1)


def fusion_call(torch_npu, q, k, v):
    # Match the existing baseline exactly, including default inner_precise=0.
    return torch_npu.npu_fusion_attention(
        q, k, v, q.shape[1], "BNSD", scale=1 / math.sqrt(q.shape[-1]), keep_prob=1.0,
        pre_tockens=2147483647, next_tockens=2147483647, sparse_mode=0)[0].contiguous()


def sample_payload(torch, output, ids, fingerprints):
    return {"query_ids": ids.cpu(), "output": output.detach().cpu(),
            "shape_bnsd": list(SHAPE), "seed": 0, "input_sha256": fingerprints}


def tensor_sha256(tensor):
    # Use a uint8 view because NumPy does not support BF16. Parent needs no NumPy.
    import torch
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def run_worker(args):
    mode, run_dir = args._worker, Path(args._run_dir).resolve()
    env = worker_environment(os.environ, mode, args.vendor_root, run_dir)
    if not args._env_ready:
        # set_env.sh may have changed LD_LIBRARY_PATH/LD_PRELOAD. A fresh exec,
        # not an in-process os.environ update, establishes the final loader state.
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()),
                                 *sys.argv[1:], "--_env-ready"], env)
    directory = run_dir / mode
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    report = {"status": "running", "mode": mode, "pid": os.getpid(), "proof_scope": PROOF_SCOPE}
    save = lambda: write_json(report_path, report)
    save()
    try:
        # Optional set_env.sh has already run in this subprocess's shell.
        os.environ.clear()
        os.environ.update(env)
        for name in ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME"):
            Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
        import torch
        import torch_npu
        torch.set_num_threads(min(torch.get_num_threads(), args.cpu_threads))
        torch.npu.set_option({"ACL_OP_COMPILER_CACHE_MODE": "disable",
                              "ACL_OP_COMPILER_CACHE_DIR": os.environ["ACL_OP_COMPILER_CACHE_DIR"]})
        torch.npu.set_device(args.device)
        if hasattr(torch.npu, "matmul"):
            torch.npu.matmul.allow_hf32 = False
        device = f"npu:{args.device}"
        report["environment"] = {
            "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(args.device), "python": sys.executable,
            "selected_environment": {name: os.environ.get(name) for name in (
                "ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH",
                "ACL_OP_COMPILER_CACHE_MODE", "ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH")}}
        # Preserve config.ini evidence; env clearing cannot disable global vendors.
        opp = os.environ.get("ASCEND_OPP_PATH")
        config = Path(opp) / "vendors" / "config.ini" if opp else None
        if config and config.is_file():
            report["global_vendor_config"] = {"path": str(config), "sha256": sha256_file(config),
                                               "contents": config.read_text(encoding="utf-8")}
        save()
        with torch.no_grad():
            print(f"{mode}: small BF16 dense correctness gate", flush=True)
            small_generator = torch.Generator().manual_seed(910901)
            small = [torch.randn((1, 2, length, 128), generator=small_generator).to(torch.bfloat16)
                     for length in (33, 129, 129)]
            expected = dense_reference(*small, torch.arange(33))
            small_npu = [value.to(device) for value in small]
            small_result = fusion_call(torch_npu, *small_npu).cpu()
            report["small_accuracy"] = check_error(small_result, expected, .01)
            save()
            if not report["small_accuracy"]["passed"]:
                raise AssertionError("Small-shape dense relative RMSE exceeds 1%")
            del small, small_npu, small_result, expected
            print(f"{mode}: allocate original contiguous BNSD BF16 {SHAPE}", flush=True)
            generator = torch.Generator().manual_seed(0)
            original = [torch.randn(SHAPE, generator=generator).to(torch.bfloat16) for _ in range(3)]
            fingerprints = [tensor_sha256(value) for value in original]
            q, k, v = [value.to(device) for value in original]
            ids = torch.linspace(0, SHAPE[2] - 1, 32).long().unique()
            torch.npu.synchronize()
            marker_file = directory / "markers.jsonl"
            offset = marker_file.stat().st_size if marker_file.exists() else 0
            print(f"{mode}: first long-shape call and provider proof", flush=True)
            out = fusion_call(torch_npu, q, k, v)
            torch.npu.synchronize()
            if tuple(out.shape) != SHAPE or out.dtype != torch.bfloat16 or out.device != q.device:
                raise AssertionError("FAS returned wrong shape/dtype/device")
            if mode == "custom":
                with marker_file.open("rb") as stream:
                    stream.seek(offset)
                    marker_text = stream.read().decode("utf-8")
                (directory / "long_shape_markers.jsonl").write_text(marker_text, encoding="utf-8")
                report["host_markers"] = parse_markers(marker_text, os.getpid())
                report["tiling_keys"] = sorted({item["tiling_key"] for item in report["host_markers"]
                                                if item["phase"] == "tiling"})
            maps_text = Path("/proc/self/maps").read_text(encoding="utf-8")
            (directory / "proc_self_maps.txt").write_text(maps_text, encoding="utf-8")
            paths = mapped_libraries(maps_text)
            report["mapped_library_paths"] = paths
            save()
            relevant = verify_libraries(paths, mode, args.vendor_root)
            report["loaded_library_sha256"] = {path: sha256_file(path) for path in relevant}
            report["host_provider_proof_passed"] = True
            # Helper reads getenv on every invocation. Disable file I/O before
            # warmup/timing while preserving the exact same loaded providers.
            os.environ.pop("VC_CANN_FAS_MARKER_FILE", None)
            marker_size = marker_file.stat().st_size if marker_file.exists() else 0
            report["marker_disabled_before_timing"] = True
            sample = out.index_select(2, ids.to(device)).cpu()
            del out
            sample_path = directory / "output_sample.pt"
            torch.save(sample_payload(torch, sample, ids, fingerprints), sample_path)
            report["output_sample"] = str(sample_path)
            report["input_sha256"] = fingerprints
            print(f"{mode}: FP32 CPU reference, 32 queries and ALL {SHAPE[2]} keys", flush=True)
            reference = dense_reference(*original, ids)
            report["long_accuracy"] = check_error(sample, reference, .01)
            report["checked_query_ids"] = ids.tolist()
            report["checked_keys"] = SHAPE[2]
            save()
            if not report["long_accuracy"]["passed"]:
                raise AssertionError("Long-shape sampled dense relative RMSE exceeds 1%")
            del original, sample, reference
            for _ in range(args.warmup):
                fusion_call(torch_npu, q, k, v)
            torch.npu.synchronize()
            timings = []
            for trial in range(args.trials):
                torch.npu.synchronize()
                start = time.perf_counter()
                for _ in range(args.repeats):
                    fusion_call(torch_npu, q, k, v)
                torch.npu.synchronize()
                elapsed = (time.perf_counter() - start) * 1000 / args.repeats
                timings.append(elapsed)
                report["timing_ms"] = {"trials": timings, "median": statistics.median(timings),
                                       "min": min(timings), "max": max(timings)}
                save()
                print(f"{mode}: trial {trial + 1}/{args.trials}: {elapsed:.4f} ms/API call", flush=True)
            if (marker_file.stat().st_size if marker_file.exists() else 0) != marker_size:
                raise AssertionError("Marker file changed after instrumentation was disabled")
            final_maps = mapped_libraries(Path("/proc/self/maps").read_text(encoding="utf-8"))
            verify_libraries(final_maps, mode, args.vendor_root)
            report["status"] = "passed"
            save()
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        save()
        traceback.print_exc()
        return 1


def worker_command(args, mode, run_dir):
    command = [sys.executable, str(Path(__file__).resolve()), "--_worker", mode,
               "--_run-dir", str(run_dir), "--vendor-root", str(args.vendor_root),
               "--source-manifest", str(args.source_manifest), "--device", str(args.device),
               "--warmup", str(args.warmup), "--repeats", str(args.repeats),
               "--trials", str(args.trials), "--cpu-threads", str(args.cpu_threads)]
    if args.cann_env_script:
        # Paths are positional arguments, never interpolated into shell code.
        command = ["bash", "--noprofile", "--norc", "-c",
                   'source "$1" >/dev/null && shift && exec "$@"', "cann-fas-worker",
                   str(args.cann_env_script), *command]
    else:
        command.append("--_env-ready")
    return command


def compare_samples(system_path, custom_path):
    # torch backend entry-point autoload must not import torch_npu in the parent.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch
    system = torch.load(system_path, map_location="cpu", weights_only=True)
    custom = torch.load(custom_path, map_location="cpu", weights_only=True)
    for key in ("shape_bnsd", "seed", "input_sha256"):
        if system[key] != custom[key]:
            raise ValueError("Worker sample metadata/input mismatch: " + key)
    if not torch.equal(system["query_ids"], custom["query_ids"]):
        raise ValueError("Worker sample query indices differ")
    result = check_error(custom["output"], system["output"], .001)
    result["exact_match"] = torch.equal(custom["output"], system["output"])
    result["scope"] = "32 selected query rows, all heads/features; not an all-output equivalence check"
    return result


def run_parent(args):
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=path.stem + "_", dir=path.parent))
    report = {"status": "running", "config": vars(args), "shape_bnsd": list(SHAPE),
              "dtype": "bfloat16", "seed": 0, "input_layout": "contiguous BNSD; no transpose/padding",
              "api": "torch_npu.npu_fusion_attention(...)[0].contiguous()",
              "api_parameters": {"head_num": 28, "input_layout": "BNSD", "scale": 1 / math.sqrt(128),
                                 "keep_prob": 1., "pre_tockens": 2147483647, "next_tockens": 2147483647,
                                 "sparse_mode": 0, "inner_precise": "default 0", "masks": None},
              "process_grouped": True, "process_order": ["system", "custom"],
              "timing_scope": "Full synchronized API wall ms/call; internal allocation/dispatch included. Process-grouped trials, not randomized interleaving or device kernel time.",
              "artifact_dir": str(run_dir), "proof_scope": PROOF_SCOPE, "workers": {}}
    save = lambda: write_json(path, report)
    save()
    try:
        report["source"] = source_evidence(args.source_manifest)
        vendor = Path(args.vendor_root)
        if not (vendor / "op_api" / "lib" / "libcust_opapi.so").is_file():
            raise ValueError("--vendor-root must contain op_api/lib/libcust_opapi.so")
        if not list(vendor.rglob("op_tiling/**/*.so")):
            raise ValueError("--vendor-root must contain an op_tiling shared library")
        save()
        for mode in report["process_order"]:
            directory = run_dir / mode
            directory.mkdir(parents=True, exist_ok=True)
            command = worker_command(args, mode, run_dir)
            env = worker_environment(os.environ, mode, args.vendor_root, run_dir)
            print(f"Starting independent {mode} worker; log: {directory / 'worker.log'}", flush=True)
            with (directory / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace")
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                returncode = process.wait()
            worker_report = directory / "report.json"
            data = json.loads(worker_report.read_text(encoding="utf-8")) if worker_report.exists() else {}
            report["workers"][mode] = {"returncode": returncode, "command": command, **data}
            save()
            if returncode != 0 or data.get("status") != "passed":
                raise RuntimeError(f"{mode} worker failed; see {directory / 'worker.log'}")
        system, custom = (report["workers"][mode] for mode in ("system", "custom"))
        if system["pid"] == custom["pid"]:
            raise AssertionError("Workers did not have distinct process identities")
        report["custom_vs_system"] = compare_samples(system["output_sample"], custom["output_sample"])
        save()
        if not report["custom_vs_system"]["passed"]:
            raise AssertionError("Custom/system sampled relative RMSE exceeds 0.1%")
        report["speedup_system_over_custom"] = system["timing_ms"]["median"] / custom["timing_ms"]["median"]
        report["status"] = "passed"
        save()
        print(json.dumps({"status": report["status"], "report": str(path),
                          "system_ms": system["timing_ms"]["median"], "custom_ms": custom["timing_ms"]["median"],
                          "speedup": report["speedup_system_over_custom"]}), flush=True)
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        save()
        traceback.print_exc()
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--cann-env-script", help="Optional CANN set_env.sh sourced independently before each worker")
    parser.add_argument("--output", default="results/cann_fas_local.json")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--_worker", choices=("system", "custom"), help=argparse.SUPPRESS)
    parser.add_argument("--_run-dir", help=argparse.SUPPRESS)
    parser.add_argument("--_env-ready", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if min(args.repeats, args.trials, args.cpu_threads) <= 0 or args.warmup < 0 or args.device < 0:
        parser.error("Measurement counts/CPU threads must be positive; warmup/device nonnegative")
    args.vendor_root, args.source_manifest = str(Path(args.vendor_root).resolve()), str(Path(args.source_manifest).resolve())
    if args.cann_env_script:
        args.cann_env_script = str(Path(args.cann_env_script).resolve())
        if not Path(args.cann_env_script).is_file():
            parser.error("--cann-env-script does not exist")
    if args._worker:
        if not args._run_dir:
            parser.error("Internal worker requires --_run-dir")
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    sys.exit(main())
