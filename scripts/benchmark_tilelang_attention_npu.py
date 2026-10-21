"""Validate and time ordinary, noncausal TileLang attention on Ascend A2/A3.

The candidate is always tilelang_attention_npu.AttentionKernel.  The independent
oracle is FP32 CPU matmul/softmax; it never constructs the long S-by-S matrix.
Times are synchronized Python API wall time, including dispatch and allocation
where applicable.  They are not NPU device-kernel or CUPTI measurements.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback


DEFAULT_SHAPE = (1, 28, 37741, 128)
TIMING_SCOPE = "synchronized Python API wall time in ms; not device-kernel or CUPTI time"
LIMITS = {"float16": {"relative_rmse": 0.005, "max_abs": 0.01},
          "bfloat16": {"relative_rmse": 0.01, "max_abs": 0.03}}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=4, type=int, default=DEFAULT_SHAPE, metavar=("B", "H", "S", "D"))
    parser.add_argument("--key-length", type=int, help="Defaults to the query length S")
    parser.add_argument("--dtype", choices=tuple(LIMITS), default="bfloat16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--check-queries", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=91028)
    parser.add_argument("--include-mindie-laser", action="store_true",
                        help="Use the already installed MindIE-SD Laser API; never builds it")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/tilelang_attention_npu.json"))
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)
    args.shape = tuple(args.shape)
    args.key_length = args.key_length if args.key_length is not None else args.shape[2]
    if min(*args.shape, args.key_length) <= 0 or args.shape[-1] not in (64, 128):
        parser.error("Require nonempty BNSD and D=64 or D=128")
    if args.block_m != 64 or args.block_n != 64:
        parser.error("The pinned AscendC attention implementation requires --block-m 64 --block-n 64")
    if args.device < 0 or args.warmup < 0 or min(args.repeats, args.trials, args.check_queries, args.cpu_threads) <= 0:
        parser.error("Require nonnegative device/warmup and positive repeats/trials/check-queries/cpu-threads")
    return args


def sample_query_ids(length, count):
    """Boundary rows first, including both sides of the 64-row tile boundary."""
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    wanted = min(length, count)
    candidates = [0, length - 1, 63, 64, 65, length // 2, 1, length - 2]
    candidates += [round(i * (length - 1) / max(1, wanted - 1)) for i in range(wanted)]
    result = []
    for value in candidates:
        if 0 <= value < length and value not in result:
            result.append(value)
        if len(result) == wanted:
            break
    if len(result) < wanted:
        result.extend(i for i in range(length) if i not in result)
    return sorted(result[:wanted])


def cpu_reference(q, k, v, query_ids=None, query_chunk=16):
    """FP32 CPU oracle over ALL keys, bounded to one head and a few Q rows.

    Inputs are the original FP16/BF16 tensors on CPU. No TileLang, torch_npu,
    PyTorch attention implementation, or quantized approximation is involved.
    """
    import torch
    if any(x.device.type != "cpu" or x.ndim != 4 for x in (q, k, v)):
        raise ValueError("CPU BNSD inputs are required for the independent oracle")
    if k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q/K/V dimensions do not describe dense MHA")
    if query_chunk <= 0:
        raise ValueError("query_chunk must be positive")
    ids = torch.arange(q.shape[2]) if query_ids is None else torch.as_tensor(query_ids, dtype=torch.long)
    if ids.ndim != 1 or ids.numel() == 0 or (ids < 0).any() or (ids >= q.shape[2]).any():
        raise ValueError("query_ids must be a nonempty valid one-dimensional selection")
    output = torch.empty((*q.shape[:2], ids.numel(), v.shape[-1]), dtype=torch.float32)
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            kh = k[batch, head].float().transpose(0, 1)
            vh = v[batch, head].float()
            for offset in range(0, ids.numel(), query_chunk):
                selected = ids[offset:offset + query_chunk]
                logits = q[batch, head].index_select(0, selected).float() @ kh
                logits.mul_(1 / math.sqrt(q.shape[-1]))
                probabilities = torch.softmax(logits, dim=-1)
                output[batch, head, offset:offset + selected.numel()] = probabilities @ vh
    return output


def error_metrics(actual, expected, dtype):
    import torch
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(f"Output shape {tuple(actual.shape)} differs from reference {tuple(expected.shape)}")
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    limits = dict(LIMITS[dtype])
    if not finite:
        return {"finite": False, "passed": False, "relative_rmse": None, "rmse": None,
                "max_abs": None, "limits": limits}
    delta = actual - expected
    rmse = delta.square().mean().sqrt().item()
    reference_rms = expected.square().mean().sqrt().item()
    relative_rmse = rmse / max(reference_rms, 1e-12)
    max_abs = delta.abs().max().item()
    return {"finite": True, "rmse": rmse, "reference_rms": reference_rms,
            "relative_rmse": relative_rmse, "max_abs": max_abs, "limits": limits,
            "passed": relative_rmse <= limits["relative_rmse"] and max_abs <= limits["max_abs"]}


def enforce_metrics(actual, expected, dtype, row, records, save):
    """Publish the failed comparison before raising, so no timing can follow it."""
    row = {**row, "metrics": error_metrics(actual, expected, dtype)}
    row["status"] = "passed" if row["metrics"]["passed"] else "failed"
    records.append(row)
    save()
    if row["status"] != "passed":
        raise AssertionError(f"Accuracy validation failed: {row}")
    return row


def exact_check(actual, expected, label, row, records, save):
    import torch
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    passed = (actual.shape == expected.shape and actual.dtype == expected.dtype
              and bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
              and torch.equal(actual, expected))
    record = {**row, "check": label, "exact_match": passed, "status": "passed" if passed else "failed"}
    records.append(record)
    save()
    if not passed:
        raise AssertionError(f"Repeated/stream output changed: {record}")


def make_cpu_inputs(shape, keys, dtype, seed, pattern="random"):
    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed)
    b, h, queries, d = shape
    # Head-sized random temporaries avoid making a second full FP32 long tensor.
    tensors = []
    for length in (queries, keys, keys):
        tensor = torch.empty((b, h, length, d), dtype=dtype)
        for batch in range(b):
            for head in range(h):
                tensor[batch, head].copy_(torch.randn((length, d), generator=generator))
        tensors.append(tensor)
    q, k, v = tensors
    if pattern == "zero_q":
        q.zero_()
    elif pattern == "constant_v":
        # Different constants per head/channel expose head addressing mistakes.
        channels = (torch.arange(d).float() % 11 - 5) / 4
        heads = torch.arange(h).float().view(1, h, 1, 1) / 8
        v.copy_(channels.view(1, 1, 1, d) + heads)
    elif pattern != "random":
        raise ValueError("Unknown validation pattern: " + pattern)
    return q, k, v


def small_cases():
    return [(pattern, queries, keys) for queries, keys in ((65, 129), (129, 65))
            for pattern in ("random", "zero_q", "constant_v")] + [("random", 1, 1)]


def fusion_call(torch_npu, q, k, v):
    return torch_npu.npu_fusion_attention(
        q, k, v, q.shape[1], "BNSD", scale=1 / math.sqrt(q.shape[-1]), keep_prob=1.0,
        pre_tockens=2147483647, next_tockens=2147483647, sparse_mode=0)[0]


def synchronized_api_time(call, synchronize, warmup, repeats, trials, clock=time.perf_counter):
    if warmup < 0 or repeats <= 0 or trials <= 0:
        raise ValueError("Invalid timing repetition count")
    for _ in range(warmup):
        value = call()
        synchronize()
        del value
    times = []
    for _ in range(trials):
        elapsed = 0.0
        for _ in range(repeats):
            synchronize()
            start = clock()
            value = call()
            synchronize()
            elapsed += clock() - start
            del value
        times.append(elapsed * 1000 / repeats)
    return {"scope": TIMING_SCOPE, "trial_mean_ms": times, "median_ms": statistics.median(times),
            "min_ms": min(times), "max_ms": max(times), "warmup": warmup,
            "repeats_per_trial": repeats, "trials": trials,
            "synchronization": "device synchronize before and after every timed API call"}


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_metadata(torch, torch_npu, backend, device):
    import ctypes
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    raw = acl.aclrtGetSocName()
    if not raw:
        raise RuntimeError("aclrtGetSocName returned null")
    soc = raw.decode()
    properties = torch.npu.get_device_properties(device)
    return {"soc": soc, "device": str(device), "device_name": torch.npu.get_device_name(device),
            "device_properties": str(properties), "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", _version("torch-npu")),
            "tilelang": _version("tilelang"), "python": sys.version, "python_executable": sys.executable,
            "backend_source_commit": getattr(backend, "SOURCE_COMMIT", None),
            "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
            "provider_environment": {name: os.environ.get(name) for name in
                                     ("ASCEND_CUSTOM_OPP_PATH", "ASCEND_OPP_PATH", "LD_PRELOAD",
                                      "ACLNN_EXTENSION_PATH", "ACLNN_EXTENSION_SWITCH", "TL_RUN_MODE")},
            "torch_loaded_libraries_before_validation": sorted(str(path) for path in torch.ops.loaded_libraries),
            "cache_environment": {name: os.environ.get(name) for name in
                                  ("TILELANG_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME",
                                   "ACL_OP_COMPILER_CACHE_DIR")}}


def isolate_runtime_cache(torch, torch_npu):
    """Select the runtime cache before importing TileLang or its kernel module."""
    import ctypes
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    raw = acl.aclrtGetSocName()
    if not raw:
        raise RuntimeError("aclrtGetSocName returned null")
    soc = raw.decode()
    root = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann")).resolve()
    files = {}
    for base in (root, root.parent):
        # CANN releases use arm64-linux, aarch64-linux or x86_64-linux, and
        # recent combined packages have yet another manifest placement.
        for pattern in ("version.info", "*install.info", "*/version.info", "*/*install.info", "*/*/version.info"):
            for candidate in base.glob(pattern):
                if candidate.is_file():
                    files[str(candidate.resolve())] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    compiler = {}
    candidates = []
    for executable in ("bisheng", "ccec", "ccec_cc"):
        found = shutil.which(executable)
        if found:
            candidates.append(Path(found))
    for base in (root, root.parent):
        for pattern in ("bin/bisheng", "compiler/ccec_compiler/bin/bisheng", "*/ccec_compiler/bin/bisheng",
                        "*/bin/bisheng", "*/compiler/ccec_compiler/bin/bisheng"):
            candidates.extend(base.glob(pattern))
    for path in candidates:
        if not path.is_file():
            continue
        path = path.resolve()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        try:
            version = subprocess.run([str(path), "--version"], capture_output=True, text=True,
                                     timeout=15, check=False)
            compiler = {"path": str(path), "sha256": digest.hexdigest(), "version_exit_code": version.returncode,
                        "version_output": (version.stdout + version.stderr).strip()[:8192]}
        except (OSError, subprocess.SubprocessError) as exc:
            compiler = {"path": str(path), "sha256": digest.hexdigest(), "version_error": str(exc)}
        break
    if not files and not compiler:
        raise RuntimeError(f"Cannot fingerprint CANN under {root}: no version manifest or bisheng/ccec compiler found")
    # Read the adjacent source pin without importing a module that could load TL.
    import ast
    source = Path(__file__).with_name("tilelang_attention_npu.py").read_text(encoding="utf-8")
    commit = None
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SOURCE_COMMIT"
                                                    for target in statement.targets):
            commit = ast.literal_eval(statement.value)
            break
    if not commit:
        raise RuntimeError("TileLang kernel SOURCE_COMMIT pin is missing")
    identity = {"source_commit": commit, "soc": soc, "cann_root": str(root), "cann_manifests": files,
                "compiler": compiler,
                "torch": torch.__version__, "torch_npu": getattr(torch_npu, "__version__", _version("torch-npu")),
                "tilelang": _version("tilelang"), "python": sys.version}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    cache_root = Path(os.environ.get("TILELANG_ATTENTION_CACHE_ROOT",
                                   str(Path(__file__).resolve().parents[1] / ".cache/tilelang-attention-npu/runtime")))
    cache_root = (cache_root / f"{soc}_{commit[:12]}_{digest}").resolve()
    for variable, subdirectory in (("TILELANG_CACHE_DIR", "tilelang"), ("ASCEND_CACHE_PATH", "ascend"),
                                   ("XDG_CACHE_HOME", "xdg"), ("ACL_OP_COMPILER_CACHE_DIR", "compiler")):
        path = cache_root / subdirectory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    return {"identity": identity, "key": digest, "directory": str(cache_root)}


def archive_sources(directory):
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    evidence = {}
    for name in (Path(__file__).name, "tilelang_attention_npu.py", "run_tilelang_attention_npu.sh",
                 "setup_tilelang_attention_npu.sh", "setup_tilelang_attention_npu.py"):
        path = root / name
        if path.is_file():
            target = directory / name
            shutil.copy2(path, target)
            evidence[name] = {"path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    return evidence


def validate_contract(output, q):
    if tuple(output.shape) != tuple(q.shape) or output.dtype != q.dtype or output.device != q.device:
        raise AssertionError("Attention output must retain input BNSD, dtype, and NPU device")


def validate_small(torch, backend, providers, args, device, report, save, artifact_dir):
    records = report["validation"]["small"]
    stream_supported = all(hasattr(torch.npu, attr) for attr in ("Stream", "stream", "current_stream"))
    report["validation"]["nondefault_stream"] = "supported" if stream_supported else "API unavailable; skipped"
    index = 0
    for dtype_name in ("float16", "bfloat16"):
        for dim in (64, 128):
            for pattern, queries, keys in small_cases():
                index += 1
                # One-key MHA has an exact V answer and separately exercises
                # batch/head addressing and fewer tasks than physical cores.
                shape = (2, 3, queries, dim) if (queries, keys) == (1, 1) else (1, 2, queries, dim)
                label = {"dtype": dtype_name, "shape_bnsd": list(shape), "keys": keys, "pattern": pattern,
                         "oracle": "independent CPU FP32; all query rows and all keys"}
                print(f"Validate {dtype_name} BNSD={shape} K={keys} {pattern}", flush=True)
                original = make_cpu_inputs(shape, keys, getattr(torch, dtype_name), args.seed + index, pattern)
                expected = cpu_reference(*original)
                q, k, v = [x.to(device) for x in original]
                kernel = backend.build_attention(*shape[:3], keys, dim, dtype=dtype_name, device=device,
                                                 block_m=args.block_m, block_n=args.block_n)
                source_key = f"small_{dtype_name}_q{queries}_k{keys}_d{dim}"
                if source_key not in report["kernel_sources"]:
                    report["kernel_sources"][source_key] = kernel.export_source(artifact_dir / source_key)
                    save()
                prepared = kernel.prepare(q, k, v)
                output = kernel.run_prepared(prepared)
                torch.npu.synchronize()
                validate_contract(output, q)
                first = output.detach().cpu().clone()
                enforce_metrics(first, expected, dtype_name, {**label, "provider": "tilelang", "path": "prepared"}, records, save)
                for repeat in range(2):
                    current = kernel.run_prepared(prepared)
                    torch.npu.synchronize()
                    exact_check(current, first, f"prepared_repeat_{repeat + 1}",
                                {**label, "provider": "tilelang"}, records, save)
                full = kernel(q, k, v)
                torch.npu.synchronize()
                validate_contract(full, q)
                exact_check(full, first, "full_vs_prepared", {**label, "provider": "tilelang"}, records, save)
                if stream_supported:
                    stream = torch.npu.Stream(device=device)
                    stream.wait_stream(torch.npu.current_stream(device))
                    with torch.npu.stream(stream):
                        # PreparedAttention owns stream-local scratch/output. A
                        # different stream gets distinct storage, used serially.
                        alternate_prepared = kernel.prepare(q, k, v)
                        alternate = kernel.run_prepared(alternate_prepared)
                    stream.synchronize()
                    torch.npu.synchronize()
                    exact_check(alternate, first, "nondefault_stream", {**label, "provider": "tilelang"}, records, save)
                for name, call in providers.items():
                    other = call(q, k, v)
                    torch.npu.synchronize()
                    validate_contract(other, q)
                    enforce_metrics(other, expected, dtype_name, {**label, "provider": name, "path": "full"}, records, save)
                    del other
                del q, k, v, output, current, full, first, prepared, kernel, original, expected
                if stream_supported:
                    del alternate, alternate_prepared, stream


def run(args, report, save, artifact_dir):
    import torch
    import torch_npu

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("An available Ascend NPU and torch_npu are required; no CPU attention fallback is used")
    torch.set_num_threads(args.cpu_threads)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    report["runtime_cache"] = isolate_runtime_cache(torch, torch_npu)
    import tilelang_attention_npu as backend
    providers = {"cann_fusion_attention": lambda q, k, v: fusion_call(torch_npu, q, k, v)}
    report["providers"] = {"cann_fusion_attention": {
        "label": "CANN npu_fusion_attention API", "dispatch": "torch_npu.npu_fusion_attention",
        "provenance_scope": "installed Python API dispatch; internal vendor binary identity is not established"}}
    if args.include_mindie_laser:
        from mindie_laser_backend import load_mindie_laser
        laser = load_mindie_laser()
        providers["mindie_laser_installed"] = laser
        report["mindie_laser"] = laser.metadata
        report["mindie_laser"]["dtype_scope"] = (
            "existing nonquantized Laser API; upstream BF16 preprocessing converts to FP16 and output returns to BF16")
        report["providers"]["mindie_laser_installed"] = {"label": "installed MindIE-SD Laser API"}
    report["runtime"] = runtime_metadata(torch, torch_npu, backend, device)
    print(f"Backend=tile-ai/tilelang-ascend target=ascendc SoC={report['runtime']['soc']} "
          f"dtype={args.dtype} BNSD={args.shape} K={args.key_length}", flush=True)
    save()
    with torch.no_grad():
        validate_small(torch, backend, providers, args, device, report, save, artifact_dir)
        shape, keys = args.shape, args.key_length
        print(f"Validate main {args.dtype} BNSD={shape} K={keys}; sampled Q, all K/V", flush=True)
        original = make_cpu_inputs(shape, keys, getattr(torch, args.dtype), args.seed)
        ids = sample_query_ids(shape[2], args.check_queries)
        expected = cpu_reference(*original, query_ids=ids)
        q, k, v = [x.to(device) for x in original]
        del original
        kernel = backend.build_attention(*shape[:3], keys, shape[-1], dtype=args.dtype, device=device,
                                         block_m=args.block_m, block_n=args.block_n)
        prepared = kernel.prepare(q, k, v)
        report["kernel"] = kernel.metadata()
        report["prepared"] = prepared.metadata()
        report["kernel_sources"]["main"] = kernel.export_source(artifact_dir / "generated_main")
        selection = torch.tensor(ids, device=device, dtype=torch.long)
        records = report["validation"]["main"]
        row = {"dtype": args.dtype, "shape_bnsd": list(shape), "keys": keys, "query_ids": ids,
               "oracle": "independent CPU FP32; sampled Q, all K/V; no full S-by-S allocation"}

        def check_main(name, output):
            torch.npu.synchronize()
            validate_contract(output, q)
            if not bool(torch.isfinite(output).all()):
                records.append({**row, "provider": name, "status": "failed", "all_output_finite": False})
                save()
                raise AssertionError(name + " produced nonfinite output")
            sample = output.index_select(2, selection).detach().cpu().clone()
            artifact_dir.mkdir(parents=True, exist_ok=True)
            sample_path = artifact_dir / f"{name}_main_sample.pt"
            torch.save({"query_ids": ids, "actual": sample, "reference_fp32": expected,
                        "shape_bnsd": list(shape), "keys": keys, "seed": args.seed}, sample_path)
            enforce_metrics(sample, expected, args.dtype, {**row, "provider": name,
                            "all_output_finite": True, "sample_artifact": str(sample_path)}, records, save)
            return sample

        first_output = kernel.run_prepared(prepared)
        check_main("tilelang_prepared", first_output)
        # Oracle scores stay sampled, while complete output copies check for
        # workspace reuse, task scheduling, or uninitialized-tail regressions.
        torch.npu.synchronize()
        first_full_cpu = first_output.detach().cpu().clone()
        full_output = kernel(q, k, v)
        check_main("tilelang_full", full_output)
        torch.npu.synchronize()
        exact_check(full_output, first_full_cpu, "main_full_vs_prepared_all_output", row, records, save)
        repeated_output = kernel.run_prepared(prepared)
        torch.npu.synchronize()
        exact_check(repeated_output, first_full_cpu, "main_prepared_repeat_all_output", row, records, save)
        del first_full_cpu, full_output, repeated_output, first_output
        for name, call in providers.items():
            check_main(name, call(q, k, v))
        report["runtime"]["torch_loaded_libraries_after_validation"] = sorted(str(path) for path in torch.ops.loaded_libraries)
        report["validation"]["status"] = "passed"
        save()
        if args.validate_only:
            report["status"] = "validated"
            return
        # No provider enters timing until EVERY validation above has passed.
        report["status"] = "timing"
        save()
        calls = {"tilelang_full": (lambda: kernel(q, k, v),
                                  "Q/K/V padding/copies + workspace/output allocation + TileLang dispatch + output view"),
                 "tilelang_prepared": (lambda: kernel.run_prepared(prepared),
                                      "reused padded Q/K/V, workspace/output; TileLang dispatch + output view")}
        calls.update({name: (lambda call=call: call(q, k, v), "full installed provider Python API")
                      for name, call in providers.items()})
        names = list(calls)
        report["timing_trial_order"] = []
        samples = {name: [] for name in names}
        for name in names:
            for _ in range(args.warmup):
                output = calls[name][0]()
                torch.npu.synchronize()
                del output
        # Rotate the complete provider order for each trial to reduce drift
        # from running all trials of a provider at one temperature/time.
        for trial in range(args.trials):
            shift = trial % len(names)
            order = names[shift:] + names[:shift]
            report["timing_trial_order"].append(order)
            for name in order:
                label = report["providers"].get(name, {}).get("label", name)
                print(f"Time trial {trial + 1}/{args.trials} {label}: {TIMING_SCOPE}", flush=True)
                result = synchronized_api_time(calls[name][0], torch.npu.synchronize, 0, args.repeats, 1)
                samples[name].extend(result["trial_mean_ms"])
                report["timings"][name] = {
                    **result, "trial_mean_ms": list(samples[name]), "median_ms": statistics.median(samples[name]),
                    "min_ms": min(samples[name]), "max_ms": max(samples[name]), "warmup": args.warmup,
                    "trials": len(samples[name]), "included_work": calls[name][1],
                    "provider_order": "rotated once per trial; no interleaving inside an API call"}
                save()
        for name in names:
            label = report["providers"].get(name, {}).get("label", name)
            print(f"  {label}: median {report['timings'][name]['median_ms']:.3f} ms", flush=True)
        report["status"] = "passed"


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    artifact_dir = (args.artifact_dir or args.output.parent / "tilelang_attention_npu_artifacts" / stamp).resolve()
    report = {"status": "running", "created_utc": stamp, "shape_bnsd": list(args.shape), "keys": args.key_length,
              "dtype": args.dtype, "layout": "BNSD", "causal": False, "quantized": False,
              "attention": "softmax(QK^T / sqrt(D)) V", "timing_scope": TIMING_SCOPE,
              "artifact_dir": str(artifact_dir), "kernel_sources": {},
              "validation": {"status": "running", "small": [], "main": []}, "timings": {},
              "settings": {name: value for name, value in vars(args).items() if name not in ("output", "artifact_dir")}}
    save = lambda: write_json(args.output, report)
    save()
    try:
        report["source_archive"] = archive_sources(artifact_dir / "source")
        run(args, report, save, artifact_dir)
        save()
        print(f"Result: {args.output.resolve()} ({report['status']})", flush=True)
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        if report["validation"]["status"] != "passed":
            report["validation"]["status"] = "failed"
        save()
        print(f"FAILED: {type(exc).__name__}: {exc}\nDiagnostic JSON: {args.output.resolve()}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
