"""Compare installed MindIE Laser and the isolated local Laser source build."""
import argparse
import ast
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

import benchmark_cann_fas_local as common
from mindie_laser_provider_checks import inspect_batch_invariant
from prepare_mindie_laser_source import MARKER_TOKEN, NATIVE_API, OP_TYPE, SOURCE_LOCK


SHAPE = (1, 28, 37741, 128)
MARKER_ENV = "VC_MINDIE_LASER_MARKER_FILE"
UPSTREAM_COMMIT = SOURCE_LOCK.commit
PROOF_SCOPE = (
    "Unique native API/operator names, fresh successful host tiling marker, loaded "
    "API/tiling libraries, bridge and packaged device binary hashes establish source "
    "build and host dispatch evidence. These do not independently prove which device "
    "binary executed, nor byte identity with the installed MindIE release."
)


def laser_transforms(torch):
    """Tensor transforms copied from MindIE-SD at UPSTREAM_COMMIT.

    Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
    Licensed under Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2
    Provided AS IS, without warranties, including non-infringement,
    merchantability or fitness for a particular purpose. See that license.
    """
    class AscendLaserAttention:
        @staticmethod
        def pad(input_tensor, base=256, dim=-2):
            shape_value = input_tensor.size(dim)
            if shape_value % base != 0:
                pad_size = ((shape_value // base) + 1) * base - shape_value
                padding_shape = list(input_tensor.shape)
                padding_shape[dim] = pad_size
                padding = torch.zeros(padding_shape, dtype=input_tensor.dtype, device=input_tensor.device)
                return torch.cat([input_tensor, padding], dim=dim)
            return input_tensor

        @staticmethod
        def la_preprocess_input(query, key, value):
            if query.dtype != torch.float16:
                query = query.to(torch.float16)
                key = key.to(torch.float16)
                value = value.to(torch.float16)
            query = AscendLaserAttention.pad(query, 256, -2)
            query = AscendLaserAttention.pad(query, 128, -1)
            key = AscendLaserAttention.pad(key, 256, -2)
            key = AscendLaserAttention.pad(key, 128, -1)
            value = AscendLaserAttention.pad(value, 256, -2)
            value = AscendLaserAttention.pad(value, 128, -1)
            return query.contiguous(), key.contiguous(), value.contiguous()

        @staticmethod
        def la_postprocess_output(attention_out, dtype, qseqlen, head_dim):
            if dtype != attention_out.dtype:
                attention_out = attention_out.to(torch.float16).to(dtype)
            attention_out = attention_out[:, :, :qseqlen, :head_dim]
            return attention_out
    return AscendLaserAttention


def source_transforms(path, torch):
    """Load only the three pinned pure tensor helpers, never import mindiesd."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendLaserAttention")
    names = {"pad", "la_preprocess_input", "la_postprocess_output"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in methods} != names:
        raise ValueError("Pinned source is missing Laser preprocessing/postprocessing helpers")
    selected = ast.ClassDef(name=cls.name, bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    namespace = {"torch": torch, "SEQLEN_BASE": 256, "DIM_BASE": 128, "SEQLEN_INDEX": -2, "DIM_INDEX": -1}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[cls.name]


def verify_transforms(actual, reference, torch):
    generator = torch.Generator().manual_seed(91025)
    count = 0
    for dtype in (torch.float16, torch.bfloat16):
        for nq, nk, dim in ((256, 256, 128), (33, 129, 64)):
            inputs = [torch.randn((1, 2, n, dim), generator=generator).to(dtype) for n in (nq, nk, nk)]
            prepared = actual.la_preprocess_input(*inputs)
            expected = reference.la_preprocess_input(*inputs)
            for got, want in zip(prepared, expected):
                if got.dtype != want.dtype or got.shape != want.shape or not got.is_contiguous() or not torch.equal(got, want):
                    raise AssertionError("Local preprocess differs from pinned upstream CPU transforms")
            for native_dtype in (torch.float16, torch.float32, torch.bfloat16):
                native = torch.randn(prepared[0].shape, generator=generator).to(native_dtype)
                got = actual.la_postprocess_output(native, dtype, nq, dim)
                want = reference.la_postprocess_output(native, dtype, nq, dim)
                if got.dtype != want.dtype or got.shape != want.shape or not torch.equal(got, want):
                    raise AssertionError("Local postprocess differs from pinned upstream CPU transforms")
            count += 1
    return {"passed": True, "cpu_cases": count, "scope": "Exact CPU tensor transform equivalence to pinned source helpers"}


def worker_environment(base, mode, vendor, run_dir):
    # Reuse the existing startup cleanup, but installed MindIE may register its
    # own provider when load_mindie_laser() runs inside its dedicated worker.
    env = common.worker_environment(base, "system" if mode == "installed" else "custom", vendor, run_dir)
    env.pop("VC_CANN_FAS_MARKER_FILE", None)
    env.pop(MARKER_ENV, None)
    if mode == "local":
        env[MARKER_ENV] = str(Path(run_dir) / mode / "markers.jsonl")
    return env


def parse_markers(text, pid):
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not records:
        raise RuntimeError("First long-shape local call produced no fresh successful tiling marker")
    for item in records:
        if (item.get("token") != MARKER_TOKEN or item.get("phase") != "tiling"
                or type(item.get("pid")) is not int or item["pid"] != pid
                or type(item.get("tiling_key")) is not int or item["tiling_key"] < 0):
            raise ValueError("Malformed or foreign-process Laser tiling marker")
    return records


def verify_schema(schema):
    expected = ["query", "key", "value", "atten_mask", "alibi_mask", "drop_mask", "scale_value",
                "head_num", "input_layout", "keep_prob", "pre_tokens", "next_tokens"]
    names = [argument.name for argument in schema.arguments]
    if len(names) != 13 or names[:12] != expected or names[-1] not in ("is_highPrecision", "is_high_precision") or len(schema.returns) != 2:
        raise ValueError(f"Unexpected vc_laser::la schema: {schema}")


def checked_files(items, root, label):
    if not isinstance(items, list) or not items:
        raise ValueError(f"Build manifest needs nonempty {label}")
    checked = {}
    for item in items:
        path = Path(item["path"])
        if not path.is_absolute():
            path = Path(root) / path
        path = path.resolve()
        if not common.within(path, root):
            raise ValueError(f"{label} escapes the vendor root: {path}")
        digest = common.sha256_file(path)
        if digest != item.get("sha256"):
            raise ValueError(f"{label} SHA256 mismatch: {path}")
        checked[str(path)] = digest
    return checked


def provenance(args, *, expected_variant="baseline"):
    source_path, build_path = Path(args.source_manifest), Path(args.build_manifest)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    build = json.loads(build_path.read_text(encoding="utf-8"))
    if source.get("upstream", {}).get("commit") != UPSTREAM_COMMIT or source.get("upstream", {}).get("url") != SOURCE_LOCK.url:
        raise ValueError("Source manifest must pin the inspected MindIE-SD commit")
    if build.get("status") != "passed":
        raise ValueError("Build manifest does not report a successful build")
    if (source.get("marker_token") != MARKER_TOKEN
            or source.get("variant", "baseline") != expected_variant
            or source.get("kernel_inverse_rename_equivalent") is not (expected_variant == "baseline")):
        raise ValueError("Source manifest must attest the expected variant and inverse-rename kernel equivalence")
    for manifest in (source, build):
        if manifest.get("native_api") != NATIVE_API or manifest.get("op_type") != OP_TYPE:
            raise ValueError("Source/build manifest does not use unique VcLaserAttention symbols")
    if Path(build["vendor_root"]).resolve() != Path(args.vendor_root).resolve():
        raise ValueError("Build manifest vendor_root differs from --vendor-root")
    if build.get("source_manifest", {}).get("sha256") != common.sha256_file(source_path):
        raise ValueError("Build/source manifest SHA256 mismatch")
    result = {"source_manifest": source, "build_manifest": build,
              "source_manifest_sha256": common.sha256_file(source_path), "build_manifest_sha256": common.sha256_file(build_path),
              "device_binary_sha256": checked_files(build.get("device_binaries"), args.vendor_root, "device_binaries"),
              "host_library_sha256": checked_files(build.get("host_libraries"), args.vendor_root, "host_libraries"),
              "bridge_path": args.bridge_path, "bridge_sha256": common.sha256_file(args.bridge_path), "proof_scope": PROOF_SCOPE}
    bridge_manifest = Path(args.bridge_manifest) if args.bridge_manifest else build_path.with_name("mindie_laser_bridge.json")
    if args.bridge_manifest and not bridge_manifest.is_file():
        raise ValueError("Requested bridge manifest does not exist")
    if bridge_manifest.is_file():
        bridge = json.loads(bridge_manifest.read_text(encoding="utf-8"))
        if bridge.get("status") != "passed" or bridge.get("native_api") != NATIVE_API:
            raise ValueError("Bridge manifest does not report a successful unique-native-API build")
        if Path(bridge["bridge_path"]).resolve() != Path(args.bridge_path).resolve() or bridge["bridge_sha256"] != result["bridge_sha256"]:
            raise ValueError("Bridge manifest path/hash differs from requested bridge")
        result["bridge_manifest"] = bridge
    return result


def inspect_provider(mode, vendor, bridge, report, directory, *, cann_root=None, soc=None):
    maps_text = Path("/proc/self/maps").read_text(encoding="utf-8")
    (directory / "proc_self_maps.txt").write_text(maps_text, encoding="utf-8")
    paths = common.mapped_libraries(maps_text)
    report["mapped_library_paths"] = paths
    if mode == "local":
        if any(name == "mindiesd" or name.startswith("mindiesd.") for name in sys.modules):
            raise RuntimeError("Local worker imported mindiesd; provider isolation is invalid")
        # Other registered tiling libraries can be loaded during NPU startup.
        # A narrowly checked background provider does not establish a Laser
        # dispatch conflict. Keep FAS's same-name-provider rules unchanged.
        report["provider_check_context"] = {"cann_root": str(cann_root) if cann_root else None,
                                             "soc": soc}
        background = (inspect_batch_invariant(paths, cann_root, soc) if cann_root and soc
                      else {"libraries": [], "metadata": {}})
        report["ambient_registered_provider"] = background["metadata"]
        filtered = [path for path in paths if path not in background["libraries"]]
        relevant = common.verify_libraries(filtered, "custom", vendor)
        relevant.extend(background["libraries"])
        if str(Path(bridge).resolve()) not in paths:
            raise RuntimeError("Local bridge is absent from process library mappings")
        relevant.append(str(Path(bridge).resolve()))
    else:
        if any(common.within(path, vendor) or Path(path).resolve() == Path(bridge).resolve() for path in paths):
            raise RuntimeError("Installed worker loaded the local Laser provider")
        relevant = [path for path in paths if any(token in path.lower() for token in ("mindie", "opapi", "optiling", "op_tiling", "ptaextension"))]
    report["loaded_library_sha256"] = {path: common.sha256_file(path) for path in relevant}


def load_adapter(args, source):
    import torch
    from mindie_laser_backend import MindIELaser, load_mindie_laser
    if args._worker == "installed":
        return load_mindie_laser(), None
    if "mindiesd" in sys.modules:
        raise RuntimeError("Local worker must not import mindiesd")
    wrapper_name = "mindiesd/layers/flash_attn/ascend_laser_attention.py"
    wrapper = Path(source["source_dir"]) / wrapper_name
    wrapper_digest = common.sha256_file(wrapper)
    if source.get("source_files", {}).get(wrapper_name, {}).get("prepared_sha256") != wrapper_digest:
        raise ValueError("Laser preprocessing wrapper hash differs from the pinned source manifest")
    transforms = laser_transforms(torch)
    validation = verify_transforms(transforms, source_transforms(wrapper, torch), torch)
    validation["upstream_wrapper_sha256"] = wrapper_digest
    torch.ops.load_library(args.bridge_path)
    native = torch.ops.vc_laser.la.default
    verify_schema(native._schema)
    metadata = {"dispatch": "torch.ops.vc_laser.la.default", "native_schema": str(native._schema),
                "native_api": NATIVE_API, "op_type": OP_TYPE, "bridge": args.bridge_path,
                "preprocessing": "Pinned upstream FP16 conversion and S256/D128 zero padding"}
    return MindIELaser(transforms.la_preprocess_input, transforms.la_postprocess_output, native, metadata), validation


def record_gate(report, name, actual, expected, limit, save):
    report[name] = common.check_error(actual, expected, limit)
    save()
    if not report[name]["passed"]:
        raise AssertionError(f"{name}: relative RMSE exceeds {100 * limit:g}%")


def validate_small(adapter, device, report, save, torch):
    report["small_accuracy"] = []
    generator = torch.Generator().manual_seed(91024)
    for dtype in (torch.float16, torch.bfloat16):
        for nq, nk, dim in ((256, 256, 128), (257, 385, 64)):
            original = [torch.randn((1, 2, n, dim), generator=generator).to(dtype) for n in (nq, nk, nk)]
            original[2][..., 0], original[2][..., 1] = 0, 2
            inputs = [tensor.to(device) for tensor in original]
            payload = adapter.prepare(*inputs)
            prepared, full = adapter.run_prepared(payload).cpu(), adapter(*inputs).cpu()
            row = {"dtype": str(dtype), "queries": nq, "keys": nk, "dim": dim,
                   "payload": adapter.describe_payload(payload)}
            report["small_accuracy"].append(row)
            record_gate(row, "full_vs_prepared", full, prepared, .001, save)
            reference = common.dense_reference(*original, torch.arange(nq))
            record_gate(row, "dense_fp32_all_rows_all_keys", full, reference, .01, save)
            torch.testing.assert_close(full[..., 0], torch.zeros_like(full[..., 0]), rtol=0, atol=0)
            torch.testing.assert_close(full[..., 1], torch.full_like(full[..., 1], 2), rtol=.004, atol=.003)
            row["constant_value_padding_check"] = "passed"
            save()


def run_worker(args, *, small_validator=None, dense_limit=.01, after_long=None):
    mode, directory = args._worker, Path(args._run_dir) / args._worker
    directory.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "mode": mode, "pid": os.getpid(), "proof_scope": PROOF_SCOPE}
    save = lambda: common.write_json(directory / "report.json", report)
    save()
    try:
        # Parent sets the final loader environment before launching this process.
        import torch
        import torch_npu
        torch.set_num_threads(min(torch.get_num_threads(), args.cpu_threads))
        source = json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
        build = json.loads(Path(args.build_manifest).read_text(encoding="utf-8"))
        provider_context = {"cann_root": build["definitions"]["CUSTOM_ASCEND_CANN_PACKAGE_PATH"],
                            "soc": build["soc"]}
        adapter, transforms = load_adapter(args, source)  # Installed registration before first NPU workload.
        report["adapter"], report["source_transform_validation"] = adapter.metadata, transforms
        torch.npu.set_option({"ACL_OP_COMPILER_CACHE_MODE": "disable",
                              "ACL_OP_COMPILER_CACHE_DIR": os.environ["ACL_OP_COMPILER_CACHE_DIR"]})
        torch.npu.set_device(args.device)
        device = f"npu:{args.device}"
        report["environment"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                                 "device": torch.npu.get_device_name(args.device),
                                 "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
                                 "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
                                 "compiler_cache": "disable"}
        save()
        with torch.no_grad():
            print(f"{mode}: FP16/BF16 small aligned/tail correctness gates", flush=True)
            (small_validator or validate_small)(adapter, device, report, save, torch)
            generator = torch.Generator().manual_seed(0)
            original = [torch.randn(SHAPE, generator=generator).to(torch.bfloat16) for _ in range(3)]
            fingerprints = [common.tensor_sha256(tensor) for tensor in original]
            inputs = [tensor.to(device) for tensor in original]
            ids = torch.linspace(0, SHAPE[2] - 1, 32).long().unique()
            marker = directory / "markers.jsonl"
            torch.npu.synchronize()
            offset = marker.stat().st_size if marker.exists() else 0
            print(f"{mode}: first long-shape full API call and provider evidence", flush=True)
            output = adapter(*inputs)
            torch.npu.synchronize()
            if mode == "local":
                with marker.open("rb") as stream:
                    stream.seek(offset)
                    fresh = stream.read().decode("utf-8")
                (directory / "long_shape_markers.jsonl").write_text(fresh, encoding="utf-8")
                report["host_markers"] = parse_markers(fresh, os.getpid())
                report["tiling_keys"] = sorted({record["tiling_key"] for record in report["host_markers"]})
            inspect_provider(mode, args.vendor_root, args.bridge_path, report, directory, **provider_context)
            report["provider_evidence_passed"] = True
            os.environ.pop(MARKER_ENV, None)
            marker_size = marker.stat().st_size if marker.exists() else 0
            report["marker_disabled_before_timing"] = True
            sample = output.index_select(2, ids.to(device)).cpu()
            del output
            payload = adapter.prepare(*inputs)
            core_sample = adapter.run_prepared(payload).index_select(2, ids.to(device)).cpu()
            report["prepared_payload"] = adapter.describe_payload(payload)
            record_gate(report, "long_full_vs_prepared_sample", sample, core_sample, .001, save)
            for name, tensor in (("full", sample), ("prepared_core", core_sample)):
                path = directory / f"{name}_sample.pt"
                torch.save(common.sample_payload(torch, tensor, ids, fingerprints), path)
                report[f"{name}_sample"] = str(path)
            print(f"{mode}: CPU FP32 reference, 32 query rows and all {SHAPE[2]} keys", flush=True)
            reference = common.dense_reference(*original, ids)
            if dense_limit is None:
                report["long_dense_fp32"] = common.error_metrics(sample, reference)
                report["long_dense_fp32"]["scope"] = "Approximate algorithm quality diagnostic; no model-quality pass/fail gate"
                save()
            else:
                record_gate(report, "long_dense_fp32", sample, reference, dense_limit, save)
            report["query_ids"], report["all_keys"] = ids.tolist(), SHAPE[2]
            if after_long is not None:
                after_long(adapter, payload, report, save, torch)
            del original, sample, core_sample, reference
            calls = {"full": lambda: adapter(*inputs), "prepared_core": lambda: adapter.run_prepared(payload)}
            report["timing_ms"] = {name: {"trials": []} for name in calls}
            for call in calls.values():
                for _ in range(args.warmup):
                    call()
            torch.npu.synchronize()
            for trial in range(args.trials):
                for name, call in calls.items():
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    for _ in range(args.repeats):
                        call()
                    torch.npu.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000 / args.repeats
                    row = report["timing_ms"][name]
                    row["trials"].append(elapsed)
                    row.update(median=statistics.median(row["trials"]), min=min(row["trials"]), max=max(row["trials"]))
                    save()
                    print(f"{mode} {name}: trial {trial + 1}/{args.trials}: {elapsed:.4f} ms", flush=True)
            if (marker.stat().st_size if marker.exists() else 0) != marker_size:
                raise AssertionError("Marker writes continued after instrumentation was disabled")
            inspect_provider(mode, args.vendor_root, args.bridge_path, report, directory, **provider_context)
        report["status"] = "passed"
        save()
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save()
        traceback.print_exc()
        return 1


def worker_command(args, mode, run_dir):
    command = [sys.executable, str(Path(__file__).resolve()), "--_worker", mode, "--_run-dir", str(run_dir)]
    for name in ("vendor_root", "bridge_path", "source_manifest", "build_manifest", "device", "warmup", "repeats", "trials", "cpu_threads"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    return command


def comparison_table(report):
    lines = ["", "MindIE Laser comparison (synchronized wall ms/API call)",
             "backend       full API median   prepared core median   dense relative RMSE"]
    for name in ("installed", "local"):
        worker = report["workers"][name]
        lines.append(f"{name:<12} {worker['timing_ms']['full']['median']:>15.4f}"
                     f" {worker['timing_ms']['prepared_core']['median']:>22.4f}"
                     f" {100 * worker['long_dense_fp32']['relative_rmse']:>20.6f}%")
    speedup = report["speedup_installed_over_local"]
    comparison = report["local_vs_installed"]
    lines += [f"installed/local speedup: full API {speedup['full']:.4f}x; prepared core {speedup['prepared_core']:.4f}x",
              f"local vs installed relative RMSE: full {100 * comparison['full']['relative_rmse']:.6f}%; "
              f"prepared core {100 * comparison['prepared_core']['relative_rmse']:.6f}%",
              "Accuracy samples: 32 query rows, all 37741 keys, all heads/features; not an all-output equivalence check.",
              "Timing groups use separate processes; prepared core includes output conversion/trim/contiguous."]
    return "\n".join(lines)


def run_parent(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=output.stem + "_", dir=output.parent)).resolve()
    report = {"status": "running", "shape_bnsd": list(SHAPE), "dtype": "bfloat16", "seed": 0,
              "config": vars(args), "artifact_dir": str(run_dir), "proof_scope": PROOF_SCOPE,
              "process_grouped": True, "process_order": ["installed", "local"], "workers": {},
              "timing_scope": {"full": "Fresh upstream QKV conversion/padding + explicit Laser native API + output conversion/trim/contiguous",
                               "prepared_core": "Fixed converted/padded QKV + native allocation/dispatch + output conversion/trim/contiguous"},
              "timing_method": "Synchronized wall ms/call; independent process groups, not paired backend interleaving or kernel-only latency"}
    save = lambda: common.write_json(output, report)
    save()
    try:
        report["provenance"] = provenance(args)
        save()
        for mode in report["process_order"]:
            directory = run_dir / mode
            directory.mkdir(parents=True, exist_ok=True)
            env = worker_environment(os.environ, mode, args.vendor_root, run_dir)
            for name in ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME"):
                Path(env[name]).mkdir(parents=True, exist_ok=True)
            command = worker_command(args, mode, run_dir)
            print(f"Starting independent {mode} Laser worker: {directory}", flush=True)
            with (directory / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace")
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                returncode = process.wait()
            path = directory / "report.json"
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            report["workers"][mode] = {"command": command, "returncode": returncode, **data}
            save()
            if returncode or data.get("status") != "passed":
                raise RuntimeError(f"{mode} worker failed; see {directory / 'worker.log'}")
        installed, local = (report["workers"][name] for name in ("installed", "local"))
        if installed["pid"] == local["pid"]:
            raise AssertionError("Backend workers did not have distinct process identities")
        report["local_vs_installed"], report["speedup_installed_over_local"] = {}, {}
        for name in ("full", "prepared_core"):
            metric = common.compare_samples(installed[f"{name}_sample"], local[f"{name}_sample"])
            report["local_vs_installed"][name] = metric
            save()
            if not metric["passed"]:
                raise AssertionError(f"Local/installed {name} relative RMSE exceeds 0.1%")
            report["speedup_installed_over_local"][name] = installed["timing_ms"][name]["median"] / local["timing_ms"][name]["median"]
        report["status"] = "passed"
        save()
        print(comparison_table(report), flush=True)
        print(f"Report: {output}", flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save()
        traceback.print_exc()
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("vendor-root", "bridge-path", "source-manifest", "build-manifest"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--bridge-manifest", help="Optional bridge provenance JSON; defaults to build-manifest sibling mindie_laser_bridge.json when present")
    parser.add_argument("--output", default="results/mindie_laser_local_compare.json")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--_worker", choices=("installed", "local"), help=argparse.SUPPRESS)
    parser.add_argument("--_run-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if min(args.repeats, args.cpu_threads) <= 0 or args.trials < 3 or args.warmup < 0 or args.device < 0:
        parser.error("Trials must be at least 3; repeats/threads positive; warmup/device nonnegative")
    for name in ("vendor_root", "bridge_path", "source_manifest", "build_manifest", "output"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    if args._worker and not args._run_dir:
        parser.error("Internal worker requires --_run-dir")
    return run_worker(args) if args._worker else run_parent(args)


if __name__ == "__main__":
    sys.exit(main())
