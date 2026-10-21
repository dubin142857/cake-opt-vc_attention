#!/usr/bin/env python3
"""Interleaved original FP16 Laser / INT8 Vector-scale / selected candidate APIs."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from types import SimpleNamespace

import benchmark_mindie_laser_quant as quant
from mindie_laser_quant_pairing import run_paired, paired_comparison
from mindie_laser_scale_reuse_checks import verify_provider_paths

common = quant.common
ORIGINAL = "original_laser"
OLD = "vector_scale"
NEW = "vector_scale_bulk"
VARIANTS = (ORIGINAL, OLD, NEW)
CANDIDATES = {
    "bulk": dict(name=NEW, build_arg="bulk_build", suffix="vscale_bulk",
                 comparison="bulk_vs_previous", report_name="vector_scale_bulk",
                 contract_key="bulk_schedule_contract"),
    "rowmax": dict(name="vector_rowmax", build_arg="rowmax_build", suffix="vscale_rowmax",
                   comparison="rowmax_vs_previous", report_name="vector_rowmax",
                   contract_key="rowmax_schedule_contract"),
}
MANIFESTS = {
    "vscale_build": "results/mindie_laser_quant_vscale_build.json",
    "bulk_build": "results/mindie_laser_quant_vscale_bulk_build.json",
    "rowmax_build": "results/mindie_laser_quant_vscale_rowmax_build.json",
    "preprocess_build": "results/mindie_laser_quant_preprocess_build.json",
    "baseline_build": "results/mindie_laser_build.json",
    "bridge_manifest": "results/mindie_laser_bridge.json",
}
SCOPES = {
    "full": "Fresh per-algorithm preprocessing plus native allocation/dispatch and output conversion/trim/contiguous",
    "prepared": "Each algorithm's prepared payload plus native allocation/dispatch and output postprocess",
    "native": "Each algorithm's prepared payload; raw native API allocation/dispatch, excluding output postprocess",
}
BASELINE_DESCRIPTION = (
    "Pinned original MindIE-SD FP16 Laser source, inverse-rename kernel equivalent, "
    "with upstream tensor transforms and explicit vc_laser::la. This is the local "
    "source reproduction, not installed mindiesd auto dispatch or binary identity."
)


def candidate_config(args):
    key = getattr(args, "candidate", "bulk")
    if key not in CANDIDATES:
        raise ValueError("Unknown Vector candidate: " + str(key))
    return CANDIDATES[key]


def variant_names(candidate):
    if candidate not in {config["name"] for config in CANDIDATES.values()}:
        raise ValueError("Unknown candidate variant: " + str(candidate))
    return ORIGINAL, OLD, candidate


def group_candidate(group):
    candidates = set(group) - {ORIGINAL, OLD}
    if len(group) != 3 or len(candidates) != 1:
        raise ValueError("Three-provider benchmark requires original and both INT8 variants")
    candidate = next(iter(candidates))
    variant_names(candidate)
    return candidate


def specifications(args):
    config = candidate_config(args)
    return {
        OLD: dict(path=args.vscale_build, namespace="vc_laser_quant_vscale",
                  source=importlib.import_module("prepare_mindie_laser_quant_vscale_source")),
        config["name"]: dict(path=getattr(args, config["build_arg"]),
                  namespace="vc_laser_quant_"+config["suffix"],
                  source=importlib.import_module("prepare_mindie_laser_quant_"+config["suffix"]+"_source")),
    }


def resolve_providers(args):
    original = quant.build_arguments(args, "baseline")
    original.marker_env = quant.baseline.MARKER_ENV
    original.marker_token = quant.baseline.MARKER_TOKEN
    providers = {ORIGINAL: original}
    for name, spec in specifications(args).items():
        options = SimpleNamespace(**vars(args))
        options.quant_build = spec["path"]
        provider = quant.build_arguments(options, "qk_int8_fused")
        provider.marker_env = spec["source"].MARKER_ENV
        provider.marker_token = spec["source"].MARKER_TOKEN
        providers[name] = provider
    roots = [str(Path(p.vendor_root).resolve()) for p in providers.values()]
    if len(set(roots)) != len(VARIANTS):
        raise ValueError("Original Laser and both INT8 variants require distinct vendor roots")
    return providers


def verify_builds(args, providers=None):
    from mindie_laser_quant_fused import validate_preprocess_manifest
    providers = resolve_providers(args) if providers is None else providers
    report = {ORIGINAL: quant.baseline.provenance(providers[ORIGINAL]),
              "preprocessor": validate_preprocess_manifest(args.preprocess_build)}
    for name, spec in specifications(args).items():
        report[name] = quant.quant_provenance(providers[name],
            identity=spec["source"].IDENTITY,
            source_validator=spec["source"].validate_prepared_source)
    return report


def worker_environment(base, providers, directory):
    names = variant_names(group_candidate(providers))
    roots = [str(Path(providers[name].vendor_root).resolve()) for name in names]
    if len(set(roots)) != 3:
        raise ValueError("Three distinct vendor roots are required")
    env = quant.baseline.worker_environment(base, "local", roots[0], directory)
    paths = [p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
             if p and not any(common.within(p, root) for root in roots)]
    env["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(roots)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(Path(root) / "op_api/lib") for root in roots] + paths)
    for key in list(env):
        if key.startswith("VC_") and key.endswith("MARKER_FILE"):
            env.pop(key)
    for name, provider in providers.items():
        env[provider.marker_env] = str(Path(directory) / "local" / (name + "_markers.jsonl"))
    return env


def inspect_providers(providers, evidence, report, directory, *, cann_root, soc):
    """Both original manifests are verified before using their combined hashes."""
    if any(name == "mindiesd" or name.startswith("mindiesd.") for name in sys.modules):
        raise RuntimeError("Three-provider worker must not import mindiesd")
    text = Path("/proc/self/maps").read_text(encoding="utf-8")
    (directory / "proc_self_maps.txt").write_text(text, encoding="utf-8")
    paths = common.mapped_libraries(text)
    background = quant.baseline.inspect_batch_invariant(paths, cann_root, soc)
    relevant = verify_provider_paths(paths, providers, background=background["libraries"])
    verify_mapped_hashes(relevant, providers, evidence)
    report["mapped_library_paths"] = paths
    report["ambient_registered_provider"] = background["metadata"]
    report["loaded_library_sha256"] = {path: common.sha256_file(path) for path in relevant}


def verify_mapped_hashes(relevant, providers, evidence):
    for name, provider in providers.items():
        # provenance() has verified the original build AND separate bridge
        # manifest; quant_provenance() verifies the combined INT8 manifest.
        checked = evidence[name]
        expected = dict(checked["host_library_sha256"])
        bridge = str(Path(provider.bridge_path).resolve())
        expected[bridge] = checked["bridge_sha256"]
        for path in relevant:
            if common.within(path, provider.vendor_root) or path == bridge:
                if path not in expected or common.sha256_file(path) != expected[path]:
                    raise RuntimeError("Mapped library differs from audited manifest: " + path)


def make_calls(adapters, payloads, inputs):
    from mindie_laser_fp16_trace_worker import native_call
    candidate = group_candidate(adapters)
    if set(payloads) != set(adapters):
        raise ValueError("Payloads must match the three selected adapters")
    if payloads[OLD] is not payloads[candidate]:
        raise ValueError("INT8 variants must share identical payload objects")
    if payloads[ORIGINAL] is payloads[OLD]:
        raise ValueError("Original FP16 Laser requires its own prepared payload")
    return {
        "full": {name: lambda a=adapter: a(*inputs) for name, adapter in adapters.items()},
        "prepared": {name: lambda a=adapter, p=payloads[name]: a.run_prepared(p)
                     for name, adapter in adapters.items()},
        "native": {name: (lambda a=adapter, p=payloads[name]: native_call(a, p))
                   if name == ORIGINAL else (lambda a=adapter, p=payloads[name]: a.run_native(p))
                   for name, adapter in adapters.items()},
    }


def require_exact(error):
    if (not all(math.isfinite(error[key]) for key in ("relative_rmse", "max_abs"))
            or not error["exact_match"]):
        raise AssertionError("Vector candidate must preserve ALL padded native output elements exactly")


class ExactValidationAdapter:
    """Queue exact old/new checks without interrupting same-stream scale updates."""
    def __init__(self, adapter, reference, rows, save):
        self.adapter, self.reference, self.rows, self.save = adapter, reference, rows, save
        self.pending = []

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def __call__(self, *inputs):
        return self.adapter(*inputs)

    def run_native(self, payload):
        actual = self.adapter.run_native(payload)
        expected = self.reference.run_native(payload)
        self.pending.append((actual, expected))
        return actual

    def flush(self):
        # Only after validate_vector_scale has queued and checked every
        # in-place update scenario. Never introduce CPU sync between updates.
        from profile_mindie_laser_quant import compare_all_rows
        for actual, expected in self.pending:
            error = compare_all_rows(actual, expected)
            self.rows.append(error)
            require_exact(error)
        self.pending.clear()
        self.save()


def compare_scopes(timing):
    candidate = group_candidate(timing["rows"])
    key = next(c["comparison"] for c in CANDIDATES.values() if c["name"] == candidate)
    return {OLD: paired_comparison(timing, ORIGINAL, OLD),
            candidate: paired_comparison(timing, ORIGINAL, candidate),
            key: paired_comparison(timing, OLD, candidate)}


def table(report):
    candidate = group_candidate(report["timings"]["full"]["rows"])
    names = variant_names(candidate)
    comparison_key = next(c["comparison"] for c in CANDIDATES.values() if c["name"] == candidate)
    lines = ["Same-process randomized interleaved synchronized API ms; NO profiler:",
             "Primary baseline: pinned ORIGINAL FP16 MindIE Laser (local source reproduction).",
             f"{'variant':<21} {'full API':>11} {'Laser/full':>12} {'saved ms':>11} {'faster':>8} {'dense RMSE %':>14}"]
    timing = report["timings"]["full"]
    original = timing["rows"][ORIGINAL]["median_ms"]
    for name in names:
        value = timing["rows"][name]["median_ms"]
        dense = 100 * report["dense_fp32_samples"][name]["relative_rmse"]
        pair = report["comparisons"]["full"].get(name)
        saved = f"{pair['median_saved_ms']:+.4f}" if pair else "--"
        faster = f"{pair['faster_trials']}/{pair['trials']}" if pair else "--"
        lines.append(f"{name:<21} {value:11.4f} {original/value:11.4f}x {saved:>11} {faster:>8} {dense:14.6f}")
    lines += ["Laser/full uses FULL API only, including fresh preprocessing for every call.",
              "saved ms is the median of paired differences, not the difference of table medians.",
              "", f"{'scope':<12} {'variant':<21} {'API ms':>11} {'Laser/row':>12} {'saved ms':>11} {'faster':>8}"]
    for scope in ("prepared", "native"):
        rows = report["timings"][scope]["rows"]
        for name in names:
            pair = report["comparisons"][scope].get(name)
            saved = f"{pair['median_saved_ms']:+.4f}" if pair else "--"
            faster = f"{pair['faster_trials']}/{pair['trials']}" if pair else "--"
            value = rows[name]["median_ms"]
            lines.append(f"{scope:<12} {name:<21} {value:11.4f} "
                         f"{rows[ORIGINAL]['median_ms']/value:11.4f}x {saved:>11} {faster:>8}")
    lines.append(candidate + " / previous INT8: fixed scopes share payload; full freshly preprocesses; ALL native outputs checked:")
    for scope in SCOPES:
        pair = report["comparisons"][scope][comparison_key]
        lines.append(f"  {scope}: old/new {pair['baseline_median_ms']/pair['candidate_median_ms']:.4f}x; "
                     f"paired saved {pair['median_saved_ms']:+.4f} ms; "
                     f"faster {pair['faster_trials']}/{pair['trials']}")
    lines += [f"Exact old/new output: {report['same_payload_all_output']['exact_match']}.",
              "FP16 and INT8 use their own payloads; native output types differ. Native is not kernel-only time.",
              "Scopes are independent and not additive. Dense RMSE samples queries/all keys; no model-quality claim."]
    return "\n".join(lines)


ROWMAX_KEYS = (127, 128, 129, 1023, 1024, 1025, 2047, 2048, 2049, 3968, 4095, 4096, 4097)
ROWMAX_MODES = ("zero", "negative", "moving_winner", "finite_extreme")


def rowmax_payload(keys, dtype, dim, mode):
    """Direct INT8 payload: distinct query/channel winners and masked K tails.

    A query selects one feature, so no dot accumulation ambiguity is needed.
    The extreme case remains below 16384 absolute score, keeping both the
    scores and pairwise score differences inside FP16's finite range.
    """
    import torch
    from mindie_laser_quant_backend import prepare_quant_tensors
    if keys <= 0 or dim not in (64, 128) or mode not in ROWMAX_MODES:
        raise ValueError("Invalid rowmax boundary fixture")
    queries = 17
    q = torch.zeros(1, 2, queries, dim, dtype=dtype)
    k = torch.zeros(1, 2, keys, dim, dtype=dtype)
    tokens = torch.arange(keys)
    values = ((tokens[:, None] * 13 + torch.arange(dim)[None, :] * 7) % 29 - 14).float()/14
    v = values[None, None].expand(1, 2, -1, -1).to(dtype).clone()
    v[..., 0], v[..., 1] = 0, 2
    payload = prepare_quant_tensors(q, k, v, v_quantize=False)
    winners = []
    for head in range(2):
        head_winners = []
        for row in range(queries):
            sign = 1 if mode in ("zero", "negative") or row % 2 == 0 else -1
            payload.q[0, head, row, row] = sign
            if mode == "negative":
                codes = -((tokens * 7 + row * 11 + head) % 127 + 1)
            elif mode == "zero":
                codes = torch.zeros_like(tokens)
            else:
                codes = ((tokens * 13 + row * 19 + head * 3) % 97 - 48) * sign
            block = (row * 7 + head * 3) % ((keys + 127)//128)
            begin = block * 128
            winner = begin + min((row * 17 + head * 31) % 128, keys - begin - 1)
            if mode in ("moving_winner", "finite_extreme"):
                codes[winner] = 127 * sign
            payload.k[0, head, :keys, row] = codes.to(torch.int8)
            head_winners.append(winner)
        winners.append(head_winners)
    payload.q_scale.fill_(32 if mode == "finite_extreme" else 1)
    payload.k_scale.fill_(32 if mode == "finite_extreme" else 1)
    return payload, dict(keys=keys, queries=queries, dim=dim, dtype=str(dtype), mode=mode,
        assigned_winner_keys=winners if mode in ("moving_winner", "finite_extreme") else None,
        max_absolute_score_bound=16384 if mode == "finite_extreme" else 16,
        direct_payload=True)


def validate_rowmax_boundaries(old, new, device, report, save):
    import torch
    from dataclasses import replace
    from profile_mindie_laser_quant import FIELDS, compare_all_rows
    rows = report["rowmax_boundary_validation"] = []
    report["rowmax_boundary_scope"] = {
        "keys": list(ROWMAX_KEYS), "modes": list(ROWMAX_MODES),
        "output_gate": "Exact ALL padded native output rows/heads/features; finite outputs required",
        "purpose": "K128 tails, 8/16/32-block reduction boundaries and K4096 section transition",
        "timed": False, "dense_reference": False,
    }
    save()
    for dtype in (torch.float16, torch.bfloat16):
        for i, keys in enumerate(ROWMAX_KEYS):
            dim = (64, 128)[i % 2]
            for mode in ROWMAX_MODES:
                host, row = rowmax_payload(keys, dtype, dim, mode)
                fixed = replace(host, **{name: getattr(host, name).to(device)
                                        for name in (*FIELDS, "v_scale")})
                rows.append(row)
                row["status"] = "running"
                save()
                print(f"Rowmax boundary: {dtype} Q17 K{keys} D{dim} {mode}", flush=True)
                pair = (("old", old), ("new", new)) if len(rows) % 2 else (("new", new), ("old", old))
                outputs = {name: adapter.run_native(fixed) for name, adapter in pair}
                row["all_native_output"] = compare_all_rows(outputs["new"], outputs["old"])
                require_exact(row["all_native_output"])
                for name, adapter in (("old", old), ("new", new)):
                    actual = adapter.postprocess(outputs[name], fixed).cpu()
                    torch.testing.assert_close(actual[..., 0], torch.zeros_like(actual[..., 0]), rtol=0, atol=0,
                                               msg=name+" rowmax fixture zero channel")
                    torch.testing.assert_close(actual[..., 1], torch.full_like(actual[..., 1], 2), rtol=.004, atol=.003,
                                               msg=name+" rowmax fixture constant channel")
                row["status"] = "passed"
                save()
    return rows


def worker(args):
    config = candidate_config(args)
    candidate = config["name"]
    names = variant_names(candidate)
    namespace = "vc_laser_quant_" + config["suffix"]
    directory = Path(args._run_dir) / "paired" / "local"
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", stage="provenance", pid=os.getpid(),
        primary_baseline=ORIGINAL, original_baseline=BASELINE_DESCRIPTION, candidate=candidate,
        shape_bnsd=args.shape, dtype="bfloat16", seed=0, scopes=SCOPES,
        timings={}, comparisons={}, contract="INT8 QK, original exp and FP16 PV; no V smoothing/reordering")
    save = lambda: common.write_json(directory / "report.json", report)
    save()
    try:
        providers = resolve_providers(args)
        evidence = report["provenance"] = verify_builds(args, providers)
        import torch
        import torch_npu
        from profile_mindie_laser_quant import compare_all_rows, payload_evidence, verify_runtime_target
        from mindie_laser_quant_backend import load_quant_laser, MindIEQuantLaser
        from mindie_laser_fp16_trace_worker import verify_runtime_target as verify_fp16_target
        from mindie_laser_vscale_reference import validate_vector_scale
        torch.set_num_threads(min(args.cpu_threads, torch.get_num_threads()))
        original, transforms = quant.baseline.load_adapter(providers[ORIGINAL],
            evidence[ORIGINAL]["source_manifest"])
        report["source_transform_validation"] = transforms
        old = load_quant_laser(providers[OLD].bridge_path, preprocessing="fused",
            preprocess_build=args.preprocess_build, torch_namespace="vc_laser_quant_vscale",
            v_quantize=False, expcast=False, check_finite=False)
        from mindie_laser_quant_vscale_patch import VSCALE_CONTRACT
        candidate_contract = importlib.import_module(
            "mindie_laser_quant_"+config["suffix"]+"_patch").CONTRACT
        old.metadata["qk_result_contract"] = VSCALE_CONTRACT
        torch.ops.load_library(providers[candidate].bridge_path)
        native = getattr(torch.ops, namespace).la.default
        if ([str(x.type) for x in native._schema.arguments] != ["Tensor"]*5+["float", "int", "bool"]
                or [str(x.type) for x in native._schema.returns] != ["Tensor"]):
            raise RuntimeError(f"Unexpected candidate schema: {native._schema}")
        new = MindIEQuantLaser(native, quantizer=old.quantizer, v_quantize=False,
            expcast=False, check_finite=False, metadata=dict(old.metadata,
            dispatch=f"torch.ops.{namespace}.la.default",
            bridge_path=providers[candidate].bridge_path, native_schema=str(native._schema)))
        adapters = {ORIGINAL: original, OLD: old, candidate: new}
        new.metadata["vector_scale_schedule"] = candidate_contract
        report["numerical_contract"] = VSCALE_CONTRACT
        report[config["contract_key"]] = candidate_contract
        report["adapters"] = {name: adapter.metadata for name, adapter in adapters.items()}
        torch.npu.set_option({"ACL_OP_COMPILER_CACHE_MODE": "disable",
                              "ACL_OP_COMPILER_CACHE_DIR": os.environ["ACL_OP_COMPILER_CACHE_DIR"]})
        torch.npu.set_device(args.device)
        import ctypes
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtGetSocName.restype = ctypes.c_char_p
        soc = acl.aclrtGetSocName()
        if not soc:
            raise RuntimeError("aclrtGetSocName returned null")
        soc = soc.decode()
        cann = os.environ["ASCEND_HOME_PATH"]
        prep = json.loads(Path(args.preprocess_build).read_text(encoding="utf-8"))
        builds = {name: evidence[name]["build_manifest"] for name in names}
        report["runtime_targets"] = {
            ORIGINAL: verify_fp16_target(builds[ORIGINAL], soc, cann),
            **{name: verify_runtime_target(builds[name], prep, soc, cann) for name in (OLD, candidate)}}
        report["environment"] = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
            runtime_soc=soc, device_index=args.device, custom_opp_path=os.environ["ASCEND_CUSTOM_OPP_PATH"])
        context = dict(cann_root=cann, soc=builds[ORIGINAL]["soc"])
        device = f"npu:{args.device}"
        with torch.no_grad():
            report["stage"], report["validation"] = "structured_validation", {}
            quant.baseline.validate_small(original, device, report["validation"], save, torch)
            exact_rows = report["structured_exact_output"] = []
            exact_adapter = ExactValidationAdapter(new, old, exact_rows, save)
            validate_vector_scale({"baseline": old,
                "reuse": exact_adapter}, device, report, save)
            exact_adapter.flush()
            if args.candidate == "rowmax":
                validate_rowmax_boundaries(old, new, device, report, save)
            report["stage"] = "long_shape_validation"
            save()
            generator = torch.Generator().manual_seed(0)
            host = [torch.randn(args.shape, generator=generator).to(torch.bfloat16) for _ in range(3)]
            report["input_sha256"] = [common.tensor_sha256(x) for x in host]
            inputs = [x.to(device) for x in host]
            fixed = old.prepare(*inputs)
            payloads = {ORIGINAL: original.prepare(*inputs), OLD: fixed, candidate: fixed}
            report["payloads"] = {ORIGINAL: original.describe_payload(payloads[ORIGINAL]),
                                  "shared_int8": payload_evidence(fixed)}
            calls = make_calls(adapters, payloads, inputs)
            raw, marker_sizes = {}, {}
            report["host_markers"] = {}
            for name in names:
                marker = Path(os.environ[providers[name].marker_env])
                offset = marker.stat().st_size if marker.exists() else 0
                raw[name] = calls["native"][name]()
                torch.npu.synchronize()
                with marker.open("rb") as stream:
                    stream.seek(offset)
                    fresh = stream.read().decode("utf-8")
                report["host_markers"][name] = (quant.baseline.parse_markers(fresh, os.getpid())
                    if name == ORIGINAL else quant.parse_quant_markers(
                        fresh, os.getpid(), providers[name].marker_token))
                marker_sizes[marker] = marker.stat().st_size
            inspect_providers(providers, evidence, report, directory, **context)
            quant.inspect_fused_preprocessor(providers[OLD], report)
            for p in providers.values():
                os.environ.pop(p.marker_env, None)
            report["same_payload_all_output"] = compare_all_rows(raw[candidate], raw[OLD])
            require_exact(report["same_payload_all_output"])
            ids = torch.linspace(0, args.shape[2]-1, min(32, args.shape[2])).long().unique()
            dense = common.dense_reference(*host, ids)
            report["dense_fp32_samples"], report["full_prepared_samples"] = {}, {}
            for name, adapter in adapters.items():
                payload = payloads[name]
                if name == ORIGINAL:
                    if not isinstance(raw[name], (tuple, list)) or len(raw[name]) != 2:
                        raise RuntimeError("Original FP16 native API must return statistics/output")
                    out = adapter.postprocess(raw[name][1], payload.dtype, payload.shape[2], payload.shape[3])
                else:
                    out = adapter.postprocess(raw[name], payload)
                sampled = out.index_select(2, ids.to(device)).cpu()
                error = common.error_metrics(sampled, dense)
                report["dense_fp32_samples"][name] = error
                if name == ORIGINAL:
                    quant.baseline.record_gate(report["validation"], "original_dense_long",
                                               sampled, dense, .01, save)
                full = calls["full"][name]().index_select(2, ids.to(device)).cpu()
                prepared = calls["prepared"][name]().index_select(2, ids.to(device)).cpu()
                quant.baseline.record_gate(report["full_prepared_samples"], name, full, prepared, .001, save)
                quant.baseline.record_gate(report["full_prepared_samples"], name+"_native",
                                           sampled, prepared, .001, save)
                del out, sampled, full, prepared
            report["accuracy_scope"] = dict(query_indices=ids.tolist(), all_keys=args.shape[2],
                exact_int8_scope="ALL padded native output rows/features", no_model_quality_claim=True)
            del raw, host, dense
            save()
            for scope, group in calls.items():
                report["stage"] = "timing_"+scope
                def progress(snapshot, scope=scope):
                    report["timings"][scope] = snapshot
                    save()
                    print(f"{scope}: randomized paired trial {snapshot['completed_trials']}/{args.trials}", flush=True)
                report["timings"][scope] = run_paired(group, synchronize=torch.npu.synchronize,
                    warmup=args.warmup, repeats=args.repeats, trials=args.trials,
                    seed=args.seed, on_trial=progress)
                report["comparisons"][scope] = compare_scopes(report["timings"][scope])
                save()
            if any(path.stat().st_size != size for path, size in marker_sizes.items()):
                raise RuntimeError("Source marker writes continued during timing")
            report["timing_status"] = "passed"
            print(table(report), flush=True)
            save()
            if args.trace:
                from mindie_laser_quant_trace import capture_native_traces
                report["stage"] = "optional_profiler"
                report["trace"] = capture_native_traces(calls["native"], torch=torch, torch_npu=torch_npu,
                                                       directory=directory/"profiler", repeats=3)
                save()
            inspect_providers(providers, evidence, report, directory, **context)
            quant.inspect_fused_preprocessor(providers[OLD], report)
        report.update(status="passed", stage="complete")
        save()
        return 0
    except Exception as exc:
        if getattr(exc, "partial_report", None) is not None:
            report["trace"] = exc.partial_report
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save()
        traceback.print_exc()
        return 1


def launch(args, env, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for key in ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker",
               "--_run-dir", args._run_dir]
    for key in (*MANIFESTS, "candidate", "device", "warmup", "repeats", "trials", "seed", "cpu_threads"):
        command += ["--"+key.replace("_", "-"), str(getattr(args, key))]
    command += ["--shape", *map(str, args.shape)]
    if args.trace:
        command.append("--trace")
    with (directory/"worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    path = directory/"report.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status": "missing_report"}
    result.update(returncode=code, command=command, worker_log=str(directory/"worker.log"))
    return result


def parent(args):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args._run_dir = tempfile.mkdtemp(prefix="mindie_laser_"+candidate_config(args)["report_name"]+"_", dir=output.parent)
    report = dict(status="running", primary_baseline=ORIGINAL, candidate=candidate_config(args)["name"],
                  original_baseline=BASELINE_DESCRIPTION, run_dir=args._run_dir, workers={})
    save = lambda: common.write_json(output, report)
    save()
    try:
        providers = resolve_providers(args)
        report["provenance"] = verify_builds(args, providers)
        save()
        directory = Path(args._run_dir)/"paired"
        env = worker_environment(os.environ, providers, directory)
        result = report["workers"]["paired"] = launch(args, env, directory/"local")
        save()
        if result.get("returncode") != 0 or result.get("status") != "passed":
            raise RuntimeError("Interleaved original-Laser comparison failed; see worker_log")
        report["status"] = "passed"
        save()
        print(f"Report: {output}", flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        traceback.print_exc()
        return 1


def parser(default_candidate="bulk"):
    result = argparse.ArgumentParser(description=__doc__)
    for key, default in MANIFESTS.items():
        result.add_argument("--"+key.replace("_", "-"), default=default)
    result.add_argument("--candidate", choices=tuple(CANDIDATES), default=default_candidate)
    result.add_argument("--output", default=None)
    result.add_argument("--shape", nargs=4, type=int, default=list(quant.baseline.SHAPE))
    for key, default in (("device", 0), ("warmup", 5), ("repeats", 10), ("trials", 9),
                         ("seed", 2026), ("cpu-threads", 4)):
        result.add_argument("--"+key, type=int, default=default)
    result.add_argument("--trace", action="store_true")
    result.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--_run-dir", help=argparse.SUPPRESS)
    return result


def main(argv=None, *, default_candidate="bulk"):
    cli = parser(default_candidate)
    args = cli.parse_args(argv)
    if args.output is None:
        args.output = "results/mindie_laser_"+candidate_config(args)["report_name"]+".json"
    if (min(args.shape) <= 0 or args.shape[-1] not in (64, 128) or args.device < 0
            or args.warmup < 0 or min(args.repeats, args.trials, args.cpu_threads) <= 0):
        cli.error("Need nonempty BNSD D64/128, nonnegative device/warmup and positive repeats/trials/threads")
    return worker(args) if args._worker else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
