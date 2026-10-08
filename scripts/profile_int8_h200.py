"""H200 main route: INT8 QK + FP8 PV, with independent VC feature switches.

Four vc15 variants isolate ExpCast and V smoothing. vc22_both is a separate
ExpCast-only performance candidate. Full APIs preprocess fresh original inputs;
fixed-core APIs reuse quantized payloads and still allocate their outputs.
"""
import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace

import torch

from benchmark_cuda import measure_trial, query_indices, fp32_reference
from profile_sage_vc import stats, load_sage_core_ops, sage_core_calls, trace_layout_samples


# name, ExpCast, V smoothing, native schedule
VARIANTS = (
    ("int8_control", False, False, "vc15"),
    ("int8_expcast", True, False, "vc15"),
    ("int8_v_smooth", False, True, "vc15"),
    ("int8_vc", True, True, "vc15"),
    ("int8_expcast_vc22", True, False, "vc22_both"),
)


def require_int8_state(state):
    """Fail before attention/timing if the preparation takes a different route."""
    if state.get("qk_format", "int8") != "int8":
        raise ValueError("This H200 workflow requires INT8 QK")
    for key in ("q", "k"):
        if state[key].dtype != torch.int8:
            raise ValueError(f"{key} must be torch.int8, got {state[key].dtype}")
    if state["v"].dtype != torch.float8_e4m3fn:
        raise ValueError("V must remain E4M3 FP8")
    for key in ("qs", "ks", "vs"):
        if state[key].dtype != torch.float32:
            raise ValueError(f"{key} must remain FP32")


def make_calls(vc, sage_api, sage_native, qkv, bases):
    core, full, states = {}, {}, {}
    for smooth, state in bases.items():
        require_int8_state(state)
        if state["block_values"] != smooth:
            raise ValueError("V smoothing state does not match its comparison arm")
    for name, cast, smooth, mode in VARIANTS:
        state = {**bases[smooth], "expcast": cast}
        states[name] = state
        core[name] = lambda s=state, m=mode: vc.forward_prepared(s, kernel_mode=m)
        full[name] = lambda c=cast, s=smooth, m=mode: vc.sageattn_vc(
            *qkv, tensor_layout="HND", expcast=c, v_smooth=s,
            kernel_mode=m, v_scale_mode="channel", grouping_mode="fused")
    core["sage_cpp"] = sage_core_calls(
        SimpleNamespace(vc=vc), bases[False], sage_native, sage_native)["sage_cpp"]
    full["sage_cpp"] = lambda: sage_api(*qkv, tensor_layout="HND", is_causal=False)
    return core, full, states


def metrics(actual, expected, *, limit=None):
    if actual.shape != expected.shape or actual.numel() == 0:
        raise AssertionError("Output shapes must match and be nonempty")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("Nonfinite attention output/reference")
    delta = actual.float() - expected.float()
    error = {
        "relative_rmse": (delta.square().mean().sqrt() /
                          expected.float().square().mean().sqrt().clamp_min(1e-12)).item(),
        "max_abs": delta.abs().max().item(),
    }
    if limit is not None and error["relative_rmse"] > limit:
        raise AssertionError(f"Implementation RMSE exceeds {100*limit:g}%: {error}")
    return error


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name, value in (("batch", 1), ("queries", 37741), ("tokens", 37741), ("heads", 28),
                        ("warmup", 3), ("repeats", 5), ("trials", 5), ("check-queries", 128),
                        ("kernel-samples", 20), ("kernel-trials", 3), ("seed", 0)):
        p.add_argument("--" + name, type=int, default=value)
    p.add_argument("--dim", type=int, choices=(64, 128), default=128)
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--trace-dir", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/h200_int8_main.json"))
    args = p.parse_args(argv)
    for name in ("batch", "queries", "tokens", "heads", "warmup", "repeats", "trials",
                 "kernel_samples", "kernel_trials"):
        if getattr(args, name) < 1:
            p.error(name + " must be positive")
    if args.check_queries < 0:
        p.error("check-queries must be nonnegative (0 means all)")
    return args


@torch.no_grad()
def main():
    args = parse_args()
    from sageattention_local import vc, vc_reference, sageattn
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("Hopper required; no CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    q = torch.randn(args.batch, args.heads, args.queries, args.dim, device=device, dtype=dtype)
    k = torch.randn(args.batch, args.heads, args.tokens, args.dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    qkv = (q, k, v)
    bases = {smooth: vc.prepare(*qkv, v_smooth=smooth, v_scale_mode="channel",
                               grouping_mode="fused") for smooth in (False, True)}
    native, _ = load_sage_core_ops()
    core, full, states = make_calls(vc, sageattn, native, qkv, bases)
    ids = query_indices(args.queries, args.check_queries)
    index = torch.tensor(ids, device=device)
    print(f"INT8 QK + FP8 PV; equivalent NHD shape: {(args.batch,args.queries,args.heads,args.dim)}; "
          f"all {args.tokens} keys. APIs receive contiguous HND inputs.", flush=True)
    print(f"Accuracy: {len(ids)}/{args.queries} query rows and ALL keys; outside timing", flush=True)
    golden = fp32_reference(*qkv, ids, 32)
    accuracy = {}
    baselines = {}
    for name, call in core.items():
        out = call()
        if out.shape != q.shape or out.dtype != dtype or out.device != q.device:
            raise AssertionError(name + " returned the wrong shape/dtype/device")
        if not torch.isfinite(out).all():
            raise AssertionError(name + " produced NaN/Inf")
        # Includes every row and a fresh preprocessing run on the same inputs.
        torch.testing.assert_close(full[name](), out, rtol=0, atol=0)
        sample = out.index_select(2, index).float().cpu()
        entry = {"vs_dense_fp32": metrics(sample, golden), "full_core_exact": True}
        if name in states:
            oracle = vc_reference.reference(states[name], query_indices=ids, chunk=32).to(dtype).float().cpu()
            entry["vs_decoded_quantized_reference"] = metrics(sample, oracle, limit=.01)
        if name in ("int8_control", "int8_expcast"):
            baselines[name] = out
        if name == "int8_expcast_vc22":
            entry["vs_vc15_all_rows"] = metrics(out, baselines.pop("int8_expcast"), limit=.001)
        if name == "sage_cpp":
            entry["control_vs_sage_all_rows"] = metrics(baselines.pop("int8_control"), out, limit=.001)
        accuracy[name] = entry
        print("Checked", name, json.dumps(entry), flush=True)
    del out, sample, oracle
    for call in [*core.values(), *full.values()]:
        for _ in range(args.warmup):
            call()
    samples = {kind: {name: [] for name in core} for kind in ("core", "full_api")}
    rng, orders = random.Random(args.seed + 2800), []
    for trial in range(args.trials):
        order = [(kind, name) for kind in samples for name in core]
        rng.shuffle(order)
        orders.append(order)
        for kind, name in order:
            _, elapsed = measure_trial((core if kind == "core" else full)[name], args.repeats, device)
            samples[kind][name].append(elapsed)
        print(f"Trial {trial+1}/{args.trials} complete", flush=True)
    rows = {name: {kind: stats(samples[kind][name]) for kind in samples} for name in core}
    print("Uninstrumented synchronized API wall time; milliseconds:")
    print(f"{'variant':<25} {'full API':>12} {'fixed core':>12} {'Sage/API':>10} {'FP32 RMSE %':>14}")
    sage_ms = rows["sage_cpp"]["full_api"]["median_ms"]
    for name, row in rows.items():
        latency = row["full_api"]["median_ms"]
        row["full_api_speedup_vs_sage"] = sage_ms / latency
        print(f"{name:<25} {latency:>12.4f} {row['core']['median_ms']:>12.4f} "
              f"{sage_ms/latency:>9.3f}x {100*accuracy[name]['vs_dense_fp32']['relative_rmse']:>14.6f}")
    print("Sage/API >1 means faster. Four vc15 arms isolate feature switches; vc22 is separate.")
    print("Full API includes fresh preprocessing/grouping; fixed core includes allocation/dispatch.")
    print("1% decoded-reference / 0.1% same-algorithm gates check implementation, NOT model quality.")
    report = {
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(device), "device": str(device),
                        "native_kernel_version": vc._extension().sage_vc_kernel_version,
                        "tf32_enabled": False},
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "quantization": {"qk": "Sage per-thread INT8, input-dtype K centering, FP32 scales",
                         "pv": "FP8 E4M3, two-level FP32 accumulation",
                         "v": "per-channel FP8; optional block residuals and separate means",
                         "hadamard": False, "rope_fused": False, "denoising_schedule": False},
        "variants": {name: {"expcast": cast, "v_smooth": smooth, "kernel_mode": mode}
                     for name, cast, smooth, mode in VARIANTS},
        "rows": rows, "accuracy": accuracy, "trial_order": orders,
        "scope": "Random inputs, contiguous HND API calls. Not full paper reproduction or model end-to-end speed. "
                 "Dense-FP32 errors use selected queries/all keys; same-algorithm gates use all output rows.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.trace_dir:
        report["cupti"] = trace_layout_samples(core, args, device)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved:", args.output, flush=True)


if __name__ == "__main__":
    main()
