"""H200 main route: INT8 QK + FP8 PV, with independent VC feature switches.

Four vc15 variants isolate ExpCast and V smoothing. vc22/vc28/vc29/vc30 are
separate ExpCast-only performance candidates. vc31 has four matched feature arms.
FA4 has a full-API row and optional CUPTI kernel timing, not a prepared-core row.
Full APIs preprocess fresh original inputs;
fixed-core APIs reuse quantized payloads and still allocate their outputs.
"""
import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace

import torch

from benchmark_cuda import measure_trial, query_indices, fp32_reference, external_backend, output_hnd
from profile_sage_vc import stats, load_sage_core_ops, sage_core_calls, trace_layout_samples
from kernel_timing import profile_kernels


# name, ExpCast, V smoothing, native schedule
VARIANTS = (
    ("int8_control", False, False, "vc15"),
    ("int8_expcast", True, False, "vc15"),
    ("int8_v_smooth", False, True, "vc15"),
    ("int8_vc", True, True, "vc15"),
    ("int8_expcast_vc22", True, False, "vc22_both"),
    ("int8_expcast_vc28", True, False, "vc28_stream_i2f"),
    ("int8_expcast_vc29", True, False, "vc29_interleave_rows"),
    ("int8_expcast_vc30", True, False, "vc30_desc"),
    ("vc31_control", False, False, "vc31_unified"),
    ("vc31_expcast", True, False, "vc31_unified"),
    ("vc31_smooth", False, True, "vc31_unified"),
    ("vc31_both", True, True, "vc31_unified"),
)

KERNEL_BACKENDS = ("sage_cpp", "int8_expcast_vc28", "vc31_control", "vc31_expcast",
                   "vc31_smooth", "vc31_both", "flash4")


def load_inputs(args, device):
    if args.qkv_file is None:
        dtype = getattr(torch, args.dtype)
        q = torch.randn(args.batch, args.heads, args.queries, args.dim, device=device, dtype=dtype)
        k = torch.randn(args.batch, args.heads, args.tokens, args.dim, device=device, dtype=dtype)
        return (q, k, torch.randn_like(k)), {"kind": "random", "post_rope": False}
    data = torch.load(args.qkv_file, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or not all(isinstance(data.get(k), torch.Tensor) for k in ("q", "k", "v")):
        raise ValueError("QKV file must contain tensors under q, k, v")
    qkv = tuple(data[k] for k in ("q", "k", "v"))
    if any(x.ndim != 4 for x in qkv):
        raise ValueError("Captured Q/K/V must be four-dimensional")
    if args.qkv_layout == "NHD":
        qkv = tuple(x.transpose(1, 2) for x in qkv)
    q, k, v = qkv
    if (q.dtype not in (torch.float16, torch.bfloat16) or
            any(x.dtype != q.dtype for x in qkv) or k.shape != v.shape or
            q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1] or
            q.shape[-1] not in (64, 128) or any(x.numel() == 0 for x in qkv)):
        raise ValueError("Captured QKV require matching FP16/BF16 MHA, D64/128, nonempty shapes")
    if not all(torch.isfinite(x).all() for x in qkv):
        raise ValueError("Captured QKV contain NaN/Inf")
    args.batch, args.heads, args.queries, args.dim = q.shape
    args.tokens, args.dtype = k.shape[2], str(q.dtype).removeprefix("torch.")
    return tuple(x.to(device).contiguous() for x in qkv), {
        "kind": "captured", "file": str(args.qkv_file), "layout": args.qkv_layout,
        "post_rope": "caller must capture after RoPE; no transform is applied by this benchmark"}


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


def make_flash4(qkv):
    # Each library receives its native contiguous layout; layout copies and
    # first-use JIT are excluded. FA4 still receives original BF16/FP16 values.
    qkv_nhd = tuple(x.transpose(1, 2).contiguous() for x in qkv)
    return external_backend("flash4", qkv, qkv_nhd)


def summarize_samples(samples):
    # External APIs do not have VC's prepared-payload interface. An absent core
    # measurement stays absent, rather than copying its full-API time.
    return {name: {kind: stats(values[name]) for kind, values in samples.items() if name in values}
            for name in samples["full_api"]}


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
    p.add_argument("--paper-kernels", action="store_true", help="Additional CUPTI windows; no ncu/nvdisasm needed")
    p.add_argument("--kernel-backends", nargs="+", choices=KERNEL_BACKENDS, default=list(KERNEL_BACKENDS))
    p.add_argument("--kernel-warmup-seconds", type=float, default=20.)
    p.add_argument("--kernel-seconds", type=float, default=10.)
    p.add_argument("--qkv-file", type=Path, help="Optional local torch.save dict(q=q,k=k,v=v), captured after RoPE")
    p.add_argument("--qkv-layout", choices=("HND", "NHD"), default="NHD")
    p.add_argument("--skip-flash4", action="store_true", help="Explicitly omit FA4; default requires the installed flash-attn-4")
    p.add_argument("--output", type=Path, default=Path("results/h200_int8_main.json"))
    args = p.parse_args(argv)
    for name in ("batch", "queries", "tokens", "heads", "warmup", "repeats", "trials",
                 "kernel_samples", "kernel_trials"):
        if getattr(args, name) < 1:
            p.error(name + " must be positive")
    if args.check_queries < 0:
        p.error("check-queries must be nonnegative (0 means all)")
    for value in (args.kernel_warmup_seconds, args.kernel_seconds):
        if not 0 < value < float("inf"):
            p.error("kernel windows must be positive finite seconds")
    return args


@torch.no_grad()
def main():
    args = parse_args()
    from sageattention_local import vc, vc_reference, sageattn
    if vc._extension().sage_vc_kernel_version < 29:
        raise RuntimeError("vc31 requires rebuilding: bash scripts/validate_int8_h200.sh (without --skip-build)")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("Hopper required; no CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    qkv, input_info = load_inputs(args, device)
    q, k, v = qkv
    dtype = q.dtype
    flash4 = None if args.skip_flash4 else make_flash4(qkv)
    if flash4:
        print("FlashAttention-4:", json.dumps(flash4.info), flush=True)
    bases = {smooth: vc.prepare(*qkv, v_smooth=smooth, v_scale_mode="channel",
                               grouping_mode="fused") for smooth in (False, True)}
    native, _ = load_sage_core_ops()
    core, full, states = make_calls(vc, sageattn, native, qkv, bases)
    if flash4:
        full["flash4"] = flash4.call
    ids = query_indices(args.queries, args.check_queries)
    index = torch.tensor(ids, device=device)
    print(f"INT8 QK + FP8 PV; equivalent NHD shape: {(args.batch,args.queries,args.heads,args.dim)}; "
          f"all {args.tokens} keys. Sage/VC: contiguous HND; FA4: contiguous NHD.", flush=True)
    print(f"Accuracy: {len(ids)}/{args.queries} query rows and ALL keys; outside timing", flush=True)
    golden = fp32_reference(*qkv, ids, 32)
    accuracy = {}
    baselines = {}
    for name in full:
        print("Checking", name, "(first-use JIT is outside timing)", flush=True)
        raw = (core[name] if name in core else full[name])()
        out = output_hnd(raw, "BSHD" if name == "flash4" else "BHND", q.shape, dtype)
        if out.shape != q.shape or out.dtype != dtype or out.device != q.device:
            raise AssertionError(name + " returned the wrong shape/dtype/device")
        if not torch.isfinite(out).all():
            raise AssertionError(name + " produced NaN/Inf")
        # Includes every row and a fresh preprocessing run on the same inputs.
        if name in core:
            torch.testing.assert_close(full[name](), out, rtol=0, atol=0)
        sample = out.index_select(2, index).float().cpu()
        entry = {"vs_dense_fp32": metrics(sample, golden), "full_core_exact": True if name in core else None}
        if name in states:
            oracle = vc_reference.reference(states[name], query_indices=ids, chunk=32).to(dtype).float().cpu()
            entry["vs_decoded_quantized_reference"] = metrics(sample, oracle, limit=.01)
        if name in ("int8_control", "int8_expcast", "int8_v_smooth", "int8_vc"):
            baselines[name] = out
        if name == "int8_expcast_vc22":
            entry["vs_vc15_all_rows"] = metrics(out, baselines.pop("int8_expcast"), limit=.001)
            baselines[name] = out
        if name == "int8_expcast_vc28":
            entry["vs_vc22_all_rows"] = metrics(out, baselines.pop("int8_expcast_vc22"), limit=.001)
            baselines[name] = out
        if name == "int8_expcast_vc29":
            entry["vs_vc28_all_rows"] = metrics(out, baselines["int8_expcast_vc28"], limit=.001)
        if name == "int8_expcast_vc30":
            entry["exact_vc28_all_rows"] = torch.equal(out, baselines["int8_expcast_vc28"])
            entry["vs_vc28_all_rows"] = metrics(out, baselines["int8_expcast_vc28"], limit=.001)
        if name.startswith("vc31_"):
            baseline = {"vc31_control": "int8_control", "vc31_expcast": "int8_expcast_vc28",
                        "vc31_smooth": "int8_v_smooth", "vc31_both": "int8_vc"}[name]
            entry["vs_previous_all_rows"] = metrics(out, baselines[baseline], limit=.001)
            entry["previous_variant"] = baseline
            entry["exact_previous_all_rows"] = torch.equal(out, baselines[baseline])
            if name == "vc31_expcast" and not entry["exact_previous_all_rows"]:
                raise AssertionError("Unsmoothed vc31 must use the identical vc28 native kernel")
        if name == "sage_cpp":
            entry["control_vs_sage_all_rows"] = metrics(baselines.pop("int8_control"), out, limit=.001)
        accuracy[name] = entry
        print("Checked", name, json.dumps(entry), flush=True)
    del raw, out, sample, oracle
    baselines.clear()
    for call in [*core.values(), *full.values()]:
        for _ in range(args.warmup):
            call()
    samples = {"core": {name: [] for name in core}, "full_api": {name: [] for name in full}}
    rng, orders = random.Random(args.seed + 2800), []
    for trial in range(args.trials):
        order = [(kind, name) for kind, calls in samples.items() for name in calls]
        rng.shuffle(order)
        orders.append(order)
        for kind, name in order:
            _, elapsed = measure_trial((core if kind == "core" else full)[name], args.repeats, device)
            samples[kind][name].append(elapsed)
        print(f"Trial {trial+1}/{args.trials} complete", flush=True)
    rows = summarize_samples(samples)
    print("Uninstrumented synchronized API wall time; milliseconds:")
    print(f"{'variant':<25} {'full API':>12} {'fixed core':>12} {'Sage/API':>10} {'FA4/API':>10} {'FP32 RMSE %':>14}")
    sage_ms = rows["sage_cpp"]["full_api"]["median_ms"]
    fa4_ms = rows["flash4"]["full_api"]["median_ms"] if flash4 else None
    for name, row in rows.items():
        latency = row["full_api"]["median_ms"]
        row["full_api_speedup_vs_sage"] = sage_ms / latency
        row["full_api_speedup_vs_flash4"] = fa4_ms / latency if fa4_ms is not None else None
        core_text = f"{row['core']['median_ms']:.4f}" if "core" in row else "--"
        fa4_text = f"{fa4_ms/latency:.3f}x" if fa4_ms is not None else "--"
        print(f"{name:<25} {latency:>12.4f} {core_text:>12} "
              f"{sage_ms/latency:>9.3f}x {fa4_text:>10} {100*accuracy[name]['vs_dense_fp32']['relative_rmse']:>14.6f}")
    print("Sage/API and FA4/API >1 mean faster than that baseline's FULL API.")
    print("Four vc15 and four vc31 arms isolate feature switches separately; vc22/vc28/vc29/vc30 are ExpCast-only. FA4 has no prepared-core row.")
    print("Full API includes fresh preprocessing/grouping; fixed core includes allocation/dispatch.")
    print("1% decoded-reference / 0.1% same-algorithm gates check implementation, NOT model quality.")
    for before, after in (("vc22", "vc28"), ("vc28", "vc29"), ("vc28", "vc30")):
        old, new = rows["int8_expcast_"+before], rows["int8_expcast_"+after]
        error = accuracy["int8_expcast_"+after][f"vs_{before}_all_rows"]["relative_rmse"]
        print(f"{before}/{after}: full API {old['full_api']['median_ms']/new['full_api']['median_ms']:.4f}x; "
              f"fixed core {old['core']['median_ms']/new['core']['median_ms']:.4f}x; "
              f"ALL-row same-payload RMSE {100*error:.6f}%")
    report = {
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(device), "device": str(device),
                        "native_kernel_version": vc._extension().sage_vc_kernel_version,
                        "tf32_enabled": False},
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "inputs": input_info,
        "quantization": {"qk": "Sage per-thread INT8, input-dtype K centering, FP32 scales",
                         "pv": "FP8 E4M3, two-level FP32 accumulation",
                         "v": "per-channel FP8; optional block residuals and separate means",
                         "hadamard": False, "rope_fused": False, "denoising_schedule": False},
        "variants": {name: {"expcast": cast, "v_smooth": smooth, "kernel_mode": mode}
                     for name, cast, smooth, mode in VARIANTS},
        "external_backends": {"flash4": flash4.info} if flash4 else {},
        "compiled_resources_d128": {key: value for key, value in vc.kernel_resources(0 if dtype==torch.float16 else 1).items()
                                   if key in ("expcast_sage_layout_27", "expcast_sage_layout_35", "expcast_sage_layout_36", "expcast_sage_layout_37", "vc31_smooth", "vc31_both")},
        "rows": rows, "accuracy": accuracy, "trial_order": orders,
        "scope": "See inputs for random/captured origin; native contiguous layouts (Sage/VC HND, FA4 NHD); layout copies and first-use JIT excluded. "
                 "VC uses INT8 QK/FP8 PV; FA4 receives original input dtype, no external quantization. "
                 "Not full paper reproduction or model end-to-end speed. "
                 "Dense-FP32 errors use selected queries/all keys; same-algorithm gates use all output rows.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.paper_kernels:
        calls = {name: core[name] for name in args.kernel_backends if name != "flash4"}
        if "flash4" in args.kernel_backends and flash4:
            calls["flash4"] = flash4.call
        def save_windows(progress):
            report["kernel_windows"] = progress
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        try:
            profile_kernels(calls, args, device, on_result=save_windows)
        except Exception as exc:
            report.setdefault("kernel_windows", {}).update(status="failed", error=str(exc))
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"CUPTI timing failed; API/accuracy results preserved in {args.output}. {exc}") from exc
    elif args.trace_dir:
        report["cupti"] = trace_layout_samples(core, args, device)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved:", args.output, flush=True)


if __name__ == "__main__":
    main()
