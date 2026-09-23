"""Isolate VC Q/K, P and V quantization error on identical post-RoPE inputs.

This is a numerical diagnostic, not a performance benchmark. Partial quantization
runs decode E4M3 operands and multiply in FP32. The native comparisons exercise
the installed, unmodified CUDA API separately.
"""
import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform

import torch

from vc_attention.core import (
    Config, LayoutCache, _quantize_qk, _validate, attention, expcast_codes,
    prepare_value_tile, preprocess_qk,
)
from vc_attention.replay import metrics


# P=FP32 means no probability encoding. Ordinary E4M3 and ExpCast deliberately
# have different denominator rules, exactly as in core.attention.
STAGES = {
    "none": (False, "fp32", False),
    "qk": (True, "fp32", False),
    "p": (False, "e4m3", False),
    "v": (False, "fp32", True),
    "qk+p": (True, "e4m3", False),
    "qk+v": (True, "fp32", True),
    "p+v": (False, "e4m3", True),
    "qk+p+v": (True, "e4m3", True),
    "expcast": (False, "expcast", False),
    "qk+expcast": (True, "expcast", False),
    "expcast+v": (False, "expcast", True),
    "qk+expcast+v": (True, "expcast", True),
}


@dataclass
class Prepared:
    config: Config
    q: torch.Tensor
    k: torch.Tensor
    quantized_q: torch.Tensor
    quantized_k: torch.Tensor
    values: list
    quantized_values: list
    permutation: object
    smoothing_active: bool
    output_dtype: torch.dtype
    scale: float


@torch.no_grad()
def prepare(qkv, config, scale):
    """Compute one layout, full-Q quantization and value encoding for all stages."""
    q, k, v = qkv
    _validate(q, k, v, False, None)
    if config.backend != "reference" or not config.quantize:
        raise ValueError("diagnostics require a quantized reference Config")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    qf, kf, vf = q.float(), k.float(), v.float()
    pi, active, _ = LayoutCache(config).layout(
        vf, step=0, total_steps=1, request_id="diagnostic", layer_id=0,
    )
    if pi is not None:
        kf = kf.gather(-2, pi.unsqueeze(-1).expand_as(kf))
        vf = vf.gather(-2, pi.unsqueeze(-1).expand_as(vf))
    qf, kf = preprocess_qk(qf, kf, k_smooth=config.k_smooth, qk_hadamard=config.qk_hadamard)
    values, quantized_values = [], []
    unquantized = replace(config, quantize=False)
    for start in range(0, vf.shape[-2], config.kv_block):
        tile = vf[..., start:start + config.kv_block, :]
        values.append(prepare_value_tile(tile, active, unquantized))
        quantized_values.append(prepare_value_tile(tile, active, config))
    return Prepared(
        config, qf, kf, _quantize_qk(qf, config.q_block),
        _quantize_qk(kf, config.k_quant_block), values, quantized_values,
        pi, active, q.dtype, scale,
    )


@torch.no_grad()
def stage_output(prepared, *, quantize_qk, probability, quantize_v):
    """Retain FP32 output to distinguish algorithm error from final dtype cast."""
    if probability not in ("fp32", "e4m3", "expcast"):
        raise ValueError("Unknown probability path")
    cfg = prepared.config
    q = prepared.quantized_q if quantize_qk else prepared.q
    k = prepared.quantized_k if quantize_qk else prepared.k
    values = prepared.quantized_values if quantize_v else prepared.values
    output = torch.empty((*q.shape[:-1], values[0][0].shape[-1]), device=q.device, dtype=torch.float32)
    for qs in range(0, q.shape[-2], cfg.q_block):
        qt = q[..., qs:qs + cfg.q_block, :]
        shape = (*qt.shape[:-1], 1)
        maximum = torch.full(shape, -math.inf, device=q.device)
        denom = torch.zeros(shape, device=q.device)
        accum = torch.zeros((*qt.shape[:-1], output.shape[-1]), device=q.device)
        for j, ks in enumerate(range(0, k.shape[-2], cfg.kv_block)):
            scores = (qt @ k[..., ks:ks + cfg.kv_block, :].transpose(-2, -1)) * prepared.scale
            new_max = torch.maximum(maximum, scores.amax(-1, keepdim=True))
            alpha = torch.exp(maximum - new_max)
            shifted = scores - new_max
            if probability == "expcast":
                operand = expcast_codes(shifted).view(torch.float8_e4m3fn).float()
                p = operand / 256.0
            else:
                p = torch.exp(shifted)
                operand = p * 256.0
                if probability == "e4m3":
                    operand = operand.to(torch.float8_e4m3fn).float()
            # Ordinary exp retains its pre-encoding row sum. ExpCast uses its
            # decoded probabilities for both the row sum and mean restoration.
            mass = p.sum(-1, keepdim=True)
            residual, vscale, scaled_mean = values[j]
            contribution = (operand @ residual.float() / 256.0 + mass * scaled_mean) * vscale
            accum = alpha * accum + contribution
            denom = alpha * denom + mass
            maximum = new_max
        output[..., qs:qs + cfg.q_block, :] = accum / denom
    return output


def query_indices(length, count):
    count = length if count == 0 else min(count, length)
    return [0] if count == 1 else [i * (length - 1) // (count - 1) for i in range(count)]


@torch.no_grad()
def dense_reference(qkv, indices, chunk, scale):
    """Independent dense softmax oracle, query-chunked but using every key."""
    q, k, v = qkv
    k, v = k.float(), v.float()
    outputs = []
    for start in range(0, len(indices), chunk):
        ids = torch.tensor(indices[start:start + chunk], device=q.device)
        scores = (q.index_select(2, ids).float() @ k.transpose(-2, -1)) * scale
        outputs.append((scores.softmax(-1) @ v).cpu())
    return torch.cat(outputs, dim=2)


def select_output(output, indices):
    return output.index_select(2, torch.tensor(indices, device=output.device)).float().cpu()


def tensor_digest(tensor):
    # uint8 view also supports BF16 and avoids a numerical conversion in the hash.
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def load_inputs(args, device):
    if args.capture:
        path = Path(args.capture)
        record = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(record, dict) or not all(isinstance(record.get(x), torch.Tensor) for x in ("q", "k", "v")):
            raise ValueError("capture must contain Tensor entries q, k, v in [B,H,N,D] layout")
        qkv = tuple(record[x].to(device) for x in ("q", "k", "v"))
        capture_metadata = record.get("metadata", {})
        stored_scale = capture_metadata.get("scale")
        scale = args.scale if args.scale is not None else stored_scale
        source = {"kind": "captured_post_rope_qkv", "file": str(path.resolve()),
                  "stored_scale": stored_scale, "dtype_preserved": True}
    else:
        # Same generation order/device/dtype as benchmark_cuda.py, after preflight.
        torch.manual_seed(args.seed)
        dtype = getattr(torch, args.dtype)
        qkv = tuple(torch.randn(args.batch, args.heads, n, args.dim, device=device, dtype=dtype)
                    for n in (args.queries, args.tokens, args.tokens))
        scale = args.scale
        source = {"kind": "synthetic_gaussian", "seed": args.seed,
                  "generator": "torch.randn on target device, same recipe as benchmark_cuda.py"}
    _validate(*qkv, False, None)
    scale = qkv[0].shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    source["inputs"] = {name: {"shape": list(x.shape), "dtype": str(x.dtype), "sha256": tensor_digest(x)}
                        for name, x in zip(("q", "k", "v"), qkv)}
    return qkv, scale, source


class FixedLayout:
    """Replay a prepared layout through the public API, without reclustering.

    Matches the cache.config / cache.layout protocol used by attention. It only
    supplies the permutation; Q/K preprocessing and V encoding run in the API.
    """
    def __init__(self, config, prepared):
        self.config = config
        self.prepared = prepared

    def layout(self, v, **kwargs):
        return self.prepared.permutation, self.prepared.smoothing_active, False


@torch.no_grad()
def diagnose(qkv, scale, config, indices, chunk, native, progress=print):
    golden = dense_reference(qkv, indices, chunk, scale)
    if not torch.isfinite(golden).all().item():
        raise ValueError("Nonfinite dense reference")
    results = {}
    for smooth in (False, True):
        cfg = replace(config, backend="reference", quantize=True, v_smooth=smooth)
        group_name = "v_smooth_on" if smooth else "v_smooth_off"
        progress(f"Preparing {group_name}: one fixed layout; full Q/K/V")
        prepared = prepare(qkv, cfg, scale)
        group = {"config": asdict(cfg), "smoothing_active": prepared.smoothing_active,
                 "permutation_sha256": None if prepared.permutation is None else tensor_digest(prepared.permutation),
                 "stages": {}, "reference_checks": {}, "native_checks": {}}
        full_samples = {}
        for name, (qk, probability, v) in STAGES.items():
            progress(f"  {group_name}: {name}")
            output = stage_output(prepared, quantize_qk=qk, probability=probability, quantize_v=v)
            sample = select_output(output, indices)
            cast_sample = sample.to(qkv[0].dtype).float()
            group["stages"][name] = {
                "quantize_qk": qk, "probability": probability, "quantize_v": v,
                "fp32_output_vs_dense": metrics(sample, golden),
                "cast_output_vs_dense": metrics(cast_sample, golden),
                "cast_vs_fp32_output": metrics(cast_sample, sample),
            }
            if name in ("qk+p+v", "qk+expcast+v"):
                full_samples[probability] = cast_sample
            del output
        # Verify diagnostic reconstruction against the existing Python API using
        # the exact same layout. This is a meaningful cross-check, not author code.
        for probability, cast in (("e4m3", False), ("expcast", True)):
            api_cfg = replace(cfg, expcast=cast)
            progress(f"  {group_name}: public reference check ({probability}, fixed layout)")
            reference, _ = attention(*qkv, api_cfg, cache=FixedLayout(api_cfg, prepared), scale=scale)
            ref_sample = select_output(reference, indices)
            group["reference_checks"][probability] = metrics(full_samples[probability], ref_sample)
            if native:
                native_cfg = replace(api_cfg, backend="cuda_fp8")
                progress(f"  {group_name}: native CUDA ({probability}, fixed layout)")
                output, stats = attention(*qkv, native_cfg, cache=FixedLayout(native_cfg, prepared), scale=scale)
                sample = select_output(output, indices)
                group["native_checks"][probability] = {
                    "stats": stats, "cuda_vs_dense": metrics(sample, golden),
                    "reference_vs_dense": metrics(ref_sample, golden),
                    "cuda_vs_reference": metrics(sample, ref_sample),
                    "scope": "identical frozen permutation; each backend independently preprocesses and quantizes",
                }
        results[group_name] = group
    return results


def parse_args(argv=None, *, description=None):
    parser = argparse.ArgumentParser(description=__doc__ if description is None else description)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-only", action="store_true", help="explicitly skip native CUDA comparisons; supports CPU or CUDA")
    parser.add_argument("--capture", help="trusted .pt file with post-RoPE q/k/v [B,H,N,D]; preserve dtype, full context")
    parser.add_argument("--scale", type=float, help="override softmax scale; otherwise capture metadata.scale or original D**-0.5")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--check-queries", type=int, default=0, help="0 checks ALL queries (default); sampling is after full-Q quantization")
    parser.add_argument("--reference-chunk", type=int, default=32)
    parser.add_argument("--output", default="results/h200_vc_stages.json")
    args = parser.parse_args(argv)
    if min(args.batch, args.heads, args.queries, args.tokens, args.dim, args.reference_chunk) <= 0 or args.check_queries < 0:
        parser.error("shapes/chunk must be positive and check-queries must be nonnegative")
    if args.scale is not None and (not math.isfinite(args.scale) or args.scale <= 0):
        parser.error("scale must be positive and finite")
    return args


def run(args):
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This diagnostic supports CPU and CUDA only")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA is unavailable; no CPU fallback")
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "torch_cuda": torch.version.cuda, "device": str(device), "tf32_enabled": False}
    if device.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(device)
    try:
        environment["vc_version"] = metadata.version("vc-attention")
    except metadata.PackageNotFoundError:
        environment["vc_version"] = "source checkout"
    if not args.reference_only:
        from vc_attention.cuda_backend import preflight
        environment["preflight"] = preflight(device)
    print(json.dumps(environment, indent=2), flush=True)
    qkv, scale, source = load_inputs(args, device)
    indices = query_indices(qkv[0].shape[-2], args.check_queries)
    print(f"Checking {len(indices)}/{qkv[0].shape[-2]} queries, ALL {qkv[1].shape[-2]} keys; no latency measurement", flush=True)
    results = diagnose(qkv, scale, Config(), indices, args.reference_chunk,
                       native=not args.reference_only, progress=lambda s: print(s, flush=True))
    return {
        "schema_version": 1, "environment": environment, "arguments": vars(args), "source": source,
        "scale": scale, "query_indices": indices, "full_q_before_sampling": True, "all_keys": True,
        "native_enabled": not args.reference_only, "results": results,
        "notes": [
            "Partial quantization uses decoded E4M3 and FP32 matmuls, not partial native FP8 kernels. No timings.",
            "QK means quantizing both Q and K after shared centering/Hadamard preprocessing.",
            "P uses ordinary exp with E4M3 probabilities for PV but an unrounded row sum. ExpCast uses decoded row sums.",
            "V includes residual E4M3 and scaled mean storage precision when smoothing is active.",
            "Each smoothing group shares one permutation across every stage and both public-API checks.",
            "Native checks isolate attention/preprocessing agreement with grouping held fixed; not end-to-end clustering agreement.",
            "This isolated call uses step 0 in the smoothing window, including for captured inputs; not schedule replay.",
            "fp32_output retains the FP32 accumulator result; cast_output includes the original input dtype output cast.",
            "Relative errors are ratios in JSON and PERCENT in tables; stage errors do not add linearly. No quality pass/fail threshold.",
            "Synthetic inputs do not establish real-model quality; capture mode uses all supplied Q/K/V without requantizing inputs.",
        ],
    }


def print_report(report):
    for group_name, group in report["results"].items():
        print(f"\n{group_name}: relative RMSE in PERCENT vs dense FP32 (fixed layout)")
        print(f"{'stage':<20} {'FP32-out %':>12} {'cast-out %':>12} {'cast max abs':>14}")
        for name, row in group["stages"].items():
            before, after = row["fp32_output_vs_dense"], row["cast_output_vs_dense"]
            print(f"{name:<20} {100 * before['relative_rmse']:12.6f} {100 * after['relative_rmse']:12.6f} {after['max_abs']:14.8f}")
        print("Diagnostic/full Python reference check (same layout):")
        for name, row in group["reference_checks"].items():
            print(f"  {name}: rel RMSE {100 * row['relative_rmse']:.6f}%, max abs {row['max_abs']:.8f}")
        if group["native_checks"]:
            print(f"{'native P mode':<20} {'CUDA/FP32 %':>14} {'REF/FP32 %':>14} {'CUDA/REF %':>14} {'CUDA/REF max':>14}")
            for name, row in group["native_checks"].items():
                print(f"{name:<20} {100 * row['cuda_vs_dense']['relative_rmse']:14.6f} "
                      f"{100 * row['reference_vs_dense']['relative_rmse']:14.6f} "
                      f"{100 * row['cuda_vs_reference']['relative_rmse']:14.6f} {row['cuda_vs_reference']['max_abs']:14.8f}")
    print("QK/P/V stages use FP32 matmuls on decoded operands. They are accuracy diagnostics, not kernel performance.")
    print("FP32-out precedes the final output dtype cast; cast-out matches the API output dtype.")
    print("Native comparisons hold grouping fixed. Errors are not additive; no quality pass/fail threshold.")


def main():
    args = parse_args()
    report = run(args)
    print_report(report)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report saved to {path}")


if __name__ == "__main__":
    main()
