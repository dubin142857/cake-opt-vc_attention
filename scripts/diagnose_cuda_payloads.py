"""Trace production CUDA kernels with fixed E4M3 bytes, scales and means.

Also sweep Q/K quantization blocks numerically without changing production
defaults. Requires rebuilding the 0.2.1 extension for diagnostic entry points.
"""
from dataclasses import replace
import json
import math
from pathlib import Path

import torch

import diagnose_cuda_accuracy as stages
from vc_attention.core import Config, _encode, _quantize_qk, attention, expcast_codes, prepare_value_tile, preprocess_qk
from vc_attention.cuda_backend import _extension, preflight
from vc_attention.replay import metrics


class Differences:
    """Aggregate element-weighted errors across tiles, not a mean of tile RMSEs."""
    def __init__(self):
        self.rows = {}

    def add(self, name, actual, reference, location=None):
        a, r = actual.detach().double(), reference.detach().double()
        if a.shape != r.shape or not bool(torch.isfinite(a).all() & torch.isfinite(r).all()):
            raise ValueError(f"{name}: incompatible/nonfinite diagnostic tensors")
        delta = a - r
        count = a.numel()
        different = int((actual != reference).sum().item())
        maximum = delta.abs().max().item()
        row = self.rows.setdefault(name, dict(count=0, different=0, error2=0., reference2=0., max_abs=0., first_difference=None))
        row["count"] += count
        row["different"] += different
        row["error2"] += delta.square().sum().item()
        row["reference2"] += r.square().sum().item()
        row["max_abs"] = max(row["max_abs"], maximum)
        if different and row["first_difference"] is None:
            row["first_difference"] = location or "full tensor"

    def report(self):
        return {name: {"elements": r["count"], "different_elements": r["different"],
                       "different_fraction": r["different"] / r["count"],
                       "relative_rmse": math.sqrt(r["error2"] / max(r["reference2"], 1e-30)),
                       "max_abs": r["max_abs"], "first_difference": r["first_difference"]}
                for name, r in self.rows.items()}


def decode(codes):
    return codes.view(torch.float8_e4m3fn).float()


@torch.no_grad()
def reference_qk(x, block):
    n, d = x.shape[-2:]
    tiles = math.ceil(n / block)
    codes = torch.zeros((*x.shape[:2], tiles * block, math.ceil(d / 32) * 32), device=x.device, dtype=torch.uint8)
    scales = torch.empty((*x.shape[:2], tiles), device=x.device)
    for i in range(tiles):
        tile = x[..., i * block:(i + 1) * block, :]
        p, s = _encode(tile, (-2, -1))
        codes[..., i * block:i * block + tile.shape[-2], :d] = p.view(torch.uint8)
        scales[..., i] = s[..., 0, 0]
    return codes, scales


@torch.no_grad()
def reference_values(v, cfg, active):
    nt = math.ceil(v.shape[-2] / cfg.kv_block)
    vp = math.ceil(v.shape[-1] / 32) * 32
    shape = (*v.shape[:2], nt, vp)
    codes = torch.zeros((*shape, cfg.kv_block), device=v.device, dtype=torch.uint8)
    scales = torch.ones(shape, device=v.device)
    means = torch.zeros(shape, device=v.device, dtype=getattr(torch, cfg.mean_dtype))
    for j in range(nt):
        p, s, mu = prepare_value_tile(v[..., j * cfg.kv_block:(j + 1) * cfg.kv_block, :], active, cfg)
        codes[..., j, :v.shape[-1], :p.shape[-2]] = p.view(torch.uint8).transpose(-1, -2)
        scales[..., j, :v.shape[-1]] = s[..., 0, :]
        means[..., j, :v.shape[-1]] = mu[..., 0, :].to(means.dtype)
    return codes, scales, means


@torch.no_grad()
def prepare_packs(qkv, cfg, prepared, ext):
    q, k, v = (x.float() for x in qkv)
    if prepared.permutation is not None:
        k = k.gather(-2, prepared.permutation.unsqueeze(-1).expand_as(k))
        v = v.gather(-2, prepared.permutation.unsqueeze(-1).expand_as(v))
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    qr, kr = preprocess_qk(q, k, k_smooth=cfg.k_smooth, qk_hadamard=cfg.qk_hadamard)
    qn, kn = ext.preprocess_qk(q, k, cfg.k_smooth, cfg.qk_hadamard)
    ref = dict(zip(("q8", "qs"), reference_qk(qr, cfg.q_block)))
    ref.update(zip(("k8", "ks"), reference_qk(kr, cfg.k_quant_block)))
    ref.update(zip(("v8", "vs", "mean"), reference_values(v, cfg, prepared.smoothing_active)))
    native = dict(zip(("q8", "qs"), ext.debug_quantize(qn.contiguous(), cfg.q_block)))
    native.update(zip(("k8", "ks"), ext.debug_quantize(kn.contiguous(), cfg.k_quant_block)))
    native.update(zip(("v8", "vs", "mean"), ext.debug_values(v, cfg.kv_block, prepared.smoothing_active,
                                                             {"float32": 0, "float16": 1, "bfloat16": 2}[cfg.mean_dtype])))
    comparisons = Differences()
    comparisons.add("Q_transform", qn, qr)
    comparisons.add("K_transform", kn, kr)
    for key in ref:
        comparisons.add(key, native[key], ref[key])
    # Repeat encoding with the very same transformed FP32 values. This separates
    # Hadamard/centering from scaling division and byte conversion differences.
    for prefix, x, block in (("q", qr, cfg.q_block), ("k", kr, cfg.k_quant_block)):
        same_codes, same_scales = ext.debug_quantize(x.contiguous(), block)
        comparisons.add(f"{prefix.upper()}_bytes_same_transform", same_codes, ref[f"{prefix}8"])
        comparisons.add(f"{prefix.upper()}_scale_same_transform", same_scales, ref[f"{prefix}s"])
    return ref, native, comparisons.report()


def python_softmax(raw, qs, ks, maximum, denom, scale, rows, keys, cast):
    scores = ((raw * qs) * ks) * scale  # Match native post-GEMM scale order.
    scores = scores.clone()
    scores[:, keys:] = -math.inf
    new_max = torch.maximum(maximum, scores.amax(-1))
    new_max[rows:] = 0
    alpha = torch.exp(maximum - new_max)
    shifted = scores - new_max[:, None]
    if cast:
        codes = expcast_codes(shifted)
        p = decode(codes) / 256
    else:
        p = shifted.exp()
        codes = (p * 256).to(torch.float8_e4m3fn).view(torch.uint8)
    codes = codes.clone()
    codes[rows:, :] = 0
    codes[:, keys:] = 0
    p[rows:, :] = 0
    p[:, keys:] = 0
    alpha[rows:] = 0
    mass = p.sum(-1)
    return codes, alpha, mass, new_max, alpha * denom + mass


def python_accumulate(accum, pv, mass, alpha, vs, mean):
    return alpha[:, None] * accum + (pv / 256 + mass[:, None] * mean.float()) * vs


@torch.no_grad()
def replay_packed(pack, qkv, cfg, scale, cast, ext=None, progress=lambda _: None):
    """Replay Python, and optionally native, with identical encoded operands.

    Isolated checks feed the native intermediate/state into the corresponding
    Python step, so a QK difference is not mislabeled as a Softmax bug.
    """
    q, k, v = qkv
    bq, bk, vp = cfg.q_block, cfg.kv_block, pack["v8"].shape[-2]
    output = torch.empty((*q.shape[:-1], v.shape[-1]), device=q.device)
    native_output = torch.empty_like(output) if ext is not None else None
    differences = Differences()
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            progress(f"    batch {b}, head {h}: all query/key tiles")
            for qi, start in enumerate(range(0, q.shape[2], bq)):
                rows = min(bq, q.shape[2] - start)
                qa = pack["q8"][b, h, start:start + bq].contiguous()
                qs = pack["qs"][b, h, qi].contiguous()
                maximum = torch.full((bq,), -math.inf, device=q.device)
                denom = torch.zeros_like(maximum)
                accum = torch.zeros((bq, vp), device=q.device)
                if ext is not None:
                    nm, nd, na = maximum.clone(), denom.clone(), accum.clone()
                for ki, ks_start in enumerate(range(0, k.shape[2], bk)):
                    keys = min(bk, k.shape[2] - ks_start)
                    ka = pack["k8"][b, h, ks_start:ks_start + bk].contiguous()
                    ks = pack["ks"][b, h, ks_start // cfg.k_quant_block].contiguous()
                    va = pack["v8"][b, h, ki].contiguous()  # [value channel, token]
                    vs, mean = pack["vs"][b, h, ki], pack["mean"][b, h, ki]
                    raw = decode(qa) @ decode(ka).T
                    p, alpha, mass, maximum, denom = python_softmax(raw, qs, ks, maximum, denom, scale, rows, keys, cast)
                    pv = decode(p) @ decode(va).T
                    accum = python_accumulate(accum, pv, mass, alpha, vs, mean)
                    if ext is None:
                        continue
                    location = dict(batch=b, head=h, q_tile=qi, kv_tile=ki)
                    nr = ext.debug_gemm(qa, ka)
                    differences.add("QK_same_bytes", nr[:rows, :keys], raw[:rows, :keys], location)
                    isolated = python_softmax(nr, qs, ks, nm, nd, scale, rows, keys, cast)
                    np, nal, nma, nm, nd = ext.debug_softmax(nr, qs, ks, nm, nd, scale, rows, keys, cast)
                    differences.add("P_bytes_same_scores_state", np[:rows, :keys], isolated[0][:rows, :keys], location)
                    for name, actual, expected in zip(("alpha", "rowmass", "maximum", "denom"), (nal, nma, nm, nd), isolated[1:]):
                        differences.add(f"Softmax_{name}_same_inputs", actual[:rows], expected[:rows], location)
                    nv = ext.debug_gemm(np.contiguous(), va)
                    expected_pv = decode(np) @ decode(va).T
                    differences.add("PV_same_bytes", nv[:rows, :v.shape[-1]], expected_pv[:rows, :v.shape[-1]], location)
                    expected_accum = python_accumulate(na, nv, nma, nal, vs, mean)
                    na, no = ext.debug_accumulate(na, nv, nma, nal, vs, mean, nd, rows)
                    differences.add("accum_same_inputs", na[:rows, :v.shape[-1]], expected_accum[:rows, :v.shape[-1]], location)
                    differences.add("normalize_same_state", no[:rows, :v.shape[-1]],
                                    (na[:rows] / nd[:rows, None])[:, :v.shape[-1]], location)
                output[b, h, start:start + rows] = (accum[:rows] / denom[:rows, None])[:, :v.shape[-1]]
                if ext is not None:
                    native_output[b, h, start:start + rows] = no[:rows, :v.shape[-1]]
    return output, native_output, differences.report()


def qk_granularity_sweep(prepared, qkv, indices, golden):
    """Change only numerical Q/K blocks. Streaming Q/KV tile geometry is fixed."""
    rows = {}
    for qb, kb in ((128, 256), (128, 128), (64, 64), (32, 32)):
        current = replace(prepared, quantized_q=_quantize_qk(prepared.q, qb),
                          quantized_k=_quantize_qk(prepared.k, kb))
        row = {"q_quant_block": qb, "k_quant_block": kb, "qk_format": "E4M3",
               "attention_q_tile": prepared.config.q_block, "attention_kv_tile": prepared.config.kv_block}
        for name, probability, qv in (("qk_only", "fp32", False), ("full_expcast", "expcast", True)):
            output = stages.stage_output(current, quantize_qk=True, probability=probability, quantize_v=qv)
            sample = stages.select_output(output, indices)
            row[name] = {"fp32_output": metrics(sample, golden),
                         "cast_output": metrics(sample.to(qkv[0].dtype).float(), golden)}
        rows[f"q{qb}_k{kb}"] = row
    return rows


def parse_args(argv=None):
    # Share the existing input/capture/scale options, but native kernels are
    # mandatory here. --reference-only belongs to the stage diagnostic.
    args = stages.parse_args(argv, description=__doc__)
    if args.reference_only:
        raise ValueError("Payload tracing requires native CUDA; use diagnose_cuda_accuracy.py for reference-only")
    if args.output == "results/h200_vc_stages.json":
        args.output = "results/h200_vc_payloads.json"
    return args


@torch.no_grad()
def run(args):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Payload tracing requires CUDA and a Hopper GPU")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    ext = _extension()
    if getattr(ext, "diagnostic_api_version", None) != 1:
        raise RuntimeError("Rebuild vc-attention 0.2.1 first: bash scripts/validate_h800.sh")
    environment = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(device), "tf32_enabled": False,
                   "preflight": preflight(device)}
    print(json.dumps(environment, indent=2), flush=True)
    qkv, scale, source = stages.load_inputs(args, device)
    if max(qkv[0].shape[-1], qkv[2].shape[-1]) > 128:
        raise ValueError("Native payload diagnostics require Q/K/V widths <= 128")
    indices = stages.query_indices(qkv[0].shape[2], args.check_queries)
    golden = stages.dense_reference(qkv, indices, args.reference_chunk, scale)
    report = dict(schema_version=1, environment=environment, source=source, arguments=vars(args),
                  query_indices=indices, scale=scale, all_keys=True, full_q_before_sampling=True, results={})
    for smooth in (False, True):
        label = "v_smooth_on" if smooth else "v_smooth_off"
        print(f"\n{label}: preprocessing and Q/K granularity audit", flush=True)
        cfg = Config(v_smooth=smooth)
        prepared = stages.prepare(qkv, cfg, scale)
        ref_pack, native_pack, prep_errors = prepare_packs(qkv, cfg, prepared, ext)
        group = {"preprocessing": prep_errors,
                 "qk_granularity": qk_granularity_sweep(prepared, qkv, indices, golden), "paths": {}}
        for cast in (False, True):
            name = "expcast" if cast else "e4m3"
            print(f"{label}/{name}: shared native payload replay", flush=True)
            shared_ref, shared_cuda, tile_errors = replay_packed(
                native_pack, qkv, cfg, scale, cast, ext, progress=lambda s: print(s, flush=True))
            reference_pack_output, _, _ = replay_packed(ref_pack, qkv, cfg, scale, cast)
            ref_cfg = replace(cfg, expcast=cast)
            api_ref, _ = attention(*qkv, ref_cfg, cache=stages.FixedLayout(ref_cfg, prepared), scale=scale)
            native_cfg = replace(ref_cfg, backend="cuda_fp8")
            api_cuda, _ = attention(*qkv, native_cfg, cache=stages.FixedLayout(native_cfg, prepared), scale=scale)
            outputs = {"shared_ref": shared_ref, "shared_cuda": shared_cuda, "ref_pack": reference_pack_output,
                       "api_ref": api_ref, "api_cuda": api_cuda}
            samples = {key: stages.select_output(x, indices) for key, x in outputs.items()}
            cast_samples = {key: x.to(qkv[0].dtype).float() for key, x in samples.items()}
            comparisons = {}
            for check, actual, expected in (
                ("shared_payload_CUDA_vs_FP32_replay", "shared_cuda", "shared_ref"),
                ("payload_encoding_effect", "shared_ref", "ref_pack"),
                ("reference_scale_order_and_output_cast", "ref_pack", "api_ref"),
                ("replay_vs_production_CUDA", "shared_cuda", "api_cuda"),
                ("production_CUDA_vs_reference", "api_cuda", "api_ref"),
            ):
                comparisons[check] = {"fp32_or_api_output": metrics(samples[actual], samples[expected]),
                                      "cast_output": metrics(cast_samples[actual], cast_samples[expected])}
            group["paths"][name] = {"tile_errors": tile_errors, "comparisons": comparisons,
                                     "outputs_vs_dense": {key: metrics(value, golden) for key, value in cast_samples.items()}}
        report["results"][label] = group
    report["notes"] = [
        "All stages reuse production CUDA kernels; debug_gemm uses the same cuBLASLt descriptor and FAST_ACCUM=0.",
        "Native and FP32 shared-payload replay consume identical Q/K/V bytes, scales and stored means.",
        "Isolated Softmax and accumulation tests consume identical native intermediate inputs and previous state.",
        "The existing Python reference dequantizes Q/K before GEMM; packed replay applies scales after raw GEMM, matching CUDA.",
        "Raw comparisons against API outputs include final output dtype rounding; cast_output aligns both output dtypes.",
        "Granularity sweep is E4M3 numerical sensitivity analysis, not a claim of author kernel defaults or performance.",
        "Byte mismatch RMSE measures byte numbers, not decoded value error; use different_elements/fraction for byte rows.",
        "All tiles are traced; check-queries only selects final-output metrics. Grouping is fixed within each smoothing group.",
        "JSON relative errors are ratios; printed tables use PERCENT. No universal quality acceptance threshold.",
    ]
    return report


def print_report(report):
    for name, group in report["results"].items():
        print(f"\n{name}: preprocessing (different %, max abs; byte rows count payload differences)")
        for key, row in group["preprocessing"].items():
            print(f"  {key:<32} {100 * row['different_fraction']:10.6f}% {row['max_abs']:14.8g}")
        print("Q/K block sweep: FP32 output relative RMSE % vs dense (reference only)")
        for key, row in group["qk_granularity"].items():
            print(f"  {key:<16} QK only {100 * row['qk_only']['fp32_output']['relative_rmse']:.6f}% "
                  f" full ExpCast {100 * row['full_expcast']['fp32_output']['relative_rmse']:.6f}%")
        for mode, path in group["paths"].items():
            print(f"{mode}: isolated tile checks (rel RMSE %, different %, max abs)")
            for key, row in path["tile_errors"].items():
                print(f"  {key:<34} {100 * row['relative_rmse']:10.6f}% {100 * row['different_fraction']:10.6f}% {row['max_abs']:14.8g}")
            print("Output comparisons after matching API output dtype (relative RMSE %)")
            for key, row in path["comparisons"].items():
                print(f"  {key:<42} {100 * row['cast_output']['relative_rmse']:10.6f}% max {row['cast_output']['max_abs']:.8g}")
    print("Errors are pairwise, not additive. Packed replay restores scales after GEMM; existing reference restores Q/K before GEMM.")


def main():
    report = run(parse_args())
    print_report(report)
    path = Path(report["arguments"]["output"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report saved to {path}")


if __name__ == "__main__":
    main()
