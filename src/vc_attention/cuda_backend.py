"""Hopper entry point; named to avoid shadowing NVIDIA's cuda namespace."""
import importlib
import math

import torch


def _extension():
    try:
        ext = importlib.import_module("vc_attention._cuda_ext")
    except ImportError as exc:
        raise RuntimeError(
            "CUDA extension is missing or failed to load. On the H800 host run "
            "TORCH_CUDA_ARCH_LIST=9.0 python3 -m pip install . --no-build-isolation. "
            "No reference fallback was used. Original error: " + str(exc)
        ) from exc
    if getattr(ext, "numerical_contract_version", None) != 2:
        raise RuntimeError("Stale CUDA extension: rebuild with python3 -m pip install . --no-build-isolation --no-deps")
    if getattr(ext, "scheduling_version", None) != 1:
        raise RuntimeError("Rebuild vc-attention 0.3.0 for batched scheduling: bash scripts/validate_h800.sh")
    return ext


def group_values(v, *, clusters, iterations, centers=None):
    return _extension().group_values(v, clusters, iterations, centers)


def _fused_variant(q_dim, v_dim, cfg):
    if not cfg.cuda_fused_pipeline:
        return "baseline"
    if (cfg.cuda_fused_specialize and q_dim == v_dim == 128 and cfg.q_block == 128
            and cfg.k_quant_block == 256 and cfg.kv_block == 128 and cfg.mean_dtype == "float16"):
        return "interleaved" if cfg.cuda_fused_interleave and cfg.expcast else "d128"
    return "pipeline"


@torch.no_grad()
def _attention(q, k, v, cfg, *, cache, step, total_steps, request_id, layer_id, scale):
    # Shape/dtype/finite checks run in core.attention before dispatch. C++ also
    # validates bounds required for memory safety when called directly.
    if q.device.type != "cuda":
        raise ValueError("cuda_fp8 requires CUDA tensors; no CPU/NPU fallback")
    if torch.cuda.get_device_capability(q.device)[0] != 9:
        raise ValueError("cuda_fp8 requires Hopper (H800/H100/H200, compute capability 9.x)")
    if q.shape[-1] > 128 or v.shape[-1] > 128:
        raise ValueError("cuda_fp8 supports Q/K/V head dimensions <= 128")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    from .core import LayoutCache

    ext = _extension()
    if cfg.cuda_schedule == "fused" and getattr(ext, "fused_api_version", None) != 1:
        raise RuntimeError("Rebuild vc-attention 0.4.0 for fused FP8 attention: bash scripts/validate_h800.sh")
    if cfg.cuda_schedule == "fused" and getattr(ext, "fused_pipeline_version", None) != 1:
        raise RuntimeError("Rebuild vc-attention 0.5.0 for the fused pipeline and baseline: bash scripts/validate_h800.sh")
    if cfg.cuda_schedule == "fused" and getattr(ext, "fused_d128_version", None) != 1:
        raise RuntimeError("Rebuild vc-attention 0.6.0 for D128 specialization: bash scripts/validate_h800.sh")
    if cfg.cuda_schedule == "fused" and getattr(ext, "fused_interleaved_version", None) != 1:
        raise RuntimeError("Rebuild vc-attention 0.7.0 for the interleaved schedule: bash scripts/validate_h800.sh")
    if cache is None:
        cache = LayoutCache(cfg)
        request_id = "one-shot" if request_id is None else request_id
    elif cache.config != cfg:
        raise ValueError("cache.config must match config")
    pi, active, refreshed = cache.layout(
        v.float(), step=step, total_steps=total_steps, request_id=request_id, layer_id=layer_id
    )
    if pi is not None:
        k = k.gather(-2, pi.unsqueeze(-1).expand_as(k))
        v = v.gather(-2, pi.unsqueeze(-1).expand_as(v))
    mean_type = {"float32": 0, "float16": 1, "bfloat16": 2}[cfg.mean_dtype]
    variant = _fused_variant(q.shape[-1], v.shape[-1], cfg) if cfg.cuda_schedule == "fused" else None
    launch = ext.forward
    if variant is not None:
        launch = {"baseline": ext.forward_fused_baseline, "pipeline": ext.forward_fused,
                  "d128": ext.forward_fused_d128, "interleaved": ext.forward_fused_interleaved}[variant]
    output = launch(q, k, v, cfg.q_block, cfg.k_quant_block, cfg.kv_block,
                         active, cfg.expcast, mean_type, scale, cfg.k_smooth, cfg.qk_hadamard,
                         cfg.cuda_schedule == "batched")
    return output, {
        "backend": "cuda_fp8",
        "smoothing_active": active,
        "layout_refreshed": refreshed,
        "expcast": cfg.expcast,
        "k_smooth": cfg.k_smooth,
        "qk_hadamard": cfg.qk_hadamard,
        "numerical_contract_version": 2,
        "tiles": math.ceil(q.shape[-2] / cfg.q_block) * math.ceil(k.shape[-2] / cfg.kv_block),
        "native_fused_attention": cfg.cuda_schedule == "fused",
        "preprocessing_fused": False,
        "native_fp8_matmul": True,
        "cuda_schedule": cfg.cuda_schedule,
        "cuda_fused_pipeline": cfg.cuda_schedule == "fused" and cfg.cuda_fused_pipeline,
        "cuda_fused_variant": variant,
    }


def preflight(device="cuda:0", *, schedule="batched"):
    """Fail loudly unless both native FP8 GEMMs and ExpCast work on Hopper."""
    from .core import Config, attention, expcast_codes

    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] != 9:
        raise RuntimeError("Preflight requires an H800/H100/H200-class Hopper GPU")
    ext = _extension()
    x = torch.tensor([0, -1, -1.6, -8, -14, -100, -float("inf")], device=device) / math.log2(math.e)
    torch.testing.assert_close(ext.expcast_codes(x), expcast_codes(x), rtol=0, atol=0)
    # Exactly representable FP8 data gives a useful independent GEMM oracle.
    q = torch.zeros(1, 1, 19, 16, device=device)
    k = torch.zeros(1, 1, 35, 16, device=device)
    v = torch.full((1, 1, 35, 16), 3.0, device=device)
    for cast in (False, True):
        actual, _ = attention(q, k, v, Config(backend="cuda_fp8", expcast=cast, cuda_schedule=schedule))
        torch.testing.assert_close(actual, torch.full_like(actual, 3), rtol=2e-5, atol=2e-5)
    if schedule == "fused":
        q = torch.zeros(1, 1, 65, 128, device=device)
        k = torch.zeros(1, 1, 129, 128, device=device)
        v = torch.full_like(k, 3)
        for cast in (False, True):
            actual, stats = attention(q, k, v, Config(backend="cuda_fp8", expcast=cast, cuda_schedule="fused"))
            assert stats["cuda_fused_variant"] == ("interleaved" if cast else "d128")
            torch.testing.assert_close(actual, torch.full_like(actual, 3), rtol=2e-5, atol=2e-5)
    torch.cuda.synchronize(device)
    info = dict(ext.build_info())
    if schedule == "fused":
        info["matmul"] = info["fused_matmul"]
    return {"device": str(device), "name": torch.cuda.get_device_name(device),
            "preflight": "passed", **info, "cuda_schedule": schedule,
            "native_fused_attention": schedule == "fused"}


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--schedule", choices=("batched", "fused", "serial"), default="batched")
    args = parser.parse_args()
    print(json.dumps(preflight(args.device, schedule=args.schedule), indent=2))
