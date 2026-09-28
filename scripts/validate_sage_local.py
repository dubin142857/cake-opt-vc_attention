"""Hopper smoke/accuracy checks for the independently compiled SageAttention.

The random-input RMSE gate detects broken execution; it is not model quality validation.
If upstream sageattention is installed, import both and report pairwise errors.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import torch


def relative_rmse(actual, expected):
    diff = (actual.float() - expected.float()).square().mean().sqrt()
    return (diff / expected.float().square().mean().sqrt().clamp_min(1e-12)).item()


@torch.no_grad()
def validate(device="cuda:0"):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("This validation requires a Hopper GPU (H100/H800/H200)")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Import upstream first to detect operator namespace collisions on local import.
    original = None
    if importlib.util.find_spec("sageattention") is not None:
        import sageattention as original
    import sageattention_local as local
    info = local.local_build_info()
    report = {"environment": {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(device)}, "local": info,
              "original_package": original.__file__ if original else None, "cases": [],
              "scope": "Small dense, noncausal HND forward tests; synthetic RMSE <=0.10 is a regression gate, not model quality."}
    for dtype in (torch.float16, torch.bfloat16):
        for dim in (64, 128):
            for nq, nk in ((1, 127), (65, 129), (257, 385), (1024, 1024)):
                torch.manual_seed(220)
                q, k, v = (torch.randn(1, 2, n, dim, device=device, dtype=dtype) for n in (nq, nk, nk))
                expected = torch.softmax((q.float() @ k.float().transpose(-2, -1)) * dim**-0.5, dim=-1) @ v.float()
                actual = local.sageattn(q, k, v, tensor_layout="HND", is_causal=False)
                torch.cuda.synchronize(device)
                if actual.shape != q.shape or actual.dtype != dtype or not torch.isfinite(actual).all().item():
                    raise AssertionError(f"Invalid local output for {dtype}, D={dim}, Q/K={nq}/{nk}")
                error = relative_rmse(actual, expected)
                if error > 0.10:
                    raise AssertionError(f"Local random-input regression gate failed: relative RMSE={error:.6f}")
                row = {"dtype": str(dtype), "dim": dim, "queries": nq, "keys": nk, "local_vs_fp32_rel_rmse": error}
                if original is not None:
                    control = original.sageattn(q, k, v, tensor_layout="HND", is_causal=False)
                    if not torch.isfinite(control).all().item():
                        raise AssertionError("Installed upstream Sage returned NaN/Inf; check that installation")
                    row.update(original_vs_fp32_rel_rmse=relative_rmse(control, expected),
                               local_vs_original_rel_rmse=relative_rmse(actual, control),
                               local_vs_original_max_abs=(actual.float() - control.float()).abs().max().item(),
                               exact_match=torch.equal(actual, control))
                report["cases"].append(row)
                print(json.dumps(row), flush=True)
    # Known answer also detects an output allocation returned without a kernel launch.
    for dtype in (torch.float16, torch.bfloat16):
        q = torch.zeros(1, 2, 65, 128, device=device, dtype=dtype)
        k = torch.zeros(1, 2, 129, 128, device=device, dtype=dtype)
        v = torch.full_like(k, 1.5)
        actual = local.sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        torch.testing.assert_close(actual, torch.full_like(q, 1.5), rtol=0, atol=0.02)
    report["known_answer_cases"] = 2
    report["status"] = "passed"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/sage_local_validation.json")
    args = parser.parse_args()
    report = validate(args.device)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Local Sage validation passed: 16 random cases + 2 known answers; saved to {path}")
