"""NPU acceptance against independent decoded operands; never relaxed silently."""
import argparse
import json
from pathlib import Path
import traceback
import torch


def error(actual, expected):
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("Nonfinite output/reference")
    diff = a - b
    return {"relative_rmse": float(diff.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12)),
            "max_abs": float(diff.abs().max())}


def to_cpu(payload):
    return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in payload.items()}


@torch.no_grad()
def validate(device, save):
    from vc_attention_ascend import prepare, run_prepared, attention, load_native
    from vc_attention_ascend.reference import payload_reference, dense_reference
    load_native()
    results = []
    generator = torch.Generator().manual_seed(91023)
    # Covers both AIV halves, whole/tail query tiles and several online K updates.
    shapes = [(1, 1), (17, 65), (31, 127), (32, 128), (33, 129), (65, 257), (97, 513)]
    for dtype in (torch.float16, torch.bfloat16):
        for d in (64, 128):
            for nq, nk in shapes:
                q, k, v = [torch.randn((1, 2, n, d), generator=generator).to(dtype) for n in (nq, nk, nk)]
                if nq == 33:
                    v[..., 0] = 0.0
                    v[..., 1] = 2.0
                cpu_inputs = (q, k, v)
                q, k, v = [x.to(device) for x in cpu_inputs]
                payload = prepare(q, k, v, layout="HND")
                cpu_payload = to_cpu(payload)
                for cast in (False, True):
                    case = {"dtype": str(dtype), "dim": d, "queries": nq, "keys": nk, "expcast": cast}
                    print("Checking 910B:", json.dumps(case), flush=True)
                    save(results, case)
                    native = run_prepared(payload, expcast=cast)
                    ref = payload_reference(cpu_payload, expcast=cast).to(dtype)
                    metric = error(native, ref)
                    if metric["relative_rmse"] > .01:
                        raise AssertionError(f"Decoded-reference implementation RMSE exceeds 1%: {case}, {metric}")
                    if nq == 33 and cast:
                        torch.testing.assert_close(native[..., 0].cpu(), torch.zeros_like(ref[..., 0]), rtol=0, atol=0)
                        torch.testing.assert_close(native[..., 1].cpu(), torch.full_like(ref[..., 1], 2), rtol=.004, atol=.003)
                    dense = dense_reference(*cpu_inputs)
                    row = {**case, "decoded_reference": metric, "dense_fp32": error(native, dense)}
                    results.append(row)
                    save(results, None)
    # Uniform attention, zero Q/K/V and independently chosen B/H dimensions.
    for value in (0.0, 2.0):
        q = torch.zeros((2, 33, 3, 64), device=device, dtype=torch.float16)
        k = torch.zeros((2, 129, 3, 64), device=device, dtype=torch.float16)
        v = torch.full_like(k, value)
        for cast in (False, True):
            out = attention(q, k, v, expcast=cast)
            torch.testing.assert_close(out, torch.full_like(q, value), rtol=.004, atol=.003)
            results.append({"uniform_value": value, "expcast": cast, "status": "passed"})
    # Nondefault stream checks host queue ordering and temporary-buffer lifetimes.
    stream = torch.npu.Stream(device=device)
    with torch.npu.stream(stream):
        q = torch.randn((1, 33, 2, 64), device=device, dtype=torch.float16)
        k = torch.randn((1, 129, 2, 64), device=device, dtype=torch.float16)
        v = torch.full_like(k, 1.0)
        out = attention(q * 2, k, v)
        stream.synchronize()
        torch.testing.assert_close(out.cpu(), torch.ones_like(out.cpu()), rtol=.004, atol=.003)
    results.append({"nondefault_stream": "passed"})
    save(results, None)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="results/ascend910b_validation.json")
    args = parser.parse_args()
    import torch_npu  # noqa: F401
    torch.npu.set_device(args.device)
    report = {"status": "running", "implementation_gate": .01, "model_quality_claim": False,
              "device": torch.npu.get_device_name(args.device)}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save(results, current):
        report.update(results=results, current_case=current)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        validate(f"npu:{args.device}", save)
        report["status"] = "passed"
        save(report["results"], None)
        print("910B validation passed:", path, flush=True)
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
        save(report.get("results", []), report.get("current_case"))
        raise


if __name__ == "__main__":
    main()
