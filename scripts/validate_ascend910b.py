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


def reuse_errors(actual, expected):
    """Keep sparse row/head corruption visible despite very different V scales."""
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    metric = error(a, b)
    per_row = ((a - b).square().mean(-1).sqrt()
               / b.square().mean(-1).sqrt().clamp_min(1e-12))
    flat = int(per_row.argmax())
    heads, queries = a.shape[1:3]
    batch, head, query = flat // (heads * queries), flat // queries % heads, flat % queries
    metric["per_head"] = [{"batch": bi, "head": hi, **error(a[bi, hi], b[bi, hi])}
                          for bi in range(a.shape[0]) for hi in range(heads)]
    metric["worst_row"] = {
        "batch": batch, "head": head, "query": query, "query_block": query // 32,
        "aiv_half": query % 32 // 16,
        "task": (batch * heads + head) * ((queries + 31) // 32) + query // 32,
        **error(a[batch, head, query], b[batch, head, query]),
    }
    return metric


def require_reuse_accuracy(metric, label):
    if (metric["relative_rmse"] > .01
            or any(head["relative_rmse"] > .01 for head in metric["per_head"])
            or metric["worst_row"]["relative_rmse"] > .01):
        raise AssertionError(f"{label} RMSE exceeds 1% (global, per-head or per-row): {metric}")


@torch.no_grad()
def check_reuse_payload(forward, payload, expcast, case, results, save, *, repeats=5, constant_v=None):
    """Validate every repeat of one immutable payload, including raw FP32 bits.

    ``forward(payload, expcast)`` returns native padded HND FP32 output. Tests
    may inject a CPU fake; doing so tests this checker, never the NPU kernel.
    """
    from vc_attention_ascend.reference import payload_reference
    if repeats < 2:
        raise ValueError("Core reuse validation requires at least two repeats")
    case = {**case, "expcast": bool(expcast), "repeats": repeats}
    save(results, {**case, "phase": "decoded_reference"})
    reference = payload_reference(to_cpu(payload), expcast=expcast)
    cast_reference = reference.to(payload["dtype"])
    baseline = None
    for repeat in range(1, repeats + 1):
        row = {**case, "repeat": repeat, "status": "running"}
        save(results, row)
        try:
            out = forward(payload, expcast)
            if out.dtype != torch.float32:
                raise AssertionError(f"Expected native FP32 output, got {out.dtype}")
            out = out[:, :, :payload["queries"]]
            if out.shape != reference.shape:
                raise AssertionError(f"Native output shape {out.shape} != reference {reference.shape}")
            # Cast on the NPU exactly as run_prepared does; compare raw bits on
            # CPU after synchronization, excluding uninitialized padded rows.
            cast = out.to(payload["dtype"]).cpu()
            raw = out.detach().cpu().contiguous()
            row["raw_fp32_vs_decoded"] = error(raw, reference)
            row["decoded_reference"] = reuse_errors(cast, cast_reference)
            if baseline is None:
                baseline = raw.clone()
            changed = raw.view(torch.int32) != baseline.view(torch.int32)
            row["determinism"] = {
                "baseline_repeat": 1, "bitwise_equal": not bool(changed.any()),
                "different_elements": int(changed.sum()), **error(raw, baseline),
                "first_changed_indices": changed.nonzero()[:8].tolist(),
            }
            if constant_v is not None:
                constant = constant_v.float().cpu().expand_as(reference)
                row["constant_v_invariant"] = reuse_errors(raw, constant)
            require_reuse_accuracy(row["decoded_reference"], "Decoded-reference implementation")
            if constant_v is not None:
                require_reuse_accuracy(row["constant_v_invariant"], "Constant-V invariant")
            if not row["determinism"]["bitwise_equal"]:
                raise AssertionError(f"Fixed-payload native FP32 output is not bitwise deterministic: {row['determinism']}")
            row["status"] = "passed"
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            results.append(row)
            save(results, row)
            raise
        results.append(row)
        save(results, None)


def reuse_inputs(heads, queries, keys, dim, dtype, generator):
    """Random scores, constant V along K, and sharply differing head scales."""
    q, k = [torch.randn((1, heads, n, dim), generator=generator).to(dtype)
            for n in (queries, keys)]
    head_gain = torch.pow(2.0, torch.linspace(-5, 5, heads)).reshape(1, heads, 1, 1)
    channel = torch.arange(dim)
    channel_gain = torch.pow(2.0, (channel % 5 - 2).float()) * (1 - 2 * (channel % 2))
    v = (head_gain * channel_gain).expand(1, heads, keys, dim).to(dtype).contiguous()
    return q, k, v


@torch.no_grad()
def validate_core_reuse(device, save, results, native, prepare):
    # 16 cases, five launches each. Q1025/H2 reproduces the observed failure;
    # H65/Q33 gives 130 tasks and forces multiple tasks per core on A2/A3.
    # Every task transition across heads loads substantially different V scales.
    generator = torch.Generator().manual_seed(91024)

    def forward(p, cast):
        return native.forward(p["q"], p["k"], p["v"], p["q_scale"], p["k_scale"],
                              p["v_scale"], p["lut"], p["queries"], p["keys"],
                              p["scale"], bool(cast))

    for dtype in (torch.float16, torch.bfloat16):
        for dim in (64, 128):
            for heads, queries in ((2, 1025), (65, 33)):
                q, k, v = reuse_inputs(heads, queries, 129, dim, dtype, generator)
                constant_v = v[:, :, :1].clone()
                payload = prepare(*(x.to(device) for x in (q, k, v)), layout="HND")
                for cast in (False, True):
                    case = {"suite": "core_reuse", "dtype": str(dtype), "dim": dim,
                            "batch": 1, "heads": heads, "queries": queries, "keys": 129,
                            "tasks": heads * ((queries + 31) // 32), "expcast": cast,
                            "v_pattern": "constant_keys_head_scales_2^-5_to_2^5"}
                    print("Checking 910B core reuse:", json.dumps(case), flush=True)
                    check_reuse_payload(forward, payload, cast, case, results, save,
                                        constant_v=constant_v)


@torch.no_grad()
def validate(device, save):
    from vc_attention_ascend import prepare, run_prepared, attention, load_native
    from vc_attention_ascend.reference import payload_reference, dense_reference
    native_ops = load_native()
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
    validate_core_reuse(device, save, results, native_ops, prepare)
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
    import vc_attention_ascend as vc
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.npu.set_device(args.device)
    report = {"status": "running", "implementation_gate": .01, "model_quality_claim": False,
              "device": torch.npu.get_device_name(args.device), "vc_version": vc.__version__}
    print("910B validation:", json.dumps(report), flush=True)
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
