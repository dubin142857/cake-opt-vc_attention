"""Observable native ExpCast checks with an independent structured CPU oracle.

These cases deliberately separate the changed probability algorithm from plain
softmax. A dense-reference tolerance alone would accept an unchanged kernel.
The inputs make every native 128-key block contain a zero maximum, so Laser's
unchanged online-history factor rounds to one and no section-boundary model is
needed. Only valid keys appear in the oracle: padding must contribute zero.
"""
import math

if __package__:
    from .mindie_laser_expcast_reference import expcast_probability, CONTRACT
else:
    from mindie_laser_expcast_reference import expcast_probability, CONTRACT


# K=4097 reaches the native K4096 section plus a tail, and the probability
# helper's second 8192-element chunk (4 query rows x 4096 keys).
SHAPES = ((17, 129), (256, 256), (33, 385), (65, 513), (1, 1), (17, 4097))


def structured_case(dtype, dim, queries, keys, torch):
    """Exact single-product QK and distinct V columns, with two constant channels."""
    if dim not in (64, 128) or min(queries, keys) < 1:
        raise ValueError("Structured Laser case requires positive lengths and D64/128")
    q = torch.zeros((1, 2, queries, dim), dtype=dtype)
    k = torch.zeros((1, 2, keys, dim), dtype=dtype)
    amplitudes = torch.tensor([1.0, 2.0, 0.5], dtype=dtype)
    q[..., 0] = amplitudes[torch.arange(queries) % 3]
    k[..., 0] = -((torch.arange(keys) % 128).float() / 2).to(dtype)
    v = torch.zeros_like(k)
    v[:, :, torch.arange(keys), torch.arange(keys) % dim] = 1
    v[..., 0] = 0
    v[..., 1] = 2
    return q, k, v


def structured_reference(q, k, v, torch, *, expcast=True):
    """Laser QK/scale/shift rounding followed by decoded-P FP32 sum and PV.

    This is restricted to the deliberate single-product cases, not a purported
    emulator of the full Laser Cube kernel or its reduction rounding.
    """
    if any(x.device.type != "cpu" for x in (q, k, v)):
        raise ValueError("Structured Laser reference is CPU-only")
    if torch.count_nonzero(q[..., 1:]) or torch.count_nonzero(k[..., 1:]):
        raise ValueError("Oracle requires exact single-feature QK inputs")
    q16, k16, v16 = (x.to(torch.float16) for x in (q, k, v))
    scores = (q16[..., :1].float() * k16[..., :1].transpose(-1, -2).float()).half()
    scale = torch.tensor(1 / math.sqrt(q.shape[-1]), dtype=torch.float16)
    scores = (scores.float() * scale.float()).half()
    for start in range(0, k.shape[-2], 128):
        block_max = scores[..., start:start + 128].amax(-1)
        if not torch.equal(block_max, torch.zeros_like(block_max)):
            raise ValueError("Every 128-key block must have a zero maximum")
    shifted = (scores.float() - scores.amax(-1, keepdim=True).float()).half()
    p = expcast_probability(shifted).float() if expcast else shifted.float().exp()
    mass = p.sum(-1, keepdim=True)
    if not torch.isfinite(mass).all() or not torch.all(mass > 0):
        raise AssertionError("Structured probability denominator is invalid")
    # Existing wrapper always passes through FP16 before the requested dtype.
    return ((p @ v16.float()) / mass).half().to(q.dtype)


def error_metrics(actual, expected, torch, limit=None):
    if actual.shape != expected.shape:
        raise AssertionError(f"Output shape differs: {actual.shape} vs {expected.shape}")
    actual, expected = actual.float(), expected.float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    difference = actual - expected
    if finite:
        rmse = difference.square().mean().sqrt()
        scale = expected.square().mean().sqrt().clamp_min(1e-30)
        relative = float(rmse / scale)
        max_abs = float(difference.abs().max())
    else:
        relative, max_abs = float("inf"), float("inf")
    result = {"relative_rmse": relative, "max_abs": max_abs, "finite": finite}
    if limit is not None:
        result.update(limit=limit, passed=finite and relative <= limit)
    return result


def _gate(row, name, actual, expected, torch, limit, save):
    row[name] = error_metrics(actual, expected, torch, limit)
    save()
    if not row[name]["passed"]:
        raise AssertionError(f"Laser ExpCast {name} failed {100 * limit:g}% implementation gate: {row[name]}")


def validate_small(adapter, device, report, save, torch):
    """Callback accepted by benchmark_mindie_laser_local.run_worker."""
    report["small_accuracy"] = []
    report["small_accuracy_contract"] = {
        "probability": CONTRACT,
        "inputs": "Q first feature repeats [1,2,0.5]; K first feature -(key%128)/2; V one-hot key%dim",
        "constant_channels": "V[...,0]=0 and V[...,1]=2; checked separately",
        "decoded_gate_scope": "All output rows and channels 2 onward; FP16 QK/scale/shift, decoded P, FP32 sum/PV, FP16 output round-trip",
        "dense_error": "Diagnostic only; different probability algorithm, no quality threshold",
        "implementation_limit": 0.01,
        "full_prepared_limit": 0.001,
    }
    save()
    with torch.no_grad():
        for dtype in (torch.float16, torch.bfloat16):
            for dim in (64, 128):
                for nq, nk in SHAPES:
                    print(f"Validate Laser ExpCast: {dtype}, Q={nq}, K={nk}, D={dim}", flush=True)
                    original = structured_case(dtype, dim, nq, nk, torch)
                    expected = structured_reference(*original, torch)
                    ordinary = structured_reference(*original, torch, expcast=False)
                    q, k, v = original
                    dense_scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(dim)
                    dense = dense_scores.softmax(-1) @ v.float()
                    row = {"dtype": str(dtype), "queries": nq, "keys": nk, "dim": dim,
                           "status": "running",
                           "ordinary_softmax_vs_decoded_reference": error_metrics(ordinary[..., 2:], expected[..., 2:], torch)}
                    report["small_accuracy"].append(row)
                    # For K=1 the algorithms intentionally coincide. Every other
                    # case must visibly reject an unchanged softmax implementation.
                    if nk > 1 and row["ordinary_softmax_vs_decoded_reference"]["relative_rmse"] <= .01:
                        save()
                        raise AssertionError("Structured case cannot distinguish unchanged softmax from ExpCast")
                    inputs = [x.to(device) for x in original]
                    payload = adapter.prepare(*inputs)
                    row["payload"] = adapter.describe_payload(payload)
                    prepared = adapter.run_prepared(payload).cpu()
                    full = adapter(*inputs).cpu()
                    _gate(row, "full_vs_prepared", full, prepared, torch, .001, save)
                    _gate(row, "decoded_reference_all_rows_nonconstant_channels",
                          full[..., 2:], expected[..., 2:], torch, .01, save)
                    row["dense_fp32_all_rows_all_keys"] = error_metrics(full, dense, torch)
                    row["dense_fp32_nonconstant_channels"] = error_metrics(full[..., 2:], dense[..., 2:], torch)
                    try:
                        torch.testing.assert_close(full[..., 0], torch.zeros_like(full[..., 0]), rtol=0, atol=0)
                        torch.testing.assert_close(full[..., 1], torch.full_like(full[..., 1], 2), rtol=.004, atol=.003)
                    except AssertionError:
                        row["constant_value_padding_check"] = "failed"
                        save()
                        raise
                    row["constant_value_padding_check"] = "passed"
                    row["status"] = "passed"
                    save()
    return report["small_accuracy"]
