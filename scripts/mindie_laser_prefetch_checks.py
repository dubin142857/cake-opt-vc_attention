"""Untimed all-output regression gates for the AIV score-prefetch experiment."""
from __future__ import annotations


# All cases use dense attention. The native adapter pads Q/K to 256 and D to
# 128, so even short inputs exercise several four-row Vector batches.
PREFETCH_CASES = (
    dict(name="aligned_key_block", batch=1, heads=2, queries=32, keys=128, dim=64, pattern="random"),
    dict(name="row_batches_and_key_tail", batch=1, heads=2, queries=17, keys=129, dim=64, pattern="random"),
    dict(name="full_key_section", batch=1, heads=2, queries=257, keys=4096, dim=128, pattern="random"),
    dict(name="section_tail_and_batch_offsets", batch=2, heads=1, queries=257, keys=4097, dim=64, pattern="outliers"),
    dict(name="multiple_key_sections", batch=1, heads=2, queries=513, keys=8193, dim=128, pattern="random"),
    dict(name="zero_scores", batch=1, heads=2, queries=33, keys=129, dim=64, pattern="zero_qk"),
    dict(name="constant_inputs", batch=1, heads=2, queries=65, keys=257, dim=128, pattern="constant"),
)


def _make_inputs(case, dtype, device, revision):
    """Generate finite inputs deterministically; never construct a score matrix."""
    import torch

    generator = torch.Generator().manual_seed(831 + revision * 101)
    b, h, qn, kn, d = (case[key] for key in ("batch", "heads", "queries", "keys", "dim"))
    inputs = [torch.randn(shape, generator=generator) for shape in
              ((b, h, qn, d), (b, h, kn, d), (b, h, kn, d))]
    q, k, v = inputs
    if case["pattern"] == "zero_qk":
        q.zero_()
        k.zero_()
    elif case["pattern"] == "constant":
        q.fill_(0.25 if revision == 0 else -0.375)
        k.fill_(-0.5 if revision == 0 else 0.75)
        channels = torch.linspace(-1.0, 1.0, d)
        v.copy_(channels if revision == 0 else -channels * 0.5)
    elif case["pattern"] == "outliers":
        # Differing score ranges next to row, quantization and section edges
        # expose stale ping-pong contents without creating FP16 overflow.
        for tensor in (q, k):
            for index in (3, 4, 127, 128, 255, 256, 4095, 4096):
                if index < tensor.shape[2]:
                    tensor[:, :, index, (index + revision) % d] = 8.0 if revision == 0 else -6.0
    return tuple(tensor.to(dtype).to(device) for tensor in inputs)


def validate_score_prefetch(adapters, device, report, save):
    """Compare every padded native FP32 output, including repeated buffer use.

    Each pair receives the identical payload object. Two orders run before and
    after an in-place Q/K/V/scale refresh, with no host synchronization between
    that refresh and the first launch. This is an implementation regression
    check; existing decoded-reference gates check the shared algorithm.
    """
    import torch
    from profile_mindie_laser_quant import compare_all_rows, FIELDS

    if set(adapters) != {"baseline", "reuse"}:
        raise ValueError("Score prefetch requires baseline and reuse adapters")
    rows = report.setdefault("score_prefetch_validation", [])
    report["score_prefetch_validation_scope"] = {
        "output": "All padded native FP32 output rows, heads and features; exact match required",
        "timed": False,
        "same_payload_object": True,
        "cases": [dict(case) for case in PREFETCH_CASES],
        "limit": "Checks implementation agreement, not dense/model accuracy or speed",
    }
    save()
    for dtype in (torch.float16, torch.bfloat16):
        for case in PREFETCH_CASES:
            print(f"Score prefetch regression: {dtype} Q{case['queries']} K{case['keys']} "
                  f"D{case['dim']} {case['pattern']} ({case['name']})", flush=True)
            inputs = _make_inputs(case, dtype, device, revision=0)
            payload = adapters["baseline"].prepare(*inputs)
            pointers = {name: getattr(payload, name).data_ptr() for name in FIELDS}
            for iteration in range(4):
                order = ("baseline", "reuse") if iteration % 2 == 0 else ("reuse", "baseline")
                row = dict(case, dtype=str(dtype), iteration=iteration, order=list(order),
                           payload_update=("initial", "unchanged", "in_place_refresh", "unchanged")[iteration],
                           status="running")
                rows.append(row)
                save()
                try:
                    if iteration == 2:
                        changed = _make_inputs(case, dtype, device, revision=1)
                        fresh = adapters["baseline"].prepare(*changed)
                        for name in FIELDS:
                            getattr(payload, name).copy_(getattr(fresh, name))
                        del fresh, changed
                    outputs = {name: adapters[name].run_native(payload) for name in order}
                    row.update(compare_all_rows(outputs["reuse"], outputs["baseline"]))
                    del outputs
                    row["pointers_unchanged"] = all(
                        getattr(payload, name).data_ptr() == pointers[name] for name in FIELDS)
                    if not row["pointers_unchanged"] or not row["exact_match"]:
                        raise AssertionError("Score prefetch changed padded native outputs or payload storage")
                    row["status"] = "passed"
                    save()
                except Exception as error:
                    row.update(status="failed", error=f"{type(error).__name__}: {error}")
                    save()
                    raise
            del payload, inputs
