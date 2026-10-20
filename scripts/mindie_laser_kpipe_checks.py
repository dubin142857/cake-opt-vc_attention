"""Untimed correctness gates for QK K GM-to-L1 one-tile lookahead."""
from __future__ import annotations

from copy import copy


K_PREFETCH_CASES = (
    dict(name="short_key_tail", batch=1, heads=2, queries=17, keys=127, dim=64),
    dict(name="second_valid_key_tile", batch=1, heads=3, queries=257, keys=129, dim=128),
    dict(name="multiple_tiles_batch_tail", batch=2, heads=3, queries=65, keys=257, dim=64),
    dict(name="section_last_key_tail", batch=1, heads=2, queries=257, keys=4095, dim=64),
    dict(name="section_exact", batch=1, heads=3, queries=513, keys=4096, dim=128),
    dict(name="section_plus_tail", batch=2, heads=3, queries=257, keys=4097, dim=64),
    dict(name="multiple_sections", batch=1, heads=2, queries=513, keys=8193, dim=128),
    # 68 Q256 groups exceed 64 Cube cores; some cores must visit multiple
    # query groups. Kpad1024 gives eight K128 tiles in each group.
    dict(name="descriptor_head_batch_switch", batch=2, heads=17, queries=257, keys=769, dim=64),
)
FIELDS = ("q", "k", "v", "q_scale", "k_scale")


def _make_inputs(case, dtype, device, revision):
    """Distinct adjacent K128 blocks and V markers expose a wrong lookahead tile."""
    import torch

    b, h, qn, kn, d = (case[key] for key in ("batch", "heads", "queries", "keys", "dim"))
    generator = torch.Generator().manual_seed(833 + revision * 97 + kn)
    q, k, v = [torch.randn(shape, generator=generator) for shape in
               ((b, h, qn, d), (b, h, kn, d), (b, h, kn, d))]
    tile = torch.arange(kn) // 128
    amplitudes = torch.tensor((.25, .5, 1., 2., 1.5, .75, 1.25, 1.75))
    # The sequence of scale ranges crosses descriptor, ping/pong and K32
    # boundaries. No two neighboring full K128 blocks have the same amplitude.
    k.mul_(amplitudes[(tile + revision) % amplitudes.numel()][None, None, :, None])
    q[..., 0] = 1 + ((torch.arange(qn) // 128 + revision) % 4).float() * .25
    k[..., 0] = ((tile * 5 + revision * 3) % 17 - 8).float() * .125
    v[..., 0] = ((tile * 3 + revision * 2) % 13 - 6).float() * .125
    # Mark batch/head independently, including the descriptor-switch case.
    phase = torch.arange(b * h).reshape(b, h, 1).float()
    k[..., 1] += phase * .03125
    v[..., 1] += phase * .015625
    return tuple(tensor.to(dtype).to(device) for tensor in (q, k, v))


def _require_native_output(output, payload):
    import torch

    if (not isinstance(output, torch.Tensor) or output.shape != payload.q.shape
            or output.dtype != torch.float32 or output.device != payload.q.device):
        raise AssertionError("K lookahead requires the entire padded native FP32 output")


def validate_k_prefetch(adapters, device, report, save):
    """Exact same-payload checks, including two calls before/after in-place refresh.

    All eight native calls in a case are queued before the first CPU comparison.
    The stream orders writes, QK and PV; no host synchronize is inserted in that
    launch loop. Both algorithms use the same native tensors and scale pointers.
    """
    import torch
    from profile_mindie_laser_quant import compare_all_rows

    if set(adapters) != {"baseline", "reuse"}:
        raise ValueError("K lookahead requires baseline and reuse adapters")
    rows = report.setdefault("k_prefetch_validation", [])
    report["k_prefetch_validation_scope"] = {
        "output": "ALL padded native FP32 rows, heads and features; exact match",
        "same_payload_object": True,
        "updates": "Isolated Q/K/V codes and Q/K scales rewritten in place; unchanged pointers",
        "timed": False,
        "queued_native_calls_before_cpu_comparison": 8,
        "cases": [dict(case) for case in K_PREFETCH_CASES],
        "parity": "Native K pads to 256, so dense descriptor K128 counts are even; "
                  "odd slot parity is covered by host event-model tests, not this public API",
        "limits": "Output checks do not expose hardware event state or prove speedup",
    }
    save()
    for dtype in (torch.float16, torch.bfloat16):
        for case in K_PREFETCH_CASES:
            print(f"K lookahead validation: {dtype} B{case['batch']} H{case['heads']} "
                  f"Q{case['queries']} K{case['keys']} D{case['dim']} ({case['name']})", flush=True)
            inputs = _make_inputs(case, dtype, device, 0)
            original = adapters["baseline"].prepare(*inputs)
            changed_inputs = _make_inputs(case, dtype, device, 1)
            changed = adapters["baseline"].prepare(*changed_inputs)
            fixed = copy(original)
            for field in FIELDS:
                setattr(fixed, field, getattr(original, field).clone())
            pointers = {field: getattr(fixed, field).data_ptr() for field in FIELDS}
            row = dict(case, dtype=str(dtype), padded_shape=list(fixed.q.shape),
                       padded_key_tokens=fixed.k.shape[2],
                       padded_k128_blocks=fixed.k.shape[2] // 128,
                       rounds=[], status="running")
            rows.append(row)
            save()
            pending = []
            try:
                for iteration in range(4):
                    if iteration == 2:
                        for field in FIELDS:
                            getattr(fixed, field).copy_(getattr(changed, field))
                    order = ("baseline", "reuse") if iteration % 2 == 0 else ("reuse", "baseline")
                    outputs = {name: adapters[name].run_native(fixed) for name in order}
                    for output in outputs.values():
                        _require_native_output(output, fixed)
                    pending.append((iteration, order, outputs))
                # CPU comparison may synchronize, only after the launch loop.
                for iteration, order, outputs in pending:
                    error = compare_all_rows(outputs["reuse"], outputs["baseline"])
                    unchanged = all(getattr(fixed, field).data_ptr() == pointer
                                    for field, pointer in pointers.items())
                    baseline_change = None if iteration == 0 else compare_all_rows(
                        outputs["baseline"], pending[iteration - 1][2]["baseline"])
                    result = dict(iteration=iteration, order=list(order),
                        payload_update=("initial", "unchanged", "in_place_refresh", "unchanged")[iteration],
                        pointers_unchanged=unchanged, baseline_change=baseline_change, **error)
                    row["rounds"].append(result)
                    save()
                    if not unchanged or not error["exact_match"]:
                        raise AssertionError("K lookahead changed padded native outputs or payload storage")
                    if baseline_change is not None:
                        expected_same = iteration != 2
                        if baseline_change["exact_match"] != expected_same:
                            raise AssertionError("K lookahead baseline reuse/refresh check is ineffective "
                                                 "or nondeterministic")
                row["status"] = "passed"
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                # Restore only our isolated buffers, also after a failure.
                for field in FIELDS:
                    getattr(fixed, field).copy_(getattr(original, field))
                save()
            del pending, original, changed, fixed, inputs, changed_inputs
