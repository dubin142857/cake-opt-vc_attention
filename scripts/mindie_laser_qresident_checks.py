"""Untimed exact-output gates for keeping one Q256 descriptor in L0A."""
from __future__ import annotations

from copy import copy


Q_RESIDENT_CASES = (
    dict(name="q_pair_and_key_tail", batch=1, heads=2, queries=257, keys=257, dim=64),
    dict(name="multiple_q_pairs_before_key_boundary", batch=2, heads=3, queries=513, keys=4095, dim=128),
    dict(name="aligned_key_section", batch=1, heads=3, queries=257, keys=4096, dim=128),
    dict(name="q_pairs_across_pv_sections", batch=2, heads=3, queries=513, keys=4097, dim=64),
    dict(name="multiple_q_and_k_sections", batch=1, heads=2, queries=769, keys=8193, dim=128),
    # 68 Q256 groups exceed 64 Cube cores, exercising multiple descriptor
    # visits on at least some cores on the intended A3 target.
    dict(name="descriptor_head_batch_switch", batch=2, heads=17, queries=257, keys=769, dim=64),
)
FIELDS = ("q", "k", "v", "q_scale", "k_scale")


def _make_inputs(case, dtype, device, revision):
    """Different Q0/Q1, descriptor/head markers and changing key sections."""
    import torch

    b, h, qn, kn, d = (case[key] for key in ("batch", "heads", "queries", "keys", "dim"))
    generator = torch.Generator().manual_seed(839 + 101 * revision + qn + kn)
    q, k, v = [torch.randn(shape, generator=generator) for shape in
               ((b, h, qn, d), (b, h, kn, d), (b, h, kn, d))]
    qb, kb = torch.arange(qn) // 128, torch.arange(kn) // 128
    amplitude = torch.tensor((.25, 1.5, .5, 2.0))
    q.mul_(amplitude[(qb + revision) % 4][None, None, :, None])
    k.mul_(amplitude[(kb + 1 + revision) % 4][None, None, :, None])
    # Opposite signs distinguish the two Q128 blocks resident in the two L0A
    # slots. Additional markers distinguish later pairs, heads and batches.
    sign = 1 if revision == 0 else -1
    q[..., 0] = torch.where(qb % 2 == 0, .75, -1.25) * sign
    head = torch.arange(b * h).view(b, h, 1)
    q[..., 1] = ((qb[None, None, :] * 3 + head * 5 + revision) % 19).float() * .0625
    k[..., 0] = ((kb * 5 + revision * 3) % 17 - 8).float() * .125
    k[..., 1] += head.float() * .03125
    v[..., 0] = ((kb * 3 + revision * 2) % 13 - 6).float() * .125
    v[..., 1] += head.float() * .015625
    return tuple(tensor.to(dtype).to(device) for tensor in (q, k, v))


def _require_output(output, payload):
    import torch

    if (not isinstance(output, torch.Tensor) or output.shape != payload.q.shape
            or output.dtype != torch.float32 or output.device != payload.q.device):
        raise AssertionError("Q residency requires the entire padded native FP32 output")


def validate_q_resident(adapters, device, report, save):
    """Compare every padded output and refresh same storage between native calls.

    Four paired calls per case cover initial, unchanged, refreshed, unchanged
    inputs. All calls are queued before CPU comparisons; the launch loop adds
    no host synchronization. Both algorithms share the same payload object.
    """
    import torch
    from profile_mindie_laser_quant import compare_all_rows

    if set(adapters) != {"baseline", "reuse"}:
        raise ValueError("Q residency requires baseline and reuse adapters")
    rows = report.setdefault("q_resident_validation", [])
    report["q_resident_validation_scope"] = {
        "output": "ALL padded native FP32 rows, heads and features; exact equality",
        "same_payload_object": True,
        "timed": False,
        "queued_native_calls_before_cpu_comparison": 8,
        "input_markers": "Opposite Q0/Q1 signs; different Q128 amplitudes and descriptor/head/batch markers",
        "updates": "Isolated Q/K/V and scale storage refreshed in place, with unchanged pointers",
        "cases": [dict(case) for case in Q_RESIDENT_CASES],
        "public_mapping": "Pinned dense native mapping has ky=2; ky!=2 fallback is tested in the host event model",
        "limits": "Numerical agreement does not prove hardware event state, overlap, or speedup",
    }
    save()
    for dtype in (torch.float16, torch.bfloat16):
        for case in Q_RESIDENT_CASES:
            print(f"Q residency validation: {dtype} B{case['batch']} H{case['heads']} "
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
                        _require_output(output, fixed)
                    pending.append((iteration, order, outputs))
                for iteration, order, outputs in pending:
                    error = compare_all_rows(outputs["reuse"], outputs["baseline"])
                    pointers_unchanged = all(getattr(fixed, field).data_ptr() == pointer
                                             for field, pointer in pointers.items())
                    baseline_change = None if iteration == 0 else compare_all_rows(
                        outputs["baseline"], pending[iteration - 1][2]["baseline"])
                    row["rounds"].append(dict(iteration=iteration, order=list(order),
                        payload_update=("initial", "unchanged", "in_place_refresh", "unchanged")[iteration],
                        pointers_unchanged=pointers_unchanged, baseline_change=baseline_change, **error))
                    save()
                    if not pointers_unchanged or not error["exact_match"]:
                        raise AssertionError("Q residency changed padded native outputs or payload storage")
                    if baseline_change is not None and baseline_change["exact_match"] != (iteration != 2):
                        raise AssertionError("Q residency baseline reuse/refresh gate is ineffective "
                                             "or nondeterministic")
                row["status"] = "passed"
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                for field in FIELDS:
                    getattr(fixed, field).copy_(getattr(original, field))
                save()
            del pending, original, changed, fixed, inputs, changed_inputs
