"""Untimed exact-output checks for per-invocation scale cache-line tracking."""
from __future__ import annotations

from copy import copy


# Q/K scale tensors are flattened B/H/block arrays.  K=4097 pads to 4352,
# hence 34 scales/head: successive heads start at different 64-byte offsets.
# The scale views deliberately do NOT all start on a cache-line boundary.
CASES = (
    (1, 3, 257, 129, 64, 1, 15),
    (1, 3, 257, 4097, 128, 15, 16),
    (2, 3, 129, 4097, 64, 16, 1),
)
SCALE_FIELDS = ("q_scale", "k_scale")
PAYLOAD_FIELDS = ("q", "k", "v", *SCALE_FIELDS)
PATTERN = (0.5, 1.0, 1.5, 2.0)
# Change Q alone, then K alone, then both. This catches independently stale
# Q/K line tags as well as incorrect sharing of the two tensors' cache state.
PHASES = ((0, 0), (1, 0), (1, 2), (3, 1))


def offset_scale(source, offset):
    """A contiguous scale tensor whose data pointer starts inside its storage.

    The prefix/suffix are initialized so an incorrect address calculation
    reads finite, deliberately different values rather than uninitialized data.
    The returned view retains its storage owner throughout all native calls.
    """
    import torch

    if source.dtype != torch.float32 or offset not in (1, 15, 16):
        raise ValueError("Scale-cache validation requires FP32 scales and offsets 1/15/16")
    owner = torch.full((offset + source.numel() + 16,), 1.75,
                       dtype=source.dtype, device=source.device)
    result = owner[offset:offset + source.numel()].view(source.shape)
    result.copy_(source)
    if not result.is_contiguous() or result.storage_offset() != offset:
        raise AssertionError("The scale view lost its requested contiguous storage offset")
    return result


def scale_location(tensor):
    """Record actual addresses, not the assumption that block index / 16 suffices."""
    blocks = tensor.shape[-1]
    return {
        "data_ptr": tensor.data_ptr(),
        "storage_offset_elements": tensor.storage_offset(),
        "address_mod_64": tensor.data_ptr() % 64,
        "blocks_per_head": blocks,
        "head_start_address_mod_64": [
            (tensor.data_ptr() + head * blocks * tensor.element_size()) % 64
            for head in range(tensor.numel() // blocks)
        ],
    }


def _require_native_output(output, payload):
    import torch

    if (not isinstance(output, torch.Tensor) or output.shape != payload.q.shape
            or output.dtype != torch.float32 or output.device != payload.q.device):
        raise AssertionError("Scale-cache gate requires the whole padded native FP32 output")


def validate_scale_cache(adapters, device, report, save):
    """Check cache-line boundaries and invalidation after same-pointer rewrites.

    Each case queues *all* scale rewrites and both native APIs on the caller's
    current stream before the first CPU output comparison. There is no explicit
    device/host synchronization between consecutive rewrites and launches.
    Scales stay immutable during each attention invocation. Both operators see
    exactly the same payload object, pointers, INT8 codes and FP16 V.

    This is an implementation gate, not a model-quality or performance test.
    """
    import torch
    from profile_mindie_laser_quant import compare_all_rows

    if set(adapters) != {"baseline", "reuse"}:
        raise ValueError("Scale-cache gate requires baseline and reuse adapters")
    rows = report.setdefault("scale_cache_storage", [])
    report["scale_cache_gate_scope"] = {
        "output": "ALL padded FP32 output elements; exact agreement with scale-reuse baseline",
        "same_payload_objects": True,
        "queued_mutation_rounds_before_cpu_comparison": len(PHASES),
        "scale_mutations": "Q alone, then K alone, then both; same storage and pointers",
        "cache_line_bytes": 64,
        "v_reordering": False,
        "expcast": False,
        "timed": False,
    }
    for dtype in (torch.float16, torch.bfloat16):
        for case_index, (batch, heads, queries, keys, dim, q_offset, k_offset) in enumerate(CASES):
            print(f"Scale-cache validation: {dtype}, B{batch} H{heads} Q{queries} K{keys} "
                  f"D{dim}, scale offsets Q{q_offset}/K{k_offset}", flush=True)
            generator = torch.Generator().manual_seed(831 + case_index)
            inputs = [torch.randn(shape, generator=generator).to(dtype).to(device) for shape in (
                (batch, heads, queries, dim), (batch, heads, keys, dim), (batch, heads, keys, dim))]
            original = adapters["baseline"].prepare(*inputs)
            fixed = copy(original)
            fixed.q_scale = offset_scale(original.q_scale, q_offset)
            fixed.k_scale = offset_scale(original.k_scale, k_offset)
            saved = {name: getattr(fixed, name).clone() for name in SCALE_FIELDS}
            pointers = {name: getattr(fixed, name).data_ptr() for name in PAYLOAD_FIELDS}
            case = {"dtype": str(dtype), "shape_bhqd": [batch, heads, queries, dim],
                    "keys": keys, "scales": {name: scale_location(getattr(fixed, name))
                                             for name in SCALE_FIELDS}, "rounds": [],
                    "status": "queued"}
            rows.append(case)
            save()
            pending = []
            try:
                pattern = torch.tensor(PATTERN, dtype=torch.float32, device=device)
                indices = {name: torch.arange(getattr(fixed, name).numel(), device=device)
                           for name in SCALE_FIELDS}
                for iteration, phases in enumerate(PHASES):
                    for name, phase in zip(SCALE_FIELDS, phases):
                        target = getattr(fixed, name)
                        factors = pattern[(indices[name] + phase) % len(PATTERN)].view(target.shape)
                        target.copy_(saved[name] * factors)
                    order = ("reuse", "baseline") if iteration % 2 == 0 else ("baseline", "reuse")
                    outputs = {name: adapters[name].run_native(fixed) for name in order}
                    for output in outputs.values():
                        _require_native_output(output, fixed)
                    pending.append((iteration, phases, order, outputs))

                # CPU copies below may synchronize. None occur inside the launch loop.
                previous_baseline = None
                for iteration, phases, order, outputs in pending:
                    error = compare_all_rows(outputs["reuse"], outputs["baseline"])
                    unchanged = all(getattr(fixed, name).data_ptr() == pointer
                                    for name, pointer in pointers.items())
                    change = None if previous_baseline is None else compare_all_rows(
                        outputs["baseline"], previous_baseline)
                    row = dict(iteration=iteration, q_phase=phases[0], k_phase=phases[1],
                               order=list(order), pointers_unchanged=unchanged,
                               baseline_change_after_rewrite=change, **error)
                    case["rounds"].append(row)
                    save()
                    if not unchanged or not error["exact_match"]:
                        raise AssertionError("Scale-cache candidate changed padded native outputs "
                                             "after offset-scale or same-pointer rewrite")
                    if change is not None and change["exact_match"]:
                        raise AssertionError("Scale-cache mutation test is ineffective: baseline "
                                             "did not respond to rewritten scales")
                    previous_baseline = outputs["baseline"]
                case["status"] = "passed"
            finally:
                # Only the isolated views were mutated. Restore them as well, including
                # on a failing gate; original quantized payloads were never modified.
                for name in SCALE_FIELDS:
                    getattr(fixed, name).copy_(saved[name])
                save()
