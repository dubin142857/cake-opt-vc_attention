"""Independent CPU arithmetic and untimed gates for Vector-side QK scaling.

The candidate changes FP16 score rounding. Its decoded oracle is deliberately
separate from the existing Laser oracle. General random inputs use a global
softmax maximum and remain diagnostics; structured cases hold each K128
maximum fixed, avoiding online-history ambiguity in the implementation gate.
"""
from __future__ import annotations

from copy import copy
from dataclasses import replace

import torch

from mindie_laser_quant_backend import QK_BLOCK, prepare_quant_tensors
from mindie_laser_quant_reference import cpu_payload, structured_case


IMPLEMENTATION_LIMIT = .01
ROUNDING_DRIFT_LIMIT = .01
HALF_NORMAL_MIN = 2.0 ** -14
HALF_MAX = 65504.0
TRANSPORT_GAIN = 1.0 / 64.0
FIELDS = ("q", "k", "v", "q_scale", "k_scale")

# Dense mapper coverage: Q128/Q256 boundaries, K4096 section transitions,
# physical padding and enough batch/head descriptors to revisit Cube cores.
MAPPING_CASES = (
    dict(name="tiny", batch=1, heads=2, queries=1, keys=1, dim=128),
    dict(name="qk_tails", batch=1, heads=2, queries=17, keys=129, dim=64),
    dict(name="q_pair", batch=2, heads=3, queries=257, keys=385, dim=128),
    dict(name="key_section_tail", batch=1, heads=2, queries=257, keys=4097, dim=64),
    dict(name="two_key_sections", batch=1, heads=2, queries=513, keys=8193, dim=128),
    dict(name="descriptor_revisit", batch=2, heads=17, queries=257, keys=257, dim=64),
)


def gain_coefficient(query_scale, key_scale, attention_scale):
    """Mirror the specified FP32 operation order, including DEQ bit truncation."""
    gain = (query_scale.float() * key_scale.float()).contiguous()
    gain = (gain.view(torch.int32) & -8192).view(torch.float32)
    scale = torch.tensor(attention_scale, dtype=torch.float16).float()
    coefficient = (gain * 64.0) * scale
    fast = (coefficient == 0) | (
        (coefficient.abs() >= HALF_NORMAL_MIN) & (coefficient.abs() <= HALF_MAX))
    return gain, coefficient, fast


def scale_scores(dots, query_scale, key_scale, attention_scale, *, details=False):
    """INT32 dot -> constant DEQF16 transport -> fused Vector score gain.

    Only CPU tensors are accepted. The FP32 branch preserves coefficients
    which would underflow/overflow a half scalar; it does not remove the
    candidate's initial FP16 transport rounding.
    """
    if any(value.device.type != "cpu" for value in (dots, query_scale, key_scale)):
        raise ValueError("Vector-scale arithmetic oracle is CPU-only")
    gain, coefficient, fast = gain_coefficient(query_scale, key_scale, attention_scale)
    if not bool(torch.isfinite(coefficient).all()):
        raise ValueError("Vector-scale coefficient must be finite")
    transport = (dots.float() * TRANSPORT_GAIN).half()
    # Select the scalar before multiplying: a half overflow in an unused
    # branch must never produce NaN through zero * infinity.
    multiplier = torch.where(fast, coefficient.half().float(), coefficient)
    scores = (transport.float() * multiplier).half()
    if details:
        return scores, dict(transport=transport, gain=gain, coefficient=coefficient,
                            fast=fast)
    return scores


@torch.no_grad()
def decoded_reference(payload, query_indices=None, *, expcast=False, chunk=4,
                      structured=False):
    """Candidate-specific CPU attention, over every valid key.

    ``structured=True`` requires zero maxima in each native K128 tile.
    ``structured="shared_max"`` permits the same finite maximum in every tile
    (used only for the full signed-INT8 transport bound fixture).
    Native Cube accumulation and online FP32 summation are not emulated.
    """
    if expcast:
        raise ValueError("Vector-scale experiment validates original exp only")
    if payload.q.device.type != "cpu":
        raise ValueError("The explicit Vector-scale oracle is CPU-only")
    if payload.v_quantize:
        raise ValueError("Vector-scale experiment requires unquantized FP16 V")
    if structured not in (False, True, "shared_max"):
        raise ValueError("Unrecognized structured maximum contract")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    ids = (torch.arange(payload.shape[2]) if query_indices is None
           else query_indices.detach().cpu().to(torch.long))
    if ids.ndim != 1 or not ids.numel() or bool(((ids < 0) | (ids >= payload.shape[2])).any()):
        raise ValueError("Query indices must select valid nonempty query rows")
    results = []
    for head in range(payload.q.shape[1]):
        key = payload.k[:, head, :payload.keys].float().transpose(-1, -2)
        value = payload.v[:, head, :payload.keys].float()
        key_ids = torch.arange(payload.keys) // QK_BLOCK
        key_scale = payload.k_scale[:, head].index_select(1, key_ids)
        rows = []
        for selected in ids.split(chunk):
            query = payload.q[:, head].index_select(1, selected).float()
            query_scale = payload.q_scale[:, head].index_select(1, selected // QK_BLOCK)
            # All products and sums of D128 INT8 values fit exactly in FP32.
            scores = scale_scores(query @ key, query_scale[:, :, None],
                                  key_scale[:, None, :], payload.scale)
            if not bool(torch.isfinite(scores).all()):
                raise ValueError("Candidate score is outside the finite FP16 score contract")
            if structured:
                maxima = torch.stack([
                    scores[..., begin:begin + QK_BLOCK].amax(-1)
                    for begin in range(0, payload.keys, QK_BLOCK)], dim=-1)
                expected = (torch.zeros_like(maxima) if structured is True
                            else maxima[..., :1].expand_as(maxima))
                if not torch.equal(maxima, expected):
                    raise ValueError("Structured oracle needs zero maximum in every native key tile"
                                     if structured is True else
                                     "Structured oracle needs the same maximum in every native key tile")
            shifted = (scores.float() - scores.amax(-1, keepdim=True).float()).half()
            probability = shifted.float().exp().half().float()
            out = (probability @ value) / probability.sum(-1, keepdim=True)
            rows.append(out[..., :payload.shape[3]])
        results.append(torch.cat(rows, dim=1))
    return torch.stack(results, dim=1).half().to(payload.dtype)


def _mapping_inputs(case, dtype):
    """Keep K128 zero maxima while giving each batch/head its own scale."""
    q, k, v = structured_case(dtype, case["dim"], case["queries"], case["keys"])
    b, h = case["batch"], case["heads"]
    outputs = []
    for index, source in enumerate((q, k, v)):
        result = source[:, :1].expand(b, h, -1, -1).clone()
        marker = torch.arange(b * h).view(b, h, 1, 1)
        factor = torch.pow(2., ((marker + index) % 5 - 2).float())
        result.mul_(factor.to(dtype))
        if index == 2:
            result[..., 0], result[..., 1] = 0, 2
        outputs.append(result)
    return tuple(outputs)


def _selected_queries(count):
    # ALL rows in small cases; transition rows in large mapping cases.
    if count <= 33:
        return torch.arange(count)
    values = {0, 1, count - 1}
    for boundary in range(128, count, 128):
        values.update((boundary - 1, boundary, min(boundary + 1, count - 1)))
    return torch.tensor(sorted(values), dtype=torch.long)


def range_payload(dtype, kind):
    """Direct payload probes outside public quantizer's [-127,127] code range."""
    if kind not in ("tiny_gain", "large_gain", "signed_int8_bound", "zero"):
        raise ValueError("Unknown Vector-scale range probe")
    dim, keys = (128, 256) if kind == "signed_int8_bound" else (64, 129)
    q = torch.zeros((1, 2, 17, dim), dtype=dtype)
    k = torch.zeros((1, 2, keys, dim), dtype=dtype)
    values = ((torch.arange(keys)[:, None] * 13 + torch.arange(dim)[None, :] * 7) % 29 - 14).float() / 14
    v = values[None, None].expand(1, 2, -1, -1).to(dtype).clone()
    v[..., 0], v[..., 1] = 0, 2
    payload = prepare_quant_tensors(q, k, v, v_quantize=False)
    if kind == "zero":
        return payload, True
    if kind == "signed_int8_bound":
        payload.q[:, :, :17] = -128
        payload.k[:, :, :keys] = -128
        payload.q_scale.fill_(.001)
        payload.k_scale.fill_(.001)
        return payload, "shared_max"
    if kind == "tiny_gain":
        payload.q[:, :, :17, :dim] = -128
        payload.k[:, :, :keys, :dim] = 127
        payload.q_scale.fill_(.0001)
        payload.k_scale.fill_(.0001)
    else:
        payload.q[:, :, :17, 0] = 1
        payload.k[:, :, :keys, 0] = -1
        payload.q_scale.fill_(128)
        payload.k_scale.fill_(64)
    payload.k[:, :, torch.arange(0, keys, 128)] = 0
    return payload, True


def _on_device(payload, device):
    return replace(payload, **{
        name: getattr(payload, name).to(device)
        for name in (*FIELDS, "v_scale")})


def _checked(error, limit, label):
    if error["relative_rmse"] > limit:
        raise AssertionError(f"{label} exceeds {100 * limit:g}% RMSE: {error}")
    return dict(error, limit=limit, passed=True)


@torch.no_grad()
def validate_vector_scale(adapters, device, report, save):
    """Own-oracle gates, scale updates at unchanged pointers, and range probes.

    Baseline drift is an explicit changed-rounding acceptance gate, not an
    exact-equivalence assertion or a model-quality requirement.
    """
    import benchmark_cann_fas_local as common
    from profile_mindie_laser_quant import compare_all_rows

    if set(adapters) != {"baseline", "reuse"}:
        raise ValueError("Vector scale requires baseline and reuse adapters")
    if any(adapter.expcast or adapter.v_quantize for adapter in adapters.values()):
        raise ValueError("Vector scale requires original exp and unquantized FP16 V")
    rows = report["vector_scale_validation"] = []
    report["vector_scale_validation_scope"] = dict(
        decoded_limit=IMPLEMENTATION_LIMIT, baseline_rounding_drift_limit=ROUNDING_DRIFT_LIMIT,
        baseline_drift_scope="ALL padded native FP32 output elements",
        decoded_scope="Selected valid Q rows around Q128/Q256 boundaries, all keys/heads/features; all Q rows for Q<=33",
        maxima="Zero in every K128 tile, or identical tile maxima for signed INT8 bound probe",
        scales="Same Q/K scale pointers updated separately between queued calls",
        timing=False, no_model_quality_claim=True,
        contract="half(dot/64) times half(64*truncated_gain*half(SCALE)); FP32 scalar fallback outside normal-half range",
    )
    save()
    for dtype in (torch.float16, torch.bfloat16):
        for case in MAPPING_CASES:
            label = f"{dtype} {case['name']} B{case['batch']} H{case['heads']} Q{case['queries']} K{case['keys']}"
            print("Vector-scale validation: " + label, flush=True)
            row = dict(case, dtype=str(dtype), status="running", rounds=[])
            rows.append(row)
            save()
            try:
                inputs = tuple(x.to(device) for x in _mapping_inputs(case, dtype))
                initial = adapters["baseline"].prepare(*inputs)
                fixed = copy(initial)
                for field in FIELDS:
                    setattr(fixed, field, getattr(initial, field).clone())
                pointers = {field: getattr(fixed, field).data_ptr() for field in FIELDS}
                host = cpu_payload(initial)
                selected = _selected_queries(case["queries"])
                row["query_indices"] = selected.tolist()
                # Copies below have fixed contents before any native calls.
                # All five calls per adapter are queued before CPU comparison.
                changed_qs = initial.q_scale * 1.5
                changed_ks = initial.k_scale * .75
                hosts = [host, host,
                         replace(host, q_scale=host.q_scale * 1.5),
                         replace(host, q_scale=host.q_scale * 1.5, k_scale=host.k_scale * .75)]
                hosts.append(hosts[-1])
                pending = []
                for iteration in range(5):
                    if iteration == 2:
                        fixed.q_scale.copy_(changed_qs)
                    if iteration == 3:
                        fixed.k_scale.copy_(changed_ks)
                    order = ("baseline", "reuse") if iteration % 2 == 0 else ("reuse", "baseline")
                    pending.append({name: adapters[name].run_native(fixed) for name in order})
                for iteration, outputs in enumerate(pending):
                    drift = compare_all_rows(outputs["reuse"], outputs["baseline"])
                    record = dict(iteration=iteration,
                        update=("initial", "unchanged", "q_scale_in_place", "k_scale_in_place", "unchanged")[iteration],
                        baseline_drift=_checked(drift, ROUNDING_DRIFT_LIMIT, label + " changed-rounding drift"),
                        pointers_unchanged=all(getattr(fixed, field).data_ptr() == pointer
                                               for field, pointer in pointers.items()))
                    if not record["pointers_unchanged"]:
                        raise AssertionError("Scale-update test changed input pointers")
                    # Test unchanged calls exactly. Baseline/candidate comparisons
                    # intentionally use different arithmetic and a separate gate.
                    if iteration in (1, 4):
                        for name in adapters:
                            same = compare_all_rows(outputs[name], pending[iteration - 1][name])
                            if not same["exact_match"]:
                                raise AssertionError(f"{label}: {name} unchanged payload is nondeterministic")
                        record["unchanged_output"] = "exact"
                    else:
                        expected = decoded_reference(hosts[iteration], selected, structured=True)
                        actual = adapters["reuse"].postprocess(outputs["reuse"], fixed).cpu().index_select(2, selected)
                        record["decoded_reference"] = _checked(
                            common.error_metrics(actual[..., 2:], expected[..., 2:]),
                            IMPLEMENTATION_LIMIT, label + " own decoded reference")
                        torch.testing.assert_close(actual[..., 0], torch.zeros_like(actual[..., 0]), rtol=0, atol=0)
                        torch.testing.assert_close(actual[..., 1], torch.full_like(actual[..., 1], 2), rtol=.004, atol=.003)
                    row["rounds"].append(record)
                    save()
                # Full/prepared consistency uses the ORIGINAL fresh inputs.
                full = adapters["reuse"](*inputs).cpu()
                prepared = adapters["reuse"].postprocess(pending[0]["reuse"], initial).cpu()
                row["full_prepared"] = _checked(common.error_metrics(full, prepared), .001, label + " full/prepared")
                row["status"] = "passed"
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                save()

        for kind in ("zero", "tiny_gain", "large_gain", "signed_int8_bound"):
            print(f"Vector-scale numeric range: {dtype} {kind}", flush=True)
            row = dict(name=kind, dtype=str(dtype), direct_payload=True, status="running")
            rows.append(row)
            save()
            try:
                host, structure = range_payload(dtype, kind)
                payload = _on_device(host, device)
                output = adapters["reuse"].run_native(payload)
                baseline = adapters["baseline"].run_native(payload)
                row["baseline_drift"] = _checked(compare_all_rows(output, baseline),
                    ROUNDING_DRIFT_LIMIT, kind + " changed-rounding drift")
                actual = adapters["reuse"].postprocess(output, payload).cpu()
                expected = decoded_reference(host, structured=structure)
                row["decoded_reference"] = _checked(common.error_metrics(actual[..., 2:], expected[..., 2:]),
                    IMPLEMENTATION_LIMIT, kind + " own decoded reference")
                torch.testing.assert_close(actual[..., 0], torch.zeros_like(actual[..., 0]), rtol=0, atol=0)
                torch.testing.assert_close(actual[..., 1], torch.full_like(actual[..., 1], 2), rtol=.004, atol=.003)
                row["status"] = "passed"
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                save()
    return rows
