"""CPU implementation gates for binary16 ExpCast on Vector-scale Laser.

The approximation is not multiplicative. Strict reference checks therefore
require an identical maximum in every K128 block, including the valid tail.
Moving-max cases are separate diagnostics, never checked with this global map.
"""
from dataclasses import replace

import torch

from mindie_laser_fp16_expcast_math import probability, CONTRACT
from mindie_laser_quant_backend import prepare_quant_tensors, QK_BLOCK
from mindie_laser_vscale_reference import scale_scores, _mapping_inputs, _selected_queries

LIMIT = .01
FIELDS = ("q", "k", "v", "q_scale", "k_scale", "v_scale")
CASES = (
    dict(name="tiny", batch=1, heads=2, queries=1, keys=1, dim=128),
    dict(name="tail64", batch=1, heads=2, queries=17, keys=129, dim=64),
    dict(name="tail128", batch=1, heads=2, queries=65, keys=257, dim=128),
    dict(name="section64", batch=1, heads=2, queries=257, keys=4097, dim=64),
    dict(name="section128", batch=1, heads=2, queries=257, keys=4097, dim=128),
)


@torch.no_grad()
def decoded_reference(payload, query_indices=None, *, chunk=4, probability_fn=probability):
    """CPU shared-max oracle; FP16 scores/P, FP32 numerator and denominator.

Native Cube/reduction rounding and Laser's small history epsilon are not
emulated. A 1% implementation tolerance accommodates those differences.
"""
    if any(getattr(payload, field).device.type != "cpu" for field in FIELDS):
        raise ValueError("Decoded reference requires CPU payload tensors")
    if payload.v_quantize or chunk < 1:
        raise ValueError("Require plain FP16 V and positive chunk")
    ids = torch.arange(payload.shape[2]) if query_indices is None else query_indices.cpu().long()
    if ids.ndim != 1 or not ids.numel() or ((ids < 0) | (ids >= payload.shape[2])).any():
        raise ValueError("Require nonempty valid query indices")
    heads = []
    for head in range(payload.q.shape[1]):
        key = payload.k[:, head, :payload.keys].float().transpose(-1, -2)
        value = payload.v[:, head, :payload.keys].float()
        ks = payload.k_scale[:, head].index_select(1, torch.arange(payload.keys)//QK_BLOCK)
        rows = []
        for selected in ids.split(chunk):
            query = payload.q[:, head].index_select(1, selected).float()
            qs = payload.q_scale[:, head].index_select(1, selected//QK_BLOCK)
            scores = scale_scores(query @ key, qs[:, :, None], ks[:, None, :], payload.scale)
            if not torch.isfinite(scores).all():
                raise ValueError("Finite FP16 scores required")
            maxima = torch.stack([scores[..., begin:begin+QK_BLOCK].amax(-1)
                                  for begin in range(0, payload.keys, QK_BLOCK)], -1)
            if not torch.equal(maxima, maxima[..., :1].expand_as(maxima)):
                raise ValueError("Strict ExpCast reference requires shared maximum in every K128 block")
            shifted = (scores.float()-maxima[..., :1].float()).half()
            p = probability_fn(shifted).float()
            out = (p @ value)/p.sum(-1, keepdim=True)
            rows.append(out[..., :payload.shape[3]])
        heads.append(torch.cat(rows, 1))
    return torch.stack(heads, 1).half().to(payload.dtype)


def signal_payload(dtype, *, tail=False, zero=False):
    """Direct exact score codes; tail V amplifies probabilities below 2**-14.

    q=64, attention scale=1/64, scales=1 make scores equal integer K codes.
    Every K128 block starts at max0. Variable channels exclude constants from
    the RMSE gate, so FTZ or unchanged exp cannot hide behind V=2 channels.
    """
    dim, queries, keys = 128, 17, 257
    q = torch.zeros((1, 2, queries, dim), dtype=dtype)
    k = torch.zeros((1, 2, keys, dim), dtype=dtype)
    payload = prepare_quant_tensors(q, k, torch.zeros_like(k), v_quantize=False)
    payload.scale = 1/64
    payload.q[:, :, :queries, 0] = 64
    payload.q_scale.fill_(1)
    payload.k_scale.fill_(1)
    payload.v[:, :, :keys, 1] = 2
    if zero:
        return payload
    magnitudes = [12, 14, 16, 17] if tail else [1, 2, 3, 4]
    for token in range(keys):
        if token % 128 == 0:
            continue
        slot = (token-1) % 4
        payload.k[:, :, token, 0] = -magnitudes[slot]
        payload.v[:, :, token, 2+slot] = 32768 if tail else 1
    return payload


def on_device(payload, device):
    return replace(payload, **{field: getattr(payload, field).to(device) for field in FIELDS})


def checked(error, label, limit=LIMIT):
    import math
    if not all(math.isfinite(error[key]) for key in ("relative_rmse", "max_abs")) or error["relative_rmse"] > limit:
        raise AssertionError(f"{label}: implementation RMSE exceeds {100*limit:g}%: {error}")
    return error


@torch.no_grad()
def validate(adapters, device, report, save, call):
    """All device calls go through the runner's stop-on-first-failure guard."""
    from mindie_laser_quant_reference import cpu_payload
    from profile_mindie_laser_quant import compare_all_rows
    from profile_mindie_laser_vector_scale_bulk import require_exact
    import benchmark_mindie_laser_quant as quant
    stable, control, candidate = (adapters[name] for name in
        ("vector_pv_prefetch", "fp16_expcast_control", "fp16_expcast"))
    rows = report["structured_validation"] = []
    report["reference_contract"] = CONTRACT
    for dtype in (torch.float16, torch.bfloat16):
        fixtures = [(case["name"], case) for case in CASES] + [(name, None) for name in ("zero", "signal", "subnormal_tail")]
        for name, case in fixtures:
            row = dict(name=name, dtype=str(dtype), status="running")
            rows.append(row)
            report["active_validation"] = row
            save()
            print(f"FP16 ExpCast validation: {dtype} {name}", flush=True)
            inputs = None
            if case is not None:
                inputs = call("validation_inputs", lambda: tuple(x.to(device) for x in _mapping_inputs(case, dtype)))
                fixed = call("validation_prepare", lambda: stable.prepare(*inputs))
                host = call("payload_cpu", lambda: cpu_payload(fixed))
            else:
                host = signal_payload(dtype, tail=name == "subnormal_tail", zero=name == "zero")
                fixed = call("direct_payload", lambda: on_device(host, device))
            row.update(shape_bnsd=list(host.shape), keys=host.keys)
            outputs = {}
            for label, adapter in (("stable", stable), ("control", control), ("expcast", candidate)):
                outputs[label] = call(label+"_native", lambda a=adapter: a.run_native(fixed))
            error = call("all_padded_control_compare", lambda: compare_all_rows(outputs["control"], outputs["stable"]))
            require_exact(error)
            row["control_vs_stable_all_padded"] = error
            ids = _selected_queries(host.shape[2])
            actual = call("expcast_postprocess_cpu", lambda: candidate.postprocess(outputs["expcast"], fixed).cpu())
            expected = decoded_reference(host, ids)
            row["decoded_reference"] = checked(quant.common.error_metrics(
                actual.index_select(2, ids)[..., 2:], expected[..., 2:]), name)
            torch.testing.assert_close(actual[..., 0], torch.zeros_like(actual[..., 0]), rtol=0, atol=0)
            torch.testing.assert_close(actual[..., 1], torch.full_like(actual[..., 1], 2), rtol=.004, atol=.003)
            if name == "signal":
                wrong = decoded_reference(host, ids, probability_fn=lambda x: x.float().exp().half())
                distinction = quant.common.error_metrics(wrong[..., 2:], expected[..., 2:])
                if distinction["relative_rmse"] <= LIMIT:
                    raise AssertionError("Signal fixture cannot distinguish old exp from FP16 ExpCast")
                row["old_exp_negative_control"] = distinction
            if name == "subnormal_tail":
                truncated = decoded_reference(host, ids, probability_fn=lambda x:
                    torch.where(probability(x) < 2**-14, 0, probability(x)))
                distinction = quant.common.error_metrics(truncated[..., 2:], expected[..., 2:])
                if distinction["relative_rmse"] <= LIMIT:
                    raise AssertionError("Tail fixture cannot distinguish flushed subnormal probabilities")
                row["ftz_negative_control"] = distinction
            if inputs is not None:
                full = call("expcast_full_cpu", lambda: candidate(*inputs).cpu())
                row["full_prepared"] = checked(quant.common.error_metrics(full, actual), name+" full/prepared", .001)
            row["status"] = "passed"
            save()
    # Moving maxima: a global approximation is not a valid strict online oracle.
    generator = torch.Generator().manual_seed(2208)
    for dtype in (torch.float16, torch.bfloat16):
        host = [torch.randn((1, 2, n, 128), generator=generator).to(dtype) for n in (17, 4097, 4097)]
        host[1][:, :, 2048:] *= 3
        row = dict(name="changing_max", dtype=str(dtype), status="running", strict_decoded_reference=False)
        rows.append(row)
        report["active_validation"] = row
        inputs = call("changing_max_inputs", lambda: tuple(x.to(device) for x in host))
        fixed = call("changing_max_prepare", lambda: stable.prepare(*inputs))
        outputs = {name: call(name+"_changing_native", lambda a=a: a.run_native(fixed))
                   for name, a in (("stable", stable), ("control", control), ("expcast", candidate))}
        row["control_vs_stable_all_padded"] = call("changing_max_exact", lambda: compare_all_rows(outputs["control"], outputs["stable"]))
        require_exact(row["control_vs_stable_all_padded"])
        prepared = call("changing_max_postprocess", lambda: candidate.postprocess(outputs["expcast"], fixed).cpu())
        full = call("changing_max_full", lambda: candidate(*inputs).cpu())
        row["full_prepared"] = checked(quant.common.error_metrics(full, prepared), "changing max", .001)
        row["dense_fp32_diagnostic"] = quant.common.error_metrics(prepared, quant.common.dense_reference(*host, torch.arange(17)))
        checked(row["dense_fp32_diagnostic"], "finite changing-max diagnostic", float("inf"))
        row["status"] = "passed"
        save()
    return rows
