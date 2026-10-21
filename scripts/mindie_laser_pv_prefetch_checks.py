"""Untimed exact gates for the V-only PV-prefetch schedule.

The existing Vector-scale own oracle and queued scale-pointer tests remain
mandatory. These additional cases exercise V-buffer identities and key-section
boundaries, without reusing the unrelated row-max reduction fixtures.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from mindie_laser_quant_backend import prepare_quant_tensors


BOUNDARY_KEYS = (1, 127, 128, 129, 255, 256, 257, 383, 384, 385,
                 1023, 1024, 1025, 4095, 4096, 4097, 8193)
CASES = tuple(dict(name=f"key_{keys}", batch=1, heads=2, queries=17,
                   keys=keys, dim=(64, 128)[i % 2])
              for i, keys in enumerate(BOUNDARY_KEYS)) + (
    dict(name="query_pair_history", batch=2, heads=3, queries=257, keys=4097, dim=128),
    dict(name="descriptor_revisit", batch=2, heads=17, queries=257, keys=1025, dim=64),
)
QUEUE = ("A", "B", "A", "B")
FIELDS = ("q", "k", "v", "q_scale", "k_scale", "v_scale")


def boundary_payload(case, dtype, identity):
    """Direct bounded INT8 payloads with token/head/channel-specific V values.

    A and B use independent allocations and distinct codes, scales and V.
    Logical K128 boundaries include 1/2/3/8/32 blocks. Public native payloads
    pad K to 256 tokens, so logical odd block counts are NOT a claim that a
    device descriptor had odd Kx. The loop/event host model checks those cases.
    """
    if identity not in QUEUE or min(case[k] for k in ("batch", "heads", "queries", "keys")) < 1:
        raise ValueError("Invalid PV boundary fixture")
    if case["dim"] not in (64, 128) or dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("PV boundary fixture requires FP16/BF16 D64/128")
    b, h, qn, kn, dim = (case[k] for k in ("batch", "heads", "queries", "keys", "dim"))
    variant = 0 if identity == "A" else 1
    q = torch.zeros((b, h, qn, dim), dtype=dtype)
    k = torch.zeros((b, h, kn, dim), dtype=dtype)
    v = torch.zeros_like(k)
    payload = prepare_quant_tensors(q, k, v, v_quantize=False)
    head = torch.arange(b * h).view(b, h, 1, 1)
    features = torch.arange(dim).view(1, 1, 1, dim)
    qt = torch.arange(qn).view(1, 1, qn, 1)
    kt = torch.arange(kn).view(1, 1, kn, 1)
    payload.q[..., :qn, :dim] = ((qt * 3 + head * 5 + features * 7 + variant * 11) % 31 - 15).to(torch.int8)
    payload.k[..., :kn, :dim] = ((kt * 11 + head * 3 + features * 5 + variant * 7) % 29 - 14).to(torch.int8)
    # Exact binary fractions; distinguish blocks/heads without FP16 overflow.
    values = ((kt * 13 + (kt // 128) * 7 + head * 19 + features * 3 + variant * 23) % 127 - 63).float() / 32
    payload.v[..., :kn, :dim] = values.half()
    payload.v[..., :kn, 0] = 1 if identity == "A" else -1
    payload.v[..., :kn, 1] = 0
    qblock = torch.arange(payload.q_scale.shape[-1]).view(1, 1, -1)
    kblock = torch.arange(payload.k_scale.shape[-1]).view(1, 1, -1)
    head_scale = head[..., 0].float()
    payload.q_scale.copy_((1 + (qblock + head_scale + variant) % 3) / 128)
    payload.k_scale.copy_((1 + (kblock * 2 + head_scale + variant) % 5) / 128)
    description = dict(case, dtype=str(dtype), identity=identity,
        logical_key_blocks=(kn + 127) // 128,
        physical_key_blocks=payload.k.shape[2] // 128,
        valid_keys=kn, padded_keys=payload.k.shape[2],
        history_section_input=kn > 4096, direct_payload=True)
    return payload, description


@torch.no_grad()
def validate_pv_prefetch(old, new, device, report, save):
    from profile_mindie_laser_quant import compare_all_rows
    from profile_mindie_laser_vector_scale_bulk import require_exact

    rows = report["pv_prefetch_boundary_validation"] = []
    report["pv_prefetch_validation_scope"] = dict(
        candidate="V-only GM-to-L1 prefetch; P copies and MAD order unchanged",
        cases=list(CASES), queue=list(QUEUE),
        output_gate="Finite exact ALL padded native output rows/heads/features",
        queued_calls="Both adapters A/B/A/B without intervening host reads or synchronization",
        physical_padding="256-token padding; logical odd K128 counts do not prove device odd Kx",
        oracle="Existing Vector-scale own oracle and unchanged-pointer Q/K scale updates also run",
        timed=False, dense_reference=False)
    save()
    for dtype in (torch.float16, torch.bfloat16):
        for case in CASES:
            row = dict(case, dtype=str(dtype), status="running", queued_outputs=[])
            rows.append(row)
            save()
            print(f"PV V-prefetch boundary: {dtype} {case['name']} "
                  f"B{case['batch']} H{case['heads']} Q{case['queries']} K{case['keys']}", flush=True)
            payloads, descriptions = {}, {}
            for identity in ("A", "B"):
                host, descriptions[identity] = boundary_payload(case, dtype, identity)
                payloads[identity] = replace(host, **{
                    field: getattr(host, field).to(device) for field in FIELDS})
            row["payloads"] = descriptions
            # All allocations/copies complete in stream order before the queue.
            # Native outputs stay alive until every launch has been enqueued.
            pending = []
            for iteration, identity in enumerate(QUEUE):
                pair = (("old", old), ("new", new)) if iteration % 2 == 0 else (("new", new), ("old", old))
                outputs = {name: adapter.run_native(payloads[identity]) for name, adapter in pair}
                pending.append((identity, outputs))
            first = {}
            for iteration, (identity, outputs) in enumerate(pending):
                comparison = compare_all_rows(outputs["new"], outputs["old"])
                require_exact(comparison)
                entry = dict(iteration=iteration, identity=identity, all_native_output=comparison)
                for name in ("old", "new"):
                    key = (name, identity)
                    if key in first:
                        repeated = compare_all_rows(outputs[name], first[key])
                        require_exact(repeated)
                        entry[name+"_repeated_payload"] = repeated
                    else:
                        first[key] = outputs[name]
                row["queued_outputs"].append(entry)
            # A/B have opposite constant first channels. Distinct results make
            # accidental reuse detectable; this is not a model-quality gate.
            for name in ("old", "new"):
                if torch.equal(first[(name, "A")], first[(name, "B")]):
                    raise AssertionError(name+" returned identical outputs for distinct A/B PV payloads")
            row["status"] = "passed"
            save()
    return rows
