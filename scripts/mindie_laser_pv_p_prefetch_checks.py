"""Exact P+V/V-only gates, including different probabilities across P128 tiles."""
from __future__ import annotations

import torch

from mindie_laser_pv_prefetch_checks import CASES, boundary_payload, validate_pv_prefetch


P_CASES = tuple(dict(name=f"probability_query_pair_k{keys}", batch=1, heads=2,
                    queries=257, keys=keys, dim=128, probability_pattern=True)
                for keys in (385, 4097))


def probability_payload(case, dtype, identity):
    """Adjacent Q128 groups favor opposite K128 blocks before masking.

    Scores are approximately +/-5.657 for the first two Q128 groups, so a
    misplaced probability tile changes block mass substantially. The third
    partial query group uses a distinct period-four key-block pattern.
    """
    payload, description = boundary_payload(case, dtype, identity)
    if not case.get("probability_pattern"):
        return payload, description
    payload.q.zero_()
    payload.k.zero_()
    qn, kn = case["queries"], case["keys"]
    variant = 0 if identity == "A" else 1
    qrows = torch.arange(qn)
    features = qrows // 128
    payload.q[:, :, qrows, features] = 16
    kblocks = torch.arange(kn) // 128
    for head in range(case["heads"]):
        parity = (kblocks + head + variant) % 2
        codes = torch.where(parity == 0, 16, -16).to(torch.int8)
        payload.k[:, head, :kn, 0] = codes
        payload.k[:, head, :kn, 1] = -codes
        payload.k[:, head, :kn, 2] = torch.where(
            (kblocks + head + variant) % 4 < 2, 16, -16).to(torch.int8)
    payload.q_scale.fill_(.5)
    payload.k_scale.fill_(.5)
    description.update(probability_pattern="Q128 groups favor alternating K128 blocks",
                       score_bound=256*.25/128**.5,
                       third_query_group="Period-four K128 pattern",
                       compared_to="V-only PV prefetch with identical Q/K/V payload")
    return payload, description


def validate_pv_p_prefetch(old, new, device, report, save):
    return validate_pv_prefetch(old, new, device, report, save,
        cases=CASES+P_CASES, payload_factory=probability_payload,
        scope_name="pv_p_prefetch",
        description="P+V prefetch versus V-only PV prefetch; same-descriptor P lookahead, unchanged math")
