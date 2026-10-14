"""Accuracy diagnostics on unchanged quantized payloads; never used for timing."""
import math
import json
import torch
from validate_ascend910b import error, to_cpu


def payload_difference(left, right):
    result = {}
    for name in ("q", "k", "v", "q_scale", "k_scale", "v_scale", "lut"):
        a, b = left[name], right[name]
        same = bool((a == b).all().item())
        result[name] = {"exact_match": same,
                        "max_abs": 0.0 if same else float((a.float() - b.float()).abs().amax().item())}
    return result


def print_check(row):
    print(json.dumps({k: v for k, v in row.items() if k not in ("worst_rows", "per_head")}), flush=True)


def row_errors(actual, expected, query_ids, queries, top=8):
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    error(a, b)  # reject nonfinite data before ranking
    squared = (a - b).square().sum(-1)
    total = squared.sum().item()
    indices = squared.flatten().argsort(descending=True)[:min(top, squared.numel())]
    heads, rows = a.shape[1:3]
    result = []
    for flat in indices.tolist():
        batch, head, row = flat // (heads * rows), (flat // rows) % heads, flat % rows
        query = int(query_ids[row])
        result.append({"batch": batch, "head": head, "query": query,
                       "query_block": query // 32, "aiv_half": (query % 32) // 16,
                       "task": (batch * heads + head) * math.ceil(queries / 32) + query // 32,
                       "squared_error_fraction": float(squared[batch, head, row]) / max(total, 1e-30),
                       **error(a[batch, head, row], b[batch, head, row])})
    return {"worst_rows": result, "per_head": [
        {"head": h, **error(a[:, h], b[:, h])} for h in range(heads)]}


def isolate_query_tile(payload, batch, head, query):
    """Copy a whole Q32 tile and its ORIGINAL scale; do not requantize Q/K/V."""
    block = query // 32
    begin = block * 32
    p = dict(payload)
    for name in ("q", "k", "v", "q_scale", "k_scale", "v_scale"):
        value = payload[name][batch:batch + 1, head:head + 1]
        if name == "q":
            value = value[:, :, begin:begin + 32]
        elif name == "q_scale":
            value = value[:, :, block:block + 1]
        p[name] = value.contiguous()
    p["queries"] = min(32, payload["queries"] - begin)
    p["layout"] = "HND"
    return p


def key_prefix(payload, keys):
    """Shorten the valid-key range while retaining original K/V scales and mean."""
    if not 0 < keys <= payload["keys"]:
        raise ValueError("Key prefix must be within the prepared payload")
    padded = math.ceil(keys / 128) * 128
    return {**payload, "keys": keys, "k": payload["k"][:, :, :padded].contiguous(),
            "v": payload["v"][:, :, :padded].contiguous(),
            "k_scale": payload["k_scale"][:, :, :padded // 128].contiguous()}


def raw_forward(vc, payload, expcast):
    """Expose the registered op's FP32 output, before the Python output cast."""
    p = payload
    return vc.load_native().forward(p["q"], p["k"], p["v"], p["q_scale"], p["k_scale"],
                                    p["v_scale"], p["lut"], p["queries"], p["keys"],
                                    p["scale"], bool(expcast))[:, :, :p["queries"]]


@torch.no_grad()
def check_payload(vc, payload, expcast, query_ids=None):
    from vc_attention_ascend.reference import payload_reference
    ids = torch.arange(payload["queries"]) if query_ids is None else query_ids.cpu()
    expected = payload_reference(to_cpu(payload), expcast=expcast, query_indices=ids)
    raw = raw_forward(vc, payload, expcast)
    sample = raw.index_select(2, ids.to(raw.device)).cpu()
    # Keep API casting on the NPU, matching the public operator contract.
    cast = raw.to(payload["dtype"])
    cast_sample = cast.index_select(2, ids.to(raw.device)).cpu()
    repeated = raw_forward(vc, payload, expcast)
    repeat_error = error(repeated, raw)  # all output rows, not just sampled queries
    decoded_error = error(cast_sample, expected.to(payload["dtype"]))
    row = {"queries": payload["queries"], "keys": payload["keys"], "expcast": expcast,
           "checked_rows": len(ids), "all_keys": True,
           "raw_fp32_vs_decoded": error(sample, expected),
           "cast_vs_decoded": decoded_error,
           "repeat_all_rows": repeat_error,
           "passed": decoded_error["relative_rmse"] <= .01 and repeat_error["relative_rmse"] <= .001,
           **row_errors(cast_sample, expected.to(payload["dtype"]), ids, payload["queries"])}
    return row, cast_sample


@torch.no_grad()
def diagnose_failure(vc, payload, actual_sample, expected_sample, query_ids, expcast, save,
                     *, original_scope="fixed_payload"):
    """Run bounded probes around the worst original rows, leaving the gate failed."""
    ranking = row_errors(actual_sample, expected_sample, query_ids, payload["queries"])
    report = {"scope": "diagnostic execution only; no timings or relaxed accuracy gates",
              "original_output_scope": original_scope,
              "isolation_scope": "stored payload; Q/K/V bytes and scales retained",
              "comparison_note": (
                  "Original full API used independent preprocessing. A later fresh-payload check does not prove "
                  "the original call's bytes were identical; a difference does not by itself prove a scheduling bug."
                  if original_scope == "full_api_fresh_preprocessing" else
                  "Original and isolated executions use the same quantized operands and scales."),
              "original": ranking, "isolated_tiles": [], "prefixes": [], "invariants": []}
    save(report)
    seen = set()
    first = None
    for bad in ranking["worst_rows"]:
        key = (bad["batch"], bad["head"], bad["query_block"])
        if key in seen:
            continue
        seen.add(key)
        tile = isolate_query_tile(payload, bad["batch"], bad["head"], bad["query"])
        if first is None:
            first = tile
        print("Isolate original failing tile:", key, flush=True)
        row, sample = check_payload(vc, tile, expcast)
        checked_index = (query_ids.cpu() == bad["query"]).nonzero().item()
        row["original_location"] = bad
        row["original_run_vs_isolated_row"] = error(
            actual_sample[bad["batch"], bad["head"], checked_index], sample[0, 0, bad["query"] % 32])
        report["isolated_tiles"].append(row)
        print_check(row)
        save(report)
        if len(seen) == 2:
            break
    # Same bytes/scales, increasing K-loop length: no re-centering or regrouping.
    for keys in sorted({min(k, first["keys"]) for k in (128, 513, 4097, first["keys"])}):
        for cast in (False, True):
            print(f"Fixed-payload prefix: K={keys}, expcast={cast}", flush=True)
            row, _ = check_payload(vc, key_prefix(first, keys), cast)
            report["prefixes"].append(row)
            print_check(row)
            save(report)
    for mode in ("zero_q", "constant_decoded_v"):
        variant = dict(first)
        if mode == "zero_q":
            variant["q"] = torch.zeros_like(first["q"])
        else:
            variant["v"] = torch.ones_like(first["v"])
            variant["v_scale"] = torch.ones_like(first["v_scale"])
        for cast in (False, True):
            print(f"Fixed-payload invariant: {mode}, expcast={cast}", flush=True)
            row, _ = check_payload(vc, variant, cast)
            report["invariants"].append({"mode": mode, **row})
            print_check(row)
            save(report)
    return report
