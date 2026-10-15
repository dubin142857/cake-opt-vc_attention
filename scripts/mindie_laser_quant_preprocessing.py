"""CPU-only, boundary-aware checks of NPU Laser quantization payloads.

This is a validation policy, not an NPU accuracy guarantee. It permits at most
one FP32 ULP in positive scales and division, only where that changes INT8 RNE
at a half-integer boundary. It never permits an arbitrary one-code error.
Production preprocessing and the native attention arithmetic are unchanged.
"""
import math

import torch
import torch.nn.functional as F

from mindie_laser_quant_backend import QK_BLOCK, SEQUENCE_PAD, prepare_quant_tensors


POLICY = {
    "scale_max_fp32_ulps": 1,
    "division_neighbor_fp32_ulps": 1,
    "int8_rule": "RNE of CPU division or either immediate FP32 neighbor, using validated actual scale; adjacent codes only",
    "power_of_two_scale": "Exact RNE required at quantization boundaries",
    "zero_and_padding": "Exact zero operands and unit scales for zero blocks/channels",
    "v_rule": "Exact decoded FP16 V; no INT8 boundary exception",
    "attention_reference": "Actual device payload, not CPU-requantized operands",
}


def _padded(x):
    return F.pad(x.float(), (0, 128 - x.shape[-1], 0, (-x.shape[2]) % SEQUENCE_PAD))


def _diagnostic_number(value):
    number = float(value)
    return number if math.isfinite(number) else str(number)


@torch.no_grad()
def check_preprocessing(original, actual, expected=None):
    """Return JSON diagnostics, including failures, before any native call.

    All tensors must already be on CPU. Scale validation is independent of
    code validation; recomputing with a wrong scale cannot pass this check.
    """
    report = {"passed": False, "policy": dict(POLICY), "fields": {}, "errors": []}
    if any(x.device.type != "cpu" for x in original):
        report["errors"].append("Preprocessing oracle requires original CPU inputs")
        return report
    if expected is None:
        expected = prepare_quant_tensors(*original, v_quantize=actual.v_quantize)
    for name in ("dtype", "shape", "keys", "scale", "v_quantize"):
        if getattr(actual, name) != getattr(expected, name):
            report["errors"].append(f"Payload metadata differs: {name}")
    for name in ("q", "k", "v", "q_scale", "k_scale", "v_scale"):
        a, e = getattr(actual, name), getattr(expected, name)
        valid = (isinstance(a, torch.Tensor) and a.device.type == "cpu"
                 and a.shape == e.shape and a.dtype == e.dtype and a.is_contiguous())
        report["fields"][name] = {"layout_dtype_passed": valid}
        if not valid:
            report["errors"].append(f"Wrong shape, dtype, device or contiguous layout: {name}")
    if report["errors"]:
        return report

    padded = {name: _padded(x) for name, x in zip(("q", "k", "v"), original)}
    zero_scales = {
        name + "_scale": values.reshape(*values.shape[:2], -1, QK_BLOCK, 128)
        .abs().amax(dim=(-2, -1)) == 0
        for name, values in padded.items() if name in ("q", "k")}
    zero_scales["v_scale"] = (padded["v"].abs().amax(dim=2) == 0 if actual.v_quantize
                              else torch.ones_like(actual.v_scale, dtype=torch.bool))
    for name in ("q_scale", "k_scale", "v_scale"):
        a, e = getattr(actual, name), getattr(expected, name)
        positive = torch.isfinite(a) & (a > 0)
        # Positive FP32 values have ordered integer encodings. No absolute
        # tolerance floor: even a tiny scale must satisfy the same ULP bound.
        ulps = (a.view(torch.int32).long() - e.view(torch.int32).long()).abs()
        unit_ok = ~zero_scales[name] | (a == 1)
        allowed = positive & (ulps <= POLICY["scale_max_fp32_ulps"]) & unit_ok
        field = report["fields"][name]
        field.update(passed=bool(allowed.all()), different_elements=int((a != e).sum()),
                     max_fp32_ulps=int(ulps.max()),
                     invalid_elements=int((~allowed).sum()))
        if not field["passed"]:
            field["samples"] = [{"index": index, "actual": _diagnostic_number(a[tuple(index)]),
                                  "expected": _diagnostic_number(e[tuple(index)]),
                                  "fp32_ulps": int(ulps[tuple(index)])}
                                 for index in (~allowed).nonzero()[:8].tolist()]
            report["errors"].append(f"Invalid independently checked {name}: {field}")
    if report["errors"]:
        return report  # Never use unvalidated scales to justify integer codes.

    for name in ("q", "k"):
        a, e = getattr(actual, name), getattr(expected, name)
        scale = getattr(actual, name + "_scale")
        source = padded[name]
        blocks = source.reshape(*source.shape[:2], -1, QK_BLOCK, 128)
        ratio = (blocks / scale[..., None, None]).reshape(source.shape)
        recomputed = ratio.round().clamp(-127, 127).to(torch.int8)
        mismatch = a != recomputed
        indices = mismatch.nonzero()
        r = ratio[mismatch]
        code = a[mismatch].long()
        cpu_code = recomputed[mismatch].long()
        low = torch.nextafter(r, torch.full_like(r, -float("inf"))).round().clamp(-127, 127)
        high = torch.nextafter(r, torch.full_like(r, float("inf"))).round().clamp(-127, 127)
        changed_scales = scale[indices[:, 0], indices[:, 1], indices[:, 2] // QK_BLOCK]
        # Division by a power of two is exact near half-integers for these
        # FP16/BF16 inputs. Do not excuse an incorrect tie-rounding mode.
        power_of_two = torch.frexp(changed_scales)[0] == .5
        allowed = ((code - cpu_code).abs() == 1) & (code >= low) & (code <= high) & ~power_of_two
        zeros_ok = bool((a[source == 0] == 0).all())
        field = report["fields"][name]
        field.update(passed=bool(allowed.all()) and zeros_ok,
                     elements=a.numel(), cpu_payload_differences=int((a != e).sum()),
                     actual_scale_rne_differences=int(mismatch.sum()),
                     accepted_boundary_differences=int(allowed.sum()),
                     rejected_code_differences=int((~allowed).sum()),
                     zero_and_padding_passed=zeros_ok)
        field["samples"] = []
        # Prefer rejected entries so the persisted report explains a failure.
        order = torch.cat(((~allowed).nonzero().flatten(), allowed.nonzero().flatten()))[:8]
        for item in order.tolist():
            index = tuple(indices[item].tolist())
            scale_index = index[:2] + (index[2] // QK_BLOCK,)
            field["samples"].append({
                "index": list(index), "source": float(source[index]),
                "actual_scale": float(scale[scale_index]),
                "cpu_scale": float(getattr(expected, name + "_scale")[scale_index]),
                "normalized_cpu_fp32": float(r[item]),
                "normalized_cpu_fp64": float(source[index].double() / scale[scale_index].double()),
                "actual_code": int(code[item]), "actual_scale_cpu_code": int(cpu_code[item]),
                "cpu_payload_code": int(e[index]),
                "allowed_code_range": [int(low[item]), int(high[item])],
                "accepted": bool(allowed[item])})
        if not field["passed"]:
            report["errors"].append(f"Unexplained INT8 {name} code mismatch: {field}")

    # The INT8 exception does not apply to the optional E4M3 experiment.
    v_equal = actual.v == expected.v
    report["fields"]["v"].update(passed=bool(v_equal.all()),
                                different_elements=int((~v_equal).sum()))
    if not bool(v_equal.all()):
        report["errors"].append("Decoded FP16 V differs from CPU preprocessing")
    report["passed"] = not report["errors"]
    return report
