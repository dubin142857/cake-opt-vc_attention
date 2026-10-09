"""Hopper integration gates for unified vc31 and vc32 PV/mean candidates."""
import argparse
import json
import math
from pathlib import Path
import torch

from profile_int8_h200 import metrics


def check_constant_channels(out, decoded, values, *, expcast, v_smooth):
    """Strict gate for controlled payloads with an independent known answer.

    Ordinary Sage uses FP8(P) in PV but unrounded P in the denominator.
    With constant V=c, its answer is c_q * sum(FP8(P)) / sum(P), not c.
    ExpCast sums the decoded P in both places. With V smoothing, these
    constant channels have zero residual and are restored by the mean term.
    """
    actual = out[..., :2]
    target = values[:, :, :1, :2].expand_as(actual)
    torch.testing.assert_close(actual, decoded[..., :2], rtol=.004, atol=.003,
                               msg=lambda detail: "Constant channels differ from decoded quantized reference\n" + detail)
    preserves_constant = expcast or v_smooth
    if preserves_constant:
        torch.testing.assert_close(actual, target, rtol=.004, atol=.003,
                                   msg=lambda detail: "ExpCast/mean restoration failed to preserve constant channels\n" + detail)
    return {
        "contract": "constant_preservation" if preserves_constant else "rounded_P_over_unrounded_mass",
        "vs_original_values": metrics(actual, target),
        "vs_decoded_reference": metrics(actual, decoded[..., :2]),
    }


def constant_channel_diagnostics(out, decoded, values, previous):
    """Report local errors without adding a tighter random-input oracle gate.

    FP32 decoded matmul is not a bit-exact emulator of Hopper FP8 WGMMA.
    Separately rounding both results to BF16 also affects elementwise errors.
    The random suite uses the established full-output gates; the controlled
    suite below independently checks normalization, including constant V.
    """
    actual, target = out[..., :2], decoded[..., :2]
    delta = (actual.float() - target.float()).abs()
    flat_index = int(delta.argmax().item())
    index, remaining = [], flat_index
    for size in reversed(delta.shape):
        index.append(remaining % size)
        remaining //= size
    index = tuple(reversed(index))
    return {
        "vs_original_values": metrics(actual, values[:, :, :1, :2].expand_as(actual)),
        "vs_decoded_reference": metrics(actual, target),
        "previous_vs_decoded_reference": metrics(previous[..., :2], target),
        "previous_exact_match": torch.equal(actual, previous[..., :2]),
        "pointwise_reference_within_legacy_tolerance": bool(
            torch.isclose(actual, target, rtol=.004, atol=.003).all().item()),
        "worst_reference_element": {
            "index": list(index), "actual": actual[index].item(),
            "decoded": target[index].item(), "previous": previous[..., :2][index].item(),
            "abs_error": delta[index].item(),
        },
        "scope": "Diagnostics only; random cases use full-output regression/reference gates.",
    }


def check_random_output(out, previous, decoded, *, v_smooth):
    """The original random-input gates, shared by all four feature arms."""
    metrics(out, previous, limit=.001)
    if not v_smooth:
        torch.testing.assert_close(out, previous, rtol=0, atol=0)
    metrics(out, decoded, limit=.01)


def known_answer_payload(dtype, dim, keys, expcast, v_smooth, device="cpu"):
    """Two logits (0, -0.3), exactly represented V/means and poison padding.

    Bypass preprocessing to isolate the native probability/mean contract.
    Keep this separate from random full-pipeline checks using prepare().
    """
    queries, tiles = 65, (keys + 127) // 128
    q = torch.zeros(1, 1, queries, dim, device=device, dtype=torch.int8)
    k = torch.zeros(1, 1, keys, dim, device=device, dtype=torch.int8)
    q[..., 0], k[..., 0], k[:, :, 0, 0] = 1, -3, 0
    # Nonrepeating halves distinguish the two D128 PV fragments.
    channel = (torch.arange(dim, device=device) % 13 - 6).float() * .25
    channel[0], channel[1] = 1.5, -2.
    values = channel.view(1, 1, 1, dim).expand(1, 1, keys, dim).to(dtype)
    logical = torch.full((1, 1, dim, tiles * 128), 64., device=device)
    logical[..., :keys] = 0. if v_smooth else channel[:, None] * 2
    i = torch.arange(tiles * 128, device=device)
    r = i % 16
    permutation = (i // 16) * 16 + (r // 8) * 2 + ((r // 2) % 4) * 4 + r % 2
    packed = torch.empty_like(logical).index_copy_(-1, permutation, logical)
    means = ((channel * 2).view(1, 1, 1, dim).expand(1, 1, tiles, dim).half().contiguous()
             if v_smooth else torch.empty(0, device=device, dtype=torch.float16))
    state = dict(q=q, k=k, v=packed.to(torch.float8_e4m3fn),
                 qs=torch.ones(1, 1, 64, device=device),
                 ks=torch.ones(1, 1, tiles * 4, device=device),
                 vs=torch.full((1, 1, dim), .5, device=device), means=means,
                 expcast=expcast, block_values=v_smooth,
                 output_type=0 if dtype == torch.float16 else 1, scale=.1)
    # Independent scalar expression; no tiled reference or native result.
    ratio = 1.
    if not expcast and not v_smooth:
        peak = 2. ** float(torch.tensor(8.807, dtype=torch.float32))
        small = peak * math.exp(-.3)
        rounded = torch.tensor([peak, small]).to(torch.float8_e4m3fn).double()
        ratio = float((rounded[0] + (keys - 1) * rounded[1]) / (peak + (keys - 1) * small))
    answer = (channel * ratio).view(1, 1, 1, dim).expand_as(q).contiguous()
    return state, values, answer


@torch.no_grad()
def validate_known_answers(vc, reference, device, report, kernel_mode="vc31_unified"):
    for dtype in (torch.float16, torch.bfloat16):
        for dim in (64, 128):
            for nk in (1, 128, 129, 513):
                for smooth in (False, True):
                    for cast in (False, True):
                        case = dict(suite="known_answer", dtype=str(dtype), dim=dim,
                                    queries=65, keys=nk, expcast=cast, v_smooth=smooth)
                        row = dict(case)
                        report["active_case"] = row
                        print(f"Checking {kernel_mode} known answer:", json.dumps(case), flush=True)
                        state, values, answer = known_answer_payload(dtype, dim, nk, cast, smooth, device)
                        decoded = reference.reference(state, chunk=32)
                        torch.testing.assert_close(decoded, answer, rtol=2e-6, atol=2e-6)
                        out = vc.forward_prepared(state, kernel_mode=kernel_mode)
                        row["vs_analytic_answer"] = metrics(out, answer.to(dtype))
                        # Check ALL channels, including D128's second PV fragment.
                        torch.testing.assert_close(out, answer.to(dtype), rtol=.004, atol=.003,
                            msg=lambda detail: "Controlled-payload analytical answer mismatch\n" + detail)
                        row["constant_channels"] = check_constant_channels(
                            out, decoded.to(dtype), values, expcast=cast, v_smooth=smooth)
                        report["known_answers"].append(row)


@torch.no_grad()
def validate(vc, reference, device, report=None, kernel_mode="vc31_unified"):
    if report is None:
        report = {}
    report.update(status="running", kernel_mode=kernel_mode, cases=[], known_answers=[])
    required = 29 if kernel_mode == "vc31_unified" else 30
    if vc._extension().sage_vc_kernel_version < required:
        raise RuntimeError("Rebuild sageattention-local for " + kernel_mode)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = report["cases"]
    for dtype in (torch.float16, torch.bfloat16):
        for dim in (64, 128):
            # Single key, exact tile, masked tail, multiple tiles and multiple heads/batches.
            for nq, nk in ((1, 1), (65, 128), (65, 129), (193, 513)):
                torch.manual_seed(3100 + nq + dim)
                q = torch.randn(2, 2, nq, dim, device=device, dtype=dtype)
                k = torch.randn(2, 2, nk, dim, device=device, dtype=dtype)
                v = torch.randn_like(k)
                # Different means per tile/channel/head expose metadata addressing mistakes.
                v += (torch.arange(nk, device=device) // 128)[None, None, :, None] * 2
                v[..., 0], v[..., 1] = 1.5, -2.
                for smooth in (False, True):
                    report["active_case"] = dict(suite="random", stage="prepare", dtype=str(dtype),
                                                 dim=dim, queries=nq, keys=nk, v_smooth=smooth)
                    base = vc.prepare(q, k, v, v_smooth=smooth, v_scale_mode="channel")
                    for cast in (False, True):
                        case = dict(dtype=str(dtype), dim=dim, queries=nq, keys=nk,
                                    expcast=cast, v_smooth=smooth)
                        print(f"Checking {kernel_mode}:", json.dumps(case), flush=True)
                        row = dict(suite="random", **case)
                        report["active_case"] = row
                        state = {**base, "expcast": cast}
                        old_mode = "vc15" if smooth else "vc28_stream_i2f"
                        if kernel_mode != "vc31_unified":
                            old_mode = "vc31_unified"
                        row["previous_mode"] = old_mode
                        expected = vc.forward_prepared(state, kernel_mode=old_mode)
                        out = vc.forward_prepared(state, kernel_mode=kernel_mode)
                        decoded = reference.reference(state, chunk=32).to(dtype)
                        row.update(previous=metrics(out, expected), decoded_reference=metrics(out, decoded),
                                   constant_channels=constant_channel_diagnostics(out, decoded, v, expected))
                        # Log before any assertion, so failures retain actual/expected values.
                        print(json.dumps(row), flush=True)
                        check_random_output(out, expected, decoded, v_smooth=smooth)
                        # Prepared API supports a caller stream; preprocessing is on the default stream.
                        caller = torch.cuda.current_stream(device)
                        worker = torch.cuda.Stream(device=device)
                        worker.wait_stream(caller)
                        with torch.cuda.stream(worker):
                            again = vc.forward_prepared(state, kernel_mode=kernel_mode)
                        caller.wait_stream(worker)
                        again.record_stream(caller)
                        torch.testing.assert_close(again, out, rtol=0, atol=0)
                        results.append(row)
    validate_known_answers(vc, reference, device, report, kernel_mode)
    report.pop("active_case", None)
    report.update(status="passed", scope=(
        "Random: all rows, default/side streams, INT8 QK/FP8 PV; 0.1% prior-implementation, "
        "bitwise unsmoothed, 1% decoded-reference gates. Controlled: independent analytic "
        "answers, all-channel rtol=0.004/atol=0.003. Not model quality."))
    return report


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel-mode", choices=("vc31_unified", "vc32_full_pv", "vc32_packed_mean"), default="vc31_unified")
    p.add_argument("--output", type=Path, default=Path("results/sage_vc31_validation.json"))
    args = p.parse_args()
    from sageattention_local import vc, vc_reference
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Hopper required; no CPU fallback")
    report = {}
    try:
        validate(vc, vc_reference, torch.device("cuda:0"), report, kernel_mode=args.kernel_mode)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_report(args.output, report)
        print(args.kernel_mode, "failure diagnostics saved:", args.output, flush=True)
        raise
    save_report(args.output, report)
    print(args.kernel_mode, "validation passed:", args.output)


if __name__ == "__main__":
    main()
