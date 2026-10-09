"""Hopper integration gates for the unified vc28 + V-Smooth candidate."""
import argparse
import json
from pathlib import Path
import torch

from profile_int8_h200 import metrics


def check_constant_channels(out, decoded, values, *, expcast, v_smooth):
    """Check the actual probability/normalizer contract, not dense softmax's.

    Ordinary Sage uses FP8(P) in PV but unrounded P in the denominator.
    With constant V=c, its answer is c_q * sum(FP8(P)) / sum(P), not c.
    ExpCast sums the decoded P in both places. With V smoothing, these
    constant channels have zero residual and are restored by the mean term.
    """
    actual = out[..., :2]
    target = values[:, :, :1, :2].expand_as(actual)
    torch.testing.assert_close(actual, decoded[..., :2], rtol=.004, atol=.003,
                               msg="Constant channels differ from decoded quantized reference")
    preserves_constant = expcast or v_smooth
    if preserves_constant:
        torch.testing.assert_close(actual, target, rtol=.004, atol=.003,
                                   msg="ExpCast/mean restoration failed to preserve constant channels")
    return {
        "contract": "constant_preservation" if preserves_constant else "rounded_P_over_unrounded_mass",
        "vs_original_values": metrics(actual, target),
        "vs_decoded_reference": metrics(actual, decoded[..., :2]),
    }


@torch.no_grad()
def validate(vc, reference, device):
    if vc._extension().sage_vc_kernel_version < 29:
        raise RuntimeError("Rebuild sageattention-local vc31 first")
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
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
                    base = vc.prepare(q, k, v, v_smooth=smooth, v_scale_mode="channel")
                    for cast in (False, True):
                        case = dict(dtype=str(dtype), dim=dim, queries=nq, keys=nk,
                                    expcast=cast, v_smooth=smooth)
                        print("Checking vc31:", json.dumps(case), flush=True)
                        state = {**base, "expcast": cast}
                        old_mode = "vc15" if smooth else "vc28_stream_i2f"
                        expected = vc.forward_prepared(state, kernel_mode=old_mode)
                        out = vc.forward_prepared(state, kernel_mode="vc31_unified")
                        old_error = metrics(out, expected, limit=.001)
                        if not smooth:
                            torch.testing.assert_close(out, expected, rtol=0, atol=0)
                        decoded = reference.reference(state, chunk=32).to(dtype)
                        decoded_error = metrics(out, decoded, limit=.01)
                        constant_check = check_constant_channels(
                            out, decoded, v, expcast=cast, v_smooth=smooth)
                        # Prepared API supports a caller stream; preprocessing is on the default stream.
                        caller = torch.cuda.current_stream(device)
                        worker = torch.cuda.Stream(device=device)
                        worker.wait_stream(caller)
                        with torch.cuda.stream(worker):
                            again = vc.forward_prepared(state, kernel_mode="vc31_unified")
                        caller.wait_stream(worker)
                        again.record_stream(caller)
                        torch.testing.assert_close(again, out, rtol=0, atol=0)
                        row = dict(**case, previous=old_error, decoded_reference=decoded_error,
                                   constant_channels=constant_check)
                        results.append(row)
                        print(json.dumps(row), flush=True)
    return {"status": "passed", "cases": results,
            "scope": "All rows, default/side streams, INT8 QK/FP8 PV. 0.1% prior-implementation and 1% decoded-reference gates; not model quality."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/sage_vc31_validation.json"))
    args = p.parse_args()
    from sageattention_local import vc, vc_reference
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Hopper required; no CPU fallback")
    report = validate(vc, vc_reference, torch.device("cuda:0"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("vc31 validation passed:", args.output)


if __name__ == "__main__":
    main()
