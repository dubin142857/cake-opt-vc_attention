"""Boundary/scale-index validation for the actual Hopper scale-prefetch kernel."""
import torch
from profile_int8_h200 import metrics


def assert_exact(actual, expected):
    result = metrics(actual, expected, limit=0.)
    # Compare bits too (signed zero); this experiment changes no arithmetic.
    result["exact_match"] = torch.equal(actual.contiguous().view(torch.uint8),
                                       expected.contiguous().view(torch.uint8))
    if not result["exact_match"]:
        raise AssertionError(f"Scale prefetch changed output bits: {result}")
    return result


@torch.no_grad()
def validate(ext, backend, vc, reference, device, progress):
    rows = []
    gen = torch.Generator(device=device).manual_seed(3700)
    for dtype in (torch.float16, torch.bfloat16):
        for dim in (64, 128):
            # No main tile, full tail, partial tail, repeated main tiles, and
            # the real long-sequence tile count. Every lane/tile has its own
            # scale so stale/advanced scale reads cannot hide behind constants.
            for n in (1, 128, 129, 256, 257, 513, 37741):
                qkv = tuple(torch.randn(1, 2, length, dim, device=device,
                                       dtype=dtype, generator=gen) for length in (65, n, n))
                state = vc.prepare(*qkv, expcast=True, v_smooth=False)
                ks = state["ks"].clone().view(1, 2, -1, 4)
                factors = torch.tensor([.5, 1., 2., .75, 1.5], device=device)
                blocks = ks.shape[2]
                ks *= factors[torch.arange(blocks, device=device) % len(factors)][None, None, :, None]
                ks *= torch.tensor([.5, .75, 1.25, 1.5], device=device)
                state = {**state, "ks": ks.reshape_as(state["ks"])}
                row = {"dtype": str(dtype), "dim": dim, "queries": 65,
                       "keys": n, "scale_pattern": "distinct lanes and alternating tiles", "status": "running"}
                rows.append(row); progress(rows)
                print("Validate scale prefetch", str(dtype), dim, n, flush=True)
                old = backend.forward(ext, state, False)
                out = backend.forward(ext, state, True)
                row["same_payload"] = assert_exact(out, old)
                oracle = reference.reference(state, chunk=32).to(dtype)
                row["decoded_reference"] = metrics(out, oracle, limit=.01)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    side = backend.forward(ext, state, True)
                stream.synchronize()
                row["nondefault_stream"] = assert_exact(side, out)
                row["status"] = "passed"; progress(rows)
    return rows
