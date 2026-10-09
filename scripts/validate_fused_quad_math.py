"""Bitwise helper gates before H200 attention timing; no pytest dependency."""
import torch


def inputs(pattern, steps, device):
    gen = torch.Generator(device=device).manual_seed(3600+steps)
    x = torch.randint(-(2**21), 2**21+1, (steps, 2, 128, 8, 8),
                      device=device, dtype=torch.int32, generator=gen)
    scale = (torch.arange(128, device=device).remainder(4)+1).float().expand(steps, 2, 128).contiguous()*1e-6
    initial = torch.full((2, 128, 2), -5e6, device=device)
    if pattern == "stable":
        x[:] = x[:1].clone()
    elif pattern == "fallback":
        scale *= 1000
    elif pattern == "tiny":
        scale.fill_(1e-38)
    elif pattern == "mixed":
        scale[::2] = 1e-38
    elif pattern in ("codes", "rounding_ties"):
        x.copy_(-(torch.arange(x.numel(), device=device).remainder(129).reshape_as(x)))
        scale.fill_(1./11.541560173034668)
        initial.fill_(0. if pattern == "codes" else (119.6500015258789-119.5)/11.541560173034668)
    elif pattern in ("all_zero", "all_max"):
        x.zero_()
        initial.fill_(127.5 if pattern == "all_zero" else 0.)
    elif pattern != "random":
        raise ValueError("Unknown input pattern: " + pattern)
    return x, scale, initial


@torch.no_grad()
def check_case(ext, pattern, steps, device, side_stream=False):
    args = inputs(pattern, steps, device)
    old = ext.check_probability(*args, False)
    stream = torch.cuda.Stream(device=device) if side_stream else torch.cuda.current_stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        new = ext.check_probability(*args, True)
    stream.synchronize()
    for label, a, b in zip(("P bytes", "max/alpha/mass"), new, old):
        if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
            raise AssertionError(f"Fused quad changed {label}: {pattern}, {steps} tiles, side_stream={side_stream}")
    if not torch.isfinite(new[1]).all():
        raise AssertionError("Nonfinite probability state")
    codes = new[0].view(torch.uint8)
    if (codes > 120).any():
        raise AssertionError("Positive E4M3 probability range violated")
    decoded = codes.view(torch.float8_e4m3fn).float().reshape(steps, 2, 128, 4, 2, 2, 4)
    # FP64 oracle, independent of the pair-chain addition order.
    oracle = decoded.double().sum((3, 4, 6)).float()
    torch.testing.assert_close(new[1][..., 4:6], oracle, rtol=0, atol=0)
    if pattern == "codes":
        torch.testing.assert_close(codes.unique().cpu(), torch.arange(121, dtype=torch.uint8), rtol=0, atol=0)
    if pattern == "rounding_ties" and (codes.remainder(2) != 0).any():
        raise AssertionError("Constructed .5 boundaries did not round to even")
    if pattern == "all_zero" and torch.count_nonzero(new[1][..., 4:6]):
        raise AssertionError("All-zero tile has nonzero mass")
    if pattern == "all_max":
        torch.testing.assert_close(new[1][..., 4:6], torch.full_like(oracle, 8192.), rtol=0, atol=0)
    return {"pattern": pattern, "tiles": steps, "side_stream": side_stream,
            "status": "passed", "P_and_state_bitwise_equal": True, "fp64_mass_oracle_exact": True}


@torch.no_grad()
def validate(ext, device, progress=lambda rows: None):
    rows = []
    cases = [(p, s) for p in ("random", "stable", "fallback", "tiny", "mixed") for s in (1, 7, 296)]
    cases += [(p, 1) for p in ("codes", "rounding_ties", "all_zero", "all_max")]
    for pattern, steps in cases:
        for side in (False, True):
            row = {"pattern": pattern, "tiles": steps, "side_stream": side, "status": "running"}
            rows.append(row); progress(rows)
            row.update(check_case(ext, pattern, steps, device, side)); progress(rows)
    native = torch.cuda.get_device_capability(device) >= (8, 9)
    print(f"Fused quad probability: {len(rows)} exact cases passed; native FP8 decode={native}", flush=True)
    return {"status": "passed", "native_fp8_decode": native, "cases": rows}
