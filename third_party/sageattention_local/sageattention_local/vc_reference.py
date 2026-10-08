"""Independent decoded-payload FP32 oracle, for diagnostics only (never dispatched).

Quantization is done on full Q/K/V first; selected queries retain their original
INT8 or FP8 block scale indices. Uses all keys in 128-token blocks, never N x N.
"""
import torch


def expcast_codes(shifted):
    u = (shifted.double() * 11.541560173034668 + 119.6500015258789).float()
    return u.round().clamp(0, 120).to(torch.uint8)


def value_permutation(n, device):
    i = torch.arange(n, device=device)
    r = i % 16
    return (i // 16) * 16 + (r // 8) * 2 + ((r // 2) % 4) * 4 + r % 2


@torch.no_grad()
def reference(state, query_indices=None, chunk=32):
    q, k, v = (state[x] for x in ("q", "k", "v"))
    fp8_qk = state.get("qk_format") == "fp8_block"
    ids = torch.arange(q.shape[2], device=q.device) if query_indices is None else torch.as_tensor(query_indices, device=q.device, dtype=torch.long)
    if chunk < 1 or ids.numel() == 0:
        raise ValueError("Reference requires positive chunk and nonempty queries")
    v = v.float().index_select(-1, value_permutation(v.shape[-1], v.device)).transpose(-1, -2)
    outputs = []
    tiles = (k.shape[2] + 127) // 128
    scale = torch.tensor(state["scale"], device=q.device, dtype=torch.float32)
    if not state["expcast"]:
        scale = scale * torch.tensor(1.4426950408889634, device=q.device, dtype=torch.float32)
    for begin in range(0, ids.numel(), chunk):
        rows = ids[begin:begin + chunk]
        query = q.index_select(2, rows).float()
        qs = state["qs"].index_select(2, rows // 64 if fp8_qk else (rows // 16) * 8 + rows % 8).unsqueeze(-1)
        maximum = torch.full((*query.shape[:-1], 1), -5e6, device=q.device)
        denom = torch.zeros_like(maximum)
        out = torch.zeros_like(query)
        for tile in range(tiles):
            start, end = tile * 128, min((tile + 1) * 128, k.shape[2])
            keys = torch.arange(start, end, device=q.device)
            ks = (state["ks"][:, :, tile, None, None] if fp8_qk else
                  state["ks"].index_select(2, tile * 4 + (keys % 8) // 2).unsqueeze(-2))
            raw = query @ k[:, :, start:end].float().transpose(-1, -2)
            dequant = qs * ks
            base = raw * dequant if tile == tiles - 1 else raw
            factor = scale if tile == tiles - 1 else scale * dequant
            logits = base * factor
            if state["expcast"]:
                new_max = torch.maximum(maximum, logits.amax(-1, keepdim=True))
                alpha = (maximum - new_max).exp()
                p = expcast_codes(logits - new_max).view(torch.float8_e4m3fn).float()
                if fp8_qk:
                    # vc15 folded affine encoding: FP32 multiply, then one FMA.
                    # Its fallback is warp-wide. On extreme ranges this oracle
                    # uses the unfused path per row; differences near byte ties
                    # are included in the native/decoded-oracle error budget.
                    a = torch.tensor(11.541560173034668, device=q.device)
                    b = torch.tensor(119.6500015258789, device=q.device)
                    gain = factor * a
                    bias = (-new_max.double() * a.double() + b.double()).float()
                    affine = (base.double() * gain.double() + bias.double()).float()
                    safe = (factor >= torch.finfo(torch.float32).tiny) & torch.isfinite(gain) & (new_max.abs() <= 128)
                    codes = affine.round().clamp(0, 120).to(torch.uint8)
                    p = torch.where(safe, codes.view(torch.float8_e4m3fn).float(), p)
                denominator_p = p
            else:
                # Sage offset is applied by a single FP32 FMA before the row maximum.
                local_max = (base.double() * factor.double() - float(torch.tensor(8.807, dtype=torch.float32))).float().amax(-1, keepdim=True)
                new_max = torch.maximum(maximum, local_max)
                alpha = torch.exp2(maximum - new_max)
                shifted = (base.double() * factor.double() - new_max.double()).float()
                denominator_p = torch.exp2(shifted)
                p = denominator_p.to(torch.float8_e4m3fn).float()
            mass = denominator_p.sum(-1, keepdim=True)
            product = p @ v[:, :, start:end]
            if state["block_values"]:
                contribution = product + mass * state["means"][:, :, tile].float().unsqueeze(-2)
                if state["vs"].ndim == 4:
                    contribution *= state["vs"][:, :, tile].unsqueeze(-2)
            else:
                contribution = product
            out = out * alpha + contribution
            denom = denom * alpha + mass
            maximum = new_max
        out = out / denom
        if state["vs"].ndim == 3:
            out *= state["vs"].unsqueeze(-2)
        outputs.append(out)
    return torch.cat(outputs, dim=2)
