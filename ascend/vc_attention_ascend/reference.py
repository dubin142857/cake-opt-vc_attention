"""Independent, explicit CPU/CUDA/NPU numerical contract (never an API fallback)."""

import math
import torch

Q_BLOCK = 32
K_BLOCK = 128
#  内核的 K 方向 matmul tile 宽度。必须与 ascend/csrc/vc_tiling.h 的 N 一致,
#  且为 K_BLOCK 的整数倍 —— K/V 按它填充, 而量化块仍按 K_BLOCK 切。
#  ★K_TILE 是内核常量 N 的 python 镜像 —— 同一个量写两处, 改一处必须改另一处。
#    2026-10-29 由 256 改 128, 随手写 cube 的几何 (M=64 x N=128)。
#    它决定 K/V 的填充粒度, 与 K_BLOCK (量化块, 128) 不是一回事。
K_TILE = 128
assert K_TILE % K_BLOCK == 0
EXPCAST_GAIN = 11.541560173034668  # FP32(8 * log2(e))


def e4m3_table(device="cpu"):
    # Only positive, finite codes needed by ExpCast, 0..120. Pad to 128 entries.
    values = []
    for code in range(128):
        exponent, fraction = code >> 3, code & 7
        values.append(fraction / 512 if exponent == 0 else (1 + fraction / 8) * 2.0 ** (exponent - 7))
    return torch.tensor(values, dtype=torch.float32, device=device)


def expcast_decode(shifted):
    # Deliberately TWO FP32 roundings, as Ascend Muls + Adds. Not CUDA FMA.
    code = torch.round(shifted.float() * EXPCAST_GAIN + 120.0).clamp(0, 120).long()
    return e4m3_table(shifted.device)[code]


def quantize_e4m3_decoded(x):
    """Finite saturation + RNE, returning FP32 decoded E4M3; no float8 support needed."""
    a = x.float().abs().clamp(max=448.0)
    exponent = torch.floor(torch.log2(a.clamp_min(2.0 ** -6))).clamp(-6, 8)
    step = torch.pow(2.0, exponent - 3)
    return (torch.round(a / step) * step).clamp(max=448.0) * x.sign()


@torch.no_grad()
def payload_reference(payload, *, expcast=True, query_indices=None, tile_width=None):
    """All keys, optionally selected queries; decoded operands and FP32 matmuls.

    tile_width 是**在线更新的粒度**, 默认取 K_TILE, 即内核的 K 方向 matmul tile 宽度。
    ★它不是自由参数: ExpCast 的 E4M3 编码相对**当前 tile 的行最大值**定义
    (低于 -120/EXPCAST_GAIN 的分数一律压成索引 0), 所以换一个宽度就是换一个数值契约。
    本函数要作为内核的同契约 oracle, 宽度必须与 vc_tiling.h 的 N 相同。
    量化块仍是 K_BLOCK: 一个 tile 覆盖 tile_width/K_BLOCK 个块, 各自的 scale 按列分段施加,
    与内核一致。普通 exp 路径对宽度不变, 故只有 ExpCast 受影响。
    """
    q, k, v = payload["q"].float(), payload["k"].float(), payload["v"].float()
    qs, ks, vs = payload["q_scale"], payload["k_scale"], payload["v_scale"]
    n, d = payload["queries"], q.shape[-1]
    ids = torch.arange(n, device=q.device) if query_indices is None else query_indices.to(q.device)
    q = q.index_select(2, ids)
    qscale = qs[..., 0].index_select(2, ids // Q_BLOCK)
    shape = (*q.shape[:-1], 1)
    m = torch.full(shape, -float("inf"), device=q.device)
    mass = torch.zeros(shape, device=q.device)
    out = torch.zeros_like(q)
    width = K_TILE if tile_width is None else tile_width
    if width % K_BLOCK:
        raise ValueError("tile_width must cover whole K quantization blocks")
    for begin in range(0, payload["keys"], width):
        count = min(width, payload["keys"] - begin)
        scores = q @ k[:, :, begin:begin + count].transpose(-1, -2)
        # Match native scalar multiplication order; QK dot product is exact.
        # 一个 tile 可跨多个量化块, 逐块取自己的 scale 并按列拼接, 与内核的列分段施加一致。
        segments = []
        for sub in range(0, count, K_BLOCK):
            block = (begin + sub) // K_BLOCK
            span = min(K_BLOCK, count - sub)
            g = (qscale * ks[:, :, block, 0].unsqueeze(-1)) * payload["scale"]
            segments.append(g.unsqueeze(-1).expand(*g.shape, span))
        scores *= torch.cat(segments, dim=-1)
        new_m = torch.maximum(m, scores.amax(-1, keepdim=True))
        alpha = torch.exp(m - new_m)
        shifted = scores - new_m
        if expcast:
            p = expcast_decode(shifted)
            tile_mass = p.sum(-1, keepdim=True)
        else:
            p = torch.exp(shifted)
            tile_mass = p.sum(-1, keepdim=True)
            p = p.half().float()  # PV uses FP16; denominator is pre-cast FP32.
        out = out * alpha + p @ v[:, :, begin:begin + count]
        mass = mass * alpha + tile_mass
        m = new_m
    return out / mass * vs.unsqueeze(2)


@torch.no_grad()
def dense_reference(q, k, v, query_indices=None, chunk=32):
    """HND inputs. Chunk queries so the long-shape reference stays bounded."""
    if query_indices is not None:
        q = q.index_select(2, query_indices.to(q.device))
    pieces = []
    for begin in range(0, q.shape[2], chunk):
        logits = (q[:, :, begin:begin + chunk].float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        pieces.append(logits.softmax(-1) @ v.float())
    return torch.cat(pieces, dim=2)
