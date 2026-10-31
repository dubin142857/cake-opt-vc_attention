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
#  ★CHUNK 是内核常量 vc910::map::CHUNK 的 python 镜像 —— 又一处"同一个量写两处"。
#    内核把一个 chunk 内的全部 PV 在 L0C 上累加, 于是**在线更新的粒度由 K_TILE 变成
#    K_TILE * CHUNK**。参考的窗口按定义要跟随内核 (见 payload_reference 的文档串),
#    否则 decoded_reference 不再是"隔离实现误差"的尺子。
#    2026-10-29 实撞: 漏改它使 expcast 在 3 个 tile 一组时报 1.87% 超闸 —— 那不是实现错,
#    是尺子与被测者用了不同的窗口 (F190)。普通 exp 路径对窗口不变, 故只有 expcast 受影响。
CHUNK = 4
ONLINE_TILE = K_TILE * CHUNK
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
def payload_reference(payload, *, expcast=True, query_indices=None, tile_width=None,
                      score_dtype=None, deq_trunc_bits=None):
    """All keys, optionally selected queries; decoded operands and FP32 matmuls.

    tile_width 是**在线更新的粒度**, 默认取 ONLINE_TILE = K_TILE * CHUNK ——
    内核把一个 chunk 内的全部 PV 在 L0C 上累加, 故它的归一化窗口是整个 chunk,
    不是单个 matmul tile。
    ★它不是自由参数: ExpCast 的 E4M3 编码相对**当前 tile 的行最大值**定义
    (低于 -120/EXPCAST_GAIN 的分数一律压成索引 0), 所以换一个宽度就是换一个数值契约。
    本函数要作为内核的同契约 oracle, 宽度必须与 vc_tiling.h 的 N 相同。
    量化块仍是 K_BLOCK: 一个 tile 覆盖 tile_width/K_BLOCK 个块, 各自的 scale 按列分段施加,
    与内核一致。普通 exp 路径对宽度不变, 故只有 ExpCast 受影响。

    score_dtype 是**趟 1 与趟 2 之间那份分数暂存的精度**, 默认 None = fp32 (现行)。
    内核的 scoreBuf 按 chunk 整存, 在 tile_m=128 上它要 131072 B, 是 UB 预算里
    唯一压得动的大头; 给它换 torch.float16 就是在这里离线量出代价 (不碰内核)。
    ★取整必须排在行最大值**之前**: 否则 fp16 可能向上取整到超过 fp32 算出的基准,
    scores - new_m 变正, ExpCast 那条'上界永不触到'的不变式 (内核据此删掉了
    Mins(sc,120)) 就不再成立。此处模型与内核要做的事一致 —— 存完再求最大值。

    deq_trunc_bits 是**反量化 scale 被硬件截掉的低位尾数位数**, 默认 None = 不截 (现行)。
    ★它只在"把反量化挪进 fixpipe"这条路上才有意义, 且**不是可选项**:
      F233 实测确证, 本机 (`__NPU_ARCH__` 2201) 的 fixpipe 反量化把那条 fp32 scale
      的低 **13** 位尾数**截断**(不是进位), 只用 bits[31:13] —— 剩 10 位尾数,
      恰是 fp16 的尾数宽度。传 13 就是在离线量这条路的数值代价 (不碰内核)。
    ★截的是**合成之后**的那个数 (qscale * kscale * scale), 因为进 deq 项的正是它。
      而 qscale 在一个 32 行的量化块内恒定, 故逐元素截与按块截同果。
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
    width = ONLINE_TILE if tile_width is None else tile_width
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
            if deq_trunc_bits:
                #  硬件把 fp32 的低 deq_trunc_bits 位尾数**截断**。按位与, 不是四舍五入 ——
                #  F233 在 8 个候选模型里把"截断"与"就近偶数"分开过: 截断 0/8192 不符,
                #  就近偶数 3528/8192 不符。
                mask = -(1 << deq_trunc_bits)          # int32 下即 0xFFFF_E000 (13 位时)
                g = (g.float().view(torch.int32) & mask).view(torch.float32)
            segments.append(g.unsqueeze(-1).expand(*g.shape, span))
        scores *= torch.cat(segments, dim=-1)
        if score_dtype is not None:
            #  趟 1 写进 scoreBuf、趟 2 再读出来 —— 取整在这里, 且排在求最大值之前。
            scores = scores.to(score_dtype).float()
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
