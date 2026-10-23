"""CPU oracle for an experimental binary16 adaptation of ExpCast.

This is not the paper's E4M3 implementation. Its normalized Mitchell offset
(-0.35 / 8) is adapted to the ten binary16 mantissa bits. Encoding in a 2**12
scaled domain, followed by FP16 multiplication by 2**-12, avoids prematurely
discarding the FP16 subnormal probability tail. Two separate FP32 arithmetic
roundings are intentional; no FMA is used. Online-history exp stays upstream.
"""

GAIN = 1477.3197021484375  # float32(1024 * log2(e))
BIAS = 27603.19921875     # float32((15 + 12 - 0.35 / 8) * 1024)
CODE_MAX = 27648         # (15 + 12) * 1024; binary16 value 4096
OUTPUT_SCALE = 0.000244140625  # 2**-12, exactly representable in binary16
CONTRACT = {
    "version": "laser_binary16_expcast_scaled12_v1",
    "input": "FP16 shifted scores <= 0; finite or negative infinity",
    "gain_fp32": GAIN,
    "bias_fp32": BIAS,
    "affine_rounding": "separate FP32 multiply then FP32 add, not FMA",
    "integer_rounding": "round to nearest, ties to even",
    "code_range": [0, CODE_MAX],
    "code_format": "positive IEEE binary16 bits, 5 exponent and 10 fraction bits",
    "output_scale": OUTPUT_SCALE,
    "output_rounding": "FP16 multiply by 2^-12, including rounded subnormals",
    "bias_origin": "paper beta -0.35 / 8 adapted to 1024 codes per binade",
    "paper_exact": False,
    "fp8": False,
    "history": "unchanged original Laser exponential rescaling",
    "recurrence": "approximate map is not multiplicative; online partition matters",
    "reference_scope": "CPU numerical specification; remote CCE/device validation required",
}


def encode(shifted_scores):
    """Return scaled-domain FP16 bit codes; negative-infinity masks map to 0."""
    import torch
    if shifted_scores.device.type != "cpu" or shifted_scores.dtype != torch.float16:
        raise ValueError("FP16 ExpCast oracle requires CPU FP16 shifted scores")
    if torch.isnan(shifted_scores).any() or (shifted_scores > 0).any():
        raise ValueError("Shifted scores must be nonpositive and cannot contain NaN")
    product = shifted_scores.float() * GAIN
    affine = product + BIAS
    return affine.clamp(0, CODE_MAX).round().to(torch.int16)


def decode(codes):
    """Reinterpret binary16 bits and undo scaling with FP16 output rounding."""
    import torch
    if codes.device.type != "cpu" or codes.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("FP16 ExpCast decode requires CPU integer codes")
    if ((codes < 0) | (codes > CODE_MAX)).any():
        raise ValueError("FP16 ExpCast code is outside [0, 27648]")
    scaled = codes.to(torch.int16).contiguous().view(torch.float16)
    # FP32 represents both operands and their exact product here. The final
    # cast models the FP16 vmuls result, particularly its subnormal rounding.
    return (scaled.float() * OUTPUT_SCALE).half()


def probability(shifted_scores):
    """Approximate unnormalized FP16 probability, with no common scale left."""
    return decode(encode(shifted_scores))


expcast_codes = encode
decode_codes = decode
expcast_probability = probability
