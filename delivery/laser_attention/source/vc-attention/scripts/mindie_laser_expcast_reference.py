"""CPU probability oracle for the Laser-only ExpCast experiment.

This is not a dense-attention quality threshold or a model of Laser's Cube
rounding. Laser supplies FP16 shifted scores; its online-history exp remains
unchanged. Separate FP32 multiply and add deliberately match the CCE sequence.
"""
import torch

GAIN = 11.541560173034668
BIAS = 119.6500015258789
CONTRACT = "laser_fp16_scores_fp32_mul_add_rne_e4m3_decode_div256_v1"


def expcast_codes(shifted_scores):
    """Clamp/RNE code for FP16 scores; -inf is permitted as a masked value."""
    if shifted_scores.device.type != "cpu" or shifted_scores.dtype != torch.float16:
        raise ValueError("Laser ExpCast oracle expects CPU FP16 shifted scores")
    if torch.isnan(shifted_scores).any() or (shifted_scores > 0).any():
        raise ValueError("Shifted scores must be nonpositive and cannot contain NaN")
    product = shifted_scores.float() * GAIN
    affine = product + BIAS
    return affine.clamp(0, 120).round().to(torch.int16)


def decode_codes(codes):
    """Return exact FP16 decode(E4M3(code))/256 via the bit-layout identity."""
    if codes.device.type != "cpu" or codes.dtype not in (torch.uint8, torch.int16,
                                                         torch.int32, torch.int64):
        raise ValueError("Expected CPU integer E4M3 codes")
    if ((codes < 0) | (codes > 120)).any():
        raise ValueError("Laser probability codes must be in [0, 120]")
    return (codes.to(torch.int16) << 7).contiguous().view(torch.float16)


def expcast_probability(shifted_scores):
    """Approximate unnormalized probability for the existing FP16 PV path."""
    return decode_codes(expcast_codes(shifted_scores))
