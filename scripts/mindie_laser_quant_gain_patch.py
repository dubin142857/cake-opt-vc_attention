"""Prepare scalar dequantization bits ahead of QK loads without moving FIX state.

Apply after the scale-reuse transform. Only ordinary FP32 multiplication and
bit masking move: set_quant_pre, its PIPE_FIX barrier and the DEQF16 transfer
stay at the same point relative to MAD, flags and skipped tiles.
"""
from __future__ import annotations

from scripts.mindie_laser_quant_patch import once
from scripts.mindie_laser_quant_reuse_patch import QK_BEGIN, RUN_BEGIN, TILING

OLD_SIGNATURE = b"float q_scale_first, float q_scale_second, float k_scale"
NEW_SIGNATURE = b"uint32_t gain_bits_first, uint32_t gain_bits_second"
K_READ = b"                const float k_scale = ReadQuantScale(k_scales_ + k_scale_index);"
PREPARE_BITS = b"""
                // Prepare both scalar coefficients before the K transfer/MAD.
                // Hardware FIX state is still written only at the original
                // guarded output point; no synchronization is moved or removed.
                float gain_first = q_scale_first * k_scale;
                float gain_second = q_scale_second * k_scale;
                const uint32_t gain_bits_first =
                    *reinterpret_cast<uint32_t *>(&gain_first) & 0xffffe000u;
                const uint32_t gain_bits_second =
                    *reinterpret_cast<uint32_t *>(&gain_second) & 0xffffe000u;"""
OLD_GAIN = b"""                float gain = (m_offset == 0 ? q_scale_first : q_scale_second) * k_scale;
                uint32_t bits = *reinterpret_cast<uint32_t *>(&gain) & 0xffffe000u;"""
NEW_GAIN = b"                uint32_t bits = m_offset == 0 ? gain_bits_first : gain_bits_second;"


def patch_cube(data: bytes) -> bytes:
    if data.count(TILING) != 1 or data.count(b"address.set_tiling(") != 1:
        raise ValueError("Gain preparation requires the pinned 64-block, ky=2 address mapping")
    data = once(data, b"        " + OLD_SIGNATURE + b");",
                b"        " + NEW_SIGNATURE + b");")
    start = data.index(QK_BEGIN)
    end = data.index(RUN_BEGIN, start)
    chunk = data[start:end]
    chunk = once(chunk, K_READ, K_READ + PREPARE_BITS)
    chunk = once(chunk,
        b"                    q_scale_first, q_scale_second, k_scale);",
        b"                    gain_bits_first, gain_bits_second);")
    chunk = once(chunk, b"    " + OLD_SIGNATURE + b")", b"    " + NEW_SIGNATURE + b")")
    chunk = once(chunk, OLD_GAIN, NEW_GAIN)
    return data[:start] + chunk + data[end:]


def apply(relative_path: str, data: bytes) -> bytes:
    return patch_cube(data) if relative_path.rsplit("/", 1)[-1] == "CubeForward.h" else data
