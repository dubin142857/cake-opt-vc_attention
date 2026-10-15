"""Exact, reversible probability-only transform for pinned MindIE Laser source.

This module does not write a checkout or alter the installed operator. The
preparation driver owns variant isolation, provenance and copied helper headers.
"""

HEADER_NAME = "vc_laser_probability.h"
CONTRACT = "laser_fp16_scores_fp32_mul_add_rne_e4m3_decode_div256_v1"
INCLUDE_ANCHOR = b'#include "AddressMappingVectorForwardOnline.h"\n'
INCLUDE_ADDITION = b'#include "vc_laser_probability.h"\n'
BULK_VEXP = b"""    vexp(cur_buf_for_attn_score_fp16, cur_buf_for_attn_score_fp16, param.section_block_num * rowPerTime *
        BASE_BLOCK_SIDE_LEN / BLOCK_NUM_8 / data_num_per_blk,   // 5 * 128 * 8   / 64    5 * 2 * 8
        1, 1, 8, 8);
    pipe_barrier(PIPE_V);"""
LEGACY_EXPERIMENTAL_PROBABILITY = b"""    // ExpCast only: reuse completed max-reduction scratch, then rebuild it for row sum.
    static_assert(MAX_LENG_PER_UB_PROC * SIZE_FP16 >= 8192 * sizeof(float),
                  "ExpCast needs 32 KiB of existing row-max scratch");
    static_assert(BASE_BLOCK_SIDE_LEN == 128, "ExpCast chunks require 128-element blocks");
    vc_laser_expcast_probability(
        reinterpret_cast<__ubuf__ half *>(cur_buf_for_attn_score_fp16),
        reinterpret_cast<__ubuf__ float *>(ub_attn.buf_for_cacl_rowmax_fp16),
        param.section_block_num * rowPerTime * BASE_BLOCK_SIDE_LEN);"""
# SIZE_FP16 is an upstream instance member, not a constant expression. Keep
# the original allocation and use the instantiated element type for this check.
# Retain the exact v1 bytes solely to recognize and migrate generated sources.
EXPERIMENTAL_PROBABILITY = LEGACY_EXPERIMENTAL_PROBABILITY.replace(
    b"    static_assert(MAX_LENG_PER_UB_PROC * SIZE_FP16 >= 8192 * sizeof(float),",
    b"    static_assert(sizeof(INPUT_T) == 2, \"ExpCast requires 16-bit Laser input\");\n"
    b"    static_assert(MAX_LENG_PER_UB_PROC * sizeof(INPUT_T) >= 8192 * sizeof(float),",
)


def _anchors(source: bytes):
    # Preserve the caller's exact newline convention rather than normalizing
    # the complete upstream file. Mixed newlines fail the exact anchor check.
    newline = b"\r\n" if b"\r\n" in source else b"\n"
    return tuple(x.replace(b"\n", newline) for x in
                 (INCLUDE_ANCHOR, INCLUDE_ADDITION, BULK_VEXP, EXPERIMENTAL_PROBABILITY))


def patch_vector_forward(source: bytes) -> bytes:
    include, addition, original, replacement = _anchors(source)
    if addition in source or b"vc_laser_expcast_probability(" in source:
        raise ValueError("Laser ExpCast probability patch was already applied")
    if source.count(include) != 1 or source.count(original) != 1:
        raise ValueError("Pinned Laser bulk-vexp/include anchors changed; refusing a broad replacement")
    return source.replace(include, include + addition, 1).replace(original, replacement, 1)


# Preparation driver's concise entry point.
patch_vector = patch_vector_forward


def unpatch_vector_forward(source: bytes) -> bytes:
    """Recover the exact input bytes for provenance checks (not a source writer)."""
    include, addition, original, replacement = _anchors(source)
    if source.count(include + addition) != 1 or source.count(replacement) != 1:
        raise ValueError("Laser ExpCast inverse anchors changed")
    return source.replace(include + addition, include, 1).replace(replacement, original, 1)
