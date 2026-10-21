"""Dense per-K128 row reduction for the fixed Vector-scale Laser candidate.

This changes only row-max reduction scheduling. Inputs, quantization, score
masking, online-history operations, exponentials, row sums and PV stay intact.
"""
from __future__ import annotations

from scripts.mindie_laser_quant_patch import once
from scripts import mindie_laser_quant_vscale_patch as baseline


CONTRACT = {
    "baseline": "Pinned fixed-vscale source; not the bulk coefficient candidate",
    "scope": "Dense four-row sections with 8..32 physical K128 blocks and at most 32 blocks after padding",
    "change": "Reduce each K128 row first; reduce its compact per-K maxima second",
    "first_stage": "Four vcmax calls; repeat=actual+padding blocks, dst stride=1 half, src block stride=1x32B, src repeat stride=32x32B",
    "second_stage": "vcmax with mask=actual+padding blocks, repeat=4, dst stride=1 half, src repeat stride=8x32B",
    "scratch": "Existing row-max scratch first 1024 bytes; partial row starts spaced by 128 half elements",
    "extra_ub_bytes": 0,
    "masking": "Original score-tail masking and -65504 K-block padding run before the candidate; padded values participate",
    "fallback": "Original process_cacl_max and vcmax for small, sparse, triangular or non-four-row sections",
    "numerics": "Same finite FP16 maximum values; no new score rounding or row-sum reassociation; signed-zero/NaN tie bit patterns not promised",
    "synchronization": "Existing V-only dataflow; barrier after first-stage outputs and second-stage results; restore full vector mask",
    "unchanged": "Scale preparation, QK/PV math, input payloads, attention ABI, sum reduction and Cube/Vector handoffs",
    "performance": "Unmeasured candidate: targets wide intermediate UB traffic and dependency stages, not fewer mathematical maximum comparisons",
}


METHOD = b"""    // Dense K128 row maxima: partials[row][k] occupy only 1024B of the
    // existing 32KiB max scratch. Keep the scalar/FP16 scale contract intact.
    __aicore__ inline bool TryDenseTileRowMax(__ubuf__ INPUT_T *maxima,
        __ubuf__ INPUT_T *score, __ubuf__ INPUT_T *partials,
        int32_t block_count, int32_t padding_count, int32_t triangle, bool sparse)
    {
        const int32_t reduced_blocks = block_count + padding_count;
        if (rowPerTime != 4 || triangle != 0 || sparse || block_count < 8 ||
            block_count > 32 || padding_count < 0 || reduced_blocks > 32) {
            return false;
        }
        // Layout: score[k][row][128]. A repeat advances one physical K block,
        // 4 * 128 half = 32 datablocks. ONLY_VALUE's dst stride is in halfs.
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        #pragma unroll
        for (int32_t row = 0; row < 4; ++row) {
            vcmax(partials + row * BASE_BLOCK_SIDE_LEN,
                score + row * BASE_BLOCK_SIDE_LEN,
                reduced_blocks, 1, 1, 32, (Order_t)0b10);
        }
        pipe_barrier(PIPE_V);
        // Rows are spaced by 256B; only initialized partials participate.
        // Include the original padding blocks to retain the original operand set.
        __set_mask(reduced_blocks);
        vcmax(maxima, partials, 4, 1, 1, 8, (Order_t)0b10);
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        return true;
    }

"""

START = b"    process_cacl_max(param.section_block_num, param.section_padding_block_num, true, cur_buf_for_attn_score_fp16,"
END = b"    if (!is_head_section) {"
FINAL = b"    vcmax(maxJVectorUbAddr, cur_buf_for_final_rowmax, rowPerTime, 1, 1, 8, (Order_t)0b10);"
CALL = b"""    if (!TryDenseTileRowMax(maxJVectorUbAddr, cur_buf_for_attn_score_fp16,
        cur_buf_for_rowmax, param.section_block_num, param.section_padding_block_num,
        qk_triangle, sparse_flag)) {
"""


def patch_vector(data: bytes) -> bytes:
    for anchor in (baseline.VECTOR_METHODS, b"\n            rowPerTime = 4;",
                   START, FINAL):
        if data.count(anchor) != 1:
            raise ValueError("rowmax candidate requires the pinned fixed-vscale Vector")
    if b"TryDenseTileRowMax" in data or b"vscale_bulk_compact" in data:
        raise ValueError("rowmax candidate cannot stack on itself or bulk coefficients")
    begin = data.index(START)
    end = data.index(END, begin)
    if end <= begin or FINAL not in data[begin:end]:
        raise ValueError("rowmax reduction boundaries changed")
    original = data[begin:end]
    wrapped = CALL + b"".join(b"    " + line if line.strip() else line
                            for line in original.splitlines(keepends=True)) + b"    }\n\n"
    data = data[:begin] + wrapped + data[end:]
    return once(data, b"    struct UB_FOR_NORMALIZE {", METHOD + b"    struct UB_FOR_NORMALIZE {")


def apply(relative_path: str, data: bytes) -> bytes:
    return patch_vector(data) if relative_path.rsplit("/", 1)[-1] == "VectorForward.h" else data
