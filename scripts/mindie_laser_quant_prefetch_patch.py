"""One-batch score lookahead on the pinned scale-reuse Laser Vector path.

This pure byte transform changes VectorForward.h only. It neither prepares a
checkout nor changes Cube/FIX configuration, score layout, arithmetic or masks.
The dense path already has two 32 KiB score slots; this schedules the next
four-row GM read while the current slot is consumed. The other paths retain
their original serialized load/compute sequence.
"""
from __future__ import annotations


CONTRACT = {
    "scope": "AIV score DMA one four-row batch ahead within one dense section",
    "guard": "qk_triangle == 0; !sparse_flag; rowPerTime == 4; vector lines == 64; key blocks in [1,32]",
    "prologue": "initialize V-to-MTE2 free tokens for both slots; consume V and MTE3 free tokens for slot 0; load batch 0; set ready 0",
    "loop": "wait current ready; consume opposite V and MTE3 free tokens and load next score batch if present; original arithmetic and P store; release both current-slot dependencies",
    "drain": "last iteration enqueues no next read; consume both slots' final V-to-MTE2 and MTE3-to-MTE2 free tokens",
    "storage": "existing two disjoint 32 KiB score slots; no added UB; numeric IDs 0/1 also used in previously unused V-to-MTE2 direction",
    "padding": "unchanged is_tail_section/windowLen score masking, including non-aligned K",
    "masks": "mask DMAs never prefetched; triangular and sparse paths retain serialized loads",
    "unchanged": "Cube, QK scales, DEQF16, GM layout, Vector math order, original exp, FP16 PV, cross-core flags",
    "performance": "scheduling hypothesis requiring paired device measurements; no local NPU speedup claim",
}

MARKER = b"vc_score_prefetch"
HELPER_ANCHOR = b"    __aicore__ __inline__ void process_line_phase_one_for_short_seq_max(bool is_head_section,"
HELPER = b"""    // The caller enables lookahead only for the audited dense four-row path.
    // MTE3->MTE2 protects the P store; V->MTE2 protects the later row-sum
    // reads. Both must finish before slot reuse. MTE2->V owns load completion.
    // Numeric event IDs 0/1 are independent in these event directions.
    __aicore__ __inline__ void vc_prefetch_score_batch(
        __ubuf__ INPUT_T *dst, __gm__ WORKSPACE_T *src,
        int32_t key_blocks, event_t event_id)
    {
        wait_flag(PIPE_MTE3, PIPE_MTE2, event_id);
        wait_flag(PIPE_V, PIPE_MTE2, event_id);
        copy_gm_to_ubuf(dst, src, 0, key_blocks,
            BASE_BLOCK_SIDE_LEN * rowPerTime / 16,
            (BASE_BLOCK_DATA_NUM - rowPerTime * BASE_BLOCK_SIDE_LEN) / 16, 0);
        set_flag(PIPE_MTE2, PIPE_V, event_id);
    }

"""
DECLARATION_OLD = b"        __ubuf__ INPUT_T* subMaxValueResultVectorUbAddr, int rowsPerDiag, int rowmax_build_step);"
DECLARATION_NEW = DECLARATION_OLD[:-2] + b", bool score_preloaded);"
DEFINITION_OLD = b"    int rowsPerDiag,\n    int rowmax_build_step)"
DEFINITION_NEW = b"    int rowsPerDiag,\n    int rowmax_build_step, bool score_preloaded)"
DMA_START = b"    copy_gm_to_ubuf(\n        cur_buf_for_attn_score_fp16,"
DMA_END = b"\n\n    int32_t cur_core_index = get_block_idx();"
READY_OLD = b"    set_flag(PIPE_MTE2, PIPE_V, event_id);\n    wait_flag(PIPE_MTE2, PIPE_V, event_id);"
READY_NEW = b"""    // Prefetched scores were waited for by the section driver. The guarded
    // path has no mask DMA; fallback retains the original score/mask fence.
    if (!score_preloaded) {
        set_flag(PIPE_MTE2, PIPE_V, event_id);
        wait_flag(PIPE_MTE2, PIPE_V, event_id);
    }"""
LOOP_ANCHOR = b"    for (int32_t lines = 0; lines < each_vector_proc_line_num; lines+=rowPerTime)"
PROLOGUE = b"""    // Padding via windowLen is applied to each loaded score batch below;
    // it is independent of triangular/sparse mask DMA and remains enabled.
    // Mask scratch slots overlap in the pinned kernel, so never prefetch masks.
    const bool vc_score_prefetch = qk_triangle == 0 && !sparse_flag &&
        rowPerTime == 4 && each_vector_proc_line_num == 64 &&
        section_block_nums > 0 && section_block_nums <= 32;
    if (vc_score_prefetch) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        vc_prefetch_score_batch(ub_attn.pp_buf_for_attn_score_fp16[0],
            attn_score_gm + param_ping_pong[0].section_start_line_offset,
            section_block_nums, EVENT_ID0);
    }

"""
DRIVER_OLD = b"""        wait_flag(PIPE_MTE3, PIPE_MTE2, event_id);
        process_line_phase_one_for_short_seq_max(is_head_section, is_tail_section, qk_triangle,"""
DRIVER_NEW = b"""        if (vc_score_prefetch) {
            wait_flag(PIPE_MTE2, PIPE_V, event_id);
            if (lines + rowPerTime < each_vector_proc_line_num) {
                const int32_t next_slot = 1 - ping_pong_flag;
                const auto next_event = next_slot == 0 ? EVENT_ID0 : EVENT_ID1;
                // Opposite-slot params already identify the next four rows:
                // each consumed slot advances by eight rows at loop end.
                // Its source rows differ from the current in-place P store.
                vc_prefetch_score_batch(ub_attn.pp_buf_for_attn_score_fp16[next_slot],
                    attn_score_gm + param_ping_pong[next_slot].section_start_line_offset,
                    section_block_nums, next_event);
            }
        } else {
            wait_flag(PIPE_MTE3, PIPE_MTE2, event_id);
        }
        process_line_phase_one_for_short_seq_max(is_head_section, is_tail_section, qk_triangle,"""
CALL_OLD = b"                                                rowmax_build_step);"
CALL_NEW = b"                                                rowmax_build_step, vc_score_prefetch);"
RELEASE_OLD = b"        set_flag(PIPE_MTE3, PIPE_MTE2, event_id);"
RELEASE_NEW = b"""        if (vc_score_prefetch) {
            // process_calc_sum runs after the P DMA and still reads this slot.
            set_flag(PIPE_V, PIPE_MTE2, event_id);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, event_id);"""
DRAIN_OLD = b"""    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);"""
DRAIN_NEW = b"""    if (vc_score_prefetch) {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);"""


def _once(data: bytes, old: bytes, new: bytes) -> bytes:
    if data.count(old) != 1:
        raise ValueError(f"Score prefetch expected one pinned anchor: {old[:100]!r}")
    return data.replace(old, new, 1)


def _changes(source: bytes):
    """Exact substitutions, including the untouched score DMA's original text."""
    newline = b"\r\n" if b"\r\n" in source else b"\n"

    def n(value):
        return value.replace(b"\n", newline)

    start, end = n(DMA_START), n(DMA_END)
    if source.count(start) != 1:
        raise ValueError("Expected one pinned four-row score DMA")
    begin = source.index(start)
    finish = source.find(end, begin)
    if finish < 0:
        raise ValueError("Score DMA end anchor changed")
    dma = source[begin:finish]
    expected = (
        b"param.section_block_num,",
        b"BASE_BLOCK_SIDE_LEN * rowPerTime / 16,",
        b"(BASE_BLOCK_DATA_NUM - rowPerTime * BASE_BLOCK_SIDE_LEN) / 16,",
    )
    if dma.count(b"copy_gm_to_ubuf(") != 1 or any(x not in dma for x in expected):
        raise ValueError("Pinned score DMA layout changed")
    # No reindent of original bytes: inverse transform restores the exact file.
    guarded_dma = n(b"    if (!score_preloaded) {\n") + dma + n(b"\n    }")
    return [
        (n(HELPER_ANCHOR), n(HELPER) + n(HELPER_ANCHOR)),
        (n(DECLARATION_OLD), n(DECLARATION_NEW)),
        (n(DEFINITION_OLD), n(DEFINITION_NEW)),
        (dma, guarded_dma),
        (n(READY_OLD), n(READY_NEW)),
        (n(LOOP_ANCHOR), n(PROLOGUE) + n(LOOP_ANCHOR)),
        (n(DRIVER_OLD), n(DRIVER_NEW)),
        (n(CALL_OLD), n(CALL_NEW)),
        (n(RELEASE_OLD), n(RELEASE_NEW)),
        (n(DRAIN_OLD), n(DRAIN_NEW)),
    ]


def patch_vector(data: bytes) -> bytes:
    if MARKER in data or b"vc_prefetch_score_batch" in data:
        raise ValueError("Score prefetch was already applied")
    if b"PIPE_V, PIPE_MTE2" in data:
        raise ValueError("V-to-MTE2 event direction must be unused before score prefetch")
    # Guard the assumptions used by the scheduling proof, not just edit sites.
    required = (
        b"MAX_LENG_PER_UB_PROC * SIZE_FP16 * 2",
        b"MAX_LENG_PER_UB_PROC;",
        b"rowPerTime = 4;",
        b"set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);",
        b"set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);",
        b"param_ping_pong[ping_pong_flag].section_start_line_offset += BASE_BLOCK_SIZE_DOUBLE * rowPerTime;",
    )
    if any(anchor not in data for anchor in required):
        raise ValueError("Pinned score slots, row size or event lifecycle changed")
    for old, new in _changes(data):
        data = _once(data, old, new)
    return data


def unpatch_vector(data: bytes, original: bytes) -> bytes:
    """Verify an exact inverse using the pinned pre-transform source."""
    for old, new in reversed(_changes(original)):
        data = _once(data, new, old)
    return data


def apply(relative_path: str, data: bytes) -> bytes:
    return patch_vector(data) if relative_path.replace("\\", "/").rsplit("/", 1)[-1] == "VectorForward.h" else data
