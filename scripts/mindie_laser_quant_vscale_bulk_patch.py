"""Batch the existing Vector-scale coefficient broadcast; retain its arithmetic.

Input is the fully prepared vscale source (possibly renamed to the bulk
identity). This patch does not change QK, scale cache, score rounding, softmax,
PV, or their dispatch. The independent build remains an experiment.
"""
from __future__ import annotations

from scripts.mindie_laser_quant_patch import once
from scripts import mindie_laser_quant_vscale_patch as baseline


BULK_CONTRACT = CONTRACT = {
    "baseline": "Pinned vector-scale source with the CANN 9.0.1 scalar-value fix",
    "change": "For 8..32 K blocks, replace one vector_dup per block by four vbrcb instructions",
    "arithmetic": "Identical scalar FP32 gain truncation, coefficient evaluation, FP16 rounding and fallback mask",
    "short_sections": "1..7 blocks retain the original coefficient preparation method",
    "broadcast": "32 packed half coefficients, vbrcb dstBlockStride=4, dstRepeatStride=32, ceil(block_count/8) repeats",
    "padding": "Initialize all compact coefficients read by rounded-up Brcb repeats; extra output stays within 4096-byte gains",
    "extra_ub_bytes": 64,
    "ub_end_bytes": 179136,
    "synchronization": "Paired V->S EVENT_ID0 before compact reuse; paired S->V EVENT_ID0 after scalar stores; V barrier before gain consumption",
    "event_scope": "Immediate paired events at section boundary; original pipelined V->S events use IDs 1..6",
    "unchanged": "Original vscale numerical contract, full-range INT8 QK, FP16 PV, normal softmax, no V reorder or ExpCast",
    "performance": "Unmeasured candidate; target CCE build and original-FP16 paired NPU benchmark required",
}


# Keep the scalar arithmetic literally inherited from the validated vscale
# method. Only its final store/launch operation changes below.
PREPARE_START = b"    // vscale methods: dense MHA only, each section is one physical Q128.\n"
APPLY_START = b"    __aicore__ inline void ApplyQuantScoreScale("
OLD_PREPARE = baseline.VECTOR_METHODS[:baseline.VECTOR_METHODS.index(APPLY_START)]
LEGACY_PREPARE = OLD_PREPARE.replace(
    b"PrepareQuantScaleSection(", b"PrepareQuantScaleSectionLegacy(", 1)

NEW_PREPARE = OLD_PREPARE.replace(
    b"// vscale methods: dense MHA only, each section is one physical Q128.",
    b"// Bulk broadcast preserves every vscale scalar operation and FP16 bit pattern.",
    1,
)
NEW_PREPARE = once(
    NEW_PREPARE,
    b"""        // Finish the preceding section's reads before reusing the gain region.
        pipe_barrier(PIPE_V);""",
    b"""        if (block_count < 8) {
            PrepareQuantScaleSectionLegacy(q_index, k_index, block_count, ub_attn);
            return;
        }
        // A preceding Brcb may still read this compact region. V->S is
        // necessary before scalar writes; a V-only barrier cannot do that.
        // EVENT_ID0 has no outstanding V->S/S->V event at this section boundary.
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);""",
)
NEW_PREPARE = once(NEW_PREPARE, b"        __set_mask(rowPerTime * 16);\n", b"")
NEW_PREPARE = once(
    NEW_PREPARE,
    b"""            // One 32-byte block per 128-element score repeat. Four equal
            // blocks serve the four packed rows belonging to this K128.
            const INPUT_T packed_gain = fast ? static_cast<INPUT_T>(coefficient) : static_cast<INPUT_T>(1.0f);
            // CCE intrinsic checking needs a concrete scalar value, not a
            // const template-typed lvalue (INFERCHANNEL in CANN 9.0.1).
            vector_dup(ub_attn.vscale_gain_fp16 + block * rowPerTime * 16,
                static_cast<half>(packed_gain), 1, 1, 1, 8, 1);""",
    b"""            const INPUT_T packed_gain = fast ? static_cast<INPUT_T>(coefficient) : static_cast<INPUT_T>(1.0f);
            ub_attn.vscale_bulk_compact_fp16[block] = packed_gain;""",
)
NEW_PREPARE = once(
    NEW_PREPARE,
    b"""        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }

""",
    b"""        const int32_t repeats = (block_count + 7) / 8;
        for (int32_t block = block_count; block < repeats * 8; ++block) {
            ub_attn.vscale_bulk_compact_fp16[block] = static_cast<INPUT_T>(0.0f);
        }
        // Scalar-produced compact values must be visible to Brcb.
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        // Brcb consumes eight consecutive values per repeat. Each call fills
        // one of four 32-byte row blocks for every K128 coefficient.
        // All addresses are 32B aligned; input/output do not overlap.
        #pragma unroll
        for (int32_t row = 0; row < 4; ++row) {
            vbrcb(reinterpret_cast<__ubuf__ uint16_t *>(ub_attn.vscale_gain_fp16) + row * 16,
                reinterpret_cast<__ubuf__ uint16_t *>(ub_attn.vscale_bulk_compact_fp16),
                4, 32, repeats);
        }
        pipe_barrier(PIPE_V);
    }

""",
)

BULK_ALLOCATION = b"""    // 32 packed half coefficients, disjoint from gains and FP32 fallback.
    static_assert(179136 <= 192 * 1024, "vscale bulk UB exceeds 192 KiB");
    ub_attn_score -> vscale_bulk_compact_fp16 = reinterpret_cast<__ubuf__ INPUT_T *>((uintptr_t)offset);
    offset += 32 * 2;
"""


def patch_vector(data: bytes) -> bytes:
    for anchor in (baseline.VECTOR_METHODS, baseline.GAIN_ALLOCATION,
                   b"\n            rowPerTime = 4;",
                   b"    uint32_t vscale_slow_mask_ = 0;"):
        if data.count(anchor) != 1:
            raise ValueError("bulk broadcast requires the pinned fixed-vscale Vector")
    data = once(data, b"        __ubuf__ float* vscale_coeff_fp32;",
        b"""        __ubuf__ float* vscale_coeff_fp32;
        __ubuf__ INPUT_T* vscale_bulk_compact_fp16;""")
    data = once(data, OLD_PREPARE, LEGACY_PREPARE + NEW_PREPARE)
    return once(data, baseline.GAIN_ALLOCATION,
                baseline.GAIN_ALLOCATION + BULK_ALLOCATION)


def apply(relative_path: str, data: bytes) -> bytes:
    return patch_vector(data) if relative_path.rsplit("/", 1)[-1] == "VectorForward.h" else data
