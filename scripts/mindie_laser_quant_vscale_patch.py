"""Fixed INT8 score transport with per-block dequantization in Vector.

Apply only to the pinned qresident candidate. This changes score rounding and
requires the vscale numerical oracle; it is not bitwise equivalent to qresident.
"""
from __future__ import annotations

from scripts.mindie_laser_quant_patch import once
from scripts.mindie_laser_quant_cache_patch import NEW_SCALE_BLOCK
from scripts.mindie_laser_quant_reuse_patch import QK_BEGIN, RUN_BEGIN, TILING


VSCALE_CONTRACT = {
    "baseline": "Pinned qresident, dense MHA, native D128, ky=2 and 64 blocks per Cube round",
    "transport": "FP16(INT32 QK / 64); fixed exact DEQF16 coefficient 0x3c800000 per invocation",
    "int8_range": "Full signed INT8 including -128; abs(dot/64) <= 32768 for D128",
    "gain": "truncate low 13 bits of float32(q_scale*k_scale), preserving previous gain truncation",
    "coefficient": "float32(float32(64*gain)*float32(half(SCALE)))",
    "fast": "Zero or 2^-14 <= abs(coefficient) <= 65504; half(transport*half(coefficient))",
    "fallback": "Unsafe half coefficient: half(float32(transport)*float32(coefficient)) using rowmax scratch",
    "rounding": "New score rounding contract; not bitwise equivalent to original DEQF16",
    "scale_mapping": "Explicit dense section qBlockIndex and kBlockIndex; both AIVs share Q128 scale",
    "cache": "Separate absolute 64-byte Q/K scalar cache tags reset each invocation; DCCI before first read/line transition",
    "gain_ub_bytes": 4096,
    "coefficient_ub_bytes": 128,
    "ub_end_bytes": 179072,
    "broadcast": "One vmul per four-row score chunk; block strides 1,1,0 and repeat strides 8,8,1",
    "synchronization": "Preserve MAD unit flags, Q residency, L1/L0 events and Cube/Vector round handoffs",
    "unchanged": "Eight-argument torch bridge ABI, Q/K payload, FP16 PV, ordinary softmax, no new V quantization/smoothing",
    "performance": "Unmeasured candidate; requires target CCE compilation and paired NPU validation",
}


def replace_span(data: bytes, start: bytes, end: bytes, replacement: bytes = b"") -> bytes:
    if data.count(start) != 1 or data.count(end) != 1:
        raise ValueError("vscale requires the pinned unique source span")
    begin, finish = data.index(start), data.index(end)
    if finish <= begin:
        raise ValueError("vscale source span order changed")
    return data[:begin] + replacement + data[finish:]


CUBE_SCALE_SETUP = b"""    // Scales have one value per physical 128-token x D128 block.
    __aicore__ inline void SetQuantScales(__gm__ uint8_t *q, __gm__ uint8_t *k)
    {
        q_scales_ = reinterpret_cast<__gm__ float *>(q);
        k_scales_ = reinterpret_cast<__gm__ float *>(k);
        // The preceding quantizer may overwrite the same allocator addresses.
        // Never carry a scalar-cache validity decision across invocations.
        q_scale_cache_line_ = ~uint64_t(0);
        k_scale_cache_line_ = ~uint64_t(0);
    }
"""
FIXED_QUANT_INIT = b"""    // Exact power of two: even D128 dot(-128, -128)/64 is finite FP16.
    // PV and diagonal stores use NoQuant and do not update this register.
    set_quant_pre(static_cast<uint64_t>(0x3c800000u));  // 1/64
    pipe_barrier(PIPE_FIX);
"""


def patch_cube(data: bytes) -> bytes:
    for anchor in (TILING, b"bool load_q)\n{", b"n_index == 0 || ky != 2"):
        if data.count(anchor) != 1:
            raise ValueError("vscale requires the pinned qresident Cube")
    data = once(data, CUBE_SCALE_SETUP, b"")
    data = once(data, NEW_SCALE_BLOCK + b"\n", b"")
    data = once(data,
        b"        float q_scale_first, float q_scale_second, float k_scale, bool load_q);",
        b"        bool load_q);")
    data = replace_span(data,
        b"        // The pinned descriptor has ky=2. Read Q scales once per descriptor,",
        b"        for (int m_index = 0; m_index < m_loop; m_index++) {")
    data = replace_span(data,
        b"                // One K scale serves both Q128 rows in this K128 tile.",
        b"                bool upper_right_flag = (upper_right && n_index == n_loop - 1);")
    data = once(data,
        b"                    q_scale_first, q_scale_second, k_scale, n_index == 0 || ky != 2);",
        b"                    n_index == 0 || ky != 2);")
    data = once(data,
        b"    float q_scale_first, float q_scale_second, float k_scale, bool load_q)\n{",
        b"    bool load_q)\n{")
    data = replace_span(data,
        b"                // FIX supports a scalar dequantization coefficient per Q128/K128 tile.",
        b"                copy_matrix_cc_to_gm(gm_out + out_offset,")
    data = once(data, b"DEQF16,   // INT32 QK -> scaled FP16 score",
        b"DEQF16,   // INT32 QK -> FP16 transport, fixed factor 1/64")
    data = once(data, RUN_BEGIN + b" {\n", RUN_BEGIN + b" {\n" + FIXED_QUANT_INIT)
    return data


VECTOR_SCALE_SETUP = b"""    __aicore__ inline void SetQuantScales(__gm__ uint8_t *q, __gm__ uint8_t *k)
    {
        q_scales_ = reinterpret_cast<__gm__ float *>(q);
        k_scales_ = reinterpret_cast<__gm__ float *>(k);
        // Reused addresses may contain newly quantized scales on this stream.
        q_scale_cache_line_ = ~uint64_t(0);
        k_scale_cache_line_ = ~uint64_t(0);
    }
"""

VECTOR_METHODS = b"""    // vscale methods: dense MHA only, each section is one physical Q128.
    __aicore__ inline void PrepareQuantScaleSection(int64_t q_index, int64_t k_index,
        int32_t block_count, UB_FOR_SHORT_LEN_ATTN_SCORE ub_attn)
    {
        static_assert(sizeof(INPUT_T) == 2, "vscale requires FP16 score storage");
        // Finish the preceding section's reads before reusing the gain region.
        pipe_barrier(PIPE_V);
        const float q_scale = ReadQuantScale(q_scales_ + q_index, q_scale_cache_line_);
        const float attention_scale = static_cast<float>(static_cast<INPUT_T>(SCALE));
        vscale_slow_mask_ = 0;
        __set_mask(rowPerTime * 16);
        for (int32_t block = 0; block < block_count; ++block) {
            const float k_scale = ReadQuantScale(k_scales_ + k_index + block, k_scale_cache_line_);
            float gain = q_scale * k_scale;
            uint32_t bits = *reinterpret_cast<uint32_t *>(&gain) & 0xffffe000u;
            gain = *reinterpret_cast<float *>(&bits);
            const float restored_gain = 64.0f * gain;
            const float coefficient = restored_gain * attention_scale;
            ub_attn.vscale_coeff_fp32[block] = coefficient;
            const float magnitude = coefficient < 0.0f ? -coefficient : coefficient;
            const bool fast = coefficient == 0.0f ||
                (magnitude >= 0.00006103515625f && magnitude <= 65504.0f);
            if (!fast) {
                vscale_slow_mask_ |= uint32_t(1) << block;
            }
            // One 32-byte block per 128-element score repeat. Four equal
            // blocks serve the four packed rows belonging to this K128.
            const INPUT_T packed_gain = fast ? static_cast<INPUT_T>(coefficient) : static_cast<INPUT_T>(1.0f);
            // CCE intrinsic checking needs a concrete scalar value, not a
            // const template-typed lvalue (INFERCHANNEL in CANN 9.0.1).
            vector_dup(ub_attn.vscale_gain_fp16 + block * rowPerTime * 16,
                static_cast<half>(packed_gain), 1, 1, 1, 8, 1);
        }
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }

    __aicore__ inline void ApplyQuantScoreScale(__ubuf__ INPUT_T *score,
        int32_t block_count, UB_FOR_SHORT_LEN_ATTN_SCORE ub_attn)
    {
        // src1 block stride 0 broadcasts one 32B gain block over all eight
        // score blocks. src1 repeat stride 1 advances to the next packed row.
        vmul(score, score, ub_attn.vscale_gain_fp16, block_count * rowPerTime,
            1, 1, 0, 8, 8, 1);
        pipe_barrier(PIPE_V);
        if (vscale_slow_mask_ != 0) {
            // The max-reduction scratch is not live until this method returns.
            // One four-row K128 block occupies only 2048 of its 32768 bytes.
            auto scratch = reinterpret_cast<__ubuf__ float *>(ub_attn.buf_for_cacl_rowmax_fp16);
            for (int32_t block = 0; block < block_count; ++block) {
                if ((vscale_slow_mask_ & (uint32_t(1) << block)) == 0) {
                    continue;
                }
                auto values = score + block * rowPerTime * BASE_BLOCK_SIDE_LEN;
                vconv_f162f32(scratch, values, rowPerTime * 2, 1, 1, 8, 4);
                pipe_barrier(PIPE_V);
                // Load out of UB address space before passing the scalar
                // operand to the raw intrinsic; preserve its FP32 value.
                float scalar_coefficient = ub_attn.vscale_coeff_fp32[block];
                vmuls(scratch, scratch, static_cast<float>(scalar_coefficient),
                    rowPerTime * 2, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vconv_f322f16(values, scratch, rowPerTime * 2, 1, 1, 4, 8);
                pipe_barrier(PIPE_V);
            }
        }
    }

"""

GAIN_ALLOCATION = b"""
    // Existing dense rowPerTime=4 allocation ends at byte 174848.
    // Gains survive max/exp/rowsum scratch reuse until the section is complete.
    static_assert(174848 + 4096 + 128 <= 192 * 1024, "vscale UB exceeds 192 KiB");
    ub_attn_score -> vscale_gain_fp16 = reinterpret_cast<__ubuf__ INPUT_T *>((uintptr_t)offset);
    offset += 32 * 4 * 16 * 2;
    ub_attn_score -> vscale_coeff_fp32 = reinterpret_cast<__ubuf__ float *>((uintptr_t)offset);
    offset += 32 * 4;
"""

OLD_SCORE_SCALE = b"""    vmuls(cur_buf_for_attn_score_fp16, cur_buf_for_attn_score_fp16,
        (INPUT_T)SCALE, param.section_block_num * rowPerTime, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);"""
NEW_SCORE_SCALE = b"""    ApplyQuantScoreScale(cur_buf_for_attn_score_fp16, param.section_block_num, ub_attn);"""


def patch_vector(data: bytes) -> bytes:
    for anchor in (b"\n            rowPerTime = 4;", b"    int32_t block_per_core = 64;",
                   b"    int32_t ky = 2;", b"    int32_t offset = 512;\n    int32_t section_num = 8;"):
        if data.count(anchor) != 1:
            raise ValueError("vscale requires pinned four-row Vector and 64/2 mapping")
    data = once(data, b"    __aicore__ inline VectorForward() {};\n",
        b"    __aicore__ inline VectorForward() {};\n" + VECTOR_SCALE_SETUP)
    data = once(data,
        b"        __ubuf__ float* diagExpMaxJMatPongUbAddr ;",
        b"""        __ubuf__ float* diagExpMaxJMatPongUbAddr ;
        __ubuf__ INPUT_T* vscale_gain_fp16;
        __ubuf__ float* vscale_coeff_fp32;""")
    data = once(data, b"    struct UB_FOR_NORMALIZE {", VECTOR_METHODS + b"    struct UB_FOR_NORMALIZE {")
    data = once(data, b"private:\n    __aicore__ __inline__ void initWorkSpace();",
        b"private:\n" + NEW_SCALE_BLOCK +
        b"\n    uint32_t vscale_slow_mask_ = 0;\n    __aicore__ __inline__ void initWorkSpace();")
    data = once(data,
        b"            allocate_ubuf_for_short_seq_attn_score(&ub_short_seq_attn);",
        b"""            allocate_ubuf_for_short_seq_attn_score(&ub_short_seq_attn);
            PrepareQuantScaleSection(section.qScaleBlockIndex[sectionLoop],
                section.kScaleBlockIndex[sectionLoop], sectionBlockNum, ub_short_seq_attn);""")
    data = once(data,
        b"    offset += BASE_BLOCK_SIDE_LEN/2*BASE_BLOCK_SIDE_LEN* SIZE_FP32;\n",
        b"    offset += BASE_BLOCK_SIDE_LEN/2*BASE_BLOCK_SIDE_LEN* SIZE_FP32;\n" + GAIN_ALLOCATION)
    data = once(data, OLD_SCORE_SCALE, NEW_SCORE_SCALE)
    return data


INDEX_FIELDS = b"""        // Physical Q128/K128 scale indices for the dense MHA vscale candidate.
        int64_t qScaleBlockIndex[FORWARD_MAX_SWITCH_TIME] = {0};
        int64_t kScaleBlockIndex[FORWARD_MAX_SWITCH_TIME] = {0};
"""

INDEX_MAPPING = b"""                // The two AIVs process different 64-row halves of the same Q128.
                const int64_t q_scale_base = (b * this->headNum_ + n) *
                    (this->querySequenceLen_ / SIZE_128) + ir;
                const int64_t k_scale_base = (b * this->headNum_ + n) *
                    (this->keyValueSequenceLen_ / SIZE_128) + ic;
                section.qScaleBlockIndex[2 * i] = q_scale_base;
                section.qScaleBlockIndex[2 * i + 1] = q_scale_base + 1;
                section.kScaleBlockIndex[2 * i] = k_scale_base;
                section.kScaleBlockIndex[2 * i + 1] = k_scale_base;

"""


def patch_address_const(data: bytes) -> bytes:
    if b"qScaleBlockIndex" in data:
        raise ValueError("vscale section metadata already present")
    anchor = b"    struct FORWARD_SECTION_INFO {\n"
    return once(data, anchor, anchor + INDEX_FIELDS)


def patch_vector_address(data: bytes) -> bytes:
    if b"qScaleBlockIndex" in data:
        raise ValueError("vscale section mapping already present")
    anchor = b"                section.rowmaxOffset[2 * i + 1] = section.rowmaxOffset[2 * i] + SIZE_128; // ky = 2,"
    if data.count(anchor) != 1:
        raise ValueError("vscale requires the pinned dense section mapper")
    begin = data.index(b"        addrMapping_nomask(")
    end = data.index(b"        __aicore__ __inline__ void addrMapping_mask(", begin)
    part = data[begin:end]
    part = once(part, b"                // section", INDEX_MAPPING + b"                // section")
    return data[:begin] + part + data[end:]


def patch_entry(data: bytes) -> bytes:
    vector_begin = data.index(b"#elif __DAV_C220_VEC__")
    if b"op.SetQuantScales" in data[vector_begin:]:
        raise ValueError("vscale entry already routes scales to Vector")
    data = once(data, b"    op.SetQuantScales(q_scale_gm, k_scale_gm);\n", b"")
    return once(data, b"    op.SetHighPrecision(true);",
        b"    op.SetQuantScales(q_scale_gm, k_scale_gm);\n    op.SetHighPrecision(true);")


def apply(relative_path: str, data: bytes) -> bytes:
    name = relative_path.rsplit("/", 1)[-1]
    transformations = {
        "CubeForward.h": patch_cube,
        "VectorForward.h": patch_vector,
        "address_const.h": patch_address_const,
        "AddressMappingVectorForwardOnline.h": patch_vector_address,
        "vc_laser_quant_qresident_attention.cpp": patch_entry,
        "vc_laser_quant_vscale_attention.cpp": patch_entry,
    }
    return transformations[name](data) if name in transformations else data
