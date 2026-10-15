"""Audited INT8-QK delta on the pinned MindIE Laser; FP16 PV stays intact.

This is an A2/A3 port experiment, not a native FP8 implementation. All edits
are scoped to exact upstream anchors so a source update cannot silently alter
the addressing or cross-core protocol.
"""
from __future__ import annotations

from scripts.mindie_laser_expcast_patch import (
    BULK_VEXP, EXPERIMENTAL_PROBABILITY, INCLUDE_ANCHOR, INCLUDE_ADDITION,
)


def once(data: bytes, old: bytes, new: bytes) -> bytes:
    if data.count(old) != 1:
        raise ValueError(f"Quant Laser expected one anchor: {old[:100]!r}")
    return data.replace(old, new, 1)


def patch_cube(data: bytes) -> bytes:
    data = once(data, b"    __aicore__ inline void Run();", b"""    // Scales have one value per physical 128-token x D128 block.
    __aicore__ inline void SetQuantScales(__gm__ uint8_t *q, __gm__ uint8_t *k)
    {
        q_scales_ = reinterpret_cast<__gm__ float *>(q);
        k_scales_ = reinterpret_cast<__gm__ float *>(k);
    }
    __aicore__ inline void Run();""")
    data = once(data, b"    // CUBE1\n", b"""    // Input scales can reuse allocator addresses across fresh API calls.
    // Invalidate the scalar cache line before a GM scalar read; MTE does not
    // keep the scalar data cache coherent with a preceding quantization op.
    __aicore__ inline float ReadQuantScale(__gm__ float *ptr)
    {
        AscendC::GlobalTensor<float> scale;
        scale.SetGlobalBuffer(ptr, 1);
        AscendC::DataCacheCleanAndInvalid<float,
            AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(scale);
        return scale.GetValue(0);
    }
    __gm__ float *q_scales_;
    __gm__ float *k_scales_;
    // CUBE1
""")
    for name in (b"Q", b"K"):
        data = once(data, b"__gm__ TYPE *__restrict__  gm_" + name,
                    b"__gm__ int8_t *__restrict__  gm_" + name)
        data = once(data, b"this->gm_" + name + b" = (__gm__ TYPE *__restrict__)gm_" + name,
                    b"this->gm_" + name + b" = (__gm__ int8_t *__restrict__)gm_" + name)
    data = once(data,
        b"cube1_base_matmul(__cbuf__ TYPE *l1_a, __cbuf__ TYPE *l1_b, __gm__ TYPE *gm_out,\n"
        b"        int32_t ky, int32_t out_put_matrix_line_strid, bool upper_right_flag);",
        b"cube1_base_matmul(__cbuf__ int8_t *l1_a, __cbuf__ int8_t *l1_b, __gm__ TYPE *gm_out,\n"
        b"        int32_t ky, int32_t out_put_matrix_line_strid, bool upper_right_flag,\n"
        b"        int64_t q_scale_base, int64_t k_scale_index);")
    begin = data.index(b"__aicore__ __inline__ void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::cube1_matmul_op(")
    end = data.index(b"__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::Run()", begin)
    chunk = data[begin:end]
    for side in (b"a", b"b"):
        chunk = once(chunk, b"auto l1_" + side + b" = ping_pong_flag_l1_" + side + b"_ ? l1_" + side + b"_pong_ : l1_" + side + b"_ping_;",
            b"auto l1_" + side + b" = reinterpret_cast<__cbuf__ int8_t *>(ping_pong_flag_l1_" + side + b"_ ? l1_" + side + b"_pong_ : l1_" + side + b"_ping_);")
    if chunk.count(b"copy_gm_to_cbuf_multi_nd2nz_b16(") != 2:
        raise ValueError("Expected exactly Q and K GM-to-L1 transfers")
    chunk = chunk.replace(b"copy_gm_to_cbuf_multi_nd2nz_b16(", b"copy_gm_to_cbuf_multi_nd2nz_b8(")
    chunk = once(chunk,
        b"cube1_base_matmul(l1_a, l1_b, gm_c, ky, line_stride, upper_right_flag);",
        b"cube1_base_matmul(l1_a, l1_b, gm_c, ky, line_stride, upper_right_flag,\n"
        b"                    (gm_a - gm_Q) / (128 * 128), (gm_b - gm_K) / (128 * 128) + n_index);")
    chunk = once(chunk,
        b"    __cbuf__ TYPE *l1_a, __cbuf__ TYPE *l1_b,\n"
        b"    __gm__ TYPE *gm_out, int32_t ky, int32_t out_put_matrix_line_strid, bool upper_right_flag)",
        b"    __cbuf__ int8_t *l1_a, __cbuf__ int8_t *l1_b,\n"
        b"    __gm__ TYPE *gm_out, int32_t ky, int32_t out_put_matrix_line_strid, bool upper_right_flag,\n"
        b"    int64_t q_scale_base, int64_t k_scale_index)")
    for side, mem in ((b"a", b"ca"), (b"b", b"cb"), (b"c", b"cc")):
        typ = b"int32_t" if side == b"c" else b"int8_t"
        chunk = once(chunk,
            b"auto l0_" + side + b" = ping_pong_flag_l0_" + side + b"_ ? l0_" + side + b"_pong_ : l0_" + side + b"_ping_;",
            b"auto l0_" + side + b" = reinterpret_cast<__" + mem + b"__ " + typ + b" *>(ping_pong_flag_l0_" + side + b"_ ? l0_" + side + b"_pong_ : l0_" + side + b"_ping_);")
    # K ND[N,D] -> NZ[K1,N1,16,32] already is Zn for logical K^T.
    start_b = chunk.index(b"        if (l1_n_size_ == SIZE_128) {")
    end_b = chunk.index(b"        set_flag(PIPE_MTE1, PIPE_M, ping_pong_flag_l0_b_ + 2);", start_b)
    chunk = chunk[:start_b] + b"""        // INT8 fractal is 16 x 32 bytes; no transpose on Load2D.
        load_cbuf_to_cb(l0_b, l1_b, 0, k0_ * n0_ / 512, 1, 0, 0, false, inc);

""" + chunk[end_b:]
    chunk = once(chunk, b"l1_a + m_offset * SIZE_16 + i * SIZE_256,",
                  b"l1_a + m_offset * SIZE_32 + i * 512,")
    chunk = once(chunk, b"k0_ / SIZE_16,                             // repeat",
                  b"k0_ / SIZE_32,                             // repeat (INT8 K0=32)")
    chunk = once(chunk, b"            auto trans_dtype = IF_BF16 ? F322BF16 : F322F16;\n", b"")
    chunk = once(chunk, b"                copy_matrix_cc_to_gm(gm_out + out_offset,", b"""                // FIX supports a scalar dequantization coefficient per Q128/K128 tile.
                // Hardware keeps float bits 31:13 (sign/exponent/10 mantissa bits).
                float gain = ReadQuantScale(q_scales_ + q_scale_base + m_offset / 128) *
                             ReadQuantScale(k_scales_ + k_scale_index);
                uint32_t bits = *reinterpret_cast<uint32_t *>(&gain) & 0xffffe000u;
                set_quant_pre(static_cast<uint64_t>(bits));
                pipe_barrier(PIPE_FIX);
                copy_matrix_cc_to_gm(gm_out + out_offset,""")
    chunk = once(chunk, b"                    F322F16,  // QuantPRE  F322F16",
                  b"                    DEQF16,   // INT32 QK -> scaled FP16 score")
    data = data[:begin] + chunk + data[end:]
    old = b"Address::PhyAddrForwardCube1Online<TYPE, TYPE, WORKSPACE_TYPE> src[16];"
    if data.count(old) != 3:
        raise ValueError("Unexpected number of QK address mapping call sites")
    return data.replace(old, b"Address::PhyAddrForwardCube1Online<int8_t, int8_t, WORKSPACE_TYPE> src[16];")


def patch_vector(data: bytes) -> bytes:
    data = once(data, INCLUDE_ANCHOR, INCLUDE_ANCHOR + INCLUDE_ADDITION)
    data = once(data, b"    __aicore__ inline void Run();", b"""    __aicore__ inline void SetExpCast(bool enabled) { use_expcast_ = enabled; }
    __aicore__ inline void Run();""")
    data = once(data, b"    bool isHighPrecision = true;", b"    bool isHighPrecision = true;\n    bool use_expcast_ = false;")
    return once(data, BULK_VEXP, b"    if (use_expcast_) {\n" + EXPERIMENTAL_PROBABILITY +
                b"\n    } else {\n" + BULK_VEXP + b"\n    }")


def patch_entry(data: bytes) -> bytes:
    data = data.replace(b"atten_mask_gm", b"q_scale_gm").replace(b"alibi_mask_gm", b"k_scale_gm")
    data = once(data, b"    op.Run();\n#elif", b"    op.SetQuantScales(q_scale_gm, k_scale_gm);\n    op.Run();\n#elif")
    data = once(data, b"op.Init(q_gm, k_gm, v_gm, q_scale_gm, score_gm,", b"op.Init(q_gm, k_gm, v_gm, nullptr, score_gm,")
    return once(data, b"    op.SetHighPrecision(true);", b"    op.SetHighPrecision(true);\n    op.SetExpCast(tiling_data->useExpCast);")


def patch_definition(data: bytes) -> bytes:
    for name, dtype in ((b"query", b"DT_INT8"), (b"key", b"DT_INT8"), (b"value", b"DT_FLOAT16")):
        data = once(data, b'this->Input("' + name + b'")\n                .ParamType(REQUIRED)\n                .DataType({ge::DT_FLOAT16, ge::DT_BF16})',
                    b'this->Input("' + name + b'")\n                .ParamType(REQUIRED)\n                .DataType({ge::' + dtype + b', ge::' + dtype + b'})')
    for old, new in ((b"atten_mask", b"q_scale"), (b"alibi_mask", b"k_scale")):
        data = once(data, b'this->Input("' + old + b'")\n                .ParamType(OPTIONAL)\n                .DataType({ge::DT_FLOAT16, ge::DT_BF16})',
                    b'this->Input("' + new + b'")\n                .ParamType(REQUIRED)\n                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})')
    # One dtype signature, avoiding duplicate binary configurations.
    for dtype in (b"DT_INT8", b"DT_FLOAT16", b"DT_FLOAT", b"DT_UINT8"):
        data = data.replace(b"{ge::" + dtype + b", ge::" + dtype + b"}", b"{ge::" + dtype + b"}")
    data = data.replace(b"{ge::FORMAT_ND, ge::FORMAT_ND}", b"{ge::FORMAT_ND}")
    data = once(data, b'        this->AICore().AddConfig("ascend910");',
                b'        this->Attr("use_expcast").AttrType(OPTIONAL).Bool(false);')
    return data


def patch_tiling(data: bytes) -> bytes:
    data = once(data, b"    tilingData.set_isHighPrecision(*isHighPrecision);", b"""    tilingData.set_isHighPrecision(*isHighPrecision);
    const auto useExpCast = context->GetAttrs()->GetAttrPointer<bool>(7);
    if (useExpCast == nullptr) return ge::GRAPH_FAILED;
    tilingData.set_useExpCast(*useExpCast);""")
    return data


def apply(relative_path: str, data: bytes) -> bytes:
    patches = {"CubeForward.h": patch_cube, "VectorForward.h": patch_vector,
               "vc_laser_quant_attention.cpp": patch_entry,
               "vc_laser_quant_attention_def.cpp": patch_definition,
               "vc_laser_quant_attention_tiling.cpp": patch_tiling}
    basename = relative_path.rsplit("/", 1)[-1]
    if basename == "vc_laser_quant_attention_tiling.h":
        return once(data, b"END_TILING_DATA_DEF;", b"TILING_DATA_FIELD_DEF(bool, useExpCast);\nEND_TILING_DATA_DEF;")
    return patches.get(basename, lambda x: x)(data)
