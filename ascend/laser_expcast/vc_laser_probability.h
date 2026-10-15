// Experimental probability transform for the pinned MindIE Laser vector kernel.
// QK, V, online-history scaling, denominator reduction and PV remain upstream.
#ifndef VC_LASER_PROBABILITY_H
#define VC_LASER_PROBABILITY_H

#ifdef __DAV_C220_VEC__
// Old CCE interfaces (not Reg/SIMT APIs):
// https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/API/cceintrinsicapi/cceapi_0049.html
// https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/API/cceintrinsicapi/cceapi_0016.html
// https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/API/cceintrinsicapi/cceapi_0042.html
// Float -> int16 RINT is also listed for Atlas A2 in the CANN 8.0 Cast table.
// Intrinsic existence/documentation is not a substitute for remote CCE validation.

// Preconditions supplied by the pinned VectorForward patch:
// - count is a positive multiple of 128; shifted scores are FP16 S - online max.
// - scratch is the 32 KiB, 32-byte-aligned former row-max reduction region.
// - score and scratch do not overlap. Row max has finished; row sum rebuilds
//   its scratch after this function. No newly allocated UB or events are used.
__aicore__ inline void vc_laser_expcast_probability(
    __ubuf__ half *score, __ubuf__ float *scratch, const int32_t count)
{
    constexpr int32_t CHUNK_ELEMENTS = 8192;
    constexpr float GAIN = 11.541560173034668f;
    constexpr float BIAS = 119.6500015258789f;
    for (int32_t offset = 0; offset < count; offset += CHUNK_ELEMENTS) {
        const int32_t remaining = count - offset;
        const int32_t chunk = remaining < CHUNK_ELEMENTS ? remaining : CHUNK_ELEMENTS;
        const uint8_t repeats32 = static_cast<uint8_t>(chunk / 64);
        const uint8_t repeats16 = static_cast<uint8_t>(chunk / 128);
        __ubuf__ int16_t *codes = reinterpret_cast<__ubuf__ int16_t *>(score + offset);

        // Wide conversion processes 64 elements/repeat. Source/destination
        // repeat strides are in 32-byte blocks: half 4, float 8.
        set_vector_mask(0, static_cast<uint64_t>(-1));
        vconv_f162f32(scratch, score + offset, repeats32, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        // Explicit two-round FP32 contract, unlike the CUDA FMA reference.
        vmuls(scratch, scratch, static_cast<float>(GAIN), repeats32, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadds(scratch, scratch, static_cast<float>(BIAS), repeats32, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmaxs(scratch, scratch, 0.0f, repeats32, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmins(scratch, scratch, 120.0f, repeats32, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322s16r(codes, scratch, repeats32, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);

        // For EVERY E4M3 code c in [0,120], including subnormals:
        // bits_fp16(decode_e4m3(c) / 256) == uint16(c << 7).
        // Therefore the in-place bit shift is the complete FP16 decoder.
        // P <= 1 retains Laser's FP16 per-section row-sum range and the
        // unchanged online recurrence; no factor 256 enters PV or row sums.
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        vshl(codes, codes, 7, repeats16, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }
    // Caller expects the full ordinary mask, including process_calc_sum.
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}
#endif
#endif
