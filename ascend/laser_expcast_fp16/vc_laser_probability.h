// Experimental binary16 ExpCast probability transform. QK/PV, online history
// and denominator reduction remain in the separately preserved Laser kernel.
#ifndef VC_LASER_FP16_PROBABILITY_H
#define VC_LASER_FP16_PROBABILITY_H

#ifdef __DAV_C220_VEC__
// Legacy CCE signatures and RNE conversion:
// https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/API/cceintrinsicapi/cceapi_0049.html
// Half/float vmuls signatures:
// https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/cceintrinsicapi/cceapi_0015.html
// These declarations are documented, but this candidate still requires actual
// CANN 9.0.1 compilation and device validation, including subnormal outputs.
// ExpCast derivation: https://arxiv.org/html/2609.15810v1#S3.SS3
// This binary16 adaptation is NOT the paper's E4M3 encoding or native FP8 PV.

// Caller contract: count > 0 and divisible by 128, score contains FP16 values
// <= 0 (finite or masked -inf). Scratch is the existing, now-dead 32 KiB
// row-max region, aligned to 32 B and disjoint from score. The subsequent row
// sum rebuilds its scratch. No new UB allocation, GM trace, or events are used.
__aicore__ inline void vc_laser_expcast_probability(
    __ubuf__ half *score, __ubuf__ float *scratch, const int32_t count)
{
    constexpr int32_t CHUNK_ELEMENTS = 8192;
    constexpr float GAIN = 1477.3197021484375f;
    constexpr float BIAS = 27603.19921875f;
    constexpr float CODE_MAX = 27648.0f;
    for (int32_t offset = 0; offset < count; offset += CHUNK_ELEMENTS) {
        const int32_t remaining = count - offset;
        const int32_t chunk = remaining < CHUNK_ELEMENTS ? remaining : CHUNK_ELEMENTS;
        const uint8_t repeats32 = static_cast<uint8_t>(chunk / 64);
        const uint8_t repeats16 = static_cast<uint8_t>(chunk / 128);
        __ubuf__ int16_t *codes = reinterpret_cast<__ubuf__ int16_t *>(score + offset);

        // Conversions process 64 elements/repeat; strides are 32-byte blocks.
        set_vector_mask(0, static_cast<uint64_t>(-1));
        vconv_f162f32(scratch, score + offset, repeats32, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        // CANN 9.0.1 legacy builtin matching requires an explicit float value
        // here; ordinary host C++ accepts the constexpr lvalue but CCE did not.
        vmuls(scratch, scratch, static_cast<float>(GAIN), repeats32, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadds(scratch, scratch, static_cast<float>(BIAS), repeats32, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmaxs(scratch, scratch, 0.0f, repeats32, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmins(scratch, scratch, static_cast<float>(CODE_MAX), repeats32, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322s16r(codes, scratch, repeats32, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);

        // The int16 bytes ALREADY encode binary16 approx(exp(x) * 4096).
        // Offset +12 keeps values contributing to a nonzero FP16 result in
        // the normal encoded range. This multiply supplies rounded subnormal
        // P instead of the premature cutoff from a bias-15 direct mapping.
        // No integer-to-half numeric conversion, E4M3 decode, or byte shift.
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        vmuls(score + offset, score + offset, static_cast<half>(0.000244140625f),
              repeats16, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}
#endif
#endif
