// D128 lookahead: QK(next) and PV(current) are separate WGMMA groups.
// After wait_group 1, Softmax(next) runs with PV(current) still outstanding.
// Two P buffers keep the next probabilities separate from pending PV readers.
#pragma once

#include "wgmma_async.cuh"

template<bool ExpCast, bool KeyTail>
__device__ __forceinline__ void interleaved_probability(
    float (&scores)[16][4], uint8_t* sp, const Shape& s,
    float qscale, float kscale, int tile, int row, int t,
    float (&maximum)[2], float (&denom)[2], float (&alpha)[2], float (&mass)[2]) {
    float tilemax[2] = {-CUDART_INF_F, -CUDART_INF_F};
    #pragma unroll
    for (int n = 0; n < 16; ++n) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const bool valid = !KeyTail || tile * 128 + n * 8 + t * 2 + j % 2 < s.nk;
            scores[n][j] = valid ? ((scores[n][j] * qscale) * kscale) * s.scale : -CUDART_INF_F;
            tilemax[j / 2] = fmaxf(tilemax[j / 2], scores[n][j]);
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const float m = fmaxf(maximum[r], rowmax(tilemax[r]));
        alpha[r] = expf(maximum[r] - m);
        maximum[r] = m;
        mass[r] = 0;
    }
    #pragma unroll
    for (int n = 0; n < 16; ++n) {
        uint32_t packed = 0;
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float shifted = scores[n][r * 2 + j] - maximum[r];
                uint8_t code;
                float p;
                if constexpr (ExpCast) {
                    const float u = __fmaf_rn(shifted, 11.541560173034668f, 119.6500015258789f);
                    code = static_cast<uint8_t>(__float2int_rn(fminf(120.0f, fmaxf(0.0f, u))));
                    p = decode(code) * (1.0f / 256.0f);
                } else {
                    p = expf(shifted);
                    code = __nv_cvt_float_to_fp8(p * 256.0f, __NV_SATFINITE, __NV_E4M3);
                }
                packed |= uint32_t(code) << ((r * 2 + j) * 8);
                mass[r] += p;
            }
        }
        *reinterpret_cast<uint16_t*>(cell(sp, row, n * 8 + t * 2)) = uint16_t(packed);
        *reinterpret_cast<uint16_t*>(cell(sp, row + 8, n * 8 + t * 2)) = uint16_t(packed >> 16);
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        mass[r] = rowsum(mass[r]);
        denom[r] = alpha[r] * denom[r] + mass[r];
    }
}

__device__ __forceinline__ void interleaved_accumulate(
    float (&out)[16][4], const float (&product)[16][4], const Shape& s, const Buffers& b,
    int head, int tile, int t, const float (&alpha)[2], const float (&mass)[2]) {
    const int64_t meta_base = (int64_t(head) * s.nk_tiles + tile) * 128;
    #pragma unroll
    for (int n = 0; n < 16; ++n) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t meta = meta_base + n * 8 + t * 2 + j % 2;
            const float mean = __half2float(static_cast<const __half*>(b.mean)[meta]);
            const float contribution = (product[n][j] * (1.0f / 256.0f) + mass[j / 2] * mean) * b.vs[meta];
            out[n][j] = alpha[j / 2] * out[n][j] + contribution;
        }
    }
}

template<bool ExpCast, bool Emulate, bool NextKeyTail>
__device__ __forceinline__ void interleaved_step(
    const Shape& s, const Buffers& b, int head, int tile, int row, int t, float qscale,
    uint8_t* sq, uint8_t* sk, uint8_t* sv, uint8_t* sp, uint8_t* next_p,
    float (&out)[16][4], float (&maximum)[2], float (&denom)[2], float (&alpha)[2], float (&mass)[2]) {
    float scores[16][4] = {}, product[16][4] = {};
    if constexpr (Emulate) tile_product<true>(scores, sq, sk);
    else {
        wgmma_issue(scores, sq, sk);    // Oldest group: QK(next).
        wgmma_issue(product, sp, sv);   // Newest group: PV(current).
        wgmma_ready<1>(scores);        // PV may still be running.
    }
    __syncthreads(); // All QK readers have released K; P/V remain read-only.
    if (tile + 2 < s.nk_tiles) {
        copy_d128<128>(sk, b.k8 + (int64_t(head) * s.pk + (tile + 2) * 128) * 128);
        copy_commit();
    }
    const float previous_alpha[2] = {alpha[0], alpha[1]}, previous_mass[2] = {mass[0], mass[1]};
    interleaved_probability<ExpCast, NextKeyTail>(scores, next_p, s, qscale,
        b.ks[head * s.nk_quant_tiles + ((tile + 1) >> 1)], tile + 1, row, t, maximum, denom, alpha, mass);
    __syncthreads(); // Publish NEW P, while PV still reads the separate OLD P.
    // The emulator deliberately consumes OLD P/V after Softmax(next) to catch
    // accidental early overwrites. It cannot validate native WGMMA concurrency.
    if constexpr (Emulate) tile_product<true>(product, sp, sv);
    else wgmma_ready<0>(product);
    __syncthreads(); // All PV consumers have released P and V.
    copy_d128<128>(sv, b.v8 + (int64_t(head) * s.nk_tiles + tile + 1) * (128 * 128));
    copy_commit();
    interleaved_accumulate(out, product, s, b, head, tile, t, previous_alpha, previous_mass);
    copy_wait(); // Next K/V and published P are ready before the next issue.
}

template<bool ExpCast, bool Emulate>
__global__ __launch_bounds__(128, 2) void fused_attention_interleaved(Shape s, Buffers b) {
    __shared__ __align__(1024) uint8_t sq[64 * 128];
    __shared__ __align__(1024) uint8_t sk[128 * 128];
    __shared__ __align__(1024) uint8_t sv[128 * 128];
    __shared__ __align__(1024) uint8_t sp[64 * 128];
    extern __shared__ __align__(1024) uint8_t alternate_p[];
    const int t = threadIdx.x % 4, row = threadIdx.x / 32 * 16 + (threadIdx.x % 32) / 4;
    const int full_qblocks = s.nq / 64;
    const int head = blockIdx.x / full_qblocks, start = blockIdx.x % full_qblocks * 64;
    copy_d128<64>(sq, b.q8 + (int64_t(head) * s.pq + start) * 128);
    copy_d128<128>(sk, b.k8 + int64_t(head) * s.pk * 128);
    copy_commit();
    float out[16][4] = {}, maximum[2] = {-CUDART_INF_F, -CUDART_INF_F}, denom[2] = {0, 0};
    float alpha[2], mass[2];
    const float qscale = b.qs[head * s.nq_tiles + (start >> 7)];
    copy_wait();
    // Prologue: prepare P(0), V(0), K(1).
    {
        copy_d128<128>(sv, b.v8 + int64_t(head) * s.nk_tiles * (128 * 128));
        copy_commit();
        float scores[16][4] = {};
        tile_product<Emulate>(scores, sq, sk);
        copy_wait(); // V ready; all QK consumers have released K.
        if (s.nk_tiles > 1) {
            copy_d128<128>(sk, b.k8 + (int64_t(head) * s.pk + 128) * 128);
            copy_commit();
        }
        const float kscale = b.ks[head * s.nk_quant_tiles];
        if (s.nk < 128)
            interleaved_probability<ExpCast, true>(scores, sp, s, qscale, kscale, 0, row, t, maximum, denom, alpha, mass);
        else
            interleaved_probability<ExpCast, false>(scores, sp, s, qscale, kscale, 0, row, t, maximum, denom, alpha, mass);
        copy_wait();
    }
    const int full_ktiles = s.nk / 128;
    #pragma unroll 1
    for (int tile = 0; tile + 1 < full_ktiles; ++tile)
        interleaved_step<ExpCast, Emulate, false>(s, b, head, tile, row, t, qscale,
            sq, sk, sv, tile % 2 ? alternate_p : sp, tile % 2 ? sp : alternate_p,
            out, maximum, denom, alpha, mass);
    if (s.nk % 128 && full_ktiles > 0)
        interleaved_step<ExpCast, Emulate, true>(s, b, head, full_ktiles - 1, row, t, qscale,
            sq, sk, sv, (full_ktiles - 1) % 2 ? alternate_p : sp, (full_ktiles - 1) % 2 ? sp : alternate_p,
            out, maximum, denom, alpha, mass);
    // Drain the final PV; all preceding outputs were updated in key-tile order.
    float product[16][4] = {};
    tile_product<Emulate>(product, (s.nk_tiles - 1) % 2 ? alternate_p : sp, sv);
    interleaved_accumulate(out, product, s, b, head, s.nk_tiles - 1, t, alpha, mass);
    #pragma unroll
    for (int n = 0; n < 16; ++n) {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            b.output[(int64_t(head) * s.nq + start + row + (j / 2) * 8) * 128 + n * 8 + t * 2 + j % 2]
                = out[n][j] / denom[j / 2];
    }
}

template<bool ExpCast, bool Emulate>
void launch_interleaved(const Shape& s, const Buffers& b, cudaStream_t stream) {
    if (s.nq >= 64) {
        // 48 KiB static + 8 KiB dynamic: opt in to >48 KiB shared per CTA.
        const auto status = cudaFuncSetAttribute(fused_attention_interleaved<ExpCast, Emulate>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 128);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("Interleaved shared memory: ") + cudaGetErrorString(status));
        fused_attention_interleaved<ExpCast, Emulate><<<s.heads * (s.nq / 64), 128, 64 * 128, stream>>>(s, b);
    }
    if (s.nq % 64)
        fused_attention_d128<ExpCast, Emulate, true><<<s.heads, 128, 0, stream>>>(s, b);
}
