// D=DV=128, Q quant block=128, K quant block=256, KV tile=128, FP16 mean.
// Keep the generic 0.5.0 pipeline intact as a same-run numerical/speed control.
#pragma once

template<int Rows, bool Tail = false>
__device__ __forceinline__ void copy_d128(uint8_t* dst, const uint8_t* src, int rows = Rows) {
    #pragma unroll 1
    for (int i = threadIdx.x * 16; i < Rows * 128; i += 128 * 16) {
        const int row = i / 128, col = i % 128;
        if constexpr (Tail) {
            const bool valid = row < rows;
            copy16_async(cell(dst, row, col), valid ? src + i : src, valid);
        } else copy16_async(cell(dst, row, col), src + i, true);
    }
}

template<bool ExpCast, bool Emulate, bool QueryTail, bool KeyTail>
__device__ __forceinline__ void d128_step(
    const Shape& s, const Buffers& b, int head, int query, int row, int t, int tile,
    uint8_t* sq, uint8_t* sk, uint8_t* sv, uint8_t* sp,
    float qscale, float (&out)[16][4], float (&maximum)[2], float (&denom)[2]) {
    const int64_t tile_head = int64_t(head) * s.nk_tiles + tile;
    copy_d128<128>(sv, b.v8 + tile_head * (128 * 128));
    copy_commit();
    float fragment[16][4] = {};
    tile_product<Emulate>(fragment, sq, sk);
    const float kscale = b.ks[head * s.nk_quant_tiles + (tile >> 1)];
    float tilemax[2] = {-CUDART_INF_F, -CUDART_INF_F};
    #pragma unroll
    for (int n = 0; n < 16; ++n) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            bool valid = true;
            if constexpr (QueryTail) valid = query + (j / 2) * 8 < s.nq;
            if constexpr (KeyTail) valid = valid && tile * 128 + n * 8 + t * 2 + j % 2 < s.nk;
            fragment[n][j] = valid ? ((fragment[n][j] * qscale) * kscale) * s.scale : -CUDART_INF_F;
            tilemax[j / 2] = fmaxf(tilemax[j / 2], fragment[n][j]);
        }
    }
    float alpha[2], mass[2] = {0, 0};
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const float reduced = rowmax(tilemax[r]);
        float m = fmaxf(maximum[r], reduced);
        if constexpr (QueryTail) if (query + 8 * r >= s.nq) m = 0;
        alpha[r] = expf(maximum[r] - m);
        maximum[r] = m;
    }
    #pragma unroll
    for (int n = 0; n < 16; ++n) {
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            uint16_t packed = 0;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float shifted = fragment[n][r * 2 + j] - maximum[r];
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
                packed |= uint16_t(code) << (j * 8);
                mass[r] += p;
            }
            *reinterpret_cast<uint16_t*>(cell(sp, row + r * 8, n * 8 + t * 2)) = packed;
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        mass[r] = rowsum(mass[r]);
        denom[r] = alpha[r] * denom[r] + mass[r];
    }
    copy_wait(); // V and P ready; every QK consumer has released K.
    if constexpr (!KeyTail) {
        if (tile + 1 < s.nk_tiles) {
            copy_d128<128>(sk, b.k8 + (int64_t(head) * s.pk + (tile + 1) * 128) * 128);
            copy_commit();
        }
    }
    float product[16][4] = {};
    tile_product<Emulate>(product, sp, sv);
    #pragma unroll
    for (int n = 0; n < 16; ++n) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = n * 8 + t * 2 + j % 2;
            const int64_t meta = tile_head * 128 + col;
            const float mean = __half2float(static_cast<const __half*>(b.mean)[meta]);
            // Preserve each FP32 rounding point of the generic contract.
            const float contribution = (product[n][j] * (1.0f / 256.0f) + mass[j / 2] * mean) * b.vs[meta];
            out[n][j] = alpha[j / 2] * out[n][j] + contribution;
        }
    }
    copy_wait(); // Next K ready; release V/P; drain all copies at the tail.
}

template<bool ExpCast, bool Emulate, bool QueryTail>
__global__ __launch_bounds__(128, 2) void fused_attention_d128(Shape s, Buffers b) {
    __shared__ __align__(1024) uint8_t sq[64 * 128];
    __shared__ __align__(1024) uint8_t sk[128 * 128];
    __shared__ __align__(1024) uint8_t sv[128 * 128];
    __shared__ __align__(1024) uint8_t sp[64 * 128];
    const int lane = threadIdx.x % 32, t = lane % 4;
    const int row = threadIdx.x / 32 * 16 + lane / 4;
    const int full_qblocks = s.nq / 64;
    // The full-row kernel is never launched when full_qblocks==0.
    const int head = QueryTail ? blockIdx.x : blockIdx.x / full_qblocks;
    const int start = (QueryTail ? full_qblocks : blockIdx.x % full_qblocks) * 64;
    const int query = start + row;
    copy_d128<64, QueryTail>(sq, b.q8 + (int64_t(head) * s.pq + start) * 128, s.nq - start);
    copy_d128<128>(sk, b.k8 + int64_t(head) * s.pk * 128);
    copy_commit();
    float out[16][4] = {};
    float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F}, denom[2] = {0, 0};
    // A 64-row CTA cannot cross a 128-row Q quantization group.
    const float qscale = b.qs[head * s.nq_tiles + (start >> 7)];
    copy_wait();
    const int full_ktiles = s.nk / 128;
    #pragma unroll 1
    for (int tile = 0; tile < full_ktiles; ++tile)
        d128_step<ExpCast, Emulate, QueryTail, false>(s, b, head, query, row, t, tile,
            sq, sk, sv, sp, qscale, out, maximum, denom);
    if (s.nk % 128)
        d128_step<ExpCast, Emulate, QueryTail, true>(s, b, head, query, row, t, full_ktiles,
            sq, sk, sv, sp, qscale, out, maximum, denom);
    #pragma unroll
    for (int n = 0; n < 16; ++n) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = query + (j / 2) * 8, col = n * 8 + t * 2 + j % 2;
            if (!QueryTail || r < s.nq)
                b.output[(int64_t(head) * s.nq + r) * 128 + col] = out[n][j] / denom[j / 2];
        }
    }
}

template<bool ExpCast, bool Emulate>
void launch_d128(const Shape& s, const Buffers& b, cudaStream_t stream) {
    if (s.nq >= 64)
        fused_attention_d128<ExpCast, Emulate, false><<<s.heads * (s.nq / 64), 128, 0, stream>>>(s, b);
    if (s.nq % 64)
        fused_attention_d128<ExpCast, Emulate, true><<<s.heads, 128, 0, stream>>>(s, b);
}
