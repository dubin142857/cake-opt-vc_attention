// Two shared-memory buffers overlap V loads with QK/softmax and next-K loads
// with PV/output updates. The 0.4.0 synchronous kernel remains an A/B control.
#pragma once

__device__ __forceinline__ void copy16_async(uint8_t* dst, const uint8_t* src, bool valid) {
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    // Padded channels/rows are zero-filled, using an in-bounds source address
    // even for ignored copies. All payload pitches and destinations align to 16B.
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

template<int Rows>
__device__ __forceinline__ void copy_tile_async(uint8_t* dst, const uint8_t* src,
                                               int valid_rows, int width) {
    #pragma unroll 1
    for (int i = threadIdx.x * 16; i < Rows * 128; i += 128 * 16) {
        const int row = i / 128, col = i % 128;
        const bool valid = row < valid_rows && col < width;
        copy16_async(cell(dst, row, col), valid ? src + int64_t(row) * width + col : src, valid);
    }
}

__device__ __forceinline__ void copy_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void copy_wait() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    // wait_group only makes this thread's copies visible to itself. Publish
    // all producers to the CTA before any warp reads or overwrites a buffer.
    __syncthreads();
}

template<bool ExpCast, bool Emulate>
__global__ __launch_bounds__(128, 2) void fused_attention_pipeline(Shape s, Buffers b) {
    __shared__ __align__(1024) uint8_t sq[64 * 128];
    __shared__ __align__(1024) uint8_t sk[128 * 128];
    __shared__ __align__(1024) uint8_t sv[128 * 128];
    __shared__ __align__(1024) uint8_t sp[64 * 128];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int group = lane / 4, t = lane % 4;
    const int qblocks = (s.nq + 63) / 64;
    const int head = blockIdx.x / qblocks, start = blockIdx.x % qblocks * 64;
    const int row = warp * 16 + group, query = start + row;
    const uint8_t* kp = b.k8 + int64_t(head) * s.pk * s.dp;
    const uint8_t* vp = b.v8 + int64_t(head) * s.nk_tiles * s.vp * 128;
    copy_tile_async<64>(sq, b.q8 + (int64_t(head) * s.pq + start) * s.dp,
                        s.nq - start, s.dp);
    copy_tile_async<128>(sk, kp, 128, s.dp);
    copy_commit();
    float out[16][4] = {};
    float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F}, denom[2] = {0, 0};
    const float qscale[2] = {
        query < s.nq ? b.qs[head * s.nq_tiles + query / s.bq] : 1,
        query + 8 < s.nq ? b.qs[head * s.nq_tiles + (query + 8) / s.bq] : 1};
    copy_wait();
    for (int tile = 0; tile < s.nk_tiles; ++tile) {
        // The prior iteration's final barrier guarantees all PV consumers
        // have released sv. Copies run while QK and probability work proceed.
        copy_tile_async<128>(sv, vp + int64_t(tile) * s.vp * 128, s.vp, 128);
        copy_commit();
        float fragment[16][4] = {};
        tile_product<Emulate>(fragment, sq, sk);
        const float kscale = b.ks[head * s.nk_quant_tiles + tile * 128 / s.k_quant];
        float tilemax[2] = {-CUDART_INF_F, -CUDART_INF_F};
        #pragma unroll
        for (int n = 0; n < 16; ++n) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const bool valid = query + (j / 2) * 8 < s.nq && tile * 128 + n * 8 + t * 2 + j % 2 < s.nk;
                fragment[n][j] = valid ? ((fragment[n][j] * qscale[j / 2]) * kscale) * s.scale : -CUDART_INF_F;
                tilemax[j / 2] = fmaxf(tilemax[j / 2], fragment[n][j]);
            }
        }
        float alpha[2], mass[2] = {0, 0};
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const float reduced = rowmax(tilemax[r]);
            const float m = query + 8 * r < s.nq ? fmaxf(maximum[r], reduced) : 0;
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
        // Wait for V, publish P, and ensure every warp has finished reading K.
        copy_wait();
        if (tile + 1 < s.nk_tiles) {
            copy_tile_async<128>(sk, kp + int64_t(tile + 1) * 128 * s.dp, 128, s.dp);
            copy_commit();
        }
        float product[16][4] = {};
        tile_product<Emulate>(product, sp, sv);
        #pragma unroll
        for (int n = 0; n < 16; ++n) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = n * 8 + t * 2 + j % 2;
                if (col < s.dv) {
                    const int64_t meta = (int64_t(head) * s.nk_tiles + tile) * s.vp + col;
                    const float contribution = (product[n][j] * (1.0f / 256.0f) + mass[j / 2] * mean_at(b, s, meta)) * b.vs[meta];
                    out[n][j] = alpha[j / 2] * out[n][j] + contribution;
                }
            }
        }
        // Drain next-K copies before use, and release V/P before reuse.
        // This also drains all outstanding copies on the final iteration.
        copy_wait();
    }
    #pragma unroll
    for (int n = 0; n < 16; ++n) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = query + (j / 2) * 8, col = n * 8 + t * 2 + j % 2;
            if (r < s.nq && col < s.dv)
                b.output[(int64_t(head) * s.nq + r) * s.dv + col] = out[n][j] / denom[j / 2];
        }
    }
}
