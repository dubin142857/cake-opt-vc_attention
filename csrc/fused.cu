// Hopper WGMMA FP8 attention. Preprocessing remains separate.
#include "vc_cuda.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <stdexcept>
#include <string>

#if __CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 4)
#error "VC fused attention build requires nvcc >= 12.4."
#endif

namespace vc {
namespace {

#include "wgmma.cuh"

__device__ __forceinline__ float decode(uint8_t x) {
    const int e = (x & 127) >> 3, m = x & 7;
    // Exact finite E4M3 decode with integer exponent/mantissa placement. In
    // particular, ExpCast's row sum must not introduce another exponential.
    const float f = e ? __uint_as_float(uint32_t(e + 120) << 23 | uint32_t(m) << 20)
                      : m * (1.0f / 512.0f);
    return x & 128 ? -f : f;
}

// The scalar version is exposed ONLY by debug_fused_emulated for tests on
// non-Hopper development GPUs. Production always instantiates Emulate=false.
__device__ __forceinline__ void mma_emulated(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
        const int lane = threadIdx.x & 31, group = lane >> 2, t = lane & 3;
        #pragma unroll 1
        for (int k = 0; k < 32; ++k) {
            const int src_a = group * 4 + (k % 16) / 4;
            const uint32_t a0 = __shfl_sync(0xffffffff, a[k / 16 * 2], src_a);
            const uint32_t a1 = __shfl_sync(0xffffffff, a[k / 16 * 2 + 1], src_a);
            const uint32_t b0 = __shfl_sync(0xffffffff, b[k / 16], (t * 2) * 4 + (k % 16) / 4);
            const uint32_t b1 = __shfl_sync(0xffffffff, b[k / 16], (t * 2 + 1) * 4 + (k % 16) / 4);
            const int shift = (k % 4) * 8;
            const float x = decode(uint8_t(a0 >> shift)), y = decode(uint8_t(a1 >> shift));
            const float u = decode(uint8_t(b0 >> shift)), v = decode(uint8_t(b1 >> shift));
            c[0] = __fmaf_rn(x, u, c[0]); c[1] = __fmaf_rn(x, v, c[1]);
            c[2] = __fmaf_rn(y, u, c[2]); c[3] = __fmaf_rn(y, v, c[3]);
        }
}

__device__ __forceinline__ uint32_t word(const uint8_t* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ uint8_t* cell(uint8_t* p, int row, int col) {
    // XOR-swizzle 16-byte groups: avoid the 8-way bank conflict that a plain
    // 128-byte row pitch gives the four-lanes-per-row MMA register loads.
    return p + row * 128 + (col ^ ((row & 7) * 16));
}
template<bool Emulate>
__device__ __forceinline__ void tile_product(float (&c)[16][4], uint8_t* a, uint8_t* b) {
    if constexpr (Emulate) {
        const int lane = threadIdx.x % 32, group = lane / 4, t = lane % 4;
        const int row = threadIdx.x / 32 * 16 + group;
        #pragma unroll
        for (int n = 0; n < 16; ++n) {
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const uint32_t ar[4] = {
                    word(cell(a, row, kk * 32 + t * 4)), word(cell(a, row + 8, kk * 32 + t * 4)),
                    word(cell(a, row, kk * 32 + t * 4 + 16)), word(cell(a, row + 8, kk * 32 + t * 4 + 16))};
                const uint32_t br[2] = {
                    word(cell(b, n * 8 + group, kk * 32 + t * 4)),
                    word(cell(b, n * 8 + group, kk * 32 + t * 4 + 16))};
                mma_emulated(c[n], ar, br);
            }
        }
    } else wgmma_product(c, a, b);
}
__device__ __forceinline__ float rowmax(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    return fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
}
__device__ __forceinline__ float rowsum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v + __shfl_xor_sync(0xffffffff, v, 2);
}
__device__ __forceinline__ float mean_at(const Buffers& b, const Shape& s, int64_t i) {
    return s.mean_type == 1 ? __half2float(static_cast<const __half*>(b.mean)[i])
         : s.mean_type == 2 ? __bfloat162float(static_cast<const __nv_bfloat16*>(b.mean)[i])
         : static_cast<const float*>(b.mean)[i];
}

// Four warps own 16 query rows each. KV tiles are shared by all four warps;
// scores, output accumulators and row state stay in registers. Q and P have
// shared-memory storage only. KV storage is reused between the two products.
template<bool ExpCast, bool Emulate>
__global__ void fused_attention(Shape s, Buffers b) {
    __shared__ __align__(1024) uint8_t sq[64 * 128];
    __shared__ __align__(1024) uint8_t skv[128 * 128];
    __shared__ __align__(1024) uint8_t sp[64 * 128];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int group = lane / 4, t = lane % 4;
    const int qblocks = (s.nq + 63) / 64;
    const int head = blockIdx.x / qblocks, start = blockIdx.x % qblocks * 64;
    const int row = warp * 16 + group, query = start + row;
    for (int i = tid * 4; i < 64 * 128; i += 128 * 4) {
        const int r = i / 128, d = i % 128;
        *reinterpret_cast<uint32_t*>(cell(sq, r, d)) = start + r < s.nq && d < s.dp
            ? word(b.q8 + (int64_t(head) * s.pq + start + r) * s.dp + d) : 0;
    }
    __syncthreads();
    float out[16][4] = {};
    float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F}, denom[2] = {0, 0};
    const float qscale[2] = {
        query < s.nq ? b.qs[head * s.nq_tiles + query / s.bq] : 1,
        query + 8 < s.nq ? b.qs[head * s.nq_tiles + (query + 8) / s.bq] : 1};
    for (int tile = 0; tile < s.nk_tiles; ++tile) {
        for (int i = tid * 4; i < 128 * 128; i += 128 * 4) {
            const int r = i / 128, d = i % 128;
            *reinterpret_cast<uint32_t*>(cell(skv, r, d)) = d < s.dp
                ? word(b.k8 + (int64_t(head) * s.pk + tile * 128 + r) * s.dp + d) : 0;
        }
        __syncthreads();
        float scores[16][4] = {};
        tile_product<Emulate>(scores, sq, skv);
        const float kscale = b.ks[head * s.nk_quant_tiles + tile * 128 / s.k_quant];
        float tilemax[2] = {-CUDART_INF_F, -CUDART_INF_F};
        #pragma unroll
        for (int n = 0; n < 16; ++n) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const bool valid = query + (j / 2) * 8 < s.nq && tile * 128 + n * 8 + t * 2 + j % 2 < s.nk;
                scores[n][j] = valid ? ((scores[n][j] * qscale[j / 2]) * kscale) * s.scale : -CUDART_INF_F;
                tilemax[j / 2] = fmaxf(tilemax[j / 2], scores[n][j]);
            }
        }
        float alpha[2], mass[2] = {0, 0};
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const float reduced = rowmax(tilemax[r]);  // All lanes participate, including padded rows.
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
        // Every warp must finish reading K before the cooperative V load.
        __syncthreads();
        for (int i = tid * 4; i < 128 * 128; i += 128 * 4) {
            const int d = i / 128, r = i % 128;
            *reinterpret_cast<uint32_t*>(cell(skv, d, r)) = d < s.vp
                ? word(b.v8 + ((int64_t(head) * s.nk_tiles + tile) * s.vp + d) * 128 + r) : 0;
        }
        __syncthreads();
        float product[16][4] = {};
        tile_product<Emulate>(product, sp, skv);
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
        __syncthreads();  // No warp may overwrite V while another reads it.
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

#include "fused_pipeline.cuh"
#include "fused_d128.cuh"

template<bool Emulate>
void launch(const Shape& s, const Buffers& b, cudaStream_t stream, bool pipeline, bool specialized) {
    const int blocks = s.heads * ((s.nq + 63) / 64);
    if (specialized) {
        if (s.expcast) launch_d128<true, Emulate>(s, b, stream);
        else launch_d128<false, Emulate>(s, b, stream);
    } else if (pipeline) {
        if (s.expcast) fused_attention_pipeline<true, Emulate><<<blocks, 128, 0, stream>>>(s, b);
        else fused_attention_pipeline<false, Emulate><<<blocks, 128, 0, stream>>>(s, b);
    } else if (s.expcast) fused_attention<true, Emulate><<<blocks, 128, 0, stream>>>(s, b);
    else fused_attention<false, Emulate><<<blocks, 128, 0, stream>>>(s, b);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(std::string("Fused attention launch: ") + cudaGetErrorString(status));
}
}  // namespace

void forward_fused(const float* q, const float* k, const float* v,
                   const Shape& s, const Buffers& b, cudaStream_t stream,
                   bool debug_emulate, bool pipeline, bool specialized) {
    debug_quantize(q, b.q8, b.qs, s.heads, s.nq, s.pq, s.d, s.dp, s.bq, s.nq_tiles, stream);
    debug_quantize(k, b.k8, b.ks, s.heads, s.nk, s.pk, s.d, s.dp, s.k_quant, s.nk_quant_tiles, stream);
    debug_values(v, s, b, stream);
    if (debug_emulate) launch<true>(s, b, stream, pipeline, specialized);
    else launch<false>(s, b, stream, pipeline, specialized);
}
}  // namespace vc
