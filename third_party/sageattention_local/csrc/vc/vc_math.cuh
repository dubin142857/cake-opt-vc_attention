// VC additions to the SageAttention Hopper core. Apache-2.0; see NOTICE.
#pragma once
#include "expcast.cuh"

__device__ __forceinline__ float2 vc_decode_pair(uint32_t codes) {
  // One packed FP8 -> half2 instruction on Hopper, then two exact FP32
  // conversions. Decoding each byte separately discards half of each result.
  __half2 pair = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(codes), __NV_E4M3);
  return __half22float2(pair);
}

template<bool ExpCast, bool BlockValues, bool Optimized=false, bool PackedProbability=false, uint32_t NQ, uint32_t NK, uint32_t NV>
__device__ __forceinline__ void vc_probability(
    float (&scores)[NQ][NK][8], float (&out)[NQ][NV][8],
    float (&maximum)[NQ][2], float (&denom)[NQ][2], float scale,
    uint32_t (&packed)[NQ][NK / 2][4], float (&mass)[NQ][2]) {
  if constexpr (ExpCast) {
    #pragma unroll
    for (int q = 0; q < NQ; ++q) {
      #pragma unroll
      for (int r = 0; r < 2; ++r) {
        float local_max = -CUDART_INF_F;
        if constexpr (Optimized) {
          // Four independent max chains instead of one 32-element chain.
          // Scale is nonnegative and identical within each lane: RN(scale*x)
          // is monotone, so scale(max(x)) == max(scale(x)). Other lanes can
          // have different K scales; multiply BEFORE the cross-lane shuffle.
          float partial[4] = {-CUDART_INF_F,-CUDART_INF_F,-CUDART_INF_F,-CUDART_INF_F};
          #pragma unroll
          for (int k = 0; k < NK; ++k)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
              partial[j] = fmaxf(partial[j], scores[q][k][r*2+(j/2)*4+j%2]);
          local_max = __fmul_rn(fmaxf(fmaxf(partial[0],partial[1]),fmaxf(partial[2],partial[3])),scale);
        } else {
          #pragma unroll
          for (int k = 0; k < NK; ++k) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
              int slot = r * 2 + (j / 2) * 4 + j % 2;
              float value = __fmul_rn(scores[q][k][slot], scale);
              scores[q][k][slot] = value;
              local_max = fmaxf(local_max, value);
            }
          }
        }
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 1));
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 2));
        float new_max = fmaxf(maximum[q][r], local_max);
        float alpha = expf(maximum[q][r] - new_max);
        maximum[q][r] = new_max;
        denom[q][r] *= alpha;
        #pragma unroll
        for (int v = 0; v < NV; ++v) {
          #pragma unroll
          for (int j = 0; j < 4; ++j)
            out[q][v][r * 2 + (j / 2) * 4 + j % 2] *= alpha;
        }
      }
      if constexpr (PackedProbability) mass[q][0] = mass[q][1] = 0.0f;
      #pragma unroll
      for (int k = 0; k < NK; ++k) {
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
          if constexpr (PackedProbability) {
            // Decode only four probabilities at a time, then immediately
            // consume them. Preserve vc6's left-to-right additions exactly.
            float a = scores[q][k][r*2], b = scores[q][k][r*2+1];
            float c = scores[q][k][r*2+4], e = scores[q][k][r*2+5];
            if constexpr (Optimized) {
              a = __fmul_rn(a,scale); b = __fmul_rn(b,scale);
              c = __fmul_rn(c,scale); e = __fmul_rn(e,scale);
            }
            uint32_t low = vc_expcast_pair(__fsub_rn(a,maximum[q][r]),__fsub_rn(b,maximum[q][r]));
            uint32_t high = vc_expcast_pair(__fsub_rn(c,maximum[q][r]),__fsub_rn(e,maximum[q][r]));
            float2 first = vc_decode_pair(low), second = vc_decode_pair(high);
            float sum = __fadd_rn(__fadd_rn(__fadd_rn(first.x,first.y),second.x),second.y);
            denom[q][r] = __fadd_rn(denom[q][r],sum);
            if constexpr (BlockValues) mass[q][r] = __fadd_rn(mass[q][r],sum);
            packed[q][k/2][(k%2)*2+r] = __byte_perm(low,high,0x5410);
          } else {
            uint32_t bytes = 0;
            #pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
              int j = r * 2 + pair * 4;
              float a = scores[q][k][j], b = scores[q][k][j + 1];
              if constexpr (Optimized) { a = __fmul_rn(a,scale); b = __fmul_rn(b,scale); }
              uint32_t lo = vc_expcast_nonpositive(__fsub_rn(a, maximum[q][r]));
              uint32_t hi = vc_expcast_nonpositive(__fsub_rn(b, maximum[q][r]));
              uint32_t codes = lo | (hi << 8);
              float2 decoded = vc_decode_pair(codes);
              // Preserve decoded-P denominator, mass and addition order exactly.
              scores[q][k][j] = decoded.x;
              scores[q][k][j + 1] = decoded.y;
              bytes |= codes << (16 * pair);
            }
            packed[q][k / 2][(k % 2) * 2 + r] = bytes;
          }
        }
      }
    }
  } else {
    // Preserve upstream's 2^8.807 P scaling and unrounded denominator.
    update_mdo<NQ, NK, NV, false, true, false>(scores, out, maximum, denom, scale);
    RS_32_to_8<NQ, NK>(scores, packed);
  }
  #pragma unroll
  for (int q = 0; q < NQ; ++q) {
    if constexpr (!(ExpCast && PackedProbability)) {
      mass[q][0] = mass[q][1] = 0.0f;
      #pragma unroll
      for (int k = 0; k < NK; ++k) {
        float a = scores[q][k][0] + scores[q][k][1] + scores[q][k][4] + scores[q][k][5];
        float b = scores[q][k][2] + scores[q][k][3] + scores[q][k][6] + scores[q][k][7];
        denom[q][0] += a;
        denom[q][1] += b;
        if constexpr (BlockValues) { mass[q][0] += a; mass[q][1] += b; }
      }
    }
    if constexpr (BlockValues) {
      #pragma unroll
      for (int r = 0; r < 2; ++r) {
        mass[q][r] += __shfl_xor_sync(0xffffffff, mass[q][r], 1);
        mass[q][r] += __shfl_xor_sync(0xffffffff, mass[q][r], 2);
      }
    }
  }
}

// Keep one half2 mean and its four output updates together. Fully unrolled
// C++ lets the compiler expand many FP32 means before consuming any of them.
// Explicit RN instructions retain vc7's FMA then FP32-history-add contract.
__device__ __forceinline__ void vc_mean_pair(
    float& o0, float& o1, float& o2, float& o3,
    float p0, float p1, float p2, float p3, float m0, float m1,
    const half* mean) {
  asm volatile(
      "{\n"
      " .reg .b32 packed; .reg .b16 lo, hi;\n"
      " .reg .f32 a, b, x0, x1, x2, x3;\n"
      " ld.global.b32 packed, [%10];\n"
      " mov.b32 {lo, hi}, packed;\n"
      " cvt.f32.f16 a, lo; cvt.f32.f16 b, hi;\n"
      " fma.rn.f32 x0, %8, a, %4; fma.rn.f32 x1, %8, b, %5;\n"
      " fma.rn.f32 x2, %9, a, %6; fma.rn.f32 x3, %9, b, %7;\n"
      " add.rn.f32 %0, %0, x0; add.rn.f32 %1, %1, x1;\n"
      " add.rn.f32 %2, %2, x2; add.rn.f32 %3, %3, x3;\n"
      "}\n"
      : "+f"(o0), "+f"(o1), "+f"(o2), "+f"(o3)
      : "f"(p0), "f"(p1), "f"(p2), "f"(p3), "f"(m0), "f"(m1), "l"(mean)
      : "memory");
}

template<bool BlockValues, bool Optimized=false, bool ChannelScale=false, bool StreamMean=false, uint32_t NQ, uint32_t NV, uint32_t NP>
__device__ __forceinline__ void vc_accumulate(
    float (&out)[NQ][NV][8], const float (&product)[NQ][NP][8],
    const float (&mass)[NQ][2], const float* scales, const half* means,
    int head, int tile, int tiles, int dim, int lane, int value_offset=0) {
  #pragma unroll
  for (int v = 0; v < NP; ++v) {
    if constexpr (BlockValues) {
      #pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        int channel = (v + value_offset) * 16 + (lane % 4) * 2 + pair * 8;
        int64_t meta = (int64_t(head) * tiles + tile) * dim + channel;
        if constexpr (StreamMean && Optimized && ChannelScale) {
          #pragma unroll
          for (int q = 0; q < NQ; ++q) {
            int j = pair * 4;
            vc_mean_pair(out[q][v + value_offset][j], out[q][v + value_offset][j+1], out[q][v + value_offset][j+2], out[q][v + value_offset][j+3],
                product[q][v][j], product[q][v][j+1], product[q][v][j+2], product[q][v][j+3],
                mass[q][0], mass[q][1], means + meta);
          }
        } else {
          // Both rows share these two channels. Torch allocations and even
          // channel indices guarantee float2/half2 alignment, including D=64.
          int64_t scale_meta = ChannelScale ? int64_t(head) * dim + channel : meta;
          float2 scale;
          if constexpr (!(ChannelScale && Optimized))
            scale = *reinterpret_cast<const float2*>(scales + scale_meta);
          float2 mean = __half22float2(*reinterpret_cast<const half2*>(means + meta));
          #pragma unroll
          for (int q = 0; q < NQ; ++q) {
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
              int j = pair * 4 + r * 2;
              if constexpr (Optimized) {
                // Fuse the mean correction; legacy block scales need a second FMA.
                // Rounds once per FMA; intentionally not bitwise vc4 arithmetic.
                float a = __fmaf_rn(mass[q][r], mean.x, product[q][v][j]);
                float b = __fmaf_rn(mass[q][r], mean.y, product[q][v][j + 1]);
                if constexpr (ChannelScale) {
                  // Shared scale is applied once after online normalization.
                  // Keep the FP32 history outside the FP8 WGMMA accumulator.
                  out[q][v + value_offset][j] = __fadd_rn(out[q][v + value_offset][j], a);
                  out[q][v + value_offset][j + 1] = __fadd_rn(out[q][v + value_offset][j + 1], b);
                } else {
                  out[q][v + value_offset][j] = __fmaf_rn(a, scale.x, out[q][v + value_offset][j]);
                  out[q][v + value_offset][j + 1] = __fmaf_rn(b, scale.y, out[q][v + value_offset][j + 1]);
                }
              } else {
                float a = __fadd_rn(product[q][v][j], __fmul_rn(mass[q][r], mean.x));
                float b = __fadd_rn(product[q][v][j + 1], __fmul_rn(mass[q][r], mean.y));
                out[q][v + value_offset][j] = __fadd_rn(out[q][v + value_offset][j], __fmul_rn(a, scale.x));
                out[q][v + value_offset][j + 1] = __fadd_rn(out[q][v + value_offset][j + 1], __fmul_rn(b, scale.y));
              }
            }
          }
        }
      }
    } else {
      #pragma unroll
      for (int j = 0; j < 8; ++j)
        #pragma unroll
        for (int q = 0; q < NQ; ++q) out[q][v + value_offset][j] += product[q][v][j];
    }
  }
}
