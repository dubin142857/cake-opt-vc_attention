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

// ExpCast without V block means: form P and independent tile sums first.
// History scaling is delayed until PV is in flight. Never read P's register
// fragments between wgmma.mma_async and wait_group (PTX forbids that too).
template<bool EncodeOnly=false, uint32_t NK>
__device__ __forceinline__ void vc_probability_affine_tile(
    const float (&scores)[NK][8],const float (&maximum)[2],float gain,
    uint32_t (&packed)[NK/2][4],float (&mass)[2]) {
  float bias[2]={__fmaf_rn(-maximum[0],11.541560173034668f,119.6500015258789f),
                 __fmaf_rn(-maximum[1],11.541560173034668f,119.6500015258789f)};
  float sums[2][2]={};
  #pragma unroll
  for(int base=0;base<NK;base+=2) {
    uint32_t codes[2][2][2]; __half2 decoded[2][2][2];
    #pragma unroll
    for(int k=0;k<2;++k)
      #pragma unroll
      for(int r=0;r<2;++r)
        #pragma unroll
        for(int pair=0;pair<2;++pair) {
          int j=r*2+pair*4;
          codes[k][r][pair]=vc_expcast_affine_pair(scores[base+k][j],scores[base+k][j+1],gain,bias[r]);
        }
    #pragma unroll
    for(int k=0;k<2;++k)
      #pragma unroll
      for(int r=0;r<2;++r) {
        packed[base/2][k*2+r]=__byte_perm(codes[k][r][0],codes[k][r][1],0x5410);
        if constexpr(!EncodeOnly) {
          #pragma unroll
          for(int pair=0;pair<2;++pair)
            decoded[k][r][pair]=__nv_cvt_fp8x2_to_halfraw2(
                static_cast<__nv_fp8x2_storage_t>(codes[k][r][pair]),__NV_E4M3);
        }
      }
    if constexpr(!EncodeOnly) {
    #pragma unroll
    for(int k=0;k<2;++k)
      #pragma unroll
      for(int r=0;r<2;++r)
        #pragma unroll
        for(int pair=0;pair<2;++pair) {
          float2 values=__half22float2(decoded[k][r][pair]);
          sums[r][pair]=__fadd_rn(sums[r][pair],__fadd_rn(values.x,values.y));
        }
    }
  }
  if constexpr(!EncodeOnly) {
    mass[0]=__fadd_rn(sums[0][0],sums[0][1]);
    mass[1]=__fadd_rn(sums[1][0],sums[1][1]);
  }
}

template<bool Interleaved=false, bool FoldAffine=false, bool EncodeOnly=false, uint32_t NQ, uint32_t NK>
__device__ __forceinline__ void vc_probability_prepare(
    const float (&scores)[NQ][NK][8], float (&maximum)[NQ][2], float scale,
    uint32_t (&packed)[NQ][NK / 2][4], float (&alpha)[NQ][2], float (&mass)[NQ][2]) {
  static_assert(NK == 8, "Exact FP32 tile sum bound assumes K128");
  #pragma unroll
  for (int q = 0; q < NQ; ++q) {
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      float partial[4] = {-CUDART_INF_F,-CUDART_INF_F,-CUDART_INF_F,-CUDART_INF_F};
      #pragma unroll
      for (int k = 0; k < NK; ++k)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
          partial[j] = fmaxf(partial[j], scores[q][k][r*2+(j/2)*4+j%2]);
      float local_max = __fmul_rn(fmaxf(fmaxf(partial[0],partial[1]),fmaxf(partial[2],partial[3])),scale);
      local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 1));
      local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 2));
      float new_max = fmaxf(maximum[q][r],local_max);
      alpha[q][r] = expf(maximum[q][r]-new_max);
      maximum[q][r] = new_max;
    }
    if constexpr(FoldAffine) {
      static_assert(Interleaved,"Affine path preserves vc11's two-fragment reduction");
      float gain=__fmul_rn(scale,11.541560173034668f);
      // Bound cancellation and avoid coefficient overflow/underflow. Maxima
      // cover all current scores and history; on this domain the error near
      // nonzero codes is far below the 0.85 gap from code 120 to code 121.
      // All lanes take the same branch. Extreme logits use the vc11 path.
      bool safe=scale>=1.1754943508222875e-38f && isfinite(gain) &&
          fabsf(maximum[q][0])<=128.f && fabsf(maximum[q][1])<=128.f;
      if(__all_sync(0xffffffff,safe)) {
        vc_probability_affine_tile<EncodeOnly>(scores[q],maximum[q],gain,packed[q],mass[q]);
        continue;
      }
    }
    // Two independent chains per row instead of updating the long-lived
    // denominator eight times. All E4M3 codes 0..120 are multiples of 2^-9;
    // 32 values <=256 sum exactly in FP32 (at most 2^22 integer units).
    // Only the final addition to the historical denominator changes rounding.
    float sums[2][2] = {};
    if constexpr (Interleaved) {
      // Two K fragments at a time: expose independent encodes and packed
      // FP8->half2 conversions before consuming FP32 values. Keep the batch
      // small to bound temporary lifetimes. All operations retain vc10's
      // explicit RN arithmetic and the same per-row/per-chain K order.
      #pragma unroll
      for (int base = 0; base < NK; base += 2) {
        uint32_t codes[2][2][2];
        __half2 decoded[2][2][2];
        #pragma unroll
        for (int k = 0; k < 2; ++k)
          #pragma unroll
          for (int r = 0; r < 2; ++r)
            #pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
              int j = r*2+pair*4;
              float a = __fmul_rn(scores[q][base+k][j],scale);
              float b = __fmul_rn(scores[q][base+k][j+1],scale);
              codes[k][r][pair] = vc_expcast_pair(__fsub_rn(a,maximum[q][r]),__fsub_rn(b,maximum[q][r]));
            }
        #pragma unroll
        for (int k = 0; k < 2; ++k)
          #pragma unroll
          for (int r = 0; r < 2; ++r) {
            packed[q][base/2][k*2+r] = __byte_perm(codes[k][r][0],codes[k][r][1],0x5410);
            if constexpr(!EncodeOnly) {
              #pragma unroll
              for (int pair = 0; pair < 2; ++pair)
                decoded[k][r][pair] = __nv_cvt_fp8x2_to_halfraw2(
                    static_cast<__nv_fp8x2_storage_t>(codes[k][r][pair]),__NV_E4M3);
            }
          }
        if constexpr(!EncodeOnly) {
        #pragma unroll
        for (int k = 0; k < 2; ++k)
          #pragma unroll
          for (int r = 0; r < 2; ++r)
            #pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
              float2 values = __half22float2(decoded[k][r][pair]);
              sums[r][pair] = __fadd_rn(sums[r][pair],__fadd_rn(values.x,values.y));
            }
        }
      }
    } else {
    #pragma unroll
    for (int k = 0; k < NK; ++k) {
      #pragma unroll
      for (int r = 0; r < 2; ++r) {
        float a = __fmul_rn(scores[q][k][r*2],scale);
        float b = __fmul_rn(scores[q][k][r*2+1],scale);
        float c = __fmul_rn(scores[q][k][r*2+4],scale);
        float e = __fmul_rn(scores[q][k][r*2+5],scale);
        uint32_t low = vc_expcast_pair(__fsub_rn(a,maximum[q][r]),__fsub_rn(b,maximum[q][r]));
        uint32_t high = vc_expcast_pair(__fsub_rn(c,maximum[q][r]),__fsub_rn(e,maximum[q][r]));
        if constexpr(!EncodeOnly) {
          float2 first = vc_decode_pair(low), second = vc_decode_pair(high);
          sums[r][0] = __fadd_rn(sums[r][0],__fadd_rn(first.x,first.y));
          sums[r][1] = __fadd_rn(sums[r][1],__fadd_rn(second.x,second.y));
        }
        packed[q][k/2][(k%2)*2+r] = __byte_perm(low,high,0x5410);
      }
    }
    }
    if constexpr(!EncodeOnly) {
      #pragma unroll
      for (int r = 0; r < 2; ++r)
        mass[q][r] = __fadd_rn(sums[r][0],sums[r][1]);
    }
  }
}

// A private shared-memory copy lets CUDA cores decode P while WGMMA reads
// the original register operands. Reading those operands before wait_group
// would be undefined behavior. Each thread owns its strided scratch words;
// no cross-thread communication or extra barrier is needed. Volatile PTX
// retains the actual shared copy and its placement relative to WGMMA.
template<uint32_t NQ, uint32_t NP>
__device__ __forceinline__ void vc_probability_store(
    const uint32_t (&packed)[NQ][NP][4], uint32_t* scratch) {
  #pragma unroll
  for(int q=0;q<NQ;++q)
    #pragma unroll
    for(int k=0;k<NP;++k)
      #pragma unroll
      for(int j=0;j<4;++j) {
        uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(
            scratch+((q*NP+k)*4+j)*128+threadIdx.x));
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(address), "r"(packed[q][k][j]) : "memory");
      }
}

template<uint32_t NQ>
__device__ __forceinline__ void vc_probability_load_mass(
    const uint32_t* scratch, float (&mass)[NQ][2]) {
  // Same order and exact FP32 tile sums as vc12 (K128, codes 0..120).
  #pragma unroll
  for(int q=0;q<NQ;++q) {
    float sums[2][2]={};
    #pragma unroll
    for(int k=0;k<8;++k)
      #pragma unroll
      for(int r=0;r<2;++r) {
        uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(
            scratch+(q*16+k*2+r)*128+threadIdx.x));
        uint32_t word;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(word) : "r"(address) : "memory");
        float2 low=vc_decode_pair(word),high=vc_decode_pair(word>>16);
        sums[r][0]=__fadd_rn(sums[r][0],__fadd_rn(low.x,low.y));
        sums[r][1]=__fadd_rn(sums[r][1],__fadd_rn(high.x,high.y));
      }
    mass[q][0]=__fadd_rn(sums[0][0],sums[0][1]);
    mass[q][1]=__fadd_rn(sums[1][0],sums[1][1]);
  }
}

template<uint32_t NQ, uint32_t NV>
__device__ __forceinline__ void vc_probability_finish(
    float (&out)[NQ][NV][8], float (&denom)[NQ][2],
    const float (&alpha)[NQ][2], const float (&mass)[NQ][2]) {
  // Expose independent CUDA-core work after the PV commit. Ptxas can schedule
  // it among outstanding PV instructions; RO is distinct from PV accumulators.
  #pragma unroll
  for (int q = 0; q < NQ; ++q) {
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      asm volatile("mul.rn.f32 %0, %0, %1; add.rn.f32 %0, %0, %2;"
          : "+f"(denom[q][r]) : "f"(alpha[q][r]), "f"(mass[q][r]));
      #pragma unroll
      for (int v = 0; v < NV; ++v)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
          asm volatile("mul.rn.f32 %0, %0, %1;"
              : "+f"(out[q][v][r*2+(j/2)*4+j%2]) : "f"(alpha[q][r]));
    }
  }
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
