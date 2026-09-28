// VC additions to the SageAttention Hopper core. Apache-2.0; see NOTICE.
#pragma once
#include "expcast.cuh"

__device__ __forceinline__ float vc_decode(uint8_t code) {
  __nv_fp8_e4m3 value;
  value.__x = code;
  return static_cast<float>(value);
}

template<bool ExpCast, bool BlockValues, uint32_t NQ, uint32_t NK, uint32_t NV>
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
      #pragma unroll
      for (int k = 0; k < NK; ++k) {
        uint32_t bytes[2] = {0, 0};
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          int r = (j % 4) / 2;
          uint8_t code = vc_expcast_code(__fsub_rn(scores[q][k][j], maximum[q][r]));
          scores[q][k][j] = vc_decode(code); // Both denominator and mean restoration use decoded P.
          bytes[r] |= uint32_t(code) << (8 * ((j / 4) * 2 + j % 2));
        }
        packed[q][k / 2][(k % 2) * 2] = bytes[0];
        packed[q][k / 2][(k % 2) * 2 + 1] = bytes[1];
      }
    }
  } else {
    // Preserve upstream's 2^8.807 P scaling and unrounded denominator.
    update_mdo<NQ, NK, NV, false, true, false>(scores, out, maximum, denom, scale);
    RS_32_to_8<NQ, NK>(scores, packed);
  }
  #pragma unroll
  for (int q = 0; q < NQ; ++q) {
    mass[q][0] = mass[q][1] = 0.0f;
    #pragma unroll
    for (int k = 0; k < NK; ++k) {
      float a = scores[q][k][0] + scores[q][k][1] + scores[q][k][4] + scores[q][k][5];
      float b = scores[q][k][2] + scores[q][k][3] + scores[q][k][6] + scores[q][k][7];
      denom[q][0] += a;
      denom[q][1] += b;
      if constexpr (BlockValues) { mass[q][0] += a; mass[q][1] += b; }
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

template<bool BlockValues, uint32_t NQ, uint32_t NV>
__device__ __forceinline__ void vc_accumulate(
    float (&out)[NQ][NV][8], const float (&product)[NQ][NV][8],
    const float (&mass)[NQ][2], const float* scales, const half* means,
    int head, int tile, int tiles, int dim, int lane) {
  #pragma unroll
  for (int v = 0; v < NV; ++v) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      float scale = 1, mean = 0;
      if constexpr (BlockValues) {
        int channel = v * 16 + (lane % 4) * 2 + (j / 4) * 8 + j % 2;
        int64_t meta = (int64_t(head) * tiles + tile) * dim + channel;
        scale = scales[meta];
        mean = __half2float(means[meta]);
      }
      #pragma unroll
      for (int q = 0; q < NQ; ++q) {
        if constexpr (BlockValues) {
          float centered = __fadd_rn(product[q][v][j], __fmul_rn(mass[q][(j % 4) / 2], mean));
          out[q][v][j] = __fadd_rn(out[q][v][j], __fmul_rn(centered, scale));
        } else out[q][v][j] += product[q][v][j];
      }
    }
  }
}
