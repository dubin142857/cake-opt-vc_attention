// Exact work-reduction candidates for the vc16 ExpCast loop. Apache-2.0.
// Included after vc_math.cuh and probability_variants.cuh.
#pragma once
#include "bit_encode.cuh"

template<uint32_t NK>
__device__ __forceinline__ int vc_integer_row_max(const int32_t (&scores)[NK][8], int r) {
  static_assert(NK == 8);
  // Four independent eight-element chains; keep the last element of each
  // chain for the final tree. 15 three-way max + one two-way max per row
  // in source (CUDA 12.4 lowers the last pair to a three-way instruction too).
  int partial[4];
  #pragma unroll
  for (int j=0;j<4;++j) {
    int c=r*2+(j/2)*4+j%2;
    partial[j]=__vimax3_s32(scores[0][c],scores[1][c],scores[2][c]);
    partial[j]=__vimax3_s32(partial[j],scores[3][c],scores[4][c]);
    partial[j]=__vimax3_s32(partial[j],scores[5][c],scores[6][c]);
  }
  int a=__vimax3_s32(partial[0],partial[1],partial[2]);
  int b=__vimax3_s32(partial[3],scores[7][r*2],scores[7][r*2+1]);
  int c=__vimax3_s32(a,scores[7][r*2+4],scores[7][r*2+5]);
  return max(b,c);
}

template<bool BitEncode=false, uint32_t NQ, uint32_t NK>
__device__ __forceinline__ void vc_probability_prepare_integer(
    const int32_t (&scores)[NQ][NK][8], float (&maximum)[NQ][2], float scale,
    uint32_t (&packed)[NQ][NK/2][4], float (&alpha)[NQ][2], float (&mass)[NQ][2]) {
  // Full tiles only. INT8 D<=128 bounds every dot product by 2^21, so
  // conversion to FP32 is exact. Scale BEFORE shuffling: K scales differ
  // between lanes. The masked tail retains vc16's floating-point path.
  #pragma unroll
  for (int q=0;q<NQ;++q) {
    #pragma unroll
    for (int r=0;r<2;++r) {
      float local=__fmul_rn(__int2float_rz(vc_integer_row_max(scores[q],r)),scale);
      local=fmaxf(local,__shfl_xor_sync(0xffffffff,local,1));
      local=fmaxf(local,__shfl_xor_sync(0xffffffff,local,2));
      float next=fmaxf(maximum[q][r],local);
      alpha[q][r]=expf(maximum[q][r]-next);
      maximum[q][r]=next;
    }
    float converted[NK][8];
    #pragma unroll
    for (int k=0;k<NK;++k)
      #pragma unroll
      for (int j=0;j<8;++j) converted[k][j]=__int2float_rz(scores[q][k][j]);
    float gain=__fmul_rn(scale,11.541560173034668f);
    bool safe=scale>=1.1754943508222875e-38f && isfinite(gain) &&
        fabsf(maximum[q][0])<=128.f && fabsf(maximum[q][1])<=128.f;
    if constexpr (BitEncode) {
      // Ensure division by 128 leaves gain normal even with --use_fast_math.
      // The old affine path still handles smaller valid scales unchanged.
      if (__all_sync(0xffffffff,safe && scale>=0x1p-119f)) {
        vc_probability_bit_tile(converted,maximum[q],gain,packed[q],mass[q]);
        continue;
      }
    }
    if (__all_sync(0xffffffff,safe))
      vc_probability_affine_tile<false,true,true>(converted,maximum[q],gain,packed[q],mass[q]);
    else
      vc_probability_variant_tile<2,2,false,false>(converted,maximum[q],scale,packed[q],mass[q]);
  }
}

template<bool SkipIdentity, uint32_t NQ, uint32_t NV>
__device__ __forceinline__ void vc_probability_finish_work(
    float (&out)[NQ][NV][8], float (&denom)[NQ][2],
    const float (&alpha)[NQ][2], const float (&mass)[NQ][2]) {
  if constexpr (!SkipIdentity) {
    vc_probability_finish(out,denom,alpha,mass);
  } else {
    static_assert(NQ==1);
    // Uniform within each warp. This reads only history/alpha, NEVER the
    // P register operands or the PV accumulators while WGMMA is pending.
    if (__all_sync(0xffffffff,alpha[0][0]==1.f && alpha[0][1]==1.f)) {
      #pragma unroll
      for (int r=0;r<2;++r)
        asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(denom[0][r]) : "f"(mass[0][r]));
    } else vc_probability_finish(out,denom,alpha,mass);
  }
}
