// vc23: reuse unchanged online-max state in the vc22 ExpCast loop.
// Apache-2.0. Included after work_reduction.cuh.
#pragma once

template<uint32_t NK>
__device__ __forceinline__ void vc_probability_cached_affine_tile(
    const float (&scores)[NK][8],const float (&bias)[2],float gain,
    uint32_t (&packed)[NK/2][4],float (&mass)[2]) {
  // Same vc22 two-fragment encoding and FP32 sum order. Only the row bias
  // comes from persistent state instead of being recomputed every K tile.
  float sums[2][2]={};
  #pragma unroll
  for (int base=0;base<NK;base+=2) {
    uint32_t codes[2][2][2]; __half2 decoded[2][2][2];
    #pragma unroll
    for (int k=0;k<2;++k)
      #pragma unroll
      for (int r=0;r<2;++r) {
        int j=r*2;
        vc_expcast_quad_rne(__fmaf_rn(scores[base+k][j],gain,bias[r]),
            __fmaf_rn(scores[base+k][j+1],gain,bias[r]),
            __fmaf_rn(scores[base+k][j+4],gain,bias[r]),
            __fmaf_rn(scores[base+k][j+5],gain,bias[r]),codes[k][r][0],codes[k][r][1]);
      }
    #pragma unroll
    for (int k=0;k<2;++k)
      #pragma unroll
      for (int r=0;r<2;++r) {
        packed[base/2][k*2+r]=codes[k][r][0];
        #pragma unroll
        for (int pair=0;pair<2;++pair)
          decoded[k][r][pair]=__nv_cvt_fp8x2_to_halfraw2(
              static_cast<__nv_fp8x2_storage_t>(codes[k][r][pair]),__NV_E4M3);
      }
    #pragma unroll
    for (int k=0;k<2;++k)
      #pragma unroll
      for (int r=0;r<2;++r)
        #pragma unroll
        for (int pair=0;pair<2;++pair) {
          float2 values=__half22float2(decoded[k][r][pair]);
          sums[r][pair]=__fadd_rn(sums[r][pair],__fadd_rn(values.x,values.y));
        }
  }
  mass[0]=__fadd_rn(sums[0][0],sums[0][1]);
  mass[1]=__fadd_rn(sums[1][0],sums[1][1]);
}

template<bool CacheBias,uint32_t NQ,uint32_t NK>
__device__ __forceinline__ void vc_probability_prepare_maxstate(
    const int32_t (&scores)[NQ][NK][8],float (&maximum)[NQ][2],float scale,
    uint32_t (&packed)[NQ][NK/2][4],float (&alpha)[NQ][2],float (&mass)[NQ][2],
    float (&bias)[NQ][2],bool& rescale) {
  static_assert(NQ==1 && NK==8);
  float next[2];
  #pragma unroll
  for (int r=0;r<2;++r) {
    float local=__fmul_rn(__int2float_rz(vc_integer_row_max(scores[0],r)),scale);
    local=fmaxf(local,__shfl_xor_sync(0xffffffff,local,1));
    local=fmaxf(local,__shfl_xor_sync(0xffffffff,local,2));
    next[r]=fmaxf(maximum[0][r],local);
  }
  // Reuse this warp-uniform decision after PV. When no row changes, exp(0)
  // is exactly one; no SFU call or later alpha==1 vote is needed.
  rescale=__any_sync(0xffffffff,next[0]!=maximum[0][0] || next[1]!=maximum[0][1]);
  if (rescale) {
    #pragma unroll
    for (int r=0;r<2;++r) {
      alpha[0][r]=expf(maximum[0][r]-next[r]);
      if constexpr (CacheBias)
        bias[0][r]=__fmaf_rn(-next[r],11.541560173034668f,119.6500015258789f);
    }
  } else { alpha[0][0]=1.f; alpha[0][1]=1.f; }
  maximum[0][0]=next[0]; maximum[0][1]=next[1];
  float converted[NK][8];
  #pragma unroll
  for (int k=0;k<NK;++k)
    #pragma unroll
    for (int j=0;j<8;++j) converted[k][j]=__int2float_rz(scores[0][k][j]);
  float gain=__fmul_rn(scale,11.541560173034668f);
  bool safe=scale>=1.1754943508222875e-38f && isfinite(gain) &&
      fabsf(maximum[0][0])<=128.f && fabsf(maximum[0][1])<=128.f;
  if (__all_sync(0xffffffff,safe)) {
    if constexpr (CacheBias)
      vc_probability_cached_affine_tile(converted,bias[0],gain,packed[0],mass[0]);
    else vc_probability_affine_tile<false,true,true>(converted,maximum[0],gain,packed[0],mass[0]);
  } else vc_probability_variant_tile<2,2,false,false>(converted,maximum[0],scale,packed[0],mass[0]);
}

template<uint32_t NQ,uint32_t NV>
__device__ __forceinline__ void vc_probability_finish_maxstate(
    float (&out)[NQ][NV][8],float (&denom)[NQ][2],
    const float (&alpha)[NQ][2],const float (&mass)[NQ][2],bool rescale) {
  static_assert(NQ==1);
  if (rescale) vc_probability_finish(out,denom,alpha,mass);
  else {
    #pragma unroll
    for (int r=0;r<2;++r)
      asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(denom[0][r]) : "f"(mass[0][r]));
  }
}
