// Derived from vc28 streamed_probability.cuh. Apache-2.0.
// Carry decoded FP16 representations, never pending WGMMA P operands.
#pragma once

struct VCLateMass {
  __half2 last[2][2] = {};
};

template<uint32_t NK>
__device__ __forceinline__ void vc_probability_mass_tail_tile(
    const int32_t (&scores)[NK][8], const float (&maximum)[2], float gain,
    uint32_t (&packed)[NK/2][4], float (&mass)[2], VCLateMass& late) {
  static_assert(NK==8);
  float bias[2]={__fmaf_rn(-maximum[0],11.541560173034668f,119.6500015258789f),
                 __fmaf_rn(-maximum[1],11.541560173034668f,119.6500015258789f)};
  float sums[2][2]={};
  #pragma unroll
  for (int base=0;base<NK;base+=2) {
    uint32_t codes[2][2][2]; __half2 decoded[2][2][2];
    #pragma unroll
    for (int k=0;k<2;++k)
      #pragma unroll
      for (int r=0;r<2;++r) {
        int j=r*2;
        vc_int_affine_quad(scores[base+k][j],scores[base+k][j+1],
            scores[base+k][j+4],scores[base+k][j+5],gain,bias[r],
            codes[k][r][0],codes[k][r][1]);
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
          if (base+k==NK-1) late.last[r][pair]=decoded[k][r][pair];
          else {
            float2 values=__half22float2(decoded[k][r][pair]);
            sums[r][pair]=__fadd_rn(sums[r][pair],__fadd_rn(values.x,values.y));
          }
        }
  }
  mass[0]=__fadd_rn(sums[0][0],sums[0][1]);
  mass[1]=__fadd_rn(sums[1][0],sums[1][1]);
}

__device__ __forceinline__ void vc_probability_finish_mass_tail(
    const VCLateMass& late, float (&mass)[2]) {
  // Each local row has 32 nonnegative E4M3 values <=256, all multiples
  // of 2^-9. Every partial sum fits in 2^22 integer units: FP32 additions
  // are exact, including this regrouping. No FP16 addition is performed.
  // Fallback leaves last[] at +0 and has already computed the full mass.
  #pragma unroll
  for (int r=0;r<2;++r)
    #pragma unroll
    for (int pair=0;pair<2;++pair) {
      float2 values=__half22float2(late.last[r][pair]);
      mass[r]=__fadd_rn(mass[r],__fadd_rn(values.x,values.y));
    }
}
