// Balanced-max candidate: encode four P bytes and retire their FP32 pair sums.
// Included inside sage_vc after work_reduction.cuh. Apache-2.0.
#pragma once

__device__ __forceinline__ void vc_int_affine_quad_sums(
    int a, int b, int c, int d, float gain, float bias,
    uint32_t& word, float& low_sum, float& high_sum) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  // Same FP32 FMA, RNE and upper-pair-then-append byte encoding as vc28.
  // Start decoding the upper pair before encoding the lower pair. Half is
  // only an exact decoded representation; every addition remains FP32 RN.
  // Keep intermediates private to this four-value group instead of exposing
  // separate arrays of upper bytes, decoded halves and promoted floats.
  asm volatile(
      "{ .reg .f32 x0,x1,x2,x3; .reg .s32 c0,c1,c2,c3;"
      " .reg .b32 upper,half_hi,half_lo; .reg .b16 pair,h0,h1;"
      " cvt.rz.f32.s32 x2,%5; cvt.rz.f32.s32 x3,%6;"
      " fma.rn.f32 x2,x2,%7,%8; fma.rn.f32 x3,x3,%7,%8;"
      " cvt.rni.s32.f32 c2,x2; cvt.rni.s32.f32 c3,x3;"
      " cvt.pack.sat.u8.s32.b32 upper,c3,c2,0;"
      " cvt.u16.u32 pair,upper; cvt.rn.f16x2.e4m3x2 half_hi,pair;"
      " cvt.rz.f32.s32 x0,%3; cvt.rz.f32.s32 x1,%4;"
      " fma.rn.f32 x0,x0,%7,%8; fma.rn.f32 x1,x1,%7,%8;"
      " cvt.rni.s32.f32 c0,x0; cvt.rni.s32.f32 c1,x1;"
      " cvt.pack.sat.u8.s32.b32 %0,c1,c0,upper;"
      " mov.b32 {h0,h1},half_hi; cvt.f32.f16 x2,h0; cvt.f32.f16 x3,h1;"
      " add.rn.f32 %2,x2,x3;"
      " cvt.u16.u32 pair,%0; cvt.rn.f16x2.e4m3x2 half_lo,pair;"
      " mov.b32 {h0,h1},half_lo; cvt.f32.f16 x0,h0; cvt.f32.f16 x1,h1;"
      " add.rn.f32 %1,x0,x1; }"
      : "=&r"(word), "=&f"(low_sum), "=&f"(high_sum)
      : "r"(a), "r"(b), "r"(c), "r"(d), "f"(gain), "f"(bias));
#else
  // SM86 helper checks use CUDA's software FP8 conversion. Full attention
  // still requires SM90a and exercises the native instruction block above.
  uint32_t upper;
  vc_int_affine_quad(a,b,c,d,gain,bias,word,upper);
  float2 lo=vc_decode_pair(word), hi=vc_decode_pair(upper);
  low_sum=__fadd_rn(lo.x,lo.y); high_sum=__fadd_rn(hi.x,hi.y);
#endif
}

template<uint32_t NK>
__device__ __forceinline__ void vc_probability_fused_quad_tile(
    const int32_t (&scores)[NK][8], const float (&maximum)[2], float gain,
    uint32_t (&packed)[NK/2][4], float (&mass)[2]) {
  static_assert(NK==8);
  float bias[2]={__fmaf_rn(-maximum[0],11.541560173034668f,119.6500015258789f),
                 __fmaf_rn(-maximum[1],11.541560173034668f,119.6500015258789f)};
  float sums[2][2];
  #pragma unroll
  for (int base=0;base<NK;base+=2) {
    float pair_sums[2][2][2];
    #pragma unroll
    for (int k=0;k<2;++k)
      #pragma unroll
      for (int r=0;r<2;++r) {
        int j=r*2;
        vc_int_affine_quad_sums(scores[base+k][j],scores[base+k][j+1],
            scores[base+k][j+4],scores[base+k][j+5],gain,bias[r],
            packed[base/2][k*2+r],pair_sums[k][r][0],pair_sums[k][r][1]);
      }
    #pragma unroll
    for (int k=0;k<2;++k)
      #pragma unroll
      for (int r=0;r<2;++r)
        #pragma unroll
        for (int pair=0;pair<2;++pair) {
          // Nonnegative finite P: first pair sum is already the result of
          // +0 + pair_sum, including positive zero. Remove four neutral adds.
          if (base==0 && k==0) sums[r][pair]=pair_sums[k][r][pair];
          else sums[r][pair]=__fadd_rn(sums[r][pair],pair_sums[k][r][pair]);
        }
  }
  mass[0]=__fadd_rn(sums[0][0],sums[0][1]);
  mass[1]=__fadd_rn(sums[1][0],sums[1][1]);
}
