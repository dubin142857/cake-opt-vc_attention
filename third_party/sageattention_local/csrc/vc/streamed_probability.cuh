// vc28: consume INT32 scores in short conversion/encoding groups. Apache-2.0.
// Included after vc_math.cuh. Only the guarded INT8 ExpCast path uses this.
#pragma once

__device__ __forceinline__ void vc_int_affine_quad(
    int a, int b, int c, int d, float gain, float bias,
    uint32_t& word, uint32_t& upper_pair) {
  // Keep I2F temporaries inside the encode group rather than materializing a
  // whole FP32 score tile. INT8 D<=128 scores are exact in FP32. The FMA,
  // RNE integer conversion and saturated packing match vc22 byte-for-byte.
  asm volatile(
      "{ .reg .f32 x0,x1,x2,x3; .reg .s32 c0,c1,c2,c3;"
      " cvt.rz.f32.s32 x2,%4; cvt.rz.f32.s32 x3,%5;"
      " fma.rn.f32 x2,x2,%6,%7; fma.rn.f32 x3,x3,%6,%7;"
      " cvt.rni.s32.f32 c2,x2; cvt.rni.s32.f32 c3,x3;"
      " cvt.pack.sat.u8.s32.b32 %1,c3,c2,0;"
      " cvt.rz.f32.s32 x0,%2; cvt.rz.f32.s32 x1,%3;"
      " fma.rn.f32 x0,x0,%6,%7; fma.rn.f32 x1,x1,%6,%7;"
      " cvt.rni.s32.f32 c0,x0; cvt.rni.s32.f32 c1,x1;"
      " cvt.pack.sat.u8.s32.b32 %0,c1,c0,%1; }"
      : "=&r"(word), "=&r"(upper_pair)
      : "r"(a), "r"(b), "r"(c), "r"(d), "f"(gain), "f"(bias));
}

// vc29: interleave the two rows' independent conversion chains in one group.
// Each row retains the vc28 FMA/RNE and upper-pair-then-append packing order.
// Early-clobber outputs may be written before all inputs have been consumed.
__device__ __forceinline__ void vc_int_affine_rows(
    const int32_t (&s)[8], float gain, const float (&bias)[2],
    uint32_t (&codes)[2][2]) {
  asm volatile(
      "{ .reg .f32 x0,x1,x2,x3; .reg .s32 c0,c1,c2,c3;"
      " cvt.rz.f32.s32 x0,%8; cvt.rz.f32.s32 x1,%9;"
      " cvt.rz.f32.s32 x2,%10; cvt.rz.f32.s32 x3,%11;"
      " fma.rn.f32 x0,x0,%12,%13; fma.rn.f32 x1,x1,%12,%13;"
      " fma.rn.f32 x2,x2,%12,%14; fma.rn.f32 x3,x3,%12,%14;"
      " cvt.rni.s32.f32 c0,x0; cvt.rni.s32.f32 c1,x1;"
      " cvt.rni.s32.f32 c2,x2; cvt.rni.s32.f32 c3,x3;"
      " cvt.pack.sat.u8.s32.b32 %1,c1,c0,0;"
      " cvt.pack.sat.u8.s32.b32 %3,c3,c2,0;"
      " cvt.rz.f32.s32 x0,%4; cvt.rz.f32.s32 x1,%5;"
      " cvt.rz.f32.s32 x2,%6; cvt.rz.f32.s32 x3,%7;"
      " fma.rn.f32 x0,x0,%12,%13; fma.rn.f32 x1,x1,%12,%13;"
      " fma.rn.f32 x2,x2,%12,%14; fma.rn.f32 x3,x3,%12,%14;"
      " cvt.rni.s32.f32 c0,x0; cvt.rni.s32.f32 c1,x1;"
      " cvt.rni.s32.f32 c2,x2; cvt.rni.s32.f32 c3,x3;"
      " cvt.pack.sat.u8.s32.b32 %0,c1,c0,%1;"
      " cvt.pack.sat.u8.s32.b32 %2,c3,c2,%3; }"
      : "=&r"(codes[0][0]), "=&r"(codes[0][1]),
        "=&r"(codes[1][0]), "=&r"(codes[1][1])
      : "r"(s[0]), "r"(s[1]), "r"(s[2]), "r"(s[3]),
        "r"(s[4]), "r"(s[5]), "r"(s[6]), "r"(s[7]),
        "f"(gain), "f"(bias[0]), "f"(bias[1]));
}

template<bool InterleaveRows=false, uint32_t NK>
__device__ __forceinline__ void vc_probability_int_affine_tile(
    const int32_t (&scores)[NK][8], const float (&maximum)[2], float gain,
    uint32_t (&packed)[NK/2][4], float (&mass)[2]) {
  static_assert(NK==8);
  float bias[2]={__fmaf_rn(-maximum[0],11.541560173034668f,119.6500015258789f),
                 __fmaf_rn(-maximum[1],11.541560173034668f,119.6500015258789f)};
  float sums[2][2]={};
  #pragma unroll
  for (int base=0;base<NK;base+=2) {
    uint32_t codes[2][2][2]; __half2 decoded[2][2][2];
    #pragma unroll
    for (int k=0;k<2;++k) {
      if constexpr (InterleaveRows)
        vc_int_affine_rows(scores[base+k],gain,bias,codes[k]);
      else {
        #pragma unroll
        for (int r=0;r<2;++r) {
          int j=r*2;
          vc_int_affine_quad(scores[base+k][j],scores[base+k][j+1],
              scores[base+k][j+4],scores[base+k][j+5],gain,bias[r],
              codes[k][r][0],codes[k][r][1]);
        }
      }
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
