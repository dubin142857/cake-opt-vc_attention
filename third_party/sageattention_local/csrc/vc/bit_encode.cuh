// vc24: exact RNE byte encoding on the bounded affine path. Apache-2.0.
// The decoded-P FP32 reduction stays identical to vc22.
#pragma once

__device__ __forceinline__ uint32_t vc_affine_bit_code(float score,float gain128,float bias128) {
  // Coefficients are the ORIGINAL rounded coefficients divided by 128.
  // Power-of-two scaling preserves FMA rounding away from underflow. Values
  // near underflow round to byte zero either way. Caller guards coefficients.
  float unit;
  asm("fma.rn.sat.f32 %0,%1,%2,%3;" : "=f"(unit) : "f"(score),"f"(gain128),"f"(bias128));
  // 98304 = 1.5 * 2^16; its FP32 ULP is 1/128. Adding unit rounds 128*unit
  // to nearest-even in the low bits. The magic constant has a zero low byte.
  // Do NOT contract the addition with the preceding FMA (double rounding is
  // intentional: it matches affine-FMA followed by integer RNE).
  return __float_as_uint(__fadd_rn(unit,98304.f));
}

template<uint32_t NK>
__device__ __forceinline__ void vc_probability_bit_tile(
    const float (&scores)[NK][8],const float (&maximum)[2],float gain,
    uint32_t (&packed)[NK/2][4],float (&mass)[2]) {
  float gain128=__fmul_rn(gain,0.0078125f);
  float bias[2]={__fmul_rn(__fmaf_rn(-maximum[0],11.541560173034668f,119.6500015258789f),0.0078125f),
                 __fmul_rn(__fmaf_rn(-maximum[1],11.541560173034668f,119.6500015258789f),0.0078125f)};
  float sums[2][2]={};
  #pragma unroll
  for (int base=0;base<NK;base+=2) {
    uint32_t codes[2][2][2]; __half2 decoded[2][2][2];
    #pragma unroll
    for (int k=0;k<2;++k)
      #pragma unroll
      for (int r=0;r<2;++r) {
        int j=r*2;
        uint32_t a=vc_affine_bit_code(scores[base+k][j],gain128,bias[r]);
        uint32_t b=vc_affine_bit_code(scores[base+k][j+1],gain128,bias[r]);
        uint32_t c=vc_affine_bit_code(scores[base+k][j+4],gain128,bias[r]);
        uint32_t d=vc_affine_bit_code(scores[base+k][j+5],gain128,bias[r]);
        uint32_t low=__byte_perm(a,b,0x4440);
        uint32_t high=__byte_perm(c,d,0x4440);
        codes[k][r][0]=__byte_perm(low,high,0x5410);
        codes[k][r][1]=high;
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
