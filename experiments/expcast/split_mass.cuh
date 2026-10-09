// vc28 candidate: split decoded-P component sums, keep FP32 throughout.
// Included inside sage_vc after work_reduction.cuh. Apache-2.0.
#pragma once

template<uint32_t NK>
__device__ __forceinline__ void vc_probability_split_mass_tile(
    const int32_t (&scores)[NK][8], const float (&maximum)[2], float gain,
    uint32_t (&packed)[NK/2][4], float (&mass)[2]) {
  static_assert(NK==8);
  float bias[2]={__fmaf_rn(-maximum[0],11.541560173034668f,119.6500015258789f),
                 __fmaf_rn(-maximum[1],11.541560173034668f,119.6500015258789f)};
  float sums[2][4]={};
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
          float2 values=__half22float2(decoded[k][r][pair]);
          // Independent component chains remove the inner pair-sum dependency.
          // Unlike vc17's four PAIR chains, no x+y is needed in this loop.
          sums[r][2*pair]=__fadd_rn(sums[r][2*pair],values.x);
          sums[r][2*pair+1]=__fadd_rn(sums[r][2*pair+1],values.y);
        }
  }
  #pragma unroll
  for (int r=0;r<2;++r)
    mass[r]=__fadd_rn(__fadd_rn(sums[r][0],sums[r][1]),
                    __fadd_rn(sums[r][2],sums[r][3]));
  // Positive E4M3 codes 0..120 are multiples of 2^-9 and <=256.
  // All partial sums of 32 such values fit in 2^22 integer units, so both
  // this tree and vc28's tree are exact in FP32. History arithmetic is untouched.
}
