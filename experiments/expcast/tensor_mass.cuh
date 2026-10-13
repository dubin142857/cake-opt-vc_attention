// Tensor-Core probability reduction; same vc28 encoding. Apache-2.0.
// Included in namespace sage_vc after streamed_probability.cuh.
#pragma once

__device__ __forceinline__ void vc_tensor_mass_step(
    const uint32_t (&a)[4], float& row0, float& row1) {
  const unsigned lane = threadIdx.x & 31;
  // Packed FP8 A owns four K values per lane, in each half of K32.
  // Columns 2*t and 2*t+1 select ONLY source lane t within each row group.
  // This preserves each thread's original partial denominator, not a
  // replicated warp sum. No cross-thread d recurrence is changed.
  const bool selected = (lane & 3) == (lane >> 3);
  // Decode each P pair exactly to FP16, then perform two FP16 MMAs with
  // FP32 accumulation. This avoids the compiler's FP8 MMA expansion into
  // independent half MMAs plus extra FP32 merges and B conversions.
  // All nonzero E4M3 codes 0..120 decode to NORMAL FP16 values (>=2^-9),
  // so MMA's unspecified handling of subnormal inputs is irrelevant here.
  const uint32_t b = selected ? 0x3c003c00u : 0u;
  float c0=0.f,c1=0.f,c2=0.f,c3=0.f;
  #pragma unroll
  for (int pair=0; pair<2; ++pair) {
    uint32_t h[4];
    #pragma unroll
    for (int j=0; j<4; ++j) {
      __half2_raw decoded=__nv_cvt_fp8x2_to_halfraw2(
          static_cast<__nv_fp8x2_storage_t>(a[j] >> (pair*16)),__NV_E4M3);
      h[j]=uint32_t(decoded.x) | (uint32_t(decoded.y)<<16);
    }
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%8}, {%0,%1,%2,%3};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(h[0]), "r"(h[1]), "r"(h[2]), "r"(h[3]), "r"(b));
  }
  row0=c0; row1=c2;
}

template<uint32_t NK>
__device__ __forceinline__ void vc_probability_tensor_mass_tile(
    const int32_t (&scores)[NK][8], const float (&maximum)[2], float gain,
    uint32_t (&packed)[NK/2][4], float (&mass)[2]) {
  static_assert(NK==8);
  const float bias[2]={__fmaf_rn(-maximum[0],11.541560173034668f,119.6500015258789f),
                       __fmaf_rn(-maximum[1],11.541560173034668f,119.6500015258789f)};
  #pragma unroll
  for (int base=0; base<NK; base+=2) {
    #pragma unroll
    for (int k=0; k<2; ++k)
      #pragma unroll
      for (int r=0; r<2; ++r) {
        int j=r*2; uint32_t upper;
        vc_int_affine_quad(scores[base+k][j],scores[base+k][j+1],
            scores[base+k][j+4],scores[base+k][j+5],gain,bias[r],
            packed[base/2][k*2+r],upper);
      }
    float partial[2];
    vc_tensor_mass_step(packed[base/2],partial[0],partial[1]);
    #pragma unroll
    for (int r=0; r<2; ++r) {
      if (base==0) mass[r]=partial[r];
      else mass[r]=__fadd_rn(mass[r],partial[r]);
    }
  }
  // Each step sums 8 nonnegative E4M3 values <=256, in units of 2^-9.
  // At most 2^20 units/step and 2^22 units/local row: FP32 regrouping is
  // exact. Each K32 group starts at zero; FP32 adds join four partials.
  // FP8-to-FP16 decoding remains. Per-value FP32 promotion/adds disappear.
  // Hopper byte-exhaustive and adversarial exact-result tests MUST pass
  // before any performance claim, including E4M3 subnormal input bytes.
}
