// VC ExpCast byte encoding shared by attention and the diagnostic probe.
// Apache-2.0; see NOTICE.
#pragma once
#include <cuda_fp8.h>

__device__ __forceinline__ uint8_t vc_expcast_code(float shifted) {
  float u = __fmaf_rn(shifted, 11.541560173034668f, 119.6500015258789f);
  return static_cast<uint8_t>(__float2int_rn(fminf(120.0f, fmaxf(0.0f, u))));
}

// Attention has already subtracted max(previous_max, tile_max): shifted <= 0.
// Therefore u <= 119.65 and the upper clamp is redundant. Keep the lower
// clamp (including masked -inf) and the exact same FP32 FMA/RNE contract.
__device__ __forceinline__ uint32_t vc_expcast_nonpositive(float shifted) {
  float u = __fmaf_rn(shifted, 11.541560173034668f, 119.6500015258789f);
  return __float2uint_rn(fmaxf(0.0f, u));
}

// Same FMA and ties-to-even byte as above for shifted <= 0, including -inf.
// PTX's saturating u8 conversion supplies the lower clamp, and zero-extends
// into the 32-bit destination. u <= 119.65 so its upper bound is never active.
__device__ __forceinline__ uint32_t vc_expcast_u8(float shifted) {
  float u = __fmaf_rn(shifted, 11.541560173034668f, 119.6500015258789f);
  uint32_t code;
  asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(code) : "f"(u));
  return code;
}

__device__ __forceinline__ uint32_t vc_expcast_pair(float a, float b) {
  uint32_t lo = vc_expcast_u8(a), hi = vc_expcast_u8(b);
  // Select the low byte of each zero-extended code, then two zero bytes.
  return __byte_perm(lo, hi, 0x1140);
}

// Algebraically fold RN(score*scale), subtraction and ExpCast's linear map.
// FP32 coefficients change rounding at byte boundaries; this is NOT bitwise
// the legacy encoder. Only used by the bounded affine probability path.
__device__ __forceinline__ uint32_t vc_expcast_affine_pair(float a,float b,float gain,float bias) {
  float x=__fmaf_rn(a,gain,bias),y=__fmaf_rn(b,gain,bias);
  uint32_t lo,hi;
  asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(lo) : "f"(x));
  asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(hi) : "f"(y));
  return __byte_perm(lo,hi,0x1140);
}
