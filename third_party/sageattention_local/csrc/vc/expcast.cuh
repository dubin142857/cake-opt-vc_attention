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

// Keep the FP32 inputs and RNE result of the scalar u8 conversions, but expose
// both conversions and their pack together. Ptxas can fuse this sequence into
// ONE Hopper F2IP.U8.F32.NTZ instead of two F2IP instructions plus a PRMT.
// Round to signed integers first: negative logits (including masked -inf)
// must saturate to zero, not reinterpret as unsigned. Integer overflow
// saturates at INT_MIN/MAX, then packing clamps to [0,255]. No FP16 rounding.
__device__ __forceinline__ uint32_t vc_expcast_pack_rne(float a,float b) {
  uint32_t packed;
  asm("{.reg .s32 lo,hi;"
      " cvt.rni.s32.f32 lo,%1; cvt.rni.s32.f32 hi,%2;"
      " cvt.pack.sat.u8.s32.b32 %0,hi,lo,0;}"
      : "=r"(packed) : "f"(a),"f"(b));
  return packed;
}

// Pack the new pair into the low half and copy upper_pair's LOW two bytes
// into the high half, using the third input of Hopper's F2IP instruction.
// Retain upper_pair separately for FP8 decoding; no shift or PRMT is needed.
__device__ __forceinline__ uint32_t vc_expcast_pack_rne_append(float a,float b,uint32_t upper_pair) {
  uint32_t packed;
  asm("{.reg .s32 lo,hi;"
      " cvt.rni.s32.f32 lo,%1; cvt.rni.s32.f32 hi,%2;"
      " cvt.pack.sat.u8.s32.b32 %0,hi,lo,%3;}"
      : "=r"(packed) : "f"(a),"f"(b),"r"(upper_pair));
  return packed;
}

__device__ __forceinline__ void vc_expcast_quad_rne(float a,float b,float c,float d,
    uint32_t& word,uint32_t& upper_pair) {
  upper_pair=vc_expcast_pack_rne(c,d);
  word=vc_expcast_pack_rne_append(a,b,upper_pair);
}

template<bool PairedEncode>
__device__ __forceinline__ uint32_t vc_expcast_encode_pair(float a,float b) {
  if constexpr(PairedEncode)
    return vc_expcast_pack_rne(__fmaf_rn(a,11.541560173034668f,119.6500015258789f),
                               __fmaf_rn(b,11.541560173034668f,119.6500015258789f));
  else return vc_expcast_pair(a,b);
}

template<bool PairedEncode>
__device__ __forceinline__ uint32_t vc_expcast_encode_affine_pair(float a,float b,float gain,float bias) {
  if constexpr(PairedEncode) return vc_expcast_pack_rne(__fmaf_rn(a,gain,bias),__fmaf_rn(b,gain,bias));
  else return vc_expcast_affine_pair(a,b,gain,bias);
}
