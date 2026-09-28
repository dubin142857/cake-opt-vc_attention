// VC ExpCast byte encoding shared by attention and the diagnostic probe.
// Apache-2.0; see NOTICE.
#pragma once
#include <cuda_fp8.h>

__device__ __forceinline__ uint8_t vc_expcast_code(float shifted) {
  float u = __fmaf_rn(shifted, 11.541560173034668f, 119.6500015258789f);
  return static_cast<uint8_t>(__float2int_rn(fminf(120.0f, fmaxf(0.0f, u))));
}
