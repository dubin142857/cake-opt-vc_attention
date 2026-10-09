// VC-Attention Eq. (7), explicit FP32 evaluation contract. Apache-2.0.
// This is our implementation of the published formula, not author kernel code.
#pragma once
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float vc_paper_u(float score, float maximum, float scale) {
  float delta=__fsub_rn(__fmul_rn(score,scale),maximum);
  return fminf(120.f,fmaxf(0.f,__fmaf_rn(delta,11.541560173034668f,119.6500015258789f)));
}
__device__ __forceinline__ uint8_t vc_paper_code(float score, float maximum) {
  return static_cast<uint8_t>(__float2int_rn(vc_paper_u(score,maximum,1.f)));
}
__device__ __forceinline__ float vc_paper_mul(float a,float b) { return __fmul_rn(a,b); }
__device__ __forceinline__ float vc_paper_add(float a,float b) { return __fadd_rn(a,b); }
__device__ __forceinline__ float vc_paper_alpha(float old,float next) { return expf(__fsub_rn(old,next)); }
