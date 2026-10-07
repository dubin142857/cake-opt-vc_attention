#pragma once

// Included within sage_vc, with CUDA vector types already declared.
// Each float4 is 16-byte aligned and belongs exclusively to one CUDA thread.
// Explicit vector accesses keep the loop-carried values in shared memory.
__device__ __forceinline__ float4 vc_history_load(const float4* ptr) {
  float4 value;
  uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
      : "=f"(value.x),"=f"(value.y),"=f"(value.z),"=f"(value.w) : "r"(address) : "memory");
  return value;
}

__device__ __forceinline__ void vc_history_store(float4* ptr, float4 value) {
  uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(address),
      "f"(value.x),"f"(value.y),"f"(value.z),"f"(value.w) : "memory");
}

__device__ __forceinline__ float4 vc_history_update(
    float4 h, float4 tile, float alpha0, float alpha1) {
  // x/y and z/w belong to different query rows. Preserve separate FP32
  // multiplication and addition, matching the two-level register recurrence.
  h.x=__fadd_rn(__fmul_rn(h.x,alpha0),tile.x);
  h.y=__fadd_rn(__fmul_rn(h.y,alpha0),tile.y);
  h.z=__fadd_rn(__fmul_rn(h.z,alpha1),tile.z);
  h.w=__fadd_rn(__fmul_rn(h.w,alpha1),tile.w);
  return h;
}

__device__ __forceinline__ void vc_history_accumulate(
    float4* ptr, float4 tile, float alpha0, float alpha1) {
  vc_history_store(ptr,vc_history_update(vc_history_load(ptr),tile,alpha0,alpha1));
}
