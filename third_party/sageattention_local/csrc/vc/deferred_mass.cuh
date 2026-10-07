// vc25: decode completed P while the following QK is pending. Apache-2.0.
// Included after vc_math.cuh. No access to an in-flight WGMMA operand.
#pragma once

template<uint32_t NK>
__device__ __forceinline__ void vc_probability_encode_fallback(
    const float (&scores)[NK][8],const float (&maximum)[2],float scale,
    uint32_t (&packed)[NK/2][4]) {
  #pragma unroll
  for (int k=0;k<NK;++k)
    #pragma unroll
    for (int r=0;r<2;++r) {
      float u[4];
      #pragma unroll
      for (int j=0;j<4;++j)
        u[j]=__fmaf_rn(__fsub_rn(__fmul_rn(scores[k][r*2+(j/2)*4+j%2],scale),maximum[r]),
            11.541560173034668f,119.6500015258789f);
      uint32_t high;
      vc_expcast_quad_rne(u[0],u[1],u[2],u[3],packed[k/2][(k%2)*2+r],high);
    }
}

template<uint32_t NQ,uint32_t NP>
__device__ __forceinline__ void vc_probability_register_mass(
    const uint32_t (&packed)[NQ][NP][4],float (&mass)[NQ][2]) {
  static_assert(NP==4);
  #pragma unroll
  for (int q=0;q<NQ;++q) {
    float sums[2][2]={};
    #pragma unroll
    for (int k=0;k<NP*2;++k)
      #pragma unroll
      for (int r=0;r<2;++r) {
        uint32_t word;
        // PTX read occurs at the requested source position, after PREVIOUS PV
        // wait. ptxas can reschedule independent instructions; source-serial
        // and source-overlap modes may produce identical machine code.
        asm volatile("mov.b32 %0,%1;" : "=r"(word) : "r"(packed[q][k/2][(k%2)*2+r]));
        float2 low=vc_decode_pair(word),high=vc_decode_pair(word>>16);
        sums[r][0]=__fadd_rn(sums[r][0],__fadd_rn(low.x,low.y));
        sums[r][1]=__fadd_rn(sums[r][1],__fadd_rn(high.x,high.y));
      }
    mass[q][0]=__fadd_rn(sums[0][0],sums[0][1]);
    mass[q][1]=__fadd_rn(sums[1][0],sums[1][1]);
  }
}

template<uint32_t NQ,uint32_t NP>
__device__ __forceinline__ void vc_probability_drain_mass(
    const uint32_t (&packed)[NQ][NP][4],const float (&alpha)[NQ][2],float (&denom)[NQ][2]) {
  float mass[NQ][2];
  vc_probability_register_mass(packed,mass);
  #pragma unroll
  for (int q=0;q<NQ;++q)
    #pragma unroll
    for (int r=0;r<2;++r)
      asm volatile("mul.rn.f32 %0,%0,%1; add.rn.f32 %0,%0,%2;"
          : "+f"(denom[q][r]) : "f"(alpha[q][r]),"f"(mass[q][r]));
}

template<uint32_t NQ,uint32_t NV>
__device__ __forceinline__ void vc_probability_scale_history(
    float (&out)[NQ][NV][8],const float (&alpha)[NQ][2]) {
  static_assert(NQ==1);
  if (!__all_sync(0xffffffff,alpha[0][0]==1.f && alpha[0][1]==1.f)) {
    #pragma unroll
    for (int r=0;r<2;++r)
      #pragma unroll
      for (int v=0;v<NV;++v)
        #pragma unroll
        for (int j=0;j<4;++j)
          asm volatile("mul.rn.f32 %0,%0,%1;"
              : "+f"(out[0][v][r*2+(j/2)*4+j%2]) : "f"(alpha[0][r]));
  }
}
