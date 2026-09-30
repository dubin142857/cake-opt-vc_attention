// VC probability candidates, Apache-2.0; see NOTICE.
// Included after vc_math.cuh. Historical helpers remain unchanged.
#pragma once

// Positive E4M3 codes 0..120 only. Normal values are exponent-rebiased;
// codes 0..7 are exact multiples of 2^-9, including zero.
__device__ __forceinline__ float vc_decode_direct(uint32_t code) {
  float normal = __uint_as_float((code << 20) + (120u << 23));
  float subnormal = __uint2float_rn(code) * 0.001953125f;
  return code >= 8 ? normal : subnormal;
}

template<int Group, int Chains, bool Direct, bool Affine, uint32_t NK>
__device__ __forceinline__ void vc_probability_variant_tile(
    const float (&scores)[NK][8], const float (&maximum)[2], float scale,
    uint32_t (&packed)[NK/2][4], float (&mass)[2]) {
  static_assert(NK == 8 && (Group == 1 || Group == 2 || Group == 4));
  static_assert(Chains == 2 || Chains == 4);
  float gain = __fmul_rn(scale,11.541560173034668f);
  float bias[2] = {__fmaf_rn(-maximum[0],11.541560173034668f,119.6500015258789f),
                   __fmaf_rn(-maximum[1],11.541560173034668f,119.6500015258789f)};
  float sums[2][Chains] = {};
  #pragma unroll
  for (int base=0; base<NK; base+=Group) {
    uint32_t codes[Group][2][2];
    using Decoded = typename std::conditional<Direct,float2,__half2>::type;
    Decoded decoded[Group][2][2];
    #pragma unroll
    for (int k=0; k<Group; ++k)
      #pragma unroll
      for (int r=0; r<2; ++r) {
        float u[4];
        #pragma unroll
        for (int pair=0; pair<2; ++pair) {
          int j=r*2+pair*4;
          #pragma unroll
          for (int t=0; t<2; ++t) {
            if constexpr (Affine)
              u[pair*2+t]=__fmaf_rn(scores[base+k][j+t],gain,bias[r]);
            else
              u[pair*2+t]=__fmaf_rn(__fsub_rn(__fmul_rn(scores[base+k][j+t],scale),maximum[r]),11.541560173034668f,119.6500015258789f);
          }
        }
        vc_expcast_quad_rne(u[0],u[1],u[2],u[3],codes[k][r][0],codes[k][r][1]);
      }
    #pragma unroll
    for (int k=0; k<Group; ++k)
      #pragma unroll
      for (int r=0; r<2; ++r) {
        packed[(base+k)/2][((base+k)%2)*2+r]=codes[k][r][0];
        #pragma unroll
        for (int pair=0; pair<2; ++pair) {
          if constexpr (Direct)
            decoded[k][r][pair]=make_float2(vc_decode_direct(codes[k][r][pair]&255u),
                vc_decode_direct((codes[k][r][pair]>>8)&255u));
          else
            decoded[k][r][pair]=__nv_cvt_fp8x2_to_halfraw2(
                static_cast<__nv_fp8x2_storage_t>(codes[k][r][pair]),__NV_E4M3);
        }
      }
    #pragma unroll
    for (int k=0; k<Group; ++k)
      #pragma unroll
      for (int r=0; r<2; ++r)
        #pragma unroll
        for (int pair=0; pair<2; ++pair) {
          float2 values;
          if constexpr (Direct) values=decoded[k][r][pair];
          else values=__half22float2(decoded[k][r][pair]);
          int chain=((base+k)*2+pair)%Chains;
          sums[r][chain]=__fadd_rn(sums[r][chain],__fadd_rn(values.x,values.y));
        }
  }
  #pragma unroll
  for (int r=0; r<2; ++r) {
    mass[r]=__fadd_rn(sums[r][0],sums[r][1]);
    if constexpr (Chains==4)
      mass[r]=__fadd_rn(mass[r],__fadd_rn(sums[r][2],sums[r][3]));
  }
}

// Mode 0 is exactly vc15/vc16. Candidates vary one dimension at a time:
// 1: one fragment; 2: four fragments; 3: four FP32 chains; 4: direct decode.
template<int Mode, uint32_t NQ, uint32_t NK>
__device__ __forceinline__ void vc_probability_prepare_variant(
    const float (&scores)[NQ][NK][8], float (&maximum)[NQ][2], float scale,
    uint32_t (&packed)[NQ][NK/2][4], float (&alpha)[NQ][2], float (&mass)[NQ][2]) {
  static_assert(Mode>=0 && Mode<=4 && NK==8);
  if constexpr (Mode==0) {
    vc_probability_prepare<true,true,false,true,true>(scores,maximum,scale,packed,alpha,mass);
  } else {
    #pragma unroll
    for (int q=0; q<NQ; ++q) {
      #pragma unroll
      for (int r=0; r<2; ++r) {
        float partial[4]={-CUDART_INF_F,-CUDART_INF_F,-CUDART_INF_F,-CUDART_INF_F};
        #pragma unroll
        for (int k=0; k<NK; ++k)
          #pragma unroll
          for (int j=0; j<4; ++j)
            partial[j]=fmaxf(partial[j],scores[q][k][r*2+(j/2)*4+j%2]);
        float local_max=__fmul_rn(fmaxf(fmaxf(partial[0],partial[1]),fmaxf(partial[2],partial[3])),scale);
        local_max=fmaxf(local_max,__shfl_xor_sync(0xffffffff,local_max,1));
        local_max=fmaxf(local_max,__shfl_xor_sync(0xffffffff,local_max,2));
        float new_max=fmaxf(maximum[q][r],local_max);
        alpha[q][r]=expf(maximum[q][r]-new_max);
        maximum[q][r]=new_max;
      }
      float gain=__fmul_rn(scale,11.541560173034668f);
      bool safe=scale>=1.1754943508222875e-38f && isfinite(gain) &&
          fabsf(maximum[q][0])<=128.f && fabsf(maximum[q][1])<=128.f;
      constexpr int group=Mode==1 ? 1 : Mode==2 ? 4 : 2;
      constexpr int chains=Mode==3 ? 4 : 2;
      if (__all_sync(0xffffffff,safe))
        vc_probability_variant_tile<group,chains,(Mode==4),true>(scores[q],maximum[q],scale,packed[q],mass[q]);
      else
        vc_probability_variant_tile<group,chains,(Mode==4),false>(scores[q],maximum[q],scale,packed[q],mass[q]);
    }
  }
}
