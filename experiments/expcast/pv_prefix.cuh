// Experimental late K32 decoded-P mass; original vc28 arithmetic. Apache-2.0.
#pragma once

struct VCLateProbability {
  float sums[2][2];
  bool active;
};

// Two K16 score fragments make one K32 PV operand. Decode and sum before
// handing this operand to WGMMA; never inspect it again while MMA is pending.
template<bool Sum=true>
__device__ __forceinline__ void vc_encode_probability_group(
    const int32_t (&scores)[2][8], const float (&bias)[2], float gain,
    uint32_t (&packed)[4], float (&sums)[2][2]) {
  uint32_t codes[2][2][2]; __half2 decoded[2][2][2];
  #pragma unroll
  for (int k=0;k<2;++k)
    #pragma unroll
    for (int r=0;r<2;++r) {
      int j=r*2;
      vc_int_affine_quad(scores[k][j],scores[k][j+1],scores[k][j+4],scores[k][j+5],
                         gain,bias[r],codes[k][r][0],codes[k][r][1]);
    }
  #pragma unroll
  for (int k=0;k<2;++k)
    #pragma unroll
    for (int r=0;r<2;++r) {
      packed[k*2+r]=codes[k][r][0];
      if constexpr (Sum) {
        #pragma unroll
        for (int pair=0;pair<2;++pair)
          decoded[k][r][pair]=__nv_cvt_fp8x2_to_halfraw2(
              static_cast<__nv_fp8x2_storage_t>(codes[k][r][pair]),__NV_E4M3);
      }
    }
  if constexpr (Sum) {
  #pragma unroll
  for (int k=0;k<2;++k)
    #pragma unroll
    for (int r=0;r<2;++r)
      #pragma unroll
      for (int pair=0;pair<2;++pair) {
        float2 v=__half22float2(decoded[k][r][pair]);
        sums[r][pair]=__fadd_rn(sums[r][pair],__fadd_rn(v.x,v.y));
      }
  }
}

template<uint32_t NQ,uint32_t NK>
__device__ __forceinline__ void vc_probability_prepare_prefix(
    const int32_t (&scores)[NQ][NK][8],float (&maximum)[NQ][2],float scale,
    uint32_t (&packed)[NQ][NK/2][4],float (&alpha)[NQ][2],float (&mass)[NQ][2],
    VCLateProbability& late) {
  static_assert(NQ==1 && NK==8);
  #pragma unroll
  for(int r=0;r<2;++r) {
    float local=__fmul_rn(__int2float_rz(vc_integer_row_max(scores[0],r)),scale);
    local=fmaxf(local,__shfl_xor_sync(0xffffffff,local,1));
    local=fmaxf(local,__shfl_xor_sync(0xffffffff,local,2));
    float next=fmaxf(maximum[0][r],local);
    alpha[0][r]=expf(maximum[0][r]-next);
    maximum[0][r]=next;
  }
  float gain=__fmul_rn(scale,11.541560173034668f);
  bool safe=scale>=1.1754943508222875e-38f && isfinite(gain) &&
      fabsf(maximum[0][0])<=128.f && fabsf(maximum[0][1])<=128.f;
  late.active=__all_sync(0xffffffff,safe);
  if(late.active) {
    float bias[2];
    #pragma unroll
    for(int r=0;r<2;++r) {
      bias[r]=__fmaf_rn(-maximum[0][r],11.541560173034668f,119.6500015258789f);
      late.sums[r][0]=late.sums[r][1]=0.f;
    }
    #pragma unroll
    for(int group=0;group<3;++group) {
      int32_t part[2][8];
      #pragma unroll
      for(int k=0;k<2;++k)
        #pragma unroll
        for(int j=0;j<8;++j) part[k][j]=scores[0][group*2+k][j];
      vc_encode_probability_group(part,bias,gain,packed[0][group],late.sums);
    }
    int32_t last[2][8];
    #pragma unroll
    for(int k=0;k<2;++k)
      #pragma unroll
      for(int j=0;j<8;++j) last[k][j]=scores[0][6+k][j];
    vc_encode_probability_group<false>(last,bias,gain,packed[0][3],late.sums);
  } else {
    // Same full-tile fallback as vc28; it has no unfinished probability work.
    float converted[NK][8];
    #pragma unroll
    for(int k=0;k<NK;++k)
      #pragma unroll
      for(int j=0;j<8;++j) converted[k][j]=__int2float_rz(scores[0][k][j]);
    vc_probability_variant_tile<2,2,false,false>(converted,maximum[0],scale,packed[0],mass[0]);
  }
}

__device__ __forceinline__ void vc_probability_finish_prefix(
    VCLateProbability& late,const uint32_t (&last_packed)[4],float (&mass)[2]) {
  if(late.active) {
    #pragma unroll
    for(int k=0;k<2;++k)
      #pragma unroll
      for(int r=0;r<2;++r) {
        float2 low=vc_decode_pair(last_packed[k*2+r]);
        float2 high=vc_decode_pair(last_packed[k*2+r]>>16);
        late.sums[r][0]=__fadd_rn(late.sums[r][0],__fadd_rn(low.x,low.y));
        late.sums[r][1]=__fadd_rn(late.sums[r][1],__fadd_rn(high.x,high.y));
      }
    mass[0]=__fadd_rn(late.sums[0][0],late.sums[0][1]);
    mass[1]=__fadd_rn(late.sums[1][0],late.sums[1][1]);
  }
}
