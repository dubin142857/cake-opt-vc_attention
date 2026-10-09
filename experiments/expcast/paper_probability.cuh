// Included inside sage_vc after the existing probability helpers. Apache-2.0.
template<bool Interleaved,typename Input,uint32_t NQ,uint32_t NK>
__device__ __forceinline__ void vc_paper_prepare(
    const Input (&scores)[NQ][NK][8],float (&maximum)[NQ][2],float scale,
    uint32_t (&packed)[NQ][NK/2][4],float (&alpha)[NQ][2],float (&mass)[NQ][2]) {
  static_assert(NK==8);
  #pragma unroll
  for(int q=0;q<NQ;++q) {
    #pragma unroll
    for(int r=0;r<2;++r) {
      float local;
      if constexpr(std::is_same<Input,int32_t>::value)
        local=__fmul_rn(__int2float_rz(vc_integer_row_max(scores[q],r)),scale);
      else {
        float mx=-CUDART_INF_F;
        #pragma unroll
        for(int k=0;k<NK;++k)
          #pragma unroll
          for(int j=0;j<4;++j) mx=fmaxf(mx,scores[q][k][r*2+(j/2)*4+j%2]);
        local=__fmul_rn(mx,scale);
      }
      local=fmaxf(local,__shfl_xor_sync(0xffffffff,local,1));
      local=fmaxf(local,__shfl_xor_sync(0xffffffff,local,2));
      float next=fmaxf(maximum[q][r],local);
      alpha[q][r]=::vc_paper_alpha(maximum[q][r],next);
      maximum[q][r]=next;
    }
    float sums[2][2]={};
    #pragma unroll
    for(int base=0;base<NK;base+=2) {
      uint32_t codes[2][2][2];
      #pragma unroll
      for(int k=0;k<2;++k)
        #pragma unroll
        for(int r=0;r<2;++r) {
          int j=r*2;
          vc_expcast_quad_rne(
            ::vc_paper_u(float(scores[q][base+k][j]),maximum[q][r],scale),
            ::vc_paper_u(float(scores[q][base+k][j+1]),maximum[q][r],scale),
            ::vc_paper_u(float(scores[q][base+k][j+4]),maximum[q][r],scale),
            ::vc_paper_u(float(scores[q][base+k][j+5]),maximum[q][r],scale),
            codes[k][r][0],codes[k][r][1]);
          packed[q][base/2][k*2+r]=codes[k][r][0];
          if constexpr(Interleaved) {
            // Consume the pair immediately; avoid materializing the half2 tile.
            #pragma unroll
            for(int pair=0;pair<2;++pair) {
              float2 v=vc_decode_pair(codes[k][r][pair]);
              sums[r][pair]=__fadd_rn(sums[r][pair],__fadd_rn(v.x,v.y));
            }
          }
        }
      if constexpr(!Interleaved) {
        __half2 decoded[2][2][2];
        #pragma unroll
        for(int k=0;k<2;++k)
          #pragma unroll
          for(int r=0;r<2;++r)
            #pragma unroll
            for(int pair=0;pair<2;++pair)
              decoded[k][r][pair]=__nv_cvt_fp8x2_to_halfraw2(
                static_cast<__nv_fp8x2_storage_t>(codes[k][r][pair]),__NV_E4M3);
        #pragma unroll
        for(int k=0;k<2;++k)
          #pragma unroll
          for(int r=0;r<2;++r)
            #pragma unroll
            for(int pair=0;pair<2;++pair) {
              float2 v=__half22float2(decoded[k][r][pair]);
              sums[r][pair]=__fadd_rn(sums[r][pair],__fadd_rn(v.x,v.y));
            }
      }
    }
    mass[q][0]=__fadd_rn(sums[0][0],sums[0][1]);
    mass[q][1]=__fadd_rn(sums[1][0],sums[1][1]);
  }
}
