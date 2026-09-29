// Diagnostic candidates only. Production attention does not include this file.
#pragma once

// Four already-shifted, nonpositive logits. Modes 0/1 share the exact FP32
// FMA/RNE contract. Mode 2 is exp2 + FP8 at the same scale (256), not Sage's
// 2^8.807 scale and not an accuracy-equivalent encoding.
template<int Mode>
__device__ __forceinline__ uint32_t component_encode4(float a,float b,float c,float d) {
  if constexpr(Mode==0) {
    return __byte_perm(vc_expcast_pair(a,b),vc_expcast_pair(c,d),0x5410);
  } else if constexpr(Mode==1) {
    uint32_t lo=__byte_perm(vc_expcast_nonpositive(a),vc_expcast_nonpositive(b),0x1140);
    uint32_t hi=__byte_perm(vc_expcast_nonpositive(c),vc_expcast_nonpositive(d),0x1140);
    return __byte_perm(lo,hi,0x5410);
  } else {
    float2 lo=make_float2(exp2f(a*math::log2e+8.f),exp2f(b*math::log2e+8.f));
    float2 hi=make_float2(exp2f(c*math::log2e+8.f),exp2f(d*math::log2e+8.f));
    return uint32_t(__nv_cvt_float2_to_fp8x2(lo,__NV_SATFINITE,__NV_E4M3)) |
        (uint32_t(__nv_cvt_float2_to_fp8x2(hi,__NV_SATFINITE,__NV_E4M3))<<16);
  }
}

// Exact value*512 for positive E4M3 codes 0..120, including all subnormals.
// Thirty-two <=256 weights need at most 2^22 units. Convert the tile sum once.
__device__ __forceinline__ uint32_t component_fp8_units(uint32_t code) {
  uint32_t exponent=code>>3, normal=uint32_t(exponent!=0);
  return ((code&7u)|(normal<<3)) << (exponent-normal);
}

template<int Mode>
__device__ __forceinline__ float component_sum32(const uint32_t (&words)[8]) {
  if constexpr(Mode==2) {
    uint32_t even=0,odd=0;
    #pragma unroll
    for(int k=0;k<8;++k) {
      uint32_t word=words[k];
      even += component_fp8_units(word&255u)+component_fp8_units((word>>8)&255u);
      odd += component_fp8_units((word>>16)&255u)+component_fp8_units(word>>24);
    }
    return __uint2float_rn(even+odd)*0.001953125f;
  } else {
    float first=0.f,second=0.f;
    #pragma unroll
    for(int k=0;k<8;++k) {
      float2 lo=vc_decode_pair(words[k]),hi=vc_decode_pair(words[k]>>16);
      if constexpr(Mode==0) {
        float group=__fadd_rn(__fadd_rn(__fadd_rn(lo.x,lo.y),hi.x),hi.y);
        first=__fadd_rn(first,group);
      } else {
        first=__fadd_rn(first,__fadd_rn(lo.x,lo.y));
        second=__fadd_rn(second,__fadd_rn(hi.x,hi.y));
      }
    }
    return __fadd_rn(first,second);
  }
}

template<int Group,int Mode>
__global__ __maxnreg__(168) void component_bench(float* output,int iterations) {
  uint32_t state=(blockIdx.x*128+threadIdx.x+1)*747796405u;
  float shifted[64]; uint32_t words[2][8];
  #pragma unroll
  for(int i=0;i<64;++i) {
    state=state*1664525u+1013904223u;
    if constexpr(Group==0) shifted[i]=-float((state>>16)+1)*0.00018f;
    else {
      // Exponents 0..14 with all mantissas. XOR below changes only mantissas.
      if(i%4==0) words[i/32][(i%32)/4]=0;
      words[i/32][(i%32)/4] |= ((state>>16)%120)<<(8*(i%4));
    }
  }
  if constexpr(Group==1) {
    // Include the maximum code 120; keep its mantissa unchanged by the mask.
    words[0][0]=(words[0][0]&0xffffff00u)|120u;
    words[1][0]=(words[1][0]&0xffffff00u)|120u;
  }
  uint32_t checksum=0; float total=0.f;
  #pragma unroll 1
  for(int step=0;step<iterations;++step) {
    if constexpr(Group==0) {
      float delta=-float(step&7)*0.03125f;
      #pragma unroll
      for(int k=0;k<16;++k) {
        float a,b,c,d;
        // Opaque, changing inputs prevent loop-invariant code elimination.
        // The same four additions and checksum work occur in every encoder.
        asm volatile("add.rn.f32 %0,%4,%8; add.rn.f32 %1,%5,%8;"
                     "add.rn.f32 %2,%6,%8; add.rn.f32 %3,%7,%8;"
          : "=&f"(a),"=&f"(b),"=&f"(c),"=&f"(d)
          : "f"(shifted[k*4]),"f"(shifted[k*4+1]),"f"(shifted[k*4+2]),"f"(shifted[k*4+3]),"f"(delta));
        checksum+=component_encode4<Mode>(a,b,c,d)^uint32_t(step*17+k);
      }
    } else {
      uint32_t mask=uint32_t(step&7)*0x01010101u;
      #pragma unroll
      for(int row=0;row<2;++row) {
        uint32_t current[8];
        #pragma unroll
        for(int k=0;k<8;++k) {
          uint32_t active_mask=k==0?(mask&0xffffff00u):mask;
          asm volatile("xor.b32 %0,%1,%2;" : "=r"(current[k]) : "r"(words[row][k]),"r"(active_mask));
        }
        total=__fadd_rn(total,component_sum32<Mode>(current));
      }
    }
  }
  output[blockIdx.x*128+threadIdx.x]=Group==0?__uint_as_float(checksum):total;
}

void component_run(torch::Tensor output,int group,int mode,int iterations) {
  TORCH_CHECK(output.is_cuda() && output.scalar_type()==at::kFloat && output.is_contiguous() &&
      output.dim()==2 && output.size(1)==128 && output.size(0)>0 && output.size(0)<=65535,"Expected CUDA [blocks,128] FP32 output");
  TORCH_CHECK(group>=0 && group<=1 && mode>=0 && mode<=2 && iterations>0 && iterations<=100000,"Invalid component/iterations");
  const c10::cuda::CUDAGuard guard(output.device());
  auto stream=c10::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto kernel=group==0 ? (mode==0?component_bench<0,0>:mode==1?component_bench<0,1>:component_bench<0,2>) :
      (mode==0?component_bench<1,0>:mode==1?component_bench<1,1>:component_bench<1,2>);
  kernel<<<output.size(0),128,0,stream>>>(output.data_ptr<float>(),iterations);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<int64_t> component_resources(int group,int mode) {
  TORCH_CHECK(group>=0 && group<=1 && mode>=0 && mode<=2,"Invalid component");
  auto kernel=group==0 ? (mode==0?component_bench<0,0>:mode==1?component_bench<0,1>:component_bench<0,2>) :
      (mode==0?component_bench<1,0>:mode==1?component_bench<1,1>:component_bench<1,2>);
  cudaFuncAttributes attr{}; C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,kernel));
  return {attr.numRegs,int64_t(attr.localSizeBytes)};
}

template<int Mode>
__global__ void component_encode_probe(const float* shifted,uint32_t* packed,int64_t words) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<words) packed[i]=component_encode4<Mode>(shifted[4*i],shifted[4*i+1],shifted[4*i+2],shifted[4*i+3]);
}

torch::Tensor component_encode_check(torch::Tensor shifted,int mode) {
  TORCH_CHECK(shifted.is_cuda() && shifted.scalar_type()==at::kFloat && shifted.is_contiguous() &&
      shifted.numel()>0 && shifted.numel()%4==0 && mode>=0 && mode<=2,"Expected nonempty CUDA FP32 groups of four and mode 0..2");
  const c10::cuda::CUDAGuard guard(shifted.device());
  TORCH_CHECK((shifted<=0).all().item<bool>(),"Shifted logits must be nonpositive, without NaN");
  auto result=torch::empty_like(shifted,shifted.options().dtype(at::kByte));
  int64_t words=shifted.numel()/4;
  auto kernel=mode==0?component_encode_probe<0>:mode==1?component_encode_probe<1>:component_encode_probe<2>;
  kernel<<<(words+255)/256,256,0,c10::cuda::getCurrentCUDAStream().stream()>>>(
      shifted.data_ptr<float>(),reinterpret_cast<uint32_t*>(result.data_ptr()),words);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return result;
}

__global__ void component_decode_probe(const uint32_t* input,float* output,int64_t rows) {
  int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(row>=rows) return;
  uint32_t words[8];
  #pragma unroll
  for(int k=0;k<8;++k) words[k]=input[row*8+k];
  output[row*3]=component_sum32<0>(words);
  output[row*3+1]=component_sum32<1>(words);
  output[row*3+2]=component_sum32<2>(words);
}

torch::Tensor component_decode_check(torch::Tensor codes) {
  TORCH_CHECK(codes.is_cuda() && codes.scalar_type()==at::kByte && codes.is_contiguous() &&
      codes.dim()==2 && codes.size(0)>0 && codes.size(1)==32 && uintptr_t(codes.data_ptr())%4==0,"Expected aligned nonempty CUDA uint8 [rows,32]");
  const c10::cuda::CUDAGuard guard(codes.device());
  TORCH_CHECK(codes.max().item<int>()<=120,"Probability codes must be 0..120");
  auto result=torch::empty({codes.size(0),3},codes.options().dtype(at::kFloat));
  component_decode_probe<<<(codes.size(0)+255)/256,256,0,c10::cuda::getCurrentCUDAStream().stream()>>>(
      reinterpret_cast<const uint32_t*>(codes.data_ptr()),result.data_ptr<float>(),codes.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return result;
}
