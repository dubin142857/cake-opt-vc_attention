// vc17 diagnostics: actual probability helpers, not attention performance.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include "../third_party/sageattention_local/csrc/math.cuh"
#include "../third_party/sageattention_local/csrc/qattn/attn_utils.cuh"
#include "../third_party/sageattention_local/csrc/vc/vc_math.cuh"
#include "../third_party/sageattention_local/csrc/vc/probability_variants.cuh"

template<int Mode>
__device__ __forceinline__ float probability_sum(const uint32_t (&words)[8]) {
  constexpr int chains=Mode==1 ? 4 : 2;
  float sums[chains]={};
  #pragma unroll
  for (int k=0;k<8;++k)
    #pragma unroll
    for (int pair=0;pair<2;++pair) {
      uint32_t code=words[k]>>(16*pair);
      float2 values;
      if constexpr (Mode==2)
        values=make_float2(vc_decode_direct(code&255u),vc_decode_direct((code>>8)&255u));
      else values=vc_decode_pair(code);
      int chain=(k*2+pair)%chains;
      sums[chain]=__fadd_rn(sums[chain],__fadd_rn(values.x,values.y));
    }
  float result=__fadd_rn(sums[0],sums[1]);
  if constexpr (chains==4) result=__fadd_rn(result,__fadd_rn(sums[2],sums[3]));
  return result;
}

template<int Group,int Mode>
__global__ __maxnreg__(168) void probability_candidate_bench(float* output,int iterations) {
  float input[1][8][8],out[1][8][8],maximum[1][2]={{-5e6f,-5e6f}},denom[1][2]={{1.f,1.f}};
  uint32_t words[2][8]={};
  uint32_t state=(blockIdx.x*128+threadIdx.x+1)*747796405u;
  #pragma unroll
  for (int i=0;i<64;++i) {
    state=state*1664525u+1013904223u;
    input[0][i/8][i%8]=float(int(state>>16)-32768)*.001f;
    out[0][i/8][i%8]=float(i%8+1)*.01f;
    words[i/32][(i%32)/4]|=((state>>16)%120)<<(8*(i%4));
  }
  // Keep maximum-code mantissas fixed when varying the other words below.
  words[0][0]=(words[0][0]&0xffffff00u)|120u;
  words[1][0]=(words[1][0]&0xffffff00u)|120u;
  uint32_t checksum=0;float total=0.f;
  #pragma unroll 1
  for (int step=0;step<iterations;++step) {
    if constexpr (Group==0) {
      float mass[1][2],alpha[1][2];uint32_t packed[1][4][4];
      vc_probability_prepare_variant<Mode>(input,maximum,.125f+float(step%23)*.0125f,packed,alpha,mass);
      vc_probability_finish(out,denom,alpha,mass);
      #pragma unroll
      for (int k=0;k<4;++k)
        #pragma unroll
        for (int j=0;j<4;++j) checksum^=packed[0][k][j];
      #pragma unroll
      for (int k=0;k<8;++k)
        #pragma unroll
        for (int j=0;j<8;++j) out[0][k][j]=__fadd_rn(out[0][k][j],.001f*(j+1));
    } else if constexpr (Group==1) {
      uint32_t mask=uint32_t(step&7)*0x01010101u;
      #pragma unroll
      for (int row=0;row<2;++row) {
        uint32_t current[8];
        #pragma unroll
        for (int k=0;k<8;++k) {
          uint32_t active=k==0 ? mask&0xffffff00u : mask;
          asm volatile("xor.b32 %0,%1,%2;" : "=r"(current[k]) : "r"(words[row][k]),"r"(active));
        }
        total=__fadd_rn(total,probability_sum<Mode>(current));
      }
    } else {
      // Current fused quad encoding only; both pairs contribute to checksum.
      #pragma unroll
      for (int k=0;k<16;++k) {
        float x[4];
        #pragma unroll
        for (int j=0;j<4;++j) {
          float shifted=-fabsf(input[0][(4*k+j)/8][(4*k+j)%8]);
          asm volatile("add.rn.f32 %0,%1,%2;" : "=f"(x[j]) : "f"(shifted),"f"(-float(step&7)*.03125f));
          x[j]=__fmaf_rn(x[j],11.541560173034668f,119.6500015258789f);
        }
        uint32_t word,upper;vc_expcast_quad_rne(x[0],x[1],x[2],x[3],word,upper);
        checksum+=word^uint32_t(step*17+k);
      }
    }
  }
  if constexpr (Group==0) {
    total=denom[0][0]+denom[0][1]+float(checksum&255);
    #pragma unroll
    for (int k=0;k<8;++k)
      #pragma unroll
      for (int j=0;j<8;++j) total+=out[0][k][j];
  }
  output[blockIdx.x*128+threadIdx.x]=Group==2 ? __uint_as_float(checksum) : total;
}

auto select_bench(int group,int mode) {
  TORCH_CHECK(group>=0 && group<=2 && mode>=0 &&
      ((group==0 && mode<=4)||(group==1 && mode<=2)||(group==2 && mode==0)),"Invalid benchmark group/mode");
  if (group==1) return mode==0?probability_candidate_bench<1,0>:mode==1?probability_candidate_bench<1,1>:probability_candidate_bench<1,2>;
  if (group==2) return probability_candidate_bench<2,0>;
  return mode==0?probability_candidate_bench<0,0>:mode==1?probability_candidate_bench<0,1>:
      mode==2?probability_candidate_bench<0,2>:mode==3?probability_candidate_bench<0,3>:probability_candidate_bench<0,4>;
}
void candidate_run(torch::Tensor output,int group,int mode,int iterations) {
  TORCH_CHECK(output.is_cuda() && output.scalar_type()==at::kFloat && output.is_contiguous() &&
      output.dim()==2 && output.size(1)==128 && output.size(0)>0 && output.size(0)<=65535 && iterations>0 && iterations<=100000,"Expected CUDA FP32 [blocks,128] and positive iterations");
  const c10::cuda::CUDAGuard guard(output.device());
  auto kernel=select_bench(group,mode);
  kernel<<<output.size(0),128,0,c10::cuda::getCurrentCUDAStream().stream()>>>(output.data_ptr<float>(),iterations);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int64_t> candidate_resources(int group,int mode) {
  cudaFuncAttributes attr{};C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,select_bench(group,mode)));
  return {attr.numRegs,int64_t(attr.localSizeBytes)};
}

template<int Mode>
__global__ void candidate_probe(const float* inputs,const float* scales,const float* history,
    int32_t* codes,float* stats,int rows,int steps) {
  int row=blockIdx.x*128+threadIdx.x;
  if(row>=rows) return; // rows multiple of 32: all participating warps are full.
  float maximum[1][2]={{history[row*2],history[row*2+1]}};
  for (int step=0;step<steps;++step) {
    float scores[1][8][8],alpha[1][2],mass[1][2];uint32_t packed[1][4][4];
    #pragma unroll
    for(int i=0;i<64;++i) scores[0][i/8][i%8]=inputs[(int64_t(step)*rows+row)*64+i];
    vc_probability_prepare_variant<Mode>(scores,maximum,scales[step],packed,alpha,mass);
    #pragma unroll
    for(int i=0;i<16;++i) codes[(int64_t(step)*rows+row)*16+i]=packed[0][i/4][i%4];
    #pragma unroll
    for(int r=0;r<2;++r) {
      int64_t offset=(int64_t(step)*rows+row)*6+r;
      stats[offset]=mass[0][r];stats[offset+2]=maximum[0][r];stats[offset+4]=alpha[0][r];
    }
  }
}
std::vector<torch::Tensor> candidate_check(torch::Tensor inputs,torch::Tensor scales,torch::Tensor history,int mode) {
  TORCH_CHECK(inputs.is_cuda() && inputs.dim()==4 && inputs.size(2)==8 && inputs.size(3)==8 &&
      inputs.size(0)>0 && inputs.size(1)>0 && inputs.size(1)%32==0,"Expected CUDA [steps,rows,8,8], rows multiple of 32");
  for(const auto& x:{inputs,scales,history})
    TORCH_CHECK(x.device()==inputs.device() && x.scalar_type()==at::kFloat && x.is_contiguous(),"Expected contiguous FP32 on one device");
  TORCH_CHECK(scales.numel()==inputs.size(0) && history.numel()==inputs.size(1)*2 && mode>=0 && mode<=4,"Invalid probe metadata or mode");
  TORCH_CHECK(inputs.numel()<INT32_MAX,"Probe exceeds bounds");
  TORCH_CHECK((scales>0).all().item<bool>() && at::isfinite(scales).all().item<bool>(),"Scales require positive finite values");
  const c10::cuda::CUDAGuard guard(inputs.device());
  auto codes=torch::empty({inputs.size(0),inputs.size(1),16},inputs.options().dtype(at::kInt));
  auto stats=torch::empty({inputs.size(0),inputs.size(1),6},inputs.options());
  auto kernel=mode==0?candidate_probe<0>:mode==1?candidate_probe<1>:mode==2?candidate_probe<2>:mode==3?candidate_probe<3>:candidate_probe<4>;
  kernel<<<(inputs.size(1)+127)/128,128,0,c10::cuda::getCurrentCUDAStream().stream()>>>(
      inputs.data_ptr<float>(),scales.data_ptr<float>(),history.data_ptr<float>(),codes.data_ptr<int32_t>(),stats.data_ptr<float>(),inputs.size(1),inputs.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {codes,stats};
}

__global__ void direct_decode_probe(const uint8_t* codes,float* output,int64_t count) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<count) output[i]=vc_decode_direct(codes[i]);
}
torch::Tensor direct_decode_check(torch::Tensor codes) {
  TORCH_CHECK(codes.is_cuda() && codes.scalar_type()==at::kByte && codes.is_contiguous() && codes.numel()>0,"Expected nonempty CUDA uint8 codes");
  const c10::cuda::CUDAGuard guard(codes.device());
  TORCH_CHECK(codes.max().item<int>()<=120,"Codes must be 0..120");
  auto result=torch::empty_like(codes,codes.options().dtype(at::kFloat));
  direct_decode_probe<<<(codes.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream().stream()>>>(codes.data_ptr<uint8_t>(),result.data_ptr<float>(),codes.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.attr("probability_api_version")=1;
  m.def("run",&candidate_run);m.def("resources",&candidate_resources);
  m.def("check",&candidate_check);m.def("decode",&direct_decode_check);
}
