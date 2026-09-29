// Isolate probability helpers; run through scripts/benchmark_expcast.py.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include "../third_party/sageattention_local/csrc/math.cuh"
#include "../third_party/sageattention_local/csrc/qattn/attn_utils.cuh"
#include "../third_party/sageattention_local/csrc/vc/vc_math.cuh"
#include "expcast_components.cuh"

template<int Mode>
// Match the D128 production register ceiling for both old and new helpers.
__global__ __maxnreg__(168) void probability_bench(float* result,int iterations) {
  float input[1][8][8],o[1][8][8],m[1][2],d[1][2],mass[1][2],alpha[1][2];
  uint32_t packed[1][4][4];
  unsigned state=(blockIdx.x*128+threadIdx.x+1)*747796405u;
  #pragma unroll
  for(int k=0;k<8;++k)
    #pragma unroll
    for(int j=0;j<8;++j) {
      state=state*1664525u+1013904223u;
      input[0][k][j]=float(int(state>>16)-32768)*0.001f;
      o[0][k][j]=float(j+1)*0.01f;
    }
  m[0][0]=m[0][1]=-5e6f; d[0][0]=d[0][1]=1.f;
  unsigned checksum=0;
  #pragma unroll 1
  for(int step=0;step<iterations;++step) {
    // Vary online maxima and encodings; do not benchmark a loop-invariant P.
    float scale=0.125f+float(step%23)*0.0125f;
    float s[1][8][8];
    #pragma unroll
    for(int k=0;k<8;++k)
      #pragma unroll
      for(int j=0;j<8;++j) s[0][k][j]=input[0][k][j];
    if constexpr(Mode>=2) {
      vc_probability_prepare<(Mode==3)>(s,m,scale,packed,alpha,mass);
      vc_probability_finish(o,d,alpha,mass);
    } else vc_probability<(Mode==1),false,true,true>(s,o,m,d,
        Mode==0 ? scale*math::log2e : scale,packed,mass);
    #pragma unroll
    for(int k=0;k<4;++k)
      #pragma unroll
      for(int j=0;j<4;++j) checksum^=packed[0][k][j];
    // Touch every history accumulator, as the real PV contribution does.
    #pragma unroll
    for(int k=0;k<8;++k)
      #pragma unroll
      for(int j=0;j<8;++j) o[0][k][j]=__fadd_rn(o[0][k][j],0.001f*(j+1));
  }
  float sum=d[0][0]+d[0][1]+float(checksum&255);
  #pragma unroll
  for(int k=0;k<8;++k)
    #pragma unroll
    for(int j=0;j<8;++j) sum+=o[0][k][j];
  result[blockIdx.x*128+threadIdx.x]=sum;
}

void run(torch::Tensor output,int mode,int iterations) {
  TORCH_CHECK(output.is_cuda() && output.scalar_type()==at::kFloat && output.is_contiguous() &&
      output.dim()==2 && output.size(1)==128 && output.size(0)>0 && output.size(0)<=65535,"Expected CUDA [blocks,128] FP32 output");
  TORCH_CHECK(mode>=0 && mode<=3 && iterations>0 && iterations<=100000,"Invalid mode/iterations");
  const c10::cuda::CUDAGuard guard(output.device());
  auto stream=c10::cuda::getCurrentCUDAStream(output.get_device()).stream();
  if(mode==0) probability_bench<0><<<output.size(0),128,0,stream>>>(output.data_ptr<float>(),iterations);
  if(mode==1) probability_bench<1><<<output.size(0),128,0,stream>>>(output.data_ptr<float>(),iterations);
  if(mode==2) probability_bench<2><<<output.size(0),128,0,stream>>>(output.data_ptr<float>(),iterations);
  if(mode==3) probability_bench<3><<<output.size(0),128,0,stream>>>(output.data_ptr<float>(),iterations);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int64_t> resources(int mode) {
  TORCH_CHECK(mode>=0 && mode<=3,"Invalid mode");
  cudaFuncAttributes attr{};
  auto kernel=mode==0?probability_bench<0>:mode==1?probability_bench<1>:mode==2?probability_bench<2>:probability_bench<3>;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,kernel));
  return {attr.numRegs,int64_t(attr.localSizeBytes)};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
  module.attr("benchmark_api_version")=3;
  module.def("run",&run); module.def("resources",&resources);
  module.def("component_run",&component_run); module.def("component_resources",&component_resources);
  module.def("encode_check",&component_encode_check); module.def("decode_check",&component_decode_check);
}
