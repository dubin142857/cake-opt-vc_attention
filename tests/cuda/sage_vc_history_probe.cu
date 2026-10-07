// Test the production shared FP32 history helper without Hopper WGMMA.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
namespace sage_vc {
#include "../../third_party/sageattention_local/csrc/vc/history_storage.cuh"
}

__global__ void history_probe(const float* products,const float* alphas,
                              float* output,int steps,int blocks,bool retain_last) {
  __shared__ __align__(16) float4 history[8*128];
  int lane=threadIdx.x%32;
  int row=(threadIdx.x/32)*16+lane/4;
  int block=blockIdx.x;
  float4 retained=make_float4(0,0,0,0);
  #pragma unroll
  for (int slot=0;slot<8;++slot)
    sage_vc::vc_history_store(history+slot*128+threadIdx.x,make_float4(0,0,0,0));
  for (int step=0;step<steps;++step) {
    const float* tile=products+((step*blocks+block)*64+row)*64;
    const float* a=alphas+(step*blocks+block)*64+row;
    #pragma unroll
    for (int fv=0;fv<4;++fv) {
      #pragma unroll
      for (int chunk=0;chunk<2;++chunk) {
        int col=fv*16+(lane%4)*2+chunk*8;
        float4 value=make_float4(tile[col],tile[col+1],tile[8*64+col],tile[8*64+col+1]);
        if (retain_last && fv*2+chunk==7)
          retained=sage_vc::vc_history_update(retained,value,a[0],a[8]);
        else sage_vc::vc_history_accumulate(history+(fv*2+chunk)*128+threadIdx.x,value,a[0],a[8]);
      }
    }
  }
  float* dst=output+(block*64+row)*64;
  #pragma unroll
  for (int fv=0;fv<4;++fv) {
    #pragma unroll
    for (int chunk=0;chunk<2;++chunk) {
      float4 h=retain_last && fv*2+chunk==7 ? retained :
          sage_vc::vc_history_load(history+(fv*2+chunk)*128+threadIdx.x);
      int col=fv*16+(lane%4)*2+chunk*8;
      dst[col]=h.x; dst[col+1]=h.y; dst[8*64+col]=h.z; dst[8*64+col+1]=h.w;
    }
  }
}

torch::Tensor run(torch::Tensor products,torch::Tensor alphas,bool retain_last) {
  TORCH_CHECK(products.is_cuda() && alphas.device()==products.device());
  TORCH_CHECK(products.scalar_type()==torch::kFloat32 && alphas.scalar_type()==torch::kFloat32);
  TORCH_CHECK(products.is_contiguous() && alphas.is_contiguous());
  TORCH_CHECK(products.dim()==4 && products.size(2)==64 && products.size(3)==64);
  TORCH_CHECK(alphas.dim()==3 && alphas.size(0)==products.size(0) &&
              alphas.size(1)==products.size(1) && alphas.size(2)==64);
  TORCH_CHECK(products.size(0)>0 && products.size(1)>0);
  c10::cuda::CUDAGuard guard(products.device());
  auto output=torch::empty({products.size(1),64,64},products.options());
  history_probe<<<products.size(1),128,0,c10::cuda::getCurrentCUDAStream(products.get_device())>>>(
      products.data_ptr<float>(),alphas.data_ptr<float>(),output.data_ptr<float>(),products.size(0),products.size(1),retain_last);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("run",&run); }
