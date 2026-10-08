// Execute vc31 mean restoration on CUDA independently of Hopper WGMMA.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "../../third_party/sageattention_local/csrc/math.cuh"
#include "../../third_party/sageattention_local/csrc/qattn/attn_utils.cuh"
namespace sage_vc {
#include "../../third_party/sageattention_local/csrc/vc/vc_math.cuh"
}

template<int Dim>
__global__ void mean_probe(const float* history,const float* products,const float* mass,
                          const half* means,float* output,int tiles) {
  constexpr int NV=Dim/16, NP=4;
  int lane=threadIdx.x, block=blockIdx.x, row=block*128+lane;
  float out[1][NV][8], local_mass[1][2];
  #pragma unroll
  for (int v=0;v<NV;++v)
    #pragma unroll
    for (int j=0;j<8;++j) out[0][v][j]=history[row*NV*8+v*8+j];
  local_mass[0][0]=mass[row*2];local_mass[0][1]=mass[row*2+1];
  #pragma unroll
  for (int part=0;part<NV/NP;++part) {
    float product[1][NP][8];
    #pragma unroll
    for (int v=0;v<NP;++v)
      #pragma unroll
      for (int j=0;j<8;++j) product[0][v][j]=products[row*NV*8+(part*NP+v)*8+j];
    sage_vc::vc_restore_block_means(out,product,local_mass,means,block/tiles,block%tiles,
                                  tiles,Dim,lane%32,part*NP);
  }
  #pragma unroll
  for (int v=0;v<NV;++v)
    #pragma unroll
    for (int j=0;j<8;++j) output[row*NV*8+v*8+j]=out[0][v][j];
}

torch::Tensor run(torch::Tensor history,torch::Tensor products,torch::Tensor mass,torch::Tensor means) {
  TORCH_CHECK(history.is_cuda() && history.dim()==5 && history.size(0)>0 && history.size(1)>0);
  TORCH_CHECK(history.size(2)==128 && (history.size(3)==4 || history.size(3)==8) && history.size(4)==8);
  for (auto& t : {history,products,mass})
    TORCH_CHECK(t.device()==history.device() && t.scalar_type()==torch::kFloat32 && t.is_contiguous());
  int h=history.size(0), tiles=history.size(1), dim=history.size(3)*16;
  TORCH_CHECK(products.sizes()==history.sizes());
  TORCH_CHECK(mass.sizes()==torch::IntArrayRef({h,tiles,128,2}));
  TORCH_CHECK(means.device()==history.device() && means.scalar_type()==torch::kFloat16 && means.is_contiguous());
  TORCH_CHECK(means.sizes()==torch::IntArrayRef({h,tiles,dim}));
  c10::cuda::CUDAGuard guard(history.device());
  auto out=torch::empty_like(history);
  auto stream=c10::cuda::getCurrentCUDAStream(history.get_device());
  if (dim==64) mean_probe<64><<<h*tiles,128,0,stream>>>(history.data_ptr<float>(),products.data_ptr<float>(),mass.data_ptr<float>(),reinterpret_cast<const half*>(means.data_ptr()),out.data_ptr<float>(),tiles);
  else mean_probe<128><<<h*tiles,128,0,stream>>>(history.data_ptr<float>(),products.data_ptr<float>(),mass.data_ptr<float>(),reinterpret_cast<const half*>(means.data_ptr()),out.data_ptr<float>(),tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);}
