// Execute production probability/history helpers independently of WGMMA.
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
#include "../../third_party/sageattention_local/csrc/vc/probability_variants.cuh"
#include "../../third_party/sageattention_local/csrc/vc/work_reduction.cuh"
}

template<int Mode>
__global__ void work_probe(const int32_t* scores,const float* scales,const float* initial,
    const float* products,int32_t* packed,float* stats,float* output,int steps) {
  int row=blockIdx.x*128+threadIdx.x;
  int threads=gridDim.x*128;
  float m[1][2]={{initial[row*2],initial[row*2+1]}},d[1][2]={{1,1}},out[1][8][8]={};
  for (int step=0;step<steps;++step) {
    int32_t x[1][8][8];float f[1][8][8],alpha[1][2],mass[1][2];uint32_t p[1][4][4];
    #pragma unroll
    for (int k=0;k<8;++k)
      #pragma unroll
      for (int c=0;c<8;++c) {
        x[0][k][c]=scores[((step*threads+row)*8+k)*8+c];
        f[0][k][c]=__int2float_rz(x[0][k][c]);
      }
    if constexpr (Mode==1 || Mode==3)
      sage_vc::vc_probability_prepare_integer(x,m,scales[step*threads+row],p,alpha,mass);
    else sage_vc::vc_probability_prepare<true,true,false,true,true>(f,m,scales[step*threads+row],p,alpha,mass);
    sage_vc::vc_probability_finish_work<(Mode==2 || Mode==3)>(out,d,alpha,mass);
    #pragma unroll
    for (int k=0;k<8;++k)
      #pragma unroll
      for (int c=0;c<8;++c)
        out[0][k][c]=__fadd_rn(out[0][k][c],products[((step*threads+row)*8+k)*8+c]);
    #pragma unroll
    for (int k=0;k<4;++k)
      #pragma unroll
      for (int c=0;c<4;++c) packed[(step*threads+row)*16+k*4+c]=p[0][k][c];
    #pragma unroll
    for (int r=0;r<2;++r) {
      stats[(step*threads+row)*8+r]=m[0][r];
      stats[(step*threads+row)*8+2+r]=alpha[0][r];
      stats[(step*threads+row)*8+4+r]=mass[0][r];
      stats[(step*threads+row)*8+6+r]=d[0][r];
    }
  }
  #pragma unroll
  for (int k=0;k<8;++k)
    #pragma unroll
    for (int c=0;c<8;++c) output[row*64+k*8+c]=out[0][k][c];
}

std::vector<torch::Tensor> run(torch::Tensor scores,torch::Tensor scales,torch::Tensor initial,
    torch::Tensor products,int mode) {
  TORCH_CHECK(scores.is_cuda() && scores.scalar_type()==torch::kInt32 && scores.is_contiguous());
  TORCH_CHECK(scores.dim()==5 && scores.size(0)>0 && scores.size(1)>0 &&
      scores.size(2)==128 && scores.size(3)==8 && scores.size(4)==8);
  for (auto& t : {scales,initial,products})
    TORCH_CHECK(t.device()==scores.device() && t.scalar_type()==torch::kFloat32 && t.is_contiguous());
  TORCH_CHECK(scales.sizes()==torch::IntArrayRef({scores.size(0),scores.size(1),128}));
  TORCH_CHECK(initial.sizes()==torch::IntArrayRef({scores.size(1),128,2}));
  TORCH_CHECK(products.sizes()==scores.sizes() && mode>=0 && mode<=3);
  c10::cuda::CUDAGuard guard(scores.device());
  auto p=torch::empty({scores.size(0),scores.size(1),128,16},scores.options());
  auto stats=torch::empty({scores.size(0),scores.size(1),128,8},scales.options());
  auto out=torch::empty({scores.size(1),128,8,8},scales.options());
  auto stream=c10::cuda::getCurrentCUDAStream(scores.get_device());
  #define LAUNCH(M) work_probe<M><<<scores.size(1),128,0,stream>>>(scores.data_ptr<int32_t>(),scales.data_ptr<float>(),initial.data_ptr<float>(),products.data_ptr<float>(),p.data_ptr<int32_t>(),stats.data_ptr<float>(),out.data_ptr<float>(),scores.size(0))
  switch(mode) {case 0: LAUNCH(0);break;case 1: LAUNCH(1);break;case 2: LAUNCH(2);break;case 3: LAUNCH(3);break;}
  #undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {p,stats,out};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("run",&run); }
