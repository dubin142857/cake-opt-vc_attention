// Tests vc28 and the balanced-max candidate on SM86+, without Hopper attention.
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
#include "balanced_max_generated.cuh"
}
template<bool BalancedMax>
__global__ void balanced_max_kernel(const int32_t* input,const float* scale,const float* initial,
    int32_t* packed,float* stats,int steps) {
  int row=blockIdx.x*128+threadIdx.x,threads=gridDim.x*128;
  float m[1][2]={{initial[row*2],initial[row*2+1]}};
  for(int step=0;step<steps;++step) {
    int32_t x[1][8][8];uint32_t p[1][4][4];float alpha[1][2],mass[1][2];
    #pragma unroll
    for(int k=0;k<8;++k)
      #pragma unroll
      for(int c=0;c<8;++c)x[0][k][c]=input[(step*threads+row)*64+k*8+c];
    if constexpr (BalancedMax) sage_vc::vc_probability_prepare_balanced_max<false,false,true,false>(x,m,scale[step*threads+row],p,alpha,mass);
    else sage_vc::vc_probability_prepare_integer<false,false,true,false>(x,m,scale[step*threads+row],p,alpha,mass);
    #pragma unroll
    for(int k=0;k<4;++k)
      #pragma unroll
      for(int c=0;c<4;++c)packed[(step*threads+row)*16+k*4+c]=p[0][k][c];
    for(int r=0;r<2;++r) {
      stats[(step*threads+row)*6+r]=m[0][r];
      stats[(step*threads+row)*6+2+r]=alpha[0][r];
      stats[(step*threads+row)*6+4+r]=mass[0][r];
    }
  }
}
std::vector<torch::Tensor> run(torch::Tensor x,torch::Tensor scale,torch::Tensor initial,bool optimized) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kInt32 && x.is_contiguous() && x.dim()==5);
  auto S=x.size(0),B=x.size(1);
  TORCH_CHECK(S>0 && B>0 && x.size(2)==128 && x.size(3)==8 && x.size(4)==8);
  for(auto t:{scale,initial})TORCH_CHECK(t.device()==x.device() && t.is_contiguous() && t.scalar_type()==torch::kFloat32);
  TORCH_CHECK(scale.sizes()==torch::IntArrayRef({S,B,128}) && initial.sizes()==torch::IntArrayRef({B,128,2}));
  c10::cuda::CUDAGuard guard(x.device());
  auto p=torch::empty({S,B,int64_t(128),int64_t(16)},x.options());
  auto stats=torch::empty({S,B,int64_t(128),int64_t(6)},scale.options());
  auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());
  if(optimized)balanced_max_kernel<true><<<B,128,0,stream>>>(x.data_ptr<int32_t>(),scale.data_ptr<float>(),initial.data_ptr<float>(),p.data_ptr<int32_t>(),stats.data_ptr<float>(),S);
  else balanced_max_kernel<false><<<B,128,0,stream>>>(x.data_ptr<int32_t>(),scale.data_ptr<float>(),initial.data_ptr<float>(),p.data_ptr<int32_t>(),stats.data_ptr<float>(),S);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {p,stats};
}

__global__ void max_tree_kernel(const int32_t* input,int32_t* output,int rows) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=rows)return;
  int32_t values[8][8];
  #pragma unroll
  for(int i=0;i<64;++i) values[i/8][i%8]=input[row*64+i];
  #pragma unroll
  for(int r=0;r<2;++r) {
    output[row*4+r*2]=sage_vc::vc_integer_row_max(values,r);
    output[row*4+r*2+1]=sage_vc::vc_integer_row_max_balanced(values,r);
  }
}
torch::Tensor max_tree(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.scalar_type()==torch::kInt32);
  TORCH_CHECK(input.dim()==3 && input.size(0)>0 && input.size(1)==8 && input.size(2)==8);
  c10::cuda::CUDAGuard guard(input.device());
  auto out=torch::empty({input.size(0),int64_t(2),int64_t(2)},input.options());
  max_tree_kernel<<<(input.size(0)+127)/128,128,0,c10::cuda::getCurrentCUDAStream(input.get_device())>>>(input.data_ptr<int32_t>(),out.data_ptr<int32_t>(),input.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("max_tree",&max_tree);}
