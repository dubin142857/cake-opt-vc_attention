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
#include "../../third_party/sageattention_local/csrc/vc/max_state.cuh"
}

template<int Mode>
__global__ void work_probe(const int32_t* scores,const float* scales,const float* initial,
    const float* products,int32_t* packed,float* stats,float* output,int steps,bool masked_tail) {
  int row=blockIdx.x*128+threadIdx.x;
  int threads=gridDim.x*128;
  float m[1][2]={{initial[row*2],initial[row*2+1]}},d[1][2]={{1,1}},out[1][8][8]={};
  float bias[1][2]={{__fmaf_rn(-m[0][0],11.541560173034668f,119.6500015258789f),
                    __fmaf_rn(-m[0][1],11.541560173034668f,119.6500015258789f)}};
  uint32_t carried[1][4][4]={};float carried_alpha[1][2]={{1.f,1.f}};
  for (int step=0;step<steps;++step) {
    if constexpr (Mode>=7) {
      sage_vc::vc_probability_drain_mass(carried,carried_alpha,d);
      if (step>0) {
        for (int r=0;r<2;++r) stats[((step-1)*threads+row)*8+6+r]=d[0][r];
      }
    }
    int32_t x[1][8][8];float f[1][8][8],alpha[1][2],mass[1][2];uint32_t p[1][4][4];
    #pragma unroll
    for (int k=0;k<8;++k)
      #pragma unroll
      for (int c=0;c<8;++c) {
        x[0][k][c]=scores[((step*threads+row)*8+k)*8+c];
        f[0][k][c]=__int2float_rz(x[0][k][c]);
        if (masked_tail && step+1==steps && k*4+(c/4)*2+c%2>=1+row%32)
          f[0][k][c]=-CUDART_INF_F;
      }
    bool rescale=false;
    if (masked_tail && step+1==steps)
      sage_vc::vc_probability_prepare<true,true,(Mode>=7),true,true>(f,m,scales[step*threads+row],p,alpha,mass);
    else if constexpr (Mode==4 || Mode==5)
      sage_vc::vc_probability_prepare_maxstate<(Mode==5)>(x,m,scales[step*threads+row],p,alpha,mass,bias,rescale);
    else if constexpr (Mode==1 || Mode==3 || Mode>=6)
      sage_vc::vc_probability_prepare_integer<(Mode==6),(Mode>=7)>(x,m,scales[step*threads+row],p,alpha,mass);
    else sage_vc::vc_probability_prepare<true,true,false,true,true>(f,m,scales[step*threads+row],p,alpha,mass);
    if constexpr (Mode>=7) {
      sage_vc::vc_probability_scale_history(out,alpha);
      // Diagnostic mass output only; real denominator is drained next step.
      sage_vc::vc_probability_register_mass(p,mass);
      for (int r=0;r<2;++r) carried_alpha[0][r]=alpha[0][r];
      for (int k=0;k<4;++k) for (int r=0;r<4;++r) carried[0][k][r]=p[0][k][r];
    } else if constexpr (Mode==4 || Mode==5) sage_vc::vc_probability_finish_maxstate(out,d,alpha,mass,rescale);
    else sage_vc::vc_probability_finish_work<(Mode==2 || Mode==3 || Mode==6)>(out,d,alpha,mass);
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
      if constexpr (Mode<7) stats[(step*threads+row)*8+6+r]=d[0][r];
    }
  }
  if constexpr (Mode>=7) {
    sage_vc::vc_probability_drain_mass(carried,carried_alpha,d);
    for (int r=0;r<2;++r) stats[((steps-1)*threads+row)*8+6+r]=d[0][r];
  }
  #pragma unroll
  for (int k=0;k<8;++k)
    #pragma unroll
    for (int c=0;c<8;++c) output[row*64+k*8+c]=out[0][k][c];
}

std::vector<torch::Tensor> run(torch::Tensor scores,torch::Tensor scales,torch::Tensor initial,
    torch::Tensor products,int mode,bool masked_tail) {
  TORCH_CHECK(scores.is_cuda() && scores.scalar_type()==torch::kInt32 && scores.is_contiguous());
  TORCH_CHECK(scores.dim()==5 && scores.size(0)>0 && scores.size(1)>0 &&
      scores.size(2)==128 && scores.size(3)==8 && scores.size(4)==8);
  for (auto& t : {scales,initial,products})
    TORCH_CHECK(t.device()==scores.device() && t.scalar_type()==torch::kFloat32 && t.is_contiguous());
  TORCH_CHECK(scales.sizes()==torch::IntArrayRef({scores.size(0),scores.size(1),128}));
  TORCH_CHECK(initial.sizes()==torch::IntArrayRef({scores.size(1),128,2}));
  TORCH_CHECK(products.sizes()==scores.sizes() && mode>=0 && mode<=8);
  c10::cuda::CUDAGuard guard(scores.device());
  auto p=torch::empty({scores.size(0),scores.size(1),128,16},scores.options());
  auto stats=torch::empty({scores.size(0),scores.size(1),128,8},scales.options());
  auto out=torch::empty({scores.size(1),128,8,8},scales.options());
  auto stream=c10::cuda::getCurrentCUDAStream(scores.get_device());
  #define LAUNCH(M) work_probe<M><<<scores.size(1),128,0,stream>>>(scores.data_ptr<int32_t>(),scales.data_ptr<float>(),initial.data_ptr<float>(),products.data_ptr<float>(),p.data_ptr<int32_t>(),stats.data_ptr<float>(),out.data_ptr<float>(),scores.size(0),masked_tail)
  switch(mode) {case 0: LAUNCH(0);break;case 1: LAUNCH(1);break;case 2: LAUNCH(2);break;
    case 3: LAUNCH(3);break;case 4: LAUNCH(4);break;case 5: LAUNCH(5);break;case 6: LAUNCH(6);break;
    case 7: LAUNCH(7);break;case 8: LAUNCH(8);break;}
  #undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {p,stats,out};
}
__global__ void encoding_probe(const float* input,int32_t* output,float* affine,int64_t n) {
  int64_t i=blockIdx.x*256+threadIdx.x;
  if (i>=n) return;
  float score=input[i*3],gain=input[i*3+1],bias=input[i*3+2];
  float u=__fmaf_rn(score,gain,bias);
  uint32_t old;
  asm("cvt.rni.sat.u8.f32 %0,%1;" : "=r"(old) : "f"(u));
  output[i*2]=old;
  output[i*2+1]=sage_vc::vc_affine_bit_code(score,__fmul_rn(gain,0.0078125f),__fmul_rn(bias,0.0078125f))&255u;
  affine[i]=u;
}

std::vector<torch::Tensor> encode(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type()==torch::kFloat32 && input.is_contiguous());
  TORCH_CHECK(input.dim()==2 && input.size(0)>0 && input.size(1)==3);
  c10::cuda::CUDAGuard guard(input.device());
  auto out=torch::empty({input.size(0),2},input.options().dtype(torch::kInt32));
  auto affine=torch::empty({input.size(0)},input.options());
  auto stream=c10::cuda::getCurrentCUDAStream(input.get_device());
  encoding_probe<<<(input.size(0)+255)/256,256,0,stream>>>(input.data_ptr<float>(),out.data_ptr<int32_t>(),affine.data_ptr<float>(),input.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out,affine};
}
__global__ void mass_probe(const int32_t* input,float* output) {
  int row=blockIdx.x*128+threadIdx.x;
  uint32_t p[1][4][4];float mass[1][2];
  for (int k=0;k<4;++k) for (int r=0;r<4;++r) p[0][k][r]=input[row*16+k*4+r];
  sage_vc::vc_probability_register_mass(p,mass);
  output[row*2]=mass[0][0];output[row*2+1]=mass[0][1];
}

torch::Tensor sum_p(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type()==torch::kInt32 && input.is_contiguous());
  TORCH_CHECK(input.dim()==3 && input.size(0)>0 && input.size(1)==128 && input.size(2)==16);
  c10::cuda::CUDAGuard guard(input.device());
  auto out=torch::empty({input.size(0),128,2},input.options().dtype(torch::kFloat32));
  auto stream=c10::cuda::getCurrentCUDAStream(input.get_device());
  mass_probe<<<input.size(0),128,0,stream>>>(input.data_ptr<int32_t>(),out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("run",&run,pybind11::arg("scores"),pybind11::arg("scales"),pybind11::arg("initial"),
      pybind11::arg("products"),pybind11::arg("mode"),pybind11::arg("masked_tail")=false);
  m.def("encode",&encode); m.def("sum_p",&sum_p);
}
