// Exact probability-state check; never called inside a timed attention API.
template<bool Fused>
__global__ void fused_quad_check_kernel(const int32_t* input,const float* scale,const float* initial,
    int32_t* packed,float* stats,int steps) {
  int row=blockIdx.x*128+threadIdx.x,threads=gridDim.x*128;
  float m[1][2]={{initial[row*2],initial[row*2+1]}};
  for(int step=0;step<steps;++step) {
    int32_t x[1][8][8];uint32_t p[1][4][4];float alpha[1][2],mass[1][2];
    #pragma unroll
    for(int k=0;k<8;++k)
      #pragma unroll
      for(int c=0;c<8;++c)x[0][k][c]=input[(step*threads+row)*64+k*8+c];
    if constexpr (Fused) sage_vc::vc_probability_prepare_fused_quad<false,false,true,false>(x,m,scale[step*threads+row],p,alpha,mass);
    else sage_vc::vc_probability_prepare_balanced_max<false,false,true,false>(x,m,scale[step*threads+row],p,alpha,mass);
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

std::vector<torch::Tensor> check_probability(torch::Tensor x,torch::Tensor scale,torch::Tensor initial,bool optimized) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kInt32 && x.is_contiguous() && x.dim()==5);
  auto S=x.size(0),B=x.size(1);
  TORCH_CHECK(S>0 && B>0 && x.size(2)==128 && x.size(3)==8 && x.size(4)==8);
  for(auto t:{scale,initial})TORCH_CHECK(t.device()==x.device() && t.is_contiguous() && t.scalar_type()==torch::kFloat32);
  TORCH_CHECK(scale.sizes()==torch::IntArrayRef({S,B,128}) && initial.sizes()==torch::IntArrayRef({B,128,2}));
  c10::cuda::CUDAGuard guard(x.device());
  auto p=torch::empty({S,B,int64_t(128),int64_t(16)},x.options());
  auto stats=torch::empty({S,B,int64_t(128),int64_t(6)},scale.options());
  auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());
  if(optimized)fused_quad_check_kernel<true><<<B,128,0,stream>>>(x.data_ptr<int32_t>(),scale.data_ptr<float>(),initial.data_ptr<float>(),p.data_ptr<int32_t>(),stats.data_ptr<float>(),S);
  else fused_quad_check_kernel<false><<<B,128,0,stream>>>(x.data_ptr<int32_t>(),scale.data_ptr<float>(),initial.data_ptr<float>(),p.data_ptr<int32_t>(),stats.data_ptr<float>(),S);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {p,stats};
}
