// Direct decoded-P denominator gate, outside timed APIs.
__global__ void tensor_mass_bytes_kernel(const int32_t* p,float* mass,int tiles) {
  int row=blockIdx.x*128+threadIdx.x;
  if(row>=tiles)return; // host guarantees complete 128-thread blocks
  float total[2]={};
  #pragma unroll
  for(int k=0;k<4;++k) {
    uint32_t a[4];
    #pragma unroll
    for(int j=0;j<4;++j)a[j]=static_cast<uint32_t>(p[row*16+k*4+j]);
    float partial[2];sage_vc::vc_tensor_mass_step(a,partial[0],partial[1]);
    #pragma unroll
    for(int r=0;r<2;++r)total[r]=__fadd_rn(total[r],partial[r]);
  }
  mass[row*2]=total[0];mass[row*2+1]=total[1];
}

torch::Tensor check_mass(torch::Tensor p) {
  TORCH_CHECK(p.is_cuda() && p.is_contiguous() && p.scalar_type()==torch::kInt32);
  TORCH_CHECK(p.dim()==2 && p.size(0)>0 && p.size(0)%128==0 && p.size(1)==16);
  c10::cuda::CUDAGuard guard(p.device());
  auto out=torch::empty({p.size(0),int64_t(2)},p.options().dtype(torch::kFloat32));
  auto stream=c10::cuda::getCurrentCUDAStream(p.get_device());
  tensor_mass_bytes_kernel<<<p.size(0)/128,128,0,stream>>>(p.data_ptr<int32_t>(),out.data_ptr<float>(),p.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
