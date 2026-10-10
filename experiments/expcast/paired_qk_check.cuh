// Small Hopper-only check of the N256 accumulator-to-matrix mapping.
// Uses the exact paired_qk_mma helper used by the attention candidate.
namespace sage_vc {
template<int D>
__global__ void check_paired_qk_kernel(const __grid_constant__ CUtensorMap qm,
    const __grid_constant__ CUtensorMap km, int32_t* output) {
  extern __shared__ __align__(128) int8_t storage[];
  int8_t* q=storage;
  int8_t* k=storage+64*D;
  __shared__ __align__(8) uint64_t bar;
  if (threadIdx.x==0) init_barrier(&bar,1);
  __syncthreads();
  if (threadIdx.x==0) {
    expect_bytes<320*D>(&bar);
    load_async_4D(q,&qm,&bar,0,0,0,blockIdx.x);
    load_async_4D(k,&km,&bar,0,0,0,blockIdx.x);
  }
  wait(&bar,0);
  int32_t score[2][1][8][8];
  wgmma::warpgroup_arrive();
  paired_qk_mma<0,D>(score,q,k);
  #pragma unroll
  for (int ki=1;ki<D/32;++ki) paired_qk_mma<1,D>(score,q+ki*32,k+ki*32);
  wgmma::warpgroup_commit_batch();
  wgmma::warpgroup_wait<0>();
  int lane=threadIdx.x%32, warp=threadIdx.x/32;
  #pragma unroll
  for (int half=0;half<2;++half)
    #pragma unroll
    for (int frag=0;frag<8;++frag)
      #pragma unroll
      for (int reg=0;reg<8;++reg) {
        int row=warp*16+lane/4+8*((reg%4)/2);
        int col=half*128+frag*16+2*(lane%4)+8*(reg/4)+reg%2;
        output[(blockIdx.x*64+row)*256+col]=score[half][0][frag][reg];
      }
}
}

namespace {
template<int D>
void launch_qk_check(torch::Tensor q,torch::Tensor k,torch::Tensor out) {
  auto qm=sage_vc::create_tensor_map_4D<64,D>(q.data_ptr<int8_t>(),
      q.size(0),1,64,D,64*D,64*D,D);
  auto km=sage_vc::create_tensor_map_4D<256,D>(k.data_ptr<int8_t>(),
      k.size(0),1,256,D,256*D,256*D,D);
  C10_CUDA_CHECK(cudaFuncSetAttribute(sage_vc::check_paired_qk_kernel<D>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,320*D));
  sage_vc::check_paired_qk_kernel<D><<<q.size(0),128,320*D,
      c10::cuda::getCurrentCUDAStream(q.get_device()).stream()>>>(qm,km,out.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor check_paired_qk(torch::Tensor q,torch::Tensor k) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && q.device()==k.device() &&
      q.is_contiguous() && k.is_contiguous(),"Expected contiguous CUDA Q/K on one device");
  c10::cuda::CUDAGuard guard(q.device());
  cudaDeviceProp props; C10_CUDA_CHECK(cudaGetDeviceProperties(&props,q.get_device()));
  TORCH_CHECK(props.major==9 && props.minor==0,"N256 QK check requires SM90a");
  TORCH_CHECK(q.dim()==3 && k.dim()==3 && q.scalar_type()==torch::kInt8 &&
      k.scalar_type()==torch::kInt8 && q.size(0)>0 && q.size(0)==k.size(0) &&
      q.size(1)==64 && k.size(1)==256 && q.size(2)==k.size(2) &&
      (q.size(2)==64 || q.size(2)==128),"Expected INT8 [B,64,D] / [B,256,D], D64/128");
  auto out=torch::empty({q.size(0),int64_t(64),int64_t(256)},q.options().dtype(torch::kInt32));
  if (q.size(2)==64) launch_qk_check<64>(q,k,out);
  else launch_qk_check<128>(q,k,out);
  return out;
}
}
