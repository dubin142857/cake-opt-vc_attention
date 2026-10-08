// Check production descriptor arithmetic on actual CUDA shared addresses.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "../../third_party/sageattention_local/csrc/wgmma.cuh"
namespace sage_vc {
#include "../../third_party/sageattention_local/csrc/vc/descriptor_mma.cuh"
}

template<int Stride>
__global__ void descriptor_probe(int64_t* output) {
  __shared__ __align__(128) int8_t shared[32768];
  auto* anchor=shared+blockIdx.x*128;
  uint64_t descriptor=wgmma::make_smem_desc<Stride>(anchor);
  #pragma unroll
  for (int fragment=0;fragment<4;++fragment) {
    int offset=(blockIdx.x*4+fragment)*2;
    output[offset]=int64_t(wgmma::make_smem_desc<Stride>(anchor+fragment*32));
    output[offset+1]=int64_t(sage_vc::vc_descriptor_k32(descriptor,fragment));
  }
}

torch::Tensor run(torch::Tensor like,int stride) {
  TORCH_CHECK(like.is_cuda() && (stride==64 || stride==128));
  c10::cuda::CUDAGuard guard(like.device());
  auto result=torch::empty({256,4,2},like.options().dtype(torch::kInt64));
  auto stream=c10::cuda::getCurrentCUDAStream(like.get_device());
  if (stride==64) descriptor_probe<64><<<256,1,0,stream>>>(result.data_ptr<int64_t>());
  else descriptor_probe<128><<<256,1,0,stream>>>(result.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);}
