// Test the pre-fix zero division using the production quantization body.
#include "../../third_party/sageattention_local/csrc/vc/vc_api.h"
#include "../../third_party/sageattention_local/csrc/vc/global_values.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

std::vector<torch::Tensor> legacy(torch::Tensor packed) {
  TORCH_CHECK(packed.is_cuda() && packed.is_contiguous() && packed.dim()==4 && packed.size(3)%128==0,
      "Probe requires padded contiguous CUDA BHDN");
  TORCH_CHECK(packed.scalar_type()==at::kHalf || packed.scalar_type()==at::kBFloat16,"FP16/BF16 only");
  const c10::cuda::CUDAGuard guard(packed.device());
  auto codes=torch::empty(packed.sizes(),packed.options().dtype(at::ScalarType::Float8_e4m3fn));
  auto scales=torch::empty({packed.size(0),packed.size(1),packed.size(2)},packed.options().dtype(at::kFloat));
  auto stream=c10::cuda::getCurrentCUDAStream(packed.get_device()).stream();
  dim3 grid(packed.size(1),packed.size(0),packed.size(2));
  if (packed.scalar_type()==at::kHalf) {
    vc_global_values::global_values_kernel<64,false,half><<<grid,256,0,stream>>>(
        reinterpret_cast<half*>(packed.data_ptr()),reinterpret_cast<int8_t*>(codes.data_ptr()),nullptr,
        scales.data_ptr<float>(),448.f,packed.size(3),packed.stride(0),packed.stride(2),packed.stride(1),
        codes.stride(0),codes.stride(2),codes.stride(1),0,0,scales.stride(0),scales.stride(1));
  } else {
    vc_global_values::global_values_kernel<64,false,nv_bfloat16><<<grid,256,0,stream>>>(
        reinterpret_cast<nv_bfloat16*>(packed.data_ptr()),reinterpret_cast<int8_t*>(codes.data_ptr()),nullptr,
        scales.data_ptr<float>(),448.f,packed.size(3),packed.stride(0),packed.stride(2),packed.stride(1),
        codes.stride(0),codes.stride(2),codes.stride(1),0,0,scales.stride(0),scales.stride(1));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {codes,scales};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("legacy",&legacy);
  m.def("safe",&vc_prepare_global_values);
}
