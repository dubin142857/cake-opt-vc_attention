// Zero-safe Sage-layout V quantization. SPDX-License-Identifier: Apache-2.0
#include "vc_api.h"
#include "global_values.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <climits>

std::vector<torch::Tensor> vc_prepare_global_values(torch::Tensor packed) {
  TORCH_CHECK(packed.is_cuda() && packed.is_contiguous() && packed.dim()==4 && packed.numel()>0,
      "Expected nonempty contiguous CUDA BHDN transposed/permuted V");
  TORCH_CHECK(packed.scalar_type()==at::kHalf || packed.scalar_type()==at::kBFloat16,"V requires FP16/BF16");
  const int64_t b=packed.size(0),h=packed.size(1),d=packed.size(2),n=packed.size(3);
  TORCH_CHECK((d==64 || d==128) && n%128==0 && packed.numel()<INT32_MAX && b<=65535 && d<=65535,
      "V requires D64/128, N padded to 128, and kernel-supported bounds");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(packed.data_ptr())%16==0,"Packed V requires 16-byte alignment");
  const c10::cuda::CUDAGuard guard(packed.device());
  auto codes=torch::empty(packed.sizes(),packed.options().dtype(at::ScalarType::Float8_e4m3fn));
  auto scales=torch::empty({b,h,d},packed.options().dtype(at::kFloat));
  auto stream=c10::cuda::getCurrentCUDAStream(packed.get_device()).stream();
  if (packed.scalar_type()==at::kHalf) {
    vc_global_values::global_values_kernel<64,true,half><<<dim3(h,b,d),256,0,stream>>>(
        reinterpret_cast<half*>(packed.data_ptr()),reinterpret_cast<int8_t*>(codes.data_ptr()),nullptr,
        scales.data_ptr<float>(),448.f,n,packed.stride(0),packed.stride(2),packed.stride(1),
        codes.stride(0),codes.stride(2),codes.stride(1),0,0,scales.stride(0),scales.stride(1));
  } else {
    vc_global_values::global_values_kernel<64,true,nv_bfloat16><<<dim3(h,b,d),256,0,stream>>>(
        reinterpret_cast<nv_bfloat16*>(packed.data_ptr()),reinterpret_cast<int8_t*>(codes.data_ptr()),nullptr,
        scales.data_ptr<float>(),448.f,n,packed.stride(0),packed.stride(2),packed.stride(1),
        codes.stride(0),codes.stride(2),codes.stride(1),0,0,scales.stride(0),scales.stride(1));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {codes,scales};
}
