// VC block residual quantization, Apache-2.0; see NOTICE.
#include "vc_api.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "expcast.cuh"
#include <climits>

namespace {
__device__ float to_float(float x) { return x; }
__device__ float to_float(half x) { return __half2float(x); }
__device__ float to_float(nv_bfloat16 x) { return __bfloat162float(x); }
__device__ float sum_warp(float x) {
  for (int shift = 16; shift; shift >>= 1) x += __shfl_xor_sync(0xffffffff, x, shift);
  return x;
}
__device__ float max_warp(float x) {
  for (int shift = 16; shift; shift >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, shift));
  return x;
}
template<typename T>
__global__ void value_blocks(const T* input, uint8_t* payload, float* scales, half* means, int n, int dim, int tiles) {
  int lane = threadIdx.x % 32, channel = blockIdx.y * 8 + threadIdx.x / 32;
  int head = blockIdx.z, tile = blockIdx.x, count = min(128, n - tile * 128);
  float x[4], total = 0;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    int row = lane + j * 32;
    x[j] = row < count ? to_float(input[(int64_t(head) * n + tile * 128 + row) * dim + channel]) : 0;
    total += x[j];
  }
  float mean = __fdiv_rn(sum_warp(total), float(count)), amax = 0;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    x[j] = lane + j * 32 < count ? __fsub_rn(x[j], mean) : 0;
    amax = fmaxf(amax, fabsf(x[j]));
  }
  amax = max_warp(amax);
  float scale = amax > 0 ? __fdiv_rn(amax, 448.0f) : 1.0f;
  scale = fmaxf(scale, __fdiv_rn(fabsf(mean), 65504.0f));
  if (lane == 0) {
    int64_t meta = (int64_t(head) * tiles + tile) * dim + channel;
    scales[meta] = scale;
    means[meta] = __float2half_rn(__fdiv_rn(mean, scale));
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    int row = lane + j * 32, r = row % 16;
    // Upstream FP8 register-A packing needs this sequence permutation in V.
    int permuted = (row / 16) * 16 + (r / 8) * 2 + ((r / 2) % 4) * 4 + r % 2;
    payload[(int64_t(head) * dim + channel) * (tiles * 128) + tile * 128 + permuted]
        = __nv_cvt_float_to_fp8(__fdiv_rn(x[j], scale), __NV_SATFINITE, __NV_E4M3);
  }
}
__global__ void expcast_probe(const float* x, uint8_t* y, int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = vc_expcast_code(x[i]);
  }
}
}

std::vector<torch::Tensor> vc_prepare_values(torch::Tensor v) {
  TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.dim() == 4 && v.numel() > 0, "V requires contiguous nonempty CUDA BHND");
  TORCH_CHECK(v.scalar_type() == at::kHalf || v.scalar_type() == at::kBFloat16 || v.scalar_type() == at::kFloat, "V requires FP16/BF16/FP32");
  TORCH_CHECK(v.size(3) == 64 || v.size(3) == 128, "V dimension must be 64 or 128");
  TORCH_CHECK(v.numel() < INT32_MAX && v.size(0) * v.size(1) <= 65535, "V shape exceeds launch bounds");
  const c10::cuda::CUDAGuard guard(v.device());
  int n = v.size(2), dim = v.size(3), tiles = (n + 127) / 128;
  auto payload = torch::empty({v.size(0), v.size(1), dim, tiles * 128}, v.options().dtype(at::ScalarType::Float8_e4m3fn));
  auto scales = torch::empty({v.size(0), v.size(1), tiles, dim}, v.options().dtype(at::kFloat));
  auto means = torch::empty_like(scales, scales.options().dtype(at::kHalf));
  dim3 grid(tiles, dim / 8, v.size(0) * v.size(1));
  auto stream = c10::cuda::getCurrentCUDAStream(v.get_device()).stream();
  if (v.scalar_type() == at::kHalf)
    value_blocks<<<grid, 256, 0, stream>>>(reinterpret_cast<const half*>(v.data_ptr()), reinterpret_cast<uint8_t*>(payload.data_ptr()), scales.data_ptr<float>(), reinterpret_cast<half*>(means.data_ptr()), n, dim, tiles);
  else if (v.scalar_type() == at::kBFloat16)
    value_blocks<<<grid, 256, 0, stream>>>(reinterpret_cast<const nv_bfloat16*>(v.data_ptr()), reinterpret_cast<uint8_t*>(payload.data_ptr()), scales.data_ptr<float>(), reinterpret_cast<half*>(means.data_ptr()), n, dim, tiles);
  else
    value_blocks<<<grid, 256, 0, stream>>>(v.data_ptr<float>(), reinterpret_cast<uint8_t*>(payload.data_ptr()), scales.data_ptr<float>(), reinterpret_cast<half*>(means.data_ptr()), n, dim, tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {payload, scales, means};
}

torch::Tensor vc_expcast_codes(torch::Tensor scores) {
  TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == at::kFloat, "ExpCast requires CUDA FP32");
  const c10::cuda::CUDAGuard guard(scores.device());
  auto input = scores.contiguous();
  auto output = torch::empty_like(input, input.options().dtype(at::kByte));
  if (input.numel()) expcast_probe<<<(input.numel() + 255) / 256, 256, 0,
      c10::cuda::getCurrentCUDAStream(input.get_device()).stream()>>>(input.data_ptr<float>(), output.data_ptr<uint8_t>(), input.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
