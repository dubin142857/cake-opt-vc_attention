// Copyright 2026 vc-attention contributors. SPDX-License-Identifier: Apache-2.0
// Q64/K128 block E4M3, FP32 scales, FP32 channel centering of K. No Hadamard.
#include "vc_api.h"
#include <cuda_fp8.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <climits>

namespace {
template<typename T, int D>
__global__ void key_partial_mean(const T* k, float* partial, int n, int blocks) {
  const int channel = threadIdx.x % D, group = threadIdx.x / D;
  const int first = blockIdx.x * 128;
  float sum = 0.f;
  for (int row=group; row<128 && first+row<n; row+=256/D)
    sum += float(k[(int64_t(blockIdx.y)*n + first+row)*D + channel]);
  __shared__ float sums[256];
  sums[threadIdx.x] = sum;
  __syncthreads();
  if (threadIdx.x<D) {
    float total=0.f;
    #pragma unroll
    for (int j=0;j<256/D;++j) total += sums[j*D+channel];
    partial[(int64_t(blockIdx.y)*blocks + blockIdx.x)*D+channel] = total;
  }
}

template<int D>
__global__ void key_finish_mean(const float* partial, float* mean, int n, int blocks) {
  const int c=threadIdx.x;
  float sum=0.f;
  for (int tile=0;tile<blocks;++tile) sum += partial[(int64_t(blockIdx.x)*blocks+tile)*D+c];
  mean[int64_t(blockIdx.x)*D+c] = __fdiv_rn(sum,float(n));
}

template<typename T, int D, int Rows, bool Center>
__global__ void quantize_qk_fp8(const T* input, const float* mean, uint8_t* codes,
                               float* scales, int n, int blocks) {
  constexpr int Items=Rows*D/256;
  float values[Items], peak=0.f;
  const int first=blockIdx.x*Rows;
  const int64_t base=int64_t(blockIdx.y)*n*D;
  #pragma unroll
  for (int i=0;i<Items;++i) {
    const int pos=threadIdx.x+i*256, row=first+pos/D, c=pos%D;
    float x=0.f;
    if (row<n) {
      x=float(input[base+int64_t(row)*D+c]);
      if constexpr (Center) x=__fsub_rn(x,mean[int64_t(blockIdx.y)*D+c]);
    }
    values[i]=x;
    peak=fmaxf(peak,fabsf(x));
  }
  #pragma unroll
  for (int offset=16;offset>0;offset/=2) peak=fmaxf(peak,__shfl_xor_sync(0xffffffff,peak,offset));
  __shared__ float peaks[8];
  if ((threadIdx.x&31)==0) peaks[threadIdx.x/32]=peak;
  __syncthreads();
  peak=0.f;
  #pragma unroll
  for (int i=0;i<8;++i) peak=fmaxf(peak,peaks[i]);
  const float scale=fmaxf(__fdiv_rn(peak,448.f),1.e-12f);
  if (threadIdx.x==0) scales[int64_t(blockIdx.y)*blocks+blockIdx.x]=scale;
  #pragma unroll
  for (int i=0;i<Items;++i) {
    const int pos=threadIdx.x+i*256, row=first+pos/D;
    if (row<n) codes[base+int64_t(row)*D+pos%D]=
        __nv_cvt_float_to_fp8(__fdiv_rn(values[i],scale),__NV_SATFINITE,__NV_E4M3);
  }
}

template<typename T, int D>
void launch(torch::Tensor q, torch::Tensor k, torch::Tensor q8, torch::Tensor k8,
            torch::Tensor qs, torch::Tensor ks, torch::Tensor mean, torch::Tensor partial) {
  const int bh=q.size(0)*q.size(1), nq=q.size(2), nk=k.size(2), nb=(nk+127)/128, qb=(nq+63)/64;
  auto stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
  key_partial_mean<T,D><<<dim3(nb,bh),256,0,stream>>>(k.data_ptr<T>(),partial.data_ptr<float>(),nk,nb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  key_finish_mean<D><<<bh,D,0,stream>>>(partial.data_ptr<float>(),mean.data_ptr<float>(),nk,nb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  quantize_qk_fp8<T,D,64,false><<<dim3(qb,bh),256,0,stream>>>(q.data_ptr<T>(),nullptr,
      reinterpret_cast<uint8_t*>(q8.data_ptr()),qs.data_ptr<float>(),nq,qb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  quantize_qk_fp8<T,D,128,true><<<dim3(nb,bh),256,0,stream>>>(k.data_ptr<T>(),mean.data_ptr<float>(),
      reinterpret_cast<uint8_t*>(k8.data_ptr()),ks.data_ptr<float>(),nk,nb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}

std::vector<torch::Tensor> vc_prepare_qk_fp8(torch::Tensor q, torch::Tensor k) {
  TORCH_CHECK(q.is_cuda() && k.device()==q.device() && q.is_contiguous() && k.is_contiguous(),
      "FP8 QK preparation requires contiguous tensors on the same CUDA device");
  TORCH_CHECK(q.dim()==4 && k.dim()==4 && q.numel()>0 && k.numel()>0, "Expected nonempty BHND Q/K");
  TORCH_CHECK((q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16) && k.scalar_type()==q.scalar_type(),
      "Q/K require matching FP16/BF16 dtype");
  const int64_t b=q.size(0), h=q.size(1), d=q.size(3), nq=q.size(2), nk=k.size(2);
  TORCH_CHECK((d==64 || d==128) && k.size(0)==b && k.size(1)==h && k.size(3)==d,
      "Q/K require matching B/H/D; D=64/128; no GQA");
  TORCH_CHECK(q.numel()<INT32_MAX && k.numel()<INT32_MAX && b*h<=65535, "FP8 QK shape exceeds kernel bounds");
  const c10::cuda::CUDAGuard guard(q.device());
  const auto fp8=q.options().dtype(at::ScalarType::Float8_e4m3fn), fp32=q.options().dtype(at::kFloat);
  auto q8=torch::empty(q.sizes(),fp8), k8=torch::empty(k.sizes(),fp8);
  auto qs=torch::empty({b,h,(nq+63)/64},fp32), ks=torch::empty({b,h,(nk+127)/128},fp32);
  auto mean=torch::empty({b,h,d},fp32), partial=torch::empty({b,h,(nk+127)/128,d},fp32);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(),"vc_prepare_qk_fp8",[&] {
    if (d==64) launch<scalar_t,64>(q,k,q8,k8,qs,ks,mean,partial);
    else launch<scalar_t,128>(q,k,q8,k8,qs,ks,mean,partial);
  });
  return {q8,qs,k8,ks,mean};
}
