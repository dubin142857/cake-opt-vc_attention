// Fused Lloyd assignment/statistics for VC grouping. Apache-2.0; see NOTICE.
#include "vc_api.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <climits>

namespace {
constexpr int kThreads = 128, kRows = 128, kClusters = 8;
__device__ __forceinline__ float as_float(float x) { return x; }
__device__ __forceinline__ float as_float(half x) { return __half2float(x); }
__device__ __forceinline__ float as_float(nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float sum_four(float x) {
  x = __fadd_rn(x, __shfl_xor_sync(0xffffffff, x, 1, 4));
  return __fadd_rn(x, __shfl_xor_sync(0xffffffff, x, 2, 4));
}

template<typename T, int Dim>
__global__ void initialize_centers(const T* values, const int64_t* ids, float* centers, int n, int count) {
  int head = blockIdx.x;
  for (int i = threadIdx.x; i < count * Dim; i += blockDim.x)
    centers[head * count * Dim + i] = as_float(values[(int64_t(head) * n + ids[i / Dim]) * Dim + i % Dim]);
}

// Four lanes per row. Read each V row once; reuse registers for all centers.
// Shared-memory atomics combine only this tile, then a separate kernel merges
// the much smaller partials. Like legacy scatter_add, FP32 summation order is
// not a bitwise contract; ties in the computed distances select the lowest id.
template<typename T, int Dim, bool Statistics>
__global__ void assign_values(const T* values, const float* centers, int* labels,
    float* partial_sums, int* partial_counts, int n, int count, int tiles) {
  __shared__ float center[kClusters][Dim];
  __shared__ float center_norm[kClusters];
  // Padding distributes clusters across banks for a four-lane row group.
  __shared__ float sums[Statistics ? kClusters : 1][Statistics ? Dim + 4 : 1];
  __shared__ int counts[Statistics ? kClusters : 1];
  const int head = blockIdx.y, tile = blockIdx.x;
  const int lane = threadIdx.x % 4, group = threadIdx.x / 4;
  for (int i = threadIdx.x; i < count * Dim; i += kThreads) {
    center[i / Dim][i % Dim] = centers[head * count * Dim + i];
    if constexpr (Statistics) sums[i / Dim][i % Dim] = 0;
  }
  if constexpr (Statistics) if (threadIdx.x < count) counts[threadIdx.x] = 0;
  __syncthreads();
  // All lanes participate, including the unused center groups.
  float cn = 0;
  if (group < count) {
    #pragma unroll
    for (int j = 0; j < Dim / 4; ++j) {
      float x = center[group][j * 4 + lane];
      cn = __fadd_rn(cn, __fmul_rn(x, x));
    }
  }
  cn = sum_four(cn);
  if (group < count && lane == 0) center_norm[group] = cn;
  __syncthreads();

  #pragma unroll
  for (int offset = 0; offset < kRows; offset += kThreads / 4) {
    int row = tile * kRows + offset + group;
    float x[Dim / 4], norm = 0;
    #pragma unroll
    for (int j = 0; j < Dim / 4; ++j) {
      x[j] = row < n ? as_float(values[(int64_t(head) * n + row) * Dim + j * 4 + lane]) : 0;
      norm = __fadd_rn(norm, __fmul_rn(x[j], x[j]));
    }
    norm = sum_four(norm);
    float best_distance = CUDART_INF_F;
    int best = 0;
    for (int c = 0; c < count; ++c) {
      float dot = 0;
      #pragma unroll
      for (int j = 0; j < Dim / 4; ++j)
        dot = __fmaf_rn(x[j], center[c][j * 4 + lane], dot);
      dot = sum_four(dot);
      float distance = __fsub_rn(__fadd_rn(norm, center_norm[c]), __fmul_rn(2.f, dot));
      if (distance < best_distance) { best_distance = distance; best = c; }
    }
    // One lane chooses the label; all four must accumulate into that cluster
    // even if their equivalent reduction trees round differently.
    best = __shfl_sync(0xffffffff, best, 0, 4);
    if (row < n) {
      if (lane == 0) labels[head * n + row] = best;
      if constexpr (Statistics) {
        if (lane == 0) atomicAdd(counts + best, 1);
        #pragma unroll
        for (int j = 0; j < Dim / 4; ++j) atomicAdd(&sums[best][j * 4 + lane], x[j]);
      }
    }
  }
  if constexpr (Statistics) {
    __syncthreads();
    for (int i = threadIdx.x; i < count * Dim; i += kThreads)
      partial_sums[((int64_t(head) * tiles + tile) * count) * Dim + i] = sums[i / Dim][i % Dim];
    if (threadIdx.x < count)
      partial_counts[(head * tiles + tile) * count + threadIdx.x] = counts[threadIdx.x];
  }
}

template<int Dim>
__global__ void update_centers(float* centers, const float* sums, const int* counts, int tiles, int count) {
  const int c = blockIdx.x, head = blockIdx.y, channel = threadIdx.x;
  float sum = 0;
  int total = 0;
  for (int tile = 0; tile < tiles; ++tile) {
    int64_t index = (int64_t(head) * tiles + tile) * count + c;
    sum = __fadd_rn(sum, sums[index * Dim + channel]);
    total += counts[index];
  }
  // Empty centers retain their previous value, exactly as the legacy policy.
  if (total) centers[(head * count + c) * Dim + channel] = __fdiv_rn(sum, float(total));
}

template<typename T, int Dim>
void run_grouping(torch::Tensor v, torch::Tensor ids, torch::Tensor centers, torch::Tensor labels,
    torch::Tensor sums, torch::Tensor counts, int count, int iterations, cudaStream_t stream) {
  int n = v.size(2), heads = v.size(0) * v.size(1), tiles = (n + kRows - 1) / kRows;
  const T* values = reinterpret_cast<const T*>(v.data_ptr());
  initialize_centers<T,Dim><<<heads,kThreads,0,stream>>>(values,ids.data_ptr<int64_t>(),centers.data_ptr<float>(),n,count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  for (int iter = 0; iter < iterations; ++iter) {
    if (iter + 1 < iterations) {
      assign_values<T,Dim,true><<<dim3(tiles,heads),kThreads,0,stream>>>(values,centers.data_ptr<float>(),labels.data_ptr<int>(),
          sums.data_ptr<float>(),counts.data_ptr<int>(),n,count,tiles);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      update_centers<Dim><<<dim3(count,heads),Dim,0,stream>>>(centers.data_ptr<float>(),sums.data_ptr<float>(),counts.data_ptr<int>(),tiles,count);
    } else {
      // The public result is the labels from this assignment. No later
      // iteration consumes updated centers, so skip the unused statistics.
      assign_values<T,Dim,false><<<dim3(tiles,heads),kThreads,0,stream>>>(values,centers.data_ptr<float>(),labels.data_ptr<int>(),nullptr,nullptr,n,count,tiles);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
}

std::vector<torch::Tensor> vc_group_values_fused(torch::Tensor v, int64_t clusters, int64_t iterations) {
  TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.dim() == 4 && v.numel() > 0, "Grouping requires contiguous nonempty CUDA BHND");
  TORCH_CHECK(v.scalar_type() == at::kHalf || v.scalar_type() == at::kBFloat16 || v.scalar_type() == at::kFloat, "Grouping requires FP16/BF16/FP32");
  TORCH_CHECK(v.size(3) == 64 || v.size(3) == 128, "Grouping requires D64/128");
  TORCH_CHECK(clusters > 0 && clusters <= 256 && std::min(clusters,v.size(2)) <= kClusters && iterations > 0 && iterations <= 100, "Fused grouping requires 1..8 effective clusters (requested <=256) and 1..100 iterations");
  TORCH_CHECK(v.numel() < INT32_MAX && v.size(0) * v.size(1) <= 65535, "Grouping shape exceeds launch bounds");
  const c10::cuda::CUDAGuard guard(v.device());
  const at::NoGradGuard no_grad;
  const int count = std::min(clusters,v.size(2)), n = v.size(2), tiles = (n + kRows - 1) / kRows;
  auto fp32 = v.options().dtype(at::kFloat), ints = v.options().dtype(at::kInt);
  // Preserve ATen's FP32 linspace/truncation initialization, including its
  // endpoint rounding; do not upload CPU-generated ids or convert all V to FP32.
  auto ids = at::linspace(0,n - 1,count,fp32).to(at::kLong);
  auto centers = torch::empty({v.size(0),v.size(1),count,v.size(3)},fp32);
  auto labels = torch::empty({v.size(0),v.size(1),n},ints);
  auto sums = iterations > 1 ? torch::empty({v.size(0),v.size(1),tiles,count,v.size(3)},fp32) : torch::empty({0},fp32);
  auto counts = iterations > 1 ? torch::empty({v.size(0),v.size(1),tiles,count},ints) : torch::empty({0},ints);
  auto stream = c10::cuda::getCurrentCUDAStream(v.get_device()).stream();
  #define RUN(T) if (v.size(3) == 64) run_grouping<T,64>(v,ids,centers,labels,sums,counts,count,iterations,stream); \
                 else run_grouping<T,128>(v,ids,centers,labels,sums,counts,count,iterations,stream)
  if (v.scalar_type() == at::kHalf) { RUN(half); }
  else if (v.scalar_type() == at::kBFloat16) { RUN(nv_bfloat16); }
  else { RUN(float); }
  #undef RUN
  // Stable sorting retains token order inside each cluster. Centers are those
  // used by the final assignment, useful for untimed nearest-center checks.
  return {at::argsort(labels,true,-1,false), labels, centers};
}
