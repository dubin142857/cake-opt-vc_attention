// Copyright (c) 2026 vc-attention contributors. SPDX-License-Identifier: Apache-2.0
// Exercises production CTA handoffs and Q64 addressing on any CUDA GPU.
// This deliberately does not claim to validate Hopper TMA/WGMMA behavior.
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "../../third_party/sageattention_local/csrc/vc/dual_query_sync.cuh"

__device__ __forceinline__ void jitter(unsigned cycles) {
  auto start=clock64();
  while (clock64()-start < cycles) {}
}

template<bool Stagger>
__global__ void dual_probe(int* out,const int* scales,int queries,int heads,int steps,int delay) {
  __shared__ volatile int key[128],value[128];
  const unsigned tid=threadIdx.x,group=tid/128,lane=tid%128;
  const unsigned qblock=blockIdx.x*2+group;
  // Whole row uniquely identifies batch/head/CTA/lane in a host oracle.
  const unsigned row=((blockIdx.z*heads+blockIdx.y)*gridDim.x+blockIdx.x)*256+tid;
  const unsigned scale_index=dual_query_scale_index(blockIdx.z,blockIdx.y,heads,queries,qblock,lane);
  out[row*(2*steps+1)] = qblock < (queries+63)/64 ? scales[scale_index] : -1;
  if(tid<128) {key[tid]=int(tid);value[tid]=int(tid)+17;}
  __syncthreads();
  for(int i=0;i<steps;++i) {
    dual_query_before_qk<Stagger>(group);
    jitter(((tid+i)%7)*delay);
    int k=key[(lane*3+group)%128];
    // Scalar loads lack WGMMA's collective completion guarantee. Model the
    // caller's wait_group 0: all four warps finished this group's K reads.
    if(group==0) asm volatile("bar.sync 2, 128;" ::: "memory");
    else asm volatile("bar.sync 3, 128;" ::: "memory");
    dual_query_after_qk<Stagger>(group);
    if(tid==128 && i+1<steps)
      for(int j=0;j<128;++j) key[j]=(i+1)*257+j;
    // Surrogate probability/PV work, intentionally skewed between groups.
    jitter(((255-tid+i)%11)*delay);
    int v=value[(lane*5+group)%128];
    out[row*(2*steps+1)+2*i+1]=k;
    out[row*(2*steps+1)+2*i+2]=v;
    dual_query_after_pv();
    if(tid==0 && i+1<steps)
      for(int j=0;j<128;++j) value[j]=(i+1)*509+j+17;
    // Real kernel has TMA mbarrier waits for the refill. Here ordinary
    // shared writes need a publication barrier before the next iteration.
    __syncthreads();
  }
}

torch::Tensor run(torch::Tensor scales,int queries,int steps,int delay,bool stagger) {
  TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.scalar_type()==at::kInt);
  TORCH_CHECK(scales.dim()==3 && queries>0 && steps>0 && steps<=300 && delay>=0 && delay<=1000);
  TORCH_CHECK(scales.size(2)==((queries+63)/64)*32);
  c10::cuda::CUDAGuard guard(scales.device());
  auto out=torch::empty({scales.size(0),scales.size(1),(queries+127)/128,256,2*steps+1},scales.options());
  dim3 grid((queries+127)/128,scales.size(1),scales.size(0));
  auto stream=c10::cuda::getCurrentCUDAStream(scales.get_device()).stream();
  if(stagger) dual_probe<true><<<grid,256,0,stream>>>(out.data_ptr<int>(),scales.data_ptr<int>(),queries,scales.size(1),steps,delay);
  else dual_probe<false><<<grid,256,0,stream>>>(out.data_ptr<int>(),scales.data_ptr<int>(),queries,scales.size(1),steps,delay);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);}
