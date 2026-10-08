// Copyright (c) 2026 vc-attention contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

// Called by all 256 CTA threads. Each branch is uniform within a warpgroup.
// Barrier 1 hands the current K tile from WG0 to WG1. Barrier 0, reserved
// for __syncthreads(), joins both completed PV operations before V reuse.
// No thread may return early, including an entirely padded second Q64 block.
template<bool Stagger>
__device__ __forceinline__ void dual_query_before_qk(unsigned group) {
  if constexpr (Stagger) {
    if (group == 1) asm volatile("bar.sync 1, 256;" ::: "memory");
  }
}

// Caller must first complete its own asynchronous QK with wgmma.wait_group 0.
// In staggered mode WG0 releases WG1 and proceeds to probability/PV. Only
// WG1's leader may refill K, after its own QK has also completed.
template<bool Stagger>
__device__ __forceinline__ void dual_query_after_qk(unsigned group) {
  if constexpr (Stagger) {
    if (group == 0) asm volatile("bar.arrive 1, 256;" ::: "memory");
  } else {
    __syncthreads();
  }
}

// Each caller has completed PV before arriving. Also ensures barrier 1's
// previous generation was consumed before WG0 can signal the next tile.
__device__ __forceinline__ void dual_query_after_pv() {
  __syncthreads();
}

__host__ __device__ constexpr unsigned dual_query_scale_index(
    unsigned batch, unsigned head, unsigned heads, unsigned queries,
    unsigned logical_q_block, unsigned local_thread) {
  return ((batch * heads + head) * ((queries + 63) / 64) + logical_q_block) * 32
      + (local_thread / 32) * 8 + (local_thread % 32) / 4;
}
