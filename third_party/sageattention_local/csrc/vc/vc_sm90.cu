// Modified from SageAttention v2.2.0 Hopper kernel: VC ExpCast and block V mean/scale restoration.
/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../utils.cuh"
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <torch/extension.h>

#include "../wgmma.cuh"
#include "../math.cuh"
#include "../dispatch_utils.h"

#include "../qattn/attn_utils.cuh"
#include "vc_api.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cfloat>
#include <climits>
#include <cmath>

namespace sage_vc {
#include "vc_math.cuh"
#include "probability_variants.cuh"
#include "work_reduction.cuh"
#include "max_state.cuh"

template <int BlockMajorSize, int BlockMinorSize, bool swizzle=true, CUtensorMapL2promotion_enum promotion_mode=CU_TENSOR_MAP_L2_PROMOTION_NONE, typename T>
CUtensorMap create_tensor_map_4D(T* gmem_ptr, int d1, int d2, int d3, int d4, int stride1, int stride2, int stride3) {
    constexpr int smem_stride = BlockMinorSize * sizeof(T);
    static_assert(sizeof(T) == 2 || sizeof(T) == 1);
    static_assert(smem_stride == 32 || smem_stride == 64 || smem_stride == 128);

    CUtensorMap tma_map;
    void* gmem_address = (void*)gmem_ptr;
    uint64_t gmem_prob_shape[5] = {(uint64_t)d4, (uint64_t)d3, (uint64_t)d2, (uint64_t)d1, 1};
    uint64_t gmem_prob_stride[5] = {(uint64_t)stride3 * sizeof(T), (uint64_t)stride2 * sizeof(T), (uint64_t)stride1 * sizeof(T), 0, 0};
    uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map, (sizeof(T) == 2) ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, gmem_address, gmem_prob_shape,
        gmem_prob_stride, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        (swizzle == false) ? CU_TENSOR_MAP_SWIZZLE_NONE : (smem_stride == 128) ? CU_TENSOR_MAP_SWIZZLE_128B : (smem_stride == 64) ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_32B,
        promotion_mode, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    TORCH_CHECK(result == CUDA_SUCCESS, "Sage VC TMA descriptor failed: ", int(result));

    return tma_map;
}

__device__ __forceinline__ void init_barrier(uint64_t* bar, int thread_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_ptr), "r"(thread_count)
    );
}

template <uint32_t bytes>
__device__ __forceinline__ void expect_bytes(uint64_t* bar) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_ptr), "n"(bytes));
}

template <typename T>
__device__ __forceinline__ void load_async_4D(T *dst, void const* const src_tma_map, uint64_t* bar, int s0, int s1, int s2, int s3) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    asm volatile (
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
        "r"(s0), "r"(s1), "r"(s2), "r"(s3)
        : "memory"
    );
}

template <typename T>
__device__ __forceinline__ void store_async_4D(void const* dst_tma_map, T *src, int global_token_idx, int global_head_idx, int global_batch_idx) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(src));

    asm volatile (
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%2, %3, %4, %5}], [%1];"
        :
        : "l"(tma_ptr), "r"(src_ptr),
        "n"(0), "r"(global_token_idx), "r"(global_head_idx), "r"(global_batch_idx)
        : "memory"
    );
}

__device__ __forceinline__ void wait(uint64_t* bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

template <uint32_t count = 1>
__device__ __forceinline__ void arrive(uint64_t* bar) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "n"(count)
        : "memory"
    );
}

template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t NUM_THREADS, uint32_t head_dim, QuantGranularity Q_GRAN, QuantGranularity K_GRAN, typename DTypeOut, MaskMode mask_mode = MaskMode::kNone, bool return_lse = false, bool fuse_v_scale=false, bool ExpCast=false, bool BlockValues=false, bool Optimized=false, bool ChannelScale=false, bool PackedProbability=false, bool CompactValues=false, bool PipelineProbability=false, bool InterleavedProbability=false, bool FoldAffine=false, int ProbabilitySchedule=0, bool PairedEncode=false, bool FusedPack=false>
__device__ __forceinline__ void sage_vc_attention_body(const CUtensorMap& tensorMapQ,
                                        const CUtensorMap& tensorMapK,
                                        const CUtensorMap& tensorMapV,
                                        float *__restrict__ Q_scale, float *__restrict__ K_scale, float *__restrict__ V_scale, const half* __restrict__ V_mean,
                                        DTypeOut* O, float *__restrict__ Lse, uint32_t stride_bz_o, uint32_t stride_h_o, uint32_t stride_seq_o,
                                        const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
                                        float sm_scale)
{
#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("trap;");
#else
  static_assert(NUM_THREADS == 128);
  static_assert(!CompactValues || (head_dim == 128 && BlockValues && Optimized && ChannelScale));
  static_assert(!FusedPack || PairedEncode);
  static_assert(!PairedEncode || (PipelineProbability && FoldAffine && ProbabilitySchedule == 0));
  static_assert(ProbabilitySchedule >= 0 && ProbabilitySchedule <= 2);
  static_assert(ProbabilitySchedule == 0 || (PipelineProbability && FoldAffine));
  static_assert(!PipelineProbability || (ExpCast && !BlockValues && Optimized && PackedProbability));
  static_assert(CTA_Q <= CTA_K);

  const uint32_t warp_idx = (threadIdx.x % 128) / 32;
  const uint32_t lane_id = threadIdx.x % 32;

  constexpr uint32_t num_tiles_q = CTA_Q / 64;
  constexpr uint32_t num_tiles_k = CTA_K / 16;
  constexpr uint32_t num_tiles_qk_inner = head_dim / 32;
  constexpr uint32_t num_tiles_v = head_dim / 16;
  constexpr uint32_t num_tiles_pv_inner = CTA_K / 32;

  const uint32_t batch_id = blockIdx.z;
  const uint32_t bx = blockIdx.x;
  const uint32_t head_id = blockIdx.y;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t kv_head_id = head_id / num_kv_groups;

  if constexpr (!ExpCast) sm_scale *= math::log2e;

  extern __shared__ __align__(128) int8_t smem_[];

  int8_t *sQ = (int8_t*)smem_;
  int8_t *sK = (int8_t*)(smem_ + CTA_Q * head_dim * sizeof(int8_t));
  int8_t *sV = (int8_t*)(smem_ + CTA_Q * head_dim * sizeof(int8_t) + CTA_K * head_dim * sizeof(int8_t));
  half *sO = (half*)smem_;
  uint32_t* probability_scratch = reinterpret_cast<uint32_t*>(smem_ + (CTA_Q + 2 * CTA_K) * head_dim);

  int32_t RS[num_tiles_q][num_tiles_k][8];
  float RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2];
  float d[num_tiles_q][2];

  uint32_t q_scale_idx, k_scale_idx;

  if constexpr (Q_GRAN == QuantGranularity::kPerBlock)
  {
    const uint32_t num_block_q = gridDim.x;
    q_scale_idx = batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx;
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_warp_block_q = gridDim.x * 4;
    q_scale_idx = batch_id * num_qo_heads * num_warp_block_q + head_id * num_warp_block_q + bx * 4 + warp_idx;
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_warp_block_q = gridDim.x * 4;
    q_scale_idx = batch_id * num_qo_heads * (num_warp_block_q * 8) + head_id * (num_warp_block_q * 8) + bx * (4 * 8) + warp_idx * 8 + lane_id / 4;
  }

  if constexpr (K_GRAN == QuantGranularity::kPerBlock || K_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_block_k + (head_id / num_kv_groups) * num_block_k;
  }
  else if constexpr (K_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * (num_block_k * 4) + (head_id / num_kv_groups) * (num_block_k * 4) + lane_id % 4;
  }

  constexpr uint32_t k_scale_advance_offset = (K_GRAN == QuantGranularity::kPerBlock || K_GRAN == QuantGranularity::kPerWarp) ? 1 : 4;

  uint32_t Q_idx_lane_base = bx * CTA_Q + warp_idx * 16 + lane_id / 4;

#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
    m[fq][0] = -5000000.0f;
    m[fq][1] = -5000000.0f;
    d[fq][0] = 1.0f;
    d[fq][1] = 1.0f;
  }

#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++)
      {
        RO[fq][fv][k] = 0.0f;
      }
    }
  }

  __shared__ __align__(8) uint64_t barrier_Q;
  __shared__ __align__(8) uint64_t barrier_K;
  __shared__ __align__(8) uint64_t barrier_V;

  if (threadIdx.x == 0) {
    init_barrier(&barrier_Q, 1);
    init_barrier(&barrier_K, 1);
    init_barrier(&barrier_V, 1);
  }

  __syncthreads();

  // load Q, K, V
  if (threadIdx.x == 0)
  {
    expect_bytes<(CTA_Q * head_dim) * sizeof(int8_t)>(&barrier_Q);
    expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_K);
    expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_V);
    load_async_4D(sQ, &tensorMapQ, &barrier_Q, 0, bx * CTA_Q, head_id, batch_id);
    load_async_4D(sK, &tensorMapK, &barrier_K, 0, 0, kv_head_id, batch_id);
    load_async_4D(sV, &tensorMapV, &barrier_V, 0, 0, kv_head_id, batch_id);
  }

  float q_scale = Q_scale[q_scale_idx];
  float original_sm_scale = sm_scale;

  // wait for Q
  wait(&barrier_Q, 0);

  const uint32_t num_iterations = div_ceil(
      mask_mode == MaskMode::kCausal
          ? min(kv_len, (bx + 1) * CTA_Q)
          : kv_len,
      CTA_K);

  int p = 1;
  for (uint32_t iter = 1; iter < num_iterations; iter++)
  {
    p ^= 1;

    float dequant_scale = q_scale * K_scale[k_scale_idx + (iter - 1) * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;

    // wait for K
    wait(&barrier_K, p);

    // compute QK^T
    wgmma::warpgroup_arrive();
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      int8_t *sQ_local = sQ + fq * 64 * head_dim;
      wgmma::wgmma_s8s8s32<CTA_K, 0, head_dim>(RS[fq], sQ_local, sK);
#pragma unroll
      for (int k_it = 1; k_it < num_tiles_qk_inner; k_it++)
      {
        wgmma::wgmma_s8s8s32<CTA_K, 1, head_dim>(RS[fq], &sQ_local[k_it*32], &sK[k_it*32]);
      }
    }
    wgmma::warpgroup_commit_batch();
    wgmma::warpgroup_wait<0>();

    // load K
    if (threadIdx.x == 0)
    {
      expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_K);
      load_async_4D(sK, &tensorMapK, &barrier_K, 0, iter * CTA_K, kv_head_id, batch_id);
    }

    // convert RS to float
    float RS_f32[num_tiles_q][num_tiles_k][8];
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
        }
      }
    }

    uint32_t RS_f8[num_tiles_q][num_tiles_pv_inner][4];
    float tile_mass[num_tiles_q][2], probability_alpha[num_tiles_q][2];
    if constexpr (PipelineProbability)
      vc_probability_prepare<InterleavedProbability,FoldAffine,(ProbabilitySchedule != 0),PairedEncode,FusedPack>(RS_f32, m, sm_scale, RS_f8, probability_alpha, tile_mass);
    else vc_probability<ExpCast, BlockValues, Optimized, PackedProbability>(RS_f32, RO, m, d, sm_scale, RS_f8, tile_mass);

    if constexpr(ProbabilitySchedule != 0) {
      vc_probability_store(RS_f8, probability_scratch);
      if constexpr(ProbabilitySchedule == 2)
        vc_probability_load_mass(probability_scratch, tile_mass);
    }

    // wait for V
    wait(&barrier_V, p);

    // D128 channel smoothing: consume one 64-channel PV fragment at a time.
    // This halves the short-lived WGMMA accumulator, retaining the full FP32
    // output history and the original four K32 operations per output element.
    constexpr uint32_t pv_parts = CompactValues ? 2 : 1;
    #pragma unroll
    for (uint32_t part = 0; part < pv_parts; ++part) {
      float RO_temp[num_tiles_q][num_tiles_v / pv_parts][8];
      // Keep Sage's two-level FP32 accumulation: FP8 WGMMA must not
      // repeatedly round the long-lived output history or mean compensation.
      if constexpr(ProbabilitySchedule == 2) {
        // Controlled serial baseline: force BOTH row sums before PV starts.
        // The mass predicate is uniformly true (same bound as at the wait).
        asm volatile("{ .reg .pred p;\n"
                     "setp.ge.f32 p, %0, 0f00000000;\n"
                     "setp.ge.and.f32 p, %1, 0f00000000, p;\n"
                     "@!p trap;\n"
                     "@p wgmma.fence.sync.aligned; }"
                     :: "f"(tile_mass[0][0]), "f"(tile_mass[0][1]) : "memory");
      } else wgmma::warpgroup_arrive();
  #pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        wgmma::wgmma_f8f8f32<head_dim / pv_parts, 0, CTA_K>(RO_temp[fq], RS_f8[fq][0], &sV[part * (head_dim / pv_parts) * CTA_K]);
  #pragma unroll
        for (uint32_t v_it = 1; v_it < num_tiles_pv_inner; v_it++)
        {
          wgmma::wgmma_f8f8f32<head_dim / pv_parts, 1, CTA_K>(RO_temp[fq], RS_f8[fq][v_it], &sV[part * (head_dim / pv_parts) * CTA_K + v_it * 32]);
        }
      }

      wgmma::warpgroup_commit_batch();
      if constexpr(ProbabilitySchedule == 1)
        vc_probability_load_mass(probability_scratch, tile_mass);
      if constexpr (PipelineProbability)
        vc_probability_finish(RO, d, probability_alpha, tile_mass);
      if constexpr(ProbabilitySchedule != 0) {
        static_assert(num_tiles_q == 1);
        // Every lane's mass is a finite nonnegative sum of 32 E4M3
        // values in [0,256], including fully masked (zero-mass) tiles.
        // Thus the wait predicate is true for the WHOLE warpgroup.
        // This data dependency stops ptxas moving the sum past the wait
        // to reuse P's registers. Invalid mass traps instead of skipping
        // the wait and accessing unfinished accumulators.
        asm volatile("{ .reg .pred p;\n"
                     "setp.ge.f32 p, %0, 0f00000000;\n"
                     "setp.ge.and.f32 p, %1, 0f00000000, p;\n"
                     "@!p trap;\n"
                     "@p wgmma.wait_group.sync.aligned 0; }"
                     :: "f"(tile_mass[0][0]), "f"(tile_mass[0][1]) : "memory");
      } else wgmma::warpgroup_wait<0>();

      // Start the next V transfer only after the last fragment has finished
      // reading sV. Overlap that transfer with the remaining mean restoration.
      // per-block mean/scale restoration, which only touches registers/global
      // metadata. Keep the existing barrier/phase for the next iteration.
      if (threadIdx.x == 0 && part + 1 == pv_parts)
      {
        expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_V);
        load_async_4D(sV, &tensorMapV, &barrier_V, iter * CTA_K, 0, kv_head_id, batch_id);
      }

      vc_accumulate<BlockValues, Optimized, ChannelScale, CompactValues>(RO, RO_temp, tile_mass, V_scale, V_mean,
          batch_id * num_qo_heads + head_id, iter - 1, div_ceil(kv_len, CTA_K), head_dim, lane_id, part * (num_tiles_v / pv_parts));
    }
  }

  {
    p ^= 1;

    float dequant_scale = q_scale * K_scale[k_scale_idx + (num_iterations - 1) * k_scale_advance_offset];
    sm_scale = original_sm_scale;

    // wait for K
    wait(&barrier_K, p);

    // compute QK^T
    wgmma::warpgroup_arrive();
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      int8_t *sQ_local = sQ + fq * 64 * head_dim;
      wgmma::wgmma_s8s8s32<CTA_K, 0, head_dim>(RS[fq], sQ_local, sK);
#pragma unroll
      for (int k_it = 1; k_it < num_tiles_qk_inner; k_it++)
      {
        wgmma::wgmma_s8s8s32<CTA_K, 1, head_dim>(RS[fq], &sQ_local[k_it*32], &sK[k_it*32]);
      }
    }
    wgmma::warpgroup_commit_batch();
    wgmma::warpgroup_wait<0>();

    // convert RS to float
    float RS_f32[num_tiles_q][num_tiles_k][8];
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
        }
      }
    }

    // masking
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          const uint32_t q_idx = Q_idx_lane_base + fq * 64 + 8 * ((k % 4) / 2);
          const uint32_t k_idx = (num_iterations - 1) * CTA_K + fk * 16 + 2 * (lane_id % 4) + 8 * (k / 4) + k % 2;

          bool is_out_of_bounds;

          if constexpr (mask_mode == MaskMode::kCausal)
          {
            is_out_of_bounds = (k_idx > q_idx) || (k_idx >= kv_len);
          }
          else
          {
            is_out_of_bounds = (k_idx >= kv_len);
          }

          if (is_out_of_bounds)
          {
            RS_f32[fq][fk][k] = -CUDART_INF_F;
          }
        }
      }
    }

    uint32_t RS_f8[num_tiles_q][num_tiles_pv_inner][4];
    float tile_mass[num_tiles_q][2], probability_alpha[num_tiles_q][2];
    if constexpr (PipelineProbability)
      vc_probability_prepare<InterleavedProbability,FoldAffine,(ProbabilitySchedule != 0),PairedEncode,FusedPack>(RS_f32, m, sm_scale, RS_f8, probability_alpha, tile_mass);
    else vc_probability<ExpCast, BlockValues, Optimized, PackedProbability>(RS_f32, RO, m, d, sm_scale, RS_f8, tile_mass);

    if constexpr(ProbabilitySchedule != 0) {
      vc_probability_store(RS_f8, probability_scratch);
      if constexpr(ProbabilitySchedule == 2)
        vc_probability_load_mass(probability_scratch, tile_mass);
    }

    // wait for V
    wait(&barrier_V, p);

    // D128 channel smoothing: consume one 64-channel PV fragment at a time.
    // This halves the short-lived WGMMA accumulator, retaining the full FP32
    // output history and the original four K32 operations per output element.
    constexpr uint32_t pv_parts = CompactValues ? 2 : 1;
    #pragma unroll
    for (uint32_t part = 0; part < pv_parts; ++part) {
      float RO_temp[num_tiles_q][num_tiles_v / pv_parts][8];
      // Keep Sage's two-level FP32 accumulation: FP8 WGMMA must not
      // repeatedly round the long-lived output history or mean compensation.
      if constexpr(ProbabilitySchedule == 2) {
        asm volatile("{ .reg .pred p;\n"
                     "setp.ge.f32 p, %0, 0f00000000;\n"
                     "setp.ge.and.f32 p, %1, 0f00000000, p;\n"
                     "@!p trap;\n"
                     "@p wgmma.fence.sync.aligned; }"
                     :: "f"(tile_mass[0][0]), "f"(tile_mass[0][1]) : "memory");
      } else wgmma::warpgroup_arrive();
  #pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        wgmma::wgmma_f8f8f32<head_dim / pv_parts, 0, CTA_K>(RO_temp[fq], RS_f8[fq][0], &sV[part * (head_dim / pv_parts) * CTA_K]);
  #pragma unroll
        for (uint32_t v_it = 1; v_it < num_tiles_pv_inner; v_it++)
        {
          wgmma::wgmma_f8f8f32<head_dim / pv_parts, 1, CTA_K>(RO_temp[fq], RS_f8[fq][v_it], &sV[part * (head_dim / pv_parts) * CTA_K + v_it * 32]);
        }
      }

      wgmma::warpgroup_commit_batch();
      if constexpr(ProbabilitySchedule == 1)
        vc_probability_load_mass(probability_scratch, tile_mass);
      if constexpr (PipelineProbability)
        vc_probability_finish(RO, d, probability_alpha, tile_mass);
      if constexpr(ProbabilitySchedule != 0) {
        static_assert(num_tiles_q == 1);
        // Same uniform mass-dependent wait as the main K-tile loop.
        asm volatile("{ .reg .pred p;\n"
                     "setp.ge.f32 p, %0, 0f00000000;\n"
                     "setp.ge.and.f32 p, %1, 0f00000000, p;\n"
                     "@!p trap;\n"
                     "@p wgmma.wait_group.sync.aligned 0; }"
                     :: "f"(tile_mass[0][0]), "f"(tile_mass[0][1]) : "memory");
      } else wgmma::warpgroup_wait<0>();

      vc_accumulate<BlockValues, Optimized, ChannelScale, CompactValues>(RO, RO_temp, tile_mass, V_scale, V_mean,
          batch_id * num_qo_heads + head_id, num_iterations - 1, div_ceil(kv_len, CTA_K), head_dim, lane_id, part * (num_tiles_v / pv_parts));
    }
  }

  normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kCudaCore>(RO, m, d);

  if constexpr (fuse_v_scale && (!BlockValues || (ChannelScale && Optimized)))
  {
    float v_scale[4];
    float *V_scale_base_ptr = V_scale +  batch_id * (num_qo_heads / num_kv_groups) * head_dim + (head_id / num_kv_groups) * head_dim + (lane_id % 4 ) * 2;
  #pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      ((float2*)v_scale)[0] = *((float2*)(V_scale_base_ptr + fv * 16));
      ((float2*)v_scale)[1] = *((float2*)(V_scale_base_ptr + fv * 16 + 8));

  #pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        RO[fq][fv][0] *= v_scale[0];
        RO[fq][fv][1] *= v_scale[1];
        RO[fq][fv][2] *= v_scale[0];
        RO[fq][fv][3] *= v_scale[1];
        RO[fq][fv][4] *= v_scale[2];
        RO[fq][fv][5] *= v_scale[3];
        RO[fq][fv][6] *= v_scale[2];
        RO[fq][fv][7] *= v_scale[3];
      }
    }
  }

  DTypeOut *O_lane_ptr = O + batch_id * stride_bz_o + head_id * stride_h_o + (bx * CTA_Q + warp_idx * 16 + (lane_id / 4)) * stride_seq_o + (lane_id % 4) * 2 ;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < head_dim/16; fv++)
    {
      if (Q_idx_lane_base + fq * 64 < qo_len)
      {
        if constexpr (std::is_same<DTypeOut, half>::value)
        {
          ((half2*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16))[0] = __float22half2_rn(((float2*)(RO[fq][fv]))[0]);
          ((half2*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8))[0] = __float22half2_rn(((float2*)(RO[fq][fv]))[2]);
        }
        else
        {
          ((nv_bfloat162*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16))[0] = __float22bfloat162_rn(((float2*)(RO[fq][fv]))[0]);
          ((nv_bfloat162*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8))[0] = __float22bfloat162_rn(((float2*)(RO[fq][fv]))[2]);
        }
      }

      if (Q_idx_lane_base + fq * 64 + 8 < qo_len)
      {
        if constexpr (std::is_same<DTypeOut, half>::value)
        {
          ((half2*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 * stride_seq_o))[0] = __float22half2_rn(((float2*)(RO[fq][fv]))[1]);
          ((half2*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 + 8 * stride_seq_o))[0] = __float22half2_rn(((float2*)(RO[fq][fv]))[3]);
        }
        else
        {
          ((nv_bfloat162*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 * stride_seq_o))[0] = __float22bfloat162_rn(((float2*)(RO[fq][fv]))[1]);
          ((nv_bfloat162*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 + 8 * stride_seq_o))[0] = __float22bfloat162_rn(((float2*)(RO[fq][fv]))[3]);
        }
      }
    }

    if constexpr (return_lse)
    {
      // only works for CTA_Q = 64
      uint32_t lse_idx = bx * CTA_Q + lane_id / 4 + 8 * (lane_id % 4) + 16 * warp_idx;
      float *lse_lane_ptr = Lse + batch_id * (qo_len * num_qo_heads) + head_id * qo_len + lse_idx;
      uint32_t fq = (lane_id % 4) / 2;
      uint32_t k = (lane_id % 4) % 2;

      if (lse_idx < qo_len && (lane_id % 4) < 2)
      {
        lse_lane_ptr[0] = (math::ptx_log2(d[fq][k]) + m[fq][k]);
      }
    }
  }
#endif
}


template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t NUM_THREADS, uint32_t head_dim, QuantGranularity Q_GRAN, QuantGranularity K_GRAN, typename DTypeOut, MaskMode mask_mode = MaskMode::kNone, bool return_lse = false, bool fuse_v_scale=false, bool ExpCast=false, bool BlockValues=false, bool Optimized=false, bool ChannelScale=false, bool PackedProbability=false>
__global__ void sage_vc_attention_kernel(const __grid_constant__ CUtensorMap tensorMapQ,
                                        const __grid_constant__ CUtensorMap tensorMapK,
                                        const __grid_constant__ CUtensorMap tensorMapV,
                                        float *__restrict__ Q_scale, float *__restrict__ K_scale, float *__restrict__ V_scale, const half* __restrict__ V_mean,
                                        DTypeOut* O, float *__restrict__ Lse, uint32_t stride_bz_o, uint32_t stride_h_o, uint32_t stride_seq_o,
                                        const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
                                        float sm_scale) {
  sage_vc_attention_body<CTA_Q, CTA_K, NUM_THREADS, head_dim, Q_GRAN, K_GRAN, DTypeOut,
      mask_mode, return_lse, fuse_v_scale, ExpCast, BlockValues, Optimized, ChannelScale, PackedProbability, false>(tensorMapQ, tensorMapK, tensorMapV, Q_scale, K_scale, V_scale, V_mean,
      O, Lse, stride_bz_o, stride_h_o, stride_seq_o, qo_len, kv_len, num_kv_groups, sm_scale);
}

// Separate entry keeps the register target off every vc7/reference kernel.
// 168 registers at 128 threads permits three register-resident CTAs on SM90.
// Actual speed and spills remain toolchain-dependent; compare against vc7.
template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t NUM_THREADS, uint32_t head_dim, QuantGranularity Q_GRAN, QuantGranularity K_GRAN, typename DTypeOut, MaskMode mask_mode = MaskMode::kNone, bool return_lse = false, bool fuse_v_scale=false, bool ExpCast=false, bool BlockValues=false, bool Optimized=false, bool ChannelScale=false, bool PackedProbability=false>
__global__ __maxnreg__(168) void sage_vc_attention_kernel_compact(const __grid_constant__ CUtensorMap tensorMapQ,
                                        const __grid_constant__ CUtensorMap tensorMapK,
                                        const __grid_constant__ CUtensorMap tensorMapV,
                                        float *__restrict__ Q_scale, float *__restrict__ K_scale, float *__restrict__ V_scale, const half* __restrict__ V_mean,
                                        DTypeOut* O, float *__restrict__ Lse, uint32_t stride_bz_o, uint32_t stride_h_o, uint32_t stride_seq_o,
                                        const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
                                        float sm_scale) {
  sage_vc_attention_body<CTA_Q, CTA_K, NUM_THREADS, head_dim, Q_GRAN, K_GRAN, DTypeOut,
      mask_mode, return_lse, fuse_v_scale, ExpCast, BlockValues, Optimized, ChannelScale, PackedProbability, true>(tensorMapQ, tensorMapK, tensorMapV, Q_scale, K_scale, V_scale, V_mean,
      O, Lse, stride_bz_o, stride_h_o, stride_seq_o, qo_len, kv_len, num_kv_groups, sm_scale);
}

// Separate specialization leaves all vc9 kernels available as a same-payload
// baseline. Limit D128 register residency to the same target as the control.
template<int Dim, typename Out>
__global__ __maxnreg__(168) void sage_vc_attention_kernel_expcast_pipeline(
    const __grid_constant__ CUtensorMap tensorMapQ,
    const __grid_constant__ CUtensorMap tensorMapK,
    const __grid_constant__ CUtensorMap tensorMapV,
    float* Q_scale, float* K_scale, float* V_scale, const half* V_mean,
    Out* O, float* Lse, uint32_t stride_bz_o, uint32_t stride_h_o, uint32_t stride_seq_o,
    uint32_t qo_len, uint32_t kv_len, uint32_t num_kv_groups, float sm_scale) {
  sage_vc_attention_body<64,128,128,Dim,QuantGranularity::kPerThread,QuantGranularity::kPerThread,
      Out,MaskMode::kNone,false,true,true,false,true,false,true,false,true>(
      tensorMapQ,tensorMapK,tensorMapV,Q_scale,K_scale,V_scale,V_mean,
      O,Lse,stride_bz_o,stride_h_o,stride_seq_o,qo_len,kv_len,num_kv_groups,sm_scale);
}

// Same vc10 numerical contract; only probability instruction scheduling changes.
template<int Dim, typename Out>
__global__ __maxnreg__(168) void sage_vc_attention_kernel_expcast_interleaved(
    const __grid_constant__ CUtensorMap tensorMapQ,
    const __grid_constant__ CUtensorMap tensorMapK,
    const __grid_constant__ CUtensorMap tensorMapV,
    float* Q_scale, float* K_scale, float* V_scale, const half* V_mean,
    Out* O, float* Lse, uint32_t stride_bz_o, uint32_t stride_h_o, uint32_t stride_seq_o,
    uint32_t qo_len, uint32_t kv_len, uint32_t num_kv_groups, float sm_scale) {
  sage_vc_attention_body<64,128,128,Dim,QuantGranularity::kPerThread,QuantGranularity::kPerThread,
      Out,MaskMode::kNone,false,true,true,false,true,false,true,false,true,true>(
      tensorMapQ,tensorMapK,tensorMapV,Q_scale,K_scale,V_scale,V_mean,
      O,Lse,stride_bz_o,stride_h_o,stride_seq_o,qo_len,kv_len,num_kv_groups,sm_scale);
}

// Fold the per-score affine map; retain vc11 for extreme numerical ranges.
template<int Dim, typename Out>
__global__ __maxnreg__(168) void sage_vc_attention_kernel_expcast_affine(
    const __grid_constant__ CUtensorMap tensorMapQ,
    const __grid_constant__ CUtensorMap tensorMapK,
    const __grid_constant__ CUtensorMap tensorMapV,
    float* Q_scale, float* K_scale, float* V_scale, const half* V_mean,
    Out* O, float* Lse, uint32_t stride_bz_o, uint32_t stride_h_o, uint32_t stride_seq_o,
    uint32_t qo_len, uint32_t kv_len, uint32_t num_kv_groups, float sm_scale) {
  sage_vc_attention_body<64,128,128,Dim,QuantGranularity::kPerThread,QuantGranularity::kPerThread,
      Out,MaskMode::kNone,false,true,true,false,true,false,true,false,true,true,true>(
      tensorMapQ,tensorMapK,tensorMapV,Q_scale,K_scale,V_scale,V_mean,
      O,Lse,stride_bz_o,stride_h_o,stride_seq_o,qo_len,kv_len,num_kv_groups,sm_scale);
}

// vc14: pair the exact RNE integer conversions; keep vc12 math and schedule.
template<int Dim, typename Out>
__global__ __maxnreg__(168) void sage_vc_attention_kernel_expcast_paired(
    const __grid_constant__ CUtensorMap tensorMapQ,
    const __grid_constant__ CUtensorMap tensorMapK,
    const __grid_constant__ CUtensorMap tensorMapV,
    float* Q_scale, float* K_scale, float* V_scale, const half* V_mean,
    Out* O, float* Lse, uint32_t stride_bz_o, uint32_t stride_h_o, uint32_t stride_seq_o,
    uint32_t qo_len, uint32_t kv_len, uint32_t num_kv_groups, float sm_scale) {
  sage_vc_attention_body<64,128,128,Dim,QuantGranularity::kPerThread,QuantGranularity::kPerThread,
      Out,MaskMode::kNone,false,true,true,false,true,false,true,false,true,true,true,0,true>(
      tensorMapQ,tensorMapK,tensorMapV,Q_scale,K_scale,V_scale,V_mean,
      O,Lse,stride_bz_o,stride_h_o,stride_seq_o,qo_len,kv_len,num_kv_groups,sm_scale);
}

// vc15: carry the high pair through F2IP; remove final PRMT, same vc14 math.
template<int Dim, typename Out>
__global__ __maxnreg__(168) void sage_vc_attention_kernel_expcast_pack4(
    const __grid_constant__ CUtensorMap tensorMapQ,
    const __grid_constant__ CUtensorMap tensorMapK,
    const __grid_constant__ CUtensorMap tensorMapV,
    float* Q_scale, float* K_scale, float* V_scale, const half* V_mean,
    Out* O, float* Lse, uint32_t stride_bz_o, uint32_t stride_h_o, uint32_t stride_seq_o,
    uint32_t qo_len, uint32_t kv_len, uint32_t num_kv_groups, float sm_scale) {
  sage_vc_attention_body<64,128,128,Dim,QuantGranularity::kPerThread,QuantGranularity::kPerThread,
      Out,MaskMode::kNone,false,true,true,false,true,false,true,false,true,true,true,0,true,true>(
      tensorMapQ,tensorMapK,tensorMapV,Q_scale,K_scale,V_scale,V_mean,
      O,Lse,stride_bz_o,stride_h_o,stride_seq_o,qo_len,kv_len,num_kv_groups,sm_scale);
}

// Schedules 1/2 share encoding and scratch storage. Only mass timing differs.
template<int Dim, typename Out, int ProbabilitySchedule>
__global__ __maxnreg__(168) void sage_vc_attention_kernel_expcast_deferred(
    const __grid_constant__ CUtensorMap tensorMapQ,
    const __grid_constant__ CUtensorMap tensorMapK,
    const __grid_constant__ CUtensorMap tensorMapV,
    float* Q_scale, float* K_scale, float* V_scale, const half* V_mean,
    Out* O, float* Lse, uint32_t stride_bz_o, uint32_t stride_h_o, uint32_t stride_seq_o,
    uint32_t qo_len, uint32_t kv_len, uint32_t num_kv_groups, float sm_scale) {
  sage_vc_attention_body<64,128,128,Dim,QuantGranularity::kPerThread,QuantGranularity::kPerThread,
      Out,MaskMode::kNone,false,true,true,false,true,false,true,false,true,true,true,ProbabilitySchedule>(
      tensorMapQ,tensorMapK,tensorMapV,Q_scale,K_scale,V_scale,V_mean,
      O,Lse,stride_bz_o,stride_h_o,stride_seq_o,qo_len,kv_len,num_kv_groups,sm_scale);
}

#include "sage_layout.cuh"
#include "tma_pipeline.cuh"
#include "compute_pipeline.cuh"
#include "shared_history.cuh"

} // namespace sage_vc

namespace {
template<int Dim, typename Out, bool ExpCast, bool BlockValues, bool Optimized, bool ChannelScale, bool PackedProbability, bool CompactValues, bool PipelineProbability=false, bool InterleavedProbability=false, bool FoldAffine=false, int ProbabilitySchedule=0, bool PairedEncode=false, bool FusedPack=false>
auto select_vc_kernel() {
  if constexpr (FusedPack)
    return sage_vc::sage_vc_attention_kernel_expcast_pack4<Dim,Out>;
  else if constexpr (PairedEncode)
    return sage_vc::sage_vc_attention_kernel_expcast_paired<Dim,Out>;
  else if constexpr (ProbabilitySchedule != 0)
    return sage_vc::sage_vc_attention_kernel_expcast_deferred<Dim,Out,ProbabilitySchedule>;
  else if constexpr (PipelineProbability && InterleavedProbability && FoldAffine)
    return sage_vc::sage_vc_attention_kernel_expcast_affine<Dim,Out>;
  else if constexpr (PipelineProbability && InterleavedProbability)
    return sage_vc::sage_vc_attention_kernel_expcast_interleaved<Dim,Out>;
  else if constexpr (PipelineProbability)
    return sage_vc::sage_vc_attention_kernel_expcast_pipeline<Dim,Out>;
  else if constexpr (CompactValues)
    return sage_vc::sage_vc_attention_kernel_compact<64, 128, 128, Dim,
        QuantGranularity::kPerThread, QuantGranularity::kPerThread, Out, MaskMode::kNone, false, true,
        ExpCast, BlockValues, Optimized, ChannelScale, PackedProbability>;
  else
    return sage_vc::sage_vc_attention_kernel<64, 128, 128, Dim,
        QuantGranularity::kPerThread, QuantGranularity::kPerThread, Out, MaskMode::kNone, false, true,
        ExpCast, BlockValues, Optimized, ChannelScale, PackedProbability>;
}

// Read compiled resources on the current Hopper device, outside timing.
// Local memory includes stack/spill storage; it is not a measured traffic count.
template<typename Out, bool ExpCast, bool CompactValues>
std::map<std::string, int64_t> channel_resources() {
  auto kernel = select_vc_kernel<128, Out, ExpCast, true, true, true, ExpCast, CompactValues>();
  constexpr int dynamic_bytes = (64 + 128 + 128) * 128;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes));
  cudaFuncAttributes attr{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
  int active_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, 128, dynamic_bytes));
  return {{"registers_per_thread", attr.numRegs}, {"local_bytes_per_thread", int64_t(attr.localSizeBytes)},
          {"static_shared_bytes", int64_t(attr.sharedSizeBytes)}, {"dynamic_shared_bytes", dynamic_bytes},
          {"threads_per_block", 128}, {"predicted_active_blocks_per_sm", active_blocks}};
}

template<typename Out>
std::map<std::string, std::map<std::string, int64_t>> channel_resource_variants() {
  return {{"smooth_vc7", channel_resources<Out, false, false>()},
          {"smooth_optimized", channel_resources<Out, false, true>()},
          {"full_vc7", channel_resources<Out, true, false>()},
          {"full_optimized", channel_resources<Out, true, true>()}};
}

template<typename Out, bool Pipeline, bool Interleaved=false, bool Affine=false, int Schedule=0, bool PairedEncode=false, bool FusedPack=false>
std::map<std::string, int64_t> expcast_resources() {
  auto kernel = select_vc_kernel<128,Out,true,false,true,false,true,false,Pipeline,Interleaved,Affine,Schedule,PairedEncode,FusedPack>();
  constexpr int dynamic_bytes = (64+128+128)*128 + (Schedule ? 8192 : 0);
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,dynamic_bytes));
  cudaFuncAttributes attr{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,kernel));
  int active_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,kernel,128,dynamic_bytes));
  return {{"registers_per_thread",attr.numRegs},{"local_bytes_per_thread",int64_t(attr.localSizeBytes)},
          {"static_shared_bytes",int64_t(attr.sharedSizeBytes)},{"dynamic_shared_bytes",dynamic_bytes},
          {"threads_per_block",128},{"predicted_active_blocks_per_sm",active_blocks}};
}

template<int Dim, typename Out, bool ExpCast, bool BlockValues, bool Optimized, bool ChannelScale=false, bool PackedProbability=false, bool CompactValues=false, bool PipelineProbability=false, bool InterleavedProbability=false, bool FoldAffine=false, int ProbabilitySchedule=0, bool PairedEncode=false, bool FusedPack=false>
void launch_vc(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor qs,
    torch::Tensor ks, torch::Tensor vs, torch::Tensor means, torch::Tensor out, float scale) {
  auto qm = sage_vc::create_tensor_map_4D<64, Dim>(reinterpret_cast<int8_t*>(q.data_ptr()),
      q.size(0), q.size(1), q.size(2), Dim, q.stride(0), q.stride(1), q.stride(2));
  auto km = sage_vc::create_tensor_map_4D<128, Dim>(reinterpret_cast<int8_t*>(k.data_ptr()),
      k.size(0), k.size(1), k.size(2), Dim, k.stride(0), k.stride(1), k.stride(2));
  auto vm = sage_vc::create_tensor_map_4D<Dim, 128>(reinterpret_cast<int8_t*>(v.data_ptr()),
      v.size(0), v.size(1), Dim, v.size(3), v.stride(0), v.stride(1), v.stride(2));
  auto kernel = select_vc_kernel<Dim, Out, ExpCast, BlockValues, Optimized, ChannelScale, PackedProbability, CompactValues, PipelineProbability, InterleavedProbability, FoldAffine, ProbabilitySchedule, PairedEncode, FusedPack>();
  constexpr int shared_bytes = (64 + 128 + 128) * Dim + (ProbabilitySchedule ? 8192 : 0);
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  dim3 grid((q.size(2) + 63) / 64, q.size(1), q.size(0));
  kernel<<<grid, 128, shared_bytes, c10::cuda::getCurrentCUDAStream(q.get_device()).stream()>>>(
      qm, km, vm, qs.data_ptr<float>(), ks.data_ptr<float>(), vs.data_ptr<float>(),
      BlockValues ? reinterpret_cast<const half*>(means.data_ptr()) : nullptr,
      reinterpret_cast<Out*>(out.data_ptr()), nullptr, out.stride(0), out.stride(1), out.stride(2),
      q.size(2), k.size(2), 1, scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// The schedule mode changes no quantized operands or output allocation.
// 1: original order + vc15 overlap; 2: early V; 3/4: pack-first control;
// 5: original order, ExpCast history before PV.
// 6..9: probability helper candidates; 10..12: vc16 with register caps 128/144/192.
// 13..15: split D128 PV into two N64 fragments with register caps 128/144/168.
template<int Schedule>
constexpr int sage_register_limit() {
  return Schedule>=25 ? 168 : Schedule==23 ? 128 : Schedule==24 ? 168 : Schedule>=20 ? 224 : (Schedule==10 || Schedule==13) ? 128 : (Schedule==11 || Schedule==14) ? 144 : Schedule==12 ? 192 : 168;
}

template<int Dim, typename Out, bool Cast, int Schedule>
auto select_sage_layout() {
  static_assert(Schedule>=1 && Schedule<=29);
  static_assert(!Cast || (Schedule!=3 && Schedule!=4));
  if constexpr (Schedule>=25) {
    return sage_vc::sage_vc_attention_kernel_sage_layout<64,128,128,Dim,
        QuantGranularity::kPerThread,QuantGranularity::kPerThread,Out,
        MaskMode::kNone,false,true,Cast,false,false,Cast,0,168,false,(Cast ? Schedule-24 : 0)>;
  } else if constexpr (Schedule>=23) {
    if constexpr (Dim==128)
      return sage_vc::sage_vc_attention_kernel_shared_history<64,128,128,Dim,
          QuantGranularity::kPerThread,QuantGranularity::kPerThread,Out,Cast,sage_register_limit<Schedule>()>;
    else return select_sage_layout<Dim,Out,Cast,1>(); // rejected by public/native input checks
  } else if constexpr (Schedule>=20) {
    return sage_vc::sage_vc_attention_kernel_compute_pipeline<64,128,128,Dim,
        QuantGranularity::kPerThread,QuantGranularity::kPerThread,Out,Cast,Schedule-19>;
  } else if constexpr (Schedule>=16) {
    return sage_vc::sage_vc_attention_kernel_tma_pipeline<64,128,128,Dim,
        QuantGranularity::kPerThread,QuantGranularity::kPerThread,Out,Cast,
        (Schedule==19 ? 3 : Schedule-15),(Schedule!=19)>;
  } else return sage_vc::sage_vc_attention_kernel_sage_layout<64,128,128,Dim,
      QuantGranularity::kPerThread,QuantGranularity::kPerThread,Out,
      MaskMode::kNone,false,true,Cast,(Schedule==2 || Schedule==4),
      (Schedule==3 || Schedule==4),(Cast && Schedule!=5),
      (Schedule>=6 && Schedule<=9 ? Schedule-5 : 0),sage_register_limit<Schedule>(),(Schedule>=13 && Schedule<=15)>;
}

// K-only / V-only add one tile; KV and its late-copy control add two.
template<int Schedule, int Dim>
constexpr int sage_layout_shared_bytes() {
  return (64+128+128+(Schedule>=18 && Schedule<=19 ? 256 : Schedule>=16 && Schedule<=17 ? 128 : 0))*Dim + (Schedule>=23 && Schedule<=24 ? 14336 : 0);
}

template<int Dim, typename Out, bool Cast, int Schedule>
void launch_sage_layout(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor qs, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, float scale) {
  auto qm = sage_vc::create_tensor_map_4D<64,Dim>(q.data_ptr<int8_t>(),
      q.size(0),q.size(1),q.size(2),Dim,q.stride(0),q.stride(1),q.stride(2));
  auto km = sage_vc::create_tensor_map_4D<128,Dim>(k.data_ptr<int8_t>(),
      k.size(0),k.size(1),k.size(2),Dim,k.stride(0),k.stride(1),k.stride(2));
  auto vm = sage_vc::create_tensor_map_4D<Dim,128>(reinterpret_cast<int8_t*>(v.data_ptr()),
      v.size(0),v.size(1),Dim,v.size(3),v.stride(0),v.stride(1),v.stride(2));
  auto kernel=select_sage_layout<Dim,Out,Cast,Schedule>();
  constexpr int shared_bytes=sage_layout_shared_bytes<Schedule,Dim>();
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes));
  dim3 grid((q.size(2)+63)/64,q.size(1),q.size(0));
  kernel<<<grid,128,shared_bytes,c10::cuda::getCurrentCUDAStream(q.get_device()).stream()>>>(
      qm,km,vm,qs.data_ptr<float>(),ks.data_ptr<float>(),vs.data_ptr<float>(),
      reinterpret_cast<Out*>(out.data_ptr()),nullptr,out.stride(0),out.stride(1),out.stride(2),
      q.size(2),k.size(2),1,scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int Dim, typename Out, int Schedule>
void dispatch_sage_layout(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor qs, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, float scale, bool cast) {
  if constexpr (Schedule>=6 && Schedule<=9)
    launch_sage_layout<Dim,Out,true,Schedule>(q,k,v,qs,ks,vs,out,scale);
  else if constexpr (Schedule==3 || Schedule==4)
    launch_sage_layout<Dim,Out,false,Schedule>(q,k,v,qs,ks,vs,out,scale);
  else if (cast) launch_sage_layout<Dim,Out,true,Schedule>(q,k,v,qs,ks,vs,out,scale);
  else launch_sage_layout<Dim,Out,false,(Schedule==5 ? 1 : Schedule)>(q,k,v,qs,ks,vs,out,scale);
}

template<int Schedule>
void dispatch_sage_output(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor qs, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, float scale,
    int output_type, bool cast) {
  if (q.size(3)==64) {
    if (output_type==0) dispatch_sage_layout<64,half,Schedule>(q,k,v,qs,ks,vs,out,scale,cast);
    else dispatch_sage_layout<64,nv_bfloat16,Schedule>(q,k,v,qs,ks,vs,out,scale,cast);
  } else {
    if (output_type==0) dispatch_sage_layout<128,half,Schedule>(q,k,v,qs,ks,vs,out,scale,cast);
    else dispatch_sage_layout<128,nv_bfloat16,Schedule>(q,k,v,qs,ks,vs,out,scale,cast);
  }
}

template<typename Out, bool Cast, int Schedule>
std::map<std::string,int64_t> sage_layout_resources() {
  auto kernel=select_sage_layout<128,Out,Cast,Schedule>();
  constexpr int bytes=sage_layout_shared_bytes<Schedule,128>();
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  cudaFuncAttributes attr{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,kernel));
  int active=0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,128,bytes));
  return {{"registers_per_thread",attr.numRegs},{"local_bytes_per_thread",int64_t(attr.localSizeBytes)},
          {"static_shared_bytes",int64_t(attr.sharedSizeBytes)},{"dynamic_shared_bytes",bytes},
          {"threads_per_block",128},{"predicted_active_blocks_per_sm",active},
          {"requested_register_limit",sage_register_limit<Schedule>()},{"pv_fragment_channels",(Schedule>=13 && Schedule<=15) || (Schedule>=23 && Schedule<=24) ? 64 : 128},
          {"k_buffers",Schedule==16 || (Schedule>=18 && Schedule<=19) ? 2 : 1},
          {"v_buffers",Schedule>=17 && Schedule<=19 ? 2 : 1},
          {"early_tma_prefetch",Schedule>=16 && Schedule<=18 ? 1 : 0},
          {"history_shared_bytes",Schedule>=23 && Schedule<=24 ? 14336 : 0},
          {"integer_max",Cast && (Schedule==25 || Schedule>=27) ? 1 : 0},
          {"skip_identity_history",Cast && Schedule>=26 ? 1 : 0},
          {"max_state_mode",Cast && Schedule>=28 ? Schedule-27 : 0},
          {"compute_schedule",Schedule>=20 && Schedule<=22 ? Schedule-19 : 0}};
}

template<int Dim, typename Out, bool Optimized, bool PackedProbability, bool CompactValues, bool PipelineProbability=false, bool InterleavedProbability=false, bool FoldAffine=false, int ProbabilitySchedule=0, bool PairedEncode=false, bool FusedPack=false>
void dispatch_vc(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor qs,
    torch::Tensor ks, torch::Tensor vs, torch::Tensor means, torch::Tensor out, float scale, bool cast, bool smooth) {
  if (smooth && vs.dim() == 3) {
    if (cast) launch_vc<Dim, Out, true, true, Optimized, true, PackedProbability, (CompactValues && Dim == 128)>(q,k,v,qs,ks,vs,means,out,scale);
    else launch_vc<Dim, Out, false, true, Optimized, true, false, (CompactValues && Dim == 128)>(q,k,v,qs,ks,vs,means,out,scale);
  }
  else if (cast && smooth) launch_vc<Dim, Out, true, true, Optimized, false, PackedProbability>(q,k,v,qs,ks,vs,means,out,scale);
  else if (cast) launch_vc<Dim, Out, true, false, Optimized, false, PackedProbability, false, PipelineProbability, InterleavedProbability, FoldAffine, ProbabilitySchedule, PairedEncode, FusedPack>(q,k,v,qs,ks,vs,means,out,scale);
  else if (smooth) launch_vc<Dim, Out, false, true, Optimized>(q,k,v,qs,ks,vs,means,out,scale);
  else launch_vc<Dim, Out, false, false, false>(q,k,v,qs,ks,vs,means,out,scale);
}
template<bool Optimized, bool PackedProbability=false, bool CompactValues=false, bool PipelineProbability=false, bool InterleavedProbability=false, bool FoldAffine=false, int ProbabilitySchedule=0, bool PairedEncode=false, bool FusedPack=false>
void dispatch_output(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor qs,
    torch::Tensor ks, torch::Tensor vs, torch::Tensor means, torch::Tensor out, float scale,
    int64_t output_type, bool cast, bool smooth) {
  if (q.size(3) == 64) {
    if (output_type == 0) dispatch_vc<64, half, Optimized, PackedProbability, CompactValues, PipelineProbability, InterleavedProbability, FoldAffine, ProbabilitySchedule, PairedEncode, FusedPack>(q,k,v,qs,ks,vs,means,out,scale,cast,smooth);
    else dispatch_vc<64, nv_bfloat16, Optimized, PackedProbability, CompactValues, PipelineProbability, InterleavedProbability, FoldAffine, ProbabilitySchedule, PairedEncode, FusedPack>(q,k,v,qs,ks,vs,means,out,scale,cast,smooth);
  } else {
    if (output_type == 0) dispatch_vc<128, half, Optimized, PackedProbability, CompactValues, PipelineProbability, InterleavedProbability, FoldAffine, ProbabilitySchedule, PairedEncode, FusedPack>(q,k,v,qs,ks,vs,means,out,scale,cast,smooth);
    else dispatch_vc<128, nv_bfloat16, Optimized, PackedProbability, CompactValues, PipelineProbability, InterleavedProbability, FoldAffine, ProbabilitySchedule, PairedEncode, FusedPack>(q,k,v,qs,ks,vs,means,out,scale,cast,smooth);
  }
}
}

torch::Tensor vc_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor qs, torch::Tensor ks, torch::Tensor vs, torch::Tensor means,
    bool expcast, bool block_values, int64_t output_type, double scale, bool optimized, bool packed_probability, bool compact_values, bool pipeline_probability, bool interleaved_probability, bool affine_probability, int64_t probability_schedule, bool paired_probability, bool fused_probability, int64_t sage_schedule) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 4 && q.numel() > 0, "Q requires nonempty CUDA BHND");
  for (const auto& x : {q,k,v,qs,ks,vs,means})
    TORCH_CHECK(x.device() == q.device() && x.is_contiguous(), "Payloads must be contiguous on the same CUDA device");
  TORCH_CHECK(q.scalar_type() == at::kChar && k.scalar_type() == at::kChar, "Q/K payloads require INT8");
  TORCH_CHECK(v.scalar_type() == at::ScalarType::Float8_e4m3fn, "V payload requires E4M3");
  TORCH_CHECK(qs.scalar_type() == at::kFloat && ks.scalar_type() == at::kFloat && vs.scalar_type() == at::kFloat, "Scales require FP32");
  TORCH_CHECK(means.scalar_type() == at::kHalf, "Scaled V means require FP16");
  TORCH_CHECK(k.dim() == 4 && v.dim() == 4 && k.numel() > 0, "K/V require nonempty four-dimensional tensors");
  int64_t batch = q.size(0), heads = q.size(1), nq = q.size(2), dim = q.size(3), nk = k.size(2), tiles = (nk + 127) / 128;
  TORCH_CHECK(dim == 64 || dim == 128, "Sage VC supports D=64/128");
  TORCH_CHECK(k.size(0) == batch && k.size(1) == heads && k.size(3) == dim, "K batch/heads/dimension mismatch; no GQA");
  TORCH_CHECK(v.sizes() == torch::IntArrayRef({batch,heads,dim,tiles * 128}), "Incorrect packed V shape");
  TORCH_CHECK(qs.sizes() == torch::IntArrayRef({batch,heads,((nq + 63) / 64) * 32}), "Incorrect Q scales");
  TORCH_CHECK(ks.sizes() == torch::IntArrayRef({batch,heads,tiles * 4}), "Incorrect K scales");
  if (block_values) {
    TORCH_CHECK((vs.sizes() == torch::IntArrayRef({batch,heads,tiles,dim}) ||
        vs.sizes() == torch::IntArrayRef({batch,heads,dim})) &&
        means.sizes() == torch::IntArrayRef({batch,heads,tiles,dim}), "Incorrect block V metadata");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(vs.data_ptr()) % alignof(float2) == 0 &&
        reinterpret_cast<uintptr_t>(means.data_ptr()) % alignof(half2) == 0,
        "Block V metadata requires float2/half2 pointer alignment");
  } else TORCH_CHECK(vs.sizes() == torch::IntArrayRef({batch,heads,dim}) && means.numel() == 0, "Incorrect global V metadata");
  TORCH_CHECK(output_type == 0 || output_type == 1, "Output type must be 0 (FP16) or 1 (BF16)");
  TORCH_CHECK(std::isfinite(scale) && scale > 0 && scale <= FLT_MAX, "Scale must be positive finite FP32");
  TORCH_CHECK(q.numel() < INT32_MAX && k.numel() < INT32_MAX && v.numel() < INT32_MAX && batch <= 65535 && heads <= 65535, "Shape exceeds kernel bounds");
  TORCH_CHECK(probability_schedule >= 0 && probability_schedule <= 2, "Probability schedule must be 0 (vc12), 1 (overlap) or 2 (serial scratch)");
  TORCH_CHECK(probability_schedule == 0 || (optimized && packed_probability && compact_values && pipeline_probability && interleaved_probability && affine_probability), "Scratch schedules require the affine pipeline");
  TORCH_CHECK(!fused_probability || paired_probability,"Fused packing requires paired conversion");
  TORCH_CHECK(!paired_probability || (probability_schedule == 0 && optimized && packed_probability && compact_values && pipeline_probability && interleaved_probability && affine_probability), "Paired conversion requires the affine pipeline without scratch");
  TORCH_CHECK(sage_schedule>=0 && sage_schedule<=29,"Sage layout schedule must be 0..29");
  TORCH_CHECK(sage_schedule<23 || sage_schedule>24 || q.size(3)==128,"Shared-history candidates require D128");
  TORCH_CHECK(sage_schedule==0 || !block_values,"Sage layout candidates require v_smooth=False");
  TORCH_CHECK(sage_schedule<6 || sage_schedule>9 || expcast,"Probability candidates require expcast=True");
  TORCH_CHECK(!expcast || (sage_schedule!=3 && sage_schedule!=4),"Pack-first candidates are control-only");
  const c10::cuda::CUDAGuard guard(q.device());
  int major, minor;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, q.get_device()));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, q.get_device()));
  TORCH_CHECK(major == 9 && minor == 0, "Sage VC native attention requires Hopper; no fallback");
  auto out = torch::empty(q.sizes(), q.options().dtype(output_type == 0 ? at::kHalf : at::kBFloat16));
  if (sage_schedule==1) dispatch_sage_output<1>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==2) dispatch_sage_output<2>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==3) dispatch_sage_output<3>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==4) dispatch_sage_output<4>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==5) dispatch_sage_output<5>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==6) dispatch_sage_output<6>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==7) dispatch_sage_output<7>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==8) dispatch_sage_output<8>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==9) dispatch_sage_output<9>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==10) dispatch_sage_output<10>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==11) dispatch_sage_output<11>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==12) dispatch_sage_output<12>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==13) dispatch_sage_output<13>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==14) dispatch_sage_output<14>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==15) dispatch_sage_output<15>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==16) dispatch_sage_output<16>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==17) dispatch_sage_output<17>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==18) dispatch_sage_output<18>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==19) dispatch_sage_output<19>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==20) dispatch_sage_output<20>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==21) dispatch_sage_output<21>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==22) dispatch_sage_output<22>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==23) dispatch_sage_output<23>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==24) dispatch_sage_output<24>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==25) dispatch_sage_output<25>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==26) dispatch_sage_output<26>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==27) dispatch_sage_output<27>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==28) dispatch_sage_output<28>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (sage_schedule==29) dispatch_sage_output<29>(q,k,v,qs,ks,vs,out,scale,output_type,expcast);
  else if (fused_probability) dispatch_output<true,true,true,true,true,true,0,true,true>(q,k,v,qs,ks,vs,means,out,scale,output_type,expcast,block_values);
  else if (paired_probability) dispatch_output<true,true,true,true,true,true,0,true>(q,k,v,qs,ks,vs,means,out,scale,output_type,expcast,block_values);
  else if (probability_schedule == 1) dispatch_output<true,true,true,true,true,true,1>(q,k,v,qs,ks,vs,means,out,scale,output_type,expcast,block_values);
  else if (probability_schedule == 2) dispatch_output<true,true,true,true,true,true,2>(q,k,v,qs,ks,vs,means,out,scale,output_type,expcast,block_values);
  else if (optimized && packed_probability && compact_values && pipeline_probability && interleaved_probability && affine_probability) dispatch_output<true, true, true, true, true, true>(q,k,v,qs,ks,vs,means,out,scale,output_type,expcast,block_values);
  else if (optimized && packed_probability && compact_values && pipeline_probability && interleaved_probability) dispatch_output<true, true, true, true, true>(q,k,v,qs,ks,vs,means,out,scale,output_type,expcast,block_values);
  else if (optimized && packed_probability && compact_values && pipeline_probability) dispatch_output<true, true, true, true>(q,k,v,qs,ks,vs,means,out,scale,output_type,expcast,block_values);
  else if (optimized && packed_probability && compact_values) dispatch_output<true, true, true>(q,k,v,qs,ks,vs,means,out,scale,output_type,expcast,block_values);
  else if (optimized && packed_probability) dispatch_output<true, true>(q,k,v,qs,ks,vs,means,out,scale,output_type,expcast,block_values);
  else if (optimized) dispatch_output<true>(q,k,v,qs,ks,vs,means,out,scale,output_type,expcast,block_values);
  else dispatch_output<false>(q,k,v,qs,ks,vs,means,out,scale,output_type,expcast,block_values);
  return out;
}

std::map<std::string, std::map<std::string, int64_t>> vc_kernel_resources(int64_t output_type) {
  TORCH_CHECK(output_type == 0 || output_type == 1, "Output type must be 0 (FP16) or 1 (BF16)");
  int device, major, minor;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  TORCH_CHECK(major == 9 && minor == 0, "Compiled attention resource query requires Hopper");
  auto result = output_type == 0 ? channel_resource_variants<half>() : channel_resource_variants<nv_bfloat16>();
  result["expcast_vc9"] = output_type == 0 ? expcast_resources<half,false>() : expcast_resources<nv_bfloat16,false>();
  result["expcast_vc10"] = output_type == 0 ? expcast_resources<half,true>() : expcast_resources<nv_bfloat16,true>();
  result["expcast_vc11"] = output_type == 0 ? expcast_resources<half,true,true>() : expcast_resources<nv_bfloat16,true,true>();
  result["expcast_vc12"] = output_type == 0 ? expcast_resources<half,true,true,true>() : expcast_resources<nv_bfloat16,true,true,true>();
  result["expcast_vc14"] = output_type == 0 ? expcast_resources<half,true,true,true,0,true>() : expcast_resources<nv_bfloat16,true,true,true,0,true>();
  result["expcast_vc15"] = output_type == 0 ? expcast_resources<half,true,true,true,0,true,true>() : expcast_resources<nv_bfloat16,true,true,true,0,true,true>();
  result["expcast_serial"] = output_type == 0 ? expcast_resources<half,true,true,true,2>() : expcast_resources<nv_bfloat16,true,true,true,2>();
  result["expcast_optimized"] = output_type == 0 ? expcast_resources<half,true,true,true,1>() : expcast_resources<nv_bfloat16,true,true,true,1>();
  result["control_sage_layout_1"] = output_type==0 ? sage_layout_resources<half,false,1>() : sage_layout_resources<nv_bfloat16,false,1>();
  result["control_sage_layout_2"] = output_type==0 ? sage_layout_resources<half,false,2>() : sage_layout_resources<nv_bfloat16,false,2>();
  result["control_sage_layout_3"] = output_type==0 ? sage_layout_resources<half,false,3>() : sage_layout_resources<nv_bfloat16,false,3>();
  result["control_sage_layout_4"] = output_type==0 ? sage_layout_resources<half,false,4>() : sage_layout_resources<nv_bfloat16,false,4>();
  result["expcast_sage_layout_1"] = output_type==0 ? sage_layout_resources<half,true,1>() : sage_layout_resources<nv_bfloat16,true,1>();
  result["expcast_sage_layout_2"] = output_type==0 ? sage_layout_resources<half,true,2>() : sage_layout_resources<nv_bfloat16,true,2>();
  result["expcast_sage_layout_5"] = output_type==0 ? sage_layout_resources<half,true,5>() : sage_layout_resources<nv_bfloat16,true,5>();
  result["expcast_sage_layout_6"] = output_type==0 ? sage_layout_resources<half,true,6>() : sage_layout_resources<nv_bfloat16,true,6>();
  result["expcast_sage_layout_7"] = output_type==0 ? sage_layout_resources<half,true,7>() : sage_layout_resources<nv_bfloat16,true,7>();
  result["expcast_sage_layout_8"] = output_type==0 ? sage_layout_resources<half,true,8>() : sage_layout_resources<nv_bfloat16,true,8>();
  result["expcast_sage_layout_9"] = output_type==0 ? sage_layout_resources<half,true,9>() : sage_layout_resources<nv_bfloat16,true,9>();
  result["control_sage_layout_10"] = output_type==0 ? sage_layout_resources<half,false,10>() : sage_layout_resources<nv_bfloat16,false,10>();
  result["expcast_sage_layout_10"] = output_type==0 ? sage_layout_resources<half,true,10>() : sage_layout_resources<nv_bfloat16,true,10>();
  result["control_sage_layout_11"] = output_type==0 ? sage_layout_resources<half,false,11>() : sage_layout_resources<nv_bfloat16,false,11>();
  result["expcast_sage_layout_11"] = output_type==0 ? sage_layout_resources<half,true,11>() : sage_layout_resources<nv_bfloat16,true,11>();
  result["control_sage_layout_12"] = output_type==0 ? sage_layout_resources<half,false,12>() : sage_layout_resources<nv_bfloat16,false,12>();
  result["expcast_sage_layout_12"] = output_type==0 ? sage_layout_resources<half,true,12>() : sage_layout_resources<nv_bfloat16,true,12>();
  result["control_sage_layout_13"] = output_type==0 ? sage_layout_resources<half,false,13>() : sage_layout_resources<nv_bfloat16,false,13>();
  result["expcast_sage_layout_13"] = output_type==0 ? sage_layout_resources<half,true,13>() : sage_layout_resources<nv_bfloat16,true,13>();
  result["control_sage_layout_14"] = output_type==0 ? sage_layout_resources<half,false,14>() : sage_layout_resources<nv_bfloat16,false,14>();
  result["expcast_sage_layout_14"] = output_type==0 ? sage_layout_resources<half,true,14>() : sage_layout_resources<nv_bfloat16,true,14>();
  result["control_sage_layout_15"] = output_type==0 ? sage_layout_resources<half,false,15>() : sage_layout_resources<nv_bfloat16,false,15>();
  result["expcast_sage_layout_15"] = output_type==0 ? sage_layout_resources<half,true,15>() : sage_layout_resources<nv_bfloat16,true,15>();
  result["control_sage_layout_16"] = output_type==0 ? sage_layout_resources<half,false,16>() : sage_layout_resources<nv_bfloat16,false,16>();
  result["expcast_sage_layout_16"] = output_type==0 ? sage_layout_resources<half,true,16>() : sage_layout_resources<nv_bfloat16,true,16>();
  result["control_sage_layout_17"] = output_type==0 ? sage_layout_resources<half,false,17>() : sage_layout_resources<nv_bfloat16,false,17>();
  result["expcast_sage_layout_17"] = output_type==0 ? sage_layout_resources<half,true,17>() : sage_layout_resources<nv_bfloat16,true,17>();
  result["control_sage_layout_18"] = output_type==0 ? sage_layout_resources<half,false,18>() : sage_layout_resources<nv_bfloat16,false,18>();
  result["expcast_sage_layout_18"] = output_type==0 ? sage_layout_resources<half,true,18>() : sage_layout_resources<nv_bfloat16,true,18>();
  result["control_sage_layout_19"] = output_type==0 ? sage_layout_resources<half,false,19>() : sage_layout_resources<nv_bfloat16,false,19>();
  result["expcast_sage_layout_19"] = output_type==0 ? sage_layout_resources<half,true,19>() : sage_layout_resources<nv_bfloat16,true,19>();
  result["control_sage_layout_20"] = output_type==0 ? sage_layout_resources<half,false,20>() : sage_layout_resources<nv_bfloat16,false,20>();
  result["expcast_sage_layout_20"] = output_type==0 ? sage_layout_resources<half,true,20>() : sage_layout_resources<nv_bfloat16,true,20>();
  result["control_sage_layout_21"] = output_type==0 ? sage_layout_resources<half,false,21>() : sage_layout_resources<nv_bfloat16,false,21>();
  result["expcast_sage_layout_21"] = output_type==0 ? sage_layout_resources<half,true,21>() : sage_layout_resources<nv_bfloat16,true,21>();
  result["control_sage_layout_22"] = output_type==0 ? sage_layout_resources<half,false,22>() : sage_layout_resources<nv_bfloat16,false,22>();
  result["expcast_sage_layout_22"] = output_type==0 ? sage_layout_resources<half,true,22>() : sage_layout_resources<nv_bfloat16,true,22>();
  result["control_sage_layout_23"] = output_type==0 ? sage_layout_resources<half,false,23>() : sage_layout_resources<nv_bfloat16,false,23>();
  result["expcast_sage_layout_23"] = output_type==0 ? sage_layout_resources<half,true,23>() : sage_layout_resources<nv_bfloat16,true,23>();
  result["control_sage_layout_24"] = output_type==0 ? sage_layout_resources<half,false,24>() : sage_layout_resources<nv_bfloat16,false,24>();
  result["expcast_sage_layout_24"] = output_type==0 ? sage_layout_resources<half,true,24>() : sage_layout_resources<nv_bfloat16,true,24>();
  result["control_sage_layout_25"] = output_type==0 ? sage_layout_resources<half,false,25>() : sage_layout_resources<nv_bfloat16,false,25>();
  result["expcast_sage_layout_25"] = output_type==0 ? sage_layout_resources<half,true,25>() : sage_layout_resources<nv_bfloat16,true,25>();
  result["control_sage_layout_26"] = output_type==0 ? sage_layout_resources<half,false,26>() : sage_layout_resources<nv_bfloat16,false,26>();
  result["expcast_sage_layout_26"] = output_type==0 ? sage_layout_resources<half,true,26>() : sage_layout_resources<nv_bfloat16,true,26>();
  result["control_sage_layout_27"] = output_type==0 ? sage_layout_resources<half,false,27>() : sage_layout_resources<nv_bfloat16,false,27>();
  result["expcast_sage_layout_27"] = output_type==0 ? sage_layout_resources<half,true,27>() : sage_layout_resources<nv_bfloat16,true,27>();
  result["control_sage_layout_28"] = output_type==0 ? sage_layout_resources<half,false,28>() : sage_layout_resources<nv_bfloat16,false,28>();
  result["expcast_sage_layout_28"] = output_type==0 ? sage_layout_resources<half,true,28>() : sage_layout_resources<nv_bfloat16,true,28>();
  result["control_sage_layout_29"] = output_type==0 ? sage_layout_resources<half,false,29>() : sage_layout_resources<nv_bfloat16,false,29>();
  result["expcast_sage_layout_29"] = output_type==0 ? sage_layout_resources<half,true,29>() : sage_layout_resources<nv_bfloat16,true,29>();
  return result;
}
