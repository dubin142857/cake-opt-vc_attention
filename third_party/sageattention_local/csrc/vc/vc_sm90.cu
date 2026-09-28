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

template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t NUM_THREADS, uint32_t head_dim, QuantGranularity Q_GRAN, QuantGranularity K_GRAN, typename DTypeOut, MaskMode mask_mode = MaskMode::kNone, bool return_lse = false, bool fuse_v_scale=false, bool ExpCast=false, bool BlockValues=false, bool Optimized=false, bool ChannelScale=false, bool PackedProbability=false>
__global__ void sage_vc_attention_kernel(const __grid_constant__ CUtensorMap tensorMapQ,
                                        const __grid_constant__ CUtensorMap tensorMapK,
                                        const __grid_constant__ CUtensorMap tensorMapV,
                                        float *__restrict__ Q_scale, float *__restrict__ K_scale, float *__restrict__ V_scale, const half* __restrict__ V_mean,
                                        DTypeOut* O, float *__restrict__ Lse, uint32_t stride_bz_o, uint32_t stride_h_o, uint32_t stride_seq_o,
                                        const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
                                        float sm_scale)
{
#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("trap;");
#else
  static_assert(NUM_THREADS == 128);
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
    float tile_mass[num_tiles_q][2];
    vc_probability<ExpCast, BlockValues, Optimized, PackedProbability>(RS_f32, RO, m, d, sm_scale, RS_f8, tile_mass);

    // wait for V
    wait(&barrier_V, p);

    float RO_temp[num_tiles_q][num_tiles_v][8];
    // Keep Sage's two-level FP32 accumulation: FP8 WGMMA must not
    // repeatedly round the long-lived output history or mean compensation.
    wgmma::warpgroup_arrive();
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      wgmma::wgmma_f8f8f32<head_dim, 0, CTA_K>(RO_temp[fq], RS_f8[fq][0], &sV[0]);
#pragma unroll
      for (uint32_t v_it = 1; v_it < num_tiles_pv_inner; v_it++)
      {
        wgmma::wgmma_f8f8f32<head_dim, 1, CTA_K>(RO_temp[fq], RS_f8[fq][v_it], &sV[v_it * 32]);
      }
    }

    wgmma::warpgroup_commit_batch();
    wgmma::warpgroup_wait<0>();

    // PV has finished reading sV. Start the next TMA transfer before the
    // per-block mean/scale restoration, which only touches registers/global
    // metadata. Keep the existing barrier/phase for the next iteration.
    if (threadIdx.x == 0)
    {
      expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_V);
      load_async_4D(sV, &tensorMapV, &barrier_V, iter * CTA_K, 0, kv_head_id, batch_id);
    }

    vc_accumulate<BlockValues, Optimized, ChannelScale>(RO, RO_temp, tile_mass, V_scale, V_mean,
        batch_id * num_qo_heads + head_id, iter - 1, div_ceil(kv_len, CTA_K), head_dim, lane_id);
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
    float tile_mass[num_tiles_q][2];
    vc_probability<ExpCast, BlockValues, Optimized, PackedProbability>(RS_f32, RO, m, d, sm_scale, RS_f8, tile_mass);

    // wait for V
    wait(&barrier_V, p);

    float RO_temp[num_tiles_q][num_tiles_v][8];
    // Keep Sage's two-level FP32 accumulation: FP8 WGMMA must not
    // repeatedly round the long-lived output history or mean compensation.
    wgmma::warpgroup_arrive();
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      wgmma::wgmma_f8f8f32<head_dim, 0, CTA_K>(RO_temp[fq], RS_f8[fq][0], &sV[0]);
#pragma unroll
      for (uint32_t v_it = 1; v_it < num_tiles_pv_inner; v_it++)
      {
        wgmma::wgmma_f8f8f32<head_dim, 1, CTA_K>(RO_temp[fq], RS_f8[fq][v_it], &sV[v_it * 32]);
      }
    }

    wgmma::warpgroup_commit_batch();
    wgmma::warpgroup_wait<0>();

    vc_accumulate<BlockValues, Optimized, ChannelScale>(RO, RO_temp, tile_mass, V_scale, V_mean,
        batch_id * num_qo_heads + head_id, num_iterations - 1, div_ceil(kv_len, CTA_K), head_dim, lane_id);
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


} // namespace sage_vc

namespace {
template<int Dim, typename Out, bool ExpCast, bool BlockValues, bool Optimized, bool ChannelScale=false, bool PackedProbability=false>
void launch_vc(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor qs,
    torch::Tensor ks, torch::Tensor vs, torch::Tensor means, torch::Tensor out, float scale) {
  auto qm = sage_vc::create_tensor_map_4D<64, Dim>(reinterpret_cast<int8_t*>(q.data_ptr()),
      q.size(0), q.size(1), q.size(2), Dim, q.stride(0), q.stride(1), q.stride(2));
  auto km = sage_vc::create_tensor_map_4D<128, Dim>(reinterpret_cast<int8_t*>(k.data_ptr()),
      k.size(0), k.size(1), k.size(2), Dim, k.stride(0), k.stride(1), k.stride(2));
  auto vm = sage_vc::create_tensor_map_4D<Dim, 128>(reinterpret_cast<int8_t*>(v.data_ptr()),
      v.size(0), v.size(1), Dim, v.size(3), v.stride(0), v.stride(1), v.stride(2));
  auto kernel = sage_vc::sage_vc_attention_kernel<64, 128, 128, Dim,
      QuantGranularity::kPerThread, QuantGranularity::kPerThread, Out, MaskMode::kNone, false, true, ExpCast, BlockValues, Optimized, ChannelScale, PackedProbability>;
  constexpr int shared_bytes = (64 + 128 + 128) * Dim;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  dim3 grid((q.size(2) + 63) / 64, q.size(1), q.size(0));
  kernel<<<grid, 128, shared_bytes, c10::cuda::getCurrentCUDAStream(q.get_device()).stream()>>>(
      qm, km, vm, qs.data_ptr<float>(), ks.data_ptr<float>(), vs.data_ptr<float>(),
      BlockValues ? reinterpret_cast<const half*>(means.data_ptr()) : nullptr,
      reinterpret_cast<Out*>(out.data_ptr()), nullptr, out.stride(0), out.stride(1), out.stride(2),
      q.size(2), k.size(2), 1, scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<int Dim, typename Out, bool Optimized, bool PackedProbability>
void dispatch_vc(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor qs,
    torch::Tensor ks, torch::Tensor vs, torch::Tensor means, torch::Tensor out, float scale, bool cast, bool smooth) {
  if (smooth && vs.dim() == 3) {
    if (cast) launch_vc<Dim, Out, true, true, Optimized, true, PackedProbability>(q,k,v,qs,ks,vs,means,out,scale);
    else launch_vc<Dim, Out, false, true, Optimized, true>(q,k,v,qs,ks,vs,means,out,scale);
  }
  else if (cast && smooth) launch_vc<Dim, Out, true, true, Optimized, false, PackedProbability>(q,k,v,qs,ks,vs,means,out,scale);
  else if (cast) launch_vc<Dim, Out, true, false, Optimized, false, PackedProbability>(q,k,v,qs,ks,vs,means,out,scale);
  else if (smooth) launch_vc<Dim, Out, false, true, Optimized>(q,k,v,qs,ks,vs,means,out,scale);
  else launch_vc<Dim, Out, false, false, false>(q,k,v,qs,ks,vs,means,out,scale);
}
template<bool Optimized, bool PackedProbability=false>
void dispatch_output(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor qs,
    torch::Tensor ks, torch::Tensor vs, torch::Tensor means, torch::Tensor out, float scale,
    int64_t output_type, bool cast, bool smooth) {
  if (q.size(3) == 64) {
    if (output_type == 0) dispatch_vc<64, half, Optimized, PackedProbability>(q,k,v,qs,ks,vs,means,out,scale,cast,smooth);
    else dispatch_vc<64, nv_bfloat16, Optimized, PackedProbability>(q,k,v,qs,ks,vs,means,out,scale,cast,smooth);
  } else {
    if (output_type == 0) dispatch_vc<128, half, Optimized, PackedProbability>(q,k,v,qs,ks,vs,means,out,scale,cast,smooth);
    else dispatch_vc<128, nv_bfloat16, Optimized, PackedProbability>(q,k,v,qs,ks,vs,means,out,scale,cast,smooth);
  }
}
}

torch::Tensor vc_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor qs, torch::Tensor ks, torch::Tensor vs, torch::Tensor means,
    bool expcast, bool block_values, int64_t output_type, double scale, bool optimized, bool packed_probability) {
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
  const c10::cuda::CUDAGuard guard(q.device());
  int major, minor;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, q.get_device()));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, q.get_device()));
  TORCH_CHECK(major == 9 && minor == 0, "Sage VC native attention requires Hopper; no fallback");
  auto out = torch::empty(q.sizes(), q.options().dtype(output_type == 0 ? at::kHalf : at::kBFloat16));
  if (optimized && packed_probability) dispatch_output<true, true>(q,k,v,qs,ks,vs,means,out,scale,output_type,expcast,block_values);
  else if (optimized) dispatch_output<true>(q,k,v,qs,ks,vs,means,out,scale,output_type,expcast,block_values);
  else dispatch_output<false>(q,k,v,qs,ks,vs,means,out,scale,output_type,expcast,block_values);
  return out;
}
