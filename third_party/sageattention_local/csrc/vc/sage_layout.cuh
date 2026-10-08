// Modified from pinned SageAttention v2.2.0 qk_int_sv_f8_cuda_sm90.cu.
// Control preserves the upstream QK/PV loop structure. Changes: independent
// scheduling switches, vc15 ExpCast, register cap and true -inf tail masking.
// Included inside namespace sage_vc after its TMA helpers; no upstream edits.
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

#pragma once

template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t NUM_THREADS, uint32_t head_dim, QuantGranularity Q_GRAN, QuantGranularity K_GRAN, typename DTypeOut, MaskMode mask_mode = MaskMode::kNone, bool return_lse = false, bool fuse_v_scale=false, bool ExpCast=false, bool EarlyV=false, bool PackFirst=false, bool OverlapHistory=true, int ProbabilityVariant=0, int RegisterLimit=168, bool CompactPV=false, int WorkReduction=0, bool FP8QK=false>
__global__ __maxnreg__(RegisterLimit) void sage_vc_attention_kernel_sage_layout(const __grid_constant__ CUtensorMap tensorMapQ,
                                        const __grid_constant__ CUtensorMap tensorMapK,
                                        const __grid_constant__ CUtensorMap tensorMapV,
                                        float *__restrict__ Q_scale, float *__restrict__ K_scale, float *__restrict__ V_scale,
                                        DTypeOut* O, float *__restrict__ Lse, uint32_t stride_bz_o, uint32_t stride_h_o, uint32_t stride_seq_o,
                                        const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
                                        float sm_scale)
{
#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("trap;");
#else
  static_assert(NUM_THREADS == 128);
  static_assert(!FP8QK || (CTA_Q==64 && CTA_K==128 &&
      Q_GRAN==QuantGranularity::kPerBlock && K_GRAN==QuantGranularity::kPerBlock &&
      (WorkReduction==0 || WorkReduction==2)), "FP8 QK uses block scales and FP32 max");
  static_assert(WorkReduction>=0 && WorkReduction<=10);
  static_assert(RegisterLimit==128 || RegisterLimit==144 || RegisterLimit==168 || RegisterLimit==192);
  static_assert(!ExpCast || !PackFirst);
  static_assert(CTA_Q <= CTA_K);
  static_assert(WorkReduction<4 || (ExpCast && OverlapHistory && !CompactPV));

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

  typename std::conditional<FP8QK,float,int32_t>::type RS[num_tiles_q][num_tiles_k][8];
  float RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2];
  float d[num_tiles_q][2];
  float probability_bias[num_tiles_q][2];
  constexpr bool carry_mass = WorkReduction==7 || WorkReduction==8;
  // Neutral initial tile lets the drain execute without a branch inside the
  // first QK batch: d*1 + 0 == d. This also avoids an undefined carried value.
  uint32_t carried_probability[num_tiles_q][num_tiles_pv_inner][4] = {};
  float carried_alpha[num_tiles_q][2];
  if constexpr (carry_mass) {
    #pragma unroll
    for (int i=0;i<num_tiles_q;++i) {carried_alpha[i][0]=1.f;carried_alpha[i][1]=1.f;}
  }

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
    if constexpr (ExpCast && WorkReduction==5) {
      probability_bias[fq][0]=__fmaf_rn(-m[fq][0],11.541560173034668f,119.6500015258789f);
      probability_bias[fq][1]=__fmaf_rn(-m[fq][1],11.541560173034668f,119.6500015258789f);
    }
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

    if constexpr (WorkReduction==7)
      vc_probability_drain_mass(carried_probability,carried_alpha,d);

    // compute QK^T
    wgmma::warpgroup_arrive();
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      int8_t *sQ_local = sQ + fq * 64 * head_dim;
      if constexpr (FP8QK) vc_wgmma_fp8_qk<0,head_dim>(RS[fq],sQ_local,sK);
      else wgmma::wgmma_s8s8s32<CTA_K, 0, head_dim>(RS[fq], sQ_local, sK);
#pragma unroll
      for (int k_it = 1; k_it < num_tiles_qk_inner; k_it++)
      {
        if constexpr (FP8QK) vc_wgmma_fp8_qk<1,head_dim>(RS[fq],&sQ_local[k_it*32],&sK[k_it*32]);
        else wgmma::wgmma_s8s8s32<CTA_K, 1, head_dim>(RS[fq], &sQ_local[k_it*32], &sK[k_it*32]);
      }
    }
    wgmma::warpgroup_commit_batch();
    if constexpr (WorkReduction==8)
      vc_probability_drain_mass(carried_probability,carried_alpha,d);
    wgmma::warpgroup_wait<0>();

    // load K
    if (threadIdx.x == 0)
    {
      expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_K);
      load_async_4D(sK, &tensorMapK, &barrier_K, 0, iter * CTA_K, kv_head_id, batch_id);
    }

    // Integer-max candidates delay conversion until after their max tree.
    float RS_f32[num_tiles_q][num_tiles_k][8];
    if constexpr (!(ExpCast && (WorkReduction==1 || WorkReduction>=3))) {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          if constexpr (FP8QK) RS_f32[fq][fk][k] = RS[fq][fk][k];
          else RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
        }
      }
    }

    }
    uint32_t local_probability[num_tiles_q][num_tiles_pv_inner][4];
    float tile_mass[num_tiles_q][2], local_alpha[num_tiles_q][2];
    auto& RS_f8 = carry_mass ? carried_probability : local_probability;
    auto& probability_alpha = carry_mass ? carried_alpha : local_alpha;
    bool probability_rescale=false;
    if constexpr (ExpCast) {
      // Full-tile integer reduction is exact for INT8 D<=128. Encoding and
      // FP32 history arithmetic follow vc16; the tail keeps its original max.
      if constexpr (WorkReduction==4 || WorkReduction==5)
        vc_probability_prepare_maxstate<(WorkReduction==5)>(RS,m,sm_scale,RS_f8,
            probability_alpha,tile_mass,probability_bias,probability_rescale);
      else if constexpr (WorkReduction==1 || WorkReduction>=3)
        vc_probability_prepare_integer<(WorkReduction==6),carry_mass,(WorkReduction>=9),(WorkReduction==10)>(RS,m,sm_scale,RS_f8,probability_alpha,tile_mass);
      else
        vc_probability_prepare_variant<ProbabilityVariant>(RS_f32,m,sm_scale,RS_f8,probability_alpha,tile_mass);
      if constexpr (!OverlapHistory)
        vc_probability_finish_work<(WorkReduction>=2)>(RO,d,probability_alpha,tile_mass);
    } else {
    update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(RS_f32, RO, m, d, sm_scale);

      if constexpr (PackFirst) RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);
    // accumulate d on thread basis
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
        d[fq][0] += (RS_f32[fq][fk][0] + RS_f32[fq][fk][1] + RS_f32[fq][fk][4] + RS_f32[fq][fk][5]);
        d[fq][1] += (RS_f32[fq][fk][2] + RS_f32[fq][fk][3] + RS_f32[fq][fk][6] + RS_f32[fq][fk][7]);
      }
    }

      if constexpr (!PackFirst) RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);
    }

    // wait for V
    wait(&barrier_V, p);

    // A separate N64 accumulator reduces PV live registers without changing
    // the four K32 products per output element or two-level FP32 accumulation.
    constexpr uint32_t pv_parts = (CompactPV && head_dim==128) ? 2 : 1;
#pragma unroll
    for (uint32_t part = 0; part < pv_parts; ++part) {
    float RO_temp[num_tiles_q][num_tiles_v / pv_parts][8];
    wgmma::warpgroup_arrive();
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
    // Scale history while PV is live; carry_mass defers d to the next QK.
    if constexpr (ExpCast && OverlapHistory) {
      if (part==0) {
        if constexpr (carry_mass)
          vc_probability_scale_history(RO,probability_alpha);
        else if constexpr (WorkReduction==4 || WorkReduction==5)
          vc_probability_finish_maxstate(RO,d,probability_alpha,tile_mass,probability_rescale);
        else vc_probability_finish_work<(WorkReduction>=2)>(RO,d,probability_alpha,tile_mass);
      }
    }
    wgmma::warpgroup_wait<0>();

    // load V
    if (EarlyV && part + 1 == pv_parts && threadIdx.x == 0)
    {
      expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_V);
      load_async_4D(sV, &tensorMapV, &barrier_V, iter * CTA_K, 0, kv_head_id, batch_id);
    }

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v / pv_parts; fv++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RO[fq][part * (num_tiles_v / pv_parts) + fv][k] += RO_temp[fq][fv][k];
        }
      }
    }
    } // all PV channel fragments completed; sV can now be reused

    // load V
    if (!EarlyV && threadIdx.x == 0)
    {
      expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_V);
      load_async_4D(sV, &tensorMapV, &barrier_V, iter * CTA_K, 0, kv_head_id, batch_id);
    }
  }

  {
    p ^= 1;

    float dequant_scale = q_scale * K_scale[k_scale_idx + (num_iterations - 1) * k_scale_advance_offset];
    sm_scale = original_sm_scale;

    // wait for K
    wait(&barrier_K, p);

    if constexpr (WorkReduction==7)
      vc_probability_drain_mass(carried_probability,carried_alpha,d);

    // compute QK^T
    wgmma::warpgroup_arrive();
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      int8_t *sQ_local = sQ + fq * 64 * head_dim;
      if constexpr (FP8QK) vc_wgmma_fp8_qk<0,head_dim>(RS[fq],sQ_local,sK);
      else wgmma::wgmma_s8s8s32<CTA_K, 0, head_dim>(RS[fq], sQ_local, sK);
#pragma unroll
      for (int k_it = 1; k_it < num_tiles_qk_inner; k_it++)
      {
        if constexpr (FP8QK) vc_wgmma_fp8_qk<1,head_dim>(RS[fq],&sQ_local[k_it*32],&sK[k_it*32]);
        else wgmma::wgmma_s8s8s32<CTA_K, 1, head_dim>(RS[fq], &sQ_local[k_it*32], &sK[k_it*32]);
      }
    }
    wgmma::warpgroup_commit_batch();
    if constexpr (WorkReduction==8)
      vc_probability_drain_mass(carried_probability,carried_alpha,d);
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
          if constexpr (FP8QK) RS_f32[fq][fk][k] = RS[fq][fk][k] * dequant_scale;
          else RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
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
    if constexpr (ExpCast) {
      // vc15's exact encoder and FP32 recurrence, with a controlled schedule.
      if constexpr (carry_mass)
        vc_probability_prepare<true,true,true,true,true>(RS_f32,m,sm_scale,RS_f8,probability_alpha,tile_mass);
      else vc_probability_prepare_variant<ProbabilityVariant>(RS_f32,m,sm_scale,RS_f8,probability_alpha,tile_mass);
      if constexpr (!OverlapHistory)
        vc_probability_finish_work<(WorkReduction>=2)>(RO,d,probability_alpha,tile_mass);
    } else {
    update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(RS_f32, RO, m, d, sm_scale);

      if constexpr (PackFirst) RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);
    // accumulate d on thread basis
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
        d[fq][0] += (RS_f32[fq][fk][0] + RS_f32[fq][fk][1] + RS_f32[fq][fk][4] + RS_f32[fq][fk][5]);
        d[fq][1] += (RS_f32[fq][fk][2] + RS_f32[fq][fk][3] + RS_f32[fq][fk][6] + RS_f32[fq][fk][7]);
      }
    }

      if constexpr (!PackFirst) RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);
    }

    // wait for V
    wait(&barrier_V, p);

    // A separate N64 accumulator reduces PV live registers without changing
    // the four K32 products per output element or two-level FP32 accumulation.
    constexpr uint32_t pv_parts = (CompactPV && head_dim==128) ? 2 : 1;
#pragma unroll
    for (uint32_t part = 0; part < pv_parts; ++part) {
    float RO_temp[num_tiles_q][num_tiles_v / pv_parts][8];
    wgmma::warpgroup_arrive();
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
    // The final carry_mass denominator is drained after this PV completes.
    if constexpr (ExpCast && OverlapHistory) {
      if (part==0) {
        if constexpr (carry_mass) vc_probability_scale_history(RO,probability_alpha);
        else vc_probability_finish_work<(WorkReduction>=2)>(RO,d,probability_alpha,tile_mass);
      }
    }
    wgmma::warpgroup_wait<0>();

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v / pv_parts; fv++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RO[fq][part * (num_tiles_v / pv_parts) + fv][k] += RO_temp[fq][fv][k];
        }
      }
    }
    } // all PV channel fragments completed; sV can now be reused
    // No following QK: drain the final masked tile after PV and history add.
    if constexpr (carry_mass) vc_probability_drain_mass(RS_f8,probability_alpha,d);
  }

  normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kCudaCore>(RO, m, d);

  if constexpr (fuse_v_scale)
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
