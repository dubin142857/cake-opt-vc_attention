/**
 * Adapted from MindIE-SD csrc/plugin/la.cpp at
 * 71098c20ff90838d1fd1ca0a40515c751a3869b0.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <tuple>
#include <ATen/Functions.h>
#include <torch/library.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"


namespace {
constexpr std::string_view CUBE_LASER_API = "aclnnVcLaserCubeVscaleProbeAttention";

void CheckInput(const at::Tensor &tensor, const at::Tensor &query,
                at::ScalarType dtype, const char *name)
{
    TORCH_CHECK(tensor.device() == query.device(), name, " must share query's NPU device");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous BNSD/scale data");
    TORCH_CHECK(!tensor.requires_grad(), "Cube diagnostics are inference only");
}

std::tuple<at::Tensor, at::Tensor> CubeLaser(const at::Tensor &query, const at::Tensor &key,
                     const at::Tensor &value,
                     const at::Tensor &query_scale, const at::Tensor &key_scale,
                     double scale, int64_t keys, bool expcast, int64_t probe_mode)
{
    TORCH_CHECK(probe_mode >= 0 && probe_mode <= 2, "probe_mode must be 0, 1 or 2");
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1,
                "Cube diagnostics require NPU tensors");
    TORCH_CHECK(!expcast, "Cube diagnostics require the original Laser exponential");
    CheckInput(query, query, at::kChar, "query");
    CheckInput(key, query, at::kChar, "key");
    CheckInput(query_scale, query, at::kFloat, "query_scale");
    CheckInput(key_scale, query, at::kFloat, "key_scale");
    CheckInput(value, query, at::kHalf, "value");
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
                "Cube diagnostics require rank-four BNSD query, key and value");
    const int64_t batch = query.size(0), heads = query.size(1);
    const int64_t queries_padded = query.size(2), keys_padded = key.size(2);
    TORCH_CHECK(batch > 0 && heads > 0 && queries_padded > 0 && keys_padded > 0,
                "Cube diagnostic dimensions must be positive");
    TORCH_CHECK(query.size(3) == 128 && key.size(3) == 128,
                "Native head dimension must be padded to 128");
    TORCH_CHECK(key.size(0) == batch && key.size(1) == heads && value.sizes() == key.sizes(),
                "Cube diagnostics require matching MHA batch/heads and key/value shapes");
    TORCH_CHECK(queries_padded % 256 == 0 && keys_padded % 256 == 0,
                "Native sequence dimensions must be padded to 256");
    TORCH_CHECK(keys > 0 && keys <= keys_padded && keys_padded - keys < 256,
                "keys must identify the valid tail of the padded key tensor");
    TORCH_CHECK(batch <= std::numeric_limits<int32_t>::max() && heads <= std::numeric_limits<int32_t>::max() &&
                queries_padded <= std::numeric_limits<int32_t>::max() && keys_padded <= std::numeric_limits<int32_t>::max(),
                "Dimensions exceed the native int32 tiling contract");
    TORCH_CHECK(query_scale.sizes() == at::IntArrayRef({batch, heads, queries_padded / 128}),
                "query_scale must be FP32 [B,H,Qpad/128]");
    TORCH_CHECK(key_scale.sizes() == at::IntArrayRef({batch, heads, keys_padded / 128}),
                "key_scale must be FP32 [B,H,Kpad/128]");
    TORCH_CHECK(std::isfinite(scale) && scale > 0 && scale <= std::numeric_limits<float>::max(),
                "scale must be finite and positive");
    const at::Tensor drop_mask = at::Tensor();
    double scale_value = scale;
    int64_t head_num = heads;
    std::string input_layout = "BNSD";
    double keep_prob = 1.0;
    int64_t pre_tokens = keys_padded - keys;
    int64_t next_tokens = 1;
    bool is_high_precision = true;
    int64_t cube_probe_mode = probe_mode;
    auto statistics = at_npu::native::empty_with_format(
        {batch, heads, queries_padded}, query.options().dtype(at::kFloat), at_npu::native::get_npu_format(query));
    auto output = at_npu::native::empty_with_format(
        query.sizes(), query.options().dtype(at::kFloat), at_npu::native::get_npu_format(query));
    // Identical trace allocation and deterministic initialization in all modes.
    auto trace = at::zeros({4, 2, 3, 24}, query.options().dtype(at::kLong));
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(trace.data_ptr()) % 64 == 0,
                "Cube trace requires 64-byte cache-line alignment");
    bool use_expcast = false;
    EXEC_NPU_CMD<CUBE_LASER_API>(query, key, value, query_scale, key_scale, drop_mask,
        scale_value, head_num, input_layout, keep_prob, pre_tokens, next_tokens,
        is_high_precision, use_expcast, cube_probe_mode, statistics, output, trace);
    return std::make_tuple(output, trace);
}
} // namespace

TORCH_LIBRARY(vc_laser_cube_vscale_probe, module)
{
    module.def("la(Tensor q, Tensor k, Tensor v, Tensor q_scale, Tensor k_scale, "
               "float scale, int keys, bool expcast, int probe_mode) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(vc_laser_cube_vscale_probe, PrivateUse1, module)
{
    module.impl("la", TORCH_FN(CubeLaser));
}
