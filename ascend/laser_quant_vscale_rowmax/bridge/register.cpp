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
#include <torch/library.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"

namespace {
constexpr std::string_view QUANT_LASER_API = "aclnnVcLaserQuantVscaleRowmaxAttention";

void CheckInput(const at::Tensor &tensor, const at::Tensor &query,
                at::ScalarType dtype, const char *name)
{
    TORCH_CHECK(tensor.device() == query.device(), name, " must share query's NPU device");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous BNSD/scale data");
    TORCH_CHECK(!tensor.requires_grad(), "Quantized Laser is inference only");
}

at::Tensor QuantLaser(const at::Tensor &query, const at::Tensor &key,
                     const at::Tensor &value, const at::Tensor &query_scale,
                     const at::Tensor &key_scale, double scale, int64_t keys,
                     bool expcast)
{
    TORCH_CHECK(!expcast,
                "Vector-scale Laser currently supports original-exp mode only (expcast=false)");
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1,
                "Quantized Laser requires NPU tensors");
    CheckInput(query, query, at::kChar, "query");
    CheckInput(key, query, at::kChar, "key");
    CheckInput(value, query, at::kHalf, "value");
    CheckInput(query_scale, query, at::kFloat, "query_scale");
    CheckInput(key_scale, query, at::kFloat, "key_scale");
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
                "Quantized Laser requires rank-four BNSD query, key and value");
    const int64_t batch = query.size(0);
    const int64_t heads = query.size(1);
    const int64_t queries_padded = query.size(2);
    const int64_t keys_padded = key.size(2);
    TORCH_CHECK(batch > 0 && heads > 0 && queries_padded > 0 && keys_padded > 0,
                "Quantized Laser dimensions must be positive");
    TORCH_CHECK(query.size(3) == 128 && key.size(3) == 128,
                "Quantized Laser native head dimension must be padded to 128");
    TORCH_CHECK(key.size(0) == batch && key.size(1) == heads && value.sizes() == key.sizes(),
                "Quantized Laser requires matching MHA batch/heads and key/value shapes");
    TORCH_CHECK(queries_padded % 256 == 0 && keys_padded % 256 == 0,
                "Quantized Laser native sequence dimensions must be padded to 256");
    TORCH_CHECK(keys > 0 && keys <= keys_padded && keys_padded - keys < 256,
                "keys must identify the valid tail of the padded key tensor");
    TORCH_CHECK(batch <= std::numeric_limits<int32_t>::max() &&
                heads <= std::numeric_limits<int32_t>::max() &&
                queries_padded <= std::numeric_limits<int32_t>::max() &&
                keys_padded <= std::numeric_limits<int32_t>::max(),
                "Quantized Laser dimensions exceed the native int32 tiling contract");
    TORCH_CHECK(query_scale.sizes() == at::IntArrayRef({batch, heads, queries_padded / 128}),
                "query_scale must be FP32 [B,H,Qpad/128]");
    TORCH_CHECK(key_scale.sizes() == at::IntArrayRef({batch, heads, keys_padded / 128}),
                "key_scale must be FP32 [B,H,Kpad/128]");
    TORCH_CHECK(std::isfinite(scale) && scale > 0 &&
                scale <= std::numeric_limits<float>::max(), "scale must be finite and positive");

    // Keep every ACLNN attribute as an lvalue, matching the upstream helper.
    const at::Tensor drop_mask = at::Tensor();
    double scale_value = scale;
    int64_t head_num = heads;
    std::string input_layout = "BNSD";
    double keep_prob = 1.0;
    int64_t pre_tokens = keys_padded - keys;
    int64_t next_tokens = 1;
    bool is_high_precision = true;
    bool use_expcast = expcast;
    auto statistics = at_npu::native::empty_with_format(
        {batch, heads, queries_padded}, query.options().dtype(at::kFloat),
        at_npu::native::get_npu_format(query));
    auto output = at_npu::native::empty_with_format(
        query.sizes(), query.options().dtype(at::kFloat),
        at_npu::native::get_npu_format(query));

    // The pinned EXEC_NPU_CMD helper preserves torch_npu's current stream and
    // resolves the unique quantized native API. No standalone ACL stream.
    EXEC_NPU_CMD<QUANT_LASER_API>(query, key, value, query_scale, key_scale, drop_mask,
        scale_value, head_num, input_layout, keep_prob, pre_tokens, next_tokens,
        is_high_precision, use_expcast, statistics, output);
    return output;
}
}  // namespace

TORCH_LIBRARY(vc_laser_quant_vscale_rowmax, module)
{
    module.def("la(Tensor q, Tensor k, Tensor v, Tensor q_scale, Tensor k_scale, "
               "float scale, int keys, bool expcast) -> Tensor");
}

TORCH_LIBRARY_IMPL(vc_laser_quant_vscale_rowmax, PrivateUse1, module)
{
    module.impl("la", TORCH_FN(QuantLaser));
}
