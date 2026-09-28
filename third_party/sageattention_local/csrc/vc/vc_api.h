// VC additions to SageAttention, Apache-2.0; see NOTICE.
#pragma once
#include <torch/extension.h>
std::vector<torch::Tensor> vc_prepare_values(torch::Tensor v);
torch::Tensor vc_expcast_codes(torch::Tensor scores);
torch::Tensor vc_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor qs, torch::Tensor ks, torch::Tensor vs, torch::Tensor means,
    bool expcast, bool block_values, int64_t output_type, double scale, bool optimized);
