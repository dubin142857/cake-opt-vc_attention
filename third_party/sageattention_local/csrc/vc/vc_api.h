// VC additions to SageAttention, Apache-2.0; see NOTICE.
#pragma once
#include <torch/extension.h>
#include <map>
#include <string>
std::vector<torch::Tensor> vc_prepare_values(torch::Tensor v, bool channel_scale);
torch::Tensor vc_expcast_codes(torch::Tensor scores);
torch::Tensor vc_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor qs, torch::Tensor ks, torch::Tensor vs, torch::Tensor means,
    bool expcast, bool block_values, int64_t output_type, double scale, bool optimized, bool packed_probability, bool compact_values, bool pipeline_probability, bool interleaved_probability);

std::map<std::string, std::map<std::string, int64_t>> vc_kernel_resources(int64_t output_type);
std::vector<torch::Tensor> vc_group_values_fused(torch::Tensor v, int64_t clusters, int64_t iterations);
