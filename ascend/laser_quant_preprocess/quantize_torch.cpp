#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "quantize_contract.h"
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include <algorithm>
#include <climits>
#include <string>
#include <tuple>

extern "C" {
uint32_t aclrtlaunch_vc_laser_quantize_fp16(uint32_t, void*, void*, void*, void*,
    uint32_t, uint32_t, uint32_t, uint64_t, uint32_t);
decltype(aclrtlaunch_vc_laser_quantize_fp16) aclrtlaunch_vc_laser_quantize_bf16;
}

namespace {
using namespace vc_laser_quant_preprocess;
void Check(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 4, "Expected BNSD rank-four input");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "Expected FP16/BF16 input");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous ND");
    TORCH_CHECK(!x.requires_grad(), "Inference-only quantization");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0,
                "Empty batch, heads or sequence not supported");
    TORCH_CHECK(x.size(2) <= INT32_MAX - 256, "Sequence indexing limit exceeded");
    TORCH_CHECK(x.size(3) == 64 || x.size(3) == 128, "D must be 64 or 128");
}
std::tuple<at::Tensor, at::Tensor> Allocate(const at::Tensor& x) {
    Check(x);
    const auto padded = (x.size(2) + 255) / 256 * 256;
    auto output = at::empty({x.size(0), x.size(1), padded, 128}, x.options().dtype(at::kChar));
    auto scales = at::empty({x.size(0), x.size(1), padded / 128}, x.options().dtype(at::kFloat));
    return {output, scales};
}
std::tuple<at::Tensor, at::Tensor> Quantize(const at::Tensor& x) {
    Check(x);
    TORCH_CHECK(x.is_privateuseone(), "Ascend NPU input required");
    c10::OptionalDeviceGuard guard(x.device());
    TORCH_CHECK(at_npu::native::get_npu_format(x) == ACL_FORMAT_ND,
                "Input must have physical ACL_FORMAT_ND; use npu_format_cast(x, 2)");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 32 == 0,
                "Input storage offset must preserve 32-byte alignment; clone this view");
    const char* soc = aclrtGetSocName();
    TORCH_CHECK(soc && std::string(soc) == VC_QUANT_SOC_VERSION,
                "Quantizer built for ", VC_QUANT_SOC_VERSION, "; runtime SoC: ",
                soc ? soc : "unknown");
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform && platform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND910B,
                "Quantizer requires A2/A3 DAV-2201");
    uint64_t ub = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    TORCH_CHECK(ub >= kUbBytes, "Insufficient UB for quantization tile");
    const auto padded = (x.size(2) + 255) / 256 * 256;
    auto output = at_npu::native::empty_with_format(
        {x.size(0), x.size(1), padded, 128}, x.options().dtype(at::kChar), ACL_FORMAT_ND);
    auto scales = at_npu::native::empty_with_format(
        {x.size(0), x.size(1), padded / 128}, x.options().dtype(at::kFloat), ACL_FORMAT_ND);
    const uint64_t tasks = scales.numel();
    const uint64_t groups = (tasks + kScaleGroup - 1) / kScaleGroup;
    const uint32_t cores = std::min<uint64_t>(platform->GetCoreNumAiv(), groups);
    TORCH_CHECK(cores > 0, "No AIV cores available");
    // Drain the PyTorch host queue before the direct ACL launch. No device-wide
    // synchronization, host tensor copy or data-dependent host read is needed.
    auto current = c10_npu::getCurrentNPUStream(x.device().index());
    c10_npu::NPUCachingAllocator::recordStream(x.storage().data_ptr(), current);
    auto stream = current.stream(true);
    auto launch = x.scalar_type() == at::kHalf ? aclrtlaunch_vc_laser_quantize_fp16
                                              : aclrtlaunch_vc_laser_quantize_bf16;
    auto ret = launch(cores, stream, x.data_ptr(), output.data_ptr(), scales.data_ptr(),
                      x.size(2), x.size(3), output.size(2), tasks, cores);
    TORCH_CHECK(ret == ACL_SUCCESS, "Fused block quantization launch failed: ", ret);
    return {output, scales};
}
std::string Metadata() {
    return "{\"contract_version\":1,\"soc\":\"" VC_QUANT_SOC_VERSION
           "\",\"qk_block\":128,\"sequence_pad\":256,\"padded_dim\":128,"
           "\"rounding\":\"FP32 Div then INT32 RNE\",\"kernel\":\"pure AIV\","
           "\"ub_bytes\":" + std::to_string(kUbBytes) + "}";
}
}

TORCH_LIBRARY(vc_laser_quant_preprocess, m) {
    m.def("quantize(Tensor x) -> (Tensor, Tensor)");
    m.def("metadata() -> str", TORCH_FN(Metadata));
}
TORCH_LIBRARY_IMPL(vc_laser_quant_preprocess, PrivateUse1, m) {
    m.impl("quantize", TORCH_FN(Quantize));
}
TORCH_LIBRARY_IMPL(vc_laser_quant_preprocess, Meta, m) {
    m.impl("quantize", TORCH_FN(Allocate));
}
