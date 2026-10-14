#include "acl/acl.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include "vc_tiling.h"

extern "C" uint32_t aclrtlaunch_vc910_forward(uint32_t, void*, void*, void*, void*,
    void*, void*, void*, void*, void*, void*, void*, void*, void*);

namespace {
using namespace vc910;

void CheckTensor(const at::Tensor& x, const at::Tensor& q, at::ScalarType type, const char* name) {
    TORCH_CHECK(x.device() == q.device(), name, " must share Q's device");
    TORCH_CHECK(x.scalar_type() == type, name, " has wrong dtype");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous ND");
    TORCH_CHECK(!x.requires_grad(), "Inference-only operator");
}

void Check(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
           const at::Tensor& qs, const at::Tensor& ks, const at::Tensor& vs,
           const at::Tensor& lut, int64_t queries, int64_t keys, double scale) {
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "HND rank-four payload required");
    CheckTensor(q, q, at::kChar, "Q"); CheckTensor(k, q, at::kChar, "K");
    CheckTensor(v, q, at::kHalf, "V"); CheckTensor(qs, q, at::kFloat, "Q scale");
    CheckTensor(ks, q, at::kFloat, "K scale"); CheckTensor(vs, q, at::kFloat, "V scale");
    CheckTensor(lut, q, at::kFloat, "LUT");
    const auto b = q.size(0), h = q.size(1), d = q.size(3);
    TORCH_CHECK(b > 0 && h > 0 && b <= UINT32_MAX && h <= UINT32_MAX, "Invalid B/H");
    TORCH_CHECK(d == 64 || d == 128, "D must be 64 or 128");
    TORCH_CHECK(queries > 0 && keys > 0 && queries < INT32_MAX - M && keys < INT32_MAX - N,
                "Sequence length outside supported range");
    const auto qp = (queries + M - 1) / M * M, kp = (keys + N - 1) / N * N;
    TORCH_CHECK(q.size(2) == qp && k.size(0) == b && k.size(1) == h && k.size(2) == kp && k.size(3) == d,
                "Q/K padding or MHA shape mismatch");
    TORCH_CHECK(v.sizes() == k.sizes(), "K/V shape mismatch");
    TORCH_CHECK(qs.sizes() == at::IntArrayRef({b, h, qp / M, 8}), "Q scales must be [B,H,Qpad/32,8]");
    TORCH_CHECK(ks.sizes() == at::IntArrayRef({b, h, kp / N, 8}), "K scales must be [B,H,Kpad/128,8]");
    TORCH_CHECK(vs.sizes() == at::IntArrayRef({b, h, d}), "V scales must be [B,H,D]");
    TORCH_CHECK(lut.dim() == 1 && lut.numel() == 128, "LUT must contain 128 FP32 entries");
    TORCH_CHECK(std::isfinite(scale) && scale > 0 && scale <= std::numeric_limits<float>::max(), "Invalid scale");
}

at::Tensor CubeTiling(platform_ascendc::PlatformAscendC* platform, bool qk, int d,
                      uint64_t l1, uint64_t l0c, const at::TensorOptions& options) {
    using namespace matmul_tiling;
    MatmulApiTiling api(*platform);
    api.SetAType(TPosition::GM, CubeFormat::ND, qk ? DataType::DT_INT8 : DataType::DT_FLOAT16, false);
    api.SetBType(TPosition::GM, CubeFormat::ND, qk ? DataType::DT_INT8 : DataType::DT_FLOAT16, qk);
    api.SetCType(TPosition::GM, CubeFormat::ND, qk ? DataType::DT_INT32 : DataType::DT_FLOAT);
    api.SetBiasType(TPosition::GM, CubeFormat::ND, qk ? DataType::DT_INT32 : DataType::DT_FLOAT);
    const int n = qk ? N : d, k = qk ? d : N;
    api.SetShape(M, n, k);
    api.SetOrgShape(M, n, k);
    api.SetFixSplit(M, n, k);
    api.SetBias(false);
    // Two sequential local matmul objects share the physical Cube memories.
    api.SetBufferSpace(l1 / 2, l0c / 2, -1);
    optiling::TCubeTiling t;
    TORCH_CHECK(api.GetTiling(t) == 0, "CANN MatmulApiTiling failed for ", qk ? "INT8 QK" : "FP16 PV");
    t.set_usedCoreNum(1);
    t.set_singleCoreM(M); t.set_singleCoreN(n); t.set_singleCoreK(k);
    t.set_dbL0A(1); t.set_dbL0B(1); t.set_dbL0C(1);
    t.set_shareMode(0); t.set_shareL1Size(l1 / 2); t.set_shareL0CSize(l0c / 2);
    auto host = at::empty({int64_t(t.GetDataSize())}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    t.SaveToBuffer(host.data_ptr(), host.numel());
    return host.to(options.device());
}

at::Tensor Forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& qs, const at::Tensor& ks, const at::Tensor& vs, const at::Tensor& lut,
    int64_t queries, int64_t keys, double scale, bool expcast) {
    Check(q, k, v, qs, ks, vs, lut, queries, keys, scale);
    TORCH_CHECK(q.is_privateuseone(), "NPU payload required");
    c10::OptionalDeviceGuard guard(q.device());
    const char* soc = aclrtGetSocName();
    TORCH_CHECK(soc && std::string(soc) == VC_ASCEND_SOC_VERSION,
                "Native extension built for ", VC_ASCEND_SOC_VERSION,
                "; runtime SoC: ", soc ? soc : "unknown",
                ". Rebuild on the selected device with scripts/run_ascend910b.sh");
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform, "PlatformAscendC unavailable");
    // A2 910B and A3 910_93xx both map to this CANN platform enum.
    TORCH_CHECK(platform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND910B,
                "This kernel requires the A2/A3 DAV-2201 platform");
    uint64_t ub = 0, l1 = 0, l0c = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
    auto core_count = platform->GetCoreNumAic();
    TORCH_CHECK(core_count > 0 && ub > 0 && l1 > 0 && l0c > 0, "Cannot query device resources");
    TORCH_CHECK(platform->GetCoreNumAiv() >= 2 * core_count,
                "This mixed kernel requires two AIV cores per AIC; check the device partition");
    uint32_t max_tmp = 0, min_tmp = 0, sum_tmp = 0;
    ge::Shape shape({ROWS, N});
    AscendC::GetReduceMaxMaxMinTmpSize(shape, ge::DT_FLOAT, AscendC::ReducePattern::AR,
                                     true, false, max_tmp, min_tmp);
    AscendC::GetReduceSumMaxMinTmpSize(shape, ge::DT_FLOAT, AscendC::ReducePattern::AR,
                                     true, false, sum_tmp, min_tmp);
    auto reduce_bytes = (std::max(max_tmp, sum_tmp) + 31) / 32 * 32;
    reduce_bytes = std::max(reduce_bytes, uint32_t(32));
    TORCH_CHECK(VectorBytes(reduce_bytes) <= ub, "Vector UB budget exceeds device capacity");
    const uint64_t tasks = uint64_t(q.size(0)) * q.size(1) * (q.size(2) / M);
    const uint32_t cores = std::min<uint64_t>(core_count, tasks);
    Tiling t{uint32_t(q.size(0)), uint32_t(q.size(1)), uint32_t(queries), uint32_t(keys),
             uint32_t(q.size(2)), uint32_t(k.size(2)), uint32_t(q.size(3)), cores,
             reduce_bytes, uint32_t(expcast), float(scale), 0, tasks};
    auto qt = CubeTiling(platform, true, t.dim, l1, l0c, q.options());
    auto vt = CubeTiling(platform, false, t.dim, l1, l0c, q.options());
    auto meta_cpu = at::empty({int64_t(sizeof(t))}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    std::memcpy(meta_cpu.data_ptr(), &t, sizeof(t));
    auto meta = meta_cpu.to(q.device());
    auto workspace = at::empty({int64_t(cores * WorkspaceBytes(t.dim))}, q.options().dtype(at::kByte));
    auto out = at::empty(q.sizes(), q.options().dtype(at::kFloat));
    // Drain PTA's host queue before direct launch, including all .to() above.
    auto stream = c10_npu::getCurrentNPUStream(q.device().index()).stream(true);
    auto ret = aclrtlaunch_vc910_forward(cores, stream, q.data_ptr(), k.data_ptr(), v.data_ptr(),
        qs.data_ptr(), ks.data_ptr(), vs.data_ptr(), lut.data_ptr(), out.data_ptr(), workspace.data_ptr(),
        qt.data_ptr(), vt.data_ptr(), meta.data_ptr());
    TORCH_CHECK(ret == ACL_SUCCESS, "vc910_forward launch failed: ", ret);
    // Workspaces allocated on the current stream; caching allocator retains stream ordering.
    return out;
}

at::Tensor Meta(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& qs, const at::Tensor& ks, const at::Tensor& vs, const at::Tensor& lut,
    int64_t queries, int64_t keys, double scale, bool expcast) {
    Check(q, k, v, qs, ks, vs, lut, queries, keys, scale);
    return at::empty(q.sizes(), q.options().dtype(at::kFloat));
}
}

TORCH_LIBRARY(vc_ascend, m) {
    m.def("forward(Tensor q, Tensor k, Tensor v, Tensor q_scale, Tensor k_scale, Tensor v_scale, Tensor lut, int queries, int keys, float scale, bool expcast) -> Tensor");
}
TORCH_LIBRARY_IMPL(vc_ascend, PrivateUse1, m) { m.impl("forward", TORCH_FN(Forward)); }
TORCH_LIBRARY_IMPL(vc_ascend, Meta, m) { m.impl("forward", TORCH_FN(Meta)); }
