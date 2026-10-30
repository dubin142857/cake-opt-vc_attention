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
#include <tuple>
#include "vc_tiling_host.h"

extern "C" {
uint32_t aclrtlaunch_vc910_forward(uint32_t blocks, void* stream, void* q, void* k, void* v,
    void* qs, void* ks, void* vs, void* lut, void* out, void* ws, void* qt, void* vt, void* meta, void* trace);
decltype(aclrtlaunch_vc910_forward) aclrtlaunch_vc910_forward_m64;
decltype(aclrtlaunch_vc910_forward) aclrtlaunch_vc910_profile;
decltype(aclrtlaunch_vc910_forward) aclrtlaunch_vc910_profile_m64;
//  CubeV2 的数值探针: 参数表与主内核不同, 单独声明。
uint32_t aclrtlaunch_vc910_cube_v2_probe(uint32_t blocks, void* stream,
    void* s_out, void* o_out, void* q, void* k, void* p, void* v);
}

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
    //  ★这里的填充单位是**客户端的量化块** Q_SCALE_BLOCK, 不是我们内部的计算 tile M。
    //    原文写的是 M —— M=32 时两者恰好重合所以一直没暴露, 而 M 改 64 后会错误地
    //    要求客户端按 64 行填充并按 qp/64 给 scale。注释里本来就写着 "Qpad/32",
    //    即本意一直是量化块。计算 tile 的填充另有一处 (下方的 extra_q), 两者互不相干。
    const auto qp = (queries + Q_SCALE_BLOCK - 1) / Q_SCALE_BLOCK * Q_SCALE_BLOCK,
               kp = (keys + N - 1) / N * N;
    TORCH_CHECK(q.size(2) == qp && k.size(0) == b && k.size(1) == h && k.size(2) == kp && k.size(3) == d,
                "Q/K padding or MHA shape mismatch");
    TORCH_CHECK(v.sizes() == k.sizes(), "K/V shape mismatch");
    TORCH_CHECK(qs.sizes() == at::IntArrayRef({b, h, qp / Q_SCALE_BLOCK, 8}),
                "Q scales must be [B,H,Qpad/32,8]");
    //  K 的量化块是 K_SCALE_BLOCK, 与 matmul tile 宽度 N 解耦; scale 数按前者算。
    TORCH_CHECK(ks.sizes() == at::IntArrayRef({b, h, kp / K_SCALE_BLOCK, 8}),
                "K scales must be [B,H,Kpad/K_SCALE_BLOCK,8]");
    TORCH_CHECK(vs.sizes() == at::IntArrayRef({b, h, d}), "V scales must be [B,H,D]");
    TORCH_CHECK(lut.dim() == 1 && lut.numel() == 128, "LUT must contain 128 FP32 entries");
    TORCH_CHECK(std::isfinite(scale) && scale > 0 && scale <= std::numeric_limits<float>::max(), "Invalid scale");
}

at::Tensor CubeTiling(platform_ascendc::PlatformAscendC* platform, bool qk, int d, int tile_m,
                      uint64_t l1, uint64_t l0c, const at::TensorOptions& options) {
    using namespace matmul_tiling;
    MatmulApiTiling api(*platform);
    api.SetAType(TPosition::GM, CubeFormat::ND, qk ? DataType::DT_INT8 : DataType::DT_FLOAT16, false);
    //  H16: PV 的 B (即 V) 改由内核自管 L1。TSCM 只接受 NZ ——
    //  CANN 安装包里 TSCM 的全部内建用例均为 CubeFormat::NZ。
    //  实测 (本机 CANN 9.0.0, /tmp/tscmprobe): TSCM/NZ 与 GM/ND 的切分参数完全相同
    //  (baseM/baseN/baseK/stepM/stepN/stepKa/stepKb/depthA1/depthB1 逐项一致),
    //  把 L1 预算再扣掉内核那 128 KB 后仍然相同, GetTiling 均返回 0。
    if (qk) {
        api.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, true);
    } else {
        api.SetBType(TPosition::TSCM, CubeFormat::NZ, DataType::DT_FLOAT16, false);
    }
    api.SetCType(TPosition::GM, CubeFormat::ND, qk ? DataType::DT_INT32 : DataType::DT_FLOAT);
    api.SetBiasType(TPosition::GM, CubeFormat::ND, qk ? DataType::DT_INT32 : DataType::DT_FLOAT);
    const int n = qk ? N : d, k = qk ? d : N;
    api.SetShape(tile_m, n, k);
    api.SetOrgShape(tile_m, n, k);
    api.SetFixSplit(tile_m, n, k);
    api.SetBias(false);
    // Two sequential local matmul objects share the physical Cube memories.
    //  H16: 非 qk 的那一路把 V 的两个 L1 槽从预算里扣掉 (见 vc_tiling.h 的 VSCM_BYTES)。
    const uint64_t l1_share = qk ? l1 / 2 : l1 / 2 - vc910::VSCM_BYTES;
    api.SetBufferSpace(l1_share, l0c / 2, -1);
    optiling::TCubeTiling t;
    TORCH_CHECK(api.GetTiling(t) == 0, "CANN MatmulApiTiling failed for ", qk ? "INT8 QK" : "FP16 PV");
    t.set_usedCoreNum(1);
    t.set_singleCoreM(tile_m); t.set_singleCoreN(n); t.set_singleCoreK(k);
    t.set_dbL0A(1); t.set_dbL0B(1); t.set_dbL0C(1);
    t.set_shareMode(0); t.set_shareL1Size(l1_share); t.set_shareL0CSize(l0c / 2);
    auto host = at::empty({int64_t(t.GetDataSize())}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    t.SaveToBuffer(host.data_ptr(), host.numel());
    return host.to(options.device());
}

std::tuple<at::Tensor, at::Tensor> ForwardImpl(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& qs, const at::Tensor& ks, const at::Tensor& vs, const at::Tensor& lut,
    int64_t queries, int64_t keys, double scale, bool expcast, int64_t tile_m, bool profile) {
    Check(q, k, v, qs, ks, vs, lut, queries, keys, scale);
    //  ★只剩 64: 手写 cube 的几何是 M=64 x N=128 (vc_l1l0_map.h 的断言守着 L0 预算),
    //    且 ROWS=M/2=32 恰好是一个 Q_SCALE_BLOCK, 向量侧的 scale 逻辑依赖这一条。
    TORCH_CHECK(tile_m == 64, "tile_m must be 64 (手写 cube 的几何); Q 量化仍是 Q32");
    TORCH_CHECK(queries < INT32_MAX - tile_m, "Query padding exceeds the native indexing contract");
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
    ge::Shape shape({tile_m / 2, N});
    AscendC::GetReduceMaxMaxMinTmpSize(shape, ge::DT_FLOAT, AscendC::ReducePattern::AR,
                                     true, false, max_tmp, min_tmp);
    AscendC::GetReduceSumMaxMinTmpSize(shape, ge::DT_FLOAT, AscendC::ReducePattern::AR,
                                     true, false, sum_tmp, min_tmp);
    auto reduce_bytes = (std::max(max_tmp, sum_tmp) + 31) / 32 * 32;
    reduce_bytes = std::max(reduce_bytes, uint32_t(32));
    TORCH_CHECK(VectorBytes(reduce_bytes, uint32_t(tile_m), map::CHUNK) <= ub,
                "Vector UB budget exceeds device capacity");
    // Payloads retain Q32 quantization. Only append a dummy Q32 block when the
    // M64 Cube would otherwise read beyond Q's allocation. Account for this in
    // both fixed-core and full-API timing; never requantize the valid operands.
    const int64_t extra_q = (tile_m - q.size(2) % tile_m) % tile_m;
    auto qp = extra_q ? at::constant_pad_nd(q, {0, 0, 0, extra_q}, 0) : q;
    auto qsp = extra_q ? at::constant_pad_nd(qs, {0, 0, 0, extra_q / Q_SCALE_BLOCK}, 1) : qs;
    const uint64_t tasks = uint64_t(q.size(0)) * q.size(1) * (qp.size(2) / tile_m);
    const uint32_t cores = std::min<uint64_t>(core_count, tasks);
    Tiling t{uint32_t(q.size(0)), uint32_t(q.size(1)), uint32_t(queries), uint32_t(keys),
             uint32_t(qp.size(2)), uint32_t(k.size(2)), uint32_t(q.size(3)), cores,
             reduce_bytes, uint32_t(expcast), float(scale), 0, tasks};
    //  ★改用 V2 布局 (S/P 各 2xCHUNK 槽 + O 双槽) —— 与 CubeV2 和向量侧的 chunk 分阶段配套。
    t.workspace_bytes = uint32_t(WorkspaceBytesV2(t.dim, uint32_t(tile_m), N, map::CHUNK));
    auto qt = CubeTiling(platform, true, t.dim, int(tile_m), l1, l0c, q.options());
    auto vt = CubeTiling(platform, false, t.dim, int(tile_m), l1, l0c, q.options());
    auto meta_cpu = at::empty({int64_t(sizeof(t))}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    std::memcpy(meta_cpu.data_ptr(), &t, sizeof(t));
    auto meta = meta_cpu.to(q.device());
    auto workspace = at::empty({int64_t(uint64_t(cores) * t.workspace_bytes)}, q.options().dtype(at::kByte));
    auto out = at::empty(qp.sizes(), q.options().dtype(at::kFloat));
    at::Tensor trace;
    if (profile) {
        trace = at::zeros({cores, PROFILE_ROLES, PROFILE_SAMPLES, PROFILE_FIELDS}, q.options().dtype(at::kLong));
        TORCH_CHECK(reinterpret_cast<uintptr_t>(trace.data_ptr()) % 64 == 0,
                    "Profile records must start on a 64-byte cache-line boundary");
    }
    // Drain PTA's host queue before direct launch, including all .to() above.
    auto stream = c10_npu::getCurrentNPUStream(q.device().index()).stream(true);
    auto launch = profile ? (tile_m == 64 ? aclrtlaunch_vc910_profile_m64 : aclrtlaunch_vc910_profile)
                          : (tile_m == 64 ? aclrtlaunch_vc910_forward_m64 : aclrtlaunch_vc910_forward);
    auto ret = launch(cores, stream, qp.data_ptr(), k.data_ptr(), v.data_ptr(),
        qsp.data_ptr(), ks.data_ptr(), vs.data_ptr(), lut.data_ptr(), out.data_ptr(), workspace.data_ptr(),
        qt.data_ptr(), vt.data_ptr(), meta.data_ptr(), profile ? trace.data_ptr() : nullptr);
    TORCH_CHECK(ret == ACL_SUCCESS, "vc910_forward launch failed: ", ret);
    // Workspaces allocated on the current stream; caching allocator retains stream ordering.
    return {out, trace};
}

at::Tensor Forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& qs, const at::Tensor& ks, const at::Tensor& vs, const at::Tensor& lut,
    int64_t queries, int64_t keys, double scale, bool expcast) {
    //  ★非 tiled 入口原来把 32 写死 —— M 改 64 后它会被自己的 TORCH_CHECK 拒掉。
    //    改用 M, 这样"默认 tile"只有一个定义处 (vc_tiling.h), 不再有第二份字面量。
    return std::get<0>(ForwardImpl(q, k, v, qs, ks, vs, lut, queries, keys, scale, expcast,
                                   int64_t(M), false));
}

at::Tensor ForwardTiled(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& qs, const at::Tensor& ks, const at::Tensor& vs, const at::Tensor& lut,
    int64_t queries, int64_t keys, double scale, bool expcast, int64_t tile_m) {
    return std::get<0>(ForwardImpl(q, k, v, qs, ks, vs, lut, queries, keys, scale, expcast, tile_m, false));
}

std::tuple<at::Tensor, at::Tensor> ProfileTiled(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& qs, const at::Tensor& ks, const at::Tensor& vs, const at::Tensor& lut,
    int64_t queries, int64_t keys, double scale, bool expcast, int64_t tile_m) {
    return ForwardImpl(q, k, v, qs, ks, vs, lut, queries, keys, scale, expcast, tile_m, true);
}

at::Tensor Meta(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& qs, const at::Tensor& ks, const at::Tensor& vs, const at::Tensor& lut,
    int64_t queries, int64_t keys, double scale, bool expcast) {
    Check(q, k, v, qs, ks, vs, lut, queries, keys, scale);
    return at::empty(q.sizes(), q.options().dtype(at::kFloat));
}

at::Tensor MetaTiled(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& qs, const at::Tensor& ks, const at::Tensor& vs, const at::Tensor& lut,
    int64_t queries, int64_t keys, double scale, bool expcast, int64_t tile_m) {
    Check(q, k, v, qs, ks, vs, lut, queries, keys, scale);
    TORCH_CHECK(tile_m == 32 || tile_m == 64, "tile_m must be 32 or 64");
    TORCH_CHECK(queries < INT32_MAX - tile_m, "queries exceed execution-tile indexing range");
    return at::empty({q.size(0), q.size(1), (q.size(2) + tile_m - 1) / tile_m * tile_m, q.size(3)},
                     q.options().dtype(at::kFloat));
}

//  ---- CubeV2 数值探针 --------------------------------------------------------
//  ★目的不是跑算子, 是确证手写原语里 repeat / src_stride / dst_gap /
//    dst_frac_stride 这几个参数的**语义**。公开 API 有声明, 编译器能查类型,
//    但 CANN 树里没有这几个函数的调用样例 (c_api 与 impl 两层都查过),
//    语义只能从内建推 —— 而"推"正是 F183 里烧掉三次构建的失败模式。
//  形状固定 (d=128, M=64, N=128), 单核启动 —— 探针要尽量少的活动部件。
std::tuple<at::Tensor, at::Tensor> CubeV2Probe(const at::Tensor& q, const at::Tensor& k,
                                               const at::Tensor& pmat, const at::Tensor& v) {
    constexpr int64_t MB = 64, NB = 128, D = 128;
    TORCH_CHECK(q.scalar_type() == at::kChar && k.scalar_type() == at::kChar,
                "probe Q/K must be int8 (QK 走 int8 通路)");
    TORCH_CHECK(pmat.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf,
                "probe P/V must be half (PV 走 half 通路)");
    TORCH_CHECK(q.sizes() == at::IntArrayRef({MB, D}), "probe Q must be [64,128]");
    TORCH_CHECK(k.sizes() == at::IntArrayRef({NB, D}), "probe K must be [128,128]");
    TORCH_CHECK(pmat.sizes() == at::IntArrayRef({MB, NB}), "probe P must be [64,128]");
    TORCH_CHECK(v.sizes() == at::IntArrayRef({NB, D}), "probe V must be [128,128]");
    for (const auto& t : {q, k, pmat, v}) {
        TORCH_CHECK(t.is_contiguous(), "probe inputs must be contiguous ND");
        TORCH_CHECK(t.is_privateuseone(), "probe inputs must live on the NPU");
    }
    auto s_out = at::zeros({MB, NB}, q.options().dtype(at::kInt));
    auto o_out = at::zeros({MB, D}, q.options().dtype(at::kFloat));
    auto stream = c10_npu::getCurrentNPUStream(q.device().index()).stream(true);
    auto ret = aclrtlaunch_vc910_cube_v2_probe(1, stream, s_out.data_ptr(), o_out.data_ptr(),
        q.data_ptr(), k.data_ptr(), pmat.data_ptr(), v.data_ptr());
    TORCH_CHECK(ret == ACL_SUCCESS, "vc910_cube_v2_probe launch failed: ", ret);
    return {s_out, o_out};
}
}

TORCH_LIBRARY(vc_ascend, m) {
    m.def("forward(Tensor q, Tensor k, Tensor v, Tensor q_scale, Tensor k_scale, Tensor v_scale, Tensor lut, int queries, int keys, float scale, bool expcast) -> Tensor");
    m.def("forward_tiled(Tensor q, Tensor k, Tensor v, Tensor q_scale, Tensor k_scale, Tensor v_scale, Tensor lut, int queries, int keys, float scale, bool expcast, int tile_m) -> Tensor");
    m.def("profile_tiled(Tensor q, Tensor k, Tensor v, Tensor q_scale, Tensor k_scale, Tensor v_scale, Tensor lut, int queries, int keys, float scale, bool expcast, int tile_m) -> (Tensor, Tensor)");
    m.def("cube_v2_probe(Tensor q, Tensor k, Tensor p, Tensor v) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(vc_ascend, PrivateUse1, m) {
    m.impl("forward", TORCH_FN(Forward));
    m.impl("forward_tiled", TORCH_FN(ForwardTiled));
    m.impl("profile_tiled", TORCH_FN(ProfileTiled));
    m.impl("cube_v2_probe", TORCH_FN(CubeV2Probe));
}
TORCH_LIBRARY_IMPL(vc_ascend, Meta, m) {
    m.impl("forward", TORCH_FN(Meta));
    m.impl("forward_tiled", TORCH_FN(MetaTiled));
}
