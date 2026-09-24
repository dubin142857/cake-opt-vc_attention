#include <torch/extension.h>
#include <ATen/ops/argsort.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

#include "vc_cuda.h"

namespace {
using at::Tensor;

// Own only the cuBLASLt resource we use. ATen's CUDAContext.h also includes
// cuSPARSE/cuSOLVER development headers, which this extension does not need.
struct LocalLtHandle {
    cublasLtHandle_t value = nullptr;
    int device;
    explicit LocalLtHandle(int device_index) : device(device_index) {
        TORCH_CHECK(cublasLtCreate(&value) == CUBLAS_STATUS_SUCCESS, "cublasLtCreate failed");
    }
    LocalLtHandle(const LocalLtHandle&) = delete;
    LocalLtHandle& operator=(const LocalLtHandle&) = delete;
    ~LocalLtHandle() {
        // Thread teardown may occur after CUDA has already shut down. Do not
        // throw from a destructor, and release on the handle's original device.
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (previous != device && cudaSetDevice(device) != cudaSuccess) return;
        if (value) cublasLtDestroy(value);
        if (previous != device) cudaSetDevice(previous);
    }
};

cublasLtHandle_t blaslt_handle(int device) {
    // Destruction synchronizes the device, so keep handles between forward
    // calls. Each host thread/device gets its own handle; stream is passed to
    // every matmul explicitly. The caller has already set its CUDAGuard.
    thread_local std::unordered_map<int, std::unique_ptr<LocalLtHandle>> handles;
    auto& handle = handles[device];
    if (!handle) handle = std::make_unique<LocalLtHandle>(device);
    return handle->value;
}

int small_int(int64_t value, const char* name) {
    TORCH_CHECK(value > 0 && value <= INT_MAX, name, " must fit a positive int32");
    return static_cast<int>(value);
}

void require_hopper(const Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "cuda_fp8 requires CUDA tensors; there is no CPU fallback");
    int major = 0, minor = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, x.get_device()));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, x.get_device()));
    TORCH_CHECK(major == 9, "cuda_fp8 targets Hopper (H800/H100/H200, compute capability 9.x); got ",
                major, ".", minor);
}

void block_size(int64_t value, const char* name) {
    TORCH_CHECK(value >= 32 && value <= 256 && value % 32 == 0,
                name, " must be a multiple of 32 in [32, 256]");
}

bool all_finite(Tensor q, Tensor k, Tensor v) {
    TORCH_CHECK(q.is_cuda(), "Finite check requires CUDA");
    TORCH_CHECK(q.scalar_type() == at::kFloat || q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16, "Invalid finite-check dtype");
    const c10::cuda::CUDAGuard guard(q.device());
    auto view = [&q](const Tensor& t) {
        TORCH_CHECK(t.device() == q.device() && t.scalar_type() == q.scalar_type() && t.dim() == 4 && t.numel() > 0,
                    "Finite check requires nonempty four-dimensional tensors of one CUDA dtype/device");
        vc::InputView result{};
        result.data = t.data_ptr(); result.size = t.numel(); result.contiguous = t.is_contiguous();
        for (int dim = 0; dim < 4; ++dim) { result.shape[dim] = t.size(dim); result.stride[dim] = t.stride(dim); }
        return result;
    };
    const auto qv = view(q), kv = view(k), vv = view(v);
    auto invalid = at::empty({}, q.options().dtype(at::kInt));
    const int dtype = q.scalar_type() == at::kFloat ? 0 : q.scalar_type() == at::kHalf ? 1 : 2;
    vc::check_finite(qv, kv, vv, dtype, invalid.data_ptr<int>(), c10::cuda::getCurrentCUDAStream(q.get_device()).stream());
    return invalid.item<int>() == 0;  // Preserve validation, with one host wait instead of three.
}

std::vector<Tensor> preprocess_qk(Tensor q, Tensor k, bool k_smooth, bool hadamard) {
    TORCH_CHECK(q.is_cuda() && k.device() == q.device(), "Q/K preprocessing requires one CUDA device");
    const c10::cuda::CUDAGuard guard(q.device());
    const at::NoGradGuard no_grad;
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && q.numel() && k.numel(), "Expected nonempty [B,H,N,D]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1) && q.size(3) == k.size(3), "Q/K shapes must match");
    const int width = small_int(q.size(3), "head dimension");
    TORCH_CHECK(width <= 128, "Q/K preprocessing supports head dimensions <= 128");
    q = q.to(at::kFloat).contiguous();
    k = k.to(at::kFloat).contiguous();
    if (!k_smooth && !hadamard) return {q, k};
    int out_width = width;
    if (hadamard) {
        out_width = 1;
        while (out_width < width) out_width *= 2;
    }
    const int heads = small_int(q.size(0) * q.size(1), "B*H");
    const int nq = small_int(q.size(2), "Q sequence length");
    const int nk = small_int(k.size(2), "K sequence length");
    small_int(int64_t(heads) * nq, "Q transform grid");
    small_int(int64_t(heads) * nk, "K transform grid");
    auto mean = k_smooth ? k.mean(2, true) : Tensor();
    auto qr = hadamard ? at::empty({q.size(0), q.size(1), nq, out_width}, q.options()) : q;
    auto kr = at::empty({k.size(0), k.size(1), nk, out_width}, k.options());
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if (hadamard)
        vc::transform_qk(q.data_ptr<float>(), nullptr, qr.data_ptr<float>(), heads, nq, width, out_width, true, stream);
    vc::transform_qk(k.data_ptr<float>(), k_smooth ? mean.data_ptr<float>() : nullptr,
                     kr.data_ptr<float>(), heads, nk, width, out_width, hadamard, stream);
    return {qr, kr};
}

template<int Mode>  // 0=cuBLAS, 1/2=generic pipeline, 3/4=0.4.0, 5/6=D128 (native/emulator).
Tensor forward(Tensor q, Tensor k, Tensor v, int64_t bq, int64_t k_quant,
               int64_t bk, bool smooth, bool expcast, int64_t mean_type, double scale,
               bool k_smooth, bool hadamard, bool batched) {
    constexpr bool fused = Mode != 0;
    if constexpr (Mode == 2 || Mode == 4 || Mode == 6) {
        TORCH_CHECK(q.is_cuda(), "Diagnostic emulator requires CUDA tensors");
        TORCH_CHECK(q.numel() <= 1024 * 128 && k.numel() <= 1024 * 128,
                    "Diagnostic emulator is restricted to small test tensors; it is not a performance backend");
    } else require_hopper(q);
    const c10::cuda::CUDAGuard guard(q.device());
    const at::NoGradGuard no_grad;
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "Expected [B,H,N,D] tensors");
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device(), "Q/K/V devices must match");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(), "Q/K/V dtypes must match");
    TORCH_CHECK(q.scalar_type() == at::kFloat || q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
                "Input dtype must be FP32, FP16, or BF16");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0) &&
                q.size(1) == k.size(1) && q.size(1) == v.size(1), "MHA batch/head counts must match; GQA is unsupported");
    TORCH_CHECK(q.size(3) == k.size(3) && k.size(2) == v.size(2), "Incompatible Q/K/V shapes");
    TORCH_CHECK(q.size(3) <= 128 && v.size(3) <= 128, "CUDA v1 requires Q/K/V head dimensions <= 128");
    TORCH_CHECK(q.numel() && k.numel() && v.numel(), "Empty inputs are unsupported");
    TORCH_CHECK(std::isfinite(scale) && scale > 0.0 && std::isfinite(static_cast<float>(scale)), "scale must be positive finite FP32");
    block_size(bq, "q_block");
    block_size(bk, "kv_block");
    TORCH_CHECK(!fused || bk == 128, "Fused CUDA currently requires kv_block=128; no fallback was used");
    TORCH_CHECK(k_quant >= bk && k_quant <= 4096 && k_quant % bk == 0,
                "k_quant_block must be a multiple of kv_block and <= 4096");
    TORCH_CHECK(mean_type >= 0 && mean_type <= 2, "Invalid mean dtype");
    if constexpr (Mode >= 5) {
        TORCH_CHECK(q.size(3) == 128 && v.size(3) == 128 && bq == 128 && k_quant == 256 &&
                    bk == 128 && mean_type == 1,
                    "D128 specialization requires D=DV=128, q_block=128, k_quant_block=256, kv_block=128, mean_dtype=float16");
    }

    const auto original_type = q.scalar_type();
    auto transformed = preprocess_qk(q, k, k_smooth, hadamard);
    q = transformed[0]; k = transformed[1];
    v = v.to(at::kFloat).contiguous();
    vc::Shape s{};
    s.heads = small_int(q.size(0) * q.size(1), "B*H");
    s.nq = small_int(q.size(2), "Q sequence length");
    s.nk = small_int(k.size(2), "KV sequence length");
    s.d = small_int(q.size(3), "head dimension");
    s.dv = small_int(v.size(3), "V head dimension");
    s.bq = static_cast<int>(bq); s.bk = static_cast<int>(bk); s.k_quant = static_cast<int>(k_quant);
    s.nq_tiles = small_int((int64_t(s.nq) + bq - 1) / bq, "Q tile count");
    s.nk_tiles = small_int((int64_t(s.nk) + bk - 1) / bk, "KV tile count");
    s.nk_quant_tiles = small_int((int64_t(s.nk) + k_quant - 1) / k_quant, "K quantization tile count");
    s.pq = small_int(int64_t(s.nq_tiles) * bq, "padded Q length");
    s.pk = small_int(int64_t(s.nk_quant_tiles) * k_quant, "padded K length");
    s.dp = (s.d + 31) / 32 * 32; s.vp = (s.dv + 31) / 32 * 32;
    small_int(int64_t(s.heads) * s.nq_tiles, "Q preprocessing grid");
    small_int(int64_t(s.heads) * s.nk_quant_tiles, "K preprocessing grid");
    small_int(int64_t(s.heads) * s.nk_tiles * s.vp, "V preprocessing grid");
    s.smooth = smooth; s.expcast = expcast; s.mean_type = static_cast<int>(mean_type); s.scale = static_cast<float>(scale);
    s.query_chunk = std::min(s.nq_tiles, std::max(1, 1024 / s.bq)) * s.bq;
    if (fused) small_int(int64_t(s.heads) * ((int64_t(s.nq) + 63) / 64), "Fused grid");
    const int scratch_heads = batched ? s.heads : 1;
    const int scratch_rows = batched ? s.query_chunk : s.bq;
    small_int(int64_t(scratch_heads) * scratch_rows, "batched Softmax grid");

    // Every temporary is allocated on the current PyTorch stream, and all
    // kernels/cuBLASLt calls use that same stream. Allocator reuse is ordered.
    const auto f = q.options();
    const auto bytes = f.dtype(at::kByte);
    auto q8 = at::empty({s.heads, s.pq, s.dp}, bytes);
    auto k8 = at::empty({s.heads, s.pk, s.dp}, bytes);
    auto v8 = at::empty({s.heads, s.nk_tiles, s.vp, s.bk}, bytes);
    auto p8 = at::empty({fused ? 0 : scratch_heads, scratch_rows, s.bk}, bytes);
    auto qs = at::empty({s.heads, s.nq_tiles}, f);
    auto ks = at::empty({s.heads, s.nk_quant_tiles}, f);
    auto vs = at::empty({s.heads, s.nk_tiles, s.vp}, f);
    const auto mean_dtype = mean_type == 1 ? at::kHalf : mean_type == 2 ? at::kBFloat16 : at::kFloat;
    auto means = at::empty(vs.sizes(), f.dtype(mean_dtype));
    auto scores = at::empty({fused ? 0 : scratch_heads, s.bk, scratch_rows}, f);
    auto pv = at::empty({fused ? 0 : scratch_heads, s.vp, scratch_rows}, f);
    auto maximum = at::empty({fused ? 0 : scratch_heads, scratch_rows}, f);
    auto denom = at::empty_like(maximum);
    auto alpha = at::empty_like(maximum);
    auto mass = at::empty_like(maximum);
    auto accum = at::empty({fused ? 0 : scratch_heads, scratch_rows, s.vp}, f);
    auto output = at::empty({q.size(0), q.size(1), q.size(2), v.size(3)}, f);
    auto workspace = at::empty({fused ? 0 : 32 * 1024 * 1024}, bytes);
    vc::Buffers buffers{
        q8.data_ptr<uint8_t>(), k8.data_ptr<uint8_t>(), v8.data_ptr<uint8_t>(), p8.data_ptr<uint8_t>(),
        qs.data_ptr<float>(), ks.data_ptr<float>(), vs.data_ptr<float>(), means.data_ptr(),
        scores.data_ptr<float>(), pv.data_ptr<float>(), maximum.data_ptr<float>(), denom.data_ptr<float>(),
        alpha.data_ptr<float>(), mass.data_ptr<float>(), accum.data_ptr<float>(), output.data_ptr<float>(),
        workspace.data_ptr(), static_cast<size_t>(workspace.numel())
    };
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if constexpr (fused) {
        vc::forward_fused(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), s, buffers, stream,
                          Mode == 2 || Mode == 4 || Mode == 6, Mode != 3 && Mode != 4, Mode >= 5);
    } else {
        auto handle = blaslt_handle(q.get_device());
        auto launch = batched ? vc::forward_batched : vc::forward;
        launch(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), s, buffers, handle, stream);
    }
    return output.to(original_type);
}

// The small Lloyd loop is host C++; ATen executes its arithmetic and stable
// sort on CUDA. No CPU copies and no Python iteration over heads/tokens.
std::vector<Tensor> group_values(Tensor v, int64_t clusters, int64_t iterations,
                                c10::optional<Tensor> old_centers) {
    require_hopper(v);
    const c10::cuda::CUDAGuard guard(v.device());
    const at::NoGradGuard no_grad;
    TORCH_CHECK(v.dim() == 4 && v.numel() > 0, "group_values expects nonempty [B,H,N,D]");
    TORCH_CHECK(clusters > 0 && iterations > 0, "clusters and iterations must be positive");
    auto x = v.to(at::kFloat);
    const int64_t count = std::min(clusters, x.size(2));
    Tensor centers;
    if (old_centers.has_value()) {
        centers = old_centers.value();
        TORCH_CHECK(centers.device() == x.device() && centers.scalar_type() == at::kFloat &&
                    centers.dim() == 4 && centers.size(0) == x.size(0) && centers.size(1) == x.size(1) &&
                    centers.size(2) == count && centers.size(3) == x.size(3), "Invalid cached centers");
    } else {
        auto indices = at::linspace(0, x.size(2) - 1, count, x.options()).to(at::kLong);
        centers = x.index_select(2, indices).clone();
    }
    auto labels = at::empty({x.size(0), x.size(1), x.size(2)}, x.options().dtype(at::kLong));
    for (int64_t iter = 0; iter < iterations; ++iter) {
        auto sums = at::zeros_like(centers);
        auto counts = at::zeros({x.size(0), x.size(1), count, 1}, x.options());
        auto center_norms = centers.square().sum(-1).unsqueeze(-2);
        for (int64_t start = 0; start < x.size(2); start += 2048) {
            auto rows = x.slice(2, start, std::min(start + 2048, x.size(2)));
            auto distances = rows.square().sum(-1, true) + center_norms - 2 * at::matmul(rows, centers.transpose(-2, -1));
            auto ids = distances.argmin(-1);
            labels.slice(2, start, start + rows.size(2)).copy_(ids);
            sums.scatter_add_(2, ids.unsqueeze(-1).expand_as(rows), rows);
            counts.scatter_add_(2, ids.unsqueeze(-1), at::ones_like(rows.slice(3, 0, 1)));
        }
        centers = at::where(counts > 0, sums / counts.clamp_min(1), centers);
    }
    return {at::argsort(labels, true, -1, false), centers};
}

Tensor expcast_codes(Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kFloat, "ExpCast expects CUDA FP32 input");
    const c10::cuda::CUDAGuard guard(input.device());
    auto x = input.contiguous();
    auto result = at::empty(x.sizes(), x.options().dtype(at::kByte));
    vc::expcast(x.data_ptr<float>(), result.data_ptr<uint8_t>(), x.numel(),
                c10::cuda::getCurrentCUDAStream(x.get_device()).stream());
    return result;
}

void debug_tensor(const Tensor& x, const Tensor& owner, at::ScalarType type, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.device() == owner.device() && x.scalar_type() == type &&
                x.is_contiguous() && x.numel() > 0, name, " requires nonempty contiguous CUDA tensors of the expected dtype on one device");
}

std::vector<Tensor> debug_quantize(Tensor x, int64_t block) {
    debug_tensor(x, x, at::kFloat, "debug_quantize");
    TORCH_CHECK(x.dim() == 4 && x.size(3) <= 128 && block > 0 && block <= 4096, "Invalid Q/K diagnostic shape or block");
    const c10::cuda::CUDAGuard guard(x.device());
    const int heads = small_int(x.size(0) * x.size(1), "B*H");
    const int n = small_int(x.size(2), "tokens"), d = small_int(x.size(3), "channels");
    const int tiles = small_int((int64_t(n) + block - 1) / block, "tiles");
    const int pn = small_int(int64_t(tiles) * block, "padded tokens"), pd = (d + 31) / 32 * 32;
    small_int(int64_t(heads) * tiles, "quantization grid");
    auto codes = at::empty({x.size(0), x.size(1), pn, pd}, x.options().dtype(at::kByte));
    auto scales = at::empty({x.size(0), x.size(1), tiles}, x.options());
    vc::debug_quantize(x.data_ptr<float>(), codes.data_ptr<uint8_t>(), scales.data_ptr<float>(),
                       heads, n, pn, d, pd, static_cast<int>(block), tiles,
                       c10::cuda::getCurrentCUDAStream(x.get_device()).stream());
    return {codes, scales};
}

std::vector<Tensor> debug_values(Tensor x, int64_t block, bool smooth, int64_t mean_type) {
    debug_tensor(x, x, at::kFloat, "debug_values");
    TORCH_CHECK(x.dim() == 4 && x.size(3) <= 128 && mean_type >= 0 && mean_type <= 2, "Invalid V diagnostic shape or mean type");
    block_size(block, "V block");
    const c10::cuda::CUDAGuard guard(x.device());
    vc::Shape s{};
    s.heads = small_int(x.size(0) * x.size(1), "B*H");
    s.nk = small_int(x.size(2), "tokens"); s.dv = small_int(x.size(3), "channels");
    s.bk = static_cast<int>(block); s.vp = (s.dv + 31) / 32 * 32;
    s.nk_tiles = small_int((int64_t(s.nk) + block - 1) / block, "V tiles");
    small_int(int64_t(s.heads) * s.nk_tiles * s.vp, "V grid");
    s.smooth = smooth; s.mean_type = static_cast<int>(mean_type);
    auto codes = at::empty({x.size(0), x.size(1), s.nk_tiles, s.vp, s.bk}, x.options().dtype(at::kByte));
    auto scales = at::empty({x.size(0), x.size(1), s.nk_tiles, s.vp}, x.options());
    auto means = at::empty(scales.sizes(), x.options().dtype(mean_type == 1 ? at::kHalf : mean_type == 2 ? at::kBFloat16 : at::kFloat));
    vc::Buffers b{}; b.v8 = codes.data_ptr<uint8_t>(); b.vs = scales.data_ptr<float>(); b.mean = means.data_ptr();
    vc::debug_values(x.data_ptr<float>(), s, b, c10::cuda::getCurrentCUDAStream(x.get_device()).stream());
    return {codes, scales, means};
}

Tensor debug_gemm(Tensor a, Tensor b) {
    require_hopper(a);
    debug_tensor(a, a, at::kByte, "GEMM A"); debug_tensor(b, a, at::kByte, "GEMM B");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(1), "Expected row-major [M,K] and [N,K] payload bytes");
    block_size(a.size(0), "M"); block_size(b.size(0), "N"); block_size(a.size(1), "K");
    const c10::cuda::CUDAGuard guard(a.device());
    auto output = at::empty({b.size(0), a.size(0)}, a.options().dtype(at::kFloat));
    auto workspace = at::empty({32 * 1024 * 1024}, a.options());
    vc::Buffers scratch{}; scratch.workspace = workspace.data_ptr(); scratch.workspace_bytes = workspace.numel();
    vc::debug_gemm(a.data_ptr<uint8_t>(), b.data_ptr<uint8_t>(), output.data_ptr<float>(),
                   static_cast<int>(a.size(0)), static_cast<int>(b.size(0)), static_cast<int>(a.size(1)),
                   scratch, blaslt_handle(a.get_device()), c10::cuda::getCurrentCUDAStream(a.get_device()).stream());
    return output.transpose(0, 1).contiguous();
}

std::vector<Tensor> debug_softmax(Tensor scores, Tensor qs, Tensor ks, Tensor maximum,
                                  Tensor denom, double scale, int64_t rows, int64_t keys, bool cast) {
    for (const auto& x : {scores, qs, ks, maximum, denom}) debug_tensor(x, scores, at::kFloat, "Softmax inputs");
    TORCH_CHECK(scores.dim() == 2 && qs.numel() == 1 && ks.numel() == 1, "Invalid score/scale dimensions");
    block_size(scores.size(0), "query tile"); block_size(scores.size(1), "key tile");
    TORCH_CHECK(maximum.dim() == 1 && denom.sizes() == maximum.sizes() && maximum.numel() == scores.size(0), "Invalid row state");
    TORCH_CHECK(rows > 0 && rows <= scores.size(0) && keys > 0 && keys <= scores.size(1), "Invalid valid-row/key counts");
    TORCH_CHECK(std::isfinite(scale) && scale > 0 && std::isfinite(static_cast<float>(scale)), "Invalid scale");
    const c10::cuda::CUDAGuard guard(scores.device());
    vc::Shape s{};
    s.bq = static_cast<int>(scores.size(0)); s.bk = static_cast<int>(scores.size(1)); s.k_quant = s.bk;
    s.nq = static_cast<int>(rows); s.nk = static_cast<int>(keys); s.nq_tiles = s.nk_quant_tiles = 1;
    s.expcast = cast; s.scale = static_cast<float>(scale);
    auto packed = scores.transpose(0, 1).contiguous();
    auto p = at::empty(scores.sizes(), scores.options().dtype(at::kByte));
    auto alpha = at::empty_like(maximum), mass = at::empty_like(maximum);
    maximum = maximum.clone(); denom = denom.clone();
    vc::Buffers b{};
    b.scores = packed.data_ptr<float>(); b.p8 = p.data_ptr<uint8_t>(); b.qs = qs.data_ptr<float>(); b.ks = ks.data_ptr<float>();
    b.maximum = maximum.data_ptr<float>(); b.denom = denom.data_ptr<float>(); b.alpha = alpha.data_ptr<float>(); b.mass = mass.data_ptr<float>();
    vc::debug_softmax(s, b, c10::cuda::getCurrentCUDAStream(scores.get_device()).stream());
    return {p, alpha, mass, maximum, denom};
}

std::vector<Tensor> debug_accumulate(Tensor accum, Tensor pv, Tensor mass, Tensor alpha,
                                     Tensor vs, Tensor mean, Tensor denom, int64_t rows) {
    for (const auto& x : {accum, pv, mass, alpha, vs, denom}) debug_tensor(x, accum, at::kFloat, "Accumulation inputs");
    TORCH_CHECK(mean.scalar_type() == at::kFloat || mean.scalar_type() == at::kHalf || mean.scalar_type() == at::kBFloat16, "Invalid mean dtype");
    debug_tensor(mean, accum, mean.scalar_type(), "Mean");
    TORCH_CHECK(accum.dim() == 2 && pv.sizes() == accum.sizes(), "Invalid PV/accumulator dimensions");
    block_size(accum.size(0), "query tile"); block_size(accum.size(1), "value channels");
    TORCH_CHECK(mass.dim() == 1 && alpha.sizes() == mass.sizes() && denom.sizes() == mass.sizes() && mass.numel() == accum.size(0), "Invalid accumulation row state");
    TORCH_CHECK(vs.dim() == 1 && mean.sizes() == vs.sizes() && vs.numel() == accum.size(1), "Invalid value metadata");
    TORCH_CHECK(rows > 0 && rows <= accum.size(0), "Invalid valid-row count");
    const c10::cuda::CUDAGuard guard(accum.device());
    vc::Shape s{}; s.bq = static_cast<int>(accum.size(0)); s.vp = s.dv = static_cast<int>(accum.size(1));
    s.nq = static_cast<int>(rows); s.nk_tiles = 1;
    s.mean_type = mean.scalar_type() == at::kHalf ? 1 : mean.scalar_type() == at::kBFloat16 ? 2 : 0;
    auto packed = pv.transpose(0, 1).contiguous();
    auto output = at::zeros_like(accum); accum = accum.clone();
    vc::Buffers b{}; b.accum = accum.data_ptr<float>(); b.pv = packed.data_ptr<float>(); b.mass = mass.data_ptr<float>();
    b.alpha = alpha.data_ptr<float>(); b.vs = vs.data_ptr<float>(); b.mean = mean.data_ptr(); b.denom = denom.data_ptr<float>(); b.output = output.data_ptr<float>();
    vc::debug_accumulate(s, b, c10::cuda::getCurrentCUDAStream(accum.get_device()).stream());
    return {accum, output};
}

std::vector<Tensor> debug_batched_step(std::vector<Tensor> x, int64_t qb, int64_t kquant,
                                       int64_t nq, int64_t nk, int64_t start, int64_t ki,
                                       double scale, bool cast) {
    TORCH_CHECK(x.size() == 9, "Expected scores, PV, Q scales, K scales, V scales, means, max, denom, accum");
    for (int i : {0, 1, 2, 3, 4, 6, 7, 8}) debug_tensor(x[i], x[0], at::kFloat, "Batched diagnostic");
    TORCH_CHECK(x[5].scalar_type() == at::kFloat || x[5].scalar_type() == at::kHalf || x[5].scalar_type() == at::kBFloat16, "Invalid mean dtype");
    debug_tensor(x[5], x[0], x[5].scalar_type(), "Batched mean");
    TORCH_CHECK(x[0].dim() == 3 && x[1].dim() == 3, "Expected [heads, rows, channels]");
    block_size(qb, "Q block"); block_size(x[0].size(2), "KV block"); block_size(x[1].size(2), "V width");
    const c10::cuda::CUDAGuard guard(x[0].device());
    vc::Shape s{};
    s.heads = small_int(x[0].size(0), "heads"); s.query_chunk = small_int(x[0].size(1), "rows");
    s.bq = static_cast<int>(qb); s.bk = static_cast<int>(x[0].size(2));
    s.vp = s.dv = static_cast<int>(x[1].size(2));
    s.nq = small_int(nq, "Q length"); s.nk = small_int(nk, "KV length");
    TORCH_CHECK(kquant >= s.bk && kquant <= 4096 && kquant % s.bk == 0, "Invalid K quantization block");
    s.k_quant = static_cast<int>(kquant);
    s.nq_tiles = small_int((nq + qb - 1) / qb, "Q tiles");
    s.nk_tiles = small_int((nk + s.bk - 1) / s.bk, "KV tiles");
    s.nk_quant_tiles = small_int((nk + kquant - 1) / kquant, "K scale tiles");
    TORCH_CHECK(s.query_chunk <= 1024 && s.query_chunk % qb == 0 && start >= 0 && start < nq && start % qb == 0 &&
                start + s.query_chunk <= int64_t(s.nq_tiles) * qb && ki >= 0 && ki < s.nk_tiles, "Invalid diagnostic tile");
    TORCH_CHECK(std::isfinite(scale) && scale > 0 && std::isfinite(static_cast<float>(scale)), "Invalid scale");
    small_int(int64_t(s.heads) * s.query_chunk, "Softmax grid");
    auto shape = [](const Tensor& t, std::initializer_list<int64_t> dims) { return t.sizes() == at::IntArrayRef(dims); };
    TORCH_CHECK(shape(x[1], {s.heads, s.query_chunk, s.vp}) && x[8].sizes() == x[1].sizes(), "Invalid PV/accumulator shape");
    TORCH_CHECK(shape(x[2], {s.heads, s.nq_tiles}) && shape(x[3], {s.heads, s.nk_quant_tiles}), "Invalid Q/K metadata");
    TORCH_CHECK(shape(x[4], {s.heads, s.nk_tiles, s.vp}) && x[5].sizes() == x[4].sizes(), "Invalid V metadata");
    TORCH_CHECK(shape(x[6], {s.heads, s.query_chunk}) && x[7].sizes() == x[6].sizes(), "Invalid row state");
    s.expcast = cast; s.scale = static_cast<float>(scale);
    s.mean_type = x[5].scalar_type() == at::kHalf ? 1 : x[5].scalar_type() == at::kBFloat16 ? 2 : 0;
    auto scores = x[0].transpose(1, 2).contiguous(), pv = x[1].transpose(1, 2).contiguous();
    auto maximum = x[6].clone(), denom = x[7].clone(), accum = x[8].clone();
    auto alpha = at::empty_like(maximum), mass = at::empty_like(maximum);
    auto p = at::empty(x[0].sizes(), x[0].options().dtype(at::kByte));
    auto output = at::zeros({s.heads, s.nq, s.dv}, x[0].options());
    vc::Buffers b{};
    b.scores = scores.data_ptr<float>(); b.pv = pv.data_ptr<float>(); b.p8 = p.data_ptr<uint8_t>();
    b.qs = x[2].data_ptr<float>(); b.ks = x[3].data_ptr<float>(); b.vs = x[4].data_ptr<float>(); b.mean = x[5].data_ptr();
    b.maximum = maximum.data_ptr<float>(); b.denom = denom.data_ptr<float>(); b.accum = accum.data_ptr<float>();
    b.alpha = alpha.data_ptr<float>(); b.mass = mass.data_ptr<float>(); b.output = output.data_ptr<float>();
    vc::debug_batched_update(s, b, static_cast<int>(start), static_cast<int>(ki), ki + 1 == s.nk_tiles,
                             c10::cuda::getCurrentCUDAStream(x[0].get_device()).stream());
    return {p, alpha, mass, maximum, denom, accum, output};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("numerical_contract_version") = 2;
    m.attr("scheduling_version") = 1;
    m.attr("fused_api_version") = 1;
    m.attr("fused_pipeline_version") = 1;
    m.attr("fused_d128_version") = 1;
    m.def("all_finite", &all_finite);
    m.def("debug_batched_step", &debug_batched_step);
    m.attr("diagnostic_api_version") = 1;
    m.def("debug_quantize", &debug_quantize);
    m.def("debug_values", &debug_values);
    m.def("debug_gemm", &debug_gemm);
    m.def("debug_softmax", &debug_softmax);
    m.def("debug_accumulate", &debug_accumulate);
    const char* names[] = {"forward", "forward_fused", "debug_fused_emulated",
                           "forward_fused_baseline", "debug_fused_baseline_emulated",
                           "forward_fused_d128", "debug_fused_d128_emulated"};
    using Forward = decltype(&forward<0>);
    const Forward entries[] = {&forward<0>, &forward<1>, &forward<2>, &forward<3>, &forward<4>, &forward<5>, &forward<6>};
    for (int mode = 0; mode < 7; ++mode) m.def(names[mode], entries[mode],
          "VC forward; debug_fused_emulated is a small-tensor numerical diagnostic only",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("q_block"), pybind11::arg("k_quant_block"), pybind11::arg("kv_block"),
          pybind11::arg("smooth"), pybind11::arg("expcast"), pybind11::arg("mean_type"),
          pybind11::arg("scale"), pybind11::arg("k_smooth"), pybind11::arg("hadamard"),
          pybind11::arg("batched") = false);
    m.def("preprocess_qk", &preprocess_qk, "Post-RoPE K centering and normalized Q/K Hadamard (CUDA FP32)");
    m.def("group_values", &group_values, "CUDA Lloyd clustering with stable permutation");
    m.def("expcast_codes", &expcast_codes, "ExpCast FP8 payload bytes");
    m.def("build_info", []() {
        pybind11::dict info;
        info["cuda_headers"] = CUDART_VERSION;
        info["cublaslt_version"] = cublasLtGetVersion();
        info["matmul"] = "cuBLASLt E4M3 x E4M3, FP32 output/accumulation";
        info["fully_fused_attention"] = false;
        info["fused_attention_available"] = true;
        info["fused_pipeline_version"] = 1;
        info["fused_d128_version"] = 1;
        info["fused_pipeline"] = "16-byte cp.async; separate K/V buffers; V overlaps QK, next K overlaps PV";
        info["fused_matmul"] = "Hopper WGMMA m64n128k32 E4M3 x E4M3, FP32 accumulation (sm_90a)";
        info["preprocessing_fused"] = false;
        info["numerical_contract_version"] = 2;
        info["scheduling_version"] = 1;
        info["default_schedule"] = "batched_heads_and_queries";
        info["diagnostic_api_version"] = 1;
        info["expcast_rounding"] = "FP32 FMA then RNE";
        info["value_mean_storage"] = "mu / per-channel V scale at configured mean dtype";
        return info;
    });
}
