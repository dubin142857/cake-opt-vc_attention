#include "vc_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <memory>
#include <tuple>
#include <vector>

namespace vc {
namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void blas_check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed, cuBLAS status=" + std::to_string(status));
}

__device__ float reduce_max(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

__device__ float reduce_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

__device__ uint8_t encode(float x) {
    return __nv_cvt_float_to_fp8(fminf(448.0f, fmaxf(-448.0f, x)), __NV_SATFINITE, __NV_E4M3);
}

__device__ float as_float(float x) { return x; }
__device__ float as_float(__half x) { return __half2float(x); }
__device__ float as_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ float read_view(InputView view, int64_t index) {
    int64_t offset = index;
    if (!view.contiguous) {
        offset = 0;
        for (int dim = 3; dim >= 0; --dim) {
            offset += (index % view.shape[dim]) * view.stride[dim];
            index /= view.shape[dim];
        }
    }
    return as_float(static_cast<const T*>(view.data)[offset]);
}

template <typename T>
__global__ void check_finite_kernel(InputView q, InputView k, InputView v, int* invalid) {
    const int64_t total = q.size + k.size + v.size;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += int64_t(blockDim.x) * gridDim.x) {
        const float value = i < q.size ? read_view<T>(q, i)
                          : i < q.size + k.size ? read_view<T>(k, i - q.size)
                          : read_view<T>(v, i - q.size - k.size);
        if (!isfinite(value)) { atomicExch(invalid, 1); return; }
    }
}

__device__ float decode_positive(uint8_t code) {
    const int exponent = code >> 3;
    const int mantissa = code & 7;
    return exponent == 0 ? ldexpf(static_cast<float>(mantissa), -9)
                         : ldexpf(1.0f + mantissa * 0.125f, exponent - 7);
}

__device__ uint8_t expcast_code(float x) {
    // Paper Eq. 7: one FP32 fused multiply-add, then round to nearest even.
    const float u = __fmaf_rn(x, 11.541560173034668f, 119.6500015258789f);
    return static_cast<uint8_t>(__float2int_rn(fminf(120.0f, fmaxf(0.0f, u))));
}

__global__ void expcast_kernel(const float* x, uint8_t* y, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += int64_t(blockDim.x) * gridDim.x) y[i] = expcast_code(x[i]);
}

__global__ void transform_qk_kernel(const float* input, const float* mean, float* output,
                                    int tokens, int width, int out_width, bool hadamard, float normalization) {
    __shared__ float values[128];
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    float value = col < width ? input[int64_t(row) * width + col] : 0.0f;
    if (mean && col < width) value -= mean[int64_t(row / tokens) * width + col];
    values[col] = value;
    __syncthreads();
    if (hadamard) {
        for (int stride = 1; stride < out_width; stride *= 2) {
            const float other = values[col ^ stride];
            const float own = values[col];
            __syncthreads();
            values[col] = (col & stride) ? other - own : own + other;
            __syncthreads();
        }
        value = values[col] * normalization;
    }
    if (col < out_width) output[int64_t(row) * out_width + col] = value;
}

// A scale belongs to a sequence block, independently for each batch/head.
// D <= 128, hence one channel group per Q/K block, as in the reference.
__global__ void quantize_qk(const float* input, uint8_t* output, float* scales,
                           int n, int padded_n, int d, int padded_d,
                           int tile_size, int tile_count) {
    __shared__ float scratch[256];
    const int head = blockIdx.x / tile_count;
    const int tile = blockIdx.x % tile_count;
    const int start = tile * tile_size;
    float amax = 0.0f;
    for (int i = threadIdx.x; i < tile_size * d; i += blockDim.x) {
        const int row = start + i / d;
        if (row < n) amax = fmaxf(amax, fabsf(input[(int64_t(head) * n + row) * d + i % d]));
    }
    amax = reduce_max(amax, scratch);
    const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
    if (threadIdx.x == 0) scales[head * tile_count + tile] = scale;
    for (int i = threadIdx.x; i < tile_size * padded_d; i += blockDim.x) {
        const int row = start + i / padded_d;
        const int col = i % padded_d;
        if (row < padded_n) {
            const float value = row < n && col < d
                ? input[(int64_t(head) * n + row) * d + col] : 0.0f;
            output[(int64_t(head) * padded_n + row) * padded_d + col] = encode(value / scale);
        }
    }
}

// Store each V tile column-major: [head, kv_tile, channel, token]. This is
// precisely the non-transposed B operand required by FP8 cuBLASLt TN GEMM.
__global__ void prepare_values(const float* input, uint8_t* output, float* scales,
                               void* means, Shape s) {
    __shared__ float scratch[256];
    const int channel = blockIdx.x % s.vp;
    const int tile_head = blockIdx.x / s.vp;
    const int tile = tile_head % s.nk_tiles;
    const int head = tile_head / s.nk_tiles;
    const int start = tile * s.bk;
    const int count = min(s.bk, s.nk - start);
    float sum = 0.0f;
    for (int row = threadIdx.x; row < count; row += blockDim.x) {
        if (channel < s.dv)
            sum += input[(int64_t(head) * s.nk + start + row) * s.dv + channel];
    }
    float mean = s.smooth ? reduce_sum(sum, scratch) / count : 0.0f;
    float amax = 0.0f;
    for (int row = threadIdx.x; row < count; row += blockDim.x) {
        if (channel < s.dv)
            amax = fmaxf(amax, fabsf(input[(int64_t(head) * s.nk + start + row) * s.dv + channel] - mean));
    }
    amax = reduce_max(amax, scratch);
    float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
    const float mean_limit = s.mean_type == 1 ? 65504.0f
                           : s.mean_type == 2 ? 3.3895313892515355e38f : 3.4028234663852886e38f;
    scale = fmaxf(scale, fabsf(mean) / mean_limit);
    const int64_t index = int64_t(tile_head) * s.vp + channel;
    if (threadIdx.x == 0) {
        const float scaled_mean = mean / scale;
        if (s.mean_type == 1) static_cast<__half*>(means)[index] = __float2half_rn(scaled_mean);
        else if (s.mean_type == 2) static_cast<__nv_bfloat16*>(means)[index] = __float2bfloat16_rn(scaled_mean);
        else static_cast<float*>(means)[index] = scaled_mean;
        scales[index] = scale;
    }
    for (int row = threadIdx.x; row < s.bk; row += blockDim.x) {
        const float value = row < count && channel < s.dv
            ? input[(int64_t(head) * s.nk + start + row) * s.dv + channel] - mean : 0.0f;
        output[index * s.bk + row] = encode(value / scale);
    }
}

__global__ void init_state(Buffers b, Shape s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < s.bq) { b.maximum[i] = -CUDART_INF_F; b.denom[i] = 0.0f; }
    if (i < s.bq * s.vp) b.accum[i] = 0.0f;
}

// QK GEMM produces column-major scores. P is written row-major so it can be
// interpreted as a transposed column-major A operand in the next GEMM.
__device__ void softmax_row(Buffers b, Shape s, int head, int query,
                            int row, int matrix_rows, int k_tile,
                            bool first, float* scratch) {
    const int j = threadIdx.x;
    const bool valid_row = query < s.nq;
    const bool valid_key = j < s.bk && k_tile * s.bk + j < s.nk;
    const float qs = b.qs[head * s.nq_tiles + query / s.bq];
    const float ks = b.ks[head * s.nk_quant_tiles + (k_tile * s.bk) / s.k_quant];
    float score = -CUDART_INF_F;
    if (valid_row && valid_key)
        score = ((b.scores[int64_t(j) * matrix_rows + row] * qs) * ks) * s.scale;
    const float tile_max = reduce_max(score, scratch);
    const float old_max = first ? -CUDART_INF_F : b.maximum[row];
    const float new_max = valid_row ? fmaxf(tile_max, old_max) : 0.0f;
    const float alpha = valid_row ? expf(old_max - new_max) : 0.0f;
    float probability = 0.0f;
    uint8_t payload = 0;
    if (valid_row && valid_key) {
        const float shifted = score - new_max;
        if (s.expcast) {
            payload = expcast_code(shifted);
            probability = decode_positive(payload) / 256.0f;
        } else {
            probability = expf(shifted);
            payload = encode(probability * 256.0f);
        }
    }
    if (j < s.bk) b.p8[row * s.bk + j] = payload;
    const float mass = reduce_sum(probability, scratch);
    if (j == 0) {
        b.alpha[row] = alpha;
        b.mass[row] = mass;
        b.maximum[row] = new_max;
        b.denom[row] = alpha * (first ? 0.0f : b.denom[row]) + mass;
    }
}

__global__ void softmax_update(Buffers b, Shape s, int head, int q_tile, int k_tile) {
    __shared__ float scratch[256];
    softmax_row(b, s, head, q_tile * s.bq + blockIdx.x,
                blockIdx.x, s.bq, k_tile, false, scratch);
}

__global__ void batched_softmax(Buffers b, Shape s, int query_start, int k_tile) {
    __shared__ float scratch[256];
    const int head = blockIdx.x / s.query_chunk;
    const int row = blockIdx.x % s.query_chunk;
    const int64_t base = int64_t(head) * s.query_chunk;
    b.scores += base * s.bk; b.p8 += base * s.bk;
    b.maximum += base; b.denom += base; b.alpha += base; b.mass += base;
    softmax_row(b, s, head, query_start + row, row, s.query_chunk,
                k_tile, k_tile == 0, scratch);
}

__global__ void batched_accumulate(Buffers b, Shape s, int query_start, int k_tile, bool finish) {
    const int64_t count = int64_t(s.heads) * s.query_chunk * s.vp;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += int64_t(blockDim.x) * gridDim.x) {
        const int col = i % s.vp;
        const int64_t state = i / s.vp;
        const int head = state / s.query_chunk, row = state % s.query_chunk;
        const int64_t meta = (int64_t(head) * s.nk_tiles + k_tile) * s.vp + col;
        const float mean = s.mean_type == 1 ? __half2float(static_cast<const __half*>(b.mean)[meta])
                         : s.mean_type == 2 ? __bfloat162float(static_cast<const __nv_bfloat16*>(b.mean)[meta])
                         : static_cast<const float*>(b.mean)[meta];
        const float product = b.pv[(int64_t(head) * s.vp + col) * s.query_chunk + row];
        const float contribution = (product / 256.0f + b.mass[state] * mean) * b.vs[meta];
        const float result = b.alpha[state] * (k_tile == 0 ? 0.0f : b.accum[i]) + contribution;
        if (finish) {
            if (query_start + row < s.nq && col < s.dv)
                b.output[(int64_t(head) * s.nq + query_start + row) * s.dv + col] = result / b.denom[state];
        } else {
            b.accum[i] = result;
        }
    }
}

__global__ void accumulate(Buffers b, Shape s, int head, int k_tile) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.bq * s.vp) return;
    const int row = i / s.vp;
    const int col = i % s.vp;
    const int64_t meta = (int64_t(head) * s.nk_tiles + k_tile) * s.vp + col;
    const float mean = s.mean_type == 1 ? __half2float(static_cast<const __half*>(b.mean)[meta])
                     : s.mean_type == 2 ? __bfloat162float(static_cast<const __nv_bfloat16*>(b.mean)[meta])
                     : static_cast<const float*>(b.mean)[meta];
    const float contribution = (b.pv[col * s.bq + row] / 256.0f + b.mass[row] * mean) * b.vs[meta];
    b.accum[i] = b.alpha[row] * b.accum[i] + contribution;
}

__global__ void write_output(Buffers b, Shape s, int head, int q_tile) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.bq * s.dv) return;
    const int row = i / s.dv;
    const int col = i % s.dv;
    const int q = q_tile * s.bq + row;
    if (q < s.nq)
        b.output[(int64_t(head) * s.nq + q) * s.dv + col] = b.accum[row * s.vp + col] / b.denom[row];
}

// Descriptors/heuristics are built twice per invocation, not per tile. Scratch
// is reused only on the same CUDA stream, after its previous consumer.
class Fp8Gemm {
    cublasLtMatmulDesc_t operation_ = nullptr;
    cublasLtMatrixLayout_t a_ = nullptr, b_ = nullptr, c_ = nullptr;
    cublasLtMatmulPreference_t preference_ = nullptr;
    cublasLtMatmulAlgo_t algorithm_{};
    void release() noexcept {
        if (preference_) cublasLtMatmulPreferenceDestroy(preference_);
        if (c_) cublasLtMatrixLayoutDestroy(c_);
        if (b_) cublasLtMatrixLayoutDestroy(b_);
        if (a_) cublasLtMatrixLayoutDestroy(a_);
        if (operation_) cublasLtMatmulDescDestroy(operation_);
    }
public:
    Fp8Gemm(cublasLtHandle_t handle, int m, int n, int k, size_t workspace_bytes,
            int batch = 1, int64_t stride_a = 0, int64_t stride_b = 0, int64_t stride_c = 0) {
        try {
            blas_check(cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F), "MatmulDescCreate");
            cublasOperation_t trans_a = CUBLAS_OP_T, trans_b = CUBLAS_OP_N;
            blas_check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)), "TRANSA");
            blas_check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)), "TRANSB");
            int8_t fast_accum = 0;
            blas_check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)), "FAST_ACCUM");
            // Scale pointers are unset: FP8 payload products use scale 1.
            // Q/K block and V channel scales are restored by our CUDA kernels.
            blas_check(cublasLtMatrixLayoutCreate(&a_, CUDA_R_8F_E4M3, k, m, k), "A layout");
            blas_check(cublasLtMatrixLayoutCreate(&b_, CUDA_R_8F_E4M3, k, n, k), "B layout");
            blas_check(cublasLtMatrixLayoutCreate(&c_, CUDA_R_32F, m, n, m), "C/D layout");
            if (batch > 1) {
                for (auto layout : {a_, b_, c_})
                    blas_check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)), "batch count");
                blas_check(cublasLtMatrixLayoutSetAttribute(a_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)), "A batch stride");
                blas_check(cublasLtMatrixLayoutSetAttribute(b_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)), "B batch stride");
                blas_check(cublasLtMatrixLayoutSetAttribute(c_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)), "C batch stride");
            }
            blas_check(cublasLtMatmulPreferenceCreate(&preference_), "PreferenceCreate");
            blas_check(cublasLtMatmulPreferenceSetAttribute(preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)), "workspace preference");
            cublasLtMatmulHeuristicResult_t candidates[8]{};
            int count = 0;
            blas_check(cublasLtMatmulAlgoGetHeuristic(handle, operation_, a_, b_, c_, c_, preference_, 8, candidates, &count), "FP8 heuristic");
            bool found = false;
            for (int i = 0; i < count; ++i) {
                if (candidates[i].state == CUBLAS_STATUS_SUCCESS && candidates[i].workspaceSize <= workspace_bytes) {
                    algorithm_ = candidates[i].algo;
                    found = true;
                    break;
                }
            }
            if (!found) throw std::runtime_error("No FP8->FP32 cuBLASLt algorithm for this shape/runtime. No fallback was used.");
        } catch (...) { release(); throw; }
    }
    ~Fp8Gemm() { release(); }
    Fp8Gemm(const Fp8Gemm&) = delete;
    Fp8Gemm& operator=(const Fp8Gemm&) = delete;
    void run(cublasLtHandle_t handle, const uint8_t* a, const uint8_t* b,
             float* out, const Buffers& buffers, cudaStream_t stream) const {
        const float alpha = 1.0f, beta = 0.0f;
        blas_check(cublasLtMatmul(handle, operation_, &alpha, a, a_, b, b_,
                                 &beta, out, c_, out, c_, &algorithm_, buffers.workspace,
                                 buffers.workspace_bytes, stream), "FP8 Tensor Core matmul");
    }
};

// Cache only immutable descriptors/algorithms, never input data or workspace.
// Handles belong to a device and host thread; include the handle in the key.
std::shared_ptr<Fp8Gemm> cached_gemm(cublasLtHandle_t handle, int m, int n, int k,
                                    size_t workspace, int batch, int64_t sa, int64_t sb, int64_t sc) {
    using Key = std::tuple<cublasLtHandle_t, int, int, int, size_t, int, int64_t, int64_t, int64_t>;
    using Entry = std::pair<Key, std::shared_ptr<Fp8Gemm>>;
    static thread_local std::vector<Entry> cache;
    const Key key{handle, m, n, k, workspace, batch, sa, sb, sc};
    for (const auto& entry : cache) if (entry.first == key) return entry.second;
    auto plan = std::make_shared<Fp8Gemm>(handle, m, n, k, workspace, batch, sa, sb, sc);
    if (cache.size() == 32) cache.erase(cache.begin());
    cache.emplace_back(key, plan);
    return plan;
}

}  // namespace

void check_finite(InputView q, InputView k, InputView v, int dtype, int* invalid, cudaStream_t stream) {
    cuda_check(cudaMemsetAsync(invalid, 0, sizeof(int), stream), "finite flag initialization");
    const int blocks = static_cast<int>(std::min<int64_t>((q.size + k.size + v.size + 255) / 256, 65535));
    if (dtype == 0) check_finite_kernel<float><<<blocks, 256, 0, stream>>>(q, k, v, invalid);
    else if (dtype == 1) check_finite_kernel<__half><<<blocks, 256, 0, stream>>>(q, k, v, invalid);
    else check_finite_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(q, k, v, invalid);
    cuda_check(cudaGetLastError(), "Q/K/V finite check");
}

void expcast(const float* input, uint8_t* output, int64_t size, cudaStream_t stream) {
    if (!size) return;
    expcast_kernel<<<static_cast<unsigned>(std::min<int64_t>((size + 255) / 256, 65535)), 256, 0, stream>>>(input, output, size);
    cuda_check(cudaGetLastError(), "ExpCast launch");
}

void transform_qk(const float* input, const float* mean, float* output,
                  int heads, int tokens, int width, int out_width,
                  bool hadamard, cudaStream_t stream) {
    const float normalization = static_cast<float>(1.0 / std::sqrt(static_cast<double>(out_width)));
    transform_qk_kernel<<<heads * tokens, 128, 0, stream>>>(input, mean, output, tokens, width, out_width, hadamard, normalization);
    cuda_check(cudaGetLastError(), "Q/K centering and Hadamard launch");
}

void debug_quantize(const float* input, uint8_t* output, float* scales,
                    int heads, int n, int padded_n, int d, int padded_d,
                    int block, int tiles, cudaStream_t stream) {
    quantize_qk<<<heads * tiles, 256, 0, stream>>>(input, output, scales, n, padded_n, d, padded_d, block, tiles);
    cuda_check(cudaGetLastError(), "diagnostic Q/K quantization");
}

void debug_values(const float* input, const Shape& s, const Buffers& b, cudaStream_t stream) {
    prepare_values<<<s.heads * s.nk_tiles * s.vp, 256, 0, stream>>>(input, b.v8, b.vs, b.mean, s);
    cuda_check(cudaGetLastError(), "diagnostic V quantization");
}

void debug_gemm(const uint8_t* a, const uint8_t* b, float* output, int m, int n, int k,
                const Buffers& scratch, cublasLtHandle_t handle, cudaStream_t stream) {
    Fp8Gemm gemm(handle, m, n, k, scratch.workspace_bytes);
    gemm.run(handle, a, b, output, scratch, stream);
}

void debug_softmax(const Shape& s, const Buffers& b, cudaStream_t stream) {
    softmax_update<<<s.bq, 256, 0, stream>>>(b, s, 0, 0, 0);
    cuda_check(cudaGetLastError(), "diagnostic Softmax");
}

void debug_accumulate(const Shape& s, const Buffers& b, cudaStream_t stream) {
    accumulate<<<(s.bq * s.vp + 255) / 256, 256, 0, stream>>>(b, s, 0, 0);
    write_output<<<(s.bq * s.dv + 255) / 256, 256, 0, stream>>>(b, s, 0, 0);
    cuda_check(cudaGetLastError(), "diagnostic accumulation and normalization");
}

void debug_batched_update(const Shape& s, const Buffers& b, int query_start,
                          int key_tile, bool finish, cudaStream_t stream) {
    batched_softmax<<<s.heads * s.query_chunk, 256, 0, stream>>>(b, s, query_start, key_tile);
    cuda_check(cudaGetLastError(), "batched Softmax");
    const int blocks = static_cast<int>(std::min<int64_t>((int64_t(s.heads) * s.query_chunk * s.vp + 255) / 256, 65535));
    batched_accumulate<<<blocks, 256, 0, stream>>>(b, s, query_start, key_tile, finish);
    cuda_check(cudaGetLastError(), "batched accumulation");
}

void forward_batched(const float* q, const float* k, const float* v,
                     const Shape& s, const Buffers& b, cublasLtHandle_t handle, cudaStream_t stream) {
    debug_quantize(q, b.q8, b.qs, s.heads, s.nq, s.pq, s.d, s.dp, s.bq, s.nq_tiles, stream);
    debug_quantize(k, b.k8, b.ks, s.heads, s.nk, s.pk, s.d, s.dp, s.k_quant, s.nk_quant_tiles, stream);
    debug_values(v, s, b, stream);
    for (int start = 0; start < s.pq; start += s.query_chunk) {
        Shape chunk = s;
        const int rows = chunk.query_chunk = std::min(s.query_chunk, s.pq - start);
        auto qk = cached_gemm(handle, rows, s.bk, s.dp, b.workspace_bytes, s.heads,
                               int64_t(s.pq) * s.dp, int64_t(s.pk) * s.dp, int64_t(rows) * s.bk);
        auto pv = cached_gemm(handle, rows, s.vp, s.bk, b.workspace_bytes, s.heads,
                               int64_t(rows) * s.bk, int64_t(s.nk_tiles) * s.vp * s.bk, int64_t(rows) * s.vp);
        for (int ki = 0; ki < s.nk_tiles; ++ki) {
            qk->run(handle, b.q8 + int64_t(start) * s.dp,
                     b.k8 + int64_t(ki) * s.bk * s.dp, b.scores, b, stream);
            batched_softmax<<<s.heads * rows, 256, 0, stream>>>(b, chunk, start, ki);
            cuda_check(cudaGetLastError(), "batched Softmax/ExpCast");
            pv->run(handle, b.p8, b.v8 + int64_t(ki) * s.vp * s.bk, b.pv, b, stream);
            const int blocks = static_cast<int>(std::min<int64_t>((int64_t(s.heads) * rows * s.vp + 255) / 256, 65535));
            batched_accumulate<<<blocks, 256, 0, stream>>>(b, chunk, start, ki, ki + 1 == s.nk_tiles);
            cuda_check(cudaGetLastError(), "batched online output update");
        }
    }
}

void forward(const float* q, const float* k, const float* v,
             const Shape& s, const Buffers& b, cublasLtHandle_t handle, cudaStream_t stream) {
    Fp8Gemm qk(handle, s.bq, s.bk, s.dp, b.workspace_bytes);
    Fp8Gemm pv(handle, s.bq, s.vp, s.bk, b.workspace_bytes);
    quantize_qk<<<s.heads * s.nq_tiles, 256, 0, stream>>>(q, b.q8, b.qs, s.nq, s.pq, s.d, s.dp, s.bq, s.nq_tiles);
    cuda_check(cudaGetLastError(), "Q quantization launch");
    quantize_qk<<<s.heads * s.nk_quant_tiles, 256, 0, stream>>>(k, b.k8, b.ks, s.nk, s.pk, s.d, s.dp, s.k_quant, s.nk_quant_tiles);
    cuda_check(cudaGetLastError(), "K quantization launch");
    prepare_values<<<s.heads * s.nk_tiles * s.vp, 256, 0, stream>>>(v, b.v8, b.vs, b.mean, s);
    cuda_check(cudaGetLastError(), "V preprocessing launch");
    for (int head = 0; head < s.heads; ++head) {
        for (int qi = 0; qi < s.nq_tiles; ++qi) {
            init_state<<<(s.bq * s.vp + 255) / 256, 256, 0, stream>>>(b, s);
            for (int ki = 0; ki < s.nk_tiles; ++ki) {
                const uint8_t* qt = b.q8 + (int64_t(head) * s.pq + qi * s.bq) * s.dp;
                const uint8_t* kt = b.k8 + (int64_t(head) * s.pk + ki * s.bk) * s.dp;
                qk.run(handle, qt, kt, b.scores, b, stream);
                softmax_update<<<s.bq, 256, 0, stream>>>(b, s, head, qi, ki);
                cuda_check(cudaGetLastError(), "Softmax/ExpCast launch");
                const uint8_t* vt = b.v8 + (int64_t(head) * s.nk_tiles + ki) * s.vp * s.bk;
                pv.run(handle, b.p8, vt, b.pv, b, stream);
                accumulate<<<(s.bq * s.vp + 255) / 256, 256, 0, stream>>>(b, s, head, ki);
                cuda_check(cudaGetLastError(), "online output update launch");
            }
            write_output<<<(s.bq * s.dv + 255) / 256, 256, 0, stream>>>(b, s, head, qi);
            cuda_check(cudaGetLastError(), "output normalization launch");
        }
    }
}

}  // namespace vc
