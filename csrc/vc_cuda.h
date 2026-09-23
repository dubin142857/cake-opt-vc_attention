#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace vc {

struct Shape {
    int heads, nq, nk, d, dv;
    int bq, bk, k_quant;
    int pq, pk, dp, vp;  // Padded sequence lengths and channel dimensions.
    int nq_tiles, nk_tiles, nk_quant_tiles;
    bool smooth, expcast;
    int mean_type;  // 0=FP32, 1=FP16, 2=BF16.
    float scale;
};

struct Buffers {
    uint8_t *q8, *k8, *v8, *p8;
    float *qs, *ks, *vs;
    void *mean;  // mu / V-scale, physically stored at mean_type precision.
    float *scores, *pv, *maximum, *denom, *alpha, *mass, *accum;
    float *output;
    void *workspace;
    size_t workspace_bytes;
};

// Inputs/outputs and scratch buffers belong to PyTorch's allocator. All work
// uses the caller's current stream; no cudaDeviceSynchronize or default stream.
void forward(const float* q, const float* k, const float* v,
             const Shape& s, const Buffers& b,
             cublasLtHandle_t handle, cudaStream_t stream);
void expcast(const float* input, uint8_t* output, int64_t size, cudaStream_t stream);
void transform_qk(const float* input, const float* mean, float* output,
                  int heads, int tokens, int width, int out_width,
                  bool hadamard, cudaStream_t stream);

// Diagnostic entry points reuse the production kernels and GEMM descriptor.
// They do not alter the forward path or select alternate arithmetic.
void debug_quantize(const float* input, uint8_t* output, float* scales,
                    int heads, int n, int padded_n, int d, int padded_d,
                    int block, int tiles, cudaStream_t stream);
void debug_values(const float* input, const Shape& s, const Buffers& b, cudaStream_t stream);
void debug_gemm(const uint8_t* a, const uint8_t* b, float* output, int m, int n, int k,
                const Buffers& scratch, cublasLtHandle_t handle, cudaStream_t stream);
void debug_softmax(const Shape& s, const Buffers& b, cudaStream_t stream);
void debug_accumulate(const Shape& s, const Buffers& b, cudaStream_t stream);

}  // namespace vc
