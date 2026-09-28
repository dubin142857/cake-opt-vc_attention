// Diagnostic only: execute the actual ExpCast/mean-restoration device helpers
// without Hopper matrix instructions. This file is not linked into the package.
#include <torch/extension.h>
#include <math_constants.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include "../../third_party/sageattention_local/csrc/math.cuh"
#include "../../third_party/sageattention_local/csrc/qattn/attn_utils.cuh"
#include "../../third_party/sageattention_local/csrc/vc/vc_math.cuh"

template<bool Optimized, bool ChannelScale=false, bool PackedProbability=false, bool StreamMean=false>
__global__ void probe(const float* scores, const float* prev_out, const float* product,
    const float* prev_max, const float* prev_d, const float* scale, const half* mean,
    uint8_t* bytes, float* output, float* maximum, float* denom, bool mixed_scales) {
  int lane = threadIdx.x % 32, row_base = (threadIdx.x / 32) * 16 + lane / 4;
  float s[1][8][8], o[1][8][8], pv[1][8][8], m[1][2], d[1][2], mass[1][2];
  uint32_t packed[1][4][4];
  for (int r = 0; r < 2; ++r) {
    m[0][r] = prev_max[row_base + r * 8];
    d[0][r] = prev_d[row_base + r * 8] * 0.25f;
  }
  for (int t = 0; t < 8; ++t)
    for (int j = 0; j < 8; ++j) {
      int row = row_base + ((j % 4) / 2) * 8;
      int column = t * 16 + (lane % 4) * 2 + (j / 4) * 8 + j % 2;
      int pos = row * 128 + column;
      s[0][t][j] = scores[pos]; o[0][t][j] = prev_out[pos]; pv[0][t][j] = product[pos];
    }
  float score_scale = mixed_scales ? 0.137f * (1 + lane % 4) : 0.25f;
  vc_probability<true, true, Optimized, PackedProbability>(s,o,m,d,score_scale,packed,mass);
  if constexpr (ChannelScale && Optimized) {
    if constexpr (StreamMean) {
      #pragma unroll
      for (int part = 0; part < 2; ++part) {
        float half_pv[1][4][8];
        #pragma unroll
        for (int t = 0; t < 4; ++t)
          #pragma unroll
          for (int j = 0; j < 8; ++j) half_pv[0][t][j] = pv[0][part*4+t][j];
        vc_accumulate<true, true, true, true>(o,half_pv,mass,scale,mean,0,0,1,128,lane,part*4);
      }
    } else vc_accumulate<true, true, true>(o,pv,mass,scale,mean,0,0,1,128,lane);
    for (int t = 0; t < 8; ++t)
      for (int j = 0; j < 8; ++j) {
        int column = t * 16 + (lane % 4) * 2 + (j / 4) * 8 + j % 2;
        o[0][t][j] *= scale[column];
      }
  } else vc_accumulate<true, Optimized, ChannelScale>(o,pv,mass,scale,mean,0,0,1,128,lane);
  for (int t = 0; t < 8; ++t)
    for (int j = 0; j < 8; ++j) {
      int r = (j % 4) / 2;
      int row = row_base + r * 8;
      int column = t * 16 + (lane % 4) * 2 + (j / 4) * 8 + j % 2;
      int shift = 8 * ((j / 4) * 2 + j % 2);
      bytes[row * 128 + column] = (packed[0][t / 2][(t % 2) * 2 + r] >> shift) & 255;
      output[row * 128 + column] = o[0][t][j];
    }
  for (int r = 0; r < 2; ++r) {
    float sum = d[0][r];
    sum += __shfl_xor_sync(0xffffffff,sum,1);
    sum += __shfl_xor_sync(0xffffffff,sum,2);
    if (lane % 4 == 0) {
      maximum[row_base + r * 8] = m[0][r];
      denom[row_base + r * 8] = sum;
    }
  }
}

std::vector<torch::Tensor> run(torch::Tensor scores, torch::Tensor out, torch::Tensor product,
    torch::Tensor m, torch::Tensor d, torch::Tensor scale, torch::Tensor mean,
    bool optimized, bool mixed_scales, bool channel_scale, bool packed_probability, bool stream_mean) {
  TORCH_CHECK(scores.is_cuda() && scores.sizes() == at::IntArrayRef({64,128}), "Probe shape");
  auto bytes = torch::empty_like(scores,scores.options().dtype(at::kByte));
  auto output = torch::empty_like(out), maximum = torch::empty_like(m), denom = torch::empty_like(d);
  auto kernel = channel_scale ? (optimized ? probe<true,true> : probe<false,true>)
      : (optimized ? probe<true> : probe<false>);
  if (packed_probability) kernel = channel_scale ? (optimized ? probe<true,true,true> : probe<false,true,true>)
      : (optimized ? probe<true,false,true> : probe<false,false,true>);
  if (stream_mean && optimized && channel_scale) kernel = packed_probability ? probe<true,true,true,true> : probe<true,true,false,true>;
  kernel<<<1,128,0,c10::cuda::getCurrentCUDAStream().stream()>>>(scores.data_ptr<float>(),out.data_ptr<float>(),
      product.data_ptr<float>(),m.data_ptr<float>(),d.data_ptr<float>(),scale.data_ptr<float>(),
      reinterpret_cast<half*>(mean.data_ptr()),bytes.data_ptr<uint8_t>(),output.data_ptr<float>(),
      maximum.data_ptr<float>(),denom.data_ptr<float>(),mixed_scales);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {bytes,output,maximum,denom};
}
__global__ void helpers(const float* shifted, const int32_t* codes,
    int64_t n, uint8_t* encoded, float* decoded, bool packed_probability) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  encoded[i] = packed_probability ? vc_expcast_u8(shifted[i]) : vc_expcast_nonpositive(shifted[i]);
  float2 pair = vc_decode_pair(codes[i]);
  decoded[i * 2] = pair.x;
  decoded[i * 2 + 1] = pair.y;
}

std::vector<torch::Tensor> run_helpers(torch::Tensor shifted, torch::Tensor codes, bool packed_probability) {
  TORCH_CHECK(shifted.is_cuda() && codes.device() == shifted.device() &&
      shifted.scalar_type() == at::kFloat && codes.scalar_type() == at::kInt &&
      shifted.is_contiguous() && codes.is_contiguous() &&
      shifted.numel() > 0 && shifted.numel() == codes.numel(), "Probe inputs");
  auto encoded = torch::empty_like(shifted, shifted.options().dtype(at::kByte));
  auto decoded = torch::empty({shifted.numel(), 2}, shifted.options());
  helpers<<<(shifted.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream().stream()>>>(
      shifted.data_ptr<float>(), codes.data_ptr<int32_t>(), shifted.numel(),
      encoded.data_ptr<uint8_t>(), decoded.data_ptr<float>(), packed_probability);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {encoded, decoded};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("run",&run,pybind11::arg("scores"),pybind11::arg("out"),pybind11::arg("product"),
      pybind11::arg("m"),pybind11::arg("d"),pybind11::arg("scale"),pybind11::arg("mean"),
      pybind11::arg("optimized")=false,pybind11::arg("mixed_scales")=false,
      pybind11::arg("channel_scale")=false,pybind11::arg("packed_probability")=false,pybind11::arg("stream_mean")=false);
  m.def("run_helpers",&run_helpers,pybind11::arg("shifted"),pybind11::arg("codes"),pybind11::arg("packed_probability")=false);
}
