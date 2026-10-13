// FP16 MMA/FP32 denominator: software FP8 decode on SM86, native on SM89+.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "../../third_party/sageattention_local/csrc/math.cuh"
#include "../../third_party/sageattention_local/csrc/qattn/attn_utils.cuh"
namespace sage_vc {
#include "../../third_party/sageattention_local/csrc/vc/vc_math.cuh"
#include "../../third_party/sageattention_local/csrc/vc/probability_variants.cuh"
#include "../../third_party/sageattention_local/csrc/vc/work_reduction.cuh"
#include "tensor_mass_helper.cuh"
}
#include "tensor_mass_check.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("check_probability",&check_probability);
  m.def("check_mass",&check_mass);
}
