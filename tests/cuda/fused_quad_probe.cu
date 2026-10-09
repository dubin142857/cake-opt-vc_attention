// Small helper test on SM86+; SM89+ executes the candidate's native FP8 PTX.
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
#include "fused_quad_generated.cuh"
}
#include "../../experiments/expcast/fused_quad_check.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("check_probability",&check_probability);}
