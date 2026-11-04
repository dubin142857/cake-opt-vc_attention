#include "sparse_finalize_kernel_impl.h"
extern "C" __global__ __aicore__ void vc_sparse_finalize_fp16(VC_FINALIZE_ARGS) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 SparseFinalizeVector op;op.Run(VC_FINALIZE_PASS);
}
