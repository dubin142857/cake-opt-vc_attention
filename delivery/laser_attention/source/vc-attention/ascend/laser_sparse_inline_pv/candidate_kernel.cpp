#include "candidate_contract.h"
#ifdef __DAV_C220_CUBE__
#include "candidate_cube.h"
#else
#include "candidate_vector.h"
#endif
extern "C" __global__ __aicore__ void vc_sparse_inline_pv_fp16(VC_CANDIDATE_ARGS) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#ifdef __DAV_C220_CUBE__
    CandidateCube op;
#else
    CandidateVector op;
#endif
    op.Run(VC_CANDIDATE_PASS);
}
