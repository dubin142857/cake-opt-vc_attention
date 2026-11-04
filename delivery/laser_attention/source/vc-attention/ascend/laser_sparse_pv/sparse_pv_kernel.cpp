#include "sparse_pv_kernel_impl.h"
extern "C" __global__ __aicore__ void vc_sparse_pv_fp16(
    GM_ADDR value,GM_ADDR indices,GM_ADDR weights,GM_ADDR output,GM_ADDR status,
    uint32_t rows,uint32_t keys,uint32_t dimension,uint32_t rank,uint32_t cores) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SparsePvVector op;op.Run(value,indices,weights,output,status,rows,keys,dimension,rank,cores);
}
