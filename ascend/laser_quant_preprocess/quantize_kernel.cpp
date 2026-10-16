/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE-CANN.txt. Adapted from CANNBot's direct-invoke Vector template;
 * quantization, block mapping and buffer reuse are the VC project adaptation.
 */
#include "quantize_kernel_impl.h"

// One global entry per translation unit: the CANN 9.0.1 legacy generator
// numbers entry aliases per function, but KTYPE metadata per source file.
// Keeping two explicit-AIV entries together produces a mismatched second key.
// The standard entry plus task marker also avoids its __vector__ parser bug.
extern "C" __global__ __aicore__ void vc_laser_quantize_fp16(
    GM_ADDR input, GM_ADDR output, GM_ADDR scales, uint32_t sequence,
    uint32_t dimension, uint32_t paddedSequence, uint64_t tasks, uint32_t cores) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelBlockQuantize<half> op;
    op.Run(input, output, scales, sequence, dimension, paddedSequence, tasks, cores);
}
