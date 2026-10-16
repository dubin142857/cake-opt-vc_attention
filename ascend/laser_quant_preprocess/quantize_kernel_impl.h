/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE-CANN.txt. Adapted from CANNBot's direct-invoke Vector template;
 * quantization, block mapping and buffer reuse are the VC project adaptation.
 */
#pragma once
#include "kernel_operator.h"
#include "quantize_contract.h"

using namespace AscendC;
using namespace vc_laser_quant_preprocess;

template <typename Input>
class KernelBlockQuantize {
public:
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR quantized, GM_ADDR scales,
                               uint32_t sequence, uint32_t dimension,
                               uint32_t paddedSequence, uint64_t tasks,
                               uint32_t coreCount) {
        TPipe pipe;
        pipe.InitBuffer(inputBuffer_, kElements * 2);
        pipe.InitBuffer(valueBuffer_, kElements * 4);
        pipe.InitBuffer(workBuffer_, kElements * 4);
        pipe.InitBuffer(outputBuffer_, kElements);
        pipe.InitBuffer(reductionBuffer_, kReductionFloats * 4);
        pipe.InitBuffer(scaleBuffer_, kScaleGroup * 4);
        const int eventVMte2 = static_cast<int>(pipe.FetchEventID(HardEvent::V_MTE2));
        const int eventMte2V = static_cast<int>(pipe.FetchEventID(HardEvent::MTE2_V));
        const int eventVS = static_cast<int>(pipe.FetchEventID(HardEvent::V_S));
        const int eventSV = static_cast<int>(pipe.FetchEventID(HardEvent::S_V));
        const int eventVMte3 = static_cast<int>(pipe.FetchEventID(HardEvent::V_MTE3));
        const int eventMte3S = static_cast<int>(pipe.FetchEventID(HardEvent::MTE3_S));
        const int eventSMte3 = static_cast<int>(pipe.FetchEventID(HardEvent::S_MTE3));
        auto in = inputBuffer_.Get<Input>();
        auto halfValues = inputBuffer_.Get<half>();
        auto values = valueBuffer_.Get<float>();
        auto work = workBuffer_.Get<float>();
        auto integers = workBuffer_.Get<int32_t>();
        auto out = outputBuffer_.Get<int8_t>();
        auto reduction = reductionBuffer_.Get<float>();
        auto scaleValues = scaleBuffer_.Get<float>();
        GlobalTensor<Input> inputGm;
        GlobalTensor<int8_t> outputGm;
        GlobalTensor<float> scaleGm;
        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ Input*>(input));
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(quantized));
        scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
        const uint32_t blocksPerHead = paddedSequence / kRows;
        const uint64_t groups = (tasks + kScaleGroup - 1) / kScaleGroup;

        // Assign adjacent groups of eight scales to one AIV. MTE3 writers never
        // share a 32-byte scale block, including the final partial group.
        for (uint64_t group = GetBlockIdx(); group < groups; group += coreCount) {
            const uint64_t first = group * kScaleGroup;
            const uint32_t count = tasks - first < kScaleGroup ? tasks - first : kScaleGroup;
            for (uint32_t slot = 0; slot < count; ++slot) {
                const uint64_t task = first + slot;
                const uint64_t head = task / blocksPerHead;
                const uint32_t row = (task % blocksPerHead) * kRows;
                const uint32_t rows = row >= sequence ? 0 :
                    (sequence - row < kRows ? sequence - row : kRows);
                if (rows != kRows || dimension != kColumns) {
                    // BF16/FP16 both encode +0 with zero bits.
                    Duplicate(halfValues, half(0), kElements);
                    SetFlag<HardEvent::V_MTE2>(eventVMte2);
                    WaitFlag<HardEvent::V_MTE2>(eventVMte2);
                }
                if (rows) {
                    DataCopy(in, inputGm[(head * sequence + row) * dimension],
                             DataCopyParams{static_cast<uint16_t>(rows),
                                 static_cast<uint16_t>(dimension * 2 / 32), 0,
                                 static_cast<uint16_t>((kColumns - dimension) * 2 / 32)});
                    SetFlag<HardEvent::MTE2_V>(eventMte2V);
                    WaitFlag<HardEvent::MTE2_V>(eventMte2V);
                }
                // The all-padding tile has no MTE2_V handoff; retain a Vector
                // barrier for Duplicate -> Cast in that branch as well.
                PipeBarrier<PIPE_V>();
                Cast(values, in, RoundMode::CAST_NONE, kElements);
                PipeBarrier<PIPE_V>();
                Abs(work, values, kElements);
                PipeBarrier<PIPE_V>();
                // Max-only reduction has no accumulated rounding. Split the
                // first 256 repeats into two legal (<=255) aligned calls.
                WholeReduceMax(reduction, work, 64, 128, 1, 1, 8,
                               ReduceOrder::ORDER_ONLY_VALUE);
                WholeReduceMax(reduction[128], work[8192], 64, 128, 1, 1, 8,
                               ReduceOrder::ORDER_ONLY_VALUE);
                PipeBarrier<PIPE_V>();
                WholeReduceMax(reduction[256], reduction, 64, 4, 1, 1, 8,
                               ReduceOrder::ORDER_ONLY_VALUE);
                PipeBarrier<PIPE_V>();
                WholeReduceMax(reduction[264], reduction[256], 4, 1, 1, 1, 8,
                               ReduceOrder::ORDER_ONLY_VALUE);
                SetFlag<HardEvent::V_S>(eventVS);
                WaitFlag<HardEvent::V_S>(eventVS);
                const float maximum = reduction.GetValue(264);
                float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
                if (scale < 1.1754943508222875e-38f) scale = 1.1754943508222875e-38f;
                scaleValues.SetValue(slot, scale);
                SetFlag<HardEvent::S_V>(eventSV);
                WaitFlag<HardEvent::S_V>(eventSV);
                Duplicate(work, scale, kElements);
                PipeBarrier<PIPE_V>();
                // Div intentionally retains the existing FP32 x / scale
                // contract; this is not reciprocal-multiply quantization.
                Div(values, values, work, kElements);
                PipeBarrier<PIPE_V>();
                Maxs(values, values, -127.0f, kElements);
                PipeBarrier<PIPE_V>();
                Mins(values, values, 127.0f, kElements);
                PipeBarrier<PIPE_V>();
                Cast(integers, values, RoundMode::CAST_RINT, kElements);
                PipeBarrier<PIPE_V>();
                // Every subsequent cast is exact in [-127,127]. In particular,
                // never narrow unrounded FP32 to FP16 before integer rounding.
                Cast(values, integers, RoundMode::CAST_RINT, kElements);
                PipeBarrier<PIPE_V>();
                Cast(halfValues, values, RoundMode::CAST_NONE, kElements);
                PipeBarrier<PIPE_V>();
                Cast(out, halfValues, RoundMode::CAST_NONE, kElements);
                SetFlag<HardEvent::V_MTE3>(eventVMte3);
                WaitFlag<HardEvent::V_MTE3>(eventVMte3);
                DataCopy(outputGm[task * kElements], out, kElements);
                // Drain before next input copy or scalar scale-buffer updates;
                // this also protects the input buffer reused for final casts.
                SetFlag<HardEvent::MTE3_S>(eventMte3S);
                WaitFlag<HardEvent::MTE3_S>(eventMte3S);
            }
            SetFlag<HardEvent::S_MTE3>(eventSMte3);
            WaitFlag<HardEvent::S_MTE3>(eventSMte3);
            DataCopyPad(scaleGm[first], scaleValues,
                        DataCopyExtParams{1, count * uint32_t(sizeof(float)), 0, 0, 0});
            SetFlag<HardEvent::MTE3_S>(eventMte3S);
            WaitFlag<HardEvent::MTE3_S>(eventMte3S);
        }
    }
private:
    TBuf<TPosition::VECCALC> inputBuffer_, valueBuffer_, workBuffer_;
    TBuf<TPosition::VECCALC> outputBuffer_, reductionBuffer_, scaleBuffer_;
};

