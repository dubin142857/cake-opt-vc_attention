#pragma once
#include <cstdint>

namespace vc_laser_quant_preprocess {
constexpr uint32_t kContractVersion = 1;
constexpr uint32_t kRows = 128;
constexpr uint32_t kColumns = 128;
constexpr uint32_t kElements = kRows * kColumns;
constexpr uint32_t kScaleGroup = 8;
constexpr uint32_t kReductionFloats = 272;
constexpr uint32_t kUbBytes = kElements * (2 + 4 + 4 + 1) +
                              kReductionFloats * 4 + kScaleGroup * 4;
static_assert(kUbBytes < 192 * 1024, "A2/A3 UB budget");
}
