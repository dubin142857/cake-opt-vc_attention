"""Keep Q/K scalar scale cache lines valid within one read-only invocation.

Apply to the pinned scale-reuse candidate. Only the scalar cache maintenance
changes; all scale values and the original QK/PV pipeline remain untouched.
"""
from __future__ import annotations

from scripts.mindie_laser_quant_patch import once


CACHE_CONTRACT = {
    "scope": "Separate last absolute 64-byte cache-line tags for Q and K scales per Cube instance",
    "reset": "SetQuantScales invalidates both tags on every native invocation, including reused addresses",
    "invalidation": "DCCI on first read or line transition; a return to an earlier line invalidates again",
    "addressing": "Compare absolute pointer & ~63; DCCI and GetValue use the original in-bounds float pointer",
    "lifetime": "Preceding same-stream quantizer completes before attention; Q/K scales are read-only until device kernel completion; same-stream rewrites after the queued call are allowed",
    "unchanged": "Scale reads and values, FP32 gain/truncation, MAD/FIX instructions, barriers, Vector/PV and cross-core schedule",
    "storage": "Two uint64_t tags; no scale array, persistent cache or cross-invocation reuse",
    "performance": "DCCI call reduction is a source-level hypothesis; NPU timing is required",
}

OLD_RESET = b"""        q_scales_ = reinterpret_cast<__gm__ float *>(q);
        k_scales_ = reinterpret_cast<__gm__ float *>(k);"""
NEW_RESET = OLD_RESET + b"""
        // The preceding quantizer may overwrite the same allocator addresses.
        // Never carry a scalar-cache validity decision across invocations.
        q_scale_cache_line_ = ~uint64_t(0);
        k_scale_cache_line_ = ~uint64_t(0);"""

OLD_SCALE_BLOCK = b"""    // Input scales can reuse allocator addresses across fresh API calls.
    // Invalidate the scalar cache line before a GM scalar read; MTE does not
    // keep the scalar data cache coherent with a preceding quantization op.
    __aicore__ inline float ReadQuantScale(__gm__ float *ptr)
    {
        AscendC::GlobalTensor<float> scale;
        scale.SetGlobalBuffer(ptr, 1);
        AscendC::DataCacheCleanAndInvalid<float,
            AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(scale);
        return scale.GetValue(0);
    }
    __gm__ float *q_scales_;
    __gm__ float *k_scales_;"""

NEW_SCALE_BLOCK = b"""    // Q/K scales are read-only for this invocation. Keep separate last-line
    // tags so interleaved Q and K reads can reuse each array's current line.
    // A line revisit after another line is conservatively invalidated again.
    __aicore__ inline float ReadQuantScale(__gm__ float *ptr, uint64_t &last_line)
    {
        AscendC::GlobalTensor<float> scale;
        scale.SetGlobalBuffer(ptr, 1);
        const uint64_t line = reinterpret_cast<uint64_t>(ptr) & ~uint64_t(63);
        if (line != last_line) {
            // SINGLE_CACHE_LINE selects the containing physical 64-byte line.
            // Keep ptr in bounds; the aligned tag is never dereferenced.
            AscendC::DataCacheCleanAndInvalid<float,
                AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(scale);
            last_line = line;
        }
        return scale.GetValue(0);
    }
    __gm__ float *q_scales_;
    __gm__ float *k_scales_;
    uint64_t q_scale_cache_line_;
    uint64_t k_scale_cache_line_;"""

READ_REPLACEMENTS = (
    (b"ReadQuantScale(q_scales_ + q_scale_base)",
     b"ReadQuantScale(q_scales_ + q_scale_base, q_scale_cache_line_)"),
    (b"ReadQuantScale(q_scales_ + q_scale_base + 1)",
     b"ReadQuantScale(q_scales_ + q_scale_base + 1, q_scale_cache_line_)"),
    (b"ReadQuantScale(k_scales_ + k_scale_index)",
     b"ReadQuantScale(k_scales_ + k_scale_index, k_scale_cache_line_)"),
)
OLD_COMMENT = b"""        // The pinned descriptor has ky=2. Read Q scales once per descriptor,
        // not once per Q128/K128 result. ReadQuantScale retains the DCCI needed
        // when a preceding quantizer reuses allocator addresses on this stream."""
NEW_COMMENT = b"""        // The pinned descriptor has ky=2. Read Q scales once per descriptor,
        // not once per Q128/K128 result. ReadQuantScale invalidates the current
        // Q line once after setup or a line transition, then reuses that line."""


def patch_cube(data: bytes) -> bytes:
    if data.count(b"ReadQuantScale(") != 4:
        raise ValueError("Cache-line reuse requires exactly the pinned three Q/K scale read sites")
    data = once(data, OLD_RESET, NEW_RESET)
    data = once(data, OLD_SCALE_BLOCK, NEW_SCALE_BLOCK)
    data = once(data, OLD_COMMENT, NEW_COMMENT)
    for old, new in READ_REPLACEMENTS:
        data = once(data, old, new)
    return data


def apply(relative_path: str, data: bytes) -> bytes:
    return patch_cube(data) if relative_path.rsplit("/", 1)[-1] == "CubeForward.h" else data
