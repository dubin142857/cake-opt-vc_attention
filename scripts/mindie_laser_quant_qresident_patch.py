"""Keep the two Q128 INT8 sub-tiles resident in L0A within one QK descriptor.

Only suppress repeated Q L1->L0A loads. Retain every existing event, MAD,
scalar dequantization, FIX store and ping-pong transition.
"""
from __future__ import annotations

from scripts.mindie_laser_quant_patch import once
from scripts.mindie_laser_quant_reuse_patch import QK_BEGIN, RUN_BEGIN, TILING


QRESIDENT_CONTRACT = {
    "baseline": "INT8 scale-cache Laser; no K lookahead, ExpCast or V smoothing",
    "scope": "Q L1-to-L0A reuse within a single ky=2 QK descriptor",
    "buffers": "Existing L0A byte offsets 0 and 32KiB; each INT8 Q128x128 occupies 16KiB",
    "load_policy": "Load Q on n_index==0, or on every K iteration when ky!=2",
    "residency": "For ky=2 each K iteration toggles L0A twice, preserving the Q-row-to-slot mapping",
    "reset": "Every new descriptor starts at n_index=0 and reloads Q; no reuse across PV or descriptors",
    "skip": "Original upper-right skip remains active; a skipped first Q row has no MAD consumer",
    "synchronization": "All existing M-to-MTE1 and MTE1-to-M events, MAD/FIX ordering and toggles remain unchanged",
    "unchanged": "Q/K/V payloads, scales/cache helper, MAD/FIX/DEQF16, K transfers, Vector/PV and cross-core schedule",
    "performance": "Removes repeated Q loads, not a prediction of device speedup",
}

OLD_DECL = b"        float q_scale_first, float q_scale_second, float k_scale);"
NEW_DECL = b"        float q_scale_first, float q_scale_second, float k_scale, bool load_q);"
OLD_CALL = b"                    q_scale_first, q_scale_second, k_scale);"
NEW_CALL = b"                    q_scale_first, q_scale_second, k_scale, n_index == 0 || ky != 2);"
OLD_DEF = b"    float q_scale_first, float q_scale_second, float k_scale)\n{"
NEW_DEF = b"    float q_scale_first, float q_scale_second, float k_scale, bool load_q)\n{"
OLD_LOAD = b"""            if (!l0_skip_flag) {
                for (int32_t i = 0; i < m0_ / SIZE_16; i++) {
                    load_cbuf_to_ca(l0_a + i * k0_ * SIZE_16,"""
NEW_LOAD = b"""            // With ky=2 the two toggles retain each Q128 tile in its L0A
            // slot until this descriptor ends. Keep all original events even
            // when reusing data; a new descriptor always reloads its first K.
            if (!l0_skip_flag && load_q) {
                for (int32_t i = 0; i < m0_ / SIZE_16; i++) {
                    load_cbuf_to_ca(l0_a + i * k0_ * SIZE_16,"""
L0A_LAYOUT = (
    b"    __ca__ TYPE *l0_a_ping_ = reinterpret_cast<__ca__ TYPE *>((uintptr_t)0);",
    b"    __ca__ TYPE *l0_a_pong_ = reinterpret_cast<__ca__ TYPE *>((uintptr_t)(32 * 1024));",
)


def patch_cube(data: bytes) -> bytes:
    for anchor in (TILING, b"q_scale_cache_line_ = ~uint64_t(0);", *L0A_LAYOUT):
        if data.count(anchor) != 1:
            raise ValueError("Q residency requires the pinned scale-cache mapping and L0A layout")
    if b"next_k_slot" in data:
        raise ValueError("Q residency must not be stacked on the K-lookahead experiment")
    begin, end = data.index(QK_BEGIN), data.index(RUN_BEGIN)
    chunk = data[begin:end]
    chunk = once(chunk, OLD_CALL, NEW_CALL)
    chunk = once(chunk, OLD_DEF, NEW_DEF)
    chunk = once(chunk, OLD_LOAD, NEW_LOAD)
    return once(data[:begin], OLD_DECL, NEW_DECL) + chunk + data[end:]


def apply(relative_path: str, data: bytes) -> bytes:
    return patch_cube(data) if relative_path.rsplit("/", 1)[-1] == "CubeForward.h" else data
