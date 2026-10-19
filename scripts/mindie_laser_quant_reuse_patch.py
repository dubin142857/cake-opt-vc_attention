"""Move repeated QK scale reads, preserving the pinned quantized Laser arithmetic.

Apply only after mindie_laser_quant_patch. The two-Q-block addressing contract
is checked before transforming any bytes. Scale values remain invocation-local;
the cache invalidation helper and all FIX/cross-core synchronization stay intact.
"""
from __future__ import annotations

from scripts.mindie_laser_quant_patch import once


QK_BEGIN = b"__aicore__ __inline__ void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::cube1_matmul_op("
RUN_BEGIN = b"__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::Run()"
TILING = b"address.set_tiling(this->Y * this->F, cur_core_index, 64, 2);"
OLD_GAIN = b"""float gain = ReadQuantScale(q_scales_ + q_scale_base + m_offset / 128) *
                             ReadQuantScale(k_scales_ + k_scale_index);"""
NEW_GAIN = b"""float gain = (m_offset == 0 ? q_scale_first : q_scale_second) * k_scale;"""


def patch_cube(data: bytes) -> bytes:
    # This candidate caches two Q scales because the pinned address mapper is
    # configured with ky=2. Refuse a source update that changes that contract.
    if data.count(TILING) != 1 or data.count(b"address.set_tiling(") != 1:
        raise ValueError("Scale reuse requires the pinned 64-block, ky=2 address mapping")
    data = once(data,
        b"        int64_t q_scale_base, int64_t k_scale_index);",
        b"        float q_scale_first, float q_scale_second, float k_scale);")
    start = data.index(QK_BEGIN)
    end = data.index(RUN_BEGIN, start)
    chunk = data[start:end]
    chunk = once(chunk, b"        bool lower_left = src[idx].lowerLeft;", b"""        bool lower_left = src[idx].lowerLeft;

        // The pinned descriptor has ky=2. Read Q scales once per descriptor,
        // not once per Q128/K128 result. ReadQuantScale retains the DCCI needed
        // when a preceding quantizer reuses allocator addresses on this stream.
        const int64_t q_scale_base = (gm_a - gm_Q) / (128 * 128);
        const float q_scale_first = ReadQuantScale(q_scales_ + q_scale_base);
        const float q_scale_second = ky > 1
            ? ReadQuantScale(q_scales_ + q_scale_base + 1) : q_scale_first;""")
    chunk = once(chunk,
        b"            for (int n_index = 0; n_index < n_loop; n_index++) {",
        b"""            for (int n_index = 0; n_index < n_loop; n_index++) {
                // One K scale serves both Q128 rows in this K128 tile.
                const int64_t k_scale_index = (gm_b - gm_K) / (128 * 128) + n_index;
                const float k_scale = ReadQuantScale(k_scales_ + k_scale_index);""")
    chunk = once(chunk,
        b"                    (gm_a - gm_Q) / (128 * 128), (gm_b - gm_K) / (128 * 128) + n_index);",
        b"                    q_scale_first, q_scale_second, k_scale);")
    chunk = once(chunk,
        b"    int64_t q_scale_base, int64_t k_scale_index)",
        b"    float q_scale_first, float q_scale_second, float k_scale)")
    chunk = once(chunk, OLD_GAIN, NEW_GAIN)
    return data[:start] + chunk + data[end:]


def apply(relative_path: str, data: bytes) -> bytes:
    return patch_cube(data) if relative_path.rsplit("/", 1)[-1] == "CubeForward.h" else data
