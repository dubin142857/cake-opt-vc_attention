"""One-tile V GM->L1 lookahead on the pinned Vector-scale Laser PV path.

Only the existing B-side transfer/reuse events move. P transfers, MAD/FIX,
history/normalization and the Vector implementation retain their ordering.
"""
from __future__ import annotations

from scripts.mindie_laser_quant_patch import once
from scripts.mindie_laser_quant_vscale_patch import FIXED_QUANT_INIT

CONTRACT = {
    "baseline": "Pinned vector_scale; no bulk/rowmax/K-lookahead candidate stacking",
    "scope": "PV only; one next V128x128 GM-to-L1 transfer ahead of current Ky MADs",
    "guard": "D128, FP16-sized TYPE, M/K/N128, dense noncausal, Ky2, no upper-right skip, one V core and positive Kx",
    "prime": "After complete history diag call, acquire current L1B slot, copy V0, publish ready",
    "lookahead": "After all current V L1-to-L0B loads, release its L1B slot; acquire and fill opposite slot iff i+1<Kx",
    "buffers": "Existing L1B 128KiB/192KiB slots; each V tile 32KiB, disjoint from P and diag scratch",
    "events": "Existing MTE1-to-MTE2 free and MTE2-to-MTE1 ready B-slot tokens; no new event IDs",
    "lifetime": "Ready consumed once before read; release after all L1 reads; final tile creates no pending prefetch",
    "handoff": "Every descriptor ends with both B free tokens, no outstanding ready token; original ping-pong parity",
    "unchanged": "Q/K/V values, scales, P copies, all MAD/FIX/diag order, Vector, QK and cross-core schedule",
    "fallback": "Original PV transfer schedule outside the guard",
    "validation": "Requires all-output exact same-payload checks and target-device compilation/timing",
}

PV_BEGIN = b"__aicore__ __inline__ void  CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::cube2_matmul_op("
PV_END = b"__aicore__ __inline__ void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::matmul_fp32("
LOOP = b"        // \xe5\xa4\x96\xe5\xbe\xaa\xe7\x8e\xaf\xe6\x8c\x89x\xe6\x96\xb9\xe5\x90\x91\xe9\x81\x8d\xe5\x8e\x86\n"
FREE_WAIT = b"            wait_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_b_);"
READY_SET = b"            set_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_b_);"
READY_WAIT = b"            wait_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_b_);"
FREE_SET = b"            set_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_b_);"
L0_READY_WAIT = b"                wait_flag(PIPE_MTE1, PIPE_M, ping_pong_flag_l0_b_ + 2);"
COPY = b"""            copy_gm_to_cbuf_multi_nd2nz_b16(l1_b_buf,
                right_start_addr + i * l1_k * n_,
                0,     // sid
                1,     // ndNum
                l1_k,  // nValue   \xe5\xae\x9e\xe9\x99\x85\xe6\x8b\xb7\xe8\xb4\x9d\xe7\x9a\x84\xe8\xa1\x8c\xe6\x95\xb0
                l1_n,  // dValue   \xe5\xae\x9e\xe9\x99\x85\xe6\x8b\xb7\xe8\xb4\x9d\xe7\x9a\x84\xe5\x88\x97\xe6\x95\xb0
                0,     // srcNdMatrixStride, unused
                n_,    // srcDValue \xe5\xa4\xa7\xe7\x9f\xa9\xe9\x98\xb5\xe7\x9a\x84\xe5\x88\x97\xe6\x95\xb0
                l1_k,  // dstNzC0Stride \xe7\x9b\xae\xe6\xa0\x87\xe8\xa1\x8c\xe6\x95\xb0
                1,     // dstNzNStride  \xe7\x9b\xae\xe6\xa0\x87\xe8\xa1\x8c\xe4\xb9\x8b\xe9\x97\xb4\xe9\x97\xb4\xe9\x9a\x94\xef\xbc\x881\xe4\xb8\xba\xe8\xbf\x9e\xe7\xbb\xad\xef\xbc\x89
                0      // dstNzMatrixStride, unused
            );"""
PRIME = b"""        // PV V lookahead: both P slots and diagonal scratch are disjoint.
        // The preceding history helper must finish before touching PV L0.
        const bool pv_v_prefetch = D == 128 && sizeof(TYPE) == 2 && Ky == 2 &&
            l1_m == 128 && l1_k == 128 && l1_n == 128 && qk_triangle == 0 &&
            sparseMode == 0 && !upper_right && vcore_num_per_head == 1 && Kx > 0;
        if (pv_v_prefetch) {
            auto pv_first_v = ping_pong_flag_l1_b_ ? l1_b_pong_ : l1_b_ping_;
            wait_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_b_);
            copy_gm_to_cbuf_multi_nd2nz_b16(pv_first_v, right_start_addr,
                0, 1, l1_k, l1_n, 0, n_, l1_k, 1, 0);
            set_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_b_);
        }

"""
LOOKAHEAD = b"""
                if (pv_v_prefetch) {
                    // MTE1 has consumed all current V fragments. Free this L1
                    // slot now; the MADs below read their independent L0B copy.
                    set_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_b_);
                    if (i + 1 < Kx) {
                        const uint32_t pv_next_v_slot = 1 - ping_pong_flag_l1_b_;
                        auto pv_next_v = pv_next_v_slot ? l1_b_pong_ : l1_b_ping_;
                        wait_flag(PIPE_MTE1, PIPE_MTE2, pv_next_v_slot);
                        copy_gm_to_cbuf_multi_nd2nz_b16(pv_next_v,
                            right_start_addr + (i + 1) * l1_k * n_,
                            0, 1, l1_k, l1_n, 0, n_, l1_k, 1, 0);
                        set_flag(PIPE_MTE2, PIPE_MTE1, pv_next_v_slot);
                    }
                }
"""
LAYOUT = (
    b"((uintptr_t)0);                    // 64 KB",
    b"((uintptr_t)(64 * 1024));         // 64 KB",
    b"((uintptr_t)128 * 1024);           // 64 KB",
    b"((uintptr_t)(192 * 1024));         // 64 KB",
    b"reinterpret_cast<__cbuf__ float *>(256 * 1024)",
    b"reinterpret_cast<__cbuf__ float *>(320 * 1024)",
    b"reinterpret_cast<__cbuf__ float *>(384 * 1024)",
    b"reinterpret_cast<__cbuf__ float *>(448 * 1024)",
)


def guarded(data: bytes) -> bytes:
    return b"            if (!pv_v_prefetch) {\n" + b"\n".join(b"    " + x for x in data.split(b"\n")) + b"\n            }"


def patch_cube(data: bytes) -> bytes:
    for anchor in (FIXED_QUANT_INIT, *LAYOUT, b"n_index == 0 || ky != 2"):
        if data.count(anchor) != 1:
            raise ValueError("PV prefetch requires pinned vector_scale and disjoint L1 mapping")
    if b"pv_v_prefetch" in data:
        raise ValueError("PV prefetch is already present")
    if data.count(PV_BEGIN) != 1 or data.count(PV_END) != 1:
        raise ValueError("Pinned PV method boundary changed")
    start, end = data.index(PV_BEGIN), data.index(PV_END)
    chunk = data[start:end]
    for old in (LOOP, FREE_WAIT, READY_SET, READY_WAIT, FREE_SET, L0_READY_WAIT, COPY):
        if chunk.count(old) != 1:
            raise ValueError("Pinned PV transfer schedule changed")
    # Guard the original sites before inserting the lookahead's extra sites.
    for old in (FREE_WAIT, COPY + b"\n" + READY_SET, FREE_SET):
        chunk = once(chunk, old, guarded(old))
    chunk = once(chunk, LOOP, PRIME + LOOP)
    chunk = once(chunk, L0_READY_WAIT, L0_READY_WAIT + b"\n" + LOOKAHEAD)
    return data[:start] + chunk + data[end:]


def apply(relative_path: str, data: bytes) -> bytes:
    return patch_cube(data) if relative_path.rsplit("/", 1)[-1] == "CubeForward.h" else data
