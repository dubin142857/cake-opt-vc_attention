"""Prefetch one QK K tile into the existing alternate L1 slot.

Apply only to the pinned scale-cache candidate. Keep the QK base matmul and
FIX output byte-identical; change only when the existing K DMA is issued.
"""
from __future__ import annotations

from scripts.mindie_laser_quant_patch import once
from scripts.mindie_laser_quant_reuse_patch import QK_BEGIN, RUN_BEGIN, TILING


KPIPE_CONTRACT = {
    "baseline": "INT8 scale-cache Laser; no ExpCast or V smoothing",
    "scope": "One-descriptor QK K128xD128 GM-to-L1 lookahead; no prefetch across descriptor or QK/PV boundaries",
    "buffers": "Reuse K L1 ping/pong at byte addresses 128KiB and 192KiB; active INT8 K tile is 16KiB",
    "prologue": "If kx>0, acquire current slot, load K0, publish MTE2-to-MTE1 readiness",
    "steady": "Wait current ready; if next exists acquire alternate free slot and issue next K load before current base matmul",
    "ownership": "Existing MTE1-to-MTE2 free events 0/1 and MTE2-to-MTE1 ready events 0/1; no new IDs",
    "epilogue": "Every loaded tile is consumed and released; no speculative tail load; same final slot parity",
    "unchanged": "Q/K/V payloads, scale values/cache helper, base matmul, MAD/FIX/DEQF16, Vector/PV and cross-core schedule",
    "performance": "Test of overlapping K movement with QK computation/output; no predicted device speedup",
}

N_LOOP = b"            for (int n_index = 0; n_index < n_loop; n_index++) {"
SLOT = b"ping_pong_flag_l1_b_"
READY_WAIT = b"                wait_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_b_);"
ORIGINAL_K_LOAD = b"""                wait_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_b_);

                // Load B to L1
                copy_gm_to_cbuf_multi_nd2nz_b8(l1_b,
                    gm_b + n_index * l1_n_size_ * k_,
                    0,           // sid
                    1,           // ndNum
                    l1_k_size_,  // nValue   """ + "实际拷贝的行数".encode() + b"""
                    l1_n_size_,  // dValue   """ + "实际拷贝的列数".encode() + b"""
                    0,           // srcNdMatrixStride, unused
                    l1_n_size_,  // srcDValue """ + "大矩阵的列数".encode() + b"""
                    l1_k_size_,  // dstNzC0Stride """ + "目标行数".encode() + b"""
                    1,           // dstNzNStride  """ + "目标行之间间隔（1为连续）".encode() + b"""
                    0            // dstNzMatrixStride, unused
                );

                set_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_b_);
                wait_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_b_);"""
ISSUE_ONLY = ORIGINAL_K_LOAD.removesuffix(b"\n" + READY_WAIT)
PROLOGUE = b"""            // One K tile ahead, confined to this descriptor. Reuse the original
            // free/ready events; an empty descriptor must not acquire a slot.
            if (n_loop > 0) {
                auto l1_b = reinterpret_cast<__cbuf__ int8_t *>(ping_pong_flag_l1_b_ ? l1_b_pong_ : l1_b_ping_);
""" + ISSUE_ONLY.replace(b"gm_b + n_index * l1_n_size_ * k_", b"gm_b") + b"""
            }

"""
NEXT_PREFETCH = b"""                // Current K is ready. Queue the next K in the other L1 slot
                // before current L1->L0/MAD/FIX work; never read past the tail.
                if (n_index + 1 < n_loop) {
                    const uint32_t next_k_slot = 1 - ping_pong_flag_l1_b_;
                    auto l1_b_next = reinterpret_cast<__cbuf__ int8_t *>(next_k_slot ? l1_b_pong_ : l1_b_ping_);
""" + b"\n".join(b"    " + line if line else b"" for line in ISSUE_ONLY.replace(
    b"copy_gm_to_cbuf_multi_nd2nz_b8(l1_b,", b"copy_gm_to_cbuf_multi_nd2nz_b8(l1_b_next,"
).replace(b"gm_b + n_index * l1_n_size_ * k_", b"gm_b + (n_index + 1) * l1_n_size_ * k_"
).replace(SLOT, b"next_k_slot").splitlines()) + b"""
                }"""
NEW_K_LOAD = READY_WAIT + b"\n\n" + NEXT_PREFETCH


def patch_cube(data: bytes) -> bytes:
    if data.count(TILING) != 1 or data.count(b"q_scale_cache_line_ = ~uint64_t(0);") != 1:
        raise ValueError("K lookahead requires the pinned scale-cache QK mapping")
    begin = data.index(QK_BEGIN)
    end = data.index(RUN_BEGIN, begin)
    chunk = data[begin:end]
    chunk = once(chunk, ORIGINAL_K_LOAD, NEW_K_LOAD)
    chunk = once(chunk, N_LOOP, PROLOGUE + N_LOOP)
    return data[:begin] + chunk + data[end:]


def apply(relative_path: str, data: bytes) -> bytes:
    return patch_cube(data) if relative_path.rsplit("/", 1)[-1] == "CubeForward.h" else data
