"""Add same-descriptor P lookahead to the validated PV V-lookahead schedule.

The Vector producer publishes the completed round before any Cube PV call.
Current L1A release remains at its original position after all Ky consumers.
"""
from __future__ import annotations

import hashlib

from scripts.mindie_laser_quant_patch import once
from scripts import mindie_laser_quant_vscale_pv_prefetch_patch as parent

CONTRACT = {
    "baseline": "Pinned vector_scale PV V-prefetch candidate; keep its V schedule",
    "scope": "One same-descriptor P tile ahead: two original P128x128 GM-to-L1 copies",
    "guard": "pv_v_prefetch && Kx>1; all parent dense noncausal D128/Ky2/MKN128 guards retained",
    "producer": "Vector completes all round sections then publishes AIV2AICFLAGID on PIPE_MTE3; all three Cube PV sites wait for that publication",
    "boundary": "Only i+1<Kx in the current descriptor; no future round/descriptor or post-normalization speculation",
    "prime": "After history helper and V0 prime, acquire current L1A and copy both P0 rows; publish ready",
    "lookahead": "After existing next-V issue, acquire opposite L1A and copy next two P rows before current Ky MADs",
    "lifetime": "Current L1A remains owned/readable; opposite slot requires previous owner's original release; original current release stays after both Ky reads",
    "buffers": "Existing two 64KiB L1A slots at 0/64KiB; P tile exactly64KiB; L1B and diagonal scratch disjoint",
    "copy_parameters": "Original vcore_num_per_head expressions retained; deployed native path passes1, therefore each copy ND1/n128",
    "events": "Original A free/ready IDs slot+2; no new event IDs or buffers",
    "unchanged": "P/V values, L1-to-L0 loads, MAD/FIX/diag order, V-prefetch, QK/Vector/cross-core schedule and rounding",
    "fallback": "Parent V-prefetch with original P schedule when independent P guard is false",
    "validation": "Exact all-output own-payload gate and target CCE/NPU measurement required",
}

PV_BEGIN, PV_END = parent.PV_BEGIN, parent.PV_END
P_ACQUIRE = b"            wait_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_a_ + 2);"
P_READY_SET = b"            set_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_a_ + 2);"
P_READY_WAIT = b"            wait_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_a_ + 2);"
P_RELEASE = b"            set_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_a_ + 2);"
P_LOAD_END = "            // 右矩阵搬运搬运一块".encode()
P_LOAD_SHA256 = "1e3239605abd9160623fd28bea10bd39b8347e8af9471196a08c459335342c4a"


def copies(destination: str, tile: str, indent: int) -> bytes:
    """Keep every original ND-to-NZ argument; change only slot/tile index."""
    code = f"""copy_gm_to_cbuf_multi_nd2nz_b16({destination},
    (__gm__ TYPE *)(left_start_addr + {tile} * l1_m * l1_k),
    0, vcore_num_per_head, l1_m / vcore_num_per_head, l1_k,
    128 / vcore_num_per_head * 128 * 2, l1_k, l1_m, 1,
    128 * BLOCK_SIZE / vcore_num_per_head);
copy_gm_to_cbuf_multi_nd2nz_b16({destination} + SIZE_128 * SIZE_128,
    (__gm__ TYPE *)(left_start_addr + {tile} * l1_k * l1_m + line_stride),
    0, vcore_num_per_head, l1_m / vcore_num_per_head, l1_k,
    128 / vcore_num_per_head * 128 * 2, l1_k, l1_m, 1,
    128 * BLOCK_SIZE / vcore_num_per_head);"""
    return b"\n".join(b" " * indent + line for line in code.encode().split(b"\n"))


PRIME = b"""        // Run() already waited for Vector's completed-round MTE3 publication.
        // Never prefetch P from another descriptor or a future round.
        const bool pv_p_prefetch = pv_v_prefetch && Kx > 1;
        if (pv_p_prefetch) {
            auto pv_first_p = ping_pong_flag_l1_a_ ? l1_a_pong_ : l1_a_ping_;
            wait_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_a_ + 2);
""" + copies("pv_first_p", "0", 12) + b"""
            set_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_a_ + 2);
        }

"""

LOOKAHEAD = b"""                if (pv_p_prefetch && i + 1 < Kx) {
                    // The opposite slot was released by tile i-1 (or preset).
                    // Current P remains in its own L1A slot until both Ky reads.
                    const uint32_t pv_next_p_slot = 1 - ping_pong_flag_l1_a_;
                    auto pv_next_p = pv_next_p_slot ? l1_a_pong_ : l1_a_ping_;
                    wait_flag(PIPE_MTE1, PIPE_MTE2, pv_next_p_slot + 2);
""" + copies("pv_next_p", "(i + 1)", 20) + b"""
                    set_flag(PIPE_MTE2, PIPE_MTE1, pv_next_p_slot + 2);
                }

"""


def p_load_block(data: bytes) -> bytes:
    start = data.index(P_ACQUIRE)
    end = data.index(P_LOAD_END, start)
    block = data[start:end]
    if hashlib.sha256(block).hexdigest() != P_LOAD_SHA256:
        raise ValueError("P lookahead requires the pinned original two-copy P load block")
    return block


def guarded(data: bytes) -> bytes:
    return b"            if (!pv_p_prefetch) {\n" + b"\n".join(
        b"    " + line if line else line for line in data.rstrip(b"\n").split(b"\n")) + b"\n            }\n\n"


def patch_cube(data: bytes) -> bytes:
    for anchor in (parent.PRIME, parent.LOOKAHEAD, parent.FIXED_QUANT_INIT, *parent.LAYOUT):
        if data.count(anchor) != 1:
            raise ValueError("P lookahead requires pinned V-prefetch and disjoint L1 layout")
    if b"pv_p_prefetch" in data:
        raise ValueError("P lookahead is already present")
    if data.count(PV_BEGIN) != 1 or data.count(PV_END) != 1:
        raise ValueError("Pinned PV method boundary changed")
    start, end = data.index(PV_BEGIN), data.index(PV_END)
    chunk = data[start:end]
    block = p_load_block(chunk)
    for anchor in (P_READY_SET, P_READY_WAIT, P_RELEASE):
        if chunk.count(anchor) != 1:
            raise ValueError("Pinned P event lifecycle changed")
    chunk = once(chunk, block, guarded(block))
    chunk = once(chunk, P_READY_SET, parent.guarded(P_READY_SET).replace(b"pv_v_prefetch", b"pv_p_prefetch"))
    chunk = once(chunk, parent.LOOP, PRIME + parent.LOOP)
    chunk = once(chunk, parent.LOOKAHEAD, parent.LOOKAHEAD + b"\n" + LOOKAHEAD)
    return data[:start] + chunk + data[end:]


def apply(relative_path: str, data: bytes) -> bytes:
    return patch_cube(data) if relative_path.rsplit("/", 1)[-1] == "CubeForward.h" else data
