"""Passive sparse PV issue/dependency windows on the V-only parent schedule.

Instrumentation is additive and removable byte-for-byte. Internal windows add
only scalar GetSystemCycle reads. The only diagnostic drain is after the entire
cube2 invocation, never between a MAD unitFlag and its corresponding FIX copy.
"""
from pathlib import Path
import re

from scripts import mindie_laser_quant_vscale_pv_prefetch_patch as parent

ROOT = Path(__file__).resolve().parents[1]
HEADER_NAME = "vc_laser_pv_probe.h"


def _block(label: str, code: bytes, indent: bytes = b"    ") -> bytes:
    return (indent + b"// VC_PV_PROBE_BEGIN " + label.encode() + b"\n" + code +
            (b"" if code.endswith(b"\n") else b"\n") +
            indent + b"// VC_PV_PROBE_END " + label.encode() + b"\n")


def _insert(data: bytes, anchor: bytes, code: bytes, label: str, *, after=False) -> bytes:
    matches = list(re.finditer(rb"^" + re.escape(anchor), data, re.M))
    if not anchor.endswith(b"\n") or len(matches) != 1:
        raise ValueError(f"PV probe requires unique complete-line anchor: {label}")
    indent = re.match(rb"[ \t]*", anchor).group()
    block = _block(label, code, indent)
    found = matches[0]
    replacement = anchor + block if after else block + anchor
    return data[:found.start()] + replacement + data[found.end():]


def _around(data, anchor, before, after, label):
    data = _insert(data, anchor, before, label + "_BEGIN")
    return _insert(data, anchor, after, label + "_END", after=True)


def _phase(data, anchor, field, label, count=b""):
    indent = re.match(rb"[ \t]*", anchor).group()
    return _around(data, anchor,
        indent + f"pv_probe_.Start({field});\n".encode(),
        indent + f"pv_probe_.Stop({field});\n".encode() + count, label)


def _group(data, anchor, field, label):
    indent = re.match(rb"[ \t]*", anchor).group()
    return _around(data, anchor, indent + b"pv_probe_.StartGroup();\n",
                   indent + f"pv_probe_.StopGroup({field});\n".encode(), label)


def strip_hooks(data: bytes) -> bytes:
    result = re.sub(
        rb"^[ \t]*// VC_PV_PROBE_BEGIN ([A-Z0-9_]+)\n.*?^[ \t]*// VC_PV_PROBE_END \1\n",
        b"", data, flags=re.M | re.S)
    if b"VC_PV_PROBE_" in result or b"pv_probe_." in result:
        raise ValueError("Unmatched PV diagnostic hook marker")
    return result


remove_hooks = strip_hooks


def patch_cube(data: bytes) -> bytes:
    original = data
    if b"pv_probe_" in data or HEADER_NAME.encode() in data:
        raise ValueError("PV probe already present")
    for anchor in (parent.PRIME, parent.LOOKAHEAD, parent.FIXED_QUANT_INIT, *parent.LAYOUT):
        if data.count(anchor) != 1:
            raise ValueError("PV diagnostics require the pinned V-only parent")
    data = _insert(data, b'#include "kernel_operator.h"\n',
                   b'#include "vc_laser_pv_probe.h"\n', "INCLUDE")
    data = _insert(data, b"    __aicore__ inline void Run();\n", b"""    __aicore__ inline void SetupCubeProbe(__gm__ uint8_t *trace, int32_t mode) {
        pv_probe_.Init(trace, mode, get_block_idx());
    }
""", "SETUP")
    data = _insert(data, b"    __aicore__ inline void ClearFlag();\n\nprivate:\n",
                   b"    vc_laser_pv_probe::PvProbe pv_probe_;\n", "MEMBER", after=True)
    for index, logical in enumerate((b"roundId", b"roundId - 2", b"roundId + Z - 2")):
        mapping = b"                address.addrMapping_cube2(gm_S, gm_V, gm_O, src, src_len, " + logical + b");\n"
        call = b"                cube2_matmul_op(src, src_len, 1, " + logical + b");\n"
        data = _insert(data, mapping, b"                pv_probe_.BeginCall(" + logical + b", Z);\n", f"CALL_{index}_BEGIN")
        data = _insert(data, call, b"                pv_probe_.EndCall();\n", f"CALL_{index}_END", after=True)

    start, end = data.index(parent.PV_BEGIN), data.index(parent.PV_END)
    chunk = data[start:end]
    chunk = _insert(chunk, b"        bool lower_left = false;\n", b"""        pv_probe_.Descriptor(idx, src_len, Kx, Ky, src[idx].onStartSection, src[idx].onEndSection);
""", "DESCRIPTOR", after=True)
    chunk = _group(chunk, b"            diag_matmul(result_gm, curr_rowmax, 2, false);\n", 23, "HISTORY")
    chunk = _group(chunk, b"            diag_matmul(src[idx].out, curr_rowsum, src[idx].ky, true);\n", 24, "OUTPUT")
    chunk = _insert(chunk,
        b"            sparseMode == 0 && !upper_right && vcore_num_per_head == 1 && Kx > 0;\n",
        b"        pv_probe_.SetGuard(pv_v_prefetch);\n", "GUARD", after=True)
    chunk = _group(chunk, b"            wait_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_b_);\n", 40, "V_PRIME_FREE")
    chunk = _group(chunk, b"""            copy_gm_to_cbuf_multi_nd2nz_b16(pv_first_v, right_start_addr,
                0, 1, l1_k, l1_n, 0, n_, l1_k, 1, 0);
""", 41, "V_PRIME_COPY")
    chunk = _insert(chunk, b"        for (int32_t i = 0; i < Kx; i++) {\n",
                    b"            pv_probe_.BeginTile(i);\n", "TILE_BEGIN", after=True)

    p_free = b"            wait_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_a_ + 2);\n"
    copy_start = chunk.index(p_free) + len(p_free)
    copy_end = chunk.index("            // 右矩阵搬运搬运一块".encode(), copy_start)
    p_copies = chunk[copy_start:copy_end]
    chunk = _phase(chunk, p_free, 26, "P_FREE")
    chunk = _phase(chunk, p_copies, 27, "P_COPIES",
                   b"            pv_probe_.Count(43, l1_skip_flag ? 1 : 2);\n")
    chunk = _phase(chunk, b"            wait_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_b_);\n", 28, "V_READY")
    chunk = _phase(chunk, b"""            set_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_a_ + 2);
            wait_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_a_ + 2);
""", 29, "P_READY")
    b_free = b"                wait_flag(PIPE_M, PIPE_MTE1, ping_pong_flag_l0_b_ + 2);\n"
    b_ready = b"""                set_flag(PIPE_MTE1, PIPE_M, ping_pong_flag_l0_b_ + 2);
                wait_flag(PIPE_MTE1, PIPE_M, ping_pong_flag_l0_b_ + 2);
"""
    b_loads = chunk[chunk.index(b_free) + len(b_free):chunk.index(b_ready)]
    chunk = _phase(chunk, b_free, 30, "L0B_FREE")
    chunk = _phase(chunk, b_loads, 31, "V_L0_LOADS", b"                pv_probe_.Count(44, 8);\n")
    chunk = _phase(chunk, b_ready, 32, "L0B_READY")
    chunk = _phase(chunk, b"                        wait_flag(PIPE_MTE1, PIPE_MTE2, pv_next_v_slot);\n", 33, "V_NEXT_FREE")
    chunk = _phase(chunk, b"""                        copy_gm_to_cbuf_multi_nd2nz_b16(pv_next_v,
                            right_start_addr + (i + 1) * l1_k * n_,
                            0, 1, l1_k, l1_n, 0, n_, l1_k, 1, 0);
""", 34, "V_NEXT_COPY", b"                        pv_probe_.Count(48, 1);\n")
    a_free = b"                    wait_flag(PIPE_M, PIPE_MTE1, ping_pong_flag_l0_a_);\n"
    a_ready_start = b"                    set_flag(PIPE_MTE1, PIPE_M, ping_pong_flag_l0_a_);\n"
    a_ready_end = b"                    wait_flag(PIPE_MTE1, PIPE_M, ping_pong_flag_l0_a_);\n"
    a_loads = chunk[chunk.index(a_free) + len(a_free):chunk.index(a_ready_start)]
    a_ready = chunk[chunk.index(a_ready_start):chunk.index(a_ready_end) + len(a_ready_end)]
    mad_start = b"                    mad(l0_c_buf,\n"
    mad_end = b"                    set_flag(PIPE_M, PIPE_MTE1, ping_pong_flag_l0_a_);\n"
    mad_pair = chunk[chunk.index(mad_start):chunk.index(mad_end)]
    chunk = _phase(chunk, a_free, 35, "L0A_FREE")
    chunk = _phase(chunk, a_loads, 36, "P_L0_LOADS", b"                    pv_probe_.Count(45, l0_skip_flag ? 0 : 8);\n")
    chunk = _phase(chunk, a_ready, 37, "L0A_READY")
    # Keep the whole MAD/optional terminal FIX plus existing dependency pair intact.
    chunk = _phase(chunk, mad_pair, 38, "MAD_FIX",
                    b"                    pv_probe_.Count(46, 1);\n                    pv_probe_.Count(47, last_k ? 1 : 0);\n")
    release = b"""            if (!pv_v_prefetch) {
                set_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_b_);
            }
            set_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_a_ + 2);
"""
    chunk = _phase(chunk, release, 39, "RELEASE")
    chunk = _insert(chunk, b"            ping_pong_flag_l1_b_ = 1 - ping_pong_flag_l1_b_;\n",
                    b"            pv_probe_.EndTile();\n", "TILE_END", after=True)
    data = data[:start] + chunk + data[end:]
    if strip_hooks(data) != original:
        raise AssertionError("PV probe modified parent bytes")
    return data


def apply(path: str, data: bytes) -> bytes:
    return patch_cube(data) if path.endswith("/op_kernel/CubeForward.h") else data
