"""Add sparse Cube call-boundary diagnostics to pinned FP16 or INT8 Laser.

All original bytes, arithmetic, flags and cross-core calls are retained. The
companion header's optional PIPE_ALL drains are only outside whole cube1/cube2
calls, where matching MAD/FIX unitFlag operations have already been issued.
"""
from pathlib import Path

from scripts.mindie_laser_quant_patch import once

ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "ascend/laser_cube_probe/vc_laser_cube_probe.h").read_bytes()
HEADER_NAME = "vc_laser_cube_probe.h"


def patch_cube(data: bytes) -> bytes:
    if HEADER_NAME.encode() in data or b"SetupCubeProbe" in data:
        raise ValueError("Cube probe already present")
    data = once(data, b'#include "kernel_operator.h"',
                b'#include "kernel_operator.h"\n#include "vc_laser_cube_probe.h"')
    data = once(data, b"    __aicore__ inline void Run();", b"""    __aicore__ inline void SetupCubeProbe(__gm__ uint8_t *trace, int32_t mode) {
        cube_probe_.Init(trace, mode, get_block_idx());
    }
    __aicore__ inline void Run();""")
    data = once(data, b"    __aicore__ inline void ClearFlag();\n\nprivate:",
                b"    __aicore__ inline void ClearFlag();\n\nprivate:\n    vc_laser_cube_probe::CubeProbe cube_probe_;")
    qk_map = b"                address.addrMapping_cube1(gm_Q, gm_K, gm_S,  src, src_len, roundId);"
    qk_call = b"                cube1_matmul_op(src, src_len);"
    if data.count(qk_map) != 3 or data.count(qk_call) != 3:
        raise ValueError("Cube probe requires three pinned complete QK call sites")
    data = data.replace(qk_map, b"                cube_probe_.Begin(0, roundId, Z);\n" + qk_map)
    data = data.replace(qk_call, qk_call + b"\n                cube_probe_.EndQK(src, src_len);")
    for logical in (b"roundId", b"roundId - 2", b"roundId + Z - 2"):
        mapping = b"                address.addrMapping_cube2(gm_S, gm_V, gm_O, src, src_len, " + logical + b");"
        call = b"                cube2_matmul_op(src, src_len, 1, " + logical + b");"
        data = once(data, mapping, b"                cube_probe_.Begin(1, " + logical + b", Z);\n" + mapping)
        data = once(data, call, call + b"\n                cube_probe_.EndPV(src, src_len);")
    return data


def apply(path: str, data: bytes) -> bytes:
    if path.endswith("op_kernel/CubeForward.h"):
        return patch_cube(data)
    return data
