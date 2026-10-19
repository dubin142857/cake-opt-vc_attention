"""Sparse diagnostic hooks over locked scale-reuse Laser bytes.

No arithmetic, buffer addressing, original flags, or cross-core signals change.
The companion header owns new sampled V_S0 completion pairs. Probe mode 0 is
not a claim of identical compiled resources: calibrate against production.
"""
from pathlib import Path

from scripts.mindie_laser_quant_patch import once

ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "ascend/laser_quant_probe/vc_laser_phase_probe.h").read_bytes()


def _function(data, start, end, transform):
    if data.count(start) != 1 or data.count(end) != 1:
        raise ValueError("Phase probe requires unique pinned function boundaries")
    first, last = data.index(start), data.index(end)
    if last <= first:
        raise ValueError("Invalid pinned phase function order")
    return data[:first] + transform(data[first:last]) + data[last:]


def patch_cube(data: bytes) -> bytes:
    data = once(data, b'#include "kernel_operator.h"',
                b'#include "kernel_operator.h"\n#include "vc_laser_phase_probe.h"')
    data = once(data, b"    __aicore__ inline void Run();", b"""    __aicore__ inline void SetupProbe(__gm__ uint8_t *trace, int32_t mode) {
        phase_probe_.Init(trace, mode, get_block_idx(), 0);
    }
    __aicore__ inline void Run();""")
    data = once(data, b"private:\n    // Input scales can reuse allocator addresses across fresh API calls.",
                b"private:\n    vc_laser_probe::PhaseProbe phase_probe_;\n    // Input scales can reuse allocator addresses across fresh API calls.")
    needle = b"                wait_flag_dev(AIV2AICFLAGID);"
    if data.count(needle) != 3:
        raise ValueError("Expected three original Cube handoff wait sites")
    # logical round is the probability tile being awaited, not the lookahead
    # QK round currently being scheduled by the two-round Cube pipeline.
    for logical in (b"roundId", b"roundId - 2", b"roundId + Z - 2"):
        replacement = (b"                phase_probe_.WaitBegin(" + logical + b", Z, address.is_running(" + logical + b"));\n"
                       + needle + b"\n                phase_probe_.WaitEnd();")
        # Skip previously replaced sites by replacing the remaining suffix.
        pos = data.find(needle, 0 if logical == b"roundId" else previous_end)
        data = data[:pos] + replacement + data[pos + len(needle):]
        previous_end = pos + len(replacement)
    return data


def patch_vector(data: bytes) -> bytes:
    data = once(data, b'#include "vc_laser_probability.h"',
                b'#include "vc_laser_probability.h"\n#include "vc_laser_phase_probe.h"')
    data = once(data, b"    __aicore__ inline void Run();", b"""    __aicore__ inline void SetupProbe(__gm__ uint8_t *trace, int32_t mode) {
        phase_probe_.Init(trace, mode, get_block_idx(), 1 + get_subblockid());
    }
    __aicore__ inline void Run();""")
    data = once(data, b"private:\n    __gm__ INPUT_T* __restrict__ gm_a_cube1;",
                b"private:\n    vc_laser_probe::PhaseProbe phase_probe_;\n    __gm__ INPUT_T* __restrict__ gm_a_cube1;")
    data = once(data, b"        wait_flag_dev(AIC2AIVFLAGID);", b"""        phase_probe_.SetRound(times_sync_cube, address.get_total_round());
        phase_probe_.WaitBegin(times_sync_cube, address.get_total_round(), address.is_running(times_sync_cube));
        wait_flag_dev(AIC2AIVFLAGID);
        phase_probe_.WaitEnd();""")

    max_start = b"    IF_BF16, WORKSPACE_T>::process_line_phase_one_for_short_seq_max(bool is_head_section,"
    exp_start = b"    IF_BF16, WORKSPACE_T>::process_line_phase_one_for_short_seq_exp_and_rowsum(int32_t qk_triangle,"
    def max_hooks(chunk):
        chunk = once(chunk, b"    wait_flag(PIPE_MTE2, PIPE_V, event_id);",
                     b"    wait_flag(PIPE_MTE2, PIPE_V, event_id);\n    phase_probe_.Mark(16, 2); // score-load completed")
        chunk = once(chunk, b"    vconv_f322f16(cur_buf_for_vbrcb_rowmax_fp16, cur_buf_for_vbrcb_rowmax_fp32, 1, 1, 1, 4, 8);\n    pipe_barrier(PIPE_V);",
                     b"    vconv_f322f16(cur_buf_for_vbrcb_rowmax_fp16, cur_buf_for_vbrcb_rowmax_fp32, 1, 1, 1, 4, 8);\n    pipe_barrier(PIPE_V);\n    phase_probe_.Mark(17, 3); // scale/mask/max/online-history/broadcast completed")
        return chunk
    data = _function(data, max_start, exp_start, max_hooks)

    exp_end = b" >::add_max_wrap(__ubuf__ INPUT_T * dest_addr"
    def exp_hooks(chunk):
        chunk = once(chunk, b"    //  128/64=2", b"    phase_probe_.Mark(18, 4); // subtract completed\n    //  128/64=2")
        chunk = once(chunk, b"    set_flag(PIPE_V, PIPE_MTE3, event_id);",
                     b"    phase_probe_.Mark(19, 5); // bulk probability transform completed\n    set_flag(PIPE_V, PIPE_MTE3, event_id);")
        chunk = once(chunk, b"    process_calc_sum(qk_triangle,",
                     b"    phase_probe_.StartSum();\n    process_calc_sum(qk_triangle,")
        chunk = once(chunk, b"                        sparse_flag,\n                        lines);",
                     b"                        sparse_flag,\n                        lines);\n    phase_probe_.Mark(20, 6); // sum completed\n    phase_probe_.PhaseFinish();")
        return chunk
    data = _function(data, exp_start, exp_end, exp_hooks)
    data = once(data, b"        process_line_phase_one_for_short_seq_max(is_head_section, is_tail_section, qk_triangle,",
                b"""        phase_probe_.PhaseBegin(sectionLoop, is_head_section, is_tail_section,
                                section_block_nums, rowPerTime, lines);
        process_line_phase_one_for_short_seq_max(is_head_section, is_tail_section, qk_triangle,""")
    return data


def apply(path: str, data: bytes) -> bytes:
    if path.endswith("op_kernel/CubeForward.h"):
        return patch_cube(data)
    if path.endswith("op_kernel/VectorForward.h"):
        return patch_vector(data)
    return data
