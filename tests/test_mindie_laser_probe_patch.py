"""CPU checks for a reversible, opt-in Laser instrumentation transformation."""
from pathlib import Path
import struct

import pytest

from scripts import mindie_laser_probe_patch as probe
from scripts import mindie_laser_expcast_patch as expcast


def source(payload=expcast.BULK_VEXP):
    return (b'#include "AddressMappingVectorForwardOnline.h"\n'
            b'    __gm__ float* __restrict__ log_sum_max_gm;\n'
            b'    address.set_core_info(core_num, cur_core_index, vector_id);\n'
            b'            attention_score_short_double_line_one(sectionLoop, section.sectionNum, section.isTriangle,\n'
            b'        ffts_cross_core_sync(PIPE_MTE3, config4);\n    }\n}\n'
            + payload + b'\n')


@pytest.mark.parametrize("payload", [expcast.BULK_VEXP, expcast.EXPERIMENTAL_PROBABILITY])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_round_trip_preserves_probability_payload_and_all_existing_code(payload, newline):
    original = source(payload).replace(b"\n", newline)
    patched = probe.patch_vector(original)
    assert probe.unpatch_vector(patched) == original
    assert payload.replace(b"\n", newline) in patched
    assert patched.count(b"vc_probability_probe_.Before(") == 1
    assert patched.count(b"vc_probability_probe_.After();") == 1
    assert patched.count(b"ffts_cross_core_sync") == original.count(b"ffts_cross_core_sync")
    assert b"PIPE_ALL" not in probe.HEADER
    assert b"floatElements >= 256" in probe.HEADER
    assert probe.TRACE_BYTES == 256 * 4


def test_duplicate_application_and_changed_event_or_anchor_are_rejected():
    with pytest.raises(ValueError, match="already applied"):
        probe.patch_vector(probe.patch_vector(source()))
    with pytest.raises(ValueError, match="EVENT_ID7"):
        probe.patch_vector(source() + b"\nset_flag(PIPE_V, PIPE_S, EVENT_ID7);\n")
    with pytest.raises(ValueError, match="anchor changed"):
        probe.patch_vector(source().replace(b"log_sum_max_gm;", b"new_stats;"))
    with pytest.raises(ValueError, match="payload changed or is ambiguous"):
        probe.patch_vector(source() + expcast.EXPERIMENTAL_PROBABILITY)


def test_actual_pinned_source_keeps_original_handshakes_and_supports_both_variants():
    path = Path(__file__).resolve().parents[1] / "build/mindie-sd-reference/csrc/ops/laser_attention/op_kernel/VectorForward.h"
    if not path.is_file():
        pytest.skip("Optional pinned checkout unavailable")
    original = path.read_bytes()
    for variant in (original, expcast.patch_vector(original)):
        patched = probe.patch_vector(variant)
        assert probe.unpatch_vector(patched) == variant
        for token in (b"wait_flag_dev(AIC2AIVFLAGID)", b"ffts_cross_core_sync", b"wait_flag(", b"set_flag("):
            assert patched.count(token) == variant.count(token)
        assert patched.count(b"vc_probability_probe_.Finish();") == 1


def trace_bytes():
    data = bytearray(probe.TRACE_BYTES)
    for role in range(2):
        for slot in range(4):
            fields = [probe.MAGIC, 1, role, slot] + [0] * 12
            if slot == 3:
                fields[4:16] = [0, 0, 2, 4, 0, 7, 50, 100, 150, 1024, 0, 1]
            struct.pack_into("<16Q", data, (role * 4 + slot) * 128, *fields)
    return data


def test_decode_raw_ticks_retains_empty_calibration_without_subtracting_it():
    result = probe.decode_records(trace_bytes())
    assert result["tick_hz"] == 50_000_000
    assert len(result["records"]) == 2
    assert all(r["section_class"] == "head_and_tail" for r in result["records"])
    assert all(r["probability_ticks"] == 50 and r["empty_bracket_ticks"] == 7 for r in result["records"])
    assert "perturbs overlap" in result["scope"]


@pytest.mark.parametrize("offset,value,message", [
    (0, 0, "Invalid Laser probe"),
    ((3 * 16 + 7) * 8, 8, "Inconsistent Laser probe"),
    ((3 * 16 + 10) * 8, 49, "Inconsistent Laser probe"),
    ((7 * 16 + 15) * 8, 0, "each AIV role"),
])
def test_decode_rejects_uninitialized_inconsistent_or_missing_samples(offset, value, message):
    data = trace_bytes()
    struct.pack_into("<Q", data, offset, value)
    with pytest.raises(ValueError, match=message):
        probe.decode_records(data)


def test_decode_rejects_small_output_instead_of_reading_past_end():
    with pytest.raises(ValueError, match="requires 1024 output bytes"):
        probe.decode_records(bytes(1023))
