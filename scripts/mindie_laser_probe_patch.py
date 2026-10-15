"""Optional, reversible bulk-probability timing probe for the pinned Laser kernel.

This module never writes a source checkout. The preparation driver must build
probe variants in separate directories/processes and mark output 0 as trace
storage. Production variants must not include this patch or its header.
"""
from __future__ import annotations

import struct

try:
    from scripts.mindie_laser_expcast_patch import BULK_VEXP, EXPERIMENTAL_PROBABILITY
except ModuleNotFoundError:
    from mindie_laser_expcast_patch import BULK_VEXP, EXPERIMENTAL_PROBABILITY


HEADER_NAME = "vc_laser_probability_probe.h"
CONTRACT = "laser_core0_aiv_probability_completion_ticks_v1"
MAGIC = 0x56434C4150524F42
ROLES, SLOTS, FIELDS = 2, 4, 16
TRACE_BYTES = ROLES * SLOTS * FIELDS * 8
TICK_HZ = 50_000_000
FIELD_NAMES = (
    "magic", "version", "role", "section_flags", "round", "section",
    "key_blocks", "rows", "line", "empty_bracket_ticks", "probability_ticks",
    "begin_tick", "end_tick", "element_count", "reserved", "valid",
)
SCOPE = (
    "Core 0, both AIV subblocks; first line of the first section of each "
    "head/tail class. PIPE_V-to-PIPE_S event 7 brackets completed bulk "
    "probability conversion. Extra synchronization perturbs overlap; these "
    "samples are not an uninstrumented softmax share or end-to-end speedup. "
    "The otherwise unwritten first return tensor is diagnostic storage only."
)

HEADER = r'''// Optional diagnostic only; do not include in production variants.
#ifndef VC_LASER_PROBABILITY_PROBE_H
#define VC_LASER_PROBABILITY_PROBE_H
#include "kernel_operator.h"

#ifdef __DAV_C220_VEC__
class VcLaserProbabilityProbe {
public:
    __aicore__ inline VcLaserProbabilityProbe() {}

    __aicore__ inline void Init(__gm__ float *output, uint64_t floatElements,
                                uint32_t core, uint32_t role)
    {
        enabled_ = core == 0 && role < 2 && floatElements >= 256;
        active_ = false;
        seen_ = 0;
        role_ = role;
        output_ = reinterpret_cast<__gm__ uint64_t *>(output);
        if (enabled_) {
            for (uint32_t s = 0; s < 4; ++s) {
                for (uint32_t i = 0; i < 16; ++i) records_[s][i] = 0;
                records_[s][0] = 0x56434c4150524f42ULL;
                records_[s][1] = 1;
                records_[s][2] = role;
                records_[s][3] = s;
            }
        }
    }

    __aicore__ inline void Section(uint64_t round, uint64_t section, bool head, bool tail)
    {
        round_ = round;
        section_ = section;
        slot_ = (head ? 1U : 0U) | (tail ? 2U : 0U);
    }

    __aicore__ inline void Before(int32_t line, int32_t keyBlocks, int32_t rows)
    {
        active_ = enabled_ && line == 0 && (seen_ & (1U << slot_)) == 0;
        if (!active_) return;
        seen_ |= 1U << slot_;
        records_[slot_][4] = round_;
        records_[slot_][5] = section_;
        records_[slot_][6] = keyBlocks;
        records_[slot_][7] = rows;
        records_[slot_][8] = line;
        records_[slot_][13] = static_cast<uint64_t>(keyBlocks) * rows * 128;
        // Drain earlier Vector work before measuring the empty bracket.
        SyncVector();
        const uint64_t emptyBegin = Tick();
        SyncVector();
        records_[slot_][9] = Tick() - emptyBegin;
        // Scalar record writes are outside the measured probability interval.
        begin_ = Tick();
    }

    __aicore__ inline void After()
    {
        if (!active_) return;
        SyncVector();
        const uint64_t end = Tick();
        records_[slot_][10] = end - begin_;
        records_[slot_][11] = begin_;
        records_[slot_][12] = end;
        records_[slot_][15] = 1;
        active_ = false;
    }

    __aicore__ inline void Finish()
    {
        if (!enabled_) return;
        // Called after Run's final cross-core signal. Each AIV owns 512 bytes;
        // no writes to the score/diagonal workspace or attention output.
        for (uint32_t s = 0; s < 4; ++s) {
            AscendC::GlobalTensor<uint64_t> record;
            record.SetGlobalBuffer(output_ + (role_ * 4 + s) * 16, 16);
            for (uint32_t i = 0; i < 16; ++i) record.SetValue(i, records_[s][i]);
            for (uint32_t i = 0; i < 16; i += 8) {
                AscendC::DataCacheCleanAndInvalid<uint64_t,
                    AscendC::CacheLine::SINGLE_CACHE_LINE,
                    AscendC::DcciDst::CACHELINE_OUT>(record[i]);
            }
        }
    }

private:
    __aicore__ inline uint64_t Tick()
    {
        return static_cast<uint64_t>(AscendC::GetSystemCycle());
    }
    __aicore__ inline void SyncVector()
    {
        // Audited unused in pinned VectorForward; do not borrow IDs 0..6.
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    }
    __gm__ uint64_t *output_;
    bool enabled_, active_;
    uint32_t seen_, role_, slot_;
    uint64_t round_, section_, begin_;
    uint64_t records_[4][16];
};
#endif
#endif
'''.encode("utf-8")
PROBE_HEADER = HEADER


def _changes(source: bytes):
    nl = b"\r\n" if b"\r\n" in source else b"\n"
    def n(data):
        return data.replace(b"\n", nl)
    changes = [
        (b'#include "AddressMappingVectorForwardOnline.h"\n',
         b'#include "AddressMappingVectorForwardOnline.h"\n#include "vc_laser_probability_probe.h"\n'),
        (b'    __gm__ float* __restrict__ log_sum_max_gm;\n',
         b'    __gm__ float* __restrict__ log_sum_max_gm;\n    VcLaserProbabilityProbe vc_probability_probe_;\n'),
        (b'    address.set_core_info(core_num, cur_core_index, vector_id);\n',
         b'    address.set_core_info(core_num, cur_core_index, vector_id);\n'
         b'    vc_probability_probe_.Init(log_sum_max_gm,\n'
         b'        static_cast<uint64_t>(B) * H * qSeqLength, cur_core_index, vector_id);\n'),
        (b'            attention_score_short_double_line_one(sectionLoop, section.sectionNum, section.isTriangle,\n',
         b'            vc_probability_probe_.Section(times_sync_cube, sectionLoop,\n'
         b'                section.isHeadSection[sectionLoop], section.isTailSection[sectionLoop]);\n'
         b'            attention_score_short_double_line_one(sectionLoop, section.sectionNum, section.isTriangle,\n'),
        (b'        ffts_cross_core_sync(PIPE_MTE3, config4);\n    }\n}',
         b'        ffts_cross_core_sync(PIPE_MTE3, config4);\n    }\n    vc_probability_probe_.Finish();\n}'),
    ]
    payloads = [n(BULK_VEXP), n(EXPERIMENTAL_PROBABILITY)]
    found = [payload for payload in payloads if source.count(payload) == 1]
    if len(found) != 1:
        raise ValueError("Pinned Laser probability payload changed or is ambiguous")
    payload = found[0]
    changes = [(n(old), n(new)) for old, new in changes]
    changes.append((payload,
        n(b'    vc_probability_probe_.Before(lines, param.section_block_num, rowPerTime);\n')
        + payload + n(b'\n    vc_probability_probe_.After();')))
    return changes


def patch_vector(source: bytes) -> bytes:
    """Accept one exact baseline or ExpCast payload; preserve all original code."""
    if b"vc_probability_probe_" in source or HEADER_NAME.encode() in source:
        raise ValueError("Laser probability probe was already applied")
    if b"EVENT_ID7" in source:
        raise ValueError("VectorForward now uses EVENT_ID7; probe event must be re-audited")
    changes = _changes(source)
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError("Pinned Laser probe anchor changed; refusing a broad replacement")
        source = source.replace(old, new, 1)
    return source


def unpatch_vector(source: bytes) -> bytes:
    """Recover exact pre-probe bytes for preparation-manifest verification."""
    for old, new in reversed(_changes(source)):
        if source.count(new) != 1:
            raise ValueError("Laser probability probe inverse anchor changed")
        source = source.replace(new, old, 1)
    return source


def decode_records(data: bytes) -> dict:
    """Decode CPU contiguous output-0 bytes after stream synchronization.

    Caller should copy only the first 1024 bytes, never interpret the records
    as softmax statistics. An undersized/uninitialized output is a hard error.
    """
    if len(data) < TRACE_BYTES:
        raise ValueError(f"Laser probe requires {TRACE_BYTES} output bytes, got {len(data)}")
    records = []
    for role in range(ROLES):
        for slot in range(SLOTS):
            offset = (role * SLOTS + slot) * FIELDS * 8
            record = dict(zip(FIELD_NAMES, struct.unpack_from("<16Q", data, offset)))
            if (record["magic"] != MAGIC or record["version"] != 1
                    or record["role"] != role or record["section_flags"] != slot
                    or record["valid"] not in (0, 1)):
                raise ValueError(f"Invalid Laser probe record at role {role}, slot {slot}")
            if not record["valid"]:
                continue
            if (record["rows"] != 4 or not 1 <= record["key_blocks"] <= 32
                    or record["line"] != 0
                    or record["element_count"] != record["key_blocks"] * record["rows"] * 128
                    or record["end_tick"] < record["begin_tick"]
                    or record["probability_ticks"] != record["end_tick"] - record["begin_tick"]):
                raise ValueError(f"Inconsistent Laser probe sample at role {role}, slot {slot}")
            record["section_class"] = ("steady", "head", "tail", "head_and_tail")[slot]
            records.append(record)
    if {record["role"] for record in records} != {0, 1}:
        raise ValueError("Laser probe must contain at least one valid sample for each AIV role")
    return {"contract": CONTRACT, "scope": SCOPE, "tick_hz": TICK_HZ,
            "trace_bytes": TRACE_BYTES, "records": records}
