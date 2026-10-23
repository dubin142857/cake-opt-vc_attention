"""Execute the real PV trace writer, then decode its serialized records.

The functional host stubs validate the C++/Python schema and hook state machine.
They do not emulate CCE, engine execution, timing, or NPU synchronization.
"""
import ctypes
import json
from pathlib import Path

import pytest

from scripts import mindie_laser_pv_records as records
from tests.test_mindie_laser_expcast_cpp import cpp_compiler
from tests.test_mindie_laser_quant_vscale_patch import compile_library

ROOT = Path(__file__).resolve().parents[1]

STUB = r"""
typedef unsigned char uint8_t; typedef unsigned int uint32_t;
typedef signed int int32_t; typedef signed long long int64_t;
typedef unsigned long long uint64_t;
#define __aicore__
#define __gm__
#define __DAV_C220_CUBE__ 1
#ifdef _MSC_VER
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif
enum { PIPE_ALL };
static uint64_t trace_data[4*3*3*64];
static uint64_t counter=101;
static int faults=0, barriers=0, flushes=0;
namespace AscendC {
uint64_t GetSystemCycle() { counter+=7;return counter; }
template<int Pipe> void PipeBarrier() {
    // Any diagnostic barrier is a regression, even at a whole-call boundary.
    ++faults;
    ++barriers;counter+=13;
}
template<class T> class GlobalTensor {
public:
    T *base; uint64_t size;
    void SetGlobalBuffer(T *p,uint64_t n) { base=p;size=n; }
    void SetValue(uint32_t i,T value) {
        if(i>=size || base+i<trace_data || base+i>=trace_data+4*3*3*64) {++faults;return;}
        base[i]=value;
    }
    GlobalTensor<T> operator[](uint32_t i) const {
        GlobalTensor<T> result;result.SetGlobalBuffer(base+i,size-i);return result;
    }
};
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
template<class T,CacheLine L,DcciDst D> void DataCacheCleanAndInvalid(const GlobalTensor<T>&) {
    ++flushes;
}
}
"""

DRIVER = r"""
void phase(vc_laser_pv_probe::PvProbe &probe,uint32_t field) {
    probe.Start(field);counter+=11;probe.Stop(field);
}
void group(vc_laser_pv_probe::PvProbe &probe,uint32_t field) {
    probe.StartGroup();counter+=17;probe.StopGroup(field);
}
EXPORT int capture(int mode,int total,int kx,int core,int descriptors,int guard,int repeat) {
    for(int i=0;i<4*3*3*64;++i) trace_data[i]=0;
    counter=101;faults=barriers=flushes=0;
    vc_laser_pv_probe::PvProbe probe;
    probe.Init(reinterpret_cast<uint8_t*>(trace_data),mode,core);
    for(int pass=0;pass<repeat;++pass) for(int round=0;round<total;++round) {
        probe.BeginCall(round,total);
        for(int descriptor=0;descriptor<descriptors;++descriptor) {
            const bool start=descriptor==0?round==0:false;
            const bool end=descriptor==0?round==total-1:true;
            probe.Descriptor(descriptor,descriptors,kx,2,start,end);
            probe.SetGuard(guard!=0);
            if(!start) group(probe,23);
            if(guard) {group(probe,40);group(probe,41);}
            for(int k=0;k<kx;++k) {
                probe.BeginTile(k);
                for(uint32_t field=26;field<=39;++field) {
                    if((field==33||field==34) && !(guard && k+1<kx)) continue;
                    phase(probe,field);
                }
                probe.Count(43,2);probe.Count(44,8);probe.Count(45,16);
                probe.Count(46,2);probe.Count(47,k+1==kx?2:0);
                probe.Count(48,guard && k+1<kx?1:0);
                probe.EndTile();
            }
            if(end) group(probe,24);
        }
        probe.EndCall();
    }
    return faults;
}
EXPORT uint64_t trace_value(int index) { return trace_data[index]; }
EXPORT int barrier_count() { return barriers; }
EXPORT int flush_count() { return flushes; }
EXPORT int run_model() {return faults;}
"""


@pytest.fixture(scope="module")
def writer(cpp_compiler, tmp_path_factory):
    directory = tmp_path_factory.mktemp("pv_writer_roundtrip")
    (directory / "kernel_operator.h").write_text("// Functional host fixture only.\n", encoding="utf-8")
    header = (ROOT / "ascend/laser_cube_probe/vc_laser_pv_probe.h").read_text(encoding="utf-8")
    library = compile_library(cpp_compiler, directory, STUB + header + DRIVER)
    library.capture.argtypes = [ctypes.c_int] * 7
    library.capture.restype = ctypes.c_int
    library.trace_value.argtypes = [ctypes.c_int]
    library.trace_value.restype = ctypes.c_uint64
    library.barrier_count.restype = library.flush_count.restype = ctypes.c_int
    return library


def capture(writer, mode, *, total=5, kx=8, core=0, descriptors=2, guard=True, repeat=1):
    assert writer.capture(mode, total, kx, core, descriptors, guard, repeat) == 0
    flat = [writer.trace_value(i) for i in range(4*3*3*64)]
    nested = [[[[flat[((c*3+r)*3+k)*64+f] for f in range(64)]
                 for k in range(3)] for r in range(3)] for c in range(4)]
    # Force the actual exported format through JSON, independent of tensor APIs.
    return json.loads(json.dumps(nested))


def expected_positions(size):
    positions = {}
    for position, index in enumerate((0, size//2, size-1)):
        if 0 <= index < size:
            positions.setdefault(index, position)
    return positions


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("total,kx", [(1, 1), (2, 2), (3, 3), (5, 8)])
def test_real_writer_roundtrip_modes_and_first_match_sampling(writer, mode, total, kx):
    trace = capture(writer, mode, total=total, kx=kx, repeat=2)
    decoded = records.decode_trace(trace, mode, require_records=mode>0)
    rounds, keys = expected_positions(total), expected_positions(kx)
    expected = set() if mode == 0 else {(r, k) for r in rounds for k in keys}
    assert {(r["round"], r["k"]) for r in decoded} == expected
    assert len(decoded) == len(expected)  # duplicate calls do not overwrite/add records
    assert writer.barrier_count() == 0
    assert writer.flush_count() == len(decoded)*8
    assert all(r["round_position"] == rounds[r["round"]] and
               r["k_position"] == keys[r["k"]] for r in decoded)
    assert all(r["descriptor"] == 0 and r["descriptors"] == 2 for r in decoded)
    for record in decoded:
        assert record["version"] == 2
        assert not record["valid_flags"] & 2
        assert record["call_completed_end"] == record["call_completion_ticks"] == record["call_total_ticks"] == 0
        if record["group_valid"]:
            assert record["call_begin"] > 0 and record["call_issue_ticks"] > 0
        else:
            assert all(record[field] == 0 for field in records.FIELD_NAMES[17:25])
        if mode == 2:
            assert record["phase_mask"] != 0


@pytest.mark.parametrize("case", [dict(total=0), dict(kx=0), dict(core=4), dict(descriptors=0)])
def test_inactive_real_writer_does_not_invent_zero_duration_samples(writer, case):
    trace = capture(writer, 2, **case)
    assert writer.barrier_count() == 0
    assert records.decode_trace(trace, 2) == []
    with pytest.raises(ValueError, match="required|records"):
        records.decode_trace(trace, 2, require_records=True)


def test_full_four_core_real_writer_coverage_and_aggregation(writer):
    combined = [[[[0]*64 for _ in range(3)] for _ in range(3)] for _ in range(4)]
    for core in range(4):
        trace = capture(writer, 2, core=core)
        combined[core] = trace[core]
    decoded = records.decode_trace(combined, 2, require_records=True)
    records.require_long_coverage(decoded)
    assert len(decoded) == 36
    summary = records.summarize_records([decoded])
    assert sum(row["samples"] for row in summary["calls"]) == 12  # not 36
    assert sum(row["samples"] for row in summary["tiles"]) == 36
    assert all(not {"call_completion_ticks", "call_total_ticks"} & row["ticks"].keys()
               for row in summary["calls"])
    assert all(row["inactive"][field] == row["samples"]
               for row in summary["calls"]
               for field in ("call_completion_ticks", "call_total_ticks"))
    first_call = next(row for row in summary["calls"] if row["round_position"] == 0)
    assert first_call["inactive"]["history_issue_ticks"] == 4
    assert "history_issue_ticks" not in first_call["ticks"]
    middle_call = next(row for row in summary["calls"] if row["round_position"] == 1)
    assert middle_call["ticks"]["history_issue_ticks"]["samples"] == 4
    assert "output_issue_ticks" not in middle_call["ticks"]
    last_tiles = [row for row in summary["tiles"] if row["k_position"] == 2]
    assert all("next_v_copy_issue_ticks" not in row["ticks"] and
               row["inactive"]["next_v_copy_issue_ticks"] == 4 for row in last_tiles)


def test_guard_disabled_has_no_prefetch_samples_but_keeps_regular_tile_windows(writer):
    trace = capture(writer, 2, guard=False)
    decoded = records.decode_trace(trace, 2, require_records=True)
    assert all(not r["v_prefetch"] and not r["next_v_copy_calls"] for r in decoded)
    summary = records.summarize_records([decoded])
    assert all("prime_v_copy_issue_ticks" not in row["ticks"] for row in summary["calls"])
    assert all("p_copy_issue_ticks" in row["ticks"] and
               "next_v_copy_issue_ticks" not in row["ticks"] for row in summary["tiles"])


@pytest.mark.parametrize("damage", [
    "magic", "version", "tile_count", "phase_hook", "group_duplicate", "group_missing",
    "mode", "storage", "round", "k", "flags", "reserved", "timestamp_delta", "cross_k_metadata",
])
def test_partial_or_inconsistent_real_records_are_rejected(writer, damage):
    trace = capture(writer, 2)
    fields = {name: i for i, name in enumerate(records.FIELD_NAMES)}
    first = trace[0][0][0]
    if damage in ("magic", "version"):
        first[fields[damage]] = 0
    elif damage == "tile_count":
        first[fields["mad_calls"]] = 0
    elif damage == "phase_hook":
        first[fields["p_copy_issue_ticks"]] = 0
        first[fields["phase_mask"]] &= ~(1 << (fields["p_copy_issue_ticks"]-26))
    elif damage == "group_duplicate":
        trace[0][0][1][fields["history_issue_ticks"]] = 7
    elif damage == "group_missing":
        trace[0][0][0] = [0]*64
    elif damage == "mode":
        first[fields["mode"]] = 1
    elif damage == "storage":
        first[fields["core"]] = 1
    elif damage == "round":
        first[fields["round"]] = 1
    elif damage == "k":
        first[fields["k"]] = 1
    elif damage == "flags":
        first[fields["valid_flags"]] = 0
    elif damage == "reserved":
        first[fields["reserved0"]] = 1
    elif damage == "timestamp_delta":
        first[fields["call_issue_ticks"]] += 1
    elif damage == "cross_k_metadata":
        trace[0][0][1][fields["descriptors"]] += 1
    with pytest.raises(ValueError):
        records.decode_trace(trace, 2, require_records=True)


@pytest.mark.parametrize("mode", [1, 2])
def test_v1_real_records_are_rejected_even_when_all_other_metadata_is_valid(writer, mode):
    trace = capture(writer, mode)
    for keys in trace[0]:
        for row in keys:
            if row[0]:
                row[1] = 1
    with pytest.raises(ValueError, match="version"):
        records.decode_trace(trace, mode, require_records=True)


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("field", ["call_completed_end", "call_completion_ticks", "call_total_ticks", "valid_flags"])
def test_no_fence_schema_rejects_any_claim_of_completed_work(writer, mode, field):
    trace = capture(writer, mode)
    row = trace[0][0][0]
    index = records.FIELD_NAMES.index(field)
    if field == "valid_flags":
        row[index] |= 2
    else:
        row[index] = 1
    with pytest.raises(ValueError):
        records.decode_trace(trace, mode, require_records=True)


@pytest.mark.parametrize("damage", ["missing_core", "missing_k", "fallback_guard"])
def test_long_coverage_refuses_incomplete_or_fallback_real_samples(writer, damage):
    combined = [[[[0]*64 for _ in range(3)] for _ in range(3)] for _ in range(4)]
    for core in range(4):
        combined[core] = capture(writer, 2, core=core, guard=damage!="fallback_guard")[core]
    if damage == "missing_core":
        combined[3] = [[[0]*64 for _ in range(3)] for _ in range(3)]
    elif damage == "missing_k":
        combined[0][0][2] = [0]*64
    decoded = records.decode_trace(combined, 2, require_records=True)
    with pytest.raises(ValueError, match="Long PV trace"):
        records.require_long_coverage(decoded)
