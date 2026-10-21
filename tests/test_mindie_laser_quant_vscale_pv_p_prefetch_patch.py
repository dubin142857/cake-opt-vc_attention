"""Actual PV-method host model for same-round P lookahead.

Reuses the unchanged prior V-lookahead model and strengthens A-slot ownership.
These are host event/operand checks, not device synchronization or speed claims.
"""
import ctypes
from pathlib import Path
import re

import pytest

from scripts import mindie_laser_quant_vscale_pv_p_prefetch_patch as patch
from scripts import prepare_mindie_laser_quant_vscale_pv_prefetch_source as baseline
from scripts.prepare_cann_fas_source import git
from tests import test_mindie_laser_quant_vscale_pv_prefetch_patch as prior
from tests.test_mindie_laser_expcast_cpp import cpp_compiler
from tests.test_mindie_laser_quant_vscale_patch import compile_library, function

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source_files():
    reference = ROOT / "build/mindie-sd-reference"
    if not (reference / ".git").exists():
        pytest.skip("Pinned cached source unavailable; no network fetch")
    assert git(reference, "rev-parse", "HEAD").decode().strip() == baseline.SOURCE_LOCK.commit
    files, _ = baseline.expected_files(reference)
    return {Path(name).name: data for name, data in files.items() if "/op_kernel/" in name}


def restore(new, old):
    new = new.replace(patch.PRIME, b"").replace(b"\n" + patch.LOOKAHEAD, b"")
    block = patch.p_load_block(old[old.index(patch.PV_BEGIN):old.index(patch.PV_END)])
    new = new.replace(patch.guarded(block), block)
    guarded_set = patch.parent.guarded(patch.P_READY_SET).replace(b"pv_v_prefetch", b"pv_p_prefetch")
    return new.replace(guarded_set, patch.P_READY_SET)


def test_only_p_gm_l1_schedule_changes(source_files):
    old = source_files["CubeForward.h"]
    new = patch.patch_cube(old)
    assert restore(new, old) == old
    assert patch.parent.PRIME in new and patch.parent.LOOKAHEAD in new
    before, after = old.index(patch.PV_BEGIN), old.index(patch.PV_END)
    assert new[:before] == old[:before]
    assert new[new.index(patch.PV_END):] == old[after:]
    ordered = rb"(?:load_cbuf_to_ca|load_cbuf_to_cb|mad|copy_matrix_cc_to_gm|diag_matmul|set_quant_pre)\([^;]*\);"
    assert re.findall(ordered, new) == re.findall(ordered, old)
    for name, data in source_files.items():
        if name != "CubeForward.h":
            assert patch.apply(name, data) is data
    assert new.count(patch.P_RELEASE) == old.count(patch.P_RELEASE)
    assert b"const bool pv_p_prefetch = pv_v_prefetch && Kx > 1;" in new


def test_p_producer_publishes_completed_sections_before_all_pv_calls(source_files):
    vector = source_files["VectorForward.h"].decode()
    run = function(vector, "::Run()")
    section = function(run, "for (int32_t sectionLoop = 0;")
    publish = "ffts_cross_core_sync(PIPE_MTE3, config4);"
    assert run.count(publish) == 1
    assert run.index(section) + len(section) < run.index(publish)
    assert "attention_score_short_double_line_one(" in section
    assert "AIV2AICFLAGID << 8" in run
    cube = source_files["CubeForward.h"].decode()
    run = function(cube, "::Run()")
    assert run.count("wait_flag_dev(AIV2AICFLAGID);") == 3
    calls = list(re.finditer(r"cube2_matmul_op\(src, src_len, 1, [^;]+;", run))
    assert len(calls) == 3
    previous_call = 0
    for call in calls:
        # Covers Z==1, steady pipeline and both drain rounds independently.
        prefix = run[previous_call:call.start()]
        assert "wait_flag_dev(AIV2AICFLAGID);" in prefix
        assert prefix.rfind("wait_flag_dev(AIV2AICFLAGID);") < prefix.rfind("address.addrMapping_cube2(")
        previous_call = call.end()


@pytest.mark.parametrize("mutation", [
    lambda d: d.replace(patch.parent.LOOKAHEAD, b"old_v_schedule"),
    lambda d: d.replace(patch.parent.LAYOUT[1], patch.parent.LAYOUT[1].replace(b"64", b"32")),
    lambda d: d.replace(b"vcore_num_per_head,                    // ndNum", b"2, // forced wrong ND"),
    lambda d: d.replace(patch.P_READY_WAIT, patch.P_READY_WAIT + patch.P_READY_WAIT),
    lambda d: d.replace(patch.P_RELEASE, b"changed_release();"),
    patch.patch_cube,
])
def test_source_drift_and_stacking_fail_closed(source_files, mutation):
    with pytest.raises(ValueError):
        patch.patch_cube(mutation(source_files["CubeForward.h"]))


def replace_exact(source, old, new):
    assert source.count(old) == 1
    return source.replace(old, new)


def host_model():
    source = prior.MODEL
    source = replace_exact(source, "uint64_t sequence_hash;", """uint64_t sequence_hash,p_copy_hash;
int p_prefetch_hits,producer_ready[4];
void mix_p(uint64_t x) {p_copy_hash=p_copy_hash*1099511628211ull+x+1469598103934665603ull;}""")
    source = replace_exact(source,
        "if(id<2 && s.read_count!=8) fail(25);",
        """if(id<2 && s.read_count!=8) fail(25);
   if(id>=2) {
     int expected=8*(ky[s.desc]-(upper[s.desc]&&s.tile==kx[s.desc]-1?1:0));
     if(s.read_count!=expected) fail(75);
   }""")
    source = replace_exact(source,
        "if(history[desc]&&!history_done[desc]) fail(33);",
        """if(history[desc]&&!history_done[desc]) fail(33);
 if(!is_v&&!producer_ready[desc]) fail(76);""")
    source = replace_exact(source,
        "mix(4000+desc*100+tile*2+row);mix(slot*100+nd);",
        """mix_p(4000+desc*100+tile*2+row);mix_p(slot*100+nd);
   if(row==0&&active_desc==desc&&tile==active_tile+1&&active_row==0) {
     int current=-1;
     for(int a=2;a<4;a++) if(s1[a].owned&&s1[a].readable&&
       s1[a].desc==desc&&s1[a].tile==active_tile) current=a;
     // Current P is live and not yet read when next P is issued elsewhere.
     if(current<2||current==slot||s1[current].read_count!=0) fail(77);
     ++p_prefetch_hits;
   }""")
    return source


def scenarios():
    source = prior.SCENARIOS
    source = replace_exact(source,
        "total_mad=0;total_fix=0;total_v=0;total_p=0;prefetch_hits=0;",
        "total_mad=0;total_fix=0;total_v=0;total_p=0;prefetch_hits=0;p_prefetch_hits=0;")
    source = replace_exact(source,
        "history_done[i]=tail_done[i]=v_loaded[i]=v_consumed[i]=0;last_v_tile[i]=-1;",
        "history_done[i]=tail_done[i]=v_loaded[i]=v_consumed[i]=0;last_v_tile[i]=-1;producer_ready[i]=0;")
    source = replace_exact(source,
        "num_tiles+=kx[i];num_rows+=kx[i]*ky[i];",
        """num_tiles+=kx[i];num_rows+=kx[i]*ky[i];
   // Models Run's completed-round MTE3 publication, verified above from source.
   producer_ready[i]=1;""")
    source = replace_exact(source, "sequence_hash=0;", "sequence_hash=0;p_copy_hash=0;")
    source = replace_exact(source,
        "if((!CANDIDATE||guard!=0)&&prefetch_hits) return 63;",
        """if((!CANDIDATE||guard!=0)&&prefetch_hits) return 63;
   if(P_CANDIDATE&&guard==0&&prefetched_expected(kx,4)!=p_prefetch_hits) return 78;
   if((!P_CANDIDATE||guard!=0)&&p_prefetch_hits) return 79;""")
    return source + "\nEXPORT uint64_t get_p_copies() {return p_copy_hash;}\n"


def compile_model(compiler, directory, data, candidate, *, model_mutation=None):
    model = host_model()
    if model_mutation:
        model = model_mutation(model)
    source = (f"#define CANDIDATE 1\n#define P_CANDIDATE {int(candidate)}\n" + model + prior.methods(data) +
              "\nint prefetched_expected(int *ks,int n) {int c=0;for(int i=0;i<n;i++)if(ks[i]>0)c+=ks[i]-1;return c;}\n" + scenarios())
    lib = compile_library(compiler, directory, source)
    lib.run_model.restype = ctypes.c_int
    lib.get_sequence.restype = ctypes.c_uint64
    lib.get_p_copies.restype = ctypes.c_uint64
    return lib.run_model(), lib.get_sequence(), lib.get_p_copies()


def test_actual_pv_preserves_math_operands_copy_arguments_and_slot_handoffs(cpp_compiler, source_files, tmp_path):
    old = source_files["CubeForward.h"]
    before = compile_model(cpp_compiler, tmp_path / "v_only", old, False)
    after = compile_model(cpp_compiler, tmp_path / "p_and_v", patch.patch_cube(old), True)
    assert before[0] == 0
    assert after[0] == 0
    # Transfer interleaving changes intentionally; arithmetic and per-P-copy
    # sequences are checked separately without falsely requiring old timing.
    assert before[1:] == after[1:]


@pytest.mark.parametrize("name,old,new", [
    ("missing_p_ready", patch.P_READY_WAIT, b""),
    ("missing_next_p_free", b"                    wait_flag(PIPE_MTE1, PIPE_MTE2, pv_next_p_slot + 2);", b""),
    ("wrong_p_slot", b"const uint32_t pv_next_p_slot = 1 - ping_pong_flag_l1_a_;", b"const uint32_t pv_next_p_slot = ping_pong_flag_l1_a_;"),
    ("p_tail_overread", b"if (pv_p_prefetch && i + 1 < Kx)", b"if (pv_p_prefetch && i + 1 <= Kx)"),
    ("missing_next_p_publish", b"                    set_flag(PIPE_MTE2, PIPE_MTE1, pv_next_p_slot + 2);", b""),
    ("wrong_p_second_row", b"left_start_addr + (i + 1) * l1_k * l1_m + line_stride", b"left_start_addr + (i + 1) * l1_k * l1_m"),
    ("wrong_p_parity", b"ping_pong_flag_l1_a_ = 1 - ping_pong_flag_l1_a_;", b"ping_pong_flag_l1_a_ = 0;"),
    ("wrong_nd2", b"0, vcore_num_per_head, l1_m / vcore_num_per_head, l1_k,", b"0, 2, l1_m / vcore_num_per_head, l1_k,"),
])
def test_model_detects_p_lifetime_address_ready_tail_and_copy_faults(cpp_compiler, source_files, tmp_path, name, old, new):
    data = patch.patch_cube(source_files["CubeForward.h"])
    begin, end = data.index(patch.PV_BEGIN), data.index(patch.PV_END)
    pv = data[begin:end]
    assert pv.count(old) >= 1
    damaged = data[:begin] + pv.replace(old, new) + data[end:]
    assert compile_model(cpp_compiler, tmp_path / name, damaged, True)[0] != 0


def test_model_rejects_p_release_before_both_current_rows_were_read(cpp_compiler, source_files, tmp_path):
    data = patch.patch_cube(source_files["CubeForward.h"])
    begin, end = data.index(patch.PV_BEGIN), data.index(patch.PV_END)
    pv = data[begin:end]
    assert pv.count(patch.P_RELEASE) == 1
    pv = pv.replace(patch.P_RELEASE, b"")
    assert pv.count(patch.LOOKAHEAD) == 1
    pv = pv.replace(patch.LOOKAHEAD, patch.P_RELEASE + b"\n" + patch.LOOKAHEAD)
    assert compile_model(cpp_compiler, tmp_path / "early_p_release", data[:begin] + pv + data[end:], True)[0] != 0


def test_model_requires_completed_vector_publication(cpp_compiler, source_files, tmp_path):
    data = patch.patch_cube(source_files["CubeForward.h"])
    # This deliberately revokes the external producer guarantee; host modeling
    # detects it, while the original device Run/Vector source check proves where
    # the unchanged publication/wait lives. It is not a device memory-model test.
    damage = lambda model: replace_exact(model, "if(!is_v&&!producer_ready[desc]) fail(76);",
        "if(!is_v) producer_ready[desc]=0; if(!is_v&&!producer_ready[desc]) fail(76);")
    assert compile_model(cpp_compiler, tmp_path / "unpublished_p", data, True, model_mutation=damage)[0] == 76
