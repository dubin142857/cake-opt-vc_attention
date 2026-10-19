"""Exact source delta, offline preparation and actual QK-method C++ checks."""
import json
from pathlib import Path
import subprocess

import pytest

from scripts import prepare_mindie_laser_quant_reuse_source as baseline
from scripts import prepare_mindie_laser_quant_gain_source as prepare
from scripts import mindie_laser_quant_gain_patch as patch
from scripts.prepare_cann_fas_source import RepositoryLock, git
from tests.test_mindie_laser_expcast_cpp import cpp_compiler, compile_syntax

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pinned():
    reference = ROOT / "build/mindie-sd-reference"
    if not (reference / ".git").exists():
        pytest.skip("Pinned MindIE source cache unavailable; tests never fetch")
    assert git(reference, "rev-parse", "HEAD").decode().strip() == prepare.SOURCE_LOCK.commit
    return reference


@pytest.fixture(scope="module")
def baseline_cube(pinned):
    files, _ = baseline.expected_files(pinned)
    return files[f"{baseline.CUSTOM_OP_ROOT}/op_kernel/CubeForward.h"]


def test_only_pure_gain_arithmetic_and_signature_move(baseline_cube):
    new = patch.patch_cube(baseline_cube)
    reverted = new.replace(patch.NEW_SIGNATURE, patch.OLD_SIGNATURE)
    reverted = reverted.replace(patch.PREPARE_BITS, b"")
    reverted = reverted.replace(b"                    gain_bits_first, gain_bits_second);",
        b"                    q_scale_first, q_scale_second, k_scale);")
    reverted = reverted.replace(patch.NEW_GAIN, patch.OLD_GAIN)
    assert reverted == baseline_cube
    # This strict inversion covers all Cube/PV/Vector handshakes, DCCI, unit
    # flags, buffers and addressing; the remaining checks explain the schedule.
    start, end = new.index(patch.QK_BEGIN), new.index(patch.RUN_BEGIN)
    caller, callee = new[start:end].split(b"::cube1_base_matmul(", 1)
    assert caller.index(patch.PREPARE_BITS) < caller.index(b"// Load B to L1")
    assert caller.count(b"* k_scale;") == 2
    assert b"* k_scale" not in callee and b"ReadQuantScale(" not in callee
    old_callee = baseline_cube[start:].split(b"::cube1_base_matmul(", 1)[1]
    for name in (b"set_quant_pre(", b"pipe_barrier(PIPE_FIX);", b"ReadQuantScale("):
        assert new.count(name) == baseline_cube.count(name)
    # Hardware configuration still follows MAD and is guarded by the same
    # skipped-tile predicate, immediately before the same DEQF16 transfer.
    copy = b"                copy_matrix_cc_to_gm(gm_out + out_offset,"
    setup = b"                set_quant_pre(static_cast<uint64_t>(bits));\n                pipe_barrier(PIPE_FIX);\n"
    assert setup + copy in callee and setup + copy in old_callee
    assert callee.index(b"            auto out_offset") < callee.index(setup)
    assert b"            if (!l0_skip_flag) {\n                // FIX" in callee


def test_anchor_changes_repeated_patch_and_wrong_mapping_fail_closed(baseline_cube):
    with pytest.raises(ValueError, match="ky=2"):
        patch.patch_cube(baseline_cube.replace(patch.TILING, patch.TILING.replace(b"64, 2", b"64, 4")))
    with pytest.raises(ValueError):
        patch.patch_cube(patch.patch_cube(baseline_cube))
    with pytest.raises(ValueError):
        patch.patch_cube(baseline_cube.replace(patch.OLD_GAIN, b"// different numerical contract"))
    with pytest.raises(ValueError):
        patch.patch_cube(baseline_cube + b"\n" + patch.TILING)


def test_independent_identity_and_single_kernel_delta(pinned):
    old_files, old_records = baseline.expected_files(pinned)
    new_files, new_records = prepare.expected_files(pinned)
    changed = []
    for old_path, old_data in old_files.items():
        path = prepare.rename(old_path.encode()).decode()
        if new_files[path] != prepare.rename(old_data):
            changed.append(path)
        if old_path in old_records:
            assert new_records[path]["original_sha256"] == old_records[old_path]["original_sha256"]
            assert new_records[path]["scale_reuse_prepared_sha256"] == prepare.sha256(old_data)
    assert changed == [f"{prepare.CUSTOM_OP_ROOT}/op_kernel/CubeForward.h"]
    marker = new_files[f"{prepare.CUSTOM_OP_ROOT}/op_host/vc_source_marker.h"]
    assert prepare.MARKER_TOKEN.encode() in marker and prepare.MARKER_ENV.encode() in marker
    assert baseline.MARKER_TOKEN.encode() not in marker
    assert all(not path.startswith(baseline.CUSTOM_OP_ROOT + "/") for path in new_files)


@pytest.fixture
def prepared(tmp_path, monkeypatch, pinned):
    reference = tmp_path / "reference"
    reference.mkdir()
    def run(*args):
        return subprocess.run(["git", "-C", str(reference), *args], capture_output=True, check=True).stdout
    run("init", "--quiet")
    run("config", "core.autocrlf", "false")
    run("config", "user.name", "gain preparation source test")
    run("config", "user.email", "gain-preparation@example.invalid")
    names = git(pinned, "ls-tree", "-r", "--name-only", "HEAD", "--", prepare.OP_ROOT).decode().splitlines()
    for name in [*names, prepare.PREPARE_SCRIPT]:
        path = reference / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(git(pinned, "show", f"HEAD:{name}"))
    run("add", ".")
    run("commit", "--quiet", "-m", "locked bytes fixture")
    lock = RepositoryLock("https://example.invalid/laser.git", run("rev-parse", "HEAD").decode().strip())
    monkeypatch.setattr(prepare, "SOURCE_LOCK", lock)
    source, manifest = tmp_path / "candidate", tmp_path / "candidate.json"
    report = prepare.prepare(source, reference, manifest, offline=True)
    return reference, source, manifest, report


def test_preparation_is_incremental_and_does_not_modify_baseline(prepared):
    reference, source, manifest, report = prepared
    before = {name: (source / name).stat().st_mtime_ns for name in report["source_files"]}
    manifest_before = manifest.stat().st_mtime_ns
    assert prepare.validate_prepared_source(source, manifest) == report
    assert prepare.prepare(source, reference, manifest, offline=True) == report
    assert before == {name: (source / name).stat().st_mtime_ns for name in before}
    assert manifest.stat().st_mtime_ns == manifest_before
    assert git(source, "diff", "--name-only").decode().splitlines() == [prepare.PREPARE_SCRIPT]
    assert not git(reference, "status", "--porcelain")


def test_unknown_kernel_edits_fail_without_overwrite(prepared):
    reference, source, manifest, _ = prepared
    cube = source / prepare.CUSTOM_OP_ROOT / "op_kernel/CubeForward.h"
    data = cube.read_bytes() + b"// user edit\n"
    cube.write_bytes(data)
    prior = manifest.read_bytes()
    with pytest.raises(prepare.PreparationError):
        prepare.prepare(source, reference, manifest)
    with pytest.raises(prepare.PreparationError):
        prepare.validate_prepared_source(source, manifest)
    assert cube.read_bytes() == data and manifest.read_bytes() == prior


def test_identity_cannot_claim_hardware_configuration_moved(prepared):
    _, source, manifest, report = prepared
    for key, value in (("variant", "quant_scale_reuse"), ("native_api", baseline.NATIVE_API),
                       ("gain_preparation_contract", {**prepare.GAIN_CONTRACT, "hardware_configuration": "early"})):
        manifest.write_text(json.dumps({**report, key: value}), encoding="utf-8")
        with pytest.raises(prepare.PreparationError, match="identity"):
            prepare.validate_prepared_source(source, manifest)


def test_preparation_cannot_enable_network_with_legacy_offline_false(prepared, monkeypatch):
    reference, source, manifest, _ = prepared
    original = prepare.ensure_checkout
    seen = []
    def verify(path, lock, offline):
        seen.append(offline)
        assert offline is True
        return original(path, lock, offline)
    monkeypatch.setattr(prepare, "ensure_checkout", verify)
    prepare.prepare(source, reference, manifest, offline=False)
    assert seen == [True]


def test_existing_reuse_source_is_discovered_before_old_baselines(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    reuse = tmp_path / "custom reuse source"
    (tmp_path / "results/mindie_laser_quant_reuse_build.json").write_text(
        json.dumps(dict(status="passed", source_dir=str(reuse))), encoding="utf-8")
    missing = tmp_path / "missing reference"
    assert prepare.cached_references(missing)[:2] == [missing, reuse]


def test_actual_changed_qk_methods_compile(cpp_compiler, baseline_cube, tmp_path):
    data = patch.patch_cube(baseline_cube).decode()
    template = "template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>\n"
    begin = data.rfind(template, 0, data.index(patch.QK_BEGIN.decode()))
    end = data.rfind(template, 0, data.index(patch.RUN_BEGIN.decode()))
    methods = data[begin:end]
    preamble = r"""
typedef signed char int8_t; typedef signed int int32_t;
typedef signed long long int64_t; typedef unsigned int uint32_t;
typedef unsigned long long uint64_t; struct half { unsigned short bits; };
#define __aicore__
#define __inline__ inline
#define __gm__
#define __cbuf__
#define __ca__
#define __cb__
#define __cc__
enum {SIZE_16=16,SIZE_32=32,SIZE_128=128,SIZE_256=256};
enum {PIPE_MTE1,PIPE_MTE2,PIPE_M,PIPE_FIX,inc,DEQF16};
void set_flag(int,int,int); void wait_flag(int,int,int);
void set_quant_pre(uint64_t); void pipe_barrier(int);
void copy_gm_to_cbuf_multi_nd2nz_b8(int8_t*,int8_t*,int,int,int,int,int,int,int,int,int);
void load_cbuf_to_ca(int8_t*,int8_t*,int,int,int,int,int,bool,int);
void load_cbuf_to_cb(int8_t*,int8_t*,int,int,int,int,int,bool,int);
void mad(int32_t*,int8_t*,int8_t*,int,int,int,int,int,int,int);
void copy_matrix_cc_to_gm(half*,int32_t*,int,int,int,int,int,int,int,int,bool,bool);
namespace Address {
template<class L,class R,class O> struct PhyAddrForwardCube1Online {
    int kx,ky,lineStride; bool upperRight,lowerLeft; L*left; R*right; O*out;
};
}
template<class TYPE,bool IF_BF16,class WORKSPACE_TYPE> class CubeForward {
public:
    template<class L,class R,class O> void cube1_matmul_op(const Address::PhyAddrForwardCube1Online<L,R,O>*,int64_t);
    void cube1_base_matmul(int8_t*,int8_t*,TYPE*,int32_t,int32_t,bool,uint32_t,uint32_t);
    float ReadQuantScale(float*); float *q_scales_,*k_scales_; int8_t *gm_Q,*gm_K;
    TYPE *l1_a_ping_,*l1_a_pong_,*l1_b_ping_,*l1_b_pong_;
    TYPE *l0_a_ping_,*l0_a_pong_,*l0_b_ping_,*l0_b_pong_;
    float *l0_c_ping_,*l0_c_pong_;
    int ping_pong_flag_l1_a_,ping_pong_flag_l1_b_,ping_pong_flag_l0_a_,ping_pong_flag_l0_b_,ping_pong_flag_l0_c_;
};
"""
    invoke = """
void instantiate(CubeForward<half,false,half>& op,
                 Address::PhyAddrForwardCube1Online<int8_t,int8_t,half>* src) {
    op.cube1_matmul_op(src,2);
}
"""
    code, log = compile_syntax(cpp_compiler, tmp_path, preamble + methods + invoke)
    assert code == 0, log
