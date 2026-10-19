"""Pinned scale-read transform, source isolation and host-only signature checks."""
import json
from pathlib import Path
import subprocess

import pytest

from scripts import prepare_mindie_laser_quant_source as baseline
from scripts import prepare_mindie_laser_quant_reuse_source as prepare
from scripts import mindie_laser_quant_reuse_patch as patch
from scripts.prepare_cann_fas_source import RepositoryLock, git
from tests.test_mindie_laser_expcast_cpp import cpp_compiler, compile_syntax

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pinned():
    reference = ROOT / "build/mindie-sd-reference"
    if not (reference / ".git").exists():
        pytest.skip("Pinned MindIE source cache unavailable; source tests do not fetch")
    assert git(reference, "rev-parse", "HEAD").decode().strip() == prepare.SOURCE_LOCK.commit
    return reference


@pytest.fixture(scope="module")
def baseline_cube(pinned):
    files, _ = baseline.expected_files(pinned)
    return files[f"{baseline.CUSTOM_OP_ROOT}/op_kernel/CubeForward.h"]


def test_only_scale_read_schedule_and_qk_signature_change(baseline_cube):
    new = patch.patch_cube(baseline_cube)
    # Strict inversion proves no extra changes to MMAs, transfers, FIX product
    # truncation or barriers, PV, Vector synchronization or the address mapping.
    reverted = new.replace(
        b"        float q_scale_first, float q_scale_second, float k_scale);",
        b"        int64_t q_scale_base, int64_t k_scale_index);")
    q_start = reverted.index(b"\n\n        // The pinned descriptor has ky=2.")
    q_end = reverted.index(b"\n\n        for (int m_index", q_start)
    reverted = reverted[:q_start] + reverted[q_end:]
    k_start = reverted.index(b"\n                // One K scale serves")
    k_end = reverted.index(b"\n                bool upper_right_flag", k_start)
    reverted = reverted[:k_start] + reverted[k_end:]
    reverted = reverted.replace(b"                    q_scale_first, q_scale_second, k_scale);",
        b"                    (gm_a - gm_Q) / (128 * 128), (gm_b - gm_K) / (128 * 128) + n_index);")
    reverted = reverted.replace(b"    float q_scale_first, float q_scale_second, float k_scale)",
        b"    int64_t q_scale_base, int64_t k_scale_index)")
    reverted = reverted.replace(patch.NEW_GAIN, patch.OLD_GAIN)
    assert reverted == baseline_cube
    # Q reads occur outside n_index; K read occurs outside both Q-block MMAs.
    main = new[new.index(patch.QK_BEGIN):new.index(patch.RUN_BEGIN)]
    outer, inner = main.split(b"            for (int n_index", 1)
    caller, callee = inner.split(b"::cube1_base_matmul(", 1)
    assert outer.count(b"ReadQuantScale(") == 2
    assert caller.count(b"ReadQuantScale(") == 1
    assert b"ReadQuantScale(" not in callee
    assert b"const float q_scale_second = ky > 1" in outer
    assert b"static " not in main
    assert b"DataCacheCleanAndInvalid" in new
    assert new.count(b"pipe_barrier(PIPE_FIX);") == baseline_cube.count(b"pipe_barrier(PIPE_FIX);")


def test_changed_addressing_or_second_patch_fail_closed(baseline_cube):
    with pytest.raises(ValueError, match="ky=2"):
        patch.patch_cube(baseline_cube.replace(patch.TILING, patch.TILING.replace(b"64, 2", b"64, 4")))
    with pytest.raises(ValueError, match="ky=2"):
        patch.patch_cube(baseline_cube + b"\n" + patch.TILING)
    with pytest.raises(ValueError):
        patch.patch_cube(patch.patch_cube(baseline_cube))


def test_independent_identity_and_only_cube_algorithm_delta(pinned):
    old_files, old_records = baseline.expected_files(pinned)
    new_files, new_records = prepare.expected_files(pinned)
    changed = []
    for old_path, old_data in old_files.items():
        path = prepare.rename(old_path.encode()).decode()
        assert path in new_files
        if new_files[path] != prepare.rename(old_data):
            changed.append(path)
        if old_path in old_records:
            assert new_records[path]["original_sha256"] == old_records[old_path]["original_sha256"]
            assert new_records[path]["baseline_prepared_sha256"] == prepare.sha256(old_data)
    assert changed == [f"{prepare.CUSTOM_OP_ROOT}/op_kernel/CubeForward.h"]
    marker = new_files[f"{prepare.CUSTOM_OP_ROOT}/op_host/vc_source_marker.h"]
    assert prepare.MARKER_TOKEN.encode() in marker
    assert prepare.MARKER_ENV.encode() in marker
    assert baseline.MARKER_TOKEN.encode() not in marker
    assert all(not p.startswith(baseline.CUSTOM_OP_ROOT + "/") for p in new_files)


@pytest.fixture
def prepared(tmp_path, monkeypatch, pinned):
    """A small real Git repository containing the locked operator bytes."""
    reference = tmp_path / "reference"
    reference.mkdir()
    def run(*args):
        return subprocess.run(["git", "-C", str(reference), *args], capture_output=True, check=True).stdout
    run("init", "--quiet")
    run("config", "core.autocrlf", "false")
    run("config", "user.name", "scale reuse source test")
    run("config", "user.email", "scale-reuse@example.invalid")
    names = git(pinned, "ls-tree", "-r", "--name-only", "HEAD", "--", baseline.OP_ROOT).decode().splitlines()
    for name in [*names, baseline.PREPARE_SCRIPT]:
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


def test_prepare_is_incremental_and_original_files_untouched(prepared):
    reference, source, manifest, report = prepared
    names = list(report["source_files"])
    before = {name: (source / name).stat().st_mtime_ns for name in names}
    manifest_before = manifest.stat().st_mtime_ns
    assert prepare.validate_prepared_source(source, manifest) == report
    assert prepare.prepare(source, reference, manifest, offline=True) == report
    assert before == {name: (source / name).stat().st_mtime_ns for name in names}
    assert manifest.stat().st_mtime_ns == manifest_before
    assert git(source, "diff", "--name-only").decode().splitlines() == [prepare.PREPARE_SCRIPT]
    assert not git(reference, "status", "--porcelain")


def test_unknown_edits_are_not_overwritten(prepared):
    reference, source, manifest, _ = prepared
    cube = source / prepare.CUSTOM_OP_ROOT / "op_kernel/CubeForward.h"
    changed = cube.read_bytes() + b"// user change\n"
    cube.write_bytes(changed)
    before_manifest = manifest.read_bytes()
    with pytest.raises(prepare.PreparationError):
        prepare.prepare(source, reference, manifest, offline=True)
    with pytest.raises(prepare.PreparationError):
        prepare.validate_prepared_source(source, manifest)
    assert cube.read_bytes() == changed
    assert manifest.read_bytes() == before_manifest


def test_manifest_cannot_claim_baseline_or_omit_cache_invalidation(prepared):
    _, source, manifest, report = prepared
    for key, value in (("variant", "quant"), ("native_api", baseline.NATIVE_API),
                       ("scale_reuse_contract", {**prepare.REUSE_CONTRACT, "cache_invalidation": "removed"})):
        manifest.write_text(json.dumps({**report, key: value}), encoding="utf-8")
        with pytest.raises(prepare.PreparationError, match="identity"):
            prepare.validate_prepared_source(source, manifest)


def test_actual_cached_qk_methods_have_valid_cpp_signatures(cpp_compiler, baseline_cube, tmp_path):
    """Instantiate actual methods; CCE address spaces/hardware still need NPU tests."""
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
    void cube1_base_matmul(int8_t*,int8_t*,TYPE*,int32_t,int32_t,bool,float,float,float);
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
