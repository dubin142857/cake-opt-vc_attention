"""Scan P+V prefetch against the measured V-only parent, using local objects."""
import importlib.util
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import prepare_mindie_laser_quant_vscale_pv_prefetch_source as baseline
from scripts import prepare_mindie_laser_quant_vscale_pv_p_prefetch_source as candidate


def main():
    # Reuse the same complete-family source comparison and analyzer protocol.
    # Override destinations before execution; the previous audit stays intact.
    driver = ROOT / "reports/ascend-pv-prefetch-sync-audit/regenerate.py"
    spec = importlib.util.spec_from_file_location("pv_family_audit", driver)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.baseline = baseline
    audit.candidate = candidate
    audit.REPORT = Path(__file__).resolve().parent
    audit.BUILD = ROOT / "build/ascend_pv_p_prefetch_audit"
    audit.main()
    old, _ = baseline.expected_files(audit.REFERENCE)
    new, _ = candidate.expected_files(audit.REFERENCE)
    before = old[baseline.CUSTOM_OP_ROOT + "/op_kernel/CubeForward.h"]
    after = new[candidate.CUSTOM_OP_ROOT + "/op_kernel/CubeForward.h"]
    old_pv = audit.method(before, "cube2_matmul_op")
    new_pv = audit.method(after, "cube2_matmul_op")
    loads = rb"load_cbuf_to_c[ab]\([^;]*\);"
    a, b = re.findall(loads, old_pv), re.findall(loads, new_pv)
    assert a == b, "P/V L1-to-L0 operations changed"
    release = b"set_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_a_ + 2);"
    # This includes the original A release, slot toggles and tail normalization.
    assert old_pv[old_pv.index(release):] == new_pv[new_pv.index(release):]
    vector = new[candidate.CUSTOM_OP_ROOT + "/op_kernel/VectorForward.h"]
    run = audit.method(after, "Run")
    assert b"is_syn = true;" in run
    assert run.count(b"wait_flag_dev(AIV2AICFLAGID);") == 3
    assert run.count(b"cube2_matmul_op(src, src_len, 1,") == 3
    assert b"ffts_cross_core_sync(PIPE_MTE3, config4);" in vector
    audit.write_json(audit.REPORT / "p_lifecycle_comparison.json", dict(
        l1_to_l0_call_sites=len(a), l1_to_l0_identical=True,
        a_release_toggles_tail_identical=True, run_identical=True,
        vector_identical=True, cube_pv_sites=3, cube_publication_wait_sites=3,
        vector_publication="ffts_cross_core_sync(PIPE_MTE3, config4)",
        boundary="Publication ownership also manually traced through Run and the unchanged round mapper; counts alone are not a synchronization proof."))


if __name__ == "__main__":
    main()
