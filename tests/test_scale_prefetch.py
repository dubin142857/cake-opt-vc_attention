"""Scale carry source boundaries, strict output gates and timing audit contracts."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_cuda_expcast as bench
import scale_prefetch_native as native
from validate_scale_prefetch import assert_exact


def test_isolated_experiment_and_no_installed_dispatch_change():
    args = bench.parse_args(["--experiment", "scale_prefetch"])
    assert args.output.name == "h200_cuda_expcast_scale_prefetch.json"
    assert args.trace_dir.name == "cuda_expcast_scale_prefetch_traces"
    assert bench.EXPERIMENTS[args.experiment][0] is native
    assert bench.parse_args([]).experiment == "balanced_max"
    assert native.EXACT_OUTPUT


def test_production_baseline_signature_and_math_remain_intact():
    original = (native.VC / "sage_layout.cuh").read_text(encoding="utf-8").replace("#pragma once", "")
    source = native.generate_source()
    assert original in source
    assert "bool SplitMass" not in source
    assert "clock64" not in source and "torch::zeros" not in source
    assert "if constexpr (Trace) return sage_vc::sage_vc_attention_kernel_scale_prefetch" in source
    assert "else return sage_vc::sage_vc_attention_kernel_sage_layout" in source
    # Whole regions must survive verbatim, rather than just checking names.
    for begin, end in (("    // Integer-max candidates", "    // wait for V"),
                       ("    // convert RS to float", "    // masking")):
        region = original[original.index(begin):original.index(end, original.index(begin))]
        assert region in native.candidate_kernel()


def test_scale_reads_are_bounded_and_after_pv_issue_before_wait():
    source = native.candidate_kernel()
    assert source.count("current_k_scale = K_scale[") == 2  # initial + NEXT
    assert source.count("q_scale * current_k_scale") == 2  # MAIN + masked tail
    main = source[source.index("for (uint32_t iter = 1;"):source.index("(num_iterations - 1)")]
    load = main.index("current_k_scale = K_scale[k_scale_idx + iter * k_scale_advance_offset]")
    assert "iter < num_iterations" in main
    assert main.rfind("warpgroup_commit_batch();", 0, load) > main.index("// wait for V")
    assert main.index("warpgroup_wait<0>();", load) > load
    assert "current_k_scale = K_scale" not in source[source.index("(num_iterations - 1)"):]


def test_patch_fails_closed_if_pipeline_anchor_changes(monkeypatch):
    read = Path.read_text
    def changed(path, *args, **kwargs):
        s = read(path, *args, **kwargs)
        if path.name == "sage_layout.cuh":
            s = s.replace("// Scale history while PV is live;", "// changed schedule")
        return s
    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(RuntimeError, match="Kernel source changed"):
        native.candidate_kernel()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gate_rejects_one_ulp_and_signed_zero_changes(dtype):
    expected = torch.ones(2, 2, dtype=dtype)
    assert assert_exact(expected.clone(), expected)["exact_match"]
    changed = expected.clone()
    changed[0, 0] = torch.nextafter(changed[0, 0], torch.tensor(float("inf"), dtype=dtype))
    with pytest.raises(AssertionError):
        assert_exact(changed, expected)
    with pytest.raises(AssertionError, match="bits"):
        assert_exact(torch.tensor([-0.], dtype=dtype), torch.tensor([0.], dtype=dtype))
    with pytest.raises(AssertionError, match="Nonfinite"):
        assert_exact(torch.tensor([float("nan")], dtype=dtype), torch.ones(1, dtype=dtype))


def test_paired_median_is_not_difference_of_medians():
    row = bench.paired_timings([1., 10., 11.], [2., 3., 12.])
    assert row["paired_saved_ms"] == [-1., 7., -1.]
    assert row["median_saved_ms"] == -1.
    assert row["faster_trials"] == 1 and row["trials"] == 3


@pytest.mark.parametrize("a,b", [([], []), ([1], [1, 2]), ([0], [1]),
                                    ([1], [float("nan")]), ([float("inf")], [1])])
def test_invalid_paired_timings(a, b):
    with pytest.raises(ValueError):
        bench.paired_timings(a, b)


@pytest.mark.parametrize("dim", [64, 128])
def test_build_audit_identifies_binaries_and_does_not_mislabel_d64(tmp_path, dim):
    (tmp_path / "build.ninja").write_text("nvcc = /cuda/nvcc\ncuda_cflags = -O3\nrule ignored\n")
    jit = tmp_path / "jit.so"; jit.write_bytes(b"jit")
    installed = tmp_path / "installed.so"; installed.write_bytes(b"installed")
    queried = []
    def resources(dtype):
        queried.append(dtype)
        return {"expcast_sage_layout_35": {"registers_per_thread": 168}}
    vc = SimpleNamespace(_extension=lambda: SimpleNamespace(__file__=installed), kernel_resources=resources)
    r = native.build_audit(SimpleNamespace(__file__=jit), vc,
                           {"generated_source": str(tmp_path / "kernel.cu")}, dim, 1)
    assert r["jit_binary"]["sha256"] != r["installed_binary"]["sha256"]
    assert r["jit_build_flags"] == ["nvcc = /cuda/nvcc", "cuda_cflags = -O3"]
    assert r["launch"]["dynamic_shared_bytes"] == 320 * dim
    assert queried == ([1] if dim == 128 else [])
    assert ("installed_resources" in r) == (dim == 128)
