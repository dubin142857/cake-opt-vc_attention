"""CPU verification of independent oracle, validation gates, timing and launcher."""
import builtins
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def benchmark(monkeypatch):
    original_import = builtins.__import__

    def isolated(name, *args, **kwargs):
        if name.split(".")[0] in ("tilelang", "torch_npu", "tilelang_attention_npu", "vc_attention"):
            raise AssertionError("CPU benchmark utility imported a device backend: " + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", isolated)
    path = ROOT / "scripts/benchmark_tilelang_attention_npu.py"
    spec = importlib.util.spec_from_file_location("tilelang_npu_benchmark_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_defaults_are_plain_bf16_main_shape(benchmark):
    args = benchmark.parse_args([])
    assert args.shape == (1, 28, 37741, 128)
    assert args.dtype == "bfloat16" and args.key_length == 37741
    assert (args.warmup, args.repeats, args.trials) == (1, 5, 3)
    assert (args.target_trial_ms, args.timing_budget_seconds) == (200, 10)
    assert (args.block_m, args.block_n) == (128, 128)
    assert not args.include_mindie_laser
    assert args.output == Path("results/tilelang_attention_npu.json")
    assert set(benchmark.small_cases()) == {
        (pattern, q, k) for pattern in ("random", "zero_q", "constant_v") for q, k in ((65, 129), (129, 65))
    } | {("random", 1, 1), ("random", 129, 257), ("random", 128, 256),
         ("random", 129, 512), ("random", 129, 513)}


@pytest.mark.parametrize("argv", [
    ["--shape", "1", "2", "0", "128"], ["--shape", "1", "2", "129", "32"],
    ["--key-length", "0"], ["--device", "-1"], ["--warmup", "-1"],
    ["--repeats", "0"], ["--trials", "0"], ["--check-queries", "0"], ["--block-m", "32"],
    ["--block-m", "64", "--block-n", "128"],
    ["--target-trial-ms", "nan"], ["--target-trial-ms", "0"], ["--timing-budget-seconds", "inf"],
])
def test_invalid_inputs_fail_before_backend_import(benchmark, argv):
    with pytest.raises(SystemExit) as error:
        benchmark.parse_args(argv)
    assert error.value.code == 2


@pytest.mark.parametrize("block", [64, 128])
def test_cli_supports_current_tiles_and_explicit_64_baseline(benchmark, block):
    args = benchmark.parse_args(["--block-m", str(block), "--block-n", str(block)])
    assert (args.block_m, args.block_n) == (block, block)


@pytest.mark.parametrize("cores", [1, 20, 24])
def test_packet4_cases_then_one_persistent_case_use_actual_runtime_metadata(benchmark, cores):
    queried = []

    def actual_metadata():
        queried.append(True)
        return {"runtime": {"cube_core_num": cores}, "spec": {"block_m": 128}}

    cases = benchmark.validation_cases(actual_metadata)
    static = [next(cases) for _ in range(44)]
    assert queried == []
    for dtype in ("float16", "bfloat16"):
        for dim in (64, 128):
            for keys in (512, 513):
                assert {"dtype": dtype, "shape_bnsd": [1, 2, 129, dim], "keys": keys, "pattern": "random"} in static
    persistent = next(cases)
    assert len(queried) == 1
    assert persistent["shape_bnsd"] == [2, 3, (cores // 6) * 128 + 1, 128]
    assert persistent["dtype"] == "bfloat16" and persistent["keys"] == 513
    assert persistent["query_tiles"] > cores
    with pytest.raises(StopIteration):
        next(cases)


def test_small_kernel_object_cache_keeps_all_patterns_and_fresh_payloads(benchmark, monkeypatch, tmp_path):
    builds, payloads = [], []
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    # This test exercises orchestration, not attention arithmetic. Tiny tensors
    # keep the fake CPU kernel separate from the production NPU implementation.
    monkeypatch.setattr(benchmark, "make_cpu_inputs", lambda shape, keys, dtype, seed, pattern:
                        tuple(torch.zeros(1, 1, 1, 1, dtype=dtype) for _ in range(3)))
    monkeypatch.setattr(benchmark, "cpu_reference", lambda q, k, v: q.float())

    def build(batch, heads, queries, keys, dim, *, dtype, device, block_m, block_n):
        builds.append((batch, heads, queries, keys, dim, dtype, str(device), block_m, block_n))

        class FakeKernel:
            def metadata(self):
                return {"spec": {"block_m": block_m, "block_n": block_n},
                        "runtime": {"cube_core_num": 24}, "packet_size": 4}

            def export_source(self, directory):
                return {}

            def prepare(self, q, k, v):
                payload = SimpleNamespace(output=q.clone())
                payloads.append(payload)
                return payload

            def run_prepared(self, payload):
                return payload.output

            def __call__(self, q, k, v):
                return self.run_prepared(self.prepare(q, k, v))

        return FakeKernel()

    report = {"validation": {"small": []}, "kernel_sources": {}}
    benchmark.validate_small(torch, SimpleNamespace(build_attention=build),
                             {"comparison": lambda q, k, v: q.clone()}, benchmark.parse_args([]),
                             torch.device("cpu"), report, lambda: None, tmp_path)
    assert report["validation"]["small_cases_count"] == report["validation"]["small_cases_expected"] == 45
    assert len(builds) == len(set(builds)) == 29
    stats = report["validation"]["small_kernel_cache"]
    assert (stats["requests"], stats["unique_kernel_builds"], stats["object_reuses"]) == (45, 29, 16)
    assert len(payloads) == len({id(payload) for payload in payloads}) == 90
    assert "current_small_case" not in report["validation"]
    assert report["validation"]["small"][-1]["persistent_task_reuse"]
    assert all(row["packet_size"] == 4 for row in report["validation"]["small"])
    assert len(report["generated_source_audits"]) == 29
    assert all(row["status"] == "unavailable" for row in report["generated_source_audits"].values())


def test_small_compile_failure_retains_current_case_label(benchmark, monkeypatch, tmp_path):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    monkeypatch.setattr(benchmark, "make_cpu_inputs", lambda shape, keys, dtype, seed, pattern:
                        tuple(torch.zeros(1, 1, 1, 1, dtype=dtype) for _ in range(3)))
    monkeypatch.setattr(benchmark, "cpu_reference", lambda q, k, v: q.float())

    def failed_build(*args, **kwargs):
        raise RuntimeError("intentional compile failure")

    report = {"validation": {"small": []}, "kernel_sources": {}}
    with pytest.raises(RuntimeError, match="intentional compile failure"):
        benchmark.validate_small(torch, SimpleNamespace(build_attention=failed_build), {}, benchmark.parse_args([]),
                                 torch.device("cpu"), report, lambda: None, tmp_path)
    pending = report["validation"]["current_small_case"]
    assert pending["dtype"] == "float16" and pending["shape_bnsd"] == [1, 2, 65, 64]
    assert pending["keys"] == 129 and pending["pattern"] == "random"
    assert report["validation"]["small_cases_count"] == 0


def test_generated_sync_audit_counts_body_forms_and_ignores_noncode_and_includes(benchmark, tmp_path):
    path = tmp_path / "attention.cpp"
    (tmp_path / "sync_helper.h").write_text("AscendC::PipeBarrier<PIPE_ALL>();\n" * 100, encoding="utf-8")
    path.write_text(r'''
        #include "sync_helper.h"
        // AscendC::PipeBarrier<PIPE_ALL>(); SetFlag<HardEvent::MTE2_V>(0);
        /* pipe_barrier(PIPE_V); CrossCoreWaitFlag(0); */
        const char *text = "PipeBarrier<PIPE_ALL>(); WaitFlag<HardEvent::MTE2_V>(0);";
        const char *escaped = "\" pipe_barrier(PIPE_ALL); \"";
        const char *raw = R"audit(SyncAll<false>(); CrossCoreSetFlag<0x2, PIPE_FIX>(0);)audit";
        char quote = '\'';
        AscendC::PipeBarrier<PIPE_ALL>();
        PipeBarrier < AscendC::PIPE_V > ();
        pipe_barrier(PIPE_ALL);
        PIPE_BARRIER (PIPE_V);
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        SetFlag<HardEvent::V_MTE3>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::V_MTE3>(1);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0);
        AscendC::CrossCoreWaitFlag(0);
        AscendC::SyncAll<false>();
    ''', encoding="utf-8")
    audit = benchmark.audit_generated_source({"ascendc": str(path)})
    assert audit["status"] == "ok" and audit["source_bytes"] == path.stat().st_size
    assert len(audit["sha256"]) == 64 and "includes are not expanded" in audit["scope"]
    calls = audit["static_calls"]
    assert calls["pipe_barrier_by_pipe"] == {"PIPE_ALL": 2, "PIPE_V": 2, "PIPE_MTE2": 1}
    assert calls["set_flag_calls"] == calls["wait_flag_calls"] == 2
    assert calls["set_flag_by_hard_event"] == calls["wait_flag_by_hard_event"] == {"MTE2_V": 1, "V_MTE3": 1}
    assert calls["cross_core_set_flag_calls"] == calls["cross_core_wait_flag_calls"] == calls["sync_all_calls"] == 1


def test_generated_sync_audit_missing_empty_or_bad_source_is_diagnostic(benchmark, tmp_path, monkeypatch):
    for export in (None, {}, {"ascendc": str(tmp_path / "missing.cpp")}):
        assert benchmark.audit_generated_source(export)["status"] == "unavailable"
    empty = tmp_path / "empty.cpp"
    empty.write_bytes(b"")
    assert benchmark.audit_generated_source({"ascendc": str(empty)})["status"] == "empty_source"
    invalid = tmp_path / "invalid.cpp"
    invalid.write_bytes(b"\xff")
    audit = benchmark.audit_generated_source({"ascendc": str(invalid)})
    assert audit["status"] == "error" and audit["error"]["type"] == "UnicodeDecodeError"
    source = tmp_path / "valid.cpp"
    source.write_text("pipe_barrier(PIPE_ALL);", encoding="utf-8")

    def parser_failure(*args, **kwargs):
        raise RuntimeError("intentional lexical audit failure")

    monkeypatch.setattr(benchmark.re, "findall", parser_failure)
    audit = benchmark.audit_generated_source({"ascendc": str(source)})
    assert audit["status"] == "error" and audit["error"]["message"] == "intentional lexical audit failure"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [64, 128])
def test_zero_q_oracle_is_all_value_mean_and_constant_v_is_preserved(benchmark, dtype, dim):
    q, k, v = benchmark.make_cpu_inputs((1, 2, 65, dim), 129, dtype, 9, "zero_q")
    assert q.dtype == k.dtype == v.dtype == dtype
    expected = v.float().mean(dim=2, keepdim=True).expand(1, 2, 65, dim)
    torch.testing.assert_close(benchmark.cpu_reference(q, k, v), expected, rtol=1e-5, atol=1e-6)
    q, k, v = benchmark.make_cpu_inputs((1, 2, 129, dim), 65, dtype, 9, "constant_v")
    expected = v[:, :, :1].float().expand(1, 2, 129, dim)
    torch.testing.assert_close(benchmark.cpu_reference(q, k, v), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [64, 128])
def test_single_key_case_preserves_each_batch_and_head_value(benchmark, dtype, dim):
    q, k, v = benchmark.make_cpu_inputs((2, 3, 1, dim), 1, dtype, 19)
    assert q.shape == k.shape == v.shape == (2, 3, 1, dim)
    actual = benchmark.cpu_reference(q, k, v)
    torch.testing.assert_close(actual, v.float(), rtol=0, atol=0)
    assert not torch.equal(v[0, 0], v[1, 0])
    assert not torch.equal(v[0, 0], v[0, 1])


def test_sampled_oracle_checks_all_keys_with_bounded_score_blocks(benchmark, monkeypatch):
    q, k, v = benchmark.make_cpu_inputs((2, 3, 129, 64), 257, torch.bfloat16, 10)
    ids = benchmark.sample_query_ids(129, 8)
    assert {0, 1, 63, 64, 65, 127, 128}.issubset(ids)
    full = benchmark.cpu_reference(q, k, v)
    shapes = []
    original = torch.softmax

    def checked_softmax(scores, *args, **kwargs):
        shapes.append(tuple(scores.shape))
        assert scores.ndim == 2 and scores.shape[0] <= 2 and scores.shape[1] == 257
        return original(scores, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", checked_softmax)
    selected = benchmark.cpu_reference(q, k, v, ids, query_chunk=2)
    torch.testing.assert_close(selected, full.index_select(2, torch.tensor(ids)), rtol=1e-5, atol=1e-6)
    assert len(shapes) == 2 * 3 * 4


@pytest.mark.parametrize("length,count", [(1, 8), (2, 8), (65, 8), (129, 129), (37741, 8)])
def test_sample_queries_are_distinct_in_bounds_and_cover_ends(benchmark, length, count):
    ids = benchmark.sample_query_ids(length, count)
    assert len(ids) == min(length, count)
    assert ids == sorted(set(ids))
    assert ids[0] == 0 and ids[-1] == length - 1


def test_sample_queries_follow_actual_128_row_tile_boundary(benchmark):
    ids = benchmark.sample_query_ids(37741, 8, block_m=128)
    assert ids == [0, 1, 127, 128, 129, 18870, 37739, 37740]
    assert benchmark.sample_query_ids(37741, 8) == [0, 1, 63, 64, 65, 18870, 37739, 37740]


def test_nonfinite_and_inaccurate_outputs_are_saved_before_failure(benchmark):
    records, saved = [], []
    reference = torch.ones((1, 1, 2, 64))
    for output in (torch.full_like(reference, float("nan")), reference + 1):
        with pytest.raises(AssertionError, match="Accuracy validation failed"):
            benchmark.enforce_metrics(output, reference, "bfloat16", {"provider": "candidate"},
                                      records, lambda: saved.append(records[-1].copy()))
        assert records[-1]["status"] == "failed" and saved[-1] == records[-1]
        json.dumps(records[-1], allow_nan=False)
    with pytest.raises(ValueError, match="shape"):
        benchmark.error_metrics(reference[:, :, :1], reference, "float16")


def test_repeat_gate_catches_aliased_or_changed_results(benchmark):
    records = []
    original = torch.ones(1, 2, 65, 64, dtype=torch.bfloat16)
    first = original.clone()
    benchmark.exact_check(original, first, "repeat", {}, records, lambda: None)
    original[..., -1] = 2
    with pytest.raises(AssertionError, match="output changed"):
        benchmark.exact_check(original, first, "stream", {}, records, lambda: None)
    assert [row["status"] for row in records] == ["passed", "failed"]


def test_timing_syncs_each_call_and_reports_ms_without_kernel_claim(benchmark):
    operations = []
    ticks = iter(i * .002 for i in range(12))
    result = benchmark.synchronized_api_time(lambda: operations.append("call"),
                                            lambda: operations.append("sync"), 2, 3, 2,
                                            clock=lambda: next(ticks))
    assert operations[:4] == ["call", "sync", "call", "sync"]
    assert operations[4:] == ["sync", "call", "sync"] * 6
    assert result["median_ms"] == pytest.approx(2)
    assert len(result["trial_mean_ms"]) == 2
    assert "Python API wall time" in result["scope"] and "not device-kernel" in result["scope"]


def test_calibrated_plan_credits_validation_and_bounds_slow_provider_work(benchmark):
    slow = benchmark.timing_plan(40000, warmup=1, repeats_cap=5, trials=3,
                                 target_trial_ms=200, budget_seconds=10, verified_calls=2)
    assert (slow["extra_warmup"], slow["effective_repeats"], slow["effective_trials"]) == (0, 1, 1)
    assert slow["requested_trials"] == 3 and slow["validation_warmup_credit"] == 1
    assert slow["minimum_one_call_exceeds_budget"]
    assert slow["estimated_timing_and_extra_warmup_seconds"] == 40
    assert not slow["calibration_in_final_statistics"]
    fast = benchmark.timing_plan(1, warmup=1, repeats_cap=5, trials=3,
                                 target_trial_ms=200, budget_seconds=10)
    assert (fast["extra_warmup"], fast["effective_repeats"], fast["effective_trials"]) == (0, 5, 3)


def test_provider_timing_adds_no_calibration_or_duplicate_warmup_calls(benchmark):
    seconds = {"tilelang_full": 40.0, "tilelang_prepared": 39.5, "cann_fusion_attention": 0.128}
    now, called = [0.0], []

    def call(name):
        called.append(name)
        now[0] += seconds[name]

    report = {"validation": {"status": "passed"}, "providers": {}, "timings": {},
              "calibration": {name: {"ms": duration * 1000, "verified_calls": 1}
                              for name, duration in seconds.items()}}
    benchmark.time_providers({name: (lambda name=name: call(name), "test API") for name in seconds},
                             report, benchmark.parse_args([]), lambda: None, lambda: None, clock=lambda: now[0])
    assert called.count("tilelang_full") == called.count("tilelang_prepared") == 1
    assert called.count("cann_fusion_attention") == 3
    assert report["timings"]["tilelang_full"]["trial_mean_ms"] == [40000]
    assert report["timings"]["tilelang_full"]["plan"]["effective_trials"] == 1
    assert report["timings"]["tilelang_full"]["calibration_in_final_statistics"] is False
    assert report["comparisons"]["tilelang_full"]["tilelang_over_cann_time_ratio"] == pytest.approx(312.5)
    assert report["comparisons"]["tilelang_prepared"]["tilelang_over_cann_time_ratio"] == pytest.approx(39.5 / .128)
    json.dumps(report, allow_nan=False)


def test_provider_timing_cannot_run_after_failed_validation(benchmark):
    called = []
    with pytest.raises(RuntimeError, match="validation must pass"):
        benchmark.time_providers({"candidate": (lambda: called.append(True), "test API")},
                                 {"validation": {"status": "failed"}}, benchmark.parse_args([]),
                                 lambda: None, lambda: None)
    assert called == []


def test_main_writes_failed_json_and_nonzero_without_device(benchmark, tmp_path, monkeypatch):
    output = tmp_path / "failure.json"

    def failing_run(args, report, save, artifact_dir):
        raise RuntimeError("intentional validation failure before timing")

    monkeypatch.setattr(benchmark, "run", failing_run)
    result = benchmark.main(["--output", str(output), "--artifact-dir", str(tmp_path / "artifacts")])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert result == 1 and report["status"] == "failed"
    assert report["validation"]["status"] == "failed" and report["timings"] == {}
    assert report["error"]["message"] == "intentional validation failure before timing"
    assert report["source_archive"]["benchmark_tilelang_attention_npu.py"]["sha256"]


@pytest.mark.parametrize("layout", ["arm64-linux", "aarch64-linux", "x86_64-linux"])
def test_runtime_cache_fingerprints_cann_layouts_and_version_changes(benchmark, tmp_path, monkeypatch, layout):
    import ctypes

    def soc():
        return b"Ascend910B3"

    monkeypatch.setattr(ctypes, "CDLL", lambda name: SimpleNamespace(aclrtGetSocName=soc))
    monkeypatch.setattr(benchmark.shutil, "which", lambda executable: None)
    monkeypatch.setattr(benchmark, "_version", lambda name: "test-version")
    cann = tmp_path / "cann"
    manifest = cann / layout / "ascend_toolkit_install.info"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("version=8.3.RC1", encoding="utf-8")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann))
    monkeypatch.setenv("TILELANG_ATTENTION_CACHE_ROOT", str(tmp_path / "cache"))
    torch_info = SimpleNamespace(__version__="2.6.0")
    npu_info = SimpleNamespace(__version__="2.6.0.RC1")
    first = benchmark.isolate_runtime_cache(torch_info, npu_info)
    assert str(manifest.resolve()) in first["identity"]["cann_manifests"]
    assert "Ascend910B3_1d7525399141_" in first["directory"]
    assert benchmark.isolate_runtime_cache(torch_info, npu_info)["key"] == first["key"]
    manifest.write_text("version=9.0.1", encoding="utf-8")
    second = benchmark.isolate_runtime_cache(torch_info, npu_info)
    assert second["key"] != first["key"]
    assert os.environ["TILELANG_CACHE_DIR"].startswith(second["directory"])


def test_runtime_cache_uses_compiler_hash_when_manifests_are_absent(benchmark, tmp_path, monkeypatch):
    import ctypes

    def soc():
        return b"Ascend910_9382"

    monkeypatch.setattr(ctypes, "CDLL", lambda name: SimpleNamespace(aclrtGetSocName=soc))
    monkeypatch.setattr(benchmark.shutil, "which", lambda executable: None)
    monkeypatch.setattr(benchmark, "_version", lambda name: "test-version")
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="bisheng compiler version 9.0.1", stderr=""))
    cann = tmp_path / "cann"
    compiler = cann / "compiler/ccec_compiler/bin/bisheng"
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes(b"compiler build 1")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann))
    monkeypatch.setenv("TILELANG_ATTENTION_CACHE_ROOT", str(tmp_path / "cache"))
    info = SimpleNamespace(__version__="2.6.0")
    first = benchmark.isolate_runtime_cache(info, info)
    assert first["identity"]["cann_manifests"] == {}
    assert first["identity"]["compiler"]["path"] == str(compiler.resolve())
    assert first["identity"]["compiler"]["version_output"] == "bisheng compiler version 9.0.1"
    compiler.write_bytes(b"compiler build 2")
    assert benchmark.isolate_runtime_cache(info, info)["key"] != first["key"]


def test_one_bash_entry_sources_setup_and_preserves_isolated_paths(benchmark, tmp_path):
    git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
    bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
    if not bash:
        pytest.skip("Bash required to exercise launcher")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    launcher = scripts / "run_tilelang_attention_npu.sh"
    shutil.copy2(ROOT / "scripts/run_tilelang_attention_npu.sh", launcher)
    (scripts / "setup_tilelang_attention_npu.sh").write_text(
        ': "${TILELANG_NPU_PYTHON:?}"\nexport TILELANG_NPU_PYTHON\n', encoding="utf-8", newline="\n")
    fake_python = tmp_path / "prepared python.sh"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "LAUNCH_ARG:%s\\n" "$@"\n'
                           'printf "CACHE_ROOT:%s\\n" "$TILELANG_ATTENTION_CACHE_ROOT"\n'
                           'printf "RUN_MODE:%s\\n" "$TL_RUN_MODE"\n',
                           encoding="utf-8", newline="\n")
    fake_python.chmod(fake_python.stat().st_mode | 0o111)
    env = dict(os.environ, TILELANG_NPU_PYTHON=fake_python.as_posix(), ASCEND_HOME_PATH=tmp_path.as_posix(), NPU_DEVICE="2")
    process = subprocess.run([bash, launcher.as_posix()], env=env, cwd=tmp_path, capture_output=True,
                             text=True, encoding="utf-8", timeout=30)
    assert process.returncode == 0, process.stderr
    argv = [line.removeprefix("LAUNCH_ARG:") for line in process.stdout.splitlines() if line.startswith("LAUNCH_ARG:")]
    assert argv[:2] == ["-u", "scripts/benchmark_tilelang_attention_npu.py"]
    args = benchmark.parse_args(argv[2:])
    assert args.device == 2 and args.shape == benchmark.DEFAULT_SHAPE and args.dtype == "bfloat16"
    assert (args.block_m, args.block_n) == (128, 128)
    cache = [line for line in process.stdout.splitlines() if line.startswith("CACHE_ROOT:")][0]
    assert cache.endswith("/.cache/tilelang-attention-npu/runtime")
    assert "RUN_MODE:npu" in process.stdout.splitlines()
