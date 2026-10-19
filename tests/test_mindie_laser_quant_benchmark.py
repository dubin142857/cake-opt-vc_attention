"""Provider isolation, manifest evidence and timing semantics without NPU."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_mindie_laser_quant as benchmark


@pytest.fixture
def arguments(tmp_path):
    paths = {}
    for name in ("baseline", "quant"):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps({"vendor_root": str(tmp_path / name / "vendors"),
                                    "source_manifest": {"path": str(tmp_path / name / "source.json")},
                                    "bridge_path": str(tmp_path / (name + ".so"))}))
        paths[name + "_build"] = str(path)
    bridge = tmp_path / "bridge.json"
    bridge.write_text(json.dumps({"bridge_path": str(tmp_path / "baseline.so")}))
    return SimpleNamespace(**paths, preprocess_build=str(tmp_path / "preprocess.json"),
                           bridge_manifest=str(bridge), device=2, warmup=5,
                           repeats=10, trials=3, cpu_threads=4, _run_dir=str(tmp_path / "run"))


def test_each_algorithm_has_an_independent_worker_but_uses_the_correct_provider(arguments):
    rows = {name: benchmark.build_arguments(arguments, name) for name in benchmark.VARIANTS}
    assert len({row._run_dir for row in rows.values()}) == 6
    assert rows["qk_int8_fused"].preprocess_build == str(Path(arguments.preprocess_build).resolve())
    assert all(row.preprocess_build is None for name, row in rows.items() if name != "qk_int8_fused")
    assert rows["installed"]._worker == "installed"
    assert rows["baseline"].bridge_path == rows["installed"].bridge_path
    quant = [rows[name] for name in benchmark.VARIANTS[2:]]
    assert len({row.vendor_root for row in quant}) == 1
    assert len({row.bridge_path for row in quant}) == 1
    assert quant[0].bridge_path != rows["baseline"].bridge_path


def test_quant_environment_excludes_original_marker_and_uses_its_own_vendor(arguments, monkeypatch):
    monkeypatch.setenv(benchmark.baseline.MARKER_ENV, "stale-original")
    monkeypatch.setenv(benchmark.QUANT_MARKER_ENV, "stale-quant")
    quant = benchmark.worker_environment(arguments, "qk_int8", arguments._run_dir)
    original = benchmark.worker_environment(arguments, "installed", arguments._run_dir)
    assert benchmark.baseline.MARKER_ENV not in quant
    assert "qk_int8" in quant[benchmark.QUANT_MARKER_ENV]
    assert Path(quant["ASCEND_CUSTOM_OPP_PATH"]).parent.name == "quant"
    assert benchmark.QUANT_MARKER_ENV not in original
    assert "ASCEND_CUSTOM_OPP_PATH" not in original


@pytest.mark.parametrize("field,value", [("pid", 999), ("pid", True), ("token", "old-laser"),
                                         ("tiling_key", -1), ("phase", "api")])
def test_quant_marker_cannot_be_reused_from_wrong_provider_or_process(field, value):
    marker = {"pid": 3, "token": "quant", "phase": "tiling", "tiling_key": 128}
    assert benchmark.parse_quant_markers(json.dumps(marker), 3, "quant") == [marker]
    marker[field] = value
    with pytest.raises(ValueError):
        benchmark.parse_quant_markers(json.dumps(marker), 3, "quant")
    with pytest.raises(RuntimeError):
        benchmark.parse_quant_markers("", 3, "quant")


def test_timing_reports_preprocessing_directly_and_ratios_use_full_api():
    def row(preprocessing, full, prepared):
        return {"timing_ms": {"preprocessing": {"median": preprocessing}, "full": {"median": full},
                              "prepared_core": {"median": prepared}},
                "long_dense_fp32": {"relative_rmse": .02}}
    report = {"process_order": ["baseline", "qkv_int8_e4m3"],
              "workers": {"baseline": row(30, 100, 70), "qkv_int8_e4m3": row(40, 50, 10)}}
    text = benchmark.table(report)
    line = next(line for line in text.splitlines() if line.startswith("qkv_int8_e4m3 "))
    assert "2.0000x" in line and "7.0000x" not in line
    assert "40.0000" in line
    assert "full minus prepared is not used" in text
    assert "no native FP8" in text and "not paired/interleaved" in text


def test_quant_provenance_revalidates_source_and_rejects_changed_bridge(tmp_path, monkeypatch):
    import prepare_mindie_laser_quant_source as prepare
    calls = []
    monkeypatch.setattr(prepare, "validate_prepared_source", lambda *args: calls.append(args))
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    host, device, bridge = vendor / "host.so", vendor / "kernel.o", tmp_path / "bridge.so"
    for path in (host, device, bridge):
        path.write_bytes(path.name.encode())
    record = lambda path: {"path": str(path), "sha256": benchmark.common.sha256_file(path)}
    source_path, build_path = tmp_path / "source.json", tmp_path / "build.json"
    identity = {"native_api": benchmark.QUANT_API, "op_type": benchmark.QUANT_OP}
    source_path.write_text(json.dumps(dict(identity, source_dir=str(tmp_path / "source"),
        marker_token="quant", upstream={"commit": benchmark.baseline.UPSTREAM_COMMIT,
                                       "url": benchmark.baseline.SOURCE_LOCK.url})))
    build_path.write_text(json.dumps(dict(identity, status="passed", vendor_root=str(vendor), source_manifest=record(source_path),
        bridge_sha256=record(bridge)["sha256"], device_binaries=[record(device)], host_libraries=[record(host)])))
    args = SimpleNamespace(build_manifest=str(build_path), source_manifest=str(source_path),
                           vendor_root=str(vendor), bridge_path=str(bridge))
    evidence = benchmark.quant_provenance(args)
    assert calls == [(tmp_path / "source", source_path)]
    assert evidence["bridge_sha256"] == record(bridge)["sha256"]
    bridge.write_bytes(b"changed native provider")
    with pytest.raises(ValueError, match="bridge SHA256"):
        benchmark.quant_provenance(args)


def test_ablation_identity_checks_exact_qk_and_probability_inputs():
    fields = ("q", "k", "v", "q_scale", "k_scale", "v_scale")
    base = {field: str(index) * 64 for index, field in enumerate(fields)}
    workers = {name: {"prepared_input_sha256": dict(base)} for name in benchmark.VARIANTS[2:]}
    workers["qk_int8"]["prepared_input_sha256"]["v"] = "f" * 64
    workers["qk_int8"]["prepared_input_sha256"]["v_scale"] = "e" * 64
    checks = benchmark.verify_quant_payload_identity(workers)
    assert len(checks) == 2 and all(row["exact_sha256_match"] for row in checks)
    assert checks[0]["fields"] == ["q", "k", "q_scale", "k_scale"]
    assert checks[1]["fields"] == list(fields)
    workers["qkv_int8_e4m3_expcast"]["prepared_input_sha256"]["k_scale"] = "d" * 64
    with pytest.raises(AssertionError, match="k_scale"):
        benchmark.verify_quant_payload_identity(workers)


def test_ablation_identity_cannot_pass_with_missing_hashes():
    with pytest.raises(AssertionError, match="SHA256"):
        benchmark.verify_quant_payload_identity({"qk_int8": {}, "qkv_int8_e4m3": {}})
    assert benchmark.verify_quant_payload_identity({"qk_int8": {}}) == []


@pytest.mark.parametrize("extra,expected", [
    ([], ["installed", "baseline", "qk_int8", "qk_int8_fused"]),
    (["--variants", *benchmark.VARIANTS], list(benchmark.VARIANTS)),
])
def test_cli_defaults_to_qk_mainline_and_accepts_all_experiments(monkeypatch, extra, expected):
    monkeypatch.setattr(benchmark, "parent", lambda args: args)
    args = benchmark.main(extra)
    assert args.variants == expected
    assert args.output == "results/mindie_laser_quant_compare.json"
    assert args.preprocess_build == "results/mindie_laser_quant_preprocess_build.json"


@pytest.mark.parametrize("extra", [["--shape", "1", "2", "0", "128"], ["--trials", "1"],
                                    ["--_worker", "qk_int8"], ["--variants", "qk_int8", "qk_int8"]])
def test_invalid_cli_fails_before_device_or_filesystem_work(extra):
    with pytest.raises(SystemExit) as exc:
        benchmark.main(extra)
    assert exc.value.code == 2


def test_component_timings_keep_primary_timings_and_store_actual_synchronized_calls(monkeypatch):
    events, saves = [], []
    fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: events.append("sync")))
    args = SimpleNamespace(warmup=1, repeats=2, trials=3)
    clock = iter([0., .002, 1., 1.004, 2., 2.006])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    report = {"mode": "qk_int8_fused", "timing_ms": {"full": {"median": 101.}}}
    benchmark.benchmark_calls({"native_api": lambda: events.append("native")}, args,
                              fake_torch, report, lambda: saves.append(True),
                              timing_key="component_timing_ms")
    assert report["timing_ms"] == {"full": {"median": 101.}}
    assert report["component_timing_ms"]["native_api"]["trials"] == pytest.approx([1., 2., 3.])
    assert report["component_timing_ms"]["native_api"]["median"] == pytest.approx(2.)
    assert events == ["native", "sync", "sync", "native", "native", "sync",
                      "sync", "native", "native", "sync", "sync", "native", "native", "sync"]
    assert len(saves) == 3


@pytest.mark.parametrize("fail_postprocess", [False, True])
def test_large_raw_output_is_retained_only_for_postprocess_measurement(monkeypatch, fail_postprocess):
    alive, scopes = [], []

    class Raw:
        def __init__(self):
            alive.append(1)
        def __del__(self):
            alive.pop()

    class Adapter:
        def quantize_qk(self, q, k):
            assert (q, k) == ("query", "key") and not alive
        def run_native(self, payload):
            assert payload == "fixed-payload" and not alive
            return Raw()
        def postprocess(self, raw, payload):
            assert isinstance(raw, Raw) and alive == [1] and payload == "fixed-payload"
            if fail_postprocess:
                raise RuntimeError("device failed")

    def measured(calls, args, torch, report, save, *, timing_key):
        assert timing_key == "component_timing_ms"
        assert report["timing_ms"]["full"]["median"] == 99.
        scopes.extend(calls)
        for name, call in calls.items():
            call()
            report.setdefault(timing_key, {})[name] = {"median": 1.}

    monkeypatch.setattr(benchmark, "benchmark_calls", measured)
    fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
    report = {"timing_ms": {"full": {"median": 99.}}}
    call = lambda: benchmark.benchmark_quant_components(
        Adapter(), ["query", "key", "value"], "fixed-payload", None, fake_torch, report, lambda: None)
    if fail_postprocess:
        with pytest.raises(RuntimeError, match="device failed"):
            call()
    else:
        call()
        assert set(report["component_timing_ms"]) == {"qk_quantize", "native_api", "output_postprocess"}
    assert scopes == ["qk_quantize", "native_api", "output_postprocess"]
    assert alive == []
    assert "not additive" in report["component_timing_scope"]["limits"]


def test_component_table_keeps_quantize_native_and_postprocess_distinct():
    timing = {name: {"median": value} for name, value in (
        ("preprocessing", 16.), ("full", 107.), ("prepared_core", 90.))}
    report = {"process_order": ["qk_int8_fused"], "workers": {"qk_int8_fused": {
        "timing_ms": timing, "long_dense_fp32": {"relative_rmse": .01},
        "component_timing_ms": {name: {"median": value} for name, value in (
            ("qk_quantize", 12.), ("native_api", 81.), ("output_postprocess", 8.))}}}}
    text = benchmark.table(report)
    component_line = [line for line in text.splitlines() if line.startswith("qk_int8_fused ")][1]
    assert component_line.split()[1:] == ["12.0000", "81.0000", "8.0000"]
    assert "not additive or kernel-only" in text


def test_fused_provider_requires_its_own_library_in_actual_worker_mappings(tmp_path, monkeypatch):
    import mindie_laser_quant_fused as fused
    library = tmp_path / "libqk.so"
    library.write_bytes(b"quantizer")
    manifest = tmp_path / "preprocess.json"
    manifest.write_text(json.dumps({"library_path": str(library)}))
    checks = []
    monkeypatch.setattr(fused, "validate_preprocess_manifest", lambda path: checks.append(path) or {"passed": True})
    args = SimpleNamespace(preprocess_build=str(manifest))
    report = {"mapped_library_paths": [], "loaded_library_sha256": {}}
    with pytest.raises(RuntimeError, match="absent from process"):
        benchmark.inspect_fused_preprocessor(args, report)
    report["mapped_library_paths"] = [str(library.resolve())]
    benchmark.inspect_fused_preprocessor(args, report)
    assert report["loaded_library_sha256"][str(library.resolve())] == benchmark.common.sha256_file(library)
    assert report["preprocessor_provider_evidence"]["manifest"] == {"passed": True}
    assert checks == [str(manifest), str(manifest)]
