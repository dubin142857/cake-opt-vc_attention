"""CPU-only isolation, transform, contract and evidence tests; no NPU validation."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_mindie_laser_local as benchmark
import mindie_laser_backend as backend


def reference_transforms():
    class Reference:
        @staticmethod
        def la_preprocess_input(q, k, v):
            return tuple(functional.pad(x.half(), (0, (-x.shape[-1]) % 128, 0, (-x.shape[-2]) % 256)).contiguous()
                         for x in (q, k, v))
        @staticmethod
        def la_postprocess_output(output, dtype, nq, dim):
            converted = output if output.dtype == dtype else output.half().to(dtype)
            return converted[:, :, :nq, :dim]
    return Reference


def test_local_transforms_match_independent_padding_and_conversion_reference():
    result = benchmark.verify_transforms(benchmark.laser_transforms(torch), reference_transforms(), torch)
    assert result["passed"] and result["cpu_cases"] == 4


def test_transform_verifier_rejects_changed_roundtrip():
    changed = reference_transforms()
    changed.la_postprocess_output = lambda out, dtype, nq, dim: out.to(dtype)[:, :, :nq, :dim]
    with pytest.raises(AssertionError, match="postprocess"):
        benchmark.verify_transforms(changed, benchmark.laser_transforms(torch), torch)


def test_source_helper_loader_does_not_execute_package_or_module_imports(tmp_path):
    path = tmp_path / "wrapper.py"
    path.write_text('''import missing_mindiesd_dependency
raise AssertionError("module level must not execute")
class AscendLaserAttention:
    @staticmethod
    def pad(x, base=256, dim=-2):
        return x
    @staticmethod
    def la_preprocess_input(q, k, v):
        return q, k, v
    @staticmethod
    def la_postprocess_output(out, dtype, nq, dim):
        return out[:, :, :nq, :dim]
    @staticmethod
    def unrelated():
        raise AssertionError("unrelated method")
''', encoding="utf-8")
    loaded = benchmark.source_transforms(path, torch)
    tensor = torch.ones((1, 1, 2, 4))
    assert loaded.la_preprocess_input(tensor, tensor, tensor)[0] is tensor
    assert not hasattr(loaded, "unrelated")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("nq,nk,dim,tail", [(17, 256, 128, 2147483647), (33, 257, 64, 255)])
def test_local_adapter_contract_uses_second_output_original_scale_and_key_tail(monkeypatch, dtype, nq, nk, dim, tail):
    monkeypatch.setattr(backend, "_require_npu", lambda tensor: None)
    calls = []
    def native(q, k, v, *args):
        calls.append(args)
        mask, alibi, drop, scale, heads, layout, keep, pre, next_tokens, precise = args
        assert (mask, alibi, drop) == (None, None, None)
        assert (heads, layout, keep, pre, next_tokens, precise) == (2, "BNSD", 1., tail, 1, True)
        assert scale == pytest.approx(dim ** -.5)
        valid = k.shape[2] - (0 if pre == 2147483647 else pre)
        output = torch.softmax((q.float() @ k[:, :, :valid].float().transpose(-1, -2)) * scale, -1) @ v[:, :, :valid].float()
        return torch.full(q.shape[:-1], float("nan")), output
    transforms = benchmark.laser_transforms(torch)
    adapter = backend.MindIELaser(transforms.la_preprocess_input, transforms.la_postprocess_output, native, {})
    generator = torch.Generator().manual_seed(29)
    q, k, v = [torch.randn((1, 2, n, dim), generator=generator).to(dtype) for n in (nq, nk, nk)]
    payload = adapter.prepare(q, k, v)
    assert payload.pre_tokens == tail and all(x.dtype == torch.float16 for x in payload.tensors)
    result = adapter.run_prepared(payload)
    assert tuple(result.shape) == tuple(q.shape) and result.dtype == dtype and result.is_contiguous()
    expected = torch.softmax((q.half().float() @ k.half().float().transpose(-1, -2)) * dim ** -.5, -1) @ v.half().float()
    torch.testing.assert_close(result, expected.half().to(dtype), rtol=.004, atol=.002)
    torch.testing.assert_close(adapter(q, k, torch.zeros_like(v)), torch.zeros_like(result), rtol=0, atol=0)
    adapter.native = lambda *args: (result,)
    with pytest.raises(RuntimeError, match="must return"):
        adapter.run_prepared(payload)


def test_provider_environments_are_clean_before_python_start(tmp_path):
    vendor, foreign = tmp_path / "vendor", tmp_path / "foreign"
    base = {"ASCEND_CUSTOM_OPP_PATH": str(foreign), "LD_PRELOAD": "/foreign/inject.so",
            "LD_LIBRARY_PATH": os.pathsep.join((str(foreign / "lib"), "/system/lib")),
            benchmark.MARKER_ENV: "/old/marker", "VC_CANN_FAS_MARKER_FILE": "/old/cannmarker"}
    installed = benchmark.worker_environment(base, "installed", vendor, tmp_path)
    local = benchmark.worker_environment(base, "local", vendor, tmp_path)
    assert "ASCEND_CUSTOM_OPP_PATH" not in installed and benchmark.MARKER_ENV not in installed
    assert local["ASCEND_CUSTOM_OPP_PATH"] == str(vendor)
    assert local[benchmark.MARKER_ENV] == str(tmp_path / "local" / "markers.jsonl")
    assert all("LD_PRELOAD" not in env and "VC_CANN_FAS_MARKER_FILE" not in env for env in (installed, local))
    assert local["ACL_OP_COMPILER_CACHE_MODE"] == installed["ACL_OP_COMPILER_CACHE_MODE"] == "disable"
    assert local["ACL_OP_COMPILER_CACHE_DIR"] != installed["ACL_OP_COMPILER_CACHE_DIR"]


def test_marker_requires_current_process_unique_token_and_success_phase():
    valid = {"token": benchmark.MARKER_TOKEN, "phase": "tiling", "pid": 47, "tiling_key": 18446744073709551615}
    assert benchmark.parse_markers(json.dumps(valid), 47) == [valid]
    with pytest.raises(RuntimeError, match="no fresh"):
        benchmark.parse_markers("", 47)
    for override in ({"token": "cann"}, {"pid": 48}, {"phase": "api"}, {"tiling_key": True}, {"tiling_key": -1}):
        with pytest.raises(ValueError):
            benchmark.parse_markers(json.dumps({**valid, **override}), 47)


def test_native_schema_rejects_changed_arity_or_output_contract():
    names = ["query", "key", "value", "atten_mask", "alibi_mask", "drop_mask", "scale_value", "head_num",
             "input_layout", "keep_prob", "pre_tokens", "next_tokens", "is_highPrecision"]
    schema = SimpleNamespace(arguments=[SimpleNamespace(name=name) for name in names], returns=[1, 2])
    benchmark.verify_schema(schema)
    schema.returns = [1]
    with pytest.raises(ValueError, match="schema"):
        benchmark.verify_schema(schema)


def make_provenance(tmp_path):
    vendor = tmp_path / "vendor"
    files = [vendor / "op_api/lib/libcust_opapi.so", vendor / "op_impl/ai_core/tbe/op_tiling/liboptiling.so",
             vendor / "op_impl/ai_core/tbe/kernel/vc_laser_attention/device.o"]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(path).encode())
    bridge = tmp_path / "libvc_laser_bridge.so"
    bridge.write_bytes(b"unique bridge")
    source = {"upstream": {"commit": benchmark.UPSTREAM_COMMIT, "url": benchmark.SOURCE_LOCK.url}, "marker_token": benchmark.MARKER_TOKEN,
              "native_api": benchmark.NATIVE_API, "op_type": benchmark.OP_TYPE,
              "kernel_inverse_rename_equivalent": True, "source_dir": str(tmp_path / "source")}
    source_path, build_path = tmp_path / "source.json", tmp_path / "build.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    entry = lambda path: {"path": str(path), "sha256": benchmark.common.sha256_file(path)}
    build = {"status": "passed", "native_api": benchmark.NATIVE_API, "op_type": benchmark.OP_TYPE, "vendor_root": str(vendor),
             "source_manifest": entry(source_path), "device_binaries": [entry(files[-1])], "host_libraries": list(map(entry, files[:2]))}
    build_path.write_text(json.dumps(build), encoding="utf-8")
    args = argparse.Namespace(source_manifest=str(source_path), build_manifest=str(build_path), vendor_root=str(vendor),
                              bridge_path=str(bridge), bridge_manifest=None)
    return args, files, source, build


def test_provenance_checks_native_identity_and_actual_binary_hashes(tmp_path):
    args, files, source, build = make_provenance(tmp_path)
    report = benchmark.provenance(args)
    assert len(report["device_binary_sha256"]) == 1 and len(report["host_library_sha256"]) == 2
    files[-1].write_bytes(b"changed binary")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        benchmark.provenance(args)
    build["native_api"] = "aclnnLaserAttention"
    Path(args.build_manifest).write_text(json.dumps(build), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        benchmark.provenance(args)


def test_empty_or_escaping_binary_manifest_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        benchmark.checked_files([], tmp_path, "device_binaries")
    outside = tmp_path.parent / "not-the-vendor.o"
    with pytest.raises(ValueError, match="escapes"):
        benchmark.checked_files([{"path": str(outside), "sha256": "unused"}], tmp_path, "device_binaries")


def test_provider_proof_requires_local_libraries_and_rejects_installed_contamination(monkeypatch, tmp_path):
    args, files, _, _ = make_provenance(tmp_path)
    paths = [str(path.resolve()) for path in files[:2]] + [str(Path(args.bridge_path).resolve())]
    original_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **kw:
                        "mapped libraries" if path.as_posix() == "/proc/self/maps" else original_read(path, *a, **kw))
    # Exercise the proof independently of Linux maps syntax on a Windows test host.
    monkeypatch.setattr(benchmark.common, "mapped_libraries", lambda text: paths)
    report = {}
    benchmark.inspect_provider("local", args.vendor_root, args.bridge_path, report, tmp_path)
    assert len(report["loaded_library_sha256"]) == 3
    with pytest.raises(RuntimeError, match="Installed worker loaded"):
        benchmark.inspect_provider("installed", args.vendor_root, args.bridge_path, {}, tmp_path)
    monkeypatch.setitem(sys.modules, "mindiesd", SimpleNamespace())
    with pytest.raises(RuntimeError, match="imported mindiesd"):
        benchmark.inspect_provider("local", args.vendor_root, args.bridge_path, {}, tmp_path)


def test_gate_records_failure_before_raising():
    report, snapshots = {}, []
    with pytest.raises(AssertionError, match="0.1%"):
        benchmark.record_gate(report, "same_algorithm", torch.full((4,), 1.01), torch.ones(4), .001,
                              lambda: snapshots.append(json.loads(json.dumps(report))))
    assert snapshots and snapshots[0]["same_algorithm"]["passed"] is False


def test_parent_provenance_error_is_saved_without_launching_workers(monkeypatch, tmp_path):
    args, _, _, _ = make_provenance(tmp_path)
    args.output = str(tmp_path / "result.json")
    Path(args.source_manifest).unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError("Workers must not start after invalid provenance")
    monkeypatch.setattr(benchmark.subprocess, "Popen", forbidden)
    assert benchmark.run_parent(args) == 1
    result = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert result["status"] == "failed" and "FileNotFoundError" in result["error"]
    assert result["workers"] == {}


def test_worker_commands_use_distinct_modes_and_default_measurement_contract(tmp_path):
    args = argparse.Namespace(vendor_root="vendor", bridge_path="bridge.so", source_manifest="source.json", build_manifest="build.json",
                              device=0, warmup=5, repeats=10, trials=3, cpu_threads=4)
    installed = benchmark.worker_command(args, "installed", tmp_path)
    local = benchmark.worker_command(args, "local", tmp_path)
    assert installed != local and installed[installed.index("--_worker") + 1] == "installed"
    assert local[local.index("--trials") + 1] == "3"


def test_comparison_table_reports_both_timing_scopes_accuracy_and_sample_limit():
    worker = {"timing_ms": {"full": {"median": 83.9254}, "prepared_core": {"median": 80.}},
              "long_dense_fp32": {"relative_rmse": .003}}
    report = {"workers": {"installed": worker, "local": worker},
              "speedup_installed_over_local": {"full": 1., "prepared_core": 1.},
              "local_vs_installed": {"full": {"relative_rmse": 0.}, "prepared_core": {"relative_rmse": 0.}}}
    table = benchmark.comparison_table(report)
    assert "installed" in table and "local" in table and "83.9254" in table and "80.0000" in table
    assert "0.300000%" in table and "1.0000x" in table
    assert "32 query rows" in table and "all 37741 keys" in table and "not an all-output" in table
