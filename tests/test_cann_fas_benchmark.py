"""CPU-only tests of benchmark isolation/evidence; no claim of NPU validation."""
import argparse
import json
import os
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_cann_fas_local as benchmark


def manifest():
    return {"upstream": {"url": benchmark.SOURCE_LOCK.url,
                         "tag": "v9.0.1", "commit": benchmark.UPSTREAM_COMMIT},
            "marker_token": benchmark.MARKER_TOKEN, "kernel_source_unchanged": True,
            "source_dir": "source", "third_party_dir": "third_party"}


@pytest.mark.parametrize("change", [
    {"marker_token": "wrong"}, {"kernel_source_unchanged": False}, {"source_dir": ""},
    {"upstream": {"url": "https://example.com/ops-transformer", "tag": "v9.0.1", "commit": benchmark.UPSTREAM_COMMIT}},
    {"upstream": {"url": "https://github.com/Ascend/ops-transformer", "tag": "master", "commit": "bad"}},
])
def test_manifest_rejects_unpinned_or_unattested_sources(change):
    with pytest.raises(ValueError):
        benchmark.validate_manifest({**manifest(), **change})


def test_source_hashes_and_version_claim(tmp_path):
    source = tmp_path / "source" / "attention" / "flash_attention_score" / "op_kernel"
    source.mkdir(parents=True)
    kernel = source / "kernel.h"
    kernel.write_text("original kernel", encoding="utf-8")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()), encoding="utf-8")
    evidence = benchmark.source_evidence(path)
    assert evidence["manifest_sha256"] == benchmark.sha256_file(path)
    assert evidence["source_file_sha256"][str(kernel.relative_to(tmp_path / "source"))] == benchmark.sha256_file(kernel)
    assert "does not assert byte identity" in evidence["version_scope"]


def test_environments_remove_inherited_vendors_without_losing_system_libraries(tmp_path):
    vendor, inherited = tmp_path / "vendors" / "ours", tmp_path / "other_vendor"
    builtin = tmp_path / "toolkit" / "opp" / "built-in" / "op_impl" / "op_tiling" / "lib"
    toolkit = tmp_path / "toolkit" / "lib64"
    base = {"ASCEND_CUSTOM_OPP_PATH": str(inherited), "LD_PRELOAD": "/bad/inject.so",
            "LD_AUDIT": "/bad/audit.so", "VC_CANN_FAS_MARKER_FILE": "/stale/marker",
            "LD_LIBRARY_PATH": os.pathsep.join(map(str, (inherited / "lib", vendor / "op_api" / "lib", toolkit, builtin))),
            "ACLNN_EXTENSION_PATH": "/bad/extension", "OTHER_SETTING": "preserved"}
    system = benchmark.worker_environment(base, "system", vendor, tmp_path)
    assert not any(name in system for name in ("ASCEND_CUSTOM_OPP_PATH", "LD_PRELOAD", "LD_AUDIT", "VC_CANN_FAS_MARKER_FILE", "ACLNN_EXTENSION_PATH"))
    assert system["LD_LIBRARY_PATH"] == os.pathsep.join(map(str, (toolkit, builtin)))
    assert system["OTHER_SETTING"] == "preserved"
    assert system["ACL_OP_COMPILER_CACHE_MODE"] == "disable"
    custom = benchmark.worker_environment(base, "custom", vendor, tmp_path)
    assert custom["ASCEND_CUSTOM_OPP_PATH"] == str(vendor)
    assert custom["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(vendor / "op_api" / "lib")
    assert custom["VC_CANN_FAS_MARKER_FILE"].endswith("markers.jsonl")
    assert custom["ASCEND_CACHE_PATH"] != system["ASCEND_CACHE_PATH"]
    assert base["LD_PRELOAD"] == "/bad/inject.so"  # Caller environment is not mutated.


def marker(phase, **kwargs):
    return {"token": benchmark.MARKER_TOKEN, "phase": phase, "pid": 123, "tiling_key": 17 if phase == "tiling" else 0, **kwargs}


def test_markers_require_fresh_api_and_tiling_with_same_process_and_token():
    text = "\n".join(json.dumps(item) for item in (marker("api"), marker("tiling")))
    assert benchmark.parse_markers(text, 123)[1]["tiling_key"] == 17
    for records in ([], [marker("api")], [marker("tiling")]):
        with pytest.raises(RuntimeError, match="both fresh"):
            benchmark.parse_markers("\n".join(map(json.dumps, records)), 123)
    for changes in ({"token": "foreign"}, {"pid": 999}, {"tiling_key": -1}, {"tiling_key": True}):
        with pytest.raises(ValueError):
            benchmark.parse_markers(json.dumps(marker("api", **changes)) + "\n" + json.dumps(marker("tiling")), 123)


def test_system_rejects_global_vendor_and_custom_requires_both_libraries(tmp_path):
    vendor = tmp_path / "vendors" / "ours"
    api = str(vendor / "op_api" / "lib" / "libcust_opapi.so")
    tiling = str(vendor / "op_impl" / "ai_core" / "tbe" / "op_tiling" / "lib" / "liboptiling.so")
    builtin = str(tmp_path / "toolkit" / "opp" / "built-in" / "op_impl" / "op_tiling" / "lib" / "liboptiling.so")
    assert benchmark.verify_libraries([builtin], "system", vendor) == [builtin]
    with pytest.raises(RuntimeError, match="contamination"):
        benchmark.verify_libraries([builtin, api], "system", vendor)
    with pytest.raises(RuntimeError, match="requires"):
        benchmark.verify_libraries([builtin, api], "custom", vendor)
    assert benchmark.verify_libraries([builtin, api, tiling], "custom", vendor) == [builtin, api, tiling]
    foreign = str(tmp_path / "vendors" / "foreign" / "op_tiling" / "libforeign.so")
    with pytest.raises(RuntimeError, match="contamination"):
        benchmark.verify_libraries([api, tiling, foreign], "custom", vendor)


def test_map_parser_deduplicates_library_mappings_and_preserves_spaces():
    text = "\n".join(("000-100 r-xp 000 00:00 1 /opt/cann libs/libopapi.so",
                       "100-200 rw-p 000 00:00 1 /opt/cann libs/libopapi.so",
                       "200-300 rw-p 000 00:00 0 [heap]"))
    assert benchmark.mapped_libraries(text) == [str(Path("/opt/cann libs/libopapi.so").resolve())]


def test_dense_reference_samples_only_queries_and_uses_all_keys():
    generator = torch.Generator().manual_seed(9)
    q = torch.randn((1, 2, 7, 4), generator=generator).bfloat16()
    k = torch.randn((1, 2, 13, 4), generator=generator).bfloat16()
    v = torch.randn((1, 2, 13, 4), generator=generator).bfloat16()
    ids = torch.tensor([0, 3, 6])
    full = torch.softmax(q.float() @ k.float().transpose(-1, -2) / 2, -1) @ v.float()
    result = benchmark.dense_reference(q, k, v, ids, chunk=2)
    torch.testing.assert_close(result, full.index_select(2, ids))
    assert result.shape == (1, 2, 3, 4)


def test_same_algorithm_gate_rejects_changes_and_input_mismatch(tmp_path):
    ids = torch.tensor([0, 3])
    payload = {"output": torch.ones((1, 2, 2, 4)), "query_ids": ids,
               "shape_bnsd": list(benchmark.SHAPE), "seed": 0, "input_sha256": ["q", "k", "v"]}
    system, custom = tmp_path / "system.pt", tmp_path / "custom.pt"
    torch.save(payload, system)
    torch.save(payload, custom)
    result = benchmark.compare_samples(system, custom)
    assert result["passed"] and result["exact_match"]
    torch.save({**payload, "output": payload["output"] + .002}, custom)
    assert not benchmark.compare_samples(system, custom)["passed"]
    torch.save({**payload, "input_sha256": ["wrong", "k", "v"]}, custom)
    with pytest.raises(ValueError, match="input mismatch"):
        benchmark.compare_samples(system, custom)
    with pytest.raises(ValueError, match="non-finite"):
        benchmark.error_metrics(torch.tensor([float("nan")]), torch.ones(1))


def test_worker_command_has_distinct_provider_and_safe_shell_arguments(tmp_path):
    args = argparse.Namespace(vendor_root=str(tmp_path / "vendor"), source_manifest="manifest.json",
                              device=0, warmup=2, repeats=3, trials=5, cpu_threads=4, cann_env_script=None)
    system = benchmark.worker_command(args, "system", tmp_path)
    custom = benchmark.worker_command(args, "custom", tmp_path)
    assert system != custom and "--_env-ready" in system
    args.cann_env_script = '/tmp/CANN environment;$(literal)/set_env.sh'
    sourced = benchmark.worker_command(args, "custom", tmp_path)
    assert sourced[:4] == ["bash", "--noprofile", "--norc", "-c"]
    assert args.cann_env_script not in sourced[4]
    assert args.cann_env_script in sourced
    assert "--_env-ready" not in sourced  # Worker execs again after cleaning sourced LD variables.


def test_sourced_environment_is_cleaned_by_exec_before_worker_imports(monkeypatch, tmp_path):
    captured = {}
    class ExecReached(Exception):
        pass
    def fake_exec(executable, argv, env):
        captured.update(executable=executable, argv=argv, env=env)
        raise ExecReached()
    monkeypatch.setattr(benchmark.os, "execve", fake_exec)
    monkeypatch.setenv("LD_PRELOAD", "/inherited/inject.so")
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/inherited/custom")
    args = argparse.Namespace(_worker="system", _run_dir=str(tmp_path), _env_ready=False,
                              vendor_root=str(tmp_path / "vendor"))
    with pytest.raises(ExecReached):
        benchmark.run_worker(args)
    assert "LD_PRELOAD" not in captured["env"]
    assert "ASCEND_CUSTOM_OPP_PATH" not in captured["env"]
    assert captured["argv"][-1] == "--_env-ready"
    assert not (tmp_path / "system").exists()  # Re-exec precedes initialization/imports.
