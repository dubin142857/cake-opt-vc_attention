"""Packaging and entrypoint regressions; Hopper execution uses validate_sage_local.py."""
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party/sageattention_local"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vendored_kernels_match_pinned_sources_and_detect_edits(tmp_path):
    verifier = load_file("verify_sage_local_source", ROOT / "scripts/verify_sage_local_source.py")
    result = verifier.verify(VENDOR)
    assert result["verified_files"] == 49
    assert result["unchanged_kernel_files"] > 20
    assert result["build_only_patches"] == ["csrc/fused/fused.cu"]
    assert result["verified_experiment_files"] == 9
    copy = tmp_path / "source"
    shutil.copytree(VENDOR, copy, ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__"))
    target = copy / "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu"
    with target.open("a", encoding="utf-8") as file:
        file.write("\n// changed\n")
    with pytest.raises(RuntimeError, match="Source hash mismatch"):
        verifier.verify(copy)


def test_include_patch_cannot_hide_kernel_body_edits(tmp_path):
    verifier = load_file("verify_sage_local_source", ROOT / "scripts/verify_sage_local_source.py")
    copy = tmp_path / "source"
    shutil.copytree(VENDOR, copy, ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__"))
    target = copy / "csrc/fused/fused.cu"
    data = target.read_bytes().replace(b"\r\n", b"\n") + b"\n// unintended kernel edit\n"
    target.write_bytes(data)
    manifest = json.loads((copy / "UPSTREAM.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        if row["local_path"] == "csrc/fused/fused.cu":
            row["local_sha256"] = hashlib.sha256(data).hexdigest()
    for path in (copy / "UPSTREAM.json", copy / "sageattention_local/UPSTREAM.json"):
        path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Kernel body differs from upstream"):
        verifier.verify(copy)


def test_sm90_fake_registration_does_not_shadow_native_entry_or_original_namespace(monkeypatch):
    calls = []
    original_name = "sageattention_sm90::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf"
    native_name = original_name.replace("sageattention_sm90", "sageattention_local_sm90")
    original = object()
    registered, fakes = {original_name: original}, {}

    def custom_op(name, **kwargs):
        def register(fn):
            assert name not in registered
            registered[name] = fn
            return fn
        return register

    def register_fake(name):
        def register(fn):
            assert name in registered
            fakes[name] = fn
            return fn
        return register

    def cuda_entry(*args):
        calls.append(args)
        return "native called"

    package = ModuleType("sageattention_local")
    package._qattn_sm90 = SimpleNamespace(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf=cuda_entry)
    torch = SimpleNamespace(Tensor=object, library=SimpleNamespace(custom_op=custom_op, register_fake=register_fake))
    monkeypatch.setitem(sys.modules, "sageattention_local", package)
    monkeypatch.setitem(sys.modules, "torch", torch)
    module = load_file("sageattention_local.sm90_compile", VENDOR / "sageattention_local/sm90_compile.py")
    entry = module.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
    assert entry is registered[native_name]
    assert entry is not fakes[native_name]
    assert registered[original_name] is original
    arguments = tuple(range(12))
    assert entry(*arguments) == "native called"
    assert calls == [arguments]


def test_source_provenance_separates_baseline_and_vc_derivative():
    manifest = json.loads((VENDOR / "UPSTREAM.json").read_text(encoding="utf-8"))
    assert manifest["commit"] == "eb615cf6cf4d221338033340ee2de1c37fbdba4a"
    assert manifest["local_distribution"] == "sageattention-local"
    assert manifest["local_version"] == "2.2.0+vc14"
    kernel = next(row for row in manifest["experiment_files"] if row["local_path"] == "csrc/vc/vc_sm90.cu")
    assert kernel["derived_from"] == "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu"
    assert {row["upstream_path"] for row in manifest["files"] if row["modified"]} == {
        "sageattention/__init__.py", "sageattention/sm80_compile.py",
        "sageattention/sm89_compile.py", "sageattention/sm90_compile.py",
        "csrc/fused/fused.cu",
    }


def test_vc_derivative_source_edits_are_detected(tmp_path):
    verifier = load_file("verify_sage_local_source", ROOT / "scripts/verify_sage_local_source.py")
    copy = tmp_path / "source"
    shutil.copytree(VENDOR,copy,ignore=shutil.ignore_patterns("build","*.egg-info","__pycache__"))
    with (copy / "csrc/vc/vc_math.cuh").open("a",encoding="utf-8") as file:
        file.write("\n// changed\n")
    with pytest.raises(RuntimeError,match="Experiment source hash mismatch"):
        verifier.verify(copy)
