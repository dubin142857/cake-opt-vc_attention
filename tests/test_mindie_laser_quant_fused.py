"""Fused-preprocessing build evidence checks, independent of NPU execution."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mindie_laser_quant_fused as fused


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    source = tmp_path / "src"
    source.mkdir()
    paths = [source / "CMakeLists.txt", source / "kernel.cpp", source / "torch.cpp"]
    for path in paths:
        path.write_text("fixture " + path.name, encoding="utf-8")
    library = tmp_path / "libvc_laser_quant_preprocess.so"
    library.write_bytes(b"fixture binary")
    build = {"status": "passed", "contract_version": 1, "soc": "Ascend910_9382",
             "library_path": str(library), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
             "source_files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                              for path in paths]}
    target = tmp_path / "build.json"
    monkeypatch.setattr(fused, "SOURCE_ROOT", source)

    def write():
        target.write_text(json.dumps(build), encoding="utf-8")
        return target

    return build, write, library, paths


def test_fused_manifest_records_source_binary_and_exact_soc(manifest):
    build, write, library, _ = manifest
    result = fused.validate_preprocess_manifest(write())
    assert result["soc"] == "Ascend910_9382"
    assert Path(result["library_path"]) == library
    assert result["source_files"] == build["source_files"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("key,value", [("status", "failed"), ("contract_version", 2),
                                        ("source_files", []), ("soc", ""),
                                        ("library_sha256", "0" * 64)])
def test_incomplete_or_stale_fused_manifest_is_rejected(manifest, key, value):
    build, write, _, _ = manifest
    build[key] = value
    with pytest.raises(ValueError):
        fused.validate_preprocess_manifest(write())


@pytest.mark.parametrize("which", ["binary", "source", "missing_source_record", "duplicate_source"])
def test_fused_evidence_detects_changed_artifacts(manifest, which):
    build, write, library, sources = manifest
    if which == "binary":
        library.write_bytes(b"changed binary")
    elif which == "source":
        sources[0].write_text("changed source", encoding="utf-8")
    elif which == "missing_source_record":
        build["source_files"].pop()
    else:
        build["source_files"].append(dict(build["source_files"][0]))
    with pytest.raises(ValueError):
        fused.validate_preprocess_manifest(write())
