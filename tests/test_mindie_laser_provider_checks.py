"""CPU filesystem tests for the narrowly scoped ambient provider exception."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mindie_laser_provider_checks import inspect_batch_invariant


def provider(tmp_path, arch="aarch64", soc="ascend910_93", data=None):
    cann = tmp_path / "cann-9.0.1"
    vendor = cann / "opp/vendors/batch_invariant"
    library = vendor / "op_impl/ai_core/tbe/op_tiling/lib/linux" / arch / "libcust_opmaster_rt2.0.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"existing batch-invariant tiling library")
    config = vendor / "op_impl/ai_core/tbe/config" / soc / f"aic-{soc}-ops-info.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(data if data is not None else {"MatMulBatchInvariant": {"opFile": {"value": "matmul_batch_invariant"}}}), encoding="utf-8")
    return cann, vendor, library, config


@pytest.mark.parametrize("arch,soc", [("aarch64", "ascend910_93"), ("x86_64", "ascend910b")])
def test_canonical_provider_is_accepted_and_hashes_are_recorded(tmp_path, arch, soc):
    cann, vendor, library, config = provider(tmp_path, arch, soc)
    result = inspect_batch_invariant([str(library), str(library)], cann, soc)
    assert result["libraries"] == [str(library.resolve())]
    evidence = result["metadata"]
    assert evidence["classification"] == "environment_existing_provider"
    assert evidence["config_path"] == str(config.resolve())
    assert evidence["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert evidence["library_sha256"][str(library.resolve())] == hashlib.sha256(library.read_bytes()).hexdigest()
    assert evidence["op_names"] == ["MatMulBatchInvariant"]


def test_same_filename_elsewhere_and_api_library_are_not_exempted(tmp_path):
    cann, vendor, library, config = provider(tmp_path)
    foreign = tmp_path / "foreign/vendors/batch_invariant/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(library.read_bytes())
    api = vendor / "op_api/lib/libcust_opapi.so"
    api.parent.mkdir(parents=True)
    api.write_bytes(b"API must remain subject to strict checks")
    result = inspect_batch_invariant([str(foreign), str(api)], cann, "ascend910_93")
    assert result == {"libraries": [], "metadata": {}}
    # A matching allowed file must not accidentally exempt its foreign sibling.
    result = inspect_batch_invariant([str(foreign), str(library)], cann, "ascend910_93")
    assert str(foreign.resolve()) not in result["libraries"]


def test_no_matching_library_needs_no_provider_or_configuration(tmp_path):
    assert inspect_batch_invariant([], tmp_path / "missing-cann", "unknown") == {"libraries": [], "metadata": {}}
    assert inspect_batch_invariant([str(tmp_path / "libcust_opmaster_rt2.0.so")], tmp_path / "missing-cann", "ascend910_93")["libraries"] == []


@pytest.mark.parametrize("data", [
    {"LaserAttention": {}}, {"VcLaserAttention": {}},
    {"Other": {"opFile": {"value": "vc_laser_attention"}}},
    {"Other": {"nested": [{"aliases": ["aclnnLaserAttentionGetWorkspaceSize"]}] }},
])
def test_laser_registration_in_any_nested_key_or_value_is_rejected(tmp_path, data):
    cann, vendor, library, config = provider(tmp_path, data=data)
    with pytest.raises(ValueError, match="LaserAttention/VcLaserAttention") as failure:
        inspect_batch_invariant([str(library)], cann, "ascend910_93")
    assert str(config.resolve()) in str(failure.value)


@pytest.mark.parametrize("content", ["{}", "[]", "not json", '["MatMul"]'])
def test_invalid_or_empty_configuration_is_rejected(tmp_path, content):
    cann, vendor, library, config = provider(tmp_path)
    config.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        inspect_batch_invariant([str(library)], cann, "ascend910_93")


def test_missing_configuration_is_rejected_with_exact_path(tmp_path):
    cann, vendor, library, config = provider(tmp_path)
    config.unlink()
    with pytest.raises(ValueError, match="configuration is missing") as failure:
        inspect_batch_invariant([str(library)], cann, "ascend910_93")
    assert str(config) in str(failure.value) and str(library) in str(failure.value)


def test_symlink_escape_is_rejected_even_when_maps_reports_the_resolved_target(monkeypatch, tmp_path):
    cann, vendor, library, config = provider(tmp_path)
    escaped = tmp_path / "foreign/libcust_opmaster_rt2.0.so"
    escaped.parent.mkdir()
    escaped.write_bytes(b"foreign")
    original_resolve = Path.resolve
    def resolve(path, *args, **kwargs):
        return original_resolve(escaped, *args, **kwargs) if path == library else original_resolve(path, *args, **kwargs)
    # Model filesystem resolution on Windows without requiring symlink privileges.
    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="escapes its canonical location"):
        inspect_batch_invariant([str(escaped)], cann, "ascend910_93")


def test_normal_cann_root_symlink_is_resolved(monkeypatch, tmp_path):
    cann, vendor, library, config = provider(tmp_path)
    alias = tmp_path / "latest"
    original_resolve = Path.resolve
    def resolve(path, *args, **kwargs):
        if path == alias:
            return original_resolve(cann, *args, **kwargs)
        return original_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    assert inspect_batch_invariant([str(library)], alias, "ascend910_93")["libraries"] == [str(library.resolve())]


def test_normal_opp_symlink_is_resolved(monkeypatch, tmp_path):
    cann, vendor, library, config = provider(tmp_path)
    shared_opp = tmp_path / "shared-opp"
    shutil.copytree(cann / "opp", shared_opp)
    mapped = shared_opp / library.relative_to(cann / "opp")
    original_resolve = Path.resolve
    def resolve(path, *args, **kwargs):
        return original_resolve(shared_opp, *args, **kwargs) if path == cann / "opp" else original_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    result = inspect_batch_invariant([str(mapped)], cann, "ascend910_93")
    assert result["libraries"] == [str(mapped.resolve())]
    assert result["metadata"]["opp_root"] == str(shared_opp.resolve())


def test_vendor_symlink_to_another_provider_is_rejected(monkeypatch, tmp_path):
    cann, vendor, library, config = provider(tmp_path)
    foreign = vendor.parent / "foreign"
    original_resolve = Path.resolve
    def resolve(path, *args, **kwargs):
        return original_resolve(foreign, *args, **kwargs) if path == vendor else original_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="vendor symlink escapes"):
        inspect_batch_invariant([str(library)], cann, "ascend910_93")


def test_configuration_symlink_outside_vendor_is_rejected(monkeypatch, tmp_path):
    cann, vendor, library, config = provider(tmp_path)
    foreign = tmp_path / "foreign-config.json"
    foreign.write_text('{"MatMul": {}}', encoding="utf-8")
    original_resolve = Path.resolve
    def resolve(path, *args, **kwargs):
        return original_resolve(foreign, *args, **kwargs) if path == config else original_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="configuration escapes"):
        inspect_batch_invariant([str(library)], cann, "ascend910_93")
