"""Exercise isolated source variants and provenance against real Git objects."""
import json
from pathlib import Path
import subprocess

import pytest

from scripts import prepare_mindie_laser_source as prepare
from scripts import mindie_laser_expcast_patch as patch


ROOT = Path(__file__).resolve().parents[1]
VECTOR = f"{prepare.OP_ROOT}/op_kernel/VectorForward.h"
CUSTOM_VECTOR = prepare.custom_path(VECTOR)
HEADER = f"{prepare.CUSTOM_OP_ROOT}/op_kernel/vc_laser_probability.h"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


@pytest.fixture
def variants(tmp_path, monkeypatch):
    source = tmp_path / "baseline"
    source.mkdir()
    git(source, "init", "--quiet")
    git(source, "config", "core.autocrlf", "false")
    git(source, "config", "user.name", "Laser variant test")
    git(source, "config", "user.email", "laser-variant@example.invalid")
    reference = ROOT / "build/mindie-sd-reference"
    vector = None
    if (reference / ".git").exists():
        result = subprocess.run(["git", "-C", str(reference), "show", f"{prepare.SOURCE_LOCK.commit}:{VECTOR}"],
                                capture_output=True)
        if result.returncode == 0:
            vector = result.stdout
    if vector is None:
        # Minimal pinned bulk-exp/include anchors; history update stays separate.
        vector = (b"// Copyright upstream\n" + patch.INCLUDE_ANCHOR +
                  b"void bulk() {\n" + patch.BULK_VEXP + b"\n}\n"
                  b"void history() { vexp(history, history, 1, 1, 1, 8, 8); }\n")
    files = {
        "README.md": b"upstream unchanged\n",
        prepare.PREPARE_SCRIPT: b'#!/bin/bash\nCPU_NUM=$(($(cat /proc/cpuinfo | grep "^processor" | wc -l)*2))\nJOB_NUM="-j${CPU_NUM}"\n',
        prepare.TILING_FILE: b'#include "laser_attention_tiling.h"\nint LaserAttentionTilingFunc(Context *context) {\n    return tiling.CheckTiling(context);\n}\n',
        f"{prepare.OP_ROOT}/op_kernel/CubeForward.h": b"// Copyright upstream\nvoid CubeForward() {}\n",
        VECTOR: vector,
    }
    for name, data in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    git(source, "add", ".")
    git(source, "commit", "--quiet", "-m", "variant fixture")
    lock = prepare.RepositoryLock("https://example.invalid/mindie.git", git(source, "rev-parse", "HEAD").decode().strip())
    monkeypatch.setattr(prepare, "SOURCE_LOCK", lock)
    baseline_manifest = tmp_path / "baseline_manifest.json"
    candidate, candidate_manifest = tmp_path / "expcast", tmp_path / "expcast_manifest.json"
    baseline = prepare.prepare_source(source, baseline_manifest, offline=True, source_lock=lock, variant="baseline")
    expcast = prepare.prepare_source(candidate, candidate_manifest, offline=True, reference_dir=source,
                                    source_lock=lock, variant="expcast")
    return source, baseline_manifest, baseline, candidate, candidate_manifest, expcast, lock


def test_variants_keep_original_sources_and_change_only_candidate_probability(variants):
    source, base_path, baseline, candidate, candidate_path, expcast, _ = variants
    assert baseline["variant"] == "baseline" and expcast["variant"] == "expcast"
    assert baseline["kernel_inverse_rename_equivalent"] is True
    assert expcast["kernel_inverse_rename_equivalent"] is False
    assert baseline["native_api"] == expcast["native_api"] == "aclnnVcLaserAttention"
    assert baseline["kernel_tree_sha256"] == expcast["kernel_tree_sha256"]
    assert (source / VECTOR).read_bytes() == (candidate / VECTOR).read_bytes()
    assert patch.unpatch_vector_forward((candidate / CUSTOM_VECTOR).read_bytes()) == (source / CUSTOM_VECTOR).read_bytes()
    changed = [name for name, item in baseline["generated_files"].items()
               if item["prepared_sha256"] != expcast["generated_files"][name]["prepared_sha256"]]
    assert changed == [CUSTOM_VECTOR]
    assert set(expcast["generated_files"]) - set(baseline["generated_files"]) == {HEADER}
    assert not (source / HEADER).exists()
    assert prepare.validate_prepared_source(source, base_path)["variant"] == "baseline"
    assert prepare.validate_prepared_source(candidate, candidate_path)["variant"] == "expcast"


def test_variant_switch_refuses_existing_checkout_and_preserves_manifest(variants):
    source, manifest, _, _, _, _, lock = variants
    before = manifest.read_bytes()
    vector = (source / CUSTOM_VECTOR).read_bytes()
    with pytest.raises(prepare.PreparationError, match="separate checkout/manifest"):
        prepare.prepare_source(source, manifest, offline=True, source_lock=lock, variant="expcast")
    assert manifest.read_bytes() == before
    assert (source / CUSTOM_VECTOR).read_bytes() == vector


@pytest.mark.parametrize("name", [CUSTOM_VECTOR, HEADER])
def test_candidate_source_tampering_is_rejected_without_overwriting(variants, name):
    _, _, _, source, manifest, _, lock = variants
    path = source / name
    data = path.read_bytes() + b"// unexpected local change\n"
    path.write_bytes(data)
    with pytest.raises(prepare.PreparationError):
        prepare.validate_prepared_source(source, manifest)
    with pytest.raises(prepare.PreparationError):
        prepare.prepare_source(source, manifest, offline=True, source_lock=lock, variant="expcast")
    assert path.read_bytes() == data


def test_candidate_manifest_cannot_claim_baseline_or_another_checkout(variants):
    base, _, _, source, manifest, expcast, _ = variants
    wrong = dict(expcast, source_dir=str(base))
    manifest.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(prepare.PreparationError, match="another checkout"):
        prepare.validate_prepared_source(source, manifest)
    wrong = dict(expcast, variant="baseline", kernel_inverse_rename_equivalent=True)
    manifest.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(prepare.PreparationError):
        prepare.validate_prepared_source(source, manifest)


def test_candidate_reprepare_is_incremental_and_keeps_baseline_unchanged(variants):
    base, base_manifest, _, source, manifest, expcast, lock = variants
    before_base = base_manifest.read_bytes()
    before_vector = (base / CUSTOM_VECTOR).read_bytes()
    mtimes = {name: (source / name).stat().st_mtime_ns for name in expcast["generated_files"]}
    result = prepare.prepare_source(source, manifest, offline=True, source_lock=lock, variant="expcast")
    assert result == expcast
    assert mtimes == {name: (source / name).stat().st_mtime_ns for name in mtimes}
    assert base_manifest.read_bytes() == before_base
    assert (base / CUSTOM_VECTOR).read_bytes() == before_vector
