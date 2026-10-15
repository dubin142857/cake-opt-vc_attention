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
                  b"    __gm__ float* __restrict__ log_sum_max_gm;\n"
                  b"    address.set_core_info(core_num, cur_core_index, vector_id);\n"
                  b"            attention_score_short_double_line_one(sectionLoop, section.sectionNum, section.isTriangle,\n"
                  b"        ffts_cross_core_sync(PIPE_MTE3, config4);\n    }\n}\n" +
                  b"void bulk() {\n" + patch.BULK_VEXP + b"\n}\n"
                  b"void history() { vexp(history, history, 1, 1, 1, 8, 8); }\n")
    files = {
        ".gitignore": b"/build/build/\n",
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


def previous_probability_version(source, manifest_path):
    """Model the exact file and hash records emitted before the constexpr fix."""
    path = source / CUSTOM_VECTOR
    current = path.read_bytes()
    newline = b"\r\n" if b"\r\n" in current else b"\n"
    new = patch.EXPERIMENTAL_PROBABILITY.replace(b"\n", newline)
    old = patch.LEGACY_EXPERIMENTAL_PROBABILITY.replace(b"\n", newline)
    assert current.count(new) == 1
    previous = current.replace(new, old, 1)
    path.write_bytes(previous)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_files"][CUSTOM_VECTOR]["prepared_sha256"] = prepare.sha256(previous)
    manifest["generated_files"][CUSTOM_VECTOR]["prepared_sha256"] = prepare.sha256(previous)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return current, previous


@pytest.mark.parametrize("variant", ["expcast", "expcast_probe"])
def test_known_probability_revision_migrates_only_vector_and_manifest(variants, tmp_path, variant):
    baseline, _, _, source, manifest, _, lock = variants
    if variant.endswith("_probe"):
        source, manifest = tmp_path / variant, tmp_path / f"{variant}_manifest.json"
        prepare.prepare_source(source, manifest, offline=True, reference_dir=baseline,
                               source_lock=lock, variant=variant)
    current, previous = previous_probability_version(source, manifest)
    assert current != previous
    sentinel = source / "build/build/cached-device.o"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"partial compiler cache")
    mtimes = {path: path.stat().st_mtime_ns for path in source.rglob("*")
              if path.is_file() and ".git" not in path.relative_to(source).parts
              and path != source / CUSTOM_VECTOR}
    updated = prepare.prepare_source(source, manifest, offline=True, source_lock=lock, variant=variant)
    assert (source / CUSTOM_VECTOR).read_bytes() == current
    assert updated["generated_files"][CUSTOM_VECTOR]["prepared_sha256"] == prepare.sha256(current)
    assert updated["source_files"][CUSTOM_VECTOR]["prepared_sha256"] == prepare.sha256(current)
    assert sentinel.read_bytes() == b"partial compiler cache"
    assert all(path.stat().st_mtime_ns == mtime for path, mtime in mtimes.items())
    assert prepare.validate_prepared_source(source, manifest)["variant"] == variant
    stable_manifest, stable_mtime = manifest.read_bytes(), (source / CUSTOM_VECTOR).stat().st_mtime_ns
    assert prepare.prepare_source(source, manifest, offline=True, source_lock=lock, variant=variant) == updated
    assert manifest.read_bytes() == stable_manifest
    assert (source / CUSTOM_VECTOR).stat().st_mtime_ns == stable_mtime


@pytest.mark.parametrize("edited", [CUSTOM_VECTOR, HEADER, "README.md"])
def test_migration_rejects_manual_changes_even_with_matching_forged_hashes(variants, edited):
    _, _, _, source, manifest, _, lock = variants
    _, legacy = previous_probability_version(source, manifest)
    path = source / edited
    modified = path.read_bytes() + b"// user change must survive\n"
    path.write_bytes(modified)
    forged = json.loads(manifest.read_text(encoding="utf-8"))
    forged["source_files"][edited]["prepared_sha256"] = prepare.sha256(modified)
    if edited in forged["generated_files"]:
        forged["generated_files"][edited]["prepared_sha256"] = prepare.sha256(modified)
    manifest.write_text(json.dumps(forged), encoding="utf-8")
    manifest_before = manifest.read_bytes()
    with pytest.raises(prepare.PreparationError):
        prepare.prepare_source(source, manifest, offline=True, source_lock=lock, variant="expcast")
    assert path.read_bytes() == modified
    assert manifest.read_bytes() == manifest_before
    if edited != CUSTOM_VECTOR:
        assert (source / CUSTOM_VECTOR).read_bytes() == legacy


@pytest.mark.parametrize("manifest_problem", ["missing", "wrong_checkout", "wrong_commit", "old_hash_mismatch"])
def test_migration_requires_matching_historical_manifest(variants, manifest_problem):
    _, _, _, source, manifest, _, lock = variants
    _, legacy = previous_probability_version(source, manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_problem == "missing":
        manifest.unlink()
    else:
        if manifest_problem == "wrong_checkout":
            data["source_dir"] += "-another"
        elif manifest_problem == "wrong_commit":
            data["upstream"]["commit"] = "f" * 40
        else:
            data["source_files"][CUSTOM_VECTOR]["prepared_sha256"] = "f" * 64
        manifest.write_text(json.dumps(data), encoding="utf-8")
    before = manifest.read_bytes() if manifest.exists() else None
    with pytest.raises(prepare.PreparationError):
        prepare.prepare_source(source, manifest, offline=True, source_lock=lock, variant="expcast")
    assert (source / CUSTOM_VECTOR).read_bytes() == legacy
    assert (manifest.read_bytes() if manifest.exists() else None) == before
