"""Pinned Laser source isolation and provenance; no Toolkit or network needed."""
from pathlib import Path
import subprocess

import pytest

from scripts import prepare_mindie_laser_source as prepare


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True).stdout.decode().strip()


@pytest.fixture
def checkout(tmp_path):
    source = tmp_path / "build/source"
    source.mkdir(parents=True)
    git(source, "init", "--quiet")
    git(source, "config", "core.autocrlf", "false")
    git(source, "config", "user.name", "Laser source test")
    git(source, "config", "user.email", "laser@example.invalid")
    files = {
        ".gitignore": "build/build/\n",
        "README.md": "upstream stays intact\n",
        "csrc/ops/CMakeLists.txt": "project(test)\n",
        prepare.PREPARE_SCRIPT: '#!/bin/bash\nCPU_NUM=$(($(cat /proc/cpuinfo | grep "^processor" | wc -l)*2))\nJOB_NUM="-j${CPU_NUM}"\n',
        prepare.TILING_FILE: '#include "laser_attention_tiling.h"\n'
            'int LaserAttentionTilingFunc(Context *context) {\n'
            '    return tiling.CheckTiling(context);\n}\n',
        f"{prepare.OP_ROOT}/op_host/laser_attention_tiling.h": "// copyright retained\nclass LaserAttentionTiling {};\n",
        f"{prepare.OP_ROOT}/op_host/CMakeLists.txt": "set(laser_attention_depends laser_attention PARENT_SCOPE)\n",
        f"{prepare.OP_ROOT}/op_kernel/laser_attention.cpp": "// Copyright upstream\nvoid laser_attention() { /* LaserAttention */ }\n",
        f"{prepare.OP_ROOT}/op_kernel/CubeForward.h": "// Copyright upstream\nvoid CubeForward() {}\n",
    }
    for name, data in files.items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data.encode())
    git(source, "add", ".")
    git(source, "commit", "--quiet", "-m", "fixture")
    lock = prepare.RepositoryLock("https://example.invalid/mindie.git", git(source, "rev-parse", "HEAD"))
    return source, tmp_path / "manifest.json", lock


def run_prepare(checkout):
    source, manifest, lock = checkout
    return prepare.prepare_source(source, manifest, offline=True, source_lock=lock)


def test_renamed_kernel_is_inverse_equivalent_and_upstream_is_untouched(checkout):
    source, manifest_path, _ = checkout
    first = run_prepare(checkout)
    files = first["generated_files"]
    assert first["native_api"] == "aclnnVcLaserAttention"
    assert first["kernel_inverse_rename_equivalent"] is True
    assert first["marker_token"] == "vc_mindie_laser_v1"
    assert first["build_parallelism_patch"] == prepare.PREPARE_SCRIPT
    for name, item in files.items():
        assert prepare.sha256((source / name).read_bytes()) == item["prepared_sha256"]
        if "/op_kernel/" in name:
            assert item["inverse_rename_equivalent"] is True
            original = (source / item["original_path"]).read_bytes()
            assert prepare.inverse_rename((source / name).read_bytes()) == original
    assert git(source, "diff", "--name-only") == prepare.PREPARE_SCRIPT
    mtimes = {name: (source / name).stat().st_mtime_ns for name in files}
    manifest_bytes = manifest_path.read_bytes()
    assert run_prepare(checkout) == first
    assert manifest_path.read_bytes() == manifest_bytes
    assert mtimes == {name: (source / name).stat().st_mtime_ns for name in files}
    tiling = (source / prepare.custom_path(prepare.TILING_FILE)).read_text()
    assert 'if (result == ge::GRAPH_SUCCESS)' in tiling
    assert 'VcMindieLaserSourceMarker("tiling"' in tiling


@pytest.mark.parametrize("kind", ["kernel", "host", "upstream", "prepare", "extra"])
def test_unknown_source_changes_are_preserved_and_rejected(checkout, kind):
    source, manifest, _ = checkout
    run_prepare(checkout)
    paths = {
        "kernel": f"{prepare.CUSTOM_OP_ROOT}/op_kernel/CubeForward.h",
        "host": prepare.custom_path(prepare.TILING_FILE),
        "upstream": "README.md", "prepare": prepare.PREPARE_SCRIPT,
        "extra": f"{prepare.CUSTOM_OP_ROOT}/op_kernel/unexpected.h",
    }
    path = source / paths[kind]
    old = path.read_bytes() if path.exists() else b""
    path.write_bytes(old + b"user edit\n")
    manifest_before = manifest.read_bytes()
    with pytest.raises(prepare.PreparationError):
        run_prepare(checkout)
    assert path.read_bytes() == old + b"user edit\n"
    assert manifest.read_bytes() == manifest_before


def test_staged_patch_is_rejected(checkout):
    source, _, _ = checkout
    run_prepare(checkout)
    git(source, "add", prepare.PREPARE_SCRIPT)
    with pytest.raises(prepare.PreparationError, match="staged"):
        run_prepare(checkout)


def test_copy_from_reference_uses_objects_without_modifying_reference(checkout, tmp_path):
    reference, _, lock = checkout
    original_status = git(reference, "status", "--porcelain")
    destination = tmp_path / "isolated"
    manifest = prepare.prepare_source(destination, tmp_path / "copy.json", offline=True,
                                     reference_dir=reference, source_lock=lock)
    assert manifest["upstream"]["commit"] == lock.commit
    assert git(reference, "status", "--porcelain") == original_status
    assert not (reference / prepare.CUSTOM_OP_ROOT).exists()


def test_marker_is_optional_and_has_pid_and_key():
    marker = prepare.MARKER_HEADER.decode()
    assert 'getenv("VC_MINDIE_LASER_MARKER_FILE")' in marker
    assert "getpid()" in marker and "tiling_key" in marker
    assert "std::abort()" in marker


def test_prepare_parallelism_patch_honors_jobs(checkout):
    source, _, _ = checkout
    run_prepare(checkout)
    script = (source / prepare.PREPARE_SCRIPT).read_text()
    assert 'CPU_NUM=${BUILD_JOB_NUM:-1}' in script
    assert '^[1-9][0-9]*$' in script
    assert '/proc/cpuinfo' not in script
