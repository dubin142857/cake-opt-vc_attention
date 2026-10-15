"""Source provenance preparation tests; no CANN, NPU, or network required."""

from pathlib import Path
import subprocess

import pytest

from scripts import prepare_cann_fas_source as prepare


def git(repo, *args):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    )
    return result.stdout.decode().strip()


def init_repo(repo: Path, files: dict[str, str]) -> str:
    repo.mkdir(parents=True)
    git(repo, "init", "--quiet")
    git(repo, "config", "core.autocrlf", "false")
    git(repo, "config", "user.name", "FAS source test")
    git(repo, "config", "user.email", "fas-source-test@example.invalid")
    for name, data in files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data.encode("utf-8"))
    git(repo, "add", ".")
    git(repo, "commit", "--quiet", "-m", "test fixture")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def checkout(tmp_path):
    source = tmp_path / "workspace/build/source"
    # CMake's optional ../../ops-base search must not shadow the locked copy.
    dep_repo = tmp_path / "opbase-origin"
    dep_commit = init_repo(dep_repo, {"README.md": "dependency\n"})
    source_commit = init_repo(source, {
        ".gitignore": "/third_party/\n/build/\n",
        prepare.API_FILE: '#include "aclnn_flash_attention_score.h"\n'
            "aclnnStatus aclnnFlashAttentionScoreGetWorkspaceSize(int input)\n"
            "{\n    return input;\n}\n",
        prepare.TILING_FILE: "#include <queue>\n"
            "int TilingFlashAttentionScore(Context *context)\n{\n"
            "        if (IsEmptyInputRegbase(context)) {\n            return ge::GRAPH_SUCCESS;\n        }\n"
            "        if (IsEmptyInput(context)) {\n            return ge::GRAPH_SUCCESS;\n        }\n"
            "    auto resultCode = TilingRegistryArch::GetInstance().DoTilingImpl(context);\n"
            "    return resultCode;\n}\n",
        prepare.OPBASE_CMAKE: f"set(OPBASE_TAG_ID {dep_commit})\n",
        f"{prepare.OP_ROOT}/op_kernel/arch32/kernel.cpp": "// untouched device kernel\n",
        "common/include/helper.h": "// unchanged common code\n",
    })
    dep = source / "third_party/opbase"
    subprocess.run(["git", "clone", "--quiet", str(dep_repo), str(dep)], check=True)
    locks = {
        "source_lock": prepare.RepositoryLock("https://example.invalid/source.git", source_commit, "test-tag"),
        "opbase_lock": prepare.RepositoryLock("https://example.invalid/opbase.git", dep_commit),
    }
    return source, tmp_path / "manifest.json", locks


def run_prepare(checkout):
    source, manifest, locks = checkout
    return prepare.prepare_source(source, manifest, offline=True, **locks)


def test_offline_preparation_is_idempotent_and_audits_every_source(checkout):
    source, manifest_path, _ = checkout
    first = run_prepare(checkout)
    first_bytes = manifest_path.read_bytes()
    mtimes = {name: (source / name).stat().st_mtime_ns for name in first["modified_host_files"]}
    second = run_prepare(checkout)
    assert first == second
    assert manifest_path.read_bytes() == first_bytes
    assert mtimes == {name: (source / name).stat().st_mtime_ns for name in first["modified_host_files"]}
    assert first["marker_token"] == "vc_cann_fas_v901"
    assert first["kernel_source_unchanged"] is True
    assert len(first["modified_host_files"]) == 3
    files = first["source_files"]
    assert files[prepare.MARKER_FILE]["original_sha256"] is None
    assert files[prepare.API_FILE]["original_sha256"] != files[prepare.API_FILE]["prepared_sha256"]
    for name, hashes in files.items():
        assert hashes["prepared_sha256"] == prepare.sha256((source / name).read_bytes())
        if "/op_kernel/" in name:
            assert hashes["original_sha256"] == hashes["prepared_sha256"]
    assert set(files) == set(git(source, "ls-files").splitlines()) | {prepare.MARKER_FILE}


@pytest.mark.parametrize("name", [
    prepare.API_FILE, prepare.TILING_FILE, prepare.MARKER_FILE,
    f"{prepare.OP_ROOT}/op_kernel/arch32/kernel.cpp", "common/include/helper.h",
])
def test_refuses_unknown_modified_contents_without_overwriting(checkout, name):
    source, manifest_path, _ = checkout
    run_prepare(checkout)
    old_manifest = manifest_path.read_bytes()
    path = source / name
    modified = path.read_bytes() + b"// user edit\n"
    path.write_bytes(modified)
    with pytest.raises(prepare.PreparationError, match="unknown dirty|Unknown contents|Unknown generated"):
        run_prepare(checkout)
    assert path.read_bytes() == modified
    assert manifest_path.read_bytes() == old_manifest


def test_refuses_staged_changes_even_known_marker_patch(checkout):
    source, _, _ = checkout
    run_prepare(checkout)
    git(source, "add", prepare.API_FILE)
    with pytest.raises(prepare.PreparationError, match="staged changes"):
        run_prepare(checkout)


def test_refuses_unknown_untracked_source(checkout):
    source, _, _ = checkout
    (source / "unexpected.cpp").write_text("// user file\n")
    with pytest.raises(prepare.PreparationError, match="unknown dirty"):
        run_prepare(checkout)


def test_refuses_dirty_dependency_before_patching_source(checkout):
    source, _, _ = checkout
    original = (source / prepare.API_FILE).read_bytes()
    (source / "third_party/opbase/README.md").write_text("modified dependency\n")
    with pytest.raises(prepare.PreparationError, match="unknown dirty"):
        run_prepare(checkout)
    assert (source / prepare.API_FILE).read_bytes() == original
    assert not (source / prepare.MARKER_FILE).exists()


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_dependency_index_flags_cannot_hide_modified_sources(checkout, flag):
    source, _, _ = checkout
    dependency = source / "third_party/opbase"
    git(dependency, "update-index", flag, "README.md")
    (dependency / "README.md").write_text("hidden change\n")
    with pytest.raises(prepare.PreparationError, match="index flags"):
        run_prepare(checkout)
    assert not (source / prepare.MARKER_FILE).exists()


def test_refuses_shadowing_opbase_source(checkout):
    source, _, _ = checkout
    (source.parent.parent / "ops-base").mkdir()
    with pytest.raises(prepare.PreparationError, match="prefer unlocked dependency"):
        run_prepare(checkout)


def test_refuses_wrong_head(checkout):
    source, _, _ = checkout
    git(source, "commit", "--quiet", "--allow-empty", "-m", "another commit")
    with pytest.raises(prepare.PreparationError, match="Refusing HEAD"):
        run_prepare(checkout)


def test_offline_missing_cache_does_not_invoke_network(tmp_path, monkeypatch):
    def no_commands(*args, **kwargs):
        raise AssertionError("offline missing cache must not run Git/network commands")

    monkeypatch.setattr(prepare.subprocess, "run", no_commands)
    with pytest.raises(prepare.PreparationError, match="Offline cache missing"):
        prepare.ensure_checkout(tmp_path / "missing", prepare.SOURCE_LOCK, offline=True)


def test_refuses_incorrect_dependency_lock_before_patching(checkout):
    source, manifest, locks = checkout
    locks["opbase_lock"] = prepare.RepositoryLock("https://example.invalid/opbase.git", "f" * 40)
    with pytest.raises(prepare.PreparationError, match="pin differs"):
        prepare.prepare_source(source, manifest, offline=True, **locks)
    assert not (source / prepare.MARKER_FILE).exists()


def test_retries_interrupted_fetch_without_touching_existing_files(tmp_path):
    origin = tmp_path / "origin"
    commit = init_repo(origin, {"README.md": "pinned source\n"})
    git(origin, "tag", "fixed-tag")
    destination = tmp_path / "interrupted"
    destination.mkdir()
    git(destination, "init", "--quiet")
    git(destination, "remote", "add", "origin", str(origin))
    lock = prepare.RepositoryLock(str(origin), commit, "fixed-tag")
    with pytest.raises(prepare.PreparationError, match="Offline cache incomplete"):
        prepare.ensure_checkout(destination, lock, offline=True)
    prepare.ensure_checkout(destination, lock, offline=False)
    assert git(destination, "rev-parse", "HEAD") == commit
    assert (destination / "README.md").read_bytes() == b"pinned source\n"


def test_interrupted_checkout_with_user_file_is_not_retried(tmp_path):
    destination = tmp_path / "interrupted"
    destination.mkdir()
    git(destination, "init", "--quiet")
    (destination / "important.txt").write_text("user data")
    with pytest.raises(prepare.PreparationError, match="nonempty checkout without HEAD"):
        prepare.ensure_checkout(destination, prepare.SOURCE_LOCK, offline=False)
    assert (destination / "important.txt").read_text() == "user data"
