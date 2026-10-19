"""Local Git object reuse and interrupted checkout recovery, without a network."""
from pathlib import Path
import os
import subprocess

import pytest

from scripts import mindie_laser_offline_cache as cache
from scripts.prepare_cann_fas_source import PreparationError, RepositoryLock


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True).stdout.decode().strip()


def init(repo):
    repo.mkdir(parents=True)
    git(repo, "init", "--quiet")
    git(repo, "config", "core.autocrlf", "false")
    git(repo, "config", "user.name", "Offline cache test")
    git(repo, "config", "user.email", "offline@example.invalid")


@pytest.fixture
def donor(tmp_path):
    repo = tmp_path / "donor with spaces"
    init(repo)
    (repo / "kernel.cpp").write_bytes(b"// first commit\n")
    git(repo, "add", ".")
    git(repo, "commit", "--quiet", "-m", "first")
    (repo / "kernel.cpp").write_bytes(b"// pinned upstream bytes\n")
    git(repo, "add", ".")
    git(repo, "commit", "--quiet", "-m", "pinned")
    lock = RepositoryLock("https://example.invalid/MindIE-SD.git", git(repo, "rev-parse", "HEAD"))
    return repo, lock


def snapshot(path):
    return {p.relative_to(path).as_posix(): p.read_bytes()
            for p in path.rglob("*") if p.is_file()}


def test_dirty_reference_imports_commit_only_and_preserves_donor(donor, tmp_path):
    repo, lock = donor
    (repo / "kernel.cpp").write_bytes(b"// locally patched kernel\n")
    (repo / "generated.cpp").write_bytes(b"// generated candidate\n")
    before = snapshot(repo)
    source = tmp_path / "new checkout"
    result = cache.seed_local_checkout(source, [tmp_path / "missing", repo, repo], lock)
    assert result["reference"] == str(repo.resolve())
    assert (source / "kernel.cpp").read_bytes() == b"// pinned upstream bytes\n"
    assert not (source / "generated.cpp").exists()
    assert git(source, "rev-parse", "HEAD") == lock.commit
    assert git(source, "remote", "get-url", "origin") == lock.url
    assert snapshot(repo) == before
    # Independent objects: no dependency on the donor's object directory.
    assert not (source / ".git/objects/info/alternates").exists()


def test_shallow_reference_can_seed_without_its_ancestors(donor, tmp_path):
    repo, lock = donor
    shallow = tmp_path / "shallow donor"
    subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--no-hardlinks",
                    repo.as_uri(), str(shallow)], check=True, capture_output=True)
    assert git(shallow, "rev-parse", "--is-shallow-repository") == "true"
    before = snapshot(shallow)
    source = tmp_path / "from shallow"
    cache.seed_local_checkout(source, [shallow], lock)
    assert git(source, "rev-parse", "HEAD") == lock.commit
    assert (source / "kernel.cpp").read_bytes() == b"// pinned upstream bytes\n"
    assert snapshot(shallow) == before


@pytest.mark.parametrize("origin", [False, True])
def test_recovers_empty_interrupted_git_without_deletion(donor, tmp_path, origin):
    repo, lock = donor
    source = tmp_path / "interrupted"
    init(source)
    if origin:
        git(source, "remote", "add", "origin", lock.url)
    # An unrelated repository config entry must survive recovery.
    git(source, "config", "test.preserve", "user-value")
    cache.seed_local_checkout(source, [repo], lock)
    assert git(source, "rev-parse", "HEAD") == lock.commit
    assert git(source, "config", "test.preserve") == "user-value"


def test_existing_pinned_source_is_untouched_without_donor(donor):
    source, lock = donor
    (source / "kernel.cpp").write_bytes(b"// already patched\n")
    before = snapshot(source)
    report = cache.seed_local_checkout(source, [], lock)
    assert report["mode"] == "existing"
    assert snapshot(source) == before


@pytest.mark.parametrize("case", ["untracked", "indexed", "foreign_origin", "wrong_head", "plain_dir"])
def test_rejects_unexpected_destination_without_mutation(donor, tmp_path, case):
    repo, lock = donor
    source = tmp_path / case
    if case == "plain_dir":
        source.mkdir()
        (source / "notes.txt").write_bytes(b"preserve me")
    else:
        init(source)
        if case == "foreign_origin":
            git(source, "remote", "add", "origin", "https://example.invalid/other.git")
        if case in ("untracked", "indexed", "wrong_head"):
            (source / "notes.txt").write_bytes(b"preserve me")
        if case in ("indexed", "wrong_head"):
            git(source, "add", ".")
        if case == "wrong_head":
            git(source, "commit", "--quiet", "-m", "unrelated")
    before = snapshot(source)
    with pytest.raises(PreparationError):
        cache.seed_local_checkout(source, [repo], lock)
    assert snapshot(source) == before


def test_missing_commit_fails_before_creating_destination(donor, tmp_path, monkeypatch):
    repo, lock = donor
    absent = RepositoryLock(lock.url, "0" * 40)
    source = tmp_path / "not created"
    calls = []
    original = cache.local_git

    def record(repo, *args, **kwargs):
        calls.append(args)
        return original(repo, *args, **kwargs)

    monkeypatch.setattr(cache, "local_git", record)
    with pytest.raises(PreparationError, match="network access is disabled"):
        cache.seed_local_checkout(source, [tmp_path / "missing", repo], absent)
    assert not source.exists()
    assert all(args[0] not in ("fetch", "clone", "init") for args in calls)


def test_commands_forbid_network_and_lazy_fetch(donor, tmp_path, monkeypatch):
    repo, lock = donor
    original = subprocess.run
    calls = []

    def guarded(command, **kwargs):
        env = kwargs["env"]
        assert env["GIT_ALLOW_PROTOCOL"] == "file"
        assert env["GIT_NO_LAZY_FETCH"] == "1"
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        calls.append(command)
        return original(command, **kwargs)

    monkeypatch.setattr(cache.subprocess, "run", guarded)
    cache.seed_local_checkout(tmp_path / "candidate", [repo], lock)
    transfers = [command for command in calls if command[3] == "fetch"]
    assert len(transfers) == 1
    assert transfers[0][-2:] == [str(repo.resolve()), lock.commit]
    assert "origin" not in transfers[0]


@pytest.mark.parametrize("failure", [False, True])
def test_offline_scope_restores_absent_and_existing_environment(monkeypatch, failure):
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "https:ssh")
    monkeypatch.delenv("GIT_NO_LAZY_FETCH", raising=False)
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "1")
    before = {key: os.environ.get(key) for key in
              ("GIT_ALLOW_PROTOCOL", "GIT_NO_LAZY_FETCH", "GIT_TERMINAL_PROMPT")}

    def exercise():
        with cache.local_objects_only():
            assert os.environ["GIT_ALLOW_PROTOCOL"] == "file"
            assert os.environ["GIT_NO_LAZY_FETCH"] == "1"
            assert os.environ["GIT_TERMINAL_PROMPT"] == "0"
            if failure:
                raise RuntimeError("local audit failure")

    if failure:
        with pytest.raises(RuntimeError, match="local audit failure"):
            exercise()
    else:
        exercise()
    assert {key: os.environ.get(key) for key in before} == before
