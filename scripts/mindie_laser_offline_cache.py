"""Seed an independent Laser checkout using local Git objects only."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
import subprocess

from scripts.prepare_cann_fas_source import PreparationError


@contextmanager
def local_objects_only():
    """Also cover shared preparation helpers that invoke Git directly."""
    settings = {"GIT_ALLOW_PROTOCOL": "file", "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"}
    previous = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def local_git(repo, *args, check=True):
    # Disallow HTTP/SSH even if a cached repository is a partial clone or a
    # user's Git configuration rewrites the local source URL.
    env = dict(os.environ, GIT_ALLOW_PROTOCOL="file", GIT_NO_LAZY_FETCH="1",
               GIT_TERMINAL_PROMPT="0")
    result = subprocess.run(["git", "-C", str(repo), *args], env=env,
                            capture_output=True, check=False)
    if check and result.returncode:
        raise PreparationError(f"Local-only git {' '.join(args)} failed in {repo}: "
                               + result.stderr.decode("utf-8", errors="replace").strip())
    return result


def seed_local_checkout(source, references, lock):
    """Recover an empty interrupted checkout without deleting any user files.

    References may have generated/modified worktrees: only the locked Git
    commit is imported. A completed candidate checkout is left untouched.
    """
    source = Path(source).resolve()
    if source.exists():
        if not source.is_dir():
            raise PreparationError(f"Source is not a directory: {source}")
        if (source / ".git").exists():
            top = local_git(source, "rev-parse", "--show-toplevel").stdout.decode().strip()
            if Path(top).resolve() != source:
                raise PreparationError(f"Source must be its own checkout: {source}")
            head = local_git(source, "rev-parse", "--verify", "HEAD", check=False)
            if head.returncode == 0:
                if head.stdout.decode().strip() != lock.commit:
                    raise PreparationError(f"Refusing non-pinned HEAD in {source}")
                return {"mode": "existing", "source": str(source)}
            if not (source / ".git").is_dir() or (source / ".git").is_symlink():
                raise PreparationError(f"Cannot repair an external Git directory: {source}")
            if local_git(source, "ls-files").stdout:
                raise PreparationError(f"Refusing indexed files in incomplete checkout: {source}")
            origin = local_git(source, "remote", "get-url", "origin", check=False)
            if origin.returncode == 0 and origin.stdout.decode().strip() != lock.url:
                raise PreparationError(f"Refusing incomplete checkout with unexpected origin: {source}")
        if any(item.name != ".git" for item in source.iterdir()):
            raise PreparationError(f"Refusing nonempty incomplete source directory: {source}")

    # Select a complete pinned commit before initializing a new destination.
    rejected = []
    reference = None
    seen = set()
    for candidate in references:
        candidate = Path(candidate).resolve()
        if candidate in seen or candidate == source:
            continue
        seen.add(candidate)
        if not candidate.is_dir():
            rejected.append(f"{candidate}: missing")
            continue
        commit = local_git(candidate, "rev-parse", "--verify", f"{lock.commit}^{{commit}}", check=False)
        if commit.returncode == 0 and commit.stdout.decode().strip() == lock.commit:
            reference = candidate
            break
        rejected.append(f"{candidate}: pinned commit absent")
    if reference is None:
        raise PreparationError("No local pinned MindIE source cache found; network access is disabled. "
            "Use --reference-dir with the existing Laser source checkout. Checked: " + "; ".join(rejected))

    source.mkdir(parents=True, exist_ok=True)
    if not (source / ".git").exists():
        local_git(source, "init", "--quiet")
    local_git(source, "config", "core.autocrlf", "false")
    origin = local_git(source, "remote", "get-url", "origin", check=False)
    if origin.returncode:
        local_git(source, "remote", "add", "origin", lock.url)
    # This fetch is from an absolute local path, not from origin/GitHub. It
    # also handles the empty .git left by the previous failed network fetch.
    print(f"Reuse local MindIE Git objects: {reference} -> {source} (network disabled)", flush=True)
    local_git(source, "fetch", "--update-shallow", "--no-tags", "--no-recurse-submodules", "--", str(reference), lock.commit)
    fetched = local_git(source, "rev-parse", "FETCH_HEAD^{commit}").stdout.decode().strip()
    if fetched != lock.commit:
        raise PreparationError("Local object transfer returned an unexpected commit")
    local_git(source, "checkout", "--quiet", "--detach", lock.commit)
    return {"mode": "local_objects", "source": str(source), "reference": str(reference)}
