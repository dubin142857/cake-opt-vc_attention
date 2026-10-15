"""Bridge source/provenance checks, without a CANN compiler or NPU."""
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_mindie_laser_bridge as bridge


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    repo = tmp_path / "upstream"
    plugin = repo / "csrc/plugin"
    plugin.mkdir(parents=True)
    for name in bridge.ADAPTER_FILES:
        content = 'constexpr auto name = "aclnnLaserAttention";\n' if name == "la.cpp" else "// upstream unchanged\n"
        (plugin / name).write_text(content, encoding="utf-8")
    (repo / "LICENSE.md").write_text("Mulan PSL v2\n", encoding="utf-8")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.STDOUT).decode().strip()
    git("init", "-q")
    git("config", "core.autocrlf", "false")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("add", ".")
    git("commit", "-qm", "fixture")
    monkeypatch.setattr(bridge, "UPSTREAM_COMMIT", git("rev-parse", "HEAD"))
    return repo


def test_preserves_adapter_except_unique_native_symbol(upstream, tmp_path):
    destination = tmp_path / "adapter"
    records = bridge.prepare_adapter(upstream, destination)
    assert b'"aclnnVcLaserAttention"' in (destination / "la.cpp").read_bytes()
    assert b'"aclnnLaserAttention"' not in (destination / "la.cpp").read_bytes()
    for name in bridge.ADAPTER_FILES[1:]:
        assert records[name]["compiled_sha256"] == records[name]["upstream_sha256"]
    mtimes = {p.name: p.stat().st_mtime_ns for p in destination.iterdir()}
    assert bridge.prepare_adapter(upstream, destination) == records
    assert mtimes == {p.name: p.stat().st_mtime_ns for p in destination.iterdir()}
    assert (destination / "LICENSE.md").is_file()


def test_reads_committed_source_not_unknown_worktree_adapter(upstream, tmp_path):
    (upstream / "csrc/plugin/la.cpp").write_text("unreviewed code", encoding="utf-8")
    bridge.prepare_adapter(upstream, tmp_path / "adapter")
    assert b"unreviewed" not in (tmp_path / "adapter/la.cpp").read_bytes()


def test_refuses_unpinned_revision(upstream, tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "UPSTREAM_COMMIT", "0" * 40)
    with pytest.raises(ValueError, match="pinned"):
        bridge.prepare_adapter(upstream, tmp_path / "adapter")
