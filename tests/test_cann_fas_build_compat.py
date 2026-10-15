"""Exercise the actual child-Bash logging fallback without CANN or an NPU."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
FILTER = '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'


@pytest.fixture
def shell(tmp_path):
    if os.name == "nt":
        bash = Path("C:/Program Files/Git/bin/bash.exe")
        if not bash.is_file():
            pytest.skip("Git Bash is required on Windows")
    else:
        bash = shutil.which("bash")
        if not bash:
            pytest.skip("Bash is required")
    # Deliberately exclude any installed gawk. Keep only a cat forwarder in PATH.
    cat = tmp_path / "cat"
    cat.write_bytes(b'#!/bin/bash\nexec /bin/cat "$@"\n')
    cat.chmod(0o755)

    def run(script, *args):
        # Git Bash accepts Windows paths for cd, but PATH needs POSIX spelling.
        script = 'fixture_bin=$(cd "$1" && pwd -P)\n' + script
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("BASH_FUNC_") and key not in ("BASH_ENV", "ENV", "SHELLOPTS")}
        return subprocess.run(
            [str(bash), "--noprofile", "--norc", "-c", script, "compat-test",
             str(tmp_path), str(ROOT / "scripts/cann_fas_build_compat.sh"), FILTER, *args],
            env=env, capture_output=True, text=True, encoding="utf-8", timeout=15,
        )
    return run


def test_missing_gawk_child_pipeline_preserves_logs_and_build_failure(shell):
    result = shell(r'''
set -euo pipefail
export PATH="$fixture_bin"
source "$2"
/bin/bash -c 'set -o pipefail; { printf "%s\n" "  compiler error: %s \\ literal" "" "last line"; exit 23; } | gawk "$1"' child "$3"
''')
    assert result.returncode == 23
    assert result.stdout == "  compiler error: %s \\ literal\n\nlast line\n"
    assert "without timestamp prefixes" in result.stderr


def test_missing_gawk_success_and_unknown_expression(shell):
    success = shell(r'''
export PATH="$fixture_bin"
source "$2"
/bin/bash -c 'printf "%s" "no trailing newline" | gawk "$1"' child "$3"
''')
    assert success.returncode == 0 and success.stdout == "no trailing newline"
    rejected = shell(r'''
export PATH="$fixture_bin"
source "$2"
/bin/bash -c 'gawk "{print NF}"'
''')
    assert rejected.returncode == 64
    assert "Unsupported gawk invocation" in rejected.stderr


def test_existing_gawk_is_preserved(shell):
    result = shell(r'''
export PATH="$fixture_bin"
gawk() { printf "%s\n" "existing-gawk:$*"; }
export -f gawk
source "$2"
/bin/bash -c 'gawk "$1"' child "$3"
''')
    assert result.returncode == 0
    assert result.stdout == f"existing-gawk:{FILTER}\n"
    assert not result.stderr
