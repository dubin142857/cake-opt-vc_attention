"""CPU-only checks for the isolated AscendC bootstrap and cmake forwarding."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tilelang_attention_npu_setup_cpu", ROOT / "scripts/setup_tilelang_attention_npu.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


@pytest.mark.parametrize("arguments", [
    ["--build", ".", "--config", "Release", "-j128"],
    ["--build", "build", "--parallel", "96"],
    ["--build", "build", "-j", "48"],
    ["--build", "build", "--parallel=96"],
])
def test_cmake_wrapper_replaces_upstream_parallelism(arguments):
    result = setup.limit_build_jobs(arguments, 4)
    assert result[-2:] == ["--parallel", "4"]
    assert all(value not in result for value in ("-j128", "96", "48", "--parallel=96"))
    assert result[:2] == arguments[:2]


def test_cmake_configure_arguments_are_unchanged():
    arguments = ["/source", "-DUSE_ASCEND=ON", "-DUSE_CUDA=OFF"]
    assert setup.limit_build_jobs(arguments) == arguments
    assert setup.limit_build_jobs(["--build", ".", "--", "VERBOSE=1"]) == [
        "--build", ".", "--parallel", "4", "--", "VERBOSE=1"]


def test_existing_unowned_venv_is_never_modified(tmp_path):
    sentinel = tmp_path / "user-file"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unowned"):
        setup.require_owned_venv(tmp_path)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(tmp_path.iterdir()) == [sentinel]


def test_owned_venv_must_belong_to_this_workspace(tmp_path):
    (tmp_path / setup.MARKER).write_text(json.dumps({"owner": setup.OWNER, "root": "/another"}),
                                        encoding="utf-8")
    with pytest.raises(RuntimeError, match="different owner"):
        setup.require_owned_venv(tmp_path)


def test_preserves_torch_binaries_and_versions():
    before = {"torch_version": "2.6.0+cpu", "torch_file": "/old/torch/__init__.py",
              "torch_npu_version": "2.6.0.post3", "torch_npu_file": "/old/torch_npu/__init__.py"}
    setup.assert_torch_preserved(before, dict(before))
    for key in before:
        with pytest.raises(RuntimeError, match="preserve"):
            setup.assert_torch_preserved(before, {**before, key: "changed"})


def test_outer_venv_stack_is_exposed_without_reinstallation(monkeypatch, tmp_path):
    site_dir = tmp_path / "lib/python3.10/site-packages"
    monkeypatch.setattr(setup, "probe", lambda *args, **kwargs: {"purelib": str(site_dir)})
    parent = tmp_path / "outer/lib/python3.10/site-packages"
    before = {"torch_file": str(parent / "torch/__init__.py"),
              "torch_npu_file": str(parent / "torch_npu/__init__.py")}
    setup.expose_existing_stack("python", before, tmp_path)
    paths = (site_dir / "tilelang_attention_existing_npu_stack.pth").read_text(encoding="utf-8").splitlines()
    assert paths == [str(parent.resolve())]
    monkeypatch.setattr(setup, "probe", lambda *args, **kwargs: {"purelib": str(tmp_path.parent)})
    with pytest.raises(RuntimeError, match="outside"):
        setup.expose_existing_stack("python", before, tmp_path)


def test_dependency_install_list_cannot_request_torch():
    assert not any(requirement.lower().startswith(("torch", "tilelang"))
                   for requirement in setup.PYTHON_REQUIREMENTS)


def test_probe_output_ignores_import_logging():
    assert setup.parse_probe_output("log\n" + setup.JSON_PREFIX + '{"ok":true}\n') == {"ok": True}
    with pytest.raises(RuntimeError, match="JSON"):
        setup.parse_probe_output("no sentinel")


def test_cuda_or_unpinned_backend_is_rejected(tmp_path):
    backend = {"file": str(tmp_path / "tilelang/__init__.py"), "version": "0.1.15"}
    with pytest.raises(RuntimeError, match="pinned"):
        setup.check_local_backend(backend, tmp_path)
    backend["version"] = "0.1.4+" + setup.SOURCE_COMMIT
    backend["commit_evidence"] = [setup.SOURCE_COMMIT]
    setup.check_local_backend(backend, tmp_path)
    with pytest.raises(RuntimeError, match="inherited"):
        setup.check_local_backend(backend, tmp_path / "another")


def test_sourceable_shell_does_not_read_callers_arguments():
    source = (ROOT / "scripts/setup_tilelang_attention_npu.sh").read_text(encoding="utf-8")
    assert '"$@"' not in source and '"${@}"' not in source
    assert 'export TILELANG_NPU_PYTHON="$selected_python"' in source
    assert "--no-deps" in inspect_source()


def test_probe_command_logging_hides_multiline_python(monkeypatch, capsys):
    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="{}"))
    setup.run(["python", "-c", "private_long_probe_code\nsecond_line"], capture=True)
    output = capsys.readouterr().out
    assert "<environment probe>" in output
    assert "private_long_probe_code" not in output


def test_build_environment_clears_only_backend_overrides(monkeypatch, tmp_path):
    for variable in setup.BACKEND_ENV_OVERRIDES:
        monkeypatch.setenv(variable, "stale-backend")
    monkeypatch.setenv("ASCEND_HOME_PATH", "/selected-cann")
    monkeypatch.setenv("TL_RUN_MODE", "sim")
    monkeypatch.setattr(setup.shutil, "which", lambda *args, **kwargs: "/real/cmake")
    environment = setup.build_environment(tmp_path / "venv/bin/python", tmp_path)
    assert not any(variable in environment for variable in setup.BACKEND_ENV_OVERRIDES)
    assert environment["ASCEND_HOME_PATH"] == "/selected-cann"
    assert environment["TL_RUN_MODE"] == "npu"
    assert environment["USE_ASCEND"] == "true"
    assert environment["USE_LLVM"] == environment["USE_ROCM"] == "false"
    source = (tmp_path / ".setup-tools/cmake").read_text(encoding="utf-8")
    compile(source, "cmake-wrapper", "exec")


def inspect_source():
    return (ROOT / "scripts/setup_tilelang_attention_npu.py").read_text(encoding="utf-8")
