"""Scope, provider isolation, cache-read validation and failure preservation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mindie_laser_scale_reuse_checks as checks
import profile_mindie_laser_scale_reuse as profile


def providers_at(root):
    return {name: SimpleNamespace(vendor_root=str(root / "vendors" / name),
            bridge_path=str(root / name / "bridge.so"), marker_env="VC_"+name.upper()+"_MARKER_FILE")
            for name in ("baseline", "reuse")}


def library_paths(providers):
    paths = []
    for p in providers.values():
        paths.extend([str(Path(p.vendor_root) / "op_api/lib/libcust_opapi.so"),
                      str(Path(p.vendor_root) / "op_impl/ai_core/tbe/op_tiling/lib/libcust_opmaster_rt2.0.so"),
                      str(Path(p.bridge_path).resolve())])
    return paths


def test_two_known_providers_allowed_but_third_is_rejected(tmp_path):
    providers = providers_at(tmp_path)
    paths = library_paths(providers)
    assert len(checks.verify_provider_paths(paths, providers)) == 6
    rogue = str(tmp_path / "vendors/other/op_api/lib/libcust_opapi.so")
    with pytest.raises(RuntimeError, match="contamination"):
        checks.verify_provider_paths(paths+[rogue], providers)
    with pytest.raises(RuntimeError, match="requires vendor"):
        checks.verify_provider_paths([p for p in paths if "op_tiling" not in p], providers)
    with pytest.raises(RuntimeError, match="bridge absent"):
        checks.verify_provider_paths([p for p in paths if p != providers["reuse"].bridge_path], providers)


def test_environment_scrubs_ambient_vendor_and_markers(tmp_path):
    providers = providers_at(tmp_path)
    alien = tmp_path / "vendors/alien"
    env = checks.paired_environment({"ASCEND_CUSTOM_OPP_PATH": str(alien),
        "LD_PRELOAD": "alien.so", "LD_LIBRARY_PATH": str(alien / "op_api/lib"),
        "VC_OLD_MARKER_FILE": "old"}, providers, tmp_path / "run")
    assert env["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep) == [str(Path(p.vendor_root).resolve()) for p in providers.values()]
    assert "alien" not in env["LD_LIBRARY_PATH"]
    assert "LD_PRELOAD" not in env and "VC_OLD_MARKER_FILE" not in env
    for name, p in providers.items():
        assert env[p.marker_env].endswith(name+"_markers.jsonl")
    with pytest.raises(ValueError, match="distinct"):
        checks.paired_environment({}, {"baseline": providers["baseline"], "reuse": providers["baseline"]}, tmp_path)


def test_fixed_native_and_prepared_share_same_object_without_prepare():
    seen = []
    class Adapter:
        def __init__(self, name): self.name = name
        def run_native(self, payload): seen.append((self.name, "native", payload))
        def run_prepared(self, payload): seen.append((self.name, "prepared", payload))
        def __call__(self, *inputs): seen.append((self.name, "full", inputs))
    adapters = {name: Adapter(name) for name in ("baseline", "reuse")}
    payload, inputs = object(), (object(), object(), object())
    calls = checks.make_calls(adapters, payload, inputs)
    for scope in ("native", "prepared", "full"):
        for name in adapters:
            calls[scope][name]()
            assert seen[-1] == (name, scope, inputs if scope == "full" else payload)


def test_reuse_validation_detects_stale_scalar_cache(monkeypatch):
    # Use tiny fake payloads, but real device-ordered copy_/native comparison logic.
    original_randn = torch.randn
    def small_randn(shape, **kwargs):
        return original_randn((1, 1, 2, 2), **kwargs)
    monkeypatch.setattr(torch, "randn", small_randn)
    class Adapter:
        def __init__(self, stale=False): self.stale = stale; self.saved = None
        def prepare(self, q, k, v):
            return SimpleNamespace(q=q.clone(), k=k.clone(), v=v.clone(),
                q_scale=q.abs().amax().reshape(1), k_scale=k.abs().amax().reshape(1))
        def run_native(self, payload):
            gain = payload.q_scale * payload.k_scale
            if self.stale:
                if self.saved is None: self.saved = gain.clone()
                gain = self.saved
            return torch.ones(1, 1, 2, 2) * gain
    report = {}
    checks.validate_scale_reuse({"baseline": Adapter(), "reuse": Adapter()}, "cpu", report, lambda: None)
    assert len(report["scale_storage_reuse"]) == 10
    assert all(row["exact_match"] and row["pointers_unchanged"] for row in report["scale_storage_reuse"])
    with pytest.raises(AssertionError, match="in-place"):
        checks.validate_scale_reuse({"baseline": Adapter(), "reuse": Adapter(stale=True)}, "cpu", {}, lambda: None)


def test_parent_failure_before_launch_preserves_report(tmp_path, monkeypatch):
    def fail(args): raise ValueError("stale candidate")
    monkeypatch.setattr(profile, "verify_builds", fail)
    monkeypatch.setattr(profile, "launch", lambda *args: pytest.fail("must not launch"))
    output = tmp_path / "result.json"
    assert profile.parent(SimpleNamespace(output=output)) == 1
    assert "stale candidate" in json.loads(output.read_text(encoding="utf-8"))["error"]


def test_import_parent_does_not_initialize_torch():
    code = "import sys;sys.path.insert(0,'scripts');import profile_mindie_laser_scale_reuse;assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fail_stage", [None, "preflight", "build", "profile"])
def test_one_command_only_builds_candidate_and_stops_on_failure(tmp_path, fail_stage):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing")
    if not bash.is_file(): pytest.skip("Bash required")
    root = tmp_path / "space in project"
    for name in ("scripts", "results", "cann"):
        (root / name).mkdir(parents=True)
    for name in ("mindie_laser_quant_build", "mindie_laser_quant_preprocess_build", "mindie_laser_build", "mindie_laser_bridge"):
        (root / "results" / (name+".json")).write_text("{}")
    script = root / "scripts/run_mindie_laser_scale_reuse.sh"
    script.write_bytes((ROOT / "scripts/run_mindie_laser_scale_reuse.sh").read_bytes())
    stub = root / "stub.py"
    stub.write_text("import json,sys\nfrom pathlib import Path\n"
        "a=sys.argv[1:]\ns='preflight' if a[0]=='-' else 'soc' if a[0]=='-c' else 'build' if 'build_' in a[1] else 'profile'\n"
        "with Path('calls.jsonl').open('a') as f:f.write(json.dumps({'stage':s,'args':a})+'\\n')\n"
        "if s=='soc':print('ascend910_93')\n"
        f"raise SystemExit(23 if s=={fail_stage!r} else 0)\n", encoding="utf-8")
    wrapper = root / "mock-python"
    wrapper.write_text(f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "{stub.as_posix()}" "$@"\n', encoding="utf-8", newline="\n")
    wrapper.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("BASH_ENV", "ENV")}
    env.update(PYTHON_BIN=wrapper.as_posix(), ASCEND_HOME_PATH=(root / "cann").as_posix(), NPU_DEVICE="2")
    result = subprocess.run([str(bash), "--noprofile", "--norc", str(script)], cwd=root, env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == (23 if fail_stage else 0), result.stdout+result.stderr
    calls = [json.loads(row) for row in (root / "calls.jsonl").read_text().splitlines()]
    if fail_stage:
        assert calls[-1]["stage"] == fail_stage
    if any(c["stage"] == "build" for c in calls):
        build = next(c["args"] for c in calls if c["stage"] == "build")
        assert build[:2] == ["-u", "scripts/build_mindie_laser_quant_reuse.py"]
    if any(c["stage"] == "profile" for c in calls):
        command = calls[-1]["args"]
        assert command[command.index("--shape")+1:command.index("--shape")+5] == ["1", "28", "37741", "128"]
        assert command[command.index("--device")+1] == "2" and "--trace" in command
