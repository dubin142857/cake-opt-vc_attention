"""Gain candidate wiring, unchanged-input gates and offline one-command flow."""
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
import profile_mindie_laser_scale_reuse as profile
import mindie_laser_gain_checks as checks
import mindie_laser_quant_backend as backend


@pytest.mark.parametrize("kind, candidate", [("gain-precompute", "gain"), ("score-prefetch", "prefetch")])
def test_candidate_binds_reuse_and_own_identity_and_namespace(tmp_path, kind, candidate):
    args = SimpleNamespace(experiment=kind, reuse_build="reuse.json",
                           **{candidate+"_build": candidate+".json"})
    specs = profile.experiment(args)
    assert specs["baseline"]["namespace"] == "vc_laser_quant_reuse"
    assert specs["reuse"]["namespace"] == "vc_laser_quant_"+candidate
    assert specs["baseline"]["path"] == "reuse.json"
    assert specs["reuse"]["path"] == candidate+".json"
    for spec in specs.values():
        assert spec["source"].IDENTITY["vendor"] == spec["namespace"]
    with pytest.raises(ValueError, match="Unknown"):
        profile.experiment(SimpleNamespace(experiment="unrecognized"))


def test_existing_scale_experiment_still_uses_original_baseline():
    specs = profile.experiment(SimpleNamespace(quant_build="old.json", reuse_build="reuse.json"))
    assert specs["baseline"]["namespace"] == "vc_laser_quant"
    assert specs["reuse"]["namespace"] == "vc_laser_quant_reuse"


@pytest.mark.parametrize("kind, candidate", [("gain-precompute", "gain"), ("score-prefetch", "prefetch")])
def test_candidate_provenance_uses_each_sources_validator(monkeypatch, tmp_path, kind, candidate):
    args = SimpleNamespace(experiment=kind, reuse_build="reuse.json",
                           preprocess_build="preprocess.json", **{candidate+"_build": candidate+".json"})
    seen = []
    def build_arguments(options, name):
        if name == "baseline":
            return "fp16"
        return SimpleNamespace(vendor_root=str(tmp_path / options.quant_build), path=options.quant_build)
    def provenance(provider, *, identity, source_validator):
        seen.append((provider.path, identity, source_validator))
        return {"checked": provider.path}
    monkeypatch.setattr(profile.quant, "build_arguments", build_arguments)
    monkeypatch.setattr(profile.quant, "quant_provenance", provenance)
    monkeypatch.setattr(profile.quant.baseline, "provenance", lambda p: {"fp16": p})
    import mindie_laser_quant_fused
    monkeypatch.setattr(mindie_laser_quant_fused, "validate_preprocess_manifest", lambda p: {"manifest": p})
    result = profile.verify_builds(args)
    specs = profile.experiment(args)
    for (key, spec), (path, identity, validator) in zip(specs.items(), seen):
        assert path == spec["path"] and identity == spec["source"].IDENTITY
        assert validator is spec["source"].validate_prepared_source
    assert result["baseline"] == {"checked": "reuse.json"}
    assert result["reuse"] == {"checked": candidate+".json"}


def test_explicit_namespace_loader_does_not_select_original_operator(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(torch.ops, "load_library", lambda path: None)
    lib = torch.library.Library("vc_gain_loader_test", "DEF")
    lib.define("la(Tensor q, Tensor k, Tensor v, Tensor qs, Tensor ks, float scale, int keys, bool expcast) -> Tensor")
    adapter = backend.load_quant_laser("candidate.so", torch_namespace="vc_gain_loader_test",
                                      v_quantize=False, expcast=False)
    assert adapter.native is torch.ops.vc_gain_loader_test.la.default
    assert adapter.metadata["dispatch"] == "torch.ops.vc_gain_loader_test.la.default"
    assert adapter.metadata["bridge_path"] == "candidate.so"


def test_block_gains_checks_all_rows_and_detects_wrong_q_pair_selection(monkeypatch):
    # Keep multi-head/block shapes but reduce feature allocation in this CPU test.
    original_randn = torch.randn
    monkeypatch.setattr(torch, "randn", lambda shape, **kw: original_randn((*shape[:3], 1), **kw))
    class Adapter:
        def __init__(self, wrong=False): self.wrong = wrong
        def prepare(self, q, k, v):
            b, h = q.shape[:2]
            return SimpleNamespace(q_scale=torch.ones(b, h, 4)*.02,
                                   k_scale=torch.ones(b, h, 34)*.03)
        def run_native(self, payload):
            q = payload.q_scale
            if self.wrong: q = q.roll(1, dims=-1)
            weights = torch.arange(1, payload.k_scale.shape[-1]+1)
            value = (payload.k_scale * weights).sum(-1, keepdim=True)
            return (q * value).repeat_interleave(128, dim=-1).unsqueeze(-1)
    report = {}
    checks.validate_block_gains({"baseline": Adapter(), "reuse": Adapter()}, "cpu", report, lambda: None)
    assert len(report["alternating_block_gains"]) == 8
    assert all(row["exact_match"] and row["pointers_unchanged"] for row in report["alternating_block_gains"])
    with pytest.raises(AssertionError, match="padded native"):
        checks.validate_block_gains({"baseline": Adapter(), "reuse": Adapter(True)}, "cpu", {}, lambda: None)


@pytest.mark.parametrize("kind", ["gain-precompute", "score-prefetch"])
def test_candidate_worker_launch_forwards_experiment_and_manifests(monkeypatch, tmp_path, kind):
    args = SimpleNamespace(_run_dir=str(tmp_path), experiment=kind, quant_build="old",
        reuse_build="reuse", gain_build="gain", prefetch_build="prefetch", preprocess_build="prep", baseline_build="fp16",
        bridge_manifest="bridge", device=2, warmup=5, repeats=10, trials=9, cpu_threads=4, seed=2026,
        shape=[1, 28, 37741, 128], trace=True)
    seen = []
    class Process:
        stdout = []
        def wait(self): return 0
    def popen(command, **kwargs): seen.append(command); return Process()
    monkeypatch.setattr(profile.subprocess, "Popen", popen)
    env = {name: str(tmp_path / name) for name in ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME")}
    result = profile.launch(args, "paired", env, tmp_path / "worker")
    command = result["command"]
    assert command[command.index("--experiment")+1] == kind
    assert command[command.index("--reuse-build")+1] == "reuse"
    assert command[command.index("--gain-build")+1] == "gain"
    assert command[command.index("--prefetch-build")+1] == "prefetch"
    assert "--trace" in command


@pytest.mark.parametrize("fail_stage", [None, "preflight", "build", "profile"])
@pytest.mark.parametrize("kind, candidate", [("gain-precompute", "gain"), ("score-prefetch", "prefetch")])
def test_one_command_builds_only_candidate_using_local_reuse_source(tmp_path, fail_stage, kind, candidate):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing")
    if not bash.is_file(): pytest.skip("Bash required")
    root = tmp_path / "space in project"
    for name in ("scripts", "results", "cann"):
        (root / name).mkdir(parents=True)
    for name in ("mindie_laser_quant_reuse_build", "mindie_laser_quant_preprocess_build", "mindie_laser_build", "mindie_laser_bridge"):
        (root / "results" / (name+".json")).write_text("{}")
    script_name = "scripts/run_mindie_laser_"+kind.replace("-", "_")+".sh"
    script = root / script_name
    script.write_bytes((ROOT / script_name).read_bytes())
    stub = root / "stub.py"
    stub.write_text("import json,sys\nfrom pathlib import Path\n"
        "a=sys.argv[1:]\ns='preflight' if a[0]=='-' else 'soc' if a[0]=='-c' else 'build' if 'build_' in a[1] else 'profile'\n"
        "with Path('calls.jsonl').open('a') as f:f.write(json.dumps({'stage':s,'args':a})+'\\n')\n"
        "if s=='soc':print('cached reuse source' if 'source_dir' in a[-1] else 'ascend910_93')\n"
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
    for call in calls:
        a = call["args"]
        if call["stage"] == "build":
            assert a[:2] == ["-u", "scripts/build_mindie_laser_quant_"+candidate+".py"]
            assert a[a.index("--reference-dir")+1] == "cached reuse source"
        if call["stage"] == "profile":
            assert a[a.index("--experiment")+1] == kind
            assert a[a.index("--shape")+1:a.index("--shape")+5] == ["1", "28", "37741", "128"]
            assert a[a.index("--device")+1] == "2" and "--trace" in a
