"""Current-vscale provenance, deferred exact checks, and offline one-command route."""
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
import profile_mindie_laser_cube_phases as profile
import profile_mindie_laser_vscale_phases as entry


def test_wrapper_selects_current_family_without_mutating_legacy_defaults(monkeypatch):
    calls = []
    monkeypatch.setattr(profile, "parent", lambda args: calls.append(args) or 0)
    assert entry.main([]) == 0
    assert profile.main([]) == 0
    assert tuple(calls[0].families) == ("vscale", "fp16")
    assert calls[0].output == "results/mindie_laser_vscale_phases.json"
    assert tuple(calls[1].families) == ("int8", "fp16")
    assert calls[1].output == "results/mindie_laser_cube_phases.json"
    for bad in (["--families", "fp16", "vscale"], ["--families", "vscale", "int8"]):
        with pytest.raises(SystemExit):
            entry.main(bad)


def test_vscale_production_selects_current_manifest_and_source():
    args = SimpleNamespace(vscale_build="current.json")  # deliberately no reuse_build
    spec = profile.production_spec(args, "vscale")
    assert spec["build"] == "current.json"
    assert spec["namespace"] == "vc_laser_quant_vscale"
    assert spec["source"].IDENTITY["op_type"] == "VcLaserQuantVscaleAttention"
    assert profile.production_spec(args, "fp16") is None
    with pytest.raises(ValueError):
        profile.production_spec(args, "rowmax")


def test_provider_resolution_and_provenance_use_current_vscale(monkeypatch):
    import prepare_mindie_laser_quant_vscale_source as current
    import prepare_mindie_laser_cube_probe_source as source
    import mindie_laser_quant_fused as fused
    seen = []
    def build(args, kind):
        seen.append((args.quant_build, kind))
        return SimpleNamespace(vendor_root=args.quant_build, bridge_path=args.quant_build+".so")
    monkeypatch.setattr(profile.quant, "build_arguments", build)
    providers = profile.resolve_providers(SimpleNamespace(vscale_build="current", vscale_probe_build="probe"), "vscale")
    assert seen == [("current", "qk_int8_fused"), ("probe", "qk_int8_fused")]
    assert providers["baseline"].marker_env == current.MARKER_ENV
    assert providers["probe"].marker_env == source.marker_environment("vscale")
    checked = []
    def provenance(provider, *, identity, source_validator):
        checked.append((identity, source_validator))
        return {"identity": identity}
    monkeypatch.setattr(profile.quant, "quant_provenance", provenance)
    monkeypatch.setattr(fused, "validate_preprocess_manifest", lambda path: {"manifest": path})
    report = profile.verify_family(SimpleNamespace(vscale_build="current", vscale_probe_build="probe", preprocess_build="quantizer"), "vscale")
    assert checked[0] == (current.IDENTITY, current.validate_prepared_source)
    assert checked[1][0] == source.identity("vscale")
    assert report["preprocessor"] == {"manifest": "quantizer"}


@pytest.mark.parametrize("bad_mode", [None, 0, 1, 2])
def test_queued_scale_updates_are_preserved_and_each_mode_has_exact_gate(monkeypatch, bad_mode):
    import mindie_laser_vscale_reference as reference
    seen, checks = [], []
    state = {"queued": False}
    class Adapter:
        expcast = False
        v_quantize = False
        def run_native(self, payload):
            seen.append(("production", payload.q_scale.item(), payload.q_scale.data_ptr()))
            return payload.q_scale.clone().reshape(1, 1, 1, 1).float()
    payload = SimpleNamespace(q_scale=torch.ones(1), k_scale=torch.ones(1), keys=1,
                              q=None, k=None, v=None, scale=1.)
    def native(q, k, v, qs, ks, scale, keys, expcast, mode):
        seen.append((mode, qs.item(), qs.data_ptr()))
        out = qs.clone().reshape(1, 1, 1, 1).float()
        if mode == bad_mode:
            out += 1
        return out, object()
    def checked(*args, **kwargs):
        assert state["queued"], "trace copy synchronized before all same-address updates were queued"
        checks.append(args[3])
        return []
    def oracle(adapters, device, report, save):
        assert adapters["baseline"].__class__ is Adapter
        for step in range(5):
            if step == 2:
                payload.q_scale.mul_(1.5)
            if step == 3:
                payload.k_scale.mul_(.75)
            adapters["reuse"].run_native(payload)
        assert len(seen) == 20 and not checks
        assert {row[2] for row in seen} == {payload.q_scale.data_ptr()}
        assert [row[1] for row in seen[::4]] == [1., 1., 1.5, 1.5, 1.5]
        state["queued"] = True
        save()
    monkeypatch.setattr(reference, "validate_vector_scale", oracle)
    monkeypatch.setattr(profile, "checked_trace", checked)
    report = {}
    if bad_mode is None:
        profile.validate_vscale(Adapter(), native, "cpu", report, lambda: None)
        assert len(report["vscale_probe_exact_validation"]) == 15
        assert all(row["exact_match"] for row in report["vscale_probe_exact_validation"])
        assert len(checks) == 15
    else:
        with pytest.raises(AssertionError, match="ALL padded native outputs"):
            profile.validate_vscale(Adapter(), native, "cpu", report, lambda: None)
        assert not report["vscale_probe_exact_validation"][-1]["exact_match"]


def test_current_worker_failure_stops_before_original_worker(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(profile, "verify_family", lambda *args: {"checked": True})
    monkeypatch.setattr(profile, "resolve_providers", lambda *args: {})
    def launch(args, family, providers):
        calls.append(family)
        return dict(status="failed", returncode=9, worker_log=family+".log")
    monkeypatch.setattr(profile, "launch", launch)
    output = tmp_path / "report.json"
    assert profile.parent(SimpleNamespace(output=output, families=("vscale", "fp16"))) == 1
    assert calls == ["vscale"]
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["workers"]["vscale"]["returncode"] == 9
    assert "fp16" not in report["workers"]


def test_wrapper_import_does_not_initialize_npu_or_torch():
    result = subprocess.run([sys.executable, "-c",
        "import sys;sys.path.insert(0,'scripts');import profile_mindie_laser_vscale_phases;"
        "assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fail_stage", [None, "preflight", "vscale", "fp16", "profile"])
def test_offline_driver_has_no_legacy_rowmax_or_bulk_dependency(tmp_path, fail_stage):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing")
    if not bash.is_file():
        pytest.skip("Bash required")
    root = tmp_path / "space in project"
    for name in ("scripts", "results", "cann"):
        (root / name).mkdir(parents=True)
    for name in ("mindie_laser_quant_vscale_build", "mindie_laser_quant_preprocess_build", "mindie_laser_build", "mindie_laser_bridge"):
        (root / "results" / (name+".json")).write_text("{}")
    script = root / "scripts/run_mindie_laser_vscale_phases.sh"
    script.write_bytes((ROOT / "scripts/run_mindie_laser_vscale_phases.sh").read_bytes())
    stub = root / "stub.py"
    stub.write_text("import json,sys\nfrom pathlib import Path\n"
        "a=sys.argv[1:]\ns='preflight' if a[0]=='-' else 'cache' if a[0]=='-c' else a[a.index('--kind')+1] if '--kind' in a else 'profile'\n"
        "with Path('calls.jsonl').open('a') as f:f.write(json.dumps({'stage':s,'args':a})+'\\n')\n"
        "if s=='cache':print('local current source' if 'source_dir' in a[-1] else 'ascend910_93')\n"
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
    else:
        assert [c["stage"] for c in calls if c["stage"] != "cache"] == ["preflight", "vscale", "fp16", "profile"]
    for call in calls:
        a = call["args"]
        assert not any("rowmax" in s or "bulk" in s or "reuse_build" in s for s in a)
        if call["stage"] in ("vscale", "fp16"):
            assert a[:2] == ["-u", "scripts/build_mindie_laser_cube_probe.py"]
            assert a[a.index("--reference-dir")+1] == "local current source"
        if call["stage"] == "profile":
            assert a[:2] == ["-u", "scripts/profile_mindie_laser_vscale_phases.py"]
            assert a[a.index("--shape")+1:a.index("--shape")+5] == ["1", "28", "37741", "128"]
            assert a[a.index("--device")+1] == "2"
