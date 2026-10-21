"""V-only PV candidate routing, exact native gates and queued payload reuse."""
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import profile_mindie_laser_vector_scale_bulk as profile
import mindie_laser_pv_prefetch_checks as checks


def test_selected_source_is_current_vscale_and_pv_candidate(monkeypatch):
    original = profile.importlib.import_module
    imported = []
    def load(name):
        imported.append(name)
        return original(name)
    monkeypatch.setattr(profile.importlib, "import_module", load)
    args = profile.parser("pv_prefetch").parse_args([])
    specs = profile.specifications(args)
    assert tuple(specs) == ("vector_scale", "vector_pv_prefetch")
    assert specs["vector_scale"]["path"] == "results/mindie_laser_quant_vscale_build.json"
    new = specs["vector_pv_prefetch"]
    assert new["path"] == "results/mindie_laser_quant_vscale_pv_prefetch_build.json"
    assert new["namespace"] == "vc_laser_quant_vscale_pv_prefetch"
    assert new["source"].IDENTITY["op_type"] == "VcLaserQuantVscalePvPrefetchAttention"
    assert not any("bulk" in name or "rowmax" in name for name in imported)


def test_thin_wrapper_fixed_defaults_and_own_report(monkeypatch):
    seen = []
    monkeypatch.setattr(profile, "parent", lambda args: seen.append(args) or 0)
    monkeypatch.setattr(sys, "argv", ["profile_mindie_laser_pv_prefetch.py"])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(ROOT / "scripts/profile_mindie_laser_pv_prefetch.py"), run_name="__main__")
    assert result.value.code == 0
    args = seen[0]
    assert args.candidate == "pv_prefetch"
    assert args.output == "results/mindie_laser_pv_prefetch.json"
    assert args.shape == [1, 28, 37741, 128]
    assert (args.warmup, args.repeats, args.trials) == (5, 10, 9)


def test_selected_build_failure_aborts_before_child(monkeypatch, tmp_path):
    args = profile.parser("pv_prefetch").parse_args(["--output", str(tmp_path / "report.json")])
    providers = {name: SimpleNamespace(vendor_root=str(tmp_path / name)) for name in
                 (profile.ORIGINAL, profile.OLD, "vector_pv_prefetch")}
    monkeypatch.setattr(profile, "resolve_providers", lambda a: providers)
    seen = []
    def failed(a, p):
        seen.append(a.pv_prefetch_build)
        raise RuntimeError("Candidate source manifest is stale")
    monkeypatch.setattr(profile, "verify_builds", failed)
    monkeypatch.setattr(profile, "launch", lambda *a: pytest.fail("Must fail before launching"))
    assert profile.parent(args) == 1
    report = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["candidate"] == "vector_pv_prefetch"
    assert seen == [args.pv_prefetch_build] and not report["workers"]


def test_pv_provenance_selects_own_validator_and_original(monkeypatch):
    args = profile.parser("pv_prefetch").parse_args([])
    specs = profile.specifications(args)
    providers = {name: SimpleNamespace(name=name) for name in
                 (profile.ORIGINAL, profile.OLD, "vector_pv_prefetch")}
    import mindie_laser_quant_fused
    monkeypatch.setattr(mindie_laser_quant_fused, "validate_preprocess_manifest", lambda p: {"prep": p})
    monkeypatch.setattr(profile.quant.baseline, "provenance", lambda p: {"original": p.name})
    monkeypatch.setattr(profile.quant, "quant_provenance", lambda p, **kw: kw)
    result = profile.verify_builds(args, providers)
    assert result[profile.ORIGINAL] == {"original": profile.ORIGINAL}
    for name, spec in specs.items():
        assert result[name]["identity"] == spec["source"].IDENTITY
        assert result[name]["source_validator"] is spec["source"].validate_prepared_source


def test_child_launch_keeps_pv_selection(monkeypatch, tmp_path):
    args = profile.parser("pv_prefetch").parse_args(["--trace"])
    args._run_dir = str(tmp_path)
    class Process:
        stdout = []
        def wait(self): return 5
    monkeypatch.setattr(profile.subprocess, "Popen", lambda *a, **kw: Process())
    env = {key: str(tmp_path / key) for key in
           ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME")}
    result = profile.launch(args, env, tmp_path / "local")
    command = result["command"]
    assert command[command.index("--candidate") + 1] == "pv_prefetch"
    assert command[command.index("--pv-prefetch-build") + 1] == args.pv_prefetch_build
    assert "--trace" in command


def test_three_provider_paths_and_markers_are_isolated(tmp_path):
    rows = {name: SimpleNamespace(vendor_root=str(tmp_path / name / "vendors" / name),
                marker_env=f"VC_{name.upper()}_MARKER_FILE")
            for name in (profile.ORIGINAL, profile.OLD, "vector_pv_prefetch")}
    env = profile.worker_environment({}, rows, tmp_path / "run")
    assert env["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep) == [p.vendor_root for p in rows.values()]
    assert env["VC_VECTOR_PV_PREFETCH_MARKER_FILE"].endswith("vector_pv_prefetch_markers.jsonl")
    rows["vector_pv_prefetch"].vendor_root = rows[profile.OLD].vendor_root
    with pytest.raises(ValueError, match="distinct"):
        profile.worker_environment({}, rows, tmp_path)


def test_primary_full_comparison_is_original_laser():
    from mindie_laser_quant_pairing import run_paired
    clock = [0.]
    durations = {profile.ORIGINAL: .085, profile.OLD: .080, "vector_pv_prefetch": .075}
    def call(name):
        clock[0] += durations[name]
    timing = run_paired({name: lambda n=name: call(n) for name in durations},
        synchronize=lambda: None, warmup=0, repeats=1, trials=9, clock=lambda: clock[0])
    comparison = profile.compare_scopes(timing)
    assert comparison["vector_pv_prefetch"]["baseline"] == profile.ORIGINAL
    assert comparison["pv_prefetch_vs_previous"]["baseline"] == profile.OLD
    assert "rowmax_vs_previous" not in comparison and "bulk_vs_previous" not in comparison
    report = dict(timings={s: timing for s in profile.SCOPES},
        comparisons={s: comparison for s in profile.SCOPES},
        dense_fp32_samples={n: {"relative_rmse": .012} for n in durations},
        same_payload_all_output={"exact_match": True})
    text = profile.table(report)
    assert "vector_pv_prefetch" in text and "1.1333x" in text
    assert "FULL API only" in text and "not kernel-only" in text


@pytest.mark.parametrize("case", checks.CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_boundary_payloads_distinguish_v_slots_and_zero_physical_padding(case, dtype):
    a, da = checks.boundary_payload(case, dtype, "A")
    b, db = checks.boundary_payload(case, dtype, "B")
    kn, qn, dim = case["keys"], case["queries"], case["dim"]
    assert da["logical_key_blocks"] == (kn + 127) // 128
    assert da["physical_key_blocks"] % 2 == 0
    assert da["history_section_input"] == (kn > 4096)
    assert da["logical_key_blocks"] <= da["physical_key_blocks"]
    for payload in (a, b):
        assert not bool(payload.k[..., kn:, :].any())
        assert not bool(payload.v[..., kn:, :].any())
        assert not bool(payload.q[..., qn:, :].any())
        assert not bool(payload.v[..., dim:].any())
        assert bool(torch.isfinite(payload.v).all())
    assert a.v.data_ptr() != b.v.data_ptr()
    assert not torch.equal(a.v, b.v)
    assert not torch.equal(a.k, b.k)
    assert not torch.equal(a.q_scale, b.q_scale)
    assert (a.v[..., :kn, 0] == 1).all() and (b.v[..., :kn, 0] == -1).all()
    if kn > 128:
        assert not torch.equal(a.v[..., 0, 2:dim], a.v[..., 128, 2:dim])


def test_queued_validation_defers_all_comparisons_and_checks_padding(monkeypatch):
    import profile_mindie_laser_quant
    case = dict(name="tail", batch=1, heads=2, queries=17, keys=129, dim=64)
    monkeypatch.setattr(checks, "CASES", (case,))
    events = []
    original_compare = profile_mindie_laser_quant.compare_all_rows
    def compare(a, b):
        events.append("compare")
        return original_compare(a, b)
    monkeypatch.setattr(profile_mindie_laser_quant, "compare_all_rows", compare)
    class Adapter:
        def __init__(self, corrupt=False): self.corrupt = corrupt
        def run_native(self, payload):
            events.append("launch")
            out = payload.v[..., :1, :].float().expand_as(payload.q).clone()
            if self.corrupt:
                out[..., -1, -1] += .0001
            return out
    report = {}
    checks.validate_pv_prefetch(Adapter(), Adapter(), "cpu", report, lambda: None)
    assert events[:8] == ["launch"] * 8 and events[8] == "compare"
    rows = report["pv_prefetch_boundary_validation"]
    assert len(rows) == 2 and all(row["status"] == "passed" for row in rows)
    assert [x["identity"] for x in rows[0]["queued_outputs"]] == list(checks.QUEUE)
    assert rows[0]["queued_outputs"][-1]["new_repeated_payload"]["exact_match"]
    with pytest.raises(AssertionError, match="ALL padded"):
        checks.validate_pv_prefetch(Adapter(), Adapter(True), "cpu", {}, lambda: None)


def test_repeated_payload_gate_catches_state_leak(monkeypatch):
    monkeypatch.setattr(checks, "CASES", (dict(name="tiny", batch=1, heads=2, queries=1, keys=1, dim=64),))
    class Adapter:
        def __init__(self): self.calls = 0
        def run_native(self, payload):
            self.calls += 1
            output = payload.v[..., :1, :].float().expand_as(payload.q).clone()
            if self.calls > 2: output[..., -1, -1] += .125
            return output
    # Both providers sharing the same state bug still fail A/B/A/B repeat gates.
    with pytest.raises(AssertionError, match="ALL padded"):
        checks.validate_pv_prefetch(Adapter(), Adapter(), "cpu", {}, lambda: None)
