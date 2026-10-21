"""Row-max candidate routing, paired Laser baseline and native edge validation."""
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


def test_rowmax_configuration_does_not_import_bulk_source(monkeypatch):
    original = profile.importlib.import_module
    imported = []
    def load(name):
        imported.append(name)
        return original(name)
    monkeypatch.setattr(profile.importlib, "import_module", load)
    args = profile.parser("rowmax").parse_args([])
    specs = profile.specifications(args)
    assert tuple(specs) == ("vector_scale", "vector_rowmax")
    assert specs["vector_rowmax"]["path"] == "results/mindie_laser_quant_vscale_rowmax_build.json"
    assert specs["vector_rowmax"]["namespace"] == "vc_laser_quant_vscale_rowmax"
    assert specs["vector_rowmax"]["source"].IDENTITY["op_type"] == "VcLaserQuantVscaleRowmaxAttention"
    assert not any("bulk" in name for name in imported)


def test_thin_wrapper_selects_rowmax_and_own_report(monkeypatch):
    seen = []
    monkeypatch.setattr(profile, "parent", lambda args: seen.append(args) or 0)
    monkeypatch.setattr(sys, "argv", ["profile_mindie_laser_vector_rowmax.py"])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(ROOT / "scripts/profile_mindie_laser_vector_rowmax.py"), run_name="__main__")
    assert result.value.code == 0
    assert seen[0].candidate == "rowmax"
    assert seen[0].output == "results/mindie_laser_vector_rowmax.json"
    assert seen[0].shape == [1, 28, 37741, 128]


def test_rowmax_three_provider_environment_isolated(tmp_path):
    rows = {name: SimpleNamespace(vendor_root=str(tmp_path / name / "vendors" / name),
                marker_env=f"VC_{name.upper()}_MARKER_FILE")
            for name in (profile.ORIGINAL, profile.OLD, "vector_rowmax")}
    env = profile.worker_environment({}, rows, tmp_path / "run")
    assert env["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep) == [p.vendor_root for p in rows.values()]
    assert env["VC_VECTOR_ROWMAX_MARKER_FILE"].endswith("vector_rowmax_markers.jsonl")
    rows["other"] = rows.pop("vector_rowmax")
    with pytest.raises(ValueError, match="Unknown candidate"):
        profile.worker_environment({}, rows, tmp_path)


def test_rowmax_full_api_uses_original_laser_comparisons():
    from mindie_laser_quant_pairing import run_paired
    clock = [0.]
    durations = {profile.ORIGINAL: .085, profile.OLD: .080, "vector_rowmax": .075}
    def call(name):
        clock[0] += durations[name]
    timing = run_paired({name: lambda n=name: call(n) for name in durations},
        synchronize=lambda: None, warmup=0, repeats=1, trials=9, clock=lambda: clock[0])
    comparison = profile.compare_scopes(timing)
    assert comparison["vector_rowmax"]["baseline"] == profile.ORIGINAL
    assert comparison["rowmax_vs_previous"]["baseline"] == profile.OLD
    assert "bulk_vs_previous" not in comparison
    assert comparison["vector_rowmax"]["faster_trials"] == 9
    report = dict(timings={s: timing for s in profile.SCOPES},
        comparisons={s: comparison for s in profile.SCOPES},
        dense_fp32_samples={n: {"relative_rmse": .012} for n in durations},
        same_payload_all_output={"exact_match": True})
    text = profile.table(report)
    assert "vector_rowmax" in text and "1.1333x" in text and "1.200000" in text
    assert "bulk" not in text.lower()


@pytest.mark.parametrize("keys", profile.ROWMAX_KEYS)
@pytest.mark.parametrize("mode", profile.ROWMAX_MODES)
def test_rowmax_fixtures_cover_masked_tails_and_different_winning_blocks(keys, mode):
    from mindie_laser_vscale_reference import scale_scores
    payload, description = profile.rowmax_payload(keys, torch.bfloat16, 64, mode)
    scores = payload.q[:, :, :17].float() @ payload.k[:, :, :keys].float().transpose(-1, -2)
    assert bool(torch.isfinite(scores).all())
    assert not bool(payload.k[:, :, keys:].any())
    if mode == "zero":
        assert not bool(scores.any())
    elif mode == "negative":
        assert bool((scores < 0).all())
    else:
        assert scores.argmax(-1).tolist()[0] == description["assigned_winner_keys"]
        if keys >= 1024:
            assert len({i // 128 for head in description["assigned_winner_keys"] for i in head}) >= 8
    scaled = scale_scores(scores, payload.q_scale[..., :1, None],
                          payload.k_scale[..., :1, None], payload.scale)
    assert bool(torch.isfinite(scaled).all())
    assert scaled.abs().max().item() <= description["max_absolute_score_bound"]
    assert scaled.amax(-1).float().sub(scaled.amin(-1).float()).max().item() < 65504


def test_boundary_validator_checks_all_padded_native_outputs(monkeypatch):
    monkeypatch.setattr(profile, "ROWMAX_KEYS", (129,))
    monkeypatch.setattr(profile, "ROWMAX_MODES", ("moving_winner",))
    class Adapter:
        def __init__(self, corrupt=False): self.corrupt = corrupt
        def run_native(self, payload):
            output = torch.zeros_like(payload.q, dtype=torch.float32)
            output[..., 1] = 2
            if self.corrupt:
                output[..., -1, -1] = .001  # Only the padded Q tail differs.
            return output
        def postprocess(self, output, payload):
            return output[..., :payload.shape[2], :payload.shape[3]].to(payload.dtype)
    report = {}
    profile.validate_rowmax_boundaries(Adapter(), Adapter(), "cpu", report, lambda: None)
    assert len(report["rowmax_boundary_validation"]) == 2
    assert all(row["status"] == "passed" for row in report["rowmax_boundary_validation"])
    with pytest.raises(AssertionError, match="ALL padded"):
        profile.validate_rowmax_boundaries(Adapter(), Adapter(True), "cpu", {}, lambda: None)


def test_child_launch_keeps_rowmax_selection(monkeypatch, tmp_path):
    args = profile.parser("rowmax").parse_args(["--trace"])
    args._run_dir = str(tmp_path)
    class Process:
        stdout = []
        def wait(self): return 5
    monkeypatch.setattr(profile.subprocess, "Popen", lambda *a, **kw: Process())
    env = {key: str(tmp_path / key) for key in
           ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME")}
    result = profile.launch(args, env, tmp_path / "local")
    command = result["command"]
    assert command[command.index("--candidate") + 1] == "rowmax"
    assert command[command.index("--rowmax-build") + 1] == args.rowmax_build
    assert "--trace" in command
