"""Keep all timing paths on the selected kernel; reject misleading calibration."""
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import attention_phase_probe as phase
import profile_expcast_h200 as focus


def payload():
    return dict(q=torch.zeros(1, 1, 1, 64, dtype=torch.int8),
                k=torch.zeros(1, 1, 1, 64, dtype=torch.int8),
                v=torch.zeros(1, 1, 64, 128).to(torch.float8_e4m3fn),
                qs=torch.ones(1), ks=torch.ones(1), vs=torch.ones(1),
                means=torch.empty(0), permutation=None, block_values=False,
                output_type=1, scale=.125, expcast=True)


def test_balanced_routing_shares_core_payload_and_reprepares_every_full_call():
    base = payload()
    prepared = []
    def prepare(*qkv, **kw):
        assert qkv == (1, 2, 3) and kw == dict(expcast=True, v_smooth=False)
        state = {**base, "fresh": len(prepared)}
        prepared.append(state)
        return state
    vc = SimpleNamespace(prepare=prepare,
        forward_prepared=lambda s, **kw: ("installed", s),
        sageattn_vc=lambda *qkv, **kw: ("installed_full", kw))
    states, core, full = focus.make_calls(vc, None, None, (1, 2, 3), base,
        lambda s: ("balanced", s), lambda s: ("baseline", s))
    for key in ("q", "k", "v", "qs", "ks", "vs"):
        assert states["control"][key] is states["expcast"][key]
    for name, route in (("control", "installed"), ("expcast", "balanced"),
                        ("vc28_jit", "baseline"), ("vc28_installed", "installed")):
        assert core[name]()[0] == route
    assert core["vc28_jit"]()[1] is core["expcast"]()[1]
    for name in ("expcast", "vc28_jit", "expcast"):
        assert full[name]()[1] is prepared[-1]
    assert [s["fresh"] for s in prepared] == [0, 1, 2]
    assert full["control"]()[1]["expcast"] is False
    assert full["vc28_installed"]()[1]["expcast"] is True
    with pytest.raises(ValueError, match="same-build baseline"):
        focus.make_calls(vc, None, None, (), base, lambda s: s)


@pytest.mark.parametrize("cast", [False, True])
def test_staged_events_and_cupti_use_selected_kernel(monkeypatch, cast):
    state = payload()
    def staged(*args, **kw):
        assert kw == dict(expcast=cast, smooth=False)
        return state
    monkeypatch.setattr(focus, "staged_prepare", staged)
    vc = SimpleNamespace(forward_prepared=lambda s, **kw: ("installed", s))
    recorder = SimpleNamespace(call=lambda name, fn: (name, fn()))
    out = focus.staged_call((), SimpleNamespace(vc=vc), recorder, cast, lambda s: ("balanced", s))
    assert out == ("attention", ("balanced" if cast else "installed", state))


@pytest.mark.parametrize("detail", [False, True])
def test_plain_coarse_and_detail_sources_use_balanced_tree_with_unchanged_sync(detail):
    original = phase.generate_source(probability_detail=detail)
    balanced = phase.generate_source(probability_detail=detail, row_max="balanced_max")
    assert balanced.count("vc_probability_prepare_balanced_max<") == 1
    assert "vc_probability_prepare_integer<" not in balanced
    assert original.count("warpgroup_wait<0>()") == balanced.count("warpgroup_wait<0>()")
    assert original.count("__syncthreads()") == balanced.count("__syncthreads()")
    work = phase.instrument_probability_helpers("balanced_max")
    assert "vc_integer_row_max_balanced(scores[q],r)" in work
    assert "vc_integer_row_max(scores[q],r)" not in work
    assert "vc_probability_int_affine_tile_detail" in work
    assert "vc_probability_variant_tile<2,2,false,false>" in work


def test_backend_paths_and_bad_backend():
    old = focus.parse_args([])
    new = focus.parse_args(["--expcast-backend", "balanced_max", "--probability-detail"])
    assert old.output != new.output and old.trace_dir != new.trace_dir
    assert new.kernel_phases and new.output.name == "h200_expcast_balanced_profile.json"
    with pytest.raises(ValueError, match="Unknown"):
        phase.generate_source(row_max="typo")
    with pytest.raises(ValueError, match="Unknown"):
        phase.instrument_probability_helpers("typo")


def test_calibration_separates_compiler_drift_from_probe_cost():
    samples = {name+suffix: [value, value+1, value+2]
               for name in ("control", "expcast")
               for suffix, value in (("_production", 20.), ("_plain", 24.), ("_probe", 25.))}
    samples.update(expcast_detail_plain=[23., 24., 25.], expcast_detail=[26., 27., 28.])
    result = focus.probe_calibration(samples, detail=True)
    drift = result["expcast_production -> expcast_plain"]
    cost = result["expcast_plain -> expcast_probe"]
    assert drift["warning"] and drift["median_added_ms"] == 4.
    assert not cost["warning"] and cost["median_added_ms"] == 1.
    assert cost["paired_trial_added_ms"] == [1., 1., 1.]
    assert result["expcast_detail_plain -> expcast_detail"]["median_added_ms"] == 3.
    assert "GPU" in cost["scope"]


@pytest.mark.parametrize("bad", [[], [0.], [-1.], [float("nan")], [float("inf")]])
def test_paired_stats_fail_on_invalid_measurements(bad):
    with pytest.raises(ValueError):
        focus.expcast_savings({"old": [1.], "new": bad}, "old", "new")


def test_paired_stats_do_not_hide_regression_trials():
    result = focus.expcast_savings({"old": [23., 24., 25.], "new": [22., 25., 25.]}, "old", "new")
    assert result["faster_trials"] == 1 and result["trials"] == 3
    assert result["paired_trial_saved_ms"] == [1., -1., 0.]


def test_phase_session_calibrates_selected_production_in_each_trial(tmp_path, monkeypatch):
    args = focus.parse_args(["--expcast-backend", "balanced_max", "--probability-detail",
                             "--phase-trials", "2", "--trace-dir", str(tmp_path)])
    builds = []
    def load(cta, tile, detail=False, row_max="vc28"):
        builds.append((detail, row_max))
        return SimpleNamespace(detail=detail, resources=lambda *a: {}), {"row_max": row_max}
    monkeypatch.setattr(focus, "load_probe", load)
    selected = lambda s: torch.ones(1)
    def validate(ext, vc, device, forward=None):
        assert forward is selected
        return []
    monkeypatch.setattr(focus, "validate_probe", validate)
    def call(ext, state, trace):
        def run():
            clocks = torch.tensor([100, 200, 300, 1000, 3000, 4000, 5000, 6000, 7000, 8000]).repeat(2, 2, 1)
            if ext.detail:
                detail = torch.zeros(2, 2, 28, dtype=torch.int64)
                detail[..., :10] = clocks
                if state["expcast"]:
                    detail[:, 0, 10:27] = torch.arange(1100, 2800, 100)
                    detail[:, 0, 27], detail[:, 1, 27] = 1, 4
                else:
                    detail[..., 27] = 3
                clocks = detail
            return torch.ones(1), clocks
        return run
    monkeypatch.setattr(focus, "probe_call", call)
    monkeypatch.setattr(focus, "run_window", lambda *a: None)
    monkeypatch.setattr(focus, "measure_trial", lambda fn, *a: (fn(), 20.))
    stream = SimpleNamespace(wait_stream=lambda s: None, synchronize=lambda: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kw: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: nullcontext())
    calls = {"control": 0, "expcast": 0}
    def production(name):
        calls[name] += 1
        return torch.ones(1)
    base = {**payload(), "k": torch.zeros(1, 1, 129, 64, dtype=torch.int8)}
    states = {name: {**base, "expcast": name == "expcast"} for name in calls}
    core = {name: lambda n=name: production(n) for name in calls}
    result = focus.phase_diagnosis(states, core, None, args, torch.device("cpu"), lambda r: None, selected)
    assert builds == [(False, "balanced_max"), (True, "balanced_max")]
    assert result["status"] == "passed" and len(result["calibration"]) == 9
    for order in result["trial_order"]:
        assert len(order) == 8
        assert "expcast_production" in order and "control_production" in order
        assert "expcast_detail_plain" in order and "expcast_detail" in order
    for name in calls:
        assert len(result["samples"][name+"_production"]) == 2
        assert result["accuracy"][name+"_probe"]["exact"]
    assert len(list(tmp_path.glob("*.pt"))) == 6
