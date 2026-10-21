"""P lookahead must compare against validated V lookahead, not Vector-scale."""
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
import mindie_laser_pv_p_prefetch_checks as checks

OLD, NEW = "vector_pv_prefetch", "vector_pv_p_prefetch"
NAMES = (profile.ORIGINAL, OLD, NEW)


def test_source_and_bridge_selection_is_v_prefetch_parent_and_p_candidate():
    args = profile.parser("pv_p_prefetch").parse_args([])
    specs = profile.specifications(args)
    assert tuple(specs) == (OLD, NEW)
    assert specs[OLD]["path"] == args.pv_prefetch_build
    assert specs[NEW]["path"] == args.pv_p_prefetch_build
    assert specs[OLD]["namespace"] == "vc_laser_quant_vscale_pv_prefetch"
    assert specs[NEW]["namespace"] == "vc_laser_quant_vscale_pv_p_prefetch"
    assert specs[NEW]["source"].IDENTITY["op_type"] == "VcLaserQuantVscalePvPPrefetchAttention"
    assert specs[OLD]["source"].IDENTITY["op_type"] == "VcLaserQuantVscalePvPrefetchAttention"
    config = profile.candidate_config(args)
    assert config["contract_key"] == "pv_p_prefetch_contract"
    assert config["old_contract_key"] == "pv_prefetch_contract"


def test_old_candidate_triples_are_unchanged_and_wrong_parent_is_rejected():
    for candidate in ("vector_scale_bulk", "vector_rowmax", OLD):
        assert profile.variant_names(candidate) == (profile.ORIGINAL, "vector_scale", candidate)
        assert profile.group_candidate(profile.variant_names(candidate)) == candidate
    assert profile.variant_names(NEW) == NAMES
    assert profile.group_candidate(dict.fromkeys(reversed(NAMES))) == NEW
    with pytest.raises(ValueError, match="selected INT8 parent"):
        profile.group_candidate((profile.ORIGINAL, "vector_scale", NEW))


def test_resolve_paths_markers_and_provenance_select_new_parent(monkeypatch, tmp_path):
    args = profile.parser("pv_p_prefetch").parse_args([])
    args._run_dir = str(tmp_path)
    def resolve(options, name):
        key = "original" if name == "baseline" else Path(options.quant_build).stem
        return SimpleNamespace(vendor_root=str(tmp_path/key), selected=name,
                               quant_build=getattr(options, "quant_build", None))
    monkeypatch.setattr(profile.quant, "build_arguments", resolve)
    providers = profile.resolve_providers(args)
    assert tuple(providers) == NAMES
    assert providers[OLD].quant_build == args.pv_prefetch_build
    assert providers[NEW].quant_build == args.pv_p_prefetch_build
    assert len({p.marker_env for p in providers.values()}) == 3
    env = profile.worker_environment({}, providers, tmp_path)
    assert env[providers[OLD].marker_env].endswith(OLD+"_markers.jsonl")
    assert env[providers[NEW].marker_env].endswith(NEW+"_markers.jsonl")
    assert env["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep) == [p.vendor_root for p in providers.values()]
    import mindie_laser_quant_fused
    monkeypatch.setattr(mindie_laser_quant_fused, "validate_preprocess_manifest", lambda p: {})
    monkeypatch.setattr(profile.quant.baseline, "provenance", lambda p: {"original": True})
    monkeypatch.setattr(profile.quant, "quant_provenance", lambda p, **kw: kw)
    evidence = profile.verify_builds(args, providers)
    specs = profile.specifications(args)
    for name in (OLD, NEW):
        assert evidence[name]["identity"] == specs[name]["source"].IDENTITY
        assert evidence[name]["source_validator"] is specs[name]["source"].validate_prepared_source


def test_shared_payload_native_and_fresh_full_calls_use_v_parent(monkeypatch):
    import mindie_laser_fp16_trace_worker
    log = []
    class Adapter:
        def __init__(self, name): self.name = name
        def __call__(self, *xs): log.append((self.name, "full", xs))
        def run_prepared(self, payload): log.append((self.name, "prepared", payload))
        def run_native(self, payload): log.append((self.name, "native", payload))
    monkeypatch.setattr(mindie_laser_fp16_trace_worker, "native_call", lambda a, p: a.run_native(p))
    adapters = {n: Adapter(n) for n in NAMES}
    original_payload, fixed_payload = object(), object()
    payloads = {profile.ORIGINAL: original_payload, OLD: fixed_payload, NEW: fixed_payload}
    inputs = (object(), object(), object())
    calls = profile.make_calls(adapters, payloads, inputs)
    for scope in calls:
        for call in calls[scope].values(): call()
    assert (OLD, "native", fixed_payload) in log and (NEW, "native", fixed_payload) in log
    assert (profile.ORIGINAL, "native", original_payload) in log
    assert (NEW, "full", inputs) in log
    with pytest.raises(ValueError, match="identical payload"):
        profile.make_calls(adapters, dict(payloads, vector_pv_p_prefetch=object()), inputs)


def test_comparisons_and_report_keep_original_primary_and_v_only_previous():
    from mindie_laser_quant_pairing import run_paired
    clock = [0.]
    durations = dict(zip(NAMES, (.085, .079, .077)))
    def call(name): clock[0] += durations[name]
    timing = run_paired({name: lambda n=name: call(n) for name in NAMES},
        synchronize=lambda: None, warmup=0, repeats=1, trials=9, clock=lambda: clock[0])
    comparison = profile.compare_scopes(timing)
    assert comparison[NEW]["baseline"] == profile.ORIGINAL
    assert comparison["pv_p_prefetch_vs_previous"]["baseline"] == OLD
    assert "vector_scale" not in comparison
    report = dict(timings={s: timing for s in profile.SCOPES},
        comparisons={s: comparison for s in profile.SCOPES},
        dense_fp32_samples={n: {"relative_rmse": .012} for n in NAMES},
        same_payload_all_output={"exact_match": True})
    text = profile.table(report)
    assert OLD in text and NEW in text and "1.1039x" in text
    assert "vector_scale " not in text and "FULL API only" in text


def test_wrapper_defaults_and_parent_failure_evidence(monkeypatch, tmp_path):
    observed = []
    real_parent = profile.parent
    monkeypatch.setattr(profile, "parent", lambda args: observed.append(args) or 0)
    monkeypatch.setattr(sys, "argv", ["profile_mindie_laser_pv_p_prefetch.py"])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(ROOT/"scripts/profile_mindie_laser_pv_p_prefetch.py"), run_name="__main__")
    assert result.value.code == 0
    args = observed[0]
    assert args.candidate == "pv_p_prefetch"
    assert args.output == "results/mindie_laser_pv_p_prefetch.json"
    assert args.shape == [1, 28, 37741, 128]
    assert (args.warmup, args.repeats, args.trials) == (5, 10, 9)
    args.output = str(tmp_path/"failure.json")
    monkeypatch.setattr(profile, "resolve_providers", lambda args: {})
    def fail(*a): raise RuntimeError("Parent V-prefetch manifest is stale")
    monkeypatch.setattr(profile, "verify_builds", fail)
    monkeypatch.setattr(profile, "launch", lambda *a: pytest.fail("Must verify before launching"))
    assert real_parent(args) == 1
    report = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert report["previous_int8_variant"] == OLD and report["candidate"] == NEW
    assert report["status"] == "failed" and not report["workers"]


@pytest.mark.parametrize("case", checks.P_CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_p_fixture_has_opposite_probabilities_in_adjacent_query_and_key_tiles(case, dtype):
    payload, info = checks.probability_payload(case, dtype, "A")
    kn = case["keys"]
    # Decode three query representatives and all valid K, independently of
    # candidate kernels. This proves P differences, not only V decorations.
    q = payload.q[0, 0, [0, 128, 256]].float() * .5
    k = payload.k[0, 0, :kn].float() * .5
    scores = q @ k.T / 128**.5
    assert scores[0, 0] > 5 and scores[0, 128] < -5
    assert scores[1, 0] < -5 and scores[1, 128] > 5
    assert scores[2, 128] > 5 and scores[2, 256] < -5
    probs = scores.softmax(-1)
    even = (torch.arange(kn)//128) % 2 == 0
    assert probs[0, even].sum() > .999
    assert probs[1, ~even].sum() > .999
    assert not torch.equal(probs[0], probs[2])
    assert info["compared_to"].startswith("V-only")
    assert not payload.q[..., 257:, :].any()
    assert not payload.k[..., kn:, :].any()
    assert not payload.v[..., kn:, :].any()


def test_p_suite_reuses_all_38_cases_with_four_new_directed_cases(monkeypatch):
    seen = {}
    def validate(*a, **kw): seen.update(kw); return "passed"
    monkeypatch.setattr(checks, "validate_pv_prefetch", validate)
    assert checks.validate_pv_p_prefetch(object(), object(), "cpu", {}, lambda: None) == "passed"
    assert seen["cases"] == checks.CASES + checks.P_CASES
    assert 2*len(seen["cases"]) == 42
    assert seen["payload_factory"] is checks.probability_payload
    assert seen["scope_name"] == "pv_p_prefetch"
    assert "versus V-only" in seen["description"]


def test_p_suite_checks_all_padded_outputs_before_timing(monkeypatch):
    monkeypatch.setattr(checks, "CASES", ())
    monkeypatch.setattr(checks, "P_CASES", (checks.P_CASES[0],))
    class Adapter:
        def __init__(self, corrupt=False): self.corrupt = corrupt
        def run_native(self, payload):
            out = payload.v[..., :1, :].float().expand_as(payload.q).clone()
            if self.corrupt: out[..., -1, -1] += .001
            return out
    report = {}
    checks.validate_pv_p_prefetch(Adapter(), Adapter(), "cpu", report, lambda: None)
    assert len(report["pv_p_prefetch_boundary_validation"]) == 2
    assert "pv_prefetch_boundary_validation" not in report
    assert all(row["status"] == "passed" for row in report["pv_p_prefetch_boundary_validation"])
    with pytest.raises(AssertionError, match="ALL padded"):
        checks.validate_pv_p_prefetch(Adapter(), Adapter(True), "cpu", {}, lambda: None)
