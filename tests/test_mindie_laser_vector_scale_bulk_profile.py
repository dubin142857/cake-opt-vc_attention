"""Original-Laser primary comparison, three-provider isolation and remote driver."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import profile_mindie_laser_vector_scale_bulk as profile


def providers(tmp_path):
    return {name: SimpleNamespace(vendor_root=str(tmp_path/name/"vendors"/name),
            bridge_path=str(tmp_path/name/"bridge.so"), marker_env=f"VC_{name.upper()}_MARKER_FILE")
            for name in profile.VARIANTS}


def test_default_matches_long_bf16_workload_and_nine_trials():
    args = profile.parser().parse_args([])
    assert args.shape == [1, 28, 37741, 128]
    assert (args.warmup, args.repeats, args.trials) == (5, 10, 9)
    assert args.baseline_build == "results/mindie_laser_build.json"
    assert args.bridge_manifest == "results/mindie_laser_bridge.json"
    assert not args.trace


def test_resolve_providers_binds_unique_identity_markers_and_original_bridge(monkeypatch, tmp_path):
    args = profile.parser().parse_args([])
    args._run_dir = str(tmp_path)
    def resolve(options, name):
        if name == "baseline":
            return SimpleNamespace(vendor_root=str(tmp_path/"fp16"),
                bridge_manifest=options.bridge_manifest)
        return SimpleNamespace(vendor_root=str(tmp_path/Path(options.quant_build).stem),
                               quant_build=options.quant_build)
    monkeypatch.setattr(profile.quant, "build_arguments", resolve)
    rows = profile.resolve_providers(args)
    assert rows[profile.ORIGINAL].bridge_manifest == args.bridge_manifest
    assert rows[profile.ORIGINAL].marker_env == profile.quant.baseline.MARKER_ENV
    assert rows[profile.OLD].quant_build == args.vscale_build
    assert rows[profile.NEW].quant_build == args.bulk_build
    assert len({row.marker_token for row in rows.values()}) == 3
    assert len({row.marker_env for row in rows.values()}) == 3
    specs = profile.specifications(args)
    assert specs[profile.NEW]["source"].IDENTITY["native_api"] == "aclnnVcLaserQuantVscaleBulkAttention"


def test_three_provider_environment_scrubs_external_injection(tmp_path):
    rows = providers(tmp_path)
    env = profile.worker_environment(dict(ASCEND_CUSTOM_OPP_PATH="/foreign/vendors/extra",
        LD_PRELOAD="/foreign/preload.so", LD_AUDIT="audit.so",
        LD_LIBRARY_PATH=os.pathsep.join(("/foreign/vendors/extra/op_api/lib", "/cann/lib")),
        VC_OTHER_MARKER_FILE="unrelated"), rows, tmp_path/"run")
    roots = [str(Path(rows[name].vendor_root).resolve()) for name in profile.VARIANTS]
    assert env["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep) == roots
    assert "LD_PRELOAD" not in env and "LD_AUDIT" not in env
    assert "VC_OTHER_MARKER_FILE" not in env
    assert "/foreign" not in env["LD_LIBRARY_PATH"]
    for name, provider in rows.items():
        assert env[provider.marker_env] == str(tmp_path/"run"/"local"/(name+"_markers.jsonl"))


def test_duplicate_or_missing_provider_is_rejected(tmp_path):
    rows = providers(tmp_path)
    rows[profile.NEW].vendor_root = rows[profile.OLD].vendor_root
    with pytest.raises(ValueError, match="distinct"):
        profile.worker_environment({}, rows, tmp_path)
    rows.pop(profile.NEW)
    with pytest.raises(ValueError, match="Three-provider"):
        profile.worker_environment({}, rows, tmp_path)


def test_mapped_hashes_use_verified_separate_original_bridge(tmp_path):
    rows = providers(tmp_path)
    paths, evidence = [], {}
    for name, provider in rows.items():
        library = Path(provider.vendor_root)/"op_api"/"lib"/"libcust_opapi.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(name.encode())
        bridge = Path(provider.bridge_path)
        bridge.write_bytes((name+"_bridge").encode())
        paths.extend((str(library.resolve()), str(bridge.resolve())))
        evidence[name] = dict(host_library_sha256={str(library.resolve()): profile.common.sha256_file(library)},
                              bridge_sha256=profile.common.sha256_file(bridge))
    profile.verify_mapped_hashes(paths, rows, evidence)
    Path(rows[profile.ORIGINAL].bridge_path).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="audited manifest"):
        profile.verify_mapped_hashes(paths, rows, evidence)


def test_provider_provenance_binds_original_and_two_distinct_validators(monkeypatch, tmp_path):
    rows = providers(tmp_path)
    seen = []
    specs = {name: dict(source=SimpleNamespace(IDENTITY={"variant": name},
        validate_prepared_source=object())) for name in (profile.OLD, profile.NEW)}
    monkeypatch.setattr(profile, "specifications", lambda args: specs)
    monkeypatch.setattr(profile.quant.baseline, "provenance",
                        lambda p: seen.append(("original", p)) or {"original": True})
    monkeypatch.setattr(profile.quant, "quant_provenance", lambda p, **kw: seen.append((p, kw)) or kw)
    import mindie_laser_quant_fused
    monkeypatch.setattr(mindie_laser_quant_fused, "validate_preprocess_manifest", lambda p: {"prep": p})
    result = profile.verify_builds(SimpleNamespace(preprocess_build="preprocessor"), rows)
    assert seen[0] == ("original", rows[profile.ORIGINAL])
    assert result[profile.ORIGINAL] == {"original": True}
    for name, spec in specs.items():
        assert result[name]["identity"] == spec["source"].IDENTITY
        assert result[name]["source_validator"] is spec["source"].validate_prepared_source


def test_fixed_payloads_separate_fp16_but_share_int8_and_full_is_fresh(monkeypatch):
    import mindie_laser_fp16_trace_worker
    logs = []
    class Adapter:
        def __init__(self, name): self.name = name
        def __call__(self, *xs): logs.append(("full", self.name, xs))
        def run_prepared(self, p): logs.append(("prepared", self.name, p))
        def run_native(self, p): logs.append(("native", self.name, p))
    monkeypatch.setattr(mindie_laser_fp16_trace_worker, "native_call",
                        lambda a, p: logs.append(("original_native", a.name, p)))
    adapters = {name: Adapter(name) for name in profile.VARIANTS}
    fp16, int8 = object(), object()
    payloads = {profile.ORIGINAL: fp16, profile.OLD: int8, profile.NEW: int8}
    inputs = (object(), object(), object())
    calls = profile.make_calls(adapters, payloads, inputs)
    for scope in calls:
        for call in calls[scope].values():
            call()
            call()
    assert logs.count(("full", profile.ORIGINAL, inputs)) == 2
    assert logs.count(("full", profile.NEW, inputs)) == 2
    assert logs.count(("original_native", profile.ORIGINAL, fp16)) == 2
    assert logs.count(("native", profile.OLD, int8)) == 2
    assert logs.count(("native", profile.NEW, int8)) == 2
    with pytest.raises(ValueError, match="identical"):
        profile.make_calls(adapters, dict(payloads, vector_scale_bulk=object()), inputs)
    with pytest.raises(ValueError, match="own"):
        profile.make_calls(adapters, dict(payloads, original_laser=int8), inputs)


@pytest.mark.parametrize("error", [
    dict(relative_rmse=0., max_abs=0., exact_match=False),
    dict(relative_rmse=float("nan"), max_abs=0., exact_match=True),
    dict(relative_rmse=0., max_abs=float("inf"), exact_match=True),
])
def test_bulk_gate_cannot_accept_drift_or_nonfinite(error):
    with pytest.raises(AssertionError, match="ALL padded"):
        profile.require_exact(error)


def test_validation_adapter_exact_gate_catches_drift(monkeypatch):
    import torch
    import profile_mindie_laser_quant
    comparisons = []
    original_compare = profile_mindie_laser_quant.compare_all_rows
    monkeypatch.setattr(profile_mindie_laser_quant, "compare_all_rows",
                        lambda a, b: comparisons.append(True) or original_compare(a, b))
    class Adapter:
        expcast = False
        def __init__(self, values): self.values = values
        def run_native(self, p): return self.values.clone()
    rows = []
    old = Adapter(torch.zeros(1, 1, 2, 2))
    new = Adapter(torch.zeros(1, 1, 2, 2))
    check = profile.ExactValidationAdapter(new, old, rows, lambda: None)
    payload = object()
    # Queued calls permit intervening same-pointer value changes without
    # comparison/CPU sync from the additional exact-equivalence instrumentation.
    check.run_native(payload)
    old.values.fill_(2)
    new.values.fill_(2)
    check.run_native(payload)
    assert comparisons == [] and rows == [] and len(check.pending) == 2
    check.flush()
    assert len(comparisons) == 2 and check.pending == []
    assert rows[-1]["exact_match"]
    new.values[..., -1] = .001
    check.run_native(payload)
    assert len(comparisons) == 2
    with pytest.raises(AssertionError, match="ALL padded"):
        check.flush()
    assert len(rows) == 3


def test_primary_table_uses_full_laser_and_preserves_per_scope_ratios():
    from mindie_laser_quant_pairing import run_paired
    timings = {}
    for scope, speeds in (("full", (90, 84, 80)), ("prepared", (85, 81, 77)), ("native", (83, 79, 75))):
        durations = dict(zip(profile.VARIANTS, speeds))
        elapsed = [0.]
        def call(name):
            elapsed[0] += durations[name]/1000
        calls = {name: lambda n=name: call(n) for name in profile.VARIANTS}
        timings[scope] = run_paired(calls, synchronize=lambda: None, warmup=0,
            repeats=2, trials=9, clock=lambda: elapsed[0])
    report = dict(timings=timings,
        comparisons={scope: profile.compare_scopes(timing) for scope, timing in timings.items()},
        dense_fp32_samples={name: {"relative_rmse": .012} for name in profile.VARIANTS},
        same_payload_all_output={"exact_match": True})
    text = profile.table(report)
    assert "1.1250x" in text  # Full 90/80, not native 83/75.
    assert "1.200000" in text
    assert "ORIGINAL FP16" in text and "faster 9/9" in text
    assert "own payloads" in text and "not kernel-only" in text
    for scope in timings:
        pair = report["comparisons"][scope][profile.NEW]
        assert pair["baseline"] == profile.ORIGINAL and pair["faster_trials"] == 9
        assert len(timings[scope]["orders"]) == 9
        for name in profile.VARIANTS:
            assert [order.index(name) for order in timings[scope]["orders"]].count(0) == 3


def test_child_launch_forwards_every_manifest_and_fixed_shape(monkeypatch, tmp_path):
    args = profile.parser().parse_args(["--trace", "--device", "2"])
    args._run_dir = str(tmp_path)
    class Process:
        stdout = []
        def wait(self): return 5
    monkeypatch.setattr(profile.subprocess, "Popen", lambda *a, **kw: Process())
    env = {key: str(tmp_path/key) for key in
           ("ACL_OP_COMPILER_CACHE_DIR", "ASCEND_CACHE_PATH", "XDG_CACHE_HOME")}
    result = profile.launch(args, env, tmp_path/"local")
    assert result["status"] == "missing_report" and result["returncode"] == 5
    command = result["command"]
    assert "--_worker" in command and "--trace" in command
    for key, default in profile.MANIFESTS.items():
        assert command[command.index("--"+key.replace("_", "-"))+1] == default
    assert command[command.index("--shape")+1:command.index("--shape")+5] == ["1", "28", "37741", "128"]


@pytest.mark.parametrize("child", [
    {"status": "passed", "returncode": 2},
    {"status": "failed", "returncode": 0},
    {"status": "missing_report", "returncode": 0},
])
def test_parent_preserves_failed_child_and_never_reports_success(monkeypatch, tmp_path, child):
    args = profile.parser().parse_args(["--output", str(tmp_path/"report.json")])
    monkeypatch.setattr(profile, "resolve_providers", lambda a: providers(tmp_path))
    monkeypatch.setattr(profile, "verify_builds", lambda a, p: {"passed": True})
    monkeypatch.setattr(profile, "worker_environment", lambda *a: {})
    monkeypatch.setattr(profile, "launch", lambda *a: child)
    assert profile.parent(args) == 1
    report = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["workers"]["paired"] == child
    assert report["primary_baseline"] == profile.ORIGINAL


@pytest.mark.parametrize("failure", [None, "preflight", "build", "profile"])
@pytest.mark.parametrize("candidate", ["bulk", "rowmax", "pv_prefetch"])
def test_one_click_builds_only_candidate_and_stops_at_failure(tmp_path, failure, candidate):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing")
    if not bash.is_file():
        pytest.skip("Bash required")
    root = tmp_path/"project with spaces"
    for part in ("scripts", "results", "cann"):
        (root/part).mkdir(parents=True)
    for name in ("mindie_laser_quant_vscale_build", "mindie_laser_quant_preprocess_build",
                 "mindie_laser_build", "mindie_laser_bridge"):
        (root/"results"/(name+".json")).write_text("{}")
    script_suffix = {"bulk": "vector_scale_bulk", "rowmax": "vector_rowmax", "pv_prefetch": "pv_prefetch"}[candidate]
    driver = root/("scripts/run_mindie_laser_"+script_suffix+".sh")
    driver.write_bytes((ROOT/("scripts/run_mindie_laser_"+script_suffix+".sh")).read_bytes())
    stub = root/"stub.py"
    stub.write_text("import json,sys\nfrom pathlib import Path\n"
        "a=sys.argv[1:]\ns='preflight' if a[0]=='-' else 'metadata' if a[0]=='-c' else 'build' if 'build_' in a[1] else 'profile'\n"
        "with Path('calls.jsonl').open('a') as f:f.write(json.dumps({'stage':s,'args':a})+'\\n')\n"
        "if s=='metadata':print('cached vector scale source' if 'source_dir' in a[-1] else 'ascend910_93')\n"
        f"raise SystemExit(23 if s=={failure!r} else 0)\n", encoding="utf-8")
    wrapper = root/"mock-python"
    wrapper.write_text(f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "{stub.as_posix()}" "$@"\n',
                       encoding="utf-8", newline="\n")
    wrapper.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("BASH_ENV", "ENV")}
    env.update(PYTHON_BIN=wrapper.as_posix(), ASCEND_HOME_PATH=(root/"cann").as_posix(), NPU_DEVICE="2")
    result = subprocess.run([str(bash), "--noprofile", "--norc", str(driver)],
        cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == (23 if failure else 0), result.stdout+result.stderr
    calls = [json.loads(row) for row in (root/"calls.jsonl").read_text().splitlines()]
    if failure:
        assert calls[-1]["stage"] == failure
    else:
        assert [c["stage"] for c in calls] == ["preflight", "metadata", "metadata", "build", "profile"]
        build, profile_call = calls[-2]["args"], calls[-1]["args"]
        assert build[1] == "scripts/build_mindie_laser_quant_vscale_"+candidate+".py"
        assert build[build.index("--reference-dir")+1] == "cached vector scale source"
        assert profile_call[1] == "scripts/profile_mindie_laser_"+script_suffix+".py"
        assert "--trace" in profile_call
