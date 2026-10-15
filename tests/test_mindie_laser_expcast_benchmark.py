"""CPU-only manifest routing, algorithm identity and benchmark scope checks."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_mindie_laser_expcast as benchmark


@pytest.fixture
def arguments(tmp_path):
    paths = {}
    for name in ("baseline", "expcast", "baseline_probe", "expcast_probe"):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps({"vendor_root": str(tmp_path / name / "vendors/vc_laser"),
                                    "source_manifest": {"path": str(tmp_path / name / "source.json")}}))
        paths[name + "_build"] = str(path)
    bridge = tmp_path / "bridge.json"
    bridge.write_text(json.dumps({"bridge_path": str(tmp_path / "bridge.so")}))
    return SimpleNamespace(**paths, bridge_manifest=str(bridge), device=2, warmup=5,
                           repeats=10, trials=3, cpu_threads=4, _run_dir=str(tmp_path / "run"))


def test_each_variant_routes_to_its_own_install_and_worker_directory(arguments):
    results = {name: benchmark.build_arguments(arguments, name)
               for name in ("installed", "baseline", "expcast", "baseline_probe", "expcast_probe")}
    assert results["installed"].vendor_root == results["baseline"].vendor_root
    assert results["installed"]._worker == "installed"
    assert all(result._worker == "local" for name, result in results.items() if name != "installed")
    assert len({row.vendor_root for name, row in results.items() if name != "installed"}) == 4
    assert len({row._run_dir for row in results.values()}) == 5
    assert len({row.bridge_path for row in results.values()}) == 1
    assert all(row.device == 2 and row.repeats == 10 and row.trials == 3 for row in results.values())


@pytest.mark.parametrize("name", ["installed", "baseline", "expcast", "baseline_probe", "expcast_probe"])
def test_worker_chooses_algorithm_gate_and_instrumentation_scope(arguments, monkeypatch, name):
    arguments._worker = name
    calls = []
    monkeypatch.setattr(benchmark.baseline, "run_worker", lambda args, **kwargs: calls.append((args, kwargs)) or 0)
    assert benchmark.worker(arguments) == 0
    args, options = calls[0]
    assert options["dense_limit"] == (None if name.startswith("expcast") else .01)
    assert bool(options["small_validator"]) == name.startswith("expcast")
    assert bool(options["after_long"]) == name.endswith("_probe")
    if options["small_validator"]:
        assert options["small_validator"].__module__.endswith("mindie_laser_expcast_validation")
    assert Path(args._run_dir).name == name


def sample_files(tmp_path, *, mismatch=None, nonfinite=False):
    common = {"shape_bnsd": [1, 28, 37741, 128], "seed": 0, "input_sha256": ["q", "k", "v"],
              "query_ids": torch.tensor([0, 5]), "output": torch.ones((1, 1, 2, 4))}
    candidate = dict(common)
    candidate["output"] = torch.full((1, 1, 2, 4), float("nan") if nonfinite else 1.1)
    if mismatch == "query_ids":
        candidate[mismatch] = torch.tensor([0, 6])
    elif mismatch is not None:
        candidate[mismatch] = {"input_sha256": ["changed", "k", "v"], "seed": 2,
                               "shape_bnsd": [1, 2, 3, 4]}[mismatch]
    paths = tmp_path / "baseline.pt", tmp_path / "candidate.pt"
    torch.save(common, paths[0])
    torch.save(candidate, paths[1])
    return paths


def test_different_algorithm_comparison_reports_error_without_same_algorithm_gate(tmp_path):
    metric = benchmark.compare_algorithm_samples(*sample_files(tmp_path))
    assert metric["relative_rmse"] == pytest.approx(.1)
    assert not metric["exact_match"]
    assert "passed" not in metric and "limit" not in metric
    assert "Different probability algorithms" in metric["scope"]


@pytest.mark.parametrize("field", ["input_sha256", "seed", "shape_bnsd", "query_ids"])
def test_different_algorithm_comparison_still_requires_identical_inputs(tmp_path, field):
    with pytest.raises(ValueError):
        benchmark.compare_algorithm_samples(*sample_files(tmp_path, mismatch=field))


def test_algorithm_diagnostic_rejects_nonfinite_output(tmp_path):
    with pytest.raises(ValueError, match="non-finite"):
        benchmark.compare_algorithm_samples(*sample_files(tmp_path, nonfinite=True))


def timing_row(full, core):
    return {"timing_ms": {"full": {"median": full}, "prepared_core": {"median": core}},
            "long_dense_fp32": {"relative_rmse": .02}}


def test_table_ratios_use_full_api_and_keep_probes_separate():
    workers = {"installed": timing_row(90, 80), "baseline": timing_row(100, 70),
               "expcast": timing_row(50, 10), "expcast_probe": timing_row(999, 12)}
    workers["expcast_probe"]["probability_probe"] = {"samples": 3}
    text = benchmark.table({"workers": workers})
    candidate_line = next(line for line in text.splitlines() if line.startswith("expcast "))
    assert "2.0000x" in candidate_line  # 100/50, never core 70/10.
    assert "7.0000x" not in candidate_line
    assert "probe/production prepared API=1.2000x" in text
    assert "perturbation, not speedup" in text
    assert "not paired/interleaved" in text


@pytest.mark.parametrize("extra", [["--baseline-probe-build", "one.json"], ["--trials", "1"],
                                    ["--_worker", "expcast"]])
def test_cli_rejects_partial_probe_or_invalid_trial_worker_arguments(extra):
    with pytest.raises(SystemExit) as exc:
        benchmark.main(extra)
    assert exc.value.code == 2
