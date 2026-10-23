"""Host-only fixtures matching documented A2/A3 msOpProf CSV layouts."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("laser_msopprof_summary", ROOT / "scripts/summarize_mindie_laser_msopprof.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
KERNEL = "VcLaserQuantVscalePvPrefetchAttention_deadbeef_0_mix_aic"


def csv_file(directory, name, headers, rows, *, bom=False):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with path.open("w", encoding="utf-8-sig" if bom else "utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def make_manifest(tmp_path, *, status="passed", returncode=0, prefix="VcLaserQuantVscalePvPrefetchAttention"):
    collection = {"variant": "vector_pv_prefetch", "output_dir": "stable/profile", "kernel_prefix": prefix,
                  "command": ["msopprof", "--aic-metrics=Default"], "status": status,
                  "returncode": returncode, "worker_report": "stable/worker.json"}
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "laser-msopprof-v1", "collections": [collection]}))
    return tmp_path / "stable/profile/OPPROF_20261023_001"


def basic(directory, *, name=KERNEL, suffix="", rows=None):
    return csv_file(directory, "OpBasicInfo" + suffix + ".csv",
        ["Op Name", "Op Type", "Task Duration(us)", "Block Dim", "Mix Block Dim", "Device ID", "PID", "Current Freq", "Rated Freq"],
        rows or [[name, "mix", "78000.125", "24", "48", "0", "123", "1650", "1650"]], bom=True)


def pipe(directory, suffix=""):
    # The documented per-core table does not repeat the kernel name. NA entries
    # are not zero. AIC, vector0 and vector1 have different per-core domains.
    return csv_file(directory, "PipeUtilization" + suffix + ".csv",
        ["block_id", "sub_block_id", "aic_time(us)", "aic_cube_ratio", "aic_mte1_time(us)",
         "aiv_time(us)", "aiv_vec_ratio", "aiv_mte2_active_bw(GB/s)"],
        [[0, "cube", "100", ".45", "15", "NA", "NA", "NA"],
         [1, "cube", "300", ".60", "20", "NA", "NA", "NA"],
         [0, "vector0", "NA", "NA", "NA", "110", ".33", "4.5"],
         [1, "vector0", "NA", "NA", "NA", "120", ".34", "4.6"],
         [0, "vector1", "NA", "NA", "NA", "130", ".35", "4.7"]])


def rest(directory, suffix=""):
    groups = {
        "Memory": ("GM_to_L1_datas(KB)", "123.25"),
        "MemoryL0": ("aic_l0a_read_bw(GB/s)", "220.5"),
        "MemoryUB": ("aiv_ub_read_bw_vector(GB/s)", "30.5"),
        "L2Cache": ("aic_total_hit_rate(%)", "91.25"),
        "ArithmeticUtilization": ("aic_cube_int8_ratio", ".5"),
        "ResourceConflictRatio": ("aiv_vec_bank_cflt_ratio", ".012")}
    for group, (metric, value) in groups.items():
        csv_file(directory, group + suffix + ".csv", ["block_id", "sub_block_id", metric], [[0, "cube", value]])


def get_file(report, group, index=0):
    return [x for x in report["collections"][0]["files"] if x["metric_group"] == group][index]


def test_official_layout_identity_domains_raw_units_and_core_imbalance(tmp_path):
    directory = make_manifest(tmp_path)
    raw_basic = basic(directory)
    raw_pipe = pipe(directory)
    rest(directory)
    report = MODULE.summarize(tmp_path)
    assert report["status"] == "verified"
    collection = report["collections"][0]
    assert collection["target_verification"] == "verified"
    item = get_file(report, "PipeUtilization")
    assert item["sha256"] == hashlib.sha256(raw_pipe.read_bytes()).hexdigest()
    assert get_file(report, "OpBasicInfo")["sha256"] == hashlib.sha256(raw_basic.read_bytes()).hexdigest()
    assert item["matched_rows"] == 5
    assert item["raw_rows"][2][2] == "NA"
    assert "sibling OpBasicInfo:" in item["rows"][0]["target"]["evidence"]
    assert item["rows"][0]["identifiers"]["block_id"] == "0"
    aic = next(x for x in item["distributions"] if x["field"] == "aic_time(us)")
    assert (aic["minimum"], aic["median"], aic["maximum"]) == (100, 200, 300)
    assert aic["header_units"] == ["us"] and aic["domain"] == "AIC"
    assert aic["maximum_rows"] == [3]
    assert aic["samples"][1]["identifiers"]["block_id"] == "1"
    ratios = [x for x in item["distributions"] if x["field"] == "aiv_vec_ratio"]
    assert len(ratios) == 2  # AIV0 and AIV1 do not disappear into one median.
    assert ratios[0]["maximum"] == .34 and ratios[0]["header_units"] == []
    cache = get_file(report, "L2Cache")["distributions"][0]
    assert cache["maximum"] == 91.25 and cache["header_units"] == ["%"]
    assert all("sum" not in x for x in item["distributions"])


def test_default_missing_groups_are_partial_even_when_basic_and_pipe_match(tmp_path):
    directory = make_manifest(tmp_path)
    basic(directory)
    pipe(directory)
    report = MODULE.summarize(tmp_path)
    assert report["status"] == "partial"
    row = report["collections"][0]
    assert row["target_verification"] == "verified"
    assert row["missing_groups"] == list(MODULE.GROUPS[2:])


def test_multitask_timestamp_files_do_not_pool_durations_or_counters(tmp_path):
    directory = make_manifest(tmp_path)
    first, second = directory / KERNEL / "0", directory / KERNEL / "1"
    for task, suffix in ((first, "_202610230101"), (second, "_202610230102")):
        basic(task, suffix=suffix)
        pipe(task, suffix=suffix)
        rest(task, suffix=suffix)
    report = MODULE.summarize(tmp_path)
    assert report["status"] == "verified"
    pipes = [f for f in report["collections"][0]["files"] if f["metric_group"] == "PipeUtilization"]
    assert len(pipes) == 2 and pipes[0]["task_directory"] != pipes[1]["task_directory"]
    assert all(next(d for d in f["distributions"] if d["field"] == "aic_time(us)")["count"] == 2 for f in pipes)


def test_wrong_kernel_is_not_verified_by_exit_zero(tmp_path):
    directory = make_manifest(tmp_path)
    basic(directory, name="TorchNpuCopy_kernel")
    pipe(directory)
    rest(directory)
    report = MODULE.summarize(tmp_path)
    assert report["status"] == "partial"
    assert report["collections"][0]["target_verification"] == "partial"
    assert get_file(report, "PipeUtilization")["mismatched_rows"] == 5
    assert not get_file(report, "PipeUtilization")["distributions"]


def test_basic_identity_is_not_borrowed_from_a_different_task_directory(tmp_path):
    directory = make_manifest(tmp_path)
    basic(directory / "expected" / "0")
    pipe(directory / "other" / "0")
    report = MODULE.summarize(tmp_path)
    assert report["collections"][0]["target_verification"] == "partial"
    assert get_file(report, "PipeUtilization")["unconfirmed_rows"] == 5


def test_multiple_basic_tasks_make_unnamed_pipe_ambiguous(tmp_path):
    directory = make_manifest(tmp_path)
    rows = [[KERNEL, "mix", 100, 24, 48, 0, 123, 1650, 1650],
            [KERNEL, "mix", 150, 24, 48, 0, 123, 1650, 1650]]
    basic(directory, rows=rows)
    pipe(directory)
    report = MODULE.summarize(tmp_path)
    assert get_file(report, "PipeUtilization")["unconfirmed_rows"] == 5
    task_durations = [d for d in get_file(report, "OpBasicInfo")["distributions"] if d["field"] == "Task Duration(us)"]
    assert len(task_durations) == 2 and all(d["count"] == 1 for d in task_durations)


@pytest.mark.parametrize("which", ["duration", "block_dim", "core_id", "ratio", "core_time"])
def test_essential_missing_fields_make_target_partial(tmp_path, which):
    directory = make_manifest(tmp_path)
    basic_path = basic(directory)
    pipe_path = pipe(directory)
    path, header = (basic_path, "Task Duration(us)" if which == "duration" else "Block Dim") if which in {"duration", "block_dim"} else (pipe_path,
        {"core_id": "sub_block_id", "ratio": "aic_cube_ratio", "core_time": "aic_time(us)"}[which])
    rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
    targets = [rows[0].index(header)]
    if which == "ratio":
        targets.append(rows[0].index("aiv_vec_ratio"))
    if which == "core_time":
        targets.append(rows[0].index("aiv_time(us)"))
    for i in sorted(targets, reverse=True):
        for row in rows:
            del row[i]
    csv_file(path.parent, path.name, rows[0], rows[1:])
    report = MODULE.summarize(tmp_path)
    assert report["collections"][0]["target_verification"] == "partial"


def test_nonfinite_percent_text_and_malformed_csv_are_retained_not_coerced(tmp_path):
    directory = make_manifest(tmp_path)
    basic(directory)
    pipe(directory)
    csv_file(directory, "L2Cache.csv", ["block_id", "sub_block_id", "aic_total_hit_rate(%)"],
             [[0, "cube", "91.5%"], [1, "cube", "NaN"], [2, "cube", "Inf"], [3, "cube", "N/A"]])
    csv_file(directory, "Memory.csv", ["block_id", "sub_block_id", "bytes(KB)", "bytes(KB)"],
             [[0, "cube", 10, 20], [1, "cube", 15]])
    report = MODULE.summarize(tmp_path)
    cache = get_file(report, "L2Cache")
    assert cache["raw_rows"][1][2] == "NaN"
    assert cache["distributions"][0]["value_suffix"] == "%"
    assert cache["distributions"][0]["maximum"] == 91.5
    memory = get_file(report, "Memory")
    assert len(memory["headers"]) == 4 and len(memory["raw_rows"][1]) == 3
    assert len(memory["errors"]) == 2 and not memory["usable_target_metrics"]
    json.dumps(report, allow_nan=False)


def test_failed_collection_retains_partial_artifacts_and_offline_cli(tmp_path):
    directory = make_manifest(tmp_path, status="failed", returncode=9)
    basic(directory)
    pipe(directory)
    rest(directory)
    report = MODULE.write_summary(tmp_path)
    assert report["status"] == "partial"
    assert (tmp_path / "summary.json").is_file() and (tmp_path / "summary.txt").is_file()
    text = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    assert "No register-spill diagnosis" in text and "max CSV rows=[3]" in text
    output = tmp_path / "nested/saved.json"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/summarize_mindie_laser_msopprof.py"),
        str(tmp_path), "--output", str(output)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0 and output.is_file() and output.with_suffix(".txt").is_file()
    assert "evidence: partial" in result.stdout


def test_absent_manifest_and_absent_output_are_partial_not_exceptions(tmp_path):
    assert MODULE.summarize(tmp_path)["status"] == "partial"
    make_manifest(tmp_path)
    report = MODULE.write_summary(tmp_path)
    assert report["status"] == "partial"
    assert report["collections"][0]["missing_groups"] == list(MODULE.GROUPS)


def test_two_variants_with_same_csv_filenames_stay_separate(tmp_path):
    directory = make_manifest(tmp_path)
    basic(directory)
    pipe(directory)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    other = dict(manifest["collections"][0], variant="original_laser", output_dir="original/profile", kernel_prefix="LaserAttention")
    manifest["collections"].append(other)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    other_dir = tmp_path / "original/profile/OPPROF_20261023_002"
    basic(other_dir, name="LaserAttention_abcdef")
    pipe(other_dir)
    report = MODULE.summarize(tmp_path)
    assert [x["variant"] for x in report["collections"]] == ["vector_pv_prefetch", "original_laser"]
    assert all(x["target_verification"] == "verified" for x in report["collections"])
    assert report["collections"][0]["files"][0]["path"] != report["collections"][1]["files"][0]["path"]
