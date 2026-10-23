"""The paste-friendly msOpProf view is bounded and never pools task contexts."""
import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compact_msopprof", ROOT / "scripts/summarize_mindie_laser_msopprof.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def entry(field, role="cube0", median=.5, *, task=0):
    context = {"attributed_kernel_names": ["VcLaserAttention_123_0_mix_aic"]}
    if role != "task":
        context["sub_block_id"] = role
    else:
        context["csv_row"] = 2
    if task:
        context["Task ID"] = str(task)
    return {"field": field, "context": context, "minimum": median, "median": median, "maximum": median,
            "count": 1 if role == "task" else 24, "value_suffix": "", "domain": "exported",
            "minimum_rows": [2], "maximum_rows": [2], "samples": []}


def fixture():
    pipe = []
    for role, domain in (("cube0", "aic"), ("vector0", "aiv"), ("vector1", "aiv")):
        fields = [f"{domain}_time(us)", f"{domain}_scalar_ratio", f"{domain}_mte2_ratio"]
        fields += (["aic_cube_ratio", "aic_mte1_ratio", "aic_fixpipe_ratio", "aic_scalar_mte1_stall_time(us)"]
                   if domain == "aic" else ["aiv_vec_ratio", "aiv_mte3_ratio", "aiv_scalar_wait_time(us)"])
        for field in fields:
            value = 76242.796875 if field.endswith("time(us)") else .769512
            pipe.append(entry(field, role, value))
    groups = {
        "OpBasicInfo": [entry(f, "task", v) for f, v in (("Task Duration(us)", 81792.25), ("Block Dim", 24),
                        ("Mix Block Dim", 48), ("Current Freq", 1800), ("Rated Freq", 1800))],
        "PipeUtilization": pipe,
        "ArithmeticUtilization": [entry("aic_cube_fp16_ratio"), entry("aic_cube_int8_ratio", median=0)] +
            [entry(f"aiv_vec_{kind}_ratio", role, .7 if kind == "fp16" else .3)
             for role in ("vector0", "vector1") for kind in ("fp16", "fp32", "misc")],
        "Memory": [entry("aic_l1_read_bw(GB/s)", median=400), entry("aic_main_mem_read_bw(GB/s)", median=20)] +
            [entry("aiv_gm_to_ub_bw(GB/s)", role, 15) for role in ("vector0", "vector1")],
        "MemoryL0": [entry("aic_l0a_read_bw(GB/s)", median=800)],
        "MemoryUB": [entry(f"aiv_ub_{rw}_bw_vector(GB/s)", role, 30)
                     for role in ("vector0", "vector1") for rw in ("read", "write")],
        "L2Cache": [entry("aic_total_hit_rate(%)", median=99)] +
            [entry("aiv_total_hit_rate(%)", role, 97) for role in ("vector0", "vector1")],
        "ResourceConflictRatio": [entry("aiv_vec_" + kind + "_cflt_ratio", role, .001)
            for role in ("vector0", "vector1") for kind in ("total", "bankgroup", "bank")],
    }
    files = [{"metric_group": group, "task_directory": "OPPROF_task0", "relative_path": f"OPPROF_task0/{group}.csv",
              "distributions": entries, "errors": [], "matched_rows": 1, "mismatched_rows": 0, "unconfirmed_rows": 0}
             for group, entries in groups.items()]
    row = {"variant": "original_laser", "status": "verified", "target_verification": "verified",
           "verified_task_directories": ["OPPROF_task0"], "missing_groups": [], "partial_groups": [],
           "errors": [], "files": files}
    return {"status": "verified", "collections": [row], "scope": "scope", "aggregation": "aggregation", "errors": []}


def test_four_variants_are_under_100_lines_and_no_line_is_unbounded():
    report = fixture()
    report["collections"] *= 4
    text = M.render_summary(report)
    assert 60 <= len(text.splitlines()) <= 100
    assert max(map(len, text.splitlines())) <= 240
    assert len(text.encode("utf-8")) < 18000
    assert text.count("Arithmetic[f2] cube0") == 4
    assert "cube_int8_ratio=0[n=24]" in text
    assert "mte1_ratio=0.769512[n=24]" in text
    assert "fixpipe_ratio=0.769512[n=24]" in text
    assert "scalar_ratio=0.769512[n=24]" in text


def test_zero_and_missing_are_distinct_and_ratios_are_not_percent_scaled():
    report = fixture()
    arithmetic = report["collections"][0]["files"][2]
    arithmetic["distributions"] = [e for e in arithmetic["distributions"] if e["field"] != "aic_cube_fp16_ratio"]
    text = M.render_summary(report)
    assert "cube_fp16_ratio=NA" in text
    assert "cube_int8_ratio=0[n=24]" in text
    assert "cube_ratio=0.769512[n=24]" in text
    assert "aic_total_hit_rate(%)=99[n=24]" in text


def test_multiple_tasks_and_passes_are_not_pooled_or_selected_silently():
    report = fixture()
    row = report["collections"][0]
    other_task = copy.deepcopy(row["files"][1])
    other_task["task_directory"] = "OPPROF_task1"
    other_task["relative_path"] = "OPPROF_task1/PipeUtilization.csv"
    other_task["distributions"] = [entry("aic_time(us)", median=999999)]
    other_pass = copy.deepcopy(row["files"][1])
    other_pass["relative_path"] = "OPPROF_task0/PipeUtilization_123.csv"
    other_pass["distributions"] = [entry("aic_time(us)", median=888888)]
    row["files"].extend([other_task, other_pass])
    row["files"][1]["distributions"].append(entry("aic_time(us)", median=777777, task=2))
    text = M.render_summary(report)
    assert "other tasks omitted=1" in text and "extra files=1" in text
    assert "contexts/fields=1" in text
    assert all(str(number) not in text for number in (999999, 888888, 777777))
    assert "76242.8/76242.8/76242.8[n=24]" in text


def test_vector0_and_vector1_keep_independent_statistics():
    report = fixture()
    for e in report["collections"][0]["files"][1]["distributions"]:
        if e["field"] == "aiv_vec_ratio":
            e.update(minimum=.11, median=.11, maximum=.11) if e["context"]["sub_block_id"] == "vector0" else e.update(minimum=.91, median=.91, maximum=.91)
    lines = M.render_summary(report).splitlines()
    assert any("Pipe[f1] vector0" in line and "vec_ratio=0.11[n=24]" in line for line in lines)
    assert any("Pipe[f1] vector1" in line and "vec_ratio=0.91[n=24]" in line for line in lines)
    assert not any("vec_ratio=0.51" in line for line in lines)


def test_long_names_errors_and_unbounded_collections_have_explicit_omissions():
    report = fixture()
    row = report["collections"][0]
    row["variant"] = "v" * 900
    row["errors"] = ["x" * 1000] * 1000
    report["collections"] *= 10
    report["errors"] = ["y" * 2000]
    text = M.render_summary(report)
    assert len(text.splitlines()) <= 100
    assert max(map(len, text.splitlines())) <= 240
    assert "Collections omitted=6" in text and "errors=1000" in text
    assert "full in JSON" in text


def test_detailed_renderer_keeps_original_data_and_compact_does_not_mutate():
    report = fixture()
    previous = copy.deepcopy(report)
    compact = M.render_summary(report)
    detailed = M.render_detailed_summary(report)
    assert report == previous
    assert "min CSV rows=[2] max CSV rows=[2]" in detailed
    assert "min CSV rows" not in compact
    assert "No register-spill diagnosis" in detailed


def test_empty_capture_remains_partial_and_missing_fields_are_explicit():
    report = fixture()
    report.update(status="partial")
    report["collections"][0].update(status="partial", target_verification="partial", files=[],
        verified_task_directories=[], missing_groups=list(M.GROUPS))
    text = M.render_summary(report)
    assert "kernel=NA (identity unconfirmed)" in text
    assert "Task Duration(us)=NA" in text
    assert "Basic[missing]" in text and "Pipe[missing]" in text
