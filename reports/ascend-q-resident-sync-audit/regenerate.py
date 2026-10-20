"""Regenerate local-only, unfiltered Q-residency synchronization evidence."""
from concurrent.futures import ThreadPoolExecutor
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import prepare_mindie_laser_quant_cache_source as baseline
from scripts import prepare_mindie_laser_quant_qresident_source as candidate
from scripts import mindie_laser_quant_qresident_patch as patch

REPORT = Path(__file__).resolve().parent
BUILD = ROOT / "build/ascend_q_resident_sync_audit"
REFERENCE = ROOT / "build/mindie-sd-reference"
TOOLS = ROOT / "build/cannbot-skills/ops/ascendc-sync-audit/scripts"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def restore_cube(data):
    for old, new in ((patch.OLD_DECL, patch.NEW_DECL), (patch.OLD_CALL, patch.NEW_CALL),
                     (patch.OLD_DEF, patch.NEW_DEF), (patch.OLD_LOAD, patch.NEW_LOAD)):
        data = data.replace(new, old)
    return data


def main():
    jobs, trees = [], {}
    for tag, module in (("baseline", baseline), ("candidate", candidate)):
        prepared, _ = module.expected_files(REFERENCE)
        selected = {}
        for name, data in prepared.items():
            if name.startswith("csrc/ops/") and Path(name).suffix in (".h", ".hpp", ".cpp"):
                relative = "/".join(name.split("/")[3:])
                selected[relative] = data
                destination = BUILD / tag / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
        trees[tag] = selected
        paths = [str(BUILD / tag / name) for name in sorted(selected)]
        for tool in ("sync_audit", "ascendc_flow_analyzer"):
            jobs.append((tag, tool, paths, selected))

    before, after = trees["baseline"], trees["candidate"]
    raw = rb"(?:set_flag|wait_flag|wait_flag_dev|ffts_cross_core_sync|pipe_barrier|mad|copy_matrix_cc_to_gm|set_quant_pre)\([^;]*\);"
    checks = {}
    for name in ("CubeForward.h", "VectorForward.h", "AddressMappingForwardOnline.h", "AddressMappingVectorForwardOnline.h"):
        key = "op_kernel/" + name
        first, second = before[key], after[key]
        first_calls, second_calls = re.findall(raw, first), re.findall(raw, second)
        checks[name] = dict(baseline_sha256=sha(first), candidate_sha256=sha(second),
                           byte_identical=first == second, baseline_raw_static_calls=len(first_calls),
                           candidate_raw_static_calls=len(second_calls), raw_static_calls_identical=first_calls == second_calls)
        assert first_calls == second_calls
        if name == "CubeForward.h":
            checks[name]["inverse_byte_identical"] = restore_cube(second) == first
            checks[name]["run_preset_clear_and_following_bytes_identical"] = first[first.index(patch.RUN_BEGIN):] == second[second.index(patch.RUN_BEGIN):]
            checks[name]["prefix_including_pv_identical_after_declaration_restore"] = first[:first.index(patch.QK_BEGIN)] == second[:second.index(patch.QK_BEGIN)].replace(patch.NEW_DECL, patch.OLD_DECL)
            checks[name]["dynamic_count_note"] = "Every event, MAD and FIX retains the same dynamic count/order; only guarded Q load instructions are suppressed."
            diff = "".join(difflib.unified_diff(first.decode().splitlines(keepends=True),
                                               second.decode().splitlines(keepends=True),
                                               f"baseline/{name}", f"candidate/{name}", n=0))
            (REPORT / "CubeForward.h.diff").write_text(diff, encoding="utf-8")
            assert checks[name]["inverse_byte_identical"]
            assert checks[name]["run_preset_clear_and_following_bytes_identical"]
            assert checks[name]["prefix_including_pv_identical_after_declaration_restore"]
        else:
            assert first == second
    write_json(REPORT / "source_comparison.json", checks)

    def run(job):
        tag, name, paths, files = job
        script = TOOLS / (name + ".py")
        command = [sys.executable, "-X", "utf8", str(script), *paths, "--format", "json"]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=180,
                                env={**os.environ, "PYTHONUTF8": "1"})
        (REPORT / f"{tag}_{name}.json").write_bytes(result.stdout)
        (REPORT / f"{tag}_{name}.stderr.txt").write_bytes(result.stderr)
        raw_result = json.loads(result.stdout.decode("utf-8"))
        record = dict(command=command, returncode=result.returncode, tool_sha256=sha(script.read_bytes()),
                      files={key: sha(value) for key, value in sorted(files.items())},
                      findings=len(raw_result.get("findings", [])), event_count=raw_result.get("event_count"),
                      operation_count=len(raw_result.get("operations", [])))
        write_json(REPORT / f"{tag}_{name}_run.json", record)
        return dict(tree=tag, tool=name, returncode=result.returncode, files=len(files),
                    findings=record["findings"], events=record["event_count"], operations=record["operation_count"])

    with ThreadPoolExecutor(max_workers=4) as workers:
        summary = list(workers.map(run, jobs))
    write_json(REPORT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
