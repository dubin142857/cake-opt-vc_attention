"""Rebuild and scan the isolated FP16 ExpCast candidate from local Git objects."""
from concurrent.futures import ThreadPoolExecutor
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import prepare_mindie_laser_quant_vscale_pv_prefetch_source as baseline
from scripts import prepare_mindie_laser_quant_vscale_pv_expcast_source as candidate

REPORT = Path(__file__).resolve().parent
BUILD = ROOT / "build/ascend_fp16_expcast_audit"
REFERENCE = ROOT / "build/mindie-sd-reference"
TOOLS = ROOT / "build/cannbot-skills/ops/ascendc-sync-audit/scripts"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    old, _ = baseline.expected_files(REFERENCE)
    new, _ = candidate.expected_files(REFERENCE)
    jobs, trees = [], {}
    for variant, files in (("baseline", old), ("candidate", new)):
        selected = {}
        for name, data in files.items():
            if name.startswith("csrc/ops/") and Path(name).suffix in (".h", ".hpp", ".cpp"):
                relative = "/".join(name.split("/")[3:])
                selected[relative] = data
                path = BUILD / variant / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        trees[variant] = selected
        paths = [str(BUILD / variant / key) for key in sorted(selected)]
        for tool in ("sync_audit", "ascendc_flow_analyzer"):
            jobs.append((variant, tool, paths, selected))

    # Check every generated file, not only device headers. The optional helper
    # is the sole source change once operator identity strings are normalized.
    checks, changed = {}, []
    assert set(new) == {candidate.rename(k.encode()).decode() for k in old}
    helper = f"{candidate.CUSTOM_OP_ROOT}/op_kernel/vc_laser_probability.h"
    for name, data in old.items():
        key = candidate.rename(name.encode()).decode()
        unchanged = candidate.rename(data) == new[key]
        checks[name] = dict(baseline_sha256=sha(data), candidate_sha256=sha(new[key]),
                           inverse_rename_identical=unchanged)
        if not unchanged:
            changed.append(key)
    assert changed == [helper], changed
    before = old[f"{baseline.CUSTOM_OP_ROOT}/op_kernel/vc_laser_probability.h"]
    after = new[helper]
    vector = new[f"{candidate.CUSTOM_OP_ROOT}/op_kernel/VectorForward.h"]
    from scripts.mindie_laser_expcast_patch import BULK_VEXP, EXPERIMENTAL_PROBABILITY
    assert vector.count(BULK_VEXP) == vector.count(EXPERIMENTAL_PROBABILITY) == 1
    write_json(REPORT / "source_comparison.json", dict(
        upstream_commit=baseline.SOURCE_LOCK.commit, generated_files=checks,
        changed_after_identity_rename=changed,
        unchanged_cube=True, unchanged_vector_caller=True,
        original_exp_branch_count=vector.count(BULK_VEXP),
        optional_helper_call_count=vector.count(EXPERIMENTAL_PROBABILITY),
        diagnostic_instrumentation=False,
        probability_contract=candidate.CONTRACT))
    diff = "".join(difflib.unified_diff(before.decode().splitlines(keepends=True),
        after.decode().splitlines(keepends=True), "baseline/vc_laser_probability.h",
        "candidate/vc_laser_probability.h", n=0))
    (REPORT / "vc_laser_probability.h.diff").write_text(diff, encoding="utf-8")

    def run(job):
        variant, name, paths, files = job
        script = TOOLS / (name + ".py")
        command = [sys.executable, "-X", "utf8", str(script), *paths, "--format", "json"]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60,
                                env={**os.environ, "PYTHONUTF8": "1"})
        (REPORT / f"{variant}_{name}.json").write_bytes(result.stdout)
        (REPORT / f"{variant}_{name}.stderr.txt").write_bytes(result.stderr)
        output = json.loads(result.stdout.decode("utf-8"))
        record = dict(command=command, returncode=result.returncode, tool_sha256=sha(script.read_bytes()),
            files={key: sha(value) for key, value in sorted(files.items())},
            findings=len(output.get("findings", [])), event_count=output.get("event_count"),
            operation_count=len(output.get("operations", [])))
        write_json(REPORT / f"{variant}_{name}_run.json", record)
        return dict(variant=variant, tool=name, returncode=result.returncode, files=len(files),
                    findings=record["findings"], events=record["event_count"], operations=record["operation_count"])

    with ThreadPoolExecutor(max_workers=4) as workers:
        summary = list(workers.map(run, jobs))
    write_json(REPORT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
