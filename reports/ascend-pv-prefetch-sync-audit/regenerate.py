"""Rebuild and scan the isolated PV V-prefetch candidate from local Git objects."""
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
from scripts import prepare_mindie_laser_quant_vscale_source as baseline
from scripts import prepare_mindie_laser_quant_vscale_pv_prefetch_source as candidate

REPORT = Path(__file__).resolve().parent
BUILD = ROOT / "build/ascend_pv_prefetch_audit"
REFERENCE = ROOT / "build/mindie-sd-reference"
TOOLS = ROOT / "build/cannbot-skills/ops/ascendc-sync-audit/scripts"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def method(data, name):
    start = data.index(b"::" + name.encode() + b"(")
    opening = data.index(b"{", start)
    depth = 1
    for index in range(opening + 1, len(data)):
        depth += (data[index] == ord("{")) - (data[index] == ord("}"))
        if depth == 0:
            return data[start:index + 1]
    raise ValueError("Unclosed method " + name)


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

    first, second = trees["baseline"], trees["candidate"]
    checks = {}
    for name, data in first.items():
        if not name.startswith("op_kernel/"):
            continue
        key = candidate.rename(name.encode()).decode()
        unchanged = candidate.rename(data) == second[key]
        checks[name] = dict(baseline_sha256=sha(data), candidate_sha256=sha(second[key]),
                           inverse_rename_identical=unchanged)
        if not name.endswith("CubeForward.h"):
            assert unchanged, name

    before, after = first["op_kernel/CubeForward.h"], second["op_kernel/CubeForward.h"]
    methods = {}
    for name in ("cube1_matmul_op", "cube1_base_matmul", "Run", "Init", "PresetFlag", "ClearFlag",
                 "matmul_fp32", "diag_matmul"):
        a, b = method(before, name), method(after, name)
        methods[name] = dict(baseline_sha256=sha(a), candidate_sha256=sha(b), identical=a == b)
        assert a == b, name
    raw = rb"(?:mad|copy_matrix_cc_to_gm|diag_matmul)\([^;]*\);"
    old_calls = re.findall(raw, method(before, "cube2_matmul_op"))
    new_calls = re.findall(raw, method(after, "cube2_matmul_op"))
    assert old_calls == new_calls, "PV math or output sites changed"
    write_json(REPORT / "source_comparison.json", dict(
        upstream_commit=baseline.SOURCE_LOCK.commit, kernel_files=checks, unchanged_methods=methods,
        pv_math_output_calls=dict(baseline=len(old_calls), candidate=len(new_calls), identical=True)))
    diff = "".join(difflib.unified_diff(before.decode().splitlines(keepends=True),
        after.decode().splitlines(keepends=True), "baseline/CubeForward.h", "candidate/CubeForward.h", n=0))
    (REPORT / "CubeForward.h.diff").write_text(diff, encoding="utf-8")

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
