"""Audit current fixed-vscale Cube diagnostics using local pinned source only."""
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
from scripts import prepare_mindie_laser_cube_probe_source as source

REPORT = Path(__file__).resolve().parent
BUILD = ROOT / "build/ascend_vscale_cube_phase_audit"
REFERENCE = ROOT / "build/mindie-sd-reference"
TOOLS = ROOT / "build/cannbot-skills/ops/ascendc-sync-audit/scripts"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def remove_hooks(data):
    data = data.replace(b'#include "vc_laser_cube_probe.h"\n', b"")
    data = data.replace(b"    vc_laser_cube_probe::CubeProbe cube_probe_;\n", b"")
    data = re.sub(rb"    __aicore__ inline void SetupCubeProbe\(__gm__ uint8_t \*trace, int32_t mode\) \{.*?\n    \}\n",
                  b"", data, flags=re.S)
    return b"".join(line for line in data.splitlines(keepends=True) if b"cube_probe_." not in line)


def main():
    jobs, comparison = [], {}
    # Both current INT8 and original FP16 are audited. Legacy scale-reuse is
    # deliberately absent; no source or build directory is overwritten.
    for family in ("vscale", "fp16"):
        original, _ = source.baseline_files(REFERENCE, family)
        candidate, _ = source.expected_files(REFERENCE, family)
        trees = {}
        for variant, files in (("baseline", original), ("probe", candidate)):
            tag = f"{family}_{variant}"
            selected = {}
            for name, data in files.items():
                if name.startswith("csrc/ops/") and Path(name).suffix in (".h", ".hpp", ".cpp"):
                    relative = "/".join(name.split("/")[3:])
                    selected[relative] = data
                    destination = BUILD / tag / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
            trees[variant] = selected
            paths = [str(BUILD / tag / name) for name in sorted(selected)]
            for tool in ("sync_audit", "ascendc_flow_analyzer"):
                jobs.append((tag, tool, paths, selected))
        before, after = trees["baseline"], trees["probe"]
        raw = rb"(?:set_flag|wait_flag|wait_flag_dev|ffts_cross_core_sync|pipe_barrier|set_quant_pre|mad|copy_matrix_cc_to_gm)\([^;]*\);"
        checks = {}
        for name in ("CubeForward.h", "VectorForward.h", "AddressMappingForwardOnline.h", "AddressMappingVectorForwardOnline.h"):
            key = "op_kernel/" + name
            first, second = before[key], after[key]
            first_calls, second_calls = re.findall(raw, first), re.findall(raw, second)
            checks[name] = dict(baseline_sha256=sha(first), probe_sha256=sha(second),
                byte_identical=first == second, baseline_raw_calls=len(first_calls),
                probe_raw_calls=len(second_calls), raw_calls_identical=first_calls == second_calls)
            if name == "CubeForward.h":
                checks[name]["inverse_byte_identical"] = remove_hooks(second) == first
                diff = "".join(difflib.unified_diff(first.decode().splitlines(keepends=True),
                    second.decode().splitlines(keepends=True),
                    f"{family}/baseline/{name}", f"{family}/probe/{name}", n=0))
                (REPORT / f"{family}_CubeForward.h.diff").write_text(diff, encoding="utf-8")
                assert checks[name]["inverse_byte_identical"]
            else:
                assert first == second
            assert first_calls == second_calls
        comparison[family] = checks
    write_json(REPORT / "source_comparison.json", comparison)

    def run(job):
        tag, name, paths, files = job
        script = TOOLS / (name + ".py")
        command = [sys.executable, "-X", "utf8", str(script), *paths, "--format", "json"]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60,
                                env={**os.environ, "PYTHONUTF8": "1"})
        (REPORT / f"{tag}_{name}.json").write_bytes(result.stdout)
        (REPORT / f"{tag}_{name}.stderr.txt").write_bytes(result.stderr)
        raw = json.loads(result.stdout.decode("utf-8"))
        record = dict(command=command, returncode=result.returncode, tool_sha256=sha(script.read_bytes()),
            files={key: sha(value) for key, value in sorted(files.items())},
            findings=len(raw.get("findings", [])), event_count=raw.get("event_count"),
            operation_count=len(raw.get("operations", [])))
        write_json(REPORT / f"{tag}_{name}_run.json", record)
        return dict(tree=tag, tool=name, returncode=result.returncode, files=len(files),
            findings=record["findings"], events=record["event_count"], operations=record["operation_count"])

    with ThreadPoolExecutor(max_workers=4) as workers:
        summary = list(workers.map(run, jobs))
    write_json(REPORT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
