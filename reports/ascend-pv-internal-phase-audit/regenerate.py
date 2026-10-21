"""Audit a separate V-only PV diagnostic against its pinned production parent."""
from concurrent.futures import ThreadPoolExecutor
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import prepare_mindie_laser_cube_probe_source as source

REPORT = Path(__file__).resolve().parent
BUILD = ROOT / "build/ascend_pv_internal_phase_audit"
REFERENCE = ROOT / "build/mindie-sd-reference"
TOOLS = ROOT / "build/cannbot-skills/ops/ascendc-sync-audit/scripts"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    from scripts import mindie_laser_pv_probe_patch as patch
    prior_code = subprocess.run(["git", "show", "67b9e75:scripts/prepare_mindie_laser_cube_probe_source.py"],
                                cwd=ROOT, capture_output=True, check=True).stdout
    prior = types.ModuleType("pv_audit_prior_preparer")
    prior.__file__ = str(ROOT / "scripts/prepare_mindie_laser_cube_probe_source.py")
    exec(compile(prior_code, prior.__file__, "exec"), prior.__dict__)
    legacy = {}
    for family in ("int8", "vscale", "fp16"):
        old_files, old_records = prior.expected_files(REFERENCE, family)
        new_files, new_records = source.expected_files(REFERENCE, family)
        legacy[family] = dict(files_identical=old_files == new_files,
            records_identical=old_records == new_records,
            identity_identical=prior._manifest_identity(REFERENCE, family) == source._manifest_identity(REFERENCE, family))
        assert all(legacy[family].values()), "Existing probe source/cache changed: " + family
    write_json(REPORT / "legacy_equivalence.json", dict(prior_commit="67b9e75", checks=legacy))
    original, _ = source.baseline_files(REFERENCE, "pv")
    candidate, _ = source.expected_files(REFERENCE, "pv")
    jobs, trees = [], {}
    for tag, files in (("baseline", original), ("probe", candidate)):
        selected = {}
        for name, data in files.items():
            if name.startswith("csrc/ops/") and Path(name).suffix in (".h", ".hpp", ".cpp"):
                relative = "/".join(name.split("/")[3:])
                selected[relative] = data
                path = BUILD / tag / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        trees[tag] = selected
        paths = [str(BUILD / tag / name) for name in sorted(selected)]
        for tool in ("sync_audit", "ascendc_flow_analyzer"):
            jobs.append((tag, tool, paths, selected))

    before, after = trees["baseline"], trees["probe"]
    # Preserve production instruction/event sequence, not just event counts.
    raw = rb"(?:set_flag|wait_flag|wait_flag_dev|ffts_cross_core_sync|pipe_barrier|set_quant_pre|mad|copy_matrix_cc_to_gm|copy_gm_to_cbuf_multi_nd2nz_b16|load_cbuf_to_ca|load_cbuf_to_cb)\([^;]*\);"
    checks = {}
    for name in ("CubeForward.h", "VectorForward.h", "AddressMappingForwardOnline.h", "AddressMappingVectorForwardOnline.h"):
        key = "op_kernel/" + name
        a, b = before[key], after[key]
        calls_a, calls_b = re.findall(raw, a), re.findall(raw, b)
        checks[name] = dict(baseline_sha256=sha(a), probe_sha256=sha(b),
            byte_identical=a == b, baseline_raw_calls=len(calls_a), probe_raw_calls=len(calls_b),
            raw_calls_identical=calls_a == calls_b)
        assert calls_a == calls_b, "Production instruction sequence changed: " + name
        if name == "CubeForward.h":
            checks[name]["inverse_byte_identical"] = patch.remove_hooks(b) == a
            assert checks[name]["inverse_byte_identical"], "Hooks do not invert to V-only"
            diff = "".join(difflib.unified_diff(a.decode().splitlines(keepends=True),
                b.decode().splitlines(keepends=True), "baseline/" + name, "probe/" + name, n=0))
            (REPORT / "CubeForward.h.diff").write_text(diff, encoding="utf-8")
        else:
            assert a == b, name
    write_json(REPORT / "source_comparison.json", dict(
        upstream_commit=source.SOURCE_LOCK.commit, parent="vector_pv_prefetch", checks=checks))

    def run(job):
        tag, name, paths, files = job
        script = TOOLS / (name + ".py")
        command = [sys.executable, "-X", "utf8", str(script), *paths, "--format", "json"]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60,
                                env={**os.environ, "PYTHONUTF8": "1"})
        (REPORT / f"{tag}_{name}.json").write_bytes(result.stdout)
        (REPORT / f"{tag}_{name}.stderr.txt").write_bytes(result.stderr)
        output = json.loads(result.stdout.decode("utf-8"))
        record = dict(command=command, returncode=result.returncode, tool_sha256=sha(script.read_bytes()),
            files={key: sha(value) for key, value in sorted(files.items())},
            findings=len(output.get("findings", [])), event_count=output.get("event_count"),
            operation_count=len(output.get("operations", [])))
        write_json(REPORT / f"{tag}_{name}_run.json", record)
        return dict(tree=tag, tool=name, returncode=result.returncode, files=len(files),
                    findings=record["findings"], events=record["event_count"], operations=record["operation_count"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        summary = list(pool.map(run, jobs))
    write_json(REPORT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
