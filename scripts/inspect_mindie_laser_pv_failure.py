#!/usr/bin/env python3
"""Collect existing PV probe failure evidence without loading or running NPU code.

Only standard-library modules and an existing CANN llvm-objdump are used. An
object must match a recorded build SHA256 before objdump reads it. This checks
local build artifacts, not the identity of the object loaded when a fault arose.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_SECONDS = 30
SCOPE = (
    "Offline collection of existing logs/build artifacts. No NPU imports, device calls, builds or network access. "
    "Build SHA agreement does not establish which object the runtime loaded. PC windows assume the reported "
    "current-start delta is a symbol-relative byte offset; they are static candidates, not confirmed fault PCs "
    "or a diagnosis. FIXP and other unknown error bitfields are not decoded. Resource values retain exported "
    "names and units; missing values are unknown, not zero or inferred spills."
)
SYMBOL_SCOPE = (
    "Offline build-object symbols only; no fault log, PC mapping, disassembly, NPU calls or network. "
    "Objects must match recorded build hashes. Function size differences are auxiliary code-generation "
    "evidence, not proof of dead-code elimination, identical instructions, register spills or a fault cause. "
    "Build SHA agreement does not identify a runtime-loaded object. Missing resource values stay unknown."
)
RESOURCE_KEY = re.compile(r"stack|local|smem|lmem|shared|register|spill|(^|_)ub($|_)|ubsize|ub_size", re.I)
PC_RE = re.compile(
    r"\bpc[ _]+start\s*[:=]\s*(0x[0-9a-f]+).*?\b(?:current(?:[ _]+pc)?)\s*[:=]\s*(0x[0-9a-f]+)",
    re.I,
)


def _error(report, context, error):
    report["errors"].append({"context": context, "error": str(error)})


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value, owner):
    """Resolve explicitly recorded paths; never search the filesystem by basename."""
    result = Path(value).expanduser()
    if not result.is_absolute():
        nearby = Path(owner).parent / result
        result = nearby if nearby.exists() else ROOT / result
    return result.resolve()


def _walk(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            location = f"{prefix}.{key}" if prefix else key
            yield location, key, child
            yield from _walk(child, location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{prefix}[{index}]")


def _read_json(path, report, context):
    try:
        with Path(path).open("r", encoding="utf-8-sig") as stream:
            result = json.load(stream)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object")
        return result
    except (OSError, ValueError) as error:
        _error(report, context, f"{type(error).__name__}: {error}")
        return None


def parse_pc_records(text, source="log"):
    result = []
    for number, line in enumerate(text.splitlines(), 1):
        for match in PC_RE.finditer(line):
            start, current = (int(value, 16) for value in match.groups())
            delta = current - start
            core = re.search(r"\bcore\s+id\s*(?:(?:is|[:=])\s*)?(\d+)", line, re.I)
            result.append({
                "source": str(source), "line_number": number, "line": line,
                "start": start, "current": current, "direct_delta": delta,
                "start_hex": hex(start), "current_hex": hex(current), "direct_delta_hex": hex(delta),
                "reported_core_id": int(core.group(1)) if core else None,
                "nonnegative_delta": delta >= 0, "delta_aligned_4": delta % 4 == 0,
                "address_domain_confirmed": False,
                "interpretation": "Arithmetic current-start only; neither a confirmed ELF VMA nor confirmed symbol offset.",
            })
    return result


def parse_symbols(text):
    """Read positive/zero-size function symbols from LLVM's --syms output."""
    result = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 5 or not re.fullmatch(r"[0-9a-fA-F]+", fields[0]):
            continue
        section = next((i for i, part in enumerate(fields[1:], 1) if part.startswith(".")), None)
        if section is None or "F" not in fields[1:section] or section + 2 >= len(fields):
            continue
        if not re.fullmatch(r"[0-9a-fA-F]+", fields[section+1]):
            continue
        start, size = int(fields[0], 16), int(fields[section+1], 16)
        name = fields[-1]
        lower = name.lower()
        result.append(dict(name=name, start=start, size=size, end=start+size, section=fields[section],
                           core_type="mix_aic" if "mix_aic" in lower else "mix_aiv" if "mix_aiv" in lower else None,
                           raw=line))
    return result


def parse_instructions(text):
    """Keep actual disassembled instruction lines, including their section/name."""
    result, section, symbol = [], None, None
    for line in text.splitlines():
        section_match = re.match(r"Disassembly of section (.+):\s*$", line)
        if section_match:
            section, symbol = section_match.group(1), None
            continue
        label = re.match(r"\s*[0-9a-fA-F]+\s+<([^>]+)>:\s*$", line)
        if label:
            symbol = label.group(1)
            continue
        instruction = re.match(r"\s*([0-9a-fA-F]+):\s+(.+)$", line)
        if not instruction:
            continue
        encoded, assembly = [], instruction.group(2).split()
        while assembly and re.fullmatch(r"(?:[0-9a-fA-F]{2}|[0-9a-fA-F]{4}|[0-9a-fA-F]{8}|[0-9a-fA-F]{16})", assembly[0]):
            encoded.append(assembly.pop(0))
        # Never turn a label, raw data word or <unknown> into an instruction hit.
        if not assembly or not re.match(r"[A-Za-z_]", assembly[0]):
            continue
        if encoded and sum(map(len, encoded)) < 8:
            continue
        result.append(dict(address=int(instruction.group(1), 16), text=line,
                           section=section, symbol=symbol, assembly=" ".join(assembly)))
    return result


def disassembly_diagnostics(text, instructions):
    """Distinguish absent tool mnemonics from an unsupported text layout."""
    address_lines = [line for line in text.splitlines()
                     if re.match(r"\s*[0-9a-fA-F]+:\s+", line)]
    unavailable = sum("<not available>" in line.lower() for line in address_lines)
    unknown = sum("<unknown>" in line.lower() for line in address_lines)
    status = ("unavailable" if unavailable and not instructions else
              "partial" if unavailable or unknown else
              "parsed" if instructions else "unparsed")
    return dict(status=status, address_rows=len(address_lines), parsed_rows=len(instructions),
                not_available_rows=unavailable, unknown_rows=unknown,
                scope="Address placeholders are not instructions. Output alone does not explain why decoding is unavailable.")


def candidate_windows(pc_records, symbols, instructions):
    """Bound/check each hypothesis, without choosing an unproven AIC/AIV side."""
    windows, rejected = [], []
    for index, pc in enumerate(pc_records):
        delta = pc["direct_delta"]
        for symbol in symbols:
            if symbol.get("core_type") not in ("mix_aic", "mix_aiv"):
                continue
            common = {"pc_record": index, "symbol": symbol["name"], "section": symbol["section"],
                      "core_type": symbol["core_type"], "symbol_start": symbol["start"],
                      "symbol_size": symbol["size"], "symbol_end": symbol["end"], "assumed_offset": delta}
            target = symbol["start"] + delta
            reason = None
            if delta < 0:
                reason = "negative current-start delta"
            elif delta % 4 or target % 4:
                reason = "hypothesis is not 4-byte aligned"
            elif symbol["size"] <= 0 or not symbol["start"] <= target < symbol["end"]:
                reason = "hypothesis is outside the sized function symbol"
            scoped = [row for row in instructions if row["section"] == symbol["section"] and
                      symbol["start"] <= row["address"] < symbol["end"] and
                      row.get("symbol") in (None, symbol["name"])]
            matches = [row for row in scoped if row["address"] == target]
            if reason is None and len(matches) != 1:
                reason = "no unique actual disassembled instruction at the hypothesized address"
            if reason:
                rejected.append(dict(common, reason=reason))
                continue
            window = [row for row in scoped if max(symbol["start"], target-64) <= row["address"] < min(symbol["end"], target+68)]
            windows.append(dict(common, target_address=target, target_hex=hex(target), instruction=matches[0],
                instructions=window, confirmed_fault_pc=False, runtime_object_identity_confirmed=False,
                hypothesis="symbol.start + (reported_current_pc - reported_start_pc); byte offset and runtime symbol identity are UNCONFIRMED"))
    return {"windows": windows, "rejected": rejected}


def _resources(metadata):
    return [{"path": location, "key": key, "value": value}
            for location, key, value in _walk(metadata) if RESOURCE_KEY.search(key)]


def _cann_roots(documents):
    roots = []
    for owner, document in documents:
        for _, key, value in _walk(document):
            if key in ("cann_root", "CANN_ROOT", "CUSTOM_ASCEND_CANN_PACKAGE_PATH", "ASCEND_HOME_PATH") and isinstance(value, str) and value:
                path = _path(value, owner)
                if path not in roots:
                    roots.append(path)
    return roots


def discover_objdump(documents, override=None):
    """Search bounded CANN compiler directories, never PATH or the whole system."""
    roots = _cann_roots(documents)
    result = {"path": None, "explicit_override": bool(override), "cann_roots": list(map(str, roots)), "checked": []}
    if override:
        candidate = Path(override).expanduser().resolve()
        result["checked"].append(str(candidate))
        if candidate.is_file():
            result["path"] = str(candidate)
        else:
            result["error"] = f"Explicit llvm-objdump path is not a file: {candidate}"
        return result
    directories = ("compiler/ccec_compiler/bin", "compiler/bin", "compiler/bisheng/bin",
                   "compiler/bisheng_compiler/bin", "tools/ccec_compiler/bin",
                   "tools/bisheng_compiler/bin", "tools/bisheng/bin", "bin",
                   "aarch64-linux/compiler/ccec_compiler/bin", "x86_64-linux/compiler/ccec_compiler/bin")
    for root in roots:
        for directory in directories:
            for filename in ("llvm-objdump", "llvm-objdump.exe"):
                candidate = (root / directory / filename).resolve()
                result["checked"].append(str(candidate))
                # A symlink escaping this manifest's CANN root is not trusted implicitly.
                if candidate.is_relative_to(root) and candidate.is_file():
                    result["path"] = str(candidate)
                    return result
    result["error"] = "No llvm-objdump found inside the recorded CANN compiler/tool directories; explicit --objdump is available."
    return result


def _run_objdump(tool, flag, object_path, directory, runner):
    name = {"--syms": "symbols", "--disassemble": "disassembly", "--version": "version"}[flag]
    stdout, stderr = directory / (name+".txt"), directory / (name+".stderr.txt")
    command = [str(tool), flag] + ([str(object_path)] if object_path is not None else [])
    evidence = {"command": command, "timeout_seconds": TIMEOUT_SECONDS,
                "stdout_path": str(stdout), "stderr_path": str(stderr), "returncode": None}
    try:
        with stdout.open("wb") as out, stderr.open("wb") as err:
            completed = runner(command, stdout=out, stderr=err, timeout=TIMEOUT_SECONDS, check=False, shell=False)
        evidence["returncode"] = completed.returncode
        if completed.returncode:
            evidence["error"] = f"llvm-objdump exited with code {completed.returncode}"
    except (OSError, subprocess.SubprocessError) as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
    return evidence


def raw_output_preview(command, *, lines=45, line_chars=300):
    """Bounded verbatim evidence when a tool succeeds but yields no parsed ISA.

    This is deliberately not a second instruction parser: headers, raw words,
    unknown opcodes and error messages remain uninterpreted text.
    """
    result = {"command": command["command"], "returncode": command["returncode"],
              "interpretation": "Raw tool output only; no instruction or PC mapping implied."}
    for key in ("stdout_path", "stderr_path"):
        path = Path(command[key])
        item = result[key] = {"path": str(path), "lines": []}
        try:
            item["bytes"] = path.stat().st_size
            with path.open(encoding="utf-8", errors="replace") as stream:
                for _ in range(lines):
                    line = stream.readline(line_chars+2)
                    if not line:
                        break
                    content = line.rstrip("\r\n")
                    if len(content) > line_chars:
                        item["lines"].append(content[:line_chars]+" ... [line truncated]")
                        # Do not scan an arbitrarily large single line to skip it.
                        item["truncated"] = True
                        break
                    item["lines"].append(content)
                else:
                    item["truncated"] = bool(stream.read(1))
        except OSError as error:
            item["error"] = f"{type(error).__name__}: {error}"
    return result


def _profile(path, directory, report):
    evidence = {"path": str(path), "failure_contexts": [], "logs": [], "pc_records": []}
    document = _read_json(path, report, "profile")
    if document is None:
        return evidence
    try:
        evidence["sha256"] = _sha(path)
        shutil.copyfile(path, directory / "profile.json")
    except OSError as error:
        _error(report, "profile snapshot", error)
    seen_logs, text_fields = set(), []
    for location, key, value in _walk(document):
        if key in ("probe_validation_failure", "first_device_failure", "active_probe_validation"):
            evidence["failure_contexts"].append({"location": location, "as_exported": value})
        if isinstance(value, str) and PC_RE.search(value):
            text_fields.append((f"{path}:{location}", value))
        if key == "worker_log" and isinstance(value, str) and value:
            log_path = _path(value, path)
            if log_path in seen_logs:
                continue
            seen_logs.add(log_path)
            log = {"path": str(log_path)}
            evidence["logs"].append(log)
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                target = directory / f"worker_{len(seen_logs):02d}.log"
                shutil.copyfile(log_path, target)
                log.update(sha256=_sha(log_path), snapshot=str(target))
                text_fields.append((str(log_path), text))
            except OSError as error:
                log["error"] = f"{type(error).__name__}: {error}"
                _error(report, "worker log", log["error"])
    for source, text in text_fields:
        evidence["pc_records"].extend(parse_pc_records(text, source))
    unique = {}
    for pc in evidence["pc_records"]:
        key = (pc["start"], pc["current"], pc["reported_core_id"])
        source = {field: pc[field] for field in ("source", "line_number", "line")}
        if key not in unique:
            unique[key] = dict(pc, sources=[])
        if source not in unique[key]["sources"]:
            unique[key]["sources"].append(source)
    evidence["pc_records"] = list(unique.values())
    if not evidence["pc_records"]:
        _error(report, "PC evidence", "No reported pc start/current pair in available profile fields or referenced worker logs")
    return evidence


def _documents(path, report, label):
    """Read the requested manifest and its explicit operator_manifest links only."""
    todo, result, seen = [path], [], set()
    while todo:
        owner = todo.pop(0).resolve()
        if owner in seen:
            continue
        if len(seen) >= 8:
            _error(report, label, "More than eight linked manifests; remaining explicit links not followed")
            break
        seen.add(owner)
        document = _read_json(owner, report, f"{label} manifest {owner}")
        if document is None:
            continue
        result.append((owner, document))
        for _, key, value in _walk(document):
            if key == "operator_manifest":
                link = value if isinstance(value, str) else value.get("path") if isinstance(value, dict) else None
                if isinstance(link, str) and link:
                    todo.append(_path(link, owner))
    return result


def _objects(documents, report, label):
    result, seen = [], set()
    for owner, document in documents:
        for location, key, value in _walk(document):
            if key != "device_binaries":
                continue
            if not isinstance(value, list):
                _error(report, label, f"{location} is not a list")
                continue
            for item in value:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    _error(report, label, f"Invalid device_binaries entry at {owner}:{location}")
                    continue
                path = _path(item["path"], owner)
                expected = item.get("sha256")
                key = (str(path), str(expected))
                if key in seen:
                    continue
                seen.add(key)
                result.append({"path": str(path), "expected_sha256": expected,
                               "manifest": str(owner), "manifest_location": location,
                               "fingerprint_as_exported": item, "actual_sha256": None, "sha256_matches": False,
                               "runtime_object_identity_confirmed": False})
    return result


def _metadata(object_path, directory, report, label):
    path = object_path.with_suffix(".json")
    result = {"path": str(path), "as_exported": None, "resource_fields": [], "resource_status": "unknown",
              "declared_object_sha256": [], "declared_sha256_matches": None,
              "binding": "same-stem metadata currently on disk; runtime provenance and original build hash are unconfirmed"}
    document = _read_json(path, report, f"{label} metadata {path}")
    if document is None:
        return result
    result.update(as_exported=document, resource_fields=_resources(document))
    # Only explicit whole-object digest fields bind this file. Nested source,
    # build or MIX subkernel digests may legitimately refer to different bytes.
    result["declared_object_sha256"] = [
        {"path": key, "value": value}
        for key, value in document.items()
        if key.lower().replace("_", "") in ("sha256", "binarysha256", "objectsha256", "sha256hash", "sha256sum")
        and isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value)
    ]
    result["resource_status"] = "exported" if result["resource_fields"] else "unknown: no matching resource keys exported"
    try:
        result["sha256"] = _sha(path)
        snapshot = directory / "metadata.json"
        shutil.copyfile(path, snapshot)
        result["snapshot"] = str(snapshot)
    except OSError as error:
        _error(report, label+" metadata snapshot", error)
    return result


def _new_report():
    return {"schema_version": 1, "status": "partial", "scope": SCOPE, "errors": [],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_object_identity_confirmed": False, "providers": {}, "tool_versions": {}}


def collect(args, *, runner=None, report=None):
    report = _new_report() if report is None else report
    symbols_only = getattr(args, "symbols_only", False)
    report["symbols_only"] = symbols_only
    if symbols_only:
        report["scope"] = SYMBOL_SCOPE
    input_keys = ("probe_build", "baseline_build") if symbols_only else ("profile", "probe_build", "baseline_build")
    report["requested_inputs"] = {key: str(getattr(args, key)) for key in input_keys}
    runner = subprocess.run if runner is None else runner
    output = Path(args.output).resolve()
    base = Path(args.artifact_dir).resolve() if args.artifact_dir else output.parent
    base.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="mindie_laser_pv_symbols_" if symbols_only else "mindie_laser_pv_failure_", dir=base))
    report["artifact_dir"] = str(directory)
    report["profile"] = ({"status": "not_requested", "pc_records": []} if symbols_only else
                         _profile(Path(args.profile).resolve(), directory, report))
    for label, source in (("probe", args.probe_build), ("baseline", args.baseline_build)):
        path = Path(source).resolve()
        destination = directory / label
        destination.mkdir()
        evidence = {"manifest": str(path), "objects": [], "manifest_snapshots": []}
        report["providers"][label] = evidence
        documents = _documents(path, report, label)
        for number, (owner, document) in enumerate(documents):
            try:
                snapshot = destination / f"manifest_{number:02d}.json"
                shutil.copyfile(owner, snapshot)
                evidence["manifest_snapshots"].append({"path": str(owner), "sha256": _sha(owner), "snapshot": str(snapshot)})
            except OSError as error:
                _error(report, label+" manifest snapshot", error)
        tool = discover_objdump(documents, args.objdump)
        evidence["tool"] = tool
        if not tool["path"]:
            _error(report, label+" llvm-objdump", tool["error"])
        evidence["objects"] = _objects(documents, report, label)
        if not evidence["objects"]:
            _error(report, label+" objects", "No recorded device_binaries found in available manifests")
        for number, obj in enumerate(evidence["objects"]):
            object_path = Path(obj["path"])
            print(f"Offline inspect {label} {number+1}/{len(evidence['objects'])}: {object_path.name}", flush=True)
            object_dir = destination / f"object_{number:02d}"
            object_dir.mkdir()
            obj.update(commands=[], symbols=[], pc_candidates={"windows": [], "rejected": []}, tool=tool["path"],
                       inspection_status="not_inspected")
            obj["metadata"] = _metadata(object_path, object_dir, report, label)
            try:
                obj["actual_sha256"] = _sha(object_path)
                expected = obj["expected_sha256"]
                obj["sha256_matches"] = (isinstance(expected, str) and bool(re.fullmatch(r"[a-fA-F0-9]{64}", expected))
                                         and expected.lower() == obj["actual_sha256"])
                if not obj["sha256_matches"]:
                    raise ValueError("Object does not match an available valid build SHA256; objdump skipped")
                declared = obj["metadata"]["declared_object_sha256"]
                if declared:
                    for digest in declared:
                        digest["matches_object"] = digest["value"].lower() == obj["actual_sha256"]
                    obj["metadata"]["declared_sha256_matches"] = all(row["matches_object"] for row in declared)
                    if not obj["metadata"]["declared_sha256_matches"]:
                        obj["metadata"]["binding"] = "UNTRUSTED: metadata-declared SHA256 does not match the current object"
                        raise ValueError("Metadata-declared SHA256 mismatch; objdump and PC mapping skipped")
                if object_path.suffix.lower() not in (".o", ".bin"):
                    raise ValueError("Only recorded .o/.bin device artifacts are inspected")
            except (OSError, ValueError) as error:
                obj["error"] = f"{type(error).__name__}: {error}"
                _error(report, label+" object "+str(object_path), obj["error"])
                continue
            if not tool["path"]:
                obj["inspection_status"] = "tool_unavailable"
                continue
            if tool["path"] not in report["tool_versions"]:
                version = _run_objdump(tool["path"], "--version", None, destination, runner)
                report["tool_versions"][tool["path"]] = version
                if "error" in version:
                    _error(report, label+" objdump version", version["error"])
            tool["version_command"] = report["tool_versions"][tool["path"]]
            for flag in (("--syms",) if symbols_only else ("--syms", "--disassemble")):
                # Recheck immediately before each command, in case the build tree changed.
                try:
                    if _sha(object_path) != obj["actual_sha256"]:
                        raise ValueError("Object changed after SHA verification; remaining objdump calls skipped")
                    command = _run_objdump(tool["path"], flag, object_path, object_dir, runner)
                    obj["commands"].append(command)
                    if "error" in command:
                        _error(report, label+" objdump "+flag, command["error"])
                except (OSError, ValueError) as error:
                    obj["error"] = f"{type(error).__name__}: {error}"
                    _error(report, label+" inspection", obj["error"])
                    break
            successful = {item["command"][1]: item for item in obj["commands"] if item["returncode"] == 0 and "error" not in item}
            instructions = []
            try:
                if "--syms" in successful:
                    obj["symbols"] = parse_symbols(Path(successful["--syms"]["stdout_path"]).read_text(encoding="utf-8", errors="replace"))
                    if not obj["symbols"]:
                        _error(report, label+" symbols", "No sized function symbol table rows parsed; raw output retained")
                if "--disassemble" in successful:
                    raw_disassembly = Path(successful["--disassemble"]["stdout_path"]).read_text(encoding="utf-8", errors="replace")
                    instructions = parse_instructions(raw_disassembly)
                    obj["disassembly_diagnostics"] = disassembly_diagnostics(raw_disassembly, instructions)
                    obj["instruction_count"] = len(instructions)
                    diagnostic = obj["disassembly_diagnostics"]
                    if diagnostic["status"] == "unavailable":
                        _error(report, label+" disassembly", "llvm-objdump emitted <not available> address placeholders, not instructions; changing the parser cannot decode them")
                    elif diagnostic["status"] == "partial":
                        _error(report, label+" disassembly", "Some addresses have unavailable/unknown instructions; only actual decoded instructions can form static candidates")
                    elif not instructions:
                        _error(report, label+" disassembly", "No instruction rows parsed; raw output retained")
                    obj["pc_candidates"] = candidate_windows(report["profile"]["pc_records"], obj["symbols"], instructions)
            except OSError as error:
                _error(report, label+" objdump output", error)
            if not instructions or obj.get("disassembly_diagnostics", {}).get("status") == "partial":
                obj["unparsed_disassembly_preview"] = [
                    raw_output_preview(item) for item in obj["commands"]
                    if item["command"][1] == "--disassemble"]
            diagnostic_status = obj.get("disassembly_diagnostics", {}).get("status")
            obj["inspection_status"] = ("symbols_collected" if symbols_only and "--syms" in successful and obj["symbols"] else
                "disassembly_unavailable" if diagnostic_status == "unavailable" else
                "collected" if len(successful) == 2 and obj["symbols"] and diagnostic_status == "parsed" else "partial")
    report["resource_comparison"] = [
        {"provider": label, "object": obj["path"], "sha256_matches": obj["sha256_matches"],
         "resource_status": obj["metadata"]["resource_status"], "exported_fields": obj["metadata"]["resource_fields"],
         "function_symbols": [{key: symbol[key] for key in ("name", "section", "start", "size", "core_type")}
                              for symbol in obj["symbols"]]}
        for label, provider in report["providers"].items() for obj in provider["objects"]
    ]
    report["status"] = "partial" if report["errors"] else "collected"
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--symbols-only", action="store_true", help="Inspect build symbol sizes only; ignore --profile and do not request disassembly or PC mapping")
    result.add_argument("--profile", type=Path, default=ROOT / "results/mindie_laser_pv_phases.json")
    result.add_argument("--probe-build", type=Path, default=ROOT / "results/mindie_laser_cube_pv_probe_build.json")
    result.add_argument("--baseline-build", type=Path, default=ROOT / "results/mindie_laser_quant_vscale_pv_prefetch_build.json")
    result.add_argument("--output", type=Path, default=ROOT / "results/mindie_laser_pv_failure.json")
    result.add_argument("--artifact-dir", type=Path, help="Parent for a fresh evidence directory (default: next to --output)")
    result.add_argument("--objdump", type=Path, help="Explicit existing llvm-objdump executable; permits a path outside recorded CANN roots")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    report = _new_report()
    try:
        collect(args, report=report)
    except Exception as error:
        # Keep every earlier fingerprint, log and raw output on unexpected errors.
        report["status"] = "partial"
        _error(report, "collection", f"{type(error).__name__}: {error}")
    try:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name+".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        temporary.replace(output)
    except OSError as error:
        print(f"Could not save offline evidence: {error}", file=sys.stderr)
        return 1
    label = "symbols" if report.get("symbols_only") else "evidence"
    print(f"Offline PV {label}: {report['status']}; errors={len(report['errors'])}; runtime object identity UNCONFIRMED")
    if report.get("symbols_only"):
        print("Function sizes do not prove instruction equality, eliminated instrumentation, spills or a fault cause.")
    for label, provider in report.get("providers", {}).items():
        print(f"  {label} llvm-objdump: {provider.get('tool', {}).get('path') or 'unavailable'}")
        for obj in provider.get("objects", []):
            candidates = obj.get("pc_candidates", {}).get("windows", [])
            metadata, symbols = obj.get("metadata", {}), obj.get("symbols", [])
            windows = len(candidates)
            print(f"  {label}: {Path(obj['path']).name}; SHA match={obj['sha256_matches']}; "
                  f"symbols={len(symbols)}; static PC candidates={windows}; resources={metadata.get('resource_status', 'unknown')}")
            if obj.get("disassembly_diagnostics"):
                print("    disassembly_diagnostics="+json.dumps(obj["disassembly_diagnostics"], ensure_ascii=True))
                if obj["disassembly_diagnostics"]["status"] == "unavailable":
                    print("    Tool did not provide instructions (<not available>); no parser fix or PC instruction mapping is possible from these placeholders.")
            for field in metadata.get("resource_fields", [])[:8]:
                value = json.dumps(field["value"], ensure_ascii=True)
                print(f"    exported {field['path']}={value[:180]}{' ... (see JSON)' if len(value)>180 else ''}")
            for symbol in [row for row in symbols if row["core_type"]][:4]:
                print(f"    {symbol['core_type']} {symbol['name']}: start={symbol['start']:#x}, size={symbol['size']} bytes")
            for window in candidates[:2]:
                print(f"    UNCONFIRMED candidate {window['symbol']} + {window['assumed_offset']:#x} -> {window['target_hex']}")
                rows = window["instructions"]
                hit = next(i for i, row in enumerate(rows) if row["address"] == window["target_address"])
                for row in rows[max(0, hit-3):hit+4]:
                    print(f"      {'>' if row['address'] == window['target_address'] else ' '} {row['text'].strip()}")
            for preview in obj.get("unparsed_disassembly_preview", []):
                print(f"    RAW disassembler output (uninterpreted); returncode={preview['returncode']}")
                print(f"    command={json.dumps(preview['command'], ensure_ascii=True)}")
                for key in ("stdout_path", "stderr_path"):
                    item = preview[key]
                    print(f"    {key}: {item['path']}; bytes={item.get('bytes', 'unknown')}")
                    for line in item["lines"]:
                        print("      "+line)
                    if item.get("error"):
                        print("      "+item["error"])
                    if item.get("truncated"):
                        print("      [preview truncated; full raw output retained]")
    for pc in report.get("profile", {}).get("pc_records", [])[:4]:
        print(f"  reported PC delta={pc['direct_delta_hex']} (hypothesis only)")
    for error in report["errors"][:8]:
        print(f"  {error['context']}: {error['error']}")
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
