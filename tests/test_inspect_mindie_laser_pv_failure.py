"""Offline failure inspection preserves evidence without launching NPU work."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import pytest

from scripts import inspect_mindie_laser_pv_failure as inspect


ERROR_LOG = """PV probe validation: mode=vector_pv_prefetch shape=[1, 2, 1, 128] K=1 isolated=True about_to_launch
PV probe validation: mode=plain shape=[1, 2, 1, 128] K=1 isolated=True about_to_launch
RuntimeError: AclrtSynchronizeDeviceWithTimeout, error code is 507014
EZ9999: exception of fftsplus aicore error, core id is 20, pc start: 0x12c08e614000, current: 0x12c08e6190b4, vec error info: 0, mte error info: 0x5800000093
fixp_error0 info: 0x93, fixp_error1 info: 0x58
"""

SYMBOLS = """SYMBOL TABLE:
0000000000000000 g     F .text 0000000000006000 vc_laser_cube_pv_probe_attention_mix_aic
0000000000008000 g     F .text 0000000000001000 vc_laser_cube_pv_probe_attention_mix_aiv
"""

TOOL_VERSION = "LLVM version fixture\n  Registered Targets:\n    hiipu - fixture target\n"

# Minimized from the remote CANN display: address rows contain this placeholder,
# not encoded bytes or decoded mnemonics. Addresses are sparse for a small test.
HIIPU_UNAVAILABLE = """Kernel_hash.bin: file format elf64-hiipu

Disassembly of section .text:

0000000000000000 <vc_laser_cube_pv_probe_attention_mix_aic>:
       0: <not available>
       4: <not available>
       8: <not available>
       c: <not available>
    50b4: <not available>
"""

DISASSEMBLY = """Disassembly of section .text:
0000000000000000 <vc_laser_cube_pv_probe_attention_mix_aic>:
    50ac: 00 00 00 00    fixture_before
    50b0: 00 00 00 00    fixture_before_two
    50b4: 00 00 00 00    fixture_instruction
    50b8: 00 00 00 00    fixture_after
0000000000008000 <vc_laser_cube_pv_probe_attention_mix_aiv>:
    8000: 00 00 00 00    fixture_other_core
"""


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def fingerprint(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


@pytest.fixture
def evidence(tmp_path):
    vendor = tmp_path / "installed provider with spaces"
    native = (vendor / "op_impl" / "ai_core" / "tbe" / "kernel" / "ascend910_93" /
              "vc_laser_cube_pv_probe_attention" / "Kernel_hash.bin")
    native.parent.mkdir(parents=True)
    native.write_bytes(b"\x7fELF fixture compiled kernel; do not alter")
    metadata = write_json(native.with_suffix(".json"), {
        "kernelName": "vc_laser_cube_pv_probe_attention",
        "blockDim": 24, "stackSize": 2048,
        "unknownCompilerField": {"asExported": "retained"},
    })
    manifest = write_json(tmp_path / "probe_build.json", {
        "status": "passed", "soc": "ascend910_93", "cann_root": str(tmp_path / "absent_cann"),
        "vendor_root": str(vendor), "device_binaries": [fingerprint(native)],
    })
    baseline = write_json(tmp_path / "baseline_build.json", {
        "status": "passed", "soc": "ascend910_93", "cann_root": str(tmp_path / "absent_cann"),
        "vendor_root": str(vendor), "device_binaries": [fingerprint(native)],
    })
    log = tmp_path / "previous worker" / "worker.log"
    log.parent.mkdir()
    log.write_text(ERROR_LOG, encoding="utf-8")
    failure = {"mode": "plain", "stage": "awaiting_sync", "isolated": True,
               "shape_bnsd": [1, 2, 1, 128], "keys": 1,
               "error": "RuntimeError: first device timeout 507014",
               "further_device_access_aborted": True}
    profile = write_json(tmp_path / "profile.json", {
        "status": "failed", "workers": {"paired": {"status": "failed", "returncode": 1,
            "worker_log": str(log), "probe_validation_failure": failure}},
    })
    tool = tmp_path / "tool directory" / "llvm-objdump.exe"
    tool.parent.mkdir()
    tool.write_bytes(b"fake executable: intercepted by runner")
    return SimpleNamespace(native=native, metadata=metadata, manifest=manifest, baseline=baseline,
                           profile=profile, log=log, failure=failure, tool=tool, directory=tmp_path)


def options(evidence, *, tool=True):
    argv = ["--profile", str(evidence.profile), "--probe-build", str(evidence.manifest),
            "--baseline-build", str(evidence.baseline),
            "--output", str(evidence.directory / "inspection.json"),
            "--artifact-dir", str(evidence.directory / "inspection artifacts")]
    if tool:
        argv += ["--objdump", str(evidence.tool)]
    return inspect.parser().parse_args(argv)


def fake_runner(commands):
    def run(command, **kwargs):
        commands.append(command)
        assert Path(command[0]).name == "llvm-objdump.exe"
        assert kwargs.get("shell", False) is False
        assert kwargs.get("check", False) is False
        outputs = {"--version": TOOL_VERSION, "--syms": SYMBOLS, "--disassemble": DISASSEMBLY}
        kwargs["stdout"].write(outputs[command[1]].encode())
        kwargs["stderr"].write(b"fixture stderr retained\n")
        return SimpleNamespace(returncode=0)
    return run


def test_symbols_only_ignores_failed_profile_and_never_disassembles(evidence, monkeypatch):
    args = options(evidence)
    args.symbols_only = True
    args.profile = evidence.directory / "missing-failure-report.json"
    monkeypatch.setattr(inspect, "_profile", lambda *args: pytest.fail("Must not read old failure context"))
    commands = []
    result = inspect.collect(args, runner=fake_runner(commands))
    assert result["status"] == "collected" and not result["errors"]
    assert result["symbols_only"] and "profile" not in result["requested_inputs"]
    assert result["profile"] == {"status": "not_requested", "pc_records": []}
    assert [command[1] for command in commands] == ["--version", "--syms", "--syms"]
    for provider in result["providers"].values():
        obj = provider["objects"][0]
        assert obj["inspection_status"] == "symbols_collected"
        assert [symbol["size"] for symbol in obj["symbols"]] == [0x6000, 0x1000]
        assert not obj["pc_candidates"]["windows"]
        assert "disassembly_diagnostics" not in obj


def test_symbols_only_still_requires_matching_build_and_metadata_hashes(evidence):
    args = options(evidence)
    args.symbols_only = True
    record = json.loads(evidence.metadata.read_text(encoding="utf-8"))
    record["sha256"] = "0" * 64
    write_json(evidence.metadata, record)
    commands = []
    result = inspect.collect(args, runner=fake_runner(commands))
    assert result["status"] == "partial" and not commands
    assert all(not obj["symbols"] for p in result["providers"].values() for obj in p["objects"])


def test_hash_mismatch_never_executes_disassembler(evidence):
    record = json.loads(evidence.manifest.read_text(encoding="utf-8"))
    record["device_binaries"][0]["sha256"] = "0" * 64
    write_json(evidence.manifest, record)
    # Both providers fail verification: not even a --version tool invocation.
    write_json(evidence.baseline, record)
    commands = []
    result = inspect.collect(options(evidence), runner=fake_runner(commands))
    assert commands == []
    assert result["status"] == "partial"
    for provider in result["providers"].values():
        assert provider["objects"][0]["sha256_matches"] is False
    assert evidence.native.read_bytes() == b"\x7fELF fixture compiled kernel; do not alter"


def test_missing_tool_preserves_failure_and_binary_metadata(evidence, monkeypatch):
    monkeypatch.setenv("PATH", "")
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "CANN_ROOT"):
        monkeypatch.delenv(name, raising=False)
    args = options(evidence)
    args.objdump = str(evidence.directory / "missing-objdump")
    commands = []
    result = inspect.collect(args, runner=fake_runner(commands))
    assert commands == []
    assert result["status"] == "partial"
    assert result["runtime_object_identity_confirmed"] is False
    assert result["profile"]["failure_contexts"]
    assert "plain" in json.dumps(result["profile"]["failure_contexts"])
    assert result["profile"]["pc_records"][0]["direct_delta"] == 0x50B4
    obj = result["providers"]["probe"]["objects"][0]
    assert obj["sha256_matches"] is True
    assert obj["metadata"]["as_exported"]["unknownCompilerField"] == {"asExported": "retained"}


def test_inspection_does_not_import_torch_or_open_network(evidence, monkeypatch):
    original_import = builtins.__import__
    def import_without_device(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torch_npu", "acl", "requests"}:
            raise AssertionError(f"offline inspector attempted import: {name}")
        return original_import(name, *args, **kwargs)
    def no_network(*args, **kwargs):
        raise AssertionError("offline inspector attempted network access")
    monkeypatch.setattr(builtins, "__import__", import_without_device)
    monkeypatch.setattr(socket, "socket", no_network)
    spec = importlib.util.spec_from_file_location("pv_inspector_no_device", inspect.__file__)
    reloaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reloaded)
    commands = []
    snapshot = {path: path.read_bytes() for path in (evidence.native, evidence.manifest,
                                                   evidence.baseline, evidence.profile, evidence.log)}
    result = reloaded.collect(options(evidence), runner=fake_runner(commands))
    assert commands and result["runtime_object_identity_confirmed"] is False
    assert all(path.read_bytes() == original for path, original in snapshot.items())
    assert all(Path(command[0]) == evidence.tool for command in commands)


def test_operator_manifest_link_resolves_nested_installed_binary(evidence):
    record = json.loads(evidence.manifest.read_text(encoding="utf-8"))
    operator = write_json(evidence.directory / "nested" / "operator_build.json", record)
    write_json(evidence.manifest, {"status": "passed", "operator_manifest": str(operator)})
    commands = []
    result = inspect.collect(options(evidence), runner=fake_runner(commands))
    obj = result["providers"]["probe"]["objects"][0]
    assert Path(obj["path"]) == evidence.native.resolve()
    assert obj["sha256_matches"] is True
    assert commands


def test_pc_log_offset_is_preserved_as_unconfirmed_candidate():
    pcs = inspect.parse_pc_records(ERROR_LOG, "worker.log")
    assert len(pcs) == 1
    assert pcs[0]["start"] == 0x12C08E614000
    assert pcs[0]["current"] == 0x12C08E6190B4
    assert pcs[0]["direct_delta"] == 0x50B4
    assert pcs[0]["reported_core_id"] == 20
    symbols = inspect.parse_symbols(SYMBOLS)
    assert {s["name"].rsplit("_", 1)[-1] for s in symbols} == {"aic", "aiv"}
    instructions = inspect.parse_instructions(DISASSEMBLY)
    assert any(row["address"] == 0x50B4 for row in instructions)
    candidates = inspect.candidate_windows(pcs, symbols, instructions)
    assert candidates["windows"]
    assert "fixture_instruction" in json.dumps(candidates["windows"])


@pytest.mark.parametrize("offset", [-4, 0x50B5, 0x90000])
def test_invalid_pc_offsets_cannot_become_instruction_windows(offset):
    start = 0x12C08E614000
    text = f"core id is 20, pc start: {start:#x}, current: {start + offset:#x}"
    pcs = inspect.parse_pc_records(text, "invalid.log")
    candidates = inspect.candidate_windows(pcs, inspect.parse_symbols(SYMBOLS),
                                           inspect.parse_instructions(DISASSEMBLY))
    assert not candidates["windows"]
    assert candidates["rejected"]


def test_multiple_mix_symbols_are_not_a_confirmed_loaded_object(evidence):
    commands = []
    result = inspect.collect(options(evidence), runner=fake_runner(commands))
    obj = result["providers"]["probe"]["objects"][0]
    assert len(obj["symbols"]) == 2
    assert obj["pc_candidates"]["windows"]
    assert result["runtime_object_identity_confirmed"] is False
    for command in obj["commands"]:
        assert Path(command["stdout_path"]).is_file()
        assert Path(command["stderr_path"]).read_text(encoding="utf-8") == "fixture stderr retained\n"


def test_nested_manifest_table_is_read_without_recursive_filesystem_scan(evidence):
    record = json.loads(evidence.manifest.read_text(encoding="utf-8"))
    write_json(evidence.manifest, {"provenance": {"provider": {"build_manifest": record}}})
    # An unrecorded sibling must not be inspected just because it is a device binary.
    sibling = evidence.native.with_name("Unrecorded.bin")
    sibling.write_bytes(b"unrecorded device object")
    commands = []
    result = inspect.collect(options(evidence), runner=fake_runner(commands))
    objects = result["providers"]["probe"]["objects"]
    assert len(objects) == 1 and Path(objects[0]["path"]) == evidence.native
    assert all(str(sibling) not in command for command in commands)


@pytest.mark.parametrize("expected", [None, "not-a-sha256"])
def test_missing_or_invalid_recorded_sha_cannot_authorize_disassembly(evidence, expected):
    record = json.loads(evidence.manifest.read_text(encoding="utf-8"))
    record["device_binaries"][0]["sha256"] = expected
    write_json(evidence.manifest, record)
    write_json(evidence.baseline, record)
    commands = []
    result = inspect.collect(options(evidence), runner=fake_runner(commands))
    assert commands == [] and result["status"] == "partial"
    assert result["providers"]["probe"]["objects"][0]["actual_sha256"] == fingerprint(evidence.native)["sha256"]


def test_tool_failure_keeps_partial_stdout_and_stderr(evidence):
    def fail(command, **kwargs):
        kwargs["stdout"].write(b"partial object information\n")
        kwargs["stderr"].write(b"target architecture unsupported\n")
        return SimpleNamespace(returncode=1)
    result = inspect.collect(options(evidence), runner=fail)
    assert result["status"] == "partial"
    for provider in result["providers"].values():
        obj = provider["objects"][0]
        assert obj["symbols"] == [] and obj["pc_candidates"]["windows"] == []
        for command in obj["commands"]:
            assert command["returncode"] == 1
            assert Path(command["stdout_path"]).read_bytes() == b"partial object information\n"
            assert Path(command["stderr_path"]).read_bytes() == b"target architecture unsupported\n"


def test_object_change_between_commands_stops_further_disassembly(evidence):
    commands = []
    def change_after_symbols(command, **kwargs):
        result = fake_runner(commands)(command, **kwargs)
        if command[1] == "--syms":
            evidence.native.write_bytes(b"concurrent rebuild changed object")
        return result
    result = inspect.collect(options(evidence), runner=change_after_symbols)
    assert sum(command[1] == "--syms" for command in commands) == 1
    assert not any(command[1] == "--disassemble" for command in commands)
    assert result["status"] == "partial"
    assert "changed after SHA verification" in json.dumps(result["errors"])


def test_unknown_resource_metadata_is_not_reported_as_zero_spills(evidence):
    write_json(evidence.metadata, {"kernelName": "fixture", "unknownCompilerField": None})
    result = inspect.collect(options(evidence), runner=fake_runner([]))
    metadata = result["providers"]["probe"]["objects"][0]["metadata"]
    assert metadata["resource_fields"] == []
    assert metadata["resource_status"].startswith("unknown")
    assert metadata["as_exported"]["unknownCompilerField"] is None


def test_discovery_does_not_use_arbitrary_path_tool(evidence, monkeypatch):
    monkeypatch.setenv("PATH", str(evidence.tool.parent))
    commands = []
    result = inspect.collect(options(evidence, tool=False), runner=fake_runner(commands))
    assert commands == [] and result["status"] == "partial"
    assert result["providers"]["probe"]["tool"]["path"] is None


def test_cli_writes_actionable_partial_report_when_tool_is_missing(evidence, monkeypatch):
    argv = ["--profile", str(evidence.profile), "--probe-build", str(evidence.manifest),
            "--baseline-build", str(evidence.baseline),
            "--output", str(evidence.directory / "cli_result.json"),
            "--objdump", str(evidence.directory / "not-installed")]
    def no_process(*args, **kwargs):
        raise AssertionError("Missing offline tool must not trigger installation or a build")
    monkeypatch.setattr(inspect.subprocess, "run", no_process)
    assert inspect.main(argv) == 0
    report = json.loads((evidence.directory / "cli_result.json").read_text(encoding="utf-8"))
    assert report["status"] == "partial"
    assert report["profile"]["failure_contexts"][0]["as_exported"] == evidence.failure
    assert Path(report["artifact_dir"]).is_dir()


@pytest.mark.parametrize("size,wrong_symbol", [(0, False), (0x50B4, False), (0x6000, True)])
def test_pc_requires_sized_matching_symbol_and_exact_instruction(size, wrong_symbol):
    symbols = inspect.parse_symbols(SYMBOLS)
    symbols[0]["size"], symbols[0]["end"] = size, size
    instructions = inspect.parse_instructions(DISASSEMBLY)
    if wrong_symbol:
        for instruction in instructions:
            if instruction["address"] == 0x50B4:
                instruction["symbol"] = "different_kernel_mix_aic"
    result = inspect.candidate_windows(inspect.parse_pc_records(ERROR_LOG), symbols, instructions)
    assert result["windows"] == []
    assert result["rejected"]


def test_valid_metadata_object_sha_mismatch_blocks_objdump(evidence):
    write_json(evidence.metadata, {"sha256": "0" * 64, "kernelName": "fixture"})
    commands = []
    report = inspect.collect(options(evidence), runner=fake_runner(commands))
    assert commands == []
    assert report["status"] == "partial"
    assert report["errors"]


def test_unknown_instruction_never_produces_pc_hit():
    text = DISASSEMBLY.replace("fixture_instruction", "<unknown>")
    instructions = inspect.parse_instructions(text)
    assert not any(row["address"] == 0x50B4 for row in instructions)
    result = inspect.candidate_windows(inspect.parse_pc_records(ERROR_LOG),
                                      inspect.parse_symbols(SYMBOLS), instructions)
    assert result["windows"] == []
    assert result["rejected"]


def test_identical_vmas_in_distinct_sections_remain_separate_candidates():
    symbols = inspect.parse_symbols("""SYMBOL TABLE:
00000000 g F .text.aic 00006000 kernel_mix_aic
00000000 g F .text.aiv 00006000 kernel_mix_aiv
""")
    instructions = inspect.parse_instructions("""Disassembly of section .text.aic:
00000000 <kernel_mix_aic>:
  50b0: 00 00 00 00 aic_previous
  50b4: 00 00 00 00 aic_instruction
Disassembly of section .text.aiv:
00000000 <kernel_mix_aiv>:
  50b0: 00 00 00 00 aiv_previous
  50b4: 00 00 00 00 aiv_instruction
""")
    result = inspect.candidate_windows(inspect.parse_pc_records(ERROR_LOG), symbols, instructions)
    assert len(result["windows"]) == 2
    for window in result["windows"]:
        assert window["confirmed_fault_pc"] is False
        assert window["runtime_object_identity_confirmed"] is False
        assert all(row["section"] == window["section"] for row in window["instructions"])
        expected = "aic_instruction" if window["core_type"] == "mix_aic" else "aiv_instruction"
        assert window["instruction"]["assembly"] == expected


def test_matching_metadata_sha_allows_version_and_object_inspection(evidence):
    write_json(evidence.metadata, {"sha256": fingerprint(evidence.native)["sha256"]})
    commands = []
    result = inspect.collect(options(evidence), runner=fake_runner(commands))
    assert result["status"] == "collected"
    assert any(command[1] == "--version" for command in commands)
    assert any(command[1] == "--syms" for command in commands)
    assert any(command[1] == "--disassemble" for command in commands)


def test_repeated_pc_from_error_context_and_log_retains_sources_without_duplicate_windows(evidence):
    profile = json.loads(evidence.profile.read_text(encoding="utf-8"))
    profile["workers"]["paired"]["probe_validation_failure"]["traceback"] = ERROR_LOG
    write_json(evidence.profile, profile)
    result = inspect.collect(options(evidence), runner=fake_runner([]))
    pcs = result["profile"]["pc_records"]
    assert len(pcs) == 1
    assert pcs[0]["reported_core_id"] == 20
    assert len(pcs[0]["sources"]) == 2


@pytest.mark.parametrize("error", [FileNotFoundError("tool vanished"),
                                   subprocess.TimeoutExpired("llvm-objdump", 30)])
def test_tool_launch_or_timeout_preserves_partial_evidence(evidence, error):
    def fail(command, **kwargs):
        kwargs["stdout"].write(b"partial before interruption\n")
        kwargs["stderr"].write(b"interrupted diagnostic\n")
        raise error
    report = inspect.collect(options(evidence), runner=fail)
    assert report["status"] == "partial"
    assert report["profile"]["failure_contexts"]
    for provider in report["providers"].values():
        obj = provider["objects"][0]
        for command in obj["commands"]:
            assert command["returncode"] is None
            assert command["error"]
            assert Path(command["stdout_path"]).read_bytes() == b"partial before interruption\n"


def test_unrecognized_disassembly_marks_object_partial(evidence):
    def no_symbols(command, **kwargs):
        kwargs["stdout"].write(b"unrecognized tool output without symbol table\n")
        return SimpleNamespace(returncode=0)
    report = inspect.collect(options(evidence), runner=no_symbols)
    assert report["status"] == "partial"
    assert report["providers"]["probe"]["objects"][0]["inspection_status"] == "partial"


def test_unexpected_inspection_error_does_not_erase_already_collected_failure(evidence, monkeypatch):
    def unexpected(command, **kwargs):
        raise RuntimeError("unexpected inspector failure after reading original runtime evidence")
    monkeypatch.setattr(inspect.subprocess, "run", unexpected)
    args = options(evidence)
    assert inspect.main(["--profile", str(args.profile), "--probe-build", str(args.probe_build),
                         "--baseline-build", str(args.baseline_build), "--output", str(args.output),
                         "--objdump", str(args.objdump)]) == 0
    report = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert report["status"] == "partial"
    assert report["profile"]["failure_contexts"][0]["as_exported"] == evidence.failure
    assert report["profile"]["pc_records"][0]["direct_delta"] == 0x50B4
    assert "unexpected inspector failure" in json.dumps(report["errors"])
    assert report["providers"]["probe"]["objects"][0]["sha256_matches"] is True


def test_nested_source_digest_is_not_mistaken_for_whole_object_sha(evidence):
    write_json(evidence.metadata, {"sha256": fingerprint(evidence.native)["sha256"],
                                   "source": {"sha256": "0" * 64}})
    commands = []
    report = inspect.collect(options(evidence), runner=fake_runner(commands))
    assert report["status"] == "collected" and commands
    metadata = report["providers"]["probe"]["objects"][0]["metadata"]
    assert metadata["declared_sha256_matches"] is True
    assert len(metadata["declared_object_sha256"]) == 1
    assert metadata["as_exported"]["source"]["sha256"] == "0" * 64


def unavailable_runner(commands, *, disassembly=HIIPU_UNAVAILABLE):
    def run(command, **kwargs):
        commands.append(command)
        outputs = {"--version": TOOL_VERSION, "--syms": SYMBOLS, "--disassemble": disassembly}
        kwargs["stdout"].write(outputs[command[1]].encode())
        # The real tool returned success with empty stderr and no instructions.
        return SimpleNamespace(returncode=0)
    return run


def test_real_hiipu_not_available_rows_never_become_machine_instructions(evidence):
    assert inspect.parse_instructions(HIIPU_UNAVAILABLE) == []
    commands = []
    report = inspect.collect(options(evidence), runner=unavailable_runner(commands))
    assert report["status"] == "partial"
    for provider in report["providers"].values():
        obj = provider["objects"][0]
        assert obj["sha256_matches"] is True
        assert len(obj["symbols"]) == 2
        assert obj["instruction_count"] == 0
        assert obj["inspection_status"] == "disassembly_unavailable"
        assert obj["pc_candidates"]["windows"] == []
        diag = obj["disassembly_diagnostics"]
        assert diag["status"] == "unavailable"
        assert diag["address_rows"] == diag["not_available_rows"] == 5
        assert diag["unknown_rows"] == diag["parsed_rows"] == 0
        assert all(command["returncode"] == 0 for command in obj["commands"])
        preview = obj["unparsed_disassembly_preview"][0]
        assert preview["stdout_path"]["lines"][0].endswith("file format elf64-hiipu")
        assert preview["stderr_path"]["bytes"] == 0
        assert preview["stderr_path"]["lines"] == []


def test_partly_unavailable_disassembly_is_partial_without_fake_pc_hit(evidence):
    mixed = HIIPU_UNAVAILABLE.replace("       4: <not available>",
                                     "       4: 00 00 00 00 fixture_available")
    report = inspect.collect(options(evidence), runner=unavailable_runner([], disassembly=mixed))
    assert report["status"] == "partial"
    obj = report["providers"]["probe"]["objects"][0]
    assert obj["inspection_status"] == "partial"
    assert obj["instruction_count"] == 1
    assert obj["pc_candidates"]["windows"] == []
    assert obj["disassembly_diagnostics"]["status"] == "partial"
    assert obj["disassembly_diagnostics"]["not_available_rows"] == 4
    assert obj["disassembly_diagnostics"]["parsed_rows"] == 1


def test_unavailable_cli_reports_limitation_and_bounded_raw_preview(evidence, monkeypatch, capsys):
    long_raw = ("Kernel_hash.bin: file format elf64-hiipu\nDisassembly of section .text:\n"
                "00000000 <vc_laser_cube_pv_probe_attention_mix_aic>:\n" +
                "".join(f"{i * 4:8x}: <not available>\n" for i in range(100)) +
                "TAIL_SHOULD_NOT_PRINT\n")
    monkeypatch.setattr(inspect.subprocess, "run", unavailable_runner([], disassembly=long_raw))
    args = options(evidence)
    assert inspect.main(["--profile", str(args.profile), "--probe-build", str(args.probe_build),
                         "--baseline-build", str(args.baseline_build), "--output", str(args.output),
                         "--objdump", str(args.objdump)]) == 0
    output = capsys.readouterr().out
    assert "unavailable" in output.lower()
    assert "RAW disassembler output" in output
    assert "elf64-hiipu" in output
    assert "preview truncated" in output
    assert "TAIL_SHOULD_NOT_PRINT" not in output
    report = json.loads(Path(args.output).read_text(encoding="utf-8"))
    preview = report["providers"]["probe"]["objects"][0]["unparsed_disassembly_preview"][0]
    assert preview["stdout_path"]["truncated"] is True
    assert len(preview["stdout_path"]["lines"]) == 45
    assert Path(preview["stdout_path"]["path"]).read_text(encoding="utf-8") == long_raw


def preview_command(tmp_path, stdout, stderr=b""):
    out, err = tmp_path / "raw.stdout.txt", tmp_path / "raw.stderr.txt"
    out.write_bytes(stdout)
    err.write_bytes(stderr)
    return {"command": ["llvm-objdump", "--disassemble", "kernel.bin"], "returncode": 0,
            "stdout_path": str(out), "stderr_path": str(err)}


def test_raw_preview_reports_exact_bytes_paths_and_bounds_lines(tmp_path):
    raw = "".join(f"original row {i}\n" for i in range(70)).encode()
    command = preview_command(tmp_path, raw, "工具stderr原样保留\n".encode())
    preview = inspect.raw_output_preview(command)
    stdout, stderr = preview["stdout_path"], preview["stderr_path"]
    assert stdout["path"] == command["stdout_path"]
    assert stdout["bytes"] == len(raw)
    assert stdout["lines"] == [f"original row {i}" for i in range(45)]
    assert stdout["truncated"] is True
    assert stderr["path"] == command["stderr_path"]
    assert stderr["bytes"] == len("工具stderr原样保留\n".encode())
    assert stderr["lines"] == ["工具stderr原样保留"]
    assert Path(command["stdout_path"]).read_bytes() == raw


@pytest.mark.parametrize("chars", [299, 300, 301, 4096])
def test_raw_preview_bounds_line_characters_even_with_terminating_newline(tmp_path, chars):
    command = preview_command(tmp_path, ("x" * chars + "\nNEXT\n").encode())
    preview = inspect.raw_output_preview(command)["stdout_path"]
    if chars <= 300:
        assert preview["lines"] == ["x" * chars, "NEXT"]
        assert not preview.get("truncated", False)
    else:
        assert preview["truncated"] is True
        assert preview["lines"][0] == "x" * 300 + " ... [line truncated]"
        assert len(preview["lines"]) == 1


def test_raw_preview_missing_file_keeps_path_and_error(tmp_path):
    command = preview_command(tmp_path, b"retained\n")
    Path(command["stderr_path"]).unlink()
    preview = inspect.raw_output_preview(command)
    assert preview["stdout_path"]["lines"] == ["retained"]
    assert preview["stderr_path"]["path"] == command["stderr_path"]
    assert preview["stderr_path"]["lines"] == []
    assert "FileNotFoundError" in preview["stderr_path"]["error"]


def test_discovery_finds_recorded_cann_tools_bisheng_compiler(evidence):
    manifest = json.loads(evidence.manifest.read_text(encoding="utf-8"))
    root = Path(manifest["cann_root"])
    tool = root / "tools" / "bisheng_compiler" / "bin" / "llvm-objdump.exe"
    tool.parent.mkdir(parents=True)
    tool.write_bytes(b"fixture tool, runner intercepted")
    commands = []
    report = inspect.collect(options(evidence, tool=False), runner=fake_runner(commands))
    assert report["status"] == "collected"
    assert report["providers"]["probe"]["tool"]["path"] == str(tool.resolve())
    assert commands and all(Path(command[0]) == tool for command in commands)
