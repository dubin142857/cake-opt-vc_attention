#!/usr/bin/env python3
"""Prepare an isolated, pinned MindIE Laser custom operator with provenance.

The default baseline only renames device identifiers. Explicit experimental
variants add an audited probability transform or diagnostic probe in separate
checkouts. Original Laser sources and copyright notices remain intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

try:
    from scripts.prepare_cann_fas_source import (
        PreparationError, RepositoryLock, audit_files, check_dirty,
        ensure_checkout, git, replace_once, sha256,
    )
except ModuleNotFoundError:
    from prepare_cann_fas_source import (
        PreparationError, RepositoryLock, audit_files, check_dirty,
        ensure_checkout, git, replace_once, sha256,
    )


ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = RepositoryLock("https://github.com/Ascend/MindIE-SD.git",
                             "71098c20ff90838d1fd1ca0a40515c751a3869b0")
MARKER_TOKEN = "vc_mindie_laser_v1"
NATIVE_API = "aclnnVcLaserAttention"
OP_TYPE = "VcLaserAttention"
OP_ROOT = "csrc/ops/laser_attention"
CUSTOM_OP_ROOT = "csrc/ops/vc_laser_attention"
TILING_FILE = f"{OP_ROOT}/op_host/laser_attention_tiling.cpp"
PREPARE_SCRIPT = "csrc/ops/cmake/scripts/prepare.sh"
MARKER_FILE = f"{CUSTOM_OP_ROOT}/op_host/vc_source_marker.h"
# Released helper bytes from Git (LF) before the two CCE scalar argument casts.
LEGACY_PROBABILITY_HEADER_SHA256 = "90bef085a1d516568e0916306234ddfad46bac3bdc96261e8950c30792f7655f"
VARIANTS = ("baseline", "expcast", "baseline_probe", "expcast_probe")
RENAMES = ((b"LaserAttention", b"VcLaserAttention"),
           (b"laser_attention", b"vc_laser_attention"))

MARKER_HEADER = r'''// Generated host-only provenance; no device algorithm changes.
#ifndef VC_MINDIE_LASER_SOURCE_MARKER_H
#define VC_MINDIE_LASER_SOURCE_MARKER_H
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

static inline void VcMindieLaserSourceMarker(const char *phase, unsigned long long tilingKey)
{
    const char *path = std::getenv("VC_MINDIE_LASER_MARKER_FILE");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    std::FILE *file = std::fopen(path, "a");
    if (file == nullptr) {
        std::fprintf(stderr, "vc_mindie_laser_v1: cannot open marker file (errno=%d)\n", errno);
        std::abort();
    }
    const int written = std::fprintf(file,
        "{\"token\":\"vc_mindie_laser_v1\",\"phase\":\"%s\",\"pid\":%ld,\"tiling_key\":%llu}\n",
        phase, static_cast<long>(getpid()), tilingKey);
    const int flushed = std::fflush(file);
    const int closed = std::fclose(file);
    if (written < 0 || flushed != 0 || closed != 0) {
        std::fprintf(stderr, "vc_mindie_laser_v1: cannot write marker file (errno=%d)\n", errno);
        std::abort();
    }
}
#endif
'''.encode("utf-8")


def rename(data: bytes) -> bytes:
    for original, custom in RENAMES:
        data = data.replace(original, custom)
    return data


def inverse_rename(data: bytes) -> bytes:
    for original, custom in reversed(RENAMES):
        data = data.replace(custom, original)
    return data


def custom_path(name: str) -> str:
    return rename(name.encode()).decode()


def parallelism_patch(original: bytes) -> bytes:
    return replace_once(original,
        b'CPU_NUM=$(($(cat /proc/cpuinfo | grep "^processor" | wc -l)*2))\nJOB_NUM="-j${CPU_NUM}"',
        b'CPU_NUM=${BUILD_JOB_NUM:-1}\n'
        b'if [[ ! "${CPU_NUM}" =~ ^[1-9][0-9]*$ ]]; then\n'
        b'    echo "BUILD_JOB_NUM must be a positive integer" >&2\n'
        b'    exit 1\nfi\nJOB_NUM="-j${CPU_NUM}"', "prepare parallelism")


def marker_patch(original: bytes) -> bytes:
    data = replace_once(original, b'#include "vc_laser_attention_tiling.h"',
                        b'#include "vc_laser_attention_tiling.h"\n#include "vc_source_marker.h"',
                        "Laser tiling marker include")
    return replace_once(data, b"    return tiling.CheckTiling(context);",
        b"    const auto result = tiling.CheckTiling(context);\n"
        b"    if (result == ge::GRAPH_SUCCESS) {\n"
        b'        VcMindieLaserSourceMarker("tiling", static_cast<unsigned long long>(context->GetTilingKey()));\n'
        b"    }\n    return result;", "Laser successful tiling marker")


def seed_checkout(source: Path, reference: Path | None, lock: RepositoryLock) -> None:
    """Copy Git objects, never the reference worktree or its modifications."""
    if source.exists() or reference is None or not reference.is_dir():
        return
    if git(reference, "rev-parse", f"{lock.commit}^{{commit}}").decode().strip() != lock.commit:
        raise PreparationError("Reference does not contain the pinned MindIE commit")
    source.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", "--no-checkout",
                             str(reference), str(source)], capture_output=True)
    if result.returncode:
        raise PreparationError(f"Local reference clone failed: {result.stderr.decode(errors='replace')}")
    git(source, "config", "core.autocrlf", "false")
    git(source, "remote", "set-url", "origin", lock.url)
    git(source, "checkout", "--quiet", "--detach", lock.commit)


def default_manifest(source: Path) -> Path:
    return source.parent / f"{source.name}_manifest.json"


def variant_files(variant: str, generated: dict) -> dict:
    """Explicit experimental delta, independently rebuilt during verification."""
    if variant not in VARIANTS:
        raise PreparationError(f"Unknown Laser variant: {variant}")
    result = dict(generated)
    if variant.startswith("expcast"):
        try:
            from scripts.mindie_laser_expcast_patch import patch_vector
        except ModuleNotFoundError:
            from mindie_laser_expcast_patch import patch_vector
        vector = f"{CUSTOM_OP_ROOT}/op_kernel/VectorForward.h"
        if vector not in result:
            raise PreparationError("ExpCast requires the pinned VectorForward.h")
        result[vector] = patch_vector(result[vector])
        result[f"{CUSTOM_OP_ROOT}/op_kernel/vc_laser_probability.h"] = (
            ROOT / "ascend/laser_expcast/vc_laser_probability.h").read_bytes()
    if variant.endswith("_probe"):
        try:
            from scripts.mindie_laser_probe_patch import patch_vector, HEADER_NAME, HEADER
        except ModuleNotFoundError:
            from mindie_laser_probe_patch import patch_vector, HEADER_NAME, HEADER
        vector = f"{CUSTOM_OP_ROOT}/op_kernel/VectorForward.h"
        result[vector] = patch_vector(result[vector])
        result[f"{CUSTOM_OP_ROOT}/op_kernel/{HEADER_NAME}"] = HEADER
    return result


def audit_variant_files(source_dir: Path, patches: dict, previous: dict | None,
                        source_lock: RepositoryLock, variant: str) -> dict:
    """Accept enumerated historical Vector/helper bytes, never arbitrary hashes.

    The historical file is derived from the pinned/current transformation,
    including probe brackets when present. Audit the entire old tree before
    any write, then record the new bytes which prepare_source will install.
    """
    vector = f"{CUSTOM_OP_ROOT}/op_kernel/VectorForward.h"
    header = f"{CUSTOM_OP_ROOT}/op_kernel/vc_laser_probability.h"
    audit_patches = patches
    migrating = {}
    if variant.startswith("expcast"):
        for name in (vector, header):
            path = source_dir / name
            if path.is_file() and path.read_bytes() != patches[name]:
                migrating[name] = path.read_bytes()
    if migrating:
        audit_patches = dict(patches)
        if vector in migrating:
            try:
                from scripts.mindie_laser_expcast_patch import (
                    EXPERIMENTAL_PROBABILITY, LEGACY_EXPERIMENTAL_PROBABILITY,
                )
            except ModuleNotFoundError:
                from mindie_laser_expcast_patch import (
                    EXPERIMENTAL_PROBABILITY, LEGACY_EXPERIMENTAL_PROBABILITY,
                )
            expected = patches[vector]
            newline = b"\r\n" if b"\r\n" in expected else b"\n"
            audit_patches[vector] = replace_once(
                expected, EXPERIMENTAL_PROBABILITY.replace(b"\n", newline),
                LEGACY_EXPERIMENTAL_PROBABILITY.replace(b"\n", newline),
                "historical ExpCast constexpr revision")
        if header in migrating:
            legacy_header = patches[header]
            for intrinsic, scalar in (("vmuls", "GAIN"), ("vadds", "BIAS")):
                old = f"{intrinsic}(scratch, scratch, {scalar}, repeats32, 1, 1, 8, 8);".encode()
                new = old.replace(scalar.encode(), f"static_cast<float>({scalar})".encode(), 1)
                legacy_header = replace_once(legacy_header, new, old, "historical CCE scalar argument")
            legacy_lf = legacy_header.replace(b"\r\n", b"\n")
            if sha256(legacy_lf) != LEGACY_PROBABILITY_HEADER_SHA256:
                raise PreparationError("Historical probability helper reconstruction no longer matches its fixed hash")
            # The helper is copied from the workspace, where Git autocrlf can
            # give the same historical file CRLF. Audit its actual old bytes
            # against its manifest, accepting only uniform LF or uniform CRLF.
            if migrating[header] not in (legacy_lf, legacy_lf.replace(b"\n", b"\r\n")):
                raise PreparationError(f"Unknown generated source contents; preserving {source_dir / header}")
            audit_patches[header] = migrating[header]
        for name, contents in migrating.items():
            if contents != audit_patches[name]:
                raise PreparationError(f"Unknown generated source contents; preserving {source_dir / name}")
        identity = {
            "source_dir": str(source_dir), "variant": variant,
            "native_api": NATIVE_API, "op_type": OP_TYPE, "marker_token": MARKER_TOKEN,
            "upstream": {"url": source_lock.url, "commit": source_lock.commit, "tag": source_lock.tag},
        }
        if previous is None or any(previous.get(key) != value for key, value in identity.items()):
            raise PreparationError("Known historical generated sources require their matching source manifest")
    records, _ = audit_files(source_dir, audit_patches)
    if migrating:
        generated = set(patches) - {PREPARE_SCRIPT}
        if (previous.get("source_files") != records
                or set(previous.get("generated_files", {})) != generated
                or any(previous["generated_files"][name].get("prepared_sha256") !=
                       records[name]["prepared_sha256"] for name in generated)):
            raise PreparationError("Historical source manifest differs from reconstructed audited sources")
        for name in migrating:
            records[name]["prepared_sha256"] = sha256(patches[name])
    return records


def prepare_source(source_dir: Path, manifest_path: Path, offline: bool = False, *,
                   reference_dir: Path | None = None,
                   source_lock: RepositoryLock = SOURCE_LOCK, variant: str = "baseline") -> dict:
    source_dir, manifest_path = Path(source_dir).resolve(), Path(manifest_path).resolve()
    if reference_dir is not None and source_dir == Path(reference_dir).resolve():
        raise PreparationError("Source and reference must be separate checkouts")
    if variant not in VARIANTS:
        raise PreparationError(f"Unknown Laser variant: {variant}")
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("variant", "baseline") != variant:
            raise PreparationError("Use a separate checkout/manifest for each Laser variant; existing build preserved")
    seed_checkout(source_dir, reference_dir, source_lock)
    ensure_checkout(source_dir, source_lock, offline)
    names = git(source_dir, "ls-tree", "-r", "--name-only", "HEAD", "--", OP_ROOT).decode().splitlines()
    if not names or not any("/op_kernel/" in name for name in names):
        raise PreparationError("Pinned Laser subtree is absent")
    originals = {name: git(source_dir, "show", f"HEAD:{name}") for name in names}
    generated = {}
    generated_records = {}
    kernel_hash = hashlib.sha256()
    for name, original in sorted(originals.items()):
        custom = custom_path(name)
        prepared = rename(original)
        equivalent = inverse_rename(prepared) == original
        if not equivalent:
            raise PreparationError(f"Identifier rename is not reversible: {name}")
        if name == TILING_FILE:
            prepared = marker_patch(prepared)
        generated[custom] = prepared
        generated_records[custom] = {
            "original_path": name, "original_sha256": sha256(original),
            "prepared_sha256": sha256(prepared),
            "inverse_rename_equivalent": equivalent and name != TILING_FILE,
        }
        if "/op_kernel/" in name:
            kernel_hash.update(f"{name}\0{sha256(original)}\n".encode())
    generated[MARKER_FILE] = MARKER_HEADER
    generated_records[MARKER_FILE] = {
        "original_path": None, "original_sha256": None,
        "prepared_sha256": sha256(MARKER_HEADER), "inverse_rename_equivalent": False,
    }
    generated = variant_files(variant, generated)
    for name, data in generated.items():
        record = generated_records.setdefault(name, {
            "original_path": None, "original_sha256": None,
            "inverse_rename_equivalent": False,
        })
        if record.get("prepared_sha256") != sha256(data):
            record["inverse_rename_equivalent"] = False
        record["prepared_sha256"] = sha256(data)
    patched_prepare = parallelism_patch(git(source_dir, "show", f"HEAD:{PREPARE_SCRIPT}"))
    patches = {PREPARE_SCRIPT: patched_prepare, **generated}
    check_dirty(source_dir, set(patches))
    custom_root = source_dir / CUSTOM_OP_ROOT
    if custom_root.exists():
        for path in custom_root.rglob("*"):
            if path.is_symlink() or (path.is_file() and path.relative_to(source_dir).as_posix() not in generated):
                raise PreparationError(f"Unknown generated subtree entry: {path}")
    # Audit all original sources, including helpers/CMake; reject manual edits.
    source_files = audit_variant_files(source_dir, patches, previous, source_lock, variant)
    for name, data in patches.items():
        target = source_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != data:
            target.write_bytes(data)
    manifest = {
        "schema_version": 1,
        "upstream": {"url": source_lock.url, "commit": source_lock.commit, "tag": source_lock.tag},
        "source_dir": str(source_dir), "original_operator": OP_ROOT,
        "custom_operator": CUSTOM_OP_ROOT, "native_api": NATIVE_API, "op_type": OP_TYPE,
        "marker_token": MARKER_TOKEN, "marker_environment": "VC_MINDIE_LASER_MARKER_FILE",
        "variant": variant,
        "kernel_inverse_rename_equivalent": variant == "baseline",
        "diagnostic_instrumentation": variant.endswith("_probe"),
        "algorithm_scope": ("Original Laser" if variant.startswith("baseline") else
            "Only bulk probability ExpCast; FP32 separate multiply/add then RNE; "
            "E4M3 decoded /256 as FP16; original FP16 QK/PV and history update; no V smoothing"),
        "kernel_tree_sha256": kernel_hash.hexdigest(),
        "kernel_tree_hash_format": "sha256 of sorted original Laser path + NUL + original file sha256 + newline",
        "identifier_renames": {old.decode(): new.decode() for old, new in RENAMES},
        "build_parallelism_patch": PREPARE_SCRIPT,
        "modified_original_files": [PREPARE_SCRIPT],
        "generated_files": generated_records, "source_files": source_files,
        "build_dependencies": "Installed CANN Toolkit/ops, bisheng, op_build, AscendC CMake/Python tools, CMake, Make and host C++ compiler; no full MindIE wheel or TIK build.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if not manifest_path.exists() or manifest_path.read_bytes() != data:
        temporary = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(manifest_path)
    return manifest


def validate_prepared_source(source: Path, manifest_path: Path) -> dict:
    """Re-derive provenance from the locked Git objects without changing files."""
    source, manifest_path = Path(source).resolve(), Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("upstream", {}).get("commit") != SOURCE_LOCK.commit:
        raise PreparationError("Source manifest is not the pinned MindIE commit")
    if manifest.get("upstream", {}).get("url") != SOURCE_LOCK.url:
        raise PreparationError("Source manifest has an unexpected upstream")
    ensure_checkout(source, SOURCE_LOCK, offline=True)
    if manifest.get("source_dir") != str(source):
        raise PreparationError("Source manifest belongs to another checkout")
    variant = manifest.get("variant", "baseline")
    if variant not in VARIANTS:
        raise PreparationError("Unknown source variant")
    if (manifest.get("native_api") != NATIVE_API or manifest.get("op_type") != OP_TYPE
            or manifest.get("marker_token") != MARKER_TOKEN
            or manifest.get("kernel_inverse_rename_equivalent") is not (variant == "baseline")):
        raise PreparationError("Source manifest identity/inverse-rename evidence is incomplete")
    expected_names = git(source, "ls-tree", "-r", "--name-only", "HEAD", "--", OP_ROOT).decode().splitlines()
    expected_generated = {custom_path(name) for name in expected_names} | {MARKER_FILE}
    if variant.startswith("expcast"):
        expected_generated.add(f"{CUSTOM_OP_ROOT}/op_kernel/vc_laser_probability.h")
    if variant.endswith("_probe"):
        expected_generated.add(f"{CUSTOM_OP_ROOT}/op_kernel/vc_laser_probability_probe.h")
    if set(manifest.get("generated_files", {})) != expected_generated:
        raise PreparationError("Source manifest omits or adds generated operator files")
    patches = {PREPARE_SCRIPT: parallelism_patch(git(source, "show", f"HEAD:{PREPARE_SCRIPT}")),
               MARKER_FILE: MARKER_HEADER}
    for name in expected_names:
        data = rename(git(source, "show", f"HEAD:{name}"))
        patches[custom_path(name)] = marker_patch(data) if name == TILING_FILE else data
    patches = variant_files(variant, patches)
    check_dirty(source, set(patches))
    actual, _ = audit_files(source, patches)
    if actual != manifest.get("source_files"):
        raise PreparationError("Source audit differs from the manifest")
    for name, expected in patches.items():
        if (source / name).read_bytes() != expected:
            raise PreparationError(f"Prepared source missing or stale: {name}")
        if name in expected_generated and manifest["generated_files"][name].get("prepared_sha256") != sha256(expected):
            raise PreparationError(f"Generated source hash differs from manifest: {name}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "build/mindie_laser_source")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "build/mindie-sd-reference")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    args = parser.parse_args()
    manifest_path = args.manifest or default_manifest(args.source_dir)
    try:
        result = prepare_source(args.source_dir, manifest_path, args.offline,
                                reference_dir=args.reference_dir, variant=args.variant)
    except (PreparationError, OSError, ValueError) as error:
        print(f"MindIE Laser preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"Prepared {result['op_type']} from {result['upstream']['commit']}")
    print(f"Variant: {args.variant}; {result['algorithm_scope']}")
    print(f"Manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
