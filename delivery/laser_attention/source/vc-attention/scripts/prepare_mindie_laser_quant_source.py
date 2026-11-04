#!/usr/bin/env python3
"""Prepare the separate INT8-QK/FP16-PV Laser experiment from locked Git bytes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_mindie_laser_source import (
    SOURCE_LOCK, OP_ROOT, PREPARE_SCRIPT, MARKER_HEADER, seed_checkout,
    parallelism_patch, default_manifest,
)
from scripts.prepare_cann_fas_source import (
    PreparationError, git, ensure_checkout, check_dirty, audit_files, sha256,
)
from scripts.mindie_laser_quant_patch import apply, once

OP_NAME = "vc_laser_quant_attention"
OP_TYPE = "VcLaserQuantAttention"
NATIVE_API = "aclnnVcLaserQuantAttention"
CUSTOM_OP_ROOT = f"csrc/ops/{OP_NAME}"
MARKER_TOKEN = "vc_mindie_laser_quant_v1"
MARKER_ENV = "VC_MINDIE_LASER_QUANT_MARKER_FILE"
IDENTITY = dict(op_name=OP_NAME, op_type=OP_TYPE, native_api=NATIVE_API, vendor="vc_laser_quant")
ALGORITHM = ("Q128/K128 symmetric INT8 Cube matmul, INT32 accumulation, FIX scalar DEQF16; "
             "per-channel E4M3 V decoded to FP16 before native PV; optional bulk ExpCast; "
             "no V clustering/smoothing, no native FP8 matmul, dense attention")


def rename(data):
    return data.replace(b"LaserAttention", OP_TYPE.encode()).replace(b"laser_attention", OP_NAME.encode())


def expected_files(source):
    names = git(source, "ls-tree", "-r", "--name-only", "HEAD", "--", OP_ROOT).decode().splitlines()
    if not names:
        raise PreparationError("Pinned Laser operator subtree is absent")
    patches, records = {}, {}
    for name in names:
        original = git(source, "show", f"HEAD:{name}")
        target = rename(name.encode()).decode()
        data = apply(target, rename(original))
        if name.endswith("/op_host/laser_attention_tiling.cpp"):
            anchor = f'#include "{OP_NAME}_tiling.h"'.encode()
            data = once(data, anchor, anchor + b'\n#include "vc_source_marker.h"')
            data = once(data, b"    return tiling.CheckTiling(context);", b"""    const auto result = tiling.CheckTiling(context);
    if (result == ge::GRAPH_SUCCESS) {
        VcMindieLaserSourceMarker("tiling", static_cast<unsigned long long>(context->GetTilingKey()));
    }
    return result;""")
        patches[target] = data
        records[target] = {"original_path": name, "original_sha256": sha256(original),
                           "prepared_sha256": sha256(data)}
    marker = MARKER_HEADER.replace(b"vc_mindie_laser_v1", MARKER_TOKEN.encode()).replace(
        b"VC_MINDIE_LASER_MARKER_FILE", MARKER_ENV.encode())
    additions = {
        f"{CUSTOM_OP_ROOT}/op_host/vc_source_marker.h": marker,
        f"{CUSTOM_OP_ROOT}/op_kernel/vc_laser_probability.h": (
            ROOT / "ascend/laser_expcast/vc_laser_probability.h").read_bytes(),
    }
    patches.update(additions)
    for name, data in additions.items():
        records[name] = dict(original_path=None, original_sha256=None, prepared_sha256=sha256(data))
    patches[PREPARE_SCRIPT] = parallelism_patch(git(source, "show", f"HEAD:{PREPARE_SCRIPT}"))
    return patches, records


def _audit(source, patches):
    check_dirty(source, set(patches))
    for path in (source / CUSTOM_OP_ROOT).rglob("*"):
        if path.is_symlink() or (path.is_file() and path.relative_to(source).as_posix() not in patches):
            raise PreparationError(f"Unexpected generated source: {path}")
    return audit_files(source, patches)[0]


def prepare(source_dir, reference_dir, manifest, *, offline=False):
    source, reference, destination = map(lambda x: Path(x).resolve(), (source_dir, reference_dir, manifest))
    if source == reference or source.is_relative_to(reference) or reference.is_relative_to(source):
        raise PreparationError("Quant source must be separate from the reference checkout")
    seed_checkout(source, reference, SOURCE_LOCK)
    ensure_checkout(source, SOURCE_LOCK, offline=offline)
    patches, generated = expected_files(source)
    # Audit everything before any write; never silently overwrite user changes.
    records = _audit(source, patches)
    report = {
        "schema_version": 1, "source_dir": str(source), "variant": "quant",
        "upstream": {"url": SOURCE_LOCK.url, "commit": SOURCE_LOCK.commit, "tag": SOURCE_LOCK.tag},
        "op_type": OP_TYPE, "native_api": NATIVE_API, "custom_operator": CUSTOM_OP_ROOT,
        "original_operator": OP_ROOT, "marker_token": MARKER_TOKEN, "marker_environment": MARKER_ENV,
        "algorithm_scope": ALGORITHM, "kernel_inverse_rename_equivalent": False,
        "diagnostic_instrumentation": False, "source_files": records, "generated_files": generated,
        "modified_original_files": [PREPARE_SCRIPT],
        "qk_scale_contract": "FP32(sQ*sK), clear low 13 bits, INT32 dot -> DEQF16; Vector half attention scale",
    }
    for name, data in patches.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if not destination.exists() or destination.read_bytes() != data:
        tmp = destination.with_name(destination.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(destination)
    return report


def validate_prepared_source(source, manifest_path):
    source = Path(source).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    ensure_checkout(source, SOURCE_LOCK, offline=True)
    for key, value in {"source_dir": str(source), "variant": "quant", "op_type": OP_TYPE,
                       "native_api": NATIVE_API, "marker_token": MARKER_TOKEN,
                       "marker_environment": MARKER_ENV}.items():
        if manifest.get(key) != value:
            raise PreparationError(f"Wrong quant source identity: {key}")
    if manifest.get("upstream") != {"url": SOURCE_LOCK.url, "commit": SOURCE_LOCK.commit, "tag": SOURCE_LOCK.tag}:
        raise PreparationError("Wrong quant upstream identity")
    patches, generated = expected_files(source)
    if _audit(source, patches) != manifest.get("source_files") or generated != manifest.get("generated_files"):
        raise PreparationError("Quant source audit differs from manifest")
    for name, data in patches.items():
        if (source / name).read_bytes() != data:
            raise PreparationError(f"Missing/stale quant source: {name}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "build/mindie_laser_quant_source")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "build/mindie-sd-reference")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    prepare(args.source_dir, args.reference_dir, args.manifest or default_manifest(args.source_dir), offline=args.offline)
    print(f"Prepared {OP_TYPE}: {ALGORITHM}")
