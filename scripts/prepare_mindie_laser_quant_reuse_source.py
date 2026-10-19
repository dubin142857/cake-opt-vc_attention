#!/usr/bin/env python3
"""Prepare an independently named QK-scale-reuse candidate from locked Laser bytes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_mindie_laser_quant_source as baseline
from scripts.mindie_laser_quant_reuse_patch import apply
from scripts.prepare_mindie_laser_source import SOURCE_LOCK, OP_ROOT, PREPARE_SCRIPT, seed_checkout, default_manifest
from scripts.prepare_cann_fas_source import PreparationError, ensure_checkout, check_dirty, audit_files, sha256

OP_NAME = "vc_laser_quant_reuse_attention"
OP_TYPE = "VcLaserQuantReuseAttention"
NATIVE_API = "aclnnVcLaserQuantReuseAttention"
CUSTOM_OP_ROOT = f"csrc/ops/{OP_NAME}"
MARKER_TOKEN = "vc_mindie_laser_quant_reuse_v1"
MARKER_ENV = "VC_MINDIE_LASER_QUANT_REUSE_MARKER_FILE"
IDENTITY = dict(op_name=OP_NAME, op_type=OP_TYPE, native_api=NATIVE_API, vendor="vc_laser_quant_reuse")
ALGORITHM = baseline.ALGORITHM
SCALE_CONTRACT = "FP32(sQ*sK), clear low 13 bits, INT32 dot -> DEQF16; Vector half attention scale"
REUSE_CONTRACT = {
    "scope": "Q scales per descriptor; K scale per K128 tile; no reuse across native invocations",
    "descriptor_ky": 2,
    "maximum_descriptor_kx": 32,
    "full_descriptor_old_scalar_reads": 128,
    "full_descriptor_new_scalar_reads": 34,
    "cache_invalidation": "ReadQuantScale DCCI retained before each remaining GM scalar read",
    "unchanged": "QK payloads, FP32 product/truncation, FIX barrier, DEQF16, Vector, PV and cross-core schedule",
    "performance": "Read counts are source-level estimates; no device speedup claim",
}


def rename(data: bytes) -> bytes:
    """Rename a prepared quant baseline, not upstream identifiers a second time."""
    for old, new in ((baseline.OP_TYPE, OP_TYPE), (baseline.OP_NAME, OP_NAME),
                     (baseline.MARKER_TOKEN, MARKER_TOKEN), (baseline.MARKER_ENV, MARKER_ENV)):
        data = data.replace(old.encode(), new.encode())
    return data


def expected_files(source):
    base_files, base_records = baseline.expected_files(source)
    patches, records = {}, {}
    for name, data in base_files.items():
        target = rename(name.encode()).decode()
        transformed = apply(target, rename(data))
        patches[target] = transformed
        if name in base_records:
            records[target] = {
                **base_records[name], "prepared_sha256": sha256(transformed),
                "baseline_prepared_sha256": sha256(data),
            }
    return patches, records


def _audit(source, patches):
    check_dirty(source, set(patches))
    for path in (source / CUSTOM_OP_ROOT).rglob("*"):
        if path.is_symlink() or (path.is_file() and path.relative_to(source).as_posix() not in patches):
            raise PreparationError(f"Unexpected generated source: {path}")
    return audit_files(source, patches)[0]


def _identity(source):
    return {
        "source_dir": str(source), "variant": "quant_scale_reuse", "op_type": OP_TYPE,
        "native_api": NATIVE_API, "custom_operator": CUSTOM_OP_ROOT,
        "marker_token": MARKER_TOKEN, "marker_environment": MARKER_ENV,
        "upstream": {"url": SOURCE_LOCK.url, "commit": SOURCE_LOCK.commit, "tag": SOURCE_LOCK.tag},
        "qk_scale_contract": SCALE_CONTRACT, "scale_reuse_contract": REUSE_CONTRACT,
    }


def prepare(source_dir, reference_dir, manifest, *, offline=False):
    source, reference, destination = map(lambda x: Path(x).resolve(), (source_dir, reference_dir, manifest))
    if source == reference or source.is_relative_to(reference) or reference.is_relative_to(source):
        raise PreparationError("Scale-reuse source must be separate from the reference checkout")
    seed_checkout(source, reference, SOURCE_LOCK)
    ensure_checkout(source, SOURCE_LOCK, offline=offline)
    patches, generated = expected_files(source)
    records = _audit(source, patches)
    report = {
        "schema_version": 1, **_identity(source), "original_operator": OP_ROOT,
        "algorithm_scope": ALGORITHM, "kernel_inverse_rename_equivalent": False,
        "diagnostic_instrumentation": False, "source_files": records, "generated_files": generated,
        "modified_original_files": [PREPARE_SCRIPT],
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
    for key, value in _identity(source).items():
        if manifest.get(key) != value:
            raise PreparationError(f"Wrong scale-reuse source identity: {key}")
    patches, generated = expected_files(source)
    if _audit(source, patches) != manifest.get("source_files") or generated != manifest.get("generated_files"):
        raise PreparationError("Scale-reuse source audit differs from manifest")
    for name, data in patches.items():
        if (source / name).read_bytes() != data:
            raise PreparationError(f"Missing/stale scale-reuse source: {name}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "build/mindie_laser_quant_reuse_source")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "build/mindie-sd-reference")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    prepare(args.source_dir, args.reference_dir, args.manifest or default_manifest(args.source_dir), offline=args.offline)
    print(f"Prepared {OP_TYPE}: QK scale reads reused within each descriptor; arithmetic unchanged")
