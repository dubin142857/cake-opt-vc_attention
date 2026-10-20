#!/usr/bin/env python3
"""Prepare the local-only Q-resident Laser candidate with Vector block scaling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_mindie_laser_quant_qresident_source as baseline
from scripts.mindie_laser_quant_vscale_patch import apply, VSCALE_CONTRACT
from scripts.prepare_mindie_laser_quant_reuse_source import REUSE_CONTRACT
from scripts.prepare_mindie_laser_source import SOURCE_LOCK, OP_ROOT, PREPARE_SCRIPT, default_manifest
from scripts.mindie_laser_offline_cache import local_objects_only, seed_local_checkout
from scripts.prepare_cann_fas_source import PreparationError, ensure_checkout, check_dirty, audit_files, sha256

OP_NAME = "vc_laser_quant_vscale_attention"
OP_TYPE = "VcLaserQuantVscaleAttention"
NATIVE_API = "aclnnVcLaserQuantVscaleAttention"
CUSTOM_OP_ROOT = f"csrc/ops/{OP_NAME}"
MARKER_TOKEN = "vc_mindie_laser_quant_vscale_v1"
MARKER_ENV = "VC_MINDIE_LASER_QUANT_VSCALE_MARKER_FILE"
IDENTITY = dict(op_name=OP_NAME, op_type=OP_TYPE, native_api=NATIVE_API, vendor="vc_laser_quant_vscale")
ALGORITHM = (
    "Q128/K128 symmetric INT8 Cube matmul with resident Q, INT32 accumulation, "
    "constant 1/64 FIX DEQF16 transport, Vector per-block Q/K scaling fused with "
    "attention scaling; original exp only, caller-provided FP16 V and native FP16 PV; "
    "no ExpCast support claim, no V quantization claim, no native FP8 matmul, dense MHA; "
    "new rounding path, no bitwise equivalence claim"
)
SCALE_CONTRACT = (
    "Cube INT32 dot -> DEQF16 at constant 1/64; Vector per-Q128/K128 gain is "
    "FP32(sQ*sK) with low 13 bits cleared, coefficient is "
    "FP32(64*gain)*FP32(half(attention_scale)); half coefficient vmul with "
    "FP32 scratch fallback for unsafe coefficients; new rounding path"
)
CACHE_CONTRACT = baseline.CACHE_CONTRACT
QRESIDENT_CONTRACT = baseline.QRESIDENT_CONTRACT


def rename(data: bytes) -> bytes:
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
                "q_resident_prepared_sha256": sha256(data),
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
        "source_dir": str(source), "variant": "quant_vector_scale", "op_type": OP_TYPE,
        "native_api": NATIVE_API, "custom_operator": CUSTOM_OP_ROOT,
        "marker_token": MARKER_TOKEN, "marker_environment": MARKER_ENV,
        "upstream": {"url": SOURCE_LOCK.url, "commit": SOURCE_LOCK.commit, "tag": SOURCE_LOCK.tag},
        "qk_scale_contract": SCALE_CONTRACT,
        "baseline_qk_scale_contract": baseline.SCALE_CONTRACT,
        "baseline_scale_reuse_contract": REUSE_CONTRACT,
        "scale_cache_contract": CACHE_CONTRACT,
        "q_resident_contract": QRESIDENT_CONTRACT,
        "vector_scale_contract": VSCALE_CONTRACT,
        "inherited_contract_scope": (
            "The Q-resident, scale-cache and scale-reuse contracts record parent provenance; "
            "vector_scale_contract defines this candidate's scale ownership and arithmetic"
        ),
        "algorithm_scope": ALGORITHM,
        "kernel_inverse_rename_equivalent": False,
        "bitwise_baseline_equivalent": False,
    }


def cached_references(reference):
    result = [reference]
    for name in ("mindie_laser_quant_qresident_build.json", "mindie_laser_quant_cache_build.json",
                 "mindie_laser_quant_reuse_build.json",
                 "mindie_laser_quant_build.json", "mindie_laser_build.json"):
        path = ROOT / "results" / name
        if path.is_file():
            build = json.loads(path.read_text(encoding="utf-8"))
            if build.get("status") == "passed" and build.get("source_dir"):
                result.append(Path(build["source_dir"]))
    result.extend(ROOT / "build" / name for name in
                  ("mindie_laser_quant_qresident_source", "mindie_laser_quant_cache_source",
                   "mindie_laser_quant_reuse_source",
                   "mindie_laser_quant_source", "mindie_laser_source", "mindie-sd-reference"))
    return result


@local_objects_only()
def prepare(source_dir, reference_dir, manifest, *, offline=True):
    # Retain the old keyword for callers, but never fetch remotely.
    source, reference, destination = map(lambda x: Path(x).resolve(), (source_dir, reference_dir, manifest))
    if source == reference or source.is_relative_to(reference) or reference.is_relative_to(source):
        raise PreparationError("Vector-scale source must be separate from the reference checkout")
    seed_local_checkout(source, cached_references(reference), SOURCE_LOCK)
    ensure_checkout(source, SOURCE_LOCK, offline=True)
    patches, generated = expected_files(source)
    records = _audit(source, patches)
    report = {
        "schema_version": 1, **_identity(source), "original_operator": OP_ROOT,
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


@local_objects_only()
def validate_prepared_source(source, manifest_path):
    source = Path(source).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    ensure_checkout(source, SOURCE_LOCK, offline=True)
    for key, value in _identity(source).items():
        if manifest.get(key) != value:
            raise PreparationError(f"Wrong Vector-scale source identity: {key}")
    patches, generated = expected_files(source)
    if _audit(source, patches) != manifest.get("source_files") or generated != manifest.get("generated_files"):
        raise PreparationError("Vector-scale source audit differs from manifest")
    for name, data in patches.items():
        if (source / name).read_bytes() != data:
            raise PreparationError(f"Missing/stale Vector-scale source: {name}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "build/mindie_laser_quant_vscale_source")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "build/mindie-sd-reference")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--offline", action="store_true", help="Compatibility flag; source preparation is always local-only")
    args = parser.parse_args()
    prepare(args.source_dir, args.reference_dir, args.manifest or default_manifest(args.source_dir), offline=args.offline)
    print(f"Prepared {OP_TYPE}: Q residency retained, constant 1/64 QK transport and Vector block scaling; new rounding path")
