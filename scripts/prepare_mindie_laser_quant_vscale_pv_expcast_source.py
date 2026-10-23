#!/usr/bin/env python3
"""Prepare an isolated, local-only FP16 ExpCast experiment on stable PV prefetch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_mindie_laser_quant_vscale_pv_prefetch_source as baseline
from scripts.mindie_laser_fp16_expcast_math import CONTRACT
from scripts.prepare_mindie_laser_source import SOURCE_LOCK, OP_ROOT, PREPARE_SCRIPT, default_manifest
from scripts.mindie_laser_offline_cache import local_objects_only, seed_local_checkout
from scripts.prepare_cann_fas_source import PreparationError, ensure_checkout, check_dirty, audit_files, sha256

OP_NAME = "vc_laser_quant_vscale_pv_expcast_attention"
OP_TYPE = "VcLaserQuantVscalePvExpcastAttention"
NATIVE_API = "aclnnVcLaserQuantVscalePvExpcastAttention"
CUSTOM_OP_ROOT = f"csrc/ops/{OP_NAME}"
MARKER_TOKEN = "vc_mindie_laser_quant_vscale_pv_expcast_v1"
MARKER_ENV = "VC_MINDIE_LASER_QUANT_VSCALE_PV_EXPCAST_MARKER_FILE"
IDENTITY = dict(op_name=OP_NAME, op_type=OP_TYPE, native_api=NATIVE_API,
                vendor="vc_laser_quant_vscale_pv_expcast")
ALGORITHM = (
    "Stable PV-prefetch INT8 QK/FP16 PV with original-exp control and optional "
    "FP16 exponent-bit probability approximation; no E4M3 decode, native FP8, "
    "V quantization/reordering/smoothing or diagnostic instrumentation. Cube "
    "schedule, QK scaling, online-history exponentials and PV accumulation are inherited."
)
SCALE_CONTRACT = baseline.SCALE_CONTRACT
PROBABILITY_HEADER = ROOT / "ascend/laser_expcast_fp16/vc_laser_probability.h"


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
        transformed = rename(data)
        if target == f"{CUSTOM_OP_ROOT}/op_kernel/vc_laser_probability.h":
            transformed = PROBABILITY_HEADER.read_bytes()
        patches[target] = transformed
        if name in base_records:
            records[target] = {
                **base_records[name], "prepared_sha256": sha256(transformed),
                "pv_prefetch_prepared_sha256": sha256(data),
            }
    return patches, records


def _audit(source, patches):
    check_dirty(source, set(patches))
    for path in (source / CUSTOM_OP_ROOT).rglob("*"):
        if path.is_symlink() or (path.is_file() and path.relative_to(source).as_posix() not in patches):
            raise PreparationError(f"Unexpected generated source: {path}")
    return audit_files(source, patches)[0]


def _identity(source):
    identity = baseline._identity(source)
    # Parent metadata records source provenance. Only the optional probability
    # helper is replaced; the original-exp control retains parent arithmetic.
    return {
        **identity, "variant": "quant_vector_scale_pv_expcast", "op_type": OP_TYPE,
        "native_api": NATIVE_API, "custom_operator": CUSTOM_OP_ROOT,
        "marker_token": MARKER_TOKEN, "marker_environment": MARKER_ENV,
        "upstream": {"url": SOURCE_LOCK.url, "commit": SOURCE_LOCK.commit, "tag": SOURCE_LOCK.tag},
        "fp16_expcast_contract": CONTRACT,
        "probability_helper_sha256": sha256(PROBABILITY_HEADER.read_bytes()),
        "original_exp_control": True,
        "algorithm_scope": ALGORITHM,
        "inherited_contract_scope": (
            "The inherited contracts record stable parent provenance. The QK scale and "
            "Cube PV-prefetch contracts remain operative. The parent's original-softmax "
            "statements apply only when expcast=false; fp16_expcast_contract defines the "
            "enabled probability mode. Bitwise control equivalence requires device validation."
        ),
        "kernel_inverse_rename_equivalent": False,
        "bitwise_baseline_equivalent": False,
    }


def cached_references(reference):
    result = [reference]
    names = ("mindie_laser_quant_vscale_pv_prefetch", "mindie_laser_quant_vscale",
             "mindie_laser_quant_qresident", "mindie_laser_quant_cache",
             "mindie_laser_quant_reuse", "mindie_laser_quant", "mindie_laser")
    for name in names:
        path = ROOT / "results" / f"{name}_build.json"
        if path.is_file():
            build = json.loads(path.read_text(encoding="utf-8"))
            if build.get("status") == "passed" and build.get("source_dir"):
                result.append(Path(build["source_dir"]))
    result.extend(ROOT / "build" / f"{name}_source" for name in names)
    result.append(ROOT / "build/mindie-sd-reference")
    return result


@local_objects_only()
def prepare(source_dir, reference_dir, manifest, *, offline=True):
    # Compatibility keyword only: this preparer never fetches from the network.
    source, reference, destination = map(lambda x: Path(x).resolve(), (source_dir, reference_dir, manifest))
    if source == reference or source.is_relative_to(reference) or reference.is_relative_to(source):
        raise PreparationError("FP16 ExpCast source must be separate from the reference checkout")
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
            raise PreparationError(f"Wrong FP16 ExpCast source identity: {key}")
    patches, generated = expected_files(source)
    if _audit(source, patches) != manifest.get("source_files") or generated != manifest.get("generated_files"):
        raise PreparationError("FP16 ExpCast source audit differs from manifest")
    for name, data in patches.items():
        if (source / name).read_bytes() != data:
            raise PreparationError(f"Missing/stale FP16 ExpCast source: {name}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "build/mindie_laser_quant_vscale_pv_expcast_source")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "build/mindie_laser_quant_vscale_pv_prefetch_source")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--offline", action="store_true", help="Compatibility flag; preparation is always local-only")
    args = parser.parse_args()
    prepare(args.source_dir, args.reference_dir, args.manifest or default_manifest(args.source_dir),
            offline=args.offline)
    print(f"Prepared {OP_TYPE}: stable PV prefetch, original-exp and FP16 ExpCast modes")
