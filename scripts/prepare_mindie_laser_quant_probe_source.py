#!/usr/bin/env python3
"""Prepare isolated, local-only phase diagnostics of scale-reuse INT8 Laser."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_mindie_laser_quant_reuse_source as baseline
from scripts.prepare_mindie_laser_source import SOURCE_LOCK, OP_ROOT, PREPARE_SCRIPT, default_manifest
from scripts.mindie_laser_offline_cache import local_objects_only, seed_local_checkout
from scripts.prepare_cann_fas_source import PreparationError, ensure_checkout, check_dirty, audit_files, sha256
from scripts.mindie_laser_quant_patch import once

OP_NAME = "vc_laser_quant_probe_attention"
OP_TYPE = "VcLaserQuantProbeAttention"
NATIVE_API = "aclnnVcLaserQuantProbeAttention"
CUSTOM_OP_ROOT = f"csrc/ops/{OP_NAME}"
MARKER_TOKEN = "vc_mindie_laser_quant_probe_v1"
MARKER_ENV = "VC_MINDIE_LASER_QUANT_PROBE_MARKER_FILE"
IDENTITY = dict(op_name=OP_NAME, op_type=OP_TYPE, native_api=NATIVE_API, vendor="vc_laser_quant_probe")
ALGORITHM = ("Scale-reuse INT8 QK / INT32 accumulation / FIX DEQF16, original Laser "
             "exponential, FP16 PV; no V quantization/reorder/smoothing, no ExpCast")
SCALE_CONTRACT = baseline.SCALE_CONTRACT
TRACE_SHAPE = [4, 3, 8, 24]
PROBE_CONTRACT = {
    "output_index": 2, "dtype": "int64", "shape": TRACE_SHAPE,
    "modes": {"0": "disabled", "1": "original cross-core waits", "2": "waits and completed Vector phases"},
    "allocation": "fresh zero-filled trace in every mode; original statistics allocation retained",
    "timing": "sampled system ticks; not production phase percentages or additive kernel durations",
    "calibration": "compare installed scale-reuse, compiled probe mode 0, wait mode 1 and phase mode 2",
}


def rename(data: bytes) -> bytes:
    for old, new in ((baseline.OP_TYPE, OP_TYPE), (baseline.OP_NAME, OP_NAME),
                     (baseline.MARKER_TOKEN, MARKER_TOKEN), (baseline.MARKER_ENV, MARKER_ENV)):
        data = data.replace(old.encode(), new.encode())
    return data


def patch_abi(path: str, data: bytes) -> bytes:
    """Keep descriptor, schema, inference and entry output ordering consistent."""
    name = path.replace("\\", "/")
    if name.endswith(f"/op_host/{OP_NAME}_def.cpp"):
        data = once(data, b'        this->Attr("scale_value")', b'''        this->Output("phase_trace")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND});
        this->Attr("scale_value")''')
        data = once(data, b'        this->Attr("use_expcast").AttrType(OPTIONAL).Bool(false);',
                    b'        this->Attr("use_expcast").AttrType(OPTIONAL).Bool(false);\n'
                    b'        this->Attr("probe_mode").AttrType(OPTIONAL).Int(0);')
    elif name.endswith(f"/op_host/{OP_NAME}_proto.cpp"):
        data = once(data, b"    return ge::GRAPH_SUCCESS;\n}\n\nstatic ge::graphStatus", b'''    gert::Shape *traceOut = context->GetOutputShape(2);
    if (traceOut == nullptr) return ge::GRAPH_FAILED;
    traceOut->SetDimNum(4);
    traceOut->SetDim(0, 4);
    traceOut->SetDim(1, 3);
    traceOut->SetDim(2, 8);
    traceOut->SetDim(3, 24);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus''')
        data = once(data, b"    context->SetOutputDataType(1, ge::DT_FLOAT);",
                    b"    context->SetOutputDataType(1, ge::DT_FLOAT);\n"
                    b"    context->SetOutputDataType(2, ge::DT_INT64);")
    elif name.endswith(f"/op_host/{OP_NAME}_tiling.h"):
        data = once(data, b"TILING_DATA_FIELD_DEF(bool, useExpCast);",
                    b"TILING_DATA_FIELD_DEF(bool, useExpCast);\nTILING_DATA_FIELD_DEF(int32_t, probeMode);")
    elif name.endswith(f"/op_host/{OP_NAME}_tiling.cpp"):
        data = once(data, b"    tilingData.set_useExpCast(*useExpCast);", b'''    if (*useExpCast) return ge::GRAPH_FAILED;
    const auto probeMode = context->GetAttrs()->GetAttrPointer<int64_t>(8);
    if (probeMode == nullptr || *probeMode < 0 || *probeMode > 2) return ge::GRAPH_FAILED;
    tilingData.set_useExpCast(false);
    tilingData.set_probeMode(static_cast<int32_t>(*probeMode));''')
        # Upstream discarded this status. Propagate it so malformed diagnostic
        # attributes cannot reach a kernel with incomplete tiling data.
        data = once(data, f"    tiling.Tiling4{OP_TYPE}(context);".encode(),
                    f"    if (tiling.Tiling4{OP_TYPE}(context) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;".encode())
    elif name.endswith(f"/op_kernel/{OP_NAME}.cpp"):
        data = once(data, b"    __gm__ uint8_t *__restrict__ workspace,",
                    b"    __gm__ uint8_t * __restrict__ phase_trace_gm,\n"
                    b"    __gm__ uint8_t *__restrict__ workspace,")
        if data.count(b"    op.Run();") != 2:
            raise PreparationError("Expected exactly one Cube and one Vector entry")
        data = data.replace(b"    op.Run();", b"    op.SetupProbe(phase_trace_gm, tiling_data->probeMode);\n    op.Run();")
    return data


def expected_files(source):
    from scripts.mindie_laser_quant_phase_patch import apply, HEADER
    base_files, base_records = baseline.expected_files(source)
    patches, records = {}, {}
    for name, data in base_files.items():
        target = rename(name.encode()).decode()
        transformed = apply(target, patch_abi(target, rename(data)))
        patches[target] = transformed
        if name in base_records:
            records[target] = {
                **base_records[name], "prepared_sha256": sha256(transformed),
                "scale_reuse_prepared_sha256": sha256(data),
            }
    header_name = f"{CUSTOM_OP_ROOT}/op_kernel/vc_laser_phase_probe.h"
    patches[header_name] = HEADER
    records[header_name] = dict(original_path=None, original_sha256=None, prepared_sha256=sha256(HEADER))
    return patches, records


def _audit(source, patches):
    check_dirty(source, set(patches))
    for path in (source / CUSTOM_OP_ROOT).rglob("*"):
        if path.is_symlink() or (path.is_file() and path.relative_to(source).as_posix() not in patches):
            raise PreparationError(f"Unexpected generated source: {path}")
    return audit_files(source, patches)[0]


def _identity(source):
    return {
        "source_dir": str(source), "variant": "quant_phase_probe", "op_type": OP_TYPE,
        "native_api": NATIVE_API, "custom_operator": CUSTOM_OP_ROOT,
        "marker_token": MARKER_TOKEN, "marker_environment": MARKER_ENV,
        "upstream": {"url": SOURCE_LOCK.url, "commit": SOURCE_LOCK.commit, "tag": SOURCE_LOCK.tag},
        "qk_scale_contract": SCALE_CONTRACT, "scale_reuse_contract": baseline.REUSE_CONTRACT,
        "probe_contract": PROBE_CONTRACT, "diagnostic_instrumentation": True,
    }


def cached_references(reference):
    result = [reference]
    for name in ("mindie_laser_quant_reuse_build.json", "mindie_laser_quant_build.json", "mindie_laser_build.json"):
        path = ROOT / "results" / name
        if path.is_file():
            build = json.loads(path.read_text(encoding="utf-8"))
            if build.get("status") == "passed" and build.get("source_dir"):
                result.append(Path(build["source_dir"]))
    result.extend(ROOT / "build" / name for name in
                  ("mindie_laser_quant_reuse_source", "mindie_laser_quant_source", "mindie_laser_source", "mindie-sd-reference"))
    return result


@local_objects_only()
def prepare(source_dir, reference_dir, manifest, *, offline=True):
    # Retain the old keyword for callers, but never fetch remotely.
    source, reference, destination = map(lambda x: Path(x).resolve(), (source_dir, reference_dir, manifest))
    if source == reference or source.is_relative_to(reference) or reference.is_relative_to(source):
        raise PreparationError("Phase-probe source must be separate from the reference checkout")
    seed_local_checkout(source, cached_references(reference), SOURCE_LOCK)
    ensure_checkout(source, SOURCE_LOCK, offline=True)
    patches, generated = expected_files(source)
    records = _audit(source, patches)
    report = {
        "schema_version": 1, **_identity(source), "original_operator": OP_ROOT,
        "algorithm_scope": ALGORITHM, "kernel_inverse_rename_equivalent": False,
        "source_files": records, "generated_files": generated,
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
            raise PreparationError(f"Wrong phase-probe source identity: {key}")
    patches, generated = expected_files(source)
    if _audit(source, patches) != manifest.get("source_files") or generated != manifest.get("generated_files"):
        raise PreparationError("Phase-probe source audit differs from manifest")
    for name, data in patches.items():
        if (source / name).read_bytes() != data:
            raise PreparationError(f"Missing/stale phase-probe source: {name}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "build/mindie_laser_quant_probe_source")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "build/mindie-sd-reference")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--offline", action="store_true", help="Compatibility flag; source preparation is always local-only")
    args = parser.parse_args()
    prepare(args.source_dir, args.reference_dir, args.manifest or default_manifest(args.source_dir), offline=args.offline)
    print(f"Prepared {OP_TYPE}: independent diagnostic modes; unchanged attention arithmetic")
