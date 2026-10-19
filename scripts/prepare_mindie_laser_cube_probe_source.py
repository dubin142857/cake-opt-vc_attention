#!/usr/bin/env python3
"""Prepare independent Cube diagnostics of original FP16 and scale-reuse INT8 Laser.

Only pinned local Git objects are read. Both providers retain their original
Vector arithmetic; an explicit third output contains diagnostic records.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_mindie_laser_source as original
from scripts import prepare_mindie_laser_quant_reuse_source as quant
from scripts.mindie_laser_quant_patch import once
from scripts.mindie_laser_offline_cache import local_objects_only, seed_local_checkout
from scripts.prepare_cann_fas_source import PreparationError, ensure_checkout, check_dirty, audit_files, sha256, git

SOURCE_LOCK = original.SOURCE_LOCK
OP_ROOT = original.OP_ROOT
PREPARE_SCRIPT = original.PREPARE_SCRIPT
KINDS = ("int8", "fp16")
TRACE_SHAPE = [4, 2, 3, 24]
PROBE_CONTRACT = {
    "output_index": 2, "dtype": "int64", "shape": TRACE_SHAPE,
    "modes": {"0": "disabled", "1": "issue intervals", "2": "completed Cube intervals"},
    "allocation": "fresh zero-filled trace in every mode; original statistics allocation retained",
    "timing": "sparse system ticks; not production phase percentages or additive kernel durations",
    "calibration": "each provider against its own uninstrumented baseline and compiled plain mode",
}


def identity(kind):
    if kind not in KINDS:
        raise PreparationError(f"Unknown Cube-probe kind: {kind}")
    name = f"vc_laser_cube_{kind}_probe"
    op_type = f"VcLaserCube{kind.capitalize()}ProbeAttention"
    return dict(op_name=name + "_attention", op_type=op_type,
                native_api="aclnn" + op_type, vendor=name)


def marker_token(kind):
    return f"vc_mindie_laser_cube_{kind}_probe_v1"


def marker_environment(kind):
    return f"VC_MINDIE_LASER_CUBE_{kind.upper()}_PROBE_MARKER_FILE"


def _rename(data, kind):
    target = identity(kind)
    source = quant if kind == "int8" else original
    old_name = quant.OP_NAME if kind == "int8" else "vc_laser_attention"
    old_env = quant.MARKER_ENV if kind == "int8" else "VC_MINDIE_LASER_MARKER_FILE"
    for old, new in ((source.OP_TYPE, target["op_type"]), (old_name, target["op_name"]),
                     (source.MARKER_TOKEN, marker_token(kind)), (old_env, marker_environment(kind))):
        data = data.replace(old.encode(), new.encode())
    return data


def baseline_files(source, kind):
    if kind == "int8":
        return quant.expected_files(source)
    # Reconstruct the original renamed baseline directly from locked Git bytes.
    # Do not use the INT8 patch or a generated reference worktree for FP16.
    names = git(source, "ls-tree", "-r", "--name-only", "HEAD", "--", OP_ROOT).decode().splitlines()
    if not names or not any("/op_kernel/" in name for name in names):
        raise PreparationError("Pinned Laser subtree is absent")
    patches = {PREPARE_SCRIPT: original.parallelism_patch(git(source, "show", f"HEAD:{PREPARE_SCRIPT}"))}
    records = {}
    for name in names:
        raw = git(source, "show", f"HEAD:{name}")
        data = original.rename(raw)
        if name == original.TILING_FILE:
            data = original.marker_patch(data)
        target = original.custom_path(name)
        patches[target] = data
        records[target] = dict(original_path=name, original_sha256=sha256(raw), prepared_sha256=sha256(data))
    patches[original.MARKER_FILE] = original.MARKER_HEADER
    records[original.MARKER_FILE] = dict(original_path=None, original_sha256=None,
                                       prepared_sha256=sha256(original.MARKER_HEADER))
    return patches, records


def patch_abi(path, data, kind):
    name = identity(kind)["op_name"]
    op_type = identity(kind)["op_type"]
    if path.endswith(f"/op_host/{name}_def.cpp"):
        dtypes = b"ge::DT_INT64" if kind == "int8" else b"ge::DT_INT64, ge::DT_INT64"
        formats = b"ge::FORMAT_ND" if kind == "int8" else b"ge::FORMAT_ND, ge::FORMAT_ND"
        data = once(data, b'        this->Attr("scale_value")',
            b'        this->Output("cube_trace")\n                .ParamType(REQUIRED)\n'
            b'                .DataType({' + dtypes + b'} )\n'
            b'                .Format({' + formats + b'});\n'
            b'        this->Attr("scale_value")')
        anchor = (b'        this->Attr("use_expcast").AttrType(OPTIONAL).Bool(false);'
                  if kind == "int8" else b'                .Bool(true);')
        data = once(data, anchor, anchor + b'\n        this->Attr("probe_mode").AttrType(OPTIONAL).Int(0);')
    elif path.endswith(f"/op_host/{name}_proto.cpp"):
        data = once(data, b"    return ge::GRAPH_SUCCESS;\n}\n\nstatic ge::graphStatus", b'''    gert::Shape *traceOut = context->GetOutputShape(2);
    if (traceOut == nullptr) return ge::GRAPH_FAILED;
    traceOut->SetDimNum(4);
    traceOut->SetDim(0, 4);
    traceOut->SetDim(1, 2);
    traceOut->SetDim(2, 3);
    traceOut->SetDim(3, 24);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus''')
        data = once(data, b"    context->SetOutputDataType(1, ge::DT_FLOAT);",
                    b"    context->SetOutputDataType(1, ge::DT_FLOAT);\n    context->SetOutputDataType(2, ge::DT_INT64);")
    elif path.endswith(f"/op_host/{name}_tiling.h"):
        data = once(data, b"END_TILING_DATA_DEF;", b"TILING_DATA_FIELD_DEF(int32_t, probeMode);\nEND_TILING_DATA_DEF;")
    elif path.endswith(f"/op_host/{name}_tiling.cpp"):
        attr_index = 8 if kind == "int8" else 7
        anchor = b"    tilingData.set_isHighPrecision(*isHighPrecision);"
        data = once(data, anchor, anchor + f'''
    const auto probeMode = context->GetAttrs()->GetAttrPointer<int64_t>({attr_index});
    if (probeMode == nullptr || *probeMode < 0 || *probeMode > 2) return ge::GRAPH_FAILED;
    tilingData.set_probeMode(static_cast<int32_t>(*probeMode));'''.encode())
        if kind == "int8":
            data = once(data, b"    tilingData.set_useExpCast(*useExpCast);",
                        b"    if (*useExpCast) return ge::GRAPH_FAILED;\n    tilingData.set_useExpCast(false);")
        data = once(data, f"    tiling.Tiling4{op_type}(context);".encode(),
                    f"    if (tiling.Tiling4{op_type}(context) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;".encode())
    elif path.endswith(f"/op_kernel/{name}.cpp"):
        data = once(data, b"    __gm__ uint8_t *__restrict__ workspace,",
                    b"    __gm__ uint8_t * __restrict__ cube_trace_gm,\n    __gm__ uint8_t *__restrict__ workspace,")
        if data.count(b"    op.Run();") != 2:
            raise PreparationError("Expected exactly one Cube and one Vector entry")
        # Instrument only the Cube instance. The Vector branch remains byte-for-byte unchanged.
        data = data.replace(b"    op.Run();", b"    op.SetupCubeProbe(cube_trace_gm, tiling_data->probeMode);\n    op.Run();", 1)
    return data


def expected_files(source, kind="int8"):
    from scripts.mindie_laser_cube_probe_patch import apply
    base_files, base_records = baseline_files(source, kind)
    patches, records = {}, {}
    for name, data in base_files.items():
        target = _rename(name.encode(), kind).decode()
        transformed = apply(target, patch_abi(target, _rename(data, kind), kind))
        patches[target] = transformed
        if name in base_records:
            records[target] = {**base_records[name], "prepared_sha256": sha256(transformed),
                               "baseline_prepared_sha256": sha256(data)}
    header_name = f'csrc/ops/{identity(kind)["op_name"]}/op_kernel/vc_laser_cube_probe.h'
    header = (ROOT / "ascend/laser_cube_probe/vc_laser_cube_probe.h").read_bytes()
    patches[header_name] = header
    records[header_name] = dict(original_path=None, original_sha256=None, prepared_sha256=sha256(header))
    return patches, records


def _audit(source, patches, kind):
    check_dirty(source, set(patches))
    for path in (source / "csrc/ops" / identity(kind)["op_name"]).rglob("*"):
        if path.is_symlink() or (path.is_file() and path.relative_to(source).as_posix() not in patches):
            raise PreparationError(f"Unexpected generated source: {path}")
    return audit_files(source, patches)[0]


def _manifest_identity(source, kind):
    record = dict(source_dir=str(source), variant=f"cube_{kind}_probe", **identity(kind),
                  custom_operator=f'csrc/ops/{identity(kind)["op_name"]}',
                  marker_token=marker_token(kind), marker_environment=marker_environment(kind),
                  upstream={"url": SOURCE_LOCK.url, "commit": SOURCE_LOCK.commit, "tag": SOURCE_LOCK.tag},
                  diagnostic_instrumentation=True, probe_contract=PROBE_CONTRACT, probe_kind=kind)
    if kind == "int8":
        record.update(qk_scale_contract=quant.SCALE_CONTRACT, scale_reuse_contract=quant.REUSE_CONTRACT)
    return record


@local_objects_only()
def prepare(source_dir, reference_dir, manifest, *, kind="int8", offline=True):
    identity(kind)
    source, reference, destination = map(lambda x: Path(x).resolve(), (source_dir, reference_dir, manifest))
    if source == reference or source.is_relative_to(reference) or reference.is_relative_to(source):
        raise PreparationError("Cube-probe source must be separate from the reference checkout")
    seed_local_checkout(source, quant.cached_references(reference), SOURCE_LOCK)
    ensure_checkout(source, SOURCE_LOCK, offline=True)
    patches, generated = expected_files(source, kind)
    records = _audit(source, patches, kind)
    report = {"schema_version": 1, **_manifest_identity(source, kind), "original_operator": OP_ROOT,
              "algorithm_scope": ("Original pinned FP16 QK/PV Laser" if kind == "fp16" else
                                  "Scale-reuse INT8 QK/INT32/FIX DEQF16; original FP16 PV"),
              "kernel_inverse_rename_equivalent": False, "source_files": records,
              "generated_files": generated, "modified_original_files": [PREPARE_SCRIPT]}
    for name, data in patches.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if not destination.exists() or destination.read_bytes() != data:
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(destination)
    return report


@local_objects_only()
def validate_prepared_source(source, manifest_path, *, kind="int8"):
    source = Path(source).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    ensure_checkout(source, SOURCE_LOCK, offline=True)
    for key, value in _manifest_identity(source, kind).items():
        if manifest.get(key) != value:
            raise PreparationError(f"Wrong Cube-probe source identity: {key}")
    patches, generated = expected_files(source, kind)
    if _audit(source, patches, kind) != manifest.get("source_files") or generated != manifest.get("generated_files"):
        raise PreparationError("Cube-probe source audit differs from manifest")
    for name, data in patches.items():
        if (source / name).read_bytes() != data:
            raise PreparationError(f"Missing/stale Cube-probe source: {name}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "build/mindie-sd-reference")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    prepare(args.source_dir, args.reference_dir, args.manifest or original.default_manifest(args.source_dir), kind=args.kind)
